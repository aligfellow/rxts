"""xtb and tblite runners, method dispatch, units, and the solvated g-xTB composite.

Runners return Hartree and Hartree/Bohr with gradient ``+dE/dx``. The ASE calculator in
:mod:`rxts.scoring` converts once to eV and eV/angstrom forces.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path
from typing import ClassVar
from weakref import finalize

import numpy as np

from rxts.utils import xyz_block

BOHR = 0.52917721092  # A
HARTREE_TO_KCAL = 627.5094740631
EV_PER_KCAL = 0.0433641153  # ASE speaks eV

#: Command-line flags that select each method.
METHODS = {
    "gfn0": ["--gfn", "0"],
    "gfn1": ["--gfn", "1"],
    "gfn2": ["--gfn", "2"],
    "gfnff": ["--gfnff"],
    "gxtb": ["--gxtb"],
}

_ENERGY = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)")

_INSTALL_HINT = "No xtb binary found. Install xtb (g-xTB for --gxtb) and put it on PATH, or set XTB_EXE=/path/to/xtb."


#: An explicit xtb path, set by :func:`set_xtb`; takes precedence over $XTB_EXE and PATH.
_XTB_PATH = None


def set_xtb(path: str | Path | None) -> None:
    """Point rxts at an xtb binary for the rest of the process. Precedence: this > $XTB_EXE > PATH."""
    global _XTB_PATH  # noqa: PLW0603 -- a process-wide binary path is a reasonable use of a module global
    _XTB_PATH = str(Path(path).expanduser()) if path else None
    supports_gxtb.cache_clear()


def xtb_exe():
    """Locate the xtb binary: :func:`set_xtb` if called, else ``$XTB_EXE``, else ``xtb`` on PATH."""
    exe = _XTB_PATH or os.environ.get("XTB_EXE", "xtb")
    found = shutil.which(exe)
    if found:
        return found
    return exe if os.path.isfile(exe) and os.access(exe, os.X_OK) else None


def xtb_available():
    """Whether an xtb binary can be found."""
    return xtb_exe() is not None


@cache
def supports_gxtb():
    """Whether this xtb understands --gxtb (g-xTB ships as a patched xtb; a stock build lacks it)."""
    exe = xtb_exe()
    if exe is None:
        return False
    out = subprocess.run([exe, "--help"], capture_output=True, text=True, check=False)
    return "--gxtb" in out.stdout


def _flags(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHODS)}")
    if method == "gxtb" and not supports_gxtb():
        # Stock xtb can ignore this flag and run GFN2 successfully.
        raise RuntimeError(
            f"{xtb_exe()} does not support --gxtb. Select a patched binary with rxts.set_xtb(...) "
            "or XTB_EXE. To use GFN2, request method='gfn2'."
        )
    return METHODS[method]


def _env(threads):
    """Pin xtb's OpenMP thread count, so a pool of jobs does not oversubscribe."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_STACKSIZE", "4G")
    return env


# g-xTB in solvent, built as a composite
#
# g-xTB has no ALPB parameters. The solvated surface uses the same composite as ORCA's
# `oet_gxtb_alpb`:
#
#     E(solvated g-xTB) = E_gxtb(gas) + [ E_gfn2(solvent) - E_gfn2(gas) ]
#
# The additive formula applies to energies, gradients, and Hessians.

#: The electronic structure the composite is built on. It has no ALPB parameters, which is why the
#: composite exists at all.
COMPOSITE_BASE = "gxtb"

#: The method that carries the ALPB parameters g-xTB lacks.
SOLVATION_BASE = "gfn2"


def _is_composite(method, solvent):
    """Whether this (method, solvent) pair has to be built rather than run."""
    return bool(solvent) and method == COMPOSITE_BASE


def composite_legs(solvent):
    """Return the composite's three (method, solvent) legs: base, solvated, unsolvated."""
    return ((COMPOSITE_BASE, None), (SOLVATION_BASE, solvent), (SOLVATION_BASE, None))


def _combine(base, solvated, unsolvated):
    """Combine base + (solvated - unsolvated), returning None if a leg failed."""
    if base is None or solvated is None or unsolvated is None:
        return None
    return base + (solvated - unsolvated)


def _read_engrad(path, n_atoms, flags_text):
    """Read Hartree energy and Hartree/Bohr gradient; report printing overflow with invocation flags."""
    values = []
    for line in path.read_text().splitlines():
        token = line.strip()
        if not token or token.startswith("#") or len(token.split()) > 1:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:  # `****`: a gradient too large for xtb to print
            raise RuntimeError(f"xtb could not print its gradient at this geometry ({flags_text}, {token!r}).") from exc

    if len(values) < 2 + 3 * n_atoms:
        raise RuntimeError(f"Read {len(values)} numbers from {path}, expected {2 + 3 * n_atoms}.")
    return values[1], np.array(values[2 : 2 + 3 * n_atoms]).reshape(n_atoms, 3)


class XtbBackend:
    """Run xtb in an independent scratch directory, retaining restart state between calls."""

    @classmethod
    def available(cls):
        """Whether an xtb binary can be found."""
        return xtb_available()

    @classmethod
    def supports(cls, method):
        """Every method in :data:`METHODS` runs through xtb. The g-xTB silent-fallback guard is in `_flags`."""
        return method in METHODS

    @classmethod
    def make(cls, charge=0, multiplicity=1, n_cores=1, timeout=None):
        """Build a runner owning a fresh scratch directory."""
        return cls(charge=charge, multiplicity=multiplicity, n_cores=n_cores, timeout=timeout)

    def __init__(self, charge=0, multiplicity=1, n_cores=1, timeout=None):
        """Locate the binary and claim a scratch directory this runner owns for its whole life."""
        exe = xtb_exe()
        if exe is None:
            raise RuntimeError(_INSTALL_HINT)
        self.exe = exe
        self.charge = charge
        self.multiplicity = multiplicity
        self.n_cores = n_cores
        self.timeout = timeout  # seconds per invocation; None leaves it unbounded
        self._dir = tempfile.TemporaryDirectory(prefix="rxts-gxtb-")
        finalize(self, self._dir.cleanup)
        self.path = Path(self._dir.name)
        self.n_calls = 0  # xtb invocations

    def run(self, atoms, method, solvent, gradient=True, restart=True):
        """Run one point, returning Hartree and Hartree/Bohr with gradient ``+dE/dx``."""
        # _flags() also raises if g-xTB is asked of an xtb that would silently answer as GFN2.
        flags = [
            *_flags(method),
            *(["--alpb", solvent] if solvent else []),
            *([] if restart else ["--norestart"]),
        ]
        (self.path / "in.xyz").write_text(xyz_block(atoms.get_chemical_symbols(), atoms.get_positions()))

        try:
            out = subprocess.run(
                [
                    self.exe,
                    "in.xyz",
                    *flags,
                    *(["--grad"] if gradient else []),
                    "--sp",
                    "--chrg",
                    str(self.charge),
                    "--uhf",
                    str(self.multiplicity - 1),
                    "-P",
                    str(self.n_cores),
                ],
                cwd=self.path,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
                env=_env(self.n_cores),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"xtb timed out after {self.timeout}s ({' '.join(flags)}).") from exc
        if out.returncode != 0:
            raise RuntimeError(f"xtb failed ({' '.join(flags)}):\n" + out.stdout[-2000:])
        self.n_calls += 1
        if gradient:
            return _read_engrad(self.path / "in.engrad", len(atoms), " ".join(flags))
        match = _ENERGY.search(out.stdout)
        if match is None:
            raise RuntimeError(f"xtb printed no TOTAL ENERGY ({' '.join(flags)}):\n" + out.stdout[-2000:])
        return float(match.group(1)), None


class TbliteBackend:
    """Run GFN1/GFN2 in-process and carry a density between points by default.

    Restart can change which electronic solution converges. Use restart=False for independent
    evaluations; reuse a runner only when density continuation is intended.
    """

    #: tblite's names for the methods this backend supports.
    _NAMES: ClassVar[dict[str, str]] = {"gfn1": "GFN1-xTB", "gfn2": "GFN2-xTB"}

    @classmethod
    def available(cls):
        """Whether `tblite` can be imported on this machine."""
        try:
            import tblite.interface  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def supports(cls, method):
        """Only GFN1 and GFN2 -- g-xTB stays subprocess, as tblite does not ship it."""
        return method in cls._NAMES

    @classmethod
    def make(cls, charge=0, multiplicity=1, n_cores=1, timeout=None):
        """Build a runner. `timeout` is ignored -- an in-process call cannot be killed like a process."""
        del timeout
        return cls(charge=charge, multiplicity=multiplicity, n_cores=n_cores)

    def __init__(self, charge=0, multiplicity=1, n_cores=1):
        """Record the surface; the tblite Calculator and its density are built lazily on the first run."""
        self.charge = charge
        self.multiplicity = multiplicity
        self.n_cores = n_cores
        self.path = None  # tblite has no scratch directory; GXTB.path passes this through
        self.n_calls = 0  # subprocess invocations -- always 0 for an in-process backend
        self._calc = None  # the cached tblite Calculator
        self._key = None  # (method, solvent) it was built for
        self._numbers = np.array([], dtype=int)  # the atomic numbers it was built for
        self._ref = None  # the last SCF Result; the warm-restart density for the NEXT run (None = cold)

    @property
    def density(self):
        """Last SCF result, used to warm-start the next point."""
        return self._ref

    def warm_from(self, ref):
        """Use an upstream density for the next point; ownership passes to this runner."""
        self._ref = ref

    def run(self, atoms, method, solvent, gradient=True, restart=True):
        """Run one point, optionally cold, returning Hartree and Hartree/Bohr with gradient ``+dE/dx``."""
        from tblite.interface import Calculator

        numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions() / BOHR  # tblite wants Bohr
        key = (method, solvent)
        if self._calc is None or key != self._key or not np.array_equal(numbers, self._numbers):
            # ALPB and gas are DIFFERENT Calculators (one has add("alpb-solvation"), one does not),
            # which is why the composite gives its wet and dry legs their own runners.
            changed = self._calc is not None  # a real (method/solvent/atoms) CHANGE, not the first build
            calc = Calculator(
                self._NAMES[method], numbers, positions, charge=float(self.charge), uhf=self.multiplicity - 1
            )
            calc.set("verbosity", 0)
            if solvent:
                calc.add("alpb-solvation", solvent)
            self._calc, self._key, self._numbers = calc, key, numbers
            if changed:
                # Previous density was for a different system; the first build keeps _ref (a warm_from
                # seed for exactly this system, else None).
                self._ref = None
        else:
            self._calc.update(positions)
        # copy=False: the reference is this runner's own density (never shared), consumed in place.
        from tblite.exceptions import TBLiteRuntimeError

        try:
            res = self._calc.singlepoint(self._ref if restart else None, copy=False)
        except TBLiteRuntimeError as error:
            raise RuntimeError(
                "tblite SCF failed at the default electronic temperature. Serial Fermi-smearing "
                "rescue is refused because it would change the optimiser's surface."
            ) from error
        self._ref = res
        gradient_au = np.asarray(res.get("gradient")) if gradient else None
        return float(res.get("energy")), gradient_au


def select_backend(method):
    """Choose in-process tblite for supported methods, otherwise the xtb subprocess."""
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHODS)}")
    if TbliteBackend.available() and TbliteBackend.supports(method):
        return TbliteBackend  # gfn1/gfn2 in-process when tblite is installed
    if XtbBackend.available() and XtbBackend.supports(method):
        return XtbBackend  # g-xTB, and the gfn1/gfn2 fallback when tblite is absent
    raise RuntimeError(_INSTALL_HINT)
