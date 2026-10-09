"""GFN-FF fields with topology frozen in memory or in ``gfnff_topo``.

Validate seed geometries with utils.require_topology before freezing their perceived bonds.
"""

from __future__ import annotations

import subprocess
import tempfile
from importlib import import_module
from pathlib import Path
from weakref import finalize

import numpy as np
from rdkit import Chem

from rxts.utils import to_xyz, xyz_block

# Shared constants, parser and executable helpers; keep their definitions in xtb.py.
from rxts.xtb import _ENERGY, _INSTALL_HINT, BOHR, HARTREE_TO_KCAL, _env, xtb_exe

_TOPOLOGY_REUSED = "topology read from file successfully"

#: Marker for an unusable evaluation; the sampler rejects it even with a zero gradient.
UNUSABLE = 1e9  # kcal/mol


def set_gfnff(path):
    """Add the directory containing a local `gfnff` package to this process's import path."""
    import sys

    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def in_process_available():
    """Return whether the standalone GFN-FF library is importable. See :func:`set_gfnff`."""
    try:
        import_module("gfnff")
    except ImportError:
        return False
    return True


class GFNFF:
    """Freeze a molecule's perceived topology; charge is total charge and solvent an ALPB name."""

    def __new__(cls, *args, **kwargs):
        """Pick the backend: the linked library if available, the subprocess otherwise."""
        # Dispatch only the base class, preserving explicitly selected backends.
        if cls is not GFNFF:
            return super().__new__(cls)
        return super().__new__(_LinkedGFNFF if in_process_available() else _SubprocessGFNFF)

    def __init__(self, mol, charge=0, solvent=None):
        """Take the atom count and symbols off `mol`; its coordinates are read per call, not stored."""
        formal_charge = Chem.GetFormalCharge(mol)
        if charge != formal_charge:
            raise ValueError(f"charge={charge} conflicts with the molecular graph's formal charge {formal_charge}")
        self.charge = charge
        self.solvent = solvent
        self.n_atoms = mol.GetNumAtoms()
        self.symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    def energy_grad(self, x):
        """Energy (kcal/mol) and gradient (kcal/mol/A) of this frozen topology at `x`."""
        raise NotImplementedError


class _LinkedGFNFF(GFNFF):
    """The linked C API builds topology once; singlepoint cannot rebuild it."""

    def __init__(self, mol, charge=0, solvent=None):
        super().__init__(mol=mol, charge=charge, solvent=solvent)
        constructor = vars(import_module("gfnff"))["GFNFFCalculator"]

        self.numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
        ref_charges = np.array([a.GetFormalCharge() for a in mol.GetAtoms()], dtype=np.float64)
        x = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
        try:
            # Automatic charge guessing can assign an anion to a neutral relay partner.
            # Reference charges set the EEQ net charges of the inferred frozen fragments.
            self._calc = constructor(
                self.numbers,
                x / BOHR,
                charge=charge,
                solvent=solvent or "",
                printlevel=0,
                ref_charges=ref_charges,
            )
        except TypeError as exc:
            if "ref_charges" in str(exc):
                raise RuntimeError(
                    "This gfnff installation cannot accept host-supplied fragment charges; "
                    "install a gfnff build with the `ref_charges` calculator argument."
                ) from exc
            raise RuntimeError(f"GFN-FF could not build a topology for this geometry: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"GFN-FF could not build a topology for this geometry: {exc}") from exc

    def energy_grad(self, x):
        x = np.asarray(x, dtype=float).reshape(self.n_atoms, 3)
        energy, gradient, _ = self._calc.singlepoint(self.numbers, x / BOHR)

        # Same unusable-geometry guard as the subprocess; here the number arrives intact, so test it.
        if not np.isfinite(energy) or not np.isfinite(gradient).all():
            return UNUSABLE, np.zeros((self.n_atoms, 3))

        return energy * HARTREE_TO_KCAL, gradient * (HARTREE_TO_KCAL / BOHR)


class _SubprocessGFNFF(GFNFF):
    """Use ``xtb --gfnff`` with a frozen ``gfnff_topo`` when the linked library is absent."""

    def __init__(self, mol, charge=0, solvent=None):
        super().__init__(mol=mol, charge=charge, solvent=solvent)
        fragments = Chem.GetMolFrags(mol)
        fragment_charges = [sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in fragment) for fragment in fragments]
        if len(fragments) > 1 and (charge != 0 or any(fragment_charges)):
            raise RuntimeError(
                "The xtb subprocess GFN-FF backend cannot set fragment charges for a charged "
                "multifragment molecule; use the linked gfnff calculator."
            )
        exe = xtb_exe()
        if exe is None:
            raise RuntimeError(_INSTALL_HINT)
        self.exe = exe

        # The directory holds the force field: gfnff_topo lives in it.
        self._dir = tempfile.TemporaryDirectory(prefix="rxts-gfnff-")
        finalize(self, self._dir.cleanup)
        self.path = Path(self._dir.name)
        (self.path / "in.xyz").write_text(to_xyz(mol))

        out = self._xtb(["--sp"])
        if out.returncode != 0:
            raise RuntimeError("GFN-FF could not build a topology for this geometry.\n" + out.stdout[-2000:])
        if not (self.path / "gfnff_topo").exists():
            raise RuntimeError("xtb did not write gfnff_topo; the topology cannot be frozen.")

    def _xtb(self, flags):
        cmd = [
            self.exe,
            "in.xyz",
            "--gfnff",
            *flags,
            "--chrg",
            str(self.charge),
            "--parallel",
            "1",
        ]
        if self.solvent:
            cmd += ["--alpb", self.solvent]
        return subprocess.run(
            cmd,
            cwd=self.path,
            capture_output=True,
            text=True,
            check=False,
            env=_env(1),
        )

    def energy_grad(self, x):
        """Frozen-topology energy (kcal/mol) and gradient (kcal/mol/A) at ``x``."""
        positions = np.asarray(x, dtype=float).reshape(self.n_atoms, 3)
        (self.path / "in.xyz").write_text(xyz_block(self.symbols, positions))
        out = self._xtb(["--grad", "--sp"])
        if out.returncode != 0:
            raise RuntimeError("GFN-FF failed:\n" + out.stdout[-2000:])
        if _TOPOLOGY_REUSED not in out.stdout:
            raise RuntimeError(
                "GFN-FF re-perceived its topology instead of reusing the frozen one. "
                "The interpolation would then be between two copies of the same "
                "molecule, and would mean nothing."
            )

        match = _ENERGY.search(out.stdout)
        if match is None:
            raise RuntimeError("Could not read a GFN-FF energy.\n" + out.stdout[-2000:])
        energy = float(match.group(1)) * HARTREE_TO_KCAL

        return self._parse_gradient(energy)

    def _parse_gradient(self, energy):
        """Read an ORCA-style .engrad file and convert Hartree/Bohr to kcal/mol/A."""
        engrad = (self.path / "in.engrad").read_text().splitlines()

        # A gradient too large to print. See UNUSABLE.
        if any("*" in line for line in engrad):
            return UNUSABLE, np.zeros((self.n_atoms, 3))

        numbers = []
        for token in engrad:
            try:
                numbers.append(float(token))
            except ValueError:
                continue  # a header or label line, not a number
        grad = numbers[2 : 2 + 3 * self.n_atoms]
        if len(grad) != 3 * self.n_atoms:
            raise RuntimeError(
                f"Read {len(grad)} gradient components from {self.path / 'in.engrad'}, expected {3 * self.n_atoms}."
            )
        return energy, np.array(grad).reshape(self.n_atoms, 3) * (HARTREE_TO_KCAL / BOHR)


__all__ = ["GFNFF", "in_process_available", "set_gfnff"]
