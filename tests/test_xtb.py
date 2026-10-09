"xtb parsing, backend selection, and warm restarts."

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from rxts import xtb
from rxts.xtb import METHODS, _flags


def test_switching_binary_refreshes_capabilities(monkeypatch):
    monkeypatch.setattr(xtb, "_XTB_PATH", None)
    monkeypatch.setattr(xtb.shutil, "which", lambda path: path)
    monkeypatch.setattr(
        xtb.subprocess,
        "run",
        lambda command, **kw: types.SimpleNamespace(stdout="--gxtb" if command[0].endswith("patched") else ""),
    )
    try:
        for binary, expected in (("patched", True), ("stock", False), ("~/bin/patched", True)):
            xtb.set_xtb(binary)
            assert not xtb.xtb_exe().startswith("~")
            assert xtb.supports_gxtb() is expected
    finally:
        xtb.supports_gxtb.cache_clear()


@pytest.mark.parametrize("backend", ["xtb", "gfnff"])
def test_runner_scratch_lifetime(monkeypatch, backend):
    import gc
    import warnings

    from rdkit import Chem

    from rxts import gfnff

    monkeypatch.setattr(xtb, "xtb_exe", lambda: "xtb")
    monkeypatch.setattr(gfnff, "xtb_exe", lambda: "xtb")

    def initialise(calculator, flags):
        (calculator.path / "gfnff_topo").touch()
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(gfnff._SubprocessGFNFF, "_xtb", initialise)
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        runner = xtb.XtbBackend() if backend == "xtb" else gfnff._SubprocessGFNFF(mol)
        directory = runner.path
        assert directory.is_dir()
        del runner
        gc.collect()
    assert not directory.exists()
    assert not [warning for warning in caught if issubclass(warning.category, ResourceWarning)]


class _TBLiteRuntimeError(Exception):
    """Stands in for `tblite.exceptions.TBLiteRuntimeError` -- one class, raised and caught."""


class _FakeResult:
    """A tblite ``Result``: it answers `get("energy")` and `get("gradient")` and nothing else."""

    def __init__(self, positions_bohr, temperature):
        self.positions = np.array(positions_bohr, dtype=float)
        self.temperature = temperature  # Hartree, or None at tblite's default

    def get(self, name):
        if name == "energy":
            return 0.5 * float(np.sum(self.positions**2))
        if name == "gradient":
            return self.positions.copy()  # dE/dx of the quadratic above: the geometry itself
        raise KeyError(name)


def _never_converges(positions, temperature):
    """A rescue that always says no -- module level so it is a plain function, not a dataclass default call."""
    return False


@dataclass
class _FakeEngine:
    """What the fake tblite recorded, and the knob that decides which SCFs fail."""

    built: list = field(default_factory=list)  # every Calculator constructed, in order
    calls: list = field(default_factory=list)  # (calculator, res argument, copy) per singlepoint
    returned: list = field(default_factory=list)  # what each singlepoint handed back, same order
    #: `fails(positions_bohr, temperature_hartree) -> bool`; the temperature is None at the default.
    fails: Any = field(default=_never_converges)


def _install_fake_tblite(monkeypatch):
    """Put a fake `tblite` in `sys.modules` and return the record of what it is asked to do.

    All three names are replaced, so this runs identically with tblite installed and without it -- and
    a test can never accidentally reach the real library and start doing chemistry.
    """
    engine = _FakeEngine()

    class Calculator:
        def __init__(self, name, numbers, positions, charge=0.0, uhf=0):
            self.name = name
            self.numbers = np.asarray(numbers)
            self.positions = np.array(positions, dtype=float)
            self.charge, self.uhf = charge, uhf
            self.settings, self.solvation = {}, None
            engine.built.append(self)

        def set(self, key, value):
            self.settings[key] = value

        def add(self, key, value):
            self.solvation = value

        def update(self, positions):
            self.positions = np.array(positions, dtype=float)

        def singlepoint(self, res=None, copy=False):
            engine.calls.append((self, res, copy))
            temperature = self.settings.get("temperature")
            if engine.fails(self.positions, temperature):
                raise _TBLiteRuntimeError("SCF did not converge")
            out = _FakeResult(self.positions, temperature)
            engine.returned.append(out)
            return out

    interface = types.SimpleNamespace(Calculator=Calculator)
    exceptions = types.SimpleNamespace(TBLiteRuntimeError=_TBLiteRuntimeError)
    monkeypatch.setitem(sys.modules, "tblite", types.SimpleNamespace(interface=interface, exceptions=exceptions))
    monkeypatch.setitem(sys.modules, "tblite.interface", interface)
    monkeypatch.setitem(sys.modules, "tblite.exceptions", exceptions)
    return engine


def _atoms(positions):
    """The two things a runner reads off an ASE `Atoms`: the numbers and the geometry."""
    return Atoms(numbers=[6] * len(positions), positions=np.asarray(positions, dtype=float))


# ---------------------------------------------------------------------------
# The .engrad parser: an overflow is the case a real run will not produce on demand
# ---------------------------------------------------------------------------


def _engrad(tmp_path, energy, gradient_lines, n_atoms=2):
    """An ORCA-style .engrad file, written by hand so the pathological lines can be chosen."""
    text = "\n".join(
        [
            "#",
            "# Number of atoms",
            "#",
            f"   {n_atoms}",
            "#",
            "# The current total energy in Eh",
            "#",
            f"    {energy}",
            "#",
            "# The current gradient in Eh/bohr",
            "#",
            *gradient_lines,
            "#",
            "# The atomic numbers and current coordinates in Bohr",
            "#",
            *[f"   6    {i}.0000000    0.0000000    0.0000000" for i in range(n_atoms)],
        ]
    )
    path = tmp_path / "in.engrad"
    path.write_text(text + "\n")
    return path


def test_engrad_parser_reads_energy_and_gradient(tmp_path):
    "Check energy and gradient parsing while excluding the coordinate block."
    path = _engrad(tmp_path, "-5.2500000", [f"    {v:.7f}" for v in (0.1, 0.2, 0.3, -0.1, -0.2, -0.3)])
    energy, gradient = xtb._read_engrad(path, n_atoms=2, flags_text="--gfn 2")

    assert energy == pytest.approx(-5.25)
    assert gradient.shape == (2, 3)
    assert np.allclose(gradient, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])


def test_overflowed_gradient_is_rejected(tmp_path):
    "Check an overflowed gradient fails with its invocation and invalid token."
    lines = ["    ********************"] * 3 + [f"    {v:.7f}" for v in (0.1, 0.2, 0.3)]
    path = _engrad(tmp_path, "-5.2500000", lines)

    with pytest.raises(RuntimeError, match="could not print its gradient") as caught:
        xtb._read_engrad(path, n_atoms=2, flags_text="--gfnff --alpb water")
    assert "--gfnff --alpb water" in str(caught.value), "the invocation that wrote the file has to be named"
    assert "*" in str(caught.value), "and the token that could not be read"


def test_truncated_gradient_reports_expected_count(tmp_path):
    path = _engrad(tmp_path, "-5.2500000", [f"    {v:.7f}" for v in (0.1, 0.2, 0.3)])

    with pytest.raises(RuntimeError, match="expected 8"):
        xtb._read_engrad(path, n_atoms=2, flags_text="--gfn 2")


# ---------------------------------------------------------------------------
# Method flags, and the silent-fallback guard
# ---------------------------------------------------------------------------


def test_gxtb_is_selected_by_its_own_flag(monkeypatch):
    """g-xTB is a patched-binary flag, not a `--gfn` level."""
    monkeypatch.setattr(xtb, "supports_gxtb", lambda: True)  # @cache'd; patch, do not call
    assert _flags("gxtb") == ["--gxtb"]
    assert _flags("gfn2") == ["--gfn", "2"]
    assert set(METHODS) == {"gfn0", "gfn1", "gfn2", "gfnff", "gxtb"}


def test_stock_xtb_refuses_gxtb_request(monkeypatch):
    """Reject a g-xTB request if the binary would silently use GFN2."""
    monkeypatch.setattr(xtb, "supports_gxtb", lambda: False)  # @cache'd; patch, do not call
    with pytest.raises(RuntimeError, match="does not support --gxtb"):
        _flags("gxtb")


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="Unknown method"):
        _flags("b3lyp")

    with pytest.raises(ValueError, match="Unknown method"):
        _flags(None)

    with pytest.raises(ValueError, match="Unknown method"):
        xtb.select_backend(None)


# ---------------------------------------------------------------------------
# The composite: which pairs are built, and the arithmetic that builds them
# ---------------------------------------------------------------------------


def test_solvation_dispatch():
    """g-xTB lacks ALPB parameters, so GFN2 supplies the solvation shift."""
    assert xtb._is_composite("gxtb", "h2o")
    assert not xtb._is_composite("gxtb", None), "gas-phase g-xTB runs directly"
    assert not xtb._is_composite("gfn2", "h2o"), "GFN2 has ALPB of its own"
    assert not xtb._is_composite(None, "h2o"), "an unnamed method is not a physical surface"
    assert xtb.composite_legs("h2o") == (("gxtb", None), ("gfn2", "h2o"), ("gfn2", None))


def test_stabilizing_solvent_lowers_composite():
    "Check stabilization lowers the composite and the gradient follows the solvation shift."
    base, solvated, unsolvated = -10.0, -3.0, -1.0  # GFN2 puts the solvated leg 2.0 below the gas one
    combined = xtb._combine(base, solvated, unsolvated)
    assert combined < base, f"a stabilising solvent must pull the composite DOWN, and it gave {combined}"

    gas = np.zeros((4, 3))
    shift = gas.copy()
    shift[0, 0] = 1.0  # the solvation shift pulls atom 0 along +x
    gradient = xtb._combine(gas, shift, gas)
    assert gradient[0, 0] > 0.0, f"the composite must follow the solvation shift, not oppose it: {gradient[0]}"

    # Additive, so one call serves the whole flat [energy, *gradient] array `SolvatedGXTB` packs its
    # legs into -- which is why there is one home rather than one per quantity.
    def packed(energy, forces):
        return np.concatenate([[energy], forces.ravel()])

    assert xtb._combine(packed(base, gas), packed(solvated, shift), packed(unsolvated, gas)) == pytest.approx(
        packed(combined, gradient)
    )

    # A leg that did not converge arrives as `None`, and the documented `:return:` says so propagates.
    assert xtb._combine(None, solvated, unsolvated) is None
    assert xtb._combine(base, None, unsolvated) is None
    assert xtb._combine(base, solvated, None) is None


# ---------------------------------------------------------------------------
# The registry: tblite in front, the binary behind, and an install hint when neither
# ---------------------------------------------------------------------------


def test_registry_prefers_tblite_then_binary(monkeypatch):
    "Check tblite preference and binary fallback."
    monkeypatch.setattr(xtb.TbliteBackend, "available", classmethod(lambda cls: True))
    monkeypatch.setattr(xtb.XtbBackend, "available", classmethod(lambda cls: True))
    assert xtb.select_backend("gfn2") is xtb.TbliteBackend
    assert xtb.select_backend("gxtb") is xtb.XtbBackend, "tblite ships no g-xTB"

    monkeypatch.setattr(xtb.TbliteBackend, "available", classmethod(lambda cls: False))
    assert xtb.select_backend("gfn2") is xtb.XtbBackend


def test_missing_backend_shows_install_hint(monkeypatch):
    monkeypatch.setattr(xtb.TbliteBackend, "available", classmethod(lambda cls: False))
    monkeypatch.setattr(xtb.XtbBackend, "available", classmethod(lambda cls: False))
    with pytest.raises(RuntimeError, match="XTB_EXE"):
        xtb.select_backend("gfn2")


# ---------------------------------------------------------------------------
# The warm-restart chain, on a fake tblite
# ---------------------------------------------------------------------------


def test_runner_chains_warm_density(monkeypatch):
    "Check warm restart chaining and cold starts for fresh runners."
    engine = _install_fake_tblite(monkeypatch)
    runner = xtb.TbliteBackend.make()

    runner.run(_atoms([[0.0, 0.0, 0.0]]), "gfn2", None, gradient=True)
    runner.run(_atoms([[0.1, 0.0, 0.0]]), "gfn2", None, gradient=True)
    runner.run(_atoms([[0.2, 0.0, 0.0]]), "gfn2", None, gradient=True)

    seen = [res for _calc, res, _copy in engine.calls]
    assert seen[0] is None, "the first call on a fresh runner is cold"
    assert seen[1] is engine.returned[0], "the second call warm-starts from the first result"
    assert seen[2] is engine.returned[1], "the third from the second -- a chain, not one anchor re-used"
    assert runner.density is engine.returned[2], ".density is the last link in the chain"

    fresh = xtb.TbliteBackend.make()
    fresh.run(_atoms([[0.2, 0.0, 0.0]]), "gfn2", None, gradient=True)
    assert engine.calls[-1][1] is None, "a fresh runner is cold, whatever ran before it"


def test_tblite_cold_restart(monkeypatch):
    """The shared backend contract makes a cold composite leg independent of runner history."""
    engine = _install_fake_tblite(monkeypatch)
    runner = xtb.TbliteBackend.make()

    runner.run(_atoms([[0.0, 0.0, 0.0]]), "gfn2", None, gradient=True)
    first = runner.density
    runner.run(_atoms([[0.1, 0.0, 0.0]]), "gfn2", None, gradient=True, restart=False)

    assert first is not None
    assert engine.calls[-1][1] is None, "restart=False still consumed the prior density"
    cold = runner.density
    runner.run(_atoms([[0.2, 0.0, 0.0]]), "gfn2", None, gradient=True)
    assert engine.calls[-1][1] is cold, "the cold result may seed a later explicitly warm call"


def test_warm_seed_resets_on_surface_change(monkeypatch):
    "Check explicit warm seeds and resets after a surface change."
    engine = _install_fake_tblite(monkeypatch)
    upstream = xtb.TbliteBackend.make()
    upstream.run(_atoms([[0.3, 0.0, 0.0]]), "gfn2", None, gradient=True)
    seed = upstream.density
    assert seed is not None

    runner = xtb.TbliteBackend.make()
    runner.warm_from(seed)
    runner.run(_atoms([[0.3, 0.0, 0.0]]), "gfn2", None, gradient=True)
    assert engine.calls[-1][1] is seed, "warm_from must reach the next run's starting guess"

    runner.run(_atoms([[0.3, 0.0, 0.0]]), "gfn2", "water", gradient=True)
    assert engine.calls[-1][1] is None, "a new (method, solvent) is a new surface: the chain resets to cold"
    assert engine.built[-1].solvation == "water", "and the rebuilt calculator carries the ALPB request"


def test_runner_returns_hartree_and_gradient(monkeypatch):
    """The runner contract, from the in-process side: (energy, +dE/dx) or (energy, None)."""
    _install_fake_tblite(monkeypatch)
    runner = xtb.TbliteBackend.make()

    energy, gradient = runner.run(_atoms([[1.0, 2.0, 2.0]]), "gfn2", None, gradient=True)
    assert energy == pytest.approx(0.5 * (1.0**2 + 2.0**2 + 2.0**2) / xtb.BOHR**2)
    assert np.allclose(gradient, np.array([[1.0, 2.0, 2.0]]) / xtb.BOHR), "positions in Bohr: the unit stops here"

    assert runner.run(_atoms([[1.0, 0.0, 0.0]]), "gfn2", None, gradient=False)[1] is None
    assert runner.n_calls == 0, "in-process: no subprocess was spawned"
    assert runner.path is None, "and no scratch directory to spawn one in"


# ---------------------------------------------------------------------------
def test_temperature_failure_preserves_warm_seed(monkeypatch):
    """One hot optimiser point would be a different surface wearing a successful return."""
    engine = _install_fake_tblite(monkeypatch)
    runner = xtb.TbliteBackend.make(charge=-1)
    runner.run(_atoms([[0.0, 0.0, 0.0]]), "gfn2", None, gradient=True)
    prior = runner.density

    def fail_at_default_temperature(positions, temperature):
        del positions
        return temperature is None

    engine.fails = fail_at_default_temperature

    with pytest.raises(RuntimeError, match=r"default electronic temperature.*Serial Fermi-smearing"):
        runner.run(_atoms([[1.0, 0.0, 0.0]]), "gfn2", None, gradient=True)
    assert runner.density is prior, "a failed point changed the next run's warm seed"
    assert all("temperature" not in calc.settings for calc in engine.built), "the serial path still ran hot"
