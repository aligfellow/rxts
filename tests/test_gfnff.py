"GFN-FF parsing, units, and backend selection."

import numpy as np
import pytest
from rdkit import Chem

from rxts import gfnff


def test_set_gfnff_adds_checkout_to_path(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))
    gfnff.set_gfnff(tmp_path)
    assert str(tmp_path) in sys.path
    gfnff.set_gfnff(tmp_path)
    assert sys.path.count(str(tmp_path)) == 1, "calling it twice must not grow the path"


def _parser(tmp_path, engrad, n_atoms=3):
    """A `_SubprocessGFNFF` that owns `engrad` and nothing else -- no xtb, no topology, no molecule."""
    (tmp_path / "in.engrad").write_text(engrad)
    ff = gfnff._SubprocessGFNFF.__new__(gfnff._SubprocessGFNFF)
    ff.n_atoms, ff.path = n_atoms, tmp_path
    return ff


def test_overflowed_gradient_is_rejected(tmp_path):
    "An overflowed gradient must raise with its unusable token identified."
    ff = _parser(
        tmp_path,
        "\n".join(["#", "   3", "#", "  -1.5", "#"] + [" ********************"] * 3 + ["  0.001"] * 6),
    )

    energy, gradient = gfnff._SubprocessGFNFF._parse_gradient(ff, -1.5)
    assert energy == gfnff.UNUSABLE, "an unusable geometry must cost more than any real one"
    assert np.allclose(gradient, 0.0), "and must not push the optimiser in any direction"


def test_short_gradient_reports_expected_count(tmp_path):
    ff = _parser(tmp_path, "\n".join(["#", "   3", "#", "  -1.5", "#"] + ["  0.001"] * 8))

    with pytest.raises(RuntimeError, match="expected 9"):
        gfnff._SubprocessGFNFF._parse_gradient(ff, -1.5)


def test_gradient_units_are_kcal_per_angstrom(tmp_path):
    "Check the parser's Hartree/Bohr to kcal/mol/angstrom conversion."
    ff = _parser(tmp_path, "\n".join(["#", "   3", "#", "  -1.5", "#"] + [f"  {v}" for v in np.arange(9) / 1000]))

    energy, gradient = gfnff._SubprocessGFNFF._parse_gradient(ff, -1.5)
    assert energy == -1.5, "the energy is the caller's own, not re-read from the file"
    assert gradient.shape == (3, 3)
    assert np.allclose(gradient, (np.arange(9) / 1000).reshape(3, 3) * (gfnff.HARTREE_TO_KCAL / gfnff.BOHR))


def test_backend_dispatch_preserves_subclasses(monkeypatch):
    "Check both backend branches and named subclass construction."
    monkeypatch.setattr(gfnff, "in_process_available", lambda: True)
    assert gfnff.GFNFF.__new__(gfnff.GFNFF).__class__ is gfnff._LinkedGFNFF

    monkeypatch.setattr(gfnff, "in_process_available", lambda: False)
    assert gfnff.GFNFF.__new__(gfnff.GFNFF).__class__ is gfnff._SubprocessGFNFF

    assert gfnff.GFNFF.__new__(gfnff._LinkedGFNFF).__class__ is gfnff._LinkedGFNFF, "a named subclass is honoured"


def test_linked_ff_uses_formal_charges(monkeypatch):
    """Native EEQ needs the graph's fragment charges, not just the net molecular charge."""
    from types import SimpleNamespace

    molecule = Chem.MolFromSmiles("[Na+].[Cl-]")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    molecule.AddConformer(conformer)
    captured = {}

    class Calculator:
        def __init__(self, numbers, positions, **kwargs):
            captured["numbers"] = numbers.copy()
            captured["positions"] = positions.copy()
            captured["kwargs"] = kwargs

    monkeypatch.setattr(gfnff, "import_module", lambda _: SimpleNamespace(GFNFFCalculator=Calculator))
    gfnff._LinkedGFNFF(molecule, charge=0)

    charges = captured["kwargs"]["ref_charges"]
    assert charges.dtype == np.float64
    assert charges.tolist() == [1.0, -1.0]


def test_subprocess_rejects_fragment_charge(monkeypatch):
    """The CLI exposes net charge only; do not let it guess charge placement across fragments."""
    molecule = Chem.MolFromSmiles("[Na+].[Cl-]")
    monkeypatch.setattr(gfnff, "xtb_exe", lambda: pytest.fail("reject before looking for xtb"))

    with pytest.raises(RuntimeError, match="cannot set fragment charges"):
        gfnff._SubprocessGFNFF(molecule, charge=0)


@pytest.mark.parametrize("backend", [gfnff._LinkedGFNFF, gfnff._SubprocessGFNFF])
def test_graph_and_declared_charge_must_agree(backend):
    with pytest.raises(ValueError, match="formal charge -1"):
        backend(Chem.MolFromSmiles("[Cl-]"), charge=0)


def test_subprocess_neutral_fragments(monkeypatch):
    """Neutral disconnected species need no unsupported charge placement."""
    molecule = Chem.MolFromSmiles("O.O")
    monkeypatch.setattr(gfnff, "xtb_exe", lambda: None)

    with pytest.raises(RuntimeError, match="xtb binary"):
        gfnff._SubprocessGFNFF(molecule, charge=0)
