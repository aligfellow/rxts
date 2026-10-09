from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import rxts
from rxts import structure as scan
from rxts import xtb
from rxts.structure import Geometry


def _mol(smiles="CO", seed=1):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _geometry(method="mmff", **kw):
    fields = {"forming_bonds": [(1, 5)], "breaking_bonds": [(0, 1)], "origin": "made by a test"} | kw
    return Geometry(mol=_mol(), method=method, **fields)


# -- method and FF path metadata --------------------------------------------


def test_geometry_requires_method():
    with pytest.raises(TypeError):
        Geometry(mol=_mol())


def test_ff_guess_keeps_surface_metadata():
    tsg = rxts.TransitionStateGuesser(engine="auto", multiplicity=1, n_conformers=1, n_embed=0)
    tsg.n_lambda = 5
    path = tsg.find_transition_state(["C[Br]", "[Cl-]"], ["C[Cl]", "[Br-]"])
    path.metal_angle_k = 239.0
    path.metal_angle_reference = "endpoint_graph"
    guess = path.guess("ff")
    assert guess.method == path.engine == "mmff"
    assert path.qm_method is None

    exported = [guess.to_atoms(), path.to_atoms()[path.peak_index("ff")]]
    for atoms in exported:
        context = atoms.info["rxts"]
        assert context["metal_angle_k"] == path.metal_angle_k
        assert context["metal_angle_reference"] == path.metal_angle_reference
        assert context["charge"] == path.charge
        assert context["multiplicity"] == path.multiplicity
        assert context["mapping"] == path.mapping
        assert atoms.info["charge"] == path.charge
        assert atoms.info["multiplicity"] == path.multiplicity
        np.testing.assert_array_equal(atoms.arrays["atom_maps"], [a.GetAtomMapNum() for a in path.mol.GetAtoms()])
    assert path.conformer_selections == []


# -- what dump() is for -----------------------------------------------------


def test_xyz_candidate_context(tmp_path):
    g = _geometry(method="gfn2", origin="rxts TS guess, highest MMFF94 point, lambda=0.55")
    comment = open(g.dump(tmp_path / "guess.xyz")).read().splitlines()[1]
    assert comment.strip() == "rxts TS guess, highest MMFF94 point, lambda=0.55 charge=0 multiplicity=1"


def test_isotope_export():
    source = _geometry()
    source.mol.GetAtomWithIdx(0).SetIsotope(13)
    exported = source.to_atoms()
    assert exported.info["rxts"]["isotopes"][0] == 13
    assert exported.get_masses()[0] == source.mol.GetAtomWithIdx(0).GetMass()
    np.testing.assert_array_equal(exported.positions, source.positions)


@pytest.fixture(scope="module")
def sn2():
    tsg = rxts.TransitionStateGuesser(engine="auto", multiplicity=1, n_conformers=1, n_embed=0)
    tsg.n_lambda = 11
    return tsg.find_transition_state(["C[Br]", "[Cl-]"], ["C[Cl]", "[Br-]"])


def test_path_endpoints_and_charge(sn2):
    lambdas = [f.lam for f in sn2.frames]
    assert lambdas[0] == 0.0
    assert lambdas[-1] == 1.0
    assert len(sn2.frames) == 11
    assert sn2.charge == -1  # the chloride's charge must survive


def test_guess_is_interior_path_maximum(sn2):
    i = sn2.peak_index("ff")
    assert sn2.ff_energies[i] == pytest.approx(max(sn2.ff_energies))
    assert 0 < i < len(sn2.frames) - 1
    assert sn2.peak_lambda("ff") == sn2.frames[i].lam


def test_reactive_bonds_cross_over(sn2):
    lengths = sn2.bond_lengths()
    (breaking,) = [v for (kind, _), v in lengths.items() if kind == "breaking"]
    (forming,) = [v for (kind, _), v in lengths.items() if kind == "forming"]

    assert breaking[-1] > breaking[0] + 1.0  # C-Br pulls apart
    assert forming[-1] < forming[0] - 1.0  # C-Cl closes up
    assert np.all(np.diff(breaking) > -1e-6)  # monotonic: no doubling back
    assert np.all(np.diff(forming) < 1e-6)

    # At the guess, neither bond is made or broken -- both are half-formed.
    i = sn2.peak_index("ff")
    assert breaking[0] < breaking[i] < breaking[-1]
    assert forming[-1] < forming[i] < forming[0]


def test_energy_is_referenced_to_the_reactants(sn2):
    assert sn2.ff_energies[0] == pytest.approx(0.0)
    assert sn2.barrier("ff") > 0.0


def test_guess_and_path_dump_separately(sn2, tmp_path):
    ts = sn2.guess("ff").dump(tmp_path / "ts.xyz")
    trajectory = sn2.dump(tmp_path / "path.xyz")
    n_atoms = sn2.reactant.GetNumAtoms()

    assert Path(ts).read_text().splitlines()[0].strip() == str(n_atoms)
    # Every frame writes one XYZ block, so the trajectory holds the whole path.
    blocks = Path(trajectory).read_text().count(f"\n{n_atoms}\n") + 1
    assert blocks == len(sn2.frames)
    assert f"charge={sn2.charge} multiplicity={sn2.multiplicity}" in Path(ts).read_text().splitlines()[1]
    comments = Path(trajectory).read_text().splitlines()[1 :: n_atoms + 2]
    assert all(f"charge={sn2.charge} multiplicity={sn2.multiplicity}" in comment for comment in comments)


# -- rule (a): a name means one thing ---------------------------------------


def test_ff_and_qm_surfaces_are_separate(sn2):
    assert not hasattr(sn2, "energies"), "`energies` is the bug; it must not exist in any form"
    assert not hasattr(sn2, "settled"), "the second flag added to patch the first one"

    assert len(sn2.ff_energies) == len(sn2.frames)  # always the force field, scored or not
    assert sn2.qm_energies is None  # nobody has scored this path
    assert sn2.scored is False  # derived from qm_energies, not a stored flag


def test_qm_queries_require_scoring(sn2):
    for ask in (lambda: sn2.guess("qm"), lambda: sn2.barrier("qm"), lambda: sn2.reaction_energy("qm")):
        with pytest.raises(ValueError, match="score"):
            ask()


def test_guess_requires_a_known_surface(sn2):
    with pytest.raises(TypeError):
        sn2.guess()
    with pytest.raises(ValueError, match='"ff" or "qm"'):
        sn2.guess("force field")


def test_guess_is_a_copy(sn2):
    before = sn2.ff_energies.copy()
    positions = [f.positions.copy() for f in sn2.frames]

    guess = sn2.guess("ff")
    guess.mol.GetConformer().SetAtomPosition(0, (99.0, 99.0, 99.0))  # scribble on what we got back

    assert np.array_equal(sn2.ff_energies, before)
    assert all(np.array_equal(f.positions, p) for f, p in zip(sn2.frames, positions, strict=True))
    assert sn2.qm_energies is None


def test_tblite_scores_without_binary(sn2, monkeypatch, caplog):
    path = copy.deepcopy(sn2)
    monkeypatch.setattr(scan, "xtb_available", lambda: False)
    monkeypatch.setattr(xtb, "xtb_available", lambda: False)

    with caplog.at_level(logging.INFO, logger="rxts"):
        path.score(method="gfn2")

    assert path.scored is True
    assert path.qm_method == "gfn2"
    assert len(path.qm_energies) == len(path.frames)
    # The progress narration is on the logger now, not stdout: the library does not own stdout.
    assert "Re-scoring" in caplog.text


def test_missing_xtb_error_names_install_hint(sn2, monkeypatch, caplog):
    path = copy.deepcopy(sn2)
    monkeypatch.setattr(scan, "xtb_available", lambda: False)

    with caplog.at_level(logging.WARNING, logger="rxts"), pytest.warns(RuntimeWarning, match="NOT scored"):
        path.score(method="gfnff")
    assert path.scored is False
    assert "XTB_EXE" in caplog.text  # the actionable hint, not just "not found"


@pytest.mark.skipif(not xtb.supports_gxtb(), reason="needs a g-xTB binary")
def test_scoring_selects_qm_guess_without_changing_ff():
    tsg = rxts.TransitionStateGuesser(engine="auto", multiplicity=1, n_conformers=1, n_embed=0)
    tsg.n_lambda = 11
    path = tsg.find_transition_state("[I-].CCl>>[Cl-].CI")

    ff_before = path.ff_energies.copy()
    ff_guess_before = path.guess("ff").positions.copy()

    path.score(method="gxtb", n_cores=1)

    assert np.array_equal(path.ff_energies, ff_before)
    assert np.array_equal(path.guess("ff").positions, ff_guess_before)

    assert path.scored is True
    assert path.qm_energies is not None
    assert len(path.qm_energies) == len(path.frames)

    peak = int(np.argmax(path.qm_energies))
    assert path.peak_index("qm") == peak
    np.testing.assert_array_equal(path.guess("qm").positions, path.frames[peak].positions)

    # Each geometry names the surface that picked it, and says so in the file it writes.
    assert path.guess("ff").method == path.engine
    assert path.guess("qm").method == path.qm_method
    assert path.guess("ff").method != path.guess("qm").method


@pytest.mark.parametrize(("stiffness", "combined"), [(1.0, True), (3.0, False)])
def test_external_frame_scoring(sn2, monkeypatch, stiffness, combined):
    from ase.calculators.calculator import Calculator, all_changes

    from rxts import CalculatorMethod

    class Quadratic(Calculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            x = atoms.get_positions()
            values = {"energy": stiffness * np.sum(x * x) / 2, "forces": -stiffness * x}
            self.results = values if combined else {name: values[name] for name in properties}

    made, contexts = [], []

    def factory(**context):
        contexts.append(context)
        calc = Quadratic()
        made.append(calc)
        return calc

    def no_xtb(*args, **kwargs):
        raise AssertionError("an external method reached xtb dispatch")

    monkeypatch.setattr(scan, "xtb_available", no_xtb)
    path = copy.deepcopy(sn2)
    original = [frame.positions.copy() for frame in path.frames]
    method = CalculatorMethod(f"quadratic k={stiffness}", factory)
    attempt = path.score(method, solvent="water", n_cores=2)
    expected = [stiffness * np.sum(x * x) / (2 * xtb.EV_PER_KCAL) for x in original]
    assert attempt.failures == {}
    assert attempt.method == method.label
    assert (attempt.charge, attempt.multiplicity, attempt.solvent) == (path.charge, path.multiplicity, "water")
    np.testing.assert_allclose(attempt.energies_ev, np.array(expected) * xtb.EV_PER_KCAL)
    np.testing.assert_allclose([frame.qm_energy for frame in path.frames], expected)
    for before, frame in zip(original, path.frames, strict=True):
        np.testing.assert_array_equal(frame.positions, before)
    exported = path.to_atoms()
    assert len(exported) == len(path.frames)
    assert all(image.calc is None for image in exported)
    for image, positions in zip(exported, original, strict=True):
        np.testing.assert_array_equal(image.positions, positions)
        assert image.info["rxts"]["mapping"] == path.mapping
        assert image.info["rxts"]["qm_method"] == method.label
        assert image.info["rxts"]["qm_solvent"] == "water"
    guess = path.guess("qm")
    assert guess.method is method
    assert guess.label == method.label
    assert guess.solvent == "water"
    assert len(made) == len(path.frames)
    assert len({id(calc) for calc in made}) == len(made)
    assert all(c["charge"] == path.charge and c["multiplicity"] == path.multiplicity for c in contexts)
    assert all(c["solvent"] == "water" for c in contexts)


def test_instance_scores_serially_and_keeps_frames(sn2):
    from ase.calculators.calculator import Calculator, all_changes

    seen = []

    class Surface(Calculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            seen.append(copy.deepcopy(self.atoms.info["rxts"]))
            self.results = {"energy": float(np.sum(self.atoms.positions**2)), "forces": -2 * self.atoms.positions}

    path = copy.deepcopy(sn2)
    path.n_cores = 6
    method = Surface()
    coordinates = [frame.positions.copy() for frame in path.frames]
    path.score(method=method)
    assert isinstance(path.qm_method, rxts.CalculatorMethod)
    assert path.qm_method.label == method.name
    assert path.scored
    assert [context["frame_index"] for context in seen] == list(range(len(path.frames)))
    for context in seen:
        assert context["charge"] == path.charge
        assert context["multiplicity"] == path.multiplicity
        assert context["mapping"] == path.mapping
        assert context["forming_bonds"] == path.forming_bonds
        assert context["breaking_bonds"] == path.breaking_bonds
    for frame, before in zip(path.frames, coordinates, strict=True):
        np.testing.assert_array_equal(frame.positions, before)
    with pytest.raises(ValueError, match="serial"):
        path.score(method=method, n_cores=2)


def test_failed_rescore_keeps_prior_profile(sn2):
    from ase.calculators.calculator import Calculator, all_changes

    class Surface(Calculator):
        implemented_properties = ("energy",)

        def __init__(self, fail=False):
            super().__init__()
            self.fail = fail

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if self.fail:
                raise RuntimeError("analytic provider lost its state")
            self.results = {"energy": float(np.sum(atoms.positions**2))}

    path = copy.deepcopy(sn2)
    previous = rxts.CalculatorMethod("first physical surface", calculator=Surface())
    path.score(previous)
    energies = path.qm_energies.copy()
    failed = rxts.CalculatorMethod("failed physical surface", calculator=Surface(fail=True))
    with pytest.warns(RuntimeWarning, match="earlier scores are retained"):
        attempt = path.score(failed)
    assert attempt.method == failed.label
    assert attempt.energies_ev == (None,) * len(path.frames)
    assert set(attempt.failures) == set(range(len(path.frames)))
    assert all("lost its state" in reason for reason in attempt.failures.values())
    assert path.qm_method is previous
    np.testing.assert_array_equal(path.qm_energies, energies)
