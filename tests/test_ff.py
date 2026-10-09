"Force-field path construction and sampling."

from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import rxembed
from rdkit import Chem

import rxts
from rxts import builder, ff
from rxts.builder import _geometry_known, as_molecules
from rxts.ff import conformers, embed_candidates, metal_core
from rxts.utils import PLACED, mol_from_xyz, set_positions


class _StopError(Exception):
    """Sentinel: stop the guesser at the call under test rather than pay for a scan."""


@pytest.mark.parametrize("invalid_role", ["reactant", "product"])
def test_gfnff_validates_topology_before_build(monkeypatch, invalid_role):
    from rxts.utils import mol_from_xyz

    geometry = """5
Pd complex
Pd  0.000  0.000  0.000
N   2.000  0.000  0.000
N  -2.000  0.000  0.000
Cl  0.000  2.300  0.000
Cl  0.000 -2.300  0.000
"""
    valid = mol_from_xyz(geometry)
    positions = valid.GetConformer().GetPositions()
    positions[1] = [10.0, 0.0, 0.0]
    invalid = set_positions(valid, positions)
    reactant, product = (invalid, valid) if invalid_role == "reactant" else (valid, invalid)

    monkeypatch.setattr(
        ff,
        "GFNFFPotential",
        lambda **kwargs: pytest.fail("invalid endpoint reached GFN-FF field construction"),
    )
    guesser = rxts.TransitionStateGuesser(engine="gfnff", multiplicity=1)
    with pytest.raises(ValueError, match="does not have the bonding it should"):
        guesser._build_potential("gfnff", reactant, product, [(0, 1)], [], charge=0)


def test_native_endpoints_are_prepared_once(monkeypatch):
    rx = rxembed
    from rdkit import Chem

    guesser = rxts.TransitionStateGuesser(engine="mmff", multiplicity=1, n_lambda=3, n_cores=1)
    reference = guesser.prepare_endpoints("CBr.[Cl-]>>CCl.[Br-]")
    native = tuple(
        rx.Conformers(Chem.Mol(mol), [mol.GetConformer().GetId()]) for mol in (reference.reactant, reference.product)
    )
    guesser.mapping = list(range(reference.reactant.GetNumAtoms()))

    def forbidden(*args, **kwargs):
        pytest.fail("selected endpoints were remapped or rebuilt")

    monkeypatch.setattr(builder, "search_approach", forbidden)
    monkeypatch.setattr(builder, "reactive_complex", forbidden)
    monkeypatch.setattr(builder, "derive_product_geometry", forbidden)
    prepared = guesser.prepare_endpoints(*native)
    assert isinstance(prepared, rxts.PreparedReaction)
    assert prepared.sources[0] is native[0]
    assert prepared.sources[1] is native[1]
    assert prepared.forming_bonds == reference.forming_bonds
    assert prepared.breaking_bonds == reference.breaking_bonds
    before = [mol.GetConformer().GetPositions().copy() for mol in (prepared.reactant, prepared.product)]
    monkeypatch.setattr(guesser, "prepare_endpoints", forbidden)
    build_potential = ff.TransitionStateGuesser._build_potential

    def capture_build(self, **kwargs):
        assert self.solvent == prepared.solvent
        assert self.n_cores == prepared.n_cores
        assert self.multiplicity == prepared.multiplicity
        return build_potential(self, **kwargs)

    guesser.solvent, guesser.n_cores, guesser.multiplicity = "water", 2, 2
    monkeypatch.setattr(ff.TransitionStateGuesser, "_build_potential", capture_build)
    path = rxts.ff.interpolate(prepared, engine="mmff", n_lambda=3, n_conformers=1, n_embed=0)
    assert len(path.frames) == 3
    assert path.mapping == prepared.mapping
    assert path.solvent == prepared.solvent
    assert path.n_cores == prepared.n_cores
    assert path.metal_angle_k == pytest.approx(1000.0 / 4.184)
    assert path.metal_angle_reference is None
    for mol, coords in zip((prepared.reactant, prepared.product), before, strict=True):
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(), coords)
    with pytest.raises(ValueError, match="omit product"):
        guesser.find_transition_state(prepared, native[1])
    with pytest.raises(ValueError, match="charge is fixed by prepare"):
        rxts.ff.interpolate(prepared, engine="mmff", n_lambda=3, charge=1)


def test_reaction_smiles_scan():
    tsg = rxts.TransitionStateGuesser(engine="auto", multiplicity=1, n_conformers=1, n_embed=0)
    tsg.n_lambda = 5
    path = tsg.find_transition_state("C[Br].[Cl-]>>C[Cl].[Br-]")
    np.testing.assert_array_equal([frame.lam for frame in path.frames], np.linspace(0.0, 1.0, 5))
    assert path.charge == -1
    assert len(path.breaking_bonds) == 1
    assert len(path.forming_bonds) == 1


@pytest.mark.parametrize("n_conformers", [1, 4])
def test_md_budget_includes_starting_geometry(monkeypatch, n_conformers):
    guesser = rxts.TransitionStateGuesser(n_lambda=3, n_conformers=n_conformers, n_embed=0, md_steps=1)
    mol = Chem.MolFromXYZBlock("2\n\nH 0 0 0\nH 0.7 0 0\n")
    calls = []

    def sample(**kwargs):
        calls.append(kwargs["lam"])
        return kwargs["x"]

    monkeypatch.setattr(guesser, "_best_conformer", lambda **kwargs: kwargs["x"])
    monkeypatch.setattr(guesser, "_minimize", lambda pot, x, lam: x)
    monkeypatch.setattr(guesser, "_sample", sample)
    pot = SimpleNamespace(changing=[], energy=lambda x, lam: (0.0, 0.0, 0.0))
    assert len(guesser._scan(pot, mol, charge=0)) == 3
    assert len(calls) == guesser.n_lambda * (n_conformers - 1)


def test_mmff_rejects_missing_parameters():
    tsg = rxts.TransitionStateGuesser(engine="mmff", multiplicity=1)
    with pytest.raises(ValueError, match=r"MMFF94 cannot assign parameters"):
        tsg.find_transition_state("CCCCCl.[I-]>>CCCCI.[Cl-]")


def test_auto_engine_selection():
    from rxts.builder import as_molecules, combine_spaced

    tsg = rxts.TransitionStateGuesser(engine="auto", multiplicity=1)

    iodide = combine_spaced(as_molecules(["CCCCCl", "[I-]"]))
    assert tsg._choose_engine(iodide, iodide) == "gfnff"

    organic = combine_spaced(as_molecules(["CC(=O)OCC", "O"]))
    assert tsg._choose_engine(organic, organic) == "mmff"


def test_unknown_engine():
    tsg = rxts.TransitionStateGuesser(engine="auto", multiplicity=1)
    tsg.engine = "uff"
    with pytest.raises(ValueError, match="Unknown engine"):
        tsg.find_transition_state("C[Br].[Cl-]>>C[Cl].[Br-]")


def test_placed_endpoints_skip_derivation(monkeypatch):

    def _forbidden(*a, **k):
        raise AssertionError("the embed-and-derive path must not run for placed endpoints")

    monkeypatch.setattr(builder, "search_approach", _forbidden)
    monkeypatch.setattr(builder, "reactive_complex", _forbidden)
    monkeypatch.setattr(builder, "derive_product_geometry", _forbidden)

    class _ReachedError(Exception):
        pass

    def _stop(self, pot, rea, charge):
        raise _ReachedError

    monkeypatch.setattr(ff.TransitionStateGuesser, "_choose_engine", lambda self, rea, pro: "mmff")
    monkeypatch.setattr(ff.TransitionStateGuesser, "_build_potential", lambda self, **k: None)
    monkeypatch.setattr(ff.TransitionStateGuesser, "_scan", _stop)

    # H2 + hydride -> hydride + H2: H0-H1 bonded then apart, H1-H2 apart then bonded.
    def _placed(coords):
        block = "3\n\n" + "\n".join(f"H {x} {y} {z}" for x, y, z in coords)
        mol = mol_from_xyz(block, charge=-1)
        mol.SetBoolProp(PLACED, True)
        return mol

    rea = _placed([(0.0, 0, 0), (0.74, 0, 0), (2.6, 0, 0)])
    pro = _placed([(0.0, 0, 0), (2.0, 0, 0), (2.74, 0, 0)])
    assert _geometry_known(rea) is True, "the PLACED mark is what puts the run on this branch at all"

    with pytest.raises(_ReachedError):
        ff.TransitionStateGuesser(engine="auto", multiplicity=1, charge=-1).find_transition_state(rea, pro)


def test_guesser_passes_named_bonds_to_mapper(monkeypatch):
    seen = []

    def spy(rea, pro, breaking_bonds=None, forming_bonds=None):
        seen.append((breaking_bonds, forming_bonds))
        raise _StopError

    monkeypatch.setattr(rxts.mapping, "map_reaction", spy)
    guesser = rxts.TransitionStateGuesser(
        engine="auto", multiplicity=1, charge=0, forming_bonds=[(0, 5)], breaking_bonds=[(0, 4)]
    )

    reactant = (
        "6\nmethyl bromide + chloride\nC 0 0 0\nH -0.6 0.8 0\nH -0.6 -0.4 0.7\nH -0.6 -0.4 -0.7\nBr 1.9 0 0\nCl 5 0 0\n"
    )
    product = (
        "6\nmethyl chloride + bromide\nC 0 0 0\nH -0.6 0.8 0\nH -0.6 -0.4 0.7\nH -0.6 -0.4 -0.7\nBr 5 0 0\nCl 1.8 0 0\n"
    )
    with pytest.raises(_StopError):
        guesser.find_transition_state(reactant, product)
    guesser.charge = -1
    guesser.forming_bonds = [(0, 5)]
    guesser.breaking_bonds = [(0, 1)]
    with pytest.raises(_StopError):
        guesser.find_transition_state("C[Br].[Cl-]>>C[Cl].[Br-]")

    assert seen == [([(0, 4)], [(0, 5)]), ([(0, 1)], [(0, 5)])]


def test_peak_conformer_is_held_across_path(monkeypatch):
    from rxts.ff import TransitionStateGuesser
    from rxts.structure import Frame

    tsg = TransitionStateGuesser(engine="auto", multiplicity=1)
    tsg.n_embed, tsg.embed_window = 1, 0

    lams = np.linspace(0.0, 1.0, 7)
    energies = -np.abs(lams - 0.5)  # a tent: the peak (max energy) is the middle frame
    # An untouched frame retains the sentinel, including either endpoint.
    frames = [
        Frame(lam=float(lam), positions=np.full((1, 3), 9.0), energy=float(e), e_reactant=0.0, e_product=0.0)
        for lam, e in zip(lams, energies, strict=True)
    ]

    def ramp(lam):  # the true FF minimum at each lambda: a single continuous line, never the sentinel
        return np.array([[lam, 0.0, 0.0]])

    monkeypatch.setattr(tsg, "_minimize", lambda pot, x, lam: ramp(lam))
    monkeypatch.setattr(tsg, "_pick_conformer", lambda pot, rea, candidates, lam, charge: ramp(lam))
    monkeypatch.setattr(ff, "embed_candidates", lambda *a, **k: [])

    class _Pot:
        def energy(self, x, lam):
            return 0.0, 0.0, 0.0

        def targets(self, lam):
            return None

    tsg._anchor_on_peak(_Pot(), None, frames, charge=0)

    for frame in frames:
        assert np.allclose(frame.positions, ramp(frame.lam), atol=1e-9), (
            f"frame at lam={frame.lam:.3f} is off the anchored ramp -- the hold missed an endpoint"
        )


def test_default_peak_search_reembeds_and_anchors(monkeypatch):
    from rxts.structure import Frame

    guesser = rxts.TransitionStateGuesser()
    calls, anchored = [], []
    monkeypatch.setattr(ff, "embed_candidates", lambda **kwargs: calls.append(kwargs) or [kwargs["x"]])
    monkeypatch.setattr(guesser, "_minimize", lambda pot, x, lam: np.asarray(x))
    monkeypatch.setattr(guesser, "_pick_conformer", lambda pot, rea, candidates, lam, charge: candidates[0])
    monkeypatch.setattr(guesser, "_anchor_on_peak", lambda pot, rea, frames, charge: anchored.append(frames))

    class Pot:
        def energy(self, x, lam):
            return 0.0, 0.0, 0.0

        def targets(self, lam):
            return None

    frames = [
        Frame(lam=lam, positions=np.zeros((1, 3)), energy=energy, e_reactant=0.0, e_product=0.0)
        for lam, energy in ((0.0, 0.0), (0.5, 1.0), (1.0, 0.0))
    ]
    guesser._refine_guess(Pot(), object(), frames, charge=0)

    assert len(calls) == 3  # the default +/-2-window band around lambda=0.5
    assert all(call["n"] == guesser.n_embed for call in calls)
    assert anchored == [frames]  # the default peak_anchored route


_REFUSED_SETTINGS = [
    ({"n_lambda": 0}, "n_lambda"),
    ({"n_lambda": 1}, "n_lambda"),
    ({"seed": None}, "seed"),
    ({"forming_bonds": [(0, 1, 2)]}, "forming_bonds"),
    ({"breaking_bonds": [(0, 1, 2)]}, "breaking_bonds"),
    ({"mapping": [0, -1]}, "mapping"),
    ({"reactant_geometry": 3}, "reactant_geometry"),
    ({"product_geometry": ""}, "product_geometry"),
    ({"engine": "mmf"}, "engine"),
    ({"morse_d": float("inf")}, "morse_d"),
    ({"soft_core": -0.5}, "soft_core"),
    ({"constraints": {(1, 1): 1.5}}, "constraints"),
    ({"constraints": {(0, 3): -1.0}}, "constraints"),
    ({"constraint_k": -1.0}, "constraint_k"),
    ({"metal_angle_k": -1.0}, "metal_angle_k"),
    ({"metal_angle_k": float("inf")}, "metal_angle_k"),
    ({"n_conformers": 0}, "n_conformers"),
    ({"md_steps": -1}, "md_steps"),
    ({"md_temperature": 0.0}, "md_temperature"),
    ({"md_timestep": 0.0}, "md_timestep"),
    ({"n_embed": -1}, "n_embed"),
    ({"embed_window": -1}, "embed_window"),
    ({"peak_anchored": None}, "peak_anchored"),
    ({"embed_score": "gfn3"}, "embed_score"),
    ({"discont_search": "yes"}, "discont_search"),
    ({"solvent": 5}, "solvent"),
    ({"n_cores": 0}, "n_cores"),
    ({"charge": 1.5}, "charge"),
    ({"multiplicity": 0}, "multiplicity"),
]

_ACCEPTED_SETTINGS = [
    {},
    {"n_lambda": 5, "n_cores": 2},
    {"engine": "gfnff", "solvent": "water"},
    {"reactant_geometry": "SPL", "product_geometry": "square_planar"},
    {"peak_anchored": False, "discont_search": True},
    {"n_embed": 3, "embed_window": 0, "embed_score": "gfn2"},
    {"md_steps": 1000, "md_timestep": 0.5},
    {"charge": -1, "multiplicity": 3},
    {"forming_bonds": [(1, 14)], "breaking_bonds": [(1, 3)], "mapping": [0, 1, 2]},
    {"constraints": {(0, 3): 1.5}, "constraint_k": 0.0},
    {"metal_angle_k": 0.0},
    {"soft_core": 0.0},
]


def test_invalid_settings_fail_at_init():
    for setting, name in _REFUSED_SETTINGS:
        with pytest.raises(ValueError, match=name):
            rxts.TransitionStateGuesser(**setting)


def test_configuration_error_lists_bad_fields():
    with pytest.raises(ValueError, match="cannot run with these settings") as excinfo:
        rxts.TransitionStateGuesser(n_lambda=0, engine="mmf", n_cores=-4)
    message = str(excinfo.value)
    for field_name in ("n_lambda", "engine", "n_cores"):
        assert field_name in message
    assert "'mmf'" in message, "the rejected value is quoted, so the typo is visible"


def test_valid_settings():
    """Valid values, including disabled terms and opt-in samplers, remain unchanged."""
    for setting in _ACCEPTED_SETTINGS:
        guesser = rxts.TransitionStateGuesser(**setting)
        assert all(getattr(guesser, field) == value for field, value in setting.items()), (
            f"__post_init__ rewrote {setting!r}; it checks, it must not coerce"
        )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("isotope", [False, True])
def test_external_conformer_selection(reverse, isotope):
    from ase.calculators.calculator import Calculator, all_changes

    class PositionEnergy(Calculator):
        implemented_properties = ("energy",)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            assert atoms.info["charge"] == 0
            assert atoms.info["multiplicity"] == 1
            np.testing.assert_array_equal(atoms.arrays["atom_maps"], [11])
            self.results = {"energy": float(atoms.positions[0, 0])}

    contexts = []

    def factory(**context):
        contexts.append(context)
        return PositionEnergy()

    method = rxts.CalculatorMethod("candidate position", factory)
    guesser = rxts.TransitionStateGuesser(engine="auto", multiplicity=1, embed_score=method, n_cores=1, solvent="water")
    from rdkit import Chem

    mol = Chem.MolFromXYZBlock("1\n\nH 0 0 0\n")
    mol.GetAtomWithIdx(0).SetAtomMapNum(11)
    mol.GetAtomWithIdx(0).SetIsotope(2 if isotope else 0)
    candidates = [np.array([[x, 0.0, 0.0]]) for x in (1.0, 2.0)]
    if reverse:
        candidates.reverse()
    before = [x.copy() for x in candidates]
    pot = SimpleNamespace(energy=lambda x, lam: (-x[0, 0], None))
    selected = guesser._pick_conformer(pot, mol, candidates, 0.5, charge=0)
    physical_best, ff_best = (1, 0) if reverse else (0, 1)
    np.testing.assert_array_equal(selected, candidates[physical_best])
    assert len(contexts) == 2
    assert all(c["solvent"] == "water" and c["n_cores"] == 1 for c in contexts)
    selection = guesser._conformer_selections[0]
    assert selection["lambda"] == 0.5
    assert selection["selected_index"] == physical_best
    assert selection["ff_selected_index"] == ff_best
    np.testing.assert_array_equal(selection["scores"].energies_ev, [x[0, 0] for x in before])
    np.testing.assert_array_equal(selection["ff_energies_kcal_per_mol"], [-x[0, 0] for x in before])
    for index, atoms in enumerate(selection["candidates"]):
        assert atoms.calc is None
        assert atoms.info == {"charge": 0, "multiplicity": 1}
        assert atoms.arrays["atom_maps"].tolist() == [11]
        assert ("masses" in atoms.arrays) == isotope
        if isotope:
            assert atoms.get_masses()[0] == mol.GetAtomWithIdx(0).GetMass()
        np.testing.assert_array_equal(atoms.positions, before[index])
        np.testing.assert_array_equal(candidates[index], before[index])
    selected[:] += 10
    np.testing.assert_array_equal(selection["candidates"][physical_best].positions, before[physical_best])


def test_conformer_calculator_rejects_parallel_use():
    from ase.calculators.lj import LennardJones

    provider = LennardJones()
    guesser = rxts.TransitionStateGuesser(multiplicity=1, embed_score=provider, n_cores=2)
    mol = Chem.MolFromXYZBlock("1\n\nH 0 0 0\n")
    candidates = [np.zeros((1, 3))]
    pot = SimpleNamespace(energy=lambda x, lam: (0.0, None))
    with pytest.raises(ValueError, match="serial scoring"):
        guesser._pick_conformer(pot, mol, candidates, 0.5, charge=0)
    assert not provider.results


@pytest.mark.parametrize("failed_positions", [(1.0,), (1.0, 2.0)])
def test_failed_conformer_score_keeps_evidence(failed_positions):
    from ase.calculators.calculator import Calculator, all_changes

    class FailingEnergy(Calculator):
        implemented_properties = ("energy",)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            x = float(atoms.positions[0, 0])
            if x in failed_positions:
                raise RuntimeError("physical evaluation failed")
            self.results = {"energy": x}

    method = rxts.CalculatorMethod("failing candidate energy", calculator=FailingEnergy())
    guesser = rxts.TransitionStateGuesser(multiplicity=1, embed_score=method, n_cores=1)
    from rdkit import Chem

    mol = Chem.MolFromXYZBlock("1\n\nH 0 0 0\n")
    candidates = [np.array([[x, 0.0, 0.0]]) for x in (1.0, 2.0)]
    pot = SimpleNamespace(energy=lambda x, lam: (-x[0, 0], None))
    with pytest.raises(ff.ConformerScoringError, match="Physical conformer scoring failed") as caught:
        guesser._pick_conformer(pot, mol, candidates, 0.5, charge=0)
    evidence = caught.value.result
    selection = caught.value.selection
    assert selection is guesser._conformer_selections[0]
    assert selection["scores"] is evidence
    assert selection["selected_index"] is None
    assert selection["lambda"] == 0.5
    assert set(evidence.failures) == set(range(len(failed_positions)))
    assert evidence.method == method.label
    for index, coords in enumerate(candidates):
        np.testing.assert_array_equal(caught.value.candidates[index].positions, coords)
        assert caught.value.candidates[index].calc is None
        assert caught.value.candidates[index].info == {"charge": 0, "multiplicity": 1}
    if len(failed_positions) == 1:
        assert evidence.energies_ev[1] == 2.0


@pytest.mark.parametrize("termination", ["success", "score_failure", "ff_failure", "interrupt"])
def test_path_selection_evidence(monkeypatch, termination):
    from ase.calculators.calculator import Calculator, all_changes

    from rxts.structure import Frame
    from rxts.utils import with_conformer

    class StateEnergy(Calculator):
        implemented_properties = ("energy",)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            assert atoms.info["charge"] == 1
            assert atoms.info["multiplicity"] == 1
            assert atoms.arrays["atom_maps"].tolist() == [11, 29, 41]
            distance = atoms.get_distance(0, 1)
            if termination == "score_failure" and distance < 1.5:
                raise ValueError("candidate failed")
            self.results = {"energy": (distance - 1.7) ** 2}

    mol = with_conformer(Chem.MolFromSmiles("[H:11][H:29].[H+:41]"), np.zeros((3, 3)))
    candidates = [np.array([[r, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.2, 0.0]]) for r in (1.4, 1.7)]
    prepared = rxts.PreparedReaction(
        reactant=mol,
        product=Chem.Mol(mol),
        forming_bonds=[],
        breaking_bonds=[(0, 1)],
        mapping=[0, 1, 2],
        charge=1,
        multiplicity=1,
        solvent=None,
        n_cores=1,
        sources=(mol, mol),
    )
    pot = SimpleNamespace(
        energy=lambda x, lam: ((np.linalg.norm(x[0] - x[1]) - 1.4) ** 2, None),
        metal_angle_k=0.0,
        metal_angle_source=None,
    )

    def scan(self, pot, rea, charge):
        self._sampling_failures.append({"lambda": 0.5, "reason": "rejected proposal"})
        chosen = self._pick_conformer(pot, rea, candidates, 0.5, charge)
        if termination == "ff_failure":
            raise ff._FFRelaxationError(0.75, chosen, 7.0, np.ones_like(chosen), "controlled failure")
        if termination == "interrupt":
            raise KeyboardInterrupt("controlled interruption")
        return [Frame(i / 2, chosen.copy(), 0.0, 0.0, 0.0) for i in range(3)]

    monkeypatch.setattr(ff.TransitionStateGuesser, "_build_potential", lambda *args, **kwargs: pot)
    monkeypatch.setattr(ff.TransitionStateGuesser, "_scan", scan)
    provider = StateEnergy()
    guesser = rxts.TransitionStateGuesser(engine="gfnff", n_lambda=3, n_embed=1, embed_score=provider)
    if termination == "success":
        path = guesser.find_transition_state(prepared)
        failures = path.sampling_failures
        selection = path.conformer_selections[0]
        result = path.score(method=provider)
        assert not result.failures
        assert result.n_evaluations == len(path.frames)
        np.testing.assert_array_equal(result.energies_ev, np.zeros(3))
        assert path.guess("qm").to_atoms().info["charge"] == 1
    else:
        failure = {
            "score_failure": ff.ConformerScoringError,
            "ff_failure": ff._FFRelaxationError,
            "interrupt": KeyboardInterrupt,
        }[termination]
        with pytest.raises(failure) as caught:
            guesser.find_transition_state(prepared)
        failures = caught.value.sampling_failures
        selection = caught.value.conformer_selections[0]
        if termination == "ff_failure":
            assert caught.value.lam == 0.75
            assert caught.value.energy_ev == 7.0
        if termination == "score_failure":
            assert selection["scores"] is caught.value.result
    assert failures == [{"lambda": 0.5, "reason": "rejected proposal"}]
    assert selection["lambda"] == 0.5
    assert selection["scores"].n_evaluations == 2
    assert selection["selected_index"] == (None if termination == "score_failure" else 1)
    assert selection["ff_selected_index"] == 0
    for atoms, original in zip(selection["candidates"], candidates, strict=True):
        assert atoms.calc is None
        np.testing.assert_array_equal(atoms.positions, original)
    provider.atoms.positions[:] = 10
    provider.atoms.info["charge"] = -2
    assert selection["candidates"][0].info["charge"] == 1
    np.testing.assert_array_equal(selection["candidates"][0].positions, candidates[0])


def test_sampling_calculator_uses_ase_units():
    from ase import Atoms

    from rxts.xtb import EV_PER_KCAL

    tsg = rxts.TransitionStateGuesser()
    gradient = np.arange(6.0).reshape(2, 3)
    tsg._energy_grad = lambda pot, x, lam: (4.0, gradient)
    atoms = Atoms(numbers=[6, 1], positions=np.zeros((2, 3)))
    atoms.calc = ff._SamplingCalculator(tsg, SimpleNamespace(n_atoms=2), 0.5)

    assert atoms.get_potential_energy() == pytest.approx(4.0 * EV_PER_KCAL)
    np.testing.assert_allclose(atoms.get_forces(), -gradient * EV_PER_KCAL)


def test_minimizer_failure_keeps_last_state(monkeypatch):
    from rxts.xtb import EV_PER_KCAL

    tsg = rxts.TransitionStateGuesser()
    gradient = np.arange(6.0).reshape(2, 3)
    tsg._energy_grad = lambda pot, x, lam: (3.0, gradient)
    calls = {}

    class NotConverged:
        def __init__(self, atoms, *, trajectory, logfile):
            self.atoms = atoms
            calls["atoms"] = atoms
            assert trajectory is logfile is None

        def run(self, *, fmax, steps):
            calls["settings"] = fmax, steps
            self.atoms.positions[0, 0] = 0.25
            self.atoms.get_forces()  # retain the state already evaluated by the optimizer
            return False

    monkeypatch.setattr(ff, "LBFGS", NotConverged)
    pot = SimpleNamespace(n_atoms=2, numbers=np.array([6, 1]))
    with pytest.raises(ff._FFRelaxationError, match=r"lambda=0.500.*did not converge") as caught:
        tsg._minimize(pot, np.zeros((2, 3)), 0.5)

    assert calls["settings"] == (0.05, 2000)
    np.testing.assert_array_equal(caught.value.positions, [[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert caught.value.energy_ev == pytest.approx(3.0 * EV_PER_KCAL)
    np.testing.assert_allclose(caught.value.forces_ev_per_angstrom, -gradient * EV_PER_KCAL)


def test_failed_embed_relaxation_keeps_evidence(monkeypatch):
    from rxts.structure import Frame

    tsg = rxts.TransitionStateGuesser(engine="gfnff", n_embed=2, multiplicity=1)
    bad, good = np.full((2, 3), 10.0), np.full((2, 3), 20.0)
    monkeypatch.setattr(ff, "embed_candidates", lambda **kwargs: [bad, good])

    def minimize(pot, x, lam):
        if x[0, 0] == 10.0:
            raise ff._FFRelaxationError(lam, x + 0.1, 1.25, x * 0.0, "did not converge")
        return np.full_like(x, 2.0 if x[0, 0] == 20.0 else 3.0)

    monkeypatch.setattr(tsg, "_minimize", minimize)

    class Potential:
        def targets(self, lam):
            return {(0, 1): 1.0}

        def energy(self, x, lam):
            energy = 1.0 if np.allclose(x, 2.0) else 3.0
            return energy, energy, 0.0

    frames = [
        Frame(lam=lam, positions=np.zeros((2, 3)), energy=5.0, e_reactant=0.0, e_product=0.0) for lam in (0.0, 0.5, 1.0)
    ]
    with pytest.warns(RuntimeWarning, match=r"Rejected 1 of 2 re-embedded proposal\(s\) at lambda=0.500") as caught:
        embedded, _ = tsg._conformer_scan(Potential(), object(), frames, charge=0, windows=[1])

    assert len(caught) == 1
    assert embedded == 1
    np.testing.assert_array_equal(frames[1].positions, np.full((2, 3), 2.0))
    assert len(tsg._sampling_failures) == 1
    failure = tsg._sampling_failures[0]
    assert failure["lambda"] == 0.5
    assert failure["reason"].endswith("did not converge.")
    assert failure["energy_ev"] == 1.25
    assert failure["positions_angstrom"] == (bad + 0.1).tolist()
    assert failure["forces_ev_per_angstrom"] == np.zeros_like(bad).tolist()


def test_empty_peak_sampling_retains_path(monkeypatch):
    tsg = rxts.TransitionStateGuesser(engine="auto", multiplicity=1, n_embed=1, embed_window=1)
    frames = [SimpleNamespace(energy=energy, lam=lam) for energy, lam in ((0, 0), (1, 0.5), (0, 1))]
    monkeypatch.setattr(tsg, "_conformer_scan", lambda **kwargs: (0, 0.0))
    with pytest.warns(RuntimeWarning, match="no candidates.*retained"):
        assert tsg._refine_guess(object(), object(), frames, charge=0) is frames
    tsg.embed_score = "gxtb"
    with pytest.raises(RuntimeError, match=r"scoring could not run.*no candidates"):
        tsg._refine_guess(object(), object(), frames, charge=0)


@pytest.mark.parametrize("settings", [{"n_embed": 0}, {"n_embed": 2, "peak_anchored": False}])
def test_peak_scoring_requires_sampling(monkeypatch, settings):
    guesser = rxts.TransitionStateGuesser(engine="gfnff", embed_score="gxtb", multiplicity=1, **settings)
    monkeypatch.setattr(guesser, "prepare_endpoints", lambda *args: pytest.fail("inactive scorer reached builder"))
    with pytest.raises(ValueError, match="embed_score requires n_embed > 0 and peak_anchored=True"):
        guesser.find_transition_state("CBr.[Cl-]>>CCl.[Br-]")


@pytest.mark.parametrize("case", ["constraint_index", "torsional_scoring"])
def test_invalid_prepared_request_skips_ff_build(monkeypatch, case):
    prepared = rxts.prepare_endpoints("CBr.[Cl-]>>CCl.[Br-]", multiplicity=1, n_cores=1)
    settings = {"constraints": {(0, prepared.reactant.GetNumAtoms()): 1.0}}
    message = "constraints must use atom indices"
    if case == "torsional_scoring":
        prepared = replace(prepared, forming_bonds=[], breaking_bonds=[])
        settings = {"embed_score": "gxtb"}
        message = "embed_score requires changing bonds"
    guesser = rxts.TransitionStateGuesser(engine="gfnff", **settings)
    monkeypatch.setattr(ff.TransitionStateGuesser, "_build_potential", lambda *a, **kw: pytest.fail("built FF"))
    with pytest.raises(ValueError, match=message):
        guesser.find_transition_state(prepared)


def test_empty_anchor_refuses_selection(monkeypatch):
    guesser = rxts.TransitionStateGuesser(engine="gfnff", multiplicity=1, n_lambda=2, n_embed=1, embed_score="gxtb")
    frames = [SimpleNamespace(energy=energy, lam=lam, positions=np.zeros((1, 3))) for energy, lam in ((1, 0), (0, 1))]
    monkeypatch.setattr(guesser, "_conformer_scan", lambda **kwargs: (1, 0.0))
    monkeypatch.setattr(guesser, "_embed_minimised", lambda *args: [])
    with pytest.raises(RuntimeError, match="Peak conformer selection has no candidates"):
        guesser._refine_guess(object(), object(), frames, charge=0)


def test_verbose_shows_progress_once(capsys):
    logger = logging.getLogger("rxts")
    level, handlers, propagate = logger.level, logger.handlers[:], logger.propagate
    logger.handlers.clear()
    logger.propagate = True
    try:
        rxts.set_verbose()
        logger.info("lambda E_reactant E_product V")
        assert capsys.readouterr().out == "lambda E_reactant E_product V\n"
        assert logger.propagate is False
        before = logger.handlers[:]
        rxts.set_verbose()
        assert logger.level == logging.INFO
        rxts.set_verbose(False)
        assert logger.level == logging.WARNING
        rxts.set_verbose()
        assert logger.level == logging.INFO
        assert logger.handlers == before
    finally:
        logger.setLevel(level)
        logger.handlers = handlers
        logger.propagate = propagate


_PAIR, _TARGET = (0, 1), 1.90
_PD_SMILES = "[Cl-]->[Pd+2](<-[NH3])(<-[NH3])<-[NH3]"
_PD_COMPLEX = """5
[PdCl2(NH3)2], donors perceived by distance
Pd  0.000  0.000  0.000
N   2.000  0.000  0.000
N  -2.000  0.000  0.000
Cl  0.000  2.300  0.000
Cl  0.000 -2.300  0.000
"""
_ONE_DONOR = """4
one-coordinate Pd complex
Pd  0.000  0.000  0.000
N   2.000  0.000  0.000
C   3.350  0.300  0.000
H   3.750  1.200  0.000
"""


def _distance(x, pair):
    i, j = pair
    return float(np.linalg.norm(np.asarray(x)[i] - np.asarray(x)[j]))


def _angle(x, a, center, b):
    arms = np.asarray(x)[[a, b]] - np.asarray(x)[center]
    cosine = np.dot(arms[0], arms[1]) / np.linalg.norm(arms[0]) / np.linalg.norm(arms[1])
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def _set_distance(x, pair, target):
    i, j = pair
    y = np.asarray(x).copy()
    direction = y[i] - y[j]
    y[i] = y[j] + target * direction / np.linalg.norm(direction)
    return y


def test_organic_proposals_are_reproducible():
    mol = as_molecules("CCO")[0]
    x = _set_distance(mol.GetConformer().GetPositions(), _PAIR, _TARGET)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        first = embed_candidates(mol, x, targets={_PAIR: _TARGET}, n=6, n_cores=1)
        second = embed_candidates(mol, x, targets={_PAIR: _TARGET}, n=6, n_cores=1)
    assert first
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert np.asarray(a).shape == (mol.GetNumAtoms(), 3)
        assert _distance(a, _PAIR) == pytest.approx(_TARGET, abs=0.15)
        np.testing.assert_array_equal(a, b)


def test_embedding_failure():
    mol = as_molecules("CCO")[0]
    impossible = {(0, mol.GetNumAtoms() + 90): 1.5}
    with pytest.warns(RuntimeWarning, match="Distance-geometry sampling failed"):
        assert embed_candidates(mol, mol.GetConformer().GetPositions(), impossible, n=4, n_cores=1) == []


def _capture_rxembed(monkeypatch):
    seen = {}

    def embed(source, **kw):
        seen.update(kw)
        return SimpleNamespace(mol=source, ids=[0])

    monkeypatch.setattr(rxembed, "embed", embed)
    return seen


def test_metal_core(monkeypatch):
    mol = mol_from_xyz(_PD_COMPLEX)
    seen = _capture_rxembed(monkeypatch)
    x = mol.GetConformer().GetPositions()
    targets = {(3, 4): 3.0}
    assert metal_core(mol, x, targets) == [0, 1, 2, 3, 4]
    embed_candidates(mol, x, targets, n=4, n_cores=1)
    assert seen == {"fix": [0, 1, 2, 3, 4], "n": 4, "threads": 1, "charge": 0, "seed": 0xF00D}


def test_native_metal_sampling(monkeypatch):
    mol = mol_from_xyz(_PD_COMPLEX)
    x = mol.GetConformer().GetPositions()
    x[1, 0] += 0.1
    constraints = object()
    isomer = SimpleNamespace(
        mol=Chem.Mol(mol),
        centres=(object(),),
        donor_bonds=[(1, 0)],
        base_cons=constraints,
        constrained_metals=frozenset({0}),
        lengths="model",
    )
    captured = {}

    def from_state(cls, source_mol, centres, donor_bonds, **kwargs):
        captured["source_mol"] = source_mol
        captured["centres"] = centres
        captured["donor_bonds"] = donor_bonds
        captured["state"] = kwargs
        return SimpleNamespace(mol=source_mol)

    def native_embed(source, **kwargs):
        captured["source"] = source
        captured["options"] = kwargs
        return SimpleNamespace(mol=source.mol, ids=[0])

    monkeypatch.setattr(ff.rxembed.Isomer, "from_state", classmethod(from_state))
    monkeypatch.setattr(ff.rxembed.core, "embed", native_embed)
    monkeypatch.setattr(ff.rxembed, "embed", lambda *a, **k: pytest.fail("used charge-inferring facade"))

    proposals = embed_candidates(mol, x, targets={(1, 2): 2.5}, n=2, n_cores=1, seed=42, isomer=isomer)

    np.testing.assert_array_equal(captured["source_mol"].GetConformer().GetPositions(), x)
    assert captured["centres"] is isomer.centres
    assert captured["donor_bonds"] is isomer.donor_bonds
    assert captured["state"] == {
        "constraints": constraints,
        "constrained_metals": isomer.constrained_metals,
        "lengths": "model",
        "protect_arrangement": False,
    }
    assert captured["options"] == {"fix": [0, 1, 2, 3, 4], "seed": 42, "threads": 1, "n": 2}
    np.testing.assert_array_equal(proposals[0], x)


@pytest.mark.parametrize(("smiles", "charge"), [("CCO", 0), ("CC[O-]", -1)])
def test_organic_targets_are_windows_not_fixed_atoms(monkeypatch, smiles, charge):
    mol = as_molecules(smiles)[0]
    seen = _capture_rxembed(monkeypatch)
    x = mol.GetConformer().GetPositions()
    targets = {_PAIR: _TARGET, (1, 2): 1.7}
    assert metal_core(mol, x, targets) == []
    assert embed_candidates(mol, x, targets, n=4, n_cores=1)
    windows = seen.pop("fix")
    assert seen == {"n": 4, "charge": charge, "seed": 0xF00D, "threads": 1}
    assert windows.keys() == targets.keys()
    assert windows[(0, 1)] == pytest.approx((1.85, 1.95))
    assert windows[(1, 2)] == pytest.approx((1.65, 1.75))


def test_organic_windows_leave_angle_free(monkeypatch):
    mol = as_molecules("CCI.CN")[0]
    x = mol.GetConformer().GetPositions()
    nu, center, leaving = 4, 1, 2
    axis = x[leaving] - x[center]
    axis /= np.linalg.norm(axis)
    normal = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    normal /= np.linalg.norm(normal)
    x[leaving] = x[center] + 2.5 * axis
    fragment = next(indices for indices in Chem.GetMolFrags(mol) if nu in indices)
    x[list(fragment)] += x[center] + 2.3 * normal - x[nu]
    targets = {(nu, center): 2.3, (center, leaving): 2.5}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        proposals = embed_candidates(mol, x, targets, n=25, n_cores=1, seed=42)
    assert len(proposals) > 1
    assert all(_distance(p, pair) == pytest.approx(targets[pair], abs=0.15) for p in proposals for pair in targets)
    angles = [_angle(p, nu, center, leaving) for p in proposals]
    assert max(angles) - min(angles) > 20


def test_no_targets_proposes_nothing():
    mol = mol_from_xyz(_PD_COMPLEX)
    assert embed_candidates(mol, mol.GetConformer().GetPositions(), targets={}, n=4, n_cores=1) == []


def test_disabled_sampling_makes_no_embedding_call(monkeypatch):
    mol = as_molecules("CCO")[0]
    x = mol.GetConformer().GetPositions()
    monkeypatch.setattr(rxembed, "embed", lambda *a, **kw: pytest.fail("disabled sampling reached embedding"))
    assert embed_candidates(mol, x, {_PAIR: _TARGET}, n=0, n_cores=1) == []


def test_torsion_proposals_preserve_seeded_core():
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCCC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=19) == 0
    x = mol.GetConformer().GetPositions()
    reactive = [(0, 1)]

    assert conformers(mol, x, reactive, n=1, rng=np.random.default_rng(4)) == []
    assert conformers(mol, x, reactive, n=0, rng=np.random.default_rng(4)) == []

    first = conformers(mol, x, reactive, n=4, rng=np.random.default_rng(4))
    second = conformers(mol, x, reactive, n=4, rng=np.random.default_rng(4))
    assert len(first) == 3
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:2], x[:2])
    assert any(np.max(np.abs(candidate[2:] - x[2:])) > 0.1 for candidate in first)


def test_torsion_proposals_skip_ineligible_molecules():
    mol = as_molecules("C")[0]
    x = mol.GetConformer().GetPositions()
    assert conformers(mol, x, reactive=(), n=4, rng=np.random.default_rng(4)) == []


def test_single_donor_metal_reembeds():
    mol = mol_from_xyz(_ONE_DONOR)
    pair, target = (0, 1), 2.2
    x = _set_distance(mol.GetConformer().GetPositions(), pair, target)
    proposals = embed_candidates(mol, x, targets={pair: target}, n=3, n_cores=1)
    assert proposals
    assert all(_distance(y, pair) == pytest.approx(target, abs=1e-6) for y in proposals)


def test_metal_proposals_preserve_core_and_isomer():
    mol = rxembed.embed(rxembed.metal(_PD_SMILES, "square_planar").select(index=0), n=6, threads=1).lowest(1).mol
    targets = {(0, 1): 2.6}
    x = _set_distance(mol.GetConformer().GetPositions(), (0, 1), targets[(0, 1)])
    core = metal_core(mol, x, targets)
    arrangement = rxembed.cxsmiles(set_positions(mol, x))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        proposals = embed_candidates(mol, x, targets=targets, n=4, n_cores=1)
    assert proposals
    periphery = [i for i in range(mol.GetNumAtoms()) if i not in core]
    for y in proposals:
        assert np.asarray(y).shape == (mol.GetNumAtoms(), 3)
        assert rxembed.cxsmiles(set_positions(mol, y)) == arrangement
        assert _distance(y, (0, 1)) == pytest.approx(2.6, abs=1e-6)
        drift = max(abs(np.linalg.norm(y[i] - y[j]) - np.linalg.norm(x[i] - x[j])) for i in core for j in core if i < j)
        assert drift < 1e-6
        assert max(np.linalg.norm(y[i] - x[i]) for i in periphery) > 0.1
