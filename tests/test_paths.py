"""Candidate collections keep complete profiles and independent scoring attempts."""

from dataclasses import replace

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem

import rxts
from rxts.structure import Frame
from rxts.utils import with_conformer
from rxts.xtb import EV_PER_KCAL


def _path(energies=(0.0, 2.0, 1.0)):
    frames = [Frame(i / 2, np.array([[0.0, 0.0, 0.0], [1.0 + i, 0.0, 0.0]]), e, e, e) for i, e in enumerate(energies)]
    mol = with_conformer(Chem.MolFromSmiles("[H][H]"), frames[0].positions)
    return rxts.ReactionPath(
        frames=frames,
        reactant=mol,
        product=with_conformer(mol, frames[-1].positions),
        forming_bonds=[],
        breaking_bonds=[],
        mapping=[0, 1],
        engine="gfnff",
        charge=0,
        multiplicity=1,
        solvent=None,
        n_cores=1,
        mol=mol,
        endpoint_identity=({"labels": ["reactant"]}, None),
    )


class DistanceEnergy(Calculator):
    implemented_properties = ("energy",)

    def __init__(self, fail_at=None):
        super().__init__()
        self.fail_at = fail_at

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        distance = atoms.get_distance(0, 1)
        if distance == self.fail_at:
            raise ValueError("controlled frame failure")
        self.results = {"energy": distance}


def test_collection_and_common_energy_reference():
    first, second = _path((-10, 3, 0)), _path((4, 5, 4))
    paths = rxts.ReactionPaths(iter((first, second)))
    assert list(paths) == [first, second]
    assert len(paths) == 2
    assert paths[-1] is second
    assert isinstance(paths[::2], rxts.ReactionPaths)
    assert paths[:1].best() is first
    assert first.barrier() > second.barrier()
    assert paths.best() is first  # Lowest absolute maximum, despite the larger relative barrier.
    with pytest.raises(ValueError, match="No candidate paths"):
        paths[:0].best()
    with pytest.raises(TypeError, match="ReactionPath"):
        rxts.ReactionPaths([None])


def test_individual_and_collection_scoring_are_independent():
    path = _path()
    calculator = DistanceEnergy()
    scored = path.score(calculator)
    batch = rxts.ReactionPaths([path]).score(calculator)
    assert batch.best().scores.energies_ev == scored.scores.energies_ev == (1, 2, 3)
    np.testing.assert_allclose(scored.absolute_energies, np.array([1, 2, 3]) / EV_PER_KCAL)
    assert batch.best().guess().label == scored.guess().label == calculator.name
    assert path.method == "gfnff"
    assert path.scores is None
    assert scored.peak_index() == 2
    assert path.peak_index() == 1
    scored.frames[0].positions[:] = 99
    scored.reactant.GetConformer().SetAtomPosition(0, (99, 99, 99))
    scored.mapping.reverse()
    scored.endpoint_identity[0]["labels"].append("changed")
    assert path.mapping == [0, 1]
    assert path.endpoint_identity[0]["labels"] == ["reactant"]
    np.testing.assert_array_equal(path.reactant.GetConformer().GetPositions(), path.frames[0].positions)
    np.testing.assert_array_equal(batch[0].frames[0].positions, path.frames[0].positions)


def test_partial_scores_survive_and_prevent_collection_selection():
    path = _path()
    paths = rxts.ReactionPaths([path, _path()])
    with pytest.warns(RuntimeWarning, match="1 of 3 frames failed"):
        failed = paths.score(DistanceEnergy(fail_at=2))
    for attempted in failed:
        assert attempted.scores.energies_ev == (1, None, 3)
        assert set(attempted.scores.failures) == {1}
    with pytest.raises(ValueError, match="scoring failed"):
        failed.best()
    assert all(p.scores is None for p in paths)
    recovered = failed.score(DistanceEnergy())
    assert recovered.best().scored
    assert failed[0].scores.energies_ev == (1, None, 3)


def test_interrupted_batch_scoring_keeps_completed_results(monkeypatch):
    first, second = _path(), _path()

    def stop(*args, **kwargs):
        raise KeyboardInterrupt("second path interrupted")

    monkeypatch.setattr(second, "score", stop)
    with pytest.raises(KeyboardInterrupt, match="second path") as caught:
        rxts.ReactionPaths([first, second]).score(DistanceEnergy())
    completed = caught.value.candidate_paths
    assert isinstance(completed, rxts.ReactionPaths)
    assert len(completed) == 1
    assert completed[0].scores.energies_ev == (1, 2, 3)
    assert first.scores is None


@pytest.mark.parametrize("context", [{"charge": 1}, {"multiplicity": 3}, {"solvent": "water"}, {"engine": "mmff"}])
def test_best_refuses_different_physical_contexts(context):
    path = _path()
    with pytest.raises(ValueError, match="same calculator, solvent, charge, multiplicity and atoms"):
        rxts.ReactionPaths([path, replace(_path(), **context)]).best()


def test_best_refuses_different_calculators_and_isotopes():
    path = _path()
    first, second = path.score(DistanceEnergy()), path.score(DistanceEnergy())
    with pytest.raises(ValueError, match="same calculator"):
        rxts.ReactionPaths([first, second]).best()
    deuterated = _path()
    deuterated.mol.GetAtomWithIdx(0).SetIsotope(2)
    with pytest.raises(ValueError, match="same calculator"):
        rxts.ReactionPaths([path, deuterated]).best()


@pytest.mark.parametrize("energies", [(0, np.nan, 1), ()])
def test_best_refuses_nonfinite_or_empty_profiles(energies):
    path = _path(energies) if energies else replace(_path(), frames=[])
    with pytest.raises(ValueError, match="one finite energy"):
        rxts.ReactionPaths([path]).best()
