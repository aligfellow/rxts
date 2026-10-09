from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from rxts import utils as parallel
from rxts.builder import _geometry_known, as_molecules
from rxts.utils import (
    _bounded_map,
    _worker_count,
    align_frames,
    angle,
    is_xyz_file,
    read_xyz,
    set_positions,
    to_atoms,
    with_conformer,
)


def test_angle_matches_rdkit():
    rng = np.random.default_rng(17)
    worst_rdkit = 0.0
    for _ in range(200):
        x = rng.normal(size=(3, 3)) * 1.5
        mol = Chem.RWMol()
        for _ in range(3):
            mol.AddAtom(Chem.Atom(6))
        mol.AddBond(0, 1, Chem.BondType.SINGLE)
        mol.AddBond(1, 2, Chem.BondType.SINGLE)
        conformer = Chem.Conformer(3)
        for index, position in enumerate(x):
            conformer.SetAtomPosition(index, position.tolist())
        mol.AddConformer(conformer)
        measured = angle(x, 0, 1, 2)
        worst_rdkit = max(
            worst_rdkit, abs(np.degrees(measured) - rdMolTransforms.GetAngleDeg(mol.GetConformer(), 0, 1, 2))
        )

    assert worst_rdkit < 1e-9


def test_ase_export():
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for isotope in (0, 13):
        mol = Chem.MolFromSmiles("CCl")
        mol.GetAtomWithIdx(0).SetIsotope(isotope)
        mol.GetAtomWithIdx(0).SetAtomMapNum(11)
        mol.GetAtomWithIdx(1).SetAtomMapNum(29)
        mol = with_conformer(mol, positions)
        atoms = to_atoms(mol, charge=0, multiplicity=1)
        assert atoms.get_chemical_symbols() == ["C", "Cl"]
        assert atoms.info == {"charge": 0, "multiplicity": 1}
        np.testing.assert_array_equal(atoms.arrays["atom_maps"], [11, 29])
        assert atoms.calc is None
        assert ("masses" in atoms.arrays) == bool(isotope)
        np.testing.assert_array_equal(atoms.positions, positions)
        expected = Atoms("CCl").get_masses()
        if isotope:
            expected[0] = mol.GetAtomWithIdx(0).GetMass()
        np.testing.assert_array_equal(atoms.get_masses(), expected)
        atoms.positions[0] += 1
        atoms.arrays["atom_maps"][0] = 100
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(), positions)
        assert mol.GetAtomWithIdx(0).GetAtomMapNum() == 11


# A square-planar Pd with four donors close enough to read as bonds: Pd-N at 2.0, Pd-Cl at 2.3 A.
_PD_COMPLEX = """5
[PdCl2(NH3)2], donors perceived by distance
Pd  0.000  0.000  0.000
N   2.000  0.000  0.000
N  -2.000  0.000  0.000
Cl  0.000  2.300  0.000
Cl  0.000 -2.300  0.000
"""


# XYZ content, filenames and placed endpoint geometry have different contracts.


def test_xyz_text_is_not_an_xyz_path():
    assert is_xyz_file(_PD_COMPLEX) is False
    assert _geometry_known(_PD_COMPLEX) is True
    assert (is_xyz_file("thing.xyz"), _geometry_known("thing.xyz")) == (True, True)
    assert (is_xyz_file("CCO"), _geometry_known("CCO")) == (False, False)

    # Raw conformers still need placement; selected endpoints and XYZ bypass it.
    conformer_bearing = as_molecules("CCO")[0]
    assert conformer_bearing.GetNumConformers() == 1
    assert _geometry_known(conformer_bearing) is False


def test_read_xyz_ignores_path_suffix(tmp_path):
    written = tmp_path / "frame.txt"
    written.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n")

    assert is_xyz_file(written) is False, "the suffix is not .xyz, so it does not NAME an xyz file"
    symbols, positions = read_xyz(written)
    assert symbols == ["O", "H", "H"], "...and read_xyz reads it anyway, because it is a Path"
    assert positions.shape == (3, 3)


def test_read_xyz_selects_frames_and_comments():
    trajectory = "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n\n2\nsecond\nH 0.0 0.0 0.0\nH 0.0 0.0 3.00\n"

    symbols, first = read_xyz(trajectory)
    assert symbols == ["H", "H"]
    assert first[1][2] == 0.74
    symbols, later = read_xyz(trajectory, frame=1)
    assert symbols == ["H", "H"]
    assert later[1][2] == 3.0
    with pytest.raises(IndexError, match="no frame 2"):
        read_xyz(trajectory, frame=2)


def test_read_xyz_rejects_invalid_frame_values():
    for frame in (-1, 0.5, True, "1"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            read_xyz("1\nH\nH 0 0 0\n", frame=frame)


def test_coordinate_writes_leave_source_unchanged():
    mol = as_molecules("CCO")[0]
    before = np.array(mol.GetConformer().GetPositions())
    moved = np.asarray(before) + 5.0

    written = set_positions(mol, moved)
    assert np.allclose(written.GetConformer().GetPositions(), moved)
    assert np.array_equal(mol.GetConformer().GetPositions(), before), "set_positions wrote through to the source"

    single = with_conformer(Chem.Mol(written), before)
    assert single.GetNumConformers() == 1
    assert np.allclose(single.GetConformer().GetPositions(), before)
    assert np.allclose(written.GetConformer().GetPositions(), moved), "with_conformer wrote through to the source"


def test_alignment_preserves_handedness():
    rng = np.random.default_rng(7)
    frame = rng.normal(size=(6, 3))
    # A rigid tumble of the same structure: rotate about z by 40 degrees and translate.
    c, s = np.cos(0.7), np.sin(0.7)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    tumbled = frame @ rotation.T + np.array([3.0, -2.0, 5.0])

    def internal(x):
        return np.array([np.linalg.norm(x[i] - x[j]) for i in range(len(x)) for j in range(i)])

    def chirality(x):
        return float(np.linalg.det(np.stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]])))

    aligned = align_frames([frame, tumbled, frame @ np.diag([1.0, 1.0, -1.0])])

    assert np.allclose(aligned[0].mean(axis=0), 0.0), "the first frame is centred, so the path has no drift"
    assert np.allclose(aligned[0], aligned[1], atol=1e-8), "the tumble is rigid: it must align exactly onto its frame"
    for x in aligned:
        assert np.allclose(internal(x), internal(frame)), "an alignment moves nothing relative to anything"
    assert np.sign(chirality(aligned[0])) == np.sign(chirality(frame))
    assert np.sign(chirality(aligned[2])) == -np.sign(chirality(frame)), (
        "a mirrored frame stays mirrored -- an SVD that forgets to force det=+1 unflips it silently"
    )


# Bounded parallel mapping used by FF and scoring workers.


def test_parallel_results_keep_input_order():
    items = list(range(20))

    def run_item(i):
        time.sleep(0.001 * (20 - i))
        return i * i

    assert _bounded_map(items, run_item, n_workers=4) == [i * i for i in items]


def test_workers_obey_all_budgets(monkeypatch):
    monkeypatch.setattr(parallel, "default_cores", lambda: 4)

    def peak_threads(items, n_workers):
        seen = set()
        lock = threading.Lock()

        def run_item(_):
            with lock:
                seen.add(threading.get_ident())

        _bounded_map(items, run_item, n_workers=n_workers)
        return len(seen)

    assert peak_threads(list(range(50)), 32) <= 4, "more items and request than cores -> capped at the cores"
    assert peak_threads([1, 2], 32) <= 2, "fewer items than cores -> capped by the item count"
    assert peak_threads(list(range(50)), 2) <= 2, "a request below the budget -> capped by the request"
    assert _worker_count(0, 8) == 1, "and never zero, whatever the arithmetic says"


def test_empty_batch_starts_no_pool():
    assert _bounded_map([], lambda x: x, n_workers=4) == []


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_budgets_fail_before_work(workers):
    with pytest.raises(ValueError, match="positive integer"):
        _bounded_map([1], lambda _item: pytest.fail("invalid budget started work"), n_workers=workers)


def test_workers_use_one_native_thread(monkeypatch):
    import contextlib

    seen = {}

    @contextlib.contextmanager
    def spy(limits=None):
        seen["limits"] = limits
        yield

    monkeypatch.setattr(parallel, "threadpool_limits", spy)
    _bounded_map([1, 2, 3], lambda x: x, n_workers=2)
    assert seen.get("limits") == 1
