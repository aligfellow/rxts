"""Endpoint construction, supplied geometries and approach placement."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

import rxts
from rxts import builder, gfnff, mapping, scoring, utils
from rxts.builder import (
    _check_same_atoms,
    as_molecules,
    combine_spaced,
    derive_product_geometry,
    prepare_endpoints,
    reactive_complex,
    relax_metal_endpoint,
    select_metal_state,
    selected_endpoint,
)
from rxts.evb import mmff_can_type
from rxts.mapping import bond_set, invert, map_reaction, parse_smiles, reaction_bonds
from rxts.utils import PLACED, metal_atoms, mol_from_xyz, perceived_bonds, require_topology, set_positions


def test_mapping_and_preparation_agree():
    graph = parse_smiles("CC(=O)OC.O")
    built = as_molecules("CC(=O)OC.O")[0]
    assert [a.GetSymbol() for a in graph.GetAtoms()] == [a.GetSymbol() for a in built.GetAtoms()]
    assert [a.GetAtomMapNum() for a in graph.GetAtoms()] == [a.GetAtomMapNum() for a in built.GetAtoms()]
    assert built.GetNumConformers() == 1  # the geometry builder adds coordinates on top


def test_native_embedding_preserves_identity(monkeypatch):
    graph = parse_smiles("[13CH3:7][NH3+:8]")
    embedded = Chem.Mol(graph)
    conf = Chem.Conformer(embedded.GetNumAtoms())
    for index in range(embedded.GetNumAtoms()):
        conf.SetAtomPosition(index, (float(index), 0.1 * index, 0.0))
    embedded.AddConformer(conf)

    class NativeResult:
        def __init__(self, mol):
            self.ids = [0]
            self.unrelaxed = []
            self.mol = mol

    class CandidateSet(list):
        pass

    calls = []
    mode = "ok"

    def embed(source, **options):
        calls.append((source, options))
        result = NativeResult(embedded)
        if mode == "ambiguous":
            return CandidateSet([result])
        if mode == "unrelaxed":
            result.unrelaxed = [0]
        if mode in {"changed", "radical"}:
            changed = Chem.Mol(embedded)
            if mode == "changed":
                changed.GetAtomWithIdx(0).SetIsotope(12)
            else:
                changed.GetAtomWithIdx(0).SetNumRadicalElectrons(1)
            return NativeResult(changed)
        return result

    native = SimpleNamespace(Conformers=NativeResult, EnsembleSet=CandidateSet, embed=embed)
    monkeypatch.setattr(builder, "rxembed", native)
    result = as_molecules(graph, seed=42)[0]

    assert calls[0][0].GetNumAtoms() == graph.GetNumAtoms()
    assert calls[0][1] == {"n": 1, "seed": 42, "threads": 1}
    assert [
        (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetAtomMapNum())
        for atom in result.GetAtoms()
    ] == [
        (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetAtomMapNum())
        for atom in graph.GetAtoms()
    ]
    assert result.GetNumBonds() == graph.GetNumBonds()
    assert result.GetNumConformers() == 1
    assert not result.HasProp(PLACED)

    mode = "ambiguous"
    with pytest.raises(ValueError, match="select one native isomer and conformer"):
        as_molecules(graph, seed=42)
    mode = "unrelaxed"
    with pytest.raises(ValueError, match="unrelaxed conformer"):
        as_molecules(graph, seed=42)
    mode = "changed"
    with pytest.raises(ValueError, match="changed the selected topology"):
        as_molecules(graph, seed=42)
    mode = "radical"
    with pytest.raises(ValueError, match="changed the selected topology"):
        as_molecules(graph, seed=42)


def test_metal_atoms_finds_the_transition_metal():
    pd_complex = mol_from_xyz(
        "5\n[PdCl2(NH3)2], donors perceived by distance\n"
        "Pd  0.000  0.000  0.000\nN   2.000  0.000  0.000\nN  -2.000  0.000  0.000\n"
        "Cl  0.000  2.300  0.000\nCl  0.000 -2.300  0.000\n"
    )
    assert metal_atoms(pd_complex) == [0]
    assert metal_atoms(as_molecules("CCO")[0]) == []


# -- coordinate input has no bond orders -----------------------------------


def test_xyz_endpoint_preserves_reader_metadata(tmp_path):
    xyz = tmp_path / "molecule.xyz"
    Chem.MolToXYZFile(as_molecules("CO")[0], str(xyz))
    mol = mol_from_xyz(xyz)
    assert mol.GetProp("_rxembedBondOrders") == "xyzgraph"
    assert mmff_can_type(mol) is False, (
        "Inferred endpoint bond orders are not an explicit MMFF parameterization request"
    )


def test_a_geometry_claims_no_implicit_hydrogens():
    """Coordinate input already lists every atom, so AddHs must be a no-op.

    This catches the measured 66-to-73 atom expansion.
    """
    benzene = mol_from_xyz(
        "12\n\n"
        "C  1.396  0.000 0.000\nC  0.698  1.209 0.000\nC -0.698  1.209 0.000\n"
        "C -1.396  0.000 0.000\nC -0.698 -1.209 0.000\nC  0.698 -1.209 0.000\n"
        "H  2.479  0.000 0.000\nH  1.240  2.147 0.000\nH -1.240  2.147 0.000\n"
        "H -2.479  0.000 0.000\nH -1.240 -2.147 0.000\nH  1.240 -2.147 0.000\n"
    )
    assert sum(a.GetNumImplicitHs() for a in benzene.GetAtoms()) == 0
    assert Chem.AddHs(benzene, addCoords=True).GetNumAtoms() == benzene.GetNumAtoms()


# -- which face the nucleophile comes in on ----------------------------------


def test_fragment_rotations():
    source = np.array([1.0, 0.0, 0.0])
    for target in (source, np.array([0.0, 1.0, 0.0]), -source):
        assert builder._rotation_onto(source, target) @ source == pytest.approx(target)


def test_monatomic_fragment_position():
    rng = np.random.default_rng(813)
    for _ in range(8):
        positions = rng.normal(size=(5, 3))
        target = rng.normal(size=3)
        rotation = builder._rotation_about(rng.normal(size=3), float(rng.uniform(-np.pi, np.pi)))
        placed = builder._place_fragment(positions, np.array([2]), 2, rotation, target)
        assert np.array_equal(placed[0], target)


def test_polyatomic_nucleophile_moves_rigidly():
    mol = combine_spaced(as_molecules(["[NH2-]", "CCl"], seed=0xF00D))
    before = np.asarray(mol.GetConformer().GetPositions()).copy()
    fragment = np.array(next(part for part in Chem.GetMolFrags(mol) if 0 in part))
    distances = np.linalg.norm(before[fragment][:, None] - before[fragment][None, :], axis=-1)
    seen = []

    def judge(x):
        candidate = np.linalg.norm(x[fragment][:, None] - x[fragment][None, :], axis=-1)
        seen.append(float(np.max(np.abs(candidate - distances))))
        return float(x[1, 1]), None

    placed = builder.search_approach(mol, forming=[(0, 3)], breaking=[(3, 4)], energy=judge)
    after = np.asarray(placed.GetConformer().GetPositions())

    assert len(seen) == 100  # 24 sphere points + one anti direction, each at four azimuths
    assert max(seen) < 1e-9
    assert builder._lone_pair_axis(mol, 0) is not None
    assert all(
        np.linalg.norm(after[neighbour] - after[3]) > np.linalg.norm(after[0] - after[3]) for neighbour in (1, 2)
    )


def test_ambiguous_placement_is_unchanged():
    intramolecular = combine_spaced(as_molecules(["OCCCCCl"], seed=0xF00D))
    multiple = combine_spaced(as_molecules(["[NH2-]", "CCl"], seed=0xF00D))
    cases = [
        (intramolecular, [(0, 4)], [(4, 5)]),
        (multiple, [(0, 3), (1, 4)], [(3, 5), (4, 2)]),
    ]

    for mol, forming, breaking in cases:
        before = np.asarray(mol.GetConformer().GetPositions()).copy()
        calls = 0

        def judge(_x):
            nonlocal calls
            calls += 1
            return 0.0, None

        placed = builder.search_approach(mol, forming=forming, breaking=breaking, energy=judge)
        assert calls == 0
        assert np.array_equal(placed.GetConformer().GetPositions(), before)


def test_monatomic_attack_search():
    mol = combine_spaced(as_molecules(["[Cl-]", "CBr"], seed=0xF00D))
    before = np.asarray(mol.GetConformer().GetPositions()).copy()
    away = before[1] - before[2]
    direction = away / np.linalg.norm(away)
    reach = builder._contact_distance(mol, 0, 1) + builder._CONTACT_SLACK
    expected = before[1] + reach * direction
    calls = 0

    def anti(x):
        nonlocal calls
        calls += 1
        approach = x[0] - x[1]
        leaving = x[2] - x[1]
        return float(approach @ leaving / (np.linalg.norm(approach) * np.linalg.norm(leaving))), None

    placed = builder.search_approach(mol, forming=[(0, 1)], breaking=[(1, 2)], energy=anti)
    after = np.asarray(placed.GetConformer().GetPositions())

    assert calls == 25
    assert builder._lone_pair_axis(mol, 0) is None
    assert np.array_equal(after[0], expected)
    assert np.array_equal(after[1:], before[1:])


@pytest.mark.skipif(not gfnff.in_process_available(), reason="charged fragments need linked GFN-FF")
def test_gfnff_substitution_has_backside_attack():
    import rxts
    from rxts.utils import angle

    guesser = rxts.TransitionStateGuesser(engine="gfnff", multiplicity=1, n_cores=1, n_conformers=1, n_embed=0)
    path = guesser.find_transition_state("[I-].O=c1ccc2ccc(OCCCCCl)cc2[nH]1>>O=c1ccc2ccc(OCCCCI)cc2[nH]1.[Cl-]")

    x = path.guess("ff").positions
    carbon = (set(path.forming_bonds[0]) & set(path.breaking_bonds[0])).pop()
    iodide = next(a for a in path.forming_bonds[0] if a != carbon)
    chloride = next(a for a in path.breaking_bonds[0] if a != carbon)

    attack = np.degrees(angle(x, iodide, carbon, chloride))
    assert attack > 140.0, f"I-C-Cl is {attack:.1f} degrees -- that is a frontside attack, not an SN2"


# Check endpoint connectivity before passing geometries to GFN-FF.


def _aligned_graphs(reactants, products):
    rea = combine_spaced(as_molecules(reactants))
    pro = combine_spaced(as_molecules(products))
    mechanism, _ = map_reaction(rea, pro)
    pro = Chem.RenumberAtoms(pro, invert(mechanism.mapping))
    forming, breaking = reaction_bonds(rea, pro, list(range(rea.GetNumAtoms())))
    return reactive_complex(rea, sorted(forming)), pro, forming, breaking


def test_a_geometry_with_the_wrong_bonding_is_refused():
    rea, _, _, _ = _aligned_graphs(["CC(=O)OCC", "O"], ["CC(=O)O", "OCC"])

    # Tear one bond by pulling an atom far away.
    broken = Chem.Mol(rea)
    conf = broken.GetConformer()
    pos = conf.GetPositions()
    conf.SetAtomPosition(0, (pos[0] + np.array([8.0, 0.0, 0.0])).tolist())

    with pytest.raises(ValueError, match="does not have the bonding it should"):
        require_topology(broken, bond_set(rea), "reactant")


def test_a_correct_geometry_passes():
    rea, _, _, _ = _aligned_graphs(["CC(=O)OCC", "O"], ["CC(=O)O", "OCC"])
    require_topology(rea, bond_set(rea), "reactant")  # must not raise


def test_the_error_names_the_offending_bond():
    rea, _, _, _ = _aligned_graphs(["C[Br]", "[Cl-]"], ["C[Cl]", "[Br-]"])
    broken = Chem.Mol(rea)
    conf = broken.GetConformer()
    pos = conf.GetPositions()
    conf.SetAtomPosition(1, (pos[1] + np.array([9.0, 0.0, 0.0])).tolist())  # yank the Br

    with pytest.raises(ValueError, match=r"C0-Br1"):
        require_topology(broken, bond_set(rea), "reactant")


# -- the derived product -----------------------------------------------------


def test_the_derived_product_has_the_products_bonding():
    rea, pro, forming, breaking = _aligned_graphs(["CC(=O)OCC", "O"], ["CC(=O)O", "OCC"])
    derived = derive_product_geometry(rea, pro, forming, breaking)
    assert perceived_bonds(derived) == bond_set(pro)


def test_no_spectator_bond_is_torn():
    """The earlier reactant-force-field relaxation stretched spectator C0-C1 to 3.79 A.

    Product-force-field cleanup must preserve spectator bonds.
    """
    rea, pro, forming, breaking = _aligned_graphs(["CC(=O)OCC", "O"], ["CC(=O)O", "OCC"])
    derived = derive_product_geometry(rea, pro, forming, breaking)

    reactive = {a for pair in list(forming) + list(breaking) for a in pair}
    pos = derived.GetConformer().GetPositions()
    table = Chem.GetPeriodicTable()

    for bond in pro.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in reactive and j in reactive:
            continue  # a reactive bond is *meant* to change
        length = float(np.linalg.norm(pos[i] - pos[j]))
        limit = 1.25 * (
            table.GetRcovalent(derived.GetAtomWithIdx(i).GetAtomicNum())
            + table.GetRcovalent(derived.GetAtomWithIdx(j).GetAtomicNum())
        )
        assert length < limit, f"spectator bond {i}-{j} was stretched to {length:.2f} A"


def test_product_uses_reactant_frame():
    rea, pro, forming, breaking = _aligned_graphs(["C[Br]", "[Cl-]"], ["C[Cl]", "[Br-]"])
    derived = derive_product_geometry(rea, pro, forming, breaking)

    r = rea.GetConformer().GetPositions()
    p = derived.GetConformer().GetPositions()
    rmsd = float(np.sqrt(((p - r) ** 2).sum(1).mean()))
    assert rmsd < 2.0, "the derived product drifted out of the reactant's frame"


def test_iodide_substitution_geometry_is_derived():
    rea, pro, forming, breaking = _aligned_graphs(["CCCCCl", "[I-]"], ["CCCCI", "[Cl-]"])
    derived = derive_product_geometry(rea, pro, forming, breaking)
    assert perceived_bonds(derived) == bond_set(pro)


def test_veloxchem_contact_retries(monkeypatch):
    slacks = []
    statuses = iter((1, 1, 1, 1, 0))

    def minimiser(mol):
        return SimpleNamespace(
            AddDistanceConstraint=lambda i, j, lower, upper, force: slacks.append(upper),
            Initialize=lambda: None,
            Minimize=lambda **kwargs: next(statuses),
        ), "mmff"

    monkeypatch.setattr(builder, "_contact_distance", lambda mol, i, j: 0.0)
    monkeypatch.setattr(builder, "rdkit_minimiser", minimiser)
    reactive_complex(as_molecules("CC")[0], [(0, 1)])

    # VeloxChem starts at 0.5 A and adds 0.5 A to 2.5 A (reaffbuilder.py:90, 779-780).
    assert slacks == [0.5, 1.0, 1.5, 2.0, 2.5]


def test_bond_perception_uses_strict_threshold():
    """Construct the exact boundary because sampled geometries will not distinguish < from <=."""
    table = Chem.GetPeriodicTable()
    exactly_at_threshold = utils._BOND_SCALE * (table.GetRcovalent(6) + table.GetRcovalent(6))
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, [0.0, 0.0, 0.0])
    conformer.SetAtomPosition(1, [exactly_at_threshold, 0.0, 0.0])
    mol.AddConformer(conformer)
    pair = mol.GetMol()

    positions = pair.GetConformer().GetPositions()
    measured = float(np.linalg.norm(positions[0] - positions[1]))
    assert measured == exactly_at_threshold, "the probe must sit exactly ON the boundary to test it"
    assert perceived_bonds(pair) == set(), (
        f"a pair at the strict {utils._BOND_SCALE}x covalent-radius threshold must remain unbonded"
    )


def test_builder_relaxation_failure(monkeypatch):
    mol = as_molecules("CCO")[0]
    failed = SimpleNamespace(
        Initialize=lambda: None,
        Minimize=lambda **kwargs: 1,
        AddDistanceConstraint=lambda *args: None,
        MMFFAddPositionConstraint=lambda *args: None,
    )
    monkeypatch.setattr(builder, "rdkit_minimiser", lambda mol: (failed, "mmff"))
    with pytest.warns(RuntimeWarning, match="Initial force-field relaxation did not converge"):
        assert reactive_complex(mol, []) is mol
    with pytest.warns(RuntimeWarning, match="Contact relaxation exhausted"):
        assert reactive_complex(mol, [(0, 1)]) is mol
    before = mol.GetConformer().GetPositions().copy()
    with pytest.warns(RuntimeWarning, match="Product force-field cleanup did not converge"):
        product = derive_product_geometry(mol, Chem.Mol(mol), [], [])
    np.testing.assert_array_equal(product.GetConformer().GetPositions(), before)


def test_attack_search_warns_without_potential():
    mol = combine_spaced(as_molecules(["[NH2-]", "CCl"]))
    with pytest.warns(RuntimeWarning, match=r"Attack-face search.*rxts\[ff\]"):
        assert builder.search_approach(mol, forming=[(0, 3)], breaking=[(3, 4)], energy=None) is mol
    with warnings.catch_warnings(record=True) as recorded:
        assert builder.search_approach(mol, forming=[], breaking=[], energy=None) is mol
    assert not recorded


def _placed(text):
    mol = mol_from_xyz(text, charge=-1)
    mol.SetBoolProp(PLACED, True)
    return mol


@pytest.mark.parametrize("as_xyz", [False, True])
def test_wrapper_reuses_placed_preparation(monkeypatch, as_xyz):
    def forbidden(*args, **kwargs):
        raise AssertionError("endpoint builder must not dispatch physical QM")

    monkeypatch.setattr(scoring, "make_calculator", forbidden)
    reactant = _placed("3\nreactant\nH 0 0 0\nH 0.74 0 0\nH 2.6 0 0\n")
    product = _placed("3\nproduct\nH 0 0 0\nH 2.0 0 0\nH 2.74 0 0\n")
    from rxts.utils import to_xyz

    inputs = (to_xyz(reactant), to_xyz(product)) if as_xyz else (reactant, product)
    options = {"mapping": [0, 1, 2], "charge": -1, "multiplicity": 1}

    direct = prepare_endpoints(*inputs, **options)
    wrapped = rxts.TransitionStateGuesser(**options).prepare_endpoints(*inputs)

    assert direct.mapping == wrapped.mapping == options["mapping"]
    assert direct.forming_bonds == wrapped.forming_bonds
    assert direct.breaking_bonds == wrapped.breaking_bonds
    assert direct.charge == wrapped.charge == -1
    assert direct.multiplicity == wrapped.multiplicity == 1
    np.testing.assert_array_equal(
        direct.reactant.GetConformer().GetPositions(), wrapped.reactant.GetConformer().GetPositions()
    )
    np.testing.assert_array_equal(
        direct.product.GetConformer().GetPositions(), wrapped.product.GetConformer().GetPositions()
    )
    np.testing.assert_array_equal(direct.reactant.GetConformer().GetPositions(), reactant.GetConformer().GetPositions())
    np.testing.assert_array_equal(direct.product.GetConformer().GetPositions(), product.GetConformer().GetPositions())
    endpoints = direct.endpoints
    assert all(image.calc is None for image in endpoints)
    assert all(image.info["rxts"]["mapping"] == direct.mapping for image in endpoints)
    np.testing.assert_array_equal(endpoints[0].positions, direct.reactant.GetConformer().GetPositions())
    endpoints[0].info["rxts"]["mapping"][0] = 2
    assert direct.mapping == [0, 1, 2]


def test_native_metal_identity_uses_mapped_indices(monkeypatch):
    from rdkit import Chem

    class Isomer:
        mol = Chem.MolFromSmiles("NCN")
        centres = (SimpleNamespace(atom=1, atomic_num=46, charge=2, geometry="linear", vertices=(0, 2), hand=""),)

        def restore(self):
            return Chem.MolFromSmiles("N->[Pd+2]<-N")

    monkeypatch.setattr(builder, "rxembed", SimpleNamespace(Isomer=Isomer))
    charge, graph = builder._declared_endpoint(Isomer())
    assert charge == 2
    assert graph.GetAtomWithIdx(1).GetAtomicNum() == 46
    identity = builder._metal_identity(Isomer(), [2, 0, 1])
    centre = identity["centres"][0]
    assert centre["atom"] == 1
    assert centre["mapped_atom"] == 0
    assert centre["vertices"] == [{"atom": 0, "mapped_atom": 2}, {"atom": 2, "mapped_atom": 1}]


def test_prepare_validates_spin_and_resources_first(monkeypatch):
    import pytest

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight validation reached parsing")

    monkeypatch.setattr(builder, "split_reaction_smiles", forbidden)
    for kwargs, message in (
        ({}, "multiplicity"),
        ({"multiplicity": 0}, "multiplicity"),
        ({"multiplicity": True}, "multiplicity"),
        ({"multiplicity": 1, "n_cores": 0}, "n_cores"),
    ):
        with pytest.raises(ValueError, match=message):
            prepare_endpoints("not a reaction", **kwargs)


def test_xyz_charge_matches_topology(monkeypatch):

    xyz = "1\nhydrogen\nH 0 0 0\n"

    def forbidden(*args, **kwargs):
        raise AssertionError("charge validation reached atom mapping")

    monkeypatch.setattr(mapping, "map_reaction", forbidden)
    with pytest.raises(ValueError, match="XYZ endpoints do not declare charge"):
        prepare_endpoints(xyz, xyz, multiplicity=1)
    with pytest.raises(ValueError, match="formal charges differ"):
        prepare_endpoints("[Cl-]>>Cl", multiplicity=1)
    with pytest.raises(ValueError, match="conflicts with the reactant formal charge"):
        prepare_endpoints("[Cl-]>>[Cl-]", charge=0, multiplicity=1)


def test_xyz_honours_a_complete_explicit_mapping():
    reactant = "2\nreactant\nC 0 0 0\nO 1.4 0 0\n"
    product = "2\nproduct\nO 1.4 0 0\nC 0 0 0\n"
    prepared = prepare_endpoints(reactant, product, mapping=[1, 0], charge=0, multiplicity=1)
    assert prepared.mapping == [1, 0]


def test_ambiguous_mechanism():
    import pytest

    reaction = "CC(=O)OCC.O>>CC(=O)O.OCC"
    with pytest.raises(ValueError, match="distinct mechanisms"):
        prepare_endpoints(reaction, multiplicity=1)


@pytest.mark.parametrize("solvent", [None, "acetonitrile"])
def test_attack_potential_is_optional(monkeypatch, solvent):

    from rxts import gfnff

    mol = object()
    sentinel = object()
    seen = []

    class FakeGFNFF:
        def __init__(self, molecule, *, charge, solvent):
            seen.append((molecule, charge, solvent))
            self.energy_grad = sentinel

    monkeypatch.setattr(gfnff, "GFNFF", FakeGFNFF)
    assert builder._charged_potential(mol, -1, solvent) is sentinel
    assert seen == [(mol, -1, solvent)]

    class Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no GFN-FF executable")

    monkeypatch.setattr(gfnff, "GFNFF", Unavailable)
    assert builder._charged_potential(mol, -1, solvent) is None


# Endpoint mapping, validation, and selected native metal state builder.


_PD_SMILES = "[Cl-]->[Pd+2](<-[NH3])(<-[NH3])<-[NH3]"
_PD_COMPLEX = """5
[PdCl2(NH3)2], donors perceived by distance
Pd  0.000  0.000  0.000
N   2.000  0.000  0.000
N  -2.000  0.000  0.000
Cl  0.000  2.300  0.000
Cl  0.000 -2.300  0.000
"""
_METAL_PD_COMPLEX = _PD_COMPLEX

_PD_THREE_DONOR = "CC(C)(C)[P](->[Pd+2](<-[C-]1=CC=CC=C1)<-[N-]1CCOCC1)(C(C)(C)C)C(C)(C)C"


def test_atom_maps_choose_the_mechanism():
    mapped = "[CH3:1][C:2](=[O:3])[O:4][CH2:5][CH3:6].[OH2:7]>>[CH3:1][C:2](=[O:3])[OH:7].[OH:4][CH2:5][CH3:6]"
    prepared = prepare_endpoints(mapped, multiplicity=1)

    # Atom 2 is the carbonyl carbon and atom 4 the ester oxygen: the acyl C-O is what breaks.
    broken = {tuple(sorted(p)) for p in prepared.breaking_bonds}
    assert (1, 3) in broken  # 0-indexed: C(2nd atom) - O(4th atom)


def test_missing_product_without_an_arrow_is_refused():
    with pytest.raises(ValueError, match=">>"):
        prepare_endpoints("CCO", multiplicity=1)


def test_named_bonds_match_topology(monkeypatch):

    def stop(*_args, **_kwargs):
        pytest.fail("named-bond validation reached inferred bond reporting")

    monkeypatch.setattr(builder, "reaction_bonds", stop)
    with pytest.raises(ValueError, match="does not realize the named reaction"):
        builder.prepare_endpoints(
            "C[Br].[Cl-]>>C[Cl].[Br-]", multiplicity=1, breaking_bonds=[(0, 1)], forming_bonds=[(0, 1)]
        )


def test_native_coords_use_native_cleanup(monkeypatch):
    import rxembed as rx
    from rdkit import Chem

    iso = rx.metal(_PD_SMILES, "SPL").select(index=0)
    iso.mol.AddConformer(Chem.Conformer(iso.mol.GetNumAtoms()))
    iso.mol.AddConformer(iso.mol.GetConformer(), assignId=True)
    with pytest.raises(ValueError, match="exactly one native Isomer"):
        select_metal_state(iso)
    seed = mol_from_xyz(_PD_COMPLEX)
    coords = seed.GetConformer().GetPositions()
    coords[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        relax_metal_endpoint(iso, set_positions(seed, coords), list(range(seed.GetNumAtoms())))


@pytest.mark.parametrize("placed_reactant", [False, True])
def test_native_isomers_skip_embedding(monkeypatch, placed_reactant):
    from rdkit import Chem

    module = builder
    from rxts.utils import set_positions, with_conformer

    rea = mol_from_xyz("3\n\nPd 0 0 0\nN 2 0 0\nCl 0 2.3 0\n")
    mapping = [2, 0, 1]
    pro = Chem.RenumberAtoms(rea, mapping)

    class Isomer:
        def restore(self):
            return Chem.Mol(self.mol)

        def __init__(self, mol):
            self.mol = Chem.Mol(mol)
            self.mol.RemoveAllConformers()
            self.centres = ()

    class Native:
        def __init__(self, mol, ids, iso=None):
            self._mol, self.ids, self.iso, self.unrelaxed = mol, ids, iso, []

        @property
        def mol(self):
            return Chem.Mol(self._mol)

    rea_source = Native(rea, [0], Isomer(rea)) if placed_reactant else Isomer(rea)
    sources = (rea_source, Isomer(pro))
    seen = []

    def embed(iso, **kwargs):
        assert iso is not sources[0]
        assert kwargs == {"n": 1, "seed": 61453, "threads": 1}
        seen.append("embed")
        return Native(with_conformer(iso.mol, rea.GetConformer().GetPositions()), [0], iso)

    def minimize(iso):
        assert iso is not sources[1]
        coords = rea.GetConformer().GetPositions()[mapping]
        np.testing.assert_array_equal(iso.mol.GetConformer().GetPositions(), coords)
        seen.append("minimize")
        return Native(set_positions(iso.mol, coords + np.array([5, 3, 2])), [0], iso)

    fake = SimpleNamespace(
        Isomer=Isomer,
        Conformers=Native,
        EnsembleSet=module.rxembed.EnsembleSet,
        embed=embed,
        minimize=minimize,
    )
    monkeypatch.setattr(module, "rxembed", fake)

    def forbidden(*args, **kwargs):
        pytest.fail("native metal builder reached ordinary embedding or remapping")

    for name in ("as_molecules", "search_approach", "reactive_complex", "derive_product_geometry"):
        monkeypatch.setattr(builder, name, forbidden)
    prepared = prepare_endpoints(*sources, multiplicity=1, mapping=mapping, n_cores=1)
    assert seen == (["minimize"] if placed_reactant else ["embed", "minimize"])
    assert prepared.sources[0] is sources[0]
    assert prepared.sources[1] is sources[1]
    assert prepared.mapping == mapping
    np.testing.assert_allclose(
        prepared.product.GetConformer().GetPositions(), rea.GetConformer().GetPositions(), atol=1e-12
    )
    assert sources[0].mol.GetNumConformers() == (1 if placed_reactant else 0)
    assert sources[1].mol.GetNumConformers() == 0
    if placed_reactant:
        np.testing.assert_array_equal(
            prepared.reactant.GetConformer().GetPositions(), rea.GetConformer().GetPositions()
        )


def test_native_selection_handles_ambiguity(monkeypatch):
    from rdkit import Chem

    module = builder

    source = "[Cl-]->[Pd+2].[Cl-]->[Pd+2]"
    seen = []
    iso = SimpleNamespace(centres=(SimpleNamespace(atom=1), SimpleNamespace(atom=3)))

    class Choices(list):
        def select(self, **kwargs):
            seen.append(("select", kwargs))
            if ambiguous:
                raise ValueError("matched 2 isomers; select the native arrangement")
            return iso

    class Isomer:
        pass

    class Native:
        pass

    def parse(spec, **kwargs):
        assert kwargs == {"remove_hs": False}
        seen.append(("parse", spec))
        return Chem.MolFromSmiles(spec)

    def metal(graph, geometry, **kwargs):
        assert graph.GetNumConformers() == 0
        assert kwargs == {"center": "all", "observed_only": False}
        seen.append(("metal", geometry))
        return Choices([iso])

    fake = SimpleNamespace(Isomer=Isomer, Conformers=Native, parse_smiles=parse, metal=metal, IsomerSet=Choices)
    monkeypatch.setattr(module, "rxembed", fake)
    ambiguous = False
    selected, graph, requested = module.select_metal_state(source, "SPL")
    assert selected is iso
    assert requested
    assert graph.GetNumAtoms() == 4
    assert seen == [
        ("parse", source),
        ("metal", "SPL"),
        ("select", {}),
        ("select", {"geometry": "SPL", "center": 1}),
        ("select", {"geometry": "SPL", "center": 3}),
    ]
    ambiguous = True
    with pytest.raises(ValueError, match="matched 2"):
        module.select_metal_state(source, "SPL")


def test_native_product_seed_identity(monkeypatch):
    from copy import deepcopy

    from rdkit import Chem

    module = builder

    rea = mol_from_xyz(_PD_COMPLEX)
    mapping = [2, 4, 0, 1, 3]
    pro = Chem.RenumberAtoms(rea, mapping)
    iso = SimpleNamespace(mol=pro, seating=((0, 1, 2),), hand="selected")
    original = deepcopy(iso)
    calls = []

    def minimize(seeded):
        assert seeded is not iso
        assert seeded.mol is not iso.mol
        assert seeded.seating == iso.seating
        assert seeded.hand == iso.hand
        reference = rea.GetConformer().GetPositions()[mapping]
        np.testing.assert_array_equal(seeded.mol.GetConformer().GetPositions(), reference)
        calls.append(seeded)
        transformed = reference @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + [8, 3, 2]
        return SimpleNamespace(mol=set_positions(seeded.mol, transformed))

    fake = SimpleNamespace(
        minimize=minimize,
        Conformers=lambda mol, ids, iso: SimpleNamespace(mol=mol),
    )
    monkeypatch.setattr(module, "rxembed", fake)
    monkeypatch.setattr(module, "selected_endpoint", lambda result, expected: result.mol)
    result = module.relax_metal_endpoint(iso, rea, mapping)
    assert len(calls) == 1
    np.testing.assert_allclose(
        result.GetConformer().GetPositions(), rea.GetConformer().GetPositions()[mapping], atol=1e-12
    )
    np.testing.assert_array_equal(iso.mol.GetConformer().GetPositions(), original.mol.GetConformer().GetPositions())
    assert iso.seating == original.seating
    assert iso.hand == original.hand
    assert bond_set(result) == bond_set(pro)

    def failed(seed):
        raise RuntimeError("native state cannot be realised")

    fake.minimize = failed
    with pytest.raises(RuntimeError, match="cannot be realised"):
        module.relax_metal_endpoint(iso, rea, mapping)


def test_native_selection_identity(monkeypatch):
    from rdkit import Chem

    module = builder

    class Native:
        def __init__(self, mol, ids, unrelaxed=()):
            self._mol, self.ids, self.unrelaxed = mol, ids, unrelaxed

        @property
        def mol(self):
            return Chem.Mol(self._mol)

    molecule = mol_from_xyz(_PD_COMPLEX)
    monkeypatch.setattr(
        module,
        "rxembed",
        SimpleNamespace(Conformers=Native, EnsembleSet=module.rxembed.EnsembleSet),
    )
    with pytest.raises(ValueError, match="unrelaxed"):
        module.selected_endpoint(Native(molecule, [0], [0]))
    with pytest.raises(ValueError, match="exactly one"):
        module.selected_endpoint(Native(molecule, []))
    displaced = molecule.GetConformer().GetPositions()
    displaced[1] = [10, 0, 0]
    reached = set_positions(molecule, displaced)
    selected = module.selected_endpoint(Native(reached, [0]))
    assert selected.GetBoolProp(PLACED)
    np.testing.assert_array_equal(selected.GetConformer().GetPositions(), reached.GetConformer().GetPositions())
    valid = module.selected_endpoint(Native(molecule, [0]))
    assert valid.GetBoolProp(PLACED)
    assert not molecule.HasProp(PLACED)
    changed = Chem.Mol(molecule)
    changed.GetAtomWithIdx(0).SetFormalCharge(2)
    with pytest.raises(ValueError, match="selected topology, atom maps or electronic state"):
        module.selected_endpoint(Native(changed, [0]), molecule)
    changed = Chem.Mol(molecule)
    changed.GetBondBetweenAtoms(0, 1).SetBondType(Chem.BondType.SINGLE)
    with pytest.raises(ValueError, match="selected topology, atom maps or electronic state"):
        module.selected_endpoint(Native(changed, [0]), molecule)
    changed = Chem.Mol(molecule)
    changed.GetAtomWithIdx(0).SetAtomMapNum(9)
    with pytest.raises(ValueError, match="selected topology, atom maps or electronic state"):
        module.selected_endpoint(Native(changed, [0]), molecule)
    changed = Chem.Mol(molecule)
    changed.GetAtomWithIdx(0).SetNumRadicalElectrons(1)
    with pytest.raises(ValueError, match="selected topology, atom maps or electronic state"):
        module.selected_endpoint(Native(changed, [0]), molecule)

    for field, value in (("SetIsotope", 108), ("SetChiralTag", Chem.ChiralType.CHI_TETRAHEDRAL_CW)):
        changed = Chem.Mol(molecule)
        getattr(changed.GetAtomWithIdx(0), field)(value)
        with pytest.raises(ValueError, match="selected topology, atom maps or electronic state"):
            module.selected_endpoint(Native(changed, [0]), molecule)
    for field, value in (
        ("SetIsAromatic", True),
        ("SetBondDir", Chem.BondDir.BEGINWEDGE),
        ("SetStereo", Chem.BondStereo.STEREOANY),
    ):
        changed = Chem.Mol(molecule)
        getattr(changed.GetBondBetweenAtoms(0, 1), field)(value)
        with pytest.raises(ValueError, match="selected topology, atom maps or electronic state"):
            module.selected_endpoint(Native(changed, [0]), molecule)


def test_metal_hydrogens_precede_selection(monkeypatch):
    from rdkit import Chem

    module = builder
    from rxts.utils import with_conformer

    graph = Chem.MolFromSmiles("[NH3]->[Pd+2]<-[Cl-]")
    graph = with_conformer(graph, np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.3, 0.0]]))
    iso = SimpleNamespace(centres=(SimpleNamespace(atom=1),))
    seen = []

    class Choices(list):
        def select(self, **kwargs):
            return iso

    class Isomer:
        pass

    class Native:
        pass

    def metal(placed, geometry, **kwargs):
        assert kwargs == {"center": None, "observed_only": True}
        coords = placed.GetConformer().GetPositions()
        np.testing.assert_array_equal(coords[:3], graph.GetConformer().GetPositions())
        assert placed.GetNumAtoms() == 6
        assert np.all(np.linalg.norm(coords[3:], axis=1) > 0)
        seen.append(placed)
        return Choices([iso])

    fake = SimpleNamespace(Isomer=Isomer, Conformers=Native, metal=metal, IsomerSet=Choices)
    monkeypatch.setattr(module, "rxembed", fake)
    state, _, requested = module.select_metal_state(graph, "SPL")
    assert state is iso
    assert requested
    assert len(seen) == 1
    assert graph.GetNumAtoms() == 3


def test_cx_identity_precedes_embedding():
    import rxembed as rx

    selected = rx.metal(_PD_SMILES, "SPL").select(index=0)
    cx = rx.cxsmiles(selected)
    restored, graph, requested = select_metal_state([cx])
    assert requested
    assert graph.GetNumConformers() == 0
    assert restored.geometry == selected.geometry
    assert rx.cxsmiles(restored) == cx
    with pytest.raises(ValueError, match=r"matched .* isomer"):
        select_metal_state("[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-[NH3]", "SPL")
    placed = rx.Conformers(selected.mol, [], iso=selected)
    with pytest.raises(ValueError, match="exactly one"):
        select_metal_state(placed, "SPL")


def test_native_stereo_and_coordinates():
    import rxembed as rx

    mol = as_molecules("C[C@H](F)Cl")[0]
    mol.AddConformer(mol.GetConformer(), assignId=True)
    native = rx.Conformers(mol, [0, 1])
    with pytest.raises(ValueError, match="exactly one selected conformer"):
        selected_endpoint(native)
    with pytest.raises(ValueError, match="multiple chemical states"):
        selected_endpoint(rx.EnsembleSet())
    selected = native[1]
    endpoint = selected_endpoint(selected)
    assert endpoint.GetBoolProp(PLACED)
    assert not selected.mol.HasProp(PLACED)
    assert endpoint.GetNumConformers() == 1
    assert [a.GetChiralTag() for a in endpoint.GetAtoms()] == [a.GetChiralTag() for a in mol.GetAtoms()]
    np.testing.assert_array_equal(endpoint.GetConformer().GetPositions(), mol.GetConformer(1).GetPositions())
    broken = selected.mol
    broken.GetConformer().SetAtomPosition(0, [np.nan, 0, 0])
    with pytest.raises(ValueError, match="nonfinite"):
        selected_endpoint(rx.Conformers(broken, [1]))
    with pytest.raises(ValueError, match="nonfinite"):
        selected_endpoint(broken)

    stated = rx.metal(_PD_SMILES, "SPL")[0]
    restored = rx.metal(rx.cxsmiles(stated))[0]
    assert restored.geometry == stated.geometry
    metal = rx.embed(restored, n=1, seed=1, threads=1)[0]
    placed = selected_endpoint(metal)
    assert metal.iso is restored
    assert placed.GetBoolProp(PLACED)
    assert [a.GetAtomicNum() for a in placed.GetAtoms()] == [a.GetAtomicNum() for a in metal.mol.GetAtoms()]


def test_prepare_preserves_native_endpoint(monkeypatch):
    import rxembed as rx
    from rdkit import Chem

    from rxts.utils import with_conformer

    states = rx.metal("N->[Pt+2](<-[Cl-])(<-[Cl-])<-N", "square_planar", screen=False)
    coordinates = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.4, 0.8, 0.0],
            [2.4, -0.8, 0.0],
            [1.4, 0.0, 0.8],
            [-2.4, 0.8, 0.0],
            [-2.4, -0.8, 0.0],
            [-1.4, 0.0, -0.8],
        ]
    )

    def endpoint(label, shift):
        iso = states.select(label=label)
        mol = Chem.Mol(iso.mol)
        for index, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(index + 1)
        positions = coordinates + shift
        iso.mol = with_conformer(mol, positions)
        return rx.Conformers(iso.mol, [0], iso=iso), iso, positions

    reactant, reactant_state, reactant_positions = endpoint("cis", np.zeros(3))
    product, product_state, product_positions = endpoint("trans", np.array([1.0, 2.0, 3.0]))

    def forbidden(*args, **kwargs):
        pytest.fail("placed endpoint builder reconstructed or evaluated the endpoint")

    for owner, name in (
        (rx, "embed"),
        (rx, "minimize"),
        (scoring, "make_calculator"),
        (builder, "as_molecules"),
        (builder, "combine_spaced"),
        (builder, "search_approach"),
        (builder, "derive_product_geometry"),
        (builder, "_charged_potential"),
    ):
        monkeypatch.setattr(owner, name, forbidden)

    mapping = list(range(reactant.mol.GetNumAtoms()))
    prepared = prepare_endpoints(reactant, product, mapping=mapping, charge=0, multiplicity=1, n_cores=1)

    assert prepared.mapping == mapping
    assert prepared.endpoint_identity == (
        builder._metal_identity(reactant_state),
        builder._metal_identity(product_state, mapping),
    )
    assert prepared.native_states == (reactant_state, product_state)
    for reached, source, positions in (
        (prepared.reactant, reactant.mol, reactant_positions),
        (prepared.product, product.mol, product_positions),
    ):
        np.testing.assert_array_equal(reached.GetConformer().GetPositions(), positions)
        assert [atom.GetAtomMapNum() for atom in reached.GetAtoms()] == [
            atom.GetAtomMapNum() for atom in source.GetAtoms()
        ]
        assert [
            (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetNumRadicalElectrons()) for atom in reached.GetAtoms()
        ] == [
            (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetNumRadicalElectrons()) for atom in source.GetAtoms()
        ]
        assert [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType())) for bond in reached.GetBonds()
        ] == [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType())) for bond in source.GetBonds()]


def test_formula_check_allows_reordering():
    from rdkit import Chem

    reactant = Chem.MolFromXYZBlock(_PT_REACTANT)
    with pytest.raises(ValueError, match="conserve atoms"):
        _check_same_atoms(reactant, Chem.MolFromXYZBlock(_MISSING_ATOM))

    _check_same_atoms(reactant, Chem.MolFromXYZBlock(_REORDERED_SEQUENCE))  # must not raise


def test_the_mechanism_can_be_forced():
    prepared = prepare_endpoints(
        ["CC(=O)OCC", "O"],
        ["CC(=O)O", "OCC"],
        multiplicity=1,
        breaking_bonds=[(1, 3)],
        forming_bonds=[(1, 14)],
    )
    assert prepared.breaking_bonds == [(1, 3)]
    assert prepared.forming_bonds == [(1, 14)]


def test_attack_potential_propagates_errors(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("invalid physical context")

    monkeypatch.setattr(rxts.gfnff, "GFNFF", broken)
    with pytest.raises(ValueError, match="invalid physical context"):
        builder._charged_potential(object(), charge=0, solvent=None)


# Proton transfer: C(0)-O_a(1)-H(3), the proton migrating to the bare O_b(2).
_PT_REACTANT = """4
proton on O_a
C   0.000   0.000   0.000
O   1.400   0.000   0.000
O   5.000   0.000   0.000
H   1.400   1.000   0.000
"""
_MISSING_ATOM = """3
one atom short: not the same molecule
C   0.000   0.000   0.000
O   1.400   0.000   0.000
O   5.000   0.000   0.000
"""
_REORDERED_SEQUENCE = """4
same atoms, but the symbol sequence is O,C,O,H -- the old guard refused exactly this
O   1.400   0.000   0.000
C   0.000   0.000   0.000
O   5.000   0.000   0.000
H   5.000   1.000   0.000
"""


def test_xyz_context_validates_before_read(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("context validation reached minimum perception")

    monkeypatch.setattr(builder, "_declared_endpoint", forbidden)
    with pytest.raises(ValueError, match="charge"):
        prepare_endpoints(Path("reactant.coord"), Path("product.coord"), multiplicity=1)
    with pytest.raises(ValueError, match="both endpoints"):
        prepare_endpoints("2\n\nH 0 0 0\nH .74 0 0\n", "[H][H]", charge=0, multiplicity=1)
