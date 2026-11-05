"""Endpoint construction and supplied geometries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rxembed
from rdkit import Chem

import rxts
from rxts import builder, gfnff, mapping, scoring, utils
from rxts.builder import (
    _check_same_atoms,
    prepare_endpoints,
    select_metal_state,
)
from rxts.evb import mmff_can_type
from rxts.mapping import bond_set, invert, parse_smiles
from rxts.utils import metal_atoms, mol_from_xyz, perceived_bonds, require_topology


def test_metal_atoms_finds_the_transition_metal():
    pd_complex = mol_from_xyz(
        "5\n[PdCl2(NH3)2], donors perceived by distance\n"
        "Pd  0.000  0.000  0.000\nN   2.000  0.000  0.000\nN  -2.000  0.000  0.000\n"
        "Cl  0.000  2.300  0.000\nCl  0.000 -2.300  0.000\n"
    )
    assert metal_atoms(pd_complex) == [0]
    assert metal_atoms(parse_smiles("CCO")) == []


# -- coordinate input has no bond orders -----------------------------------


def test_xyz_endpoint_preserves_reader_metadata(tmp_path):
    xyz = tmp_path / "molecule.xyz"
    Chem.MolToXYZFile(rxembed.embed(parse_smiles("CO"), n=1, seed=0xF00D, threads=1).mol, str(xyz))
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


# Check the scan still finds a backside substitution. Coembed places the iodide frontside on about
# 1 seed in 4, so this uses the default pair budget and lets the lowest peak choose.


@pytest.mark.skipif(not gfnff.in_process_available(), reason="charged fragments need linked GFN-FF")
def test_gfnff_substitution_has_backside_attack():
    import rxts
    from rxts.utils import angle

    guesser = rxts.TransitionStateGuesser(engine="gfnff", multiplicity=1, n_cores=1, n_conformers=4, n_embed=0)
    path = guesser.find_transition_state("[I-].O=c1ccc2ccc(OCCCCCl)cc2[nH]1>>O=c1ccc2ccc(OCCCCI)cc2[nH]1.[Cl-]")

    x = path.guess().positions
    carbon = (set(path.forming_bonds[0]) & set(path.breaking_bonds[0])).pop()
    iodide = next(a for a in path.forming_bonds[0] if a != carbon)
    chloride = next(a for a in path.breaking_bonds[0] if a != carbon)

    attack = np.degrees(angle(x, iodide, carbon, chloride))
    assert attack > 140.0, f"I-C-Cl is {attack:.1f} degrees -- that is a frontside attack, not an SN2"


# Check endpoint connectivity before passing geometries to GFN-FF.


def _aligned_graphs(reactants, products):
    rea, pro = parse_smiles(".".join(reactants)), parse_smiles(".".join(products))
    mechanism, _ = mapping.map_reaction(rea, pro)
    prepared = prepare_endpoints(rea, pro, mapping=mechanism.mapping, multiplicity=1)
    return prepared.reactant, prepared.product, prepared.forming_bonds, prepared.breaking_bonds


def test_a_geometry_with_the_wrong_bonding_is_refused():
    rea, _, _, _ = _aligned_graphs(["C[Br]", "[Cl-]"], ["C[Cl]", "[Br-]"])
    require_topology(rea, bond_set(rea), "reactant")  # the coembedded endpoint passes
    broken = Chem.Mol(rea)
    conf = broken.GetConformer()
    conf.SetAtomPosition(1, (conf.GetPositions()[1] + np.array([9.0, 0.0, 0.0])).tolist())  # yank the Br
    with pytest.raises(ValueError, match=r"does not have the bonding it should(.|\n)*C0-Br1"):
        require_topology(broken, bond_set(rea), "reactant")


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


def test_wrapper_samples_the_same_pairs_as_direct_preparation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("endpoint builder must not dispatch physical QM")

    monkeypatch.setattr(scoring, "make_calculator", forbidden)
    pairs = rxembed.coembed("CBr.[Cl-]>>CCl.[Br-]", n=1, seed=1, threads=1)
    reactant, product = pairs.reactant.mol, pairs.product.mol
    inputs = (reactant, product)
    options = {
        "mapping": list(range(reactant.GetNumAtoms())),
        "charge": -1,
        "multiplicity": 1,
        "n_conformers": 2,
        "n_cores": 1,
    }

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
    assert direct.reactant.GetNumConformers() == direct.product.GetNumConformers() == 2
    endpoints = direct.endpoints
    assert all(image.calc is None for image in endpoints)
    assert all(image.info["rxts"]["mapping"] == direct.mapping for image in endpoints)
    np.testing.assert_array_equal(endpoints[0].positions, direct.reactant.GetConformer().GetPositions())
    endpoints[0].info["rxts"]["mapping"][0] = 2
    assert direct.mapping == list(range(reactant.GetNumAtoms()))


def test_native_metal_identity_uses_mapped_indices():
    import rxembed as rx

    species = rx.isomers("N->[Pd+2]<-N", geometry="linear").select()
    charge, graph = builder._declared_endpoint(species)
    assert charge == 2
    assert graph.GetAtomWithIdx(1).GetAtomicNum() == 46
    identity = builder._metal_identity(species, [2, 0, 1])
    centre = identity["centres"][0]
    assert centre["atom"] == 1
    assert centre["mapped_atom"] == 0
    assert (centre["atomic_number"], centre["charge"], centre["geometry"], centre["hand"]) == (46, 2, "linear", "")
    assert centre["vertices"] == [
        {"atoms": [0], "mapped_atoms": [2], "winding": ""},
        {"atoms": [2], "mapped_atoms": [1], "winding": ""},
    ]


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
    pair = rxembed.coembed("CBr.[Cl-]>>CCl.[Br-]", n=1, seed=1, threads=1)
    reactant, product = pair.reactant.mol, pair.product.mol
    order = list(reversed(range(product.GetNumAtoms())))
    prepared = prepare_endpoints(
        Chem.MolToXYZBlock(reactant),
        Chem.MolToXYZBlock(Chem.RenumberAtoms(product, order)),
        mapping=order,
        charge=-1,
        multiplicity=1,
    )
    assert prepared.mapping == order


def test_ambiguous_mechanism():
    import pytest

    reaction = "CC(=O)OCC.O>>CC(=O)O.OCC"
    with pytest.raises(ValueError, match="distinct mechanisms"):
        prepare_endpoints(reaction, multiplicity=1)


# Endpoint mapping, validation, and selected native metal state builder.


_PD_SMILES = "[Cl-]->[Pd+2](<-[NH3])(<-[NH3])<-[NH3]"


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


def test_native_selection_rejects_inconsistent_conformers():
    import rxembed as rx

    states = rx.isomers("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", geometry="SPL")
    species = rx.embed(states.select(label="cis"), n=1, seed=1).minimize().mol
    other = rx.embed(states.select(label="trans"), n=1, seed=1).minimize().mol
    species.AddConformer(other.GetConformer(), assignId=True)
    with pytest.raises(ValueError, match="another species"):
        select_metal_state(species)


def test_declared_mol_gets_its_hydrogens_on_its_conformer():
    """Hydrogens are placed before mapping, so the selected species and the mapped graph share atoms."""
    from rxts.utils import with_conformer

    coords = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.3, 0.0]])
    graph = with_conformer(Chem.MolFromSmiles("[NH3]->[Pd+2]<-[Cl-]"), coords)
    charge, declared = builder._declared_endpoint(graph)
    assert (charge, declared.GetNumAtoms(), graph.GetNumAtoms()) == (1, 6, 3)
    placed = declared.GetConformer().GetPositions()
    np.testing.assert_array_equal(placed[:3], coords)
    assert np.all(np.linalg.norm(placed[3:] - coords[0], axis=1) < 1.5)  # on the nitrogen, not at the origin


def test_cx_identity_precedes_embedding():
    import rxembed as rx

    selected = rx.isomers(_PD_SMILES, geometry="SPL").select(index=0)
    cx = rx.cxsmiles(selected)
    graph = Chem.AddHs(rx.parse_smiles(cx, remove_hs=False))
    restored = select_metal_state(graph)
    assert graph.GetNumConformers() == 0
    assert rx.identity(restored).centres[0].geometry == rx.identity(selected).centres[0].geometry
    assert rx.cxsmiles(restored) == cx
    with pytest.raises(ValueError, match="2 isomers fit"):
        select_metal_state(Chem.AddHs(rx.parse_smiles("[Cl-]->[Pd+2](<-[Cl-])(<-[NH3])<-[NH3]")), "SPL")


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


_BETA_R = (
    "C[P:5](C)(C)->[Pd+2:4](<-[CH2-:1][CH:2]([H:3])c1ccccc1)<-[P:6](C)(C)C "
    "|atomProp:1.atomNote.s0:4.atomNote.SPL:5.atomNote.s1:14.atomNote.s3|"
)
_BETA_P = (
    "C[P:5](C)(C)->[Pd+2:4]1(<-[CH2:1]=[CH:2]->1c1ccccc1)(<-[H-:3])<-[P:6](C)(C)C "
    "|atomProp:1.atomNote.s0:4.atomNote.SPL:5.atomNote.s1+:6.atomNote.s1+:13.atomNote.s2:14.atomNote.s3|"
)


def _forbid(monkeypatch, *names):
    def forbidden(*args, **kwargs):
        pytest.fail("coembed alignment reached the sequential builder")

    for name in names:
        monkeypatch.setattr(builder.rxembed, name, forbidden)


def test_unplaced_endpoints_are_coembedded(monkeypatch):
    """Unplaced endpoints come from one co-embedded pair, already in the reactant's order."""
    _forbid(monkeypatch, "minimize")
    prepared = prepare_endpoints("CBr.[Cl-]>>CCl.[Br-]", multiplicity=1)
    assert prepared.reactant.GetNumConformers() == prepared.product.GetNumConformers() == 1
    sym = [a.GetSymbol() for a in prepared.reactant.GetAtoms()]
    assert [a.GetSymbol() for a in prepared.product.GetAtoms()] == sym
    assert [sorted((sym[i], sym[j])) for i, j in prepared.forming_bonds] == [["C", "Cl"]]
    assert [sorted((sym[i], sym[j])) for i, j in prepared.breaking_bonds] == [["Br", "C"]]
    assert all(a.GetAtomMapNum() == 0 for m in (prepared.reactant, prepared.product) for a in m.GetAtoms())


def test_coembedding_budget_retains_pairing_and_accepts_fewer(monkeypatch):
    rx = builder.rxembed
    coembed = rx.coembed
    seen = []

    def fewer(sides, **options):
        assert options["n"] == 4
        assert options["threads"] == 1
        pairs = coembed(sides, **options)[1::2]
        assert 0 < len(pairs) < 4
        for index in range(len(pairs)):
            pair = pairs[index]
            seen.append(tuple(e.mol.GetConformer().GetPositions() for e in (pair.reactant, pair.product)))
        return pairs

    monkeypatch.setattr(rx, "coembed", fewer)
    sources = (_BETA_R, _BETA_P)  # mapped, so the maps the pairs carry back are checked too
    prepared = rxts.TransitionStateGuesser(multiplicity=1, n_conformers=4, n_cores=1).prepare_endpoints(*sources)
    assert prepared.reactant.GetNumConformers() == prepared.product.GetNumConformers() == len(seen)
    for i, pair in enumerate(seen):
        for mol, expected in zip((prepared.reactant, prepared.product), pair, strict=True):
            np.testing.assert_array_equal(mol.GetConformer(i).GetPositions(), expected)
    expected_maps = [atom.GetAtomMapNum() for atom in parse_smiles(sources[0]).GetAtoms()]
    assert all(
        [atom.GetAtomMapNum() for atom in mol.GetAtoms()] == expected_maps
        for mol in (prepared.reactant, prepared.product)
    )


def test_supplied_coembedded_batch_keeps_all_pairs_and_atom_order(monkeypatch):
    rx = builder.rxembed
    sides = ("CBr.[Cl-]", "CCl.[Br-]")
    pairs = rx.coembed(*sides, n=3, seed=0xF00D, threads=1)[1::2]
    assert len(pairs) > 0
    monkeypatch.setattr(rx, "coembed", lambda *a, **k: pytest.fail("rebuilt a supplied batch"))
    prepared = prepare_endpoints(pairs, multiplicity=1, n_conformers=1)
    assert prepared.mapping == list(range(prepared.reactant.GetNumAtoms()))
    for mol, ensemble in zip((prepared.reactant, prepared.product), (pairs.reactant, pairs.product), strict=True):
        assert mol.GetNumConformers() == len(pairs)
        assert [a.GetAtomMapNum() for a in mol.GetAtoms()] == [a.GetAtomMapNum() for a in ensemble.mol.GetAtoms()]
        for conf, cid in zip(mol.GetConformers(), pairs.ids, strict=True):
            np.testing.assert_array_equal(conf.GetPositions(), ensemble.mol.GetConformer(cid).GetPositions())
    with pytest.raises(ValueError, match="omit product"):
        prepare_endpoints(pairs, sides[1], multiplicity=1)
    with pytest.raises(ValueError, match="omit mapping"):
        prepare_endpoints(pairs, mapping=prepared.mapping[::-1], multiplicity=1)


@pytest.mark.parametrize(("metal", "fixed"), [(False, "reactant"), (True, "product")])
def test_explicit_fixed_coembedded_pair_keeps_its_geometry(monkeypatch, metal, fixed):
    if metal:
        states = rxembed.isomers("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", geometry="SPL")
        sources = [states.select(label=label) for label in ("cis", "trans")]
    else:
        sources = ["CBr.[Cl-]", "CCl.[Br-]"]
    initial = rxembed.coembed(*sources, n=1, seed=1, threads=1)
    sources = [initial.reactant.mol, initial.product.mol]
    side = 0 if fixed == "reactant" else 1
    positions = sources[side].GetConformer().GetPositions() + np.array([4.0, -3.0, 2.0])
    sources[side].GetConformer().SetPositions(positions)
    pairs = rxembed.coembed(*sources, fixed=fixed, threads=1)
    cleanup = [list(endpoint.unrelaxed) for endpoint in (pairs.reactant, pairs.product)]
    _forbid(monkeypatch, "coembed", "minimize")
    geometry = {"reactant_geometry": "SPL", "product_geometry": "SPL"} if metal else {}
    prepared = prepare_endpoints(pairs, n_conformers=4, multiplicity=1, **geometry)
    assert [list(endpoint.unrelaxed) for endpoint in (pairs.reactant, pairs.product)] == cleanup
    for endpoint, ensemble in zip((prepared.reactant, prepared.product), (pairs.reactant, pairs.product), strict=True):
        assert endpoint.GetNumConformers() == 1
        np.testing.assert_array_equal(
            endpoint.GetConformer().GetPositions(), ensemble.mol.GetConformer().GetPositions()
        )
    np.testing.assert_array_equal((prepared.reactant, prepared.product)[side].GetConformer().GetPositions(), positions)


def test_paths_accept_native_coembeds_with_the_same_result_type(monkeypatch):
    n_pairs = 3
    pairs = builder.rxembed.coembed("CBr.[Cl-]>>CCl.[Br-]", n=n_pairs, seed=0xF00D, threads=1)
    monkeypatch.setattr(builder.rxembed, "coembed", lambda *a, **k: pytest.fail("rebuilt a supplied batch"))
    guesser = rxts.TransitionStateGuesser(engine="mmff", multiplicity=1, n_conformers=1, n_lambda=3, n_embed=0)
    paths = guesser.paths(pairs)
    assert isinstance(paths, rxts.ReactionPaths)
    assert len(paths) == len(pairs) == n_pairs
    assert all(isinstance(path, rxts.ReactionPath) and len(path.frames) == 3 for path in paths)
    for path, pair in zip(paths, pairs, strict=True):
        np.testing.assert_array_equal(
            path.reactant.GetConformer().GetPositions(), pair.reactant.mol.GetConformer().GetPositions()
        )
        np.testing.assert_array_equal(
            path.product.GetConformer().GetPositions(), pair.product.mol.GetConformer().GetPositions()
        )


@pytest.mark.parametrize("budget", [0, True, 1.5])
def test_invalid_pair_budget_fails_before_embedding(monkeypatch, budget):
    monkeypatch.setattr(builder.rxembed, "coembed", lambda *a, **k: pytest.fail("embedded invalid request"))
    with pytest.raises(ValueError, match="n_conformers"):
        prepare_endpoints("CBr.[Cl-]>>CCl.[Br-]", multiplicity=1, n_conformers=budget)


def test_selected_isomers_are_coembedded(monkeypatch):
    """User-selected, unmapped species co-embed in rxts's order and keep their selected states."""
    rx = builder.rxembed
    _forbid(monkeypatch, "minimize")
    isomers = [rx.isomers(Chem.AddHs(rx.parse_smiles(s, remove_hs=False))).select() for s in (_BETA_R, _BETA_P)]
    prepared = prepare_endpoints(*isomers, multiplicity=1)
    assert [a.GetSymbol() for a in prepared.reactant.GetAtoms()] == [a.GetSymbol() for a in isomers[0].GetAtoms()]
    assert [a.GetSymbol() for a in prepared.product.GetAtoms()] == [a.GetSymbol() for a in prepared.reactant.GetAtoms()]
    sym = [a.GetSymbol() for a in prepared.reactant.GetAtoms()]
    assert sorted(sorted((sym[i], sym[j])) for i, j in prepared.forming_bonds) == [["C", "Pd"], ["H", "Pd"]]
    assert [sorted((sym[i], sym[j])) for i, j in prepared.breaking_bonds] == [["C", "H"]]
    own = [a.GetAtomMapNum() for a in isomers[0].GetAtoms()]  # the caller's maps, not the pinning ones
    assert [a.GetAtomMapNum() for a in prepared.reactant.GetAtoms()] == own
    assert [a.GetAtomMapNum() for a in prepared.product.GetAtoms()] == own
    pd = sym.index("Pd")
    alkene = [n.GetIdx() for n in prepared.product.GetAtomWithIdx(pd).GetNeighbors() if n.GetSymbol() == "C"]
    x = prepared.product.GetConformer().GetPositions()
    assert len(alkene) == 2
    assert all(1.9 < np.linalg.norm(x[pd] - x[c]) < 2.5 for c in alkene)  # side-on, both carbons at Pd


@pytest.mark.skipif(not gfnff.in_process_available(), reason="metal scans need linked GFN-FF")
def test_multiple_metal_pairs_keep_a_continuous_reaction_path():
    guesser = rxts.TransitionStateGuesser(engine="gfnff", multiplicity=1, n_cores=1, n_conformers=2, n_embed=4)
    paths = guesser.paths(_BETA_R, _BETA_P)
    path = paths.best()
    assert len(paths) == 2
    assert max(frame.energy for frame in path.frames) == min(max(f.energy for f in p.frames) for p in paths)
    assert not path.sampling_failures
    assert all(identity["centres"] for identity in path.endpoint_identity)
    x = path.guess().positions
    pairs = path.forming_bonds + path.breaking_bonds
    distances = sorted(np.linalg.norm(x[i] - x[j]) for i, j in pairs)
    # A shared beta-H transfer geometry: stretching C-H, forming Pd-H and the Pd-alkene contact.
    assert 1.3 < distances[0] < 1.7
    assert 1.6 < distances[1] < 2.0
    assert 2.0 < distances[2] < 2.5
    # Continuous through the peak. Near lambda = 1 the product field may rotate the alkene upright on Pd, a real
    # product rotamer change of ~0.8 A RMS that says nothing about the guess.
    peak = path.peak_index()
    for a, b in zip(path.frames[peak - 1 : peak + 1], path.frames[peak : peak + 2], strict=True):
        assert np.sqrt(np.mean((a.positions - b.positions) ** 2)) < 0.1


@pytest.mark.parametrize("placed", [0, 1])
@pytest.mark.parametrize("metal", [False, True])
def test_one_supplied_endpoint_samples_its_species(placed, metal):
    import rxembed as rx

    from rxts.utils import with_conformer

    if metal:
        states = rx.isomers("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", geometry="SPL")
        sides = [states.select(label=label) for label in ("cis", "trans")]
    else:
        sides = [parse_smiles(smiles) for smiles in ("CBr.[Cl-]", "CCl.[Br-]")]
    pair = rx.coembed(*sides, n=1, seed=1, threads=1)
    sides = [ensemble.mol for ensemble in (pair.reactant, pair.product)]
    mapping = list(reversed(range(sides[0].GetNumAtoms())))
    sides[1] = Chem.RenumberAtoms(sides[1], mapping)
    endpoint = Chem.Mol(sides[placed])
    for side in sides:
        side.RemoveAllConformers()
    coordinates = endpoint.GetConformer().GetPositions()
    coordinates += [7.0, -3.0, 2.0]
    sides[placed] = with_conformer(endpoint, coordinates)
    expected = [rx.cxsmiles(side) for side in sides]
    prepared = prepare_endpoints(*sides, mapping=mapping, multiplicity=1, n_conformers=4)
    assert prepared.reactant.GetNumConformers() == prepared.product.GetNumConformers() == 4
    endpoints = prepared.reactant, prepared.product
    in_reactant_order = coordinates if placed == 0 else coordinates[invert(mapping)]
    assert not np.array_equal(endpoints[placed].GetConformer().GetPositions(), in_reactant_order)
    np.testing.assert_array_equal(sides[placed].GetConformer().GetPositions(), coordinates)
    assert sides[1 - placed].GetNumConformers() == 0
    for mol, species in zip(endpoints, expected, strict=True):
        assert rx.cxsmiles(mol) == species
        assert np.isfinite(mol.GetConformer().GetPositions()).all()
        assert perceived_bonds(mol) == bond_set(mol)


@pytest.mark.parametrize(
    "copper",
    [
        "[15NH3:1]->[Cu+:2]<-[NH3:3]",
        "[15NH3:1]->[Cu+:2]<-[NH3:3] |atomProp:0.atomNote.s0:1.atomNote.LIN:2.atomNote.s1|",
    ],
    ids=["native", "CX"],
)
def test_copper_fragment_substitution_keeps_mapping_and_native_atom_order(copper):
    reactant = [copper, "[13CH3:4][Br:5]", "[Cl-:6]"]
    product = [copper, "[13CH3:4][Cl:6]", "[Br-:5]"]
    prepared = prepare_endpoints(
        reactant, product, reactant_geometry="linear", product_geometry="linear", multiplicity=1, seed=1
    )

    elements = ["N", "Cu", "N", *["H"] * 6, "C", "Br", *["H"] * 3, "Cl"]
    assert prepared.mapping[10] == 14
    assert prepared.mapping[14] == 10
    for mol in (prepared.reactant, prepared.product, prepared.native_states[0]):
        assert [atom.GetSymbol() for atom in mol.GetAtoms()] == elements
        assert [mol.GetAtomWithIdx(index).GetAtomMapNum() for index in (0, 1, 2, 9, 10, 14)] == [1, 2, 3, 4, 5, 6]
        assert (mol.GetAtomWithIdx(0).GetIsotope(), mol.GetAtomWithIdx(9).GetIsotope()) == (15, 13)
    product_elements = list(elements)
    product_elements[10], product_elements[14] = "Cl", "Br"
    assert [atom.GetSymbol() for atom in prepared.native_states[1].GetAtoms()] == product_elements


@pytest.mark.parametrize("kind", ["organic", "unnoted-metal", "xyz-metal"])
def test_supplied_conformational_endpoints_sample_in_one_frame(kind):
    import rxembed as rx

    if kind == "organic":
        pairs = rx.coembed("CBr.[Cl-]>>CCl.[Br-]", n=1, seed=1, threads=1)
    else:
        species = rx.isomers("CCCCN->[Pd+2](<-[Cl-])(<-[Cl-])<-N", geometry="SPL")
        pairs = rx.coembed(*(species.select(label=label) for label in ("cis", "trans")), n=1, seed=1, threads=1)
    sources = [pairs.reactant.mol, pairs.product.mol]
    if kind == "xyz-metal":
        sources = [mol_from_xyz(Chem.MolToXYZBlock(mol), charge=0) for mol in sources]
    elif kind == "unnoted-metal":
        for mol in sources:
            for atom in mol.GetAtoms():
                if atom.HasProp("atomNote"):
                    atom.ClearProp("atomNote")
    for mol in sources:
        for index, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(101 + index)
    mapping = [*range(1, sources[0].GetNumAtoms()), 0]
    sources[1] = Chem.RenumberAtoms(sources[1], mapping)
    if kind == "xyz-metal":
        sources[1].SetBoolProp(utils.FROM_COORDINATES, True)
    positions = sources[1].GetConformer().GetPositions()
    positions = positions @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) + [4.0, -3.0, 2.0]
    sources[1].GetConformer().SetPositions(positions)
    before = [Chem.Mol(mol) for mol in sources]
    charge = Chem.GetFormalCharge(sources[0])

    prepared = prepare_endpoints(
        *sources, mapping=mapping, charge=charge, multiplicity=1, solvent="water", n_cores=2, n_conformers=3
    )

    if kind == "organic":
        assert len(prepared.forming_bonds) == len(prepared.breaking_bonds) == 1
    else:
        assert prepared.forming_bonds == prepared.breaking_bonds == []
    assert prepared.mapping == mapping
    assert (prepared.charge, prepared.multiplicity, prepared.solvent, prepared.n_cores) == (charge, 1, "water", 2)
    expected = [before[0], Chem.RenumberAtoms(before[1], invert(mapping))]
    for endpoint, original in zip((prepared.reactant, prepared.product), expected, strict=True):
        assert endpoint.GetNumConformers() == 3
        assert not np.array_equal(endpoint.GetConformer().GetPositions(), original.GetConformer().GetPositions())
        assert [atom.GetAtomMapNum() for atom in endpoint.GetAtoms()] == [
            atom.GetAtomMapNum() for atom in original.GetAtoms()
        ]
        if kind != "organic":  # cis stays cis, trans stays trans
            assert rx.identity(endpoint).name == rx.identity(original).name
    for source, original in zip(sources, before, strict=True):
        np.testing.assert_array_equal(source.GetConformer().GetPositions(), original.GetConformer().GetPositions())
        assert Chem.MolToCXSmiles(source) == Chem.MolToCXSmiles(original)


def test_supplied_conformers_establish_species_without_setting_the_pair_count():
    import rxembed as rx

    states = rx.isomers("N->[Pd+2](<-[Cl-])(<-[Cl-])<-N", geometry="SPL")
    pairs = rx.coembed(*(states.select(label=label) for label in ("cis", "trans")), n=1, seed=1, threads=1)
    source, product = pairs.reactant.mol, pairs.product.mol
    source.AddConformer(source.GetConformer(), assignId=True)
    for mol in (source, product):
        for index, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(index + 1)
    prepared = prepare_endpoints(source, product, multiplicity=1, n_conformers=3)
    assert prepared.reactant.GetNumConformers() == prepared.product.GetNumConformers() == 3
    assert source.GetNumConformers() == 2


def test_nonfinite_endpoint_coordinates_fail_before_coembedding(monkeypatch):
    pairs = rxembed.coembed("CBr.[Cl-]>>CCl.[Br-]", n=1, seed=1, threads=1)
    sources = [pairs.reactant.mol, pairs.product.mol]
    sources[1].GetConformer().SetAtomPosition(0, [np.nan, 0.0, 0.0])
    before = [mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in sources]
    _forbid(monkeypatch, "coembed")
    with pytest.raises(ValueError, match="nonfinite coordinates"):
        prepare_endpoints(*sources, multiplicity=1)
    assert [mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in sources] == before


def test_known_isomer_does_not_silently_choose_an_ambiguous_partner(monkeypatch):
    source = "N->[Pd+2](<-[Cl-])(<-[Cl-])<-N"
    known = rxembed.embed(rxembed.isomers(source, geometry="SPL").select(label="cis"), n=1, seed=1).minimize().mol
    _forbid(monkeypatch, "coembed")
    with pytest.raises(ValueError, match="2 isomers fit"):
        prepare_endpoints(
            known,
            source,
            mapping=list(range(known.GetNumAtoms())),
            product_geometry="SPL",
            n_conformers=3,
            multiplicity=1,
        )


@pytest.mark.parametrize("fragment_form", ["mol", "cxsmiles"])
def test_fragment_coordinates_do_not_change_declared_graph_preparation(fragment_form):
    fragments = [rxembed.parse_smiles(text) for text in ("[13CH3:1][Br:2]", "[Cl-:3]", "C[C@H](F)Cl")]
    fragments[0] = Chem.AddHs(fragments[0])
    product = ["[13CH3:1][Cl:3]", "[Br-:2]", "C[C@H](F)Cl"]
    supplied = []
    for graph in fragments:
        mol = Chem.Mol(graph)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetPositions(np.arange(mol.GetNumAtoms() * 3).reshape(-1, 3) * 10.0)
        mol.AddConformer(conf)
        supplied.append(mol if fragment_form == "mol" else Chem.MolToCXSmiles(mol, canonical=False))
    before = (
        [mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in supplied]
        if fragment_form == "mol"
        else supplied[:]
    )

    declared = fragments
    if fragment_form == "cxsmiles":
        declared = [rxembed.parse_smiles(text, remove_hs=False) for text in supplied]
        for graph in declared:
            graph.RemoveAllConformers()
    expected = prepare_endpoints(declared, product, multiplicity=1, seed=1)
    prepared = prepare_endpoints(supplied, product, multiplicity=1, seed=1)

    assert prepared.mapping == expected.mapping
    assert prepared.forming_bonds == expected.forming_bonds
    assert prepared.breaking_bonds == expected.breaking_bonds
    for result, reference in zip(
        (prepared.reactant, prepared.product), (expected.reactant, expected.product), strict=True
    ):
        np.testing.assert_array_equal(result.GetConformer().GetPositions(), reference.GetConformer().GetPositions())
        assert Chem.MolToCXSmiles(result) == Chem.MolToCXSmiles(reference)
        assert result.GetAtomWithIdx(0).GetIsotope() == 13
    after = (
        [mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for mol in supplied] if fragment_form == "mol" else supplied
    )
    assert after == before
    assert all(mol.GetNumConformers() == 0 for mol in fragments)


def test_metal_mol_fragments_still_require_one_combined_molecule():
    metal = rxembed.isomers("[NH3]->[Cu+]<-[NH3]", geometry="linear").select()
    with pytest.raises(ValueError, match="requires one combined molecule or selected species"):
        prepare_endpoints([metal, "CBr", "[Cl-]"], [metal, "CCl", "[Br-]"], multiplicity=1)


def test_scalar_cxsmiles_keeps_its_existing_geometry_contract():
    pair = rxembed.coembed((_BETA_R, _BETA_P), n=1, seed=7, threads=1)
    texts = []
    for mol in (pair.reactant.mol, pair.product.mol):
        mol.GetConformer().SetPositions(mol.GetConformer().GetPositions() + np.array([4.0, -3.0, 2.0]))
        for index, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(index + 1)
        texts.append(Chem.MolToCXSmiles(mol, canonical=False))
    sources = [rxembed.parse_smiles(text, remove_hs=False) for text in texts]

    expected = prepare_endpoints(*sources, multiplicity=1, seed=1)
    prepared = prepare_endpoints(*texts, multiplicity=1, seed=1)

    assert prepared.mapping == expected.mapping
    for result, reference in zip(
        (prepared.reactant, prepared.product), (expected.reactant, expected.product), strict=True
    ):
        np.testing.assert_array_equal(result.GetConformer().GetPositions(), reference.GetConformer().GetPositions())
        assert Chem.MolToCXSmiles(result) == Chem.MolToCXSmiles(reference)
