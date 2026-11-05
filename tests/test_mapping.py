"""Check mapped mechanisms and the anchors they share."""

from __future__ import annotations

import logging
from functools import reduce

import numpy as np
import pytest
from rdkit import Chem

from rxts import mapping
from rxts.mapping import (
    Mechanism,
    _anchors,
    distinct_mechanisms,
    invert,
    map_reaction,
    map_smiles,
    mechanisms,
    parse_smiles,
    reaction_bonds,
    split_reaction_smiles,
)


def react(reactants, products):
    rea = reduce(Chem.CombineMols, (parse_smiles(fragment) for fragment in reactants))
    pro = reduce(Chem.CombineMols, (parse_smiles(fragment) for fragment in products))
    mech, tied = map_reaction(rea, pro)
    return rea, mech, tied


def elements(mol, pairs):
    return sorted(tuple(sorted(mol.GetAtomWithIdx(int(i)).GetSymbol() for i in pair)) for pair in pairs)


def test_sn2_swaps_one_bond_for_another():
    rea, mech, _ = react(["C[Br]", "[Cl-]"], ["C[Cl]", "[Br-]"])
    assert elements(rea, mech.breaking) == [("Br", "C")]
    assert elements(rea, mech.forming) == [("C", "Cl")]


def test_pericyclic_array_tracks_closed_ring():
    from rxts.mapping import _pericyclic_array

    # Two changing links plus four unchanged links form a synthetic six-node ring.
    mechanism = Mechanism([], frozenset({(2, 3)}), frozenset({(0, 1)}))
    bonds = {(0, 1), (1, 4), (4, 5), (5, 2), (2, 3), (3, 0)}
    assert _pericyclic_array(mechanism, bonds) == 6

    mechanism = Mechanism([], frozenset({(0, 2)}), frozenset({(0, 1)}))
    assert _pericyclic_array(mechanism, {(0, 1)}) == 0


@pytest.mark.parametrize(
    ("reactant", "product", "breaking"),
    [("C=COCC=C", "C=CCCC=O", (2, 3)), ("C=CC=COCC=C", "O=CC=CCCC=C", (4, 5))],
)
def test_mapper_prefers_six_atom_array(reactant, product, breaking):
    _, mechanism, tied = react([reactant], [product])
    assert mechanism.forming == {(0, 5)}
    assert mechanism.breaking == {breaking}
    assert len(tied) == 2


def test_bond_and_proton_transfer():
    rea, mech, _ = react(["CC(=O)OCC", "O"], ["CC(=O)O", "OCC"])
    assert elements(rea, mech.breaking) == [("C", "O"), ("H", "O")]
    assert elements(rea, mech.forming) == [("C", "O"), ("H", "O")]


@pytest.mark.parametrize(
    ("reactants", "products", "message"),
    [
        ("[CH3:1][CH3:1]", "[CH3:1][CH3:2]", "appears more than once"),
        ("[13CH3:1]C", "[CH3:1][13CH3]", "isotope"),
        ("[CH3:1][Br:2].[Cl-:3]", "[CH3:1][Cl:2].[Br-:3]", "Br.*Cl"),
    ],
)
def test_atom_map_identity(reactants, products, message):
    with pytest.raises(ValueError, match=message):
        react([reactants], [products])


def test_automatic_mapping_preserves_isotopes():
    rea, pro = parse_smiles("[13CH3]C"), parse_smiles("C[13CH3]")
    mechanism, _ = map_reaction(rea, pro)
    assert [
        (pro.GetAtomWithIdx(p).GetAtomicNum(), pro.GetAtomWithIdx(p).GetIsotope()) for p in range(pro.GetNumAtoms())
    ] == [(rea.GetAtomWithIdx(r).GetAtomicNum(), rea.GetAtomWithIdx(r).GetIsotope()) for r in mechanism.mapping]


def test_atom_count_must_be_conserved():
    with pytest.raises(ValueError, match="conserve atoms"):
        react(["CC(=O)OCC", "O"], ["CC(=O)O"])


def test_mapped_sn2():
    rea, mech, _ = react(["[CH3:1][Br:2].[Cl-:3]"], ["[CH3:1][Cl:3].[Br-:2]"])
    assert elements(rea, mech.breaking) == [("Br", "C")]
    assert elements(rea, mech.forming) == [("C", "Cl")]


def test_same_element_reordering():
    rea = Chem.RWMol()
    for symbol in ("C", "O", "O", "H"):
        rea.AddAtom(Chem.Atom(symbol))
    rea.AddBond(0, 1, Chem.BondType.SINGLE)
    rea.AddBond(1, 3, Chem.BondType.SINGLE)
    pro = Chem.RWMol(rea)
    pro.RemoveBond(1, 3)
    pro.AddBond(2, 3, Chem.BondType.SINGLE)
    for graph in (rea, pro):
        graph.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(graph)

    def changes(product):
        mechanism, _ = map_reaction(rea, product)
        ordered = Chem.RenumberAtoms(product, invert(mechanism.mapping))
        return reaction_bonds(rea, ordered, list(range(rea.GetNumAtoms())))

    assert changes(pro) == ({(2, 3)}, {(1, 3)})
    assert changes(Chem.RenumberAtoms(pro, [0, 2, 1, 3])) == changes(pro)


def test_distinct_mechanisms():
    rea = parse_smiles("CC(=O)OCC.O")
    pro = parse_smiles("CC(=O)O.OCC")

    _, tied = map_reaction(rea, pro)
    distinct = distinct_mechanisms(rea, tied)
    assert len(distinct) == 3  # really distinct routes stay apart
    broken = {pair for mech in distinct for pair in mech.breaking}
    assert {(1, 3), (3, 4)} <= broken


_SNAR_ASSISTED = "CCOC(=O)CCCN.Clc1nc(Cl)cc(Cl)n1.CCN(CC)CC>>CCOC(=O)CCCNc1cc(Cl)nc(Cl)n1.[Cl-].CC[NH+](CC)CC"
_SNAR_ASSISTED_HEAVY_MAPS = (
    "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][CH2:7][CH2:8][NH2:9].[Cl:10][c:11]1[n:12][c:13]([Cl:14])[cH:15]"
    "[c:16]([Cl:17])[n:18]1.[CH3:19][CH2:20][N:21]([CH2:22][CH3:23])[CH2:24][CH3:25]>>[CH3:1][CH2:2][O:3]"
    "[C:4](=[O:5])[CH2:6][CH2:7][CH2:8][NH:9][c:13]1[cH:15][c:16]([Cl:17])[n:18][c:11]([Cl:10])[n:12]1."
    "[Cl-:14].[CH3:19][CH2:20][NH+:21]([CH2:22][CH3:23])[CH2:24][CH3:25]"
)

_CO = "(<-[C-]#[O+])"
_MN2_CO10 = f"[Mn]{_CO * 5}[Mn]{_CO * 5}"
_MN_AMIDO = "[Mn+]12(<-[C-]#[O+])(<-[C-]#[O+])<-[N-](CCP(C(C)C)(C(C)C)->1)CCP(C(C)C)(C(C)C)->2"
_MN_HYDRIDE = "[Mn+]12(<-[H-])(<-[C-]#[O+])(<-[C-]#[O+])<-[NH](CCP(C(C)C)(C(C)C)->1)CCP(C(C)C)(C(C)C)->2"


@pytest.mark.parametrize(
    "reaction",
    [
        _SNAR_ASSISTED,  # which N-H proton Et3N takes is not a mechanism
        _SNAR_ASSISTED_HEAVY_MAPS,  # heavy maps leave the hydrogens implicit
        "CC[N+](CC)(CC)c1cc(Cl)nc(Cl)n1.[Cl-]>>CCCl.CCN(CC)c1cc(Cl)nc(Cl)n1",  # one of three equivalent ethyls
        f"{_MN2_CO10}>>{_MN2_CO10.replace(_CO, '', 1)}.[C-]#[O+]",  # 2 x 5!^2 symmetries on connectivity
        "CC(=O)[O-].[NH4+]>>CC(=O)O.N",  # either carboxylate oxygen: one resonance form
    ],
    ids=["snar-assisted", "snar-heavy-maps", "de-ethylation", "mn2-co10-cage", "carboxylate"],
)
def test_equivalent_hydrogens_and_groups_are_one_mechanism_in_any_atom_order(reaction):
    reactants, products = split_reaction_smiles(reaction)
    rea, pro = parse_smiles(".".join(reactants)), parse_smiles(".".join(products))
    for order in (range(rea.GetNumAtoms()), reversed(range(rea.GetNumAtoms()))):
        shuffled = Chem.RenumberAtoms(rea, list(order))
        _, tied = map_reaction(shuffled, pro)
        assert len(distinct_mechanisms(shuffled, tied)) == 1


def test_unmapped_hydride_transfer_lists_its_ties_within_budget(monkeypatch):
    """Sibling hydrogens are taken in order and misplaced ones are owed up front, so a small budget holds."""
    monkeypatch.setattr(mapping, "_MAX_NODES", 10_000)  # 90k without the owed bond count, 2M+ unordered
    rea, pro = parse_smiles(f"{_MN_HYDRIDE}.CC(C)=O"), parse_smiles(f"{_MN_AMIDO}.CC(C)O")
    for seed in range(4):  # the two routes tie on ring size and atoms, so polarity, not atom order, decides
        order = list(range(rea.GetNumAtoms()))
        np.random.default_rng(seed).shuffle(order)
        shuffled = Chem.RenumberAtoms(rea, order)
        chosen, tied = map_reaction(shuffled, pro)

        def donor_and_acceptor(pair, mol=shuffled):
            h, acceptor = sorted(pair, key=lambda i: mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
            return mol.GetAtomWithIdx(h).GetNeighbors()[0].GetSymbol(), mol.GetAtomWithIdx(acceptor).GetSymbol()

        assert sorted(map(donor_and_acceptor, chosen.forming)) == [("Mn", "C"), ("N", "O")]
        assert len(distinct_mechanisms(shuffled, tied)) == 2  # hydride to O and proton to C is another graph


def test_moved_isotope_label_is_not_pinned_away_in_any_atom_order():
    """A 13C Cope changes two bonds, as a [3,3] or a [1,3] shift, whatever the atom order."""
    reactants, products = split_reaction_smiles("[13CH2]=CCCC=C>>C=C[13CH2]CC=C")
    rea, pro = parse_smiles(".".join(reactants)), parse_smiles(".".join(products))
    for seed in range(4):
        order = list(range(rea.GetNumAtoms()))
        np.random.default_rng(seed).shuffle(order)
        shuffled = Chem.RenumberAtoms(rea, order)
        chosen, tied = map_reaction(shuffled, pro)
        assert len(chosen.changing) == 2
        assert len(distinct_mechanisms(shuffled, tied)) == 2


def test_anchors_do_not_depend_on_atom_order():
    """A Meisenheimer adduct losing chloride: greedy first-match pins gave three bonds in some orders."""
    reactants, products = split_reaction_smiles(
        "CC[N+](CC)(CC)C1(Cl)C=C(Cl)[N-]C(Cl)=N1>>CC[N+](CC)(CC)c1cc(Cl)nc(Cl)n1.[Cl-]"
    )
    rea, pro = parse_smiles(".".join(reactants)), parse_smiles(".".join(products))
    for seed in range(8):
        order = list(range(rea.GetNumAtoms()))
        np.random.default_rng(seed).shuffle(order)
        chosen, _ = map_reaction(Chem.RenumberAtoms(rea, order), pro)
        assert len(chosen.changing) == 1


def test_folded_hydrogens_anchor_as_explicit_ones():
    """Hydrogens read as counts anchor exactly the atoms explicit hydrogens do (an Ugi coupling, Golden 17).

    The isocyanide N has no H and the amide N one; a bare skeleton's SMILES writes both as N unless every
    count is written, and a rim atom's count must stay out of a key as its hydrogens do.
    """
    reactants, products = split_reaction_smiles(
        "CCCCC(=O)O.Brc1nc([N+]#[C-])ccc1.NCc1ccccc1.CCCCC=O>>O.CCCCC(=O)N(Cc1ccccc1)C(C(=O)Nc1nc(Br)ccc1)CCCC"
    )
    rea, pro = parse_smiles(".".join(reactants)), parse_smiles(".".join(products))
    counts = tuple([len(h) for h in mapping._terminal_hydrogens(mol)] for mol in (rea, pro))
    explicit = {p: a for p, a in _anchors(rea, pro, {}).items() if pro.GetAtomWithIdx(p).GetAtomicNum() != 1}
    assert _anchors(rea, pro, {}, counts=counts) == explicit


def test_deuterium_is_not_merged_with_hydrogen():
    rea = parse_smiles("[2H]C([H])Cl")  # D0, C1, H2, Cl3, H4
    identity = list(range(rea.GetNumAtoms()))
    loses = [Mechanism(identity, frozenset(), frozenset({(1, h)})) for h in (0, 2, 4)]
    assert distinct_mechanisms(rea, loses) == loses[:2]  # C-D apart from C-H; the two C-H are one


def _silane_transfer():
    return (
        reduce(Chem.CombineMols, (parse_smiles(s) for s in ("C[Si](C)(C)C", "[CH3-]"))),
        reduce(Chem.CombineMols, (parse_smiles(s) for s in ("[CH2-][Si](C)(C)C", "C"))),
    )


def test_named_bonds_anchor_reordered_product():
    rea, pro = _silane_transfer()
    named = {"breaking_bonds": [(0, 5)], "forming_bonds": [(5, 17)]}  # the methyl H moves to CH3-
    shuffled = Chem.RenumberAtoms(pro, [int(i) for i in np.random.default_rng(0).permutation(pro.GetNumAtoms())])
    anchors = _anchors(rea, shuffled, {}, **named)
    assert len(anchors) == len({r for r, _ in anchors.values()}) == rea.GetNumAtoms()  # all, edited reactant
    mechanism, tied = map_reaction(rea, shuffled, **named)
    assert mechanism.changing == frozenset({(0, 5), (5, 17)})
    assert len(tied) == 1


def test_named_bond_errors(caplog):
    rea, pro = parse_smiles("CBr.[Cl-]"), parse_smiles("CCl.[Br-]")
    with pytest.raises(ValueError, match="names atom index 99"):
        map_reaction(rea, pro, forming_bonds=[(99, 100)])
    with pytest.raises(ValueError, match="not a bond of the reactant"):
        map_reaction(rea, pro, breaking_bonds=[(0, 2)])

    before = rea.GetNumBonds()
    with caplog.at_level(logging.WARNING, logger="rxts.mapping"):
        mechanism, _ = map_reaction(rea, pro, forming_bonds=[(0, 1)])
    assert "already a bond" in caplog.text
    assert "breaking_bonds" in caplog.text
    assert rea.GetNumBonds() == before
    assert mechanism.changing == frozenset({(0, 1), (0, 2)})


def labels(mech):
    return set(mech["breaking_labels"]) | set(mech["forming_labels"])


def test_water_relay():
    direct = mechanisms("CC(=O)OC.O>>CC(=O)O.CO")[0]
    relayed = mechanisms("CC(=O)OC.O>>CC(=O)O.CO", add_relay="O")[0]

    assert len(direct["forming"]) + len(direct["breaking"]) == 4
    assert len(relayed["forming"]) + len(relayed["breaking"]) == 6
    oh = [lbl for lbl in labels(relayed) if "H" in lbl]
    assert len(oh) == 4


def test_two_water_relay_cycle():
    reaction = "CC(=O)OC.O>>CC(=O)O.CO"
    relayed = mechanisms(reaction, add_relay="O.O")[0]
    rea = parse_smiles(reaction.split(">>", 1)[0])
    added = parse_smiles(relayed["smiles"].split(">>", 1)[0])

    assert len(relayed["forming"]) + len(relayed["breaking"]) == 8
    assert added.GetNumAtoms() == rea.GetNumAtoms() + 6


def test_relay_selects_the_unique_unchanged_component():
    reaction = (
        "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[OH2:99].[OH2:100]>>[CH3:1][C:2](=[O:3])[OH:99].[CH3:5][OH:4].[OH2:100]"
    )
    (relayed,) = mechanisms(reaction, relay="O")
    reactant, product = relayed["smiles"].split(">>")
    original = parse_smiles(reaction.split(">>", 1)[0])

    assert parse_smiles(reactant).GetNumAtoms() == original.GetNumAtoms()
    assert parse_smiles(product).GetNumAtoms() == original.GetNumAtoms()
    assert len(relayed["forming"]) + len(relayed["breaking"]) == 6
    assert "[O:100]" in reactant
    assert "[O:100]" in product


def test_relay_component_identity():
    reaction = "[CH3:1][OH:2].[OH-:3].[OH2:99]>>[CH3:1][O-:2].[OH2:3].[OH2:100]"
    (relayed,) = mechanisms(reaction, relay="O")

    assert len(relayed["forming"]) + len(relayed["breaking"]) == 4
    with pytest.raises(ValueError, match="must appear exactly once on each side"):
        mechanisms(reaction, relay_map=99)


def test_unchanged_dative_direction():
    def coordinated(reverse):
        mol = Chem.RWMol()
        nitrogen = mol.AddAtom(Chem.Atom("N"))
        palladium = mol.AddAtom(Chem.Atom("Pd"))
        begin, end = (palladium, nitrogen) if reverse else (nitrogen, palladium)
        mol.AddBond(begin, end, Chem.BondType.DATIVE)
        return mol.GetMol()

    reactant, same, reversed_bond = coordinated(False), coordinated(False), coordinated(True)
    same_error = mapping._changed(reactant, same, [0, 1], {0, 1})
    reverse_error = mapping._changed(reactant, reversed_bond, [0, 1], {0, 1})

    assert same_error is None
    assert reverse_error == "with changing bonds"


def test_relay_selects_unchanged_components():
    reaction = (
        "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[OH2:98].[OH2:99].[OH2:100]>>"
        "[CH3:1][C:2](=[O:3])[OH:98].[CH3:5][OH:4].[OH2:99].[OH2:100]"
    )
    (relayed,) = mechanisms(reaction, relay="O.O")
    reactant, product = relayed["smiles"].split(">>")

    assert parse_smiles(reactant).GetNumAtoms() == parse_smiles(reaction.split(">>", 1)[0]).GetNumAtoms()
    assert parse_smiles(product).GetNumAtoms() == parse_smiles(reaction.split(">>", 1)[1]).GetNumAtoms()
    assert len(relayed["forming"]) + len(relayed["breaking"]) == 8
    assert all(f"[O:{number}]" in reactant and f"[O:{number}]" in product for number in (99, 100))


def test_relay_selection_ambiguity():
    reaction = "[CH3:1][OH:2].[OH-:3].[OH2:99].[OH2:100]>>[CH3:1][O-:2].[OH2:3].[OH2:99].[OH2:100]"
    with pytest.raises(ValueError, match=r"multiple unchanged components.*relay_map="):
        mechanisms(reaction, relay="O")
    (selected,) = mechanisms(reaction, relay_map=99)
    assert len(selected["forming"]) + len(selected["breaking"]) == 4


def test_missing_relay_does_not_add_water():
    reaction = "CC(=O)OC.O>>CC(=O)O.CO"
    with pytest.raises(ValueError, match="does not match an unchanged component"):
        mechanisms(reaction, relay="O")
    with pytest.raises(ValueError, match="requested 2"):
        mechanisms(reaction, relay="O.O")


@pytest.mark.parametrize("bridge", ["OCC", "CCO"])
def test_relay_accepts_an_alcohol_bridge(bridge):
    ethanol = mechanisms("CC(=O)OC.O>>CC(=O)O.CO", add_relay=bridge)[0]
    assert len(ethanol["forming"]) + len(ethanol["breaking"]) == 6
    assert all("O" in label for label in labels(ethanol) if "H" in label)

    augmented = parse_smiles(ethanol["smiles"].split(">>")[0])
    assert augmented.GetNumAtoms() == parse_smiles("CC(=O)OC.O").GetNumAtoms() + 9


def test_relay_requires_proton_transfer():
    reaction = "[CH3:1][Br:2].[OH-:3].[OH2:99]>>[CH3:1][OH:3].[Br-:2].[OH2:99]"
    for settings in ({"add_relay": "O"}, {"relay": "O"}, {"relay_map": 99}):
        with pytest.raises(ValueError, match="nothing to relay"):
            mechanisms(reaction, **settings)


@pytest.mark.parametrize(
    ("bridge", "remedy"),
    [
        ("invalid SMILES", "relay"),
        ("CC", "protic site"),
        ("O.CC", "protic site"),
        ("CC(O)CO", "inequivalent protic sites"),
        ("[OH:1]CC[OH:2]", "inequivalent protic sites"),
    ],
)
def test_relay_failure_reason(bridge, remedy):
    with pytest.raises(ValueError, match=remedy) as caught:
        mechanisms("CC(=O)OC.O>>CC(=O)O.CO", add_relay=bridge)
    assert "nothing to relay" not in str(caught.value)


def test_equivalent_bridge_sites_are_the_same_route():
    relayed = mechanisms("CC(=O)OC.O>>CC(=O)O.CO", add_relay="OCCO")[0]
    assert len(relayed["forming"]) + len(relayed["breaking"]) == 6
    assert all("O" in label for label in labels(relayed) if "H" in label)
    (again,) = mechanisms(relayed["smiles"])
    assert len(again["forming"]) + len(again["breaking"]) == 6


def test_existing_relay_preserves_maps():
    reaction = (
        "[CH3:1][OH:2].[OH-:3].[H:150][O:99][H:151].[OH2:100]>>[CH3:1][O-:2].[OH2:3].[H:150][O:99][H:151].[OH2:100]"
    )
    (relayed,) = mechanisms(reaction, relay_map=99)
    rea_smiles, pro_smiles = relayed["smiles"].split(">>")
    rea, pro = parse_smiles(rea_smiles), parse_smiles(pro_smiles)

    def attached_hydrogen_maps(mol, atom_map):
        atom = next(a for a in mol.GetAtoms() if a.GetAtomMapNum() == atom_map)
        return {a.GetAtomMapNum() for a in atom.GetNeighbors() if a.GetAtomicNum() == 1}

    relay_h = attached_hydrogen_maps(rea, 99)
    donor_h = attached_hydrogen_maps(rea, 2)
    assert len(relayed["forming"]) + len(relayed["breaking"]) == 4
    assert len(rea.GetAtoms()) == len(parse_smiles(reaction.split(">>", 1)[0]).GetAtoms())
    assert len(pro.GetAtoms()) == len(parse_smiles(reaction.split(">>", 1)[1]).GetAtoms())
    assert attached_hydrogen_maps(pro, 3) & relay_h
    assert 150 in attached_hydrogen_maps(pro, 3)
    assert attached_hydrogen_maps(pro, 99) & donor_h
    assert attached_hydrogen_maps(pro, 100) == attached_hydrogen_maps(rea, 100)
    assert {a.GetAtomMapNum() for a in rea.GetAtoms() if a.GetAtomMapNum() == 99} == {99}
    (round_trip,) = mechanisms(relayed["smiles"])
    assert len(round_trip["forming"]) + len(round_trip["breaking"]) == 4


def test_mapped_relay_chain_identity():
    reaction = "[CH3:1][OH:2].[OH-:3].[OH2:99].[OH2:100]>>[CH3:1][O-:2].[OH2:3].[OH2:99].[OH2:100]"
    (relayed,) = mechanisms(reaction, relay_map=(99, 100))

    assert len(relayed["forming"]) + len(relayed["breaking"]) == 6
    assert relayed["smiles"].count("[O:99]") == 2
    assert relayed["smiles"].count("[O:100]") == 2


def test_added_relay_pool_capacity():
    reaction = "[CH3:1][OH:2].[CH3:5][OH:6].[OH-:3].[OH-:4]>>[CH3:1][O-:2].[CH3:5][O-:6].[OH2:3].[OH2:4]"
    direct = mechanisms(reaction)[0]
    relayed = mechanisms(reaction, add_relay="O")[0]
    reactant = parse_smiles(relayed["smiles"].split(">>", 1)[0])
    original = parse_smiles(reaction.split(">>", 1)[0])

    assert len(direct["forming"]) + len(direct["breaking"]) == 4
    assert len(relayed["forming"]) + len(relayed["breaking"]) == 8
    assert reactant.GetNumAtoms() == original.GetNumAtoms() + 3
    with pytest.raises(ValueError, match="needs 2 attached hydrogens"):
        mechanisms(reaction, add_relay="CCO")


@pytest.mark.parametrize(
    ("reaction", "relay_map", "message"),
    [
        (
            "[CH3:1][OH:2].[OH-:3].[OH2:99]>>[CH3:1][O-:2].[OH2:3].[OH2:99]",
            101,
            "exactly once",
        ),
        (
            "[CH3:1][OH:2].[OH-:3].[OH2:99].[OH2:99]>>[CH3:1][O-:2].[OH2:3].[OH2:99].[OH2:99]",
            99,
            "appears more than once",
        ),
        (
            "[OH2:9].CC(=O)OC>>CC(=O)[OH:9].CO",
            9,
            "active component",
        ),
        (
            "[CH3:1][OH:2].[CH3:5][OH:6].[OH-:3].[OH-:4].[OH-:99]>>"
            "[CH3:1][O-:2].[CH3:5][O-:6].[OH2:3].[OH2:4].[OH-:99]",
            99,
            "transfers 2 proton",
        ),
    ],
    ids=["missing-map", "duplicate-map", "active-component", "insufficient-hydrogens"],
)
def test_invalid_mapped_relay(reaction, relay_map, message):
    with pytest.raises(ValueError, match=message):
        mechanisms(reaction, relay_map=relay_map)


def test_relay_controls_are_mutually_exclusive():
    reaction = "[CH3:1][OH:2].[OH-:3].[OH2:99]>>[CH3:1][O-:2].[OH2:3].[OH2:99]"
    with pytest.raises(ValueError, match="mutually exclusive"):
        mechanisms(reaction, add_relay="O", relay_map=99)


def test_relay_preserves_cip_stereochemistry():
    same_configuration = (
        "[CH3:1][OH:2].[OH-:3].[C@H:98]([F:100])([Cl:101])[OH:99]"
        ">>[CH3:1][O-:2].[OH2:3].[OH:99][C@H:98]([Cl:101])[F:100]"
    )
    (relayed,) = mechanisms(same_configuration, relay_map=99)
    assert len(relayed["forming"]) + len(relayed["breaking"]) == 4

    opposite_configuration = same_configuration.replace("[OH:99][C@H:98]", "[OH:99][C@@H:98]")
    with pytest.raises(ValueError, match="changed atom state"):
        mechanisms(opposite_configuration, relay_map=99)


def test_mapped_structures_cannot_add_relay():
    rea, pro = parse_smiles("CC(=O)OC.O"), parse_smiles("CC(=O)O.CO")
    with pytest.raises(ValueError, match="needs a reaction SMILES"):
        mechanisms(rea, pro, add_relay="O")


def test_fully_mapped_relay_smiles():
    mapped = (
        "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[O:6]([H:7])[H:8].[O:9]([H:10])[H:11]"
        ">>[CH3:1][C:2](=[O:3])[O:6][H:7].[CH3:5][O:4][H:10].[O:9]([H:8])[H:11]"
    )
    (mech,) = mechanisms(mapped)
    assert len(mech["forming"]) + len(mech["breaking"]) == 6
    assert len([lbl for lbl in labels(mech) if "H" in lbl]) == 4


@pytest.mark.parametrize("entry", [mechanisms, map_smiles], ids=lambda f: f.__name__)
def test_reaction_required(entry):
    with pytest.raises(ValueError, match=">>") as caught:
        entry("CCO")
    assert "CCO" in str(caught.value)


def test_missing_product_names_argument():
    with pytest.raises(ValueError, match="product") as caught:
        mechanisms(parse_smiles("CCO"))

    message = str(caught.value)
    assert "Mol" in message
    assert "0x" not in message


def test_distinct_smiles_mechanism_needs_selection():
    reaction = "CC(=O)OCC.O>>CC(=O)O.OCC"
    with pytest.raises(ValueError, match="ambiguous"):
        map_smiles(reaction)

    selected = map_smiles(reaction, mechanism=0)
    assert ":1" in selected
    assert ">>" in map_smiles(reaction, strict=False)

    for mechanism in (True, -1):
        with pytest.raises(ValueError, match="mechanism must be"):
            map_smiles(reaction, mechanism=mechanism)


def test_report_atom_order():
    (report,) = mechanisms("OCCCCBr.[Cl-]>>OCCCC[Cl].[Br-]")
    reactant = parse_smiles(report["smiles"].split(">>", 1)[0])

    def label(pair):
        return "-".join(f"{reactant.GetAtomWithIdx(i).GetSymbol()}{i}" for i in pair)

    assert [label(pair) for pair in report["forming"]] == report["forming_labels"]
    assert [label(pair) for pair in report["breaking"]] == report["breaking_labels"]

    def pairs(changes):
        return [tuple(sorted(reactant.GetAtomWithIdx(i).GetSymbol() for i in pair)) for pair in changes]

    assert pairs(report["forming"]) == [("C", "Cl")]
    assert pairs(report["breaking"]) == [("Br", "C")]


def test_reaction_smiles_split():
    assert split_reaction_smiles("CC(=O)OCC.O>>CC(=O)O.OCC") == (
        ["CC(=O)OCC", "O"],
        ["CC(=O)O", "OCC"],
    )
    assert split_reaction_smiles("CBr>[Na+]>CO") == (["CBr"], ["CO"])
    assert split_reaction_smiles("CCO") is None
    metal = "[NH3]->[Pd+2](<-[Cl-])<-[NH3]"
    assert split_reaction_smiles(f"{metal}>O>{metal}") == ([metal], [metal])
    assert split_reaction_smiles(metal) is None
    cx = metal + " |$left>right..label$|"
    assert split_reaction_smiles(f"{cx}>>{cx}") == ([cx], [cx])
    assert split_reaction_smiles(cx) is None


def test_explicit_hydrogen_parsing():
    mol = parse_smiles("CC(=O)OC.O")
    assert mol.GetNumAtoms() == 14
    assert all(atom.GetAtomMapNum() == 0 for atom in mol.GetAtoms())
    mapped = parse_smiles("[O:6]([H:7])[H:8]")
    assert sorted(atom.GetAtomMapNum() for atom in mapped.GetAtoms()) == [6, 7, 8]


def test_numeric_mapping_identity():
    rea, pro = Chem.MolFromSmiles("[13C].[C]"), Chem.MolFromSmiles("[C].[13C]")
    assert mapping._validate_mapping([1, 0], rea, pro) == [1, 0]
    with pytest.raises(ValueError, match="isotope"):
        mapping._validate_mapping([0, 1], rea, pro)

    rea, pro = Chem.MolFromSmiles("[C:1].[C]"), Chem.MolFromSmiles("[C].[C:1]")
    assert mapping._validate_mapping([1, 0], rea, pro) == [1, 0]
    with pytest.raises(ValueError, match="contradicting their shared atom-map number"):
        mapping._validate_mapping([0, 1], rea, pro)


def test_mapping_needs_no_geometry_or_ff(monkeypatch):
    import sys

    for name in ("builder", "ff", "evb", "gfnff", "scoring"):
        monkeypatch.setitem(sys.modules, f"rxts.{name}", None)
    reactant, product = "CBr.[Cl-]", "CCl.[Br-]"
    reaction = f"{reactant}>>{product}"
    mapped = map_smiles(reaction, hydrogens=True)
    choices = mechanisms(reaction)
    assert mechanisms(reactant, product) == choices
    assert len(mechanisms(mapped)) == len(choices) == 1
    assert len(mechanisms("NC=O.O>>N.O=CO", add_relay="O")) == 1


def test_mapping_rejects_invalid_pairs():
    mol = parse_smiles("[CH4:1]")
    for pairs in ([(0, True)], [(0, 1.5)], [(0, 0)], [(0, 1, 2)], [(0, 1), (1, 0)], [(0, 99)]):
        with pytest.raises(ValueError, match="forming_bonds"):
            map_reaction(mol, mol, forming_bonds=pairs)


@pytest.mark.parametrize("relay", [None, "O"])
def test_mapping_retains_cx_annotations(relay):
    reaction = (
        "[CH3:1][O:2][H:3].[NH3:4] |atomProp:0.atomNote.selected|"
        ">>[CH3:1][O-:2].[NH3+:4][H:3] |atomProp:0.atomNote.selected|"
    )
    (report,) = mechanisms(reaction, add_relay=relay)
    for side in report["smiles"].split(">>"):
        mol = parse_smiles(side)
        annotated = [atom for atom in mol.GetAtoms() if atom.HasProp("atomNote")]
        assert len(annotated) == 1
        assert annotated[0].GetAtomMapNum() == 1
        assert annotated[0].GetProp("atomNote") == "selected"
    (again,) = mechanisms(report["smiles"])
    assert again["forming_labels"] == report["forming_labels"]
    assert again["breaking_labels"] == report["breaking_labels"]


def test_relay_preserves_a_protonated_donor_configuration():
    import rxembed

    reaction = "C[N-](CC)->[Pd](Cl)(Cl)<-N.CO>>C[N@H](CC)->[Pd](Cl)(Cl)<-N.C[O-]"
    (report,) = mechanisms(reaction, add_relay="O")
    expected = Chem.CombineMols(
        rxembed.parse_smiles(reaction.split(">>")[1], remove_hs=False), rxembed.parse_smiles("O", remove_hs=False)
    )
    product = parse_smiles(report["smiles"].split(">>")[1])
    for atom in product.GetAtoms():
        atom.SetAtomMapNum(0)
    assert rxembed.cxsmiles(product) == rxembed.cxsmiles(expected)
