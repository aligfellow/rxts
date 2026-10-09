"Reaction drawings and plotting data."

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rxts
from rxts.draw import _sketch, draw_mechanisms
from rxts.mapping import _mechanism_mols


def test_draws_one_panel_per_mechanism():
    img = draw_mechanisms("CC(=O)OC.O>>CC(=O)O.CO")
    # three mechanisms, two per row -> two rows tall
    assert img.size == (400 * 2, 340 * 2)


def test_added_relay_draws():
    img = draw_mechanisms("CC(=O)OC.O>>CC(=O)O.CO", add_relay="O")
    assert img.size[0] > 0
    assert img.size[1] > 0


def test_existing_relay_draws():
    reaction = (
        "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[OH2:99].[OH2:100]>>[CH3:1][C:2](=[O:3])[OH:99].[CH3:5][OH:4].[OH2:100]"
    )
    img = draw_mechanisms(reaction, relay="O")
    assert img.size[0] > 0
    assert img.size[1] > 0


def test_an_aromatic_nh_ring_still_draws():
    """The N-H hydrogen is a spectator; removing it must not cost the ring its Kekule structure."""
    img = draw_mechanisms("[I-].O=c1ccc(OCCCl)c[nH]1>>O=c1ccc(OCCI)c[nH]1.[Cl-]")
    assert img.size == (400, 340)


def test_draw_endpoint_structures():
    rea, pro, _mech = next(iter(_mechanism_mols("CC(=O)OC.O>>CC(=O)O.CO")))
    img = draw_mechanisms(rea, pro)
    assert img.size[0] > 0
    assert img.size[1] > 0


def test_drawing_preserves_endpoint_bonds():
    from math import cos, pi, sin

    from rxts.utils import mol_from_xyz

    rows = [
        f"{el} {r * cos(k * pi / 3):.3f} {r * sin(k * pi / 3):.3f} 0.0"
        for r, el in ((1.39, "C"), (2.48, "H"))
        for k in range(6)
    ]
    benzene = mol_from_xyz("12\n\n" + "\n".join(rows), charge=0)
    assert sum(bond.GetIsAromatic() for bond in benzene.GetBonds()) == 6  # the ring came back aromatic
    assert sum(atom.GetNumImplicitHs() for atom in benzene.GetAtoms()) == 0  # no invented "phantom" hydrogens
    assert sum(atom.GetNumRadicalElectrons() for atom in benzene.GetAtoms()) == 0  # nor radical dots


def test_drawing_colors_bonds_by_fate():
    "Check every changing bond is highlighted in its forming or breaking color."
    from rxts.draw import _BREAKING, _FORMING

    rea, _, mech = next(_mechanism_mols("CC(=O)OC.O>>CC(=O)O.CO", add_relay="O"))
    mol, bonds, colours = _sketch(rea, mech)
    assert len(bonds) == len(mech.forming) + len(mech.breaking) == 6
    assert set(colours) == set(bonds)  # every changing bond is coloured

    noted = {int(a.GetProp("atomNote")): a.GetIdx() for a in mol.GetAtoms() if a.HasProp("atomNote")}

    def colour_of(pair):
        i, j = pair
        return colours[mol.GetBondBetweenAtoms(noted[int(i)], noted[int(j)]).GetIdx()]

    assert _FORMING != _BREAKING, "one colour for both fates draws a mechanism that cannot be read"
    assert {colour_of(pair) for pair in mech.forming} == {_FORMING}
    assert {colour_of(pair) for pair in mech.breaking} == {_BREAKING}


def test_draw_missing_relay():
    with pytest.raises(ValueError, match="nothing to relay"):
        draw_mechanisms("CCCl.[OH-]>>CCO.[Cl-]", add_relay="O")


def test_plain_smiles_is_refused():
    "A non-reaction SMILES must be rejected before drawing."
    with pytest.raises(ValueError, match=">>"):
        draw_mechanisms("CCO")


@pytest.fixture(scope="module")
def sn2():
    """A five-point force-field scan of `CH3Br + Cl-`: the smallest path with two reactive bonds."""
    tsg = rxts.TransitionStateGuesser(engine="auto", multiplicity=1, n_conformers=1, n_embed=0)
    tsg.n_lambda = 5
    return tsg.find_transition_state(["C[Br]", "[Cl-]"], ["C[Cl]", "[Br-]"])


def test_plot_labels_match_surfaces(sn2):
    energies = rxts.draw.profile_data(sn2)
    assert list(energies) == ["MMFF94 (scan)"], "an unscored path must offer exactly one curve, named for it"
    assert energies["MMFF94 (scan)"] == [pytest.approx(v) for v in sn2.ff_energies]

    # Distinct profiles ensure each label carries its own surface's values.
    scored = SimpleNamespace(engine="mmff", ff_energies=[0.0, 5.0], qm_energies=[0.0, 9.0], qm_label="GFN2")
    curves = rxts.draw.profile_data(scored)
    assert list(curves) == ["MMFF94 (scan)", "GFN2 (single points)"], "one curve per surface, each named for it"
    assert curves["GFN2 (single points)"] == [0.0, 9.0], "the QM label must carry the QM numbers"
    assert curves["MMFF94 (scan)"] == [0.0, 5.0]

    bonds = rxts.draw.bond_data(sn2)
    assert sorted(bonds) == ["C0-Br1 (breaking)", "C0-Cl5 (forming)"]
    assert bonds["C0-Br1 (breaking)"][-1] > bonds["C0-Br1 (breaking)"][0]
    assert all(len(series) == len(sn2.frames) for series in bonds.values())
