"""Draw mapped mechanisms and prepare reaction-path data for visualization."""

from __future__ import annotations

import io
from math import ceil
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from rxts.mapping import _mechanism_mols
from rxts.structure import method_label
from rxts.utils import describe

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from rxts.structure import ReactionPath

_FORMING = (0.05, 0.50, 0.45)  # bond forming
_BREAKING = (0.55, 0.10, 0.15)  # bond breaking


def _sketch(rea, mech):
    """Build a reactant sketch highlighting forming and breaking bonds."""
    forming = [tuple(sorted(pair)) for pair in mech.forming]
    breaking = [tuple(sorted(pair)) for pair in mech.breaking]
    reactive = {i for pair in forming + breaking for i in pair}

    work = Chem.RWMol(rea)
    # Note each reactive atom's index so it survives the renumbering when hydrogens are removed.
    for i in reactive:
        work.GetAtomWithIdx(i).SetProp("atomNote", str(i))

    # Kekulize before removing spectator H atoms, which can erase aromatic N-H valence.
    Chem.Kekulize(work, clearAromaticFlags=False)

    # A forming bond is added so it can be drawn; a breaking one already exists.
    for a, b in forming:
        work.AddBond(a, b, Chem.BondType.SINGLE)

    # Descending indices preserve reactive H atoms and the remaining atom indices.
    spectators = [
        atom.GetIdx() for atom in rea.GetAtoms() if atom.GetAtomicNum() == 1 and atom.GetIdx() not in reactive
    ]
    for h in sorted(spectators, reverse=True):
        work.RemoveAtom(h)

    mol = work.GetMol()
    # This drawing graph has partial reaction bonds; retain ring/aromatic layout without valence checks.
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
    )
    rdDepictor.Compute2DCoords(mol)

    old_to_new = {int(a.GetProp("atomNote")): a.GetIdx() for a in mol.GetAtoms() if a.HasProp("atomNote")}
    bonds, colours = [], {}
    for pairs, colour in ((forming, _FORMING), (breaking, _BREAKING)):
        for a, b in pairs:
            idx = mol.GetBondBetweenAtoms(old_to_new[a], old_to_new[b]).GetIdx()
            bonds.append(idx)
            colours[idx] = colour
    return mol, bonds, colours


def draw_mechanisms(
    reaction: str | Path | Chem.Mol,
    product: str | Path | Chem.Mol | None = None,
    add_relay: str | None = None,
    size: tuple[int, int] = (400, 340),
    per_row: int = 2,
    *,
    charge: int | None = None,
    relay: str | None = None,
    relay_map: int | list[int] | tuple[int, ...] | None = None,
) -> Image.Image:
    """Draw the ordered mechanisms with green forming and red breaking bonds.

    Relay controls and panel order match mechanisms().
    """
    entries = list(_mechanism_mols(reaction, product, add_relay, charge=charge, relay=relay, relay_map=relay_map))

    mols, bonds, bond_colours, legends = [], [], [], []
    for i, (rea, _pro, mech) in enumerate(entries):
        mol, bond_hl, colours = _sketch(rea, mech)
        mols.append(mol)
        bonds.append(bond_hl)
        bond_colours.append(colours)
        legends.append(str(i))

    per_row = min(per_row, len(mols))
    rows = ceil(len(mols) / per_row)
    width, height = size
    drawer = rdMolDraw2D.MolDraw2DCairo(width * per_row, height * rows, width, height)
    options = drawer.drawOptions()
    options.legendFontSize = 20
    options.continuousHighlight = False
    options.highlightBondWidthMultiplier = 9
    drawer.DrawMolecules(
        mols,
        legends=legends,
        highlightAtoms=[[]] * len(mols),  # no atom halos -- the reactant stays plain
        highlightBonds=bonds,
        highlightBondColors=bond_colours,
    )
    drawer.FinishDrawing()

    from PIL import Image

    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def profile_data(path: ReactionPath) -> dict[str, list[float]]:
    """Relative FF and available physical energy series, in kcal/mol."""
    data = {f"{method_label(path.engine)} (scan)": [float(v) for v in path.ff_energies]}
    qm_energies = path.qm_energies
    if qm_energies is not None:
        data[f"{path.qm_label} (single points)"] = [float(v) for v in qm_energies]
    return data


def bond_data(path: ReactionPath) -> dict[str, list[float]]:
    """Reactive bond-length series, in angstrom."""
    return {
        f"{describe(path.mol, pair)} ({kind})": [float(v) for v in series]
        for (kind, pair), series in path.bond_lengths().items()
    }
