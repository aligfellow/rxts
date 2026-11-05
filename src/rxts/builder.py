"""Coembed selected endpoint species, or reuse an explicit rxembed pair batch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rxembed
from rdkit import Chem

from rxts.mapping import (
    _normalise_pairs,
    _select_mapping,
    _validate_bonds,
    invert,
    reaction_bonds,
    split_reaction_smiles,
)
from rxts.structure import Geometry
from rxts.utils import (
    FROM_COORDINATES,
    _validate_declared_context,
    _worker_count,
    describe,
    is_xyz_file,
    metal_atoms,
    mol_from_xyz,
    validate_cores,
    with_conformer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = logging.getLogger(__name__)


def _metal_identity(species, product_mapping=None):
    """Copy a selected rxembed species' coordination identity into JSON-safe atom-index data."""
    if species is None:
        return None

    def indices(values):
        values = [int(i) for i in values]
        if product_mapping is None:
            return values
        if any(i < 0 or i >= len(product_mapping) for i in values):
            raise ValueError("Selected metal identity refers to an atom outside the mapped product")
        return [int(product_mapping[i]) for i in values]

    centres = []
    for state in rxembed.identity(species).centres:
        vertices = [
            None
            if site is None
            else {"atoms": [int(i) for i in site.atoms], "mapped_atoms": indices(site.atoms), "winding": site.winding}
            for site in state.vertices
        ]
        centres.append(
            {
                "atom": int(state.atom),
                "mapped_atom": indices([state.atom])[0],
                "atomic_number": int(state.atomic_num),
                "charge": int(state.charge),
                "geometry": str(state.geometry),
                "vertices": vertices,
                "hand": str(state.hand),
            }
        )
    return {"centres": centres}


@dataclass(frozen=True)
class PreparedReaction:
    """Mapped endpoints in reactant atom order, ready for FF interpolation.

    Conformer positions pair the two endpoints; the first pair defines the frozen FF fields.
    Selected metal species keep their own atom order; `mapping` takes the product's to the reactant's."""

    reactant: Chem.Mol
    product: Chem.Mol
    forming_bonds: list[tuple[int, int]]
    breaking_bonds: list[tuple[int, int]]
    mapping: list[int]
    charge: int
    multiplicity: int
    solvent: str | None
    n_cores: int | None
    native_states: tuple[Chem.Mol | None, Chem.Mol | None] = field(default=(None, None), repr=False)
    endpoint_identity: tuple[dict | None, dict | None] = (None, None)

    @property
    def endpoints(self) -> tuple[Atoms, ...]:
        """Return the first pair as independent ASE endpoints with reaction and source identity metadata."""
        return tuple(
            Geometry(
                mol=mol,
                method="prepared",
                forming_bonds=list(self.forming_bonds),
                breaking_bonds=list(self.breaking_bonds),
                charge=self.charge,
                multiplicity=self.multiplicity,
                solvent=self.solvent,
                n_cores=self.n_cores,
                mapping=list(self.mapping),
                endpoint_identity=self.endpoint_identity,
                origin=f"rxts prepared {side} endpoint",
            ).to_atoms()
            for side, mol in (("reactant", self.reactant), ("product", self.product))
        )


def _check_same_atoms(rea, pro):
    """Require both endpoints to contain the same elements and atom counts."""
    left = sorted(a.GetSymbol() for a in rea.GetAtoms())
    right = sorted(a.GetSymbol() for a in pro.GetAtoms())
    if left != right:
        raise ValueError(
            "The two XYZ files are not the same molecule -- a reaction has to conserve atoms. "
            f"They hold [{' '.join(left)}] and [{' '.join(right)}]."
        )


def _declared_endpoint(spec, charge=None):
    """Read each declared graph in fragment order, with every hydrogen explicit.

    Fragment lists use the parsed graphs, including CX stereo, and discard conformers on copies. A complete Mol
    keeps its conformer, and hydrogens it leaves implicit are placed on it.
    """
    if _is_xyz_input(spec):
        mol = Chem.Mol(spec) if isinstance(spec, Chem.Mol) else mol_from_xyz(spec, charge=charge)
        return None, mol
    fragments = isinstance(spec, (list, tuple))
    mols = []
    for part in spec if fragments else [spec]:
        if isinstance(part, Chem.Mol):
            if fragments and metal_atoms(part):
                raise ValueError("Native metal preparation requires one combined molecule or selected species")
            mol = Chem.Mol(part)
        elif isinstance(part, str):
            mol = rxembed.parse_smiles(part, remove_hs=False)
        else:
            raise TypeError("Expected reaction SMILES, RDKit molecules or XYZ files")
        if fragments:
            mol.RemoveAllConformers()
        mols.append(Chem.AddHs(mol, addCoords=bool(mol.GetNumConformers())))
    charge = sum(atom.GetFormalCharge() for mol in mols for atom in mol.GetAtoms())
    graph = mols[0]
    for fragment in mols[1:]:
        graph = Chem.CombineMols(graph, fragment)
    Chem.GetSymmSSSR(graph)
    return charge, graph


def _is_xyz_input(spec):
    """Whether this input is XYZ text, a file, or a coordinate-perceived molecule."""
    return (
        isinstance(spec, Path)
        or is_xyz_file(spec)
        or (isinstance(spec, str) and "\n" in spec)
        or (isinstance(spec, Chem.Mol) and spec.HasProp(FROM_COORDINATES))
    )


def select_metal_state(graph, geometry=None):
    """Select the one metal species a declared `graph` can be, in its atom order; None without a metal.

    A conformer establishes the species but is not kept; a geometry name selects a shape only. Several fitting
    species raise rather than choose.
    """
    metals = metal_atoms(graph)
    if not metals:
        if geometry is not None:
            raise ValueError("Endpoint geometry requires a metal coordination state")
        return None
    observed = bool(graph.GetNumConformers())
    stated = all(graph.GetAtomWithIdx(m).HasProp("atomNote") for m in metals)
    if geometry is None and not (stated or observed):
        raise ValueError("Coordinate-free metal endpoints need geometry= or a selected species or CX state")
    found = rxembed.isomers(graph, geometry=geometry, observed=observed)
    if len(found) > 1:
        labels = ", ".join(
            "/".join(filter(None, rxembed.identity(state).labels)) or f"#{k}" for k, state in enumerate(found)
        )
        raise ValueError(f"{len(found)} isomers fit ({labels}); pass one from rxembed.isomers(...).select(...)")
    return found.select()


def _mapped(mol, maps):
    """Return a copy of an endpoint species or graph whose atom maps pin rxts's mapping."""
    out = Chem.Mol(mol)
    Chem.GetSymmSSSR(out)  # combined fragments carry no ring info
    for atom, number in zip(out.GetAtoms(), maps, strict=True):
        atom.SetAtomMapNum(number)
    return out


def _coembed_endpoints(sides, mapping, maps, seed, n_conformers, n_cores):
    """Sample paired conformers of the selected species in reactant atom order, carrying the caller's `maps`."""
    n = len(mapping)
    sides = (_mapped(sides[0], range(1, n + 1)), _mapped(sides[1], [r + 1 for r in mapping]))
    pairs = rxembed.coembed(sides, n=n_conformers, seed=seed, threads=_worker_count(n_conformers, n_cores))
    return tuple(_mapped(mol, maps) for mol in _paired_endpoints(pairs))


def _paired_endpoints(pairs):
    """Return both endpoints of a coembedded batch; conformer k of one pairs with conformer k of the other."""
    if not pairs.ids:  # `ids` also refuses a batch whose endpoints were selected apart
        raise ValueError("rxembed co-embedded no pair of these endpoints; check that each side embeds alone")
    endpoints = pairs.reactant.mol, pairs.product.mol
    for mol in endpoints:
        for k, conf in enumerate(mol.GetConformers()):  # pair k is conformer k of both
            if not np.isfinite(conf.GetPositions()).all():
                raise ValueError("The coembedded batch has nonfinite coordinates")
            conf.SetId(k)
    return endpoints


def prepare_endpoints(
    reactant: str | Path | Chem.Mol | Sequence[str | Chem.Mol] | rxembed.Coembedded,
    product: str | Path | Chem.Mol | Sequence[str | Chem.Mol] | None = None,
    *,
    seed: int = 0xF00D,
    n_conformers: int = 1,
    forming_bonds: list[tuple[int, int]] | None = None,
    breaking_bonds: list[tuple[int, int]] | None = None,
    mapping: list[int] | None = None,
    reactant_geometry: str | None = None,
    product_geometry: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    solvent: str | None = None,
    n_cores: int | None = None,
) -> PreparedReaction:
    """Map endpoint inputs, or attach reaction metadata to a coembedded batch.

    Endpoint inputs request `n_conformers` coembedded pairs in the reactant's atom order;
    fewer pairs may pass rxembed's gates. Coordinates can establish the selected isomer but are not held fixed.
    Geometry names select coordination shapes; coembedding constructs and relaxes both endpoints together.
    Fragment lists contribute graphs; their conformers are discarded on copies before construction.
    A native `Coembedded` batch supplies its own pair count and shared atom order without re-embedding.
    """
    _validate_declared_context(charge, multiplicity)
    if multiplicity is None:
        raise ValueError("prepare_endpoints requires a declared multiplicity; pass multiplicity=...")
    if n_cores is not None:
        validate_cores(n_cores)
    if isinstance(n_conformers, bool) or not isinstance(n_conformers, Integral) or n_conformers < 1:
        raise ValueError("n_conformers must be an int >= 1")

    supplied_pairs = None
    if isinstance(reactant, rxembed.Coembedded):
        if product is not None:
            raise ValueError("A coembedded result already contains both endpoints; omit product")
        supplied_pairs = _paired_endpoints(reactant)
        identity = list(range(supplied_pairs[0].GetNumAtoms()))
        if mapping is not None and list(mapping) != identity:
            raise ValueError("Coembedded endpoints already share atom order; omit mapping")
        mapping = identity
        reactant, product = (with_conformer(mol, mol.GetConformer().GetPositions()) for mol in supplied_pairs)

    if product is None:
        split = split_reaction_smiles(reactant)
        if split is None:
            raise ValueError(
                "Give both reactants and products, or a single reaction SMILES "
                'containing ">>", e.g. "CC(=O)OCC.O>>CC(=O)O.OCC".'
            )
        reactant, product = split

    reactant_xyz, product_xyz = _is_xyz_input(reactant), _is_xyz_input(product)
    if reactant_xyz != product_xyz:
        raise ValueError("Supply both endpoints as XYZ, or read them as RDKit/rxembed molecules before preparation")
    if charge is None and reactant_xyz:
        raise ValueError("XYZ endpoints do not declare charge; pass charge=...")
    reactant_charge, reactant_graph = _declared_endpoint(reactant, charge)
    product_charge, product_graph = _declared_endpoint(product, charge)
    for graph in (reactant_graph, product_graph):
        if any(not np.isfinite(conf.GetPositions()).all() for conf in graph.GetConformers()):
            raise ValueError("Endpoint has nonfinite coordinates")
    declared_charges = [value for value in (reactant_charge, product_charge) if value is not None]
    if len(set(declared_charges)) > 1:
        raise ValueError(f"Reactant and product formal charges differ ({reactant_charge:+d} and {product_charge:+d})")
    if charge is None:
        if not declared_charges:
            raise ValueError("Reaction topology does not declare formal charges; pass charge=...")
        charge = declared_charges[0]
    for label, topology_charge in (("reactant", reactant_charge), ("product", product_charge)):
        if topology_charge is not None and topology_charge != charge:
            raise ValueError(f"charge={charge:+d} conflicts with the {label} formal charge {topology_charge:+d}")

    n_atoms = reactant_graph.GetNumAtoms()
    forming_bonds = _normalise_pairs(forming_bonds, "forming_bonds", n_atoms)
    breaking_bonds = _normalise_pairs(breaking_bonds, "breaking_bonds", n_atoms)
    if reactant_xyz:
        _check_same_atoms(reactant_graph, product_graph)
    mapping = _select_mapping(reactant_graph, product_graph, mapping, forming_bonds, breaking_bonds)
    _validate_bonds(reactant_graph, Chem.RenumberAtoms(product_graph, invert(mapping)), forming_bonds, breaking_bonds)

    rea_state = select_metal_state(reactant_graph, reactant_geometry)
    pro_state = select_metal_state(product_graph, product_geometry)
    if supplied_pairs is None:
        sides = [reactant_graph if rea_state is None else rea_state, product_graph if pro_state is None else pro_state]
        maps = [atom.GetAtomMapNum() for atom in reactant_graph.GetAtoms()]
        rea, pro = _coembed_endpoints(sides, mapping, maps, seed, n_conformers, n_cores)
    else:
        rea, pro = supplied_pairs
    forming, breaking = reaction_bonds(rea, pro, list(range(rea.GetNumAtoms())))
    if forming_bonds is not None:
        forming = {tuple(sorted(p)) for p in forming_bonds}
    if breaking_bonds is not None:
        breaking = {tuple(sorted(p)) for p in breaking_bonds}

    logger.info(
        f"Reaction: {len(breaking)} bond(s) breaking, {len(forming)} forming"
        f"  |  {rea.GetNumAtoms()} atoms, charge {charge:+d}"
    )
    for pair in sorted(breaking):
        logger.info(f"  breaking  {describe(rea, pair)}")
    for pair in sorted(forming):
        logger.info(f"  forming   {describe(rea, pair)}")

    return PreparedReaction(
        reactant=rea,
        product=pro,
        forming_bonds=sorted(forming),
        breaking_bonds=sorted(breaking),
        mapping=list(mapping),
        charge=charge,
        multiplicity=multiplicity,
        solvent=solvent,
        n_cores=n_cores,
        native_states=(rea_state, pro_state),
        endpoint_identity=(_metal_identity(rea_state), _metal_identity(pro_state, mapping)),
    )
