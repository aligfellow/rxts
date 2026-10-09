"""Compatible endpoint geometry construction.

rxembed supplies molecular embeddings and minimum graphs. Reactive-complex restraints follow
VeloxChem's ReactionForceFieldBuilder; attack-face search and topology checks belong here.
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rxembed
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from rxts.mapping import (
    _normalise_pairs,
    _select_mapping,
    _validate_bonds,
    bond_set,
    invert,
    parse_smiles,
    reaction_bonds,
    split_reaction_smiles,
)
from rxts.structure import Geometry
from rxts.utils import (
    FROM_COORDINATES,
    PLACED,
    _align,
    _graph_identity,
    _validate_declared_context,
    describe,
    is_xyz_file,
    metal_atoms,
    mol_from_xyz,
    require_topology,
    set_positions,
    validate_cores,
    with_conformer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from rxembed import Conformers, Isomer

_FRAGMENT_GAP = 2.0  # A of empty space between fragments before they are pulled in
_CONTACT_SLACK = 0.5  # A beyond the vdW minimum still counts as "touching" (VeloxChem's value)
_CONTACT_K = 100.0  # kcal/mol/A^2 pulling reactive atoms into contact
_CONTACT_ATTEMPTS = 5  # retries, each loosening the flat-bottom slack, before giving up
# Soft Cartesian holds let the product topology relax without wholesale reorientation.
_PIN_K = 5.0
_PIN_K_SOFT = 1.0
_DEGENERATE = 1e-6  # a bond vector this short has no direction

logger = logging.getLogger(__name__)


def _native_conformer(result, expected=None):
    """Validate and return one relaxed native conformer."""
    if isinstance(result, rxembed.EnsembleSet):
        raise ValueError("rxembed found multiple chemical states; select one native isomer and conformer first")
    if not isinstance(result, rxembed.Conformers):
        raise TypeError("Select one rxembed isomer and conformer before preparing the reaction")
    if len(result.ids) != 1:
        raise ValueError("rxembed must return exactly one selected conformer")
    if result.unrelaxed:
        raise ValueError("rxembed returned an unrelaxed conformer; converge its native cleanup first")

    mol = result.mol
    if expected is not None and _graph_identity(mol) != _graph_identity(expected):
        raise ValueError("rxembed changed the selected topology, atom maps or electronic state")
    if mol.GetNumConformers() != 1 or not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError("rxembed returned nonfinite or multiple conformers")
    return mol


def as_molecules(spec, seed=0xF00D):
    """Embed SMILES/RDKit inputs or copy coordinate-bearing molecules."""
    if not isinstance(spec, (list, tuple)):
        spec = [spec]
    out = []
    for item in spec:
        if isinstance(item, str):
            mol = parse_smiles(item)
        elif isinstance(item, Chem.Mol):
            mol = Chem.Mol(item)
            if mol.GetNumConformers():
                out.append(mol)
                continue
            mol = Chem.AddHs(mol)
        else:
            raise TypeError(f"Expected a SMILES string or an RDKit Mol, got {type(item)}")
        result = rxembed.embed(mol, n=1, seed=int(seed), threads=1)
        out.append(_native_conformer(result, mol))
    return out


def combine_spaced(mols):
    """Merge fragments into one molecule, spaced out so nothing overlaps.

    Fragments are laid end to end along x. The gap is arbitrary; the reactive
    complex step then pulls them back together into a real geometry.
    """
    combined = None
    for original in mols:
        mol = Chem.Mol(original)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        if combined is None:
            shift = -pos[:, 0].min()
        else:
            prev = combined.GetConformer().GetPositions()
            shift = prev[:, 0].max() - pos[:, 0].min() + _FRAGMENT_GAP
        for i in range(mol.GetNumAtoms()):
            p = pos[i] + np.array([shift, 0.0, 0.0])
            conf.SetAtomPosition(i, p.tolist())
        combined = mol if combined is None else Chem.CombineMols(combined, mol)
    Chem.SanitizeMol(combined)
    return combined


def rdkit_minimiser(mol):
    """Build a minimiser for `mol`: MMFF94 where it types, UFF where it does not.

    UFF is needed because it types `[I-]`, which MMFF refuses.

    :return: (force field, "mmff" | "uff").
    """
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is not None:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, props, nonBondedThresh=1e5, ignoreInterfragInteractions=False
        )
        if ff is not None:
            return ff, "mmff"

    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, ignoreInterfragInteractions=False)
    if ff is None:
        raise ValueError(
            "Neither MMFF94 nor UFF can type this molecule, so there is nothing to "
            "build a starting geometry with. Select a native rxembed isomer with a named "
            "coordination geometry, or supply endpoint XYZ structures."
        )
    return ff, "uff"


def _contact_distance(mol, i, j):
    """Compute how far apart two atoms are when just touching.

    MMFF's R*_ij where MMFF can type the molecule (VeloxChem's choice); a sum of van der Waals radii is
    not equivalent and is only the fallback for elements MMFF cannot type.
    """
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is not None:
        return props.GetMMFFVdWParams(i, j)[2]
    table = Chem.GetPeriodicTable()
    return table.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) + table.GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())


#: Approach directions to try, spread evenly on a sphere by a Fibonacci lattice.
#: Deterministic, and each candidate costs one force-field energy.
_APPROACH_DIRECTIONS = 24
#: Axial orientations sampled for a rigid incoming fragment.
_APPROACH_SPINS = 4


def _approach_directions():
    """Evenly spread unit vectors on a sphere (Fibonacci lattice)."""
    i = np.arange(_APPROACH_DIRECTIONS) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / _APPROACH_DIRECTIONS)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def _rotation_about(axis, angle):
    """Rotate through `angle` about a unit direction."""
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= _DEGENERATE:
        raise ValueError("A rotation axis needs a direction.")
    x, y, z = axis / norm
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _rotation_onto(source, target):
    """Build the shortest rotation taking `source` onto `target`."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine <= _DEGENERATE:
        if cosine > 0.0:
            return np.eye(3)
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        return _rotation_about(np.cross(source, basis), np.pi)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + cross + (cross @ cross) * ((1.0 - cosine) / sine**2)


def _lone_pair_axis(mol, atom):
    """Point from an atom's neighbour centroid through the atom, or return `None` if undefined."""
    neighbours = [neighbour.GetIdx() for neighbour in mol.GetAtomWithIdx(int(atom)).GetNeighbors()]
    if not neighbours:
        return None
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    axis = positions[int(atom)] - positions[neighbours].mean(axis=0)
    norm = float(np.linalg.norm(axis))
    return None if norm <= _DEGENERATE else axis / norm


def _place_fragment(positions, fragment, arriving, rotation, target):
    """Rigidly place `fragment` with `arriving` exactly at `target`."""
    return (positions[fragment] - positions[int(arriving)]) @ rotation.T + target


def search_approach(mol, forming, breaking, energy):
    """Place one incoming fragment by sampling approach directions and axial rotations.

    Rank sphere and leaving-group-opposite proposals with the supplied charged FF energy.
    Intramolecular and multiple-contact reactions retain their placement; an eligible search
    without a charged potential warns and retains its original coordinates."""
    # An "incoming" atom forms a bond to a centre simultaneously breaking one -- the SN2 pattern.
    incoming: dict[int, int] = {}  # arriving atom -> the centre it attacks
    for a, c in forming:
        for centre, arriving in ((a, c), (c, a)):
            for pair in breaking:
                if centre in pair and arriving not in pair:
                    incoming.setdefault(arriving, centre)

    if len(incoming) != 1:
        return mol

    conf = mol.GetConformer()
    x0 = np.asarray(conf.GetPositions(), dtype=float)
    arriving, centre = next(iter(incoming.items()))
    fragment = np.asarray(next(part for part in Chem.GetMolFrags(mol) if arriving in part), dtype=int)
    if centre in fragment:
        return mol
    if energy is None:
        warnings.warn(
            "Attack-face search has no charged force field; install rxts[ff] or supply endpoint XYZ structures.",
            RuntimeWarning,
            stacklevel=2,
        )
        return mol

    reach = _contact_distance(mol, int(arriving), int(centre)) + _CONTACT_SLACK
    candidates = list(_approach_directions())
    # ...and the direction anti to every bond this centre is breaking; offered, not imposed.
    for pair in breaking:
        if centre not in pair:
            continue
        leaving = pair[0] if pair[1] == centre else pair[1]
        away = x0[centre] - x0[leaving]
        norm = np.linalg.norm(away)
        if norm > _DEGENERATE:
            candidates.append(away / norm)

    axis = _lone_pair_axis(mol, arriving)
    best, best_energy = None, np.inf
    for direction in candidates:
        target = x0[centre] + reach * direction
        rotations = (np.eye(3),)
        if axis is not None:
            aligned = _rotation_onto(axis, -direction)
            rotations = tuple(
                _rotation_about(direction, 2.0 * np.pi * spin / _APPROACH_SPINS) @ aligned
                for spin in range(_APPROACH_SPINS)
            )
        for rotation in rotations:
            placed = _place_fragment(x0, fragment, arriving, rotation, target)
            x = x0.copy()
            x[fragment] = placed
            try:
                candidate_energy, _ = energy(x)
                candidate_energy = float(candidate_energy)
            except Exception:
                continue
            if np.isfinite(candidate_energy) and candidate_energy < best_energy:
                best, best_energy = placed, candidate_energy

    if best is not None:
        for atom, position in zip(fragment, best, strict=True):
            conf.SetAtomPosition(int(atom), position.tolist())

    return mol


def reactive_complex(mol, contact_pairs):
    """Relax the reactive complex with flat-bottom restraints on approaching atom pairs.

    Exhausted relaxation warns and retains provisional coordinates."""
    if not contact_pairs:
        ff, _ = rdkit_minimiser(mol)
        ff.Initialize()
        if ff.Minimize(maxIts=2000) != 0:
            warnings.warn(
                "Initial force-field relaxation did not converge; coordinates remain provisional.",
                RuntimeWarning,
                stacklevel=2,
            )
        return mol

    for attempt in range(_CONTACT_ATTEMPTS):
        ff, _ = rdkit_minimiser(mol)
        slack = _CONTACT_SLACK + attempt * 0.5
        for i, j in contact_pairs:
            ff.AddDistanceConstraint(int(i), int(j), 0.0, _contact_distance(mol, int(i), int(j)) + slack, _CONTACT_K)
        ff.Initialize()
        if ff.Minimize(maxIts=2000, forceTol=1e-4) == 0:
            return mol
    warnings.warn(
        "Contact relaxation exhausted its attempts; coordinates remain provisional.", RuntimeWarning, stacklevel=2
    )
    return mol


def derive_product_geometry(reactant, product, forming, breaking):
    """Relax the product topology in the reactant frame with soft Cartesian holds.

    Try progressively weaker spectator holds if the bonding is wrong. Valid but unconverged
    cleanup warns; failure to obtain the product bonding raises."""
    intended = bond_set(product)
    reactive = {int(a) for pair in list(forming) + list(breaking) for a in pair}
    spectators = [i for i in range(product.GetNumAtoms()) if i not in reactive]

    # Hold everything first; if the reactive core cannot then reach its new bonding,
    # hold only the spectators and let the core move freely.
    ladder = [
        (range(product.GetNumAtoms()), _PIN_K),
        (spectators, _PIN_K),
        (spectators, _PIN_K_SOFT),
    ]

    failures = []
    for held, k in ladder:
        target = Chem.Mol(product)
        conf = target.GetConformer()
        source = reactant.GetConformer().GetPositions()
        for i in range(target.GetNumAtoms()):  # seed all atoms in R's frame
            conf.SetAtomPosition(i, source[i].tolist())

        ff, kind = rdkit_minimiser(target)
        # Pin each held atom near where it already is, with a soft constraint.
        pin = ff.MMFFAddPositionConstraint if kind == "mmff" else ff.UFFAddPositionConstraint
        for i in held:
            pin(int(i), 0, k)
        ff.Initialize()
        status = ff.Minimize(maxIts=2000, forceTol=1e-4)

        try:
            require_topology(target, intended, "derived product")
        except ValueError as exc:
            failures.append(str(exc).splitlines()[1])  # the first "missing/spurious bonds" line
            continue
        if status != 0:
            warnings.warn(
                "Product force-field cleanup did not converge; its bonding is valid but coordinates are provisional.",
                RuntimeWarning,
                stacklevel=2,
            )
        return target

    raise ValueError(
        "Could not derive a product geometry with the product's bonding.\n" + "\n".join(f"  {f}" for f in failures)
    )


def _metal_identity(isomer, product_mapping=None):
    """Copy selected rxembed coordination identity into JSON-safe atom-index data."""
    if isomer is None:
        return None

    def indices(values):
        values = [int(i) for i in values]
        if product_mapping is None:
            return values
        if any(i < 0 or i >= len(product_mapping) for i in values):
            raise ValueError("Selected metal identity refers to an atom outside the mapped product")
        return [int(product_mapping[i]) for i in values]

    centres = []
    for state in isomer.centres:
        vertices = []
        for site in state.vertices:
            if site is None:
                vertices.append(None)
            elif hasattr(site, "atoms"):
                atoms = [int(i) for i in site.atoms]
                vertices.append({"atoms": atoms, "mapped_atoms": indices(atoms), "winding": str(site.winding)})
            else:
                vertices.append({"atom": int(site), "mapped_atom": indices([site])[0]})
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

    Native sources and isomers retain their original index spaces and selected identities."""

    reactant: Chem.Mol
    product: Chem.Mol
    forming_bonds: list[tuple[int, int]]
    breaking_bonds: list[tuple[int, int]]
    mapping: list[int]
    charge: int
    multiplicity: int
    solvent: str | None
    n_cores: int | None
    sources: tuple[object, object] = field(repr=False)
    native_states: tuple[Isomer | None, Isomer | None] = field(default=(None, None), repr=False)
    endpoint_identity: tuple[dict | None, dict | None] = (None, None)

    @property
    def endpoints(self) -> tuple[Atoms, ...]:
        """Return independent ASE endpoints with declared reaction and source identity metadata."""
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


def _geometry_known(spec):
    """Recognise placed XYZ or selected rxembed endpoints, which bypass approach and product derivation."""
    return bool(_is_xyz_input(spec) or (isinstance(spec, Chem.Mol) and spec.HasProp(PLACED)))


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
    """Read topology and declared charge before embedding; native Isomers restore real metals."""
    if _is_xyz_input(spec):
        mol = Chem.Mol(spec) if isinstance(spec, Chem.Mol) else mol_from_xyz(spec, charge=charge)
        return None, mol
    if isinstance(spec, Chem.Mol):
        mols = [Chem.AddHs(spec) if not spec.GetNumConformers() else Chem.Mol(spec)]
    elif isinstance(spec, (list, tuple)):
        mols = []
        for part in spec:
            if isinstance(part, Chem.Mol):
                mols.append(Chem.AddHs(part) if not part.GetNumConformers() else part)
            else:
                mols.append(parse_smiles(part))
    elif isinstance(spec, (str, Path)):
        mols = [parse_smiles(spec)]
    else:
        mol = spec.restore() if isinstance(spec, rxembed.Isomer) else getattr(spec, "mol", None)
        if not isinstance(mol, Chem.Mol):
            raise TypeError("Expected reaction SMILES, RDKit molecules, XYZ files, or selected rxembed states")
        mols = [mol]
    charge = sum(atom.GetFormalCharge() for mol in mols for atom in mol.GetAtoms())
    graph = Chem.Mol(mols[0])
    for fragment in mols[1:]:
        graph = Chem.CombineMols(graph, fragment)
    return charge, graph


def _is_xyz_input(spec):
    """Whether this input is XYZ text, a file, or a coordinate-perceived molecule."""
    return (
        isinstance(spec, Path)
        or is_xyz_file(spec)
        or (isinstance(spec, str) and "\n" in spec)
        or (isinstance(spec, Chem.Mol) and spec.HasProp(FROM_COORDINATES))
    )


def _charged_potential(mol, charge, solvent):
    """Evaluate the attack-face construction energy with GFN-FF electrostatics."""
    from rxts.gfnff import GFNFF

    try:
        return GFNFF(mol, charge=charge, solvent=solvent).energy_grad
    except (RuntimeError, OSError, ImportError):
        return None


def selected_endpoint(spec, expected=None):
    """Read one selected native conformer and mark its geometry as supplied."""
    if (
        isinstance(spec, Chem.Mol)
        and spec.GetNumConformers()
        and not np.isfinite(spec.GetConformer().GetPositions()).all()
    ):
        raise ValueError("The selected endpoint has nonfinite coordinates")
    if isinstance(spec, (str, Path, Chem.Mol)) or type(spec) in (list, tuple):
        return spec
    mol = _native_conformer(spec, expected)
    mol.SetBoolProp(PLACED, True)
    return mol


def select_metal_state(spec, geometry=None):
    """Select a native metal arrangement; geometry names select a shape only.

    CX labels and Isomer inputs retain their source indexing; placed conformers keep their coordinates.
    """
    original = spec
    if isinstance(spec, (list, tuple)) and all(isinstance(part, str) for part in spec):
        spec = ".".join(spec)
    graph = spec
    if isinstance(spec, str) and not _is_xyz_input(spec):
        graph = Chem.MolFromSmiles(spec)
    if isinstance(graph, Chem.Mol):
        if not metal_atoms(graph):
            if geometry is not None:
                raise ValueError("Endpoint geometry requires a metal coordination state")
            return None, original, False
        if geometry is None and graph.HasProp(PLACED):
            return None, original, False
    elif isinstance(spec, (str, Path, list, tuple)):
        if isinstance(spec, (list, tuple)) and any(isinstance(part, Chem.Mol) and metal_atoms(part) for part in spec):
            raise ValueError("Native metal preparation requires one combined molecule or selected native Isomer")
        if geometry is not None:
            raise ValueError("Endpoint geometry requires one metal molecule or native Isomer")
        return None, original, False

    if isinstance(spec, rxembed.Conformers):
        graph = selected_endpoint(spec)
        iso = spec.iso
        if iso is None and geometry is not None:
            raise ValueError("Endpoint geometry requires a selected native metal Isomer")
        if geometry is None:
            return iso, graph, False
    elif isinstance(spec, rxembed.Isomer):
        iso = spec
        if iso.mol.GetNumConformers() > 1:
            raise ValueError("Select exactly one native Isomer conformer before preparing the reaction")
        graph = rxembed.Conformers(iso.mol, [c.GetId() for c in iso.mol.GetConformers()], iso=iso).mol
    elif isinstance(graph, Chem.Mol):
        # Native parsing binds donor notes to their bonds before AddHs appends atoms. RDKit's
        # ordinary parser loses coordinated ligand tags, so it is used only for metal detection.
        if isinstance(spec, str):
            graph = rxembed.parse_smiles(spec, remove_hs=False)
        graph = Chem.AddHs(graph, addCoords=bool(graph.GetNumConformers()))
        if geometry is None and not graph.GetNumConformers():
            metals = metal_atoms(graph)
            if not all(graph.GetAtomWithIdx(m).HasProp("atomNote") for m in metals):
                raise ValueError("Coordinate-free metal endpoints need geometry= or selected native CX/Isomer state")
        iso = rxembed.metal(
            graph,
            geometry,
            center="all" if len(metal_atoms(graph)) > 1 else None,
            observed_only=bool(graph.GetNumConformers()),
        ).select()
    else:
        raise TypeError("Select one rxembed isomer and conformer before preparing the reaction")

    if geometry is not None:
        for state in iso.centres:
            rxembed.IsomerSet([iso]).select(geometry=geometry, center=state.atom)
    return iso, graph, True


def relax_metal_endpoint(isomer, reactant, mapping):
    """Relax a copied native isomer from the mapped endpoint seed.

    rxembed restores real elements, charges and donor bonds; source identity stays unchanged."""
    seeded = deepcopy(isomer)
    reference = reactant.GetConformer().GetPositions()[mapping]
    if not np.isfinite(reference).all():
        raise ValueError("The native endpoint seed has nonfinite coordinates")
    seeded.mol = with_conformer(seeded.mol, reference)
    expected = rxembed.Conformers(seeded.mol, [0], iso=seeded).mol
    result = rxembed.minimize(seeded)
    product = selected_endpoint(result, expected)
    if [a.GetAtomicNum() for a in product.GetAtoms()] != [
        reactant.GetAtomWithIdx(int(r)).GetAtomicNum() for r in mapping
    ]:
        raise ValueError("Native endpoint cleanup changed the mapped endpoint atoms")
    return set_positions(product, _align(product.GetConformer().GetPositions(), reference))


def prepare_endpoints(
    reactant: str | Path | Chem.Mol | Sequence[str | Chem.Mol] | Conformers | Isomer,
    product: str | Path | Chem.Mol | Sequence[str | Chem.Mol] | Conformers | Isomer | None = None,
    *,
    seed: int = 0xF00D,
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
    """Map endpoints and prepare matching graphs and coordinates without scanning or QM.

    Native source objects keep their original atom order. Isomer/CX or geometry requests derive the product
    from the mapped reactant seed; placed conformers retain their coordinates.
    """
    _validate_declared_context(charge, multiplicity)
    if multiplicity is None:
        raise ValueError("prepare_endpoints requires a declared multiplicity; pass multiplicity=...")
    if n_cores is not None:
        validate_cores(n_cores)

    if product is None:
        split = split_reaction_smiles(reactant)
        if split is None:
            raise ValueError(
                "Give both reactants and products, or a single reaction SMILES "
                'containing ">>", e.g. "CC(=O)OCC.O>>CC(=O)O.OCC".'
            )
        reactant, product = split

    sources = (reactant, product)
    reactant_xyz, product_xyz = _is_xyz_input(reactant), _is_xyz_input(product)
    if reactant_xyz != product_xyz:
        raise ValueError("Supply both endpoints as XYZ, or read them as RDKit/rxembed molecules before preparation")
    if charge is None and reactant_xyz:
        raise ValueError("XYZ endpoints do not declare charge; pass charge=...")
    reactant_charge, reactant_graph = _declared_endpoint(reactant, charge)
    product_charge, product_graph = _declared_endpoint(product, charge)
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

    rea_state, reactant, rea_request = select_metal_state(reactant, reactant_geometry)
    pro_state, product, pro_request = select_metal_state(product, product_geometry)
    native_pair = rea_request or pro_request
    if native_pair:
        if rea_state is None or pro_state is None:
            raise ValueError("Native paired preparation requires a selected coordination state at both endpoints")
        if not reactant.GetNumConformers():
            reactant = selected_endpoint(rxembed.embed(deepcopy(rea_state), n=1, seed=seed, threads=1), reactant)
        elif rea_request:
            reactant = relax_metal_endpoint(rea_state, reactant, list(range(reactant.GetNumAtoms())))
        else:
            reactant = Chem.Mol(reactant)
        rea = reactant_graph if reactant_xyz else reactant
        pro = relax_metal_endpoint(pro_state, rea, mapping)
    else:
        reactant = selected_endpoint(reactant)
        product = selected_endpoint(product)
        if reactant_xyz:
            rea, pro = reactant_graph, product_graph
        else:
            rea = (
                Chem.Mol(reactant)
                if isinstance(reactant, Chem.Mol) and _geometry_known(reactant)
                else combine_spaced(as_molecules(reactant, seed=seed))
            )
            pro = (
                Chem.Mol(product)
                if isinstance(product, Chem.Mol) and _geometry_known(product)
                else combine_spaced(as_molecules(product, seed=seed))
            )

    # Keep product atoms and coordinates in the reactant's index space.
    pro = Chem.RenumberAtoms(pro, invert(mapping))
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

    if not native_pair and (not _geometry_known(reactant) or not _geometry_known(product)):
        # Choose the attack face before contact relaxation; later distance restraints do not set it.
        rea = search_approach(
            mol=rea,
            forming=sorted(forming),
            breaking=sorted(breaking),
            energy=_charged_potential(rea, charge, solvent),
        )
        rea = reactive_complex(rea, sorted(forming))
        pro = derive_product_geometry(reactant=rea, product=pro, forming=forming, breaking=breaking)

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
        sources=sources,
        native_states=(rea_state, pro_state),
        endpoint_identity=(_metal_identity(rea_state), _metal_identity(pro_state, mapping)),
    )
