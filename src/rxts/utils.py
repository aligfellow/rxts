"""Molecular I/O, geometry checks, rigid alignment and bounded independent work."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from threadpoolctl import threadpool_limits

if TYPE_CHECKING:
    from ase import Atoms

# Coordinate-perceived graphs are excluded from automatic MMFF typing.
FROM_COORDINATES = "_rxts_from_coordinates"

#: Selected endpoint coordinates are already placed; skip fragment approach and product derivation.
PLACED = "_rxts_placed"


# Transition metals handled by native coordination preparation.
_TRANSITION_METALS = frozenset(range(21, 31)) | frozenset(range(39, 49)) | {57, 71} | frozenset(range(72, 81))
_BOND_SCALE = 1.2  # bonded below this multiple of the summed covalent radii


def is_xyz_file(spec):
    """Recognise an XYZ filename; literal XYZ text and conformer-bearing molecules are different inputs."""
    if isinstance(spec, Path):
        return spec.suffix == ".xyz"
    return isinstance(spec, str) and "\n" not in spec and spec.strip().endswith(".xyz")


def read_xyz(source, frame=0):
    """Read one frame of an XYZ file or string as (symbols, coordinates)."""
    if isinstance(frame, bool) or not isinstance(frame, Integral) or frame < 0:
        raise ValueError("frame must be a nonnegative integer")
    # Strings need an .xyz suffix to mean a path; Path objects always name files.
    is_file = isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and source.strip().endswith(".xyz")
    )
    text = Path(source).read_text() if is_file else str(source)
    lines = text.splitlines()
    cursor = 0
    for selected in range(frame + 1):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor == len(lines):
            raise IndexError(f"{source!r} has no frame {frame}")
        try:
            count = int(lines[cursor].split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Malformed XYZ atom count at line {cursor + 1}") from exc
        if count < 0:
            raise ValueError(f"Malformed XYZ atom count at line {cursor + 1}: {count}")
        cursor += 1
        if cursor == len(lines):
            raise ValueError(f"Truncated XYZ frame {selected}: missing comment line")
        cursor += 1  # The comment line may be blank.
        if len(lines) - cursor < count:
            raise ValueError(f"Truncated XYZ frame {selected}: expected {count} atom lines")
        symbols, coordinates = [], []
        for line_number, row in enumerate(lines[cursor : cursor + count], cursor + 1):
            fields = row.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed XYZ atom line {line_number}")
            try:
                point = [float(value) for value in fields[1:4]]
            except ValueError as exc:
                raise ValueError(f"Malformed XYZ coordinates at line {line_number}") from exc
            if not np.isfinite(point).all():
                raise ValueError(f"Malformed XYZ coordinates at line {line_number}")
            symbols.append(fields[0])
            coordinates.append(point)
        cursor += count
        if selected == frame:
            return symbols, np.asarray(coordinates, dtype=float).reshape((-1, 3))


def mol_from_xyz(source, *, charge=0):
    """Read an endpoint minimum through rxembed, without graph-changing fallbacks."""
    import rxembed

    symbols, coords = read_xyz(source)
    # rxembed currently accepts paths only; normalize literal and multi-frame XYZ here.
    with TemporaryDirectory(prefix="rxts-endpoint-") as scratch:
        path = Path(scratch) / "endpoint.xyz"
        text = Path(source).read_text() if isinstance(source, Path) or "\n" not in str(source) else str(source)
        path.write_text("\n".join(text.lstrip().splitlines()[: len(symbols) + 2]) + "\n")
        out = rxembed.read_xyz(str(path), charge=charge, fallback=False)
    if [a.GetSymbol() for a in out.GetAtoms()] != symbols or not np.array_equal(
        out.GetConformer().GetPositions(), coords
    ):
        raise ValueError("rxembed XYZ perception changed atom order or coordinates; supply an explicit endpoint graph")
    out.SetBoolProp(FROM_COORDINATES, True)
    return out


def metal_atoms(mol):
    """Return indices of transition-metal atoms."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in _TRANSITION_METALS]


def perceived_bonds(mol):
    """Read the bonding a geometry implies, by distance against covalent radii.

    The same criterion GFN-FF uses, applied here to check it perceived the intended topology (GFN-FF
    does not report its own bond list).
    """
    pos = mol.GetConformer().GetPositions()
    table = Chem.GetPeriodicTable()
    radii = [table.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()]
    bonds = set()
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            if float(np.linalg.norm(pos[i] - pos[j])) < _BOND_SCALE * (radii[i] + radii[j]):
                bonds.add((i, j))
    return bonds


def require_topology(mol, intended, what):
    """Check seed bonding against the intended graph before building a frozen GFN-FF field.

    The covalent-radius check is necessary but does not prove GFN-FF perceives the same topology."""
    seen = perceived_bonds(mol)
    intended = {tuple(sorted(p)) for p in intended}
    if seen == intended:
        return

    missing = sorted(intended - seen)
    spurious = sorted(seen - intended)
    lines = [f"The {what} geometry does not have the bonding it should."]
    if missing:
        lines.append(f"  bonds that should be there but are not: {', '.join(describe(mol, p) for p in missing)}")
    if spurious:
        lines.append(f"  bonds that are there but should not be: {', '.join(describe(mol, p) for p in spurious)}")
    lines.append("")
    lines.append(
        "GFN-FF reads its bond list straight off the coordinates, so it would build "
        "a force field for the wrong molecule and never say so."
    )
    raise ValueError("\n".join(lines))


def xyz_block(symbols, positions, comment=""):
    """XYZ block for one geometry, from element symbols and angstrom coordinates."""
    lines = [str(len(symbols)), comment]
    for symbol, p in zip(symbols, positions, strict=True):
        lines.append(f"{symbol:<2s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}")
    return "\n".join(lines) + "\n"


def to_xyz(mol, comment=""):
    """XYZ block for one geometry."""
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return xyz_block(symbols, mol.GetConformer().GetPositions(), comment)


def set_positions(mol, x):
    """Write coordinates back onto a copy of `mol`."""
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, p in enumerate(x):
        conf.SetAtomPosition(i, p.tolist())
    return out


def with_conformer(mol, positions):
    """Return a copy of `mol` carrying exactly these coordinates, and no other conformer."""
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, [float(v) for v in p])
    out.AddConformer(conf)
    return out


def _align(positions, reference):
    """Rotate and translate `positions` onto `reference` (Kabsch).

    Removes rigid-body drift between frames; changes no bond length or energy.
    """
    p = positions - positions.mean(axis=0)
    q = reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))  # keep it a rotation, not a reflection
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return p @ rotation.T + reference.mean(axis=0)


def align_frames(frames):
    """Align a whole trajectory, each frame onto the one before it."""
    out = [frames[0] - frames[0].mean(axis=0)]
    for x in frames[1:]:
        out.append(_align(positions=x, reference=out[-1]))
    return out


def _validate_declared_context(charge, multiplicity):
    if charge is not None and (isinstance(charge, bool) or not isinstance(charge, Integral)):
        raise ValueError("charge must be an integer when declared")
    if multiplicity is not None and (
        isinstance(multiplicity, bool) or not isinstance(multiplicity, Integral) or multiplicity < 1
    ):
        raise ValueError("multiplicity must be a positive integer when declared")


def to_atoms(mol: Chem.Mol, *, charge: int | None = None, multiplicity: int | None = None) -> Atoms:
    """Copy coordinates, ordered atom maps, isotope overrides and declared state into ASE Atoms."""
    from ase import Atoms

    _validate_declared_context(charge, multiplicity)
    atoms = Atoms(
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
        positions=np.asarray(mol.GetConformer().GetPositions(), dtype=float),
    )
    atoms.set_array("atom_maps", np.array([a.GetAtomMapNum() for a in mol.GetAtoms()], dtype=int))
    if charge is not None:
        atoms.info["charge"] = int(charge)
    if multiplicity is not None:
        atoms.info["multiplicity"] = int(multiplicity)
    if any(atom.GetIsotope() for atom in mol.GetAtoms()):
        masses = atoms.get_masses()
        for atom in mol.GetAtoms():
            if atom.GetIsotope():
                masses[atom.GetIdx()] = atom.GetMass()
        atoms.set_masses(masses)
    return atoms


def bond_length(mol, pair):
    """Distance between two atoms, in angstrom."""
    return rdMolTransforms.GetBondLength(mol.GetConformer(), int(pair[0]), int(pair[1]))


def angle(x, i, j, k):
    """Measure angle i-j-k in radians, with j at the vertex."""
    a, b = x[i] - x[j], x[k] - x[j]
    cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def describe(mol, internals):
    """Name an internal coordinate by its atom symbols and indices."""
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    return "-".join(f"{sym[i]}{i}" for i in internals)


def _graph_identity(mol):
    """Chemical identity in current atom order, including isotopes and declared stereo."""
    atoms = tuple(
        (
            a.GetAtomicNum(),
            a.GetIsotope(),
            a.GetFormalCharge(),
            a.GetNumRadicalElectrons(),
            a.GetAtomMapNum(),
            a.GetChiralTag(),
        )
        for a in mol.GetAtoms()
    )
    bonds = tuple(
        sorted(
            (
                b.GetBeginAtomIdx(),
                b.GetEndAtomIdx(),
                b.GetBondType(),
                b.GetIsAromatic(),
                b.GetBondDir(),
                b.GetStereo(),
                tuple(b.GetStereoAtoms()),
            )
            for b in mol.GetBonds()
        )
    )
    return atoms, bonds


def default_cores():
    """CPUs available to this process, respecting affinity/SLURM allocations."""
    if hasattr(os, "process_cpu_count"):  # 3.13+: honours affinity, and is portable
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1  # macOS/Windows: no affinity mask to ask about


def validate_cores(n_cores):
    """Resolve an optional worker budget and reject invalid explicit values before work."""
    if n_cores is None:
        return default_cores()
    if isinstance(n_cores, bool) or not isinstance(n_cores, Integral) or n_cores < 1:
        raise ValueError("n_cores must be a positive integer")
    return int(n_cores)


def _worker_count(n_items, n_workers=None):
    """Workers to run `n_items` on: never more than the items or the core budget, and at least one."""
    return max(1, min(validate_cores(n_workers), n_items, default_cores()))


def _bounded_map(items, run_item, n_workers=None):
    """Map independent work in input order, with workers capped by items, request and CPU allocation.

    Pin numerical libraries to one thread per worker to avoid oversubscription.
    """
    if n_workers is not None:
        validate_cores(n_workers)
    items = list(items)
    if not items:
        return []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=_worker_count(len(items), n_workers)) as pool:
        return list(pool.map(run_item, items))
