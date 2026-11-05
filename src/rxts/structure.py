"""Mapped FF geometry records and molecular ASE/XYZ export."""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from rdkit import Chem

from rxts import scoring
from rxts.scoring import CalculatorMethod, ScoreResult
from rxts.utils import (
    bond_length,
    default_cores,
    describe,
    set_positions,
    to_atoms,
    to_xyz,
    validate_cores,
)
from rxts.xtb import EV_PER_KCAL, METHODS, TbliteBackend, xtb_available

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from ase.calculators.calculator import BaseCalculator

_METHOD_NAMES = {"gxtb": "g-xTB", "gfnff": "GFN-FF", "mmff": "MMFF94"}


def method_label(name: str | CalculatorMethod | BaseCalculator | None) -> str:
    """Display label for a built-in method or supplied calculator."""
    name = scoring._normalise_method(name)
    if name is None:
        return "none"
    if isinstance(name, CalculatorMethod):
        return name.label
    return _METHOD_NAMES.get(name, name.upper())


_ASE_CONTEXT = (
    "forming_bonds",
    "breaking_bonds",
    "charge",
    "multiplicity",
    "solvent",
    "n_cores",
    "mapping",
    "endpoint_identity",
    "metal_angle_k",
    "metal_angle_reference",
)


@dataclass(frozen=True)
class Geometry:
    """FF candidate coordinates with the reactant graph, selected surface and reaction context."""

    mol: Chem.Mol
    method: str | CalculatorMethod
    forming_bonds: list = field(default_factory=list)
    breaking_bonds: list = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    solvent: str | None = None
    n_cores: int | None = None
    origin: str = "rxts geometry, provenance not recorded"
    mapping: list[int] | None = None  # Original product index -> reactant index, when available.
    endpoint_identity: tuple[dict[str, Any] | None, dict[str, Any] | None] = (None, None)
    metal_angle_k: float | None = None
    metal_angle_reference: str | None = None

    def to_atoms(self) -> Atoms:
        """Export exact coordinates, isotope masses and independent reaction metadata without a calculator."""
        atoms = to_atoms(self.mol, charge=self.charge, multiplicity=self.multiplicity)
        context = {name: deepcopy(getattr(self, name)) for name in _ASE_CONTEXT}
        for name in ("forming_bonds", "breaking_bonds", "mapping"):
            value = context[name]
            context[name] = None if value is None else list(value)
        isotopes = [atom.GetIsotope() for atom in self.mol.GetAtoms()]
        if any(isotopes):
            context["isotopes"] = isotopes
        atoms.info["rxts"] = context | {"method": method_label(self.method), "position_unit": "angstrom"}
        return atoms

    @property
    def positions(self) -> np.ndarray:
        """The coordinates, (n_atoms, 3), in angstrom."""
        return np.asarray(self.mol.GetConformer().GetPositions(), dtype=float)

    @property
    def label(self) -> str:
        """How to spell this geometry's method."""
        return method_label(self.method)

    def dump(self, filename: str | Path) -> str | Path:
        """Write one XYZ candidate with its origin, charge and multiplicity."""
        comment = f"{' '.join(self.origin.split())} charge={self.charge} multiplicity={self.multiplicity}"
        with open(filename, "w") as fh:
            fh.write(to_xyz(self.mol, comment))
        return filename


logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """One point on the reaction path."""

    lam: float
    positions: np.ndarray
    energy: float  # interpolated force-field energy V, kcal/mol
    e_reactant: float  # reactant force field at this geometry, kcal/mol
    e_product: float  # product force field at this geometry, kcal/mol


def _frame(res, x, lam):
    """Store interpolated, reactant-field and product-field energies in kcal/mol."""
    e, e1, e2 = res
    return Frame(lam=lam, positions=x.copy(), energy=e, e_reactant=e1, e_product=e2)


@dataclass
class ReactionPath:
    """An ordered FF scan, or an independently scored copy carrying its calculator's profile."""

    frames: list[Frame]
    reactant: Chem.Mol
    product: Chem.Mol
    forming_bonds: list[tuple[int, int]]
    breaking_bonds: list[tuple[int, int]]
    mapping: list[int]
    engine: str
    charge: int
    multiplicity: int
    # FF construction solvent; physical scoring has its own qm_solvent.
    solvent: str | None
    n_cores: int | None
    # Reactant graph in the frames' atom order.
    mol: Chem.Mol = field(repr=False)

    # Scoring context, including failed attempts with partial frame energies.
    qm_method: str | CalculatorMethod | None = field(default=None, repr=False)
    scores: ScoreResult | None = field(default=None, repr=False)
    endpoint_identity: tuple[dict | None, dict | None] = field(default=(None, None), repr=False)
    metal_angle_k: float = field(default=0.0, repr=False)
    metal_angle_reference: str | None = field(default=None, repr=False)
    sampling_failures: list[dict[str, Any]] = field(default_factory=list, repr=False)
    conformer_selections: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def ff_energies(self) -> np.ndarray:
        """FF energies relative to the reactant frame, in kcal/mol."""
        e = np.array([f.energy for f in self.frames], dtype=float)
        return e - e[0]

    @property
    def qm_energies(self) -> np.ndarray | None:
        """Physical energies relative to the reactant frame, in kcal/mol; None until fully scored."""
        if self.scores is None or self.scores.failures or any(e is None for e in self.scores.energies_ev):
            return None
        e = self.absolute_energies
        return e - e[0]

    @property
    def qm_solvent(self) -> str | None:
        """The scoring attempt's solvent, or None before scoring."""
        return None if self.scores is None else self.scores.solvent

    @property
    def scored(self) -> bool:
        """Whether every frame has a physical score."""
        return self.qm_energies is not None

    @property
    def qm_label(self) -> str | None:
        """Physical method label, or None before scoring."""
        return None if self.qm_method is None else method_label(self.qm_method)

    @property
    def method(self) -> str | CalculatorMethod:
        """The force field or supplied calculator that defines this result's energies."""
        if self.scores is None:
            return self.engine
        assert self.qm_method is not None
        return self.qm_method

    @property
    def absolute_energies(self) -> np.ndarray:
        """Absolute energies in kcal/mol; refuse incomplete scores and nonfinite profiles."""
        if self.scores is None:
            energies = np.array([frame.energy for frame in self.frames], dtype=float)
        else:
            if self.scores.failures or any(e is None for e in self.scores.energies_ev):
                raise ValueError("Path scoring failed; inspect path.scores.failures before selecting a candidate")
            energies = np.asarray(self.scores.energies_ev, dtype=float) / EV_PER_KCAL
        if len(energies) != len(self.frames) or not len(energies) or not np.isfinite(energies).all():
            raise ValueError("A path needs one finite energy for every frame before selection")
        return energies

    @property
    def energies(self) -> np.ndarray:
        """This result's energies relative to its reactant frame, in kcal/mol."""
        values = self.absolute_energies
        return values - values[0]

    def peak_index(self) -> int:
        """Index of the highest sampled frame on this result's energy profile."""
        return int(np.argmax(self.absolute_energies))

    def peak_lambda(self) -> float:
        """Lambda at the highest sampled frame on this result's energy profile."""
        return float(self.frames[self.peak_index()].lam)

    def barrier(self) -> float:
        """Peak minus this path's reactant-frame energy, in kcal/mol."""
        return float(self.energies[self.peak_index()])

    def reaction_energy(self) -> float:
        """Products minus reactants on this result's profile, in kcal/mol."""
        return float(self.energies[-1])

    def guess(self) -> Geometry:
        """Return the highest sampled geometry, preserving the method, state and correspondence."""
        i = self.peak_index()
        frame = self.frames[i]
        solvent = self.solvent if self.scores is None else self.qm_solvent
        label = method_label(self.method)
        what = (
            f"highest point of the {label} scan"
            if self.scores is None
            else f"highest {label} point of the force-field scan ({method_label(self.engine)})"
        )
        return self._geometry(
            frame,
            method=self.method,
            solvent=solvent,
            origin=(
                f"rxts TS guess, {what} in {solvent or 'gas phase'}, "
                f"lambda={frame.lam:.2f}, barrier {self.barrier():.1f} kcal/mol"
            ),
        )

    def _geometry(self, frame, *, method: str | CalculatorMethod, solvent, origin) -> Geometry:
        context = {name: deepcopy(getattr(self, name)) for name in _ASE_CONTEXT}
        context["solvent"] = solvent
        return Geometry(mol=set_positions(self.mol, frame.positions), method=method, origin=origin, **context)

    def to_atoms(self) -> list[Atoms]:
        """Export every existing FF frame as independent ASE atoms without scoring or attaching calculators."""
        qm_e, ff_e = self.qm_energies, self.ff_energies
        atoms = []
        for i, frame in enumerate(self.frames):
            image = self._geometry(
                frame,
                method=self.method,
                solvent=self.solvent if self.scores is None else self.qm_solvent,
                origin=f"rxts force-field frame {i}, lambda={frame.lam:.3f}",
            ).to_atoms()
            image.info["rxts"].update(
                construction="ff_scan",
                construction_method=self.engine,
                frame_index=i,
                lambda_value=frame.lam,
                ff_energy_relative_kcal=float(ff_e[i]),
                qm_energy_relative_kcal=None if qm_e is None else float(qm_e[i]),
                qm_method=self.qm_label,
                qm_solvent=self.qm_solvent,
            )
            atoms.append(image)
        return atoms

    @property
    def path(self) -> list[Chem.Mol]:
        """Every frame as an RDKit molecule, reactants -> products."""
        return [set_positions(self.mol, f.positions) for f in self.frames]

    def dump(self, filename: str | Path) -> str | Path:
        """Write ordered XYZ frames with charge, multiplicity and relative energies."""
        qm_e = self.qm_energies
        ff_e = self.ff_energies
        with open(filename, "w") as fh:
            for i, (frame, mol) in enumerate(zip(self.frames, self.path, strict=True)):
                comment = f"lambda={frame.lam:.3f} E({method_label(self.engine)})={ff_e[i]:.2f} kcal/mol"
                if qm_e is not None:
                    comment += f" E({self.qm_label})={qm_e[i]:.2f} kcal/mol"
                comment += f" charge={self.charge} multiplicity={self.multiplicity}"
                fh.write(to_xyz(mol, comment))
        return filename

    def score(
        self,
        method: str | CalculatorMethod | BaseCalculator,
        solvent: str | None = None,
        n_cores: int | None = None,
    ) -> ReactionPath:
        """Return an independently scored copy; keep partial evaluations in its `scores` on failure.

        Built-ins use gas for solvent=None; supplied calculators retain their configured potential.
        Failed results cannot select a guess. The original path and any earlier scored copies stay unchanged.
        """
        method = scoring._normalise_method(method)
        if method is None:
            raise ValueError("Path scoring requires an explicit physical method")
        if not isinstance(method, CalculatorMethod) and method not in METHODS:
            raise ValueError(
                f"Unsupported physical method {method!r}; use one of {sorted(METHODS)} or CalculatorMethod"
            )
        validate_cores(n_cores)
        # GFN1/GFN2 can run through tblite; g-xTB and the solvated composite require the binary.
        # A supplied ASE provider bypasses built-in availability checks.
        if (
            not isinstance(method, CalculatorMethod)
            and not (TbliteBackend.available() and TbliteBackend.supports(method))
            and not xtb_available()
        ):
            logger.warning(
                f"{method_label(method)} needs the xtb binary, which is not on PATH -- skipping "
                f"scoring. Put xtb on PATH, or set $XTB_EXE to it (a Jupyter kernel often has neither)."
            )
            reason = "xtb was not found on PATH; put xtb on PATH or set $XTB_EXE"
            attempt = ScoreResult(
                (None,) * len(self.frames),
                dict.fromkeys(range(len(self.frames)), reason),
                0,
                0,
                str(method),
                self.charge,
                self.multiplicity,
                solvent,
            )
        else:
            if isinstance(method, CalculatorMethod) and method.calculator is not None:
                cores = 1 if n_cores is None else n_cores
                execution = "serially"
            else:
                cores = self.n_cores if n_cores is None else n_cores
                cores = default_cores() if cores is None else cores
                execution = f"on {cores} core(s)"
            logger.info(f"Re-scoring {len(self.frames)} frames with {method_label(method)} {execution}...")
            attempt = scoring.score(
                self.to_atoms(),
                charge=self.charge,
                multiplicity=self.multiplicity,
                method=method,
                solvent=solvent,
                n_cores=cores,
            )
        if attempt.failures:
            warnings.warn(
                f"Path NOT scored: {len(attempt.failures)} of {len(self.frames)} frames failed. "
                "Inspect the returned path.scores.failures; this result cannot select a candidate.",
                RuntimeWarning,
                stacklevel=2,
            )
        return replace(
            self,
            frames=[replace(frame, positions=frame.positions.copy()) for frame in self.frames],
            reactant=Chem.Mol(self.reactant),
            product=Chem.Mol(self.product),
            mol=Chem.Mol(self.mol),
            forming_bonds=list(self.forming_bonds),
            breaking_bonds=list(self.breaking_bonds),
            mapping=list(self.mapping),
            endpoint_identity=deepcopy(self.endpoint_identity),
            sampling_failures=deepcopy(self.sampling_failures),
            conformer_selections=deepcopy(self.conformer_selections),
            scores=attempt,
            qm_method=method,
        )

    def bond_lengths(self) -> dict[tuple[str, tuple[int, int]], list[float]]:
        """Reactive bond lengths over the ordered path, in angstrom."""
        return {
            (kind, pair): [float(np.linalg.norm(f.positions[pair[0]] - f.positions[pair[1]])) for f in self.frames]
            for kind, pairs in (("breaking", self.breaking_bonds), ("forming", self.forming_bonds))
            for pair in pairs
        }

    def summary(self) -> str:
        """Summarise the candidate and reactive distances on this result's energy profile."""
        guess = self.guess()
        label = guess.label
        lines = [
            f"Transition state guess    lambda = {self.peak_lambda():.2f}",
            f"Barrier ({label:^12s})  {self.barrier():8.1f} kcal/mol",
            f"Reaction energy           {self.reaction_energy():8.1f} kcal/mol",
            f"Path frames               {len(self.frames)}",
            "",
            "Reactive bonds at the transition state:",
        ]
        for kind, pairs in (("breaking", self.breaking_bonds), ("forming", self.forming_bonds)):
            for pair in pairs:
                lines.append(f"  {kind:>8s}  {describe(self.mol, pair):<10s}  {bond_length(guess.mol, pair):5.2f} A")
        return "\n".join(lines)


class ReactionPaths(tuple):
    """An iterable, indexable collection of complete candidate paths, including a one-path result.

    `score` returns independent scored paths. `best` compares absolute profile maxima on a common
    method and context; it does not compare barriers independently zeroed at different reactants,
    or certify continuity or a saddle point. Failed scores prevent selection across the collection.
    """

    def __new__(cls, paths=()):
        items = tuple(paths)
        if any(not isinstance(path, ReactionPath) for path in items):
            raise TypeError("ReactionPaths accepts ReactionPath objects")
        return super().__new__(cls, items)

    def __getitem__(self, index):
        picked = super().__getitem__(index)
        return ReactionPaths(picked) if isinstance(index, slice) else picked

    def score(
        self,
        method: str | CalculatorMethod | BaseCalculator,
        solvent: str | None = None,
        n_cores: int | None = None,
    ) -> ReactionPaths:
        """Score every frame of every path, returning independent paths with their paid results."""
        method = scoring._normalise_method(method)
        paths = []
        try:
            for path in self:
                paths.append(path.score(method, solvent=solvent, n_cores=n_cores))
        except BaseException as error:
            error.candidate_paths = ReactionPaths(paths)  # ty: ignore[unresolved-attribute]
            raise
        return ReactionPaths(paths)

    def best(self) -> ReactionPath:
        """Return the path with the lowest absolute maximum; refuse mixed methods or incomplete scores."""
        if not self:
            raise ValueError("No candidate paths to select; supply at least one endpoint pair")
        contexts, peaks = [], []
        for path in self:
            method = path.method
            if isinstance(method, CalculatorMethod) and method.calculator is not None:
                method = method.calculator
            contexts.append(
                (
                    method,
                    path.solvent if path.scores is None else path.qm_solvent,
                    path.charge,
                    path.multiplicity,
                    tuple(sorted((atom.GetAtomicNum(), atom.GetIsotope()) for atom in path.mol.GetAtoms())),
                )
            )
            peaks.append(float(np.max(path.absolute_energies)))
        if any(context != contexts[0] for context in contexts[1:]):
            raise ValueError("Compare paths with the same calculator, solvent, charge, multiplicity and atoms")
        return self[int(np.argmin(peaks))]
