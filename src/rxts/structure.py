"""Mapped FF geometry records and molecular ASE/XYZ export."""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

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
    from rdkit import Chem

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
    qm_energy: float | None = None  # physical energy, kcal/mol, if scored


def _frame(res, x, lam):
    """Store interpolated, reactant-field and product-field energies in kcal/mol."""
    e, e1, e2 = res
    return Frame(lam=lam, positions=x.copy(), energy=e, e_reactant=e1, e_product=e2)


@dataclass
class ReactionPath:
    """An ordered FF scan with optional fixed-frame physical scores.

    Select every peak/barrier surface explicitly with "ff" or "qm".
    """

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

    # Set with qm_energy only after every frame scores successfully.
    qm_method: str | CalculatorMethod | None = field(default=None, repr=False)
    qm_solvent: str | None = field(default=None, repr=False)
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
        if any(f.qm_energy is None for f in self.frames):
            return None
        e = np.array([f.qm_energy for f in self.frames], dtype=float)
        return e - e[0]

    @property
    def scored(self) -> bool:
        """Whether every frame has a physical score."""
        return self.qm_energies is not None

    @property
    def qm_label(self) -> str | None:
        """Physical method label, or None before scoring."""
        return None if self.qm_method is None else method_label(self.qm_method)

    def _energies(self, on):
        """Resolve the explicitly requested energy surface."""
        if on == "ff":
            return self.ff_energies
        if on == "qm":
            e = self.qm_energies
            if e is None:
                raise ValueError(
                    "This path has no QM energies: score() has not run, or it failed. "
                    'Run path.score(method=...) or select on="ff" explicitly.'
                )
            return e
        raise ValueError(f'on must be "ff" or "qm", not {on!r}: say which surface you are asking about.')

    def peak_index(self, on: str) -> int:
        """Highest frame on the named surface, after keeping the lowest FF candidate per window."""
        return int(np.nanargmax(self._energies(on)))

    def peak_lambda(self, on: str) -> float:
        """Lambda at the highest frame on the named surface."""
        return float(self.frames[self.peak_index(on)].lam)

    def barrier(self, on: str) -> float:
        """Peak minus reactant-frame energy on the named surface, in kcal/mol."""
        return float(self._energies(on)[self.peak_index(on)])

    def reaction_energy(self, on: str) -> float:
        """Products minus reactants on the named surface, kcal/mol. Negative means exothermic."""
        return float(self._energies(on)[-1])

    def guess(self, on: str) -> Geometry:
        """Return the peak candidate on the explicit surface, preserving state and correspondence."""
        i = self.peak_index(on)
        frame = self.frames[i]
        method = self.engine if on == "ff" else self.qm_method
        solvent = self.solvent if on == "ff" else self.qm_solvent
        label = method_label(method)
        what = (
            f"highest point of the {label} scan"
            if on == "ff"
            else f"highest {label} point of the force-field scan ({method_label(self.engine)})"
        )
        return self._geometry(
            frame,
            # Successful physical scoring sets qm_method before a QM peak can be selected.
            method=method,  # ty: ignore[invalid-argument-type]
            solvent=solvent,
            origin=(
                f"rxts TS guess, {what} in {solvent or 'gas phase'}, "
                f"lambda={frame.lam:.2f}, barrier {self.barrier(on):.1f} kcal/mol"
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
                method=self.engine,
                solvent=self.solvent,
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
    ) -> ScoreResult:
        """Score existing frames without moving them; return absolute eV energies and indexed failures.

        Replace the profile only after full success. Built-ins use gas for solvent=None;
        supplied calculators retain their configured potential.
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
            warnings.warn(
                f"Path NOT scored: {reason}. Earlier scores are retained if present.",
                RuntimeWarning,
                stacklevel=2,
            )
            return ScoreResult(
                (None,) * len(self.frames),
                dict.fromkeys(range(len(self.frames)), reason),
                0,
                0,
                str(method),
                self.charge,
                self.multiplicity,
                solvent,
            )

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
            logger.warning("Frame scoring failed; the previous profile is retained if present.")
            warnings.warn(
                f"{len(attempt.failures)} of {len(self.frames)} frames failed. "
                "No scores were replaced; earlier scores are retained if present. "
                "Inspect the returned failure reasons before using the stored profile.",
                RuntimeWarning,
                stacklevel=2,
            )
            return attempt

        # Commit a complete profile with its method and solvent.
        for frame, energy in zip(self.frames, attempt.energies_ev, strict=True):
            assert energy is not None
            frame.qm_energy = energy / EV_PER_KCAL
        self.qm_method = method
        self.qm_solvent = attempt.solvent
        return attempt

    def bond_lengths(self) -> dict[tuple[str, tuple[int, int]], list[float]]:
        """Reactive bond lengths over the ordered path, in angstrom."""
        return {
            (kind, pair): [float(np.linalg.norm(f.positions[pair[0]] - f.positions[pair[1]])) for f in self.frames]
            for kind, pairs in (("breaking", self.breaking_bonds), ("forming", self.forming_bonds))
            for pair in pairs
        }

    def summary(self, on: str = "ff") -> str:
        """Summarise the candidate and reactive distances on the named surface."""
        guess = self.guess(on)
        label = guess.label
        lines = [
            f"Transition state guess    lambda = {self.peak_lambda(on):.2f}",
            f"Barrier ({label:^12s})  {self.barrier(on):8.1f} kcal/mol",
            f"Reaction energy           {self.reaction_energy(on):8.1f} kcal/mol",
            f"Path frames               {len(self.frames)}",
            "",
            "Reactive bonds at the transition state:",
        ]
        for kind, pairs in (("breaking", self.breaking_bonds), ("forming", self.forming_bonds)):
            for pair in pairs:
                lines.append(f"  {kind:>8s}  {describe(self.mol, pair):<10s}  {bond_length(guess.mol, pair):5.2f} A")
        return "\n".join(lines)
