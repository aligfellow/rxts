"""VeloxChem-derived FF scanning, conformer sampling and candidate selection."""

from __future__ import annotations

import logging
import math
import sys
import warnings
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import rxembed
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from rxts import scoring
from rxts.builder import PreparedReaction
from rxts.builder import prepare_endpoints as _prepare_endpoints
from rxts.evb import (
    _METAL_ANGLE_K,
    _MORSE_D,
    _SOFT_CORE_FRACTION,
    GFNFFPotential,
    MMFFPotential,
    active_torsion,
    check_mmff,
    metal_angle_references,
    mmff_can_type,
    untypeable_fragments,
)
from rxts.mapping import bond_set
from rxts.scoring import CalculatorMethod
from rxts.structure import ReactionPath, _frame, method_label
from rxts.utils import (
    _worker_count,
    align_frames,
    default_cores,
    describe,
    metal_atoms,
    perceived_bonds,
    require_topology,
    set_positions,
    to_atoms,
    with_conformer,
)
from rxts.xtb import EV_PER_KCAL, METHODS

if TYPE_CHECKING:  # annotations only
    from collections.abc import Sequence
    from pathlib import Path

    from rxembed.core import Conformers, Isomer

logger = logging.getLogger(__name__)

_package_logger = logging.getLogger("rxts")


def set_verbose(verbose: bool = True) -> None:
    """Enable console progress, or reduce output to warnings with False."""
    if verbose and not _package_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _package_logger.addHandler(handler)
        _package_logger.propagate = False
    _package_logger.setLevel(logging.INFO if verbose else logging.WARNING)


_ENGINES = ("auto", "mmff", "gfnff")

_PAIR = 2


def _finite(v):
    return isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(float(v))


def _an_int(v, floor=None):
    return isinstance(v, Integral) and not isinstance(v, bool) and (floor is None or v >= floor)


def _pairs(v):
    return v is None or (
        isinstance(v, list | tuple)
        and all(isinstance(p, list | tuple) and len(p) == _PAIR and all(_an_int(i, 0) for i in p) for p in v)
    )


def _targets(v):
    return isinstance(v, dict) and all(_pairs([k]) and k[0] != k[1] and _finite(t) and t > 0 for k, t in v.items())


class ConformerScoringError(RuntimeError):
    """Retain paid scores and proposed geometries when physical ranking fails."""

    def __init__(self, result, candidates, *, selection=None):
        self.result = result
        self.selection = selection
        self.candidates = tuple(candidates)
        failed = "; ".join(f"{index}: {reason}" for index, reason in result.failures.items())
        super().__init__(f"Physical conformer scoring failed ({failed}); inspect result and candidates")


class _FFRelaxationError(RuntimeError):
    """An FF minimisation failed; retain the optimizer's last paid state."""

    def __init__(self, lam, positions, energy, forces, reason):
        self.positions = np.array(positions, copy=True)  # angstrom
        self.lam = float(lam)
        self.energy_ev = float(energy)
        self.forces_ev_per_angstrom = np.array(forces, copy=True)
        super().__init__(f"Force-field relaxation at lambda={lam:.3f} {reason}.")


class _SamplingCalculator(Calculator):
    """Expose the scan's restrained FF energy and gradient in ASE units."""

    def __init__(self, guesser, pot, lam):
        super().__init__()
        self.implemented_properties = ["energy", "forces"]
        self.guesser, self.pot, self.lam = guesser, pot, lam

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        energy, gradient = self.guesser._energy_grad(self.pot, self.atoms.positions, self.lam)
        self.results["energy"] = energy * EV_PER_KCAL
        self.results["forces"] = -gradient * EV_PER_KCAL


@dataclass
class TransitionStateGuesser:
    """Build an interpolated FF path and select a peak candidate.

    Endpoint fields are relaxed across lambda. `embed_score` optionally ranks peak candidates on a
    physical calculator; otherwise ranking stays on the force field.
    """

    n_lambda: int = 21
    seed: int = 0xF00D

    # Reaction detection: set these to override what is auto-detected.
    forming_bonds: list | None = None
    breaking_bonds: list | None = None
    mapping: list | None = None
    reactant_geometry: str | None = None
    product_geometry: str | None = None

    # "auto" uses MMFF where both endpoints can be typed, otherwise GFN-FF.
    engine: str | None = None  # explicitly select "auto", "mmff" or "gfnff" before scanning
    morse_d: float = _MORSE_D
    soft_core: float = _SOFT_CORE_FRACTION
    constraints: dict = field(default_factory=dict)  # {(i, j): target distance in angstrom}
    constraint_k: float = 100.0  # kcal/mol/A^2
    metal_angle_k: float = _METAL_ANGLE_K  # kcal/mol/rad^2; 1000 kJ/mol/rad^2 from VeloxChem

    # Optional spectator torsion proposals at lambda=0; the scan carries the selected conformer.
    n_conformers: int = 4

    # rxts uses its own Langevin sampler at every lambda; it does not reproduce VeloxChem's integrator.
    md_steps: int = 0
    md_temperature: float = 600.0  # K
    md_timestep: float = 0.5  # fs

    # VeloxChem's peak-band search with rxembed proposals in place of MD.
    n_embed: int = 4
    # Half-width in lambda windows.
    embed_window: int = 2
    # Deliberate divergence from VeloxChem's per-window max-min: carry the peak conformer
    # across the path to avoid spectator-fold jumps. False restores per-window selection.
    peak_anchored: bool = True

    # Optional physical ranking of FF-minimised anchored peak candidates; the FF stays unchanged.
    embed_score: str | CalculatorMethod | scoring.BaseCalculator | None = None

    # VeloxChem's discontinuity search, used when peak anchoring is off.
    discont_search: bool = False

    # GFN-FF carries ALPB. Optional conformer scoring uses this phase; path.score chooses its own.
    solvent: str | None = None

    # Affinity-aware budget for independent scoring and rxembed workers.
    n_cores: int | None = None

    _reactant_isomer: Isomer | None = field(default=None, init=False, repr=False)
    _product_isomer: Isomer | None = field(default=None, init=False, repr=False)
    _product_mapping: list[int] | None = field(default=None, init=False, repr=False)
    _sampling_failures: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _conformer_selections: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    charge: int | None = None
    multiplicity: int | None = None

    def __post_init__(self):
        """Reject invalid settings together, before endpoint preparation or scanning."""
        self.embed_score = scoring._normalise_method(self.embed_score)
        bad = []

        def check(name, valid, requirement):
            if not valid:
                bad.append(f"  {name}={getattr(self, name)!r} -- must be {requirement}")

        check("n_lambda", _an_int(self.n_lambda, 2), "an int >= 2")
        check("seed", _an_int(self.seed), "an int")
        check("forming_bonds", _pairs(self.forming_bonds), "None, or atom-index pairs")
        check("breaking_bonds", _pairs(self.breaking_bonds), "None, or atom-index pairs")
        check(
            "mapping",
            self.mapping is None
            or (isinstance(self.mapping, list | tuple) and all(_an_int(i, 0) for i in self.mapping)),
            "None, or non-negative atom indices",
        )
        for name in ("reactant_geometry", "product_geometry"):
            value = getattr(self, name)
            check(name, value is None or (isinstance(value, str) and bool(value.strip())), "None or a geometry name")
        check("engine", self.engine is None or self.engine in _ENGINES, f"None or one of {_ENGINES}")
        for name in ("morse_d", "md_temperature", "md_timestep"):
            value = getattr(self, name)
            check(name, _finite(value) and value > 0, "a finite positive number")
        check("soft_core", _finite(self.soft_core) and self.soft_core >= 0, "a finite non-negative number")
        check("constraints", _targets(self.constraints), "atom-pair distances with distinct atoms and positive values")
        check("constraint_k", _finite(self.constraint_k) and self.constraint_k >= 0, "a finite non-negative number")
        check("metal_angle_k", _finite(self.metal_angle_k) and self.metal_angle_k >= 0, "a finite non-negative number")
        check("n_conformers", _an_int(self.n_conformers, 1), "an int >= 1")
        check("md_steps", _an_int(self.md_steps, 0), "an int >= 0")
        check("n_embed", _an_int(self.n_embed, 0), "an int >= 0")
        check("embed_window", _an_int(self.embed_window, 0), "an int >= 0")
        check("peak_anchored", isinstance(self.peak_anchored, bool), "True or False")
        check(
            "embed_score",
            self.embed_score is None
            or isinstance(self.embed_score, CalculatorMethod)
            or (isinstance(self.embed_score, str) and self.embed_score in METHODS),
            f"None, an ASE Calculator, a CalculatorMethod, or one of {sorted(METHODS)}",
        )
        check("discont_search", isinstance(self.discont_search, bool), "True or False")
        check("solvent", self.solvent is None or isinstance(self.solvent, str), "None or a solvent name")
        check("n_cores", self.n_cores is None or _an_int(self.n_cores, 1), "None or an int >= 1")
        check("charge", self.charge is None or _an_int(self.charge), "None or an int")
        check("multiplicity", self.multiplicity is None or _an_int(self.multiplicity, 1), "None or an int >= 1")
        if bad:
            raise ValueError(
                f"TransitionStateGuesser cannot run with {'this setting' if len(bad) == 1 else 'these settings'}:\n"
                + "\n".join(bad)
            )

    def prepare_endpoints(
        self,
        reactant: str | Path | Chem.Mol | Sequence[str | Chem.Mol] | Conformers | Isomer,
        product: str | Path | Chem.Mol | Sequence[str | Chem.Mol] | Conformers | Isomer | None = None,
    ) -> PreparedReaction:
        return _prepare_endpoints(
            reactant,
            product,
            seed=self.seed,
            forming_bonds=self.forming_bonds,
            breaking_bonds=self.breaking_bonds,
            mapping=self.mapping,
            reactant_geometry=self.reactant_geometry,
            product_geometry=self.product_geometry,
            charge=self.charge,
            multiplicity=self.multiplicity,
            solvent=self.solvent,
            n_cores=self.n_cores,
        )

    def find_transition_state(
        self,
        reactant: str | Path | Chem.Mol | Sequence[str | Chem.Mol] | Conformers | Isomer | PreparedReaction,
        product: str | Path | Chem.Mol | Sequence[str | Chem.Mol] | Conformers | Isomer | None = None,
    ) -> ReactionPath:
        """Build an FF path; optionally score peak candidates on a physical calculator."""
        if self.embed_score is not None and (self.n_embed < 1 or not self.peak_anchored):
            raise ValueError(
                "embed_score requires n_embed > 0 and peak_anchored=True; "
                "it ranks the anchored peak search, not every MD proposal or the finished path"
            )
        if self.engine is None:
            raise ValueError("FF interpolation requires an explicit engine=; pass 'auto', 'mmff' or 'gfnff'")
        if self.engine not in _ENGINES:
            raise ValueError(f"Unknown engine {self.engine!r}; expected one of {', '.join(repr(e) for e in _ENGINES)}.")
        if isinstance(reactant, PreparedReaction):
            if product is not None:
                raise ValueError("A prepared reaction already contains both endpoints; omit product")
            prepared = reactant
        else:
            prepared = self.prepare_endpoints(reactant, product)
        n_atoms = prepared.reactant.GetNumAtoms()
        if any(i >= n_atoms for pair in self.constraints for i in pair):
            raise ValueError(f"constraints must use atom indices in [0, {n_atoms})")
        if self.embed_score is not None and not (prepared.forming_bonds or prepared.breaking_bonds):
            raise ValueError(
                "embed_score requires changing bonds for peak re-embedding; "
                "for a conformational transition, omit embed_score and score the finished path"
            )
        runner = replace(
            self,
            charge=prepared.charge,
            multiplicity=prepared.multiplicity,
            solvent=prepared.solvent,
            n_cores=prepared.n_cores,
        )
        runner._reactant_isomer = prepared.native_states[0]
        runner._product_isomer = prepared.native_states[1]
        runner._product_mapping = prepared.mapping
        runner._sampling_failures = []
        runner._conformer_selections = []
        rea, pro = Chem.Mol(prepared.reactant), Chem.Mol(prepared.product)
        forming, breaking = set(prepared.forming_bonds), set(prepared.breaking_bonds)
        engine = runner._choose_engine(rea, pro)
        pot = runner._build_potential(
            engine=engine, rea=rea, pro=pro, forming=forming, breaking=breaking, charge=prepared.charge
        )
        try:
            frames = runner._scan(pot, rea, prepared.charge)
        except BaseException as error:
            # Retain paid evidence when a later step fails or the caller interrupts.
            error.conformer_selections = list(runner._conformer_selections)  # ty: ignore[unresolved-attribute]
            error.sampling_failures = list(runner._sampling_failures)  # ty: ignore[unresolved-attribute]
            raise
        path = ReactionPath(
            frames=frames,
            reactant=rea,
            product=pro,
            forming_bonds=list(prepared.forming_bonds),
            breaking_bonds=list(prepared.breaking_bonds),
            mapping=list(prepared.mapping),
            engine=engine,
            charge=prepared.charge,
            multiplicity=prepared.multiplicity,
            n_cores=prepared.n_cores,
            solvent=prepared.solvent,
            mol=rea,
            endpoint_identity=prepared.endpoint_identity,
            metal_angle_k=pot.metal_angle_k,
            metal_angle_reference=pot.metal_angle_source,
            sampling_failures=list(runner._sampling_failures),
            conformer_selections=list(runner._conformer_selections),
        )
        logger.info(path.summary("ff"))
        return path

    def _choose_engine(self, rea, pro):
        """Choose MMFF when both endpoints type, otherwise GFN-FF when `engine='auto'`."""
        if self.engine == "mmff":
            check_mmff(rea, "reactants")
            check_mmff(pro, "products")
            return "mmff"
        if self.engine == "gfnff":
            return "gfnff"
        if self.engine != "auto":
            raise ValueError(f"Unknown engine {self.engine!r}; expected one of {', '.join(repr(e) for e in _ENGINES)}.")

        if mmff_can_type(rea) and mmff_can_type(pro):
            return "mmff"

        bad = untypeable_fragments(rea) + untypeable_fragments(pro)
        logger.warning(
            f"MMFF94 cannot type {', '.join(sorted(set(bad)))} -- using GFN-FF, which has no atom typing at all."
        )
        return "gfnff"

    def _build_potential(self, engine, rea, pro, forming, breaking, charge):
        # With no changing bonds, drive the most changed torsion instead.
        torsion = None
        if not forming and not breaking:
            torsion = active_torsion(rea, pro)
            indices, phi_r, phi_p = torsion
            logger.info(
                f"No bonds change: this is a conformational transition state. Driving the "
                f"torsion {describe(rea, indices)}, {phi_r:+.1f} -> {phi_p:+.1f} deg."
            )
            if engine == "mmff":
                raise NotImplementedError(
                    "Conformational transition states are implemented on the GFN-FF engine only. Pass engine='gfnff'."
                )

        if engine == "gfnff":
            # GFN-FF perceives topology from coordinates, so a mismatched endpoint gives the wrong field.
            require_topology(rea, bond_set(rea), "reactant")
            require_topology(pro, bond_set(pro), "product")

        metal_refs, metal_source = (((), ()), None)
        if self.metal_angle_k:
            metal_refs, metal_source = metal_angle_references(
                rea,
                pro,
                native_states=(self._reactant_isomer, self._product_isomer),
                product_mapping=self._product_mapping,
            )

        if engine == "mmff":
            return MMFFPotential(
                rea=rea,
                pro=pro,
                forming=forming,
                breaking=breaking,
                morse_d=self.morse_d,
                soft_core=self.soft_core,
                metal_angle_k=self.metal_angle_k,
                metal_angle_references=metal_refs,
                metal_angle_source=metal_source,
            )

        return GFNFFPotential(
            rea=rea,
            pro=pro,
            forming=forming,
            breaking=breaking,
            charge=charge,
            n_cores=self.n_cores,
            torsion=torsion,
            solvent=self.solvent,
            metal_angle_k=self.metal_angle_k,
            metal_angle_references=metal_refs,
            metal_angle_source=metal_source,
        )

    def _scan(self, pot, rea, charge):
        lambdas = np.linspace(0.0, 1.0, self.n_lambda)
        x = rea.GetConformer().GetPositions()
        rng = np.random.default_rng(self.seed)

        x = self._best_conformer(pot=pot, rea=rea, x=x, rng=rng)

        logger.info(f"{'lambda':>7s} {'E_reactant':>12s} {'E_product':>12s} {'V':>10s}")
        logger.info(f"{'':->7s} {'':->12s} {'':->12s} {'':->10s}")

        frames = []
        for lam in lambdas:
            x = self._minimize(pot, x, lam)
            candidates = [x]
            if self.md_steps:
                # MD proposals are relaxed on the same lambda sampling surface.
                for _ in range(self.n_conformers - 1):
                    shaken = self._sample(pot=pot, x=x, lam=lam, rng=rng)
                    candidates.append(self._minimize(pot, shaken, lam))

            # Minimise on the sampling surface, then score on the unrestrained interpolation.
            scored = [(pot.energy(c, lam), c) for c in candidates]
            res, x = min(scored, key=lambda s: s[0][0])
            frame = _frame(res, x, float(lam))
            frames.append(frame)
            logger.info(f"{lam:7.3f} {frame.e_reactant:12.2f} {frame.e_product:12.2f} {frame.energy:10.2f}")

        if self.n_embed > 0:
            frames = self._refine_guess(pot=pot, rea=rea, frames=frames, charge=charge)
        else:
            frames = self._fix_discontinuities(pot, frames)

        # Remove rigid-body drift without changing internal geometry.
        for frame, aligned in zip(frames, align_frames([f.positions for f in frames]), strict=True):
            frame.positions = aligned

        return frames

    def _refine_guess(self, pot, rea, frames, charge):
        """Search peak windows with rxembed proposals, then anchor or repair discontinuities."""
        searched = set()
        last = len(frames) - 1

        embedded, improved = 0, 0.0
        while len(searched) <= last:
            peak = int(np.argmax([f.energy for f in frames]))
            if {peak, max(peak - 1, 0), min(peak + 1, last)} <= searched:
                break
            band = range(max(peak - self.embed_window, 0), min(peak + self.embed_window, last) + 1)
            fresh = [i for i in band if i not in searched]
            if not fresh:
                break
            logger.debug(
                f"  peak at lambda {frames[peak].lam:.2f}: conformer search, {self.n_embed} embed(s)/"
                f"window over {len(fresh)} window(s), band +-{self.embed_window}"
            )
            n, imp = self._conformer_scan(pot=pot, rea=rea, frames=frames, charge=charge, windows=fresh, peak=peak)
            embedded += n
            improved = max(improved, imp)
            searched.update(fresh)
        if not embedded:
            if self.embed_score is not None:
                raise RuntimeError(
                    "Requested physical conformer scoring could not run: peak sampling produced no candidates. "
                    "Inspect the sampling warnings; set embed_score=None only to accept an FF-only path."
                )
            warnings.warn(
                "Peak conformer search produced no candidates; the force-field path was retained.",
                RuntimeWarning,
                stacklevel=2,
            )
            return frames
        logger.info(
            f"Peak sampling: {embedded} candidate(s) over {len(searched)} window(s) around "
            f"lambda {frames[int(np.argmax([f.energy for f in frames]))].lam:.2f}; "
            f"best improved V by {improved:.2f} kcal/mol."
        )

        if self.peak_anchored:
            self._anchor_on_peak(pot=pot, rea=rea, frames=frames, charge=charge)
        elif self.discont_search:
            while len(searched) <= last:
                bad = sorted(self._discontinuities(frames) - searched)
                if not bad:
                    break
                lams = ", ".join(f"{frames[i].lam:.2f}" for i in bad)
                logger.info(f"Discontinuity at lambda {lams}: conformer-searching there too.")
                _, imp = self._conformer_scan(pot=pot, rea=rea, frames=frames, charge=charge, windows=bad)
                logger.debug(f"  best improved V by {imp:.2f} kcal/mol.")
                searched.update(bad)

        return frames

    def _embed_minimised(self, pot, rea, x, lam):
        """Relax rxembed proposals on the scan FF; retain failed attempts without ranking them."""
        proposals = embed_candidates(
            mol=rea,
            x=x,
            targets=pot.targets(lam),
            n=self.n_embed,
            n_cores=self.n_cores,
            seed=self.seed,
            isomer=self._reactant_isomer,
        )
        relaxed, failures = [], []
        for trial in proposals:
            try:
                relaxed.append(self._minimize(pot, trial, lam))
            except _FFRelaxationError as error:
                failures.append(error)
                self._sampling_failures.append(
                    {
                        "lambda": error.lam,
                        "reason": str(error),
                        "positions_angstrom": error.positions.tolist(),
                        "energy_ev": error.energy_ev,
                        "forces_ev_per_angstrom": error.forces_ev_per_angstrom.tolist(),
                    }
                )
        if failures:
            warnings.warn(
                f"Rejected {len(failures)} of {len(proposals)} re-embedded proposal(s) at lambda={lam:.3f}: "
                "force-field relaxation did not converge.",
                RuntimeWarning,
                stacklevel=2,
            )
        return relaxed

    def _anchor_on_peak(self, pot, rea, frames, charge):
        """Select a peak conformer, then relax it outward across every lambda."""
        last = len(frames) - 1
        peak = int(np.argmax([f.energy for f in frames]))
        lam = frames[peak].lam
        band_lo = max(peak - self.embed_window, 1)
        band_hi = min(peak + self.embed_window, last - 1)

        candidates = [self._minimize(pot, frames[i].positions, lam) for i in range(band_lo, band_hi + 1)]
        candidates += self._embed_minimised(pot, rea, frames[peak].positions, lam)

        best_x = self._pick_conformer(pot=pot, rea=rea, candidates=candidates, lam=lam, charge=charge)
        frames[peak] = _frame(pot.energy(best_x, lam), best_x, lam)

        # Relax the chosen structure outward in both directions to keep one continuous path.
        for windows in (range(peak + 1, last + 1), range(peak - 1, -1, -1)):
            x = frames[peak].positions
            for i in windows:
                x = self._minimize(pot, x, frames[i].lam)
                frames[i] = _frame(pot.energy(x, frames[i].lam), x, frames[i].lam)
        logger.info("Anchored the path on the peak conformer, held across the interior.")

    def _pick_conformer(self, pot, rea, candidates, lam, charge):
        """Choose the lowest FF candidate, or rank these FF-minimised geometries with `embed_score`."""
        if not candidates:
            raise RuntimeError(
                "Peak conformer selection has no candidates; increase n_lambda or inspect the embedding warnings."
            )
        serial = isinstance(self.embed_score, CalculatorMethod) and self.embed_score.calculator is not None
        if serial and self.n_cores not in (None, 1):
            raise ValueError("Configured conformer calculators require serial scoring; pass n_cores=1")
        ff_energies = tuple(float(pot.energy(candidate, lam)[0]) for candidate in candidates)
        ff_best = min(range(len(candidates)), key=lambda i: ff_energies[i])
        if not self.embed_score:
            return candidates[ff_best]

        cores = 1 if serial else self.n_cores or default_cores()
        logger.info(
            f"Scoring {len(candidates)} peak conformer(s) with {method_label(self.embed_score)} to choose the guess..."
        )
        atoms = tuple(
            to_atoms(set_positions(rea, x), charge=charge, multiplicity=self.multiplicity) for x in candidates
        )
        result = scoring.score(
            atoms,
            charge=charge,
            multiplicity=self.multiplicity,
            method=self.embed_score,
            solvent=self.solvent,
            n_cores=cores,
        )
        selection = {
            "lambda": float(lam),
            "candidates": atoms,
            "scores": result,
            "selected_index": None,
            "ff_energies_kcal_per_mol": ff_energies,
            "ff_selected_index": ff_best,
        }
        self._conformer_selections.append(selection)
        if result.failures:
            raise ConformerScoringError(result, atoms, selection=selection)
        qm_best = int(np.argmin(np.asarray(result.energies_ev, dtype=float)))
        selection["selected_index"] = qm_best
        if qm_best != ff_best:
            logger.info("  QM chose a different conformer than the force field would have.")
        return candidates[qm_best]

    def _conformer_scan(self, pot, rea, frames, charge, windows, peak=None):
        """Re-run selected windows and retain their lowest-energy candidates."""

        def consider(best, y, lam):
            frame = _frame(pot.energy(y, lam), y, lam)
            return frame if frame.energy < best.energy else best

        def search(i, seed):
            """Embed at window i, with `seed` relaxed there as one more candidate. Keep the lowest V."""
            lam = frames[i].lam
            best = consider(frames[i], self._minimize(pot, seed, lam), lam)
            relaxed = self._embed_minimised(pot, rea, frames[i].positions, lam)
            for candidate in relaxed:
                best = consider(best, candidate, lam)
            here = len(relaxed)
            was = frames[i].energy
            where = "peak" if i == peak else "neighbour"
            verdict = f"V {was:.2f} -> {best.energy:.2f} (lower)" if best.energy < was - 1e-3 else "no lower structure"
            logger.debug(f"  lambda {lam:.2f} ({where}): {here} conformer(s), {verdict}")
            frames[i] = best
            return here, was - best.energy

        windows = sorted(windows)
        embedded, improved = 0, 0.0

        if peak is not None and peak in windows:
            n, imp = search(peak, frames[peak].positions)
            embedded += n
            improved = max(improved, imp)
            for side in ([w for w in windows if w > peak], [w for w in windows if w < peak][::-1]):
                x = frames[peak].positions
                for i in side:
                    n, imp = search(i, x)
                    embedded += n
                    improved = max(improved, imp)
                    x = frames[i].positions  # carry the winner onward
        else:
            x = frames[windows[0]].positions
            for i in windows:
                n, imp = search(i, x)
                embedded += n
                improved = max(improved, imp)
                x = frames[i].positions
        return embedded, improved

    def _discontinuities(self, frames):
        """Return interior windows where either endpoint energy moves against the scan direction."""
        bad = set()
        for i in range(len(frames) - 1):
            if frames[i].e_reactant > frames[i + 1].e_reactant:
                bad |= {i, i + 1}
            if frames[i].e_product < frames[i + 1].e_product:
                bad |= {i, i + 1}
        return bad - {0, len(frames) - 1}

    def _best_conformer(self, pot, rea, x, rng):
        """Resample spectator torsions and keep the lowest-energy reactant structure."""
        if self.n_conformers <= 1:
            return x

        candidates = conformers(mol=rea, x=x, reactive=pot.changing, n=self.n_conformers, rng=rng)
        if not candidates:
            return x

        best = self._minimize(pot, x, 0.0)
        best_e = pot.energy(best, 0.0)[0]
        starting = best_e

        for trial_x in candidates:
            trial = self._minimize(pot, trial_x, 0.0)
            e = pot.energy(trial, 0.0)[0]
            if e < best_e:
                best, best_e = trial, e

        logger.info(
            f"Conformer search: {len(candidates)} spectator conformer(s); reactive atoms excluded from torsion moves. "
            f"{starting:.2f} -> {best_e:.2f} kcal/mol"
        )
        return best

    def _energy_grad(self, pot, x, lam):
        """Evaluate the sampling surface and user distance restraints."""
        e, g = pot.sampling_energy(x, lam)
        e_user, g_user = 0.0, np.zeros_like(x)
        for (i, j), target in self.constraints.items():
            d = x[i] - x[j]
            r = float(np.linalg.norm(d))
            e_user += 0.5 * self.constraint_k * (r - target) ** 2
            de = self.constraint_k * (r - target)
            g_user[i] += de * d / r
            g_user[j] -= de * d / r
        return e + e_user, g + g_user

    def _fix_discontinuities(self, pot, frames):
        """Re-run interior windows where endpoint energies indicate a conformer hop."""
        # Identical endpoint fields make the monotonicity test unsuitable for conformational paths.
        if not pot.changing:
            return frames

        stuck = set()
        for _ in range(2):
            bad = self._discontinuities(frames) - stuck
            if not bad:
                return frames

            logger.info(f"Re-running {len(bad)} lambda window(s) where the path jumped: {sorted(bad)}")
            for i in sorted(bad):
                seed = frames[i - 1].positions if i - 1 not in bad else frames[i + 1].positions
                x = self._minimize(pot, seed.copy(), frames[i].lam)
                frame = _frame(pot.energy(x, frames[i].lam), x, frames[i].lam)
                if frame.energy < frames[i].energy:
                    frames[i] = frame
                else:
                    stuck.add(i)

        return frames

    def _minimize(self, pot, x, lam):
        """Relax one scan window with bounded Cartesian steps; failure retains its last FF state."""
        atoms = Atoms(
            numbers=pot.numbers,
            positions=np.asarray(x).reshape(pot.n_atoms, 3),
            calculator=_SamplingCalculator(self, pot, lam),
        )
        # ASE accepts None in Dynamics; LBFGS's logfile annotation omits it.
        opt = LBFGS(atoms, trajectory=None, logfile=None)  # ty: ignore[invalid-argument-type]
        converged = opt.run(fmax=0.05, steps=2000)
        positions = atoms.get_positions()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        if not converged or not (np.isfinite(positions).all() and np.isfinite(energy) and np.isfinite(forces).all()):
            reason = "did not converge" if not converged else "returned nonfinite values"
            raise _FFRelaxationError(lam, positions, energy, forces, reason)
        return positions

    def _sample(self, pot, x, lam, rng):
        """Brief Langevin dynamics, to move the structure out of a local minimum."""
        kb = 0.0019872041  # kcal/mol/K
        dt = self.md_timestep * 0.02045  # fs -> AKMA-ish time units for amu/A/kcal
        m = pot.masses[:, None]
        gamma = 0.1
        x = x.copy()
        v = rng.normal(size=x.shape) * np.sqrt(kb * self.md_temperature / m)

        f = -self._energy_grad(pot, x, lam)[1]
        for _ in range(self.md_steps):
            v += 0.5 * dt * f / m
            x += dt * v
            f = -self._energy_grad(pot, x, lam)[1]
            v += 0.5 * dt * f / m
            # Langevin thermostat
            c = np.exp(-gamma)
            v = c * v + np.sqrt((1 - c**2) * kb * self.md_temperature / m) * rng.normal(size=x.shape)
        return x


def interpolate(prepared: PreparedReaction, *, engine: str, **settings: Any) -> ReactionPath:
    """Interpolate prepared endpoints on an explicitly selected force field."""
    for name in ("charge", "multiplicity", "solvent", "n_cores"):
        if name in settings:
            if settings[name] != getattr(prepared, name):
                raise ValueError(f"{name} is fixed by prepare_endpoints(); pass the value there")
            del settings[name]
    unused = {"mapping", "forming_bonds", "breaking_bonds", "reactant_geometry", "product_geometry"} & settings.keys()
    if unused:
        raise TypeError(f"{', '.join(sorted(unused))} are endpoint options; pass them to prepare_endpoints()")

    return TransitionStateGuesser(engine=engine, **settings).find_transition_state(prepared)


__all__ = ["ConformerScoringError", "interpolate"]


# Hold reactive distances without holding the attack angle. FF relaxation then settles each proposal.
_ORGANIC_DISTANCE_PAD = 0.05


def rotatable_torsions(mol, reactive=()):
    """List acyclic spectator torsions, excluding bonds or atoms in the reaction center."""
    frozen = {int(a) for pair in reactive for a in pair}
    edges = sorted(perceived_bonds(mol))

    neighbours = {i: [] for i in range(mol.GetNumAtoms())}
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)

    ring_info = mol.GetRingInfo()
    in_ring = set()
    for ring in ring_info.AtomRings():
        for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
            in_ring.add(tuple(sorted((a, b))))

    torsions = []
    for j, k in edges:
        if (j, k) in in_ring:
            continue  # rotating a ring bond tears the ring
        if j in frozen or k in frozen:
            continue  # this bond is the reaction, not a spectator
        left = [a for a in neighbours[j] if a != k and a not in frozen]
        right = [a for a in neighbours[k] if a != j and a not in frozen]
        if not left or not right:
            continue  # a bond to a terminal atom has no torsion about it
        torsions.append((left[0], j, k, right[0]))
    return torsions


def resample_torsions(mol, x, torsions, rng):
    if not torsions:
        return x.copy()

    work = set_positions(mol, x)
    conf = work.GetConformer()

    for torsion in torsions:
        angle = float(rng.uniform(-180.0, 180.0))
        try:
            a, b, c, d = torsion
            rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, angle)
        except (ValueError, RuntimeError):
            continue  # a degenerate (collinear) torsion has no angle to set; leave it alone

    return np.asarray(conf.GetPositions())


def conformers(mol, x, reactive, n, rng):
    """Propose uniform spectator-torsion rotations for the FF scan to relax."""
    if n <= 1:
        return []
    torsions = rotatable_torsions(mol, reactive)
    if not torsions:
        return []
    return [resample_torsions(mol=mol, x=x, torsions=torsions, rng=rng) for _ in range(n - 1)]


def metal_core(mol, x, targets):
    """Return the metal, its current donors, and reactive atoms to keep fixed during embedding."""
    metals = {int(m) for m in metal_atoms(mol)}
    if not metals:
        return []
    core = {int(a) for pair in targets for a in pair} | metals
    for i, j in perceived_bonds(set_positions(mol, x)):
        if i in metals or j in metals:
            core |= {i, j}  # a donor: what the metal is holding at this geometry, not at an endpoint's
    return sorted(core)


def embed_candidates(mol, x, targets, n, *, n_cores=None, seed=0xF00D, isomer=None):
    """Propose conformers with held organic distances or a fixed metal core.

    A supplied isomer gives raw distance-geometry seeds; the reaction FF relaxes them before ranking.
    """
    if n <= 0 or not targets:
        return []

    fixed = metal_core(mol, x, targets)
    threads = _worker_count(int(n), n_cores)
    options: dict[str, Any]
    if isomer is not None and fixed:
        # TS coordinates own the rigid core; the saved state supplies its coordination model.
        source = rxembed.Isomer.from_state(
            with_conformer(isomer.mol, x),
            isomer.centres,
            isomer.donor_bonds,
            constraints=isomer.base_cons,
            constrained_metals=isomer.constrained_metals,
            lengths=isomer.lengths,
            protect_arrangement=False,
        )
        embed = rxembed.core.embed
        options = {"fix": fixed, "seed": seed, "threads": threads}
    else:
        source = set_positions(mol, x)
        embed = rxembed.embed
        options = (
            {"fix": fixed}
            if fixed
            else {
                "fix": {
                    pair: (distance - _ORGANIC_DISTANCE_PAD, distance + _ORGANIC_DISTANCE_PAD)
                    for pair, distance in targets.items()
                }
            }
        )
        options["charge"] = Chem.GetFormalCharge(mol)
        options.update(seed=seed, threads=threads)

    try:
        logger.debug(
            "  re-embedding %d targets with %s.", len(targets), "a rigid core" if fixed else "distance windows"
        )
        ensemble = embed(source, **options, n=int(n))
        placed = ensemble.mol  # a fresh copy on every access, so bind it once
        return [np.asarray(placed.GetConformer(i).GetPositions()) for i in ensemble.ids]
    except Exception as exc:
        # A proposal step that cannot propose costs the sampling, not the scan, so warn rather than raise.
        warnings.warn(
            f"Distance-geometry sampling failed at this window; the scan continued without it: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
