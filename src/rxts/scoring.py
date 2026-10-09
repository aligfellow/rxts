"""Fixed-frame scoring and ASE calculators for physical methods."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from threading import Lock
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakValueDictionary

import numpy as np
from ase import Atoms
from ase.calculators.calculator import BaseCalculator, Calculator, all_changes

from rxts.utils import _bounded_map, _validate_declared_context, validate_cores
from rxts.xtb import (
    BOHR,
    EV_PER_KCAL,
    HARTREE_TO_KCAL,
    METHODS,
    SOLVATION_BASE,
    _combine,
    _is_composite,
    composite_legs,
    select_backend,
)

if TYPE_CHECKING:
    from _thread import LockType
    from collections.abc import Callable


@dataclass(frozen=True)
class ScoreResult:
    """Ordered ASE energies and the indexed failures and context for one scoring attempt."""

    energies_ev: tuple[float | None, ...]
    failures: dict[int, str]
    n_evaluations: int
    n_gradients: int
    method: str
    charge: int | None
    multiplicity: int | None
    solvent: str | None


def _attached_label(calculator):
    provider = getattr(calculator, "calculator", calculator)
    owned = _gxtb_provider(provider)
    if owned is not None:
        return owned.method
    label = getattr(provider, "name", None)
    if not label or label == "unknown":
        label = type(provider).__name__
    return f"attached:{label}"


def _gxtb_provider(calculator):
    """Return an rxts-owned GXTB, unwrapping only rxts' own counter wrapper."""
    if isinstance(calculator, GXTB):
        return calculator
    if isinstance(calculator, _ExternalCalculator) and isinstance(calculator.calculator, GXTB):
        return calculator.calculator
    return None


def _resolve_gxtb_context(calculator, charge, multiplicity, solvent):
    """Use the configured physical context and refuse conflicting caller labels."""
    _validate_declared_context(charge, multiplicity)
    for name, requested, configured in (
        ("charge", charge, calculator.charge),
        ("multiplicity", multiplicity, calculator.multiplicity),
        ("solvent", solvent, calculator.solvent),
    ):
        if requested is not None and requested != configured:
            raise ValueError(
                f"score {name}={requested!r} conflicts with the configured xtb calculator's {name}={configured!r}"
            )
    return calculator.charge, calculator.multiplicity, calculator.solvent


def score(
    frames: Atoms | list[Atoms] | tuple[Atoms, ...],
    *,
    method: str | CalculatorMethod | BaseCalculator | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    solvent: str | None = None,
    n_cores: int | None = 1,
) -> ScoreResult:
    """Return absolute eV energies, indexed failures, and evaluation counts for fixed frames."""
    method = _normalise_method(method)
    frames = (frames,) if isinstance(frames, Atoms) else tuple(frames)
    if any(not isinstance(frame, Atoms) for frame in frames):
        raise TypeError("score accepts ASE Atoms frames")
    owned = None
    if isinstance(method, CalculatorMethod):
        owned = _gxtb_provider(method.calculator)
    elif method is None and frames:
        attached = [_gxtb_provider(frame.calc) for frame in frames]
        own_calculators = [calculator for calculator in attached if calculator is not None]
        for calculator in own_calculators:
            _resolve_gxtb_context(calculator, charge, multiplicity, solvent)
        if own_calculators and len(own_calculators) == len(attached):
            contexts = {(calculator.charge, calculator.multiplicity, calculator.solvent) for calculator in attached}
            if len(contexts) != 1:
                raise ValueError("Attached xtb calculators have different physical contexts; score them separately")
            owned = own_calculators[0]
    if owned is not None:
        charge, multiplicity, solvent = _resolve_gxtb_context(owned, charge, multiplicity, solvent)
    workers = 1 if n_cores is None else validate_cores(n_cores)
    if method is None:
        _validate_declared_context(charge, multiplicity)
    configured = isinstance(method, CalculatorMethod) and method.calculator is not None
    if (method is None or configured) and workers != 1:
        raise ValueError("Attached or configured ASE calculators require serial scoring (n_cores=1)")
    if (
        method is not None
        and not isinstance(method, CalculatorMethod)
        and (not isinstance(method, str) or method not in METHODS)
    ):
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(METHODS)}")
    if method is not None:
        _validate_builtin_context(method, charge, multiplicity)
    if isinstance(method, str) and any(np.any(frame.pbc) for frame in frames):
        raise ValueError("xtb methods in score require non-periodic molecular Atoms")
    for frame in frames:
        if len(frame) == 0:
            raise ValueError("score requires at least one atom per frame")
        if frame.positions.shape != (len(frame), 3) or not np.isfinite(frame.positions).all():
            raise ValueError("score requires finite Cartesian positions with shape (n_atoms, 3)")

    prepared = [copy_atoms(frame) for frame in frames]
    if method is None:
        wrappers = {}
        for frame in prepared:
            if frame.calc is None:
                raise ValueError("method=None requires an ASE calculator attached to every frame")
            key = id(frame.calc)
            if key not in wrappers:
                wrappers[key] = observed_calculator(frame.calc)
            frame.calc = wrappers[key]
        labels = list(dict.fromkeys(_attached_label(frame.calc) for frame in prepared))
        method_name = ", ".join(labels) if labels else "attached"
        calculators = list({id(calculator): calculator for calculator in wrappers.values()}.values())
    else:
        method_name = method.label if isinstance(method, CalculatorMethod) else method
        calculators = []
        for frame in prepared:
            frame.calc = make_calculator(
                charge=charge,
                multiplicity=multiplicity,
                method=method,
                solvent=solvent,
                n_cores=1,
                timeout=1800,
                fresh=not configured,
            )
            calculators.append(frame.calc)

    before = {
        id(calculator): (
            calculator,
            getattr(calculator, "n_evaluations", 0),
            getattr(calculator, "n_gradients", 0),
        )
        for calculator in calculators
    }

    def evaluate(frame):
        try:
            energy = np.asarray(frame.get_potential_energy())
            if energy.shape != () or not np.isfinite(energy):
                raise ValueError("calculator returned a non-scalar or non-finite energy")
            return float(energy), None
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"

    evaluated = (
        _bounded_map(prepared, evaluate, n_workers=int(workers)) if workers > 1 else list(map(evaluate, prepared))
    )
    energies = tuple(energy for energy, _reason in evaluated)
    failures = {index: reason for index, (_energy, reason) in enumerate(evaluated) if reason is not None}
    n_evaluations = sum(getattr(calculator, "n_evaluations", 0) - old for calculator, old, _ in before.values())
    n_gradients = sum(getattr(calculator, "n_gradients", 0) - old for calculator, _, old in before.values())
    return ScoreResult(energies, failures, n_evaluations, n_gradients, str(method_name), charge, multiplicity, solvent)


def copy_atoms(atoms: Atoms) -> Atoms:
    """Copy ASE state without copying the attached provider."""
    if not isinstance(atoms, Atoms):
        raise TypeError("Expected ASE Atoms")
    copied = atoms.copy()
    copied.info = deepcopy(atoms.info)
    copied.calc = atoms.calc
    return copied


def _same_state_value(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_state_value(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_same_state_value(a, b) for a, b in zip(left, right, strict=True))
    try:
        equal = left == right
    except Exception:
        return left is right
    return bool(equal) if np.ndim(equal) == 0 else bool(np.all(equal))


def _extra_state_changes(previous: Atoms, atoms: Atoms) -> list[str]:
    changes = []
    for name in previous.arrays.keys() | atoms.arrays.keys():
        if (
            name not in previous.arrays
            or name not in atoms.arrays
            or not _same_state_value(previous.arrays.get(name), atoms.arrays.get(name))
        ):
            changes.append(name)
    for name in ("cell", "pbc", "info"):
        if not _same_state_value(getattr(previous, name), getattr(atoms, name)):
            changes.append(name)
    return changes


def _validate_builtin_context(method, charge, multiplicity):
    _validate_declared_context(charge, multiplicity)
    if isinstance(method, CalculatorMethod):
        return
    if method is None:
        raise ValueError("An explicit physical method is required; pass a method such as 'gfn2'")
    if isinstance(charge, bool) or not isinstance(charge, Integral):
        raise ValueError("An explicit integer charge is required for a built-in physical method")
    if isinstance(multiplicity, bool) or not isinstance(multiplicity, Integral) or multiplicity < 1:
        raise ValueError("An explicit positive integer multiplicity is required for a built-in physical method")


@dataclass(frozen=True, eq=False)
class CalculatorMethod:
    """Name an ASE calculator factory or one configured instance for serial scoring.

    Factories receive charge, multiplicity, solvent, core count, and timeout, and must return a fresh
    calculator with independent child state and scratch. An instance keeps its existing settings.
    Energies and forces use ASE units.
    """

    label: str
    factory: Callable[..., BaseCalculator] | None = None
    calculator: BaseCalculator | None = field(default=None, kw_only=True, repr=False)
    _issued: WeakValueDictionary[int, BaseCalculator] = field(
        default_factory=WeakValueDictionary, init=False, repr=False
    )
    _lock: LockType = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        """Require an explicit physical label and exactly one calculator source."""
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("CalculatorMethod needs a nonempty label identifying its physical settings")
        if self.calculator is not None:
            if self.factory is not None:
                raise ValueError("CalculatorMethod accepts either factory or calculator, not both")
            if not isinstance(self.calculator, BaseCalculator):
                raise TypeError("CalculatorMethod.calculator must be a configured ASE Calculator")
        elif not callable(self.factory):
            raise TypeError("CalculatorMethod.factory must construct a fresh ASE calculator")


def _normalise_method(method):
    """Wrap a configured ASE calculator in the existing serial method boundary."""
    if isinstance(method, BaseCalculator):
        owned = _gxtb_provider(method)
        if owned is not None:
            return CalculatorMethod(owned.method, calculator=method)
        label = getattr(method, "name", None)
        if not isinstance(label, str) or not label.strip() or label == "unknown":
            label = type(method).__name__
        return CalculatorMethod(label, calculator=method)
    return method


class _ExternalCalculator(Calculator):
    """Adapt an ASE provider while counting requests and validating returned energy and forces."""

    def __init__(self, calculator, lock=None):
        super().__init__()
        self.calculator = calculator
        self._lock = lock if lock is not None else nullcontext()
        self.implemented_properties = list(calculator.implemented_properties or calculator.results)
        self.n_evaluations = 0
        self.n_gradients = 0

    def check_state(self, atoms, tol=1e-15):
        """Include physical arrays, metadata and periodic state in the ASE cache key."""
        changes = set(super().check_state(atoms, tol))
        changes.update(self.calculator.check_state(atoms, tol))
        if self.atoms is not None:
            # ASE owns tolerance for its standard arrays; compare only extra state exactly.
            changes.update(set(_extra_state_changes(self.atoms, atoms)) - set(all_changes))
        return sorted(changes)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        """Read only requested ASE properties, keeping any additional cached results."""
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("An ASE calculation requires Atoms")
        wrapper_changes = set(_extra_state_changes(self.atoms, atoms)) if self.atoms is not None else set()
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        self.atoms.info = deepcopy(atoms.info)
        self.n_evaluations += 1
        self.n_gradients += int("forces" in properties)
        with self._lock:
            provider_changes = set(self.calculator.check_state(atoms))
            state_changes = wrapper_changes
            if self.calculator.atoms is not None:
                state_changes.update(_extra_state_changes(self.calculator.atoms, atoms))
            missing_changes = state_changes - provider_changes
            if missing_changes:
                Calculator.reset(self.calculator)
            requested = {name: self.calculator.get_property(name, atoms) for name in properties}
            for name, value in requested.items():
                if name in {"energy", "forces"}:
                    expected = () if name == "energy" else (len(atoms), 3)
                    if np.asarray(value).shape != expected or not np.isfinite(value).all():
                        raise ValueError(
                            f"ASE provider returned nonfinite or malformed {name}; expected shape {expected}"
                        )
            if self.calculator.atoms is not None:
                self.calculator.atoms.info = deepcopy(self.calculator.atoms.info)
            # Opportunistic arrays may be a shared provider's reusable calculation buffer.
            self.results = deepcopy(self.calculator.results)
            self.results.update(requested)


class GXTB(Calculator):
    """ASE calculator for xtb-family energies and gradients; GFN-FF has its own dispatcher."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]  # ty: ignore[invalid-attribute-override]

    def __init__(self, charge=0, multiplicity=1, n_cores=1, solvent=None, method="gxtb", timeout=None, **kwargs):
        """Resolve `method` to a name and build the one backend runner that will serve this calculator."""
        super().__init__(**kwargs)
        self.solvent = solvent
        # Resolve before choosing a backend, and retain the actual method used.
        self.method = method
        self.charge = charge
        self.multiplicity = multiplicity
        self.n_cores = n_cores
        self._backend = select_backend(self.method).make(
            charge=charge, multiplicity=multiplicity, n_cores=n_cores, timeout=timeout
        )
        self.n_evaluations = 0
        self.n_gradients = 0

    @property
    def path(self):
        """The scratch directory owned by this calculator's backend."""
        return self._backend.path

    @property
    def n_calls(self):
        """Number of subprocess invocations, tracked by the backend (zero when it is in-process tblite)."""
        return self._backend.n_calls

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        """Evaluate requested ASE properties; energy-only calls do not compute gradients."""
        if atoms is not None and np.any(atoms.pbc):
            raise ValueError("xtb calculators do not support periodic ASE Atoms")
        if atoms is not None:
            contexts = [atoms.info]
            reaction = atoms.info.get("rxts")
            if isinstance(reaction, dict):
                contexts.append(reaction)
            for context in contexts:
                charge = context.get("charge")
                multiplicity = context.get("multiplicity")
                _validate_declared_context(charge, multiplicity)
                if charge is not None and charge != self.charge:
                    raise ValueError(
                        f"Atoms declare charge={charge}, but this xtb calculator uses charge={self.charge}"
                    )
                if multiplicity is not None and multiplicity != self.multiplicity:
                    raise ValueError(
                        f"Atoms declare multiplicity={multiplicity}, but this xtb calculator uses "
                        f"multiplicity={self.multiplicity}"
                    )
        super().calculate(atoms, properties, system_changes)
        forces = "forces" in properties
        energy, gradient = self._energy_gradient(atoms, forces)
        self.n_evaluations += 1
        self.n_gradients += int(forces)
        self.results["energy"] = energy * HARTREE_TO_KCAL * EV_PER_KCAL
        if forces:
            self.results["forces"] = -gradient * (HARTREE_TO_KCAL * EV_PER_KCAL / BOHR)

    def check_state(self, atoms, tol=1e-15):
        """Include full ASE state in the built-in calculator's result cache key."""
        changes = set(super().check_state(atoms, tol))
        if self.atoms is not None:
            # ASE owns tolerance for its standard arrays; compare only extra state exactly.
            changes.update(set(_extra_state_changes(self.atoms, atoms)) - set(all_changes))
        return sorted(changes)

    def _energy_gradient(self, atoms, gradient):
        """Return Hartree energy and optional (N, 3) Hartree/Bohr gradient, +dE/dx."""
        # The backend gets the semantic request; restart defaults on, so a plain GXTB warm-restarts
        # between successive geometries.
        return self._backend.run(atoms=atoms, method=self.method, solvent=self.solvent, gradient=gradient)


class SolvatedGXTB(GXTB):
    """Combine gas g-xTB with the GFN2 ALPB solvation correction.

    E = E_gxtb(gas) + E_gfn2(solvent) - E_gfn2(gas), also applied to gradients. Each leg
    owns a runner. GFN2 uses tblite where available; n_calls counts subprocesses, while
    n_gradients counts physical gradient evaluations."""

    def __init__(self, solvent, charge=0, multiplicity=1, n_cores=1, timeout=None, **kwargs):
        """Build the gas g-xTB leg, then the wet and dry GFN2 legs its solvation shift is differenced from."""
        # GXTB.__init__ resolves self._backend for method="gxtb" -- the gas leg's runner. The two GFN2
        # legs are resolved here, per leg, so they can go in-process independently of the gas leg.
        super().__init__(
            charge=charge,
            multiplicity=multiplicity,
            n_cores=n_cores,
            method="gxtb",
            timeout=timeout,
            **kwargs,
        )
        self.solvent = solvent
        make_leg = select_backend(SOLVATION_BASE).make
        self._base = self._backend  # g-xTB in the gas phase, from GXTB.__init__
        self._solvated = make_leg(
            charge=charge, multiplicity=multiplicity, n_cores=n_cores, timeout=timeout
        )  # GFN2 + ALPB
        self._unsolvated = make_leg(  # GFN2 in the gas phase -- its OWN runner
            charge=charge, multiplicity=multiplicity, n_cores=n_cores, timeout=timeout
        )

    @property
    def n_calls(self):
        """Subprocess invocations, summed over the three legs (one per evaluation on tblite, three on xtb)."""
        return self._base.n_calls + self._solvated.n_calls + self._unsolvated.n_calls

    def _energy_gradient(self, atoms, gradient):
        """Combine independent cold-start legs as gas + (wet - dry), including gradients if requested."""

        def leg(runner, leg_method, leg_solvent):
            # The three legs own separate runners, so none reads another surface's state.
            # restart=False keeps every repeated composite evaluation on the same cold field.
            energy, grad = runner.run(
                atoms=atoms, method=leg_method, solvent=leg_solvent, gradient=gradient, restart=False
            )
            return np.concatenate([[energy], grad.ravel()]) if gradient else np.array([energy])

        base_job, solvated_job, unsolvated_job = composite_legs(self.solvent)
        combined = _combine(
            leg(self._base, *base_job),
            leg(self._solvated, *solvated_job),
            leg(self._unsolvated, *unsolvated_job),
        )
        return combined[0], (combined[1:].reshape(len(atoms), 3) if gradient else None)


def make_calculator(*, charge=None, multiplicity=None, method=None, solvent=None, n_cores=1, timeout=None, fresh=True):
    """Acquire a fresh calculator; ``fresh=False`` permits a supplied instance for serial scoring.

    Built-in methods require charge/multiplicity; solvent=None selects gas phase.
    """
    if isinstance(n_cores, bool) or not isinstance(n_cores, Integral) or n_cores < 1:
        raise ValueError("n_cores must be a positive integer")
    if not isinstance(method, CalculatorMethod) and method is not None and method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(METHODS)}")
    _validate_builtin_context(method, charge, multiplicity)
    if isinstance(method, CalculatorMethod):
        if method.calculator is not None:
            if fresh:
                raise ValueError(
                    "This operation requires fresh ASE calculators; supply CalculatorMethod(label, factory) "
                    "instead of a configured instance"
                )
            return _ExternalCalculator(method.calculator, method._lock)
        assert method.factory is not None
        calculator = method.factory(
            charge=charge, multiplicity=multiplicity, solvent=solvent, n_cores=n_cores, timeout=timeout
        )
        if not isinstance(calculator, BaseCalculator):
            raise TypeError("CalculatorMethod.factory must return a fresh ASE Calculator")
        with method._lock:
            if id(calculator) in method._issued:
                raise ValueError("CalculatorMethod.factory reused a calculator; construct a fresh instance per call")
            method._issued[id(calculator)] = calculator
        return _ExternalCalculator(calculator)
    if _is_composite(method, solvent):
        return SolvatedGXTB(solvent, charge=charge, multiplicity=multiplicity, n_cores=n_cores, timeout=timeout)
    return GXTB(
        charge=charge,
        multiplicity=multiplicity,
        n_cores=n_cores,
        solvent=solvent,
        method=method,
        timeout=timeout,
    )


def observed_calculator(calculator):
    """Return an ASE calculator whose existing evaluation counters can be observed."""
    if not isinstance(calculator, BaseCalculator):
        raise TypeError("Attached providers must be ASE Calculators")
    if isinstance(calculator, (_ExternalCalculator, GXTB)):
        return calculator
    return _ExternalCalculator(calculator)
