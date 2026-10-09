"""Tests for fixed-frame physical scoring."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms

from rxts import scoring, xtb
from rxts.scoring import GXTB, SolvatedGXTB, copy_atoms, make_calculator, observed_calculator


def test_native_score_keeps_ordered_failures():
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    class Quadratic(Calculator):
        name = "quadratic"
        implemented_properties = ("energy",)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            x = float(atoms.positions[0, 0])
            if x < 0:
                raise RuntimeError("negative coordinate")
            self.results["energy"] = 0.5 * x**2

    provider = Quadratic()
    frames = [
        Atoms("H", positions=[[2, 0, 0]], calculator=provider),
        Atoms("H", positions=[[-1, 0, 0]], calculator=provider),
    ]
    before = [frame.positions.copy() for frame in frames]

    got = scoring.score(frames, method=None)

    assert got.energies_ev == (2.0, None)
    assert got.failures == {1: "RuntimeError: negative coordinate"}
    assert got.n_evaluations == 2
    assert got.n_gradients == 0
    assert got.method == "attached:quadratic"
    assert (got.charge, got.multiplicity, got.solvent) == (None, None, None)
    assert all(frame.calc is provider for frame in frames)
    assert all(np.array_equal(frame.positions, positions) for frame, positions in zip(frames, before, strict=True))


def test_native_score_preserves_state_and_context():
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    from rxts import CalculatorMethod

    contexts = []

    class MetadataSurface(Calculator):
        implemented_properties = ("energy",)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = (
                atoms.get_masses()[0]
                + atoms.arrays["initial_charges"][0]
                + atoms.arrays["initial_magmoms"][0]
                + atoms.arrays["probe"][0]
                + atoms.info["nested"]["offset"]
            )

    def factory(**context):
        contexts.append(context)
        return MetadataSurface()

    method = CalculatorMethod("metadata surface", factory)
    atoms = Atoms("H", positions=[[0.2, 0, 0]])
    atoms.set_masses([4.0])
    atoms.set_initial_charges([0.25])
    atoms.set_initial_magnetic_moments([0.5])
    atoms.new_array("probe", np.array([0.75]))
    atoms.info["nested"] = {"offset": 1.0}
    before = atoms.positions.copy()

    got = scoring.score([atoms], method=method, charge=-1, multiplicity=2, solvent="water")

    assert got.energies_ev == (6.5,)
    assert got.failures == {}
    assert got.n_evaluations == 1
    assert got.n_gradients == 0
    assert (got.method, got.charge, got.multiplicity, got.solvent) == ("metadata surface", -1, 2, "water")
    assert contexts == [{"charge": -1, "multiplicity": 2, "solvent": "water", "n_cores": 1, "timeout": 1800}]
    assert np.array_equal(atoms.positions, before)
    assert atoms.calc is None


def test_native_score_validates_before_evaluation(monkeypatch):
    from ase import Atoms

    atoms = Atoms("H", positions=[[0, 0, 0]], pbc=True)
    monkeypatch.setattr(scoring, "make_calculator", lambda **kwargs: pytest.fail("provider was constructed"))

    with pytest.raises(ValueError, match="explicit integer charge"):
        scoring.score(atoms, method="gfn2")
    with pytest.raises(ValueError, match="positive integer"):
        scoring.score(atoms, method="gfn2", charge=0, multiplicity=1, n_cores=0)

    with pytest.raises(ValueError, match="non-periodic"):
        scoring.score(atoms, method="gfn2", charge=0, multiplicity=1)

    atoms.pbc = False
    atoms.arrays["positions"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite Cartesian positions"):
        scoring.score(atoms, method="gfn2", charge=0, multiplicity=1)

    from ase import Atoms

    with pytest.raises(ValueError, match="at least one atom"):
        scoring.score(Atoms(), method="gfn2", charge=0, multiplicity=1)


def test_score_accepts_an_ase_calculator_instance(monkeypatch):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    class Surface(Calculator):
        implemented_properties = ("energy",)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": float(atoms.positions[0, 0])}

    attempt = scoring.score(Atoms("H", positions=[[2.0, 0, 0]]), method=Surface(), charge=0, multiplicity=1)
    assert attempt.energies_ev == (2.0,)
    assert attempt.method == "surface"
    assert attempt.failures == {}


# The ASE boundary: where Hartree becomes eV and a gradient becomes a force.
#
#: eV per Hartree, composed the way `GXTB.calculate` composes it rather than as a literal.
_EV_PER_HARTREE = xtb.HARTREE_TO_KCAL * xtb.EV_PER_KCAL


def _atoms(n=2):
    return Atoms(numbers=[6] * n, positions=np.arange(3 * n, dtype=float).reshape(n, 3))


class _FakeRunner:
    """A backend runner: the contract `XtbBackend.run`'s docstring writes, and a log of what was asked.

    The energy it returns depends on (method, solvent) alone, so the composite's three legs come back
    with three different numbers and a leg swapped for another is visible in the arithmetic.
    """

    #: (method, solvent) -> energy in Hartree. Distinct and not round, so no two legs can be confused.
    ENERGIES: ClassVar[dict] = {("gxtb", None): -3.0, ("gfn2", "water"): -11.0, ("gfn2", None): -7.0}

    def __init__(self, charge=0, multiplicity=1, n_cores=1, timeout=None):
        self.charge, self.multiplicity, self.n_cores, self.timeout = charge, multiplicity, n_cores, timeout
        self.path = "/scratch/fake"
        self.n_calls = 0
        self.requests = []  # (method, solvent, gradient, restart) per run

    def run(self, atoms, method, solvent, gradient=True, restart=True):
        self.requests.append((method, solvent, gradient, restart))
        self.n_calls += 1
        energy = self.ENERGIES.get((method, solvent), -1.0)
        if not gradient:
            return energy, None
        # A gradient that is a function of the leg, so `gas + (wet - dry)` is checkable componentwise.
        return energy, np.full((len(atoms), 3), energy)


class _FakeBackend:
    """A registry entry whose `make` hands back a :class:`_FakeRunner` and records it."""

    made: ClassVar[list] = []

    @classmethod
    def make(cls, charge=0, multiplicity=1, n_cores=1, timeout=None):
        runner = _FakeRunner(charge=charge, multiplicity=multiplicity, n_cores=n_cores, timeout=timeout)
        cls.made.append(runner)
        return runner


@pytest.fixture
def registry(monkeypatch):
    """Point scoring at the fake backend and hand back the list of runners it builds."""
    made: ClassVar[list] = []
    backend = type("_Fake", (_FakeBackend,), {"made": made})
    monkeypatch.setattr(scoring, "select_backend", lambda method: backend)
    return made


def test_energy_units_and_force_sign(registry):
    "Check Hartree conversion and force sign at the ASE boundary."
    calc = GXTB(method="gfn2", solvent=None)
    atoms = _atoms()
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert energy == pytest.approx(-7.0 * _EV_PER_HARTREE)
    assert np.allclose(forces, 7.0 * (_EV_PER_HARTREE / xtb.BOHR)), "forces are MINUS the gradient (-7.0)"
    assert registry[0].requests[0][:2] == ("gfn2", None), "the semantic request, not xtb flags"


def test_energy_request_skips_gradient(registry):
    calc = GXTB(method="gfn2")
    atoms = _atoms()
    atoms.calc = calc

    atoms.get_potential_energy()
    assert [gradient for _m, _s, gradient, _r in registry[0].requests] == [False]
    assert "forces" not in calc.results
    assert calc.n_evaluations == 1
    assert calc.n_gradients == 0

    atoms.get_forces()
    assert registry[0].requests[-1][2] is True
    assert calc.n_evaluations == 2
    assert calc.n_gradients == 1


def test_calculator_counters(registry):
    calc = GXTB(method="gfn2")
    atoms = _atoms()
    atoms.calc = calc

    atoms.get_forces()
    moved = atoms.copy()
    moved.positions[0, 0] += 0.1
    calc.calculate(moved, properties=("energy", "forces"))

    assert calc.n_gradients == 2
    assert calc.n_evaluations == 2
    assert calc.n_calls == registry[0].n_calls == 2
    assert calc.path == registry[0].path, "the scratch directory is the backend's, passed through"


def test_gxtb_default_ignores_path(registry, monkeypatch):
    "Resolve to the explicit g-xTB surface before backend selection."
    monkeypatch.setattr(xtb, "supports_gxtb", lambda: pytest.fail("default must not inspect PATH"))
    calc = GXTB()

    assert calc.method == "gxtb"
    calc.calculate(_atoms(), properties=("energy",))
    assert registry[0].requests[0][0] == "gxtb"
    with pytest.raises(ValueError, match="explicit physical method"):
        make_calculator(method=None)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"charge": 1}, "Atoms declare charge=1"),
        ({"rxts": {"multiplicity": 2}}, "Atoms declare multiplicity=2"),
    ],
)
def test_gxtb_refuses_mismatched_declared_ase_context(registry, metadata, message):
    atoms = _atoms()
    atoms.info.update(metadata)
    atoms.calc = GXTB(charge=0, multiplicity=1, method="gfn2")

    with pytest.raises(ValueError, match=message):
        atoms.get_potential_energy()
    assert all(not backend.requests for backend in registry)


def test_owned_xtb_method_and_state(registry):
    calculator = GXTB(charge=-1, multiplicity=2, method="gfn2")
    for method in (calculator, None):
        atoms = _atoms()
        if method is None:
            atoms.calc = calculator
        result = scoring.score(atoms, method=method)
        assert result.method == "gfn2"
        assert (result.charge, result.multiplicity, result.solvent) == (-1, 2, None)


def test_owned_xtb_rejects_state_override(registry):
    calculator = GXTB(charge=0, multiplicity=1, method="gfn2")

    with pytest.raises(ValueError, match="charge=1 conflicts with the configured xtb calculator"):
        scoring.score(_atoms(), method=calculator, charge=1, multiplicity=1)
    with pytest.raises(ValueError, match="charge must be an integer"):
        scoring.score(_atoms(), method=calculator, charge=False)
    assert all(not backend.requests for backend in registry)


def test_owned_xtb_solvent(registry):
    from rxts.ff import TransitionStateGuesser

    calculator = SolvatedGXTB("water", charge=-1, multiplicity=1)
    result = scoring.score(_atoms(), method=calculator)
    assert (result.method, result.charge, result.multiplicity, result.solvent) == ("gxtb", -1, 1, "water")

    with pytest.raises(ValueError, match="solvent='methanol' conflicts with the configured xtb calculator"):
        scoring.score(_atoms(), method=calculator, solvent="methanol")

    guesser = TransitionStateGuesser(engine="mmff", multiplicity=1, n_lambda=3, n_conformers=1, n_embed=0, n_cores=1)
    path = guesser.find_transition_state("CBr.[Cl-]>>CCl.[Br-]")
    assert not path.score(calculator).failures
    assert path.qm_solvent == "water"
    assert path.guess("qm").solvent == "water"


def test_only_solvated_gxtb_is_composite(registry, monkeypatch):
    with pytest.raises(ValueError, match="explicit integer charge"):
        make_calculator(method="gfn2")
    with pytest.raises(ValueError, match="positive integer"):
        make_calculator(method="gfn2", charge=0, multiplicity=1, n_cores=0)
    assert registry == [], "invalid chemistry and resources must fail before selecting a backend"

    assert isinstance(make_calculator(method="gxtb", charge=0, multiplicity=1, solvent="water"), SolvatedGXTB)
    for plain in (
        make_calculator(method="gxtb", charge=0, multiplicity=1),
        make_calculator(method="gfn2", charge=0, multiplicity=1, solvent="water"),
        make_calculator(method="gfn2", charge=0, multiplicity=1),
    ):
        assert isinstance(plain, GXTB)
        assert not isinstance(plain, SolvatedGXTB)


def test_composite_runs_cold_legs_and_shifts(registry):
    "Check three independent cold legs and gas + (wet - dry) for energy and gradient."
    calc = SolvatedGXTB("water")
    atoms = _atoms()
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    combined = -3.0 + (-11.0 - -7.0)
    assert energy == pytest.approx(combined * _EV_PER_HARTREE)
    assert np.allclose(forces, -combined * (_EV_PER_HARTREE / xtb.BOHR))

    assert calc._solvated is not calc._unsolvated, "one runner shared would cross-contaminate the shift"
    assert calc._base is not calc._solvated
    asked = [
        (runner.requests[0][0], runner.requests[0][1]) for runner in (calc._base, calc._solvated, calc._unsolvated)
    ]
    assert asked == [("gxtb", None), ("gfn2", "water"), ("gfn2", None)]
    assert all(request[3] is False for runner in registry for request in runner.requests), "every leg runs cold"

    assert calc.n_evaluations == 2
    assert calc.n_gradients == 1, "the energy-only read is not a gradient"
    assert calc.n_calls == 6, "and three legs' worth of invocations to whoever pays for them"


def test_external_counts_and_sharing():
    from ase.calculators.calculator import Calculator, all_changes
    from ase.calculators.mixing import SumCalculator

    from rxts import CalculatorMethod

    class SeparateProperties(Calculator):
        implemented_properties = ("energy", "forces")
        __hash__ = None

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {name: 2.5 if name == "energy" else np.full((len(atoms), 3), -0.7) for name in properties}

    made, contexts = [], []

    def factory(**context):
        contexts.append(context)
        made.append(SeparateProperties())
        return made[-1]

    method = CalculatorMethod("explicit test surface", factory)
    atoms = _atoms()
    atoms.calc = make_calculator(method=method, charge=-1, multiplicity=2, solvent="water", n_cores=3, timeout=10)
    assert contexts == [{"charge": -1, "multiplicity": 2, "solvent": "water", "n_cores": 3, "timeout": 10}]
    assert atoms.get_potential_energy() == 2.5
    assert atoms.calc.n_gradients == 0
    np.testing.assert_array_equal(atoms.get_forces(), np.full((2, 3), -0.7))
    atoms.get_forces()
    assert atoms.calc.n_gradients == 1
    atoms.positions[0, 0] += 0.1
    atoms.get_forces()
    assert atoms.calc.n_gradients == 2
    assert atoms.calc.n_evaluations == 3
    joint = atoms.get_properties(["energy", "forces"])
    assert joint["energy"] == 2.5
    np.testing.assert_array_equal(joint["forces"], np.full((2, 3), -0.7))
    assert method != CalculatorMethod(method.label, factory)
    assert make_calculator(method=method).calculator is not atoms.calc.calculator

    shared = CalculatorMethod("shared", lambda **kw: made[0])
    first = make_calculator(method=shared)
    with pytest.raises(ValueError, match="fresh instance"):
        make_calculator(method=shared)
    assert first.calculator is made[0]
    with pytest.raises(TypeError, match="fresh ASE Calculator"):
        make_calculator(method=CalculatorMethod("invalid", lambda **kw: None))
    with pytest.raises(ValueError, match="nonempty label"):
        CalculatorMethod("", factory)
    with pytest.raises(TypeError, match="factory"):
        CalculatorMethod("invalid", None)

    mixed = CalculatorMethod("two surfaces", lambda **kw: SumCalculator([SeparateProperties(), SeparateProperties()]))
    atoms.calc = make_calculator(method=mixed)
    assert atoms.get_potential_energy() == 5.0
    np.testing.assert_allclose(atoms.get_forces(), -1.4)
    assert atoms.calc.n_gradients == 1


def test_composite_units_and_lifetime():
    from ase.calculators.calculator import Calculator, all_changes
    from ase.calculators.mixing import SumCalculator

    from rxts import CalculatorMethod

    reads = []

    class Constant(Calculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            reads.append(tuple(properties))
            self.results = {name: 2.5 if name == "energy" else np.full((len(atoms), 3), -0.7) for name in properties}

    composite = SumCalculator([Constant(), Constant()])
    method = CalculatorMethod("configured summed surface", calculator=composite)
    with pytest.raises(ValueError, match=r"fresh ASE calculators.*factory"):
        make_calculator(method=method)
    assert reads == []
    with pytest.raises(ValueError, match="not both"):
        CalculatorMethod("ambiguous", lambda **context: Constant(), calculator=composite)
    with pytest.raises(TypeError, match="configured ASE Calculator"):
        CalculatorMethod("invalid", calculator=object())

    atoms = _atoms()
    atoms.calc = make_calculator(method=method, fresh=False)
    assert atoms.calc.calculator is composite
    assert atoms.get_potential_energy() == 5.0
    np.testing.assert_allclose(atoms.get_forces(), -1.4)
    atoms.get_forces()
    assert atoms.calc.n_gradients == 1
    second = make_calculator(method=method, fresh=False)
    assert second.calculator is composite
    assert second is not atoms.calc


def test_instance_serializes_reads():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from ase.calculators.calculator import Calculator, all_changes

    from rxts import CalculatorMethod

    entered, release, second_read, second_started = Event(), Event(), Event(), Event()

    class HeldRead(Calculator):
        implemented_properties = ("energy",)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if atoms.positions[0, 0] == 0:
                entered.set()
                assert release.wait(5)
            else:
                second_read.set()
            self.results = {"energy": float(self.atoms.positions[0, 0])}

    method = CalculatorMethod("held serial surface", calculator=HeldRead())
    first, second = Atoms("H", positions=[[0, 0, 0]]), Atoms("H", positions=[[1, 0, 0]])
    first.calc = make_calculator(method=method, fresh=False)
    second.calc = make_calculator(method=method, fresh=False)

    def read_second():
        second_started.set()
        return second.get_potential_energy()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first.get_potential_energy)
        try:
            assert entered.wait(5)
            second_future = pool.submit(read_second)
            assert second_started.wait(5)
            assert not second_read.wait(0.05)
        finally:
            release.set()
        assert first_future.result() == 0
        assert second_future.result() == 1
    assert second_read.is_set()


def test_serial_forces_snapshot_provider_buffer():
    from ase.calculators.calculator import Calculator, all_changes

    from rxts import CalculatorMethod

    class BufferSurface(Calculator):
        implemented_properties = ("energy", "forces")

        def __init__(self):
            super().__init__()
            self.buffer = np.zeros((1, 3))

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.buffer[:] = -2 * self.atoms.positions
            self.results = {"energy": float(np.sum(self.atoms.positions**2)), "forces": self.buffer}

    provider = BufferSurface()
    method = CalculatorMethod("buffered serial surface", calculator=provider)
    first = Atoms("H", positions=[[1, 0, 0]], calculator=make_calculator(method=method, fresh=False))
    second = Atoms("H", positions=[[2, 0, 0]], calculator=make_calculator(method=method, fresh=False))
    assert first.get_potential_energy() == 1.0
    assert second.get_potential_energy() == 4.0
    np.testing.assert_array_equal(first.get_forces(), [[-2, 0, 0]])
    assert not np.shares_memory(first.calc.results["forces"], provider.buffer)


def test_copy_atoms_preserves_state_and_identity():
    provider = GXTB(method="gfn2")
    atoms = Atoms("H", positions=[[1, 2, 3]], calculator=provider)
    atoms.set_masses([2.0])
    atoms.set_initial_charges([0.25])
    atoms.set_initial_magnetic_moments([1.0])
    atoms.new_array("probe", np.array([4.0]))
    atoms.info["nested"] = {"values": [np.array([5.0, 6.0])]}

    copied = copy_atoms(atoms)

    assert copied.calc is provider
    np.testing.assert_array_equal(copied.arrays["masses"], [2.0])
    np.testing.assert_array_equal(copied.arrays["initial_charges"], [0.25])
    np.testing.assert_array_equal(copied.arrays["initial_magmoms"], [1.0])
    np.testing.assert_array_equal(copied.arrays["probe"], [4.0])
    np.testing.assert_array_equal(copied.info["nested"]["values"][0], [5.0, 6.0])
    copied.arrays["probe"][0] = 9.0
    copied.info["nested"]["values"][0][0] = 9.0
    assert atoms.arrays["probe"][0] == 4.0
    assert atoms.info["nested"]["values"][0][0] == 5.0

    opaque = object()
    atoms.info["opaque"] = opaque
    copied = copy_atoms(atoms)
    assert copied.info["opaque"] is not opaque


def test_external_metadata_cache():
    from ase.calculators.calculator import Calculator, all_changes

    class StateSurface(Calculator):
        implemented_properties = ("energy",)

        def __init__(self):
            super().__init__()
            self.reads = 0

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.reads += 1
            self.results["energy"] = atoms.get_masses()[0] + atoms.arrays["probe"][0] + atoms.info["nested"]["offset"]

    provider = StateSurface()
    atoms = Atoms("H", positions=[[0, 0, 0]], calculator=observed_calculator(provider))
    atoms.set_masses([2.0])
    atoms.new_array("probe", np.array([3.0]))
    atoms.info["nested"] = {"offset": 5.0}
    assert atoms.get_potential_energy() == 10.0

    atoms.set_masses([4.0])
    assert atoms.get_potential_energy() == 12.0
    atoms.arrays["probe"][0] = 7.0
    assert atoms.get_potential_energy() == 16.0
    atoms.info["nested"]["offset"] = 8.0
    assert atoms.get_potential_energy() == 19.0
    assert provider.reads == 4
    atoms.positions[0, 0] += 1e-16
    assert atoms.get_potential_energy() == 19.0
    assert provider.reads == 4
    assert atoms.calc.n_evaluations == 4


def test_external_wrapper_keeps_constrained_cache():
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.constraints import FixAtoms

    atoms = Atoms("H", positions=[[0, 0, 0]], constraint=FixAtoms([0]))
    atoms.calc = SinglePointCalculator(atoms, energy=3.5, forces=np.zeros((1, 3)))
    wrapper = observed_calculator(atoms.calc)
    atoms.calc = wrapper

    assert atoms.get_potential_energy() == 3.5
    assert wrapper.n_evaluations == 1
    assert atoms.get_potential_energy() == 3.5
    assert wrapper.n_evaluations == 1


def test_native_counters():
    with pytest.raises(TypeError, match="ASE Calculators"):
        observed_calculator(type("CounterObject", (), {"n_evaluations": 0, "n_gradients": 0})())

    calculator = GXTB(method="gfn2")
    assert observed_calculator(calculator) is calculator


def test_external_boundary_rejects_bad_properties():
    from ase.calculators.calculator import Calculator, all_changes

    class InvalidSurface(Calculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": np.array([1.0]), "forces": np.zeros(3)}

    for property_name in ("energy", "forces"):
        atoms = Atoms("H", calculator=observed_calculator(InvalidSurface()))
        with pytest.raises(ValueError, match=f"malformed {property_name}"):
            atoms.calc.get_property(property_name, atoms)
    with pytest.raises(ValueError, match="Unknown method"):
        make_calculator(method="")
