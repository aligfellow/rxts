"MMFF term reconstruction, reactive-bond restraints, and torsions."

from __future__ import annotations

import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers, rdMolTransforms

from rxts import evb
from rxts.evb import (
    CONFORMER_K,
    THRESHOLD,
    GFNFFPotential,
    MMFFPotential,
    _Endpoint,
    _metal_angle_energy_grad,
    _mmff_bond,
    _mmff_ele,
    _mmff_vdw,
    _morse,
    _signed_dihedral,
    _soft_core,
    _wrap,
    active_torsion,
    candidate_torsions,
    dihedral,
    driving_restraint,
    driving_torsion,
    metal_angle_references,
    restraint_targets,
    torsion_gradient,
)

#: kJ per kcal -- the conversion every VeloxChem energy constant crosses on its way into rxts, which
#: works in RDKit's MMFF units (kcal/mol, angstrom).
KJ_PER_KCAL = 4.184

TERMS = ("Bond", "Angle", "StretchBend", "Oop", "Torsion", "VdW", "Ele")


def _only(mol, term):
    """An MMFF force field with a single energy term enabled."""
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    for name in TERMS:
        getattr(props, f"SetMMFF{name}Term")(name == term)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
        mol, props, nonBondedThresh=1e5, ignoreInterfragInteractions=False
    )
    return ff, props


@pytest.fixture
def mol():
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OCC"))
    rdDistGeom.EmbedMolecule(m, randomSeed=7)
    return m


def test_bond_term_matches_rdkit(mol):
    ff, props = _only(mol, "Bond")
    pos = mol.GetConformer().GetPositions()
    mine = 0.0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        _, kb, r0 = props.GetMMFFBondStretchParams(mol, i, j)
        mine += _mmff_bond(kb, r0, np.linalg.norm(pos[i] - pos[j]))[0]
    assert mine == pytest.approx(ff.CalcEnergy(), abs=1e-9)


def test_vdw_term_matches_rdkit(mol):
    ff, props = _only(mol, "VdW")
    pos = mol.GetConformer().GetPositions()
    topo = Chem.GetDistanceMatrix(mol)
    mine = 0.0
    for i, j in itertools.combinations(range(mol.GetNumAtoms()), 2):
        if topo[i, j] < 3:  # MMFF excludes 1-2 and 1-3 pairs
            continue
        rstar, eps = props.GetMMFFVdWParams(i, j)[2:4]
        mine += _mmff_vdw(rstar, eps, np.linalg.norm(pos[i] - pos[j]))[0]
    assert mine == pytest.approx(ff.CalcEnergy(), abs=1e-9)


def test_electrostatic_term_matches_rdkit(mol):
    ff, props = _only(mol, "Ele")
    pos = mol.GetConformer().GetPositions()
    topo = Chem.GetDistanceMatrix(mol)
    mine = 0.0
    for i, j in itertools.combinations(range(mol.GetNumAtoms()), 2):
        if topo[i, j] < 3:
            continue
        scale = 0.75 if topo[i, j] == 3 else 1.0  # MMFF scales 1-4 electrostatics
        qq = props.GetMMFFPartialCharge(i) * props.GetMMFFPartialCharge(j)
        mine += _mmff_ele(qq, np.linalg.norm(pos[i] - pos[j]), scale)[0]
    assert mine == pytest.approx(ff.CalcEnergy(), abs=1e-9)


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (_mmff_bond, (4.19, 1.492)),
        (_mmff_vdw, (3.94, 0.068)),
        (_morse, (119.5, 600.0, 1.35)),
    ],
)
def test_each_term_hands_back_its_own_derivative(fn, args):
    h = 1e-6
    for r in (0.9, 1.4, 2.0, 3.5):
        _, analytic = fn(*args, r)
        numeric = (fn(*args, r + h)[0] - fn(*args, r - h)[0]) / (2 * h)
        assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-6)


def test_soft_core_is_bounded_and_smooth():
    """MMFF's wall between a not-yet-bonded pair is 316 kcal/mol at 1.4 A -- 1300 kJ/mol, dwarfing any barrier."""
    rstar, eps = 3.94, 0.068
    rcut = 0.9 * rstar
    pair = (rstar, eps)

    assert _mmff_vdw(*pair, 1.4)[0] > 100.0
    assert abs(_soft_core(_mmff_vdw, pair, 1.4, rcut)[0]) < 10.0, "two atoms about to bond must be able to approach"

    for r in (rcut + 0.01, 5.0, 8.0):
        assert _soft_core(_mmff_vdw, pair, r, rcut)[0] == pytest.approx(_mmff_vdw(*pair, r)[0]), "untouched beyond rcut"

    lo = _soft_core(_mmff_vdw, pair, rcut - 1e-7, rcut)
    hi = _soft_core(_mmff_vdw, pair, rcut + 1e-7, rcut)
    assert lo[0] == pytest.approx(hi[0], abs=1e-5), "joins on continuously in value"
    assert lo[1] == pytest.approx(hi[1], abs=1e-4), "...and in slope"


@pytest.fixture
def sn2():
    """Cl- + CH3Br, and its product, in ONE atom order: C Br Cl H H H."""
    rea = Chem.AddHs(Chem.MolFromSmiles("CBr.[Cl-]"))
    rdDistGeom.EmbedMolecule(rea, randomSeed=11)
    edit = Chem.RWMol(rea)
    edit.RemoveBond(0, 1)
    edit.AddBond(0, 2, Chem.BondType.SINGLE)
    edit.GetAtomWithIdx(2).SetFormalCharge(0)
    edit.GetAtomWithIdx(1).SetFormalCharge(-1)
    pro = edit.GetMol()
    Chem.SanitizeMol(pro)
    return MMFFPotential(rea, pro, forming=[(0, 2)], breaking=[(0, 1)])


def test_energy_only_matches_gradient_energy(sn2):
    x = sn2.rea.mol.GetConformer().GetPositions()
    for side in ("rea", "pro"):
        for patch in ("pes", "integration"):
            cheap, g = sn2._endpoint(x, side, patch, gradient=False)
            assert g is None
            assert cheap == sn2._endpoint(x, side, patch)[0]


def test_scoring_skips_gradient_sampling_uses_it(sn2, monkeypatch):
    calls = []
    real = _Endpoint.energy_grad
    monkeypatch.setattr(_Endpoint, "energy_grad", lambda self, x: (calls.append(1), real(self, x))[1])

    x = sn2.rea.mol.GetConformer().GetPositions()
    sn2.energy(x, 0.5)
    assert len(calls) == 0

    sn2.sampling_energy(x, 0.5)
    assert len(calls) == 2


class _Analytic:
    """A GFN-FF endpoint stand-in: an energy and its exact gradient, no xtb."""

    def __init__(self, k):
        self.k = k

    def energy_grad(self, x):
        return self.k * float(np.sum(x**2)), 2.0 * self.k * x


def _fake_gfnff():
    """A `GFNFFPotential` with both xtb calculators replaced, built without running `__init__`."""
    pot = GFNFFPotential.__new__(GFNFFPotential)
    pot.rea, pot.pro = _Analytic(1.0), _Analytic(3.0)
    pot._pool = None
    pot.torsion = None
    pot.changing = [(0, 1)]
    pot._bond = {(0, 1): ("rea", 500.0, 1.5)}
    pot._contact = {(0, 1): 3.6}
    return pot


def test_gfnff_interpolates_and_reports_endpoints():
    pot = _fake_gfnff()
    x = np.array([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]])
    lam = 0.25

    e1, e2 = pot.rea.energy_grad(x)[0], pot.pro.energy_grad(x)[0]
    e, got1, got2 = pot.energy(x, lam)
    assert (got1, got2) == (e1, e2)
    assert e == pytest.approx((1.0 - lam) * e1 + lam * e2)


def test_gfnff_overflow_rejected():
    from rxts.gfnff import UNUSABLE

    pot = _fake_gfnff()
    pot.rea.energy_grad = lambda x: (UNUSABLE, np.zeros_like(x))
    with pytest.raises(RuntimeError, match="could not evaluate a frozen topology"):
        pot.sampling_energy(np.array([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]]), 0.5)


def test_gfnff_sampling_still_reads_its_own_gradient():
    """`energy` stopped returning a gradient; dropping it from `sampling_energy` too leaves the suite green."""
    pot = _fake_gfnff()
    x = np.array([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]])
    lam = 0.25

    g1 = pot.rea.energy_grad(x)[1]
    g2 = pot.pro.energy_grad(x)[1]
    _, g_r = driving_restraint(x=x, lam=lam, bond=pot._bond, contact=pot._contact)
    expected = (1.0 - lam) * g1 + lam * g2 + g_r

    assert np.allclose(pot.sampling_energy(x, lam)[1], expected, atol=0, rtol=0)


def test_haptic_centroid_gradient():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.2, 1.4, 0.3]])
    refs = (((0, (1, 2), (3,), 1.1),), ((0, (1, 2), (3,), 1.8),))
    k, lam, h = 239.0, 0.3, 1e-6
    energy, gradient = _metal_angle_energy_grad(x, lam, k, refs)
    numeric = np.empty_like(x)
    for i in range(len(x)):
        for axis in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, axis] += h
            xm[i, axis] -= h
            ep = _metal_angle_energy_grad(xp, lam, k, refs)[0]
            em = _metal_angle_energy_grad(xm, lam, k, refs)[0]
            numeric[i, axis] = (ep - em) / (2 * h)
    assert energy > 0.0
    np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gradient[1], gradient[2], atol=1e-12)


def test_trans_angle_limit_and_singular_errors():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    trans = (((0, (1,), (2,), np.pi),), ())
    energy, gradient = _metal_angle_energy_grad(x, 0.0, 239.0, trans)
    assert energy == 0.0
    np.testing.assert_array_equal(gradient, np.zeros_like(x))

    delta = 1e-5
    near = x.copy()
    near[2] = [-np.cos(delta), np.sin(delta), 0.0]
    energy, gradient = _metal_angle_energy_grad(near, 0.0, 239.0, trans)
    assert np.isfinite(energy)
    assert np.isfinite(gradient).all()
    h = 1e-7
    plus, minus = near.copy(), near.copy()
    plus[2, 1] += h
    minus[2, 1] -= h
    numeric = (
        _metal_angle_energy_grad(plus, 0.0, 239.0, trans)[0] - _metal_angle_energy_grad(minus, 0.0, 239.0, trans)[0]
    ) / (2 * h)
    assert gradient[2, 1] == pytest.approx(numeric, rel=1e-5)
    with pytest.raises(ValueError, match="nonmatching collinear metal angle"):
        _metal_angle_energy_grad(x, 0.0, 239.0, (((0, (1,), (2,), 2.0),), ()))
    zero = x.copy()
    zero[1] = zero[0]
    with pytest.raises(ValueError, match="zero-length site leg"):
        _metal_angle_energy_grad(zero, 0.0, 239.0, trans)
    nonfinite = x.copy()
    nonfinite[1, 0] = np.nan
    with pytest.raises(ValueError, match="metal angle coordinates must be finite"):
        _metal_angle_energy_grad(nonfinite, 0.0, 239.0, trans)


def test_metal_guidance_changes_sampling_only():
    from rxts.ff import TransitionStateGuesser

    pot = _fake_gfnff()
    guesser = TransitionStateGuesser()
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
    refs = (((0, (1,), (2,), 1.0),), ())
    bare = pot.energy(x, 0.25)
    unguided = guesser._energy_grad(pot, x, 0.25)
    pot.metal_angle_k, pot.metal_angle_references = 239.0, refs
    guided = guesser._energy_grad(pot, x, 0.25)
    assert pot.energy(x, 0.25) == bare
    assert guided[0] > unguided[0]
    assert not np.array_equal(guided[1], unguided[1])
    pot.metal_angle_k = 0.0
    disabled = guesser._energy_grad(pot, x, 0.25)
    assert disabled[0] == pytest.approx(unguided[0])
    np.testing.assert_array_equal(disabled[1], unguided[1])


def test_native_haptic_angle_mapping():
    class Haptic:
        atoms = (1, 2)

    state = type("Centre", (), {"atom": 0, "vertices": (Haptic(), 3, None)})()
    isomer = type("Isomer", (), {"centres": (state,)})()
    mol = Chem.RWMol()
    for number in (46, 6, 6, 6, 6, 6):
        mol.AddAtom(Chem.Atom(number))
    for donor in (1, 2, 3):
        mol.AddBond(0, donor, Chem.BondType.DATIVE)
    conf = Chem.Conformer(6)
    coords = ((0, 0, 0), (1, 0, 0), (1, 0.2, 0), (0, 1.5, 0), (3, 3, 3), (4, 4, 4))
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    mol = mol.GetMol()
    mapping = [5, 2, 4, 1, 0, 3]
    refs, source = metal_angle_references(mol, mol, (None, isomer), mapping)
    assert source == "endpoint_graph+native_state"
    assert len(refs[1]) == 1
    center, haptic, donor, angle = refs[1][0]
    assert (center, haptic, donor) == (mapping[0], (mapping[1], mapping[2]), (mapping[3],))
    p = np.asarray(coords)
    a = p[[mapping[1], mapping[2]]].mean(axis=0) - p[mapping[0]]
    b = p[mapping[3]] - p[mapping[0]]
    assert angle == pytest.approx(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))
    graph_refs, graph_source = metal_angle_references(mol, mol)
    assert graph_source == "endpoint_graph"
    assert len(graph_refs[0]) == len(graph_refs[1]) == 3
    bad = Chem.Mol(mol)
    bad.GetConformer().SetAtomPosition(1, (float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError, match="metal angle reference coordinates must be finite"):
        metal_angle_references(bad, mol)


def test_morse_energy_conversion():
    """Check the stored kcal value against the 500 kJ/mol input conversion."""
    kcal, kj = evb._MORSE_D, 500.0
    exact = kj / KJ_PER_KCAL
    assert kcal == pytest.approx(exact, abs=0.05)
    assert kcal == round(exact, 1)


# Check the restraint schedule against an independent transcription of its defining algebra.


def _veloxchem_dynamic_bond_target(lam, eq_a, eq_b, p):
    """The endpoint interpolation followed by the progress exponent::

        p = self.dynamic_bond_tightening
        eq = eqA * (1 - lam) + eqB * lam
        min_eq = min(eqA, eqB)
        dif_eq = abs(eqA - eqB)
        if dif_eq > 0:
            eq = (((eq - min_eq) / dif_eq) ** p * dif_eq) + min_eq

    `eqA` is the length at lambda = 0 and `eqB` the length at lambda = 1, so a breaking bond has
    eqA = bonded, eqB = free and a forming bond the reverse.
    """
    eq = eq_a * (1 - lam) + eq_b * lam
    min_eq = min(eq_a, eq_b)
    dif_eq = abs(eq_a - eq_b)
    if dif_eq > 0:
        eq = (((eq - min_eq) / dif_eq) ** p * dif_eq) + min_eq
    return eq


def test_restraint_schedule_matches_veloxchem():
    # A breaking C-Cl, a forming C-I and one O-H of a proton relay, with rxts contact distances.
    bond = {(0, 1): ("rea", 300.0, 1.79), (2, 3): ("pro", 250.0, 2.14), (4, 5): ("rea", 500.0, 0.97)}
    contact = {(0, 1): 3.55, (2, 3): 3.96, (4, 5): 2.72}

    worst = 0.0
    for lam in np.linspace(0.0, 1.0, 21):
        targets = restraint_targets(lam=float(lam), bond=bond, contact=contact)
        for pair, (side, _k, r_bonded) in bond.items():
            r_free = contact[pair]
            eq_a, eq_b = (r_bonded, r_free) if side == "rea" else (r_free, r_bonded)
            reference = _veloxchem_dynamic_bond_target(float(lam), eq_a, eq_b, evb._TIGHTENING)
            worst = max(worst, abs(targets[pair] - reference))

    assert worst < 1e-14, f"maximum schedule difference from VeloxChem: {worst:.3e} A"
    # Endpoints: the schedule must be exact there, not merely close.
    at_zero = restraint_targets(lam=0.0, bond=bond, contact=contact)
    at_one = restraint_targets(lam=1.0, bond=bond, contact=contact)
    assert at_zero[(0, 1)] == 1.79, "a breaking bond is at its bonded length at lambda = 0"
    assert at_one[(0, 1)] == 3.55, "a breaking bond is at contact at lambda = 1"
    assert at_zero[(2, 3)] == 3.96, "a forming bond is at contact at lambda = 0"
    assert at_one[(2, 3)] == 2.14, "a forming bond is at its bonded length at lambda = 1"


def _recovered_fc_scale(lam, k_bonded=300.0, delta=0.01):
    """`fc_scale` at one lambda, read back out of `driving_restraint`'s energy.

    It is written inline in `driving_restraint`, so the only way a test can see it is to displace one
    held bond by a known delta from its own target, recover `k_eff = 2E/delta^2`, and divide by the
    linear interpolation `(1 - lam) k_a + lam k_b`.
    """
    bond = {(0, 1): ("rea", k_bonded, 1.79)}
    contact = {(0, 1): 3.55}
    r0 = restraint_targets(lam=lam, bond=bond, contact=contact)[(0, 1)]
    x = np.array([[0.0, 0.0, 0.0], [r0 + delta, 0.0, 0.0]])
    energy, _gradient = driving_restraint(x, lam, bond, contact)
    k_a, k_b = k_bonded, evb._GHOST_FC * k_bonded  # "rea": bonded at lam=0, ghost at lam=1
    return (2.0 * energy / delta**2) / ((1.0 - lam) * k_a + lam * k_b)


@pytest.mark.parametrize(
    ("lam", "expected"),
    [
        (0.0, 1.0),
        (0.125, 0.9125),
        (0.25, 0.85),
        (0.375, 0.8125),
        (0.5, 0.8),
        (0.625, 0.8125),
        (0.75, 0.85),
        (0.875, 0.9125),
        (1.0, 1.0),
    ],
)
def test_fc_scale_matches_veloxchem_curve(lam, expected):
    gamma = evb._FC_DIP
    fc_scale = _recovered_fc_scale(lam)

    quadratic = 4.0 * (lam - 0.5) ** 2 * (1.0 - gamma) + gamma
    assert fc_scale == pytest.approx(expected, abs=1e-9)
    assert fc_scale == pytest.approx(quadratic, abs=1e-9)
    if lam not in (0.0, 0.5, 1.0):
        triangular = gamma + (1.0 - gamma) * abs(2.0 * lam - 1.0)
        assert not np.isclose(quadratic, triangular)


def test_morse_curvature_matches_harmonic():
    "Check the Morse potential has the MMFF bond curvature at its minimum."
    d_e, k, r0 = evb._MORSE_D, 600.0, 1.35
    h = 1e-4
    e_plus = _morse(d_e, k, r0, r0 + h)[0]
    e_minus = _morse(d_e, k, r0, r0 - h)[0]
    e_zero = _morse(d_e, k, r0, r0)[0]

    assert e_zero == pytest.approx(0.0, abs=1e-12), "the Morse minimum sits at r0 with zero energy"
    curvature = (e_plus - 2 * e_zero + e_minus) / h**2
    assert curvature == pytest.approx(k, rel=1e-6), (
        f"\n\nd2E/dr2 at the Morse minimum is {curvature!r}, not the harmonic k = {k}."
        "\n`a = sqrt(k / (2 D))` is what ties them together (evb.py `_morse`, VeloxChem"
        "\n`_add_morse_bond`). Lose it and D stops being a dissociation energy."
    )
    assert _morse(d_e, k, r0, 1e6)[0] == pytest.approx(d_e), "E -> D at infinite separation"


def test_gfnff_free_length_uses_vdw_radii(monkeypatch):
    """C-H: RDKit's 2.900 A radius sum, versus VeloxChem's measured 3.025 A sigma mean."""
    from rxts import gfnff

    monkeypatch.setattr(gfnff, "GFNFF", lambda mol, **kwargs: _Analytic(1.0))
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    pair = (0, 1)
    potential = GFNFFPotential(mol, mol, forming=[], breaking=[pair], n_cores=1)

    assert potential._bond[pair] == ("rea", 500.0, pytest.approx(1.07))
    assert potential._contact[pair] == pytest.approx(2.9)
    assert restraint_targets(0.0, potential._bond, potential._contact)[pair] == pytest.approx(1.07)
    assert restraint_targets(1.0, potential._bond, potential._contact)[pair] == pytest.approx(2.9)


# A transition state where no bond changes: the driven torsion.

LINE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
T = (0, 1, 2, 3)


def _well_conditioned(rng, n):
    """A geometry whose torsion is not near-collinear -- where the finite difference is unreliable too."""
    while True:
        x = rng.standard_normal((n, 3)) * 1.4
        t = tuple(int(v) for v in rng.choice(n, 4, replace=False))
        if np.abs(torsion_gradient(x, *t)).max() < 50.0:
            return x, t


def _finite_difference(f, x, h=1e-6):
    out = np.zeros_like(x)
    for a in range(x.shape[0]):
        for c in range(3):
            plus, minus = x.copy(), x.copy()
            plus[a, c] += h
            minus[a, c] -= h
            out[a, c] = (f(plus) - f(minus)) / (2.0 * h)
    return out


def test_wrap_takes_the_short_way_round():
    assert _wrap(-170.0 - 170.0) == pytest.approx(20.0)
    assert _wrap(170.0 - -170.0) == pytest.approx(-20.0)
    assert _wrap(10.0) == pytest.approx(10.0)


def test_torsion_gradient_matches_finite_differences():
    """A wrong sign here drives the torsion the wrong way round the circle and still converges."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, t = _well_conditioned(rng, int(rng.integers(4, 9)))
        numeric = _finite_difference(lambda y, t=t: dihedral(y, *t), x)
        assert np.abs(torsion_gradient(x, *t) - numeric).max() < 1e-6


def test_torsion_value_matches_rdkit_and_rigid_motion():
    rng = np.random.default_rng(17)
    worst_rdkit = 0.0
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    shift = np.array([3.1, -2.2, 0.7])
    for _ in range(200):
        x = rng.normal(size=(4, 3)) * 1.5
        mol = Chem.RWMol()
        for _ in range(4):
            mol.AddAtom(Chem.Atom(6))
        for a, b in zip(range(3), range(1, 4), strict=True):
            mol.AddBond(a, b, Chem.BondType.SINGLE)
        conformer = Chem.Conformer(4)
        for index, position in enumerate(x):
            conformer.SetAtomPosition(index, position.tolist())
        mol.AddConformer(conformer)
        value = dihedral(x, *T)
        reference = np.radians(rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *T))
        difference = (value - reference + np.pi) % (2 * np.pi) - np.pi
        worst_rdkit = max(worst_rdkit, abs(difference))
        moved = dihedral(x @ rotation + shift, *T)
        moved_difference = (moved - value + np.pi) % (2 * np.pi) - np.pi
        assert abs(moved_difference) < 1e-12

    assert worst_rdkit < 1e-12


def test_raw_dihedral_uses_rdkit_indices_and_errors():
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    torsion = (4, 1, 3, 0)
    conformer = Chem.Conformer(len(x))
    for index, point in enumerate(x):
        conformer.SetAtomPosition(index, point.tolist())
    expected = rdMolTransforms.GetDihedralRad(conformer, *torsion)
    assert _signed_dihedral(x, *torsion) == pytest.approx(expected)
    assert _signed_dihedral(LINE, *T) == 0.0

    coincident = x.copy()
    coincident[3] = coincident[1]
    with pytest.raises(ValueError, match="identical 3D coordinates"):
        _signed_dihedral(coincident, *torsion)


def test_torsion_force_and_torque_balance():
    """The half the finite difference cannot see: it would pass if `dihedral` were wrong the same way."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        x, t = _well_conditioned(rng, 6)
        g = torsion_gradient(x, *t)
        assert np.abs(g.sum(axis=0)).max() < 1e-10
        assert np.abs(np.cross(x, g).sum(axis=0)).max() < 1e-9


def test_driving_torsion_gradient():
    rng = np.random.default_rng(2)
    for _ in range(30):
        x, t = _well_conditioned(rng, int(rng.integers(4, 9)))
        lam = float(rng.random())
        phi_r, phi_p = float(rng.uniform(-180, 180)), float(rng.uniform(-180, 180))
        _, g = driving_torsion(x, lam, t, phi_r, phi_p)
        numeric = _finite_difference(lambda y, lam=lam, t=t, r=phi_r, p=phi_p: driving_torsion(y, lam, t, r, p)[0], x)
        assert np.abs(g - numeric).max() < 1e-5


def test_torsion_minimum_slides_between_endpoints():
    rng = np.random.default_rng(3)
    x, t = _well_conditioned(rng, 6)
    phi = np.degrees(dihedral(x, *t))
    e0, g0 = driving_torsion(x, 0.0, t, phi, phi + 90.0)
    assert e0 == pytest.approx(0.0, abs=1e-9), "at lambda 0 the restraint sits on the reactant's own angle"
    assert np.abs(g0).max() < 1e-9
    e1, g1 = driving_torsion(x, 1.0, t, phi, phi + 90.0)
    assert e1 > 1.0, "at lambda 1 the target is 90 degrees away and the restraint is pushing"
    assert np.abs(g1).max() > 1e-3


def test_collinear_torsion():
    """Four atoms on a line have no dihedral plane, so the published formula divides by both normals."""
    assert np.array_equal(torsion_gradient(LINE, *T), np.zeros_like(LINE))

    energy, gradient = driving_torsion(LINE, 0.5, T, 0.0, 90.0)
    assert np.isfinite(energy), "the restraint's energy must survive a geometry its derivative cannot"
    assert np.array_equal(gradient, np.zeros_like(LINE))


@pytest.mark.parametrize("side", ["ijk", "jkm"])
def test_near_collinear_gradient_uses_floor(side):
    """The utility and gradient floors agree; active/driving values retain the prior raw angle.

    The latter can leave a finite restraint energy with a zero gradient near collinearity. This known
    degeneracy is preserved for output compatibility; this dependency change does not retune it.
    """
    disagree = []
    for eps in np.logspace(-3, -12, 37):
        if side == "ijk":  # i-j-k collinear to within eps, m held out of the plane
            x = np.array([[-1.0, float(eps), 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        else:  # j-k-m collinear instead, i out of the plane
            x = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, float(eps), 0.0]])
        gradient = torsion_gradient(x, *T)
        assert np.all(np.isfinite(gradient)), f"eps={eps:.3e} divided by a normal it had just called zero"
        if (dihedral(x, *T) == 0.0) != bool(np.array_equal(gradient, np.zeros_like(x))):
            disagree.append(f"{eps:.3e}")
    assert not disagree, (
        f"{len(disagree)} of 37 geometries measure an angle whose gradient is zeroed (or the reverse), at "
        f"eps = {disagree}. `dihedral` and `torsion_gradient` compare different quantities, or different "
        "constants, to decide the same question."
    )

    eps = 1e-9
    if side == "ijk":
        x = np.array([[-1.0, eps, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    else:
        x = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, eps, 0.0]])
    raw = _signed_dihedral(x, *T)
    assert raw != 0.0
    assert dihedral(x, *T) == 0.0
    energy, driven_gradient = driving_torsion(x, 0.5, T, 0.0, 0.0)
    assert energy == pytest.approx(CONFORMER_K * (1.0 - np.cos(raw)))
    assert np.array_equal(driven_gradient, np.zeros_like(x))


def test_candidate_torsions_are_one_per_central_bond():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    centres = [tuple(sorted(t[1:3])) for t in candidate_torsions(mol)]
    assert len(centres) == len(set(centres))


def _twisted(smiles, torsion, phi, seed=3):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(), *torsion, phi)
    return mol


def test_active_torsion_finds_the_rotated_bond():
    torsion = (4, 3, 1, 0)  # O=C-N-C in DMF
    found, phi_r, phi_p = active_torsion(_twisted("CN(C)C=O", torsion, 0.0), _twisted("CN(C)C=O", torsion, 180.0))
    assert tuple(sorted(found[1:3])) == tuple(sorted(torsion[1:3]))
    assert abs(_wrap(phi_p - phi_r)) > THRESHOLD


def test_two_identical_structures_are_not_a_reaction():
    """Refusing is the right answer. Driving a torsion here would invent a reaction."""
    mol = _twisted("CN(C)C=O", (4, 3, 1, 0), 0.0)
    with pytest.raises(ValueError, match="do not differ by a reaction"):
        active_torsion(mol, Chem.Mol(mol))


def test_torsion_energy_conversion_is_unrounded():
    """Check the stored kcal value against the 200 kJ/mol input conversion."""
    kcal, kj = evb.CONFORMER_K, 200.0
    exact = kj / KJ_PER_KCAL
    assert kcal == pytest.approx(exact, abs=0.05)
    assert kcal == round(exact, 1)
