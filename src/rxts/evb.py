"""Interpolate reactant and product force fields to generate a transition-state guess.

Reactive MMFF pairs replace their endpoint terms with Morse bonds or soft-core nonbonded terms. Energies
and gradients use kcal/mol and angstrom.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from numbers import Real

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

from rxts.utils import FROM_COORDINATES, metal_atoms

_MMFF_BOND_K = 143.9325  # kb [mdyn/A] * this -> kcal/mol/A^2
_MMFF_BOND_CS = -2.0  # cubic stretch constant
_COULOMB = 332.0716  # kcal*A/mol/e^2
_MMFF_ELE_BUFFER = 0.05  # A, MMFF's electrostatic buffering distance
_MMFF_14_SCALE = 0.75  # MMFF scales 1-4 electrostatics
_ONE_FOUR = 3  # topological distance of a 1-4 pair; 1-2 and 1-3 are excluded

# l-interpolation exponent for reactive bonds, from VeloxChem's EvbSystemBuilder; see driving_restraint.
_TIGHTENING = 3.25
_GHOST_FC = 0.2  # force constant a reactive bond keeps where it does not exist
_FC_DIP = 0.8  # reactive force constants soften to this at l = 0.5
_MORSE_D = 119.5  # dissociation energy of a reactive bond, kcal/mol (= 500 kJ/mol, VeloxChem's default)
_SOFT_CORE_FRACTION = 0.9  # reactive-pair soft-core cutoff, as a fraction of the vdW minimum
_METAL_ANGLE_K = 1000.0 / 4.184  # VeloxChem's 1000 kJ/mol/rad², in the scan's kcal/mol units
_ANGLE_SINGULAR = 1e-8


def mmff_can_type(mol):
    """Check MMFF coverage, conservatively refusing graphs inferred from XYZ."""
    if mol.HasProp(FROM_COORDINATES):
        return False
    return bool(rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol))


def untypeable_fragments(mol):
    """Find the fragments of `mol` that MMFF94 cannot handle, as SMILES."""
    bad = []
    for indices in Chem.GetMolFrags(mol):
        frag = Chem.MolFragmentToSmiles(mol, atomsToUse=list(indices))
        piece = Chem.MolFromSmiles(frag)
        if piece is None or not mmff_can_type(Chem.AddHs(piece)):
            bad.append(frag)
    return bad


def check_mmff(mol, what):
    """Reject an explicitly requested MMFF field without parameter coverage."""
    if mmff_can_type(mol):
        return
    bad = untypeable_fragments(mol)
    detail = f" The trouble is with: {', '.join(bad)}." if bad else ""
    raise ValueError(f"MMFF94 cannot assign parameters to the {what}.{detail} Use engine='gfnff' with rxts[ff].")


# Match RDKit's pair terms so the reactive contributions can be replaced.
def _mmff_bond(kb, r0, r):
    """MMFF94 quartic bond stretch, and its derivative."""
    dr = r - r0
    c = _MMFF_BOND_K * kb / 2.0
    cs = _MMFF_BOND_CS
    e = c * dr**2 * (1.0 + cs * dr + (7.0 / 12.0) * cs**2 * dr**2)
    de = c * (2.0 * dr + 3.0 * cs * dr**2 + (7.0 / 3.0) * cs**2 * dr**3)
    return e, de


def _mmff_vdw(rstar, eps, r):
    """MMFF94 buffered 14-7 van der Waals, and its derivative."""
    a = 1.07 * rstar / (r + 0.07 * rstar)
    da = -a / (r + 0.07 * rstar)
    r7, rs7 = r**7, rstar**7
    b = 1.12 * rs7 / (r7 + 0.12 * rs7)
    db = -7.0 * r**6 * b / (r7 + 0.12 * rs7)
    e = eps * a**7 * (b - 2.0)
    de = eps * (7.0 * a**6 * da * (b - 2.0) + a**7 * db)
    return e, de


def _mmff_ele(qq, r, scale):
    """MMFF94 buffered electrostatics, and its derivative."""
    d = r + _MMFF_ELE_BUFFER
    e = scale * _COULOMB * qq / d
    de = -scale * _COULOMB * qq / d**2
    return e, de


def _morse(d_e, k, r0, r):
    """Morse potential matching a harmonic of curvature `k` at the minimum."""
    a = np.sqrt(k / (2.0 * d_e))
    x = np.exp(-a * (r - r0))
    e = d_e * (1.0 - x) ** 2
    de = 2.0 * d_e * a * (1.0 - x) * x
    return e, de


def _soft_core(fn, head, r, rcut, tail=()):
    """Cap a diverging pair potential below `rcut` with a smooth quadratic; untouched beyond `rcut`.

    The quadratic matches value and slope at `rcut` and flattens towards r = 0. `fn` is called as
    ``fn(*head, r, *tail)``, so the caller need not close over the loop's parameters.
    """

    def f(rr):
        return fn(*head, rr, *tail)

    if r >= rcut:
        return f(r)
    e_c, de_c = f(rcut)
    a = de_c / (2.0 * rcut)
    c = e_c - a * rcut**2
    return a * r**2 + c, 2.0 * a * r


def restraint_targets(lam, bond, contact):
    """Return the lambda-dependent target lengths, in angstrom."""
    targets = {}
    for pair, (side, _, r_bonded) in bond.items():
        r_free = contact[pair]
        # breaking: bonded at l=0, apart at l=1. Forming: the reverse.
        progress = lam if side == "rea" else 1.0 - lam
        targets[pair] = r_bonded + (progress**_TIGHTENING) * (r_free - r_bonded)
    return targets


def driving_restraint(x, lam, bond, contact):
    """Return the harmonic reactive-distance restraint and its Cartesian gradient."""
    e_total = 0.0
    g = np.zeros_like(x)
    # Force constants soften towards mid-lambda (bonding most ambiguous), recover at endpoints.
    fc_scale = 4.0 * (lam - 0.5) ** 2 * (1.0 - _FC_DIP) + _FC_DIP
    targets = restraint_targets(lam=lam, bond=bond, contact=contact)

    for pair, (side, k_bonded, _) in bond.items():
        i, j = pair

        if side == "rea":  # breaking: bonded at l=0, apart at l=1
            k_a, k_b = k_bonded, _GHOST_FC * k_bonded
        else:  # forming: apart at l=0, bonded at l=1
            k_a, k_b = _GHOST_FC * k_bonded, k_bonded

        r0 = targets[pair]
        k = ((1.0 - lam) * k_a + lam * k_b) * fc_scale

        d = x[i] - x[j]
        r = float(np.linalg.norm(d))
        dr = r - r0
        e, de = 0.5 * k * dr**2, k * dr
        e_total += e
        g[i] += de * d / r
        g[j] -= de * d / r

    return e_total, g


def _endpoint_metal_angles(mol, isomer=None, mapping=None):
    """Build metal-centred angle references in `mol`'s atom order from its geometry."""

    def mapped(i):
        i = int(i)
        return i if mapping is None else int(mapping[i])

    centres = []
    source = "endpoint_graph"
    if isomer is not None and isomer.centres:
        source = "native_state"
        for state in isomer.centres:
            sites = []
            for site in state.vertices:
                if site is None:
                    continue
                atoms = site.atoms if hasattr(site, "atoms") else (int(site),)
                sites.append(tuple(mapped(i) for i in atoms))
            centres.append((mapped(state.atom), sites))
    else:
        metals = metal_atoms(mol)
        if not metals:
            return (), None
        for center in metals:
            sites = [(int(atom.GetIdx()),) for atom in mol.GetAtomWithIdx(center).GetNeighbors()]
            centres.append((int(center), sites))

    x = mol.GetConformer().GetPositions()
    if not np.isfinite(x).all():
        raise ValueError("metal angle reference coordinates must be finite")

    angles = []
    for center, sites in centres:
        for i, left in enumerate(sites):
            for right in sites[i + 1 :]:
                a = np.mean(x[list(left)], axis=0) - x[center]
                b = np.mean(x[list(right)], axis=0) - x[center]
                ra, rb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                if not np.isfinite(ra) or not np.isfinite(rb):
                    raise ValueError(f"metal angle reference at atom {center} has a nonfinite site leg")
                if ra <= 1e-12 or rb <= 1e-12:
                    raise ValueError(f"metal angle reference at atom {center} has a zero-length site leg")
                theta = float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))
                if not np.isfinite(theta):
                    raise ValueError(f"metal angle reference at atom {center} must have finite geometry")
                angles.append((center, left, right, theta))
    return tuple(angles), source if angles else None


def metal_angle_references(rea, pro, native_states=(None, None), product_mapping=None):
    """Return endpoint angle terms and the concise source label used by a reaction scan."""
    rea_angles, rea_source = _endpoint_metal_angles(rea, native_states[0])
    pro_angles, pro_source = _endpoint_metal_angles(pro, native_states[1], product_mapping)
    sources = {source for source in (rea_source, pro_source) if source}
    return (rea_angles, pro_angles), "+".join(sorted(sources)) or None


def _metal_angle_energy_grad(x, lam, k, references):
    """Endpoint-weighted harmonic metal angles and analytic Cartesian gradient."""
    if not k or not references:
        return 0.0, np.zeros_like(x)
    if not np.isfinite(x).all():
        raise ValueError("metal angle coordinates must be finite")
    energy, gradient = 0.0, np.zeros_like(x)
    for weight, angles in ((1.0 - lam, references[0]), (lam, references[1])):
        if not weight:
            continue
        for center, left, right, target in angles:
            a = np.mean(x[list(left)], axis=0) - x[center]
            b = np.mean(x[list(right)], axis=0) - x[center]
            ra, rb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if not np.isfinite(ra) or not np.isfinite(rb):
                raise ValueError(f"metal angle at atom {center} has a nonfinite site leg")
            if ra <= 1e-12 or rb <= 1e-12:
                raise ValueError(f"metal angle at atom {center} has a zero-length site leg")
            cross = np.cross(a, b)
            sin_theta = float(np.linalg.norm(cross) / (ra * rb))
            cos_theta = float(np.clip(np.dot(a, b) / (ra * rb), -1.0, 1.0))
            theta = float(np.arctan2(sin_theta, cos_theta))
            if not np.isfinite(target) or not np.isfinite(theta):
                raise ValueError(f"metal angle at atom {center} must have finite geometry and reference")
            delta = theta - target
            energy += weight * 0.5 * k * delta**2
            if sin_theta <= _ANGLE_SINGULAR:
                if abs(delta) <= 1e-10:
                    continue  # a matching exact linear angle has zero limiting energy and gradient
                raise ValueError(f"nonmatching collinear metal angle at atom {center} has no defined gradient")
            unit_a, unit_b = a / ra, b / rb
            dtheta_a = (cos_theta * unit_a - unit_b) / (ra * sin_theta)
            dtheta_b = (cos_theta * unit_b - unit_a) / (rb * sin_theta)
            scale = weight * k * delta
            ga, gb = scale * dtheta_a, scale * dtheta_b
            gradient[list(left)] += ga / len(left)
            gradient[list(right)] += gb / len(right)
            gradient[center] -= ga + gb
    return energy, gradient


class _Endpoint:
    """One endpoint's MMFF field and parameters."""

    def __init__(self, mol):
        self.mol = mol
        self.props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        if self.props is None:
            raise ValueError(
                "MMFF94 cannot type this molecule. Every atom needs standard "
                "valences -- radicals and exotic elements are not supported."
            )
        # Fragments must interact: a bimolecular reaction IS their interaction.
        self.ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, self.props, nonBondedThresh=1e5, ignoreInterfragInteractions=False
        )
        self.topo = Chem.GetDistanceMatrix(mol)
        self.bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}

    def energy(self, x):
        return self.ff.CalcEnergy(list(x.ravel()))

    def energy_grad(self, x):
        flat = list(x.ravel())
        return self.ff.CalcEnergy(flat), np.array(self.ff.CalcGrad(flat)).reshape(-1, 3)

    def raw_bond_params(self, i, j):
        # RDKit's tabulated stretch parameters as returned: (kb in mdyn/A, r0 in A),
        # the units _mmff_bond consumes before applying _MMFF_BOND_K itself.
        _, kb, r0 = self.props.GetMMFFBondStretchParams(self.mol, i, j)
        return kb, r0

    def bond_params(self, i, j):
        kb, r0 = self.raw_bond_params(i, j)
        return _MMFF_BOND_K * kb, r0  # (curvature kcal/mol/A^2, length A)

    def vdw_params(self, i, j):
        p = self.props.GetMMFFVdWParams(i, j)
        return p[2], p[3]  # (R*_ij, epsilon)

    def pair_charge(self, i, j):
        return self.props.GetMMFFPartialCharge(i) * self.props.GetMMFFPartialCharge(j)

    def nonbonded_scale(self, i, j):
        """How MMFF treats pair (i, j): None if excluded, else the 1-4 scale."""
        d = self.topo[i, j]
        if d < _ONE_FOUR:
            return None  # 1-2 and 1-3 pairs are excluded
        return _MMFF_14_SCALE if d == _ONE_FOUR else 1.0


class _ReactionPotential:
    """Shared atom data, endpoint guidance, and reactive targets."""

    # Each endpoint engine supplies these parameters during setup.
    _bond: dict[tuple[int, int], tuple[str, float, float]]  # pair -> (side, force constant, bonded r)
    _contact: dict[tuple[int, int], float]  # pair -> the "apart" distance, in angstrom

    def __init__(
        self,
        rea,
        forming,
        breaking,
        *,
        pro=None,
        metal_angle_k=_METAL_ANGLE_K,
        metal_angle_references=None,
        metal_angle_source=None,
    ):
        if (
            isinstance(metal_angle_k, bool)
            or not isinstance(metal_angle_k, Real)
            or not np.isfinite(metal_angle_k)
            or metal_angle_k < 0
        ):
            raise ValueError("metal_angle_k must be a finite non-negative number in kcal/mol/rad²")
        self.n_atoms = rea.GetNumAtoms()
        self.numbers = np.array([atom.GetAtomicNum() for atom in rea.GetAtoms()], dtype=np.int32)
        self.forming = sorted(forming)
        self.breaking = sorted(breaking)
        self.changing = sorted(set(forming) | set(breaking))
        self.masses = np.array([a.GetMass() for a in rea.GetAtoms()])
        self.metal_angle_k = float(metal_angle_k)
        if metal_angle_references is None and self.metal_angle_k and pro is not None:
            rea_refs, rea_source = _endpoint_metal_angles(rea)
            pro_refs, pro_source = _endpoint_metal_angles(pro)
            metal_angle_references = rea_refs, pro_refs
            sources = {source for source in (rea_source, pro_source) if source}
            metal_angle_source = "+".join(sorted(sources)) or None
        self.metal_angle_references = metal_angle_references or ((), ())
        self.metal_angle_source = metal_angle_source if any(self.metal_angle_references) else None

    def targets(self, lam):
        """Length each reactive bond is held at, at this `l`. See :func:`restraint_targets`."""
        return restraint_targets(lam=lam, bond=self._bond, contact=self._contact)


class MMFFPotential(_ReactionPotential):
    """Interpolate MMFF94 endpoint fields, replacing terms for reactive pairs."""

    def __init__(
        self,
        rea,
        pro,
        forming,
        breaking,
        morse_d=_MORSE_D,
        soft_core=_SOFT_CORE_FRACTION,
        metal_angle_k=_METAL_ANGLE_K,
        metal_angle_references=None,
        metal_angle_source=None,
    ):
        """Type both endpoints with MMFF94 once, so every lambda reuses the same two fields."""
        super().__init__(
            rea,
            forming,
            breaking,
            pro=pro,
            metal_angle_k=metal_angle_k,
            metal_angle_references=metal_angle_references,
            metal_angle_source=metal_angle_source,
        )
        self.rea = _Endpoint(rea)
        self.pro = _Endpoint(pro)
        self.morse_d = morse_d
        self.soft_core = soft_core

        if not self.changing:
            raise ValueError(
                "No bonds form or break between these structures -- there is no "
                "reaction to find a transition state for."
            )

        # Reactive-bond parameters, resolved once. Built over self.changing (sorted) so
        # _bond.items() iterates in sorted order, which is what makes the driving-restraint
        # summation reproducible; GFNFFPotential builds its own the same way.
        self._bond = {}
        is_breaking = set(self.breaking)
        for pair in self.changing:
            if pair in is_breaking:
                self._bond[pair] = ("rea", *self.rea.bond_params(*pair))
            else:
                self._bond[pair] = ("pro", *self.pro.bond_params(*pair))

        # As a plain non-bonded pair, the vdW minimum is the natural "just touching" distance.
        self._contact = {p: self.pro.vdw_params(*p)[0] for p in self.changing}

    # -- endpoint energies ------------------------------------------------

    def _endpoint(self, x, side, patch, gradient=True):
        """Evaluate one endpoint; `pes` patches reactive pairs, `integration` removes their terms."""
        end = self.rea if side == "rea" else self.pro
        if gradient:
            e, g = end.energy_grad(x)
        else:
            e, g = end.energy(x), None

        for i, j in self.changing:
            d = x[i] - x[j]
            r = float(np.linalg.norm(d))
            de = 0.0

            if (i, j) in end.bonds:
                # Bonded here, so MMFF has a stretch term for it; remove it.
                kb, r0 = end.raw_bond_params(i, j)
                e_b, de_b = _mmff_bond(kb, r0, r)
                e -= e_b
                de -= de_b
                if patch == "pes":
                    # Replace it with a bond that can break.
                    k, _ = end.bond_params(i, j)
                    e_m, de_m = _morse(d_e=self.morse_d, k=k, r0=r0, r=r)
                    e += e_m
                    de += de_m
            else:
                # Not bonded here: an ordinary non-bonded pair about to be compressed. Remove MMFF's.
                scale = end.nonbonded_scale(i, j)
                if scale is None:
                    continue  # 1-2 / 1-3: MMFF already excludes it, nothing to do
                rstar, eps = end.vdw_params(i, j)
                qq = end.pair_charge(i, j)
                e_v, de_v = _mmff_vdw(rstar, eps, r)
                e_e, de_e = _mmff_ele(qq, r, scale)
                e -= e_v + e_e
                de -= de_v + de_e
                if patch == "pes":
                    rcut = self.soft_core * rstar
                    e_v, de_v = _soft_core(fn=_mmff_vdw, head=(rstar, eps), r=r, rcut=rcut)
                    e_e, de_e = _soft_core(fn=_mmff_ele, head=(qq,), r=r, rcut=rcut, tail=(scale,))
                    e += e_v + e_e
                    de += de_v + de_e

            if g is not None:
                unit = d / r
                g[i] += de * unit
                g[j] -= de * unit

        return e, g

    def energy(self, x, lam):
        """Evaluate the interpolated reaction energy V(x, l), kcal/mol; returns (E, E_rea, E_pro).

        The surface the transition state is read off -- scored, never minimised on (minimising slides
        back to reactants or products). No caller reads a gradient here, so none is computed.
        """
        e1 = self._endpoint(x, "rea", "pes", gradient=False)[0]
        e2 = self._endpoint(x, "pro", "pes", gradient=False)[0]
        return (1.0 - lam) * e1 + lam * e2, e1, e2

    # -- the potential geometries are generated on --------------------------

    def sampling_energy(self, x, lam):
        """Evaluate the potential that structures are sampled on, in kcal/mol.

        The reactive pairs' own MMFF terms are removed and replaced by the driving
        restraint, so the reaction coordinate is set by `l` and everything else
        relaxes around it.
        """
        e1, g1 = self._endpoint(x, "rea", "integration")
        e2, g2 = self._endpoint(x, "pro", "integration")
        e_r, g_r = driving_restraint(x=x, lam=lam, bond=self._bond, contact=self._contact)
        e = (1.0 - lam) * e1 + lam * e2 + e_r
        g = (1.0 - lam) * g1 + lam * g2 + g_r
        e_metal, g_metal = _metal_angle_energy_grad(
            x, lam, getattr(self, "metal_angle_k", 0.0), getattr(self, "metal_angle_references", ((), ()))
        )
        e += e_metal
        g += g_metal
        return e, g


class GFNFFPotential(_ReactionPotential):
    """Interpolate geometry-frozen GFN-FF endpoints with reactive-distance guidance."""

    def __init__(
        self,
        rea,
        pro,
        forming,
        breaking,
        charge=0,
        n_cores=None,
        bond_k=500.0,
        torsion=None,
        solvent=None,
        metal_angle_k=_METAL_ANGLE_K,
        metal_angle_references=None,
        metal_angle_source=None,
    ):
        """Build both endpoint fields with their topologies frozen, and the driving restraint between them."""
        from rxts.gfnff import GFNFF

        super().__init__(
            rea,
            forming,
            breaking,
            pro=pro,
            metal_angle_k=metal_angle_k,
            metal_angle_references=metal_angle_references,
            metal_angle_source=metal_angle_source,
        )

        #: A conformational reaction (rotation, ring flip) changes no bonds -- no distance to drive.
        #: What moves is a dihedral, driven instead. See `rxts.evb`.
        self.torsion = torsion

        if not self.changing and torsion is None:
            raise ValueError(
                "No bonds form or break between these structures -- there is no "
                "reaction to find a transition state for."
            )

        # Bonded and touching distances from radii: no force field consulted, so any element works.
        # `bond_k` is a typical single-bond stiffness, firm enough to set the reactive distances.
        table = Chem.GetPeriodicTable()
        z = [a.GetAtomicNum() for a in rea.GetAtoms()]
        self._bond, self._contact = {}, {}
        is_breaking = set(self.breaking)
        for pair in self.changing:
            i, j = pair
            side = "rea" if pair in is_breaking else "pro"
            r_bonded = table.GetRcovalent(z[i]) + table.GetRcovalent(z[j])
            self._bond[pair] = (side, bond_k, r_bonded)
            self._contact[pair] = table.GetRvdw(z[i]) + table.GetRvdw(z[j])

        # Each calculator freezes its topology from the geometry it is given.
        self.rea = GFNFF(rea, charge=charge, solvent=solvent)
        self.pro = GFNFF(pro, charge=charge, solvent=solvent)
        self._pool = ThreadPoolExecutor(max_workers=2) if (n_cores or 2) > 1 else None

    def _endpoints(self, x):
        """Both topologies at one geometry. They are independent, so run them at once."""
        from rxts.gfnff import UNUSABLE

        if self._pool is None:
            endpoints = self.rea.energy_grad(x), self.pro.energy_grad(x)
        else:
            futures = (
                self._pool.submit(self.rea.energy_grad, x),
                self._pool.submit(self.pro.energy_grad, x),
            )
            endpoints = futures[0].result(), futures[1].result()
        # An overflow sentinel has zero force; accepting it would falsely certify an ASE minimum.
        if any(energy == UNUSABLE for energy, _ in endpoints):
            raise RuntimeError("GFN-FF could not evaluate a frozen topology at this geometry.")
        return endpoints

    def _interpolated(self, x, lam):
        """Return interpolated energy/gradient and both endpoint energies."""
        (e1, g1), (e2, g2) = self._endpoints(x)
        return (1.0 - lam) * e1 + lam * e2, (1.0 - lam) * g1 + lam * g2, e1, e2

    def energy(self, x, lam):
        """Evaluate the interpolated reaction energy V(x, l), in kcal/mol; returns (E, E_rea, E_pro).

        Scored, never minimised on. The driving restraint is absent here, so the barrier
        is the force field's and not the restraint's.
        """
        e, _, e1, e2 = self._interpolated(x, lam)
        return e, e1, e2

    def sampling_energy(self, x, lam):
        """Energy sampled structures are relaxed on: V plus the driving restraint (and torsion if conformational).

        Uses VeloxChem's 3.25 tightening exponent, as the MMFF engine does.
        """
        e, g, _, _ = self._interpolated(x, lam)
        if self.changing:
            e_r, g_r = driving_restraint(x=x, lam=lam, bond=self._bond, contact=self._contact)
            e, g = e + e_r, g + g_r
        if self.torsion is not None:
            e_t, g_t = driving_torsion(x, lam, *self.torsion)
            e, g = e + e_t, g + g_t
        e_metal, g_metal = _metal_angle_energy_grad(
            x,
            lam,
            getattr(self, "metal_angle_k", 0.0),
            getattr(self, "metal_angle_references", ((), ())),
        )
        e, g = e + e_metal, g + g_metal
        return e, g


# Conformational transitions drive the most changed torsion with a periodic restraint.
_COLLINEAR = 1e-8


def _signed_dihedral(x, i, j, k, m):
    """RDKit signed radians; collinear is zero, near-coincident adjacent atoms raise ValueError."""
    x = np.asarray(x, dtype=float)
    conformer = Chem.Conformer(4)
    for index, atom in enumerate((i, j, k, m)):
        conformer.SetAtomPosition(index, x[atom].tolist())
    angle = rdMolTransforms.GetDihedralRad(conformer, 0, 1, 2, 3)
    return 0.0 if np.isnan(angle) else float(angle)


def dihedral(x, i, j, k, m):
    """Measure a torsion in radians, returning zero if either plane is collinear.

    This shares `_COLLINEAR` with `torsion_gradient`. Active torsion selection and restraint energy
    use `_signed_dihedral` without this floor.
    """
    x = np.asarray(x, dtype=float)
    b1, b2, b3 = x[j] - x[i], x[k] - x[j], x[m] - x[k]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    if np.linalg.norm(n1) < _COLLINEAR or np.linalg.norm(n2) < _COLLINEAR:
        return 0.0
    return _signed_dihedral(x, i, j, k, m)


#: VeloxChem's value: 200 kJ/mol. Stiff enough to drag the torsion round against the barrier it
#: is climbing, soft enough that the rest of the molecule is not distorted doing it.
CONFORMER_K = 47.8  # kcal/mol  (= 200 kJ/mol)

#: Below this the two structures do not meaningfully differ by a rotation.
THRESHOLD = 20.0  # degrees


def _wrap(delta):
    """Bring a difference of angles into [-180, 180), the shortest way round."""
    return (delta + 180.0) % 360.0 - 180.0


def candidate_torsions(mol):
    """Return one dihedral for each central bond with neighbors at both ends."""
    torsions = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
        right = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]
        if not left or not right:
            continue  # a terminal bond has no torsion about it
        torsions.append((left[0], j, k, right[0]))
    return torsions


def active_torsion(reactant, product):
    """Find the most changed torsion, or raise if the structures differ by less than `THRESHOLD`."""
    x_rea = np.asarray(reactant.GetConformer().GetPositions(), dtype=float)
    x_pro = np.asarray(product.GetConformer().GetPositions(), dtype=float)

    best, biggest = None, 0.0
    for torsion in candidate_torsions(reactant):
        phi_rea = np.degrees(_signed_dihedral(x_rea, *torsion))
        phi_pro = np.degrees(_signed_dihedral(x_pro, *torsion))
        delta = abs(_wrap(phi_pro - phi_rea))
        if delta > biggest:
            best, biggest = (torsion, phi_rea, phi_pro), delta

    if best is None or biggest < THRESHOLD:
        raise ValueError(
            f"No bonds form or break, and no dihedral rotates by more than {THRESHOLD:.0f} "
            f"degrees either (the largest is {biggest:.1f}). These two structures do not "
            "differ by a reaction this method can describe."
        )
    return best


def torsion_gradient(x, i, j, k, m):
    """Return the analytic dihedral derivative, d(phi)/dx, in rad/angstrom."""
    r_ij = x[i] - x[j]
    r_kj = x[k] - x[j]
    r_kl = x[k] - x[m]

    n1, n2 = np.cross(r_ij, r_kj), np.cross(r_kj, r_kl)
    n1_sq, n2_sq = n1 @ n1, n2 @ n2
    length = np.linalg.norm(r_kj)

    grad = np.zeros_like(x)
    if np.linalg.norm(n1) < _COLLINEAR or np.linalg.norm(n2) < _COLLINEAR or length < _COLLINEAR:
        return grad

    d_i = length / n1_sq * n1
    d_l = -length / n2_sq * n2

    inner1 = (r_ij @ r_kj) / (length * length)
    inner2 = (r_kl @ r_kj) / (length * length)

    grad[i] = d_i
    grad[j] = (inner1 - 1.0) * d_i - inner2 * d_l
    grad[k] = (inner2 - 1.0) * d_l - inner1 * d_i
    grad[m] = d_l
    return grad


def driving_torsion(x, lam, torsion, phi_reactant, phi_product):
    """Restrain a torsion to an interpolated periodic minimum.

        E = k * (1 - cos(phi - phi_0)),      phi_0 = phi_reactant + lambda * wrap(phi_product - phi_reactant)

    Returns energy in kcal/mol and gradient in kcal/mol/angstrom.
    """
    target = phi_reactant + lam * _wrap(phi_product - phi_reactant)
    phi = np.degrees(_signed_dihedral(x, *torsion))
    delta = np.radians(_wrap(phi - target))

    energy = CONFORMER_K * (1.0 - np.cos(delta))
    # dE/dx = dE/dphi * dphi/dx, and dE/dphi = k sin(phi - phi_0)
    return energy, CONFORMER_K * np.sin(delta) * torsion_gradient(x, *torsion)
