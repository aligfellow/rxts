"""Map reaction mechanisms: anchor the untouched atoms, then search the rest exactly.

:func:`_anchors` pairs atoms whose surroundings are unchanged and unambiguous, as VeloxChem's
``ReactionMatcher`` assist ids do (``reactionmatcher.py:346-503``) but without its greedy, atom-order
dependent pick. Branch-and-bound over bijections (:func:`_optimum`, :func:`_search`) maps the rest,
enumerates tied mechanisms, and ranks them by the pericyclic array. Naming bonds edits the reactant
the anchors read; it does not constrain that search. Mechanism selection among ties happens in the
caller.

SMILES parsing, graph mapping and relay construction are independent of endpoint embedding and FF scanning.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import permutations, product
from numbers import Integral
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import rxembed
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

from rxts.utils import describe, is_xyz_file, mol_from_xyz

logger = logging.getLogger(__name__)

# Search nodes allowed across every pass before the mapper gives up rather than hang. Most
# reactions need a few hundred; listing ties visits at least one node per tied mapping, so this bounds the
# tie list too.
_MAX_NODES = 200_000

# Search depth down to which a node is bounded by an assignment of the atoms left (see `_search`).
_BOUND_DEPTH = 8

# Ceiling on changing bonds: where `_optimum`'s ladder stops and reports no single step.
_MAX_CHANGING = 12

# Bound the factorial ring-ranking search for large changed-bond sets.
_MAX_ARRAY_BONDS = 6

# Graph-ranking heuristic favours six-atom reaction circuits.
_HUCKEL_ARRAY = 6

# Pauling electronegativity by atomic number (CRC values); He, Ne, Ar and elements past Lr have none.
_PAULING = [None] + [
    None if value == "-" else float(value)
    for value in """
    2.20 -    0.98 1.57 2.04 2.55 3.04 3.44 3.98 -    0.93 1.31 1.61 1.90 2.19 2.58 3.16 -    0.82 1.00
    1.36 1.54 1.63 1.66 1.55 1.83 1.88 1.91 1.90 1.65 1.81 2.01 2.18 2.55 2.96 3.00 0.82 0.95 1.22 1.33
    1.6  2.16 1.9  2.2  2.28 2.20 1.93 1.69 1.78 1.96 2.05 2.1  2.66 2.6  0.79 0.89 1.10 1.12 1.13 1.14
    1.13 1.17 1.2  1.2  1.1  1.22 1.23 1.24 1.25 1.1  1.27 1.3  1.5  2.36 1.9  2.2  2.20 2.28 2.54 2.00
    1.62 2.33 2.02 2.0  2.2  2.2  0.7  0.9  1.1  1.3  1.5  1.38 1.36 1.28 1.3  1.3  1.3  1.3  1.3  1.3
    1.3  1.3  1.3
    """.split()
]


_REACTION_PARTS = 3  # reactants > agents > products


def split_reaction_smiles(reaction):
    """Split reaction SMILES into reactants and products, dropping agents.

    CX atom indices span all fragments, so keep each labeled endpoint intact.
    """
    if not isinstance(reaction, str):
        return None
    # Dative arrows and CX labels belong to the molecule, not to reaction separators.
    parts = [""]
    for i, block in enumerate(re.split(r"(\|[^|]*\|)", reaction)):
        if i % 2:
            parts[-1] += block
        else:
            chunks = re.split(r"(?<!-)>", block)
            parts[-1] += chunks[0]
            parts.extend(chunks[1:])
    if len(parts) == 1:
        return None
    if len(parts) != _REACTION_PARTS:
        raise ValueError(
            f"Reaction SMILES {reaction!r} is malformed. Expected "
            "'reactants>agents>products' -- or 'reactants>>products' when there "
            "are no agents."
        )
    reactants = [parts[0]] if "|" in parts[0] else [s for s in parts[0].split(".") if s]
    products = [parts[2]] if "|" in parts[2] else [s for s in parts[2].split(".") if s]
    if not reactants or not products:
        raise ValueError(f"Reaction SMILES {reaction!r} needs at least one reactant and one product.")
    return reactants, products


def parse_smiles(smiles):
    """Parse explicit hydrogens in the shared mapper/embedding atom order."""
    return Chem.AddHs(rxembed.parse_smiles(smiles, remove_hs=False))


def _from_smiles(smiles, side):
    """Parse a reaction side and name it in parse errors."""
    if not isinstance(smiles, str):
        smiles = ".".join(smiles)
    try:
        return parse_smiles(smiles)
    except ValueError:
        raise ValueError(f"Could not parse the {side} of this reaction: {smiles!r}") from None


class _ExhaustedError(Exception):
    """The search ran out of node budget, so no answer it could give is trustworthy."""


@dataclass
class Mechanism:
    """One way the reaction could happen."""

    mapping: list[int]  # product atom k is reactant atom mapping[k]
    forming: frozenset  # pairs bonded only in the product
    breaking: frozenset  # pairs bonded only in the reactant

    @property
    def changing(self) -> frozenset:
        """Every bond the reaction touches."""
        return self.forming | self.breaking


def _adjacency(mol):
    return [{n.GetIdx() for n in a.GetNeighbors()} for a in mol.GetAtoms()]


def _pericyclic_array(mech, rea_bonds):
    """Return the shortest closed array containing all changing bonds, or zero.

    Join changing bonds in every order and orientation along unchanged paths. The first
    orientation fixes cyclic symmetry. Array size is a graph-based ranking heuristic;
    bond orders are ignored."""
    changing = sorted(tuple(sorted(pair)) for pair in mech.changing)
    if not changing or len(changing) > _MAX_ARRAY_BONDS:
        return 0

    spine = nx.Graph(rea_bonds | {tuple(sorted(pair)) for pair in mech.forming})
    spine.remove_edges_from(changing)  # the connecting runs are unchanged bonds, never changing ones
    reach = {a: nx.single_source_shortest_path_length(spine, a) for pair in changing for a in pair}

    first, rest = changing[0], changing[1:]
    best = 0
    for order in permutations(rest):
        for flipped in product((False, True), repeat=len(order)):
            walk = [first] + [(b, a) if flip else (a, b) for (a, b), flip in zip(order, flipped, strict=True)]
            ring = len(walk)
            for (_, leaving), (arriving, _) in zip(walk, walk[1:] + walk[:1], strict=True):
                run = reach[leaving].get(arriving)
                if run is None:
                    ring = None
                    break
                ring += run
            if ring is not None and (best == 0 or ring < best):
                best = ring
    return best


def _environment(mol, radius=2):
    """Fingerprint each atom by its local bonding environment out to `radius` bonds (default 2).

    Orders mapping candidates so a cheap mapping is found early; never filters, since an atom near the
    reaction legitimately differs between the two sides. Not the anchors, which fix an atom whose
    whole neighbourhood is unchanged and unambiguous; ordering never fixes an atom.
    """
    env = [(a.GetAtomicNum(),) for a in mol.GetAtoms()]
    for _ in range(radius):
        env = [env[a.GetIdx()] + tuple(sorted(env[n.GetIdx()] for n in a.GetNeighbors())) for a in mol.GetAtoms()]
    return env


def _orbits(mol, adjacency, atoms, fixed, hydrogens=None):
    """A stabiliser chain of the symmetries of `mol` on `atoms` that leave every `fixed` atom in place.

    Takes `atoms` in order. Each one some remaining symmetry moves is yielded with one symmetry carrying it
    to each other atom of its orbit, then held fixed, so later orbits are those of the symmetries left:
    two H on a CH2, the methyls of an iPr, two AcOH, and the two faces of every free phenyl. RDKit's
    untied canonical ranks propose an orbit; equal canonical SMILES rooted at two atoms prove it, and their
    output orders give the symmetry.
    """
    skeleton = _skeleton(mol, adjacency, hydrogens)
    labels = skeleton[0]
    for k, a in enumerate(fixed, 1):
        labels.GetAtomWithIdx(a).SetAtomMapNum(k)
    held = len(fixed)
    ranks = Chem.CanonicalRankAtoms(labels, breakTies=False)
    for x in atoms:
        alike = [y for y in atoms if y != x and ranks[y] == ranks[x]]
        if not alike:
            continue
        key, start = _canonical(skeleton, atoms, x)
        symmetries = []
        for y in alike:
            other, image = _canonical(skeleton, atoms, y)
            if other == key:
                symmetries.append(dict(zip(start, image, strict=True)))
        yield x, symmetries
        held += 1
        labels.GetAtomWithIdx(x).SetAtomMapNum(held)
        ranks = Chem.CanonicalRankAtoms(labels, breakTies=False)


def _symmetry(pro, pro_adj, rea, rea_adj, candidates, order, pro_hydrogens=None, rea_hydrogens=None):
    """Lex-leader constraints that search one mapping of every family related by symmetry.

    A symmetry of the product leaves every bond as it was, and one of the reactant gives the same mechanism
    relabelled, so of each family only the mapping first in this order is needed: atoms in search `order`,
    images compared by reactant index. Along the product's stabiliser chain (see `_orbits`), taken in that
    order, that mapping gives each atom the smallest image in its orbit (`below`). A reactant symmetry first
    changes the image of the first product atom placed into an atom it moves, and that atom must take the
    smaller of its target and the target's image (`entry`). Every constraint keeps the order's first
    mapping, so all of them together lose nothing.
    """
    pinned = [p for p in order if len(candidates[p]) == 1]
    below = [[] for _ in range(pro.GetNumAtoms())]
    for x, symmetries in _orbits(pro, pro_adj, order, pinned, pro_hydrogens):
        for symmetry in symmetries:
            below[symmetry[x]].append(x)
    reactant_atoms = sorted({r for p in order for r in candidates[p]})
    fixed = [candidates[p][0] for p in pinned]
    moves = [s for _, ss in _orbits(rea, rea_adj, reactant_atoms, fixed, rea_hydrogens) for s in ss]
    entry = [[] for _ in range(rea.GetNumAtoms())]
    for k, symmetry in enumerate(moves):
        for a, b in symmetry.items():
            if a != b:
                entry[a].append((k, b))
    return below, entry, len(moves)


def _terminal_hydrogens(mol):
    """Each atom's hydrogens, or None when any hydrogen is not a plain H on exactly one heavy atom."""
    found = [[] for _ in range(mol.GetNumAtoms())]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            partners = atom.GetNeighbors()
            if atom.GetIsotope() or len(partners) != 1 or partners[0].GetAtomicNum() == 1:
                return None
            found[partners[0].GetIdx()].append(atom.GetIdx())
    return found


def _with_hydrogens(heavy, rea_h, pro_h):
    """Complete a mapping of heavy atoms with their hydrogens, once per distinct way the spare ones move.

    A hydrogen stays on its atom where it can; an atom left short takes spare hydrogens from atoms holding
    more than their partner needs. Hydrogens on one atom are interchangeable, so ways differ only in how
    many each donor gives each short atom, and a short atom's slots take donors in order.
    """
    full, spare, short = list(heavy), {}, []
    for p, r in enumerate(heavy):
        if r is not None:
            kept = min(len(rea_h[r]), len(pro_h[p]))
            for a, b in zip(rea_h[r][:kept], pro_h[p][:kept], strict=True):
                full[b] = a
            if len(rea_h[r]) > kept:
                spare[r] = rea_h[r][kept:]
            short += [(p, h) for h in pro_h[p][kept:]]
    donors, given = sorted(spare), dict.fromkeys(spare, 0)

    def fill(k, start):
        if k == len(short):
            yield list(full)
            return
        p, h = short[k]
        first = start if k and short[k - 1][0] == p else 0
        for i, r in enumerate(donors[first:], first):
            if given[r] < len(spare[r]):
                full[h] = spare[r][given[r]]
                given[r] += 1
                yield from fill(k + 1, i)
                given[r] -= 1

    return fill(0, 0)


def map_reaction(
    rea: Chem.Mol,
    pro: Chem.Mol,
    breaking_bonds: list | None = None,
    forming_bonds: list | None = None,
) -> tuple[Mechanism, list[Mechanism]]:
    """Return the best product-to-reactant mapping and all tied mechanisms.

    Named bonds use reactant indices and edit the reactant the anchors read, following `get_mapping`
    and `_prepare_graph` in reactionmatcher.py. They guide the anchors; the branch-and-bound search
    still minimises bond changes.

    More than one tied mechanism means connectivity alone leaves an ambiguity. Callers can select
    among ties or provide atom-mapped SMILES.
    """
    n = rea.GetNumAtoms()
    breaking_bonds = _normalise_pairs(breaking_bonds, "breaking_bonds", n)
    forming_bonds = _normalise_pairs(forming_bonds, "forming_bonds", n)
    if n != pro.GetNumAtoms():
        raise ValueError(
            f"Reactants have {n} atoms but products have {pro.GetNumAtoms()}. "
            "A reaction must conserve atoms -- check the SMILES for a missing "
            "species, wrong stoichiometry, or a typo."
        )

    rea_identity = [(a.GetAtomicNum(), a.GetIsotope()) for a in rea.GetAtoms()]
    pro_identity = [(a.GetAtomicNum(), a.GetIsotope()) for a in pro.GetAtoms()]
    rea_elem = [identity[0] for identity in rea_identity]
    pro_elem = [identity[0] for identity in pro_identity]
    pins = _map_pins(rea, pro)
    if sorted(rea_identity) != sorted(pro_identity):
        raise ValueError("Reactants and products have different molecular formulas. A reaction must conserve atoms.")

    explicit = _explicit_mapping(rea, pro)
    if explicit is not None:
        # A fully mapped SMILES has one mechanism and no search; the tied list still holds it, so
        # callers that read the tied set see it rather than an empty list.
        mech = _mechanism(rea, pro, explicit)
        return mech, [mech]

    rea_adj, pro_adj = _adjacency(rea), _adjacency(pro)
    rea_env, pro_env = _environment(rea), _environment(pro)

    candidates = [
        sorted(
            (r for r in range(n) if rea_identity[r] == pro_identity[p]),
            key=lambda r, p=p: (rea_env[r] != pro_env[p], -len(rea_adj[r] & pro_adj[p]), r),
        )
        for p in range(n)
    ]

    # Hydrogens on one atom are interchangeable, so when every one is a plain H on one heavy atom on both
    # sides, and none is pinned or named, the search places heavy atoms only and counts each one's
    # hydrogens: a pairing then pays |h_product - h_reactant|, exactly the hydrogen bonds it must change.
    # Otherwise the hydrogens are atoms of the search like any other.
    rea_h, pro_h = _terminal_hydrogens(rea), _terminal_hydrogens(pro)
    named = set(breaking_bonds or ()) | set(forming_bonds or ())
    fold = (
        rea_h is not None
        and pro_h is not None
        and not any(pro_elem[p] == 1 for p in pins)
        and not any(rea_elem[a] == 1 for pair in named for a in pair)
    )
    if fold:
        atoms = [p for p in range(n) if pro_elem[p] != 1]
        rea_adj = [{a for a in near if rea_elem[a] != 1} for near in rea_adj]
        pro_adj = [{a for a in near if pro_elem[a] != 1} for near in pro_adj]
    else:
        atoms, rea_h, pro_h = list(range(n)), [[] for _ in range(n)], [[] for _ in range(n)]
    rea_count, pro_count = [len(h) for h in rea_h], [len(h) for h in pro_h]

    # The reactant's bonds do not depend on the mapping, so build this set once, not once per mapping.
    rea_bonds = bond_set(rea)
    spent = [0]  # one search budget across every pass and release

    def tied(anchors, ceiling):
        """Every distinct mechanism changing the fewest bonds, with the anchors fixed.

        A known `ceiling` (a mapping exists changing that many) replaces the ladder with one pass under it.
        """
        fixed = [[anchors[p][0]] if p in anchors else candidates[p] for p in range(n)]
        # Nearest the anchors first, so each atom placed has a placed neighbour and a wrong pick costs a bond;
        # the rest by rarity.
        near = _distances(pro_adj, [p for p in atoms if len(fixed[p]) == 1])
        order = sorted(atoms, key=lambda p: (near.get(p, n), len(fixed[p]), -len(pro_adj[p]), -pro_elem[p]))
        costs = _assignment_costs(order, fixed, rea_adj, pro_adj, rea_identity, pro_identity, rea_count, pro_count)
        lower = _cheapest(costs[0])
        if math.isinf(lower):
            lower = 0.0  # no complete assignment of candidates: let the search report it
        search = partial(
            _search,
            n=n,
            order=order,
            candidates=fixed,
            rea_adj=rea_adj,
            pro_adj=pro_adj,
            pro_identity=pro_identity,
            rea_identity=rea_identity,
            rea_hydrogens=rea_count,
            pro_hydrogens=pro_count,
            symmetry=_symmetry(
                pro, pro_adj, rea, rea_adj, fixed, order, *((pro_count, rea_count) if fold else (None, None))
            ),
            spent=spent,
            costs=costs,
        )
        # Pass 1: the fewest bonds any mapping can change. Pass 2: every mapping that achieves it.
        seen, mechanisms = set(), []
        optimum = _optimum(search, lower) if ceiling is None else search(bound=ceiling + 1, collect=False)
        for heavy in search(bound=optimum, collect=True):
            for m in _with_hydrogens(heavy, rea_h, pro_h):
                pro_bonds = bond_set(pro, order=m)
                forming, breaking = frozenset(pro_bonds - rea_bonds), frozenset(rea_bonds - pro_bonds)
                if (forming, breaking) not in seen:
                    seen.add((forming, breaking))
                    mechanisms.append(Mechanism(list(m), forming, breaking))
        return mechanisms

    # An anchor fixed at radius d has no changed bond within d - 1 bonds of it, or its surroundings would
    # have differed; one that has was wrong, so it is released and the search run again. Named bonds were
    # applied before anchoring, so they do not count; explicit pins carry radius 0 and are never released.
    anchors = _anchors(rea, pro, pins, breaking_bonds, forming_bonds, (rea_count, pro_count) if fold else None)

    # Releasing anchors only widens the search, so the mapping just found still bounds the next round.
    kept, ceiling = dict(anchors), None
    try:
        while True:
            mechanisms = tied(kept, ceiling)
            changed = [a for mech in mechanisms for pair in mech.changing - named for a in pair]
            near = _distances(_adjacency(rea), changed)
            doubtful = [p for p, (r, radius) in kept.items() if near.get(r, n) < radius]
            if not doubtful:
                break
            ceiling = len(mechanisms[0].changing)
            for p in doubtful:
                del kept[p]
    except _ExhaustedError:
        raise ValueError(
            "Could not map these reactants onto these products within the search "
            "budget. Map the atoms you care about in the SMILES, or pass "
            "forming_bonds/breaking_bonds, and the search will respect them."
        ) from None

    def polarity(mech):
        # A hydride moves to carbon and a proton to oxygen, not the reverse: each moving hydrogen's new
        # partner is best as close in electronegativity to its old one as it can be. An element with no
        # Pauling value (He, Ne, Ar) adds nothing.
        old = {}
        for pair in mech.breaking:
            for h, partner in (pair, pair[::-1]):
                if rea_elem[h] == 1:
                    old.setdefault(h, []).append(partner)
        total = 0.0
        for pair in mech.forming:
            for h, partner in (pair, pair[::-1]):
                gaps = [
                    abs(_PAULING[rea_elem[o]] - _PAULING[rea_elem[partner]])
                    for o in old.get(h, ())
                    if max(rea_elem[o], rea_elem[partner]) < len(_PAULING)
                    and None not in (_PAULING[rea_elem[o]], _PAULING[rea_elem[partner]])
                ]
                total += min(gaps, default=0.0)
        return round(total, 3)  # rounded so equal gaps tie exactly, whatever order they were summed in

    # Atoms in a multiple or aromatic bond of the reactant: where pi electrons can move. Bond orders only
    # break ties among equally minimal mechanisms; they never decide what is minimal.
    unsaturated = {
        a
        for bond in rea.GetBonds()
        if bond.GetBondTypeAsDouble() > 1
        for a in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    }

    def rank(mech):
        # A pericyclic shift first: a six-atom array with no hydrogen moving and pi electrons at two or more
        # of its ends (Claisen, Cope, Diels-Alder; not a proton relay closing six atoms). Then the fewest
        # heavy atoms touched, the most hydrogen bonds changed, the most changes at pi atoms, and hydrogen
        # polarity; sorted bonds make the final tie-break deterministic. Chosen on the Golden set and checked
        # on Jaworski et al.'s (2019): 69% and 68% of curated mechanisms first among ties, against 44%.
        heavy = {a for pair in mech.changing for a in pair if rea_elem[a] != 1}
        h_bonds = sum(1 for i, j in mech.changing if rea_elem[i] == 1 or rea_elem[j] == 1)
        at_pi = sum(1 for pair in mech.changing for a in pair if a in unsaturated and rea_elem[a] != 1)
        pericyclic = h_bonds == 0 and at_pi >= 2 and _pericyclic_array(mech, rea_bonds) == _HUCKEL_ARRAY
        return (
            not pericyclic,
            len(heavy),
            -h_bonds,
            -at_pi,
            polarity(mech),
            sorted(mech.forming),
            sorted(mech.breaking),
        )

    mechanisms.sort(key=rank)
    return mechanisms[0], mechanisms


def _assignment_costs(order, candidates, rea_adj, pro_adj, rea_identity, pro_identity, rea_hydrogens, pro_hydrogens):
    """Per-pair lower bounds on changed bonds, as a matrix over `order` x reactant atoms, with its indices.

    An atom's neighbour kinds differ from its image's only through changed bonds at that atom, and a bond
    between two searched atoms is counted at both its ends; so a mapping changes at least half its summed
    neighbour mismatch plus its hydrogen-count differences, a sum over its pairs. Pairs outside the
    candidates cost infinity, so the cheapest assignment of the matrix bounds every mapping the search tries.
    """
    targets = sorted({r for p in order for r in candidates[p]})
    kinds = {k: i for i, k in enumerate(sorted({*rea_identity, *pro_identity}))}

    def counts(atoms, adjacency, identity, hydrogens):
        matrix = np.zeros((len(atoms), len(kinds) + 1))
        for row, atom in enumerate(atoms):
            for q in adjacency[atom]:
                matrix[row, kinds[identity[q]]] += 0.5
            matrix[row, -1] = hydrogens[atom]
        return matrix

    mismatch = np.abs(
        counts(order, pro_adj, pro_identity, pro_hydrogens)[:, None, :]
        - counts(targets, rea_adj, rea_identity, rea_hydrogens)[None, :, :]
    ).sum(axis=2)
    row, column = {p: i for i, p in enumerate(order)}, {r: j for j, r in enumerate(targets)}
    allowed = np.zeros(mismatch.shape, dtype=bool)
    for i, p in enumerate(order):
        allowed[i, [column[r] for r in candidates[p]]] = True
    mismatch[~allowed] = np.inf
    return mismatch, row, column, targets


def _cheapest(matrix):
    """The cheapest assignment's total (scipy linear_sum_assignment), infinite when none is complete."""
    try:
        rows, columns = linear_sum_assignment(matrix)
    except ValueError:
        return math.inf
    return float(matrix[rows, columns].sum())


def _optimum(search, lower=0.0):
    """Find the fewest bonds any mapping can change, by raising the ceiling until one fits.

    A ceiling of k prunes any partial mapping that has already changed k bonds; real reactions change
    few, so starting low and raising prunes hard where no ceiling would (the incremental cost is a
    lower bound that says nothing near the root). Upstream's `_find_breaking_edges` deepens the same
    way, one edge at a time. The ceiling starts just above `lower`, a proven bound on the answer.
    """
    for bound in range(math.ceil(lower - 1e-9) + 1, _MAX_CHANGING + 2):
        cost = search(bound=bound, collect=False)
        if cost < bound:
            return cost
    raise ValueError(
        f"No mapping between these reactants and products changes fewer than "
        f"{_MAX_CHANGING} bonds, so this does not look like one reaction step. "
        "Check that the product really is the product; if it is, map the atoms "
        "you care about in the SMILES, or pass forming_bonds/breaking_bonds."
    )


def _search(
    n,
    order,
    candidates,
    rea_adj,
    pro_adj,
    pro_identity,
    rea_identity,
    rea_hydrogens,
    pro_hydrogens,
    symmetry,
    spent,
    costs,
    bound,
    collect,
):
    """Branch-and-bound over element-preserving bijections of the atoms in `order`, never exceeding `bound`.

    With `collect` false, returns the fewest changing bonds achievable, or `bound`
    if nothing beats it. With it true, `bound` is that number and every mapping
    achieving it is returned, up to the swaps of identical parts `symmetry` orders (see `_symmetry`). Atoms left out
    of `order` (folded hydrogens) stay None; each pairing also pays its hydrogen-count mismatch.
    """
    mapping = [None] * n
    inverse = [None] * n
    best = bound  # the incumbent: the ceiling until a cheaper complete mapping beats it
    found = []
    below, entry, n_pairs = symmetry
    matrix, row, column, targets = costs
    placed_cost = 0.0  # the per-pair lower bounds of the pairs placed so far
    entered = [0] * n_pairs  # atoms placed so far into each swappable reactant pair
    unplaced = [Counter(pro_identity[q] for q in pro_adj[p]) for p in range(n)]
    unassigned = [Counter(rea_identity[s] for s in rea_adj[r]) for r in range(n)]

    def owing(p):
        """Bonds a placed atom must still change: neighbours to come that its image's cannot match.

        Each is a bond between this atom and one not yet placed, so no other placed atom counts it
        and `delta` has not yet: adding these keeps the bound sound.
        """
        mine, theirs = unplaced[p], unassigned[mapping[p]]
        return sum(abs(mine[e] - theirs[e]) for e in mine.keys() | theirs.keys())

    def delta(p, r):
        """Bonds changed by committing product atom `p` to reactant `r`.

        Only pairs with both ends already assigned are counted, which keeps this a lower bound on the
        final cost, and so a sound basis for pruning.
        """
        formed = sum(1 for q in pro_adj[p] if mapping[q] is not None and mapping[q] not in rea_adj[r])
        broken = sum(1 for s in rea_adj[r] if inverse[s] is not None and inverse[s] not in pro_adj[p])
        return formed + broken + abs(pro_hydrogens[p] - rea_hydrogens[r])

    def step(depth, cost, owed):
        nonlocal best, placed_cost
        # Node budget: truncating quietly would return a first-leaf bound indistinguishable from an answer.
        spent[0] += 1
        if spent[0] > _MAX_NODES:
            raise _ExhaustedError

        # Near the root, where a subtree is largest, bound it by the pairs placed so far plus the cheapest
        # assignment of the rest (see _assignment_costs): a doomed partial mapping is cut before it grows.
        if depth <= _BOUND_DEPTH < len(order) - depth:
            rest = matrix[np.ix_([row[q] for q in order[depth:]], [column[r] for r in targets if inverse[r] is None])]
            lower = placed_cost + _cheapest(rest)
            if (lower > best + 1e-9) if collect else (lower >= best - 1e-9):
                return

        if depth == len(order):
            if collect:
                found.append(list(mapping))
            elif cost < best:
                best = cost
            return

        p = order[depth]
        for r in candidates[p]:
            if inverse[r] is not None or any(r < mapping[a] for a in below[p]):
                continue
            if any(not entered[k] and b < r for k, b in entry[r]):
                continue  # the first atom into a swappable reactant pair takes the smaller of the two
            d = cost + delta(p, r)
            # Ties are pruned while hunting the optimum, kept while enumerating.
            if (d > best) if collect else (d >= best):
                continue
            mapping[p], inverse[r] = r, p
            placed_cost += matrix[row[p], column[r]]
            for k, _ in entry[r]:
                entered[k] += 1
            # Placing p takes one neighbour of p's kind off each neighbour's count, so each placed neighbour's
            # |unplaced - unassigned| for that kind moves by exactly one: down if its side had the surplus.
            e, o = pro_identity[p], owed
            for q in pro_adj[p]:
                if mapping[q] is not None:
                    o += -1 if unplaced[q][e] > unassigned[mapping[q]][e] else 1
                unplaced[q][e] -= 1
            for s in rea_adj[r]:
                if inverse[s] is not None:
                    o += -1 if unassigned[s][e] > unplaced[inverse[s]][e] else 1
                unassigned[s][e] -= 1
            o += owing(p)
            if (d + o <= best) if collect else (d + o < best):
                step(depth + 1, d, o)
            for q in pro_adj[p]:
                unplaced[q][pro_identity[p]] += 1
            for s in rea_adj[r]:
                unassigned[s][rea_identity[r]] += 1
            for k, _ in entry[r]:
                entered[k] -= 1
            placed_cost -= matrix[row[p], column[r]]
            mapping[p], inverse[r] = None, None

    step(0, 0, 0)  # an exhausted budget raises _ExhaustedError for map_reaction to retry or report
    return found if collect else best


def _mechanism(rea, pro, mapping):
    forming, breaking = reaction_bonds(rea, pro, mapping)
    return Mechanism(list(mapping), frozenset(forming), frozenset(breaking))


def _explicit_mapping(rea, pro):
    """Use the SMILES' own atom map numbers, when every atom carries one."""
    rea_maps = [a.GetAtomMapNum() for a in rea.GetAtoms()]
    pro_maps = [a.GetAtomMapNum() for a in pro.GetAtoms()]
    if 0 in rea_maps or 0 in pro_maps or set(rea_maps) != set(pro_maps):
        return None
    by_map = {m: i for i, m in enumerate(rea_maps)}
    return [by_map[m] for m in pro_maps]


def _map_pins(rea, pro):
    """Product-atom -> reactant-atom assignments forced by the SMILES' map numbers.

    Shared numbers must be unique and identify the same isotope and element.
    """
    rea_maps, pro_maps = {}, {}
    for atom in rea.GetAtoms():
        if atom.GetAtomMapNum():
            rea_maps.setdefault(atom.GetAtomMapNum(), []).append(atom.GetIdx())
    for atom in pro.GetAtoms():
        if atom.GetAtomMapNum():
            pro_maps.setdefault(atom.GetAtomMapNum(), []).append(atom.GetIdx())

    for side, maps in (("reactants", rea_maps), ("products", pro_maps)):
        repeated = [number for number, indices in maps.items() if len(indices) > 1]
        if repeated:
            raise ValueError(f"Atom map number {repeated[0]} appears more than once in the {side}.")
    if (
        all(atom.GetAtomMapNum() for atom in rea.GetAtoms())
        and all(atom.GetAtomMapNum() for atom in pro.GetAtoms())
        and set(rea_maps) != set(pro_maps)
    ):
        raise ValueError("Fully mapped reactants and products must use the same atom map numbers.")

    pins = {}
    for number, p_idx in pro_maps.items():
        r_idx = rea_maps.get(number, [])
        if not r_idx:
            continue
        p, r = p_idx[0], r_idx[0]
        reactant_atom, product_atom = rea.GetAtomWithIdx(r), pro.GetAtomWithIdx(p)
        if (reactant_atom.GetAtomicNum(), reactant_atom.GetIsotope()) != (
            product_atom.GetAtomicNum(),
            product_atom.GetIsotope(),
        ):
            raise ValueError(
                f"Atom map number {number} identifies different element/isotope atoms: "
                f"{reactant_atom.GetIsotope() or ''}{reactant_atom.GetSymbol()} in the reactants but a "
                f"{product_atom.GetIsotope() or ''}{product_atom.GetSymbol()} in the products."
            )
        pins[p] = r
    return pins


def bond_set(mol, order=None):
    """Bonds as a set of sorted index pairs, optionally re-indexed through `order`."""
    bonds = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if order is not None:
            i, j = order[i], order[j]
        bonds.add((i, j) if i < j else (j, i))
    return bonds


def reaction_bonds(rea, pro, mapping):
    """Bonds formed and broken by the reaction, in reactant atom numbering.

    :return: (forming, breaking) -- each a set of sorted index pairs.
    """
    rea_bonds = bond_set(rea)
    pro_bonds = bond_set(pro, order=mapping)
    return pro_bonds - rea_bonds, rea_bonds - pro_bonds


def invert(mapping):
    """`mapping[p] = r` (product -> reactant) inverted to reactant -> product."""
    out = [0] * len(mapping)
    for p, r in enumerate(mapping):
        out[r] = p
    return out


def _validate_bonds(rea, pro, forming, breaking):
    """Require named reactive pairs to agree with the selected endpoint topologies."""
    rea_bonds, pro_bonds = bond_set(rea), bond_set(pro)
    unrealised = [
        (pair, role)
        for role, wanted, realised in (
            ("forming", forming, pro_bonds - rea_bonds),
            ("breaking", breaking, rea_bonds - pro_bonds),
        )
        for pair in sorted(set(wanted or ()) - realised)
    ]
    if not unrealised:
        return
    listed = ", ".join(f"{describe(rea, pair)} ({role})" for pair, role in unrealised)
    raise ValueError(
        f"The selected mapping does not realize the named reaction: {listed}. Pass a complete mapping="
        " or named bonds that match the selected endpoint topologies."
    )


def _validate_mapping(mapping, rea, pro):
    """Require a complete product-to-reactant atom permutation that preserves atom identity."""
    n_atoms = rea.GetNumAtoms()
    if pro.GetNumAtoms() != n_atoms or not isinstance(mapping, list | tuple) or len(mapping) != n_atoms:
        raise ValueError("mapping must be a complete product-to-reactant permutation for both endpoints")
    if any(isinstance(atom, bool) or not isinstance(atom, Integral) for atom in mapping):
        raise ValueError("mapping must contain integer atom indices")
    mapping = [int(atom) for atom in mapping]
    if sorted(mapping) != list(range(n_atoms)):
        raise ValueError("mapping must contain every reactant atom index exactly once")
    pins = _map_pins(rea, pro)
    for product_atom, reactant_atom in enumerate(mapping):
        product_identity = (
            pro.GetAtomWithIdx(product_atom).GetAtomicNum(),
            pro.GetAtomWithIdx(product_atom).GetIsotope(),
        )
        reactant_identity = (
            rea.GetAtomWithIdx(reactant_atom).GetAtomicNum(),
            rea.GetAtomWithIdx(reactant_atom).GetIsotope(),
        )
        if product_identity != reactant_identity:
            raise ValueError(
                f"mapping pairs product atom {product_atom} with a different element or isotope at "
                f"reactant atom {reactant_atom}"
            )
        if product_atom in pins and pins[product_atom] != reactant_atom:
            raise ValueError(
                f"mapping pairs product atom {product_atom} with reactant atom {reactant_atom}, "
                "contradicting their shared atom-map number"
            )
    return mapping


def _normalise_pairs(pairs, name, n_atoms):
    if pairs is None:
        return None
    if not isinstance(pairs, list | tuple):
        raise ValueError(f"{name} must be a list of distinct atom-index pairs")
    normalized = [tuple(sorted(_checked_pair(pair, n_atoms, name))) for pair in pairs]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not repeat an atom pair")
    return normalized


def _select_mapping(rea, pro, mapping, forming, breaking):
    if mapping is not None:
        return _validate_mapping(mapping, rea, pro)

    ranked, tied = map_reaction(rea, pro, breaking_bonds=breaking, forming_bonds=forming)
    distinct = distinct_mechanisms(rea, tied)
    named_forming, named_breaking = set(forming or ()), set(breaking or ())
    if named_forming or named_breaking:
        matches = [
            mechanism
            for mechanism in distinct
            if named_forming <= set(mechanism.forming) and named_breaking <= set(mechanism.breaking)
        ]
        if len(matches) > 1:
            raise ValueError(
                "The named forming_bonds/breaking_bonds still leave multiple distinct mechanisms; "
                "pass mapping= or provide bonds that select one mechanism."
            )
        if matches:
            return _validate_mapping(matches[0].mapping, rea, pro)
    elif len(distinct) > 1:
        raise ValueError(
            f"This reaction has {len(distinct)} distinct mechanisms on connectivity alone. "
            "Pass mapping=, provide forming_bonds/breaking_bonds that uniquely select one mechanism, "
            "or supply atom-mapped reaction SMILES."
        )
    return _validate_mapping((distinct or [ranked])[0].mapping, rea, pro)


def _mechanism_key(rea, mech):
    """Canonical SMILES of the mechanism's ITS graph: kept bonds single, broken double, formed triple.

    Atoms carry element and isotope only, as the mapper reads them, so resonance forms (the two
    carboxylate oxygens, a Kekule ring) are one. Two mechanisms are one reaction relabelled exactly when
    their ITS graphs are isomorphic, so no symmetry group is enumerated or capped. Equal keys are always
    the same graph; RDKit's ranking does not backtrack, so a pathological graph could at worst keep
    equivalent mechanisms apart.
    """
    its = Chem.RWMol()
    for atom in rea.GetAtoms():
        label = Chem.Atom(atom.GetAtomicNum())
        label.SetIsotope(atom.GetIsotope())
        label.SetNoImplicit(True)
        its.AddAtom(label)
    for bond in rea.GetBonds():
        pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        broken = pair in mech.breaking or pair[::-1] in mech.breaking
        its.AddBond(*pair, Chem.BondType.DOUBLE if broken else Chem.BondType.SINGLE)
    for i, j in mech.forming:
        its.AddBond(i, j, Chem.BondType.TRIPLE)
    return Chem.MolToSmiles(its)


def distinct_mechanisms(rea, mechanisms):
    """Merge forming/breaking bond sets related by reactant symmetry, keeping the first of each."""
    unique = {}
    for mech in mechanisms:
        unique.setdefault(_mechanism_key(rea, mech), mech)
    return list(unique.values())


# -- Anchors -------------------------------------------------------------------

# The smallest neighbourhood an anchor must hold over: VeloxChem's assist-id floor (reactionmatcher.py:79).
_ANCHOR_MIN_RADIUS = 3

# Atom-map label marking the centre of a keyed neighbourhood, above any input map number.
_CENTRE = 10**6


def _checked_pair(pair, n_atoms, parameter):
    """Validate one caller-named atom pair."""
    if (
        not isinstance(pair, list | tuple)
        or len(pair) != 2
        or any(isinstance(index, bool) or not isinstance(index, Integral) for index in pair)
        or pair[0] == pair[1]
    ):
        raise ValueError(f"{parameter} must contain distinct integer atom-index pairs")
    i, j = (int(index) for index in pair)
    for index in (i, j):
        if not 0 <= index < n_atoms:
            raise ValueError(
                f"{parameter} names atom index {index}, but this molecule has {n_atoms} atoms, so its "
                f"indices run 0-{n_atoms - 1}. These are REACTANT atom indices, 0-based; if they came "
                "from rxts.mechanisms() they are in ITS numbering, which for a multi-fragment "
                "reaction is not the guesser's -- pass that entry's `smiles` instead."
            )
    return i, j


def _named_bonds(adjacency, breaking_bonds, forming_bonds):
    """Apply caller-named bonds to the reactant's adjacency, as VeloxChem `_prepare_graph` does (211-217).

    Breaking bonds go before forming bonds. A missing breaking bond raises an actionable error; an
    already-present forming bond is logged and changes nothing.
    """
    for i, j in breaking_bonds or ():
        if j not in adjacency[i]:
            raise ValueError(
                f"breaking_bonds names ({i}, {j}), which is not a bond of the reactant, so there is "
                "nothing there to break. These are REACTANT atom indices and the pair must already be "
                "bonded; a pair that is bonded in the PRODUCT belongs in forming_bonds."
            )
        adjacency[i].discard(j)
        adjacency[j].discard(i)
    for i, j in forming_bonds or ():
        if j in adjacency[i]:
            logger.warning(
                f"forming_bonds names ({i}, {j}), which is already a bond of the reactant, so it forms "
                "nothing. The anchoring is unchanged by it. This is almost always wrong numbering -- a "
                "pair that is bonded in the reactant and not in the product belongs in breaking_bonds."
            )
        adjacency[i].add(j)
        adjacency[j].add(i)


def _ball(adjacency, centre, radius):
    """Every atom within `radius` bonds of `centre`."""
    seen = frontier = {centre}
    for _ in range(radius):
        frontier = {q for v in frontier for q in adjacency[v]} - seen
        seen = seen | frontier
    return seen


def _distances(adjacency, sources):
    """Bonds from each atom to the nearest of `sources`; atoms they cannot reach are left out."""
    distance, frontier, layer = dict.fromkeys(sources, 0), set(sources), 0
    while frontier:
        layer += 1
        frontier = {q for v in frontier for q in adjacency[v]} - distance.keys()
        distance |= dict.fromkeys(frontier, layer)
    return distance


def _depth(adjacency, centre):
    """Bonds from `centre` to the farthest atom of its molecule."""
    depth, seen, frontier = 0, {centre}, {centre}
    while frontier := {q for v in frontier for q in adjacency[v]} - seen:
        seen |= frontier
        depth += 1
    return depth


def _skeleton(mol, adjacency, hydrogens=None):
    """`mol` as a bare graph to key: element and isotope, `adjacency`'s bonds all single, and no H beyond
    the folded `hydrogens` counts given.

    Nothing is sanitised, so valence, aromaticity and charge -- where RDKit is fragile for metals and
    boron -- never enter. An anchored atom carries its reactant index + 1 as an atom-map label. Returned
    with each bond's two atoms, in bond-index order.
    """
    skeleton, ends = Chem.RWMol(), []
    for atom in mol.GetAtoms():
        bare = Chem.Atom(atom.GetAtomicNum())
        bare.SetIsotope(atom.GetIsotope())
        bare.SetNoImplicit(True)
        bare.SetNumExplicitHs(hydrogens[atom.GetIdx()] if hydrogens else 0)
        skeleton.AddAtom(bare)
    for i, neighbours in enumerate(adjacency):
        for j in neighbours:
            if i < j:
                skeleton.AddBond(i, j, Chem.BondType.SINGLE)
                ends.append((i, j))
    return skeleton, ends


def _canonical(skeleton, atoms, centre=None, bare=()):
    """Canonical SMILES of the subgraph on `atoms` (`centre` marked), and the atoms in output order.

    `bare` atoms are written without their hydrogen counts: on the rim of a ball, where a hydrogen would lie
    one bond outside it, so a key reads the same whether hydrogens are atoms or counts.

    The subgraph is copied out first (PathToSubmol): RDKit's MolFragmentToSmiles breaks ties using the
    rest of the molecule, so the same labelled fragment can be written two ways as anchors land
    elsewhere. A copy makes the string a function of the subgraph alone; equal strings are the same
    rooted graph.
    """
    mol, ends = skeleton
    inside = set(atoms)
    bonds = [k for k, (a, b) in enumerate(ends) if a in inside and b in inside]
    new = {}
    part = Chem.RWMol(Chem.PathToSubmol(mol, bonds, atomMap=new)) if bonds else Chem.RWMol()
    for atom in sorted(inside - new.keys()):  # atoms with no bond inside: a lone ion, a folded water
        new[atom] = part.AddAtom(mol.GetAtomWithIdx(atom))
    if centre is not None:
        part.GetAtomWithIdx(new[centre]).SetAtomMapNum(_CENTRE)
    for atom in bare:
        part.GetAtomWithIdx(new[atom]).SetNumExplicitHs(0)
    smiles = Chem.MolToSmiles(part, allHsExplicit=True)  # else a bare skeleton writes no H as the default count
    old = {i: a for a, i in new.items()}
    return smiles, [old[i] for i in part.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]]


def _rooted(skeleton, centre, atoms, bare=()):
    """Canonical SMILES of `atoms` with `centre` marked: equal strings are the same rooted graph."""
    return _canonical(skeleton, atoms, centre, bare)[0]


def _anchors(rea, pro, pins, breaking_bonds=None, forming_bonds=None, counts=None):
    """Pair the atoms whose surroundings show them untouched, so the search maps only the reaction.

    A heavy atom's key is the canonical SMILES of everything within `radius` bonds of it, with it
    marked and earlier anchors labelled. A key held by one atom on each side anchors that pair. Only
    when no key is unique does a shared key anchor a pair, and only if both sides hold it equally often
    and one side's holders are interchangeable -- the whole molecule rooted at each reads the same -- so
    a symmetry carries any choice to any other;
    one such pair at a time, since it relabels what the next depends on. The radius runs from half
    the widest reactant down to `_ANCHOR_MIN_RADIUS`, each repeated until nothing new anchors; atom
    order plays no part. An anchored atom's hydrogens pair by isotope and index: they are
    interchangeable. Like VeloxChem's assist ids, this reads unchanged surroundings as an unchanged
    atom. The explicit `pins` label from the start, so no anchor contradicts them, and named bonds
    edit the reactant first. Given `counts` (each atom's terminal hydrogens, reactant then product), the
    keys read hydrogens as those counts, not as atoms, and no hydrogen anchors.

    Returns product atom -> (reactant atom, radius it was fixed at): the pins, at radius 0, and every
    anchor.
    """
    mols = (rea, pro)
    adjacency = [_adjacency(rea), _adjacency(pro)]
    _named_bonds(adjacency[0], breaking_bonds, forming_bonds)
    radius = (max((_depth(adjacency[0], a) for a in range(rea.GetNumAtoms())), default=0) + 1) // 2 + 1
    if counts:  # hydrogens are counts on their atoms, never atoms of a ball
        for mol, adj in zip(mols, adjacency, strict=True):
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    h = atom.GetIdx()
                    for q in adj[h]:
                        adj[q].discard(h)
                    adj[h] = set()
    skeletons = [_skeleton(*parts) for parts in zip(mols, adjacency, counts or (None, None), strict=True)]
    anchored, used = {}, set()  # product atom -> (reactant atom, radius); the reactant atoms taken
    keys = ({}, {})  # side -> {atom: (ball size, key)}, dropped when an anchor lands within reach

    def label(p, r, radius):
        anchored[p] = r, radius
        used.add(r)
        for side, atom in ((0, r), (1, p)):
            skeletons[side][0].GetAtomWithIdx(atom).SetAtomMapNum(r + 1)
            for near in _ball(adjacency[side], atom, radius):  # every key whose ball holds it, now or later
                keys[side].pop(near, None)

    def hydrogens(side, atom):
        mol = mols[side]
        found = (q for q in adjacency[side][atom] if mol.GetAtomWithIdx(q).GetAtomicNum() == 1)
        return sorted(found, key=lambda q: (mol.GetAtomWithIdx(q).GetIsotope(), q))

    def anchor(p, r, radius):
        label(p, r, radius)
        rh, ph = hydrogens(0, r), hydrogens(1, p)
        if len(rh) == len(ph):
            for a, b in zip(rh, ph, strict=True):
                if a not in used and b not in anchored:
                    label(b, a, radius)

    def holders(side, radius):
        found = {}
        for atom in range(mols[side].GetNumAtoms()):
            if atom in (used if side == 0 else anchored) or mols[side].GetAtomWithIdx(atom).GetAtomicNum() == 1:
                continue
            ball = _ball(adjacency[side], atom, radius)
            rim = ball - _ball(adjacency[side], atom, radius - 1)
            size, key = keys[side].get(atom, (None, None))
            if size != (len(ball), len(rim)):  # balls nest as the radius shrinks: same sizes, same ball and rim
                key = _rooted(skeletons[side], atom, ball, rim)
                keys[side][atom] = (len(ball), len(rim)), key
            found.setdefault(key, []).append(atom)
        return found

    def interchangeable(side, atoms):
        everything = range(mols[side].GetNumAtoms())
        return len({_rooted(skeletons[side], atom, everything) for atom in atoms}) == 1

    for p, r in pins.items():
        label(p, r, 0)
    while radius >= _ANCHOR_MIN_RADIUS:
        while True:
            found = holders(0, radius), holders(1, radius)
            shared = [(found[1].get(key, []), atoms) for key, atoms in found[0].items()]
            pairs = [(ps[0], rs[0]) for ps, rs in shared if len(rs) == len(ps) == 1]
            if not pairs:
                pairs = next(
                    (
                        [(ps[0], rs[0])]
                        for ps, rs in shared
                        if len(rs) == len(ps) > 1 and (interchangeable(0, rs) or interchangeable(1, ps))
                    ),
                    [],
                )
            if not pairs:
                break
            for p, r in pairs:
                anchor(p, r, radius)
        radius -= 1
    return anchored


# SMILES reporting and proton relays extend VeloxChem's molecule-based mapper.


def _relay_sites(mol, atoms, label, *, maps=False):
    sites = _protic_sites(mol, atoms)
    if not sites:
        raise ValueError(f"{label}: each relay molecule needs a protic site, a heteroatom carrying hydrogen")
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False, includeAtomMaps=maps)
    if len({ranks[i] for i in sites}) > 1:
        remedy = (
            "include it in the reaction and select an atom with relay_map="
            if label.startswith("add_relay=")
            else "select one with relay_map="
        )
        raise ValueError(f"{label}: relay molecule has inequivalent protic sites; {remedy}")
    return sites[0]


def _add_relay(rea, pro, mapping, bridge_smiles):
    """Add the declared relay molecules once, then route every transferred proton through them."""
    bridge = _from_smiles(bridge_smiles, "add_relay")
    fragments = [set(fragment) for fragment in Chem.GetMolFrags(bridge)]
    donors = [_relay_sites(bridge, fragment, f"add_relay={bridge_smiles!r}", maps=True) for fragment in fragments]
    offset = rea.GetNumAtoms()
    rea2, pro2 = Chem.CombineMols(rea, bridge), Chem.CombineMols(pro, bridge)
    new_map = list(mapping) + list(range(offset, offset + bridge.GetNumAtoms()))
    return _route_relay(rea2, pro2, new_map, [offset + donor for donor in donors], f"add_relay={bridge_smiles!r}")


def _proton_transfers(rea, pro, mapping):
    """Return direct H transfers as (reactant H, reactant acceptor) pairs."""
    forming, breaking = reaction_bonds(rea, pro, mapping)

    def acceptor_of(pairs):
        return {i: j for a, b in pairs for i, j in ((a, b), (b, a)) if rea.GetAtomWithIdx(i).GetAtomicNum() == 1}

    leaving, arriving = acceptor_of(breaking), acceptor_of(forming)
    return sorted((h, arriving[h]) for h in leaving if h in arriving)


def _route_relay(rea, pro, mapping, sites, label):
    """Route direct proton transfers through the selected sites without adding atoms."""
    transfers = _proton_transfers(rea, pro, mapping)
    if not transfers:
        return None
    hydrogens = []
    for site in sites:
        attached = sorted(
            (n.GetIdx() for n in rea.GetAtomWithIdx(site).GetNeighbors() if n.GetAtomicNum() == 1),
            key=lambda i: (rea.GetAtomWithIdx(i).GetAtomMapNum() == 0, rea.GetAtomWithIdx(i).GetAtomMapNum() or i),
        )
        if len(attached) < len(transfers):
            raise ValueError(
                f"{label}: the reaction transfers {len(transfers)} proton(s); "
                f"each relay site needs {len(transfers)} attached hydrogens, but one has {len(attached)}"
            )
        hydrogens.append(attached)

    r2p = invert(mapping)
    mapping = list(mapping)
    for transfer_index, (proton, _acceptor) in enumerate(transfers):
        arriving = r2p[proton]
        for attached in hydrogens:
            spare = r2p[attached[transfer_index]]
            mapping[arriving], mapping[spare] = mapping[spare], mapping[arriving]
    product = Chem.Mol(pro)
    for p, r in enumerate(mapping):
        product.GetAtomWithIdx(p).SetAtomMapNum(rea.GetAtomWithIdx(r).GetAtomMapNum())
    return rea, product, _mechanism(rea, product, mapping)


def _protic_sites(mol, atoms=None):
    atoms = range(mol.GetNumAtoms()) if atoms is None else sorted(atoms)
    return [
        i
        for i in atoms
        if mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)
        and any(n.GetAtomicNum() == 1 for n in mol.GetAtomWithIdx(i).GetNeighbors())
    ]


def _component_smiles(mol, atoms):
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolFragmentToSmiles(work, atomsToUse=sorted(atoms), canonical=True, isomericSmiles=True)


def _unchanged_component(rea, pro, mapping, reactant_atoms):
    """Return a component's mapped product atoms if its atom and bond state is unchanged."""
    r2p = invert(mapping)
    product_atoms = {r2p[i] for i in reactant_atoms}
    if product_atoms not in (set(fragment) for fragment in Chem.GetMolFrags(pro)):
        return None, "whose identity changes in the reaction"

    rea_stereo, pro_stereo = Chem.Mol(rea), Chem.Mol(pro)
    Chem.AssignStereochemistry(rea_stereo, cleanIt=True, force=True)
    Chem.AssignStereochemistry(pro_stereo, cleanIt=True, force=True)

    def atom_state(atom, stereo_atom):
        return (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
            stereo_atom.GetProp("_CIPCode") if stereo_atom.HasProp("_CIPCode") else None,
        )

    for ri in reactant_atoms:
        pi = r2p[ri]
        if atom_state(rea.GetAtomWithIdx(ri), rea_stereo.GetAtomWithIdx(ri)) != atom_state(
            pro.GetAtomWithIdx(pi), pro_stereo.GetAtomWithIdx(pi)
        ):
            return None, "with changed atom state"

    def bond_state(mol, component, to_reactant):
        result = set()
        directed = {Chem.BondType.DATIVE, Chem.BondType.DATIVEONE, Chem.BondType.DATIVEL, Chem.BondType.DATIVER}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in component and j in component:
                i, j = to_reactant[i], to_reactant[j]
                endpoints = (i, j) if bond.GetBondType() in directed else (min(i, j), max(i, j))
                result.add((*endpoints, bond.GetBondType(), bond.GetIsAromatic(), bond.GetStereo()))
        return result

    if bond_state(rea, reactant_atoms, {i: i for i in reactant_atoms}) != bond_state(pro, product_atoms, mapping):
        return None, "with changing bonds"
    return product_atoms, None


def _existing_relay(rea, pro, mapping, relay_rs, *, label, relay_ps=None):
    r2p = invert(mapping)
    relay_ps = [r2p[i] for i in relay_rs] if relay_ps is None else relay_ps
    if len(relay_rs) != len(relay_ps) or len(set(relay_rs)) != len(relay_rs):
        raise ValueError(f"{label} must identify distinct relay atoms in reaction-map order")
    components = [
        set(next(fragment for fragment in Chem.GetMolFrags(rea) if relay_r in fragment)) for relay_r in relay_rs
    ]
    for relay_r, relay_p, r_component in zip(relay_rs, relay_ps, components, strict=True):
        if r2p[relay_r] != relay_p:
            raise ValueError(f"{label} does not identify the same atom in the reaction mapping")
        _product_component, error = _unchanged_component(rea, pro, mapping, r_component)
        if error:
            raise ValueError(f"{label} selects an active component {error}")
        r_atom, p_atom = rea.GetAtomWithIdx(relay_r), pro.GetAtomWithIdx(relay_p)
        if r_atom.GetAtomicNum() != p_atom.GetAtomicNum():
            raise ValueError(f"{label} changes element between reactants and products")
        if r_atom.GetAtomicNum() in (1, 6):
            raise ValueError(f"{label} must select protic heteroatoms")
        if not any(n.GetAtomicNum() == 1 for n in r_atom.GetNeighbors()) or not any(
            n.GetAtomicNum() == 1 for n in p_atom.GetNeighbors()
        ):
            raise ValueError(f"{label} must select heteroatoms carrying hydrogen on both sides")

    return _route_relay(rea, pro, mapping, relay_rs, label)


def _relay_component(rea, pro, mapping, component_smiles):
    """Select a unique unchanged component by full, map-insensitive isomeric SMILES."""
    query = _from_smiles(component_smiles, "relay")
    query_fragments = [set(fragment) for fragment in Chem.GetMolFrags(query)]
    if not _proton_transfers(rea, pro, mapping):
        return None
    wanted = [_component_smiles(query, fragment) for fragment in query_fragments]
    candidates = {}
    for fragment in Chem.GetMolFrags(rea):
        component = set(fragment)
        key = _component_smiles(rea, component)
        if key in wanted and _unchanged_component(rea, pro, mapping, component)[0] is not None:
            candidates.setdefault(key, []).append(component)
    chosen = {}
    for key in dict.fromkeys(wanted):
        requested, available = wanted.count(key), candidates.get(key, [])
        if len(available) < requested:
            raise ValueError(
                f"relay={component_smiles!r} does not match an unchanged component pool: "
                f"requested {requested} {key} component(s), found {len(available)}"
            )
        if len(available) > requested:
            raise ValueError(
                f"relay={component_smiles!r} matches multiple unchanged components beyond the requested pool; "
                "select atoms with relay_map="
            )
        chosen[key] = iter(available)

    relay_rs = []
    for key in wanted:
        atoms = next(chosen[key])
        relay_rs.append(_relay_sites(rea, atoms, f"relay={component_smiles!r}"))
    return _existing_relay(rea, pro, mapping, relay_rs, label=f"relay={component_smiles!r}")


def _relay_map(rea, pro, mapping, relay_map):
    """Route proton transfers through an existing, unchanged, mapped protic atom."""
    relay_maps = relay_map if isinstance(relay_map, (list, tuple)) else [relay_map]
    if not relay_maps or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in relay_maps
    ):
        raise ValueError("relay_map must be a positive atom-map number or a nonempty ordered list of them")
    if len(set(relay_maps)) != len(relay_maps):
        raise ValueError("relay_map entries must be distinct")

    def locate(mol, number, side):
        found = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == number]
        if len(found) != 1:
            raise ValueError(f"relay_map={number} must appear exactly once on each side; {side} has {len(found)}")
        return found[0]

    relay_rs = [locate(rea, number, "reactants") for number in relay_maps]
    relay_ps = [locate(pro, number, "products") for number in relay_maps]
    return _existing_relay(rea, pro, mapping, relay_rs, relay_ps=relay_ps, label=f"relay_map={relay_map}")


def map_smiles(
    reaction: str,
    hydrogens: bool = False,
    strict: bool = True,
    mechanism: int | None = None,
) -> str:
    """Return atom-mapped reaction SMILES for a selected mechanism.

    hydrogens=True prints explicit mapped H atoms. strict=True requires an explicit mechanism
    selection for ambiguous reactions; strict=False accepts the first ranked choice."""
    rea, pro, chosen, distinct = _analyse(reaction)
    if mechanism is not None and (isinstance(mechanism, bool) or not isinstance(mechanism, Integral) or mechanism < 0):
        raise ValueError("mechanism must be a non-negative integer selection")
    if mechanism is None and strict and len(distinct) > 1:
        raise ValueError(
            f"This reaction is ambiguous on connectivity alone -- {len(distinct)} distinct "
            "mechanisms change the same number of bonds. Call rxts.mechanisms() to see them, "
            "map the atoms you care about in the input SMILES, or pass mechanism=n to choose."
        )
    if mechanism is not None:
        if mechanism >= len(distinct):
            raise ValueError(
                f"This reaction has {len(distinct)} distinct mechanism(s); there is no mechanism "
                f"{mechanism}. Call rxts.mechanisms() to see them."
            )
        chosen = distinct[mechanism]

    smiles, _ = _mapped_smiles(rea=rea, pro=pro, chosen=chosen, hydrogens=hydrogens)
    return smiles


def _analyse(reaction):
    """Parse and enumerate unconstrained mapping choices once.

    Return reactant, product, chosen mechanism and distinct mechanisms.
    """
    split = split_reaction_smiles(reaction)
    if split is None:
        raise ValueError(
            f"{reaction!r} has no '>>', so it names one molecule and not a reaction. There is nothing "
            "to map a reactant onto. A reaction SMILES is spelled 'reactants>>products', e.g. "
            "'CC(=O)OCC.O>>CC(=O)O.OCC'."
        )
    reactant_smiles, product_smiles = split
    rea = _from_smiles(reactant_smiles, "reactants")
    pro = _from_smiles(product_smiles, "products")
    chosen, tied = map_reaction(rea, pro)
    return rea, pro, chosen, distinct_mechanisms(rea, tied)


def _mapped_smiles(rea, pro, chosen, hydrogens=False, preserve_maps=False):
    """Return mapped reaction SMILES and map labels in the original reactant order."""
    rea, pro = Chem.Mol(rea), Chem.Mol(pro)
    if preserve_maps:
        rea_maps = {}
        pro_maps = {}
        for atom in rea.GetAtoms():
            if atom.GetAtomMapNum():
                rea_maps.setdefault(atom.GetAtomMapNum(), []).append(atom.GetIdx())
        for atom in pro.GetAtoms():
            if atom.GetAtomMapNum():
                pro_maps.setdefault(atom.GetAtomMapNum(), []).append(atom.GetIdx())
        labels = [0] * rea.GetNumAtoms()
        r2p = invert(chosen.mapping)
        for number, reactant_indices in rea_maps.items():
            product_indices = pro_maps.get(number, [])
            if len(reactant_indices) == len(product_indices) == 1:
                r = reactant_indices[0]
                p = r2p[r]
                if p in product_indices:
                    labels[r] = number
        next_label = max(labels, default=0) + 1
        for i in range(len(labels)):
            if labels[i] == 0:
                labels[i] = next_label
                next_label += 1
    else:
        labels = list(range(1, rea.GetNumAtoms() + 1))
    for i, atom in enumerate(rea.GetAtoms()):
        atom.SetAtomMapNum(labels[i])
    for p, atom in enumerate(pro.GetAtoms()):
        atom.SetAtomMapNum(labels[chosen.mapping[p]])  # product atom p is reactant atom mapping[p]

    if not hydrogens:
        # RemoveHs keeps the heavy atoms' map numbers; it only stops the hydrogens being written.
        rea, pro = Chem.RemoveHs(rea), Chem.RemoveHs(pro)

    smiles = f"{rxembed.cxsmiles(rea)}>>{rxembed.cxsmiles(pro)}"
    return smiles, labels


def _report(rea, pro, mech, preserve_maps=False):
    """Report mapped SMILES, reactive pairs and atom-labelled bond changes for one mechanism."""
    smiles, labels = _mapped_smiles(rea, pro, mech, hydrogens=True, preserve_maps=preserve_maps)
    serialized = parse_smiles(smiles.split(">>", 1)[0])
    serialized_indices = {atom.GetAtomMapNum(): atom.GetIdx() for atom in serialized.GetAtoms()}
    source_to_report = {index: serialized_indices[label] for index, label in enumerate(labels)}

    def remap(pairs):
        return sorted(tuple(sorted(source_to_report[index] for index in pair)) for pair in pairs)

    forming, breaking = remap(mech.forming), remap(mech.breaking)
    return {
        "smiles": smiles,
        "forming": forming,
        "breaking": breaking,
        "forming_labels": [describe(serialized, pair) for pair in forming],
        "breaking_labels": [describe(serialized, pair) for pair in breaking],
    }


def mechanisms(
    reaction: str | Path | Chem.Mol,
    product: str | Path | Chem.Mol | None = None,
    add_relay: str | None = None,
    *,
    charge: int | None = None,
    relay: str | None = None,
    relay_map: int | list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """List mapped mechanisms with forming/breaking pairs and readable bond labels.

    Returned pairs use the canonical reactant order in the mapped SMILES, including explicit H.
    `add_relay` adds molecules on both sides; `relay` selects existing unchanged components;
    `relay_map` selects existing mapped protic atoms in relay-chain order."""
    return [
        _report(
            rea,
            pro,
            mech,
            preserve_maps=relay is not None or relay_map is not None,
        )
        for rea, pro, mech in _mechanism_mols(
            reaction, product, add_relay, charge=charge, relay=relay, relay_map=relay_map
        )
    ]


def _mechanism_mols(reaction, product=None, add_relay=None, *, charge=None, relay=None, relay_map=None):
    """Yield endpoint graphs and distinct mechanisms for mapping and drawing.

    Two supplied endpoints are used as given; relays require reaction SMILES and are deduplicated by graph.
    """
    if sum(value is not None for value in (add_relay, relay, relay_map)) > 1:
        raise ValueError("add_relay=, relay= and relay_map= are mutually exclusive")
    if product is not None:
        if any(value is not None for value in (add_relay, relay, relay_map)):
            raise ValueError(
                "add_relay= / relay= / relay_map= needs a reaction SMILES; two structures are mapped as given."
            )
        endpoints = []
        for source in (reaction, product):
            if isinstance(source, Chem.Mol):
                endpoints.append(source)
            elif isinstance(source, Path) or is_xyz_file(source) or "\n" in source:
                if charge is None:
                    raise ValueError(
                        "XYZ endpoint mapping needs charge=...; coordinates do not declare electronic state"
                    )
                endpoints.append(mol_from_xyz(source, charge=charge))
            else:
                endpoints.append(parse_smiles(source))
        rea, pro = endpoints
        _, tied = map_reaction(rea, pro)
        for mech in distinct_mechanisms(rea, tied):
            yield rea, pro, mech
        return

    if not isinstance(reaction, str):
        raise ValueError(
            f"a {type(reaction).__name__} was given as the reactant with no `product=`, and a "
            "structure on its own does not say what it reacts to. Pass the product structure as the "
            "second argument, or give the whole reaction as one reaction SMILES "
            "('CC(=O)OCC.O>>CC(=O)O.OCC')."
        )

    rea, pro, _, distinct = _analyse(reaction)
    if all(value is None for value in (add_relay, relay, relay_map)):
        for mech in distinct:
            yield rea, pro, mech
        return

    seen = set()
    for mech in distinct:
        if add_relay is not None:
            result = _add_relay(rea=rea, pro=pro, mapping=mech.mapping, bridge_smiles=add_relay)
        elif relay is not None:
            result = _relay_component(rea=rea, pro=pro, mapping=mech.mapping, component_smiles=relay)
        else:
            result = _relay_map(rea=rea, pro=pro, mapping=mech.mapping, relay_map=relay_map)
        if result is None:
            continue
        rea_b, pro_b, relayed = result
        smiles, _ = _mapped_smiles(rea_b, pro_b, relayed)
        if smiles not in seen:
            seen.add(smiles)
            yield rea_b, pro_b, relayed
    if not seen:
        route = (
            f"add_relay={add_relay!r}"
            if add_relay is not None
            else f"relay={relay!r}"
            if relay is not None
            else f"relay_map={relay_map}"
        )
        raise ValueError(f"{route}: none of this reaction's mechanisms move a proton, so there is nothing to relay.")
