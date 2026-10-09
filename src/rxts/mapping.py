"""Map reaction mechanisms with VeloxChem's assist-id pre-solve and rxts search.

VeloxChem's ``ReactionMatcher`` supplies the graph pre-solve (``_calculate_assist_ids``,
``reactionmatcher.py:346-503``). rxts replaces its incremental breaking-edge search with
branch-and-bound over bijections (:func:`_optimum`, :func:`_search`), enumerates tied mechanisms,
and ranks them by the pericyclic array. Naming bonds affects pre-solve pins; it does not constrain
that search. Mechanism selection among ties happens in the caller.

SMILES parsing, graph mapping and relay construction are independent of endpoint embedding and FF scanning.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from itertools import permutations, product
from numbers import Integral
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit import Chem

from rxts.utils import describe, is_xyz_file, mol_from_xyz

logger = logging.getLogger(__name__)

# Bounded so a pathological input degrades instead of hanging.
_MAX_NODES = 2_000_000
_MAX_OPTIMA = 20_000

# Explicit hydrogens can make symmetry enumeration factorial. Truncation leaves mechanism
# equivalence unestablished, so report it rather than silently merging or refusing the reaction.
_MAX_AUTOMORPHISMS = 5000

# Ceiling on changing bonds: where `_optimum`'s ladder stops and reports no single step.
_MAX_CHANGING = 12

# Bound the factorial ring-ranking search for large changed-bond sets.
_MAX_ARRAY_BONDS = 6

# Graph-ranking heuristic favours six-atom reaction circuits.
_HUCKEL_ARRAY = 6


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
    params = Chem.SmilesParserParams()
    params.removeHs = False  # Retain mapped transferring protons.
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"Could not parse SMILES {smiles!r}")
    return Chem.AddHs(mol)


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
    reaction legitimately differs between the two sides. Not the assist ids, which pin one atom to one
    atom where the neighbourhoods are provably isomorphic; ordering never fixes an atom.
    """
    env = [(a.GetAtomicNum(),) for a in mol.GetAtoms()]
    for _ in range(radius):
        env = [env[a.GetIdx()] + tuple(sorted(env[n.GetIdx()] for n in a.GetNeighbors())) for a in mol.GetAtoms()]
    return env


def map_reaction(
    rea: Chem.Mol,
    pro: Chem.Mol,
    breaking_bonds: list | None = None,
    forming_bonds: list | None = None,
) -> tuple[Mechanism, list[Mechanism]]:
    """Return the best product-to-reactant mapping and all tied mechanisms.

    Named bonds use reactant indices and reach the VeloxChem assist-id pre-solve only, following
    `get_mapping` and `_prepare_graph` in reactionmatcher.py. They guide the initial atom pins;
    the branch-and-bound search still minimises bond changes.

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

    # Pin the atoms the search need not guess, each as a one-element candidate list. Explicit maps
    # override assist ids, including dependent hydrogen assignments: keeping a hydrogen paired to the
    # parent an overridden assist pin chose creates a hybrid mapping that respects neither pin set.
    explicit_pro = {r: p for p, r in pins.items()}
    for p, r in _assist_pins(rea, pro, breaking_bonds, forming_bonds).items():
        if rea_identity[r] != pro_identity[p]:
            continue
        if (p in pins and pins[p] != r) or (r in explicit_pro and explicit_pro[r] != p):
            continue
        if pro.GetAtomWithIdx(p).GetAtomicNum() == 1:
            pro_parent = next((n.GetIdx() for n in pro.GetAtomWithIdx(p).GetNeighbors() if n.GetAtomicNum() != 1), None)
            rea_parent = next((n.GetIdx() for n in rea.GetAtomWithIdx(r).GetNeighbors() if n.GetAtomicNum() != 1), None)
            if (
                pro_parent is not None
                and rea_parent is not None
                and pro_identity[pro_parent] != rea_identity[rea_parent]
            ):
                continue
            if (pro_parent is not None and pro_parent in pins and pins[pro_parent] != rea_parent) or (
                rea_parent is not None and rea_parent in explicit_pro and explicit_pro[rea_parent] != pro_parent
            ):
                continue
        candidates[p] = [r]
    for p, r in pins.items():
        candidates[p] = [r]

    order = sorted(range(n), key=lambda p: (len(candidates[p]), -len(pro_adj[p]), -pro_elem[p]))

    # Pass 1: the fewest bonds any mapping can change. Pass 2: every mapping
    # that achieves it, so the tie-break has something to choose between.
    optimum = _optimum(n=n, order=order, candidates=candidates, rea_adj=rea_adj, pro_adj=pro_adj)
    optimal = _search(
        n=n, order=order, candidates=candidates, rea_adj=rea_adj, pro_adj=pro_adj, bound=optimum, collect=True
    )

    # The reactant's bonds do not depend on the mapping, so build this set once, not once per mapping.
    rea_bonds = bond_set(rea)
    seen, mechanisms = set(), []
    for m in optimal:
        pro_bonds = bond_set(pro, order=m)
        forming, breaking = frozenset(pro_bonds - rea_bonds), frozenset(rea_bonds - pro_bonds)
        key = (forming, breaking)
        if key not in seen:
            seen.add(key)
            mechanisms.append(Mechanism(list(m), forming, breaking))

    def rank(mech):
        # Rank by distance from a six-atom pericyclic array, then atoms touched, then hydrogen bonds
        # changed; sorted bonds make the final tie-break deterministic. The array is only a proxy:
        # bond orders and thermal/photochemical context are unavailable here.
        atoms = len({a for pair in mech.changing for a in pair})
        h_bonds = sum(1 for i, j in mech.changing if rea_elem[i] == 1 or rea_elem[j] == 1)
        off_huckel = abs(_pericyclic_array(mech, rea_bonds) - _HUCKEL_ARRAY)
        return (off_huckel, atoms, -h_bonds, sorted(mech.forming), sorted(mech.breaking))

    mechanisms.sort(key=rank)
    return mechanisms[0], mechanisms


def _optimum(n, order, candidates, rea_adj, pro_adj):
    """Find the fewest bonds any mapping can change, by raising the ceiling until one fits.

    A ceiling of k prunes any partial mapping that has already changed k bonds; real reactions change
    few, so starting low and raising prunes hard where no ceiling would (the incremental cost is a
    lower bound that says nothing near the root). Upstream's `_find_breaking_edges` deepens the same
    way, one edge at a time.
    """
    for bound in range(1, _MAX_CHANGING + 2):
        cost = _search(
            n=n, order=order, candidates=candidates, rea_adj=rea_adj, pro_adj=pro_adj, bound=bound, collect=False
        )
        if cost < bound:
            return cost
    raise ValueError(
        f"No mapping between these reactants and products changes fewer than "
        f"{_MAX_CHANGING} bonds, so this does not look like one reaction step. "
        "Check that the product really is the product; if it is, map the atoms "
        "you care about in the SMILES, or pass forming_bonds/breaking_bonds."
    )


def _search(n, order, candidates, rea_adj, pro_adj, bound, collect):
    """Branch-and-bound over element-preserving bijections, never exceeding `bound`.

    With `collect` false, returns the fewest changing bonds achievable, or `bound`
    if nothing beats it. With it true, `bound` is that number and every mapping
    achieving it is returned.
    """
    mapping = [None] * n
    inverse = [None] * n
    best, nodes = bound, 0  # the incumbent: the ceiling until a cheaper complete mapping beats it
    found = []

    def delta(p, r):
        """Bonds changed by committing product atom `p` to reactant `r`.

        Only pairs with both ends already assigned are counted, which keeps this a lower bound on the
        final cost, and so a sound basis for pruning.
        """
        formed = sum(1 for q in pro_adj[p] if mapping[q] is not None and mapping[q] not in rea_adj[r])
        broken = sum(1 for s in rea_adj[r] if inverse[s] is not None and inverse[s] not in pro_adj[p])
        return formed + broken

    def step(depth, cost):
        nonlocal best, nodes
        # Node budget: truncating quietly would return a first-leaf bound indistinguishable from an answer.
        if nodes > _MAX_NODES:
            raise _ExhaustedError
        nodes += 1

        if depth == n:
            if collect:
                found.append(list(mapping))
            elif cost < best:
                best = cost
            return

        p = order[depth]
        for r in candidates[p]:
            if inverse[r] is not None:
                continue
            d = cost + delta(p, r)
            # Ties are pruned while hunting the optimum, kept while enumerating.
            if (d > best) if collect else (d >= best):
                continue
            mapping[p], inverse[r] = r, p
            step(depth + 1, d)
            mapping[p], inverse[r] = None, None
            # Enumerating ties is bounded: the optimum is proven, so at worst a tied alternative is unlisted.
            if collect and len(found) > _MAX_OPTIMA:
                return

    try:
        step(0, 0)
    except _ExhaustedError:
        raise ValueError(
            "Could not map these reactants onto these products within the search "
            "budget. Map the atoms you care about in the SMILES, or pass "
            "forming_bonds/breaking_bonds, and the search will respect them."
        ) from None
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


def _automorphisms(mol):
    """Return reactant automorphisms and whether RDKit reached the search cap.

    RDKit cannot distinguish an exact-size group from truncation. Incomplete groups can leave
    equivalent mechanisms separate, but cannot merge chemically different mechanisms.
    """
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=_MAX_AUTOMORPHISMS)
    truncated = len(matches) >= _MAX_AUTOMORPHISMS
    if not matches:
        matches = [tuple(range(mol.GetNumAtoms()))]
    return matches, truncated


def distinct_mechanisms(rea, mechanisms):
    """Merge forming/breaking bond sets related by reactant symmetry."""
    mechanisms = list(mechanisms)
    if len(mechanisms) < 2:
        return mechanisms
    automorphisms, truncated = _automorphisms(rea)

    def canonical(mech):
        forms = []
        for sigma in automorphisms:
            f = frozenset(tuple(sorted((sigma[i], sigma[j]))) for i, j in mech.forming)
            b = frozenset(tuple(sorted((sigma[i], sigma[j]))) for i, j in mech.breaking)
            forms.append((sorted(f), sorted(b)))
        return min(forms)  # the same label whichever member of the orbit we were handed

    seen, out = set(), []
    for mech in mechanisms:
        key = str(canonical(mech))
        if key not in seen:
            seen.add(key)
            out.append(mech)

    if truncated and len(out) > 1:
        logger.warning(
            f"This reactant has at least {_MAX_AUTOMORPHISMS} automorphisms, so the symmetry quotient "
            f"is incomplete: the {len(out)} mechanism(s) reported are an UPPER BOUND and may include "
            "duplicates of the same reaction relabelled. Pin the one you want with its `smiles` and "
            "never its index pairs, or pass forming_bonds/breaking_bonds -- those are threaded into "
            "the mapper's pre-solve and usually collapse the tie outright."
        )
    return out


# -- VeloxChem assist-id pre-solve ------------------------------------------

# Upstream's floor on the neighbourhood radius an assist-id pin must hold over (reactionmatcher.py:79).
_ASSIST_MIN_DEPTH = 3


def _checked_pair(pair, n_atoms, parameter):
    """Validate pairs before NetworkX can create unintended nodes."""
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


def _apply_reaction_bonds(graph, n_atoms, breaking_bonds, forming_bonds):
    """Apply caller-named bonds to the reactant graph as VeloxChem `_prepare_graph` does (211-217).

    Breaking edges are removed before forming edges are added. A missing breaking edge raises an
    actionable error; an already-present forming edge is logged, matching NetworkX's idempotent add.
    """
    for pair in breaking_bonds or ():
        i, j = _checked_pair(pair, n_atoms, "breaking_bonds")
        if not graph.has_edge(i, j):
            raise ValueError(
                f"breaking_bonds names ({i}, {j}), which is not a bond of the reactant, so there is "
                "nothing there to break. These are REACTANT atom indices and the pair must already be "
                "bonded; a pair that is bonded in the PRODUCT belongs in forming_bonds."
            )
        graph.remove_edge(i, j)
    for pair in forming_bonds or ():
        i, j = _checked_pair(pair, n_atoms, "forming_bonds")
        if graph.has_edge(i, j):
            logger.warning(
                f"forming_bonds names ({i}, {j}), which is already a bond of the reactant, so it forms "
                "nothing. The pre-solve is unchanged by it. This is almost always wrong numbering -- a "
                "pair that is bonded in the reactant and not in the product belongs in breaking_bonds."
            )
        graph.add_edge(i, j)


def _reaction_graph(mol, breaking_bonds=None, forming_bonds=None):
    """Build the VeloxChem assist-id graph, editing named bonds before collecting hydrogen neighbours.

    Mapping search still uses the unedited endpoint molecules."""
    graph = nx.Graph()
    graph.add_nodes_from(range(mol.GetNumAtoms()))
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    _apply_reaction_bonds(graph, mol.GetNumAtoms(), breaking_bonds, forming_bonds)
    for atom in mol.GetAtoms():
        graph.nodes[atom.GetIdx()]["elem"] = atom.GetAtomicNum()
    for node in graph.nodes:
        hydrogens = [n for n in graph.neighbors(node) if graph.nodes[n]["elem"] == 1]
        if hydrogens:
            graph.nodes[node]["H_indices"] = hydrogens
    return graph


def _same_environment(a, b):
    """Match two nodes on element, and on identity where one is already pinned."""
    return a["elem"] == b["elem"] and a.get("assist_id", -1) == b.get("assist_id", -1)


def _ball(graph, centre, radius):
    """Return the subgraph of everything within `radius` bonds of `centre`: a BFS ball."""
    return graph.subgraph(nx.single_source_shortest_path_length(graph, centre, cutoff=radius))


def _assist_order(n):
    """Walk the atoms on upstream's coprime stride, so no region is searched first just by index.

    Which of several interchangeable atoms gets pinned first is arbitrary but must be reproducible;
    striding by a value coprime to `n` visits every atom once in an index-independent order.
    """
    step = 2 * n // 3
    while n and math.gcd(step, n) != 1:
        step -= 1
    return [(i * step) % n for i in range(n)]


def _record_assist(rea_graph, pro_graph, inner, pins):
    """Record the neighbourhood correspondence and equivalent hydrogens as assist-id pins."""
    for r, p in inner.items():
        pins[p] = r
    for r, p in inner.items():
        rea_graph.nodes[r]["assist_id"] = r
        pro_graph.nodes[p]["assist_id"] = r
        rea_h = rea_graph.nodes[r].get("H_indices")
        pro_h = pro_graph.nodes[p].get("H_indices")
        if rea_h and pro_h and len(rea_h) == len(pro_h):
            for rh, ph in zip(rea_h, pro_h, strict=True):
                rea_graph.nodes[rh]["assist_id"] = rh
                pro_graph.nodes[ph]["assist_id"] = rh
                pins[ph] = rh


def _assist_pins(rea, pro, breaking_bonds=None, forming_bonds=None):
    """Pin isomorphic neighbourhoods using VeloxChem ReactionMatcher._calculate_assist_ids.

    Check radii depth and depth-1 down to its minimum depth of 3. Named bonds edit only
    the reactant pre-solve graph before hydrogen lists are built."""
    n = rea.GetNumAtoms()
    rea_graph = _reaction_graph(rea, breaking_bonds, forming_bonds)
    pro_graph = _reaction_graph(pro)
    order = _assist_order(n)

    # Start one past half the widest connected component's diameter -- enough to span its radius.
    depth = 1
    for component in nx.connected_components(rea_graph):
        depth = max(depth, math.ceil(nx.diameter(rea_graph.subgraph(component)) / 2))
    depth += 1

    pins = {}
    while depth >= _ASSIST_MIN_DEPTH:
        for r in order:
            if rea_graph.nodes[r]["elem"] == 1 or rea_graph.nodes[r].get("assist_id") is not None:
                continue
            rea_ball = _ball(rea_graph, r, depth)
            for p in order:
                pro_node = pro_graph.nodes[p]
                if pro_node.get("assist_id") is not None or pro_node["elem"] != rea_graph.nodes[r]["elem"]:
                    continue
                pro_ball = _ball(pro_graph, p, depth)
                if rea_ball.number_of_nodes() != pro_ball.number_of_nodes():
                    continue
                outer = GraphMatcher(rea_ball, pro_ball, node_match=_same_environment)
                if not outer.is_isomorphic():
                    continue
                # The inner ball must match too, so the pin does not rest on the ragged outer shell.
                rea_inner = _ball(rea_graph, r, depth - 1)
                pro_inner = _ball(pro_graph, p, depth - 1)
                if not GraphMatcher(rea_inner, pro_inner, node_match=_same_environment).is_isomorphic():
                    continue
                inner = {rn: pn for rn, pn in outer.mapping.items() if rn in rea_inner}
                _record_assist(rea_graph, pro_graph, inner, pins)
                break
        depth -= 1
    return pins


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
    work = Chem.RWMol(pro)
    for transfer_index, (proton, acceptor) in enumerate(transfers):
        current, acceptor_p = r2p[proton], r2p[acceptor]
        for site, attached in zip(sites, hydrogens, strict=True):
            donor, spare = r2p[site], r2p[attached[transfer_index]]
            work.RemoveBond(acceptor_p, current)
            work.AddBond(donor, current, Chem.BondType.SINGLE)
            work.RemoveBond(donor, spare)
            work.AddBond(acceptor_p, spare, Chem.BondType.SINGLE)
            current = spare
    pro2 = work.GetMol()
    return rea, pro2, _mechanism(rea, pro2, mapping)


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
    rea, pro, chosen, distinct, truncated = _analyse(reaction)
    if mechanism is not None and (isinstance(mechanism, bool) or not isinstance(mechanism, Integral) or mechanism < 0):
        raise ValueError("mechanism must be a non-negative integer selection")
    if mechanism is None and strict and len(distinct) > 1:
        # With the symmetry quotient truncated, `len(distinct)` is an upper bound and two of these
        # may be one reaction relabelled -- so `strict` says what it can prove and no more.
        bound = (
            " That count is an UPPER BOUND: this reactant's automorphism group hit the search cap, so "
            "the symmetry quotient could not rule out that some of these are the same reaction "
            "relabelled."
            if truncated
            else ""
        )
        raise ValueError(
            f"This reaction is ambiguous on connectivity alone -- {len(distinct)} distinct "
            "mechanisms change the same number of bonds. Call rxts.mechanisms() to see them, "
            f"map the atoms you care about in the input SMILES, or pass mechanism=n to choose.{bound}"
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

    Return reactant, product, chosen mechanism, distinct mechanisms and symmetry-truncation flag.
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
    distinct = distinct_mechanisms(rea, tied)
    truncated = _automorphisms(rea)[1] if len(distinct) > 1 else False
    return rea, pro, chosen, distinct, truncated


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

    smiles = f"{Chem.MolToSmiles(rea)}>>{Chem.MolToSmiles(pro)}"
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

    rea, pro, _, distinct, _truncated = _analyse(reaction)
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
