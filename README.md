# rxts

Reaction mapping and force-field TS guesses, adapted from VeloxChem, with rxembed geometry construction for organic and metal reactions.

## Install

```sh
uv sync --extra ff  # add --extra viz for the notebook
```

GFN-FF needs a Fortran compiler and BLAS/LAPACK; MMFF works without the `ff` extra. rxembed comes from its GitHub main and GFN-FF from a pinned fork; `uv.lock` records both commits.

## Use

```python
import rxts

reaction = "CBr.[Cl-]>>CCl.[Br-]"
guesser = rxts.TransitionStateGuesser(engine="gfnff", multiplicity=1, n_cores=1)
paths = guesser.paths(reaction)
path = paths.best()
guess = path.guess()
guess.dump("guess.xyz")
path.dump("path.xyz")
```

`paths` supports indexing, slicing and iteration, even with one candidate. `.best()` selects the lowest absolute profile maximum. `guesser.find_transition_state(reaction)` returns that single FF path directly.

To score every frame with a physical calculator:

```python
scored = paths.score("gxtb")  # or "gfn2", or a configured ASE calculator
physical_guess = scored.best().guess()
```

Scoring returns new results and preserves the FF paths. `path.score(...)` scores one path; each scored path retains its energies and failures in `.scores`. Failed scores block selection. g-xTB needs the [g-xTB executable](https://github.com/grimme-lab/g-xtb) on PATH, or `rxts.set_xtb("/path/to/xtb")`.

These are candidate geometries; a numerical peak does not verify continuity or a saddle point. For an ASE handoff, use `guess.to_atoms()` or `path.to_atoms()`.

## Settings and inputs

Set options on the guesser, in its constructor or before the next call:

| Setting | Default | Meaning |
|---|---|---|
| `n_lambda` | 21 | Frames per path |
| `n_conformers` | 4 | Requested coembedded endpoint pairs; one scan per accepted pair |
| `n_embed` | 4 | Intermediate geometry proposals near each path's peak |

All candidate paths share the first pair's frozen FF fields. Counts are budgets; fewer pairs may be accepted. Use `n_conformers=1, n_embed=0` for one scan without extra proposals. Optional `embed_score` ranks peak proposals during generation; `.score(...)` evaluates finished paths.

Choose `engine="gfnff"`, `"mmff"` (requires parameters for both endpoints), or `"auto"` (MMFF where supported, otherwise GFN-FF).

Pass reaction SMILES, endpoint Mols/XYZ, selected rxembed isomers, or a `Coembedded` batch. Endpoint inputs request `n_conformers` fresh pairs within the selected isomers; coordinates can establish identity but are not held fixed. Supplied batches keep their coordinates and count. For exact geometry preservation, construct a batch with `rxembed.coembed(..., fixed="reactant")` first. Fragment lists supply graphs. Declare `multiplicity` and, for XYZ, `charge`. Metal seating must identify one isomer per endpoint; `*_geometry` selects a coordination shape.

## Examples and reference

The [notebook](examples/quickstart.ipynb) covers organic and metal reactions, sampling, calculators, plotting and proton relays. Mapping and route construction are also available independently through `rxts.map_smiles(...)` and `rxts.mechanisms(...)`; `add_relay="O"` adds a water relay.

Adapted from [van Hoorn, Norman and Ahlquist (2026), *From SMILES Codes for Reactants and Products to Transition States With VeloxChem*](https://doi.org/10.1002/jcc.70454).

BSD-3-Clause; attribution is retained in [LICENSE](LICENSE).
