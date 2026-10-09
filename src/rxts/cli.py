"""Command-line FF guesses with optional physical scoring and profile plotting."""

from __future__ import annotations

import argparse
from importlib import import_module

from rxts.ff import TransitionStateGuesser, set_verbose
from rxts.mapping import split_reaction_smiles
from rxts.xtb import METHODS


def main() -> None:
    """Find a transition state guess for a reaction SMILES."""
    parser = argparse.ArgumentParser(
        prog="rxts",
        description="Transition state guess and reaction path from a reaction SMILES.",
    )
    parser.add_argument(
        "reaction",
        help='reaction SMILES, e.g. "CBr.[Cl-]>>CCl.[Br-]"',
    )
    parser.add_argument("-o", "--prefix", default="ts", help="output file prefix (default: ts)")
    parser.add_argument("-n", "--n-lambda", type=int, default=21, help="points along the path")
    parser.add_argument("--xtb", action="store_true", help="re-score the path with xtb")
    parser.add_argument("--method", choices=sorted(METHODS), help="explicit physical method for --xtb")
    parser.add_argument("--engine", choices=("gfnff", "mmff", "auto"), help="explicit FF construction engine")
    parser.add_argument("--multiplicity", type=int, help="declared spin multiplicity")
    parser.add_argument("--n-cores", type=int, default=1, help="FF/scoring core budget (default: 1)")
    parser.add_argument("--solvent", default=None, help="ALPB solvent name, e.g. water")
    parser.add_argument("--plot", action="store_true", help="write <prefix>_profile.png")
    args = parser.parse_args()

    try:
        if split_reaction_smiles(args.reaction) is None:
            parser.error('reaction must contain ">>" or use reactants>agents>products')
    except ValueError as error:
        parser.error(str(error))
    if args.engine is None or args.multiplicity is None:
        parser.error("declare --engine and --multiplicity for FF construction")
    if args.xtb and args.method is None:
        parser.error("declare --method for --xtb")

    set_verbose()
    tsg = TransitionStateGuesser(
        n_lambda=args.n_lambda,
        solvent=args.solvent,
        engine=args.engine,
        multiplicity=args.multiplicity,
        n_cores=args.n_cores,
    )

    path = tsg.find_transition_state(args.reaction)

    on = "qm" if args.xtb else "ff"
    if args.xtb:
        print(f"Physical method: {args.method}")
        path.score(method=args.method, solvent=args.solvent)

    guess = path.guess(on)

    print()
    print(guess.dump(f"{args.prefix}.xyz"))  # the one structure, its origin in the XYZ comment
    print(path.dump(f"{args.prefix}_path.xyz"))  # the whole scan, as a trajectory

    if args.plot:
        try:
            plotprofile = import_module("plotprofile")
        except ModuleNotFoundError as error:
            if error.name != "plotprofile":
                raise
            raise ModuleNotFoundError("Plotting requires rxts[viz]; pip install 'rxts[viz]'") from error

        from rxts.draw import bond_data, profile_data

        # curviness=0 joins the points straight: a scan is data, and smoothing would draw points that
        # were never computed.
        plotter = plotprofile.ReactionProfilePlotter(
            units="kcal",
            energy="E",
            labels=False,
            curviness=0.0,
            x_label="reaction coordinate  λ",
            y2={"label": "bond length (Å)", "linestyle": "--"},
        )
        plotter.plot(
            profile_data(path),
            secondary=bond_data(path),
            x=[frame.lam for frame in path.frames],
            filename=f"{args.prefix}_profile",
            file_format="png",
        )
        print(f"{args.prefix}_profile.png")


if __name__ == "__main__":
    main()
