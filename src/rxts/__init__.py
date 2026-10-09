"""VeloxChem-derived reaction mapping and force-field TS guesses."""

from importlib.metadata import PackageNotFoundError, version

from rxts import ff as ff
from rxts import gfnff as gfnff
from rxts import scoring as scoring
from rxts.builder import PreparedReaction, prepare_endpoints
from rxts.draw import (
    bond_data,
    draw_mechanisms,
    profile_data,
)
from rxts.ff import TransitionStateGuesser, set_verbose
from rxts.mapping import Mechanism, map_reaction, map_smiles, mechanisms
from rxts.scoring import CalculatorMethod, ScoreResult
from rxts.structure import Geometry, ReactionPath
from rxts.xtb import set_xtb

try:
    __version__ = version("rxts")
except PackageNotFoundError:
    __version__ = "0.0.0+source"

__all__ = [
    "CalculatorMethod",
    "Geometry",
    "Mechanism",
    "PreparedReaction",
    "ReactionPath",
    "ScoreResult",
    "TransitionStateGuesser",
    "bond_data",
    "draw_mechanisms",
    "map_reaction",
    "map_smiles",
    "mechanisms",
    "prepare_endpoints",
    "profile_data",
    "set_verbose",
    "set_xtb",
]
