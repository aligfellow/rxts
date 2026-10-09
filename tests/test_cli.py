"CLI argument validation and output routing."

import sys

import pytest

from rxts import cli


def _run(monkeypatch, *argv):
    """Drive `cli.main()` with an argv, returning the SystemExit it raised."""
    monkeypatch.setattr(sys, "argv", ["rxts", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return exit_info.value


def test_plain_smiles_requires_reaction(monkeypatch, capsys):
    assert _run(monkeypatch, "CCO").code == 2
    assert ">>" in capsys.readouterr().err, "the error must name the arrow that is missing"


def test_empty_product_is_refused_early(monkeypatch, capsys):
    assert _run(monkeypatch, "CCO>>").code == 2
    assert "at least one reactant and one product" in capsys.readouterr().err


def test_cli_uses_method_registry(monkeypatch, capsys):
    assert _run(monkeypatch, "C[Br].[Cl-]>>C[Cl].[Br-]", "--method", "b3lyp").code == 2
    error = capsys.readouterr().err
    assert "--method" in error
    assert "gfn2" in error, "the message must list the methods that do exist"


def test_old_verify_flag_is_rejected(monkeypatch, capsys):
    assert _run(monkeypatch, "C[Br].[Cl-]>>C[Cl].[Br-]", "--verify").code == 2
    assert "unrecognized arguments: --verify" in capsys.readouterr().err


def _drive(monkeypatch, *argv):
    """Run the CLI with a stub guesser and record its calls and outputs."""
    seen = {"written": []}

    class _Guess:
        method = "gfn2"

        def dump(self, name):
            seen["written"].append(name)
            return name

    class _Path:
        def guess(self, on):
            seen["on"] = on
            return _Guess()

        def score(self, method=None, solvent=None):
            seen["scored"] = method

        def dump(self, name):
            seen["written"].append(name)
            return name

    class _Guesser:
        def __init__(self, **settings):
            seen["settings"] = settings

        def find_transition_state(self, reactant):
            seen["reaction"] = reactant
            return _Path()

    monkeypatch.setattr(cli, "TransitionStateGuesser", _Guesser)
    monkeypatch.setattr(cli, "set_verbose", lambda: None)
    monkeypatch.setattr(sys, "argv", ["rxts", *argv, "--engine", "gfnff", "--multiplicity", "1"])
    cli.main()
    return seen


def test_cli_writes_selected_surface(monkeypatch):
    plain = _drive(monkeypatch, "C[Br].[Cl-]>>C[Cl].[Br-]", "-o", "run", "-n", "7")
    assert plain["reaction"] == "C[Br].[Cl-]>>C[Cl].[Br-]"
    assert plain["settings"]["n_lambda"] == 7, "-n reaches the guesser as its own setting, not as a default"
    assert plain["written"] == ["run.xyz", "run_path.xyz"], "one structure and the whole trajectory, both named"
    assert plain["on"] == "ff"
    assert "scored" not in plain, "no --xtb is no QM: the run must not pay for a surface nobody asked for"

    scored = _drive(monkeypatch, "C[Br].[Cl-]>>C[Cl].[Br-]", "--xtb", "--method", "gfn1")
    assert scored["scored"] == "gfn1", "--method names the surface --xtb prices on"
    assert scored["on"] == "qm", "after scoring, the structure written is the QM maximum"


def test_plot_requires_viz_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotprofile", None)
    with pytest.raises(ModuleNotFoundError, match=r"rxts\[viz\]"):
        _drive(monkeypatch, "C[Br].[Cl-]>>C[Cl].[Br-]", "--plot")


def test_cli_preserves_reaction_smiles(monkeypatch):
    reaction = "N->[Pd+2].Cl |atomProp:1.geometry.SPL|>O>N->[Pd+2].Cl |atomProp:1.geometry.SPL|"
    seen = _drive(monkeypatch, reaction)
    assert seen["reaction"] == reaction
    assert seen["on"] == "ff"
