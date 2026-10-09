import inspect
import subprocess
import sys

import rxts


def test_core_import_needs_no_workflow_tools():
    probe = (
        "import sys;"
        " sys.modules.update({name: None for name in"
        " ('geometric', 'gfnff', 'openconf',"
        " 'plotprofile', 'xyzrender', 'graphrc', 'prism_pruner', 'sklearn', 'matplotlib')});"
        " import rxts; import rxts.xtb as xtb;"
        " xtb.xtb_exe = lambda: None;"
        " tsg = rxts.TransitionStateGuesser(engine='mmff', multiplicity=1, n_lambda=3, n_cores=1);"
        " path = tsg.find_transition_state('CBr.[Cl-]>>CCl.[Br-]');"
        " assert len(path.frames)==3; assert path.guess('ff').positions.shape[1]==3;"
        " assert rxts.mechanisms('CBr.[Cl-]>>CCl.[Br-]');"
        " assert 'geometric' not in {n for n,v in sys.modules.items() if v is not None};"
        " print('core passed')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith("core passed")


def test_documented_submodules_bind_on_import():
    for name in ("ff", "gfnff", "scoring"):
        assert hasattr(rxts, name), f"bare package import did not bind rxts.{name}"


def test_import_preserves_application_logging():
    probe = (
        "import logging; logger=logging.getLogger('rxts');"
        " handler=logging.StreamHandler(); logger.addHandler(handler); logger.setLevel(logging.ERROR);"
        " import rxts; assert logger.handlers == [handler]; assert logger.level == logging.ERROR"
    )
    subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)


def test_version_matches_metadata():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert rxts.__version__ == declared, f"package version {rxts.__version__!r} != project version {declared!r}"


def _public_signatures():
    for name in rxts.__all__:
        obj = getattr(rxts, name)
        if not inspect.isclass(obj):
            if callable(obj):
                yield name, obj
            continue
        for attr, member in sorted(vars(obj).items()):
            if attr.startswith("_"):
                continue
            fn = member
            if isinstance(fn, property):
                fn = fn.fget
            elif isinstance(fn, classmethod | staticmethod):
                fn = fn.__func__
            if inspect.isfunction(fn):
                yield f"{name}.{attr}", fn


def test_public_annotations():
    for label, fn in sorted(_public_signatures()):
        signature = inspect.signature(fn, eval_str=False)
        missing = [
            name
            for name, parameter in signature.parameters.items()
            if name not in ("self", "cls") and parameter.annotation is inspect.Parameter.empty
        ]
        assert not missing, f"rxts.{label} has unannotated parameters: {missing}"
        assert signature.return_annotation is not inspect.Signature.empty, f"rxts.{label} has no return annotation"


def test_public_names_are_exported():
    public_api = (
        "Geometry",
        "Mechanism",
        "ReactionPath",
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
    )
    unexported = sorted(name for name in public_api if name not in rxts.__all__ or not hasattr(rxts, name))
    assert not unexported, f"public names missing from __all__ or package namespace: {unexported}"
