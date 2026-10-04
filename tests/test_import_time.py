"""Guards on what `import brainhops` and `brainhops --help` import (#290).

Importing the package, or asking the command line for help, should not
import the optional I/O dependencies or the data model: they take
seconds to import. Each check runs in a fresh interpreter, since this
one has imported everything by the time the tests run.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import brainhops

# The directory that holds the `brainhops` package, so that the fresh
# interpreter imports the same copy as this one.
_SRC = str(Path(brainhops.__file__).resolve().parent.parent)
_DATA = Path(__file__).parent / "data"


def _run(code: str) -> str:
    """Run `code` in a fresh interpreter and return its standard output."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [_SRC, env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout


def _modules_after(code: str) -> set:
    """The modules imported once `code` has run in a fresh interpreter."""
    code += "\nimport json, sys\nprint(json.dumps(sorted(sys.modules)))"
    return set(json.loads(_run(code).splitlines()[-1]))


@pytest.fixture(scope="module")
def modules_after_import() -> set:
    return _modules_after("import brainhops")


@pytest.fixture(scope="module")
def modules_after_help() -> set:
    # Runs `python -m brainhops --help` in-process, through the same
    # entry point as the console script.
    return _modules_after(
        "import runpy, sys\n"
        "sys.argv = ['brainhops', '--help']\n"
        "try:\n"
        "    runpy.run_module('brainhops', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
    )


@pytest.mark.parametrize(
    "module",
    ["abczarr", "dask", "nibabel", "tifffile", "PIL", "h5py", "pint"],
)
def test_import_brainhops_does_not_import(
    module: str, modules_after_import: set
) -> None:
    assert module not in modules_after_import


@pytest.mark.parametrize("module", ["bagof.magic", "brainhops.datamodel"])
def test_help_does_not_import(module: str, modules_after_help: set) -> None:
    assert module not in modules_after_help


def test_load_finds_every_format_in_a_fresh_interpreter() -> None:
    # `brainhops.io.load` dispatches over the formats of every kind, which
    # register as their packages are imported; the lazy `brainhops.io`
    # must import them before dispatching.
    path = _DATA / "itk_affine3d.tfm"
    code = (
        "import brainhops\n"
        f"print(type(brainhops.io.load({str(path)!r})).__name__)\n"
    )
    assert _run(code).strip() == "TfmTransform"


@pytest.mark.parametrize(
    "package",
    [
        "brainhops",
        "brainhops.datamodel",
        "brainhops.datamodel._transformations",
        "brainhops.io",
        "brainhops.io.base",
    ],
)
def test_lazy_names_resolve(package: str) -> None:
    module = importlib.import_module(package)
    for name in module.__all__:
        assert getattr(module, name) is not None, name
        assert name in dir(module), name
