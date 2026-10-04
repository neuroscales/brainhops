"""The availability flags of submodules are answered without importing
them, from what makes them importable (#290)."""

import importlib.util

import pytest
from test_import_time import _modules_after, _run

from brainhops._core import dependencies as deps

_DERIVED = [
    "HAS_DASK_ARRAY",
    "HAS_SCIPY_NDIMAGE",
    "HAS_CUPY_NDIMAGE",
    "HAS_DASK_NDIMAGE",
]


def _importable(qualname: str) -> bool:
    try:
        importlib.import_module(qualname)
    except (ImportError, OSError):
        return False
    return True


@pytest.mark.parametrize(
    "flag, qualname",
    [
        ("HAS_DASK_ARRAY", "dask.array"),
        ("HAS_SCIPY_NDIMAGE", "scipy.ndimage"),
        ("HAS_CUPY_NDIMAGE", "cupyx.scipy.ndimage"),
        ("HAS_DASK_NDIMAGE", "brainhops._core.dask_ndimage"),
    ],
)
def test_derived_flag_matches_import(flag: str, qualname: str) -> None:
    assert getattr(deps, flag) == _importable(qualname)


def test_derived_flags_do_not_import() -> None:
    modules = _modules_after(
        "from brainhops._core import dependencies as deps\n"
        + "".join(f"deps.{flag}\n" for flag in _DERIVED)
    )
    for module in ("dask", "dask.array", "scipy", "scipy.ndimage", "cupy"):
        assert module not in modules


def test_lazy_type_imports_when_used() -> None:
    code = (
        "import sys\n"
        "from brainhops._core import dependencies as deps\n"
        "Fraction = deps.lazy_type('fractions:Fraction')\n"
        "print('fractions' in sys.modules)\n"
        "print(isinstance(Fraction(1, 2), Fraction))\n"
        "print(isinstance(0.5, Fraction), repr(Fraction))\n"
    )
    out = _run(code).split()
    assert out == ["False", "True", "False", "fractions.Fraction"]
    assert deps.lazy_type("fractions:Fraction") is deps.lazy_type(
        "fractions:Fraction"
    )


def test_has_abczarr_driver_does_not_import() -> None:
    modules = _modules_after(
        "from brainhops._core import dependencies as deps\n"
        "deps.has_abczarr_driver()\n"
    )
    assert "abczarr" not in modules
    assert "zarr" not in modules


def test_has_abczarr_driver_agrees_with_abczarr() -> None:
    # Told from the installed backends before abczarr is imported, and
    # asked of abczarr after.
    pytest.importorskip("abczarr")
    code = (
        "from brainhops._core import dependencies as deps\n"
        "print(deps.has_abczarr_driver())\n"
        "import abczarr\n"
        "print(deps.has_abczarr_driver())\n"
    )
    before, after = _run(code).split()
    assert before == after
