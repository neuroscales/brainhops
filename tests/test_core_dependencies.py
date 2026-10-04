"""The availability flags of submodules are answered without importing
them, from what makes them importable (#290)."""

import importlib.metadata
import importlib.util
from typing import Iterator

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


@pytest.fixture
def fake_metadata(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Distributions described by `{name: (extras, requirements)}`."""
    dists = {}

    def metadata(name: str) -> dict:
        if name not in dists:
            raise importlib.metadata.PackageNotFoundError(name)
        return {"Provides-Extra": dists[name][0]}

    def requires(name: str) -> list:
        metadata(name)
        return list(dists[name][1])

    def distribution(name: str) -> str:
        metadata(name)
        return name

    class _Message(dict):
        def get_all(self, key: str) -> list:
            return self.get(key)

    monkeypatch.setattr(
        importlib.metadata, "metadata", lambda n: _Message(metadata(n))
    )
    monkeypatch.setattr(importlib.metadata, "requires", requires)
    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    return dists


def test_has_extra(fake_metadata: dict) -> None:
    fake_metadata.update(
        {
            "pkg": (
                ["array", "full", "odd"],
                [
                    "base>=1",
                    'arr>=1.24; extra == "array"',
                    "pkg[array]; extra == 'full'",
                    'gone; extra == "full"',
                    'arr; python_version < "3" and extra == "odd"',
                ],
            ),
            "base": ([], []),
            "arr": ([], []),
        }
    )
    assert deps._has_extra("pkg", "array") is True
    # A required distribution is missing.
    assert deps._has_extra("pkg", "full") is False
    # A marker other than the extra cannot be told without evaluating it.
    assert deps._has_extra("pkg", "odd") is None
    # An extra that the distribution does not declare.
    assert deps._has_extra("pkg", "other") is False
    # No metadata at all.
    assert deps._has_extra("missing", "array") is None


@pytest.fixture
def fresh_flags() -> Iterator[None]:
    """Forget the derived flags before and after the test."""
    for flag in _DERIVED:
        vars(deps).pop(flag, None)
    yield
    for flag in _DERIVED:
        vars(deps).pop(flag, None)


def test_dask_array_without_metadata(
    monkeypatch: pytest.MonkeyPatch, fresh_flags: None
) -> None:
    # With no metadata to read, the flag falls back to importing.
    monkeypatch.setattr(deps, "_has_extra", lambda *_: None)
    assert deps.HAS_DASK_ARRAY == _importable("dask.array")


def test_dask_array_needs_its_extra(
    monkeypatch: pytest.MonkeyPatch, fresh_flags: None
) -> None:
    if importlib.util.find_spec("dask") is None:
        pytest.skip("dask is not installed")
    monkeypatch.setattr(deps, "_has_extra", lambda *_: False)
    assert deps.HAS_DASK_ARRAY is False
    assert deps.HAS_DASK_NDIMAGE is False


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
