"""
The completeness of the graph of conversions between transformations.

Every pair of concrete classes is converted, in both directions, from
an identity and from a general transformation. A conversion either
succeeds, reports a loss with a result of the requested class, or is
refused on purpose. The pairs that are refused on purpose are listed
below with their reason, so that a missing converter shows up as a
failure instead of going unnoticed. The rules are those of
`spec/design/conversions.md`.
"""

import warnings

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel import transformations as X
from brainhops.errors import ConversionError, DomainError

_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_AFFINE = np.concatenate([np.diag([2.0, 3.0, 4.0]), [[1.0], [2.0], [3.0]]], 1)


def _samples() -> tx.Dict[str, tx.Tuple[X.Transformation, X.Transformation]]:
    """Return an identity and a general instance of each source class."""
    rng = np.random.default_rng(0)
    return {
        "Identity": (X.Identity(), X.Identity()),
        "Translation": (
            X.Translation([0.0, 0.0, 0.0]),
            X.Translation([1.0, 2.0, 3.0]),
        ),
        "Scaling": (X.Scaling([1.0, 1.0, 1.0]), X.Scaling([2.0, 3.0, 4.0])),
        "Permutation": (X.Permutation([0, 1, 2]), X.Permutation([1, 0, 2])),
        "Rotation": (X.Rotation(np.eye(3)), X.Rotation(_ROTATION)),
        "Linear": (X.Linear(np.eye(3)), X.Linear(np.diag([2.0, 3.0, 4.0]))),
        "Affine": (X.Affine(np.eye(4)[:3]), X.Affine(_AFFINE)),
        "AffineExponential": (
            X.Affine(np.zeros((3, 4)), log=True),
            X.Affine(np.full((3, 4), 0.1), log=True),
        ),
        "Sequence": (
            X.Sequence([X.Translation([0.0, 0.0, 0.0])]),
            X.Sequence([X.Translation([1.0, 2.0, 3.0]), X.Scaling([2.0] * 3)]),
        ),
        "DisplacementField": (
            X.DisplacementField(np.zeros((4, 4, 4, 3))),
            X.DisplacementField(rng.random((4, 4, 4, 3))),
        ),
    }


_SOURCES = list(_samples())
_TARGETS = [
    *_SOURCES,
    "CoordinateField",
    "StationaryVelocityField",
    "LinearExponential",
    "RotationExponential",
    "ScalingExponential",
]
_FIELDS = {"DisplacementField", "CoordinateField", "StationaryVelocityField"}
_MATRICES = set(_TARGETS) - _FIELDS - {"Sequence"}

# The pairs that are refused on purpose, and the reason.
_REFUSED = {
    # A field is built on a grid, which a transformation without one does
    # not give.
    **{(s, t): "a field needs a grid" for s in _SOURCES for t in _FIELDS},
    # A field holds an affine map only approximately, and fitting one is
    # not a conversion.
    **{
        ("DisplacementField", t): "a field is not a matrix"
        for t in _MATRICES - {"Identity"}
    },
    # The logarithm of a displacement is not computed, and the error says
    # to use `to(log=True)` on a field that holds a velocity.
    ("DisplacementField", "StationaryVelocityField"): "no logarithm",
}
# A displacement field converts to a field of coordinates.
del _REFUSED["DisplacementField", "DisplacementField"]
del _REFUSED["DisplacementField", "CoordinateField"]

# The general samples whose map has no real logarithm, so that a tangent
# class cannot hold it: a permutation that swaps two axes has a negative
# determinant, and a scaling read from a rotation has a zero factor.
_NO_LOGARITHM = {
    ("Permutation", t)
    for t in (
        "AffineExponential",
        "LinearExponential",
        "RotationExponential",
        "ScalingExponential",
    )
} | {("Rotation", "ScalingExponential")}


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("source", _SOURCES)
def test_every_pair_converts_or_is_refused_on_purpose(
    source: str,
    target: str,
) -> None:
    cls = getattr(X, target)
    identity, general = _samples()[source]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if (source, target) in _REFUSED:
            for sample in (identity, general):
                with pytest.raises(ConversionError):
                    sample.to(cls, lossy=True)
            return
        # The identity is exact in every class.
        assert isinstance(identity.to(cls), cls)
        if (source, target) in _NO_LOGARITHM:
            with pytest.raises(DomainError):
                general.to(cls, lossy=True)
            return
        # A lossy result is still of the requested class.
        assert isinstance(general.to(cls, lossy=True), cls)


@pytest.mark.parametrize("source", _SOURCES)
def test_a_conversion_to_the_own_class_returns_the_object(
    source: str,
) -> None:
    for sample in _samples()[source]:
        assert sample.to(type(sample)) is sample
        assert sample.to() is sample
