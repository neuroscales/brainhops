"""
The rules that tie `to()`, `from_instance()` and `from_any()` together.

The rules are those of `docs/design/conversions.md`: the three spellings
give the same result, a class copies by name only the fields of a related
class, a transformation is converted by its converters, `to()` returns the
object itself when nothing changes, and lazy wrappers are not targets.
"""

import numpy as np
import pytest

from brainhops.datamodel import axes as A
from brainhops.datamodel import systems as S
from brainhops.datamodel import transformations as X
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.metadata import Metadata
from brainhops.errors import ConversionError, LossyConversionError

_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------
#   ONE OPERATION, THREE SPELLINGS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "t, cls",
    [
        (X.Linear(_ROTATION), X.Rotation),
        (X.Rotation(_ROTATION), X.Affine),
        (X.Scaling([2.0, 3.0, 4.0]), X.Affine),
        (X.Translation([1.0, 2.0, 3.0]), X.Affine),
    ],
    ids=lambda x: getattr(x, "__name__", type(x).__name__),
)
def test_the_three_spellings_convert_alike(
    t: X.Transformation, cls: type
) -> None:
    results = [t.to(cls), cls.from_instance(t), cls.from_any(t)]
    for result in results:
        assert isinstance(result, cls)
        np.testing.assert_allclose(
            np.asarray(result.matrix), np.asarray(results[0].matrix)
        )


def test_a_conversion_checks_what_the_class_promises() -> None:
    # A rotation promises an orthogonal matrix, which field names cannot
    # check.
    with pytest.raises(LossyConversionError):
        X.Rotation.from_instance(X.Linear(np.diag([2.0, 1.0, 1.0])))
    # A translation vector is not a vector of scales.
    with pytest.raises(LossyConversionError):
        X.Scaling.from_instance(X.Translation([1.0, 2.0, 3.0]))


# ----------------------------------------------------------------------
#   RELATED CLASSES
# ----------------------------------------------------------------------


def test_related_classes_are_copied_by_name_with_their_checks() -> None:
    assert isinstance(A.LeftToRightAxis.from_instance(A.SpaceAxis()), A.Axis)
    # A parent instance copied into a polymorphic parent keeps the subclass
    # that its fields select.
    assert type(A.SpaceAxis.from_instance(A.LeftToRightAxis())) is (
        A.LeftToRightAxis
    )
    with pytest.raises(ValueError):
        A.TimeAxis.from_instance(A.LeftToRightAxis())
    with pytest.raises(ValueError):
        S.RASmm.from_instance(S.LPSmm())


def test_unrelated_classes_are_refused() -> None:
    image = SingleScaleImage(np.zeros((2, 2, 2)))
    with pytest.raises(ConversionError):
        MultiScaleImage.from_instance(image)
    with pytest.raises(ConversionError):
        Metadata.from_instance(A.LeftToRightAxis())


def test_from_instance_takes_only_data_models() -> None:
    with pytest.raises(TypeError, match="from_any"):
        A.Axis.from_instance({"name": "x"})
    # `from_any` reads a mapping.
    assert A.Axis.from_any({"name": "x"}).name == "x"


# ----------------------------------------------------------------------
#   COPIES
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        A.LeftToRightAxis(),
        S.RASmm(),
        Metadata(description="d"),
        X.Affine(np.eye(4)[:3]),
        X.Sequence([X.Translation([1.0, 2.0, 3.0])]),
        X.Affine(np.eye(4)[:3]).inverse(),
    ],
    ids=lambda x: type(x).__name__,
)
def test_to_returns_the_object_unless_a_copy_is_asked(obj: object) -> None:
    assert obj.to() is obj
    assert obj.to(type(obj)) is obj
    copy = obj.to(copy=True)
    assert copy is not obj
    assert type(copy) is type(obj)


# ----------------------------------------------------------------------
#   WRAPPERS
# ----------------------------------------------------------------------


def test_a_lazy_wrapper_is_not_a_conversion_target() -> None:
    affine = X.Affine(np.eye(4)[:3])
    with pytest.raises(ConversionError, match=r"inverse\(\)"):
        affine.to(X.InverseAffine)
    inverse = affine.inverse()
    assert inverse.to(type(inverse)) is inverse
