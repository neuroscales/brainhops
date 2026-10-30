"""
The rules that tie `to()`, `from_instance()` and `from_any()` together.

The rules are those of `spec/design/conversions.md`: the three spellings
give the same result, a class copies by name only the fields of a related
class, a transformation is converted by its converters, `to()` returns the
object itself when nothing changes, and lazy wrappers are not targets.
"""

import numpy as np
import pytest
import typing_extensions as tx

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


@pytest.mark.parametrize(
    "t",
    [X.Scaling([1.0, 2.0, 3.0]), X.Permutation([1, 0, 2])],
    ids=lambda x: type(x).__name__,
)
def test_a_conversion_to_rotation_refuses_a_non_rotation(
    t: X.Transformation,
) -> None:
    # A scaling that is not a unit scaling, and an odd permutation, are
    # not rotations, although Rotation is a subclass of Linear (#622).
    with pytest.raises(LossyConversionError):
        t.to(X.Rotation)


@pytest.mark.parametrize(
    "t",
    [X.Scaling([1.0, 1.0, 1.0]), X.Permutation([1, 2, 0])],
    ids=lambda x: type(x).__name__,
)
def test_a_conversion_to_rotation_accepts_a_rotation(
    t: X.Transformation,
) -> None:
    result = t.to(X.Rotation)
    assert isinstance(result, X.Rotation)
    np.testing.assert_allclose(
        np.asarray(result.to(X.Linear).matrix),
        np.asarray(t.to(X.Linear).matrix),
    )


def test_a_conversion_from_projection_to_rotation_checks_the_result() -> None:
    # A projection that drops an axis and creates another one has a
    # square but singular matrix, which is not a rotation (#622).
    projection = X.Projection([1], [1], input=S.RASmm())
    with pytest.raises(LossyConversionError):
        projection.to(X.Rotation)
    # A projection that drops and creates no axis is the identity.
    assert isinstance(X.Projection().to(X.Rotation), X.Rotation)


def test_a_rotation_tangent_stores_the_logarithm_of_the_matrix() -> None:
    # The rotation tangent stores the logarithm of the matrix, so the
    # matrix of a Linear must not be copied into it as data.
    result = X.Linear(_ROTATION).to(X.RotationExponential)
    np.testing.assert_allclose(
        np.asarray(result.matrix), _ROTATION, atol=1e-12
    )


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
    with pytest.raises(ConversionError):
        Metadata.from_instance(A.LeftToRightAxis())
    with pytest.raises(ConversionError):
        A.Axis.from_instance(Metadata())


def test_a_single_scale_image_is_a_pyramid_of_one_level() -> None:
    image = SingleScaleImage(
        np.zeros((2, 2, 2)), transformations=[X.Scaling([2.0, 2.0, 2.0])]
    )
    for pyramid in (
        MultiScaleImage.from_instance(image),
        image.to(MultiScaleImage),
        MultiScaleImage.from_any(image),
    ):
        assert pyramid.images == [image]
        # The level keeps its transformations, so the geometry is the same.
        assert list(pyramid.transformations) == []
        assert pyramid.metadata is image.metadata


def test_a_pyramid_is_the_single_scale_image_of_its_first_level() -> None:
    levels = [
        SingleScaleImage(np.zeros((4, 4, 4))),
        SingleScaleImage(np.zeros((2, 2, 2))),
    ]
    pyramid = MultiScaleImage(levels, [X.Translation([1.0, 2.0, 3.0])])
    for image in (
        SingleScaleImage.from_instance(pyramid),
        pyramid.to(SingleScaleImage),
    ):
        assert type(image) is SingleScaleImage
        assert image.shape == (4, 4, 4)
        # The level is placed in the world space of the pyramid.
        np.testing.assert_allclose(
            np.asarray(image.geometry.transformation.to(X.Affine).matrix),
            np.asarray(pyramid.geometry.transformation.to(X.Affine).matrix),
        )


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


# ----------------------------------------------------------------------
#   FORMATS
# ----------------------------------------------------------------------


def _transformation_formats() -> tx.List[type]:
    """Return every concrete transformation format that can be imported."""
    import importlib
    import pkgutil

    import brainhops.io.transformations as package
    from brainhops.io.transformations.base import TransformationFormat

    prefix = package.__name__ + "."
    for module in pkgutil.walk_packages(package.__path__, prefix):
        try:
            importlib.import_module(module.name)
        except ImportError:
            continue
    found, stack = set(), [TransformationFormat]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in found:
                found.add(sub)
                stack.append(sub)
    # The data model leaves classes named `__magic_discard__` behind, which
    # are not formats.
    return sorted(
        (
            c
            for c in found
            if issubclass(c, X.Transformation)
            and not c.__name__.startswith("__")
        ),
        key=lambda c: (c.__module__, c.__qualname__),
    )


@pytest.mark.parametrize(
    "cls", _transformation_formats(), ids=lambda c: c.__name__
)
def test_a_format_converts_without_fields_of_its_data_model(
    cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Whatever the order of its bases, a format converts another
    # transformation as `to()` does, without the fields that its data model
    # would copy, which would replace the map that the converter builds.
    from brainhops.datamodel._transformations.compute import convert

    calls = []
    real = convert._convert

    def spy(other: tx.Any, target: type, **kwargs: tx.Any) -> tx.Any:
        calls.append(kwargs)
        return real(other, target, **kwargs)

    monkeypatch.setattr(convert, "_convert", spy)
    for source in (X.Identity(), X.Affine(np.eye(4)[:3])):
        calls.clear()
        try:
            cls.from_instance(source)
        except (ConversionError, TypeError, ValueError):
            pass
        # The first call is the conversion that was asked for. A converter
        # that goes through other classes calls the dispatcher again.
        assert calls[:1] == [{}], type(source).__name__


@pytest.mark.parametrize(
    "module, name",
    [
        ("brainhops.io.images.tiff._multiscale", "TiffMultiScaleImage"),
        ("brainhops.io.images.zarr._multiscale", "OmeZarrImage"),
    ],
    ids=["tiff", "ome-zarr"],
)
def test_a_pyramid_format_takes_a_single_scale_image(
    module: str, name: str
) -> None:
    cls = getattr(pytest.importorskip(module), name)
    image = SingleScaleImage(np.arange(8.0).reshape(2, 2, 2))
    pyramid = cls.from_instance(image)
    assert isinstance(pyramid, cls)
    assert pyramid.nscales == 1
    np.testing.assert_array_equal(
        np.asarray(pyramid.images[0].data), np.asarray(image.data)
    )


def test_a_format_that_lists_its_data_model_first_is_refused() -> None:
    from brainhops.io.base.parsers import TextFileReader
    from brainhops.io.transformations.base import TransformationFormat

    with pytest.raises(TypeError, match="before its format base"):

        class Late(X.Affine, TransformationFormat):
            pass

    # A base that only reads may come first.
    class Early(TextFileReader, TransformationFormat, X.Affine):
        pass

    assert issubclass(Early, X.Affine)


def test_a_format_without_an_exact_converter_adapts_world_systems() -> None:
    from brainhops.io.transformations.freesurfer.lta._xforms import (
        LtaTransformationRASToRAS,
        LtaTransformationVoxToVox,
    )
    from brainhops.io.transformations.minc._affine import XfmAffine

    matrix = np.concatenate([np.eye(3), [[1.0], [2.0], [3.0]]], 1)
    lps = X.Affine(matrix, input=S.LPSmm(), output=S.LPSmm())
    # LPS is RAS with the first two axes flipped, so the translation flips
    # with them.
    for cls in (XfmAffine, LtaTransformationRASToRAS):
        converted = lps.to(cls)
        assert isinstance(converted.input, S.RASmm)
        np.testing.assert_allclose(converted.matrix[:, 3], [-1.0, -2.0, 3.0])
    # A voxel system is not bridged to a world system without the geometry
    # of its volume.
    with pytest.raises(ConversionError):
        XfmAffine.from_instance(LtaTransformationVoxToVox(matrix=matrix))
