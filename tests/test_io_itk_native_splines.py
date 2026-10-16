"""Native noncubic spline encodings and 2-D field geometry."""

from pathlib import Path

import numpy as np
import pytest

from brainhops import io
from brainhops.datamodel import transformations as xforms
from brainhops.io.transformations.itk._common import (
    _TRANSFORM_TYPE_RE,
    _type_fields,
)


def _apply(transform: xforms.Transformation, points: np.ndarray) -> np.ndarray:
    field = xforms.CoordinatesField(field=points)
    result = xforms.Sequence(transformations=[field, *transform]).compute()
    return np.asarray(result.to(xforms.CoordinatesField).field)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize("extension", ["tfm", "h5", "mat"])
def test_native_spline_order_and_geometry(
    tmp_path: Path, ndim: int, order: int, extension: str
) -> None:
    sitk = pytest.importorskip("SimpleITK")
    if extension == "h5":
        pytest.importorskip("h5py")
    domain = sitk.Image([15, 17, 13][:ndim], sitk.sitkFloat32)
    domain.SetOrigin([2.0, -3.0, 5.0][:ndim])
    domain.SetSpacing([1.1, 1.7, 2.3][:ndim])
    native = sitk.BSplineTransformInitializer(domain, [3, 4, 3][:ndim], order)
    if order == 0:
        # ITK's initializer subtracts (order - 1) as an unsigned value,
        # placing order-0 grids far away. Set a valid coefficient origin.
        fixed = list(native.GetFixedParameters())
        fixed[ndim : 2 * ndim] = domain.GetOrigin()
        native.SetFixedParameters(fixed)
    params = 0.8 * np.sin(np.arange(native.GetNumberOfParameters()) * 0.37)
    native.SetParameters(params.tolist())
    path = tmp_path / f"spline.{extension}"
    sitk.WriteTransform(native, str(path))
    points = np.array(
        [
            domain.TransformContinuousIndexToPhysicalPoint(p[:ndim])
            for p in [(3.2, 4.6, 2.7), (8.1, 9.3, 6.4), (6.7, 8.2, 5.3)]
        ]
    )
    expected = np.array([native.TransformPoint(p) for p in points])
    if order == 0:
        # ITK 5's order-0 valid-region calculation also underflows and
        # TransformPoint returns identity. Check the encoded nearest
        # coefficient map independently instead of copying that defect.
        fixed = np.array(native.GetFixedParameters())
        shape = fixed[:ndim].astype(int)
        indices = np.floor(
            (points - fixed[ndim : 2 * ndim]) / fixed[2 * ndim : 3 * ndim]
            + 0.5
        ).astype(int)
        coefficients = params.reshape(ndim, *reversed(shape))
        offsets = np.array(
            [coefficients[(slice(None), *reversed(i))] for i in indices]
        )
        expected = points + offsets
    loaded = io.transformations.load(path, hint="itk")
    assert loaded[0].degree == order
    assert len(loaded[0].displacement.input.axes) == ndim
    np.testing.assert_allclose(_apply(loaded, points), expected, atol=2e-6)
    assert np.max(abs(expected - points)) > 0.01
    # MATLAB is writable: retain the order when encoding changed metadata.
    if extension == "mat":
        from bagof.magic import replace

        from brainhops.io.transformations.itk.mat import MatRaw

        block = replace(loaded[0], degree=(order + 1) % 4)
        path = tmp_path / "changed.mat"
        MatRaw(blocks=(block,)).save(path)
        back = io.transformations.load(path, hint="itk")
        assert back[0].degree == (order + 1) % 4
        readback = sitk.ReadTransform(str(path))
        assert sitk.BSplineTransform(readback).GetOrder() == (order + 1) % 4


@pytest.mark.parametrize("extension", ["tfm", "h5", "mat"])
def test_native_2d_displacement_reslices_original_png(
    tmp_path: Path, extension: str
) -> None:
    sitk = pytest.importorskip("SimpleITK")
    pil = pytest.importorskip("PIL.Image")
    if extension == "h5":
        pytest.importorskip("h5py")
    values = (np.arange(19 * 23).reshape(19, 23) % 251).astype("uint8")
    path = tmp_path / "source.png"
    pil.fromarray(values).save(path)
    field = np.zeros((19, 23, 2), dtype=float)
    field[..., 0], field[..., 1] = 2, -1
    native = sitk.DisplacementFieldTransform(
        sitk.GetImageFromArray(field, isVector=True)
    )
    transform_path = tmp_path / f"displacement.{extension}"
    sitk.WriteTransform(native, str(transform_path))
    original = io.images.load(path)
    transform = io.transformations.load(transform_path, hint="itk")
    actual = original(transform).reslice(original, degree=0, bound="constant")
    expected = sitk.Resample(
        sitk.ReadImage(str(path)), native, sitk.sitkNearestNeighbor
    )
    np.testing.assert_array_equal(
        np.asarray(actual.data), sitk.GetArrayFromImage(expected).T
    )


def test_order_suffix_is_only_valid_for_bspline() -> None:
    from brainhops.io.base.parsers import ParserContentError

    with pytest.raises(ParserContentError, match="Only B-spline"):
        _type_fields(_TRANSFORM_TYPE_RE.match("AffineTransform_double_3_3_2"))
