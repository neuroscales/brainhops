"""
Tests for the conventions shared by headerless raster image formats.

`brainhops.io.images.base._utils_raster` is backend-agnostic: it turns a
C-ordered array, its storage axes and whatever pixel size a file records
into the F-ordered data and the pixel-to-physical scaling every raster
reader returns, and back again for writers.
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling
from brainhops.datamodel.units import is_sampleunit
from brainhops.io.images.base import _utils_raster as raster

# ----------------------------------------------------------------------
#   AXES
# ----------------------------------------------------------------------


def test_storage_axes_codes() -> None:
    axes = raster.storage_axes("TZCYXS")
    assert [a.name for a in axes] == ["t", "z", "c", "y", "x", "s"]
    assert [a.type for a in axes] == [
        "time",
        "space",
        "channel",
        "space",
        "space",
        "channel",
    ]
    assert all(is_sampleunit(a.unit) for a in axes)


def test_storage_axes_samples_are_the_channel_axis() -> None:
    axes = raster.storage_axes("YXS")
    assert [a.name for a in axes] == ["y", "x", "c"]
    assert axes[-1].type == "channel"


def test_storage_axes_unknown_and_repeated_codes() -> None:
    axes = raster.storage_axes("QQYX")
    assert [a.name for a in axes] == ["q", "q1", "y", "x"]
    assert axes[0].type is None


def test_storage_axes_passes_axis_lists_through() -> None:
    axes = [Axis("y", "space"), Axis("x", "space")]
    assert raster.storage_axes(axes) == axes
    with pytest.raises(TypeError):
        raster.storage_axes(["y", "x"])


@pytest.mark.parametrize(
    "codes, expected",
    [
        ("YX", ["x", "y"]),
        ("ZYX", ["x", "y", "z"]),
        ("YXS", ["x", "y", "c"]),
        ("TZCYX", ["x", "y", "z", "t", "c"]),
        ("CYXS", ["x", "y", "s", "c"]),
        ("IYX", ["x", "y", "i"]),
    ],
)
def test_canonical_order(codes: str, expected: tx.Any) -> None:
    shape = tuple(range(2, 2 + len(codes)))
    array = np.zeros(shape)
    data, axes = raster.to_canonical(array, codes)
    assert [a.name for a in axes] == expected
    perm = raster.canonical_permutation(codes)
    assert data.shape == tuple(shape[p] for p in perm)


def test_to_canonical_is_a_view_in_f_order() -> None:
    array = np.arange(6 * 4 * 5).reshape(6, 4, 5)
    data, _ = raster.to_canonical(array, "ZYX")
    assert np.shares_memory(data, array)
    assert data.flags.f_contiguous
    assert data[1, 2, 3] == array[3, 2, 1]


def test_to_canonical_rgb_keeps_channels_last() -> None:
    array = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    data, axes = raster.to_canonical(array, "YXS")
    assert data.shape == (5, 4, 3)
    assert np.shares_memory(data, array)
    assert data[2, 1, 0] == array[1, 2, 0]


def test_to_canonical_checks_the_number_of_axes() -> None:
    with pytest.raises(ValueError):
        raster.to_canonical(np.zeros((2, 3)), "ZYX")


@pytest.mark.parametrize("codes", ["YX", "YXS", "TZCYX", "ZYXS"])
def test_storage_round_trip(codes: str) -> None:
    shape = tuple(range(2, 2 + len(codes)))
    array = np.random.default_rng(0).random(shape)
    data, axes = raster.to_canonical(array, codes)
    back = raster.to_storage(data, axes, codes)
    np.testing.assert_array_equal(back, array)


def test_storage_permutation_matches_unnamed_axes_by_role() -> None:
    axes = [
        Axis("dim0", "space"),
        Axis("dim1", "space"),
        Axis("rgb", "channel"),
    ]
    assert raster.storage_permutation(axes, "YXS") == [1, 0, 2]


def test_storage_permutation_refuses_missing_axes() -> None:
    axes = raster.default_axes(2)
    with pytest.raises(ValueError):
        raster.storage_permutation(axes, "YXS")
    with pytest.raises(ValueError):
        raster.storage_permutation(axes + [Axis("t", "time")], "YXS")


def test_default_axes() -> None:
    assert [a.name for a in raster.default_axes(2)] == ["x", "y"]
    assert [a.name for a in raster.default_axes(5)] == [
        "x",
        "y",
        "z",
        "t",
        "c",
    ]
    axes = raster.default_axes(3, channel=True)
    assert [a.name for a in axes] == ["x", "y", "c"]
    assert axes[-1].type == "channel"


def test_image_axes() -> None:
    axes = raster.default_axes(2)
    system = CoordinateSystem(axes=axes)
    assert raster.image_axes(2, system) == list(system.axes)
    assert raster.image_axes(3, system) is None
    assert raster.image_axes(2, None) is None


# ----------------------------------------------------------------------
#   SYSTEMS AND TRANSFORMATIONS
# ----------------------------------------------------------------------


def test_pixel_and_voxel_systems() -> None:
    _, axes = raster.to_canonical(np.zeros((2, 3, 4)), "YXS")
    system = raster.pixel_system(axes)
    assert system.name == "pixel"
    assert system.order == "F"
    assert all(is_sampleunit(a.unit) for a in system.axes)
    _, axes = raster.to_canonical(np.zeros((2, 3, 4)), "ZYX")
    assert raster.pixel_system(axes).name == "voxel"


def test_unknown_size_is_the_identity_with_no_unit() -> None:
    _, axes = raster.to_canonical(np.zeros((2, 3, 4)), "YXS")
    (xform,) = raster.raster_transformations(axes)
    assert isinstance(xform, Scaling)
    np.testing.assert_array_equal(xform.scale, [1, 1, 1])
    assert xform.output.name == "physical"
    assert all(a.unit is None for a in xform.output.axes)
    assert raster.physical_pixel_size(xform, axes) is None


def test_known_size_scales_the_spatial_axes() -> None:
    _, axes = raster.to_canonical(np.zeros((2, 3, 4)), "YXS")
    scales = {"x": (0.5, "mm"), "y": (0.25, "mm")}
    (xform,) = raster.raster_transformations(axes, scales)
    np.testing.assert_allclose(xform.scale, [0.5, 0.25, 1])
    units = [str(a.unit) if a.unit else None for a in xform.output.axes]
    assert units == ["millimeter", "millimeter", None]
    assert raster.physical_pixel_size(xform, axes) == {"x": 0.5, "y": 0.25}
    um = raster.physical_pixel_size(xform, axes, "um")
    assert um == pytest.approx({"x": 500.0, "y": 250.0})


def test_physical_pixel_size_of_an_affine() -> None:
    axes = raster.default_axes(2)
    pixel = raster.pixel_system(axes)
    physical = raster.physical_system(axes, {"x": "mm", "y": "mm"})
    matrix = np.array([[-0.5, 0.0, 10.0], [0.0, 2.0, 3.0]])
    xform = Affine(input=pixel, output=physical, matrix=matrix)
    assert raster.physical_pixel_size(xform, axes) == {"x": 0.5, "y": 2.0}
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    xform = Affine(input=pixel, output=physical, matrix=rotation)
    assert raster.physical_pixel_size(xform, axes) is None
    assert raster.physical_pixel_size(None, axes) is None


# ----------------------------------------------------------------------
#   PIXEL SIZE
# ----------------------------------------------------------------------


AXES2 = raster.default_axes(2)


def test_resolve_nothing_known() -> None:
    assert raster.resolve_pixel_size(AXES2) == {}
    # A unit alone does not make up a size.
    assert raster.resolve_pixel_size(AXES2, unit="mm") == {}


def test_resolve_metadata_and_unit_conversion() -> None:
    meta = {"x": (0.1, "mm"), "y": (0.2, "mm")}
    out = raster.resolve_pixel_size(AXES2, meta)
    assert {k: (v, str(u)) for k, (v, u) in out.items()} == {
        "x": (0.1, "millimeter"),
        "y": (0.2, "millimeter"),
    }
    out = raster.resolve_pixel_size(AXES2, meta, unit="um")
    assert out["x"][0] == pytest.approx(100.0)
    assert str(out["x"][1]) == "micrometer"


def test_resolve_metadata_without_unit_takes_the_given_one() -> None:
    out = raster.resolve_pixel_size(AXES2, {"x": (2.0, None)}, unit="um")
    assert out["x"][0] == 2.0
    assert str(out["x"][1]) == "micrometer"


@pytest.mark.parametrize(
    "pixel_size, expected",
    [
        (0.5, {"x": 0.5, "y": 0.5}),
        ((0.5, 2.0), {"x": 0.5, "y": 2.0}),
        ([3.0], {"x": 3.0, "y": 3.0}),
        ({"y": 4.0}, {"x": 0.1, "y": 4.0}),
    ],
)
def test_resolve_pixel_size_keyword(
    pixel_size: tx.Any, expected: tx.Any
) -> None:
    meta = {"x": (0.1, "mm"), "y": (0.2, "mm")}
    out = raster.resolve_pixel_size(AXES2, meta, pixel_size=pixel_size)
    assert {k: v for k, (v, _) in out.items()} == expected
    # The metadata's unit is kept when none is given.
    assert all(str(u) == "millimeter" for _, u in out.values())


def test_resolve_pixel_size_keyword_unit() -> None:
    out = raster.resolve_pixel_size(AXES2, pixel_size=1.5, unit="um")
    assert out["x"] == (1.5, out["x"][1])
    assert str(out["x"][1]) == "micrometer"
    out = raster.resolve_pixel_size(AXES2, pixel_size=1.5)
    assert out["x"] == (1.5, None)


def test_resolve_keeps_non_spatial_metadata() -> None:
    axes = raster.to_canonical(np.zeros((2, 3, 4)), "TYX")[1]
    meta = {"t": (0.5, "s"), "x": (1.0, "mm")}
    out = raster.resolve_pixel_size(axes, meta, pixel_size=2.0)
    assert out["t"][0] == 0.5
    assert out["x"][0] == out["y"][0] == 2.0


@pytest.mark.parametrize(
    "pixel_size", [0, -1.0, float("nan"), (1.0, 2.0, 3.0), {"z": 1.0}]
)
def test_resolve_refuses_bad_sizes(pixel_size: tx.Any) -> None:
    with pytest.raises(ValueError):
        raster.resolve_pixel_size(AXES2, pixel_size=pixel_size)


# ----------------------------------------------------------------------
#   DPI
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "dpi, default",
    [
        (None, True),
        ((72, 72), True),
        ((71.9996, 71.9996), True),
        ((72.009, 72.009), True),
        ((96.012, 96.012), True),
        ((96.0, 96.0), True),
        (1, True),
        ((0, 0), True),
        ((300, 300), False),
        ((72, 300), False),
    ],
)
def test_is_default_dpi(dpi: tx.Any, default: bool) -> None:
    assert raster.is_default_dpi(dpi) is default


def test_dpi_conversions() -> None:
    assert raster.dpi_to_size(25.4) == pytest.approx(1.0)
    assert raster.dpi_to_size(254, "um") == pytest.approx(100.0)
    assert raster.size_to_dpi(0.1) == pytest.approx(254.0)
    assert raster.size_to_dpi(1, "inch") == pytest.approx(1.0)
    assert raster.convert_length(1, "inch", "mm") == pytest.approx(25.4)
    with pytest.raises(ValueError):
        raster.dpi_to_size(0)
    with pytest.raises(ValueError):
        raster.convert_length(1, "mm", "s")
