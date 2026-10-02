"""
Tests for TIFF images (plain, BigTIFF, OME-TIFF, ImageJ, pyramidal) read
and written with tifffile, and for the Pillow fallback.

Fixtures are generated with tifffile (and Pillow, for files tifffile
always tags), so that each test states the exact layout and metadata of
the file it reads.
"""

import io
import math
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

tifffile = pytest.importorskip("tifffile")

import brainhops.io as bio  # noqa: E402
from brainhops.datamodel.images import (  # noqa: E402
    MultiScaleImage,
    SingleScaleImage,
)
from brainhops.datamodel.transformations import Affine, Scaling  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserNotImplementedError,
    WriterError,
)
from brainhops.io.images import load  # noqa: E402
from brainhops.io.images.base import _utils_raster as raster  # noqa: E402
from brainhops.io.images.tiff import (  # noqa: E402
    TiffImage,
    TiffMultiScaleImage,
)
from brainhops.io.images.tiff import _image as tiff_image  # noqa: E402
from brainhops.io.images.tiff import _utils as backend  # noqa: E402

RNG = np.random.default_rng(0)
GREY = RNG.integers(0, 2**16, (7, 12), dtype=np.uint16)  # rows, columns
STACK = RNG.integers(0, 2**16, (3, 2, 7, 12), dtype=np.uint16)  # ZCYX
RGB = RNG.integers(0, 256, (7, 12, 3), dtype=np.uint8)


def _write(tmp_path: Path, name: str, array: np.ndarray, **kw) -> Path:
    path = tmp_path / name
    tifffile.imwrite(path, array, **kw)
    return path


def _bytes(array: np.ndarray, **kw) -> bytes:
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array, **kw)
    return buffer.getvalue()


def _scale(image: tx.Any) -> np.ndarray:
    xform = image.transformation
    if isinstance(xform, Scaling):
        return np.asarray(xform.scale, dtype=float)
    return np.diag(np.asarray(xform.matrix, dtype=float)[:, :-1])


def _shift(image: tx.Any) -> np.ndarray:
    xform = image.transformation
    if isinstance(xform, Scaling):
        return np.zeros(len(xform.scale))
    return np.asarray(xform.matrix, dtype=float)[:, -1]


def _units(image: tx.Any) -> tx.List[tx.Optional[str]]:
    out = []
    for axis in image.transformation.output.axes:
        unit = axis.unit
        out.append(None if unit is None else str(unit))
    return out


def _names(image: tx.Any) -> tx.List[str]:
    return [axis.name for axis in image.transformation.input.axes]


# ----------------------------------------------------------------------
#   READING: DATA
# ----------------------------------------------------------------------


def test_read_plain_is_a_transposed_memory_map(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.tif", GREY))
    assert isinstance(image, TiffImage)
    assert image.dialect is None
    np.testing.assert_array_equal(image.data, GREY.T)
    # A view of a memory map, F-ordered: nothing copied.
    assert isinstance(image.data.base, np.memmap)
    assert image.data.flags.f_contiguous
    assert _names(image) == ["x", "y"]
    assert image.transformation.input.name == "pixel"
    # Copy-on-write: writing does not change the file.
    image.data[0, 0] = 1
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "a.tif"), GREY)


def test_read_is_lazy(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.tif", GREY))
    assert not hasattr(image, "_cache_data")
    _ = image.data
    assert hasattr(image, "_cache_data")


def test_read_compressed_is_decoded(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "a.tif", STACK, compression="zlib", metadata={"axes": "ZCYX"}
    )
    # `lazy=False`: with dask as the array backend, the pixels would
    # otherwise be read as a dask array, which has no `base`.
    image = load(path, lazy=False)
    assert not isinstance(image.data.base, np.memmap)
    np.testing.assert_array_equal(image.data, STACK.transpose(3, 2, 0, 1))


def test_read_without_mmap(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.tif", GREY), mmap=False, lazy=False)
    assert not isinstance(image.data.base, np.memmap)
    np.testing.assert_array_equal(image.data, GREY.T)


def test_read_lazily_with_dask(tmp_path: Path) -> None:
    da = pytest.importorskip("dask.array")
    pytest.importorskip("zarr")
    path = _write(
        tmp_path,
        "a.tif",
        STACK,
        compression="zlib",
        tile=(16, 16),
        metadata={"axes": "ZCYX"},
    )
    image = load(path, lazy=True)
    assert isinstance(image.data, da.Array)
    np.testing.assert_array_equal(
        np.asarray(image.data), STACK.transpose(3, 2, 0, 1)
    )


def test_read_rgb(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.tif", RGB, photometric="rgb"))
    assert image.storage_axes == "YXS"
    assert _names(image) == ["x", "y", "c"]
    np.testing.assert_array_equal(image.data, RGB.transpose(1, 0, 2))


def test_read_from_bytes_and_stream() -> None:
    content = _bytes(GREY)
    for source in (content, io.BytesIO(content)):
        image = bio.load(source)
        assert isinstance(image, TiffImage)
        np.testing.assert_array_equal(image.data, GREY.T)


def test_read_bigtiff(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.tif", GREY, bigtiff=True)
    assert path.read_bytes()[:4] == b"II+\0"
    image = load(path)
    np.testing.assert_array_equal(image.data, GREY.T)


def test_read_series(tmp_path: Path) -> None:
    path = tmp_path / "a.tif"
    volume = RNG.random((2, 5, 6)).astype(np.float32)
    with tifffile.TiffWriter(path) as writer:
        writer.write(GREY)
        writer.write(volume, metadata={"axes": "ZYX"})
    first = load(path)
    assert first.n_series == 2 and first.series == 0
    np.testing.assert_array_equal(first.data, GREY.T)
    second = load(path, series=1)
    assert second.series == 1
    np.testing.assert_array_equal(second.data, volume.transpose(2, 1, 0))
    assert _names(second) == ["x", "y", "z"]
    assert load(path, series=-1).series == 1
    with pytest.raises(Exception, match="no series 2"):
        TiffImage.from_filename(path, series=2)


def test_unknown_option(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="frame"):
        TiffImage.from_filename(_write(tmp_path, "a.tif", GREY), frame=1)


# ----------------------------------------------------------------------
#   READING: RESOLUTION TAGS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "resolution, unit, sizes, expected_unit",
    [
        ((100, 50), "CENTIMETER", [0.01, 0.02], "centimeter"),
        ((254, 254), "INCH", [0.1, 0.1], "millimeter"),
        ((2, 4), 4, [0.5, 0.25], "millimeter"),
        ((2, 4), 5, [0.5, 0.25], "micrometer"),
    ],
)
def test_resolution_tags(
    tmp_path: Path,
    resolution: tx.Any,
    unit: tx.Any,
    sizes: tx.List[float],
    expected_unit: str,
) -> None:
    path = _write(
        tmp_path, "a.tif", GREY, resolution=resolution, resolutionunit=unit
    )
    image = load(path)
    np.testing.assert_allclose(_scale(image), sizes)
    assert _units(image) == [expected_unit] * 2


@pytest.mark.parametrize(
    "options",
    [
        {},  # tifffile's default: 1/1, no unit
        {"resolution": (2, 2), "resolutionunit": "NONE"},  # aspect only
        {"resolution": (72, 72), "resolutionunit": "INCH"},  # placeholder
        {"resolution": (96, 96), "resolutionunit": "INCH"},  # placeholder
        {"resolution": (1, 1), "resolutionunit": "CENTIMETER"},
    ],
)
def test_resolution_tags_unknown(tmp_path: Path, options: tx.Any) -> None:
    image = load(_write(tmp_path, "a.tif", GREY, **options))
    np.testing.assert_array_equal(_scale(image), [1, 1])
    assert _units(image) == [None, None]


def test_resolution_tags_absent(tmp_path: Path) -> None:
    # Pillow writes no resolution tags at all; tifffile then reports 1
    # pixel per inch, which must not be taken for a size.
    from PIL import Image

    path = tmp_path / "a.tif"
    Image.fromarray(GREY).save(path)
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].tags.valueof(282) is None
        assert backend.resolution_scales(tif.pages[0]) == {}
    image = load(path)
    assert _units(image) == [None, None]


class _Tags:
    def __init__(self, values: tx.Dict[int, tx.Any]) -> None:
        self.values_ = values

    def valueof(self, code: int, default: tx.Any = None) -> tx.Any:
        return self.values_.get(code, default)


class _Page:
    def __init__(self, values: tx.Dict[int, tx.Any]) -> None:
        self.tags = _Tags(values)


def test_resolution_unit_defaults_to_inch() -> None:
    page = _Page({282: (300, 1), 283: (150, 1)})
    scales = backend.resolution_scales(page)
    assert scales["x"] == (pytest.approx(25.4 / 300), "mm")
    assert scales["y"] == (pytest.approx(25.4 / 150), "mm")
    # One tag alone is not a resolution.
    assert backend.resolution_scales(_Page({282: (300, 1)})) == {}


# ----------------------------------------------------------------------
#   READING: IMAGEJ
# ----------------------------------------------------------------------


def _imagej(tmp_path: Path, **metadata) -> Path:
    data = RNG.integers(0, 255, (4, 3, 2, 7, 12), dtype=np.uint8)  # TZCYX
    meta = {"axes": "TZCYX"}
    meta.update(metadata)
    return _write(
        tmp_path,
        "ij.tif",
        data,
        imagej=True,
        resolution=(1 / 0.5, 1 / 0.25),
        metadata=meta,
    )


def test_imagej_hyperstack(tmp_path: Path) -> None:
    path = _imagej(tmp_path, unit="micron", spacing=2.0, finterval=0.5)
    image = load(path)
    assert image.dialect == "imagej"
    assert image.imagej_metadata["spacing"] == 2.0
    assert image.storage_axes == "TZCYX"
    assert _names(image) == ["x", "y", "z", "t", "c"]
    assert image.data.shape == (12, 7, 3, 4, 2)
    expected = tifffile.imread(path).transpose(4, 3, 1, 0, 2)
    np.testing.assert_array_equal(image.data, expected)
    np.testing.assert_allclose(_scale(image), [0.5, 0.25, 2.0, 0.5, 1])
    units = _units(image)
    assert units[:3] == ["micrometer"] * 3
    assert units[3] == "second" and units[4] is None


def test_imagej_negative_spacing_flips_z(tmp_path: Path) -> None:
    image = load(_imagej(tmp_path, unit="um", spacing=-3.0))
    np.testing.assert_allclose(_scale(image)[:3], [0.5, 0.25, -3.0])


def test_imagej_escaped_micro_sign(tmp_path: Path) -> None:
    image = load(_imagej(tmp_path, unit="\\u00B5m", spacing=1.0))
    assert _units(image)[:3] == ["micrometer"] * 3


def test_imagej_without_unit_is_unknown(tmp_path: Path) -> None:
    image = load(_imagej(tmp_path, spacing=1.0))
    np.testing.assert_allclose(_scale(image), [1, 1, 1, 1, 1])
    assert _units(image) == [None] * 5


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("micron", "um"),
        ("um", "um"),
        ("µm", "um"),
        ("µm", "um"),
        ("\\u00B5m", "um"),
        ("nm", "nm"),
        ("Å", "angstrom"),
        ("inch", "inch"),
        ("pixel", None),
        ("reference frame", None),
        (None, None),
    ],
)
def test_length_unit(spelling: tx.Any, expected: tx.Any) -> None:
    assert backend.length_unit(spelling) == expected


# ----------------------------------------------------------------------
#   READING: OME-TIFF
# ----------------------------------------------------------------------


def _ome(tmp_path: Path, name: str = "a.ome.tif", **metadata) -> Path:
    meta = {"axes": "ZCYX"}
    meta.update(metadata)
    return _write(tmp_path, name, STACK, metadata=meta)


def test_ome_physical_sizes(tmp_path: Path) -> None:
    path = _ome(
        tmp_path,
        PhysicalSizeX=0.5,
        PhysicalSizeXUnit="nm",
        PhysicalSizeY=0.25,
        PhysicalSizeZ=2.0,
        PhysicalSizeZUnit="mm",
    )
    image = load(path)
    assert image.dialect == "ome"
    assert "<OME" in image.ome_xml
    assert _names(image) == ["x", "y", "z", "c"]
    np.testing.assert_array_equal(image.data, STACK.transpose(3, 2, 0, 1))
    np.testing.assert_allclose(_scale(image), [0.5, 0.25, 2.0, 1])
    # The unit of PhysicalSizeY defaults to micrometres.
    assert _units(image) == ["nanometer", "micrometer", "millimeter", None]


def test_ome_dimension_order(tmp_path: Path) -> None:
    data = RNG.integers(0, 9, (2, 3, 7, 12), dtype=np.uint8)  # CZYX
    path = _write(
        tmp_path,
        "a.ome.tif",
        data,
        metadata={"axes": "CZYX", "PhysicalSizeZ": 4.0},
    )
    with tifffile.TiffFile(path) as tif:
        assert 'DimensionOrder="XYZCT"' in tif.ome_metadata
    image = load(path)
    assert image.storage_axes == "CZYX"
    np.testing.assert_array_equal(image.data, data.transpose(3, 2, 1, 0))
    np.testing.assert_allclose(_scale(image), [1, 1, 4.0, 1])


def test_ome_plane_positions_are_the_origin(tmp_path: Path) -> None:
    planes = {
        "PositionX": [10.0] * 6,
        "PositionXUnit": ["mm"] * 6,
        "PositionY": [20.0] * 6,
        "PositionZ": [5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
    }
    path = _ome(
        tmp_path,
        PhysicalSizeX=0.5,
        PhysicalSizeY=0.5,
        PhysicalSizeZ=2.0,
        Plane=planes,
    )
    image = load(path)
    assert isinstance(image.transformation, Affine)
    # x in millimetres, converted to the micrometres of the pixel size; y
    # and z have no unit, and are taken in micrometres.
    np.testing.assert_allclose(_shift(image), [10000.0, 20.0, 5.0, 0.0])
    # The pixel centre at index 0 lands on the origin.
    assert load(path, origin=False).transformation.__class__ is Scaling


def test_ome_time_increment(tmp_path: Path) -> None:
    data = RNG.integers(0, 9, (5, 7, 12), dtype=np.uint8)
    path = _write(
        tmp_path,
        "a.ome.tif",
        data,
        metadata={"axes": "TYX", "TimeIncrement": 2.0},
    )
    image = load(path)
    assert _names(image) == ["x", "y", "t"]
    np.testing.assert_allclose(_scale(image), [1, 1, 2.0])
    assert _units(image) == [None, None, "second"]


def test_ome_wins_over_resolution_tags_axis_by_axis(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "a.ome.tif",
        STACK,
        resolution=(1000, 1000),
        resolutionunit="CENTIMETER",
        metadata={"axes": "ZCYX", "PhysicalSizeX": 3.0, "PhysicalSizeZ": 2},
    )
    with tifffile.TiffFile(path) as tif:
        meta = backend.series_metadata(tif, 0)
    assert meta.sources == {"x": "ome", "y": "resolution", "z": "ome"}
    image = load(path)
    np.testing.assert_allclose(_scale(image), [3.0, 0.001, 2.0, 1])
    assert _units(image)[:3] == ["micrometer", "centimeter", "micrometer"]


def test_ome_multi_series(tmp_path: Path) -> None:
    path = tmp_path / "a.ome.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(GREY, metadata={"PhysicalSizeX": 2.0, "Name": "a"})
        writer.write(
            np.ones((2, 5, 6), np.float32),
            metadata={"axes": "ZYX", "PhysicalSizeX": 3.0, "Name": "b"},
        )
    second = load(path, series=1)
    assert second.n_series == 2
    np.testing.assert_allclose(_scale(second)[0], 3.0)


def test_pixel_size_and_unit_override(tmp_path: Path) -> None:
    path = _ome(tmp_path, PhysicalSizeX=0.5, PhysicalSizeY=0.5)
    image = load(path, unit="mm")
    np.testing.assert_allclose(_scale(image)[:2], [0.0005, 0.0005])
    image = load(path, pixel_size={"z": 7.0})
    np.testing.assert_allclose(_scale(image)[:3], [0.5, 0.5, 7.0])
    image = load(path, pixel_size=(1, 2, 3), unit="mm")
    np.testing.assert_allclose(_scale(image)[:3], [1, 2, 3])
    assert _units(image)[:3] == ["millimeter"] * 3


# ----------------------------------------------------------------------
#   READING: PYRAMIDS
# ----------------------------------------------------------------------


def _pyramid(tmp_path: Path, name: str = "p.ome.tif", **metadata) -> Path:
    data = RNG.integers(0, 2**16, (2, 32, 40), dtype=np.uint16)  # CYX
    path = tmp_path / name
    meta = {"axes": "CYX"}
    meta.update(metadata)
    with tifffile.TiffWriter(path) as writer:
        writer.write(data, subifds=2, metadata=meta, tile=(16, 16))
        writer.write(data[:, ::2, ::2], subfiletype=1, tile=(16, 16))
        writer.write(data[:, ::4, ::4], subfiletype=1, tile=(16, 16))
    return path


def test_pyramid_is_multiscale(tmp_path: Path) -> None:
    path = _pyramid(tmp_path, PhysicalSizeX=0.5, PhysicalSizeY=0.5)
    image = load(path)
    assert isinstance(image, TiffMultiScaleImage)
    assert isinstance(image, MultiScaleImage)
    assert image.nscales == 3
    assert all(isinstance(level, TiffImage) for level in image.images)
    # Opening the pyramid reads no level.
    assert not any(hasattr(level, "_cache_data") for level in image.images)
    shapes = [level.data.shape for level in image.images]
    assert shapes == [(40, 32, 2), (20, 16, 2), (10, 8, 2)]
    full = tifffile.imread(path)
    np.testing.assert_array_equal(image.data, full.transpose(2, 1, 0))
    # Sizes grow by the downsampling factor; centres move by (f - 1) / 2.
    for level, factor in zip(image.images, (1, 2, 4)):
        np.testing.assert_allclose(
            _scale(level), [0.5 * factor, 0.5 * factor, 1]
        )
        shift = 0.5 * (factor - 1) / 2
        np.testing.assert_allclose(_shift(level), [shift, shift, 0])
        assert _units(level)[:2] == ["micrometer"] * 2
    assert not image.transformations


def test_pyramid_level_edges_coincide(tmp_path: Path) -> None:
    # A plain TIFF pyramid, of unknown pixel size.
    data = RNG.integers(0, 2**16, (32, 40), dtype=np.uint16)
    path = tmp_path / "p.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(data, subifds=2)
        writer.write(data[::2, ::2], subfiletype=1)
        writer.write(data[::4, ::4], subfiletype=1)
    image = load(path)
    assert isinstance(image, TiffMultiScaleImage)
    base, coarse = image.images[0], image.images[2]
    # The left edge of the first pixel, and the right edge of the last.
    for level in (base, coarse):
        s, t = _scale(level)[0], _shift(level)[0]
        n = level.data.shape[0]
        assert s * -0.5 + t == pytest.approx(-0.5)
        assert s * (n - 0.5) + t == pytest.approx(40 - 0.5)


def test_pyramid_level_as_single_scale(tmp_path: Path) -> None:
    path = _pyramid(tmp_path, PhysicalSizeX=0.5, PhysicalSizeY=0.5)
    level = load(path, level=1)
    assert isinstance(level, TiffImage)
    assert level.level == 1 and level.n_levels == 3
    assert level.data.shape == (20, 16, 2)
    np.testing.assert_allclose(_scale(level), [1.0, 1.0, 1])
    np.testing.assert_allclose(_shift(level), [0.25, 0.25, 0])
    with pytest.raises(Exception, match="no level 3"):
        TiffImage.from_filename(path, level=3)


def test_pyramid_sniff(tmp_path: Path) -> None:
    pyramid = _pyramid(tmp_path)
    plain = _write(tmp_path, "a.tif", GREY)
    assert TiffMultiScaleImage.sniff(pyramid) == Confidence.CERTAIN
    assert TiffMultiScaleImage.sniff(pyramid, level=0) == Confidence.NO
    assert TiffMultiScaleImage.sniff(plain) == Confidence.NO
    assert TiffImage.sniff(pyramid) == Confidence.LIKELY
    assert TiffImage.sniff(b"not a tiff file") == Confidence.NO


# ----------------------------------------------------------------------
#   PILLOW FALLBACK
# ----------------------------------------------------------------------


def test_pillow_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PIL")
    monkeypatch.setattr(tiff_image, "HAS_TIFFFILE", False)
    content = _bytes(GREY, resolution=(100, 50), resolutionunit="CENTIMETER")
    image = TiffImage.from_bytes(content)
    assert isinstance(image.data, np.ndarray)
    np.testing.assert_array_equal(image.data, GREY.T)
    np.testing.assert_allclose(_scale(image), [0.1, 0.2])
    assert _units(image) == ["millimeter"] * 2
    # Placeholders are unknown there too.
    image = TiffImage.from_bytes(_bytes(GREY))
    assert _units(image) == [None, None]
    with pytest.raises(ParserNotImplementedError):
        TiffImage.from_bytes(content, series=1)
    with pytest.raises(ParserNotImplementedError):
        TiffMultiScaleImage.from_bytes(content)
    assert TiffMultiScaleImage.sniff(content) == Confidence.NO


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _image(
    data: np.ndarray,
    scales: tx.Optional[tx.Dict[str, tx.Any]] = None,
    axes: tx.Optional[tx.List[tx.Any]] = None,
) -> SingleScaleImage:
    axes = axes or raster.default_axes(data.ndim)
    return SingleScaleImage(
        data=data, transformations=raster.raster_transformations(axes, scales)
    )


def _dialect(path: tx.Any) -> tx.Tuple[bool, bool]:
    with tifffile.TiffFile(path) as tif:
        return bool(tif.is_ome), bool(tif.is_imagej)


def test_write_plain_when_size_is_unknown(tmp_path: Path) -> None:
    data = RNG.random((12, 7)).astype(np.float64)
    path = tmp_path / "a.tif"
    bio.save(_image(data), path)
    assert _dialect(path) == (False, False)
    back = load(path)
    np.testing.assert_array_equal(back.data, data)
    assert _units(back) == [None, None]


def test_write_ome_when_size_is_known(tmp_path: Path) -> None:
    data = RNG.integers(0, 9, (12, 7, 3), dtype=np.int16)  # x, y, z
    image = _image(data, {"x": (0.5, "mm"), "y": (0.25, "mm"), "z": (2, "um")})
    path = tmp_path / "a.tif"
    bio.save(image, path)
    assert _dialect(path) == (True, False)
    back = load(path)
    assert back.storage_axes == "ZYX"
    np.testing.assert_array_equal(back.data, data)
    np.testing.assert_allclose(_scale(back), [0.5, 0.25, 2])
    assert _units(back) == ["millimeter", "millimeter", "micrometer"]
    # The resolution tags are written too, for readers that know no OME.
    with tifffile.TiffFile(path) as tif:
        assert backend.resolution_scales(tif.pages[0])["x"] == (
            pytest.approx(0.05),
            "cm",
        )


def test_write_ome_by_name(tmp_path: Path) -> None:
    path = tmp_path / "a.ome.tiff"
    bio.save(_image(GREY.T.copy()), path)
    assert _dialect(path) == (True, False)


def test_write_ome_round_trip(tmp_path: Path) -> None:
    planes = {"PositionX": [1.0] * 6, "PositionZ": [3.0, 3, 5, 5, 7, 7]}
    path = _ome(
        tmp_path,
        PhysicalSizeX=0.5,
        PhysicalSizeY=0.5,
        PhysicalSizeZ=2.0,
        Plane=planes,
        Name="sample",
        Channel={"Name": ["dapi", "gfp"]},
    )
    image = load(path)
    out = tmp_path / "b.tif"
    image.save(out)
    back = load(out)
    assert back.dialect == "ome"
    assert back.storage_axes == "ZCYX"
    np.testing.assert_array_equal(back.data, image.data)
    np.testing.assert_allclose(_scale(back), _scale(image))
    np.testing.assert_allclose(_shift(back), _shift(image))
    assert 'Name="sample"' in back.ome_xml
    assert 'Name="gfp"' in back.ome_xml
    # The position of every plane is written, not only of the first.
    assert 'PositionZ="7.0"' in back.ome_xml


def test_write_imagej_round_trip(tmp_path: Path) -> None:
    path = _imagej(
        tmp_path,
        unit="micron",
        spacing=-2.0,
        finterval=0.5,
        Ranges=(0.0, 255.0, 0.0, 100.0),
    )
    image = load(path)
    out = tmp_path / "b.tif"
    image.save(out)
    assert _dialect(out) == (False, True)
    back = load(out)
    np.testing.assert_array_equal(back.data, image.data)
    np.testing.assert_allclose(_scale(back), _scale(image))
    assert back.imagej_metadata["spacing"] == -2.0
    assert back.imagej_metadata["finterval"] == 0.5
    assert tuple(back.imagej_metadata["Ranges"]) == (0.0, 255.0, 0.0, 100.0)


def test_write_imagej_refuses_what_imagej_cannot_store(tmp_path: Path) -> None:
    image = _image(RNG.random((12, 7)))  # float64
    with pytest.raises(WriterError, match="ImageJ"):
        image_tiff = TiffImage.from_instance(image)
        image_tiff.to_bytes(dialect="imagej")


def test_write_explicit_plain_with_resolution(tmp_path: Path) -> None:
    image = TiffImage.from_instance(
        _image(GREY.T.copy(), {"x": (0.1, "mm"), "y": (0.2, "mm")})
    )
    content = image.to_bytes(dialect="plain")
    back = TiffImage.from_bytes(content)
    assert back.dialect is None
    np.testing.assert_allclose(_scale(back), [0.01, 0.02])
    assert _units(back) == ["centimeter"] * 2
    with pytest.raises(WriterError, match="dialect"):
        image.to_bytes(dialect="png")


def test_write_rgb_as_samples(tmp_path: Path) -> None:
    data = RGB.transpose(1, 0, 2).copy()
    axes = raster.default_axes(3, channel=True)
    path = tmp_path / "a.tif"
    bio.save(_image(data, axes=axes), path)
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "YXS"
        assert tif.pages[0].photometric == 2  # RGB
    np.testing.assert_array_equal(load(path).data, data)


def test_write_round_trip_keeps_storage_order(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.tif", STACK, metadata={"axes": "ZCYX"})
    image = load(path)
    out = tmp_path / "b.tif"
    image.save(out)
    with tifffile.TiffFile(out) as tif:
        assert tif.series[0].axes == "ZCYX"
    np.testing.assert_array_equal(tifffile.imread(out), STACK)


def test_write_tags_round_trip(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "a.tif",
        GREY,
        software="scope 1.0",
        extratags=[(315, "s", 0, "someone", True)],
    )
    image = load(path)
    assert image.tags["Software"] == "scope 1.0"
    out = tmp_path / "b.tif"
    image.save(out)
    back = load(out)
    assert back.tags["Software"] == "scope 1.0"
    assert back.tags["Artist"] == "someone"


def test_write_bigtiff(tmp_path: Path) -> None:
    image = TiffImage.from_instance(_image(GREY.T.copy()))
    assert image.to_bytes()[:4] == b"II*\0"
    assert image.to_bytes(bigtiff=True)[:4] == b"II+\0"
    assert tiff_image._bigtiff(2**32, None)
    assert not tiff_image._bigtiff(2**20, None)


def test_write_options_and_streams(tmp_path: Path) -> None:
    image = TiffImage.from_instance(_image(GREY.T.copy()))
    buffer = io.BytesIO()
    image.save(buffer, compression="zlib")
    with tifffile.TiffFile(io.BytesIO(buffer.getvalue())) as tif:
        assert tif.pages[0].compression == 8  # ADOBE_DEFLATE
    np.testing.assert_array_equal(load(buffer.getvalue()).data, GREY.T)


def test_write_dask_data(tmp_path: Path) -> None:
    da = pytest.importorskip("dask.array")
    data = da.from_array(GREY.T.copy(), chunks=4)
    path = tmp_path / "a.tif"
    bio.save(_image(data), path)
    np.testing.assert_array_equal(load(path).data, GREY.T)


def test_write_refusals(tmp_path: Path) -> None:
    with pytest.raises(WriterError, match="no data"):
        TiffImage().to_bytes()
    axes = [raster.Axis(n, "space", unit="sample") for n in "abcd"]
    image = _image(np.zeros((2, 2, 2, 2), np.uint8), axes=axes)
    with pytest.raises(WriterError, match="three spatial"):
        TiffImage.from_instance(image).to_bytes()
    assert not (tmp_path / "a.tif").exists()


def test_write_flip_is_kept_by_imagej_dropped_by_ome(tmp_path: Path) -> None:
    data = RNG.integers(0, 9, (12, 7, 3), dtype=np.uint8)
    image = _image(data, {"x": (1, "um"), "y": (1, "um"), "z": (-2, "um")})
    tiff = TiffImage.from_instance(image)
    back = TiffImage.from_bytes(tiff.to_bytes(dialect="imagej"))
    np.testing.assert_allclose(_scale(back), [1, 1, -2])
    back = TiffImage.from_bytes(tiff.to_bytes(dialect="ome"))
    np.testing.assert_allclose(_scale(back), [1, 1, 2])


def test_write_pyramid_round_trip(tmp_path: Path) -> None:
    path = _pyramid(tmp_path, PhysicalSizeX=0.5, PhysicalSizeY=0.5)
    pyramid = load(path)
    out = tmp_path / "b.ome.tif"
    bio.save(pyramid, out)
    back = load(out)
    assert isinstance(back, TiffMultiScaleImage)
    assert back.nscales == 3
    for a, b in zip(pyramid.images, back.images):
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_allclose(_scale(a), _scale(b))
        np.testing.assert_allclose(_shift(a), _shift(b))


def test_write_multiscale_from_datamodel(tmp_path: Path) -> None:
    base = RNG.integers(0, 9, (16, 8), dtype=np.uint8)
    levels = [
        _image(base, {"x": (1, "um"), "y": (1, "um")}),
        _image(base[::2, ::2].copy(), {"x": (2, "um"), "y": (2, "um")}),
    ]
    path = tmp_path / "a.tif"
    bio.save(MultiScaleImage(images=levels), path)
    back = load(path)
    assert isinstance(back, TiffMultiScaleImage)
    np.testing.assert_array_equal(back.images[1].data, base[::2, ::2])
    np.testing.assert_allclose(_scale(back.images[1]), [2, 2])
    assert math.isclose(_shift(back.images[1])[0], 0.5)
