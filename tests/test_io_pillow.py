"""Tests for raster images (PNG, JPEG, BMP, GIF, WebP, ...) read via Pillow.

The fixtures are written with Pillow, so that each test states the exact
pixels, mode and resolution.
"""

import io
import os
import pickle
import subprocess
import sys
import threading
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

import brainhops.io as bio  # noqa: E402
from brainhops._core.backends import backend  # noqa: E402
from brainhops.datamodel.axes import Axis  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.transformations import Scaling  # noqa: E402
from brainhops.datamodel.units import is_indexunit  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.images import load  # noqa: E402
from brainhops.io.images.base import _utils_raster as raster  # noqa: E402
from brainhops.io.images.pillow import (  # noqa: E402
    PillowImage,
    PillowMetadata,
    PillowRaw,
)
from brainhops.io.images.pillow import _data as pillow_data  # noqa: E402
from brainhops.io.images.pillow._data import PillowProxy  # noqa: E402
from brainhops.io.images.pillow._utils import (  # noqa: E402
    array_to_pillow,
    format_for_name,
    pillow_dpi,
    pillow_to_array,
    read_pillow,
    sniff_pillow,
)

RNG = np.random.default_rng(0)
GREY = RNG.integers(0, 256, (7, 12), dtype=np.uint8)  # rows, columns
RGB = RNG.integers(0, 256, (7, 12, 3), dtype=np.uint8)
RGBA = RNG.integers(0, 256, (7, 12, 4), dtype=np.uint8)
GREY16 = RNG.integers(0, 2**16, (7, 12), dtype=np.uint16)


def _write(tmp_path, name, array, **options):  # noqa: ANN001, ANN202
    path = tmp_path / name
    Image.fromarray(array).save(path, **options)
    return path


def _bytes(array, format, **options):  # noqa: ANN001, ANN202
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=format, **options)
    return buffer.getvalue()


def _read(path):  # noqa: ANN001, ANN202
    with Image.open(path) as im:
        im.load()
        return np.asarray(im), dict(im.info), im.format, im.mode


# ----------------------------------------------------------------------
#   READING: DATA
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "array, name, mode",
    [
        (GREY, "a.png", "L"),
        (RGB, "a.png", "RGB"),
        (RGBA, "a.png", "RGBA"),
        (GREY16, "a.png", "I;16"),
        (RGB, "a.bmp", "RGB"),
        (GREY, "a.pgm", "L"),
        (RGB, "a.ppm", "RGB"),
        (RGB, "a.tga", "RGB"),
        (RGB, "a.webp", "RGB"),
    ],
)
def test_read_lossless_is_transposed_view(
    tmp_path: Path, array: np.ndarray, name: str, mode: str
) -> None:
    if name.endswith(".webp"):
        path = _write(tmp_path, name, array, lossless=True)
    else:
        path = _write(tmp_path, name, array)
    image = load(path)
    assert isinstance(image, PillowImage)
    assert image.metadata.raw.mode == mode
    if array.ndim == 2:
        expected = array.T
    else:
        expected = array.transpose(1, 0, 2)
    assert image.data.shape == expected.shape
    assert image.data.dtype == array.dtype
    np.testing.assert_array_equal(image.data, expected)
    # x is the column and y the row.
    data = np.asarray(image.data)
    assert data[3, 2, ...].tolist() == array[2, 3, ...].tolist()


def test_read_axes_and_systems(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.png", RGB))
    (xform,) = image.transformations
    assert isinstance(xform, Scaling)
    pixel = xform.input
    assert pixel.name == "pixel"
    assert pixel.order == "F"
    assert [a.name for a in pixel.axes] == ["x", "y", "c"]
    assert [a.type for a in pixel.axes] == ["space", "space", "channel"]
    assert all(is_indexunit(a.unit) for a in pixel.axes)
    # With an unknown pixel size, the scaling is an identity onto unit-less
    # axes.
    np.testing.assert_array_equal(xform.scale, [1, 1, 1])
    assert xform.output.name == "physical"
    assert all(a.unit is None for a in xform.output.axes)
    # The geometry covers the whole data.
    assert image.geometry.grid.shape == image.data.shape


def test_read_grey_has_no_channel_axis(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.png", GREY))
    assert [a.name for a in image.transformation.input.axes] == ["x", "y"]
    assert image.data.ndim == 2


def test_read_data_is_a_view_of_the_decoded_array(tmp_path: Path) -> None:
    # With NumPy, the data is a view of the decoded frame, and with Dask it
    # is a lazy array of the frame.
    with backend("numpy"):
        image = load(_write(tmp_path, "a.png", GREY))
        # F-ordered grey data is the transpose of the C-ordered array.
        values = image.to_backend()
        assert values.flags.f_contiguous
        assert values.base is not None
        # The data refuse an edit in place, since the writer writes
        # `rawdata`.
        with pytest.raises(TypeError):
            image.data[0, 0] = 1


def test_read_bilevel(tmp_path: Path) -> None:
    bits = GREY > 128
    path = tmp_path / "a.png"
    Image.fromarray(bits).save(path)
    image = load(path)
    assert image.metadata.raw.mode == "1"
    assert image.data.dtype == bool
    np.testing.assert_array_equal(image.data, bits.T)


def test_read_grey_alpha(tmp_path: Path) -> None:
    la = RGBA[..., :2]
    image = load(_write(tmp_path, "a.png", np.ascontiguousarray(la)))
    assert image.metadata.raw.mode == "LA"
    np.testing.assert_array_equal(image.data, la.transpose(1, 0, 2))


def test_read_int32_and_float32_tiff(tmp_path: Path) -> None:
    # Pillow writes 32-bit integers and floats only as TIFF.
    for array in (
        RNG.integers(-(2**31), 2**31 - 1, (7, 12), dtype=np.int32),
        RNG.random((7, 12)).astype(np.float32),
    ):
        buffer = _bytes(array, "TIFF")
        image = PillowImage.from_bytes(buffer)
        assert image.data.dtype == array.dtype
        np.testing.assert_array_equal(image.data, array.T)


def test_read_jpeg_is_close(tmp_path: Path) -> None:
    smooth = np.tile(np.arange(12, dtype=np.uint8) * 20, (7, 1))
    smooth = np.stack([smooth] * 3, -1)
    image = load(_write(tmp_path, "a.jpg", smooth, quality=100))
    assert image.metadata.raw.format == "JPEG"
    assert image.data.shape == (12, 7, 3)
    assert np.abs(image.data.astype(int) - smooth.transpose(1, 0, 2)).max() < 8


def test_read_palette_is_converted(tmp_path: Path) -> None:
    rgb = Image.fromarray(RGB).quantize(colors=16)
    path = tmp_path / "a.png"
    rgb.save(path)
    expected = np.asarray(rgb.convert("RGB"))
    image = load(path)
    assert image.metadata.raw.mode == "P"
    assert image.data.shape == (12, 7, 3)
    np.testing.assert_array_equal(image.data, expected.transpose(1, 0, 2))
    # palette=False keeps the indices.
    indices = load(path, palette=False)
    assert indices.data.shape == (12, 7)
    np.testing.assert_array_equal(indices.data, np.asarray(rgb).T)


def test_read_palette_with_transparency_is_rgba(tmp_path: Path) -> None:
    p = Image.fromarray(GREY).convert("P")
    path = tmp_path / "a.png"
    p.save(path, transparency=0)
    image = load(path)
    assert image.data.shape == (12, 7, 4)


# ----------------------------------------------------------------------
#   READING: FRAMES
# ----------------------------------------------------------------------


def _gif(tmp_path):  # noqa: ANN001, ANN202
    frames = [
        Image.fromarray(np.full((5, 6), 40 * i, dtype=np.uint8))
        for i in range(3)
    ]
    path = tmp_path / "a.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    return path


def test_read_first_frame_by_default(tmp_path: Path) -> None:
    image = load(_gif(tmp_path))
    assert image.metadata.raw.format == "GIF"
    assert image.metadata.raw.n_frames == 3
    assert image.frame == 0
    assert image.data.shape == (6, 5, 3)
    assert (image.data == 0).all()


@pytest.mark.parametrize("frame, index", [(2, 2), (-1, 2), (1, 1)])
def test_read_frame(tmp_path: Path, frame: int, index: int) -> None:
    image = load(_gif(tmp_path), frame=frame)
    assert image.frame == index
    assert (image.data == 40 * index).all()


def test_read_missing_frame(tmp_path: Path) -> None:
    with pytest.raises(IndexError):
        PillowImage.load(_gif(tmp_path), frame=3)


# ----------------------------------------------------------------------
#   READING: PIXEL SIZE
# ----------------------------------------------------------------------


def test_dpi_is_ignored_by_default(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.png", GREY, dpi=(300, 150)))
    np.testing.assert_array_equal(image.transformation.scale, [1, 1])
    assert all(a.unit is None for a in image.transformation.output.axes)
    # The dpi is not applied by default but is kept as metadata.
    assert image.metadata.raw.info["dpi"] == pytest.approx(
        (300, 150), rel=1e-4
    )


@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.bmp"])
def test_dpi_true_uses_the_resolution(tmp_path: Path, name: str) -> None:
    image = load(_write(tmp_path, name, GREY, dpi=(300, 150)), dpi=True)
    xform = image.transformation
    np.testing.assert_allclose(xform.scale, [25.4 / 300, 25.4 / 150], 1e-4)
    assert all(str(a.unit) == "millimeter" for a in xform.output.axes)
    assert raster.physical_pixel_size(
        xform, list(xform.input.axes)
    ) == pytest.approx({"x": 25.4 / 300, "y": 25.4 / 150}, rel=1e-4)


@pytest.mark.parametrize("dpi", [(72, 72), (96, 96)])
def test_dpi_true_ignores_placeholders(tmp_path: Path, dpi: tx.Any) -> None:
    image = load(_write(tmp_path, "a.png", GREY, dpi=dpi), dpi=True)
    np.testing.assert_array_equal(image.transformation.scale, [1, 1])
    assert all(a.unit is None for a in image.transformation.output.axes)


def test_dpi_true_without_resolution(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.png", GREY), dpi=True)
    np.testing.assert_array_equal(image.transformation.scale, [1, 1])


def test_explicit_dpi_and_unit(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.png", GREY)
    image = load(path, dpi=254, unit="um")
    np.testing.assert_allclose(image.transformation.scale, [100, 100])
    axes = image.transformation.output.axes
    assert all(str(a.unit) == "micrometer" for a in axes)
    image = load(path, dpi=(254, 127))
    np.testing.assert_allclose(image.transformation.scale, [0.1, 0.2])


def test_pixel_size_overrides_dpi(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.png", RGB, dpi=(300, 300))
    image = load(path, dpi=True, pixel_size=(0.5, 2.0))
    xform = image.transformation
    np.testing.assert_allclose(xform.scale, [0.5, 2.0, 1.0])
    # The unit from the metadata is kept.
    assert str(xform.output.axes[0].unit) == "millimeter"
    image = load(path, pixel_size=3.0, unit="um")
    np.testing.assert_allclose(image.transformation.scale, [3.0, 3.0, 1.0])
    assert str(image.transformation.output.axes[0].unit) == "micrometer"
    # Without a unit anywhere, the size is unit-less.
    image = load(path, pixel_size=3.0)
    assert image.transformation.output.axes[0].unit is None


def test_pillow_dpi() -> None:
    assert pillow_dpi({}) is None
    assert pillow_dpi({"dpi": (300, 150)}) == (300.0, 150.0)
    assert pillow_dpi({"dpi": 300}) == (300.0, 300.0)
    assert pillow_dpi({"dpi": "nonsense"}) is None


# ----------------------------------------------------------------------
#   SNIFFING AND DISPATCH
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "format, expected",
    [
        ("PNG", Confidence.LIKELY),
        ("JPEG", Confidence.LIKELY),
        ("BMP", Confidence.LIKELY),
        ("GIF", Confidence.LIKELY),
        ("WEBP", Confidence.LIKELY),
        ("PPM", Confidence.LIKELY),
        ("TIFF", Confidence.WEAK),
    ],
)
def test_sniff(format: str, expected: tx.Any) -> None:
    assert PillowImage.sniff_bytes(_bytes(GREY, format)) == expected


def test_sniff_rejects_other_content() -> None:
    assert PillowImage.sniff_bytes(b"not an image at all") == Confidence.NO
    assert PillowImage.sniff_bytes(b"\x00" * 512) == Confidence.NO
    with pytest.raises(SnifferContentError):
        PillowImage.sniff_bytes(b"nope", error=True)


def test_sniff_leaves_the_stream_where_it_was() -> None:
    stream = io.BytesIO(_bytes(GREY, "PNG"))
    stream.seek(3)
    sniff_pillow(stream)
    assert stream.tell() == 3


def test_load_from_bytes_and_stream() -> None:
    content = _bytes(RGB, "PNG")
    for source in (content, io.BytesIO(content)):
        image = bio.load(source)
        assert isinstance(image, PillowImage)
        np.testing.assert_array_equal(image.data, RGB.transpose(1, 0, 2))


def test_load_by_content_with_wrong_extension(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(_bytes(RGB, "PNG"))
    image = load(path)
    assert image.metadata.raw.format == "PNG"


def test_tiff_falls_back_on_pillow() -> None:
    # The TIFF reader claims TIFF when registered, but Pillow reads it on
    # request.
    content = _bytes(GREY16, "TIFF")
    assert PillowImage.sniff(content) == Confidence.WEAK
    image = bio.images.load(content, hint="pillow")
    assert isinstance(image, PillowImage)
    np.testing.assert_array_equal(image.data, GREY16.T)


_WITHOUT_TIFFFILE = """
import sys

sys.modules["tifffile"] = None  # as if tifffile were not installed

import numpy as np
from PIL import Image

import brainhops.io as bio
from brainhops.io.base.parsers import ParserContentError
from brainhops.io.images.pillow import PillowImage

path = sys.argv[1]
array = np.arange(12, dtype=np.uint8).reshape(3, 4)
Image.fromarray(array).save(path, dpi=(254, 254))
assert not hasattr(bio.images, "tiff")
image = bio.load(path)
assert isinstance(image, PillowImage), type(image)
np.testing.assert_array_equal(image.data, array.T)
image = bio.images.load(path, dpi=True)
np.testing.assert_allclose(image.transformation.scale, 0.1)
try:
    bio.load(path, hint="tiff")
except ParserContentError as e:
    assert "pip install brainhops[tiff]" in str(e), str(e)
else:
    raise AssertionError("hint='tiff' should fail without tifffile")
"""


def test_pillow_reads_tiff_without_tifffile(tmp_path: Path) -> None:
    # Without tifffile, the weak TIFF sniff of Pillow is the only claim, and
    # requesting the TIFF reader says what to install.
    script = tmp_path / "script.py"
    script.write_text(_WITHOUT_TIFFFILE)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(bio.__file__).parents[2])]
        + [p for p in [env.get("PYTHONPATH")] if p]
    )
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "a.tif")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_not_an_image(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(b"this is text, not a PNG")
    with pytest.raises(ParserContentError):
        PillowImage.load(path)
    with pytest.raises(ParserContentError):
        load(path)


def test_unknown_option(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        PillowImage.load(_write(tmp_path, "a.png", GREY), colour="blue")


def test_decompression_bomb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, "a.png", GREY)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(Image.DecompressionBombError):
        PillowImage.load(path)
    # The file is still recognised as an image.
    assert PillowImage.sniff(path) == Confidence.LIKELY


def test_read_pillow_backend(tmp_path: Path) -> None:
    raw = read_pillow(_write(tmp_path, "a.png", RGB, dpi=(200, 200)))
    assert raw.axes == "YXS"
    assert raw.format == "PNG"
    assert raw.n_frames == 1
    assert raw.dpi == pytest.approx((200, 200), rel=1e-4)
    np.testing.assert_array_equal(raw.array, RGB)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "array, name",
    [
        (GREY, "b.png"),
        (RGB, "b.png"),
        (RGBA, "b.png"),
        (GREY16, "b.png"),
        (GREY > 100, "b.png"),
        (np.ascontiguousarray(RGBA[..., :2]), "b.png"),
        (RGB, "b.bmp"),
        (GREY, "b.pgm"),
        (RGB, "b.tga"),
        (GREY16, "b.tif"),
    ],
)
def test_write_round_trip(
    tmp_path: Path, array: np.ndarray, name: str
) -> None:
    data = array.T if array.ndim == 2 else array.transpose(1, 0, 2)
    path = tmp_path / name
    image = SingleScaleImage(data=data)
    if name.endswith(".tif"):
        # The Pillow writer does not claim TIFF but writes it on request.
        PillowImage.from_instance(image).save(path)
    else:
        bio.save(image, path)
    stored, _, _, _ = _read(path)
    np.testing.assert_array_equal(stored, array)
    if not name.endswith(".tif"):
        back = load(path)
        np.testing.assert_array_equal(back.data, data)


def test_write_reads_back_identically(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.png", RGBA))
    image.save(tmp_path / "b.png")
    back = load(tmp_path / "b.png")
    np.testing.assert_array_equal(back.data, image.data)
    assert back.metadata.raw.mode == image.metadata.raw.mode


def test_write_float_tiff(tmp_path: Path) -> None:
    array = RNG.random((7, 12)).astype(np.float32)
    content = PillowImage(data=array.T).to_bytes(format="TIFF")
    with Image.open(io.BytesIO(content)) as im:
        np.testing.assert_array_equal(np.asarray(im), array)


def test_write_jpeg(tmp_path: Path) -> None:
    bio.save(SingleScaleImage(data=RGB.transpose(1, 0, 2)), tmp_path / "b.jpg")
    stored, _, fmt, mode = _read(tmp_path / "b.jpg")
    assert fmt == "JPEG"
    assert stored.shape == RGB.shape


def test_write_options_are_passed_on(tmp_path: Path) -> None:
    image = PillowImage(data=RGB.transpose(1, 0, 2))
    low = image.to_bytes(format="JPEG", quality=5)
    high = image.to_bytes(format="JPEG", quality=95)
    assert len(low) < len(high)


def test_write_format_from_stream_name_or_keyword(tmp_path: Path) -> None:
    image = PillowImage(data=GREY.T)
    with open(tmp_path / "b.bmp", "wb") as f:
        image.save(f)
    assert _read(tmp_path / "b.bmp")[2] == "BMP"
    buffer = io.BytesIO()
    image.save(buffer)  # unnamed: PNG
    assert Image.open(io.BytesIO(buffer.getvalue())).format == "PNG"
    buffer = io.BytesIO()
    image.save(buffer, format="GIF")
    assert Image.open(io.BytesIO(buffer.getvalue())).format == "GIF"


def test_write_keeps_the_read_format() -> None:
    image = PillowImage.from_bytes(_bytes(GREY, "BMP"))
    assert Image.open(io.BytesIO(image.to_bytes())).format == "BMP"


@pytest.mark.parametrize(
    "data, name",
    [
        (np.zeros((4, 5), np.float64), "b.png"),
        (np.zeros((4, 5), np.int16), "b.png"),
        (np.zeros((4, 5), np.int64), "b.png"),
        (np.zeros((4, 5, 3), np.uint16), "b.png"),
        (np.zeros((4, 5, 6), np.uint8), "b.png"),
        (np.zeros((4, 5), np.uint16), "b.jpg"),
        (np.zeros((4, 5), np.float32), "b.png"),
        (np.zeros((4, 5, 4), np.uint8), "b.jpg"),
        (np.zeros((4, 5, 6, 7), np.uint8), "b.png"),
    ],
)
def test_write_refuses_unrepresentable_data(
    tmp_path: Path, data: np.ndarray, name: str
) -> None:
    path = tmp_path / name
    with pytest.raises(WriterError):
        PillowImage(data=data).save(path)
    assert not path.exists()


def test_write_refuses_a_volume(tmp_path: Path) -> None:
    xform = raster.raster_transformations(raster.default_axes(3))
    image = SingleScaleImage(
        data=np.zeros((4, 5, 3), np.uint8), transformations=xform
    )
    with pytest.raises(WriterError):
        bio.save(image, tmp_path / "b.png")


def test_write_drops_singleton_axes(tmp_path: Path) -> None:
    # A 2-D slice (x, y, z=1) of a volume, as NIfTI stores it.
    xform = raster.raster_transformations(raster.default_axes(3))
    image = SingleScaleImage(
        data=GREY.T[:, :, None].copy(), transformations=xform
    )
    bio.save(image, tmp_path / "b.png")
    np.testing.assert_array_equal(_read(tmp_path / "b.png")[0], GREY)


def test_write_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(WriterError):
        PillowImage(data=GREY.T).save(tmp_path / "b.unknownext")


def test_write_nothing() -> None:
    with pytest.raises(WriterError):
        PillowImage().to_bytes()


def test_array_to_pillow_modes() -> None:
    assert array_to_pillow(GREY).mode == "L"
    assert array_to_pillow(RGB).mode == "RGB"
    assert array_to_pillow(RGB[..., :1]).mode == "L"
    assert array_to_pillow(GREY16).mode == "I;16"
    assert array_to_pillow(GREY16.byteswap().view(">u2")).mode == "I;16"
    assert array_to_pillow(GREY > 0).mode == "1"


def test_format_for_name() -> None:
    assert format_for_name("a.PNG") == "PNG"
    assert format_for_name("dir/a.jpeg") == "JPEG"
    assert format_for_name("a.nii") is None
    assert format_for_name(None) is None


# ----------------------------------------------------------------------
#   WRITING: RESOLUTION AND METADATA
# ----------------------------------------------------------------------


def test_write_dpi_from_physical_scaling(tmp_path: Path) -> None:
    axes = raster.default_axes(2)
    scales = {"x": (0.1, "mm"), "y": (200.0, "um")}
    image = SingleScaleImage(
        data=GREY.T,
        transformations=raster.raster_transformations(axes, scales),
    )
    bio.save(image, tmp_path / "b.png")
    _, info, _, _ = _read(tmp_path / "b.png")
    assert info["dpi"] == pytest.approx((254, 127), rel=1e-3)
    back = load(tmp_path / "b.png", dpi=True)
    np.testing.assert_allclose(back.transformation.scale, [0.1, 0.2], 1e-3)


def test_write_no_dpi_when_unknown_or_refused(tmp_path: Path) -> None:
    bio.save(SingleScaleImage(data=GREY.T), tmp_path / "b.png")
    assert "dpi" not in _read(tmp_path / "b.png")[1]
    axes = raster.default_axes(2)
    image = SingleScaleImage(
        data=GREY.T,
        transformations=raster.raster_transformations(
            axes, {"x": (0.1, "mm"), "y": (0.1, "mm")}
        ),
    )
    PillowImage.from_instance(image).save(tmp_path / "c.png", dpi=False)
    assert "dpi" not in _read(tmp_path / "c.png")[1]


def test_write_explicit_dpi(tmp_path: Path) -> None:
    PillowImage(data=GREY.T).save(tmp_path / "b.png", dpi=(300, 150))
    assert _read(tmp_path / "b.png")[1]["dpi"] == pytest.approx(
        (300, 150), rel=1e-4
    )


def test_write_round_trips_file_dpi(tmp_path: Path) -> None:
    # A dpi that was read but not applied is still written back.
    image = load(_write(tmp_path, "a.png", GREY, dpi=(72, 72)))
    image.save(tmp_path / "b.png")
    assert _read(tmp_path / "b.png")[1]["dpi"] == pytest.approx(
        (72, 72), rel=1e-3
    )


def test_write_round_trips_icc_profile(tmp_path: Path) -> None:
    profile = b"\x00" * 128  # Pillow stores the profile as opaque bytes.
    path = tmp_path / "a.png"
    Image.fromarray(RGB).save(path, icc_profile=profile)
    image = load(path)
    assert image.metadata.raw.info["icc_profile"] == profile
    image.save(tmp_path / "b.png")
    assert _read(tmp_path / "b.png")[1]["icc_profile"] == profile


def test_write_a_nifti_slice(tmp_path: Path) -> None:
    nb = pytest.importorskip("nibabel")
    affine = np.diag([0.5, 0.25, 1.0, 1.0])
    nb.save(
        nb.Nifti1Image(GREY.T[:, :, None].copy(), affine), tmp_path / "a.nii"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = load(tmp_path / "a.nii")
    bio.save(image, tmp_path / "b.png")
    stored, info, _, _ = _read(tmp_path / "b.png")
    np.testing.assert_array_equal(stored, GREY)
    assert info["dpi"] == pytest.approx((25.4 / 0.5, 25.4 / 0.25), rel=1e-3)


def test_write_a_sagittal_slice(tmp_path: Path) -> None:
    # (x=1, y, z): y along the columns and z along the rows.
    xform = raster.raster_transformations(raster.default_axes(3))
    image = SingleScaleImage(
        data=GREY.T[None, :, :].copy(), transformations=xform
    )
    bio.save(image, tmp_path / "b.png")
    np.testing.assert_array_equal(_read(tmp_path / "b.png")[0], GREY)


def test_write_channel_first_axes(tmp_path: Path) -> None:
    axes = [Axis("c", "channel")] + raster.default_axes(2)
    xform = raster.raster_transformations(axes)
    data = RGB.transpose(2, 1, 0).copy()  # (c, x, y)
    image = SingleScaleImage(data=data, transformations=xform)
    bio.save(image, tmp_path / "b.png")
    np.testing.assert_array_equal(_read(tmp_path / "b.png")[0], RGB)


def test_write_falls_back_on_png_for_read_only_formats() -> None:
    record = PillowRaw(format="FITS", mode="L", size=(7, 12))
    image = PillowImage(data=GREY.T, metadata=PillowMetadata.from_raw(record))
    assert Image.open(io.BytesIO(image.to_bytes())).format == "PNG"


def test_png_scal_is_kept_but_not_applied(tmp_path: Path) -> None:
    # Pillow does not write sCAL, so the chunk is inserted after IHDR.
    png = _bytes(GREY, "PNG")
    data = b"\x01" + b"0.5\x000.25"
    chunk = len(data).to_bytes(4, "big") + b"sCAL" + data
    chunk += zlib.crc32(chunk[4:]).to_bytes(4, "big")
    end = 8 + 4 + 4 + 13 + 4  # signature, then IHDR
    path = tmp_path / "a.png"
    path.write_bytes(png[:end] + chunk + png[end:])
    for source in (path, io.BytesIO(path.read_bytes())):
        image = load(source)
        assert image.metadata.raw.info["sCAL"] == (1, 0.5, 0.25)
        np.testing.assert_array_equal(image.transformation.scale, [1, 1])
    scal = image.metadata.raw.info["sCAL"]
    image = load(path, pixel_size=scal[1:], unit="mm")
    np.testing.assert_allclose(image.transformation.scale, [0.5, 0.25])


# ----------------------------------------------------------------------
#   RECORD, LAZY ARRAY AND CONSOLIDATION
# ----------------------------------------------------------------------


def _frames(tmp_path, name, **options):  # noqa: ANN001, ANN202
    frames = [Image.fromarray(np.roll(RGB, i, axis=1)) for i in range(3)]
    path = tmp_path / name
    frames[0].save(path, save_all=True, append_images=frames[1:], **options)
    return path


_FILES = {
    "png": lambda p: _write(p, "a.png", RGB),
    "png-16": lambda p: _write(p, "a.png", GREY16),
    "apng": lambda p: _frames(p, "a.apng"),
    "jpeg": lambda p: _write(p, "a.jpg", RGB, quality=90),
    "bmp": lambda p: _write(p, "a.bmp", RGB),
    "dib": lambda p: _write(p, "a.dib", RGB),
    "gif": lambda p: _write(p, "a.gif", GREY),
    "gif-frames": lambda p: _frames(p, "a.gif"),
    "webp": lambda p: _write(p, "a.webp", RGB),
    "webp-lossless": lambda p: _write(p, "a.webp", RGB, lossless=True),
    "webp-frames": lambda p: _frames(p, "a.webp"),
    "ppm": lambda p: _write(p, "a.ppm", RGB),
    "pgm": lambda p: _write(p, "a.pgm", GREY),
    "pbm": lambda p: _write(p, "a.pbm", GREY > 100),
    "tga": lambda p: _write(p, "a.tga", RGB),
    "tga-rle": lambda p: _write(p, "a.tga", RGB, compression="tga_rle"),
    "pcx": lambda p: _write(p, "a.pcx", RGB),
    "jp2": lambda p: _write(p, "a.jp2", RGB),
    "j2k": lambda p: _write(p, "a.j2k", RGB),
}


@pytest.mark.parametrize("kind", sorted(_FILES))
def test_an_untouched_image_is_written_as_its_file(
    tmp_path: Path, kind: str
) -> None:
    # Pillow encodes again whatever it writes, so an image that is read and
    # not changed is written as the bytes of its file, in every format.
    source = _FILES[kind](tmp_path)
    content = source.read_bytes()
    image = PillowImage.load(source)
    assert isinstance(image.rawdata, PillowProxy)
    assert image.to_bytes() == content
    (tmp_path / "copy").mkdir()
    target = tmp_path / "copy" / source.name
    PillowImage.load(source).save(target)
    assert target.read_bytes() == content
    if source.suffix in PillowImage.EXTENSIONS:
        # The generic writer chooses the Pillow writer for this name.
        bio.save(PillowImage.load(source), target)
        assert target.read_bytes() == content
    assert "_cache_data" not in vars(image)
    # A changed image is encoded again, as one frame. It reads back as it
    # was given, except in JPEG and WebP, which Pillow writes lossily by
    # default.
    image.data = np.asarray(image.data)
    changed = image.consolidate()
    assert changed.metadata is not image.metadata
    assert changed.metadata.raw.n_frames == 1
    if not kind.startswith(("jpeg", "webp")):
        np.testing.assert_array_equal(changed.data, image.data)


def test_a_stream_is_written_back_from_its_start() -> None:
    content = _bytes(RGB, "PNG")
    stream = io.BytesIO(b"")
    stream.write(content)
    stream.seek(5)
    image = PillowImage.from_fileobj(stream)
    assert stream.tell() == 5
    assert image.to_bytes() == content
    np.testing.assert_array_equal(image.data, RGB.transpose(1, 0, 2))
    assert stream.tell() == 5


def test_the_record_is_the_header(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.png", RGB, dpi=(300, 150))
    record = PillowMetadata.load(path).raw
    assert record == PillowImage.load(path).metadata.raw
    assert record.format == "PNG"
    assert record.mode == "RGB"
    assert record.size == (12, 7)
    assert record.frame == 0
    assert record.n_frames == 1
    assert record.dpi == pytest.approx((300, 150), rel=1e-4)
    assert record.layout() == ((7, 12, 3), np.dtype("uint8"))
    clone = pickle.loads(pickle.dumps(record))
    assert clone == record
    with pytest.raises(TypeError):
        PillowMetadata.load(path, palette=False)


def test_the_record_describes_the_frame_that_is_read(tmp_path: Path) -> None:
    path = _gif(tmp_path)
    first = PillowMetadata.load(path).raw
    last = PillowMetadata.load(path, frame=-1).raw
    assert (first.frame, first.n_frames, first.mode) == (0, 3, "P")
    # Pillow composites the frames of a GIF after the first in RGB.
    assert (last.frame, last.n_frames, last.mode) == (2, 3, "RGB")
    image = PillowImage.load(path, frame=-1)
    assert image.frame == 2
    assert image.metadata.raw == last


def test_unpicklable_info_is_left_out() -> None:
    from brainhops.io.images.pillow._raw import picklable

    info = {"dpi": (72.0, 72.0), "lock": threading.Lock()}
    assert picklable(info) == {"dpi": (72.0, 72.0)}


@pytest.mark.parametrize(
    "mode, info, palette",
    [
        ("1", {}, True),
        ("L", {}, True),
        ("LA", {}, True),
        ("P", {}, True),
        ("P", {}, False),
        ("P", {"transparency": 0}, True),
        ("PA", {}, True),
        ("PA", {}, False),
        ("RGB", {}, True),
        ("RGBA", {}, True),
        ("CMYK", {}, True),
        ("YCbCr", {}, True),
        ("I;16", {}, True),
        ("I;16B", {}, True),
        ("I", {}, True),
        ("F", {}, True),
    ],
)
def test_the_layout_is_that_of_the_decoded_frame(
    mode: str, info: tx.Dict[str, tx.Any], palette: bool
) -> None:
    im = Image.new(mode, (5, 3))
    im.info.update(info)
    array, _ = pillow_to_array(im, palette=palette)
    record = PillowRaw(format="PNG", mode=mode, size=(5, 3), info=info)
    assert record.layout(palette) == (array.shape, array.dtype)


def test_the_shape_is_read_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, "a.png", RGB)
    image = load(path)

    def refuse(*args: tx.Any, **kwargs: tx.Any) -> None:
        raise AssertionError("The frame was decoded.")

    monkeypatch.setattr(pillow_data, "read_pillow", refuse)
    assert image.shape == (12, 7, 3)
    assert image.rawdata.shape == (7, 12, 3)
    assert image.rawdata[:0].shape == (0, 12, 3)
    assert [axis.name for axis in image.transformation.input.axes] == [
        "x",
        "y",
        "c",
    ]


def test_dpi_true_keeps_the_bytes(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.png", GREY, dpi=(254, 127))
    image = load(path, dpi=True)
    # The pixel size is decoded from the record, and is not assigned.
    assert image.dpi
    assert image._transformations is None
    np.testing.assert_allclose(image.transformation.scale, [0.1, 0.2], 1e-4)
    assert image.to_bytes() == path.read_bytes()


def test_a_pixel_size_given_to_the_reader_is_assigned(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.png", GREY)
    image = load(path, pixel_size=0.5, unit="mm")
    assert image._transformations is not None
    content = image.to_bytes()
    assert content != path.read_bytes()
    back = PillowImage.from_bytes(content, dpi=True)
    np.testing.assert_allclose(back.transformation.scale, [0.5, 0.5], 1e-4)
    # A unit alone makes up no size, so the record is kept.
    assert load(path, unit="mm").to_bytes() == path.read_bytes()


def test_a_changed_image_is_written_as_one_frame(tmp_path: Path) -> None:
    path = _gif(tmp_path)
    image = load(path, frame=1)
    assert image.to_bytes() == path.read_bytes()
    image.data = np.asarray(image.data)
    consolidated = image.consolidate()
    assert consolidated.frame == 0
    assert consolidated.metadata.raw.n_frames == 1
    np.testing.assert_array_equal(consolidated.data, image.data)


def test_the_reader_options_cannot_change_after_a_read(tmp_path: Path) -> None:
    image = load(_gif(tmp_path))
    image.frame = 1
    with pytest.raises(WriterError):
        image.consolidate()
    image = load(_gif(tmp_path))
    image.palette = False
    with pytest.raises(WriterError):
        image.to_bytes()


@pytest.mark.parametrize(
    "name, options",
    [
        ("b.png", {"quality": 90}),
        ("b.png", {"save_all": True}),
        ("b.jpg", {"compress_level": 9}),
        ("b.bmp", {"optimize": True}),
    ],
)
def test_an_option_that_pillow_ignores_is_refused(
    tmp_path: Path, name: str, options: tx.Dict[str, tx.Any]
) -> None:
    image = load(_write(tmp_path, "a.png", RGB))
    with pytest.raises(TypeError):
        image.save(tmp_path / name, **options)
    assert not (tmp_path / name).exists()


def test_another_format_encodes_the_image_again(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.png", RGB))
    image.save(tmp_path / "b.bmp")
    assert _read(tmp_path / "b.bmp")[2] == "BMP"
    content = image.to_bytes(format="BMP")
    assert Image.open(io.BytesIO(content)).format == "BMP"
    # The image itself is not changed by a write.
    assert image.metadata.raw.format == "PNG"


def test_a_changed_image_keeps_its_icc_profile(tmp_path: Path) -> None:
    profile = b"\x00" * 128
    path = tmp_path / "a.png"
    Image.fromarray(RGB).save(path, icc_profile=profile)
    image = load(path)
    image.data = np.asarray(image.data)[::-1]
    image.save(tmp_path / "b.png")
    assert _read(tmp_path / "b.png")[1]["icc_profile"] == profile


def test_saving_over_its_own_file_reads_the_new_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.png", RGB)
    image = load(path)
    flipped = np.asarray(image.data)[::-1]
    image.data = flipped
    image.save(path)
    # The image holds what the file holds, and its lazy array reads it.
    assert image._data is None
    assert isinstance(image.rawdata, PillowProxy)
    assert image.rawdata.source == path or str(image.rawdata.source) == str(
        path
    )
    back = load(path)
    assert image.metadata.raw == back.metadata.raw
    np.testing.assert_array_equal(image.data, flipped)
    np.testing.assert_array_equal(back.data, flipped)
