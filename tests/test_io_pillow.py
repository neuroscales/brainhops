"""
Tests for raster images (PNG, JPEG, BMP, GIF, WebP, ...) read and written
with Pillow.

Fixtures are generated with Pillow itself, so that each test states the
exact pixels, mode and resolution of the file it reads.
"""

import io
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

import brainhops.io as bio  # noqa: E402
from brainhops.datamodel.axes import Axis  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.transformations import Scaling  # noqa: E402
from brainhops.datamodel.units import is_sampleunit  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.images import load  # noqa: E402
from brainhops.io.images.base import _utils_raster as raster  # noqa: E402
from brainhops.io.images.pillow import PillowImage  # noqa: E402
from brainhops.io.images.pillow._utils import (  # noqa: E402
    array_to_pillow,
    format_for_name,
    pillow_dpi,
    read_pillow,
    sniff_pillow,
)

RNG = np.random.default_rng(0)
GREY = RNG.integers(0, 256, (7, 12), dtype=np.uint8)  # rows, columns
RGB = RNG.integers(0, 256, (7, 12, 3), dtype=np.uint8)
RGBA = RNG.integers(0, 256, (7, 12, 4), dtype=np.uint8)
GREY16 = RNG.integers(0, 2**16, (7, 12), dtype=np.uint16)


def _write(tmp_path, name, array, **options):  # noqa: ANN001, ANN202
    """Write `array` (C order) with Pillow, and return the path."""
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
    assert image.mode == mode
    if array.ndim == 2:
        expected = array.T
    else:
        expected = array.transpose(1, 0, 2)
    assert image.data.shape == expected.shape
    assert image.data.dtype == array.dtype
    np.testing.assert_array_equal(image.data, expected)
    # x is the column, y the row
    assert image.data[3, 2, ...].tolist() == array[2, 3, ...].tolist()


def test_read_axes_and_systems(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.png", RGB))
    (xform,) = image.transformations
    assert isinstance(xform, Scaling)
    pixel = xform.input
    assert pixel.name == "pixel"
    assert pixel.order == "F"
    assert [a.name for a in pixel.axes] == ["x", "y", "c"]
    assert [a.type for a in pixel.axes] == ["space", "space", "channel"]
    assert all(is_sampleunit(a.unit) for a in pixel.axes)
    # Unknown size: identity onto axes with no unit.
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
    image = load(_write(tmp_path, "a.png", GREY))
    # The F-ordered data of a grey image is the transpose of a C array.
    assert image.data.flags.f_contiguous
    assert image.data.base is not None
    assert image.data.flags.writeable


def test_read_bilevel(tmp_path: Path) -> None:
    bits = GREY > 128
    path = tmp_path / "a.png"
    Image.fromarray(bits).save(path)
    image = load(path)
    assert image.mode == "1"
    assert image.data.dtype == bool
    np.testing.assert_array_equal(image.data, bits.T)


def test_read_grey_alpha(tmp_path: Path) -> None:
    la = RGBA[..., :2]
    image = load(_write(tmp_path, "a.png", np.ascontiguousarray(la)))
    assert image.mode == "LA"
    np.testing.assert_array_equal(image.data, la.transpose(1, 0, 2))


def test_read_int32_and_float32_tiff(tmp_path: Path) -> None:
    # Pillow writes 32-bit integer and float images only as TIFF.
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
    assert image.image_format == "JPEG"
    assert image.data.shape == (12, 7, 3)
    assert np.abs(image.data.astype(int) - smooth.transpose(1, 0, 2)).max() < 8


def test_read_palette_is_converted(tmp_path: Path) -> None:
    rgb = Image.fromarray(RGB).quantize(colors=16)
    path = tmp_path / "a.png"
    rgb.save(path)
    expected = np.asarray(rgb.convert("RGB"))
    image = load(path)
    assert image.mode == "P"
    assert image.data.shape == (12, 7, 3)
    np.testing.assert_array_equal(image.data, expected.transpose(1, 0, 2))
    # With palette=False, the indices are kept.
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
    assert image.image_format == "GIF"
    assert image.n_frames == 3
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
    # ... but kept as metadata.
    assert image.info["dpi"] == pytest.approx((300, 150), rel=1e-4)


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
    # The unit of the metadata is kept.
    assert str(xform.output.axes[0].unit) == "millimeter"
    image = load(path, pixel_size=3.0, unit="um")
    np.testing.assert_allclose(image.transformation.scale, [3.0, 3.0, 1.0])
    assert str(image.transformation.output.axes[0].unit) == "micrometer"
    # Without a unit from anywhere, the size has none.
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
    assert image.image_format == "PNG"


def test_tiff_falls_back_on_pillow() -> None:
    image = bio.images.load(_bytes(GREY16, "TIFF"))
    assert isinstance(image, PillowImage)
    np.testing.assert_array_equal(image.data, GREY16.T)


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
    # It is still recognized as an image.
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
        # TIFF files are not claimed by the Pillow writer, but it can
        # write one when asked to.
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
    assert back.mode == image.mode


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
    # A 2D slice of a volume, (x, y, z=1), as NIfTI stores one.
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
    # Read without using the resolution: it is still written back.
    image = load(_write(tmp_path, "a.png", GREY, dpi=(72, 72)))
    image.save(tmp_path / "b.png")
    assert _read(tmp_path / "b.png")[1]["dpi"] == pytest.approx(
        (72, 72), rel=1e-3
    )


def test_write_round_trips_icc_profile(tmp_path: Path) -> None:
    profile = b"\x00" * 128  # Pillow stores the profile as opaque bytes
    path = tmp_path / "a.png"
    Image.fromarray(RGB).save(path, icc_profile=profile)
    image = load(path)
    assert image.info["icc_profile"] == profile
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
    # (x=1, y, z): y runs along the columns and z along the rows.
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
    image = PillowImage(data=GREY.T, image_format="FITS")
    assert Image.open(io.BytesIO(image.to_bytes())).format == "PNG"


def test_png_scal_is_kept_but_not_applied(tmp_path: Path) -> None:
    # Pillow does not write an sCAL chunk: insert one after IHDR.
    png = _bytes(GREY, "PNG")
    data = b"\x01" + b"0.5\x000.25"
    chunk = len(data).to_bytes(4, "big") + b"sCAL" + data
    chunk += zlib.crc32(chunk[4:]).to_bytes(4, "big")
    end = 8 + 4 + 4 + 13 + 4  # signature, then IHDR
    path = tmp_path / "a.png"
    path.write_bytes(png[:end] + chunk + png[end:])
    for source in (path, io.BytesIO(path.read_bytes())):
        image = load(source)
        assert image.info["sCAL"] == (1, 0.5, 0.25)
        np.testing.assert_array_equal(image.transformation.scale, [1, 1])
    image = load(path, pixel_size=image.info["sCAL"][1:], unit="mm")
    np.testing.assert_allclose(image.transformation.scale, [0.5, 0.25])
