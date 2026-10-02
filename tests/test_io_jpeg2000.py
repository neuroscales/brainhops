"""
Tests for JPEG 2000 images (JP2 files and raw codestreams) read as
multiscale images and written with Pillow (OpenJPEG).

Fixtures are generated with Pillow (and imagecodecs, which writes bit
depths and signed samples Pillow does not), so that each test states the
exact layout of the file it reads.
"""

import io
import struct
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

PIL = pytest.importorskip("PIL")
from PIL import Image, features  # noqa: E402

if not features.check("jpg_2000"):  # pragma: no cover
    pytest.skip("Pillow was built without OpenJPEG", allow_module_level=True)

import brainhops.io as bio  # noqa: E402
from brainhops.datamodel.images import (  # noqa: E402
    MultiScaleImage,
    SingleScaleImage,
)
from brainhops.datamodel.transformations import Affine, Scaling  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.images import load, sniff  # noqa: E402
from brainhops.io.images.base import _utils_raster as raster  # noqa: E402
from brainhops.io.images.jpeg2000 import (  # noqa: E402
    J2kImage,
    J2kMultiScaleImage,
    Jp2Image,
    Jp2MultiScaleImage,
)
from brainhops.io.images.jpeg2000 import _image as j2k_image  # noqa: E402
from brainhops.io.images.jpeg2000 import _utils as backend  # noqa: E402
from brainhops.io.images.pillow import PillowImage  # noqa: E402

RNG = np.random.default_rng(0)
# A linear ramp (rows, columns), odd-sized: the reversible wavelet keeps a
# ramp's even samples exactly, away from the edges.
ROWS, COLUMNS = 37, 45
RAMP = (
    3 * np.arange(COLUMNS)[None, :] + 100 * np.arange(ROWS)[:, None]
).astype(np.uint16)
RGB = RNG.integers(0, 256, (ROWS, COLUMNS, 3), dtype=np.uint8)


def _encode(array: np.ndarray, j2k: bool = False, **kw) -> bytes:
    buffer = io.BytesIO()
    kw.setdefault("num_resolutions", 4)
    Image.fromarray(array).save(buffer, "JPEG2000", no_jp2=j2k, **kw)
    return buffer.getvalue()


def _write(tmp_path: Path, name: str, array: np.ndarray, **kw) -> Path:
    path = tmp_path / name
    path.write_bytes(_encode(array, j2k=not name.endswith(".jp2"), **kw))
    return path


def _ceil(n: int, f: int) -> int:
    return -(-n // f)


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, multi, single",
    [
        ("a.jp2", Jp2MultiScaleImage, Jp2Image),
        ("a.j2k", J2kMultiScaleImage, J2kImage),
    ],
)
def test_dispatch(
    tmp_path: Path, name: str, multi: type, single: type
) -> None:
    path = _write(tmp_path, name, RAMP)
    image = load(path)
    assert type(image) is multi
    assert isinstance(image, MultiScaleImage)
    assert [type(level) for level in image.images] == [single] * 4
    one = load(path, level=1)
    assert type(one) is single
    assert isinstance(one, SingleScaleImage)
    # Without a name, the content alone tells the container.
    assert sniff(io.BytesIO(path.read_bytes())) is multi
    assert type(load(io.BytesIO(path.read_bytes()))) is multi


def test_dispatch_single_resolution_is_single_scale(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP, num_resolutions=1)
    image = load(path)
    assert type(image) is Jp2Image
    assert image.n_levels == 1


def test_sniff_scores(tmp_path: Path) -> None:
    jp2 = _encode(RAMP)
    j2k = _encode(RAMP, j2k=True)
    assert Jp2MultiScaleImage.sniff_bytes(jp2) == Confidence.CERTAIN
    assert Jp2MultiScaleImage.sniff_bytes(jp2, level=0) == Confidence.NO
    assert Jp2Image.sniff_bytes(jp2) == Confidence.LIKELY
    assert J2kImage.sniff_bytes(jp2) == Confidence.NO
    assert J2kMultiScaleImage.sniff_bytes(j2k) == Confidence.CERTAIN
    assert Jp2Image.sniff_bytes(j2k) == Confidence.NO
    # Pillow reads JPEG 2000, but only as a fallback.
    assert PillowImage.sniff_bytes(jp2) == Confidence.WEAK
    assert Jp2Image.sniff_bytes(b"\x89PNG\r\n\x1a\n") == Confidence.NO


def test_hints(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP)
    assert type(load(path, hint="pillow")) is PillowImage
    assert type(load(path, hint="jpeg2000")) is Jp2MultiScaleImage
    assert type(load(path, hint="jpeg2000.jp2")) is Jp2MultiScaleImage
    codestream = tmp_path / "a.j2c"
    codestream.write_bytes(_encode(RAMP, j2k=True))
    assert type(load(codestream, hint="j2c")) is J2kMultiScaleImage


def test_save_picks_jpeg2000_over_pillow(tmp_path: Path) -> None:
    axes = raster.default_axes(2)
    image = SingleScaleImage(
        data=RAMP.T, transformations=raster.raster_transformations(axes)
    )
    bio.save(image, tmp_path / "a.jp2")
    bio.save(image, tmp_path / "a.j2k")
    assert (tmp_path / "a.jp2").read_bytes() == _encode(
        RAMP, num_resolutions=6
    )
    assert (tmp_path / "a.j2k").read_bytes()[:4] == backend.J2K_SIGNATURE


# ----------------------------------------------------------------------
#   LEVELS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.jp2", "a.j2k"])
def test_level_shapes(tmp_path: Path, name: str) -> None:
    image = load(_write(tmp_path, name, RAMP))
    shapes = [level.data.shape for level in image.images]
    assert shapes == [
        (_ceil(COLUMNS, 2**r), _ceil(ROWS, 2**r)) for r in range(4)
    ]
    assert [level.level for level in image.images] == [0, 1, 2, 3]


def test_full_resolution_is_exact_and_f_ordered(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.jp2", RAMP), level=0)
    data = image.data
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data, RAMP.T)
    # A transposed view of what Pillow decodes, not a copy.
    assert data.flags.f_contiguous and not data.flags.c_contiguous


@pytest.mark.parametrize("level", [1, 2, 3])
def test_reduced_level_values(tmp_path: Path, level: int) -> None:
    data = load(_write(tmp_path, "a.jp2", RAMP), level=level).data
    f = 2**level
    # Pixel i of level r sits on the full-resolution pixel f * i: the
    # interior of a ramp is kept exactly.
    expected = RAMP[::f, ::f].T
    inner = (slice(1, -1), slice(1, -1))
    np.testing.assert_array_equal(data[inner], expected[inner])


def test_reduced_level_rgb(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.jp2", RGB))
    assert image.images[0].data.shape == (COLUMNS, ROWS, 3)
    np.testing.assert_array_equal(image.images[0].data, RGB.transpose(1, 0, 2))
    assert image.images[2].data.shape == (12, 10, 3)
    assert image.images[0].transformation.input.axes[2].name == "c"


def test_level_transformations(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.jp2", RAMP), pixel_size=0.5)
    assert image.transformations == []
    for r, level in enumerate(image.images):
        xform = level.transformation
        assert isinstance(xform, Scaling)
        np.testing.assert_allclose(xform.scale, [0.5 * 2**r] * 2)
        assert xform.output.name == "physical"


def test_negative_and_missing_levels(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP)
    assert load(path, level=-1).level == 3
    with pytest.raises(IndexError, match="no level 4"):
        Jp2Image.from_filename(path, level=4)
    with pytest.raises(ParserContentError, match="no level 4"):
        load(path, level=4)


def test_levels_are_decoded_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    decode = backend.decode_level

    def spy(source: tx.Any, header: tx.Any, level: int) -> tx.Any:
        calls.append(level)
        return decode(source, header, level)

    monkeypatch.setattr(backend, "decode_level", spy)
    image = load(_write(tmp_path, "a.jp2", RAMP))
    assert calls == []
    _ = image.images[2].data
    _ = image.images[2].data
    assert calls == [2]


def test_read_from_bytes_and_fileobj(tmp_path: Path) -> None:
    content = _encode(RAMP)
    image = Jp2MultiScaleImage.from_bytes(content)
    np.testing.assert_array_equal(image.images[0].data, RAMP.T)
    stream = io.BytesIO(content)
    single = Jp2Image.from_fileobj(stream, level=1)
    assert stream.tell() == 0
    assert single.data.shape == (23, 19)


# ----------------------------------------------------------------------
#   IMAGE OFFSET
# ----------------------------------------------------------------------


def _with_offset(content: bytes, offset: tx.Tuple[int, int]) -> bytes:
    """Move the image area (and its only tile) of a codestream Pillow
    wrote on the reference grid. With an offset that is a multiple of
    every downsampling factor, the coded data is unchanged."""
    at = content.index(b"\xff\x51") + 4
    siz = struct.unpack(">HIIIIIIII", content[at : at + 34])
    rsiz, xs, ys, _, _, xt, yt, _, _ = siz
    ox, oy = offset
    new = struct.pack(
        ">HIIIIIIII", rsiz, xs + ox, ys + oy, ox, oy, xt, yt, ox, oy
    )
    return content[:at] + new + content[at + 34 :]


def test_offset_reads_full_resolution_only() -> None:
    content = _with_offset(_encode(RAMP, j2k=True), (8, 16))
    # Pillow cannot decode the reduced levels of such an image, so it is
    # read as a single-scale image by default.
    assert J2kMultiScaleImage.sniff_bytes(content) == Confidence.NO
    image = load(io.BytesIO(content))
    assert type(image) is J2kImage
    assert image.header.image_offset == (8, 16)
    assert image.header.size == (COLUMNS, ROWS)
    np.testing.assert_array_equal(image.data, RAMP.T)
    assert isinstance(image.transformation, Scaling)
    pyramid = J2kMultiScaleImage.from_bytes(content)
    assert isinstance(pyramid.images[1].transformation, Scaling)
    with pytest.raises(ParserContentError, match="offset"):
        _ = pyramid.images[1].data


def test_odd_offset_geometry() -> None:
    content = _with_offset(_encode(RAMP, j2k=True), (3, 5))
    image = J2kMultiScaleImage.from_bytes(content)
    header = image.header
    assert header.level_shape(0) == (COLUMNS, ROWS)
    # ceil((3 + 45) / 4) - ceil(3 / 4), ceil((5 + 37) / 4) - ceil(5 / 4)
    assert header.level_shape(2) == (11, 9)
    assert header.level_start(2) == (1, 3)
    xform = image.images[2].transformation
    assert isinstance(xform, Affine)
    matrix = np.asarray(xform.homogeneous_matrix)
    np.testing.assert_allclose(matrix[:2, :2], np.diag([4, 4]))
    np.testing.assert_allclose(matrix[:2, 2], [1, 3])


# ----------------------------------------------------------------------
#   BIT DEPTHS
# ----------------------------------------------------------------------


def test_twelve_bit_and_signed() -> None:
    imagecodecs = pytest.importorskip("imagecodecs")
    if not imagecodecs.JPEG2K.available:  # pragma: no cover
        pytest.skip("imagecodecs was built without OpenJPEG")
    twelve = (RNG.integers(0, 4096, (20, 30))).astype(np.uint16)
    content = imagecodecs.jpeg2k_encode(
        twelve, level=0, codecformat="jp2", bitspersample=12
    )
    image = Jp2Image.from_bytes(content)
    assert image.header.bit_depths == (12,)
    np.testing.assert_array_equal(image.data, twelve.T)

    signed = (np.arange(600) - 300).astype(np.int16).reshape(20, 30)
    content = imagecodecs.jpeg2k_encode(signed, level=0, codecformat="j2k")
    image = J2kImage.from_bytes(content)
    assert image.header.signed == (True,)
    assert image.data.dtype == np.int16
    np.testing.assert_array_equal(image.data, signed.T)

    colour = RNG.integers(0, 4096, (20, 30, 3)).astype(np.uint16)
    content = imagecodecs.jpeg2k_encode(
        colour, level=0, codecformat="jp2", bitspersample=12
    )
    with pytest.raises(ParserContentError, match="precision"):
        _ = Jp2Image.from_bytes(content).data


# ----------------------------------------------------------------------
#   PIXEL SIZE
# ----------------------------------------------------------------------


def test_pixel_size_unknown_by_default(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.jp2", RAMP), level=0)
    assert image.header.capture_resolution is None
    np.testing.assert_allclose(image.transformation.scale, [1, 1])
    assert [a.unit for a in image.transformation.output.axes] == [None] * 2


def test_pixel_size_from_capture_resolution() -> None:
    # 2000 points per mm along x, 1000 along y: 0.5 and 1 micrometre.
    content = backend.add_boxes(_encode(RAMP), capture=(2e6, 1e6))
    image = Jp2MultiScaleImage.from_bytes(content)
    assert image.header.capture_resolution == pytest.approx((2e6, 1e6))
    np.testing.assert_allclose(
        image.images[1].transformation.scale, [1e-3, 2e-3]
    )
    units = image.images[0].transformation.output.axes
    assert [str(a.unit) for a in units] == ["millimeter"] * 2
    um = Jp2Image.from_bytes(content, unit="um")
    np.testing.assert_allclose(um.transformation.scale, [0.5, 1])


def test_pixel_size_display_fallback_and_placeholders() -> None:
    content = backend.add_boxes(_encode(RAMP), display=(1e4, 1e4))
    image = Jp2Image.from_bytes(content)
    np.testing.assert_allclose(image.transformation.scale, [0.1, 0.1])
    # 72 dpi is 2834.6 points per metre: a placeholder.
    dpi72 = 72 / 0.0254
    content = backend.add_boxes(_encode(RAMP), capture=(dpi72, dpi72))
    image = Jp2Image.from_bytes(content)
    assert image.header.capture_resolution == pytest.approx((dpi72,) * 2)
    np.testing.assert_allclose(image.transformation.scale, [1, 1])
    assert image.transformation.output.axes[0].unit is None
    # Overridden.
    image = Jp2Image.from_bytes(content, pixel_size=(2, 3), unit="um")
    np.testing.assert_allclose(image.transformation.scale, [2, 3])


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.jp2", "b.j2k"])
def test_write_round_trip_lossless(tmp_path: Path, name: str) -> None:
    source = load(_write(tmp_path, "src.jp2", RGB), level=0)
    bio.save(source, tmp_path / name)
    back = load(tmp_path / name)
    np.testing.assert_array_equal(back.images[0].data, source.data)
    # The number of resolutions of the file read is kept.
    assert back.header.n_levels == 4
    assert back.header.reversible


def test_write_pixel_size_and_boxes(tmp_path: Path) -> None:
    content = backend.add_boxes(
        _encode(RAMP, comment="hello"),
        display=(1e4, 1e4),
        boxes=[(b"xml ", b"<gml/>")],
    )
    image = Jp2Image.from_bytes(content, pixel_size=0.25, unit="um")
    out = tmp_path / "a.jp2"
    image.save(out)
    back = Jp2Image.from_filename(out)
    assert back.header.capture_resolution == pytest.approx((4e6, 4e6))
    assert back.header.display_resolution == pytest.approx((1e4, 1e4))
    assert back.header.boxes == ((b"xml ", b"<gml/>"),)
    assert back.header.comments == ("hello",)
    np.testing.assert_allclose(back.transformation.scale, [2.5e-4] * 2)
    np.testing.assert_array_equal(back.data, RAMP.T)


def test_write_options(tmp_path: Path) -> None:
    image = Jp2Image.from_bytes(_encode(RAMP))
    content = image.to_bytes(num_resolutions=2)
    assert Jp2Image.from_bytes(content).n_levels == 2
    with pytest.raises(WriterError, match="resolution levels"):
        image.to_bytes(num_resolutions=7)
    lossy = image.to_bytes(
        irreversible=True, quality_mode="rates", quality_layers=[40]
    )
    assert not Jp2Image.from_bytes(lossy).header.reversible
    with pytest.raises(WriterError):
        Jp2Image(
            data=RNG.random((5, 6)),
            transformations=raster.raster_transformations(
                raster.default_axes(2)
            ),
        ).to_bytes()


def test_write_multiscale(tmp_path: Path) -> None:
    pyramid = load(_write(tmp_path, "a.jp2", RAMP))
    out = tmp_path / "b.j2k"
    bio.save(pyramid, out)
    back = load(out)
    assert type(back) is J2kMultiScaleImage
    assert len(back.images) == 4
    for level, other in zip(back.images, pyramid.images):
        np.testing.assert_array_equal(level.data, other.data)

    axes = raster.default_axes(2)
    bad = Jp2MultiScaleImage(
        images=[
            SingleScaleImage(
                data=np.zeros((8, 8), np.uint8),
                transformations=raster.raster_transformations(axes),
            ),
            SingleScaleImage(
                data=np.zeros((3, 4), np.uint8),
                transformations=raster.raster_transformations(axes),
            ),
        ],
        transformations=[],
    )
    with pytest.raises(WriterError, match="shape"):
        bad.to_bytes()


# ----------------------------------------------------------------------
#   HEADER
# ----------------------------------------------------------------------


def test_header_fields() -> None:
    header = Jp2MultiScaleImage.from_bytes(
        _encode(RGB, tile_size=(16, 16))
    ).header
    assert header.container == "jp2"
    assert header.size == (COLUMNS, ROWS)
    assert header.tile_size == (16, 16)
    assert header.bit_depths == (8, 8, 8)
    assert header.signed == (False,) * 3
    assert header.subsampling == ((1, 1),) * 3
    assert header.n_levels == 4
    assert header.n_layers == 1


def test_corrupt_files() -> None:
    with pytest.raises(ParserContentError):
        Jp2Image.from_bytes(backend.JP2_SIGNATURE + b"\x00\x00")
    with pytest.raises(ParserContentError):
        J2kImage.from_bytes(backend.J2K_SIGNATURE + b"\x00" * 10)


def test_resolution_box_encoding() -> None:
    for value in (1.0, 2834.6, 1e4, 123456.7, 4e6, 0.001):
        box = j2k_image.backend._resolution_box(b"resc", (value, value))
        back = backend._resolution(box[8:])
        assert back == pytest.approx((value, value), rel=1e-6)
