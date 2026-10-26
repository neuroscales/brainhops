"""
Tests for JPEG 2000 images, which are JP2 files and raw codestreams read
as multiscale images and written with Pillow (OpenJPEG).

The fixtures are generated with Pillow, and with imagecodecs for the bit
depths and the signed samples that Pillow does not write, so that each
test states the exact layout of the file that it reads. The checks that
every format of the design of format records passes are in
`test_io_conformance.py`.
"""

import io
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import replace

PIL = pytest.importorskip("PIL")
from PIL import Image, features  # noqa: E402

if not features.check("jpg_2000"):  # pragma: no cover
    pytest.skip("Pillow was built without OpenJPEG", allow_module_level=True)

import brainhops.io as bio  # noqa: E402
from brainhops._core.backends import backend  # noqa: E402
from brainhops.datamodel.images import (  # noqa: E402
    MultiScaleImage,
    SingleScaleImage,
)
from brainhops.datamodel.metadata import (  # noqa: E402
    ConversionReport,
    MetadataLossError,
)
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    Scaling,
)
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.images import load, sniff  # noqa: E402
from brainhops.io.images.base import _utils_raster as raster  # noqa: E402
from brainhops.io.images.jpeg2000 import (  # noqa: E402
    Jpeg2000Image,
    Jpeg2000Metadata,
    Jpeg2000MultiScaleImage,
    Jpeg2000Raw,
    _boxes,  # noqa: E402
    _data,  # noqa: E402
)
from brainhops.io.images.jpeg2000._data import _Jpeg2000Proxy  # noqa: E402
from brainhops.io.images.jpeg2000._raw import read_record  # noqa: E402
from brainhops.io.images.pillow import PillowImage  # noqa: E402

RNG = np.random.default_rng(0)
# A linear ramp of (rows, columns), with odd sizes. The reversible wavelet
# keeps the even samples of a ramp exactly, away from the edges.
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


def _with(content: bytes, **changes) -> bytes:
    """Rewrite the header of a file with other comments, boxes or
    resolutions."""
    return _boxes.rewrite(content, replace(read_record(content), **changes))


def _ceil(n: int, f: int) -> int:
    return -(-n // f)


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.jp2", "a.j2k"])
def test_dispatch(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path, name, RAMP)
    image = load(path)
    assert type(image) is Jpeg2000MultiScaleImage
    assert isinstance(image, MultiScaleImage)
    assert [type(level) for level in image.images] == [Jpeg2000Image] * 4
    assert image.metadata.raw.container == name[-3:]
    one = load(path, level=1)
    assert type(one) is Jpeg2000Image
    assert isinstance(one, SingleScaleImage)
    # Without a name, the content alone tells the format.
    assert sniff(io.BytesIO(path.read_bytes())) is Jpeg2000MultiScaleImage
    assert type(load(io.BytesIO(path.read_bytes()))) is Jpeg2000MultiScaleImage


def test_dispatch_single_resolution_is_single_scale(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP, num_resolutions=1)
    image = load(path)
    assert type(image) is Jpeg2000Image
    assert image.metadata.raw.n_levels == 1


def test_sniff_scores() -> None:
    jp2 = _encode(RAMP)
    j2k = _encode(RAMP, j2k=True)
    assert Jpeg2000MultiScaleImage.sniff_bytes(jp2) == Confidence.CERTAIN
    assert Jpeg2000MultiScaleImage.sniff_bytes(j2k) == Confidence.CERTAIN
    assert Jpeg2000MultiScaleImage.sniff_bytes(jp2, level=0) == Confidence.NO
    assert Jpeg2000Image.sniff_bytes(jp2) == Confidence.LIKELY
    assert Jpeg2000Image.sniff_bytes(j2k) == Confidence.LIKELY
    # Pillow reads JPEG 2000, but only as a fallback.
    assert PillowImage.sniff_bytes(jp2) == Confidence.WEAK
    assert Jpeg2000Image.sniff_bytes(b"\x89PNG\r\n\x1a\n") == Confidence.NO
    assert Jpeg2000Metadata.sniff_bytes(jp2) == Confidence.LIKELY


def test_hints(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP)
    assert type(load(path, hint="pillow")) is PillowImage
    assert type(load(path, hint="jpeg2000")) is Jpeg2000MultiScaleImage
    assert type(load(path, hint="jp2")) is Jpeg2000MultiScaleImage
    codestream = tmp_path / "a.j2c"
    codestream.write_bytes(_encode(RAMP, j2k=True))
    assert type(load(codestream, hint="j2c")) is Jpeg2000MultiScaleImage


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
    assert (tmp_path / "a.j2k").read_bytes()[:4] == _boxes.J2K_SIGNATURE


def test_unknown_options_are_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP)
    with pytest.raises(TypeError, match="mmap"):
        Jpeg2000Image.from_filename(path, mmap=True)
    with pytest.raises(TypeError, match="level"):
        Jpeg2000MultiScaleImage.from_filename(path, level=1)
    with pytest.raises(TypeError, match="keep_open"):
        Jpeg2000Raw.from_fileobj(io.BytesIO(path.read_bytes()), keep_open=1)
    image = load(path, level=0)
    with pytest.raises(TypeError, match="colour"):
        image.to_bytes(colour=True)
    with pytest.raises(WriterError, match="signed"):
        image.to_bytes(signed=True)
    with pytest.raises(WriterError, match="container"):
        image.to_bytes(no_jp2=True)


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
    assert [level.shape for level in image.images] == shapes
    assert [level.level for level in image.images] == [0, 1, 2, 3]
    assert image.shape == (COLUMNS, ROWS)


def test_full_resolution_is_exact_f_ordered_and_read_only(
    tmp_path: Path,
) -> None:
    with backend("numpy"):
        image = load(_write(tmp_path, "a.jp2", RAMP), level=0)
        data = image.data
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data, RAMP.T)
    # A transposed view of what Pillow decodes, which cannot be edited in
    # place, since the writer would not see the edit.
    assert data.flags.f_contiguous and not data.flags.c_contiguous
    assert not data.flags.writeable
    assert isinstance(image.rawdata, _Jpeg2000Proxy)


def test_data_stays_lazy_with_dask(tmp_path: Path) -> None:
    da = pytest.importorskip("dask.array")
    with backend("dask"):
        image = load(_write(tmp_path, "a.jp2", RAMP), level=1)
        data = image.data
    assert isinstance(data, da.Array)
    np.testing.assert_array_equal(
        np.asarray(data), np.asarray(image.rawdata).T
    )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_reduced_level_values(tmp_path: Path, level: int) -> None:
    data = np.asarray(load(_write(tmp_path, "a.jp2", RAMP), level=level).data)
    f = 2**level
    # Pixel i of level r lies on the full-resolution pixel f * i, so the
    # interior of a ramp is kept exactly.
    expected = RAMP[::f, ::f].T
    inner = (slice(1, -1), slice(1, -1))
    np.testing.assert_array_equal(data[inner], expected[inner])


def test_reduced_level_rgb(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.jp2", RGB))
    assert image.images[0].shape == (COLUMNS, ROWS, 3)
    np.testing.assert_array_equal(
        np.asarray(image.images[0].data), RGB.transpose(1, 0, 2)
    )
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
        Jpeg2000Image.from_filename(path, level=4)
    with pytest.raises(ParserContentError, match="no level 4"):
        load(path, level=4)


def test_levels_are_decoded_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    decode = _data._decode_level

    def spy(source: tx.Any, record: tx.Any, level: int) -> tx.Any:
        calls.append(level)
        return decode(source, record, level)

    monkeypatch.setattr(_data, "_decode_level", spy)
    with backend("numpy"):
        image = load(_write(tmp_path, "a.jp2", RAMP))
        assert image._images is None
        assert image.shape == (COLUMNS, ROWS)
        assert calls == []
        _ = image.images[2].data
        _ = image.images[2].data
    assert calls == [2]


def test_an_empty_selection_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = load(_write(tmp_path, "a.jp2", RAMP), level=1)

    def fail(*args: tx.Any) -> tx.Any:
        raise AssertionError("The file was decoded.")

    monkeypatch.setattr(_data, "_decode_level", fail)
    empty = image.rawdata[0:0]
    assert empty.shape == (0, COLUMNS // 2 + 1)
    assert empty.dtype == np.uint16
    assert image.shape == (23, 19)


def test_read_from_bytes_and_fileobj() -> None:
    content = _encode(RAMP)
    image = Jpeg2000MultiScaleImage.from_bytes(content)
    np.testing.assert_array_equal(np.asarray(image.images[0].data), RAMP.T)
    stream = io.BytesIO(content)
    single = Jpeg2000Image.from_fileobj(stream, level=1)
    assert stream.tell() == 0
    assert single.data.shape == (23, 19)


# ----------------------------------------------------------------------
#   IMAGE OFFSET
# ----------------------------------------------------------------------


def _with_offset(content: bytes, offset: tx.Tuple[int, int]) -> bytes:
    """Move the image area of a codestream that Pillow wrote, and its only
    tile, on the reference grid. With an offset that is a multiple of every
    factor of the levels, the coded data are unchanged."""
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
    assert Jpeg2000MultiScaleImage.sniff_bytes(content) == Confidence.NO
    image = load(io.BytesIO(content))
    assert type(image) is Jpeg2000Image
    assert image.metadata.raw.image_offset == (8, 16)
    assert image.metadata.raw.size == (COLUMNS, ROWS)
    np.testing.assert_array_equal(np.asarray(image.data), RAMP.T)
    assert isinstance(image.transformation, Scaling)
    pyramid = Jpeg2000MultiScaleImage.from_bytes(content)
    assert isinstance(pyramid.images[1].transformation, Scaling)
    with pytest.raises(ParserContentError, match="offset"):
        np.asarray(pyramid.images[1].rawdata)


def test_odd_offset_geometry() -> None:
    content = _with_offset(_encode(RAMP, j2k=True), (3, 5))
    image = Jpeg2000MultiScaleImage.from_bytes(content)
    record = image.metadata.raw
    assert record.level_shape(0) == (COLUMNS, ROWS)
    # ceil((3 + 45) / 4) - ceil(3 / 4), ceil((5 + 37) / 4) - ceil(5 / 4)
    assert record.level_shape(2) == (11, 9)
    assert record.level_start(2) == (1, 3)
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
    image = Jpeg2000Image.from_bytes(content)
    assert image.metadata.raw.bit_depths == (12,)
    np.testing.assert_array_equal(np.asarray(image.data), twelve.T)

    signed = (np.arange(600) - 300).astype(np.int16).reshape(20, 30)
    content = imagecodecs.jpeg2k_encode(signed, level=0, codecformat="j2k")
    image = Jpeg2000Image.from_bytes(content)
    assert image.metadata.raw.signed == (True,)
    assert image.rawdata.dtype == np.int16
    np.testing.assert_array_equal(np.asarray(image.data), signed.T)

    colour = RNG.integers(0, 4096, (20, 30, 3)).astype(np.uint16)
    content = imagecodecs.jpeg2k_encode(
        colour, level=0, codecformat="jp2", bitspersample=12
    )
    with pytest.raises(ParserContentError, match="precision"):
        np.asarray(Jpeg2000Image.from_bytes(content).rawdata)


# ----------------------------------------------------------------------
#   PIXEL SIZE
# ----------------------------------------------------------------------


def test_pixel_size_unknown_by_default(tmp_path: Path) -> None:
    image = load(_write(tmp_path, "a.jp2", RAMP), level=0)
    assert image.metadata.raw.capture_resolution is None
    np.testing.assert_allclose(image.transformation.scale, [1, 1])
    assert [a.unit for a in image.transformation.output.axes] == [None] * 2


def test_pixel_size_from_capture_resolution() -> None:
    # 2000 points per mm along x and 1000 along y, which are 0.5 and 1
    # micrometre.
    content = _with(_encode(RAMP), capture_resolution=(2e6, 1e6))
    image = Jpeg2000MultiScaleImage.from_bytes(content)
    record = image.metadata.raw
    assert record.capture_resolution == pytest.approx((2e6, 1e6))
    np.testing.assert_allclose(
        image.images[1].transformation.scale, [1e-3, 2e-3]
    )
    units = image.images[0].transformation.output.axes
    assert [str(a.unit) for a in units] == ["millimeter"] * 2
    um = Jpeg2000Image.from_bytes(content, unit="um")
    np.testing.assert_allclose(um.transformation.scale, [0.5, 1])


def test_pixel_size_display_fallback_and_placeholders() -> None:
    content = _with(_encode(RAMP), display_resolution=(1e4, 1e4))
    image = Jpeg2000Image.from_bytes(content)
    np.testing.assert_allclose(image.transformation.scale, [0.1, 0.1])
    # 72 dpi is 2834.6 points per metre, which is a placeholder.
    dpi72 = 72 / 0.0254
    content = _with(_encode(RAMP), capture_resolution=(dpi72, dpi72))
    image = Jpeg2000Image.from_bytes(content)
    assert image.metadata.raw.capture_resolution == pytest.approx((dpi72,) * 2)
    np.testing.assert_allclose(image.transformation.scale, [1, 1])
    assert image.transformation.output.axes[0].unit is None
    # The options override it, and their transformations are assigned.
    image = Jpeg2000Image.from_bytes(content, pixel_size=(2, 3), unit="um")
    assert image._transformations is not None
    np.testing.assert_allclose(image.transformation.scale, [2, 3])


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.jp2", "b.j2k"])
def test_write_round_trip_lossless(tmp_path: Path, name: str) -> None:
    source = load(_write(tmp_path, "src.jp2", RGB), level=0)
    bio.save(source, tmp_path / name)
    back = load(tmp_path / name)
    np.testing.assert_array_equal(
        np.asarray(back.images[0].data), np.asarray(source.data)
    )
    # The number of resolutions of the file that was read is kept.
    assert back.metadata.raw.n_levels == 4
    assert back.metadata.raw.reversible


def test_an_untouched_file_is_written_byte_for_byte(tmp_path: Path) -> None:
    content = _with(
        _encode(RAMP, comment="hello", irreversible=True),
        capture_resolution=(4e6, 4e6),
        boxes=((b"xml ", b"<gml/>"),),
    )
    path = tmp_path / "a.jp2"
    path.write_bytes(content)
    for image in (load(path), load(path, level=0)):
        assert image.to_bytes() == content
        bio.save(image, tmp_path / "copy.jp2")
        assert (tmp_path / "copy.jp2").read_bytes() == content
        # Consolidating an untouched image shares its lazy array and its
        # metadata, and decodes nothing.
        consolidated = image.consolidate()
        assert consolidated.rawdata is image.rawdata
        assert consolidated.metadata is image.metadata


def test_a_new_container_copies_the_coded_image(tmp_path: Path) -> None:
    content = _encode(RAMP, irreversible=True, quality_layers=[20])
    codestream = content[content.index(_boxes.J2K_SIGNATURE) :]
    source = tmp_path / "a.jp2"
    source.write_bytes(content)
    load(source).save(tmp_path / "b.j2k")
    # The codestream is copied as it is, so a lossy file is not encoded
    # again, and the header of a JP2 file is made again around it.
    assert (tmp_path / "b.j2k").read_bytes() == codestream
    load(tmp_path / "b.j2k").save(tmp_path / "c.jp2")
    assert (tmp_path / "c.jp2").read_bytes().endswith(codestream)
    back = load(tmp_path / "c.jp2")
    assert back.metadata.raw.container == "jp2"
    assert not back.metadata.raw.reversible


def test_write_pixel_size_and_boxes(tmp_path: Path) -> None:
    content = _with(
        _encode(RAMP, comment="hello"),
        display_resolution=(1e4, 1e4),
        boxes=((b"xml ", b"<gml/>"),),
    )
    image = Jpeg2000Image.from_bytes(content, pixel_size=0.25, unit="um")
    out = tmp_path / "a.jp2"
    image.save(out)
    back = Jpeg2000Image.from_filename(out)
    record = back.metadata.raw
    assert record.capture_resolution == pytest.approx((4e6, 4e6))
    assert record.display_resolution == pytest.approx((1e4, 1e4))
    assert record.boxes == ((b"xml ", b"<gml/>"),)
    assert record.comments == ("hello",)
    np.testing.assert_allclose(back.transformation.scale, [2.5e-4] * 2)
    np.testing.assert_array_equal(np.asarray(back.data), RAMP.T)
    # Only the header changed, so the codestream is the same.
    assert out.read_bytes()[-200:] == content[-200:]


def test_a_geometry_without_pixel_size_drops_the_resolutions(
    tmp_path: Path,
) -> None:
    content = _with(_encode(RAMP), display_resolution=(1e4, 1e4))
    image = Jpeg2000Image.from_bytes(content)
    image.transformations = []
    image.save(tmp_path / "a.jp2")
    record = load(tmp_path / "a.jp2", level=0).metadata.raw
    assert record.capture_resolution is None
    assert record.display_resolution is None


def test_assigned_data_are_pending(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP)
    image = load(path, level=0)
    raw, metadata = image.rawdata, image.metadata
    image.data = RAMP.T // 2
    assert image.rawdata is raw
    assert image.metadata is metadata
    consolidated = image.consolidate()
    assert consolidated._data is None
    assert consolidated.metadata is not metadata
    np.testing.assert_array_equal(np.asarray(consolidated.rawdata), RAMP // 2)
    # The record of the file is the base of the new file.
    assert consolidated.metadata.raw.n_levels == 4
    image.save(path)
    assert image._data is None
    np.testing.assert_array_equal(
        np.asarray(load(path).images[0].data), RAMP.T // 2
    )


def test_a_reduced_level_is_written_as_a_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP, num_resolutions=4)
    content = path.read_bytes()
    path.write_bytes(_with(content, capture_resolution=(1e6, 1e6)))
    level = load(path, level=1)
    expected = np.asarray(level.data)
    level.save(tmp_path / "b.jp2")
    back = load(tmp_path / "b.jp2")
    # The level keeps the levels below it, and its own pixel size.
    assert back.metadata.raw.n_levels == 3
    np.testing.assert_array_equal(np.asarray(back.images[0].data), expected)
    np.testing.assert_allclose(back.images[0].transformation.scale, [2e-3] * 2)
    # Saved over its own file, the level holds what the file holds.
    level.save(path)
    assert level.level == 0
    assert level.metadata.raw.n_levels == 3
    np.testing.assert_array_equal(np.asarray(level.data), expected)


def test_write_options(tmp_path: Path) -> None:
    image = Jpeg2000Image.from_bytes(_encode(RAMP))
    content = image.to_bytes(num_resolutions=2)
    assert Jpeg2000Image.from_bytes(content).metadata.raw.n_levels == 2
    with pytest.raises(WriterError, match="resolution levels"):
        image.to_bytes(num_resolutions=7)
    lossy = image.to_bytes(
        irreversible=True, quality_mode="rates", quality_layers=[40]
    )
    assert not Jpeg2000Image.from_bytes(lossy).metadata.raw.reversible
    with pytest.raises(WriterError, match="raw codestream"):
        image.to_bytes(container="png")
    with pytest.raises(WriterError):
        Jpeg2000Image(
            data=RNG.random((5, 6)),
            transformations=raster.raster_transformations(
                raster.default_axes(2)
            ),
        ).to_bytes()


def test_write_comment(tmp_path: Path) -> None:
    image = Jpeg2000Image.from_bytes(_encode(RAMP, comment="hello"))
    content = image.to_bytes(comment="bye")
    assert read_record(content).comments == ("bye",)
    with pytest.raises(WriterError, match="Latin-1"):
        image.to_bytes(comment="☃")


def test_write_multiscale(tmp_path: Path) -> None:
    pyramid = load(_write(tmp_path, "a.jp2", RAMP))
    out = tmp_path / "b.j2k"
    bio.save(pyramid, out)
    back = load(out)
    assert type(back) is Jpeg2000MultiScaleImage
    assert len(back.images) == 4
    for level, other in zip(back.images, pyramid.images):
        np.testing.assert_array_equal(
            np.asarray(level.data), np.asarray(other.data)
        )

    # The shapes of the coarser levels are not checked: the codec derives
    # them from the first level.
    axes = raster.default_axes(2)
    odd = Jpeg2000MultiScaleImage(
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
    back = Jpeg2000MultiScaleImage.from_bytes(odd.to_bytes())
    assert [level.shape for level in back.images] == [(8, 8), (4, 4)]


def test_a_multiscale_image_of_fewer_levels(tmp_path: Path) -> None:
    pyramid = load(_write(tmp_path, "a.jp2", RAMP))
    pyramid.images = pyramid.images[:2]
    back = Jpeg2000MultiScaleImage.from_bytes(pyramid.to_bytes())
    assert back.metadata.raw.n_levels == 2
    # A pyramid built from levels of another class is encoded from its first
    # level.
    built = Jpeg2000MultiScaleImage.from_instance(
        MultiScaleImage(images=[SingleScaleImage(RAMP.T)], transformations=[])
    )
    assert built.metadata is None
    back = Jpeg2000MultiScaleImage.from_bytes(built.to_bytes())
    np.testing.assert_array_equal(np.asarray(back.data), RAMP.T)


def test_a_multiscale_image_writes_its_metadata_over_its_level() -> None:
    # The full-resolution level is read from another file, whose record
    # holds another comment, and the pyramid sets a field that the record
    # of a JPEG 2000 file cannot hold.
    content = _with(_encode(RAMP), comments=("pyramid",))
    pyramid = Jpeg2000MultiScaleImage.from_bytes(content)
    level = Jpeg2000MultiScaleImage.from_bytes(
        _with(_encode(RAMP), comments=("level",))
    ).images[0]
    pyramid = replace(
        pyramid,
        metadata=replace(pyramid.metadata, flip_angle=30.0),
        images=[level],
    )
    assert pyramid.images[0].metadata is not pyramid.metadata

    # The metadata of the pyramid is written, and the field that the
    # record cannot hold is reported as lost.
    report = ConversionReport(source="a", target="b")
    copy = pyramid.consolidate(on_loss=report)
    assert list(report.lost) == ["flip_angle"]
    assert copy.metadata.raw.comments == ("pyramid",)
    np.testing.assert_array_equal(np.asarray(copy.data), RAMP.T)
    with pytest.raises(MetadataLossError):
        pyramid.consolidate(on_loss="raise")


def test_a_multiscale_image_written_with_its_placement(
    tmp_path: Path,
) -> None:
    pyramid = load(_write(tmp_path, "a.jp2", RAMP))
    level = pyramid.images[0].transformation
    physical = raster.physical_system(
        raster.default_axes(2), {"x": "mm", "y": "mm"}
    )
    pyramid.transformations = [
        Scaling([0.5, 0.5], input=level.output, output=physical)
    ]
    consolidated = pyramid.consolidate()
    assert consolidated._transformations is None
    assert consolidated.transformations == []
    # The placement is composed with the full-resolution level, whose pixel
    # size is written, and the coarser levels follow.
    scale = consolidated.images[1].transformation.scale
    np.testing.assert_allclose(scale, [1.0, 1.0])
    # The pyramid itself is left as it was.
    assert len(pyramid.transformations) == 1
    assert pyramid.metadata.raw.capture_resolution is None


def test_copies_of_a_pyramid_stay_lazy(tmp_path: Path) -> None:
    pyramid = load(_write(tmp_path, "a.jp2", RAMP))
    for copy in (
        replace(pyramid),
        Jpeg2000MultiScaleImage.from_instance(pyramid),
    ):
        assert copy.rawdata is pyramid.rawdata
        assert copy.metadata is pyramid.metadata
        assert copy._images is None
    copy = MultiScaleImage.from_instance(pyramid)
    assert [level.shape for level in copy.images] == [
        level.shape for level in pyramid.images
    ]


def test_a_pyramid_saved_over_its_own_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP)
    pyramid = load(path, pixel_size=0.5, unit="mm")
    pyramid.save(path)
    assert pyramid._images is None
    assert pyramid.rawdata.source.is_file(path)
    np.testing.assert_allclose(
        pyramid.images[1].transformation.scale, [1.0, 1.0]
    )
    np.testing.assert_array_equal(np.asarray(pyramid.data), RAMP.T)
    assert load(path).metadata.raw == pyramid.metadata.raw


# ----------------------------------------------------------------------
#   RECORD AND METADATA
# ----------------------------------------------------------------------


def test_record_fields() -> None:
    record = Jpeg2000MultiScaleImage.from_bytes(
        _encode(RGB, tile_size=(16, 16))
    ).metadata.raw
    assert record.container == "jp2"
    assert record.size == (COLUMNS, ROWS)
    assert record.tile_size == (16, 16)
    assert record.bit_depths == (8, 8, 8)
    assert record.signed == (False,) * 3
    assert record.subsampling == ((1, 1),) * 3
    assert record.n_levels == 4
    assert record.n_layers == 1
    assert record.samples == 3
    assert record.dtype == np.uint8
    assert record.stored_shape(1) == (19, 23, 3)


def test_metadata_reads_the_header_alone(tmp_path: Path) -> None:
    content = _with(
        _encode(RAMP, comment="hello"), capture_resolution=(1e6, 2e6)
    )
    metadata = Jpeg2000Metadata.from_bytes(content)
    header = metadata.to_bytes()
    assert content.startswith(header)
    assert len(header) < len(content)
    # A file cut after its header gives the same record.
    cut = Jpeg2000Metadata.from_bytes(header)
    assert cut.raw == metadata.raw
    clone = pickle.loads(pickle.dumps(metadata))
    assert clone.raw == metadata.raw
    # An edited record writes its comments and resolutions.
    edited = replace(
        metadata.raw, comments=("edited",), capture_resolution=None
    )
    back = read_record(edited.to_bytes())
    assert back.comments == ("edited",)
    assert back.capture_resolution is None
    # The fields that the coded image decides cannot be edited.
    with pytest.raises(WriterError, match="decided by the coded image"):
        replace(metadata.raw, n_levels=2).to_bytes()
    with pytest.raises(WriterError, match="no record"):
        Jpeg2000Metadata().to_bytes()


def test_corrupt_files() -> None:
    with pytest.raises(ParserContentError):
        Jpeg2000Image.from_bytes(_boxes.JP2_SIGNATURE + b"\x00\x00")
    with pytest.raises(ParserContentError):
        Jpeg2000Image.from_bytes(_boxes.J2K_SIGNATURE + b"\x00" * 10)


def test_resolution_box_encoding() -> None:
    for value in (1.0, 2834.6, 1e4, 123456.7, 4e6, 0.001):
        box = _boxes._resolution_box(b"resc", (value, value))
        back = _boxes.read_resolution(box[8:])
        assert back == pytest.approx((value, value), rel=1e-6)


def test_a_coarser_level_that_was_assigned_is_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.jp2", RAMP)
    derived = Jpeg2000MultiScaleImage.from_bytes(load(path).to_bytes())
    # A pyramid is written from its first level, and the codec derives the
    # coarser levels again, so data assigned to a coarser level, even of
    # another shape, are ignored rather than refused.
    pyramid = load(path)
    pyramid.images[2].data = np.zeros((5, 7), np.uint16)
    level = pyramid.images[1].transformation
    pyramid.images[1].transformations = [
        Scaling([3.0, 3.0], input=level.input, output=level.output)
    ]
    content = pyramid.to_bytes()
    assert content == path.read_bytes()
    pyramid.save(tmp_path / "b.jp2")
    back = load(tmp_path / "b.jp2")
    assert len(back.images) == len(derived.images)
    for got, expected in zip(back.images, derived.images):
        np.testing.assert_array_equal(
            np.asarray(got.data), np.asarray(expected.data)
        )
        np.testing.assert_allclose(
            got.transformation.to(Affine).homogeneous_matrix,
            expected.transformation.to(Affine).homogeneous_matrix,
        )
    np.testing.assert_array_equal(
        np.asarray(back.images[0].data), np.asarray(load(path).images[0].data)
    )
    # A changed first level gives the coarser levels that the codec derives
    # from it, whatever was assigned to them.
    first = RAMP.T[:, ::-1].copy()
    expected = load(path)
    expected.images[0].data = first
    expected = Jpeg2000MultiScaleImage.from_bytes(expected.to_bytes())
    pyramid = load(path)
    pyramid.images[0].data = first
    pyramid.images[3].data = np.ones((6, 5), np.uint16)
    back = Jpeg2000MultiScaleImage.from_bytes(pyramid.to_bytes())
    np.testing.assert_array_equal(np.asarray(back.images[0].data), first)
    for got, want in zip(back.images, expected.images):
        np.testing.assert_array_equal(
            np.asarray(got.data), np.asarray(want.data)
        )


def test_a_pixel_size_is_written_from_the_first_level(tmp_path: Path) -> None:
    # The pixel size given when reading becomes the transformations of every
    # level, and the file stores that of the first level, from which level 3
    # is derived.
    pyramid = load(_write(tmp_path, "a.jp2", RAMP), pixel_size=0.5, unit="mm")
    back = Jpeg2000MultiScaleImage.from_bytes(pyramid.to_bytes())
    np.testing.assert_allclose(back.images[3].transformation.scale, [4, 4])
