"""Tests for whole-slide images read through OpenSlide.

Sample slides cannot be downloaded, so tifffile writes a generic tiled
pyramidal TIFF and an Aperio-like SVS. Other vendors need real files, so
their declarations are checked, and their sniffing is checked with a
detection that is forced. The conformance of the classes to the design of
format records is checked by `tests/test_io_conformance.py`, through the
generic TIFF vendor.
"""

import gc
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import replace

openslide = pytest.importorskip("openslide")
tifffile = pytest.importorskip("tifffile")

from brainhops.datamodel.images import (  # noqa: E402
    MultiScaleImage,
    SingleScaleImage,
)
from brainhops.datamodel.transformations import Scaling  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.base.specs import format_hints  # noqa: E402
from brainhops.io.images import load  # noqa: E402
from brainhops.io.images import openslide as bos  # noqa: E402
from brainhops.io.images.openslide import _base  # noqa: E402
from brainhops.io.images.openslide import _data as slides  # noqa: E402
from brainhops.io.images.tiff import (  # noqa: E402
    TiffImage,
    TiffMultiScaleImage,
)
from brainhops.io.images.tiff import _utils as tiff_backend  # noqa: E402

RNG = np.random.default_rng(0)
BASE = RNG.integers(0, 256, (512, 768, 3), dtype=np.uint8)  # y, x, c
APERIO = "Aperio Image Library v10.0.0\n"
MPP = 0.499

VENDOR_CLASSES = [
    (bos.AperioImage, bos.AperioMultiScaleImage, "aperio", "svs"),
    (bos.HamamatsuImage, bos.HamamatsuMultiScaleImage, "hamamatsu", "ndpi"),
    (bos.MiraxImage, bos.MiraxMultiScaleImage, "mirax", "mrxs"),
    (bos.LeicaImage, bos.LeicaMultiScaleImage, "leica", "scn"),
    (bos.PhilipsImage, bos.PhilipsMultiScaleImage, "philips", "philips"),
    (bos.VentanaImage, bos.VentanaMultiScaleImage, "ventana", "bif"),
    (bos.SakuraImage, bos.SakuraMultiScaleImage, "sakura", "svslide"),
    (bos.TrestleImage, bos.TrestleMultiScaleImage, "trestle", "trestle"),
    (bos.ZeissImage, bos.ZeissMultiScaleImage, "zeiss", "czi"),
    (bos.DicomWsiImage, bos.DicomWsiMultiScaleImage, "dicom", "dicom-wsi"),
    (
        bos.GenericTiffImage,
        bos.GenericTiffMultiScaleImage,
        "generic-tiff",
        "generic-tiff",
    ),
]


def _generic(path: Path) -> Path:
    """A tiled pyramidal TIFF with three levels in successive directories."""
    with tifffile.TiffWriter(path) as w:
        image = BASE
        for k in range(3):
            w.write(
                image,
                tile=(128, 128),
                photometric="rgb",
                compression="zlib",
                subfiletype=1 if k else 0,
                metadata=None,
            )
            image = image[::2, ::2]
    return path


def _svs(path: Path, mpp: bool = True) -> Path:
    """An Aperio-like SVS: three levels, a thumbnail, a label and a macro."""
    suffix = f"|MPP = {MPP}" if mpp else ""

    def write(
        w: tx.Any, image: np.ndarray, text: str, tiled: bool = True, **kw
    ) -> None:
        w.write(
            image,
            tile=(128, 128) if tiled else None,
            photometric="rgb",
            compression="zlib",
            description=APERIO + text + suffix,
            metadata=None,
            **kw,
        )

    with tifffile.TiffWriter(path) as w:
        write(w, BASE, "768x512 [0,0 768x512] (128x128) RGB|AppMag = 20")
        write(w, BASE[::8, ::8], "768x512 -> 96x64", tiled=False)
        write(w, BASE[::2, ::2], "768x512 -> 384x256", subfiletype=1)
        write(w, BASE[::4, ::4], "768x512 -> 192x128", subfiletype=1)
        write(w, BASE[:40, :50], "label 50x40", tiled=False, subfiletype=1)
        write(w, BASE[:30, :60], "macro 60x30", tiled=False, subfiletype=9)
    return path


def _reference(path: Path, level: int) -> np.ndarray:
    """A level as OpenSlide reads it, in (x, y, c) RGB order."""
    with openslide.OpenSlide(str(path)) as slide:
        size = slide.level_dimensions[level]
        rgba = np.asarray(slide.read_region((0, 0), level, size))
    return rgba[..., :3].transpose(1, 0, 2)


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


@pytest.fixture
def svs(tmp_path: Path) -> Path:
    return _svs(tmp_path / "slide.svs")


@pytest.fixture
def generic(tmp_path: Path) -> Path:
    return _generic(tmp_path / "slide.tif")


# ----------------------------------------------------------------------
#   CLASSES AND HINTS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("single, multi, vendor, hint", VENDOR_CLASSES)
def test_vendor_classes(
    single: type, multi: type, vendor: str, hint: str
) -> None:
    for cls in (single, multi):
        hints = format_hints(cls)
        assert "openslide" in hints
        assert hint in hints
        assert f"openslide.{hint}" in hints
        assert cls.EXTENSIONS
    assert single.EXTENSIONS == multi.EXTENSIONS
    assert issubclass(single, bos.OpenSlideImage)
    assert issubclass(single, SingleScaleImage)
    assert issubclass(multi, bos.OpenSlideMultiScaleImage)
    assert issubclass(multi, MultiScaleImage)
    # The levels of a pyramid are objects of the class of its vendor.
    level = np.zeros((4, 2, 3), dtype=np.uint8)
    pyramid = multi(rawdata=(level, level[::2, ::2]))
    assert [type(image) for image in pyramid.images] == [single, single]
    assert [image.level for image in pyramid.images] == [0, 1]


@pytest.mark.parametrize("single, multi, vendor, hint", VENDOR_CLASSES)
def test_vendor_sniffing(
    single: type,
    multi: type,
    vendor: str,
    hint: str,
    svs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each class sniffs its own vendor, which OpenSlide is made to detect.
    score = (
        Confidence.MAYBE if vendor == "generic-tiff" else Confidence.CERTAIN
    )
    detected = {"vendor": vendor, "levels": 3}
    monkeypatch.setattr(
        _base, "detect", lambda _: (detected["vendor"], detected["levels"])
    )
    assert multi.sniff(svs) == score
    assert multi.sniff(svs, level=1) == Confidence.NO
    assert single.sniff(svs, level=1) == score
    assert single.sniff(svs) == pytest.approx(0.8 * score)
    detected["levels"] = 1
    assert multi.sniff(svs) == Confidence.NO
    assert single.sniff(svs) == score
    detected["vendor"] = "another"
    for cls in (single, multi):
        assert cls.sniff(svs) == Confidence.NO
        with pytest.raises(ParserContentError):
            cls.sniff(svs, error=ParserContentError)


# ----------------------------------------------------------------------
#   APERIO
# ----------------------------------------------------------------------


def test_svs_is_aperio_pyramid(svs: Path) -> None:
    image = load(svs)
    assert isinstance(image, bos.AperioMultiScaleImage)
    record = image.metadata.raw
    assert isinstance(record, bos.OpenSlideRaw)
    assert record.vendor == "aperio"
    assert len(record.levels) == 3
    assert record.level_downsamples == (1.0, 2.0, 4.0)
    assert record.levels[1].shape == (384, 256, 3)
    assert record.properties["openslide.mpp-x"] == str(MPP)
    assert record.properties["aperio.AppMag"] == "20"
    assert set(record.associated_images) == {"thumbnail", "label", "macro"}
    assert len(image.images) == 3
    for k, level in enumerate(image.images):
        assert isinstance(level, bos.AperioImage)
        assert level.level == k
        assert level.metadata is image.metadata
        assert level.rawdata is image.rawdata[k]
        f = 2.0**k
        assert level.data.shape == (768 // 2**k, 512 // 2**k, 3)
        assert level.data.dtype == np.uint8
        np.testing.assert_allclose(_scale(level), [MPP * f, MPP * f, 1])
        np.testing.assert_allclose(
            _shift(level), [MPP * (f - 1) / 2, MPP * (f - 1) / 2, 0]
        )
        units = [str(a.unit) for a in level.transformation.output.axes[:2]]
        assert units == ["micrometer", "micrometer"]
        np.testing.assert_array_equal(
            np.asarray(level.data), _reference(svs, k)
        )


def test_svs_level(svs: Path) -> None:
    image = load(svs, level=1)
    assert isinstance(image, bos.AperioImage)
    assert image.level == 1
    assert image.data.shape == (384, 256, 3)
    assert isinstance(load(svs, level=-1), bos.AperioImage)
    assert load(svs, level=-1).level == 2
    with pytest.raises(ParserContentError):
        load(svs, level=3)


def test_svs_associated_image(svs: Path) -> None:
    image = load(svs)
    label = image.associated_image("label")
    assert label.shape == (50, 40, 3)
    assert label.dtype == np.uint8
    np.testing.assert_array_equal(label, BASE[:40, :50].transpose(1, 0, 2))
    with pytest.raises(KeyError):
        image.associated_image("nothing")


def test_svs_without_mpp(tmp_path: Path) -> None:
    image = load(_svs(tmp_path / "x.svs", mpp=False))
    level = image.images[1]
    np.testing.assert_allclose(_scale(level), [2, 2, 1])
    np.testing.assert_allclose(_shift(level), [0.5, 0.5, 0])
    assert all(a.unit is None for a in level.transformation.output.axes)


def test_pixel_size_overrides(svs: Path) -> None:
    image = load(svs, pixel_size=2.0)
    np.testing.assert_allclose(_scale(image.images[1]), [4, 4, 1])
    image = load(svs, unit="mm")
    np.testing.assert_allclose(
        _scale(image.images[0]), [MPP / 1000, MPP / 1000, 1]
    )


# ----------------------------------------------------------------------
#   LAZY READS
# ----------------------------------------------------------------------


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> tx.List[tx.Any]:
    log = []
    original = openslide.OpenSlide.read_region

    def read_region(
        self: tx.Any, location: tx.Any, level: int, size: tx.Any
    ) -> tx.Any:
        log.append((tuple(location), level, tuple(size)))
        return original(self, location, level, size)

    monkeypatch.setattr(openslide.OpenSlide, "read_region", read_region)
    return log


def test_regions_are_read_lazily(svs: Path, calls: tx.List[tx.Any]) -> None:
    image = load(svs, lazy=False)
    data = image.images[1].data
    assert isinstance(data, slides._SlideArray)
    assert calls == []
    region = data[10:20, 30:35]
    assert region.shape == (10, 5, 3)
    # Indexing reads nothing, and converting the region reads it alone.
    assert calls == []
    values = np.asarray(region)
    # Level coordinates (10, 30) are level-0 coordinates (20, 60).
    assert calls == [((20, 60), 1, (10, 5))]
    np.testing.assert_array_equal(values, _reference(svs, 1)[10:20, 30:35])


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), slice(None)),
        (5, 7),
        (5,),
        (-1, -2, 1),
        (slice(3, 50, 4), slice(None, None, -3)),
        (slice(40, 3, -5), 2, slice(None, 2)),
        (Ellipsis, 0),
        ([3, 1, 7], slice(2, 9)),
        (slice(2, 2), slice(None)),
        (np.int64(4), Ellipsis),
    ],
)
def test_indexing_matches_numpy(svs: Path, key: tx.Any) -> None:
    data = load(svs, lazy=False).images[1].data
    expected = _reference(svs, 1)
    if isinstance(key[0], list):
        want = expected[key[0]][:, key[1]]
    else:
        want = expected[key]
    np.testing.assert_array_equal(data[key], want)


def test_dask(svs: Path, calls: tx.List[tx.Any]) -> None:
    pytest.importorskip("dask.array")
    data = load(svs, lazy=True).images[0].data
    assert type(data).__module__.startswith("dask")
    assert data.shape == (768, 512, 3)
    assert data.chunksize == (768, 512, 3)  # grouped into larger chunks
    assert calls == []
    np.testing.assert_array_equal(
        data[:100, :50].compute(), _reference(svs, 0)[:100, :50]
    )


def test_background_composite(
    svs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    data = load(svs, lazy=False).images[0].data
    properties = {**data.record.properties}
    properties["openslide.background-color"] = "102030"
    data.record = replace(data.record, properties=properties)
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    rgba[0, 0, 3] = 255
    rgba[0, 1, 3] = 0

    def read_region(
        self: tx.Any, location: tx.Any, level: int, size: tx.Any
    ) -> tx.Any:
        return Image.fromarray(rgba, "RGBA")

    monkeypatch.setattr(openslide.OpenSlide, "read_region", read_region)
    out = np.asarray(data[:3, :2])
    assert out.shape == (3, 2, 3)
    assert tuple(out[0, 0]) == (200, 200, 200)
    assert tuple(out[1, 0]) == (16, 32, 48)
    assert slides._hex_colour(None).tolist() == [255, 255, 255]


# ----------------------------------------------------------------------
#   GENERIC TIFF, AND PRIORITIES OVER THE TIFF READER
# ----------------------------------------------------------------------


def test_generic_tiff_stays_with_tiff(generic: Path) -> None:
    assert openslide.OpenSlide.detect_format(str(generic)) == "generic-tiff"
    assert isinstance(load(generic), TiffMultiScaleImage)
    assert isinstance(load(generic, level=1), TiffImage)
    image = load(generic, hint="openslide")
    assert isinstance(image, bos.GenericTiffMultiScaleImage)
    assert len(image.metadata.raw.levels) == 3
    level = load(generic, hint="openslide", level=2)
    assert isinstance(level, bos.GenericTiffImage)
    np.testing.assert_array_equal(
        np.asarray(level.data), BASE[::4, ::4].transpose(1, 0, 2)
    )
    # Without a recorded pixel size, the axes have no unit.
    assert all(
        a.unit is None for a in image.images[0].transformation.output.axes
    )


def test_sniff_scores(svs: Path, generic: Path) -> None:
    assert bos.AperioMultiScaleImage.sniff(svs) == Confidence.CERTAIN
    assert bos.AperioMultiScaleImage.sniff(svs, level=0) == Confidence.NO
    assert bos.AperioImage.sniff(svs, level=0) == Confidence.CERTAIN
    assert Confidence.NO < bos.AperioImage.sniff(svs) < Confidence.CERTAIN
    assert bos.AperioMultiScaleImage.sniff(generic) == Confidence.NO
    assert bos.HamamatsuMultiScaleImage.sniff(svs) == Confidence.NO
    assert bos.GenericTiffMultiScaleImage.sniff(generic) == Confidence.MAYBE
    assert TiffMultiScaleImage.sniff(generic) > Confidence.MAYBE
    # Nothing is sniffed in memory.
    assert bos.AperioMultiScaleImage.sniff(svs.read_bytes()) == Confidence.NO


def test_tiff_yields_vendor_pyramids(
    generic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fixture cannot reproduce how tifffile reads a vendor pyramid, so
    # the whole-slide flag is forced.
    assert TiffMultiScaleImage.sniff(generic) == Confidence.CERTAIN
    monkeypatch.setattr(tiff_backend, "is_whole_slide", lambda tif: True)
    score = TiffMultiScaleImage.sniff(generic)
    assert TiffImage.sniff(generic) < score < Confidence.CERTAIN


def test_svs_by_tiff_hint(svs: Path) -> None:
    assert isinstance(load(svs, hint="tiff"), (TiffImage, TiffMultiScaleImage))


# ----------------------------------------------------------------------
#   SOURCES
# ----------------------------------------------------------------------


def test_open_file(svs: Path) -> None:
    with open(svs, "rb") as f:
        image = load(f)
    assert isinstance(image, bos.AperioMultiScaleImage)
    assert image.rawdata[0].slide.filename == svs
    assert image.images[0].rawdata.slide is image.rawdata[0].slide


def test_bytes_by_hint(svs: Path) -> None:
    content = svs.read_bytes()
    assert not isinstance(load(content), bos.OpenSlideMultiScaleImage)
    image = load(content, hint="openslide")
    assert isinstance(image, bos.AperioMultiScaleImage)
    np.testing.assert_array_equal(
        np.asarray(image.images[2].data), _reference(svs, 2)
    )
    level = load(content, hint="aperio", level=1)
    assert isinstance(level, bos.AperioImage)
    assert level.level == 1
    # The temporary copy is deleted with the images.
    tempdir = Path(image.rawdata[0].slide._state["tempdir"])
    assert tempdir.exists()
    del image
    gc.collect()
    assert not tempdir.exists()


def test_wrong_vendor(generic: Path) -> None:
    with pytest.raises(ParserContentError):
        bos.AperioMultiScaleImage.load(generic)
    with pytest.raises(ParserContentError):
        load(generic, hint="aperio")


def test_options_are_checked(svs: Path) -> None:
    with pytest.raises(TypeError):
        bos.AperioMultiScaleImage.load(svs, level=1)
    with pytest.raises(TypeError):
        bos.AperioImage.load(svs, series=1)


def test_missing_openslide() -> None:
    code = (
        "import sys; sys.modules['openslide'] = None\n"
        "from brainhops.io.images import load\n"
        "try:\n"
        "    load(b'II*\\x00', hint='svs')\n"
        "except Exception as e:\n"
        "    print(e)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "brainhops[openslide]" in out


# ----------------------------------------------------------------------
#   RECORD, METADATA AND CONSOLIDATION
# ----------------------------------------------------------------------


def test_metadata_reads_the_record_alone(svs: Path) -> None:
    metadata = bos.OpenSlideMetadata.load(svs)
    assert isinstance(metadata, bos.OpenSlideMetadata)
    assert metadata.raw == load(svs).metadata.raw
    assert bos.OpenSlideMetadata.from_bytes(svs.read_bytes()).raw == (
        metadata.raw
    )
    # The record holds no open slide, so it can be pickled.
    assert pickle.loads(pickle.dumps(metadata)).raw == metadata.raw
    record = metadata.raw
    assert record.background_color is None
    assert record.bounds is None
    properties = {**record.properties, "openslide.background-color": "AABBCC"}
    for key, value in zip("xy", (3, 4)):
        properties[f"openslide.bounds-{key}"] = str(value)
    properties["openslide.bounds-width"] = "10"
    properties["openslide.bounds-height"] = "20"
    edited = replace(record, properties=properties)
    assert edited.background_color == "AABBCC"
    assert edited.bounds == (3, 4, 10, 20)
    with pytest.raises(TypeError):
        bos.OpenSlideMetadata.load(svs, level=1)


def test_the_bases_read_any_vendor(svs: Path, generic: Path) -> None:
    for path, single in ((svs, bos.AperioImage), (generic, None)):
        pyramid = bos.OpenSlideMultiScaleImage.load(path)
        assert type(pyramid) is bos.OpenSlideMultiScaleImage
        assert all(type(i) is bos.OpenSlideImage for i in pyramid.images)
        level = bos.OpenSlideImage.load(path, level=1)
        assert type(level) is bos.OpenSlideImage
        if single is not None:
            assert np.array_equal(
                np.asarray(level.data),
                np.asarray(single.load(path, level=1).data),
            )


def test_options_are_assigned_transformations(svs: Path) -> None:
    image = load(svs, level=1, pixel_size=2.0)
    assert image._transformations is not None
    np.testing.assert_allclose(_scale(image), [4, 4, 1])
    pyramid = load(svs, pixel_size=2.0)
    assert pyramid._images is not None
    assert all(level.metadata is pyramid.metadata for level in pyramid.images)
    # Without options, nothing is assigned.
    assert load(svs, level=1)._transformations is None
    assert load(svs)._images is None


def test_lazy_is_kept_by_copies(svs: Path) -> None:
    pytest.importorskip("dask.array")
    level = load(svs, level=1, lazy=True)
    for copy in (replace(level), type(level).from_instance(level)):
        assert type(copy.data).__module__.startswith("dask")


def test_a_level_consolidates_its_data_only(svs: Path) -> None:
    image = load(svs, level=2)
    value = np.zeros(image.shape, dtype=np.uint8)
    image.data = value
    consolidated = image.consolidate()
    assert consolidated.rawdata is value
    assert consolidated._data is None
    assert consolidated.metadata is image.metadata
    assert consolidated.level == 2
    assert isinstance(image.rawdata, slides._SlideArray)
    # The associated images are still read from the slide of the record.
    assert image.associated_image("label").shape == (50, 40, 3)
    with pytest.raises(KeyError):
        consolidated.associated_image("label")
    with pytest.raises(WriterError):
        from brainhops import io

        io.save(image, svs.with_name("copy.svs"))


def test_a_pyramid_consolidates_its_data_only(svs: Path) -> None:
    pyramid = load(svs)
    raw, metadata = pyramid.rawdata, pyramid.metadata
    value = np.zeros(pyramid.images[1].shape, dtype=np.uint8)
    pyramid.images[1].data = value
    consolidated = pyramid.consolidate()
    assert consolidated.metadata is metadata
    assert consolidated.rawdata[1] is value
    assert consolidated.rawdata[0] is raw[0]
    assert consolidated._images is None
    # The associated images are read through any level of the slide.
    assert consolidated.associated_image("macro").shape == (60, 30, 3)
    # A transformation assigned to a level is kept with its level.
    pyramid = load(svs)
    pyramid.images[2].transformations = [Scaling([3.0, 3.0, 1.0])]
    consolidated = pyramid.consolidate()
    assert len(consolidated.rawdata) == len(pyramid.rawdata)
    assert all(a is b for a, b in zip(consolidated.rawdata, pyramid.rawdata))
    assert consolidated._images is not None
    np.testing.assert_allclose(_scale(consolidated.images[2]), [3, 3, 1])
    assert consolidated.images[0]._transformations is None
    twice = consolidated.consolidate()
    assert len(twice.rawdata) == len(consolidated.rawdata)
    assert all(a is b for a, b in zip(twice.rawdata, consolidated.rawdata))
    np.testing.assert_allclose(_scale(twice.images[2]), [3, 3, 1])
    # A level that is not a level of the slide cannot be described.
    pyramid = load(svs)
    pyramid.images = list(reversed(pyramid.images))
    with pytest.raises(WriterError):
        pyramid.consolidate()
    pyramid = load(svs)
    pyramid.images = pyramid.images[:2]
    with pytest.raises(WriterError):
        pyramid.consolidate()


@pytest.mark.parametrize("inplace", [False, True])
def test_kept_levels_hold_the_consolidated_metadata(
    svs: Path, inplace: bool
) -> None:
    # A field set on a pyramid whose levels are kept gives the levels the
    # metadata of the copy, so that the copy can be consolidated again.
    pyramid = load(svs)
    metadata = pyramid.metadata
    pyramid.images[2].transformations = [Scaling([3.0, 3.0, 1.0])]
    copy = pyramid.consolidate(
        inplace=inplace, on_loss="ignore", name="renamed"
    )
    assert copy._images is not None
    assert copy.metadata is not None
    assert copy.metadata is not metadata
    assert all(level.metadata is copy.metadata for level in copy.images)
    twice = copy.consolidate()
    np.testing.assert_allclose(_scale(twice.images[2]), [3, 3, 1])


def test_consolidate_keeps_a_list_changed_in_place(svs: Path) -> None:
    # A slide cannot be written, so a list decoded and then changed in
    # place is given to the copy as an assigned list.
    image = load(svs, level=1)
    decoded = image.transformations
    assert image._transformations is None
    image.transformations[:] = [Scaling([5.0, 5.0, 1.0])]
    consolidated = image.consolidate()
    assert consolidated._transformations is not None
    np.testing.assert_allclose(_scale(consolidated), [5, 5, 1])
    image.data = np.zeros(image.shape, dtype=np.uint8)
    np.testing.assert_allclose(_scale(image.consolidate()), [5, 5, 1])
    # A list that is the decoded one is left to the record.
    image = load(svs, level=1)
    assert len(image.transformations) == len(decoded)
    assert image.consolidate()._transformations is None
    # The placements of a pyramid, which a slide never stores, likewise.
    pyramid = load(svs)
    assert pyramid.consolidate()._transformations is None
    pyramid.transformations.append(Scaling([2.0, 2.0, 1.0]))
    assert pyramid._transformations is None
    consolidated = pyramid.consolidate()
    assert len(consolidated._transformations) == 1
    assert consolidated.transformations is consolidated._transformations
    pyramid.images[1].data = np.zeros(pyramid.images[1].shape, np.uint8)
    assert len(pyramid.consolidate()._transformations) == 1
