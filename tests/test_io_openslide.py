"""
Tests for whole-slide images read with OpenSlide.

OpenSlide's sample slides cannot be downloaded here, so the fixtures are
generated with tifffile: a generic tiled pyramidal TIFF, and an
Aperio-like SVS (a tiled TIFF whose `ImageDescription` starts with
`Aperio`, with a thumbnail, a label and a macro image), both of which
OpenSlide recognizes. The other vendors (Hamamatsu, MIRAX, Leica,
Philips, Ventana, Sakura, Trestle, Zeiss, DICOM) need real files and are
only checked for their declarations.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

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
)
from brainhops.io.base.specs import format_hints  # noqa: E402
from brainhops.io.images import load  # noqa: E402
from brainhops.io.images import openslide as bos  # noqa: E402
from brainhops.io.images.openslide import _base  # noqa: E402
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
    """A tiled pyramidal TIFF: three levels as successive directories."""
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
    """An Aperio-like SVS: three tiled levels, a thumbnail, a label and a
    macro image."""
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
    """A level as OpenSlide reads it, as an (x, y, c) RGB array."""
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
        assert cls.VENDOR == vendor
        hints = format_hints(cls)
        assert "openslide" in hints
        assert hint in hints
        assert f"openslide.{hint}" in hints
        assert cls.EXTENSIONS
    assert issubclass(single, bos.OpenSlideImage)
    assert issubclass(single, SingleScaleImage)
    assert issubclass(multi, bos.OpenSlideMultiScaleImage)
    assert issubclass(multi, MultiScaleImage)
    assert _base._LEVEL_CLASSES[vendor] is single


# ----------------------------------------------------------------------
#   APERIO
# ----------------------------------------------------------------------


def test_svs_is_aperio_pyramid(svs: Path) -> None:
    image = load(svs)
    assert isinstance(image, bos.AperioMultiScaleImage)
    assert image.vendor == "aperio"
    assert image.n_levels == 3
    assert image.level_downsamples == (1.0, 2.0, 4.0)
    assert image.properties["openslide.mpp-x"] == str(MPP)
    assert image.properties["aperio.AppMag"] == "20"
    assert set(image.associated_images) == {"thumbnail", "label", "macro"}
    assert len(image.images) == 3
    for k, level in enumerate(image.images):
        assert isinstance(level, bos.AperioImage)
        assert level.level == k
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
    """Record every `read_region` call."""
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
    assert isinstance(data, _base._SlideArray)
    assert calls == []
    region = data[10:20, 30:35]
    assert region.shape == (10, 5, 3)
    # Level coordinates (10, 30) are level-0 coordinates (20, 60).
    assert calls == [((20, 60), 1, (10, 5))]
    np.testing.assert_array_equal(region, _reference(svs, 1)[10:20, 30:35])


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
    assert data.chunksize == (768, 512, 3)  # tiles grouped to >= 1024
    assert calls == []
    np.testing.assert_array_equal(
        data[:100, :50].compute(), _reference(svs, 0)[:100, :50]
    )


def test_background_composite(
    svs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    data = load(svs, lazy=False).images[0].data
    data._background = _base._hex_colour("102030")
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    rgba[0, 0, 3] = 255
    rgba[0, 1, 3] = 0

    def read_region(
        self: tx.Any, location: tx.Any, level: int, size: tx.Any
    ) -> tx.Any:
        return Image.fromarray(rgba, "RGBA")

    monkeypatch.setattr(openslide.OpenSlide, "read_region", read_region)
    out = data[:3, :2]
    assert out.shape == (3, 2, 3)
    assert tuple(out[0, 0]) == (200, 200, 200)
    assert tuple(out[1, 0]) == (16, 32, 48)
    assert _base._hex_colour(None).tolist() == [255, 255, 255]


# ----------------------------------------------------------------------
#   GENERIC TIFF, AND PRIORITIES OVER THE TIFF READER
# ----------------------------------------------------------------------


def test_generic_tiff_stays_with_tiff(generic: Path) -> None:
    assert openslide.OpenSlide.detect_format(str(generic)) == "generic-tiff"
    assert isinstance(load(generic), TiffMultiScaleImage)
    assert isinstance(load(generic, level=1), TiffImage)
    image = load(generic, hint="openslide")
    assert isinstance(image, bos.GenericTiffMultiScaleImage)
    assert image.n_levels == 3
    level = load(generic, hint="openslide", level=2)
    assert isinstance(level, bos.GenericTiffImage)
    np.testing.assert_array_equal(
        np.asarray(level.data), BASE[::4, ::4].transpose(1, 0, 2)
    )
    # No pixel size recorded: the identity.
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
    # In memory, nothing is sniffed.
    assert bos.AperioMultiScaleImage.sniff(svs.read_bytes()) == Confidence.NO


def test_tiff_yields_vendor_pyramids(
    generic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tifffile reads a vendor pyramid as one series with levels; the fake
    # SVS has a single level per series, so the flag is forced here.
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
    assert image.images[0]._slide.filename == str(svs)


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
    tempdir = Path(image._slide._state["tempdir"])
    assert tempdir.exists()
    del image
    import gc

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
