"""Tests for image stacks: folders of numbered TIFF and Pillow images.

The folders are written with tifffile and Pillow in each test, so that
each test states the names, the layout and the metadata of its files.
The checks of the format-record design are in `test_io_conformance.py`.
"""

import os
import pickle
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

tifffile = pytest.importorskip("tifffile")
Image = pytest.importorskip("PIL.Image")
da = pytest.importorskip("dask.array")

import brainhops.io as io  # noqa: E402
from brainhops._core.backends import backend  # noqa: E402
from brainhops.datamodel.transformations import Scaling  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    ParserError,
    ParserExistsError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.images.base import _utils_raster as raster  # noqa: E402
from brainhops.io.images.pillow import PillowImage  # noqa: E402
from brainhops.io.images.pillow._utils import (  # noqa: E402
    PillowHeader,
    read_pillow,
    read_pillow_header,
)
from brainhops.io.images.stack import (  # noqa: E402
    StackImage,
    StackMember,
    StackMetadata,
    StackRaw,
)
from brainhops.io.images.stack import _data as stack_data  # noqa: E402
from brainhops.io.images.stack._data import _StackProxy  # noqa: E402
from brainhops.io.images.tiff import TiffImage  # noqa: E402
from brainhops.io.images.tiff import _utils as tiff_backend  # noqa: E402
from brainhops.io.metadata import MetadataFormat  # noqa: E402

RNG = np.random.default_rng(0)
VOLUME = RNG.integers(0, 2**16, (12, 3, 2), dtype=np.uint16)  # z, y, x
COLOUR = RNG.integers(0, 256, (4, 3, 2, 3), dtype=np.uint8)  # z, y, x, c


def _tiffs(
    folder: Path,
    volume: np.ndarray = VOLUME,
    name: str = "z{}.tif",
    start: int = 0,
    **kw,
) -> Path:
    """Write one TIFF file per slice of a `(z, y, x[, c])` volume."""
    folder.mkdir(parents=True, exist_ok=True)
    for k, plane in enumerate(volume):
        tifffile.imwrite(folder / name.format(start + k), plane, **kw)
    return folder


def _pngs(
    folder: Path, volume: np.ndarray = COLOUR, name: str = "s_{:03d}.png"
) -> Path:
    """Write one PNG file per slice of a `(z, y, x[, c])` volume."""
    folder.mkdir(parents=True, exist_ok=True)
    for k, plane in enumerate(volume):
        Image.fromarray(plane).save(folder / name.format(k))
    return folder


def _scale(image: tx.Any) -> np.ndarray:
    return np.asarray(image.transformation.scale, dtype=float)


def _units(image: tx.Any) -> tx.List[tx.Optional[str]]:
    return [
        None if axis.unit is None else str(axis.unit)
        for axis in image.transformation.output.axes
    ]


def _count_reads(monkeypatch: pytest.MonkeyPatch) -> tx.List[str]:
    """Record the name of every file that the lazy array decodes."""
    reads: tx.List[str] = []
    original = stack_data._read_file

    def counting(filename: str, member: StackMember) -> np.ndarray:
        reads.append(member.name)
        return original(filename, member)

    monkeypatch.setattr(stack_data, "_read_file", counting)
    return reads


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_a_folder_of_tiff_files_is_one_volume(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    image = io.load(folder, hint="stack")
    assert isinstance(image, StackImage)
    assert isinstance(image.rawdata, _StackProxy)
    assert isinstance(image.metadata, StackMetadata)
    assert image.shape == (2, 3, 12)
    assert image.rawdata.shape == (12, 3, 2)
    assert np.array_equal(np.asarray(image.data), VOLUME.transpose(2, 1, 0))
    # The class and the loaders, with the hint, read it in the same way.
    for other in (
        StackImage.load(folder),
        io.load(str(folder), hint="stack"),
        io.images.load(folder, hint="stack"),
    ):
        assert isinstance(other, StackImage)
        assert np.array_equal(np.asarray(other.data), np.asarray(image.data))


def test_numbers_are_sorted_as_integers(tmp_path: Path) -> None:
    # Unpadded numbers sort as 0, 1, 2, ..., 10, 11, not as 0, 1, 10, 11, 2.
    folder = _tiffs(tmp_path / "slices", name="z{}.tif")
    image = StackImage.load(folder)
    names = image.metadata.raw.names
    assert names[:3] == ("z0.tif", "z1.tif", "z2.tif")
    assert names[-2:] == ("z10.tif", "z11.tif")
    assert [m.number for m in image.metadata.raw.members] == list(range(12))
    assert np.array_equal(np.asarray(image.rawdata), VOLUME)


def test_numbering_may_start_anywhere(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices", name="img{:04d}.tif", start=7)
    image = StackImage.load(folder)
    assert image.metadata.raw.members[0].number == 7
    assert np.array_equal(np.asarray(image.rawdata), VOLUME)


def test_a_folder_of_png_files_with_channels(tmp_path: Path) -> None:
    folder = _pngs(tmp_path / "rgb")
    image = io.load(folder, hint="stack")
    assert isinstance(image, StackImage)
    assert image.shape == (2, 3, 4, 3)
    assert image.metadata.raw.axes == "ZYXS"
    data = np.asarray(image.data)
    assert np.array_equal(data, COLOUR.transpose(2, 1, 0, 3))
    names = [axis.name for axis in image.transformation.input.axes]
    assert names == ["x", "y", "z", "c"]
    member = image.metadata.raw.members[0]
    assert member.format == "pillow"
    assert isinstance(member.header, PillowHeader)
    assert member.header.mode == "RGB"


def test_planar_tiff_channels_come_last(tmp_path: Path) -> None:
    folder = tmp_path / "planar"
    folder.mkdir()
    for k, plane in enumerate(COLOUR):
        tifffile.imwrite(
            folder / f"p{k}.tif",
            np.ascontiguousarray(plane.transpose(2, 0, 1)),
            photometric="rgb",
            planarconfig="separate",
        )
    image = StackImage.load(folder)
    assert image.metadata.raw.members[0].axes == "SYX"
    assert np.array_equal(np.asarray(image.rawdata), COLOUR)
    assert np.array_equal(np.asarray(image.data), COLOUR.transpose(2, 1, 0, 3))


def test_a_pattern_selects_the_files(tmp_path: Path) -> None:
    folder = tmp_path / "mixed"
    _tiffs(folder, VOLUME[:5], name="ch1_z{}.tif")
    _tiffs(folder, VOLUME[5:9], name="ch2_z{}.tif")
    (folder / "notes.txt").write_text("not an image")
    (folder / ".hidden.tif").write_bytes(b"")
    # Without a pattern, the two channels are two numbers that vary.
    with pytest.raises(ParserContentError, match="2 numbers"):
        StackImage.load(folder)
    image = io.load(folder, hint="stack", pattern="ch2_*.tif")
    assert isinstance(image, StackImage)
    assert image.metadata.raw.pattern == "ch2_*.tif"
    assert np.array_equal(np.asarray(image.rawdata), VOLUME[5:9])
    metadata = MetadataFormat.load(folder, hint="stack", pattern="ch1_*.tif")
    assert metadata.raw.names == tuple(f"ch1_z{k}.tif" for k in range(5))


def test_a_pattern_may_name_a_subfolder(tmp_path: Path) -> None:
    _pngs(tmp_path / "data" / "rgb")
    image = StackImage.load(tmp_path / "data", pattern="rgb/*.png")
    assert image.metadata.raw.names[0] == "rgb/s_000.png"
    assert np.array_equal(np.asarray(image.rawdata), COLOUR)


def test_a_pattern_selects_image_files_only(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    (folder / "z.txt").write_text("text")
    with pytest.raises(ParserContentError, match="'z.txt'"):
        StackImage.load(folder, pattern="z*")
    with pytest.raises(ParserContentError, match="no image file"):
        StackImage.load(folder, pattern="*.png")


def test_hidden_files_are_never_taken(tmp_path: Path) -> None:
    # A writer that replaces a file at once writes a hidden file first.
    folder = _tiffs(tmp_path / "slices")
    tifffile.imwrite(folder / ".tmp.z12.tif", VOLUME[0])
    for pattern in (None, "*.tif", "*"):
        image = StackImage.load(folder, pattern=pattern)
        assert len(image.metadata.raw.members) == 12


def test_a_pattern_stays_in_the_folder(tmp_path: Path) -> None:
    _tiffs(tmp_path / "a")
    _tiffs(tmp_path / "b")
    with pytest.raises(ParserContentError, match="inside the folder"):
        StackImage.load(tmp_path / "a", pattern="../b/*.tif")
    with pytest.raises(ParserContentError, match="inside the folder"):
        StackImage.load(tmp_path / "a", pattern=str(tmp_path / "b/*"))


# ----------------------------------------------------------------------
#   REFUSED FOLDERS
# ----------------------------------------------------------------------


def test_a_repeated_number_is_refused(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices", VOLUME[:3], name="z{}.tif")
    tifffile.imwrite(folder / "z01.tif", VOLUME[0])
    with pytest.raises(ParserContentError, match="same number"):
        StackImage.load(folder)


def test_names_that_follow_no_pattern_are_refused(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices", VOLUME[:3])
    tifffile.imwrite(folder / "other.tif", VOLUME[0])
    with pytest.raises(ParserContentError, match="numbering pattern"):
        StackImage.load(folder)


@pytest.mark.parametrize(
    "plane, what",
    [
        (VOLUME[0, :2], "shape"),
        (VOLUME[0].astype("float32"), "type"),
        (np.stack([VOLUME[0]] * 3, axis=-1).astype("uint8"), "shape"),
    ],
)
def test_files_that_differ_are_refused(
    tmp_path: Path, plane: np.ndarray, what: str
) -> None:
    folder = _tiffs(tmp_path / "slices", VOLUME[:3])
    tifffile.imwrite(folder / "z3.tif", plane)
    with pytest.raises(ParserContentError, match=f"same {what}.*'z3.tif'"):
        StackImage.load(folder)


def test_files_with_and_without_channels_are_refused(tmp_path: Path) -> None:
    folder = _pngs(tmp_path / "pngs", COLOUR[:2])
    grey = Image.fromarray(COLOUR[2, ..., 0])
    grey.save(folder / "s_002.png")
    with pytest.raises(ParserContentError, match="channels"):
        StackImage.load(folder)


def test_a_stack_in_one_file_is_refused(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices", VOLUME[:2])
    tifffile.imwrite(folder / "z2.tif", VOLUME[:5])
    with pytest.raises(ParserContentError, match="one plane"):
        StackImage.load(folder)


def test_an_animation_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "gifs"
    folder.mkdir()
    frames = [Image.fromarray(COLOUR[k]) for k in range(2)]
    frames[0].save(folder / "a0.gif", save_all=True, append_images=frames)
    frames[1].save(folder / "a1.gif")
    with pytest.raises(ParserContentError, match="frames"):
        StackImage.load(folder)


def test_paths_that_are_not_folders_are_refused(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    with pytest.raises(ParserExistsError):
        StackImage.load(tmp_path / "missing")
    with pytest.raises(ParserContentError, match="is a file"):
        StackImage.load(folder / "z0.tif")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ParserContentError, match="no image file"):
        StackImage.load(tmp_path / "empty")


def test_streams_and_bytes_are_refused(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    content = (folder / "z0.tif").read_bytes()
    for cls in (StackImage, StackMetadata):
        assert cls.sniff(content) == Confidence.NO
        assert cls.sniff(BytesIO(content)) == Confidence.NO
        with pytest.raises(ParserError, match="folder"):
            cls.load(content)
        with pytest.raises(ParserError, match="folder"):
            cls.load(BytesIO(content))


def test_unknown_and_invalid_options_are_refused(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    with pytest.raises(TypeError, match="frame"):
        StackImage.load(folder, frame=1)
    # The fill value concerns the pixels, which metadata does not read.
    with pytest.raises(TypeError, match="fill"):
        StackMetadata.load(folder, fill=1)
    for options, match in (
        ({"pixel_size": (1.0, 2.0, 3.0)}, "pixel_size"),
        ({"pixel_size": -1.0}, "pixel_size"),
        ({"spacing": 0}, "spacing"),
        ({"spacing": "far"}, "spacing"),
        ({"unit": "s"}, "unit"),
        ({"axis": "t"}, "stacking axis"),
        ({"axis": 3}, "stacking axis"),
        ({"fill": np.nan}, "fill"),
        ({"fill": -1}, "fill"),
    ):
        with pytest.raises(ValueError, match=match):
            StackImage.load(folder, **options)


# ----------------------------------------------------------------------
#   SNIFFING
# ----------------------------------------------------------------------


def test_a_stack_is_read_only_with_its_hint(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    _pngs(tmp_path / "pngs")
    for path in (folder, tmp_path / "pngs"):
        assert io.sniff(path) is None
        assert io.images.sniff(path) is None
        assert MetadataFormat.sniff(path) is None
        assert StackImage.sniff(path) == Confidence.NO
        assert StackMetadata.sniff(path) == Confidence.NO
        for load in (io.load, io.images.load, MetadataFormat.load):
            with pytest.raises(ParserError):
                load(path)
        assert isinstance(io.load(path, hint="stack"), StackImage)
        assert isinstance(
            MetadataFormat.load(path, hint="stack"), StackMetadata
        )
        # Sniffing with the hint names the format that the hint loads.
        assert io.sniff(path, hint="stack") is StackImage
        assert io.images.sniff(path, hint="stack") is StackImage
        assert MetadataFormat.sniff(path, hint="stack") is StackMetadata
    # The error of the sniffer tells how to read the stack.
    for cls in (StackImage, StackMetadata):
        with pytest.raises(SnifferContentError, match='hint="stack"'):
            cls.sniff(folder, error=True)


def test_files_are_left_to_their_readers(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    _pngs(tmp_path / "pngs")
    assert StackImage.sniff(folder / "z0.tif") == Confidence.NO
    assert io.sniff(folder / "z0.tif") is TiffImage
    assert isinstance(io.load(folder / "z0.tif"), TiffImage)
    assert isinstance(io.load(tmp_path / "pngs" / "s_000.png"), PillowImage)
    with pytest.raises(ParserError):
        io.load(folder / "z0.tif", hint="stack")


def test_zarr_stores_are_not_claimed(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "store.zarr")
    (folder / "zarr.json").write_text("{}")
    for cls in (StackImage, StackMetadata):
        with pytest.raises(SnifferContentError, match="Zarr store"):
            cls.sniff(folder, error=True)


def test_a_folder_without_numbered_images_is_not_claimed(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "photos"
    folder.mkdir()
    Image.fromarray(COLOUR[0]).save(folder / "cat.png")
    Image.fromarray(COLOUR[1]).save(folder / "dog.png")
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "a1.txt").write_text("1")
    for path in (folder, tmp_path / "text"):
        with pytest.raises(SnifferContentError) as info:
            StackImage.sniff(path, error=True)
        assert "hint" not in str(info.value)
        with pytest.raises(ParserError):
            io.load(path, hint="stack")


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def test_the_pixel_size_comes_from_the_tags(tmp_path: Path) -> None:
    # 2 pixels per centimetre along x and 4 along y.
    folder = _tiffs(
        tmp_path / "slices", resolution=(2.0, 4.0), resolutionunit="CENTIMETER"
    )
    image = StackImage.load(folder)
    record = image.metadata.raw
    assert (record.pixel_size, record.spacing, record.unit) == (
        (0.5, 0.25),
        None,
        "cm",
    )
    assert isinstance(image.transformation, Scaling)
    assert np.allclose(_scale(image), [0.5, 0.25, 1.0])
    # The spacing of the slices is not recorded, so it stays unknown.
    assert _units(image) == ["centimeter", "centimeter", None]
    assert image.transformation.input.name == "voxel"


def test_the_options_set_the_fields_of_the_record(tmp_path: Path) -> None:
    folder = _tiffs(
        tmp_path / "slices", resolution=(2.0, 4.0), resolutionunit="CENTIMETER"
    )
    image = StackImage.load(folder, spacing=2.0)
    assert image.metadata.raw.spacing == 2.0
    assert np.allclose(_scale(image), [0.5, 0.25, 2.0])
    assert _units(image) == ["centimeter"] * 3
    # A unit converts the sizes that the files record.
    image = StackImage.load(folder, spacing=2.0, unit="mm")
    record = image.metadata.raw
    assert np.allclose(record.pixel_size, (5.0, 2.5))
    assert record.unit == "mm"
    assert np.allclose(_scale(image), [5.0, 2.5, 2.0])
    assert _units(image) == ["millimeter"] * 3
    image = StackImage.load(folder, pixel_size=(1.0, 2.0), spacing=-3.0)
    assert np.allclose(_scale(image), [1.0, 2.0, -3.0])
    # The transformations decoded again from the record use the fields.
    image.transformations = None
    assert np.allclose(_scale(image), [1.0, 2.0, -3.0])
    # The metadata takes the same options.
    metadata = StackMetadata.load(folder, pixel_size=0.1, unit="um")
    assert metadata.raw.pixel_size == (0.1, 0.1)
    assert metadata.raw.unit == "um"


def test_the_spacing_comes_from_imagej(tmp_path: Path) -> None:
    folder = _tiffs(
        tmp_path / "slices",
        imagej=True,
        resolution=(2.0, 2.0),
        metadata={"spacing": 1.5, "unit": "um"},
    )
    image = StackImage.load(folder)
    assert image.metadata.raw.spacing == 1.5
    assert np.allclose(_scale(image), [0.5, 0.5, 1.5])
    assert _units(image) == ["micrometer"] * 3
    # A negative spacing flips z, unless the option sets the spacing.
    flipped = _tiffs(
        tmp_path / "flipped",
        imagej=True,
        resolution=(2.0, 2.0),
        metadata={"spacing": -1.5, "unit": "um"},
    )
    assert np.allclose(_scale(StackImage.load(flipped)), [0.5, 0.5, -1.5])
    image = StackImage.load(flipped, spacing=3.0)
    assert np.allclose(_scale(image), [0.5, 0.5, 3.0])


def test_the_spacing_comes_from_ome(tmp_path: Path) -> None:
    # The TIFF writer records the spacing of a volume of one slice as the
    # PhysicalSizeZ of its OME-XML, and tifffile reads it as a plane.
    folder = tmp_path / "slices"
    folder.mkdir()
    for k, plane in enumerate(VOLUME[:3]):
        tifffile.imwrite(
            folder / f"z{k}.tif",
            plane,
            ome=True,
            metadata={
                "axes": "YX",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeY": 0.5,
                "PhysicalSizeZ": 4.0,
            },
        )
    record = StackMetadata.load(folder).raw
    assert (record.pixel_size, record.spacing, record.unit) == (
        (0.5, 0.5),
        4.0,
        "um",
    )


def test_slices_that_run_in_both_directions_disagree(
    tmp_path: Path,
) -> None:
    folder = _tiffs(
        tmp_path / "slices",
        VOLUME[:3],
        imagej=True,
        metadata={"spacing": 1.5, "unit": "um"},
    )
    tifffile.imwrite(
        folder / "z3.tif",
        VOLUME[3],
        imagej=True,
        metadata={"spacing": -1.5, "unit": "um"},
    )
    with pytest.warns(UserWarning, match="spacing"):
        image = StackImage.load(folder)
    assert image.metadata.raw.spacing is None
    assert np.allclose(_scale(image)[2], 1.0)


def test_sizes_on_which_the_files_disagree_are_unknown(
    tmp_path: Path,
) -> None:
    folder = _tiffs(
        tmp_path / "slices",
        VOLUME[:3],
        resolution=(2.0, 4.0),
        resolutionunit="CENTIMETER",
    )
    tifffile.imwrite(
        folder / "z3.tif",
        VOLUME[3],
        resolution=(2.0, 5.0),
        resolutionunit="CENTIMETER",
    )
    with pytest.warns(UserWarning, match="pixel size"):
        image = StackImage.load(folder)
    assert image.metadata.raw.pixel_size is None
    assert np.allclose(_scale(image), [1.0, 1.0, 1.0])
    assert _units(image) == [None, None, None]


def test_pillow_files_have_no_size(tmp_path: Path) -> None:
    folder = tmp_path / "pngs"
    folder.mkdir()
    for k, plane in enumerate(COLOUR):
        Image.fromarray(plane).save(folder / f"s{k}.png", dpi=(300, 300))
    image = StackImage.load(folder)
    # The resolution of a raster file describes a screen or a printer, so
    # the Pillow reader ignores it by default, and so does the stack.
    assert np.allclose(_scale(image), [1.0, 1.0, 1.0, 1.0])
    assert _units(image) == [None, None, None, None]
    image = StackImage.load(folder, pixel_size=0.1, spacing=0.2, unit="mm")
    assert np.allclose(_scale(image)[:3], [0.1, 0.1, 0.2])


# ----------------------------------------------------------------------
#   STACKING AXIS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis, codes, order",
    [
        ("z", "ZYX", (2, 1, 0)),
        (2, "ZYX", (2, 1, 0)),
        ("y", "YZX", (2, 0, 1)),
        (1, "YZX", (2, 0, 1)),
        ("x", "XZY", (0, 2, 1)),
        (0, "XZY", (0, 2, 1)),
    ],
)
def test_the_stacking_axis_places_the_files(
    tmp_path: Path, axis: tx.Any, codes: str, order: tx.Tuple[int, ...]
) -> None:
    # VOLUME holds 12 files of 3 rows and 2 columns.
    folder = _tiffs(
        tmp_path / "slices", resolution=(2.0, 4.0), resolutionunit="CENTIMETER"
    )
    image = StackImage.load(folder, axis=axis, spacing=3.0, unit="cm")
    assert image.metadata.raw.axes == codes
    assert image.shape == tuple(VOLUME.shape[i] for i in order)
    assert np.array_equal(np.asarray(image.data), VOLUME.transpose(order))
    # The columns of a file lie along the first of the two other axes and
    # its rows along the second, and the spacing scales the stacking axis.
    name = "xyz"[axis] if isinstance(axis, int) else axis
    expected = {"z": [0.5, 0.25, 3.0], "y": [0.5, 3.0, 0.25]}
    expected["x"] = [3.0, 0.5, 0.25]
    assert np.allclose(_scale(image), expected[name])


def test_the_stacking_axis_keeps_channels_last(tmp_path: Path) -> None:
    folder = _pngs(tmp_path / "rgb")
    image = StackImage.load(folder, axis="y")
    assert image.metadata.raw.axes == "YZXS"
    assert np.array_equal(np.asarray(image.data), COLOUR.transpose(2, 0, 1, 3))


# ----------------------------------------------------------------------
#   MISSING SLICES
# ----------------------------------------------------------------------


def test_a_missing_number_is_a_missing_slice(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    os.remove(folder / "z4.tif")
    os.remove(folder / "z7.tif")
    image = StackImage.load(folder)
    record = image.metadata.raw
    assert record.missing == (4, 7)
    assert record.numbers == tuple(range(12))
    assert len(record.members) == 10
    assert record.shape == (12, 3, 2)
    expected = VOLUME.copy()
    expected[[4, 7]] = 0
    assert np.array_equal(np.asarray(image.rawdata), expected)
    # A floating type is filled with NaN, and `fill` sets another value.
    floats = _tiffs(tmp_path / "floats", VOLUME[:3].astype("float32"))
    os.remove(floats / "z1.tif")
    assert np.isnan(np.asarray(StackImage.load(floats).rawdata)[1]).all()
    filled = StackImage.load(floats, fill=-1.5)
    assert (np.asarray(filled.rawdata)[1] == -1.5).all()
    assert (np.asarray(StackImage.load(folder, fill=9).rawdata)[4] == 9).all()


def test_a_missing_slice_reads_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _tiffs(tmp_path / "slices")
    os.remove(folder / "z4.tif")
    reads = _count_reads(monkeypatch)
    image = StackImage.load(folder)
    assert (np.asarray(image.rawdata[4]) == 0).all()
    assert reads == []
    with backend("dask"):
        data = StackImage.load(folder).to_backend()
    assert np.array_equal(data[:, :, 3:6].compute(), VOLUME[3:6].T * [1, 0, 1])
    # Dask may read the chunks in any order.
    assert sorted(reads) == ["z3.tif", "z5.tif"]


# ----------------------------------------------------------------------
#   LAZINESS
# ----------------------------------------------------------------------


def test_loading_reads_no_pixel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _tiffs(tmp_path / "slices")
    reads = _count_reads(monkeypatch)
    image = StackImage.load(folder)
    image.shape, image.transformations, repr(image)
    assert reads == []
    # An empty selection, which Dask reads to learn the type, reads nothing.
    assert image.rawdata[0:0].shape == (0, 3, 2)
    assert reads == []


def test_a_selection_reads_only_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _tiffs(tmp_path / "slices")
    reads = _count_reads(monkeypatch)
    image = StackImage.load(folder)
    assert np.array_equal(image.rawdata[5], VOLUME[5])
    assert np.array_equal(image.rawdata[2:8:3, 1], VOLUME[2:8:3, 1])
    assert np.array_equal(image.rawdata[-1, :, 0], VOLUME[-1, :, 0])
    assert reads == ["z5.tif", "z2.tif", "z5.tif", "z11.tif"]


def test_dask_reads_one_file_per_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _tiffs(tmp_path / "slices")
    reads = _count_reads(monkeypatch)
    with backend("dask"):
        data = StackImage.load(folder).to_backend()
    assert isinstance(data, da.Array)
    assert data.chunks == ((2,), (3,), (1,) * 12)
    assert reads == []
    assert np.array_equal(data[:, :, 4].compute(), VOLUME[4].T)
    assert reads == ["z4.tif"]
    assert np.array_equal(data.compute(), VOLUME.transpose(2, 1, 0))


def test_numpy_data_are_read_only(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    image = StackImage.load(folder)
    with pytest.raises(TypeError):
        image.data[(0,) * image.ndim] = 1
    with backend("numpy"):
        data = image.to_backend()
    assert isinstance(data, np.ndarray)


def test_a_file_changed_after_loading_is_refused(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    image = StackImage.load(folder)
    tifffile.imwrite(folder / "z3.tif", VOLUME[3, :2])
    with pytest.raises(ParserContentError, match="'z3.tif'"):
        np.asarray(image.rawdata)
    os.remove(folder / "z3.tif")
    with pytest.raises(ParserContentError, match="no longer exists"):
        np.asarray(image.rawdata[3])


# ----------------------------------------------------------------------
#   METADATA
# ----------------------------------------------------------------------


def test_the_record_holds_the_headers_of_the_files(tmp_path: Path) -> None:
    folder = _tiffs(
        tmp_path / "slices",
        VOLUME[:3],
        description="slice",
        resolution=(2.0, 4.0),
        resolutionunit="CENTIMETER",
    )
    metadata = StackMetadata.load(folder)
    record = metadata.raw
    assert isinstance(record, StackRaw)
    assert record.shape == (3, 3, 2)
    assert record.dtype == np.dtype("uint16")
    assert record.format == "tiff"
    member = record.members[1]
    assert (member.name, member.number, member.axes) == ("z1.tif", 1, "YX")
    assert isinstance(member.header, tiff_backend.SeriesMetadata)
    assert member.header.scales["x"] == (0.5, "cm")
    assert record.copy() == record
    clone = pickle.loads(pickle.dumps(metadata))
    assert clone.raw == record


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _bytes(folder: Path) -> tx.Dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_an_untouched_stack_is_copied_file_by_file(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices", description="slice")
    os.remove(folder / "z4.tif")
    image = StackImage.load(folder)
    image.save(tmp_path / "copy")
    # The missing slice is not written, and every other file is a copy.
    assert _bytes(tmp_path / "copy") == _bytes(folder)
    # Numbered names in the format of the files are copies too.
    image.save(tmp_path / "named" / "out.tiff")
    copied = _bytes(tmp_path / "named")
    assert sorted(copied) == [f"out_{n:02d}.tiff" for n in range(12) if n != 4]
    assert copied["out_05.tiff"] == (folder / "z5.tif").read_bytes()


def test_numbered_names_come_from_the_path(tmp_path: Path) -> None:
    image = StackImage(data=VOLUME[:3].transpose(2, 1, 0))
    image.save(tmp_path / "out.tif")
    image.save(tmp_path / "pngs" / "s.png")
    names = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert names == ["out_0.tif", "out_1.tif", "out_2.tif"]
    assert sorted(os.listdir(tmp_path / "pngs")) == [
        "s_0.png",
        "s_1.png",
        "s_2.png",
    ]
    back = StackImage.load(tmp_path / "pngs")
    assert back.metadata.raw.format == "pillow"
    assert np.array_equal(np.asarray(back.rawdata), VOLUME[:3])
    back = StackImage.load(tmp_path, pattern="out_*.tif")
    assert np.array_equal(np.asarray(back.rawdata), VOLUME[:3])


def test_a_new_stack_is_a_folder_of_tiff_files(tmp_path: Path) -> None:
    data = np.arange(2 * 3 * 11, dtype="float64").reshape(2, 3, 11)
    StackImage(data=data).save(tmp_path / "volume")
    names = sorted(os.listdir(tmp_path / "volume"))
    assert names[:2] == ["slice_00.tif", "slice_01.tif"]
    assert names[-1] == "slice_10.tif"
    back = StackImage.load(tmp_path / "volume")
    assert np.array_equal(np.asarray(back.data), data)
    # Another axis cuts the volume into other files.
    StackImage(data=data).save(tmp_path / "coronal", axis="y")
    back = StackImage.load(tmp_path / "coronal", axis="y")
    assert len(back.metadata.raw.members) == 3
    assert np.array_equal(np.asarray(back.data), data)


def test_the_geometry_is_written_to_the_files(tmp_path: Path) -> None:
    image = StackImage(data=VOLUME.transpose(2, 1, 0))
    axes = raster.default_axes(3)
    image.transformations = raster.raster_transformations(
        axes, {"x": (0.5, "mm"), "y": (0.25, "mm"), "z": (2.0, "mm")}
    )
    consolidated = image.consolidate()
    record = consolidated.metadata.raw
    assert (record.pixel_size, record.spacing, record.unit) == (
        (0.5, 0.25),
        2.0,
        "millimeter",
    )
    image.save(tmp_path / "slices")
    back = StackImage.load(tmp_path / "slices")
    assert np.allclose(back.metadata.raw.pixel_size, (0.5, 0.25))
    assert np.isclose(back.metadata.raw.spacing, 2.0)
    assert np.allclose(_scale(back), [0.5, 0.25, 2.0])
    assert _units(back) == ["millimeter"] * 3
    # A plane with channels records no spacing, and Pillow records nothing.
    colour = StackImage(data=COLOUR.transpose(2, 1, 0, 3))
    colour.transformations = raster.raster_transformations(
        raster.default_axes(4, channel=True),
        {"x": (0.5, "mm"), "y": (0.25, "mm"), "z": (2.0, "mm")},
    )
    record = colour.consolidate().metadata.raw
    assert (record.pixel_size, record.spacing) == ((0.5, 0.25), None)
    colour.save(tmp_path / "rgb")
    back = StackImage.load(tmp_path / "rgb")
    assert back.metadata.raw.members[0].axes == "YXS"
    assert np.array_equal(np.asarray(back.rawdata), COLOUR)
    assert np.allclose(_scale(back)[:3], [0.5, 0.25, 1.0])
    colour.save(tmp_path / "png" / "c.png")
    record = StackMetadata.load(tmp_path / "png").raw
    assert (record.pixel_size, record.spacing) == (None, None)


def test_assigned_data_fills_every_slice(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices", name="z{:02d}.tif")
    os.remove(folder / "z04.tif")
    image = StackImage.load(folder)
    image.data = np.asarray(image.data) + 1
    record = image.consolidate().metadata.raw
    # The data of the model takes precedence over the record: the missing
    # slice gets a file, named after the others, and the files keep their
    # names.
    assert record.missing == ()
    assert record.names == tuple(f"z{n:02d}.tif" for n in range(12))
    assert all(member.header is None for member in record.members)
    image.save(tmp_path / "copy")
    back = StackImage.load(tmp_path / "copy")
    expected = VOLUME + 1
    expected[4] = 1
    assert np.array_equal(np.asarray(back.rawdata), expected)


def test_saving_leaves_the_image_unchanged(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    image = StackImage.load(folder, spacing=2.0)
    raw, metadata = image.rawdata, image.metadata
    image.data = np.asarray(image.data) * 2
    pending = image._data
    image.save(tmp_path / "copy")
    assert image.rawdata is raw
    assert image.metadata is metadata
    assert image._data is pending


def test_saving_over_its_own_folder_reads_the_new_files(
    tmp_path: Path,
) -> None:
    folder = _tiffs(tmp_path / "slices")
    image = StackImage.load(folder)
    image.data = np.asarray(image.data) + 1
    image.save(folder)
    assert isinstance(image.rawdata, _StackProxy)
    assert image._data is None
    assert image.metadata.raw == StackMetadata.load(folder).raw
    assert np.array_equal(np.asarray(image.rawdata), VOLUME + 1)
    # Numbered names that the lazy array reads are replaced in the same
    # way, and an untouched image keeps the fields of its record.
    named = _tiffs(tmp_path / "named", name="z_{:02d}.tif")
    image = StackImage.load(named, spacing=5.0)
    image.save(named / "z.tif")
    assert image.metadata.raw.spacing == 5.0
    assert image.metadata.raw.pattern == "z_*.tif"
    assert np.array_equal(np.asarray(image.rawdata), VOLUME)
    assert sorted(os.listdir(named)) == [f"z_{n:02d}.tif" for n in range(12)]


def test_only_a_folder_of_images_is_replaced(tmp_path: Path) -> None:
    image = StackImage(data=VOLUME[:2].transpose(2, 1, 0))
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep me")
    with pytest.raises(WriterError, match="notes.txt"):
        image.save(folder)
    assert (folder / "notes.txt").read_text() == "keep me"
    (tmp_path / "file").write_text("")
    with pytest.raises(WriterError, match="is a file"):
        image.save(tmp_path / "file")
    # A folder of images is replaced, and nothing is left beside it.
    old = _tiffs(tmp_path / "old", VOLUME[:5])
    image.save(old)
    assert sorted(os.listdir(old)) == ["slice_0.tif", "slice_1.tif"]
    assert sorted(os.listdir(tmp_path)) == ["data", "file", "old"]
    # A folder from which a pattern read one channel keeps the other.
    mixed = _tiffs(tmp_path / "mixed", VOLUME[:2], name="ch1_z{}.tif")
    _tiffs(mixed, VOLUME[2:4], name="ch2_z{}.tif")
    image = StackImage.load(mixed, pattern="ch1_*.tif")
    with pytest.raises(WriterError, match="ch2_z0.tif"):
        image.save(mixed)
    assert len(os.listdir(mixed)) == 4


def test_streams_bytes_and_bad_options_are_refused(tmp_path: Path) -> None:
    image = StackImage(data=VOLUME[:2].transpose(2, 1, 0))
    with pytest.raises(WriterError, match="folder"):
        image.save(BytesIO())
    with pytest.raises(WriterError, match="folder"):
        image.to_bytes()
    with pytest.raises(TypeError, match="compression"):
        image.save(tmp_path / "copy", compression="zlib")
    with pytest.raises(WriterError, match="stacking axis"):
        image.save(tmp_path / "copy", axis="t")
    with pytest.raises(WriterError, match="axes"):
        StackImage(data=np.zeros((2, 3))).save(tmp_path / "flat")
    with pytest.raises(WriterError, match="no data"):
        StackImage().save(tmp_path / "empty")
    assert not (tmp_path / "copy").exists()


def test_a_stack_is_saved_to_other_formats(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    image = StackImage.load(folder)
    # A stack is saved to another format by its extension, here a
    # multi-page TIFF.
    io.save(image, tmp_path / "volume.tif")
    back = io.load(tmp_path / "volume.tif")
    assert isinstance(back, TiffImage)
    assert np.array_equal(np.asarray(back.data), np.asarray(image.data))
    # Any image is saved as a stack with the format itself.
    StackImage.from_instance(back).save(tmp_path / "again")
    again = StackImage.load(tmp_path / "again")
    assert np.array_equal(np.asarray(again.data), np.asarray(image.data))


# ----------------------------------------------------------------------
#   BACKENDS OF THE SINGLE-FILE READERS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, suffix, info",
    [
        ("1", ".png", {}),
        ("L", ".png", {}),
        ("LA", ".png", {}),
        ("P", ".png", {}),
        ("P", ".png", {"transparency": 0}),
        ("RGB", ".png", {}),
        ("RGBA", ".png", {}),
        ("I;16", ".png", {}),
        ("I", ".tif", {}),
        ("F", ".tif", {}),
        ("CMYK", ".jpg", {}),
    ],
)
def test_the_pillow_header_gives_the_decoded_layout(
    tmp_path: Path, mode: str, suffix: str, info: tx.Dict[str, tx.Any]
) -> None:
    path = tmp_path / ("image" + suffix)
    Image.new(mode, (5, 3)).save(path, **info)
    for palette in (True, False):
        header = read_pillow_header(path, palette=palette)
        array = read_pillow(path, palette=palette).array
        assert header.shape == array.shape, (mode, palette)
        assert np.dtype(header.dtype) == array.dtype, (mode, palette)
        assert header.axes == ("YXS" if array.ndim == 3 else "YX")


def test_imagej_scales_without_a_page() -> None:
    scales, signs = tiff_backend.imagej_scales(
        {"spacing": -2.0, "unit": "micron", "finterval": 0.5},
        None,
        ("x", "y", "z", "t"),
    )
    assert scales == {"z": (2.0, "um"), "t": (0.5, "s")}
    assert signs == {"z": -1.0}


def test_no_file_is_left_open(tmp_path: Path) -> None:
    folder = _tiffs(tmp_path / "slices")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        image = StackImage.load(folder)
        np.asarray(image.rawdata)
