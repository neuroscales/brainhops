"""Tests for `brainhops.io.save`.

The file name gives the candidates: the formats declaring the longest
extension it ends with, and the prefix they require. The object gives
the one that is used: a format it already is, a file-backed version of
its very data model that takes all of its fields, or, for a
transformation, a format it converts to exactly. Every writable format is
round-tripped through it; the conversions are tested in
`test_io_convert_formats`.
"""

import io as _io
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io as io
from brainhops._core.dependencies import HAS_NIBABEL, has_abczarr_driver
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.metadata._report import takes_error_policy
from brainhops.datamodel.systems import LPSmm, RASmm
from brainhops.datamodel.transformations import (
    Affine,
    CoordinateField,
    DisplacementField,
    Scaling,
)
from brainhops.errors import (
    ConversionError,
    UnrepresentableTransformationError,
)
from brainhops.io.base import FileFormat
from brainhops.io.base._base import (
    is_format_registry,
    register_format,
)
from brainhops.io.base.parsers import (
    AmbiguousFormatError,
    TextFileWriter,
    WriterError,
    WriterNotImplementedError,
)

needs_nibabel = pytest.mark.skipif(not HAS_NIBABEL, reason="needs nibabel")
needs_zarr = pytest.mark.skipif(
    not has_abczarr_driver(), reason="needs abczarr and a driver"
)

DATA = np.arange(24, dtype="float32").reshape(2, 3, 4)
MATRIX = np.array(
    [
        [0.0, -1.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 2.0, 5.0],
    ]
)


def _image() -> SingleScaleImage:
    return SingleScaleImage(data=DATA, transformations=[Affine(MATRIX)])


# ----------------------------------------------------------------------
#   FAKE FORMATS
# ----------------------------------------------------------------------


class Note(DataModelBase):
    """A data model with one field, for plain-text formats."""

    text: str = ""


class OtherNote(DataModelBase):
    """A data model unrelated to Note."""

    text: str = ""


def _note_format(name: str, model: type = Note, **attrs: tx.Any) -> type:
    """A text format for `model`, writing the text after a tag."""

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:  # noqa: ANN001
        yield f"{name}:{kwargs.get('suffix', '')}{self.text}"

    namespace = {"to_lines": to_lines}
    namespace.update(attrs)
    return type(name, (model, FileFormat, TextFileWriter), namespace)


@pytest.fixture
def formats() -> tx.Iterator[tx.Callable[..., type]]:
    """Register writable test formats and leave the real registry as found."""
    made: tx.List[type] = []

    def make(name: str, model: type = Note, **attrs: tx.Any) -> type:
        fmt = register_format(_note_format(name, model, **attrs))
        made.append(fmt)
        return fmt

    yield make
    for fmt in made:
        for base in fmt.__mro__:
            if is_format_registry(base):
                base._REGISTRY.discard(fmt)


def _written(path: Path) -> str:
    return path.read_text().rstrip("\n")


# ----------------------------------------------------------------------
#   CHOOSING THE FORMAT FROM THE NAME
# ----------------------------------------------------------------------


def test_the_format_declaring_the_extension_is_used(formats, tmp_path) -> None:  # noqa: ANN001
    formats("A", EXTENSIONS=(".a",))
    formats("B", EXTENSIONS=(".b",))
    io.save(Note(text="hi"), tmp_path / "x.b")
    assert _written(tmp_path / "x.b") == "B:hi"


def test_the_longest_extension_wins(formats, tmp_path) -> None:  # noqa: ANN001
    formats("Short", EXTENSIONS=(".txt",))
    formats("Long", EXTENSIONS=(".note.txt",))
    io.save(Note(text="hi"), tmp_path / "x.note.txt")
    assert _written(tmp_path / "x.note.txt") == "Long:hi"
    io.save(Note(text="hi"), tmp_path / "x.txt")
    assert _written(tmp_path / "x.txt") == "Short:hi"


@pytest.mark.parametrize("suffix", [".gz", ".bz2", ".xz"])
def test_any_format_is_compressed_by_its_suffix(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
    suffix: str,
) -> None:
    from brainhops._core.streams import decompress

    formats("A", EXTENSIONS=(".a",))
    formats("B", EXTENSIONS=(".b",))
    target = tmp_path / ("x.b" + suffix)
    io.save(Note(text="hi"), target)
    content = target.read_bytes()
    assert content != decompress(content)
    assert decompress(content).decode().rstrip("\n") == "B:hi"


def test_a_longer_extension_is_not_given_up_for_a_shorter_one(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # The format of the longer extension cannot hold the object, so nothing
    # is written, without falling back to the shorter extension.
    formats("Short", EXTENSIONS=(".txt",))
    formats("Long", model=OtherNote, EXTENSIONS=(".note.txt",))
    with pytest.raises(WriterError, match="Long"):
        io.save(Note(text="hi"), tmp_path / "x.note.txt")
    assert not (tmp_path / "x.note.txt").exists()


def test_a_required_prefix_is_more_specific(formats, tmp_path) -> None:  # noqa: ANN001
    formats("Plain", EXTENSIONS=(".n",))
    formats("Prefixed", EXTENSIONS=(".n",), PREFIXES=("p_",))
    io.save(Note(text="hi"), tmp_path / "p_x.n")
    assert _written(tmp_path / "p_x.n") == "Prefixed:hi"
    io.save(Note(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "Plain:hi"


def test_priority_breaks_a_tie(formats, tmp_path) -> None:  # noqa: ANN001
    formats("Low", EXTENSIONS=(".n",))
    formats("High", EXTENSIONS=(".n",), PRIORITY=1)
    io.save(Note(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "High:hi"


def test_a_tie_is_an_ambiguity(formats, tmp_path) -> None:  # noqa: ANN001
    formats("One", EXTENSIONS=(".n",))
    formats("Two", EXTENSIONS=(".n",))
    with pytest.raises(AmbiguousFormatError):
        io.save(Note(text="hi"), tmp_path / "x.n")
    assert not (tmp_path / "x.n").exists()


def test_an_ambiguity_tells_the_user_how_to_choose(formats, tmp_path) -> None:  # noqa: ANN001
    formats(
        "Plain",
        EXTENSIONS=(".n",),
        __doc__="""
        A note written as plain text.

        This paragraph is detail for the reference, not for the message.
        """,
    )
    formats("Bare", EXTENSIONS=(".n",))
    with pytest.raises(AmbiguousFormatError) as raised:
        io.save(Note(text="hi"), tmp_path / "x.n")
    message = str(raised.value)
    lines = message.splitlines()

    # The message names the file and the object, then lists one line per
    # candidate in order.
    assert "'x.n'" in lines[0] and "Note" in lines[0]
    assert "2 formats" in lines[0]
    # Each candidate is described by its own docstring, with the call that
    # writes the object in it.
    assert lines[1] == "  - Bare: `Bare.from_any(obj).save(path)`"
    assert lines[2] == (
        "  - Plain (A note written as plain text): "
        "`Plain.from_any(obj).save(path)`"
    )
    assert "from_any(obj).save(path)" in lines[-1]
    # A data model docstring does not describe the format, and tie-breaking
    # concerns maintainers.
    assert "data model with one field" not in message
    assert "Attributes" not in message
    assert "PRIORITY" not in message
    assert "detail for the reference" not in message


def test_a_tie_with_a_format_that_cannot_hold_it_is_not_one(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    formats("Mine", EXTENSIONS=(".n",))
    formats("Theirs", model=OtherNote, EXTENSIONS=(".n",))
    io.save(Note(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "Mine:hi"


def test_an_unknown_extension_is_refused(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(WriterError, match="no writable format"):
        io.save(Note(text="hi"), tmp_path / "x.unknown-extension")


def test_an_object_no_format_can_hold_is_refused(formats, tmp_path) -> None:  # noqa: ANN001
    formats("Theirs", model=OtherNote, EXTENSIONS=(".n",))
    with pytest.raises(WriterError, match="Theirs"):
        io.save(Note(text="hi"), tmp_path / "x.n")


def test_options_reach_the_writer(formats, tmp_path) -> None:  # noqa: ANN001
    formats("A", EXTENSIONS=(".a",))
    io.save(Note(text="hi"), tmp_path / "x.a", suffix="> ")
    assert _written(tmp_path / "x.a") == "A:> hi"


# ----------------------------------------------------------------------
#   CHOOSING THE FORMAT FROM A HINT
# ----------------------------------------------------------------------


def test_a_hint_restricts_the_candidates(formats, tmp_path) -> None:  # noqa: ANN001
    # Without the hint, the two formats would be an ambiguity.
    formats("One", EXTENSIONS=(".n",), HINTS=("one",))
    formats("Two", EXTENSIONS=(".n",), HINTS=("two",))
    io.save(Note(text="hi"), tmp_path / "x.n", hint="two")
    assert _written(tmp_path / "x.n") == "Two:hi"
    io.save(Note(text="hi"), tmp_path / "x.n", hint=["one"])
    assert _written(tmp_path / "x.n") == "One:hi"


def test_a_hint_does_not_override_a_longer_extension(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    formats("Short", EXTENSIONS=(".txt",), HINTS=("note",))
    formats("Long", EXTENSIONS=(".note.txt",), HINTS=("note",))
    io.save(Note(text="hi"), tmp_path / "x.note.txt", hint="note")
    assert _written(tmp_path / "x.note.txt") == "Long:hi"


def test_a_hinted_format_is_written_whatever_the_name(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # No format claims `.json`, but the hint names the format.
    formats("Mine", EXTENSIONS=(".mine.json",), HINTS=("mine",))
    with pytest.raises(WriterError, match="no writable format"):
        io.save(Note(text="hi"), tmp_path / "x.unclaimed-json")
    io.save(Note(text="hi"), tmp_path / "x.unclaimed-json", hint="MINE")
    assert _written(tmp_path / "x.unclaimed-json") == "Mine:hi"


def test_a_hint_no_format_answers_to_is_refused(formats, tmp_path) -> None:  # noqa: ANN001
    formats("A", EXTENSIONS=(".a",), HINTS=("a",))
    with pytest.raises(WriterError, match="'no-such-hint'"):
        io.save(Note(text="hi"), tmp_path / "x.a", hint="no-such-hint")
    assert not (tmp_path / "x.a").exists()


# ----------------------------------------------------------------------
#   WHAT THE OBJECT IS WRITTEN AS
# ----------------------------------------------------------------------


def test_an_object_of_the_format_is_written_as_it_is(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # Rebuilding would rerun the constructor and lose state kept outside the
    # fields.
    fmt = formats("A", EXTENSIONS=(".a",))
    obj = fmt(text="hi")
    obj.text = "changed"
    seen = []
    original = fmt.save

    def spy(self, file, **kwargs) -> None:  # noqa: ANN001
        seen.append(self)
        return original(self, file, **kwargs)

    fmt.save = spy
    io.save(obj, tmp_path / "x.a")
    assert seen == [obj]
    assert _written(tmp_path / "x.a") == "A:changed"


def test_the_object_own_format_is_not_an_ambiguity(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # Two formats claim .n equally, but an object already in one stays in it.
    one = formats("One", EXTENSIONS=(".n",))
    formats("Two", EXTENSIONS=(".n",))
    io.save(one(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "One:hi"


def test_a_subclass_of_the_data_model_is_not_squeezed_into_it(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # The Note format would drop what the richer subclass holds.
    class RichNote(Note):
        colour: str = "red"

    formats("A", EXTENSIONS=(".a",))
    with pytest.raises(WriterError):
        io.save(RichNote(text="hi"), tmp_path / "x.a")


def test_a_format_missing_a_field_of_the_data_model_cannot_hold_it(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # The format serves `text` itself, so building it from a note loses text.
    formats(
        "Derived",
        EXTENSIONS=(".n",),
        __annotations__={"text": tx.ClassVar[str]},
        text=property(lambda self: "derived"),
    )
    with pytest.raises(WriterError, match="does not take the field.*text"):
        io.save(Note(text="hi"), tmp_path / "x.n")


def test_an_unnamed_file_is_written_in_the_object_own_format(formats) -> None:  # noqa: ANN001
    fmt = formats("A", EXTENSIONS=(".a",))
    buffer = _io.StringIO()
    io.save(fmt(text="hi"), buffer)
    assert buffer.getvalue() == "A:hi\n"


def test_an_unnamed_file_cannot_choose_a_format(formats) -> None:  # noqa: ANN001
    formats("A", EXTENSIONS=(".a",))
    with pytest.raises(WriterError, match="no name"):
        io.save(Note(text="hi"), _io.StringIO())


# ----------------------------------------------------------------------
#   FALLING BACK WHEN A CANDIDATE REFUSES
# ----------------------------------------------------------------------

REFUSALS = [
    ConversionError("cannot convert"),
    UnrepresentableTransformationError("cannot encode"),
    WriterNotImplementedError("cannot write lines"),
    WriterError("cannot write"),
]


@pytest.mark.parametrize("error", REFUSALS, ids=lambda e: type(e).__name__)
def test_a_candidate_that_refuses_to_write_gives_way_to_the_next(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
    error: Exception,
) -> None:
    formats("Picky", EXTENSIONS=(".n",), PRIORITY=1, to_lines=_raising(error))
    formats("Plain", EXTENSIONS=(".n",))
    io.save(Note(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "Plain:hi"


@pytest.mark.parametrize("error", REFUSALS, ids=lambda e: type(e).__name__)
def test_a_candidate_that_refuses_to_build_gives_way_to_the_next(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
    error: Exception,
) -> None:
    formats(
        "Picky",
        EXTENSIONS=(".n",),
        PRIORITY=1,
        from_instance=_refusing_build(error),
    )
    formats("Plain", EXTENSIONS=(".n",))
    io.save(Note(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "Plain:hi"


def test_the_candidates_are_tried_in_order_of_priority(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    refusal = UnrepresentableTransformationError("cannot encode")
    formats(
        "First", EXTENSIONS=(".n",), PRIORITY=3, to_lines=_raising(refusal)
    )
    formats("Second", EXTENSIONS=(".n",), PRIORITY=2)
    formats("Third", EXTENSIONS=(".n",), PRIORITY=1)
    io.save(Note(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "Second:hi"


def test_when_every_candidate_refuses_each_reason_is_given(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    formats(
        "Picky",
        EXTENSIONS=(".n",),
        PRIORITY=1,
        to_lines=_raising(UnrepresentableTransformationError("no geometry")),
    )
    formats(
        "Fussy",
        EXTENSIONS=(".n",),
        from_instance=_refusing_build(ConversionError("wrong systems")),
    )
    with pytest.raises(WriterError) as raised:
        io.save(Note(text="hi"), tmp_path / "x.n")
    lines = str(raised.value).splitlines()
    assert "Fussy, Picky" in lines[0]
    assert lines[1:] == [
        "  - Picky: UnrepresentableTransformationError: no geometry",
        "  - Fussy: ConversionError: wrong systems",
    ]


def test_a_refused_candidate_leaves_no_file(formats, tmp_path) -> None:  # noqa: ANN001
    refusal = UnrepresentableTransformationError("cannot encode")
    formats("Picky", EXTENSIONS=(".n",), to_lines=_raising(refusal))
    with pytest.raises(WriterError, match="Picky"):
        io.save(Note(text="hi"), tmp_path / "x.n")
    assert list(tmp_path.iterdir()) == []
    # A file that is already there is left as it was.
    (tmp_path / "x.n").write_text("before\n")
    with pytest.raises(WriterError, match="Picky"):
        io.save(Note(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "before"
    assert list(tmp_path.iterdir()) == [tmp_path / "x.n"]


@pytest.mark.parametrize(
    "error",
    [PermissionError(13, "Permission denied"), RuntimeError("a bug")],
    ids=lambda e: type(e).__name__,
)
def test_an_error_that_is_not_a_refusal_stops_the_save(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
    error: Exception,
) -> None:
    formats("Broken", EXTENSIONS=(".n",), PRIORITY=1, to_lines=_raising(error))
    formats("Plain", EXTENSIONS=(".n",))
    with pytest.raises(type(error)):
        io.save(Note(text="hi"), tmp_path / "x.n")
    assert list(tmp_path.iterdir()) == []


def test_an_os_error_while_building_stops_the_save(formats, tmp_path) -> None:  # noqa: ANN001
    # Building a candidate may read the data of the object, from a file
    # that can no longer be read.
    error = FileNotFoundError(2, "No such file or directory", "source.nii")
    formats(
        "Broken",
        EXTENSIONS=(".n",),
        PRIORITY=1,
        from_instance=_refusing_build(error),
    )
    formats("Plain", EXTENSIONS=(".n",))
    with pytest.raises(FileNotFoundError, match="source.nii"):
        io.save(Note(text="hi"), tmp_path / "x.n")
    assert list(tmp_path.iterdir()) == []


def test_two_candidates_that_build_are_still_an_ambiguity(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # Only one candidate of a tier is written, so a tier where two
    # candidates build is not resolved by writing one and then the other.
    refusal = UnrepresentableTransformationError("cannot encode")
    formats("One", EXTENSIONS=(".n",), to_lines=_raising(refusal))
    formats("Two", EXTENSIONS=(".n",))
    with pytest.raises(AmbiguousFormatError):
        io.save(Note(text="hi"), tmp_path / "x.n")
    assert list(tmp_path.iterdir()) == []


def test_an_object_of_a_format_that_refuses_is_not_converted(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # An object of one of the candidates is written as it is, and only in
    # its own format.
    refusal = UnrepresentableTransformationError("cannot encode")
    picky = formats("Picky", EXTENSIONS=(".n",), to_lines=_raising(refusal))
    formats("Plain", EXTENSIONS=(".n",))
    with pytest.raises(UnrepresentableTransformationError):
        io.save(picky(text="hi"), tmp_path / "x.n")
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------
#   ROUND TRIPS THROUGH EVERY WRITABLE FORMAT
# ----------------------------------------------------------------------


@needs_nibabel
@pytest.mark.parametrize("name", ["image.nii", "image.nii.gz"])
def test_an_image_round_trips_through_nifti(tmp_path, name: str) -> None:  # noqa: ANN001
    from brainhops.io.images.nifti import NiftiImage

    io.save(_image(), tmp_path / name)
    back = io.images.load(tmp_path / name)
    assert isinstance(back, NiftiImage)
    assert np.array_equal(np.asarray(back.data), DATA)
    np.testing.assert_allclose(back.transformation.matrix, MATRIX)


@needs_nibabel
def test_a_nifti_affine_round_trips(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.nifti import NiftiVoxelToRAS

    affine = NiftiVoxelToRAS(matrix=MATRIX)
    io.save(affine, tmp_path / "affine.nii")
    back = NiftiVoxelToRAS.load(tmp_path / "affine.nii")
    np.testing.assert_allclose(back.matrix, MATRIX)


@needs_nibabel
def test_a_nifti_coordinate_field_round_trips(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.nifti import NiftiRASCoordinateField

    field = np.zeros((2, 3, 4, 3), dtype="float32")
    field[..., 0] = 1.0
    io.save(NiftiRASCoordinateField(field=field), tmp_path / "field.nii")
    back = io.transformations.load(tmp_path / "field.nii")
    assert isinstance(back, NiftiRASCoordinateField)
    np.testing.assert_array_equal(np.asarray(back.field), field)


@needs_zarr
def test_an_image_round_trips_through_zarr(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.images.zarr import ZarrImage

    io.save(SingleScaleImage(data=DATA), tmp_path / "image.zarr")
    back = io.images.load(tmp_path / "image.zarr")
    assert isinstance(back, ZarrImage)
    assert np.array_equal(np.asarray(back.data), DATA)


@needs_zarr
def test_a_geometry_is_not_dropped_by_plain_zarr(tmp_path) -> None:  # noqa: ANN001
    # A plain Zarr array stores no geometry, so the writer refuses an image
    # that has one rather than drop it, and writes nothing. No other format
    # that claims `.zarr` holds a single image, so the save is refused with
    # the reason of the plain Zarr writer.
    pattern = "ZarrImage: UnrepresentableTransformationError"
    with pytest.raises(WriterError, match=pattern):
        io.save(_image(), tmp_path / "image.zarr")
    assert not (tmp_path / "image.zarr").exists()
    # A store that is already there is left as it was.
    io.save(SingleScaleImage(data=DATA + 1), tmp_path / "image.zarr")
    with pytest.raises(WriterError, match=pattern):
        io.save(_image(), tmp_path / "image.zarr")
    back = io.images.load(tmp_path / "image.zarr")
    assert np.array_equal(np.asarray(back.data), DATA + 1)
    assert [path.name for path in tmp_path.iterdir()] == ["image.zarr"]


@needs_zarr
@pytest.mark.parametrize("name", ["pyramid.zarr", "pyramid.ome.zarr"])
def test_a_multiscale_image_round_trips_through_ome_zarr(
    tmp_path,  # noqa: ANN001
    name: str,
) -> None:
    from brainhops.io.images.zarr import OmeZarrImage

    # OME-Zarr 0.6 only scales and translates a level, so the affine is
    # given as the placement of the pyramid rather than of its level.
    level = SingleScaleImage(data=DATA)
    pyramid = MultiScaleImage(images=[level], transformations=[Affine(MATRIX)])
    io.save(pyramid, tmp_path / name)
    back = io.images.load(tmp_path / name)
    assert isinstance(back, OmeZarrImage)
    assert np.array_equal(np.asarray(back.images[0].data), DATA)


@needs_zarr
def test_an_ome_zarr_field_round_trips(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.zarr import OmeZarrField
    from tests.test_io_zarr_fields import write_field_store

    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 7.0
    source = write_field_store(
        tmp_path,
        field,
        [("z", "space"), ("y", "space"), ("x", "space"), ("c", "coordinate")],
        {"type": "identity"},
        name="source.ome.zarr",
    )
    read = io.transformations.load(source)
    assert isinstance(read, OmeZarrField)
    io.save(read, tmp_path / "copy.ome.zarr")
    back = io.transformations.load(tmp_path / "copy.ome.zarr")
    assert isinstance(back, OmeZarrField)
    np.testing.assert_array_equal(np.asarray(back.raw_levels[0]), field)


# ----------------------------------------------------------------------
#   CONVERTING BETWEEN FORMATS, AND WHAT IS NOT CONVERTED
# ----------------------------------------------------------------------


@needs_nibabel
@needs_zarr
def test_an_image_read_from_nifti_is_written_as_zarr(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.images.zarr import ZarrImage

    io.save(_image(), tmp_path / "image.nii")
    nifti = io.images.load(tmp_path / "image.nii")
    # The geometry of the NIfTI file cannot be stored, so it is refused
    # until it is cleared.
    with pytest.raises(
        WriterError, match="UnrepresentableTransformationError"
    ):
        io.save(nifti, tmp_path / "image.zarr")
    nifti.transformations = []
    io.save(nifti, tmp_path / "image.zarr")
    back = io.images.load(tmp_path / "image.zarr")
    assert isinstance(back, ZarrImage)
    assert np.array_equal(np.asarray(back.data), DATA)


@needs_zarr
def test_a_single_image_is_written_as_a_one_level_ome_zarr(tmp_path) -> None:  # noqa: ANN001, E501
    # .ome.zarr asks for an OME-Zarr pyramid, which holds a single image
    # as its only level, and plain Zarr is not substituted (#624).
    from brainhops.io.images.zarr import OmeZarrImage

    io.save(_image(), tmp_path / "image.ome.zarr")
    back = io.images.load(tmp_path / "image.ome.zarr")
    assert isinstance(back, OmeZarrImage)
    assert back.nscales == 1
    assert np.array_equal(np.asarray(back.images[0].data), DATA)


@needs_nibabel
@pytest.mark.parametrize(
    "obj",
    [
        Affine(MATRIX, input=RASmm(), output=RASmm()),
        Scaling([1.0, 2.0, 3.0], input=LPSmm(), output=LPSmm()),
        CoordinateField(field=np.zeros((2, 3, 4, 3)), degree=2),
        DisplacementField(field=np.zeros((2, 3, 4, 3)), degree=2),
    ],
    ids=[
        "world-affine",
        "scaling",
        "quadratic-coordinates",
        "quadratic-displacement",
    ],
)
def test_a_transformation_no_nifti_format_holds_is_refused(
    tmp_path,  # noqa: ANN001
    obj: tx.Any,
) -> None:
    # A NIfTI transformation maps voxels to RAS, or RAS to RAS through a
    # linearly interpolated field. A transformation that does not, such
    # as an affine or a scaling between world spaces, or a field
    # interpolated with quadratic splines, would be read back as a
    # different map. It is therefore refused, with the reason of each
    # format. A cubic field, of displacements or of coordinates, is held
    # exactly by a NiftyReg grid of control points, so quadratic ones are
    # refused instead.
    with pytest.raises(WriterError, match="from_any"):
        io.save(obj, tmp_path / "transform.nii")
    assert not (tmp_path / "transform.nii").exists()


@needs_nibabel
def test_the_explicit_route_writes_a_general_affine(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.nifti import NiftiVoxelToRAS

    NiftiVoxelToRAS.from_any(Affine(MATRIX)).save(tmp_path / "affine.nii")
    back = NiftiVoxelToRAS.load(tmp_path / "affine.nii")
    np.testing.assert_allclose(back.matrix, MATRIX)


@needs_nibabel
def test_an_image_is_written_to_an_open_named_file(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.images.nifti import NiftiImage

    io.save(_image(), tmp_path / "image.nii")
    nifti = io.images.load(tmp_path / "image.nii")
    with open(tmp_path / "copy.nii", "wb") as f:
        io.save(nifti, f)
    back = NiftiImage.load(tmp_path / "copy.nii")
    assert np.array_equal(np.asarray(back.data), DATA)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _raising(error: Exception) -> tx.Callable[..., tx.Iterator[str]]:
    """A `to_lines` that raises `error` after the first line is written."""

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:  # noqa: ANN001
        yield "partial"
        raise error

    return to_lines


def _refusing_build(error: Exception) -> classmethod:
    """A `from_instance` that raises `error` instead of building."""

    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Any:  # noqa: ANN001
        raise error

    return classmethod(from_instance)
