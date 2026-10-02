"""
Tests for `brainhops.io.save`, which writes an object in the format its
file name calls for.

The file name gives the candidates: the formats declaring the longest
extension it ends with, and the prefix they require. The object gives
the one that is used: a format it already is, or a file-backed version
of its very data model that takes all of its fields. Every writable
format is round-tripped through it.
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
from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Scaling,
)
from brainhops.io.base._base import (
    WritableTextFileBasedObject,
    register_format,
)
from brainhops.io.base.parsers import AmbiguousFormatError, WriterError

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
    """A data model with one field, for formats written as plain text."""

    text: str = ""


class OtherNote(DataModelBase):
    """A data model unrelated to `Note`."""

    text: str = ""


def _note_format(name: str, model: type = Note, **attrs: tx.Any) -> type:
    """A text format for `model`, written as its text after a tag."""

    def to_lines(self, **kwargs) -> tx.Iterator[str]:  # noqa: ANN001
        yield f"{name}:{kwargs.get('suffix', '')}{self.text}"

    namespace = {"to_lines": to_lines}
    namespace.update(attrs)
    return type(name, (model, WritableTextFileBasedObject), namespace)


@pytest.fixture
def formats() -> tx.Iterator[tx.Callable[..., type]]:
    """
    Register fake writable formats, and unregister them afterwards.

    `save` chooses among the real writable registry, so the fakes must go
    in it, and must leave it as they found it.
    """
    made: tx.List[type] = []

    def make(name: str, model: type = Note, **attrs: tx.Any) -> type:
        fmt = register_format(_note_format(name, model, **attrs))
        made.append(fmt)
        return fmt

    yield make
    for fmt in made:
        for base in fmt.__mro__:
            if "_REGISTRY" in base.__dict__:
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
    formats("Short", EXTENSIONS=(".gz",))
    formats("Long", EXTENSIONS=(".note.gz",))
    io.save(Note(text="hi"), tmp_path / "x.note.gz")
    assert _written(tmp_path / "x.note.gz") == "Long:hi"
    io.save(Note(text="hi"), tmp_path / "x.gz")
    assert _written(tmp_path / "x.gz") == "Short:hi"


def test_a_longer_extension_is_not_given_up_for_a_shorter_one(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # The name asks for the longer extension's format. When that format
    # cannot hold the object, nothing is written, rather than falling
    # back to the format of a shorter extension the name also ends with.
    formats("Short", EXTENSIONS=(".gz",))
    formats("Long", model=OtherNote, EXTENSIONS=(".note.gz",))
    with pytest.raises(WriterError, match="Long"):
        io.save(Note(text="hi"), tmp_path / "x.note.gz")
    assert not (tmp_path / "x.note.gz").exists()


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

    # The file and the object, then one line per candidate, in order.
    assert "'x.n'" in lines[0] and "Note" in lines[0]
    assert "2 formats" in lines[0]
    # Each candidate is described in the words of its own docstring, when
    # it has one, and comes with the call that writes the object in it.
    assert lines[1] == "  - Bare: `Bare.from_other(obj).save(path)`"
    assert lines[2] == (
        "  - Plain (A note written as plain text): "
        "`Plain.from_other(obj).save(path)`"
    )
    assert "from_other(obj).save(path)" in lines[-1]
    # Neither the docstring of the data model, nor the list of fields
    # generated for a class with none, describes the format. How
    # maintainers settle a tie is not the user's concern.
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
#   WHAT THE OBJECT IS WRITTEN AS
# ----------------------------------------------------------------------


def test_an_object_of_the_format_is_written_as_it_is(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # Rebuilding it would run its constructor again, and lose state a
    # file-based object keeps outside its fields.
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
    # Two formats of notes claim `.n` equally well. A plain note could go
    # to either, but one that already is of one of them stays in it.
    one = formats("One", EXTENSIONS=(".n",))
    formats("Two", EXTENSIONS=(".n",))
    io.save(one(text="hi"), tmp_path / "x.n")
    assert _written(tmp_path / "x.n") == "One:hi"


def test_a_subclass_of_the_data_model_is_not_squeezed_into_it(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # A format of `Note` would drop what a richer note says.
    class RichNote(Note):
        colour: str = "red"

    formats("A", EXTENSIONS=(".a",))
    with pytest.raises(WriterError):
        io.save(RichNote(text="hi"), tmp_path / "x.a")


def test_a_format_missing_a_field_of_the_data_model_cannot_hold_it(
    formats,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # This format serves `text` itself rather than taking it, so building
    # it from a note would lose the note's text.
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
    from brainhops.io.transformations.nifti import NiftiVoxelToRas

    affine = NiftiVoxelToRas(matrix=MATRIX)
    io.save(affine, tmp_path / "affine.nii")
    back = NiftiVoxelToRas.load(tmp_path / "affine.nii")
    np.testing.assert_allclose(back.matrix, MATRIX)


@needs_nibabel
def test_a_nifti_coordinate_field_round_trips(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.nifti import NiftiRasCoordinatesField

    field = np.zeros((2, 3, 4, 3), dtype="float32")
    field[..., 0] = 1.0
    io.save(NiftiRasCoordinatesField(field=field), tmp_path / "field.nii")
    back = io.transformations.load(tmp_path / "field.nii")
    assert isinstance(back, NiftiRasCoordinatesField)
    np.testing.assert_array_equal(np.asarray(back.field)[..., 0, :], field)


@needs_zarr
def test_an_image_round_trips_through_zarr(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.images.zarr import ZarrImage

    io.save(_image(), tmp_path / "image.zarr")
    back = io.images.load(tmp_path / "image.zarr")
    assert isinstance(back, ZarrImage)
    assert np.array_equal(np.asarray(back.data), DATA)


@needs_zarr
@pytest.mark.parametrize("name", ["pyramid.zarr", "pyramid.ome.zarr"])
def test_a_multiscale_image_round_trips_through_ome_zarr(
    tmp_path,  # noqa: ANN001
    name: str,
) -> None:
    from brainhops.io.images.zarr import OmeZarrImage

    io.save(MultiScaleImage(images=[_image()]), tmp_path / name)
    back = io.images.load(tmp_path / name)
    assert isinstance(back, OmeZarrImage)
    assert np.array_equal(np.asarray(back.images[0].data), DATA)


@needs_zarr
def test_an_ome_zarr_field_round_trips(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.zarr import OmeZarrField
    from tests.test_io_zarr_fields import _write_field_store

    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 7.0
    source = _write_field_store(
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
    io.save(nifti, tmp_path / "image.zarr")
    back = io.images.load(tmp_path / "image.zarr")
    assert isinstance(back, ZarrImage)
    assert np.array_equal(np.asarray(back.data), DATA)


@needs_zarr
def test_a_single_image_is_not_written_as_ome_zarr(tmp_path) -> None:  # noqa: ANN001
    # `.ome.zarr` asks for OME-Zarr, whose format holds a pyramid. A plain
    # Zarr store is not written in its place.
    with pytest.raises(WriterError, match="OmeZarrImage"):
        io.save(_image(), tmp_path / "image.ome.zarr")
    assert not (tmp_path / "image.ome.zarr").exists()


@needs_nibabel
@pytest.mark.parametrize(
    "obj",
    [
        Affine(MATRIX),
        Scaling([1.0, 2.0, 3.0]),
        CoordinatesField(field=np.zeros((2, 3, 4, 3))),
        DisplacementField(field=np.zeros((2, 3, 4, 3))),
    ],
    ids=["affine", "scaling", "coordinates", "displacement"],
)
def test_a_general_transformation_is_not_given_a_nifti_meaning(
    tmp_path,  # noqa: ANN001
    obj: tx.Any,
) -> None:
    # A NIfTI transformation means voxel-to-RAS. A general one would come
    # back meaning something it did not say, so it is refused, and the
    # caller builds the NIfTI format explicitly if that is what is meant.
    with pytest.raises(WriterError, match="from_other"):
        io.save(obj, tmp_path / "transform.nii")
    assert not (tmp_path / "transform.nii").exists()


@needs_nibabel
def test_the_explicit_route_writes_a_general_affine(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.nifti import NiftiVoxelToRas

    NiftiVoxelToRas.from_other(Affine(MATRIX)).save(tmp_path / "affine.nii")
    back = NiftiVoxelToRas.load(tmp_path / "affine.nii")
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
