"""
Tests for the format-agnostic metadata (`brainhops.datamodel.metadata`).

What is checked: the vocabulary groups and their annotations; the
`repr` and the equality of generic metadata; the typed terms (enums,
units, data types, encoding directions); `to()` between generic
metadata; the loss reports and the loss policy; and `derive`, with the
records of the image operations (`Indexed` and `Resampled`). The
metadata of the file formats is tested in `test_io_metadata_*.py`.
"""

import datetime
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest
from bagof.magic import fields

import brainhops.datamodel.metadata
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.enums import (
    AxisType,
    ContrastMethod,
    IlluminationType,
    IntentEnum,
    Manufacturer,
    SpaceEnum,
)
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.metadata import (
    Channel,
    ConversionReport,
    DiffusionVocabulary,
    DisplayVocabulary,
    EncodingDirection,
    GeneratedBy,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    MicroscopyVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    Scope,
    StorageVocabulary,
    TransformVocabulary,
    Vocabulary,
    metadata_loss_policy,
)
from brainhops.datamodel.metadata._derive import Indexed, Resampled
from brainhops.datamodel.metadata._report import apply_loss_policy
from brainhops.datamodel.metadata._vocabulary import (
    FIELDS,
    GROUPS,
    VOCABULARY,
    Along,
    Scoped,
)
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, CartesianField

# ----------------------------------------------------------------------
#   VOCABULARY GROUPS
# ----------------------------------------------------------------------


def test_the_public_names() -> None:
    assert sorted(brainhops.datamodel.metadata.__all__) == sorted(
        [
            "Metadata",
            "Vocabulary",
            *(group.__name__ for group in GROUPS),
            "Scope",
            "GeneratedBy",
            "Channel",
            "EncodingDirection",
            "ConversionReport",
            "MetadataLossWarning",
            "MetadataLossError",
            "metadata_loss_policy",
        ]
    )


def test_the_vocabulary_is_the_groups_in_order() -> None:
    assert list(GROUPS) == [
        ProvenanceVocabulary,
        MRIVocabulary,
        DiffusionVocabulary,
        DisplayVocabulary,
        StorageVocabulary,
        MicroscopyVocabulary,
        TransformVocabulary,
    ]
    for group in GROUPS:
        assert issubclass(Metadata, group)
        assert issubclass(group, Vocabulary)
    assert VOCABULARY == sum(GROUPS.values(), ())
    assert VOCABULARY[:2] == ("name", "description")
    assert GROUPS[DiffusionVocabulary] == ("bvalues", "bvectors")
    assert GROUPS[StorageVocabulary] == (
        "dtype",
        "scale_slope",
        "scale_intercept",
    )
    assert FIELDS == ("extra",) + VOCABULARY
    assert len(set(VOCABULARY)) == len(VOCABULARY)
    # The fields of a group convert, as those of the class that inherits
    # them do.
    assert Metadata(echo_time="0.03", bvalues=[0, 1000]).bvalues == (
        0.0,
        1000.0,
    )


def test_the_fields_of_metadata_are_the_vocabulary_and_extra() -> None:
    assert {f.name for f in fields(Metadata)} == set(FIELDS)
    # The data model does no input or output, and holds no record.
    for name in ("raw", "load", "from_raw", "consolidate"):
        assert not hasattr(Metadata, name)


def test_bids_and_scope_annotations() -> None:
    by_name = {f.name: f.metadata for f in fields(Metadata)}
    assert by_name["repetition_time"]["bids"] == "RepetitionTime"
    assert by_name["repetition_time"]["scope"] is Scope.ACQUISITION
    assert by_name["slice_timing"]["scope"] is Scope.SPATIAL
    assert by_name["channels"]["scope"] is Scope.AXIS
    assert by_name["channels"]["along"] is AxisType.channel
    assert by_name["bvalues"]["along"] is AxisType.time
    assert by_name["display_range"]["scope"] is Scope.FILE
    assert by_name["dtype"]["scope"] is Scope.FILE
    with pytest.raises(ValueError):
        Scoped(Scope.AXIS)
    with pytest.raises(ValueError):
        Along("space")
    assert "bids" not in by_name["display_range"]


def test_repr_hides_none_and_an_empty_extra() -> None:
    # `extra` first, only when not empty, then the vocabulary in its
    # declared order.
    assert repr(Metadata(echo_time=0.03, description="x", extra={})) == (
        "Metadata(description='x', echo_time=0.03)"
    )
    assert repr(Metadata(name="n", extra={"k": 1})) == (
        "Metadata(extra={'k': 1}, name='n')"
    )
    assert repr(Metadata()) == "Metadata()"


def test_generic_metadata_compares_by_value() -> None:
    assert Metadata(description="d") == Metadata(description="d")
    assert Metadata(description="d") != Metadata(description="e")
    assert Metadata(extra={"k": 1}) != Metadata()
    meta = Metadata(description="d", phase_encoding_direction="j-")
    assert pickle.loads(pickle.dumps(meta)) == meta


# ----------------------------------------------------------------------
#   TERMS
# ----------------------------------------------------------------------


def test_known_terms_become_enum_members() -> None:
    meta = Metadata(
        space="MNI152NLin6Asym",
        intent="label",
        manufacturer="Siemens",
        illumination_type="Epifluorescence",
        contrast_method="DIC",
        input_space="scanner",
    )
    assert meta.space is SpaceEnum.MNI152NLin6Asym
    assert meta.intent is IntentEnum.label
    assert meta.manufacturer is Manufacturer.Siemens
    assert meta.illumination_type is IlluminationType.Epifluorescence
    assert meta.contrast_method is ContrastMethod.DIC
    assert meta.input_space is SpaceEnum.scanner
    # They are strings, and compare equal to their value.
    assert meta.space == "MNI152NLin6Asym" and meta.intent == "label"
    # An unknown term stays a string: the vocabulary is not closed.
    meta.space = "my-template"
    assert type(meta.space) is str and meta.space == "my-template"
    with pytest.raises((TypeError, ValueError)):
        Metadata(space=3)


def test_data_unit_is_a_unit_when_known() -> None:
    from brainhops.datamodel.units import Unit

    meta = Metadata(data_unit="ms")
    assert isinstance(meta.data_unit, Unit)
    assert meta.data_unit == Unit("millisecond")
    # Any name the units module parses is a `Unit`, compared as a unit.
    assert Metadata(data_unit="a.u.").data_unit == Unit("au")
    assert Metadata(data_unit="mm/s").data_unit == Unit("millimeter/second")
    # A name it cannot parse stays the file's own string.
    assert Metadata(data_unit="mm2/s").data_unit == "mm2/s"
    with pytest.raises(TypeError):
        Metadata(data_unit=3)


def test_dtype_is_a_native_dtype() -> None:
    meta = Metadata(dtype=">i2")
    assert meta.dtype == np.dtype("int16")
    assert meta.dtype.isnative
    assert Metadata().dtype is None
    # How the file stores the values: kept by `derive` (`FILE` scope).
    assert meta.derive(Resampled(CartesianField())).dtype == np.int16
    assert meta.derive(_indexed(Ellipsis, 0)).dtype == np.int16


def test_a_channel_color_is_held_as_rgba() -> None:
    # One spelling per color, so that a record read back agrees with it.
    assert Channel(color="#0000ff").color == "0000FFFF"
    assert Channel(color="00FF0080").color == "00FF0080"
    assert Channel().color is None


# ----------------------------------------------------------------------
#   ENCODING DIRECTIONS
# ----------------------------------------------------------------------


def test_an_encoding_direction_is_a_vector_in_voxel_axes() -> None:
    direction = EncodingDirection("j-")
    assert direction.vector == (0.0, -1.0, 0.0)
    assert direction.space is None
    assert direction.to_bids() == "j-"
    assert direction == EncodingDirection((0, -1, 0))
    assert direction != EncodingDirection("j")
    # Equality is the fields': a direction is not its BIDS string.
    assert direction != "j-"
    assert repr(direction) == "EncodingDirection('j-')"
    assert EncodingDirection("k") == EncodingDirection((0, 0, 2))
    with pytest.raises(ValueError):
        EncodingDirection("x")
    with pytest.raises(ValueError):
        EncodingDirection((0, 0, 0))
    oblique = EncodingDirection((1, 1, 0))
    assert oblique.vector == pytest.approx((2**-0.5, 2**-0.5, 0.0))
    assert oblique.to_bids() is None
    world = EncodingDirection((0, 1, 0), space="scanner")
    assert world.space is SpaceEnum.scanner and world.to_bids() is None


def test_a_direction_may_be_in_a_coordinate_system() -> None:
    named = CoordinateSystem(name="RAS")
    direction = EncodingDirection((1, 0, 0), space=named)
    assert direction.space is named
    assert direction.to_bids() is None
    # A system equals an equal system, never a label, not even its name.
    assert direction == EncodingDirection(
        "i", space=CoordinateSystem(name="RAS")
    )
    assert direction != EncodingDirection((1, 0, 0), space="RAS")
    assert "space=CoordinateSystem(" in repr(direction)


def test_a_direction_is_mapped_by_a_transformation() -> None:
    from brainhops.datamodel.transformations import DisplacementField

    swap = np.array(
        [[0, 1, 0, 5], [1, 0, 0, -2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float
    )
    direction = EncodingDirection("j-")
    # The linear part of an affine maps it; its translation does not.
    assert direction.transform(Affine(swap)) == EncodingDirection("i-")
    assert direction.transform(swap[:3, :3]) == EncodingDirection("i-")
    with pytest.raises(TypeError, match="affine"):
        direction.transform(DisplacementField(np.zeros((2, 2, 2, 3))))


def test_the_direction_fields_take_bids_strings() -> None:
    meta = Metadata(phase_encoding_direction="j-")
    assert isinstance(meta.phase_encoding_direction, EncodingDirection)
    assert meta.phase_encoding_direction.to_bids() == "j-"
    meta.slice_encoding_direction = {"Vector": [0, 0, 1]}
    assert meta.slice_encoding_direction == EncodingDirection("k")


# ----------------------------------------------------------------------
#   to() AND from_instance
# ----------------------------------------------------------------------


def test_to_copies_generic_metadata() -> None:
    meta = Metadata(description="d", extra={"Key": 1})
    # Nothing changes, so the metadata itself is returned unless a copy is
    # asked for.
    assert meta.to() is meta
    same = meta.to(copy=True)
    assert type(same) is Metadata and same == meta and same is not meta
    # The free-form keys are copied, not shared.
    assert same.extra == meta.extra and same.extra is not meta.extra
    assert meta.to(description="other").description == "other"
    # Generic metadata supports every field: nothing is lost.
    with warnings.catch_warnings(), metadata_loss_policy("raise"):
        warnings.simplefilter("error")
        report = ConversionReport()
        Metadata(echo_time=0.03).to(Metadata, on_loss=report)
    assert not report.lossy


# ----------------------------------------------------------------------
#   POLICIES
# ----------------------------------------------------------------------


def _lossy() -> ConversionReport:
    return ConversionReport(source="a", target="b", lost={"echo_time": 0.03})


def test_the_policies() -> None:
    with pytest.warns(MetadataLossWarning) as record:
        apply_loss_policy(_lossy())
    assert record[0].message.report.lost == {"echo_time": 0.03}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_loss_policy(_lossy(), "ignore")
        # Nothing lost: silent whatever the policy.
        apply_loss_policy(ConversionReport(), "raise")
    with pytest.raises(MetadataLossError) as info:
        apply_loss_policy(_lossy(), "raise")
    assert info.value.report.lost == {"echo_time": 0.03}
    with pytest.raises(ValueError):
        apply_loss_policy(_lossy(), "loud")


def test_a_report_as_on_loss_is_filled_silently() -> None:
    sink = ConversionReport()
    with warnings.catch_warnings(), metadata_loss_policy("raise"):
        warnings.simplefilter("error")
        apply_loss_policy(_lossy(), sink)
    assert sink.lost == {"echo_time": 0.03}
    assert (sink.source, sink.target) == ("a", "b")


def test_the_policy_context_manager_nests_and_restores() -> None:
    with metadata_loss_policy("raise"):
        with metadata_loss_policy("ignore"):
            apply_loss_policy(_lossy())
        with pytest.raises(MetadataLossError):
            apply_loss_policy(_lossy())
    with pytest.warns(MetadataLossWarning):
        apply_loss_policy(_lossy())
    with pytest.raises(ValueError):
        with metadata_loss_policy("loud"):
            pass


def test_a_report_raises_and_merges() -> None:
    report = ConversionReport(source="a", target="b")
    report.raise_if_lossy()
    report.merge(ConversionReport(lost={"x": 1}))
    assert report.lost == {"x": 1}
    assert "x=1" in str(report)
    with pytest.raises(MetadataLossError):
        report.raise_if_lossy()


# ----------------------------------------------------------------------
#   DERIVE
# ----------------------------------------------------------------------


def _scoped() -> Metadata:
    return Metadata(
        description="d",
        creation_time=datetime.datetime(2020, 1, 1),
        history=("acquired",),
        echo_time=0.03,
        slice_timing=(0.0, 0.5),
        phase_encoding_direction="j-",
        channels=(Channel(name="a"), Channel(name="b"), Channel(name="c")),
        bvalues=(0, 1000, 2000),
        display_range=(0, 1),
        extra={"Key": 1},
    )


def _dwi_system() -> CoordinateSystem:
    """The voxel axes of a 4-D diffusion series: three spatial axes and a
    time axis."""
    return CoordinateSystem(
        axes=[
            Axis("i", "space"),
            Axis("j", "space"),
            Axis("k", "space"),
            Axis("t", "time"),
        ]
    )


def _indexed(*index) -> Indexed:  # noqa: ANN002
    """`image[index]`, for a diffusion series of shape (4, 5, 6, 3)."""
    return Indexed(index, (4, 5, 6, 3), _dwi_system())


def _mapped(matrix) -> Resampled:  # noqa: ANN001
    """A resampling whose map from the old voxels to the new ones has the
    linear part `matrix`."""
    matrix = np.asarray(matrix, dtype=float)
    full = np.eye(len(matrix) + 1)
    full[:-1, :-1] = matrix
    # The transformation of `Resampled` goes from the new voxels to the
    # old ones.
    return Resampled(Affine(np.linalg.inv(full)[:-1]))


def test_derive_records_provenance() -> None:
    meta = _scoped()
    derived = meta.derive(history="smooth")
    assert derived is not meta
    assert derived.description == "d"
    assert derived.creation_time is None
    assert derived.history == ("acquired", "smooth")
    assert derived.echo_time == 0.03
    assert derived.slice_timing == (0.0, 0.5)
    assert derived.phase_encoding_direction == EncodingDirection("j-")
    assert derived.bvalues == (0.0, 1000.0, 2000.0)
    assert derived.extra == {"Key": 1}
    assert derived.extra is not meta.extra
    assert [g.name for g in derived.generated_by] == ["brainhops"]
    # The brainhops entry is added once.
    assert len(derived.derive().generated_by) == 1
    # A string is one entry; a sequence, several; `None`, none.
    assert meta.derive().history == ("acquired",)
    assert meta.derive(history=["a", "b"]).history == ("acquired", "a", "b")
    assert Metadata().derive().history is None


def test_indexed_expands_its_index() -> None:
    op = _indexed(Ellipsis, 1)
    assert op.expanded == (slice(None),) * 3 + (1,)
    assert op.axes == (AxisType.space,) * 3 + (AxisType.time,)
    assert Indexed(0, (4, 5)).index == (0,)
    assert Indexed((0,), (4, 5)).axes == (None, None)


def test_indexed_positions_along_an_axis() -> None:
    assert list(_indexed(Ellipsis, [2, 0]).positions("time")) == [2, 0]
    assert list(_indexed(Ellipsis, slice(1, 3)).positions(AxisType.time)) == [
        1,
        2,
    ]
    # Untouched: the full range.
    assert list(_indexed(slice(None)).positions("time")) == [0, 1, 2]
    # A boolean mask keeps the positions it selects.
    mask = np.array([True, False, True])
    assert list(_indexed(Ellipsis, mask).positions("time")) == [0, 2]
    # A dropped axis, a selection out of range or not 1-D, or an axis the
    # image does not have: no positions.
    assert _indexed(Ellipsis, 1).positions("time") is None
    assert _indexed(Ellipsis, [5]).positions("time") is None
    assert _indexed(Ellipsis, np.array([[0, 1]])).positions("time") is None
    assert _indexed(Ellipsis).positions("channel") is None


def test_indexed_moves_space() -> None:
    assert not _indexed(Ellipsis, slice(0, 2)).moves_space
    assert not _indexed(Ellipsis, 1).moves_space
    assert not _indexed(slice(None), slice(0, 5)).moves_space
    assert _indexed(slice(None), slice(None, None, -1)).moves_space
    assert _indexed(None).moves_space
    assert _indexed(2).moves_space
    # An axis of unknown type is taken to be spatial.
    assert Indexed((Ellipsis, 1), (4, 5, 6, 3)).moves_space


def test_indexed_voxel_map() -> None:
    flip = _indexed(slice(None), slice(None, None, -1)).voxel_map
    assert np.allclose(flip, np.diag([1.0, -1.0, 1.0, 1.0]))
    # A dropped axis has a zero column.
    dropped = _indexed(slice(None), 2).voxel_map
    assert dropped.shape == (3, 4)
    assert np.allclose(dropped[:, 1], 0)
    # An index other than integers, slices and `None` has no map.
    assert _indexed(Ellipsis, [0, 2]).voxel_map is None


def test_derive_indexed_selects_the_entries_along_the_axis() -> None:
    meta = _scoped()
    selected = meta.derive(_indexed(Ellipsis, [2, 0]), history="getitem")
    assert selected.bvalues == (2000.0, 0.0)
    assert meta.derive(_indexed(Ellipsis, slice(1, 3))).bvalues == (
        1000.0,
        2000.0,
    )
    assert meta.derive(_indexed(Ellipsis, slice(0, 0))).bvalues == ()
    # A dropped axis, or a position beyond the field, clears it.
    assert meta.derive(_indexed(Ellipsis, 1)).bvalues is None
    short = Indexed((Ellipsis, [5]), (4, 5, 6, 8), _dwi_system())
    assert meta.derive(short).bvalues is None
    # The other axes, and the other fields, are untouched.
    assert [c.name for c in selected.channels] == ["a", "b", "c"]
    assert selected.slice_timing == (0.0, 0.5)
    assert selected.phase_encoding_direction == EncodingDirection("j-")
    assert selected.display_range == (0.0, 1.0)
    assert selected.history == ("acquired", "getitem")
    assert selected.creation_time is None
    channels = CoordinateSystem(
        axes=[Axis("i", "space"), Axis("c", "channel")]
    )
    channel = meta.derive(Indexed((Ellipsis, [2, 0]), (4, 3), channels))
    assert [c.name for c in channel.channels] == ["c", "a"]
    assert channel.bvalues == (0.0, 1000.0, 2000.0)
    # An image without an axis of that type keeps the field.
    assert meta.derive(Indexed((0,), (4, 3))).bvalues == meta.bvalues


def test_derive_maps_directions_through_the_voxel_map() -> None:
    meta = Metadata(
        phase_encoding_direction="j-",
        slice_encoding_direction=EncodingDirection((0, 0, 1), space="mni"),
        slice_timing=(0.0, 0.5),
        echo_time=0.03,
        bvalues=(0, 1000),
    )
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    derived = meta.derive(_mapped(swap), history="reslice")
    assert derived.phase_encoding_direction == EncodingDirection("i-")
    # A direction in a world space does not move with the voxels.
    assert derived.slice_encoding_direction == EncodingDirection(
        (0, 0, 1), space="mni"
    )
    # The slice timing is cleared, the rest is kept.
    assert derived.slice_timing is None
    assert derived.echo_time == 0.03
    assert derived.bvalues == (0.0, 1000.0)
    assert derived.history == ("reslice",)
    flip = np.diag([1.0, -1.0, 1.0])
    assert meta.derive(_mapped(flip)).phase_encoding_direction == (
        EncodingDirection("j")
    )
    # The map of a 4-D image: the direction lies in its first three axes.
    flip4 = np.diag([1.0, -1.0, 1.0, 1.0])
    assert meta.derive(_mapped(flip4)).phase_encoding_direction == (
        EncodingDirection("j")
    )
    onto_time = np.eye(4)[[0, 3, 2, 1]]
    assert meta.derive(_mapped(onto_time)).phase_encoding_direction is None
    rotated = meta.derive(
        _mapped([[1, 0, 0], [0, 2**-0.5, -(2**-0.5)], [0, 2**-0.5, 2**-0.5]])
    )
    assert rotated.phase_encoding_direction.to_bids() is None
    # Through an index of the voxels.
    indexed = meta.derive(_indexed(slice(None), slice(None, None, -1)))
    assert indexed.phase_encoding_direction == EncodingDirection("j")
    assert meta.derive(_indexed(slice(None), 2)).phase_encoding_direction is (
        None
    )


def test_derive_resampled() -> None:
    meta = Metadata(
        phase_encoding_direction="j-",
        slice_encoding_direction=EncodingDirection((0, 0, 1), space="mni"),
        slice_timing=(0.0, 0.5),
        bvalues=(0, 1000),
    )
    # The transformation goes from the new voxels to the old ones.
    swap = Affine(np.eye(4)[[1, 0, 2, 3]][:-1])
    resampled = Resampled(swap, Geometry())
    assert resampled.moves_space
    assert np.allclose(resampled.voxel_map, np.eye(3)[[1, 0, 2]])
    derived = meta.derive(resampled)
    assert derived.phase_encoding_direction == EncodingDirection("i-")
    assert derived.slice_timing is None
    assert derived.bvalues == (0.0, 1000.0)
    # Without an affine form, a direction in voxel axes is cleared, and
    # one in a world space is kept.
    warped = Resampled(CartesianField())
    assert warped.geometry is None
    assert warped.voxel_map is None
    derived = meta.derive(warped)
    assert derived.phase_encoding_direction is None
    assert derived.slice_encoding_direction.space == "mni"


def test_derive_without_an_operation_keeps_every_field() -> None:
    meta = _scoped()
    assert meta.derive() == meta.derive(None)
    assert meta.derive().slice_timing == meta.slice_timing
    assert meta.derive().phase_encoding_direction == (
        meta.phase_encoding_direction
    )
    # `derive` builds a new object: changing it leaves the source alone.
    derived = meta.derive()
    derived.extra["Other"] = 2
    assert meta.extra == {"Key": 1}


def test_the_metadata_imports_before_the_transformations() -> None:
    # The records import the transformations lazily: the transformations
    # import the metadata.
    code = (
        "import numpy as np\n"
        "from brainhops.datamodel.metadata._derive import Resampled\n"
        "from brainhops.datamodel.transformations import Affine\n"
        "from brainhops.datamodel.geometry import Geometry\n"
        "r = Resampled(Affine(np.eye(3)[:2]), Geometry())\n"
        "assert r.voxel_map.shape == (2, 2)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_generated_by_entries() -> None:
    entry = GeneratedBy(name="fMRIPrep", version="23.0")
    meta = Metadata(generated_by=(entry,))
    assert meta.derive().generated_by[0] == entry
    assert meta.derive().generated_by[1].name == "brainhops"
