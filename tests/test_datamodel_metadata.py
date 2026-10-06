"""
Tests for the format-agnostic metadata framework
(`brainhops.datamodel.metadata`), on synthetic formats.

What is checked: the `UNSUPPORTED` sentinel; the class hierarchy and the
vocabulary groups; the `supports=` class keyword; the
typed terms (enums, units, data types, encoding directions); the read-time
snapshot and the change-detecting write (cases 1-4 of section 6 of the
design memo); `to()`, conversion loss reports and the loss policies;
`derive` and the propagation hooks; the BIDS sidecar codec; and the
`metadata` field of the data model roots.
"""

import copy
import inspect
import json
import pickle
import warnings

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import Factory, Magic, fields, replace

from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.enums import (
    AxisType,
    ContrastMethod,
    IlluminationType,
    IntentEnum,
    Manufacturer,
    SpaceEnum,
)
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import (
    UNSUPPORTED,
    Channel,
    ConversionReport,
    EncodingDirection,
    GeneratedBy,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    Scope,
    metadata_loss_policy,
)
from brainhops.datamodel.metadata._dtype import preferred_dtype
from brainhops.datamodel.metadata._field import MetadataField
from brainhops.datamodel.metadata._report import (
    apply_loss_policy,
    collect_loss_reports,
)
from brainhops.datamodel.metadata._sentinel import ALL, Unsupported
from brainhops.datamodel.metadata._vocabulary import (
    GROUPS,
    VOCABULARY,
    Along,
    DiffusionVocabulary,
    DisplayVocabulary,
    MicroscopyVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    Scoped,
    StorageVocabulary,
    TransformVocabulary,
)
from brainhops.datamodel.transformations import Affine, Translation
from brainhops.io.metadata import (
    FileBasedMetadata,
    OpaqueMetadata,
)


def _to(source, target, **kwargs):  # noqa: ANN001, ANN003, ANN202
    """`source.to(target, ...)`, and the report it filled."""
    report = ConversionReport()
    return source.to(target, on_loss=report, **kwargs), report


# ----------------------------------------------------------------------
#   SYNTHETIC FORMATS
# ----------------------------------------------------------------------


class LiteMetadata(
    FileBasedMetadata,
    on={"format": "test-lite"},
    supports=("description", "slice_timing", "history", "extra"),
):
    """A format that stores three fields and free-form keys."""


class DictRecord(dict):
    """The raw record of `DictMetadata`: a dict of a type of its own, so
    that a conversion knows whose record it is."""


class DictMetadata(
    FileBasedMetadata[DictRecord],
    on={"format": "test-dict"},
    supports=("description", "display_range", "slice_timing", "extra"),
):
    """A format whose record is a dict: `desc`, `cal`, `slices`, and any
    other key is free-form."""

    _KNOWN = {"desc": "description", "cal": "display_range"}

    @classmethod
    def _decode_raw(cls, raw, *, image=None) -> dict:  # noqa: ANN001
        out = {name: raw.get(key) for key, name in cls._KNOWN.items()}
        out["slice_timing"] = raw.get("slices")
        out["extra"] = {
            k: v
            for k, v in raw.items()
            if k not in cls._KNOWN and k != "slices"
        }
        return out

    def _encode_raw(self, raw, changed, *, image=None, report) -> dict:  # noqa: ANN001
        for key, name in self._KNOWN.items():
            if name in changed:
                if changed[name] is None:
                    raw.pop(key, None)
                else:
                    raw[key] = changed[name]
        if "slice_timing" in changed:
            times = changed["slice_timing"]
            if times is None:
                raw.pop("slices", None)
            elif image is not None and len(times) != image:
                # `image` stands for the number of slices here.
                report.lost["slice_timing"] = times
            else:
                raw["slices"] = times
        for key, value in (changed.get("extra") or {}).items():
            if value is None:
                raw.pop(key, None)
            else:
                raw[key] = value
        if "description" in changed and changed["description"]:
            if len(changed["description"]) > 8:
                report.approximated["description"] = "truncated to 8"
                raw["desc"] = changed["description"][:8]
        return raw

    def _reslice(self, linear, *, history=None) -> "DictMetadata":  # noqa: ANN001
        obj = super()._reslice(linear, history=history)
        if obj.raw is not None:
            obj.raw.pop("slice_hint", None)
        return obj


class DialectMetadata(
    FileBasedMetadata,
    on={"format": "test-dialect"},
    supports=("description", "channels"),
):
    """A format whose `channels` support depends on the instance."""

    dialect: str = "rich"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dialect == "plain" and self.channels is None:
            object.__setattr__(self, "channels", UNSUPPORTED)


def _rich() -> Metadata:
    return Metadata(
        description="a scan",
        echo_time=0.03,
        slice_timing=(0.0, 0.5, 1.0),
        history=("acquired",),
        extra={"Custom": 1},
    )


# ----------------------------------------------------------------------
#   SENTINEL
# ----------------------------------------------------------------------


def test_unsupported_is_a_falsy_singleton() -> None:
    assert Unsupported() is UNSUPPORTED
    assert not UNSUPPORTED
    assert UNSUPPORTED is not None
    assert UNSUPPORTED != None  # noqa: E711
    assert repr(UNSUPPORTED) == "UNSUPPORTED"


def test_unsupported_survives_copy_and_pickle() -> None:
    assert copy.copy(UNSUPPORTED) is UNSUPPORTED
    assert copy.deepcopy(UNSUPPORTED) is UNSUPPORTED
    assert pickle.loads(pickle.dumps(UNSUPPORTED)) is UNSUPPORTED


def test_maybe_fields_convert_values_and_keep_the_sentinel() -> None:
    meta = Metadata(repetition_time="2", slice_timing=[0, 1])
    assert meta.repetition_time == 2.0
    assert meta.slice_timing == (0.0, 1.0)
    meta = Metadata(echo_time=UNSUPPORTED)
    assert meta.echo_time is UNSUPPORTED


def test_an_unsupported_field_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="echo_time"):
        LiteMetadata(echo_time=0.03)
    with pytest.raises(ValueError, match="echo_time"):
        LiteMetadata.from_dict({"echo_time": 0.03})
    # None and UNSUPPORTED both read as "nothing here".
    assert LiteMetadata(echo_time=None).echo_time is UNSUPPORTED
    assert LiteMetadata(echo_time=UNSUPPORTED).echo_time is UNSUPPORTED


def test_repr_hides_none_and_unsupported() -> None:
    assert repr(LiteMetadata(description="x")) == (
        "LiteMetadata(format='test-lite', description='x')"
    )
    assert repr(OpaqueMetadata()) == "OpaqueMetadata(format='opaque')"


# ----------------------------------------------------------------------
#   supports=
# ----------------------------------------------------------------------


def test_supports_lists_what_a_format_stores() -> None:
    vocabulary = set(VOCABULARY) | {"extra"}
    supported = {"description", "slice_timing", "history", "extra"}
    assert LiteMetadata.supported_fields == supported
    # The complement is derived from the declaration.
    assert LiteMetadata.unsupported_fields == vocabulary - supported
    assert Metadata.supported_fields == vocabulary
    assert Metadata.unsupported_fields == frozenset()
    assert FileBasedMetadata.supported_fields == vocabulary
    assert OpaqueMetadata.supported_fields == frozenset()
    assert OpaqueMetadata.unsupported_fields == vocabulary
    for name in LiteMetadata.unsupported_fields:
        field = next(f for f in fields(LiteMetadata) if f.name == name)
        assert field.default is UNSUPPORTED


def test_supports_reads_the_class_declaration() -> None:
    assert LiteMetadata.supports("description")
    assert not LiteMetadata.supports("echo_time")
    assert LiteMetadata.supports("extra")
    with pytest.raises(KeyError):
        LiteMetadata.supports("not_a_field")


def test_per_instance_capability() -> None:
    assert DialectMetadata.supports("channels")
    assert DialectMetadata().channels is not UNSUPPORTED
    plain = DialectMetadata(dialect="plain")
    assert plain.channels is UNSUPPORTED
    # `replace` runs `__post_init__` again, so the rule holds.
    assert replace(DialectMetadata(), dialect="plain").channels is UNSUPPORTED


def test_a_subclass_inherits_and_may_change_its_capabilities() -> None:
    class Inherits(LiteMetadata):
        pass

    class Widens(LiteMetadata, supports=("description", "echo_time")):
        pass

    assert Inherits.unsupported_fields == LiteMetadata.unsupported_fields
    assert Widens.supports("echo_time")
    assert not Widens.supports("slice_timing")
    assert Widens(echo_time=0.03).echo_time == 0.03
    assert Widens().echo_time is None


def test_a_field_may_be_declared_unsupported_by_hand() -> None:
    class ByHand(Metadata):
        echo_time: tx.Optional[float] = UNSUPPORTED

    assert ByHand.unsupported_fields == {"echo_time"}


def test_supports_all() -> None:
    class Everything(FileBasedMetadata, supports=ALL):
        pass

    assert Everything.unsupported_fields == frozenset()


def test_wrong_declarations_are_refused() -> None:
    with pytest.raises(TypeError, match="not vocabulary"):

        class Typo(FileBasedMetadata, supports=("descr",)):
            pass

    with pytest.raises(TypeError, match="derived"):

        class Derived(FileBasedMetadata, derived=("description",)):
            pass

    with pytest.raises(TypeError, match="typo_kw"):

        class Misspelt(FileBasedMetadata, typo_kw=1):
            pass


def test_polymorphic_construction_on_format() -> None:
    meta = Metadata(format="test-lite", description="x")
    assert type(meta) is LiteMetadata
    assert type(Metadata(description="x")) is Metadata
    assert type(Metadata(format="opaque")) is OpaqueMetadata
    # An unknown format falls back to the root, the generic metadata.
    assert type(Metadata(format="unknown")) is Metadata
    with pytest.raises(ValueError):
        LiteMetadata(format="test-dict")


def test_bids_and_scope_annotations() -> None:
    by_name = {f.name: f.metadata for f in fields(Metadata)}
    assert by_name["repetition_time"]["bids"] == "RepetitionTime"
    assert by_name["repetition_time"]["scope"] is Scope.ACQUISITION
    assert by_name["slice_timing"]["scope"] is Scope.SPATIAL
    assert by_name["channels"]["scope"] is Scope.AXIS
    assert by_name["channels"]["along"] is AxisType.channel
    assert by_name["bvalues"]["along"] is AxisType.time
    assert by_name["display_range"]["scope"] is Scope.FILE
    with pytest.raises(ValueError):
        Scoped(Scope.AXIS)
    with pytest.raises(ValueError):
        Along("space")
    assert "bids" not in by_name["display_range"]


# ----------------------------------------------------------------------
#   SNAPSHOT AND CHANGE-DETECTING WRITE (memo, section 6)
# ----------------------------------------------------------------------


def _read(record=None) -> DictMetadata:  # noqa: ANN001
    if record is None:
        record = {"desc": "short", "cal": (0.0, 1.0), "Key": "v"}
    return DictMetadata.from_raw(record)


def test_reading_decodes_and_snapshots() -> None:
    meta = _read()
    assert meta.description == "short"
    assert meta.display_range == (0.0, 1.0)
    assert meta.extra == {"Key": "v"}
    assert meta._changed_fields() == {}


def test_case1_untouched_writes_the_record_as_read() -> None:
    meta = _read()
    report = ConversionReport()
    record = meta.update_raw(on_loss=report)
    assert record == {"desc": "short", "cal": (0.0, 1.0), "Key": "v"}
    assert record is not meta.raw  # written over a copy
    assert not report.lossy


def test_case2_a_record_edit_survives() -> None:
    meta = _read()
    meta.raw["desc"] = "edited"
    assert meta.update_raw()["desc"] == "edited"


def test_case3_a_common_field_set_wins() -> None:
    meta = _read()
    meta.raw["desc"] = "edited"
    meta.description = "mine"
    assert meta._changed_fields() == {"description": "mine"}
    assert meta.update_raw()["desc"] == "mine"


def test_case4_none_clears_the_slot() -> None:
    meta = _read()
    meta.description = None
    meta.display_range = None
    assert meta._changed_fields() == {
        "description": None,
        "display_range": None,
    }
    record = meta.update_raw()
    assert "desc" not in record and "cal" not in record


def test_extra_is_compared_key_by_key() -> None:
    meta = _read({"desc": "d", "Kept": 1, "Gone": 2, "Edited": 3})
    meta.extra = {"Kept": 1, "Edited": 4, "New": 5}
    assert meta._changed_fields() == {
        "extra": {"Edited": 4, "New": 5, "Gone": None}
    }
    assert meta.update_raw() == {"desc": "d", "Kept": 1, "Edited": 4, "New": 5}


def test_a_snapshot_holds_converted_values() -> None:
    # A list decoded into a tuple field is not a change.
    meta = DictMetadata.from_raw({"slices": [0, 1, 2]})
    assert meta.slice_timing == (0.0, 1.0, 2.0)
    assert meta._changed_fields() == {}


def test_an_object_built_in_memory_has_everything_changed() -> None:
    meta = DictMetadata(description="d", display_range=(1, 2))
    assert meta._changed_fields() == {
        "description": "d",
        "display_range": (1.0, 2.0),
    }
    assert meta.update_raw() == {"desc": "d", "cal": (1.0, 2.0)}


def test_the_snapshot_survives_replace_copy_and_pickle() -> None:
    meta = _read()
    for other in (
        replace(meta, description="x"),
        copy.deepcopy(meta),
        pickle.loads(pickle.dumps(meta)),
    ):
        assert other._snapshot == meta._snapshot
    assert replace(meta, description="x")._changed_fields() == {
        "description": "x"
    }


def test_value_dependent_loss_is_reported_by_the_encoder() -> None:
    meta = DictMetadata(description="much too long", slice_timing=(0, 1))
    report = meta.check_writable(image=3)
    assert report.lost == {"slice_timing": (0.0, 1.0)}
    assert report.approximated == {"description": "truncated to 8"}
    assert not meta.check_writable(image=2).lost


def test_an_unsupported_field_assigned_later_is_reported_at_write() -> None:
    meta = LiteMetadata()
    meta.echo_time = 0.03  # attribute assignment is not validated
    report = meta.check_writable()
    assert report.lost == {"echo_time": 0.03}


# ----------------------------------------------------------------------
#   CONVERSION AND LOSS REPORTS (memo, section 7)
# ----------------------------------------------------------------------


def test_conversion_reports_what_the_target_cannot_hold() -> None:
    target, report = _to(_rich(), LiteMetadata)
    assert type(target) is LiteMetadata
    assert target.description == "a scan"
    assert target.slice_timing == (0.0, 0.5, 1.0)
    assert target.extra == {"Custom": 1}
    assert report.lost == {"echo_time": 0.03}
    assert (report.source, report.target) == ("generic", "test-lite")
    assert "echo_time" in str(report)


def test_conversion_accepts_a_format_name() -> None:
    target, _ = _to(_rich(), "test-lite")
    assert type(target) is LiteMetadata


def test_conversion_through_the_hub_loses_what_a_direct_one_does() -> None:
    lite = LiteMetadata(description="d", history=("h",), extra={"k": 1})
    hub, report = _to(lite, Metadata)
    assert not report.lossy
    # UNSUPPORTED on the source side reads as None.
    assert hub.echo_time is None
    _, direct = _to(lite, DictMetadata)
    _, via_hub = _to(hub, DictMetadata)
    assert direct.lost == via_hub.lost == {"history": ("h",)}


def test_the_record_travels_only_within_a_format() -> None:
    meta = _read()
    other, _ = _to(meta, Metadata)
    assert type(other) is Metadata
    # The hub carries the record and the snapshot...
    assert other.raw is meta.raw and other._snapshot == meta._snapshot
    # ... back to the format whose type of record it is, not to another.
    assert _to(other, DictMetadata)[0].raw is meta.raw
    assert _to(other, LiteMetadata)[0].raw is None
    same = DictMetadata.from_other(meta)
    assert same.raw is meta.raw and same._snapshot == meta._snapshot
    # A copy keeps the most specific class.
    assert type(FileBasedMetadata.from_other(meta)) is DictMetadata


def test_extra_is_lost_where_the_target_has_no_store() -> None:
    _, report = _to(Metadata(extra={"Key": 1}), OpaqueMetadata)
    assert report.lost == {"extra": {"Key": 1}}


def test_explicit_values_win_over_the_source() -> None:
    target, _ = _to(_rich(), Metadata, description="other")
    assert target.description == "other"


# ----------------------------------------------------------------------
#   POLICIES
# ----------------------------------------------------------------------


def test_the_default_policy_warns_once_per_conversion() -> None:
    source = Metadata(
        echo_time=0.03, flip_angle=90.0, magnetic_field_strength=3.0
    )
    with pytest.warns(MetadataLossWarning) as record:
        LiteMetadata.from_other(source)
    assert len(record) == 1
    report = record[0].message.report
    assert set(report.lost) == {
        "echo_time",
        "flip_angle",
        "magnetic_field_strength",
    }


def test_ignore_is_silent_and_raise_raises() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rich().to(LiteMetadata, on_loss="ignore")
        # Nothing lost: silent whatever the policy.
        Metadata(description="d").to(LiteMetadata, on_loss="raise")
    with pytest.raises(MetadataLossError) as info:
        _rich().to(LiteMetadata, on_loss="raise")
    assert info.value.report.lost == {"echo_time": 0.03}


def test_the_policy_context_manager_nests_and_restores() -> None:
    with metadata_loss_policy("raise"):
        with metadata_loss_policy("ignore"):
            LiteMetadata.from_other(_rich())
        with pytest.raises(MetadataLossError):
            LiteMetadata.from_other(_rich())
    with pytest.warns(MetadataLossWarning):
        LiteMetadata.from_other(_rich())
    with pytest.raises(ValueError):
        with metadata_loss_policy("loud"):
            pass


def test_the_policy_governs_implicit_field_conversions() -> None:
    class Holder(DataModelBase):
        meta: tx.Optional[LiteMetadata] = None

    with metadata_loss_policy("raise"):
        with pytest.raises(MetadataLossError):
            Holder(meta=_rich())
    with pytest.warns(MetadataLossWarning):
        held = Holder(meta=_rich())
    assert type(held.meta) is LiteMetadata


def test_a_report_raises_and_merges() -> None:
    report = ConversionReport(source="a", target="b")
    report.raise_if_lossy()
    other = ConversionReport(lost={"x": 1})
    report.merge(other)
    assert report.lost == {"x": 1}
    with pytest.raises(MetadataLossError):
        report.raise_if_lossy()
    with pytest.raises(MetadataLossError):
        apply_loss_policy(report, "raise")


# ----------------------------------------------------------------------
#   DERIVE (memo, section 9)
# ----------------------------------------------------------------------


def _scoped() -> Metadata:
    import datetime

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


def test_select_indexes_a_field_along_its_axis() -> None:
    # The hook of `image[index]`: the image resolves the kept positions.
    meta = _scoped()
    assert meta._select("time", [2, 0]).bvalues == (2000.0, 0.0)
    assert meta._select(AxisType.time, np.arange(1, 3)).bvalues == (
        1000.0,
        2000.0,
    )
    assert meta._select("time", np.array([], int)).bvalues == ()
    # A dropped axis, or a position beyond the field, clears it.
    assert meta._select("time", None).bvalues is None
    assert meta._select("time", [5]).bvalues is None
    assert meta._select("time", [-1]).bvalues is None
    # The other axes, and the other fields, are untouched.
    selected = meta._select("time", [2, 0], history="select")
    assert [c.name for c in selected.channels] == ["a", "b", "c"]
    assert selected.slice_timing == (0.0, 0.5)
    assert selected.display_range == (0.0, 1.0)
    assert selected.history == ("acquired", "select")
    assert selected.creation_time is None
    channel = meta._select(AxisType.channel, [2, 0])
    assert [c.name for c in channel.channels] == ["c", "a"]
    assert channel.bvalues == (0.0, 1000.0, 2000.0)
    with pytest.raises(ValueError, match="_reslice"):
        meta._select("space", [0])


def test_reslice_maps_directions_through_a_linear_map() -> None:
    # The hook of `image.reslice()`: the image computes the linear map
    # from the old voxel axes to the new ones.
    meta = Metadata(
        phase_encoding_direction="j-",
        slice_encoding_direction=EncodingDirection((0, 0, 1), space="mni"),
        slice_timing=(0.0, 0.5),
        echo_time=0.03,
        bvalues=(0, 1000),
    )
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    derived = meta._reslice(swap, history="reslice")
    assert derived.phase_encoding_direction == EncodingDirection("i-")
    # A direction in a world space does not move with the voxels.
    assert derived.slice_encoding_direction == EncodingDirection(
        (0, 0, 1), space="mni"
    )
    # The slice timing is cleared; the rest is kept.
    assert derived.slice_timing is None
    assert derived.echo_time == 0.03
    assert derived.bvalues == (0.0, 1000.0)
    assert derived.history == ("reslice",)
    flip = np.diag([1.0, -1.0, 1.0])
    assert meta._reslice(flip).phase_encoding_direction == EncodingDirection(
        "j"
    )
    # The map of a 4-D image: the direction lies in its first three axes.
    flip4 = np.diag([1.0, -1.0, 1.0, 1.0])
    assert meta._reslice(flip4).phase_encoding_direction == (
        EncodingDirection("j")
    )
    onto_time = np.eye(4)[[0, 3, 2, 1]]
    assert meta._reslice(onto_time).phase_encoding_direction is None
    rotated = meta._reslice(
        [[1, 0, 0], [0, 2**-0.5, -(2**-0.5)], [0, 2**-0.5, 2**-0.5]]
    )
    assert rotated.phase_encoding_direction.to_bids() is None
    # Without a map, a change of the spatial axes clears it.
    cleared = meta._reslice(None)
    assert cleared.phase_encoding_direction is None
    assert cleared.slice_encoding_direction.space == "mni"
    assert cleared.slice_timing is None


def test_reslice_keeps_the_record_and_clears_through_it() -> None:
    meta = DictMetadata.from_raw(
        {"desc": "d", "slices": [0, 1], "slice_hint": "x"}
    )
    derived = meta._reslice(None)
    assert type(derived) is DictMetadata
    assert derived._snapshot == meta._snapshot
    assert "slice_hint" not in derived.raw and "slice_hint" in meta.raw
    # The cleared field differs from the snapshot: it is cleared on write.
    assert derived._changed_fields()["slice_timing"] is None
    assert "slices" not in derived.update_raw()
    # Unsupported fields stay unsupported.
    assert derived.echo_time is UNSUPPORTED


def test_derive_copies_the_record_by_default() -> None:
    meta = LiteMetadata.from_raw({"key": [1]})
    derived = meta.derive()
    assert derived.raw == meta.raw and derived.raw is not meta.raw


# ----------------------------------------------------------------------
#   BIDS SIDECAR
# ----------------------------------------------------------------------


_SIDECAR = {
    "RepetitionTime": 2.0,
    "EchoTime": 0.03,
    "SliceTiming": [0.0, 0.5, 1.0, 1.5],
    "PhaseEncodingDirection": "j-",
    "Description": "a bold run",
    "Sources": ["bids:raw:sub-01_T1w.nii.gz"],
    "GeneratedBy": [{"Name": "fMRIPrep", "Version": "23.0"}],
    "TaskName": "rest",
    "DisplayRange": [0, 255],
}


def test_a_sidecar_round_trips() -> None:
    meta = Metadata.from_bids(_SIDECAR)
    assert meta.repetition_time == 2.0
    assert meta.slice_timing == (0.0, 0.5, 1.0, 1.5)
    assert meta.generated_by == (GeneratedBy(name="fMRIPrep", version="23.0"),)
    assert meta.display_range == (0.0, 255.0)
    # A key that names no vocabulary field lands in `extra`.
    assert meta.extra == {"TaskName": "rest"}
    sidecar = meta.to_bids()
    assert sidecar == {
        **_SIDECAR,
        "DisplayRange": [0.0, 255.0],
    }
    json.dumps(sidecar)


def test_a_sidecar_is_read_from_a_path_or_a_string(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "sub-01_bold.json"
    path.write_text(json.dumps(_SIDECAR))
    assert Metadata.from_bids(path) == Metadata.from_bids(_SIDECAR)
    assert Metadata.from_bids(str(path)) == Metadata.from_bids(_SIDECAR)
    assert Metadata.from_bids(json.dumps(_SIDECAR)) == Metadata.from_bids(
        _SIDECAR
    )


def test_diffusion_is_not_a_sidecar_key() -> None:
    meta = Metadata(bvalues=(0, 1000), description="dwi")
    with pytest.warns(MetadataLossWarning):
        sidecar = meta.to_bids()
    assert sidecar == {"Description": "dwi"}
    with pytest.raises(MetadataLossError):
        meta.to_bids(on_loss="raise")


def test_times_are_iso_strings() -> None:
    meta = Metadata.from_bids({"AcquisitionTime": "2020-01-02T03:04:05"})
    assert meta.acquisition_time.year == 2020
    assert meta.to_bids() == {"AcquisitionTime": "2020-01-02T03:04:05"}


# ----------------------------------------------------------------------
#   THE DATA MODEL FIELD
# ----------------------------------------------------------------------


def test_the_roots_carry_an_optional_keyword_only_field() -> None:
    image = SingleScaleImage(np.zeros((2, 3)))
    assert image.metadata is None
    signature = inspect.signature(SingleScaleImage)
    assert list(signature.parameters)[:2] == ["data", "transformations"]
    assert (
        signature.parameters["metadata"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    affine = Affine(np.eye(4)[:3])
    assert affine.metadata is None
    assert list(inspect.signature(Translation).parameters)[-1] == "metadata"


def test_the_field_converts_to_generic_metadata() -> None:
    image = SingleScaleImage(
        np.zeros((2, 3)), metadata={"description": "from a dict"}
    )
    assert type(image.metadata) is Metadata
    assert image.metadata.description == "from a dict"
    lite = LiteMetadata(description="d")
    affine = Affine(np.eye(4)[:3], metadata=lite)
    assert type(affine.metadata) is Metadata
    assert affine.metadata.description == "d"


def test_the_field_is_out_of_eq_and_repr() -> None:
    a = SingleScaleImage(np.zeros(2), metadata=Metadata(description="a"))
    b = SingleScaleImage(np.zeros(2), metadata=Metadata(description="b"))
    assert "metadata" not in repr(a)
    x = Affine(np.eye(3)[:2], metadata=Metadata(description="a"))
    assert "metadata" not in repr(x)
    for cls in (SingleScaleImage, Affine):
        field = next(f for f in fields(cls) if f.name == "metadata")
        assert not field.eq and not field.repr
    assert a.metadata != b.metadata


def test_replace_carries_the_metadata() -> None:
    affine = Affine(np.eye(3)[:2], metadata=Metadata(description="a"))
    assert replace(affine, data=2 * np.eye(3)[:2]).metadata.description == (
        "a"
    )


def test_replace_and_from_other_copy_the_metadata() -> None:
    image = SingleScaleImage(
        np.zeros((2, 3)), metadata=Metadata(description="a", extra={"k": 1})
    )
    copied = replace(image, data=np.ones((2, 3)))
    assert copied.metadata is not image.metadata
    copied.metadata.description = "edited copy"
    copied.metadata.extra["k"] = 2
    assert image.metadata.description == "a"
    assert image.metadata.extra == {"k": 1}
    other = SingleScaleImage.from_other(image)
    assert other.metadata is not image.metadata
    assert other.metadata == image.metadata
    given = Metadata(description="given")
    affine = Affine(np.eye(3)[:2], metadata=given)
    assert affine.metadata is not given


def test_a_copy_shares_the_record_and_copies_the_snapshot() -> None:
    meta = DictMetadata.from_raw({"desc": "read", "Lab": "x"})
    other = meta.copy()
    assert other.raw is meta.raw
    assert other == meta
    other.description = "changed"
    other._snapshot["description"] = "forged"
    assert meta._changed_fields() == {}
    assert meta._snapshot["description"] == "read"


def test_the_hub_holds_any_record_and_opaque_none() -> None:
    assert Metadata().raw is None
    assert Metadata(raw={"a": 1}).raw == {"a": 1}
    assert OpaqueMetadata().raw is None
    with pytest.raises(TypeError):
        OpaqueMetadata(raw={"a": 1})


def test_a_decoder_may_not_return_an_unsupported_field() -> None:
    class Wrong(
        FileBasedMetadata,
        on={"format": "test-wrong"},
        supports=("description",),
    ):
        @classmethod
        def _decode_raw(cls, raw, *, image=None) -> dict:  # noqa: ANN001
            return {"description": "d", "echo_time": 0.03}

    with pytest.raises(TypeError, match="echo_time"):
        Wrong.from_raw({})


# ----------------------------------------------------------------------
#   A NEW RECORD, FORCED FIELDS, DERIVED FIELDS
# ----------------------------------------------------------------------


def test_update_from_raw_keeps_the_changes_over_a_new_record() -> None:
    meta = DictMetadata.from_raw({"desc": "old", "cal": (0, 1), "A": 1})
    meta.description = "mine"
    meta.display_range = None
    meta.extra = {"A": 1, "B": 2}
    new = meta.update_from_raw({"desc": "new", "cal": (2, 3), "C": 3})
    assert new.description == "mine"
    assert new.display_range is None
    assert new.extra == {"C": 3, "B": 2}
    assert new._changed_fields() == {
        "description": "mine",
        "display_range": None,
        "extra": {"B": 2},
    }


def test_force_writes_a_field_equal_to_the_snapshot() -> None:
    meta = DictMetadata.from_raw({"desc": "read"})
    meta.raw["desc"] = "record edit"
    assert meta.update_raw(dict(meta.raw))["desc"] == "record edit"
    forced = meta.update_raw(dict(meta.raw), force=("description",))
    assert forced["desc"] == "read"
    meta.description = None
    assert "desc" not in meta.update_raw(
        dict(meta.raw), force=("description",)
    )


class GeoMetadata(
    FileBasedMetadata[dict],
    on={"format": "test-geo"},
    supports=("repetition_time",),
):
    """A format whose `repetition_time` is the image's time step (the
    image is a number here), or the record's when it has none."""

    @classmethod
    def _decode_raw(cls, raw, *, image=None) -> dict:  # noqa: ANN001
        if image is not None:
            return {"repetition_time": image}
        return {"repetition_time": (raw or {}).get("tr")}

    def _encode_raw(self, raw, changed, *, image=None, report) -> dict:  # noqa: ANN001
        # The time step of an image is the writer's to store.
        if "repetition_time" in changed and image is None:
            raw["tr"] = changed["repetition_time"]
        return raw


def test_a_field_the_record_does_not_hold_is_reported() -> None:
    meta = GeoMetadata(repetition_time=2.0)
    report = ConversionReport()
    raw = meta.update_raw(image=2.0, on_loss=report)
    assert raw == {}
    meta.check_raw(raw, image=2.0, on_loss=report)
    assert not report.lossy
    report = ConversionReport()
    raw = meta.update_raw(image=1.5, on_loss=report)
    assert raw == {}
    meta.check_raw(raw, image=1.5, on_loss=report)
    assert set(report.approximated) == {"repetition_time"}
    assert "1.5" in report.approximated["repetition_time"]
    # The data model says nothing: the value is the format's to write.
    raw = meta.update_raw(image=None)
    assert raw == {"tr": 2.0}
    assert not meta.check_raw(raw, on_loss="ignore").lossy
    assert meta.check_writable(image=1.5).approximated
    assert not meta.check_writable(image=2.0).lossy
    # Numbers, and sequences of numbers, agree within rounding.
    assert not meta.check_raw({"tr": 2.0 + 1e-9}, on_loss="ignore").lossy


# ----------------------------------------------------------------------
#   ONE WARNING
# ----------------------------------------------------------------------


def test_collected_reports_merge_into_one() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with collect_loss_reports() as reports:
            apply_loss_policy(ConversionReport(source="a", lost={"x": 1}))
            apply_loss_policy(
                ConversionReport(target="b", approximated={"y": "z"})
            )
            apply_loss_policy(ConversionReport(lost={"q": 1}), "ignore")
    assert len(reports) == 2
    report = ConversionReport.merged(reports)
    assert (report.source, report.target) == ("a", "b")
    assert report.lost == {"x": 1} and report.approximated == {"y": "z"}
    with pytest.raises(MetadataLossError):
        with collect_loss_reports():
            apply_loss_policy(ConversionReport(lost={"x": 1}), "raise")


def test_from_other_carries_the_metadata_of_another_family() -> None:
    affine = Affine(np.eye(4)[:3], metadata=Metadata(description="mine"))

    class Holder(DataModelBase):
        inner: tx.Any = None
        metadata: tx.Optional[Metadata] = None

    held = Holder.from_other(affine)
    assert held.inner is affine
    assert held.metadata.description == "mine"
    assert Holder.from_other(affine, metadata=None).metadata is None


# ----------------------------------------------------------------------
#   HIERARCHY AND VOCABULARY GROUPS
# ----------------------------------------------------------------------


def test_the_hierarchy_mirrors_the_images() -> None:
    assert issubclass(FileBasedMetadata, Metadata)
    assert issubclass(OpaqueMetadata, FileBasedMetadata)
    assert issubclass(LiteMetadata, FileBasedMetadata)
    for group in GROUPS:
        assert issubclass(Metadata, group)
    # Every class has a raw record and a snapshot; reading and writing
    # them is what a file-based class adds.
    names = {f.name for f in fields(Metadata)}
    assert {"raw", "_snapshot"} <= names
    assert not hasattr(Metadata, "from_raw")


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
    assert VOCABULARY == sum(GROUPS.values(), ())
    assert VOCABULARY[:2] == ("name", "description")
    assert GROUPS[DiffusionVocabulary] == ("bvalues", "bvectors")
    assert "data_type" in GROUPS[StorageVocabulary]
    assert "extra" not in VOCABULARY
    assert len(set(VOCABULARY)) == len(VOCABULARY)
    # A group's fields convert, as the class that inherits them does.
    assert Metadata(echo_time="0.03", bvalues=[0, 1000]).bvalues == (
        0.0,
        1000.0,
    )


def test_supports_takes_groups() -> None:
    class ByGroup(
        FileBasedMetadata,
        on={"format": "test-group"},
        supports=(ProvenanceVocabulary, "echo_time"),
    ):
        pass

    assert ByGroup.supported_fields == set(
        GROUPS[ProvenanceVocabulary] + ("echo_time",)
    )
    assert ByGroup.supports("history") and not ByGroup.supports("bvalues")
    with pytest.raises(TypeError, match="not a vocabulary group"):

        class NotAGroup(FileBasedMetadata, supports=(Channel,)):
            pass


def test_a_new_field_is_unsupported_until_a_format_opts_in() -> None:
    # `supports=` lists what a format stores: everything else, including
    # what the vocabulary gains later, defaults to UNSUPPORTED.
    assert LiteMetadata.supported_fields <= set(VOCABULARY) | {"extra"}
    assert "data_type" in LiteMetadata.unsupported_fields


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
    assert Metadata(intent=UNSUPPORTED).intent is UNSUPPORTED
    with pytest.raises((TypeError, ValueError)):
        Metadata(space=3)
    assert json.dumps(Metadata(space="mni").to_bids()) == (
        '{"SpatialReference": "mni"}'
    )


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
    # Written as its symbol, which parses back to the same unit.
    assert meta.to_bids() == {"DataUnit": "ms"}
    for name in ("a.u.", "mm/s", "degC", "uV", "HU"):
        bids = Metadata(data_unit=name).to_bids()
        assert Metadata.from_bids(bids).data_unit == Unit(name)
    assert Metadata(data_unit="mm2/s").to_bids() == {"DataUnit": "mm2/s"}


def test_data_type_is_a_native_dtype() -> None:
    meta = Metadata(data_type=">i2")
    assert meta.data_type == np.dtype("int16")
    assert meta.data_type.isnative
    assert meta.to_bids() == {"DataType": "int16"}
    assert Metadata.from_bids({"DataType": "uint8"}).data_type == np.uint8
    # A resampling changes the kind of the values: it is grid-bound.
    # How the file stores the values: kept by `derive` (`file` scope).
    assert meta._reslice(None).data_type == np.int16
    assert meta._select("time", [0]).data_type == np.int16


def test_preferred_dtype() -> None:
    labels = Metadata(data_type="uint8")
    # The data type wins when the values are of its kind...
    assert preferred_dtype(labels, np.int64) == np.uint8
    assert preferred_dtype(labels, np.bool_) == np.uint8
    assert preferred_dtype(Metadata(data_type="f4"), np.float64) == np.float32
    # ... an explicit dtype wins over it ...
    assert preferred_dtype(labels, np.int64, "int16") == np.int16
    # ... and floats are never quantised into it, nor integers made floats.
    report = ConversionReport()
    assert preferred_dtype(labels, np.float64, on_loss=report) == np.float64
    assert "data_type" in report.approximated
    with pytest.warns(MetadataLossWarning):
        # With no report, the policy in effect.
        assert preferred_dtype(Metadata(data_type="f4"), np.int16) == np.int16
    assert preferred_dtype(Metadata(), np.int16) == np.int16
    # A data type that was only read is dropped silently.

    class Typed(
        FileBasedMetadata, on={"format": "test-typed"}, supports=("data_type",)
    ):
        @classmethod
        def _decode_raw(cls, raw, *, image=None) -> dict:  # noqa: ANN001
            return {"data_type": raw}

    read = Typed.from_raw("uint8")
    assert read.data_type == np.uint8 and not read._changed_fields()
    report = ConversionReport()
    preferred_dtype(read, np.float64, on_loss=report)
    assert not report.lossy


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


def test_the_direction_fields_take_bids_strings() -> None:
    meta = Metadata(phase_encoding_direction="j-")
    assert isinstance(meta.phase_encoding_direction, EncodingDirection)
    assert meta.phase_encoding_direction.to_bids() == "j-"
    meta.slice_encoding_direction = {"Vector": [0, 0, 1]}
    assert meta.slice_encoding_direction == EncodingDirection("k")
    assert meta.to_bids() == {
        "PhaseEncodingDirection": "j-",
        "SliceEncodingDirection": "k",
    }
    assert Metadata.from_bids(meta.to_bids()) == meta


def test_an_oblique_direction_is_lost_in_a_sidecar() -> None:
    meta = Metadata(phase_encoding_direction=(1, 1, 0))
    with pytest.raises(MetadataLossError) as info:
        meta.to_bids(on_loss="raise")
    assert set(info.value.report.lost) == {"phase_encoding_direction"}


# ----------------------------------------------------------------------
#   to()
# ----------------------------------------------------------------------


def test_to_converts_and_reports() -> None:
    lite = _rich().to(LiteMetadata, on_loss="ignore")
    assert type(lite) is LiteMetadata
    assert lite.description == "a scan"
    report = ConversionReport()
    _rich().to("test-lite", on_loss=report)
    assert report.lost == {"echo_time": 0.03}
    assert (report.source, report.target) == ("generic", "test-lite")
    with pytest.raises(MetadataLossError):
        _rich().to(LiteMetadata, on_loss="raise")
    with pytest.warns(MetadataLossWarning):
        _rich().to(LiteMetadata)


def test_a_report_as_on_loss_is_filled_silently() -> None:
    report = ConversionReport()
    with warnings.catch_warnings(), metadata_loss_policy("raise"):
        warnings.simplefilter("error")
        _rich().to(LiteMetadata, on_loss=report)
        # A second conversion adds to the same report.
        Metadata(extra={"Key": 1}).to(OpaqueMetadata, on_loss=report)
    assert report.lost == {"echo_time": 0.03, "extra": {"Key": 1}}
    assert (report.source, report.target) == ("generic", "test-lite")
    # `report=` is gone: it would be a field, which there is not.
    with pytest.raises(TypeError):
        _rich().to(LiteMetadata, report=ConversionReport())


def test_update_raw_applies_the_policy_unless_given_a_report() -> None:
    meta = _read()
    meta.echo_time = 0.03  # not supported: lost on write
    with pytest.warns(MetadataLossWarning):
        meta.update_raw()
    with pytest.raises(MetadataLossError):
        meta.update_raw(on_loss="raise")
    report = ConversionReport()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        meta.update_raw(on_loss=report)
    assert report.lost == {"echo_time": 0.03}
    with pytest.raises(ValueError):
        _rich().to("no-such-format")


def test_to_none_keeps_the_class() -> None:
    meta = _read()
    same = meta.to(description="other")
    assert type(same) is DictMetadata
    assert same.raw is meta.raw
    assert same._changed_fields() == {"description": "other"}
    generic = Metadata(description="d").to()
    assert type(generic) is Metadata and generic.description == "d"


def test_a_format_class_given_to_the_generic_field_is_converted() -> None:
    meta = _read()
    image = SingleScaleImage(np.zeros((2, 3)), metadata=meta)
    assert type(image.metadata) is Metadata
    assert image.metadata.description == "short"


def test_metadata_field_is_an_annotation() -> None:
    class Holder(Magic):
        meta: MetadataField[LiteMetadata, Factory(), tx.Doc("Some metadata.")]

    field = next(f for f in fields(Holder) if f.name == "meta")
    assert not field.repr and not field.eq and field.kw
    assert field.doc == "Some metadata."
    assert type(Holder().meta) is LiteMetadata
    given = LiteMetadata(description="d")
    held = Holder(meta=given)
    assert held.meta is not given and held.meta == given
    # Converted even though `Holder` does not convert its fields.
    with metadata_loss_policy("ignore"):
        held.meta = _rich()
    assert type(held.meta) is LiteMetadata


def test_lazy_is_not_a_class_keyword() -> None:
    with pytest.raises(TypeError, match="unknown class keyword"):

        class Lazy(
            FileBasedMetadata, supports=("history",), lazy=("history",)
        ):
            pass


def test_the_pinned_format_narrows_the_field() -> None:
    # `on={"format": ...}` alone gives the field its literal type and its
    # default, and refuses any other format.
    field = next(f for f in fields(LiteMetadata) if f.name == "format")
    assert field.default == "test-lite"
    assert tx.get_args(field.type) == ("test-lite",)
    assert LiteMetadata().format == "test-lite"
    with pytest.raises(Exception, match="test-lite"):
        LiteMetadata(format="test-dict")
    assert type(Metadata(format="test-lite")) is LiteMetadata


def test_a_channel_color_is_held_as_rgba() -> None:
    # One spelling per color, so that a record read back agrees with it.
    assert Channel(color="#0000ff").color == "0000FFFF"
    assert Channel(color="00FF0080").color == "00FF0080"
    assert Channel().color is None
