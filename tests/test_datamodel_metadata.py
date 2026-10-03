"""
Tests for the format-agnostic metadata framework
(`brainhops.datamodel.metadata`), on synthetic formats.

What is checked: the `UNSUPPORTED` sentinel; the `supports=`/`derived=`
class keywords; the read-time snapshot and the change-detecting write
(cases 1-4 of section 6 of the design memo); conversion loss reports and
the loss policies; `derive`; the BIDS sidecar codec; and the `metadata`
field of the data model roots.
"""

import copy
import inspect
import json
import pickle
import warnings

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import fields, replace

from brainhops.datamodel import (
    UNSUPPORTED,
    ConversionReport,
    FormatMetadata,
    GeneratedBy,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    OpaqueMetadata,
    Unsupported,
    metadata_loss_policy,
)
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import (
    ALL,
    Channel,
    Lazy,
    apply_loss_policy,
    convert,
    get_metadata_loss_policy,
    one_loss_warning,
)
from brainhops.datamodel.transformations import Affine, Translation

# ----------------------------------------------------------------------
#   SYNTHETIC FORMATS
# ----------------------------------------------------------------------


class LiteMetadata(
    FormatMetadata,
    on={"format": "test-lite"},
    supports=("description", "slice_timing", "history", "extra"),
    derived=("slice_timing",),
):
    """A format that stores three fields and free-form keys."""

    format: tx.Literal["test-lite"] = "test-lite"


class DictMetadata(
    FormatMetadata,
    on={"format": "test-dict"},
    supports=("description", "display_range", "slice_timing", "extra"),
):
    """A format whose record is a dict: `desc`, `cal`, `slices`, and any
    other key is free-form."""

    format: tx.Literal["test-dict"] = "test-dict"

    _KNOWN = {"desc": "description", "cal": "display_range"}

    @classmethod
    def _default_raw(cls) -> dict:
        return {}

    @classmethod
    def _decode(cls, raw, *, image=None) -> dict:  # noqa: ANN001
        out = {name: raw.get(key) for key, name in cls._KNOWN.items()}
        out["slice_timing"] = raw.get("slices")
        out["extra"] = {
            k: v
            for k, v in raw.items()
            if k not in cls._KNOWN and k != "slices"
        }
        return out

    def _encode(self, raw, changed, *, image=None, report) -> dict:  # noqa: ANN001
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

    def _derive_raw(self, raw, *, grid_changed, volumes) -> dict:  # noqa: ANN001
        if raw is None or not grid_changed:
            return raw
        raw = dict(raw)
        raw.pop("slice_hint", None)
        return raw


class KeyvalMetadata(
    FormatMetadata,
    on={"format": "test-keyval"},
    supports=("description", "extra"),
):
    """A key/value format: what it has no slot for goes into `extra`."""

    format: tx.Literal["test-keyval"] = "test-keyval"

    @classmethod
    def _import(cls, other, values, *, report) -> None:  # noqa: ANN001
        extra = dict(values.get("extra") or {})
        for name in list(report.lost):
            if name == "extra":
                continue
            extra[name] = report.lost.pop(name)
            report.passed_through += (name,)
        values["extra"] = extra


class DialectMetadata(
    FormatMetadata,
    on={"format": "test-dialect"},
    supports=("description", "channels"),
):
    """A format whose `channels` support depends on the instance."""

    format: tx.Literal["test-dialect"] = "test-dialect"
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
#   supports= / derived=
# ----------------------------------------------------------------------


def test_supports_lists_what_a_format_stores() -> None:
    vocabulary = set(FormatMetadata.vocabulary_fields) | {"extra"}
    assert LiteMetadata.unsupported_fields == vocabulary - {
        "description",
        "slice_timing",
        "history",
        "extra",
    }
    assert LiteMetadata.derived_fields == {"slice_timing"}
    assert Metadata.unsupported_fields == frozenset()
    assert OpaqueMetadata.unsupported_fields == vocabulary
    for name in LiteMetadata.unsupported_fields:
        field = next(f for f in fields(LiteMetadata) if f.name == name)
        assert field.default is UNSUPPORTED


def test_supports_on_the_class_and_on_an_instance() -> None:
    assert LiteMetadata.supports("description")
    assert not LiteMetadata.supports("echo_time")
    meta = LiteMetadata()
    assert meta.supports("description")
    assert not meta.supports("echo_time")
    with pytest.raises(KeyError):
        LiteMetadata.supports("not_a_field")


def test_per_instance_capability() -> None:
    assert DialectMetadata.supports("channels")
    assert DialectMetadata().supports("channels")
    plain = DialectMetadata(dialect="plain")
    assert not plain.supports("channels")
    # `replace` runs `__post_init__` again, so the rule holds.
    assert not replace(DialectMetadata(), dialect="plain").supports("channels")


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
    # `derived=` is narrowed to what the subclass supports.
    assert Widens.derived_fields == frozenset()


def test_a_field_may_be_declared_unsupported_by_hand() -> None:
    class ByHand(Metadata):
        echo_time: tx.Optional[float] = UNSUPPORTED

    assert ByHand.unsupported_fields == {"echo_time"}


def test_supports_all() -> None:
    class Everything(FormatMetadata, supports=ALL):
        pass

    assert Everything.unsupported_fields == frozenset()


def test_wrong_declarations_are_refused() -> None:
    with pytest.raises(TypeError, match="not vocabulary"):

        class Typo(FormatMetadata, supports=("descr",)):
            pass

    with pytest.raises(TypeError, match="derived"):

        class NotStored(
            FormatMetadata, supports=("description",), derived=("echo_time",)
        ):
            pass


def test_polymorphic_construction_on_format() -> None:
    meta = FormatMetadata(format="test-lite", description="x")
    assert type(meta) is LiteMetadata
    assert type(FormatMetadata(description="x")) is Metadata
    assert type(FormatMetadata(format="opaque")) is OpaqueMetadata
    with pytest.raises(ValueError):
        LiteMetadata(format="test-dict")


def test_bids_and_scope_annotations() -> None:
    by_name = {f.name: f.metadata for f in fields(FormatMetadata)}
    assert by_name["repetition_time"]["bids"] == "RepetitionTime"
    assert by_name["repetition_time"]["scope"] == "acquisition"
    assert by_name["slice_timing"]["scope"] == "grid"
    assert by_name["channels"]["scope"] == "volume"
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
    assert meta.changed_fields() == {}


def test_case1_untouched_writes_the_record_as_read() -> None:
    meta = _read()
    report = ConversionReport()
    record = meta.write_raw(report=report)
    assert record == {"desc": "short", "cal": (0.0, 1.0), "Key": "v"}
    assert record is not meta.raw  # written over a copy
    assert not report.lossy


def test_case2_a_record_edit_survives() -> None:
    meta = _read()
    meta.raw["desc"] = "edited"
    assert meta.write_raw()["desc"] == "edited"


def test_case3_a_common_field_set_wins() -> None:
    meta = _read()
    meta.raw["desc"] = "edited"
    meta.description = "mine"
    assert meta.changed_fields() == {"description": "mine"}
    assert meta.write_raw()["desc"] == "mine"


def test_case4_none_clears_the_slot() -> None:
    meta = _read()
    meta.description = None
    meta.display_range = None
    assert meta.changed_fields() == {
        "description": None,
        "display_range": None,
    }
    record = meta.write_raw()
    assert "desc" not in record and "cal" not in record


def test_extra_is_compared_key_by_key() -> None:
    meta = _read({"desc": "d", "Kept": 1, "Gone": 2, "Edited": 3})
    meta.extra = {"Kept": 1, "Edited": 4, "New": 5}
    assert meta.changed_fields() == {
        "extra": {"Edited": 4, "New": 5, "Gone": None}
    }
    assert meta.write_raw() == {"desc": "d", "Kept": 1, "Edited": 4, "New": 5}


def test_a_snapshot_holds_converted_values() -> None:
    # A list decoded into a tuple field is not a change.
    meta = DictMetadata.from_raw({"slices": [0, 1, 2]})
    assert meta.slice_timing == (0.0, 1.0, 2.0)
    assert meta.changed_fields() == {}


def test_an_object_built_in_memory_has_everything_changed() -> None:
    meta = DictMetadata(description="d", display_range=(1, 2))
    assert meta.changed_fields() == {
        "description": "d",
        "display_range": (1.0, 2.0),
    }
    assert meta.write_raw() == {"desc": "d", "cal": (1.0, 2.0)}


def test_the_snapshot_survives_replace_copy_and_pickle() -> None:
    meta = _read()
    for other in (
        replace(meta, description="x"),
        copy.deepcopy(meta),
        pickle.loads(pickle.dumps(meta)),
    ):
        assert other._decoded == meta._decoded
    assert replace(meta, description="x").changed_fields() == {
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
    target, report = convert(_rich(), LiteMetadata, on_loss="ignore")
    assert type(target) is LiteMetadata
    assert target.description == "a scan"
    assert target.slice_timing == (0.0, 0.5, 1.0)
    assert target.extra == {"Custom": 1}
    assert report.lost == {"echo_time": 0.03}
    assert (report.source, report.target) == ("generic", "test-lite")
    assert "echo_time" in str(report)


def test_conversion_accepts_a_format_name() -> None:
    target, _ = convert(_rich(), "test-lite", on_loss="ignore")
    assert type(target) is LiteMetadata


def test_conversion_through_the_hub_loses_what_a_direct_one_does() -> None:
    lite = LiteMetadata(description="d", history=("h",), extra={"k": 1})
    hub, report = convert(lite, Metadata)
    assert not report.lossy
    # UNSUPPORTED on the source side reads as None.
    assert hub.echo_time is None
    _, direct = convert(lite, DictMetadata, on_loss="ignore")
    _, via_hub = convert(hub, DictMetadata, on_loss="ignore")
    assert direct.lost == via_hub.lost == {"history": ("h",)}


def test_the_record_travels_only_within_a_format() -> None:
    meta = _read()
    other, _ = convert(meta, Metadata)
    assert other.raw is None and other._decoded == {}
    same = DictMetadata.from_other(meta)
    assert same.raw is meta.raw and same._decoded == meta._decoded
    # A copy keeps the most specific class.
    assert type(FormatMetadata.from_other(meta)) is DictMetadata


def test_extra_is_lost_where_the_target_has_no_store() -> None:
    _, report = convert(
        Metadata(extra={"Key": 1}), OpaqueMetadata, on_loss="ignore"
    )
    assert report.lost == {"extra": {"Key": 1}}


def test_import_may_recover_a_loss() -> None:
    target, report = convert(_rich(), KeyvalMetadata)
    assert not report.lossy
    assert target.extra == {
        "Custom": 1,
        "echo_time": 0.03,
        "slice_timing": (0.0, 0.5, 1.0),
        "history": ("acquired",),
    }
    assert set(report.passed_through) == {
        "echo_time",
        "slice_timing",
        "history",
    }


def test_explicit_values_win_over_the_source() -> None:
    target, _ = convert(_rich(), Metadata, description="other")
    assert target.description == "other"


# ----------------------------------------------------------------------
#   POLICIES
# ----------------------------------------------------------------------


def test_the_default_policy_warns_once_per_conversion() -> None:
    assert get_metadata_loss_policy() == "warn"
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
        convert(_rich(), LiteMetadata, on_loss="ignore")
        # Nothing lost: silent whatever the policy.
        convert(Metadata(description="d"), LiteMetadata, on_loss="raise")
    with pytest.raises(MetadataLossError) as info:
        convert(_rich(), LiteMetadata, on_loss="raise")
    assert info.value.report.lost == {"echo_time": 0.03}


def test_the_policy_context_manager_nests_and_restores() -> None:
    with metadata_loss_policy("raise"):
        assert get_metadata_loss_policy() == "raise"
        with metadata_loss_policy("ignore"):
            LiteMetadata.from_other(_rich())
        with pytest.raises(MetadataLossError):
            LiteMetadata.from_other(_rich())
    assert get_metadata_loss_policy() == "warn"
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
    other = ConversionReport(lost={"x": 1}, passed_through=("y",))
    report.merge(other)
    assert report.lost == {"x": 1} and report.passed_through == ("y",)
    with pytest.raises(MetadataLossError):
        report.raise_if_lossy()
    with pytest.raises(MetadataLossError):
        apply_loss_policy(report, "raise")


# ----------------------------------------------------------------------
#   DERIVE (memo, section 9)
# ----------------------------------------------------------------------


def test_derive_follows_the_scopes() -> None:
    import datetime

    meta = Metadata(
        description="d",
        creation_time=datetime.datetime(2020, 1, 1),
        history=("acquired",),
        echo_time=0.03,
        slice_timing=(0.0, 0.5),
        phase_encoding_direction="j-",
        channels=(Channel(name="a"), Channel(name="b"), Channel(name="c")),
        diffusion_bvalues=(0, 1000, 2000),
        display_range=(0, 1),
        extra={"Key": 1},
    )
    same_grid = meta.derive(step="smooth")
    assert same_grid is not meta
    assert same_grid.description == "d"
    assert same_grid.creation_time is None
    assert same_grid.history == ("acquired", "smooth")
    assert same_grid.echo_time == 0.03
    assert same_grid.slice_timing == (0.0, 0.5)
    assert same_grid.extra == {"Key": 1}
    assert same_grid.extra is not meta.extra
    assert [g.name for g in same_grid.generated_by] == ["brainhops"]
    # The brainhops entry is added once.
    assert len(same_grid.derive().generated_by) == 1

    resampled = meta.derive(grid_changed=True)
    assert resampled.slice_timing is None
    assert resampled.phase_encoding_direction is None
    assert resampled.echo_time == 0.03

    selected = meta.derive(volumes=[2, 0])
    assert [c.name for c in selected.channels] == ["c", "a"]
    assert selected.diffusion_bvalues == (2000.0, 0.0)
    assert selected.display_range == (0.0, 1.0)

    changed = meta.derive(volumes_changed=True)
    assert changed.channels is None and changed.diffusion_bvalues is None


def test_derive_keeps_the_record_and_clears_through_it() -> None:
    meta = DictMetadata.from_raw(
        {"desc": "d", "slices": [0, 1], "slice_hint": "x"}
    )
    derived = meta.derive(grid_changed=True)
    assert type(derived) is DictMetadata
    assert derived._decoded == meta._decoded
    assert "slice_hint" not in derived.raw and "slice_hint" in meta.raw
    # The cleared field differs from the snapshot: it is cleared on write.
    assert derived.changed_fields()["slice_timing"] is None
    assert "slices" not in derived.write_raw()
    # Unsupported fields stay unsupported.
    assert derived.echo_time is UNSUPPORTED


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
    meta = Metadata(diffusion_bvalues=(0, 1000), description="dwi")
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
    assert replace(affine, matrix=2 * np.eye(3)[:2]).metadata.description == (
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
    other._decoded["description"] = "forged"
    assert meta.changed_fields() == {}
    assert meta._decoded["description"] == "read"


def test_the_hub_and_opaque_have_no_record() -> None:
    for cls in (Metadata, OpaqueMetadata):
        assert cls().raw is None
        with pytest.raises(TypeError):
            cls(raw={"a": 1})


def test_a_decoder_may_not_return_an_unsupported_field() -> None:
    class Wrong(
        FormatMetadata, on={"format": "test-wrong"}, supports=("description",)
    ):
        format: tx.Literal["test-wrong"] = "test-wrong"

        @classmethod
        def _decode(cls, raw, *, image=None) -> dict:  # noqa: ANN001
            return {"description": "d", "echo_time": 0.03}

    with pytest.raises(TypeError, match="echo_time"):
        Wrong.from_raw({})


# ----------------------------------------------------------------------
#   A NEW RECORD, FORCED FIELDS, DERIVED FIELDS
# ----------------------------------------------------------------------


def test_with_record_keeps_the_changes_over_a_new_record() -> None:
    meta = DictMetadata.from_raw({"desc": "old", "cal": (0, 1), "A": 1})
    meta.description = "mine"
    meta.display_range = None
    meta.extra = {"A": 1, "B": 2}
    new = meta.with_record({"desc": "new", "cal": (2, 3), "C": 3})
    assert new.description == "mine"
    assert new.display_range is None
    assert new.extra == {"C": 3, "B": 2}
    assert new.changed_fields() == {
        "description": "mine",
        "display_range": None,
        "extra": {"B": 2},
    }


def test_force_writes_a_field_equal_to_the_snapshot() -> None:
    meta = DictMetadata.from_raw({"desc": "read"})
    meta.raw["desc"] = "record edit"
    assert meta.write_raw(dict(meta.raw))["desc"] == "record edit"
    forced = meta.write_raw(dict(meta.raw), force=("description",))
    assert forced["desc"] == "read"
    meta.description = None
    assert "desc" not in meta.write_raw(dict(meta.raw), force=("description",))


class GeoMetadata(
    FormatMetadata,
    on={"format": "test-geo"},
    supports=("repetition_time",),
    derived=("repetition_time",),
):
    """A format whose `repetition_time` is the image's time step (the
    image is a number here), or the record's when it has none."""

    format: tx.Literal["test-geo"] = "test-geo"

    @classmethod
    def _default_raw(cls) -> dict:
        return {}

    def _geometry(self, image) -> dict:  # noqa: ANN001
        return {"repetition_time": image}

    def _encode(self, raw, changed, *, image=None, report) -> dict:  # noqa: ANN001
        if "repetition_time" in changed:
            raw["tr"] = changed["repetition_time"]
        return raw


def test_a_derived_field_the_data_model_gives_is_not_encoded() -> None:
    meta = GeoMetadata(repetition_time=2.0)
    report = ConversionReport()
    assert meta.write_raw(image=2.0, report=report) == {}
    assert not report.lossy
    report = ConversionReport()
    assert meta.write_raw(image=1.5, report=report) == {}
    assert set(report.approximated) == {"repetition_time"}
    # The data model says nothing: the value is the format's to write.
    assert meta.write_raw(image=None) == {"tr": 2.0}
    assert meta.check_writable(image=1.5).approximated


# ----------------------------------------------------------------------
#   LAZY FIELDS
# ----------------------------------------------------------------------

_LOADS = []


def _load_history() -> tuple:
    _LOADS.append(1)
    return ("cmd a", "cmd b")


class LazyMetadata(
    FormatMetadata,
    on={"format": "test-lazy"},
    supports=("description", "history"),
):
    """A format whose `history` sits in a lazy part of the record."""

    format: tx.Literal["test-lazy"] = "test-lazy"

    @classmethod
    def _default_raw(cls) -> dict:
        return {}

    @classmethod
    def _decode(cls, raw, *, image=None) -> dict:  # noqa: ANN001
        return {"description": raw.get("desc"), "history": Lazy(_load_history)}

    def _encode(self, raw, changed, *, image=None, report) -> dict:  # noqa: ANN001
        if "history" in changed:
            raw["history"] = changed["history"]
        return raw


def test_a_lazy_field_is_decoded_on_first_access() -> None:
    _LOADS.clear()
    meta = LazyMetadata.from_raw({"desc": "d"})
    copied = meta.copy()
    assert _LOADS == []
    assert meta.description == "d"
    assert meta.history == ("cmd a", "cmd b")
    assert meta.history == ("cmd a", "cmd b")
    assert _LOADS == [1]
    assert meta.changed_fields() == {}
    # The copy waits for its own read.
    assert copied.changed_fields() == {}
    assert _LOADS == [1, 1]


def test_assigning_a_lazy_field_snapshots_it_first() -> None:
    meta = LazyMetadata.from_raw({})
    meta.history = None
    assert meta.changed_fields() == {"history": None}
    meta = pickle.loads(pickle.dumps(LazyMetadata.from_raw({})))
    assert meta.history == ("cmd a", "cmd b")


# ----------------------------------------------------------------------
#   ONE WARNING
# ----------------------------------------------------------------------


def test_one_loss_warning_merges_the_reports() -> None:
    with pytest.warns(MetadataLossWarning) as caught:
        # `stacklevel=1`: this function (2 would be its caller).
        with one_loss_warning(stacklevel=1):
            apply_loss_policy(ConversionReport(source="a", lost={"x": 1}))
            apply_loss_policy(
                ConversionReport(target="b", approximated={"y": "z"})
            )
            apply_loss_policy(ConversionReport(lost={"q": 1}), "ignore")
    assert len(caught) == 1
    report = caught[0].message.report
    assert (report.source, report.target) == ("a", "b")
    assert report.lost == {"x": 1} and report.approximated == {"y": "z"}
    assert caught[0].filename == __file__
    with pytest.raises(MetadataLossError):
        with one_loss_warning():
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
