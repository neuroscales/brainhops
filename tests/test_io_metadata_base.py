"""
Tests for the base of the metadata of file formats
(`brainhops.io.metadata`), on a synthetic format, and for BIDS sidecars.

What is checked: the contract of `consolidate` (a fresh decode of the
record tells what changed, the changes are encoded into a copy, what no
slot takes is reported as lost, and the result is decoded afresh); the
overrides that `consolidate` takes; the conversion into a format, which
reports a lower bound of the loss from `SUPPORTED_FIELDS`; the BIDS
sidecar; and the storage helpers.
"""

import io as _io
import json
import pickle
import warnings

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import NoRepr, replace

from brainhops.datamodel.metadata import (
    ConversionReport,
    EncodingDirection,
    GeneratedBy,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    metadata_loss_policy,
)
from brainhops.io.base.parsers import (
    BinaryFileReader,
    TextFileReader,
    TextFileWriter,
    WriterError,
    _overrides_from_fileobj,
)
from brainhops.io.metadata import (
    BidsSidecar,
    MetadataFormat,
    MetadataReader,
    MetadataWriter,
    from_bids,
    to_bids,
)
from brainhops.io.metadata._base import (
    consolidated,
    lost_fields,
    unsupported_values,
)
from brainhops.io.metadata._json import encode_changes, to_json
from brainhops.io.metadata._storage import preferred_storage, stored_values

# ----------------------------------------------------------------------
#   A SYNTHETIC FORMAT
# ----------------------------------------------------------------------


class DictRaw(dict):
    """The record of `DictMetadata`: a dict with `desc`, `cal`, `slices`
    and free-form keys."""

    def copy(self) -> "DictRaw":
        return DictRaw(self)


_KNOWN = {"desc": "description", "cal": "display_range"}


def _decode(raw: tx.Optional[DictRaw]) -> tx.Dict[str, tx.Any]:
    if raw is None:
        return {}
    out = {name: raw.get(key) for key, name in _KNOWN.items()}
    out["slice_timing"] = raw.get("slices")
    out["extra"] = {
        k: v for k, v in raw.items() if k not in _KNOWN and k != "slices"
    }
    return out


def _encode(
    raw: DictRaw, changed: tx.Dict[str, tx.Any], report: ConversionReport
) -> DictRaw:
    for key, name in _KNOWN.items():
        if name not in changed:
            continue
        value = changed.pop(name)
        if value is None:
            raw.pop(key, None)
        elif name == "description" and len(value) > 8:
            report.approximated["description"] = "truncated to 8"
            raw[key] = value[:8]
        else:
            raw[key] = value
    if "slice_timing" in changed:
        times = changed["slice_timing"]
        if times is None:
            raw.pop("slices", None)
            changed.pop("slice_timing")
        elif len(times) <= 4:
            # More than four slices stay in `changed`: lost.
            raw["slices"] = changed.pop("slice_timing")
    if "extra" in changed:
        extra = changed.pop("extra")
        for key in [k for k in raw if k not in _KNOWN and k != "slices"]:
            del raw[key]
        raw.update(extra)
    return raw


class DictMetadata(MetadataFormat):
    """A format that stores three fields and free-form keys."""

    SUPPORTED_FIELDS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "extra",
        "description",
        "display_range",
        "slice_timing",
    )

    raw: NoRepr[tx.Optional[DictRaw]] = None

    @classmethod
    def from_raw(cls, raw, **kwargs) -> "DictMetadata":  # noqa: ANN001
        return cls(raw=raw, **{**_decode(raw), **kwargs})

    def _consolidate(self, *, inplace=False, **fields):  # noqa: ANN001, ANN003, ANN202
        return consolidated(
            self, _encode, DictRaw, inplace=inplace, fields=fields
        )


class OtherMetadata(MetadataFormat):
    """Another format, which stores only a description in its record."""

    SUPPORTED_FIELDS: tx.ClassVar[tx.Tuple[str, ...]] = ("description",)

    raw: NoRepr[tx.Optional[DictRaw]] = None

    @classmethod
    def from_raw(cls, raw, **kwargs) -> "OtherMetadata":  # noqa: ANN001
        description = None if raw is None else raw.get("desc")
        return cls(raw=raw, **{"description": description, **kwargs})


class _SharedRecordMetadata(MetadataFormat):
    """A base whose formats share their type of record and its field."""

    raw: NoRepr[tx.Optional[DictRaw]] = None


class FirstContainerMetadata(_SharedRecordMetadata):
    """A format whose record is that of its base."""


class SecondContainerMetadata(_SharedRecordMetadata):
    """Another format whose record is that of the same base."""


class TextDictRaw(DictRaw, TextFileReader, TextFileWriter):
    """A record that reads and writes itself as `key=value` lines."""

    @classmethod
    def from_lines(cls, lines, **kwargs) -> "TextDictRaw":  # noqa: ANN001
        if kwargs:
            raise TypeError(f"Unknown option(s): {', '.join(kwargs)}")
        return cls(line.split("=", 1) for line in lines if line.strip())

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        if kwargs:
            raise TypeError(f"Unknown option(s): {', '.join(kwargs)}")
        return (f"{key}={value}" for key, value in self.items())


class TextDictMetadata(MetadataReader, MetadataWriter, DictMetadata):
    """A format whose metadata is read and written through its record."""

    RAW: tx.ClassVar[type] = TextDictRaw


class StreamDictRaw(DictRaw, BinaryFileReader):
    """A record that reads a description from the start of a stream."""

    @classmethod
    def from_fileobj(cls, file, **kwargs) -> "StreamDictRaw":  # noqa: ANN001
        return cls(desc=file.read(4).decode())


class StreamDictMetadata(MetadataReader, DictMetadata):
    """A binary format whose record class reads streams."""

    RAW: tx.ClassVar[type] = StreamDictRaw


def _read() -> DictMetadata:
    return DictMetadata.from_raw(
        DictRaw(desc="short", cal=(0.0, 1.0), Key="v")
    )


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_from_raw_decodes_every_supported_field() -> None:
    meta = _read()
    assert meta.description == "short"
    assert meta.display_range == (0.0, 1.0)
    assert meta.extra == {"Key": "v"}
    # A field the format has no slot for decodes as `None`.
    assert meta.echo_time is None
    assert repr(meta) == (
        "DictMetadata(extra={'Key': 'v'}, description='short', "
        "display_range=(0.0, 1.0))"
    )
    # Given values take precedence over the decoded ones.
    given = DictMetadata.from_raw(DictRaw(desc="a"), description="b")
    assert given.description == "b"


def test_a_format_with_a_record_class_reads_every_input_through_it(
    tmp_path,  # noqa: ANN001
) -> None:
    path = tmp_path / "meta.txt"
    path.write_text("desc=short\nKey=v\n")
    inputs = (
        TextDictMetadata.from_filename(path),
        TextDictMetadata.from_fileobj(_io.StringIO("desc=short\nKey=v")),
        TextDictMetadata.from_bytes(b"desc=short\nKey=v"),
        TextDictMetadata.from_text("desc=short\nKey=v"),
        TextDictMetadata.from_lines(["desc=short", "Key=v"]),
        TextDictMetadata.load(path),
    )
    for meta in inputs:
        assert isinstance(meta, TextDictMetadata)
        assert isinstance(meta.raw, TextDictRaw)
        assert meta.description == "short"
        assert meta.extra == {"Key": "v"}
    # The options go to the record class, which refuses them.
    with pytest.raises(TypeError, match="bogus"):
        TextDictMetadata.from_text("desc=short", bogus=True)


def test_a_format_with_a_record_class_writes_every_output_through_it(
    tmp_path,  # noqa: ANN001
) -> None:
    meta = TextDictMetadata.from_text("desc=short\nKey=v")
    assert list(meta.to_lines()) == ["desc=short", "Key=v"]
    assert meta.to_text() == "desc=short\nKey=v"
    assert meta.to_bytes() == b"desc=short\nKey=v"
    stream = _io.StringIO()
    meta.to_fileobj(stream)
    assert stream.getvalue() == "desc=short\nKey=v\n"
    path = tmp_path / "meta.txt"
    meta.save(path)
    assert path.read_text() == "desc=short\nKey=v"
    # The options go to the record, which refuses them.
    with pytest.raises(TypeError, match="bogus"):
        meta.to_text(bogus=True)
    # Metadata without a record has nothing to write.
    with pytest.raises(WriterError, match="no record"):
        TextDictMetadata().to_text()


def test_a_format_with_a_record_class_opens_files_in_its_mode() -> None:
    assert TextDictMetadata._READ_MODE == TextDictRaw._READ_MODE == "rt"
    assert StreamDictMetadata._READ_MODE == StreamDictRaw._READ_MODE == "rb"
    assert TextDictMetadata._WRITE_MODE == TextDictRaw._WRITE_MODE


def test_a_record_class_that_reads_streams_reads_bytes_too() -> None:
    # A format with no record class reads no stream itself.
    assert not _overrides_from_fileobj(DictMetadata)
    assert _overrides_from_fileobj(StreamDictMetadata)
    meta = StreamDictMetadata.from_bytes(b"shortest")
    assert isinstance(meta.raw, StreamDictRaw)
    assert meta.description == "shor"


def test_metadata_of_a_format_compares_by_identity() -> None:
    meta = _read()
    assert meta == meta
    assert meta != replace(meta)
    clone = pickle.loads(pickle.dumps(meta))
    assert clone.raw == meta.raw and clone.description == meta.description


# ----------------------------------------------------------------------
#   CONSOLIDATION
# ----------------------------------------------------------------------


def test_an_untouched_object_shares_its_record() -> None:
    meta = _read()
    copy = meta.consolidate()
    assert copy is not meta
    assert copy.raw is meta.raw
    assert meta.consolidate(inplace=True) is meta
    assert meta.raw == {"desc": "short", "cal": (0.0, 1.0), "Key": "v"}


def test_a_changed_field_is_encoded_into_a_copy() -> None:
    meta = _read()
    record = meta.raw
    meta.description = "mine"
    copy = meta.consolidate()
    assert copy.raw is not record
    assert copy.raw["desc"] == "mine"
    # The record is never changed in place.
    assert record["desc"] == "short"
    assert meta.raw is record
    # A second consolidation changes nothing.
    again = copy.consolidate()
    assert again.raw is copy.raw


def test_none_clears_the_slot() -> None:
    meta = _read()
    meta.description = None
    meta.display_range = None
    copy = meta.consolidate()
    assert "desc" not in copy.raw and "cal" not in copy.raw
    assert copy.raw["Key"] == "v"


def test_the_fields_win_over_an_edit_of_the_record() -> None:
    meta = _read()
    meta.raw["Other"] = 1
    # The object no longer agrees with its record: `extra` misses the
    # key, so it counts as a change and wins.
    assert "Other" not in meta.consolidate().raw
    # Metadata built from the edited record agrees with it.
    edited = DictMetadata.from_raw(DictRaw(desc="d", Other=1))
    assert edited.consolidate().raw is edited.raw


def test_extra_is_written_whole() -> None:
    meta = DictMetadata.from_raw(DictRaw(desc="d", Kept=1, Gone=2))
    meta.extra = {"Kept": 1, "New": 3}
    assert meta.consolidate().raw == {"desc": "d", "Kept": 1, "New": 3}


def test_a_field_without_a_slot_is_lost() -> None:
    meta = _read()
    meta.echo_time = 0.03
    report = ConversionReport()
    copy = meta.consolidate(on_error=report)
    assert report.lost == {"echo_time": 0.03}
    assert copy.echo_time is None
    # The other fields are decoded afresh, and the record is shared,
    # since no field it stores changed.
    assert copy.description == "short"
    with pytest.warns(MetadataLossWarning):
        meta.consolidate()
    with pytest.raises(MetadataLossError) as info:
        meta.consolidate(on_error="raise")
    assert info.value.report.lost == {"echo_time": 0.03}
    with warnings.catch_warnings(), metadata_loss_policy("ignore"):
        warnings.simplefilter("error")
        meta.consolidate()


def test_a_value_a_slot_cannot_take_is_lost() -> None:
    meta = _read()
    meta.slice_timing = (0.0, 0.1, 0.2, 0.3, 0.4)
    report = ConversionReport()
    copy = meta.consolidate(on_error=report)
    assert report.lost == {"slice_timing": (0.0, 0.1, 0.2, 0.3, 0.4)}
    assert copy.slice_timing is None
    assert "slices" not in copy.raw


def test_an_approximation_reads_back_as_stored() -> None:
    meta = _read()
    meta.description = "much too long"
    report = ConversionReport()
    copy = meta.consolidate(on_error=report)
    assert report.approximated == {"description": "truncated to 8"}
    assert copy.description == "much too"
    assert copy.consolidate().raw is copy.raw


def test_consolidate_takes_overrides_by_name() -> None:
    meta = _read()
    copy = meta.consolidate(description="given", display_range=(1, 2))
    assert copy.raw["desc"] == "given" and copy.raw["cal"] == (1.0, 2.0)
    assert meta.description == "short"
    # An override equal to the record changes nothing.
    assert meta.consolidate(description="short").raw is meta.raw
    with pytest.raises(TypeError, match="no_such_field"):
        meta.consolidate(no_such_field=1)
    with pytest.raises(TypeError):
        meta.consolidate(raw=DictRaw())
    inplace = meta.consolidate(inplace=True, description="here")
    assert inplace is meta
    assert meta.description == "here" and meta.raw["desc"] == "here"


def test_metadata_without_a_record() -> None:
    empty = DictMetadata()
    assert empty.consolidate().raw is None
    meta = DictMetadata(description="d", echo_time=0.03)
    report = ConversionReport()
    copy = meta.consolidate(on_error=report)
    assert copy.raw == {"desc": "d"}
    assert report.lost == {"echo_time": 0.03}


def test_the_base_class_stores_nothing() -> None:
    meta = MetadataFormat(raw=object(), description="d")
    report = ConversionReport()
    copy = meta.consolidate(on_error=report)
    assert report.lost == {"description": "d"}
    assert copy.raw is meta.raw and copy.description is None
    assert MetadataFormat.SUPPORTED_FIELDS == ()
    assert MetadataFormat.from_raw(None).description is None


def test_lost_fields_reports_what_is_left() -> None:
    report = lost_fields({"echo_time": 0.03}, ConversionReport())
    assert report.lost == {"echo_time": 0.03}


# ----------------------------------------------------------------------
#   CONVERSION
# ----------------------------------------------------------------------


def _rich() -> Metadata:
    return Metadata(
        description="a scan",
        echo_time=0.03,
        slice_timing=(0.0, 0.5, 1.0),
        extra={"Custom": 1},
    )


def test_unsupported_values_reads_the_constant() -> None:
    assert unsupported_values(DictMetadata, _rich()) == {"echo_time": 0.03}
    assert unsupported_values(OtherMetadata, _rich()) == {
        "echo_time": 0.03,
        "slice_timing": (0.0, 0.5, 1.0),
        "extra": {"Custom": 1},
    }
    assert unsupported_values(OtherMetadata, Metadata(extra={})) == {}


def test_conversion_carries_every_field_and_reports_a_lower_bound() -> None:
    report = ConversionReport()
    target = _rich().to(DictMetadata, on_error=report)
    assert type(target) is DictMetadata
    assert target.description == "a scan"
    # The value rides along, and the next consolidation drops it.
    assert target.echo_time == 0.03
    assert report.lost == {"echo_time": 0.03}
    assert (report.source, report.target) == ("Metadata", "DictMetadata")
    assert target.raw is None
    assert target.extra == {"Custom": 1}
    with pytest.warns(MetadataLossWarning):
        _rich().to(DictMetadata)
    with pytest.raises(MetadataLossError):
        _rich().to(DictMetadata, on_error="raise")


def test_the_record_stays_within_its_format() -> None:
    meta = _read()
    with metadata_loss_policy("ignore"):
        other = meta.to(OtherMetadata)
        generic = meta.to(Metadata)
    assert other.raw is None
    assert type(generic) is Metadata and generic.description == "short"
    same = DictMetadata.from_instance(meta)
    assert same.raw is meta.raw
    assert meta.to().raw is meta.raw
    # Converted metadata does not share `extra` with its source.
    assert generic.extra == meta.extra and generic.extra is not meta.extra
    # Two formats that declare `raw` on a common base keep their records
    # apart too.
    first = FirstContainerMetadata(raw=DictRaw(desc="d"))
    assert SecondContainerMetadata.from_instance(first).raw is None
    assert FirstContainerMetadata.from_instance(first).raw is first.raw


def test_from_any_converts_metadata() -> None:
    with metadata_loss_policy("ignore"):
        meta = DictMetadata.from_any(_rich())
    assert type(meta) is DictMetadata and meta.description == "a scan"
    assert DictMetadata.from_any({"description": "d"}).description == "d"


# ----------------------------------------------------------------------
#   BIDS SIDECARS
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
    meta = from_bids(_SIDECAR)
    assert type(meta) is Metadata
    assert meta.repetition_time == 2.0
    assert meta.slice_timing == (0.0, 0.5, 1.0, 1.5)
    assert meta.generated_by == (GeneratedBy(name="fMRIPrep", version="23.0"),)
    assert meta.display_range == (0.0, 255.0)
    # A key that names no vocabulary field lands in `extra`.
    assert meta.extra == {"TaskName": "rest"}
    sidecar = to_bids(meta)
    assert sidecar == {**_SIDECAR, "DisplayRange": [0.0, 255.0]}
    json.dumps(sidecar)


def test_a_sidecar_is_a_metadata_format(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "sub-01_bold.json"
    path.write_text(json.dumps(_SIDECAR))
    loaded = MetadataFormat.load(path)
    assert type(loaded) is BidsSidecar
    assert loaded.raw == _SIDECAR
    assert loaded.echo_time == 0.03
    assert BidsSidecar.load(path).raw == _SIDECAR
    assert from_bids(path) == from_bids(_SIDECAR)
    assert from_bids(str(path)) == from_bids(_SIDECAR)
    assert from_bids(json.dumps(_SIDECAR)) == from_bids(_SIDECAR)
    # An untouched sidecar consolidates to the same object; a change is
    # written into a copy, and saved.
    assert loaded.consolidate().raw is loaded.raw
    loaded.echo_time = 0.05
    out = tmp_path / "out.json"
    loaded.save(out)
    assert json.loads(out.read_text())["EchoTime"] == 0.05
    assert loaded.raw["EchoTime"] == 0.03
    loaded.save(out, description="given")
    assert json.loads(out.read_text())["Description"] == "given"


def test_diffusion_is_not_a_sidecar_key() -> None:
    meta = Metadata(bvalues=(0, 1000), description="dwi")
    with pytest.warns(MetadataLossWarning) as record:
        sidecar = to_bids(meta)
    assert len(record) == 1
    assert sidecar == {"Description": "dwi"}
    with pytest.raises(MetadataLossError):
        to_bids(meta, on_error="raise")
    assert "bvalues" not in BidsSidecar.SUPPORTED_FIELDS


def test_an_oblique_direction_is_lost_in_a_sidecar() -> None:
    meta = Metadata(phase_encoding_direction=(1, 1, 0))
    with pytest.raises(MetadataLossError) as info:
        to_bids(meta, on_error="raise")
    assert set(info.value.report.lost) == {"phase_encoding_direction"}
    sidecar = BidsSidecar.from_raw({"PhaseEncodingDirection": "j"})
    report = ConversionReport()
    copy = sidecar.consolidate(
        phase_encoding_direction=(1, 1, 0), on_error=report
    )
    assert "PhaseEncodingDirection" not in copy.raw
    assert set(report.lost) == {"phase_encoding_direction"}


def test_terms_units_types_and_times_in_a_sidecar() -> None:
    from brainhops.datamodel.units import Unit

    assert to_bids(Metadata(space="mni")) == {"SpatialReference": "mni"}
    assert to_bids(Metadata(data_unit="ms")) == {"DataUnit": "ms"}
    for name in ("a.u.", "mm/s", "degC", "uV", "HU"):
        bids = to_bids(Metadata(data_unit=name))
        assert from_bids(bids).data_unit == Unit(name)
    assert to_bids(Metadata(data_unit="mm2/s")) == {"DataUnit": "mm2/s"}
    assert to_bids(Metadata(dtype=">i2")) == {"Dtype": "int16"}
    assert from_bids({"Dtype": "uint8"}).dtype == np.uint8
    meta = from_bids({"AcquisitionTime": "2020-01-02T03:04:05"})
    assert meta.acquisition_time.year == 2020
    assert to_bids(meta) == {"AcquisitionTime": "2020-01-02T03:04:05"}
    meta = Metadata(phase_encoding_direction="j-")
    meta.slice_encoding_direction = {"Vector": [0, 0, 1]}
    assert to_bids(meta) == {
        "PhaseEncodingDirection": "j-",
        "SliceEncodingDirection": "k",
    }
    assert from_bids(to_bids(meta)) == meta


def test_a_direction_in_a_coordinate_system_in_json() -> None:
    from brainhops.datamodel.systems import CoordinateSystem

    named = EncodingDirection((1, 0, 0), space=CoordinateSystem(name="RAS"))
    # JSON writes the name of the system, which reads back as a label.
    assert to_json("phase_encoding_direction", named) == {
        "Vector": [1.0, 0.0, 0.0],
        "Space": "RAS",
    }
    # Without a name, the direction cannot be written, and is lost.
    unnamed = EncodingDirection((1, 0, 0), space=CoordinateSystem())
    obj = {"PhaseEncodingDirection": "j"}
    report = ConversionReport()
    encode_changes(obj, {"phase_encoding_direction": unnamed}, report=report)
    assert obj == {}
    assert report.lost == {"phase_encoding_direction": unnamed}


# ----------------------------------------------------------------------
#   STORAGE
# ----------------------------------------------------------------------


def test_stored_values_round_only_into_integers() -> None:
    values = np.array([1.0, 2.2, 3.6])
    # Into an integer type, the unscaled values are rounded...
    stored = stored_values(values, np.int16, 2.0, 1.0)
    assert stored.dtype == np.int16
    assert stored.tolist() == [0, 1, 1]
    # ... into a floating-point type, they are kept as they are.
    stored = stored_values(values, np.float32, 2.0, 1.0)
    assert stored.dtype == np.float32
    assert stored.tolist() == pytest.approx([0.0, 0.6, 1.3])
    assert stored_values(values, np.float64).tolist() == values.tolist()


def test_preferred_storage() -> None:
    # The metadata decides the type, and the data only without one.
    labels = Metadata(dtype="uint8")
    assert preferred_storage(labels, np.zeros(3, np.int64)) == (
        np.uint8,
        None,
        None,
    )
    assert preferred_storage(Metadata(), np.zeros(3, np.int16))[0] == np.int16
    assert preferred_storage(None, np.zeros(3, np.int16))[0] == np.int16
    # A scaling is kept when the values fit it...
    scaled = Metadata(dtype="int16", scale_slope=0.5, scale_intercept=1.0)
    data = np.array([1.0, 1.5, 3.0])
    assert preferred_storage(scaled, data) == (np.int16, 0.5, 1.0)
    # ... and is reported otherwise.
    report = ConversionReport()
    result = preferred_storage(scaled, data + 0.1, report=report)
    assert result == (np.int16, None, None)
    assert set(report.approximated) == {"scale_slope", "scale_intercept"}


# ----------------------------------------------------------------------
#   DERIVATION
# ----------------------------------------------------------------------


def test_derived_metadata_keeps_its_class_and_its_record() -> None:
    meta = _read()
    derived = meta.derive(history="smooth")
    assert type(derived) is DictMetadata
    assert derived.raw is meta.raw
    # The format has no slot for the provenance that the derivation
    # adds, so the derived metadata loses nothing when consolidated.
    assert derived.history is None and derived.generated_by is None
    with warnings.catch_warnings(), metadata_loss_policy("raise"):
        warnings.simplefilter("error")
        assert derived.consolidate().raw is meta.raw
    # A value that was there is kept as it was.
    meta.history = ("acquired",)
    assert meta.derive(history="smooth").history == ("acquired",)
