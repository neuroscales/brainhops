"""
Tests for the vocabulary of MGH and MGZ files (`MghMetadata`).

The footer of acquisition parameters is decoded into the vocabulary in
BIDS units (milliseconds to seconds, radians to degrees), and the
command-line tags into `history`. `consolidate` encodes a changed field
into a copy of the record, keeps the other tags, and reports what MGH
cannot hold.
"""

import struct

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")


from brainhops.datamodel.metadata import ConversionReport  # noqa: E402
from brainhops.io.common.mgh import MghMetadata, MghRaw  # noqa: E402
from brainhops.io.common.mgh._tags import (  # noqa: E402
    decode_history,
    encode_history,
    parse_tags,
)


def _cmdline(command: bytes) -> bytes:
    payload = command + b"\0"
    return struct.pack(">iq", 3, len(payload)) + payload


# A tag FreeSurfer writes with a 64-bit length, then two command lines,
# then a legacy tag with no length (which must stay last).
OTHER_TAG = struct.pack(">iq", 43, 4) + struct.pack(">f", 3.0)
TAGS = (
    OTHER_TAG
    + _cmdline(b"mri_convert in.nii orig.mgz")
    + _cmdline(b"mri_normalize orig.mgz T1.mgz")
    + struct.pack(">i", 2)
)


# ----------------------------------------------------------------------
#   TAGS
# ----------------------------------------------------------------------


def test_tags_parse_into_chunks() -> None:
    parsed = parse_tags(TAGS)
    assert [tag for tag, _ in parsed] == [43, 3, 3, 2]
    assert b"".join(chunk for _, chunk in parsed) == TAGS
    assert parse_tags(b"") == []
    assert parse_tags(b"\0\0") is None


def test_history_replaces_only_the_command_lines() -> None:
    tags = encode_history(TAGS, ("a", "b", "c"))
    assert decode_history(tags) == ("a", "b", "c")
    parsed = parse_tags(tags)
    assert [tag for tag, _ in parsed] == [43, 3, 3, 3, 2]
    assert parsed[0][1] == OTHER_TAG
    # No command line yet: they go before the legacy tag.
    tags = encode_history(OTHER_TAG + struct.pack(">i", 2), ("a",))
    assert [tag for tag, _ in parse_tags(tags)] == [43, 3, 2]
    assert encode_history(TAGS, None) == OTHER_TAG + struct.pack(">i", 2)


# ----------------------------------------------------------------------
#   THE VOCABULARY
# ----------------------------------------------------------------------


def _read() -> MghMetadata:
    record = MghRaw(tags=TAGS)
    record.header["tr"] = 2000.0
    record.header["te"] = 30.0
    record.header["flip_angle"] = np.radians(9.0)
    record.header.set_data_dtype(np.int16)
    return MghMetadata.from_raw(record)


def test_the_footer_and_the_tags_are_decoded() -> None:
    meta = _read()
    assert meta.repetition_time == 2.0
    assert meta.echo_time == 0.03
    assert meta.inversion_time is None
    assert meta.flip_angle == 9.0
    assert meta.dtype == np.int16
    assert meta.history == (
        "mri_convert in.nii orig.mgz",
        "mri_normalize orig.mgz T1.mgz",
    )
    # Tags that do not parse give no history.
    assert MghMetadata.from_raw(MghRaw(tags=b"\0\0")).history is None


def test_a_changed_field_is_encoded_into_a_copy() -> None:
    meta = _read()
    report = ConversionReport()
    copy = meta.consolidate(
        repetition_time=1.5,
        inversion_time=0.9,
        echo_time=None,
        history=("recon-all",),
        dtype="uint8",
        on_loss=report,
    )
    assert not report.lossy
    header = copy.raw.header
    assert float(header["tr"]) == 1500.0
    assert float(header["ti"]) == 900.0
    assert float(header["te"]) == 0.0
    assert header.get_data_dtype() == np.uint8
    assert decode_history(copy.raw.tags) == ("recon-all",)
    # The other tags are kept in place.
    assert parse_tags(copy.raw.tags)[0][1] == OTHER_TAG
    # The read record is left as it was, and an untouched copy shares it.
    assert float(meta.raw.header["tr"]) == 2000.0
    assert meta.consolidate().raw is meta.raw
    assert copy.consolidate().raw is copy.raw


def test_what_mgh_cannot_hold() -> None:
    meta = _read()
    report = ConversionReport()
    copy = meta.consolidate(
        dtype="float64", description="d", extra={"K": 1}, on_loss=report
    )
    assert set(report.approximated) == {"dtype"}
    assert set(report.lost) == {"description", "extra"}
    assert copy.dtype == np.int16
    # A history over tags that do not parse is lost.
    broken = MghMetadata.from_raw(MghRaw(tags=b"\0\0"))
    report = ConversionReport()
    copy = broken.consolidate(history=("a",), on_loss=report)
    assert report.lost == {"history": ("a",)}
    assert copy.raw.tags == b"\0\0"


def test_a_changed_header_field_survives_a_save(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "header.mgz"
    _read().save(path, dtype="uint8")
    assert MghMetadata.load(path).dtype == np.uint8


# ----------------------------------------------------------------------
#   THE RECORD
# ----------------------------------------------------------------------


def test_record_copies_are_independent() -> None:
    record = MghRaw(tags=b"x")
    record.header["tr"] = 5.0
    other = record.copy()
    other.header["tr"] = 6.0
    assert float(record.header["tr"]) == 5.0
    assert other.tags == record.tags
