"""
Tests for the metadata of MGH/MGZ files (`MghMetadata`).

The footer of MRI parameters is decoded into the vocabulary in BIDS
units (ms -> s, rad -> deg), the command-line tags into `history`; a
read-then-save keeps the footer and the tags, a field set by the user is
written over them, and a conversion to or from another format reports
what the target cannot hold.
"""

import copy
import struct

import pytest

nb = pytest.importorskip("nibabel")


from brainhops.io.base._mgh_metadata import MghRaw  # noqa: E402
from brainhops.io.base._mgh_tags import (  # noqa: E402
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
#   CROSS-FORMAT
# ----------------------------------------------------------------------


def test_record_copies_are_independent() -> None:
    record = MghRaw(tags=b"x")
    record.header["tr"] = 5.0
    other = copy.deepcopy(record)
    other.header["tr"] = 6.0
    assert float(record.header["tr"]) == 5.0
    assert other != record
    other.header["tr"] = 5.0
    assert other == record
