"""Tests for orientation datamodel compatibility."""

from brainhops.datamodel.orientation import AnatomicalOrientation, LeftToRight


def test_anatomical_orientation_accepts_plain_string_values() -> None:
    obj = AnatomicalOrientation(value="left-to-right")
    assert obj.value == "left-to-right"
    assert LeftToRight().value == "left-to-right"
    assert type(LeftToRight().value) is str
