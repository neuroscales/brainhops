"""
Tests for the metadata of NIfTI files (`NiftiMetadata`), on images and
on every NIfTI-based transformation.

A NIfTI file read and saved again keeps its description, auxiliary file,
display range, slice timing and extensions; the common fields decode
them, and a common field set by the user is written over the header,
with what NIfTI cannot hold reported.
"""

import pytest


def _to(source, target, **kwargs):  # noqa: ANN001, ANN003, ANN202
    """`source.to(target, ...)`, and the report it filled."""
    report = ConversionReport()
    return source.to(target, on_loss=report, **kwargs), report


nb = pytest.importorskip("nibabel")


from brainhops.datamodel.metadata import (  # noqa: E402
    ConversionReport,
    Metadata,
)
from brainhops.io.images.nifti import NiftiMetadata  # noqa: E402

# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_capabilities() -> None:
    assert not NiftiMetadata.supports("extra")
    assert not NiftiMetadata.supports("echo_time")
    assert NiftiMetadata.supports("slice_timing")


# ----------------------------------------------------------------------
#   CONVERSION
# ----------------------------------------------------------------------


def test_what_nifti_cannot_hold_is_reported() -> None:
    generic = Metadata(
        description="d", echo_time=0.03, history=("a",), extra={"K": 1}
    )
    nifti, report = _to(generic, NiftiMetadata)
    assert nifti.description == "d"
    assert report.lost == {
        "echo_time": 0.03,
        "history": ("a",),
        "extra": {"K": 1},
    }
