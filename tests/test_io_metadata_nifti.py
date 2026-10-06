"""
Tests for the metadata of NIfTI files (`NiftiMetadata`), on images and
on every NIfTI-based transformation.

A NIfTI file read and saved again keeps its description, auxiliary file,
display range, slice timing and extensions; the common fields decode
them, and a common field set by the user is written over the header,
with what NIfTI cannot hold reported.
"""

import numpy as np
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


# ----------------------------------------------------------------------
#   DERIVATION
# ----------------------------------------------------------------------


def _header() -> "nb.Nifti1Header":
    """A 4-D header with encoding axes and a slice timing."""
    h = nb.Nifti1Header()
    h.set_data_shape((4, 5, 6, 3))
    h.set_dim_info(freq=0, phase=1, slice=2)
    h["slice_code"] = 1
    h["slice_start"], h["slice_end"] = 0, 5
    h["slice_duration"] = 0.5
    return h


def test_a_header_follows_the_spatial_axes() -> None:
    from brainhops.datamodel.axes import Axis
    from brainhops.datamodel.geometry import Geometry
    from brainhops.datamodel.metadata import Indexed, Resampled
    from brainhops.datamodel.systems import CoordinateSystem
    from brainhops.datamodel.transformations import Affine

    meta = NiftiMetadata.from_raw(_header())
    # A resampling clears the slice slots and `dim_info` of the copy.
    resampled = meta.derive(Resampled(Affine(np.eye(4)[:3]), Geometry()))
    assert resampled.raw is not meta.raw
    assert resampled.raw.get_dim_info() == (None, None, None)
    assert resampled.raw["slice_code"] == 0
    assert meta.raw.get_dim_info() == (0, 1, 2)
    # An index along time keeps them.
    system = CoordinateSystem(
        axes=[Axis(n, "space") for n in "ijk"] + [Axis("t", "time")]
    )
    indexed = meta.derive(Indexed((Ellipsis, [0, 2]), (4, 5, 6, 3), system))
    assert indexed.raw is not meta.raw
    assert indexed.raw.get_dim_info() == (0, 1, 2)
    assert indexed.raw["slice_code"] == 1
    assert indexed.raw["slice_duration"] == 0.5
