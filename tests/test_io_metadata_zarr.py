"""
Tests for the metadata of Zarr images: `ZarrMetadata` (a plain array,
the vocabulary as a sidecar under the attribute `"brainhops"`) and
`OmeZarrMetadata` (an OME-Zarr pyramid: the multiscale name, `omero` and
the other group attributes).
"""

import pytest

pytest.importorskip("abczarr")


from brainhops.datamodel.metadata import (  # noqa: E402
    ConversionReport,
    Metadata,
)
from brainhops.io.images.zarr import (  # noqa: E402
    ZarrMetadata,
)


def _to(source, target, **kwargs):  # noqa: ANN001, ANN003, ANN202
    """`source.to(target, ...)`, and the report it filled."""
    report = ConversionReport()
    return source.to(target, on_loss=report, **kwargs), report


# ----------------------------------------------------------------------
#   PLAIN ZARR
# ----------------------------------------------------------------------


def test_plain_zarr_cannot_hold_diffusion() -> None:
    assert ZarrMetadata.unsupported_fields == {
        "bvalues",
        "bvectors",
        "scale_slope",
        "scale_intercept",
    }
    _, report = _to(
        Metadata(bvalues=(0.0, 1000.0)),
        ZarrMetadata,
    )
    assert report.lost == {"bvalues": (0.0, 1000.0)}
