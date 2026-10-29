"""
Tests for the vocabulary of X5 files (`X5Metadata`).

The vocabulary of an X5 file lives in the JSON `Metadata` of the
transform that holds it: the single transform of the first chain, or of
a file without chains. Every field has a key there, and the other keys
are `extra`.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from brainhops.datamodel.metadata import (  # noqa: E402
    ConversionReport,
    EncodingDirection,
)
from brainhops.io.transformations.x5 import X5Metadata  # noqa: E402
from brainhops.io.transformations.x5._raw import (  # noqa: E402
    X5Header,
    X5Node,
    X5Raw,
)


def _raw(*metadata, chains=()) -> X5Raw:  # noqa: ANN001, ANN002
    nodes = [X5Node(transform=np.eye(4), metadata=m) for m in metadata]
    return X5Raw(header=X5Header(chains=list(chains)), nodes=nodes)


def _read() -> X5Metadata:
    return X5Metadata.from_raw(
        _raw(
            {
                "Description": "an affine",
                "PhaseEncodingDirection": "j-",
                "Moving": "sub-01_T1w.nii.gz",
                "Custom": 1,
            }
        )
    )


def test_the_json_of_the_transform_is_decoded() -> None:
    meta = _read()
    assert meta.description == "an affine"
    assert meta.phase_encoding_direction == EncodingDirection("j-")
    assert meta.moving == "sub-01_T1w.nii.gz"
    assert meta.extra == {"Custom": 1}
    assert "echo_time" in X5Metadata.SUPPORTED_FIELDS
    assert "extra" in X5Metadata.SUPPORTED_FIELDS


def test_a_changed_field_is_written_into_a_copy() -> None:
    meta = _read()
    report = ConversionReport()
    copy = meta.consolidate(
        echo_time=0.03, extra={"Other": 2}, description=None, on_error=report
    )
    assert not report.lossy
    assert copy.raw.nodes[0].metadata == {
        "PhaseEncodingDirection": "j-",
        "Moving": "sub-01_T1w.nii.gz",
        "EchoTime": 0.03,
        "Other": 2,
    }
    assert meta.raw.nodes[0].metadata["Description"] == "an affine"
    assert copy.consolidate().raw is copy.raw
    assert meta.consolidate().raw is meta.raw


def test_which_transform_holds_the_vocabulary() -> None:
    # The single transform of the first chain.
    chained = X5Metadata.from_raw(
        _raw({"Description": "a"}, {"Description": "b"}, chains=[(1,)])
    )
    assert chained.description == "b"
    # A chain of several transforms has no vocabulary of its own.
    composed = X5Metadata.from_raw(
        _raw({"Description": "a"}, {"Description": "b"}, chains=[(0, 1)])
    )
    assert composed.description is None
    report = ConversionReport()
    copy = composed.consolidate(description="c", on_error=report)
    assert report.lost == {"description": "c"}
    assert copy.raw is not composed.raw
    assert [n.metadata for n in copy.raw.nodes] == [
        {"Description": "a"},
        {"Description": "b"},
    ]
    # A record is never given a transform to hold the fields.
    report = ConversionReport()
    assert X5Metadata(description="d").consolidate(on_error=report).raw is None
    assert report.lost == {"description": "d"}


def test_metadata_that_is_not_json_is_replaced() -> None:
    meta = X5Metadata.from_raw(_raw("not json"))
    assert meta.description is None
    report = ConversionReport()
    copy = meta.consolidate(description="d", on_error=report)
    assert set(report.approximated) == {"extra"}
    assert copy.raw.nodes[0].metadata == {"Description": "d"}


def test_a_changed_field_survives_a_save(tmp_path) -> None:  # noqa: ANN001
    meta = _read()
    path = tmp_path / "affine.x5"
    meta.save(path, description="saved")
    assert X5Metadata.load(path).description == "saved"
    assert meta.description == "an affine"
