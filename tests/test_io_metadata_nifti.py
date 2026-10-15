"""
Tests for the vocabulary of NIfTI files (`NiftiMetadata`).

The header of a NIfTI file is decoded into the vocabulary when the
metadata is built, and `consolidate` encodes the fields that changed
into a copy of the header. What NIfTI cannot hold is reported as lost,
and what it holds approximately is reported as approximated.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from brainhops.datamodel.axes import Axis  # noqa: E402
from brainhops.datamodel.geometry import Geometry  # noqa: E402
from brainhops.datamodel.metadata import (  # noqa: E402
    ConversionReport,
    EncodingDirection,
    Metadata,
    MetadataLossError,
    metadata_loss_policy,
)
from brainhops.datamodel.metadata._derive import (  # noqa: E402
    Indexed,
    Resampled,
)
from brainhops.datamodel.systems import CoordinateSystem  # noqa: E402
from brainhops.datamodel.transformations import Affine  # noqa: E402
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.metadata import MetadataFormat  # noqa: E402


def _header() -> "nb.Nifti1Header":
    """A 4-D header with every slot of the vocabulary set."""
    h = nb.Nifti1Header()
    h.set_data_shape((4, 5, 6, 3))
    h.set_data_dtype(np.int16)
    h.set_xyzt_units("mm", "msec")
    h["pixdim"][4] = 2000.0
    h["descrip"] = b"a run"
    h["aux_file"] = b"sub-01_T1w.nii"
    h["cal_min"], h["cal_max"] = 0.0, 100.0
    h["scl_slope"], h["scl_inter"] = 2.0, 1.0
    h.set_dim_info(freq=0, phase=1, slice=2)
    h["slice_code"] = 1
    h["slice_start"], h["slice_end"] = 0, 5
    h["slice_duration"] = 500.0
    h.set_sform(np.eye(4), code=1)
    return h


def _read() -> NiftiMetadata:
    return NiftiMetadata.from_raw(NiftiRaw(header=_header()))


def _consolidated(meta: NiftiMetadata, **fields) -> tuple:  # noqa: ANN003
    report = ConversionReport()
    return meta.consolidate(on_loss=report, **fields), report


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_every_slot_is_decoded() -> None:
    meta = _read()
    assert meta.description == "a run"
    assert meta.sources == ("sub-01_T1w.nii",)
    assert meta.display_range == (0.0, 100.0)
    assert meta.space == "scanner"
    assert meta.intent is None
    assert meta.phase_encoding_direction == EncodingDirection("j")
    assert meta.slice_encoding_direction == EncodingDirection("k")
    assert meta.slice_timing == (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    assert meta.repetition_time == 2.0
    assert meta.dtype == np.int16
    assert (meta.scale_slope, meta.scale_intercept) == (2.0, 1.0)
    # NIfTI has no slot for these.
    assert meta.echo_time is None and meta.extra == {}
    decoded = {
        name
        for name in NiftiMetadata.SUPPORTED_FIELDS
        if getattr(meta, name) is not None
    }
    assert decoded == set(NiftiMetadata.SUPPORTED_FIELDS) - {"intent"}


def test_an_empty_header() -> None:
    meta = NiftiMetadata.from_raw(NiftiRaw())
    assert meta.description is None
    assert meta.repetition_time is None
    assert meta.slice_timing is None
    assert NiftiMetadata.from_raw(None).dtype is None


def test_a_loaded_image_decodes_its_metadata(tmp_path) -> None:  # noqa: ANN001
    data = np.zeros((4, 5, 6, 3), dtype=np.int16)
    image = nb.Nifti1Image(data, np.eye(4), _header())
    path = tmp_path / "bold.nii.gz"
    nb.save(image, path)
    loaded = NiftiImage.load(path)
    assert loaded.metadata.description == "a run"
    assert loaded.metadata.repetition_time == 2.0
    assert MetadataFormat.load(path).slice_timing == (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
    )


# ----------------------------------------------------------------------
#   CONSOLIDATION
# ----------------------------------------------------------------------


def test_an_untouched_header_is_shared() -> None:
    meta = _read()
    assert meta.consolidate().raw is meta.raw


def test_each_field_is_encoded_into_a_copy() -> None:
    meta = _read()
    header = meta.raw.header
    copy, report = _consolidated(
        meta,
        description="edited",
        sources=("other.nii",),
        display_range=(1.0, 2.0),
        slice_timing=(2.5, 2.0, 1.5, 1.0, 0.5, 0.0),
        repetition_time=1.5,
        dtype="uint8",
        scale_slope=0.5,
        scale_intercept=None,
        intent="label",
    )
    assert not report.lossy
    h = copy.raw.header
    assert h["descrip"] == b"edited"
    assert h["aux_file"] == b"other.nii"
    assert (h["cal_min"], h["cal_max"]) == (1.0, 2.0)
    assert int(h["slice_code"]) == 2  # sequential decreasing
    assert h["pixdim"][4] == pytest.approx(1500.0)
    assert h.get_data_dtype() == np.uint8
    assert (float(h["scl_slope"]), float(h["scl_inter"])) == (0.5, 0.0)
    assert int(h["intent_code"]) == nb.nifti1.intent_codes.code["label"]
    assert copy.intent == "label"
    # The header of the read metadata is left as it was.
    assert header["descrip"] == b"a run"
    assert meta.description == "a run"
    # Consolidating again changes nothing.
    assert copy.consolidate().raw is copy.raw


def test_none_clears_the_slots() -> None:
    copy, report = _consolidated(
        _read(),
        description=None,
        display_range=None,
        slice_timing=None,
        phase_encoding_direction=None,
    )
    assert not report.lossy
    h = copy.raw.header
    assert h["descrip"] == b""
    assert (h["cal_min"], h["cal_max"]) == (0.0, 0.0)
    assert int(h["slice_code"]) == 0
    assert h.get_dim_info() == (0, None, 2)


def test_what_the_header_holds_approximately() -> None:
    copy, report = _consolidated(
        _read(),
        description="x" * 90,
        sources=("a.nii", "b.nii"),
        phase_encoding_direction="j-",
        space="mni",
    )
    assert set(report.approximated) == {
        "description",
        "sources",
        "phase_encoding_direction",
        "space",
    }
    assert copy.description == "x" * 80
    assert copy.sources == ("a.nii",)
    assert copy.phase_encoding_direction == EncodingDirection("j")
    # The space is that of the codes of the forms, which the geometry
    # decides.
    assert copy.space == "scanner"
    # An intent that retypes the axes is the geometry's.
    _, report = _consolidated(_read(), intent="vector")
    assert "intent" in report.approximated


def test_what_the_header_cannot_hold_is_lost() -> None:
    copy, report = _consolidated(
        _read(),
        echo_time=0.03,
        extra={"Key": 1},
        slice_encoding_direction=(1, 1, 0),
        intent="no such intent",
    )
    assert set(report.lost) == {
        "echo_time",
        "extra",
        "slice_encoding_direction",
        "intent",
    }
    assert copy.echo_time is None and copy.extra == {}
    # The oblique direction replaced the slice axis, which is cleared.
    assert copy.raw.header.get_dim_info() == (0, 1, None)
    with pytest.raises(MetadataLossError):
        _read().consolidate(echo_time=0.03, on_loss="raise")
    # A timing that no slice order describes clears the slice slots.
    copy, report = _consolidated(
        _read(), slice_timing=(0.0, 0.3, 0.1, 0.9, 0.2, 0.5)
    )
    assert set(report.lost) == {"slice_timing"}
    assert int(copy.raw.header["slice_code"]) == 0


def test_a_repetition_time_needs_a_time_axis() -> None:
    header = nb.Nifti1Header()
    header.set_data_shape((4, 5, 6))
    meta = NiftiMetadata.from_raw(NiftiRaw(header=header))
    copy, report = _consolidated(meta, repetition_time=2.0)
    assert report.lost == {"repetition_time": 2.0}
    assert copy.repetition_time is None


def test_metadata_without_a_record_writes_a_new_header(tmp_path) -> None:  # noqa: ANN001
    meta = NiftiMetadata(description="new")
    assert NiftiMetadata().consolidate().raw is None
    copy = meta.consolidate()
    assert isinstance(copy.raw.header, nb.Nifti1Header)
    assert copy.description == "new"
    path = tmp_path / "header.nii"
    meta.save(path)
    assert NiftiMetadata.load(path).description == "new"


def test_a_changed_field_survives_a_save(tmp_path) -> None:  # noqa: ANN001
    meta = _read()
    meta.description = "saved"
    path = tmp_path / "header.nii.gz"
    meta.save(path)
    assert NiftiMetadata.load(path).description == "saved"
    assert meta.raw.header["descrip"] == b"a run"
    meta.save(path, display_range=(3.0, 4.0))
    assert NiftiMetadata.load(path).display_range == (3.0, 4.0)


# ----------------------------------------------------------------------
#   CONVERSION AND DERIVATION
# ----------------------------------------------------------------------


def test_a_conversion_reports_a_lower_bound() -> None:
    generic = Metadata(
        description="d", echo_time=0.03, history=("a",), extra={"K": 1}
    )
    report = ConversionReport()
    nifti = generic.to(NiftiMetadata, on_loss=report)
    assert nifti.description == "d"
    assert nifti.raw is None
    assert report.lost == {
        "echo_time": 0.03,
        "history": ("a",),
        "extra": {"K": 1},
    }
    with metadata_loss_policy("ignore"):
        assert nifti.consolidate().echo_time is None


def test_a_header_follows_the_spatial_axes() -> None:
    meta = _read()
    # A resampling clears the slice slots and `dim_info` of a copy.
    resampled = meta.derive(Resampled(Affine(np.eye(4)[:3]), Geometry()))
    assert resampled.raw is not meta.raw
    assert resampled.raw.header.get_dim_info() == (None, None, None)
    assert int(resampled.raw.header["slice_code"]) == 0
    assert resampled.slice_timing is None
    assert resampled.phase_encoding_direction == EncodingDirection("j")
    assert meta.raw.header.get_dim_info() == (0, 1, 2)
    # The mapped direction differs from the copy, so it is written back.
    # The derivation adds no provenance that NIfTI cannot hold.
    assert resampled.generated_by is None and resampled.history is None
    copy, report = _consolidated(resampled)
    assert not report.lossy
    assert copy.raw.header.get_dim_info() == (None, 1, 2)
    # An index along time keeps the record.
    system = CoordinateSystem(
        axes=[Axis(n, "space") for n in "ijk"] + [Axis("t", "time")]
    )
    indexed = meta.derive(Indexed((Ellipsis, [0, 2]), (4, 5, 6, 3), system))
    assert indexed.raw is meta.raw
    assert indexed.slice_timing == meta.slice_timing
