"""
Tests for the metadata of NIfTI files (`NiftiMetadata`), on images and
on every NIfTI-based transformation.

A NIfTI file read and saved again keeps its description, auxiliary file,
display range, slice timing and extensions; the common fields decode
them, and a common field set by the user is written over the header,
with what NIfTI cannot hold reported.
"""

import warnings

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from bagof.magic import fields  # noqa: E402

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.metadata import (  # noqa: E402
    UNSUPPORTED,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    convert,
    metadata_loss_policy,
)
from brainhops.io.base._base import FileBasedObject  # noqa: E402
from brainhops.io.base.nifti import NiftiParser  # noqa: E402
from brainhops.io.images.nifti import NiftiImage, NiftiMetadata  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiVoxelToRAS,
)

AFFINE = np.diag([2.0, 2.0, 2.5, 1.0])
SHAPE = (4, 5, 6, 7)


def _write_scan(path, **edits):  # noqa: ANN001, ANN003, ANN202
    """A 4-D scan with every slot `NiftiMetadata` reads filled in."""
    image = nb.Nifti1Image(np.zeros(SHAPE, "float32"), AFFINE)
    h = image.header
    h["descrip"] = b"a bold run"
    h["aux_file"] = b"sub-01_T1w.nii"
    h["cal_min"], h["cal_max"] = 0.0, 100.0
    h.set_xyzt_units("mm", "sec")
    h["pixdim"][4] = 2.0
    h.set_dim_info(freq=0, phase=1, slice=2)
    h.set_slice_duration(0.25)
    h["slice_code"] = 1  # sequential increasing
    h["slice_end"] = SHAPE[2] - 1
    h.extensions.append(nb.nifti1.Nifti1Extension(6, b"a comment"))
    for key, value in edits.items():
        h[key] = value
    nb.save(image, str(path))
    return path


@pytest.fixture
def scan(tmp_path):  # noqa: ANN001, ANN201
    return _write_scan(tmp_path / "scan.nii.gz")


def _header(path):  # noqa: ANN001, ANN202
    return nb.load(str(path)).header


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_the_header_is_decoded_into_the_vocabulary(scan) -> None:  # noqa: ANN001
    image = io.load(scan)
    assert type(image) is NiftiImage
    meta = image.metadata
    assert type(meta) is NiftiMetadata
    assert meta.description == "a bold run"
    assert meta.sources == ("sub-01_T1w.nii",)
    assert meta.display_range == (0.0, 100.0)
    assert meta.repetition_time == 2.0
    assert meta.phase_encoding_direction == "j"
    assert meta.slice_encoding_direction == "k"
    assert meta.slice_timing == (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    assert meta.space == "aligned"
    assert meta.intent is None
    assert meta.echo_time is UNSUPPORTED
    assert meta.extra is UNSUPPORTED
    assert meta.changed_fields() == {}


def test_the_record_is_the_header(scan) -> None:  # noqa: ANN001
    image = io.load(scan)
    assert image.metadata.raw is image.header
    assert image.metadata.header is image.header


def test_capabilities() -> None:
    assert NiftiMetadata.derived_fields == {
        "repetition_time",
        "intent",
        "space",
    }
    assert not NiftiMetadata.supports("extra")
    assert not NiftiMetadata.supports("echo_time")
    assert NiftiMetadata.supports("slice_timing")


def test_an_image_built_in_memory_has_empty_metadata() -> None:
    image = NiftiImage(data=np.zeros((2, 3, 4)))
    assert type(image.metadata) is NiftiMetadata
    assert image.metadata.raw is None


def test_a_header_given_later_is_read_again(scan) -> None:  # noqa: ANN001
    image = NiftiImage(data=np.zeros((2, 3, 4)))
    image.header = _header(scan)
    assert image.metadata.description == "a bold run"


def test_explicit_metadata_wins_over_the_header(scan) -> None:  # noqa: ANN001
    nib = nb.load(str(scan))
    image = NiftiImage(image=nib, metadata=NiftiMetadata(description="mine"))
    assert image.metadata.raw is image.header
    assert image.metadata.description == "mine"
    assert image.metadata.display_range == (0.0, 100.0)
    assert image.metadata.changed_fields() == {"description": "mine"}


# ----------------------------------------------------------------------
#   ROUND TRIP (snapshot cases 1-4)
# ----------------------------------------------------------------------


def test_case1_a_read_then_save_keeps_the_header(scan, tmp_path) -> None:  # noqa: ANN001
    out = tmp_path / "out.nii.gz"
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetadataLossWarning)
        io.load(scan).save(out)
    h = _header(out)
    assert h["descrip"].item() == b"a bold run"
    assert h["aux_file"].item() == b"sub-01_T1w.nii"
    assert (float(h["cal_min"]), float(h["cal_max"])) == (0.0, 100.0)
    assert h.get_dim_info() == (0, 1, 2)
    assert int(h["slice_code"]) == 1
    assert float(h["slice_duration"]) == 0.25
    assert [e.get_content() for e in h.extensions] == [b"a comment"]
    back, read = io.load(out).metadata, io.load(scan).metadata
    # The time step is geometry, which the image writer does not store
    # (`pixdim[4]` is written as 1): `repetition_time` derives from it.
    assert back.repetition_time == 1.0
    back.repetition_time = read.repetition_time
    assert back == read


def test_geometry_and_encoding_still_come_from_the_writer(
    tmp_path,  # noqa: ANN001
) -> None:
    path = _write_scan(tmp_path / "scan.nii.gz", scl_slope=4.0, scl_inter=0)
    image = io.load(path)
    image.data = np.ones(SHAPE, "int16")
    out = tmp_path / "out.nii.gz"
    image.save(out)
    h = _header(out)
    assert h.get_data_dtype() == np.int16
    assert np.isnan(h["scl_slope"]) or float(h["scl_slope"]) in (0.0, 1.0)
    assert h.get_xyzt_units()[0] == "mm"
    np.testing.assert_allclose(h.get_best_affine(), AFFINE)
    # The other slots are still the record's.
    assert h["descrip"].item() == b"a bold run"


def test_case2_a_header_edit_survives(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata.raw["descrip"] = b"edited in the header"
    image.save(tmp_path / "out.nii")
    assert _header(tmp_path / "out.nii")["descrip"].item() == (
        b"edited in the header"
    )


def test_case3_a_common_field_wins(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata.raw["descrip"] = b"edited in the header"
    image.metadata.description = "set by the user"
    image.metadata.display_range = (10, 20)
    image.metadata.slice_timing = (0.0, 0.75, 0.25, 1.0, 0.5, 1.25)
    image.save(tmp_path / "out.nii")
    h = _header(tmp_path / "out.nii")
    assert h["descrip"].item() == b"set by the user"
    assert (float(h["cal_min"]), float(h["cal_max"])) == (10.0, 20.0)
    assert h.get_value_label("slice_code") == "alternating increasing"
    back = io.load(tmp_path / "out.nii").metadata
    assert back.slice_timing == (0.0, 0.75, 0.25, 1.0, 0.5, 1.25)


def test_case4_none_clears_the_header_slot(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata.description = None
    image.metadata.display_range = None
    image.metadata.slice_timing = None
    image.metadata.sources = None
    image.save(tmp_path / "out.nii")
    h = _header(tmp_path / "out.nii")
    assert h["descrip"].item() == b""
    assert h["aux_file"].item() == b""
    assert (float(h["cal_min"]), float(h["cal_max"])) == (0.0, 0.0)
    assert int(h["slice_code"]) == 0 and float(h["slice_duration"]) == 0


def test_derive_clears_the_grid_fields_on_write(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata = image.metadata.derive(grid_changed=True)
    image.save(tmp_path / "out.nii")
    h = _header(tmp_path / "out.nii")
    assert int(h["slice_code"]) == 0
    assert h.get_dim_info() == (None, None, None)
    assert h["descrip"].item() == b"a bold run"


def test_slices_are_dropped_when_the_slice_axis_changes(
    scan,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    image = io.load(scan)
    image.data = np.zeros((4, 5, 3, 7), "float32")
    # The common field is untouched, so nothing is reported; but the
    # record's six-slice timing cannot be kept for three slices.
    image.save(tmp_path / "out.nii")
    h = _header(tmp_path / "out.nii")
    assert int(h["slice_code"]) == 0
    assert h["descrip"].item() == b"a bold run"


# ----------------------------------------------------------------------
#   VALUE-DEPENDENT LOSS AND POLICIES
# ----------------------------------------------------------------------


def test_an_81_byte_description_is_truncated_and_reported(
    tmp_path,  # noqa: ANN001
) -> None:
    image = NiftiImage(data=np.zeros((2, 3, 4), "float32"))
    image.metadata.description = "x" * 81
    report = image.metadata.check_writable(image=image)
    assert set(report.approximated) == {"description"}
    with pytest.warns(MetadataLossWarning, match="80 bytes") as record:
        image.save(tmp_path / "out.nii")
    assert len(record) == 1
    assert _header(tmp_path / "out.nii")["descrip"].item() == b"x" * 80


def test_an_irregular_slice_timing_is_lost(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata.slice_timing = (0.0, 0.1, 0.7, 0.2, 0.3, 0.4)
    with pytest.warns(MetadataLossWarning) as record:
        image.save(tmp_path / "out.nii")
    assert record[0].message.report.lost == {
        "slice_timing": (0.0, 0.1, 0.7, 0.2, 0.3, 0.4)
    }
    # The value set by the user replaced the record's, so the record's
    # timing is not written either.
    assert int(_header(tmp_path / "out.nii")["slice_code"]) == 0


def test_polarity_and_derived_fields_are_approximated(
    scan,  # noqa: ANN001
) -> None:
    image = io.load(scan)
    image.metadata.phase_encoding_direction = "j-"
    image.metadata.repetition_time = 3.0
    report = image.metadata.check_writable(image=image)
    assert set(report.approximated) == {
        "phase_encoding_direction",
        "repetition_time",
    }


def test_on_loss_is_a_save_option(tmp_path) -> None:  # noqa: ANN001
    image = NiftiImage(data=np.zeros((2, 3, 4), "float32"))
    image.metadata.description = "x" * 81
    with pytest.raises(MetadataLossError):
        image.save(tmp_path / "a.nii", on_loss="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image.save(tmp_path / "b.nii", on_loss="ignore")
    with metadata_loss_policy("raise"):
        with pytest.raises(MetadataLossError):
            image.save(tmp_path / "c.nii")


def test_an_intent_name_is_written_when_the_axes_set_none(
    tmp_path,  # noqa: ANN001
) -> None:
    image = NiftiImage(data=np.zeros((2, 3, 4), "int16"))
    image.metadata.intent = "label"
    image.save(tmp_path / "out.nii")
    assert int(_header(tmp_path / "out.nii")["intent_code"]) == 1002
    assert io.load(tmp_path / "out.nii").metadata.intent == "label"


# ----------------------------------------------------------------------
#   like= AND OVERRIDES
# ----------------------------------------------------------------------


def test_like_still_copies_its_description(scan, tmp_path) -> None:  # noqa: ANN001
    template = nb.Nifti1Header()
    template["descrip"] = b"from the template"
    image = io.load(scan)
    image.to_nibabel(like=template).to_filename(str(tmp_path / "a.nii"))
    assert _header(tmp_path / "a.nii")["descrip"].item() == (
        b"from the template"
    )
    # A common field set by the user wins over `like`...
    image.metadata.description = "mine"
    h = image.to_nibabel(like=template).header
    assert h["descrip"].item() == b"mine"
    # ... and an explicit header override wins over everything.
    h = image.to_nibabel(like=template, descrip="override").header
    assert h["descrip"].item() == b"override"


# ----------------------------------------------------------------------
#   CONVERSION
# ----------------------------------------------------------------------


def test_nifti_to_generic_and_back_is_lossless(scan) -> None:  # noqa: ANN001
    nifti = io.load(scan).metadata
    generic, report = convert(nifti, Metadata)
    assert not report.lossy
    assert generic.raw is None
    back, report = convert(generic, NiftiMetadata)
    assert not report.lossy
    assert back == nifti
    assert back.raw is None  # the record never travels across formats


def test_what_nifti_cannot_hold_is_reported() -> None:
    generic = Metadata(
        description="d", echo_time=0.03, history=("a",), extra={"K": 1}
    )
    nifti, report = convert(generic, NiftiMetadata, on_loss="ignore")
    assert nifti.description == "d"
    assert report.lost == {
        "echo_time": 0.03,
        "history": ("a",),
        "extra": {"K": 1},
    }


def test_a_sidecar_round_trips_through_nifti(scan) -> None:  # noqa: ANN001
    sidecar = convert(io.load(scan).metadata, Metadata)[0].to_bids()
    assert sidecar["SliceTiming"] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    assert sidecar["RepetitionTime"] == 2.0
    nifti, report = convert(Metadata.from_bids(sidecar), NiftiMetadata)
    assert not report.lossy
    assert nifti == io.load(scan).metadata


def test_metadata_is_a_shared_field_for_from_instance(
    tmp_path,  # noqa: ANN001
) -> None:
    generic = SingleScaleImage(
        np.zeros((2, 3, 4), "float32"),
        metadata=Metadata(description="in memory", echo_time=0.03),
    )
    with pytest.warns(MetadataLossWarning, match="echo_time"):
        nifti = NiftiImage.from_instance(generic)
    assert type(nifti.metadata) is NiftiMetadata
    assert nifti.metadata.description == "in memory"
    nifti.save(tmp_path / "out.nii")
    assert _header(tmp_path / "out.nii")["descrip"].item() == b"in memory"
    # And back: the NIfTI metadata becomes generic.
    back = SingleScaleImage.from_instance(io.load(tmp_path / "out.nii"))
    assert type(back.metadata) is Metadata
    assert back.metadata.description == "in memory"


def test_io_save_converts_and_reports(tmp_path) -> None:  # noqa: ANN001
    generic = SingleScaleImage(
        np.zeros((2, 3, 4), "float32"),
        metadata=Metadata(description="via save", flip_angle=90.0),
    )
    with pytest.warns(MetadataLossWarning, match="flip_angle"):
        io.save(generic, tmp_path / "out.nii")
    assert _header(tmp_path / "out.nii")["descrip"].item() == b"via save"


# ----------------------------------------------------------------------
#   NIfTI-BASED TRANSFORMATIONS
# ----------------------------------------------------------------------


def _nifti_formats():  # noqa: ANN202
    return sorted(
        (c for c in FileBasedObject._REGISTRY if issubclass(c, NiftiParser)),
        key=lambda c: c.__name__,
    )


@pytest.mark.parametrize("cls", _nifti_formats(), ids=lambda c: c.__name__)
def test_every_nifti_format_narrows_the_field(cls) -> None:  # noqa: ANN001
    field = next(f for f in fields(cls) if f.name == "metadata")
    assert field.type is NiftiMetadata
    assert not field.positional


def _vector_file(path, intent=1007, name="", **slots):  # noqa: ANN001, ANN003, ANN202
    vectors = np.zeros((3, 4, 5, 1, 3), "float32")
    image = nb.Nifti1Image(vectors, AFFINE)
    image.header.set_intent(intent, name=name)
    image.header["descrip"] = b"a warp"
    image.header["aux_file"] = b"moving.nii"
    for key, value in slots.items():
        image.header[key] = value
    nb.save(image, str(path))
    return path


def test_a_coordinates_field_keeps_its_metadata(tmp_path) -> None:  # noqa: ANN001
    path = _vector_file(tmp_path / "y.nii.gz", name="Mapping")
    field = io.load(path)
    assert type(field) is NiftiRASCoordinatesField
    assert field.metadata.description == "a warp"
    assert field.metadata.intent == "vector"
    field.save(tmp_path / "out.nii.gz")
    h = _header(tmp_path / "out.nii.gz")
    assert h["descrip"].item() == b"a warp"
    assert h["aux_file"].item() == b"moving.nii"
    # The writer's own intent is kept.
    assert int(h["intent_code"]) == 1007
    assert h.get_intent()[2] == "Mapping"


def test_a_transformation_built_in_memory_writes_its_metadata(
    tmp_path,  # noqa: ANN001
) -> None:
    # The generic metadata is converted when it is given, and what NIfTI
    # cannot hold is reported then.
    with pytest.warns(MetadataLossWarning, match="moving"):
        field = NiftiRASCoordinatesField(
            field=np.zeros((3, 4, 5, 3), "float32"),
            metadata=Metadata(description="made here", moving="mov.nii"),
        )
    assert type(field.metadata) is NiftiMetadata
    field.save(tmp_path / "out.nii.gz")
    assert _header(tmp_path / "out.nii.gz")["descrip"].item() == b"made here"


def test_an_affine_and_its_inverse_share_the_metadata(scan) -> None:  # noqa: ANN001
    xform = NiftiVoxelToRAS.from_file(scan)
    assert xform.metadata.description == "a bold run"
    xform.metadata.description = "edited"
    assert xform.inverse().metadata.description == "edited"


def test_spm_and_itk_fields_keep_their_metadata(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.itk.nifti import (
        ItkNiftiDisplacementField,
    )
    from brainhops.io.transformations.spm.y import SpmCoordinatesField

    spm = SpmCoordinatesField.from_file(
        _vector_file(tmp_path / "y_spm.nii", name="Mapping")
    )
    assert spm.metadata.description == "a warp"
    itk_path = _vector_file(tmp_path / "itk.nii.gz")
    itk = ItkNiftiDisplacementField.from_file(itk_path)
    assert itk.metadata.description == "a warp"
    itk.save(tmp_path / "itk_out.nii.gz")
    h = _header(tmp_path / "itk_out.nii.gz")
    assert h["descrip"].item() == b"a warp"
    assert int(h["intent_code"]) == 1007


def test_niftyreg_keeps_reading_intent_p1_and_writing_it(
    tmp_path,  # noqa: ANN001
) -> None:
    from brainhops.io.transformations.niftyreg import (
        NiftyRegDisplacementField,
    )

    path = _vector_file(
        tmp_path / "disp.nii.gz", name="NREG_TRANS", intent_p1=1
    )
    xform = io.load(path)
    assert type(xform) is NiftyRegDisplacementField
    assert xform.niftyreg_type == 1
    assert xform.metadata.description == "a warp"
    xform.save(tmp_path / "out.nii.gz")
    h = _header(tmp_path / "out.nii.gz")
    assert float(h["intent_p1"]) == 1
    assert h.get_intent()[2] == "NREG_TRANS"
    assert h["descrip"].item() == b"a warp"


def test_fnirt_keeps_reading_intent_p_from_the_header() -> None:
    from brainhops.io.transformations.fsl.fnirt import FnirtWarpField

    image = nb.Nifti1Image(np.zeros((3, 4, 5, 3), "float32"), AFFINE)
    image.header["intent_code"] = 2006
    image.header["intent_p1"] = 1.0
    image.header["descrip"] = b"fnirt"
    warp = FnirtWarpField.from_nibabel(image)
    assert warp.metadata.description == "fnirt"
    assert warp.metadata.intent == "fnirt disp field"
    assert float(warp.header["intent_p1"]) == 1.0
    assert warp.metadata.raw is warp.header


# ----------------------------------------------------------------------
#   USER GUIDE
# ----------------------------------------------------------------------


def test_the_user_guide_runs() -> None:
    """`docs/start/metadata.md` is a runnable doctest."""
    import doctest
    from pathlib import Path

    page = Path(__file__).parents[1] / "docs" / "start" / "metadata.md"
    result = doctest.testfile(
        str(page),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0
