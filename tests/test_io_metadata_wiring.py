"""Tests of the metadata of images and transformations (#233, #287).

The data model holds metadata on its roots, `Image` and
`Transformation`, and each file format takes that field over to hold
the metadata of its own class, which also holds the header of a file.
"""

import inspect
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from bagof.magic import fields, replace

import brainhops.io as io
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import (
    ConversionReport,
    EncodingDirection,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    metadata_loss_policy,
)
from brainhops.datamodel.systems import RASmm
from brainhops.datamodel.transformations import (
    Affine,
    Transformation,
)
from brainhops.io.base._base import Format, _takes_over
from brainhops.io.common.nifti import NiftiMetadata
from brainhops.io.images.afni import AfniImage, AfniMetadata
from brainhops.io.images.freesurfer.mgh import MghImage, MghMetadata
from brainhops.io.images.minc import MincImage, MincMetadata
from brainhops.io.images.mrtrix import MrtrixImage, MrtrixMetadata
from brainhops.io.images.nifti import NiftiImage
from brainhops.io.images.nrrd import NrrdImage, NrrdMetadata
from brainhops.io.transformations.c3d import C3dAffine
from brainhops.io.transformations.freesurfer.lta import (
    LtaMetadata,
    LtaTransformation,
)
from brainhops.io.transformations.freesurfer.m3z import M3zMetadata, M3zMorph
from brainhops.io.transformations.fsl.fnirt import FnirtWarpField
from brainhops.io.transformations.itk.nifti import ItkNiftiDisplacementField
from brainhops.io.transformations.nifti import (
    NiftiRASCoordinatesField,
    NiftiRASDisplacementField,
    NiftiRASToVoxel,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.niftyreg import NiftyRegDisplacementField
from brainhops.io.transformations.spm import SpmCoordinatesField
from brainhops.io.transformations.x5 import X5Metadata, X5Transform

# Each format that takes `metadata` over, with the class of its metadata.
_FORMATS = [
    (NiftiImage, NiftiMetadata),
    (MghImage, MghMetadata),
    (AfniImage, AfniMetadata),
    (MrtrixImage, MrtrixMetadata),
    (NrrdImage, NrrdMetadata),
    (MincImage, MincMetadata),
    (X5Transform, X5Metadata),
    (LtaTransformation, LtaMetadata),
    (M3zMorph, M3zMetadata),
    (NiftiRASToVoxel, NiftiMetadata),
    (NiftiVoxelToRAS, NiftiMetadata),
    (NiftiRASCoordinatesField, NiftiMetadata),
    (NiftiRASDisplacementField, NiftiMetadata),
    (NiftyRegDisplacementField, NiftiMetadata),
    (SpmCoordinatesField, NiftiMetadata),
    (ItkNiftiDisplacementField, NiftiMetadata),
    (FnirtWarpField, NiftiMetadata),
]


# ----------------------------------------------------------------------
#   ROOTS AND TAKEOVER
# ----------------------------------------------------------------------


def test_an_image_of_the_data_model_holds_generic_metadata() -> None:
    meta = Metadata(description="T1w")
    image = SingleScaleImage(np.zeros((2, 2, 2)), metadata=meta)
    assert image.metadata is meta
    assert "metadata" not in repr(image)
    # The field is keyword-only, so the positional arguments are kept.
    parameter = inspect.signature(SingleScaleImage).parameters["metadata"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_transformation_of_the_data_model_holds_generic_metadata() -> None:
    meta = Metadata(description="registration")
    affine = Affine(np.eye(4)[:-1], metadata=meta)
    assert affine.metadata is meta
    assert Affine(np.eye(4)[:-1]).metadata is None
    parameter = inspect.signature(Affine).parameters["metadata"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "cls,metadata_class", _FORMATS, ids=lambda x: x.__name__
)
def test_a_format_takes_the_metadata_of_the_root_over(
    cls: type, metadata_class: type
) -> None:
    # One field is known as `metadata`: the private field of the format,
    # which keeps the place of the field of the root.
    named = [f for f in fields(cls) if f.public_name == "metadata"]
    assert [f.name for f in named] == ["_metadata"]
    parameter = inspect.signature(cls).parameters["metadata"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    # Generic metadata is converted into the metadata of the format,
    # without a record, and the vocabulary rides along.
    with metadata_loss_policy("ignore"):
        obj = cls(metadata=Metadata(description="carried"))
    assert type(obj.metadata) is metadata_class
    assert obj.metadata.raw is None
    assert obj.metadata.description == "carried"


def test_a_format_built_from_a_data_model_converts_its_metadata() -> None:
    meta = Metadata(description="T1w", echo_time=0.003)
    image = SingleScaleImage(np.zeros((2, 3, 4), np.float32), metadata=meta)
    with metadata_loss_policy("ignore"):
        nifti = NiftiImage.from_instance(image)
        mgh = MghImage.from_instance(nifti)
    assert type(nifti.metadata) is NiftiMetadata
    assert nifti.metadata.raw is None
    assert nifti.metadata.description == "T1w"
    # The values ride along until a consolidation drops what the format
    # cannot store.
    assert type(mgh.metadata) is MghMetadata
    assert mgh.metadata.echo_time == 0.003
    assert mgh.metadata.description == "T1w"


def test_every_format_that_holds_metadata_takes_it_over() -> None:
    # A format that holds the metadata of its own class in `_metadata`
    # takes the field of the root over, whichever format it is.
    holders = [
        cls
        for cls in Format._REGISTRY
        if any(field.name == "_metadata" for field in fields(cls))
    ]
    assert holders
    for cls in holders:
        assert _takes_over(cls, "metadata"), cls.__name__


def test_metadata_fields_is_renamed_encoding_fields() -> None:
    assert not hasattr(Transformation, "metadata_fields")
    assert Affine.encoding_fields == ("log",)


# ----------------------------------------------------------------------
#   OPERATIONS AND CONVERSIONS
# ----------------------------------------------------------------------


def _nifti(tmp_path: Path) -> NiftiImage:
    """Return a 4-D NIfTI image read from a file with a described header."""
    image = nb.Nifti1Image(
        np.zeros((4, 5, 6, 3), np.float32), np.diag([2.0, 2.0, 2.0, 1.0])
    )
    header = image.header
    header["descrip"] = b"T1w"
    header.set_xyzt_units("mm", "sec")
    header["pixdim"][4] = 2.0
    header.set_dim_info(phase=1, slice=2)
    header["slice_code"] = 1
    header["slice_duration"] = 0.1
    header["slice_end"] = 5
    path = tmp_path / "series.nii.gz"
    nb.save(image, str(path))
    return NiftiImage.load(path)


def test_an_index_along_time_keeps_the_class_and_the_record(
    tmp_path: Path,
) -> None:
    image = _nifti(tmp_path)
    first = image[..., :2]
    assert type(first.metadata) is NiftiMetadata
    assert first.metadata.raw is image.metadata.raw
    assert first.metadata.description == "T1w"
    assert first.metadata.slice_timing == image.metadata.slice_timing


def test_a_reslice_clears_the_spatial_fields_and_keeps_the_class(
    tmp_path: Path,
) -> None:
    image = _nifti(tmp_path)
    resliced = image.reslice()
    assert type(resliced.metadata) is NiftiMetadata
    assert resliced.metadata.description == "T1w"
    assert resliced.metadata.slice_timing is None
    # The slots of the old voxel axes are cleared from a copy of the
    # record, and the record of the image is not changed.
    assert resliced.metadata.raw is not image.metadata.raw
    assert int(image.metadata.raw.header["slice_code"]) == 1
    flipped = image[:, ::-1]
    assert flipped.metadata.phase_encoding_direction == EncodingDirection("j-")


def test_a_conversion_into_a_nifti_affine_carries_the_metadata() -> None:
    affine = Affine(np.eye(4)[:-1], metadata=Metadata(description="reg"))
    converted = affine.to(NiftiVoxelToRAS)
    assert type(converted.metadata) is NiftiMetadata
    assert converted.metadata.description == "reg"
    assert converted.metadata.raw is None


def test_the_node_of_the_vocabulary_gives_its_block_the_metadata() -> None:
    affine = Affine(np.eye(4)[:-1], input=RASmm(), output=RASmm())
    empty = X5Transform(transformations=[affine]).consolidate()
    metadata = replace(empty.metadata, description="hi").consolidate()
    block = X5Transform(metadata=metadata).transformations[0]
    # The block holds generic metadata, without the record of the file.
    assert type(block.metadata) is Metadata
    assert block.metadata.description == "hi"


# ----------------------------------------------------------------------
#   WRITES
# ----------------------------------------------------------------------


def _int16_nifti(tmp_path: Path) -> NiftiImage:
    """Return a 3-D NIfTI image read from a file of 16-bit integers."""
    data = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    path = tmp_path / "int16.nii.gz"
    nb.save(nb.Nifti1Image(data, np.eye(4)), str(path))
    return NiftiImage.load(path)


def test_an_assigned_field_survives_a_save(tmp_path: Path) -> None:
    # The metadata is consolidated as step 2 of every write, so a field
    # assigned and never consolidated is written (tension 403).
    image = _int16_nifti(tmp_path)
    image.metadata = replace(image.metadata, description="edited")
    path = tmp_path / "edited.nii.gz"
    image.save(path)
    assert NiftiImage.load(path).metadata.description == "edited"
    assert image.consolidate().metadata.description == "edited"
    # A change made in place is seen by every copy that shares the
    # metadata, and is written too.
    image.metadata.description = "in place"
    image.save(path)
    assert NiftiImage.load(path).metadata.description == "in place"


def test_a_save_option_overrides_a_field(tmp_path: Path) -> None:
    image = _int16_nifti(tmp_path)
    path = tmp_path / "options.nii.gz"
    image.save(path, description="option", dtype="float32")
    loaded = NiftiImage.load(path)
    assert loaded.metadata.description == "option"
    assert loaded.metadata.dtype == np.dtype("float32")
    # The image itself is not changed.
    assert image.metadata.description is None
    # Any other option names a field of the header.
    with pytest.raises(ValueError, match="no_such_field"):
        image.consolidate(no_such_field=1)


def test_the_metadata_decides_the_stored_type(tmp_path: Path) -> None:
    # Data computed in another type are stored with the type of the
    # metadata, as nibabel stores them with the type of its header.
    image = _int16_nifti(tmp_path)
    image.data = np.asarray(image.data, dtype=np.float64) * 2
    path = tmp_path / "stored.nii.gz"
    image.save(path)
    assert nb.load(str(path)).get_data_dtype() == np.dtype("int16")
    # nibabel scales the values into the integer type.
    written = np.asarray(NiftiImage.load(path).data)
    assert np.allclose(written, image.data, atol=1e-2)
    image.save(path, dtype="float64")
    assert nb.load(str(path)).get_data_dtype() == np.dtype("float64")


@pytest.mark.parametrize(
    "name, field",
    [
        ("descrip", "description"),
        ("aux_file", "sources"),
        ("cal_min", "display_range"),
        ("cal_max", "display_range"),
        ("scl_slope", "scale_slope"),
        ("scl_inter", "scale_intercept"),
        ("slice_code", "slice_timing"),
        ("slice_duration", "slice_timing"),
        ("dim_info", "phase_encoding_direction"),
    ],
)
def test_a_header_field_of_the_vocabulary_is_refused(
    tmp_path: Path, name: str, field: str
) -> None:
    image = _int16_nifti(tmp_path)
    with pytest.raises(TypeError, match=f"{field}"):
        image.save(tmp_path / "refused.nii.gz", **{name: 1})


def test_the_intent_is_the_field(tmp_path: Path) -> None:
    image = _int16_nifti(tmp_path)
    path = tmp_path / "intent.nii.gz"
    image.save(path, intent="label")
    assert int(nb.load(str(path)).header["intent_code"]) == 1002
    # An intent that changes how the axes are read is decided by the
    # geometry, and is reported as approximated.
    report = ConversionReport()
    image.save(path, intent="vector", on_loss=report)
    assert "intent" in report.approximated
    assert int(nb.load(str(path)).header["intent_code"]) == 0


def test_the_time_step_owns_the_repetition_time(tmp_path: Path) -> None:
    image = _nifti(tmp_path)
    report = ConversionReport()
    image.consolidate(repetition_time=3.0, on_loss=report)
    assert "repetition_time" in report.approximated


def test_assigning_metadata_keeps_assigned_transformations(
    tmp_path: Path,
) -> None:
    image = _int16_nifti(tmp_path)
    scaling = Affine(np.diag([2.0, 3.0, 4.0, 1.0])[:-1])
    image.transformations = [scaling]
    image.metadata = replace(image.metadata, description="other")
    assert image.transformations == [scaling]
    path = tmp_path / "kept.nii.gz"
    image.save(path)
    assert np.allclose(
        nb.load(str(path)).affine, np.diag([2.0, 3.0, 4.0, 1.0])
    )
    assert NiftiImage.load(path).metadata.description == "other"


def test_a_save_that_converts_warns_once(tmp_path: Path) -> None:
    meta = Metadata(description="T1w", echo_time=0.003, extra={"key": 1})
    image = SingleScaleImage(np.zeros((2, 3, 4), np.float32), metadata=meta)
    with pytest.warns(MetadataLossWarning) as caught:
        io.save(image, tmp_path / "once.nii.gz")
    assert len(caught) == 1
    assert set(caught[0].message.report.lost) == {"echo_time", "extra"}
    assert NiftiImage.load(tmp_path / "once.nii.gz").metadata.description == (
        "T1w"
    )
    report = ConversionReport()
    io.save(image, tmp_path / "report.mgz", on_loss=report)
    assert set(report.lost) == {"description", "extra"}
    with pytest.raises(MetadataLossError):
        io.save(image, tmp_path / "raise.mgz", on_loss="raise")


def test_a_format_without_a_vocabulary_reports_its_fields_as_lost(
    tmp_path: Path,
) -> None:
    # The record of a Convert3D matrix holds no field of the vocabulary,
    # so a field that is set is reported as lost by every write and every
    # consolidation, and the consolidated copy no longer holds it.
    with metadata_loss_policy("ignore"):
        xform = C3dAffine(np.eye(4)[:-1], metadata=Metadata(description="r"))
    report = ConversionReport()
    path = tmp_path / "affine.mat"
    xform.save(path, on_loss=report)
    assert set(report.lost) == {"description"}
    assert C3dAffine.load(path).metadata.description is None
    copy = xform.consolidate(on_loss="ignore")
    assert copy.metadata.description is None
    assert xform.metadata.description == "r"
    with pytest.raises(MetadataLossError):
        xform.consolidate(on_loss="raise", history=("a step",))


def test_an_x5_transform_of_one_node_writes_its_metadata(
    tmp_path: Path,
) -> None:
    affine = Affine(np.eye(4)[:-1], input=RASmm(), output=RASmm())
    xform = X5Transform(
        transformations=[affine], metadata=Metadata(description="reg")
    )
    path = tmp_path / "one.x5"
    xform.save(path)
    assert X5Transform.load(path).metadata.description == "reg"
