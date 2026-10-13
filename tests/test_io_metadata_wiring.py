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

from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import (
    EncodingDirection,
    Metadata,
    metadata_loss_policy,
)
from brainhops.datamodel.systems import RASmm
from brainhops.datamodel.transformations import (
    Affine,
    Transformation,
)
from brainhops.io.common.nifti import NiftiMetadata
from brainhops.io.images.afni import AfniImage, AfniMetadata
from brainhops.io.images.freesurfer.mgh import MghImage, MghMetadata
from brainhops.io.images.minc import MincImage, MincMetadata
from brainhops.io.images.mrtrix import MrtrixImage, MrtrixMetadata
from brainhops.io.images.nifti import NiftiImage
from brainhops.io.images.nrrd import NrrdImage, NrrdMetadata
from brainhops.io.transformations.freesurfer.lta import (
    LtaMetadata,
    LtaTransformation,
)
from brainhops.io.transformations.freesurfer.m3z import M3zMetadata, M3zMorph
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
