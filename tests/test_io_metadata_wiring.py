"""Tests of the metadata of images and transformations (#233, #287).

The data model holds metadata on its roots, `Image` and
`Transformation`, and each file format takes that field over to hold
the metadata of its own class, which also holds the header of a file.
"""

import inspect

import numpy as np
import pytest
from bagof.magic import fields

from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import Metadata, metadata_loss_policy
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
