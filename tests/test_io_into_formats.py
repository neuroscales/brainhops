"""
Every transformation converted into a format: exactly, or refused (#343).

The converters of the data model rebuild a transformation as the class
they are asked for, so without a converter of its own, a format would be
answered with a relabelled transformation: a `Scaling` between LPS spaces
as a `NiftiVoxelToRAS`, say. Each format therefore has a converter for
each family that would reach it, which converts exactly, or refuses. A
format converted to its own class is changed as any transformation of its
family is, and is not refused.
"""

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    LPSmm,
    RASmm,
    VoxelCoordinateSystem,
)
from brainhops.errors import ConversionError  # noqa: E402
from brainhops.io.base.parsers import WriterError  # noqa: E402
from brainhops.io.transformations.elastix import (  # noqa: E402
    ElastixTransform,
)
from brainhops.io.transformations.freesurfer.lta import (  # noqa: E402
    LtaTransformation,
    LtaTransformationRASToRAS,
    LtaTransformationVoxToVox,
)
from brainhops.io.transformations.freesurfer.m3z import M3zMorph  # noqa: E402
from brainhops.io.transformations.fsl.flirt import (  # noqa: E402
    FlirtTransform,
)
from brainhops.io.transformations.fsl.fnirt import (  # noqa: E402
    FnirtWarpField,
)
from brainhops.io.transformations.itk.h5 import H5Transform  # noqa: E402
from brainhops.io.transformations.itk.nifti import (  # noqa: E402
    ItkNiftiDisplacementField,
)
from brainhops.io.transformations.itk.tfm import TfmTransform  # noqa: E402
from brainhops.io.transformations.matrix import (  # noqa: E402
    CsvMatrixAffine,
    TxtMatrixAffine,
)
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiRASToVoxel,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.niftyreg import (  # noqa: E402
    NiftyRegAffine,
    NiftyRegDisplacementField,
)
from brainhops.io.transformations.x5 import X5Transform  # noqa: E402

VOXEL = VoxelCoordinateSystem()

MATRIX = np.array(
    [
        [0.0, -2.0, 0.0, 10.0],
        [1.5, 0.0, 0.0, -20.0],
        [0.0, 0.0, 2.5, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""An affine with a permutation, a flip, a scaling and an offset."""

LPS2RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""LPS and RAS differ by the sign of their first two axes."""


def _families(input: tx.Any, output: tx.Any) -> tx.Dict[str, tx.Any]:
    """One transformation of each family, between `input` and `output`."""
    ends = dict(input=input, output=output)
    field = np.zeros((2, 3, 4, 3))
    return {
        "identity": xforms.Identity(**ends),
        "translation": xforms.Translation([1.0, 2.0, 3.0], **ends),
        "scaling": xforms.Scaling([2.0, 3.0, 4.0], **ends),
        "permutation": xforms.Permutation([1, 0, 2], **ends),
        "rotation": xforms.Rotation(matrix=np.eye(3), **ends),
        "linear": xforms.Linear(matrix=np.diag([1.0, 2.0, 3.0]), **ends),
        "affine": xforms.Affine(MATRIX[:-1], **ends),
        "affine-tangent": xforms.AffineExponential(
            data=np.zeros((3, 4)), **ends
        ),
        "sequence": xforms.Sequence(
            transformations=[xforms.Scaling([2.0, 3.0, 4.0], **ends)]
        ),
        "immutable-sequence": xforms.ImmutableSequence(
            transformations=[xforms.Scaling([2.0, 3.0, 4.0], **ends)]
        ),
        "displacement": xforms.DisplacementField(field=field, **ends),
        "coordinates": xforms.CoordinatesField(field=field, **ends),
    }


FORMATS = [
    NiftiVoxelToRAS,
    NiftiRASToVoxel,
    NiftiRASCoordinatesField,
    NiftyRegAffine,
    NiftyRegDisplacementField,
    FlirtTransform,
    LtaTransformation,
    TxtMatrixAffine,
    TfmTransform,
    ItkNiftiDisplacementField,
    FnirtWarpField,
    M3zMorph,
    X5Transform,
    ElastixTransform,
]
"""Formats with an exact conversion, and formats with none yet."""


# ----------------------------------------------------------------------
#   NOTHING IS RELABELLED
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FORMATS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("family", sorted(_families(VOXEL, VOXEL)))
def test_a_transformation_no_format_holds_is_refused(
    family: str, cls: type
) -> None:
    # Between two voxel spaces, a transformation is held by none of these
    # formats, whose spaces are RAS or their own.
    t = _families(VOXEL, VOXEL)[family]
    with pytest.raises(ConversionError):
        t.to(cls)
    assert t.to(cls, error=False) is False


@pytest.mark.parametrize(
    "family, cls",
    [
        *(
            (family, cls)
            for cls in (FlirtTransform, LtaTransformation, TxtMatrixAffine)
            for family in ("scaling", "linear", "affine", "sequence")
        ),
        *(
            (family, cls)
            for cls in (
                NiftyRegDisplacementField,
                TfmTransform,
                ItkNiftiDisplacementField,
                FnirtWarpField,
                M3zMorph,
                X5Transform,
                ElastixTransform,
            )
            for family in ("sequence", "immutable-sequence")
        ),
    ],
    ids=lambda x: getattr(x, "__name__", x),
)
def test_a_format_without_an_exact_conversion_refuses(
    family: str, cls: type
) -> None:
    # The format may hold some transformations of the family, which are
    # for its exact converter to tell (#312); until then, every one is
    # refused, even between systems the format could hold.
    t = _families(RASmm(), RASmm())[family]
    with pytest.raises(ConversionError, match="#312"):
        t.to(cls)


def test_a_scaling_between_lps_spaces_is_not_a_nifti_affine() -> None:
    scaling = xforms.Scaling([2.0, 3.0, 4.0], input=LPSmm(), output=LPSmm())
    with pytest.raises(ConversionError, match="cannot be held exactly"):
        scaling.to(NiftiVoxelToRAS)
    with pytest.raises(ConversionError, match="cannot be held exactly"):
        NiftiVoxelToRAS.from_any(scaling)


@pytest.mark.parametrize("name", ["s.nii", "s.lta", "s.txt", "s.mat", "s.tfm"])
def test_save_refuses_what_no_format_holds(tmp_path, name: str) -> None:  # noqa: ANN001
    # `io.save` converts to the formats the name calls for, and writes
    # only what one of them holds exactly.
    scaling = xforms.Scaling([2.0, 3.0, 4.0], input=VOXEL, output=VOXEL)
    with pytest.raises(WriterError):
        io.save(scaling, tmp_path / name)
    assert not (tmp_path / name).exists()


# ----------------------------------------------------------------------
#   WHAT A FORMAT HOLDS IS CONVERTED EXACTLY
# ----------------------------------------------------------------------


AFFINE_FAMILIES = [
    "identity",
    "translation",
    "scaling",
    "permutation",
    "rotation",
    "linear",
    "affine",
    "affine-tangent",
    "sequence",
]
"""The families with an affine form."""


@pytest.mark.parametrize("family", AFFINE_FAMILIES)
def test_a_voxel_to_ras_member_of_a_family_is_a_nifti_affine(
    family: str,
) -> None:
    t = _families(VOXEL, RASmm())[family]
    expected = t.to(xforms.Affine).homogeneous_matrix
    for nifti in [t.to(NiftiVoxelToRAS), NiftiVoxelToRAS.from_any(t)]:
        assert type(nifti) is NiftiVoxelToRAS
        np.testing.assert_array_equal(nifti.homogeneous_matrix, expected)


@pytest.mark.parametrize("family", AFFINE_FAMILIES)
def test_a_ras_to_voxel_member_of_a_family_is_a_nifti_affine(
    family: str,
) -> None:
    t = _families(RASmm(), VOXEL)[family]
    nifti = t.to(NiftiRASToVoxel)
    assert type(nifti) is NiftiRASToVoxel
    np.testing.assert_array_equal(
        nifti.homogeneous_matrix, t.to(xforms.Affine).homogeneous_matrix
    )


def test_a_voxel_to_lps_scaling_is_flipped_into_ras() -> None:
    scaling = xforms.Scaling([2.0, 3.0, 4.0], input=VOXEL, output=LPSmm())
    nifti = scaling.to(NiftiVoxelToRAS)
    np.testing.assert_array_equal(
        nifti.homogeneous_matrix, LPS2RAS @ np.diag([2.0, 3.0, 4.0, 1.0])
    )


@pytest.mark.parametrize("family", ["scaling", "translation", "linear"])
def test_a_family_member_round_trips_through_nifti(
    tmp_path,  # noqa: ANN001
    family: str,
) -> None:
    t = _families(VOXEL, RASmm())[family]
    io.save(t, tmp_path / "affine.nii.gz")
    back = io.transformations.load(tmp_path / "affine.nii.gz")
    assert isinstance(back, NiftiVoxelToRAS)
    np.testing.assert_array_equal(
        back.homogeneous_matrix, t.to(xforms.Affine).homogeneous_matrix
    )


def test_a_cartesian_grid_is_a_nifti_field_of_coordinates() -> None:
    grid = xforms.CartesianField(shape=(2, 3, 4))
    nifti = grid.to(NiftiRASCoordinatesField)
    assert type(nifti) is NiftiRASCoordinatesField
    np.testing.assert_array_equal(nifti.field, grid.field)


def test_a_velocity_is_not_a_nifti_field_of_coordinates() -> None:
    velocity = xforms.StationaryVelocityField(data=np.zeros((2, 3, 4, 3)))
    with pytest.raises(ConversionError, match="holds a"):
        velocity.to(NiftiRASCoordinatesField)


# ----------------------------------------------------------------------
#   NIFTYREG
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", AFFINE_FAMILIES)
def test_a_ras_to_ras_member_of_a_family_is_a_niftyreg_affine(
    family: str,
) -> None:
    t = _families(RASmm(), RASmm())[family]
    niftyreg = t.to(NiftyRegAffine)
    assert type(niftyreg) is NiftyRegAffine
    np.testing.assert_array_equal(
        niftyreg.homogeneous_matrix, t.to(xforms.Affine).homogeneous_matrix
    )


def test_an_lps_to_lps_affine_is_flipped_into_niftyreg() -> None:
    affine = xforms.Affine(MATRIX[:-1], input=LPSmm(), output=LPSmm())
    niftyreg = affine.to(NiftyRegAffine)
    np.testing.assert_array_equal(
        niftyreg.homogeneous_matrix, LPS2RAS @ MATRIX @ LPS2RAS
    )


@pytest.mark.parametrize("family", ["scaling", "affine", "sequence"])
def test_an_affine_of_undeclared_systems_is_not_a_niftyreg_affine(
    family: str,
) -> None:
    t = _families(None, None)[family]
    with pytest.raises(ConversionError, match="does not declare"):
        t.to(NiftyRegAffine)


def test_a_ras_to_ras_affine_is_saved_as_niftyreg(tmp_path) -> None:  # noqa: ANN001
    affine = xforms.Affine(MATRIX[:-1], input=RASmm(), output=RASmm())
    io.save(affine, tmp_path / "affine.txt")
    back = io.transformations.load(tmp_path / "affine.txt", hint="niftyreg")
    assert isinstance(back, NiftyRegAffine)
    np.testing.assert_array_equal(back.homogeneous_matrix, MATRIX)


# ----------------------------------------------------------------------
#   A FORMAT CONVERTED TO ITSELF
# ----------------------------------------------------------------------


AFFINE_FORMATS = [
    NiftiVoxelToRAS,
    NiftiRASToVoxel,
    NiftyRegAffine,
    TxtMatrixAffine,
    LtaTransformation,
    LtaTransformationRASToRAS,
]
"""The affine formats that are built from their matrix."""


@pytest.mark.parametrize("cls", AFFINE_FORMATS, ids=lambda cls: cls.__name__)
def test_an_affine_format_takes_a_new_matrix(cls: type) -> None:
    t = cls(matrix=MATRIX[:-1])
    changed = t.to(matrix=2 * MATRIX[:-1])
    assert type(changed) is type(t)
    np.testing.assert_array_equal(changed.matrix, 2 * MATRIX[:-1])
    np.testing.assert_array_equal(t.matrix, MATRIX[:-1])


@pytest.mark.parametrize("cls", AFFINE_FORMATS, ids=lambda cls: cls.__name__)
def test_an_affine_format_is_converted_to_itself(cls: type) -> None:
    t = cls(matrix=MATRIX[:-1])
    assert t.to(type(t)) is t
    assert type(t.to(log=False)) is type(t)
    with pytest.raises(TypeError, match="tangent"):
        t.to(log=True)


@pytest.mark.parametrize("cls", AFFINE_FORMATS, ids=lambda cls: cls.__name__)
def test_an_affine_format_is_inverted(cls: type) -> None:
    t = cls(matrix=MATRIX[:-1])
    inverse = t.inverse().compute()
    np.testing.assert_allclose(
        inverse.homogeneous_matrix, np.linalg.inv(MATRIX), atol=1e-12
    )


def test_a_niftyreg_affine_is_inverted_as_one() -> None:
    inverse = NiftyRegAffine(matrix=MATRIX[:-1]).inverse().compute()
    assert type(inverse) is NiftyRegAffine


def test_a_flirt_transform_is_converted_to_itself() -> None:
    # The matrix of a FLIRT transform is derived from its raw matrix and
    # its images, so it is the raw matrix that is changed.
    flirt = FlirtTransform(flirt_matrix=np.eye(4))
    assert flirt.to(FlirtTransform) is flirt
    changed = flirt.to(flirt_matrix=MATRIX)
    assert type(changed) is FlirtTransform
    np.testing.assert_array_equal(changed.flirt_matrix, MATRIX)


def test_a_nifti_field_of_coordinates_takes_a_new_field() -> None:
    nifti = NiftiRASCoordinatesField(field=np.zeros((2, 3, 4, 3)))
    changed = nifti.to(field=np.ones((2, 3, 4, 3)))
    assert type(changed) is NiftiRASCoordinatesField
    np.testing.assert_array_equal(changed.field, np.ones((2, 3, 4, 3)))


def test_a_chain_format_is_converted_to_itself() -> None:
    itk = TfmTransform([xforms.Scaling([1.0, 2.0, 3.0])])
    assert itk.to(TfmTransform) is itk
    chain = [xforms.Translation([1.0, 2.0, 3.0])]
    changed = itk.to(transformations=chain)
    assert type(changed) is TfmTransform
    assert list(changed.transformations) == chain


@pytest.mark.parametrize(
    "t, cls",
    [
        (TxtMatrixAffine(matrix=MATRIX[:-1]), CsvMatrixAffine),
        (
            LtaTransformationVoxToVox(matrix=MATRIX[:-1]),
            LtaTransformationRASToRAS,
        ),
        (TfmTransform([xforms.Scaling([1.0, 2.0, 3.0])]), H5Transform),
    ],
    ids=["matrix", "lta", "itk"],
)
def test_another_variant_of_a_format_is_not_relabelled(
    t: xforms.Transformation, cls: type
) -> None:
    with pytest.raises(ConversionError, match="#312"):
        t.to(cls)
