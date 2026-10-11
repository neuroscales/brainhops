"""
Exact conversions into the file formats, and their refusals (#343).

The converters of the data model rebuild a transformation as whatever
class they are asked for and keep its parameters, which is called
relabelling. Some classes are bound to coordinate systems, such as
`VoxelToRAS`, the NIfTI and NiftyReg affines, and the RAS and LPS
fields. Such a class checks that its endpoints are compatible with its
systems, so a relabel between other systems fails with a
`ConversionError` when the class is built. The exact converters of
NIfTI and NiftyReg bridge the endpoints instead, so that an affine to
LPS is flipped into RAS. A format without an exact converter (#312) is
relabelled by one converter for all formats, and only when the systems
of the transformation are not known or are compatible with those of the
format. A format with no coordinate systems of its own, such as a bare
matrix or an ITK chain, therefore holds the relabelled transformation
with its original endpoints. `t.to(Format)`, `Format.from_any(t)` and
`Format.from_instance(t)` are one conversion. A format that is
converted to its own class is changed by the rules of its family, as
any other transformation of that family is.
"""

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.datamodel._transformations.compute.convert import (  # noqa: E402
    find_converter,
)
from brainhops.datamodel.axes import Axis  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    CoordinateSystem,
    FVoxelCoordinateSystem,
    LPSmm,
    RASCoordinateSystem,
    RASmm,
    VoxelCoordinateSystem,
)
from brainhops.errors import (  # noqa: E402
    ConversionError,
    IncompatibleSystemError,
)
from brainhops.io.base.parsers import WriterError  # noqa: E402
from brainhops.io.transformations.base._converters import (  # noqa: E402
    relabel,
    relabels,
)
from brainhops.io.transformations.base.affines import (  # noqa: E402
    LPSToVoxel,
    RASToRAS,
    RASToVoxel,
    VoxelToLPS,
    VoxelToRAS,
)
from brainhops.io.transformations.base.fields import (  # noqa: E402
    LPSCoordinatesField,
    RASCoordinatesField,
)
from brainhops.io.transformations.elastix import (  # noqa: E402
    ElastixTransform,
)
from brainhops.io.transformations.freesurfer.lta import (  # noqa: E402
    LtaTransformation,
    LtaTransformationRASToRAS,
    LtaTransformationVoxToVox,
)
from brainhops.io.transformations.freesurfer.m3z import M3zMorph  # noqa: E402
from brainhops.io.transformations.fsl._fields import (  # noqa: E402
    RASToWarpField,
    WarpFieldToRAS,
)
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
    NiftiRASDisplacementField,
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
"""The matrix that flips the first two axes, turning LPS into RAS."""


def _families(input: tx.Any, output: tx.Any) -> tx.Dict[str, tx.Any]:
    """
    Return one transformation of each family.

    Each transformation maps `input` to `output`.
    """
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
    ItkNiftiDisplacementField,
    FnirtWarpField,
    M3zMorph,
    X5Transform,
    ElastixTransform,
]
"""The formats that are bound to RAS, to LPS or to their own spaces."""

CHAIN_FORMATS = [FnirtWarpField, X5Transform, ElastixTransform]
"""
The formats that hold a chain and read their coordinate systems from it,
so that they have no systems of their own to refuse.
"""

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
    "immutable-sequence",
]
"""The families whose members are affines, or chains of affines."""

RELABELLED = {
    *(
        (family, cls)
        for family in ("sequence", "immutable-sequence")
        for cls in CHAIN_FORMATS
    ),
    *((family, LtaTransformation) for family in AFFINE_FAMILIES),
}
"""
The pairs of `FORMATS` that relabel a transformation between voxel
spaces. A generic LTA file reads its systems from its type and from the
geometry of its volumes, so it has none of its own to refuse, and
whether it can write the systems of an affine is decided when it is
written.
"""


# ----------------------------------------------------------------------
#   CLASSES BOUND TO COORDINATE SYSTEMS
# ----------------------------------------------------------------------


BOUND = [
    (VoxelToRAS, LPSmm(), "output"),
    (RASToVoxel, LPSmm(), "input"),
    (VoxelToLPS, RASmm(), "output"),
    (LPSToVoxel, RASmm(), "input"),
    (RASToRAS, LPSmm(), "output"),
    (NiftiVoxelToRAS, LPSmm(), "output"),
    (NiftiRASToVoxel, VOXEL, "input"),
    (NiftyRegAffine, LPSmm(), "input"),
    (RASToWarpField, LPSmm(), "input"),
    (WarpFieldToRAS, VOXEL, "output"),
]
"""
The affine classes bound to systems, each with an endpoint that it
cannot have and the side of that endpoint.
"""


@pytest.mark.parametrize(
    "cls, system, side", BOUND, ids=[cls.__name__ for cls, _, _ in BOUND]
)
def test_a_bound_class_refuses_an_incompatible_endpoint(
    cls: type, system: CoordinateSystem, side: str
) -> None:
    with pytest.raises(IncompatibleSystemError, match="not compatible"):
        cls(matrix=MATRIX[:-1], **{side: system})


@pytest.mark.parametrize(
    "cls, output",
    [(RASCoordinatesField, LPSmm()), (LPSCoordinatesField, RASmm())],
)
def test_a_bound_field_refuses_an_incompatible_endpoint(
    cls: type, output: CoordinateSystem
) -> None:
    with pytest.raises(IncompatibleSystemError):
        cls(field=np.zeros((2, 3, 4, 3)), output=output)


def test_a_flirt_transform_refuses_an_incompatible_endpoint() -> None:
    with pytest.raises(IncompatibleSystemError):
        FlirtTransform(flirt_matrix=np.eye(4), input=LPSmm())


@pytest.mark.parametrize(
    "voxel, world",
    [
        (None, None),
        (FVoxelCoordinateSystem(), RASCoordinateSystem()),
        (VOXEL, CoordinateSystem(axes=[Axis(), Axis(), Axis()])),
        (CoordinateSystem(axes=[...]), RASmm()),
    ],
    ids=["unknown", "other-names", "unknown-axes", "open"],
)
def test_a_bound_class_accepts_a_compatible_endpoint(
    voxel: tx.Any, world: tx.Any
) -> None:
    # The names of axes are only labels, and an endpoint or an axis that
    # is not known may be the one that the class is bound to.
    t = VoxelToRAS(matrix=MATRIX[:-1], input=voxel, output=world)
    assert type(t) is VoxelToRAS


def test_a_2d_world_is_the_first_axes_of_lps() -> None:
    # ITK holds the world of a 2-D image as the L and P axes.
    world = CoordinateSystem(axes=LPSmm().axes[:2])
    pixel = CoordinateSystem(axes=VOXEL.axes[:2])
    t = VoxelToLPS(matrix=np.eye(3)[:2], input=pixel, output=world)
    assert t.output == world


def test_a_relabel_into_a_bound_class_is_a_conversion_error() -> None:
    scaling = xforms.Scaling([2.0, 3.0, 4.0], input=LPSmm(), output=LPSmm())
    with pytest.raises(ConversionError, match="not compatible"):
        scaling.to(VoxelToRAS)
    assert scaling.to(VoxelToRAS, error=False) is False


# ----------------------------------------------------------------------
#   NOTHING IS RELABELLED BETWEEN OTHER SYSTEMS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, cls",
    [
        (family, cls)
        for cls in FORMATS
        for family in sorted(_families(VOXEL, VOXEL))
        if (family, cls) not in RELABELLED
    ],
    ids=lambda x: getattr(x, "__name__", x),
)
def test_a_transformation_no_format_holds_is_refused(
    family: str, cls: type
) -> None:
    # Between two voxel spaces, a transformation is held by none of these
    # formats, whose spaces are RAS, LPS or their own.
    t = _families(VOXEL, VOXEL)[family]
    with pytest.raises(ConversionError):
        t.to(cls)
    assert t.to(cls, error=False) is False


@pytest.mark.parametrize(
    "family, cls",
    sorted(RELABELLED, key=lambda pair: (pair[0], pair[1].__name__)),
    ids=lambda x: getattr(x, "__name__", x),
)
def test_a_format_without_systems_relabels_between_voxel_spaces(
    family: str, cls: type
) -> None:
    # These formats read their systems from what they hold, so they have
    # none of their own that a transformation between voxel spaces would
    # contradict, and the transformation keeps its endpoints.
    t = _families(VOXEL, VOXEL)[family]
    for held in _entry_points(t, cls):
        assert type(held) is cls
        assert held.input == VOXEL
        assert held.output == VOXEL
        np.testing.assert_allclose(
            held.to(xforms.Affine).homogeneous_matrix,
            t.to(xforms.Affine).homogeneous_matrix,
        )


@pytest.mark.parametrize(
    "family, cls",
    [
        (family, cls)
        for cls in (ItkNiftiDisplacementField, M3zMorph)
        for family in ("sequence", "immutable-sequence")
    ],
    ids=lambda x: getattr(x, "__name__", x),
)
def test_a_format_without_an_exact_conversion_refuses(
    family: str, cls: type
) -> None:
    # A relabel would build a wrong object, such as a field format around
    # a chain from which it cannot read its own systems. The format may
    # hold some transformations of the family, and an exact converter
    # will decide which ones (#312).
    t = _families(RASmm(), RASmm())[family]
    with pytest.raises(ConversionError, match="#312"):
        t.to(cls)


@pytest.mark.parametrize(
    "family, cls",
    [
        (family, cls)
        for cls in (NiftyRegDisplacementField, FnirtWarpField)
        for family in ("sequence", "immutable-sequence")
    ],
    ids=lambda x: getattr(x, "__name__", x),
)
def test_a_format_without_an_exact_conversion_relabels_compatible_systems(
    family: str, cls: type
) -> None:
    # NiftyReg maps RAS to RAS, and FNIRT reads its systems from the chain
    # it holds, so a chain between RAS spaces is relabelled with the same
    # parameters until an exact converter is written (#312).
    t = _families(RASmm(), RASmm())[family]
    for held in _entry_points(t, cls):
        assert type(held) is cls
        assert list(held.transformations) == list(t.transformations)


@pytest.mark.parametrize("family", ["scaling", "linear", "affine", "sequence"])
@pytest.mark.parametrize("cls", [LtaTransformation, LtaTransformationRASToRAS])
def test_an_lta_relabels_an_affine_between_ras_spaces(
    family: str, cls: type
) -> None:
    # The affine that the transformation is, between RAS spaces, is what
    # an LTA of the RAS type holds.
    t = _families(RASmm(), RASmm())[family]
    for lta in _entry_points(t, cls):
        assert type(lta) is cls
        assert lta.input == RASmm()
        np.testing.assert_allclose(
            lta.homogeneous_matrix, t.to(xforms.Affine).homogeneous_matrix
        )


def test_an_lps_affine_is_not_relabelled_as_a_ras_lta() -> None:
    # An LTA of the RAS type maps RAS to RAS, so an affine between LPS
    # spaces would mean another map. It is refused by every entry point.
    t = _families(LPSmm(), LPSmm())["affine"]
    for convert in _conversions(LtaTransformationRASToRAS):
        with pytest.raises(ConversionError, match="not compatible"):
            convert(t)


def test_an_affine_without_systems_is_relabelled_as_an_lta() -> None:
    # An affine whose systems are not known is taken to map those of the
    # format, as `from_any` always took it, and `to` now takes it so.
    t = xforms.Affine(MATRIX[:-1])
    for lta in _entry_points(t, LtaTransformation):
        assert type(lta) is LtaTransformation
        np.testing.assert_array_equal(lta.matrix, MATRIX[:-1])


def test_a_chain_between_lps_spaces_is_not_relabelled_as_niftyreg() -> None:
    # NiftyReg maps RAS to RAS, so a chain between LPS spaces would mean
    # another map. It is refused by every entry point.
    t = _families(LPSmm(), LPSmm())["sequence"]
    for convert in _conversions(NiftyRegDisplacementField):
        with pytest.raises(ConversionError, match="not compatible"):
            convert(t)


@pytest.mark.parametrize(
    "cls", [NiftyRegDisplacementField, FnirtWarpField, X5Transform]
)
def test_a_chain_without_systems_is_relabelled(cls: type) -> None:
    # A chain whose systems are not known is taken to map the systems of
    # the format, as `from_any` always took it, and `to` now takes it so.
    t = _families(None, None)["sequence"]
    for held in _entry_points(t, cls):
        assert type(held) is cls
        assert list(held.transformations) == list(t.transformations)


@pytest.mark.parametrize(
    "t, cls",
    [
        (xforms.Affine(MATRIX[:-1]), NiftiVoxelToRAS),
        (xforms.Scaling([2.0, 3.0, 4.0]), NiftiRASToVoxel),
        (_families(RASmm(), RASmm())["affine"], NiftyRegAffine),
        (_families(None, None)["displacement"], NiftiRASCoordinatesField),
    ],
    ids=["nifti", "nifti-scaling", "niftyreg", "nifti-field"],
)
def test_an_exact_converter_is_chosen_over_relabelling(
    t: xforms.Transformation, cls: type
) -> None:
    # The converter of a format names its format and the families it
    # converts, so it is more specific than the converter that relabels.
    assert not relabels(t, cls)
    assert find_converter(t, cls) is not None


def test_a_format_without_an_exact_converter_is_relabelled() -> None:
    chain = _families(None, None)["sequence"]
    assert relabels(chain, NiftyRegDisplacementField)
    assert relabels(chain, X5Transform)
    assert not relabels(chain, NiftiRASDisplacementField)


def test_a_format_family_is_not_relabelled() -> None:
    from brainhops.io.transformations.niftyreg import NiftyRegField

    chain = _families(None, None)["sequence"]
    with pytest.raises(ConversionError, match="family of formats"):
        relabel(chain, NiftyRegField)


@pytest.mark.parametrize("family", ["scaling", "affine", "sequence"])
def test_flirt_says_why_it_refuses(family: str) -> None:
    t = _families(RASmm(), RASmm())[family]
    with pytest.raises(ConversionError, match="moving and a reference image"):
        t.to(FlirtTransform)


# ----------------------------------------------------------------------
#   FORMATS WITHOUT SYSTEMS OF THEIR OWN
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", ["scaling", "affine", "sequence"])
def test_a_bare_matrix_holds_a_relabelled_affine(family: str) -> None:
    # A bare matrix is given its conventions when it is read, so it keeps
    # the endpoints of the affine, and the map is unchanged.
    t = _families(LPSmm(), LPSmm())[family]
    matrix = t.to(TxtMatrixAffine)
    assert type(matrix) is TxtMatrixAffine
    assert matrix.input == LPSmm()
    np.testing.assert_array_equal(
        matrix.homogeneous_matrix, t.to(xforms.Affine).homogeneous_matrix
    )


def test_an_itk_chain_holds_a_relabelled_chain() -> None:
    chain = _families(LPSmm(), LPSmm())["sequence"]
    itk = chain.to(TfmTransform)
    assert type(itk) is TfmTransform
    assert list(itk.transformations) == list(chain.transformations)


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
def test_an_affine_of_unknown_systems_maps_niftyreg_ones(family: str) -> None:
    # Unknown endpoints are compatible with RAS, so they are taken to be
    # the endpoints of NiftyReg, as NIfTI takes them to be its own.
    t = _families(None, None)[family]
    niftyreg = t.to(NiftyRegAffine)
    np.testing.assert_array_equal(
        niftyreg.homogeneous_matrix, t.to(xforms.Affine).homogeneous_matrix
    )


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
        (
            LtaTransformationVoxToVox(matrix=MATRIX[:-1]),
            LtaTransformationRASToRAS,
        ),
    ],
    ids=["lta"],
)
def test_another_variant_of_a_format_is_not_relabelled(
    t: xforms.Transformation, cls: type
) -> None:
    with pytest.raises(ConversionError):
        t.to(cls)


def test_another_container_of_an_itk_chain_holds_it() -> None:
    # The chain has no systems, so it is relabelled into the other
    # container, which holds the same chain.
    tfm = TfmTransform([xforms.Scaling([1.0, 2.0, 3.0])])
    for h5 in _entry_points(tfm, H5Transform):
        assert type(h5) is H5Transform
        assert list(h5.transformations) == list(tfm.transformations)


def test_another_container_of_a_bare_matrix_holds_it() -> None:
    csv = TxtMatrixAffine(matrix=MATRIX[:-1]).to(CsvMatrixAffine)
    assert type(csv) is CsvMatrixAffine
    np.testing.assert_array_equal(csv.matrix, MATRIX[:-1])


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _conversions(cls: type) -> tx.List[tx.Callable[[tx.Any], tx.Any]]:
    """Return the three entry points that convert to `cls`."""
    return [lambda t: t.to(cls), cls.from_any, cls.from_instance]


def _entry_points(t: xforms.Transformation, cls: type) -> tx.List[tx.Any]:
    """Return `t` converted to `cls` by each of the three entry points."""
    return [convert(t) for convert in _conversions(cls)]
