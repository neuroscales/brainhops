"""Tests pinning the positional parameters of every transformation.

Fields are collected in MRO order, so their position in `__init__`
follows the order of the bases, and reordering the bases once made
`TfmTransform([scaling])` read the list as the input coordinate system.
Fields are therefore keyword-only by default, and only the defining
fields, such as `data` or the parsed file of a reader, are positional.
An intended change of signature must update `POSITIONAL`.
"""

import importlib
import inspect

import numpy as np
import pytest
import typing_extensions as tx

# Import every family so that the subclass walk finds all of them.
import brainhops.datamodel.geometry  # noqa: F401
import brainhops.io.transformations  # noqa: F401
from brainhops.datamodel.systems import RASmm
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Sequence,
    Transformation,
)

# Positional parameters of each public transformation, in order, keyed by
# the defining module. All other parameters are keyword-only.
POSITIONAL: tx.Dict[str, tx.Dict[str, tx.Tuple[str, ...]]] = {
    "brainhops.datamodel._transformations.concrete": {
        "Affine": ("data",),
        "CartesianField": ("shape",),
        "ConcreteTransformation": (),
        "CoordinatesField": ("data",),
        "DisplacementField": ("data",),
        "Identity": (),
        "Linear": ("data",),
        "Permutation": ("data",),
        "Rotation": ("data",),
        "Scaling": ("data",),
        "TransformationField": ("data",),
        "Translation": ("data",),
    },
    "brainhops.datamodel._transformations.tangents": {
        "AffineExponential": ("data",),
        "LinearExponential": ("data",),
        "RotationExponential": ("data",),
        "ScalingExponential": ("data",),
        "StationaryVelocityField": ("data",),
    },
    "brainhops.datamodel._transformations.inverse": {
        "Inverse": ("forward",),
        "InverseAffine": ("forward",),
        "InverseAffineExponential": ("forward",),
        "InverseCoordinatesField": ("forward",),
        "InverseDisplacementField": ("forward",),
        "InverseLinear": ("forward",),
        "InverseLinearExponential": ("forward",),
        "InversePermutation": ("forward",),
        "InverseRotation": ("forward",),
        "InverseRotationExponential": ("forward",),
        "InverseScaling": ("forward",),
        "InverseScalingExponential": ("forward",),
        "InverseStationaryVelocityField": ("forward",),
        "InverseTranslation": ("forward",),
    },
    "brainhops.datamodel._transformations.meta": {
        "Bijection": ("forward", "backward"),
        "MetaTransformation": (),
        "Projection": ("dropped", "created"),
        "SubspaceTransformation": (
            "transformation",
            "input_axes",
            "output_axes",
        ),
    },
    "brainhops.datamodel._transformations.multiscale": {
        "MultiscaleField": ("scales",),
    },
    "brainhops.datamodel._transformations.operators": {
        "Operation": ("forward",),
        "Sqrt": ("forward",),
        "SqrtAffine": ("forward",),
        "SqrtLinear": ("forward",),
        "SqrtRotation": ("forward",),
        "SqrtScaling": ("forward",),
        "SqrtTranslation": ("forward",),
    },
    "brainhops.datamodel._transformations.sequence": {
        "ImmutableSequence": ("transformations",),
        "MutableSequence": ("transformations",),
        "Sequence": ("transformations",),
    },
    "brainhops.datamodel.geometry": {
        "Geometry": ("transformations", "shape", "grid", "transformation"),
    },
    "brainhops.io.transformations.base.affines": {
        "LPSToVoxel": ("data",),
        "RASToRAS": ("data",),
        "RASToVoxel": ("data",),
        "VoxelToLPS": ("data",),
        "VoxelToRAS": ("data",),
    },
    "brainhops.io.transformations.base.fields": {
        "LPSCoordinatesField": ("data",),
        "RASCoordinatesField": ("data",),
    },
    "brainhops.io.transformations.elastix._xform": {
        "ElastixParameterTransform": ("transformations",),
        "ElastixTomlTransform": ("transformations",),
        "ElastixTransform": ("transformations",),
    },
    "brainhops.io.transformations.freesurfer.lta._xforms": {
        "LtaTransformation": ("data",),
        "LtaTransformationPhysToPhys": ("data",),
        "LtaTransformationRASToRAS": ("data",),
        "LtaTransformationVoxToVox": ("data",),
    },
    "brainhops.io.transformations.freesurfer.m3z._xform": {
        "M3zMorph": ("transformations", "struct"),
    },
    "brainhops.io.transformations.fsl._affines": {
        "ScaledMmToScaledMm": ("data",),
        "ScaledMmToVoxel": ("data",),
        "VoxelToScaledMm": ("data",),
    },
    "brainhops.io.transformations.fsl._fields": {
        "RASToWarpField": ("data",),
        "WarpFieldToRAS": ("data",),
    },
    "brainhops.io.transformations.fsl.flirt._xform": {
        "FlirtTransform": ("flirt_matrix", "moving", "reference"),
    },
    "brainhops.io.transformations.fsl.fnirt._base": {
        "FnirtWarpField": ("transformations",),
    },
    "brainhops.io.transformations.itk._common": {
        "ItkAffineBase": ("transformations",),
        "ItkAffineStruct": ("transformations",),
        "ItkBSplineStruct": ("transformations",),
        "ItkBlockBase": ("transformations",),
        "ItkDisplacementBase": ("transformations",),
        "ItkDisplacementFieldStruct": ("transformations",),
        "ItkEuler2DStruct": ("transformations",),
        "ItkEuler3DStruct": ("transformations",),
        "ItkIdentityStruct": ("transformations",),
        "ItkMatrixOffsetStruct": ("transformations",),
        "ItkScaleLogarithmicStruct": ("transformations",),
        "ItkScaleSkewVersor3DStruct": ("transformations",),
        "ItkScaleStruct": ("transformations",),
        "ItkScaleVersor3DStruct": ("transformations",),
        "ItkSimilarity2DStruct": ("transformations",),
        "ItkSimilarity3DStruct": ("transformations",),
        "ItkTranslationStruct": ("transformations",),
        "ItkVersorRigid3DStruct": ("transformations",),
        "ItkVersorStruct": ("transformations",),
    },
    "brainhops.io.transformations.itk._xform": {
        "ItkTransform": ("transformations",),
    },
    "brainhops.io.transformations.itk.h5._xform": {
        "H5Transform": ("transformations", "file", "header"),
    },
    "brainhops.io.transformations.itk.mat._xform": {
        "MatTransform": ("transformations",),
    },
    "brainhops.io.transformations.itk.nifti._fields": {
        "ItkNiftiCoordinatesField": ("transformations",),
        "ItkNiftiDisplacementField": ("transformations",),
        "ItkNiftiField": ("transformations",),
    },
    "brainhops.io.transformations.itk.tfm._xform": {
        "TfmTransform": ("transformations",),
    },
    "brainhops.io.transformations.matrix._xform": {
        name: ("data",)
        for name in (
            "CsvMatrixAffine",
            "Mat73MatrixAffine",
            "MatLegacyMatrixAffine",
            "MatMatrixAffine",
            "MatrixAffine",
            "NpyMatrixAffine",
            "NpzMatrixAffine",
            "TsvMatrixAffine",
            "TxtMatrixAffine",
        )
    },
    "brainhops.io.transformations.nifti._affines": {
        "NiftiRASToVoxel": ("data",),
        "NiftiVoxelToRAS": ("data",),
    },
    "brainhops.io.transformations.nifti._base": {
        "NiftiBasedTransformation": (),
    },
    "brainhops.io.transformations.nifti._fields": {
        "NiftiRASCoordinatesField": ("data",),
        "NiftiRASDisplacementField": ("transformations",),
    },
    "brainhops.io.transformations.niftyreg._affine": {
        "NiftyRegAffine": ("data",),
    },
    "brainhops.io.transformations.niftyreg._fields": {
        # The fields of the format precede `transformations`.
        "NiftyRegControlPointGrid": ("transformations",),
        "NiftyRegDeformationField": ("transformations",),
        "NiftyRegDisplacementField": ("transformations",),
        "NiftyRegField": (),
        "NiftyRegSequence": ("transformations",),
        "NiftyRegVelocity": ("transformations",),
        "NiftyRegVelocityField": ("transformations",),
        "NiftyRegVelocityGrid": ("transformations",),
    },
    "brainhops.io.transformations.spm._fields": {
        "SpmCoordinatesField": ("transformations",),
    },
    "brainhops.io.transformations.x5._blocks": {
        "X5BSplineField": ("transformations",),
        "X5CoordinatesField": ("transformations",),
        "X5DisplacementField": ("transformations",),
    },
    "brainhops.io.transformations.x5._xform": {
        "X5Transform": ("transformations",),
    },
    "brainhops.io.transformations.zarr._xforms": {
        "OmeZarrField": (
            "node",
            "raw_levels",
            "voxel2world",
            "level_transforms",
            "axes",
            "ome",
        ),
    },
}

PINNED = [
    (module, name) for module in POSITIONAL for name in POSITIONAL[module]
]

# Parameters that are keyword-only on every transformation.
KEYWORD_ONLY = ("input", "output")


def _subclasses(cls: type) -> tx.List[type]:
    out, todo = [], list(cls.__subclasses__())
    while todo:
        sub = todo.pop()
        if sub not in out:
            out.append(sub)
            todo.extend(sub.__subclasses__())
    return out


# Every transformation class defined by the package, but none defined by
# tests.
ALL = sorted(
    (
        cls
        for cls in _subclasses(Transformation)
        if cls.__module__.startswith("brainhops.")
    ),
    key=lambda cls: (cls.__module__, cls.__qualname__),
)

# A parameterized class such as `Inverse[Affine]` is reached through its
# origin class.
PUBLIC = [
    cls
    for cls in ALL
    if not cls.__name__.startswith("_") and cls.__name__.isidentifier()
]


def _parameters(cls: type) -> tx.List[inspect.Parameter]:
    return list(inspect.signature(cls.__init__).parameters.values())[1:]


def _positional(cls: type) -> tx.Tuple[str, ...]:
    return tuple(
        p.name
        for p in _parameters(cls)
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


def _id(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


@pytest.mark.parametrize(
    "module, name", PINNED, ids=[f"{m}.{n}" for m, n in PINNED]
)
def test_the_positional_parameters_are_pinned(module: str, name: str) -> None:
    try:
        mod = importlib.import_module(module)
    except ImportError as e:  # An optional dependency is missing.
        pytest.skip(str(e))
    assert _positional(getattr(mod, name)) == POSITIONAL[module][name]


@pytest.mark.parametrize("cls", PUBLIC, ids=_id)
def test_every_public_transformation_is_pinned(cls: type) -> None:
    assert cls.__name__ in POSITIONAL.get(cls.__module__, {}), (
        f"{_id(cls)} is missing from POSITIONAL; its positional "
        f"parameters are {_positional(cls)!r}."
    )


@pytest.mark.parametrize("cls", ALL, ids=_id)
def test_the_endpoints_are_keyword_only(cls: type) -> None:
    # A subclass that redeclares an endpoint with its own default must also
    # write `KwOnly[...]`, which is not inherited.
    kinds = {p.name: p.kind for p in _parameters(cls)}
    for name in KEYWORD_ONLY:
        if name in kinds:
            assert kinds[name] is inspect.Parameter.KEYWORD_ONLY, name


def test_a_positional_endpoint_is_refused() -> None:
    with pytest.raises(TypeError):
        Affine(np.eye(4), RASmm())


def test_a_list_of_transformations_is_the_first_argument() -> None:
    tfm = pytest.importorskip("brainhops.io.transformations.itk.tfm")
    scaling = Scaling(scale=np.array([2.0, 3.0, 4.0]))
    for cls in (Sequence, tfm.TfmTransform):
        xform = cls([scaling])
        assert xform.input is None
        assert len(xform.transformations) == 1
