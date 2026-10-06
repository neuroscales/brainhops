"""
Regression tests: the positional parameters of every transformation.

bagof collects the fields of a class in MRO order, so the place of a
parameter in `__init__` follows the order of the bases. Reordering them
once turned the ITK formats' `(transformations, input, output)` into
`(input, output, transformations)`, and `TfmTransform([scaling])` then
silently read the list as the input coordinate system.

The endpoints are therefore keyword-only on every transformation, and
the positional parameters of each public class are pinned here: a change
to a base that moves them fails this file instead of reaching users. An
intended change updates `POSITIONAL`.
"""

import importlib
import inspect

import numpy as np
import pytest
import typing_extensions as tx

# Import every family, so that walking the subclasses finds them all.
import brainhops.datamodel.geometry  # noqa: F401
import brainhops.io.transformations  # noqa: F401
from brainhops.datamodel.systems import RASmm
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Sequence,
    Transformation,
)

# The positional parameters of each public transformation class, in
# order, keyed by the module that defines the class. Every other
# parameter, `input` and `output` included, is keyword-only.
POSITIONAL: tx.Dict[str, tx.Dict[str, tx.Tuple[str, ...]]] = {
    "brainhops.datamodel._transformations.concrete": {
        "Affine": ("data",),
        "AffineExponential": ("data",),
        "CartesianField": ("shape", "degree", "bound", "coeff"),
        "ConcreteTransformation": (),
        "CoordinatesField": ("data", "degree", "bound", "coeff"),
        "DisplacementField": ("data", "degree", "bound", "coeff"),
        "Identity": (),
        "Linear": ("data",),
        "LinearExponential": ("data",),
        "Permutation": ("data",),
        "Rotation": ("data",),
        "RotationExponential": ("data",),
        "Scaling": ("data",),
        "ScalingExponential": ("data",),
        "StationaryVelocityField": ("data", "degree", "bound", "coeff"),
        "TransformationField": ("data", "degree", "bound", "coeff"),
        "Translation": ("data",),
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
    "brainhops.io.transformations.base._base": {
        "FileBasedTransformation": (),
        "WritableFileBasedTransformation": (),
    },
    "brainhops.io.transformations.base.affines": {
        "LPSToVoxel": ("data",),
        "RASToRAS": ("data",),
        "RASToVoxel": ("data",),
        "VoxelToLPS": ("data",),
        "VoxelToRAS": ("data",),
    },
    "brainhops.io.transformations.base.fields": {
        "LPSCoordinatesField": ("data", "degree", "bound", "coeff"),
        "RASCoordinatesField": ("data", "degree", "bound", "coeff"),
    },
    "brainhops.io.transformations.elastix._xform": {
        "ElastixParameterTransform": (
            "transformations",
            "parameter_map",
            "initial",
        ),
        "ElastixTomlTransform": (
            "transformations",
            "parameter_map",
            "initial",
        ),
        "ElastixTransform": ("transformations", "parameter_map", "initial"),
    },
    "brainhops.io.transformations.freesurfer.lta._xforms": {
        "LtaTransformation": ("data", "struct"),
        "LtaTransformationPhysToPhys": ("data", "struct"),
        "LtaTransformationRASToRAS": ("data", "struct"),
        "LtaTransformationVoxToVox": ("data", "struct"),
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
        # The format's own fields come before `transformations` here.
        "FnirtWarpField": (
            "moving",
            "reference",
            "deformation_type",
            "image",
            "header",
            "transformations",
        ),
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
        "ItkNiftiCoordinatesField": ("transformations", "image", "header"),
        "ItkNiftiDisplacementField": ("transformations", "image", "header"),
        "ItkNiftiField": ("transformations", "image", "header"),
    },
    "brainhops.io.transformations.itk.tfm._xform": {
        "TfmTransform": ("transformations",),
    },
    "brainhops.io.transformations.matrix._xform": {
        name: (
            "raw_matrix",
            "variable",
            "vector",
            "direction",
            "index_base",
            "data",
        )
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
    "brainhops.io.transformations.nifti.affines": {
        "NiftiRASToVoxel": ("data", "image", "header"),
        "NiftiVoxelToRAS": ("data", "image", "header"),
    },
    "brainhops.io.transformations.nifti.base": {
        "NiftiBasedTransformation": ("image", "header"),
    },
    "brainhops.io.transformations.nifti.fields": {
        "NiftiRASCoordinatesField": (
            "data",
            "degree",
            "bound",
            "coeff",
            "image",
            "header",
        ),
        "NiftiRASDisplacementField": ("transformations", "image", "header"),
    },
    "brainhops.io.transformations.niftyreg._affine": {
        "NiftyRegAffine": ("data",),
    },
    "brainhops.io.transformations.niftyreg._fields": {
        # The format's own fields come before `transformations` here.
        "NiftyRegControlPointGrid": ("image", "header", "transformations"),
        "NiftyRegDeformationField": ("image", "header", "transformations"),
        "NiftyRegDisplacementField": ("image", "header", "transformations"),
        "NiftyRegField": ("image", "header"),
        "NiftyRegSequence": ("image", "header", "transformations"),
        "NiftyRegVelocity": ("image", "header", "transformations"),
        "NiftyRegVelocityField": ("image", "header", "transformations"),
        "NiftyRegVelocityGrid": ("image", "header", "transformations"),
    },
    "brainhops.io.transformations.spm.y": {
        "SpmCoordinatesField": ("transformations", "image", "header"),
    },
    "brainhops.io.transformations.x5._blocks": {
        "X5BSplineField": ("transformations",),
        "X5CoordinatesField": ("transformations",),
        "X5DisplacementField": ("transformations",),
    },
    "brainhops.io.transformations.x5._xform": {
        "X5Transform": (
            "transformations",
            "header",
            "nodes",
            "chain",
            "position",
            "file",
        ),
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

# The parameters that are keyword-only on every transformation.
KEYWORD_ONLY = ("input", "output")


def _subclasses(cls: type) -> tx.List[type]:
    out, todo = [], list(cls.__subclasses__())
    while todo:
        sub = todo.pop()
        if sub not in out:
            out.append(sub)
            todo.extend(sub.__subclasses__())
    return out


# Every transformation class that the package defines and that the
# imports above register. The classes that tests define are left out.
ALL = sorted(
    (
        cls
        for cls in _subclasses(Transformation)
        if cls.__module__.startswith("brainhops.")
    ),
    key=lambda cls: (cls.__module__, cls.__qualname__),
)

# The classes whose name is public. A parameterized class
# (`Inverse[Affine]`) is reached through the class it parameterizes.
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
    except ImportError as e:  # an optional dependency is missing
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
    # bagof does not carry `KwOnly` over to a field that a subclass
    # declares again, so a subclass that gives an endpoint a default of
    # its own has to write `KwOnly[...]` too.
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
