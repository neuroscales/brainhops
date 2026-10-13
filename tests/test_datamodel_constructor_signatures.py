"""Tests pinning the positional parameters of every transformation and image.

Fields are collected in MRO order, so their position in `__init__`
follows the order of the bases, and reordering the bases once made
`TfmTransform([scaling])` read the list as the input coordinate system.
Fields are therefore keyword-only by default, and only the defining
fields, such as `data` or the parsed file of a reader, are positional.
An image is defined by its `data` (or its `images`, for a pyramid) and
its `transformations`, which are positional in that order. The opened
Zarr `node` of the OME-Zarr formats, which a temporary parser of
`brainhops.io.common.zarr` contributes until they follow the design of
format records, stays positional, as it does on the transformations. An
intended change of signature must update `POSITIONAL` or `IMAGES`.
"""

import importlib
import inspect

import numpy as np
import pytest
import typing_extensions as tx

# Import every family so that the subclass walk finds all of them.
import brainhops.datamodel.geometry  # noqa: F401
import brainhops.io.images  # noqa: F401
import brainhops.io.transformations  # noqa: F401
from brainhops.datamodel.images import Image, SingleScaleImage
from brainhops.datamodel.systems import RASmm
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Sequence,
    Transformation,
)
from brainhops.io.images.base import ImageFormat

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
    "brainhops.datamodel._transformations.derived": {
        "DerivedTransformation": ("forward",),
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
    "brainhops.io.transformations.afni._affine": {
        "AfniAffine": ("data",),
    },
    "brainhops.io.transformations.afni._nifti": {
        "AfniNiftiWarp": ("transformations",),
    },
    "brainhops.io.transformations.afni._warp": {
        "AfniBrikWarp": ("transformations",),
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
    "brainhops.io.transformations.c3d._affine": {
        "C3dAffine": ("data",),
    },
    "brainhops.io.transformations.elastix._xform": {
        "ElastixParameterTransform": ("transformations",),
        "ElastixTomlTransform": ("transformations",),
        "ElastixTransform": ("transformations",),
    },
    "brainhops.io.transformations.bdv._xform": {
        "BdvViewRegistration": ("transformations",),
    },
    "brainhops.io.transformations.freesurfer.lta._xforms": {
        "LtaTransformation": ("data",),
        "LtaTransformationPhysToPhys": ("data",),
        "LtaTransformationRASToRAS": ("data",),
        "LtaTransformationVoxToVox": ("data",),
    },
    "brainhops.io.transformations.trakem2._xform": {
        "TrakEM2PatchAffine": ("data",),
    },
    "brainhops.io.transformations.freesurfer.m3z._xform": {
        "M3zMorph": ("transformations",),
    },
    "brainhops.io.transformations.freesurfer.regdat._xforms": {
        "RegisterDatTransformation": ("data",),
        "RegisterDatTransformationRASToRAS": ("data",),
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
    "brainhops.io.transformations.minc._affine": {
        "XfmAffine": ("data",),
    },
    "brainhops.io.transformations.minc._chain": {
        "XfmTransform": ("transformations",),
    },
    "brainhops.io.transformations.minc._grid": {
        "XfmGridTransform": ("transformations",),
    },
    "brainhops.io.transformations.mrtrix._linear": {
        "MrtrixLinearTransform": ("data",),
    },
    "brainhops.io.transformations.mrtrix._warps": {
        "MrtrixCoordinatesField": ("transformations",),
        "MrtrixDisplacementField": ("transformations",),
        "MrtrixWarp": (),
        "MrtrixWarpFull": ("transformations",),
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

# Positional parameters of each public image, in order, keyed by the
# defining module. All other parameters are keyword-only.
IMAGES: tx.Dict[str, tx.Dict[str, tx.Tuple[str, ...]]] = {
    "brainhops.datamodel.images": {
        "Image": (),
        "MultiScaleImage": ("images", "transformations"),
        "SingleScaleImage": ("data", "transformations"),
    },
    "brainhops.io.images.afni._image": {
        "AfniImage": ("data", "transformations"),
    },
    "brainhops.io.images.base._base": {
        "ImageFormat": (),
    },
    "brainhops.io.images.bdv._image": {
        "BdvImage": ("data", "transformations"),
    },
    "brainhops.io.images.bdv._multiscale": {
        "BdvMultiScaleImage": ("images", "transformations"),
    },
    "brainhops.io.images.freesurfer.mgh._image": {
        "MghImage": ("data", "transformations"),
    },
    "brainhops.io.images.imaris._image": {
        "ImarisImage": ("data", "transformations"),
    },
    "brainhops.io.images.imaris._multiscale": {
        "ImarisMultiScaleImage": ("images", "transformations"),
    },
    "brainhops.io.images.jpeg2000._image": {
        "Jpeg2000Image": ("data", "transformations"),
    },
    "brainhops.io.images.jpeg2000._multiscale": {
        "Jpeg2000MultiScaleImage": ("images", "transformations"),
    },
    "brainhops.io.images.minc._image": {
        name: ("data", "transformations")
        for name in ("Minc1Image", "Minc2Image", "MincImage")
    },
    "brainhops.io.images.mrtrix._image": {
        "MrtrixImage": ("data", "transformations"),
    },
    "brainhops.io.images.n5._image": {
        "N5Image": ("data", "transformations"),
    },
    "brainhops.io.images.n5._multiscale": {
        "N5MultiScaleImage": ("images", "transformations"),
    },
    "brainhops.io.images.nifti._image": {
        "NiftiImage": ("data", "transformations"),
    },
    "brainhops.io.images.nrrd._image": {
        name: ("data", "transformations")
        for name in ("AttachedNrrdImage", "DetachedNrrdImage", "NrrdImage")
    },
    "brainhops.io.images.openslide._base": {
        "OpenSlideImage": ("data", "transformations"),
        "OpenSlideMultiScaleImage": ("images", "transformations"),
    },
    "brainhops.io.images.openslide._formats": {
        **{
            f"{vendor}Image": ("data", "transformations")
            for vendor in (
                "Aperio",
                "DicomWsi",
                "GenericTiff",
                "Hamamatsu",
                "Leica",
                "Mirax",
                "Philips",
                "Sakura",
                "Trestle",
                "Ventana",
                "Zeiss",
            )
        },
        **{
            f"{vendor}MultiScaleImage": ("images", "transformations")
            for vendor in (
                "Aperio",
                "DicomWsi",
                "GenericTiff",
                "Hamamatsu",
                "Leica",
                "Mirax",
                "Philips",
                "Sakura",
                "Trestle",
                "Ventana",
                "Zeiss",
            )
        },
    },
    "brainhops.io.images.pillow._image": {
        "PillowImage": ("data", "transformations"),
    },
    "brainhops.io.images.stack._image": {
        "StackImage": ("data", "transformations"),
    },
    "brainhops.io.images.precomputed._image": {
        "PrecomputedImage": ("data", "transformations"),
    },
    "brainhops.io.images.precomputed._multiscale": {
        "PrecomputedMultiScaleImage": ("images", "transformations"),
    },
    "brainhops.io.images.tiff._image": {
        "TiffImage": ("data", "transformations"),
    },
    "brainhops.io.images.tiff._multiscale": {
        "TiffMultiScaleImage": ("images", "transformations"),
    },
    "brainhops.io.images.zarr._image": {
        "ZarrImage": ("data", "transformations"),
    },
    "brainhops.io.images.zarr._multiscale": {
        "OmeZarrImage": ("images", "transformations", "node"),
        "OmeZarrLevel": ("data", "transformations"),
    },
}

PINNED_IMAGES = [
    (module, name) for module in IMAGES for name in IMAGES[module]
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


# Every image class defined by the package, but none defined by tests.
ALL_IMAGES = sorted(
    (
        cls
        for cls in {Image, ImageFormat}.union(
            _subclasses(Image), _subclasses(ImageFormat)
        )
        if cls.__module__.startswith("brainhops.")
    ),
    key=lambda cls: (cls.__module__, cls.__qualname__),
)

PUBLIC_IMAGES = [cls for cls in ALL_IMAGES if not cls.__name__.startswith("_")]


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
    # A subclass that declares an endpoint again inherits `kw_only=True`,
    # unless its first base is a file reader that is not keyword-only, in
    # which case the subclass must write `KwOnly[...]` itself.
    kinds = {p.name: p.kind for p in _parameters(cls)}
    for name in KEYWORD_ONLY:
        if name in kinds:
            assert kinds[name] is inspect.Parameter.KEYWORD_ONLY, name


def test_a_positional_endpoint_is_refused() -> None:
    with pytest.raises(TypeError):
        Affine(np.eye(4), RASmm())


def test_a_field_declared_again_inherits_kw_only() -> None:
    # The formats rely on this: they declare `raw` and `_metadata` without
    # `KwOnly[...]`, and some redeclare an endpoint with a new default.
    from bagof.magic import Magic

    class Base(Magic, kw_only=True):
        _input: tx.Optional[int] = None

    class Redeclared(Base):
        _input: tx.Optional[int] = 1
        extra: tx.Optional[int] = None

    kinds = {p.name: p.kind for p in _parameters(Redeclared)}
    assert kinds["input"] is inspect.Parameter.KEYWORD_ONLY
    assert kinds["extra"] is inspect.Parameter.KEYWORD_ONLY


def test_a_list_of_transformations_is_the_first_argument() -> None:
    tfm = pytest.importorskip("brainhops.io.transformations.itk.tfm")
    scaling = Scaling(scale=np.array([2.0, 3.0, 4.0]))
    for cls in (Sequence, tfm.TfmTransform):
        xform = cls([scaling])
        assert xform.input is None
        assert len(xform.transformations) == 1


# --- images -----------------------------------------------------------


@pytest.mark.parametrize(
    "module, name",
    PINNED_IMAGES,
    ids=[f"{m}.{n}" for m, n in PINNED_IMAGES],
)
def test_the_positional_parameters_of_an_image_are_pinned(
    module: str, name: str
) -> None:
    try:
        mod = importlib.import_module(module)
    except ImportError as e:  # An optional dependency is missing.
        pytest.skip(str(e))
    assert _positional(getattr(mod, name)) == IMAGES[module][name]


@pytest.mark.parametrize("cls", PUBLIC_IMAGES, ids=_id)
def test_every_public_image_is_pinned(cls: type) -> None:
    assert cls.__name__ in IMAGES.get(cls.__module__, {}), (
        f"{_id(cls)} is missing from IMAGES; its positional "
        f"parameters are {_positional(cls)!r}."
    )


def test_a_field_of_an_image_subclass_is_keyword_only() -> None:
    # The `kw_only=True` option of `Image` is inherited, so a field that a
    # subclass declares without `KwOnly[...]` is keyword-only, and so is a
    # field declared again without `NotKwOnly`.
    class Labelled(SingleScaleImage):
        label: str = ""

    class Redeclared(SingleScaleImage):
        transformations: tx.List[Transformation] = ()

    assert _positional(Labelled) == ("data", "transformations")
    assert _positional(Redeclared) == ("data",)
    kinds = {p.name: p.kind for p in _parameters(Labelled)}
    assert kinds["label"] is inspect.Parameter.KEYWORD_ONLY


def test_a_positional_metadata_field_is_refused() -> None:
    tiff = pytest.importorskip("brainhops.io.images.tiff")
    with pytest.raises(TypeError):
        tiff.TiffImage(np.zeros((2, 2)), [], "ome")
