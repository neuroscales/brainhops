"""Decoding of elastix transforms into ITK blocks, and encoding back.

The transforms of elastix, which elastix calls "Advanced" transforms, are
reimplementations of ITK transforms. They have the same parameters as the
ITK transforms and act on LPS coordinates. Each elastix transform is
therefore decoded into a block of [`brainhops.io.transformations.itk`][],
that is, the representation of one ITK transform as a file stores it.
Only the details that are specific to elastix are translated here: the
name under which the center of rotation is stored, direction matrices that
are stored column by column, and B-spline grids that start at a non-zero
index.
"""

import math

import numpy as np
import scipy.linalg
import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.geometry import Geometry
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserNotImplementedError,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base.affines import VoxelToLPS
from brainhops.io.transformations.itk._common import (
    ItkBSplineStruct,
    ItkStruct,
    ItkTransformClass,
)
from brainhops.io.transformations.itk._systems import _make_system

from ._parser import ParameterMap, get_bool, get_floats, get_string

_ITKT = ItkTransformClass

#: The elastix transforms that this reader decodes.
SUPPORTED = (
    "TranslationTransform",
    "EulerTransform",
    "SimilarityTransform",
    "AffineTransform",
    "AffineLogTransform",
    "AffineDTITransform",
    "BSplineTransform",
    "RecursiveBSplineTransform",
)


# ----------------------------------------------------------------------
#   B-SPLINE BLOCKS
# ----------------------------------------------------------------------
# An ITK BSplineTransform is cubic, while elastix reads the degree from
# `BSplineTransformSplineOrder`, so this module defines one block class
# per degree.


class _ElastixBSplineStruct(ItkBSplineStruct):
    """elastix B-spline block of the default degree, 3."""


class _ElastixLinearBSplineStruct(_ElastixBSplineStruct):
    """elastix B-spline block of degree 1."""

    degree: tx.ClassVar[int] = 1


class _ElastixQuadraticBSplineStruct(_ElastixBSplineStruct):
    """elastix B-spline block of degree 2."""

    degree: tx.ClassVar[int] = 2


_BSPLINES = {
    1: _ElastixLinearBSplineStruct,
    2: _ElastixQuadraticBSplineStruct,
    3: _ElastixBSplineStruct,
}


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def ndim_of(pmap: ParameterMap) -> int:
    """Return the number of dimensions that the transform maps."""
    for name in ("FixedImageDimension", "MovingImageDimension"):
        value = get_string(pmap, name)
        if value is not None:
            return int(value)
    for name in ("Size", "Spacing", "Origin", "CenterOfRotationPoint"):
        if name in pmap:
            return len(pmap[name])
    raise ParserContentError(
        "This elastix parameter map does not say how many dimensions it "
        "has (no FixedImageDimension)."
    )


def direction_of(pmap: ParameterMap, name: str, ndim: int) -> np.ndarray:
    """Read a direction matrix, which elastix stores column by column.

    elastix writes and reads the direction cosines column by column
    (`Conversion::ToVectorOfStrings`), which is the transpose of the
    row-major order of ITK.
    """
    values = get_floats(pmap, name)
    if values is None:
        return np.eye(ndim)
    if values.size != ndim * ndim:
        raise ParserContentError(
            f"The elastix parameter {name} holds {values.size} values, "
            f"not {ndim * ndim}."
        )
    return values.reshape(ndim, ndim).T


def _vector(
    pmap: ParameterMap, name: str, ndim: int, default: float = None
) -> tx.Optional[np.ndarray]:
    values = get_floats(pmap, name)
    if values is None:
        if default is None:
            return None
        return np.full(ndim, float(default))
    if values.size != ndim:
        raise ParserContentError(
            f"The elastix parameter {name} holds {values.size} values, "
            f"not {ndim}."
        )
    return values


def fixed_geometry(pmap: ParameterMap) -> tx.Optional[Geometry]:
    """Return the geometry of the fixed image onto which transformix resamples.

    The voxel-to-LPS affine places voxel zero at the first voxel of the
    largest possible region. The affine therefore includes the start
    `Index`, which is almost always zero.
    """
    if "Size" not in pmap:
        return None
    ndim = ndim_of(pmap)
    shape = tuple(int(round(v)) for v in _vector(pmap, "Size", ndim))
    index = _vector(pmap, "Index", ndim, 0)
    spacing = _vector(pmap, "Spacing", ndim, 1)
    origin = _vector(pmap, "Origin", ndim, 0)
    direction = direction_of(pmap, "Direction", ndim)
    linear = direction @ np.diag(spacing)
    matrix = np.concatenate(
        [linear, (origin + linear @ index)[:, None]], axis=1
    )
    if ndim == 3:
        vox2lps = VoxelToLPS(matrix=matrix)
    else:
        # `VoxelToLPS` names 3-D spaces only, whereas ITK blocks can have
        # any number of dimensions.
        vox2lps = _xforms.Affine(
            matrix=matrix,
            input=_systems.VoxelCoordinateSystem(),
            output=_make_system(ndim),
        )
    return Geometry(transformation=vox2lps, shape=shape)


# ----------------------------------------------------------------------
#   DECODING
# ----------------------------------------------------------------------


def _parameters(pmap: ParameterMap) -> np.ndarray:
    """Return the transform parameters of a parameter map.

    `ITKTransformParameters` takes precedence when it is present. Otherwise,
    `TransformParameters` is read and checked against `NumberOfParameters`.
    """
    params = get_floats(pmap, "ITKTransformParameters")
    if params is not None:
        return params
    params = get_floats(pmap, "TransformParameters")
    if params is None:
        raise ParserContentError(
            "This elastix parameter map has no TransformParameters."
        )
    expected = get_string(pmap, "NumberOfParameters")
    if expected is not None and int(expected) != params.size:
        raise ParserContentError(
            f"This elastix parameter map has {params.size} "
            f"TransformParameters, but says NumberOfParameters is "
            f"{expected}."
        )
    return params


def _center(pmap: ParameterMap, ndim: int) -> np.ndarray:
    """Return the center of rotation, in LPS world coordinates.

    elastix 3.402 and later write `CenterOfRotationPoint`, and maps with ITK
    fixed parameters may hold the center in `ITKTransformFixedParameters`.
    The older `CenterOfRotation`, a voxel index of the fixed image, is read
    neither by current elastix nor by this function.
    """
    center = _vector(pmap, "CenterOfRotationPoint", ndim)
    if center is not None:
        return center
    fixed = get_floats(pmap, "ITKTransformFixedParameters")
    if fixed is not None and fixed.size >= ndim:
        return fixed[:ndim]
    if "CenterOfRotation" in pmap:
        raise ParserNotImplementedError(
            "This elastix parameter map stores its center of rotation as a "
            "voxel index (CenterOfRotation), as elastix < 3.402 did. "
            "Current elastix does not read it either; only "
            "CenterOfRotationPoint is supported."
        )
    raise ParserContentError(
        "This elastix parameter map has no center of rotation "
        "(CenterOfRotationPoint)."
    )


def _check_length(name: str, params: np.ndarray, expected: int) -> None:
    if params.size != expected:
        raise ParserContentError(
            f"An elastix {name} has {expected} parameters, not {params.size}."
        )


def _struct(
    type: ItkTransformClass,
    ndim: int,
    parameters: np.ndarray,
    fixed_parameters: tx.Sequence[float] = (),
    cls: tx.Type[ItkStruct] = ItkStruct,
) -> ItkStruct:
    return cls(
        type=type,
        precision="double",
        ndim_input=ndim,
        ndim_output=ndim,
        parameters=np.asarray(parameters, dtype=np.float64),
        fixed_parameters=np.asarray(fixed_parameters, dtype=np.float64),
    )


def _affine_struct(
    ndim: int, matrix: np.ndarray, translation: np.ndarray, center: np.ndarray
) -> ItkStruct:
    """Build an ITK `AffineTransform` block, `y = M (x - c) + c + t`.

    The block stores the matrix row by row, then the translation, and it
    keeps the center as its fixed parameters.
    """
    params = np.concatenate([np.asarray(matrix).ravel(), translation])
    return _struct(_ITKT.AffineTransform, ndim, params, center)


def _dti_matrix(params: np.ndarray, ndim: int) -> np.ndarray:
    """Return the matrix of an `AffineDTITransform`.

    The matrix is the product `R @ Gx @ Gy [@ Gz] @ S`, as in
    `itkAffineDTI{2,3}DTransform`, where the rotations turn in the opposite
    direction to those of `EulerTransform`.
    """
    if ndim == 2:
        (a,), (gx, gy), (sx, sy) = params[0:1], params[1:3], params[3:5]
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, s], [-s, c]])
        Gx = np.array([[1, gx], [0, 1]])
        Gy = np.array([[1, 0], [gy, 1]])
        return R @ Gx @ Gy @ np.diag([sx, sy])
    ax, ay, az = params[0:3]
    gx, gy, gz = params[3:6]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    Ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    Rz = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    Gx = np.array([[1, 0, gx], [0, 1, 0], [0, 0, 1]])
    Gy = np.array([[1, 0, 0], [gy, 1, 0], [0, 0, 1]])
    Gz = np.array([[1, 0, 0], [0, 1, 0], [0, gz, 1]])
    return Rx @ Ry @ Rz @ Gx @ Gy @ Gz @ np.diag(params[6:9])


def _bspline(pmap: ParameterMap, ndim: int, params: np.ndarray) -> ItkStruct:
    """Decode an elastix B-spline into an ITK B-spline block.

    The coefficients form `D` scalar images of world-space displacements,
    one per dimension. As in ITK, the images are stored one after the other
    with x varying fastest, and they cover the grid region that starts at
    `GridIndex`. The first coefficient therefore lies at
    `GridOrigin + GridDirection @ diag(GridSpacing) @ GridIndex`.
    """
    if get_bool(pmap, "UseCyclicTransform"):
        raise ParserNotImplementedError(
            "Cyclic elastix B-splines (UseCyclicTransform) are not supported."
        )
    degree = int(get_string(pmap, "BSplineTransformSplineOrder", "3"))
    if degree not in _BSPLINES:
        raise ParserContentError(
            f"Unsupported elastix B-spline degree {degree}: elastix only "
            f"writes degrees 1, 2 and 3."
        )
    size = _vector(pmap, "GridSize", ndim)
    if size is None:
        raise ParserContentError(
            "This elastix B-spline has no control-point grid (GridSize)."
        )
    size = np.round(size)
    index = _vector(pmap, "GridIndex", ndim, 0)
    spacing = _vector(pmap, "GridSpacing", ndim, 1)
    origin = _vector(pmap, "GridOrigin", ndim, 0)
    direction = direction_of(pmap, "GridDirection", ndim)
    _check_length("B-spline", params, ndim * int(np.prod(size)))
    origin = origin + direction @ (spacing * index)
    # The ITK fixed parameters are the size, the origin and the spacing,
    # followed by the direction matrix row by row.
    fixed = np.concatenate([size, origin, spacing, direction.ravel()])
    return _struct(
        _ITKT.BSplineTransform, ndim, params, fixed, cls=_BSPLINES[degree]
    )


def map_to_block(pmap: ParameterMap) -> ItkStruct:
    """Decode the transform of a parameter map into an ITK block.

    The block maps LPS coordinates of the fixed image to LPS coordinates of
    the moving image, and it does not include the initial transform.
    """
    name = get_string(pmap, "Transform")
    if name is None:
        raise ParserContentError(
            "This elastix parameter map names no Transform."
        )
    if name not in SUPPORTED:
        raise ParserNotImplementedError(
            f"The elastix transform {name!r} is not supported. Supported "
            f"transforms: {', '.join(SUPPORTED)}."
        )
    ndim = ndim_of(pmap)
    params = _parameters(pmap)

    if name == "TranslationTransform":
        _check_length(name, params, ndim)
        return _struct(_ITKT.TranslationTransform, ndim, params)

    if name in ("BSplineTransform", "RecursiveBSplineTransform"):
        return _bspline(pmap, ndim, params)

    center = _center(pmap, ndim)

    if name == "EulerTransform":
        if ndim == 2:
            _check_length(name, params, 3)
            return _struct(_ITKT.Euler2DTransform, 2, params, center)
        if ndim == 3:
            _check_length(name, params, 6)
            if "ComputeZYX" in pmap:
                zyx = get_bool(pmap, "ComputeZYX")
            else:
                itk = get_floats(pmap, "ITKTransformFixedParameters")
                zyx = itk is not None and itk.size > 3 and bool(itk[3])
            fixed = np.append(center, 1.0 if zyx else 0.0)
            return _struct(_ITKT.Euler3DTransform, 3, params, fixed)

    if name == "SimilarityTransform":
        if ndim == 2:
            _check_length(name, params, 4)
            return _struct(_ITKT.Similarity2DTransform, 2, params, center)
        if ndim == 3:
            _check_length(name, params, 7)
            return _struct(_ITKT.Similarity3DTransform, 3, params, center)

    if name == "AffineTransform":
        _check_length(name, params, ndim * (ndim + 1))
        return _struct(_ITKT.AffineTransform, ndim, params, center)

    if name == "AffineLogTransform":
        # The matrix is the exponential of the first D * D parameters
        # (row-major), and the translation follows.
        _check_length(name, params, ndim * (ndim + 1))
        log = params[: ndim * ndim].reshape(ndim, ndim)
        matrix = scipy.linalg.expm(log)
        return _affine_struct(ndim, matrix, params[ndim * ndim :], center)

    if name == "AffineDTITransform" and ndim in (2, 3):
        nparams = 7 if ndim == 2 else 12
        _check_length(name, params, nparams)
        matrix = _dti_matrix(params, ndim)
        return _affine_struct(ndim, matrix, params[-ndim:], center)

    raise ParserNotImplementedError(
        f"The elastix {name} is not supported in {ndim} dimensions."
    )


# ----------------------------------------------------------------------
#   ENCODING
# ----------------------------------------------------------------------

#: Parameters that describe the transform itself. A writer that replaces
#: the transform drops these parameters, and it keeps the parameters of the
#: fixed image, of the resampler and of the pixels.
TRANSFORM_KEYS = (
    "Transform",
    "NumberOfParameters",
    "TransformParameters",
    "ITKTransformParameters",
    "ITKTransformFixedParameters",
    "InitialTransformParameterFileName",
    "InitialTransformParametersFileName",
    "HowToCombineTransforms",
    "CenterOfRotationPoint",
    "CenterOfRotation",
    "ComputeZYX",
    "GridSize",
    "GridIndex",
    "GridSpacing",
    "GridOrigin",
    "GridDirection",
    "BSplineTransformSplineOrder",
    "UseCyclicTransform",
)

#: For each class of ITK block, the elastix transform that stores the
#: parameters of the block unchanged, with the center in
#: `CenterOfRotationPoint`.
_ELASTIX_NAMES = {
    _ITKT.TranslationTransform: "TranslationTransform",
    _ITKT.Euler2DTransform: "EulerTransform",
    _ITKT.Euler3DTransform: "EulerTransform",
    _ITKT.Similarity2DTransform: "SimilarityTransform",
    _ITKT.Similarity3DTransform: "SimilarityTransform",
    _ITKT.AffineTransform: "AffineTransform",
    _ITKT.MatrixOffsetTransformBase: "AffineTransform",
}


def _column_major(direction: np.ndarray) -> tx.Tuple[float, ...]:
    return tuple(np.asarray(direction, dtype=np.float64).T.ravel().tolist())


def _floats(values: tx.Any) -> tx.Tuple[float, ...]:
    return tuple(np.asarray(values, dtype=np.float64).ravel().tolist())


def _block_map(block: ItkStruct) -> tx.Optional[ParameterMap]:
    """Return the elastix parameters of an ITK block.

    The result is `None` when elastix has no transform with the same
    parameters as the block.
    """
    ndim = int(block.ndim_input)
    params = np.asarray(block.parameters, dtype=np.float64).ravel()
    fixed = np.asarray(block.fixed_parameters, dtype=np.float64).ravel()
    if block.type == _ITKT.BSplineTransform and isinstance(
        block, ItkBSplineStruct
    ):
        size = fixed[0:ndim]
        direction = fixed[3 * ndim :].reshape(ndim, ndim)
        return {
            "Transform": ("BSplineTransform",),
            "TransformParameters": params,
            "GridSize": tuple(int(round(v)) for v in size),
            "GridIndex": (0,) * ndim,
            "GridSpacing": _floats(fixed[2 * ndim : 3 * ndim]),
            "GridOrigin": _floats(fixed[ndim : 2 * ndim]),
            "GridDirection": _column_major(direction),
            "BSplineTransformSplineOrder": (int(block.degree),),
            "UseCyclicTransform": ("false",),
        }
    name = _ELASTIX_NAMES.get(block.type)
    if name is None:
        return None
    pmap: ParameterMap = {
        "Transform": (name,),
        "TransformParameters": params,
    }
    if name != "TranslationTransform":
        center = fixed[:ndim] if fixed.size >= ndim else np.zeros(ndim)
        pmap["CenterOfRotationPoint"] = _floats(center)
    if block.type == _ITKT.Euler3DTransform:
        pmap["ComputeZYX"] = ("true" if block.compute_zyx else "false",)
    return pmap


def _affine_map(xform: _xforms.Transformation) -> ParameterMap:
    """Encode an affine as an `AffineTransform` centered on the origin.

    With the center `c` at zero, the transform maps `x` to `M x + t`, so
    the translation is the last column of the affine matrix.
    """
    affine = xform
    if not isinstance(affine, _xforms.Affine):
        affine = xform.to(_xforms.Affine, error=None)
    if affine is None:
        try:
            affine = xform.compute().to(_xforms.Affine, error=None)
        except Exception:  # noqa: BLE001
            affine = None
    matrix = None if affine is None else affine.matrix
    if matrix is None:
        raise UnrepresentableTransformationError(
            f"An elastix parameter file is written from an elastix or ITK "
            f"transform, or an affine, and a {type(xform).__name__} cannot "
            f"be written as either."
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    ndim = matrix.shape[0]
    if matrix.shape != (ndim, ndim + 1):
        raise UnrepresentableTransformationError(
            f"An elastix transform maps a space to a space of the same "
            f"dimension, of shape (D, D + 1), but the matrix has shape "
            f"{matrix.shape}."
        )
    space = _make_system(ndim)
    for end in ("input", "output"):
        system = getattr(affine, end)
        if system is not None and system != space:
            raise UnrepresentableTransformationError(
                f"An elastix transform maps LPS millimetres to LPS "
                f"millimetres, but the {end} of this one is "
                f"{type(system).__name__}. Convert it to LPS first."
            )
    return {
        "Transform": ("AffineTransform",),
        "TransformParameters": np.concatenate(
            [matrix[:, :ndim].ravel(), matrix[:, ndim]]
        ),
        "CenterOfRotationPoint": (0.0,) * ndim,
    }


def transformation_to_map(
    xform: _xforms.Transformation, base: tx.Optional[ParameterMap] = None
) -> ParameterMap:
    """Return the parameter map that encodes a single transformation.

    An ITK block whose class elastix shares (translation, Euler, similarity,
    affine or B-spline) is written as that elastix transform, with its own
    parameters and center. Anything else that reduces to an affine becomes an
    `AffineTransform` centered on the origin. The other parameters of `base`
    (fixed geometry, resampler, pixel types) are kept, so that transformix
    can still resample.
    """
    pmap = None
    if isinstance(xform, ItkStruct):
        pmap = _block_map(xform)
    if pmap is None:
        pmap = _affine_map(xform)
    ndim = None
    for key in ("CenterOfRotationPoint", "GridSize"):
        if key in pmap:
            ndim = len(pmap[key])
    if ndim is None:
        ndim = len(pmap["TransformParameters"])
    params = pmap.pop("TransformParameters")
    out: ParameterMap = {
        "Transform": pmap.pop("Transform"),
        "NumberOfParameters": (int(params.size),),
        "TransformParameters": params,
        "InitialTransformParameterFileName": ("NoInitialTransform",),
        "HowToCombineTransforms": ("Compose",),
        "FixedImageDimension": (ndim,),
        "MovingImageDimension": (ndim,),
    }
    out.update(pmap)
    for key, value in (base or {}).items():
        if key not in TRANSFORM_KEYS and key not in out:
            out[key] = value
    return out
