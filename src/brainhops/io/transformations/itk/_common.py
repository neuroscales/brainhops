import math
from warnings import warn

import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

from brainhops._core import affines as _affines
from brainhops._core.enum import StrEnum
from brainhops._core.properties import lazyproperty, smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.io.base.parsers import ParserContentError
from brainhops.io.transformations.base.affines import LPSToVoxel, VoxelToLPS

from ._systems import _make_system


class ItkTransformClass(StrEnum):
    """Names of the ITK transform classes."""

    IdentityTransform = "IdentityTransform"

    TranslationTransform = "TranslationTransform"

    ScaleTransform = "ScaleTransform"
    ScaleLogarithmicTransform = "ScaleLogarithmicTransform"

    Similarity2DTransform = "Similarity2DTransform"
    Similarity3DTransform = "Similarity3DTransform"

    Euler2DTransform = "Euler2DTransform"
    Euler3DTransform = "Euler3DTransform"
    VersorTransform = "VersorTransform"

    VersorRigid3DTransform = "VersorRigid3DTransform"
    ScaleVersor3DTransform = "ScaleVersor3DTransform"
    ScaleSkewVersor3DTransform = "ScaleSkewVersor3DTransform"

    AffineTransform = "AffineTransform"
    MatrixOffsetTransformBase = "MatrixOffsetTransformBase"

    DisplacementFieldTransform = "DisplacementFieldTransform"
    BSplineTransform = "BSplineTransform"

    CompositeTransform = "CompositeTransform"


_ITKT = ItkTransformClass


class ItkPrecision(StrEnum):
    """Precisions of ITK transforms."""

    Float = "float"
    Double = "double"


class ItkStruct(Magic, kw_only=True, convert=True, polymorphic=True, eq=False):
    """One ITK transform block, holding only what the file stores.

    Concrete blocks register their class with `on={"type": ...}`, so that
    `ItkStruct(type=..., ...)` dispatches to the right subtype; an unclaimed
    type yields a bare `ItkStruct`. Through [`ItkAffineBase`][] and
    [`ItkDisplacementBase`][], a parsed block is already a transformation.
    Blocks compare by identity, since their parameters are arrays.
    """

    type: ItkTransformClass
    """The ITK transform class, such as `"AffineTransform"`."""

    precision: ItkPrecision
    """The precision, `"float"` or `"double"`."""

    ndim_input: int
    """The number of input dimensions."""

    ndim_output: int
    """The number of output dimensions."""

    parameters: ArrayProtocol = ()
    """The optimisable parameters, such as a translation vector."""

    fixed_parameters: ArrayProtocol = ()
    """The fixed parameters, such as a center of rotation."""

    def _check_same_ndim(self, expected_ndim: tx.Optional[int] = None) -> None:
        if self.ndim_input != self.ndim_output:
            name = self.__class__.__name__
            raise ValueError(
                f"{name} must have equal input and output dimensions "
                f"({self.ndim_input} != {self.ndim_output})"
            )
        if expected_ndim is not None and self.ndim_input != expected_ndim:
            name = self.__class__.__name__
            raise ValueError(
                f"{name} must have {expected_ndim} dimensions "
                f"({self.ndim_input} != {expected_ndim})"
            )

    def _check_parameters_length(self, expected_length: int) -> None:
        if len(self.parameters) != expected_length:
            name = self.__class__.__name__
            raise ValueError(
                f"{name} parameters length {len(self.parameters)} "
                f"does not match expected length {expected_length}"
            )

    def _check_fixed_parameters_length(self, expected_length: int) -> None:
        if len(self.fixed_parameters) != expected_length:
            name = self.__class__.__name__
            raise ValueError(
                f"{name} fixed parameters length {len(self.fixed_parameters)} "
                f"does not match expected length {expected_length}"
            )


# ----------------------------------------------------------------------
#   BASES
# ----------------------------------------------------------------------


class ItkBlockBase(ItkStruct, _xforms.ImmutableSequence):
    """Common base of ITK blocks, which map LPS to LPS world coordinates.

    The endpoints come from `ndim_input` and `ndim_output` rather than from the
    chain, as a plain
    [`Sequence`][brainhops.datamodel.transformations.Sequence] would do,
    because building the chain of a warp block decodes the warp data. The chain
    is a tuple of named slots and cannot be edited in place.
    """

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The LPS space that the block maps from."""
        return _make_system(self.ndim_input)

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The LPS space that the block maps to."""
        return _make_system(self.ndim_output)

    def inverse(self, compute: bool = False, **kwargs) -> _xforms.Sequence:
        """The inverse of the block, as a plain sequence."""
        return _inverse_chain(self, compute=compute, **kwargs)


class ItkAffineBase(ItkBlockBase):
    """ITK block that encodes an affine-like transform.

    ITK stores a linear part acting about a center of rotation, then an
    optional translation. The block is a chain of these named slots:

    | Slot          | Transformation                             |
    | ------------- | ------------------------------------------ |
    | `recenter`    | moves the center of rotation to the origin |
    | `linear`      | the linear part of the block               |
    | `uncenter`    | moves the center of rotation back          |
    | `translation` | the translation, when the block has one    |

    Slots are derived from the parameters on first access, and unused slots are
    `None` and left out. Blocks whose linear part has several factors list them
    in `_SLOTS`.
    """

    _SLOTS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "recenter",
        "linear",
        "uncenter",
        "translation",
    )
    """The names of the slots that form the chain, in order."""

    @smartproperty(cache=True)
    def center(self) -> tx.Optional[ArrayProtocol]:
        """The center of rotation, or `None` if the stored vector is empty."""
        return _nonempty(self.fixed_parameters)

    @smartproperty(cache=True)
    def recenter(self) -> tx.Optional[_xforms.Transformation]:
        """Translation from the center of rotation to the origin."""
        center = self.center
        if center is None:
            return None
        return _xforms.Translation(center).inverse()

    @smartproperty(cache=True)
    def uncenter(self) -> tx.Optional[_xforms.Transformation]:
        """Translation from the origin back to the center of rotation."""
        center = self.center
        if center is None:
            return None
        return _xforms.Translation(center)

    @smartproperty(cache=True)
    def linear(self) -> tx.Optional[_xforms.Transformation]:
        """The linear part, applied about the center of rotation."""
        return None

    @smartproperty(cache=True)
    def translation(self) -> tx.Optional[_xforms.Transformation]:
        """The translation applied after the linear part."""
        return None

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of used slots, cached.

        The chain is a tuple, so that `del block[0]` cannot edit the cache.
        """
        chain = (getattr(self, name) for name in self._SLOTS)
        return tuple(child for child in chain if child is not None)


class ItkDisplacementBase(ItkBlockBase):
    """ITK block that encodes a dense or spline warp.

    The warp lives on its own voxel grid, described by the fixed parameters, so
    the block is a chain of three named slots:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `lps2voxel`    | LPS world coordinates to warp-grid voxels   |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2lps`    | warp-grid voxels back to LPS world          |

    The warp data is decoded on first access, so opening a file never reads it.
    Subclasses set the spline parameters `degree`, `store` and `bound` of the
    [`DisplacementField`][brainhops.datamodel.transformations.DisplacementField].
    """

    degree: tx.ClassVar[int] = 1
    """The spline degree of the field."""

    store: tx.ClassVar[StoreEnum] = StoreEnum.values
    """Whether the field holds spline coefficients or values."""

    bound: tx.ClassVar[tx.Union[BoundaryCondition, float]] = (
        BoundaryCondition.nearest
    )
    """The boundary condition outside the field of view."""

    interleaved: tx.ClassVar[bool] = True
    """Whether each grid point stores its whole vector contiguously.

    A dense field is a vector image, so its components are interleaved. A
    B-spline stores one coefficient image per axis, back to back. Mixing the
    two up silently transposes the warp.
    """

    @lazyproperty
    def _grid(self) -> tx.Tuple[np.ndarray, tx.Tuple[int, ...]]:
        return _vox2lps(self.fixed_parameters, self.ndim_input)

    @smartproperty(cache=True)
    def field(self) -> ArrayProtocol:
        """The warp on its own grid, as an `(Nx, Ny, Nz, D)` array.

        ITK stores flat world-space displacements. They are reordered and
        rotated into voxel units, because a
        [`DisplacementField`][brainhops.datamodel.transformations.DisplacementField]
        adds them in grid units.
        """  # noqa: E501
        vox2lps, shape = self._grid
        ndim = self.ndim_input

        parameters = self.parameters
        parameters = parameters if parameters is not None else np.array([])
        if not hasattr(parameters, "reshape"):
            parameters = get_array_backend(parameters).asarray(parameters)

        # The buffer is C-ordered with x fastest, so the spatial axes come out
        # reversed in both layouts.
        spatial = range(ndim - 1, -1, -1)
        if self.interleaved:
            disp = parameters.reshape(*reversed(shape), ndim)
            disp = disp.transpose(*spatial, ndim)
        else:
            disp = parameters.reshape(ndim, *reversed(shape))
            disp = disp.transpose(*(axis + 1 for axis in spatial), 0)

        # Only the linear part of the affine rotates a displacement.
        lps2vox = _affines.inv(vox2lps)
        backend = get_array_backend(disp)
        rotate = backend.asarray(lps2vox[:, :ndim], dtype=disp.dtype)
        return backend.matmul(rotate, disp[..., None])[..., 0]

    @smartproperty(cache=True)
    def lps2voxel(self) -> LPSToVoxel:
        """Affine from LPS world coordinates to warp-grid voxels."""
        vox2lps, _ = self._grid
        return LPSToVoxel(matrix=_affines.inv(vox2lps))

    @smartproperty(cache=True)
    def displacement(self) -> _xforms.DisplacementField:
        """The displacement field on the warp grid."""
        VOX = _systems.VoxelCoordinateSystem()
        return _xforms.DisplacementField(
            data=self.field,
            input=VOX,
            output=VOX,
            degree=self.degree,
            store=self.store,
            bound=self.bound,
        )

    @smartproperty(cache=True)
    def voxel2lps(self) -> VoxelToLPS:
        """Affine from warp-grid voxels to LPS world coordinates."""
        vox2lps, _ = self._grid
        return VoxelToLPS(matrix=vox2lps)

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of the three slots, cached as a tuple."""
        return (self.lps2voxel, self.displacement, self.voxel2lps)


# ----------------------------------------------------------------------
#   BLOCKS
# ----------------------------------------------------------------------


class ItkIdentityStruct(ItkAffineBase, on={"type": _ITKT.IdentityTransform}):
    """ITK identity transform, which has no parameters."""

    parameters: tx.Tuple[tx.Any, ...] = ()

    fixed_parameters: tx.Tuple[tx.Any, ...] = ()

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Identity:
        """The identity."""
        # A block must hold at least one transformation.
        return _xforms.Identity(input=self.input, output=self.output)


class ItkTranslationStruct(
    ItkAffineBase, on={"type": _ITKT.TranslationTransform}
):
    """ITK translation, with one parameter per dimension."""

    fixed_parameters: tx.Tuple[tx.Any, ...] = ()

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters)


class ItkScaleStruct(ItkAffineBase, on={"type": _ITKT.ScaleTransform}):
    """ITK scaling, with one factor per dimension."""

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Scaling:
        """The per-axis scaling."""
        return _xforms.Scaling(self.parameters)


class ItkScaleLogarithmicStruct(
    ItkAffineBase, on={"type": _ITKT.ScaleLogarithmicTransform}
):
    """ITK scaling that stores the logarithms of its factors."""

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Scaling:
        """The per-axis scaling."""
        return _xforms.Scaling(np.exp(self.parameters))


class ItkEuler2DStruct(ItkAffineBase, on={"type": _ITKT.Euler2DTransform}):
    """ITK 2-D rotation followed by a translation."""

    ndim_input: tx.Literal[2] = 2
    ndim_output: tx.Literal[2] = 2

    parameters: tx.Tuple[float, float, float]
    """The angle, then the translation."""

    fixed_parameters: tx.Tuple[float, float]
    """The center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Rotation:
        """The rotation."""
        return _xforms.Rotation(_angle_to_matrix(self.parameters[0]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[1:3])


class ItkEuler3DStruct(ItkAffineBase, on={"type": _ITKT.Euler3DTransform}):
    """ITK 3-D Euler rotation followed by a translation."""

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[float, float, float, float, float, float]
    """The angles `(rx, ry, rz)`, then the translation."""

    fixed_parameters: tx.Union[
        tx.Tuple[float, float, float],
        tx.Tuple[float, float, float, float],
    ]
    """The center of rotation, then optionally the `ComputeZYX` flag."""

    @smartproperty(cache=True)
    def center(self) -> tx.Optional[ArrayProtocol]:
        """The center, from the first three fixed parameters."""
        return _nonempty(self.fixed_parameters[:3])

    @lazyproperty
    def compute_zyx(self) -> bool:
        """Whether the angles compose in ZYX rather than ZXY order.

        Older files lack the flag and always use ZXY.
        """
        fixed = self.fixed_parameters
        return len(fixed) > 3 and bool(fixed[3])

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Rotation:
        """The rotation."""
        return _xforms.Rotation(
            _euler_to_matrix(self.parameters[:3], self.compute_zyx)
        )

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


class ItkVersorStruct(ItkAffineBase, on={"type": _ITKT.VersorTransform}):
    """ITK 3-D versor rotation."""

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[float, float, float]
    """The vector part of the versor."""

    fixed_parameters: tx.Tuple[float, float, float]
    """The center of rotation."""

    def __post_init__(self) -> None:
        self._check_same_ndim()

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Rotation:
        """The rotation."""
        return _xforms.Rotation(_versor_to_matrix(self.parameters[:3]))


class ItkVersorRigid3DStruct(
    ItkAffineBase, on={"type": _ITKT.VersorRigid3DTransform}
):
    """ITK 3-D versor rotation followed by a translation."""

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[float, float, float, float, float, float]
    """The versor, then the translation."""

    fixed_parameters: tx.Tuple[float, float, float]
    """The center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Rotation:
        """The rotation."""
        return _xforms.Rotation(_versor_to_matrix(self.parameters[:3]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


class ItkSimilarity2DStruct(
    ItkAffineBase, on={"type": _ITKT.Similarity2DTransform}
):
    """ITK 2-D similarity: isotropic scaling, rotation and translation."""

    ndim_input: tx.Literal[2] = 2
    ndim_output: tx.Literal[2] = 2

    parameters: tx.Tuple[float, float, float, float]
    """The scale, the angle and the translation."""

    fixed_parameters: tx.Tuple[float, float]
    """The center of rotation."""

    # ITK parameterises the linear part by a scale and an angle.
    _SLOTS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "recenter",
        "scaling",
        "rotation",
        "uncenter",
        "translation",
    )

    @smartproperty(cache=True)
    def scaling(self) -> _xforms.Scaling:
        """The isotropic scaling."""
        return _isotropic_scaling(self.parameters[0], self.ndim_input)

    @smartproperty(cache=True)
    def rotation(self) -> _xforms.Rotation:
        """The rotation."""
        return _xforms.Rotation(_angle_to_matrix(self.parameters[1]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[2:4])


class ItkSimilarity3DStruct(
    ItkAffineBase, on={"type": _ITKT.Similarity3DTransform}
):
    """ITK 3-D similarity: isotropic scaling, rotation and translation."""

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[float, float, float, float, float, float, float]
    """The versor, the translation and the scale."""

    fixed_parameters: tx.Tuple[float, float, float]
    """The center of rotation."""

    # ITK parameterises the linear part by a scale and a versor.
    _SLOTS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "recenter",
        "scaling",
        "rotation",
        "uncenter",
        "translation",
    )

    @smartproperty(cache=True)
    def scaling(self) -> _xforms.Scaling:
        """The isotropic scaling."""
        return _isotropic_scaling(self.parameters[6], self.ndim_input)

    @smartproperty(cache=True)
    def rotation(self) -> _xforms.Rotation:
        """The rotation."""
        return _xforms.Rotation(_versor_to_matrix(self.parameters[0:3]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


class ItkScaleVersor3DStruct(
    ItkAffineBase, on={"type": _ITKT.ScaleVersor3DTransform}
):
    """ITK 3-D versor rotation with per-axis scaling and a translation."""

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[
        float, float, float, float, float, float, float, float, float
    ]
    """The versor, the translation and the scales `(sx, sy, sz)`."""

    fixed_parameters: tx.Tuple[float, float, float]
    """The center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Linear:
        """The rotation matrix plus `diag(scales - 1)`."""
        R = _versor_to_matrix(self.parameters[0:3])
        S = np.diag(np.asarray(self.parameters[6:9]) - 1)
        return _xforms.Linear(R + S)

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


class ItkScaleSkewVersor3DStruct(
    ItkAffineBase, on={"type": _ITKT.ScaleSkewVersor3DTransform}
):
    """ITK 3-D versor rotation with scaling, skew and a translation."""

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    """The versor, the translation, the scales and the skews.

    The skews are ordered `(kxy, kxz, kyx, kyz, kzx, kzy)`.
    """

    fixed_parameters: tx.Tuple[float, float, float]
    """The center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Linear:
        """The rotation matrix plus `diag(scales - 1)` and the skews."""
        k = self.parameters[9:15]
        R = _versor_to_matrix(self.parameters[0:3])
        S = np.diag(np.asarray(self.parameters[6:9]) - 1)
        K = np.array(
            [[0, k[0], k[1]], [k[2], 0, k[3]], [k[4], k[5], 0]],
            dtype=np.float64,
        )
        return _xforms.Linear(R + S + K)

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


class _ItkMatrixOffsetBase(ItkAffineBase):
    """Base of blocks that store a full matrix, then a translation."""

    def __post_init__(self) -> None:
        self._check_same_ndim()
        ndim = self.ndim_input
        self._check_parameters_length((ndim + 1) * ndim)
        self._check_fixed_parameters_length(ndim)

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Linear:
        """The linear part, stored in row-major order."""
        Di, Do = self.ndim_input, self.ndim_output
        L = np.array(self.parameters[: Di * Do], dtype=np.float64)
        return _xforms.Linear(L.reshape(Do, Di))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation, stored after the linear part."""
        Do = self.ndim_output
        return _xforms.Translation(
            np.array(self.parameters[-Do:], dtype=np.float64)
        )


class ItkAffineStruct(
    _ItkMatrixOffsetBase, on={"type": _ITKT.AffineTransform}
):
    """ITK affine transform."""


class ItkMatrixOffsetStruct(
    _ItkMatrixOffsetBase, on={"type": _ITKT.MatrixOffsetTransformBase}
):
    """ITK base class of affine transforms.

    Older ANTs releases write it instead of `AffineTransform`.
    """


class ItkDisplacementFieldStruct(
    ItkDisplacementBase, on={"type": _ITKT.DisplacementFieldTransform}
):
    """ITK dense displacement field.

    The field holds one world-space displacement per voxel, interpolated
    linearly and stored interleaved.
    """

    interleaved: tx.ClassVar[bool] = True


class ItkBSplineStruct(
    ItkDisplacementBase, on={"type": _ITKT.BSplineTransform}
):
    """ITK B-spline transform.

    The field holds cubic coefficients on a control-point grid, zero outside
    it, stored as one planar image per axis.
    """

    degree: tx.ClassVar[int] = 3
    store: tx.ClassVar[StoreEnum] = StoreEnum.coefficients
    bound: tx.ClassVar[tx.Union[BoundaryCondition, float]] = (
        BoundaryCondition.zeros
    )
    interleaved: tx.ClassVar[bool] = False


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _application_order(
    blocks: tx.List[tx.Any],
    composites: tx.List[int],
    position: tx.Optional[int] = None,
) -> tx.List[tx.Any]:
    """Return the blocks of an ITK file in application order.

    ITK writes a composite as a header followed by its queue, front to back,
    but applies the queue back to front: `[T0, T1]` maps `x` to `T0(T1(x))`. A
    [`Sequence`][brainhops.datamodel.transformations.Sequence] lists
    transformations in application order, so composite blocks are reversed.
    Without a composite, every block is a separate top-level transform, and the
    first is returned by default, with a warning when there are several.

    Parameters
    ----------
    blocks
        The blocks in file order, without composite headers.
    composites
        The file positions of the skipped composite headers.
    position
        The index of the top-level transform to return.

    Raises
    ------
    ParserContentError
        If a composite header is not the first block, or if there is no
        top-level transform at `position`.
    """
    if composites:
        if list(composites) != [0]:
            raise ParserContentError(
                "ITK only writes a CompositeTransform as the first block "
                "of a file, and it cannot be nested."
            )
        transforms = [list(reversed(blocks))]
    else:
        transforms = [[block] for block in blocks]

    if position is None:
        if len(transforms) > 1:
            warn(
                f"This ITK file holds {len(transforms)} transforms and no "
                f"composite, so only the first one is read. Pass "
                f"`position=` to read another one.",
                stacklevel=2,
            )
        position = 0
    if not transforms and position == 0:
        return []
    if not 0 <= position < len(transforms):
        raise ParserContentError(
            f"This ITK file has {len(transforms)} transform(s), so it has "
            f"no transform {position}."
        )
    return transforms[position]


def _inverse_chain(
    struct: _xforms.Sequence, compute: bool = False, **kwargs
) -> _xforms.Sequence:
    """Invert a block as the reversed chain of its inverted children."""
    return _xforms.Sequence(
        transformations=[
            child.inverse(compute=compute, **kwargs)
            for child in reversed(struct.transformations or [])
        ],
        input=struct.output,
        output=struct.input,
    )


def _isotropic_scaling(scale: float, ndim: int) -> _xforms.Scaling:
    """Isotropic scaling, keeping the single ITK factor as it is stored.

    A one-element vector broadcasts to any number of axes, so the endpoints are
    given explicitly to fix the dimensionality.
    """
    system = _make_system(ndim)
    return _xforms.Scaling(
        np.asarray([scale], dtype=np.float64), input=system, output=system
    )


def _nonempty(values: tx.Optional[ArrayProtocol]) -> tx.Optional[tx.Any]:
    """Return `values`, or `None` if they are empty (an unused parameter)."""
    if values is None:
        return None
    try:
        if len(values) == 0:
            return None
    except TypeError:
        pass
    return values


def _vox2lps(
    fixed_parameters: tx.Sequence[float], ndim: int = 3
) -> tx.Tuple[np.ndarray, tx.Tuple[int, ...]]:
    """Return the voxel-to-LPS affine of a warp grid, and the grid shape.

    The fixed parameters hold the shape, origin, spacing and direction, sized
    by `ndim`. The affine is returned in the compact `(ndim, ndim + 1)` form
    expected by [`brainhops._core.affines`][].
    """

    fixed_parameters = np.asarray(fixed_parameters, dtype=np.float64)
    shape = fixed_parameters[0:ndim]
    origin = fixed_parameters[ndim : 2 * ndim]
    spacing = fixed_parameters[2 * ndim : 3 * ndim]
    direction = fixed_parameters[3 * ndim : 3 * ndim + ndim * ndim]
    direction = direction.reshape(ndim, ndim)

    shape = tuple(map(int, map(round, shape)))

    vox2lps = np.zeros((ndim, ndim + 1), dtype=np.float64)
    vox2lps[:, :ndim] = direction @ np.diag(spacing)
    vox2lps[:, ndim] = origin
    return vox2lps, shape


def _angle_to_matrix(angle: float) -> np.ndarray:
    """Convert a 2-D angle to a rotation matrix."""
    return np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )


# Rounding can put a half-turn versor a few ulps outside the unit sphere. ITK
# renormalises it, so a file that ITK opens must open here.
_VERSOR_TOLERANCE = 1e-6


def _versor_to_matrix(q: tx.Sequence[float]) -> np.ndarray:
    """Convert the vector part of a versor to a rotation matrix."""
    qx, qy, qz = q
    norm_sq = qx**2 + qy**2 + qz**2
    if norm_sq > 1.0:
        # Within tolerance, rescale onto the sphere; beyond it, refuse.
        norm = math.sqrt(norm_sq)
        if norm > 1.0 + _VERSOR_TOLERANCE:
            raise ValueError(
                f"Versor quaternion vector part has magnitude > 1 ({norm})"
            )
        qx, qy, qz = qx / norm, qy / norm, qz / norm
        norm_sq = 1.0
    qw = math.sqrt(max(0.0, 1.0 - norm_sq))
    return np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )


def _euler_to_matrix(
    angles: tx.Sequence[float], compute_zyx: bool = False
) -> np.ndarray:
    """Convert ITK Euler angles to a rotation matrix.

    The order is `Rz @ Rx @ Ry` by default and `Rz @ Ry @ Rx` when
    `compute_zyx` is set.
    """
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx if compute_zyx else Rz @ Rx @ Ry
