# stdlib
import math

# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import Magic

# core
from brainhops._core import affines as _affines
from brainhops._core.enum import StrEnum
from brainhops._core.properties import lazyproperty, smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.io.transformations.base.affines import LPSToVoxel, VoxelToLPS

# locals
from ._systems import _make_system


class ITKTransformClass(StrEnum):
    """Enumeration of ITK transform class names."""

    IdentityTransform = "IdentityTransform"

    # Translations
    TranslationTransform = "TranslationTransform"

    # Scales
    ScaleTransform = "ScaleTransform"
    ScaleLogarithmicTransform = "ScaleLogarithmicTransform"

    # Similarities
    Similarity2DTransform = "Similarity2DTransform"
    Similarity3DTransform = "Similarity3DTransform"

    # Rotations
    Euler2DTransform = "Euler2DTransform"
    Euler3DTransform = "Euler3DTransform"
    VersorTransform = "VersorTransform"

    # Quaternions+
    VersorRigid3DTransform = "VersorRigid3DTransform"
    ScaleVersor3DTransform = "ScaleVersor3DTransform"
    ScaleSkewVersor3DTransform = "ScaleSkewVersor3DTransform"

    # Affines
    AffineTransform = "AffineTransform"

    # Non-linear
    DisplacementFieldTransform = "DisplacementFieldTransform"
    BSplineTransform = "BSplineTransform"

    # Composite
    CompositeTransform = "CompositeTransform"


_ITKT = ITKTransformClass  # Alias for brevity in type hints


class ITKPrecision(StrEnum):
    """Enumeration of ITK transform precision types."""

    Float = "float"
    Double = "double"


class ITKStruct(Magic, kw_only=True, convert=True):
    """This object represents a single ITK transform block.

    It holds what an ITK file stores about one block -- its transform
    class, its precision, its dimensions, and its parameter vectors --
    and nothing else. Concrete blocks inherit from [`ITKAffineBase`][]
    or [`ITKDisplacementBase`][], which both combine this provenance with
    [`Sequence`][brainhops.datamodel.transformations.Sequence] through
    [`ITKBlockBase`][], so a parsed block is already a brainhops
    transformation.
    """

    _REGISTRY: tx.ClassVar[tx.Mapping[str, type]] = {}

    def __new__(cls, *args, **kwargs) -> None:
        if cls is not ITKStruct:
            return super().__new__(cls)
        if not hasattr(cls, "_REGISTRY"):
            cls._REGISTRY = {}
        cls = cls._REGISTRY.get(kwargs.get("type"), cls)
        return super().__new__(cls)

    type: ITKTransformClass
    """The ITK transform class name (e.g., "AffineTransform")."""

    precision: ITKPrecision
    """The ITK transform precision type (e.g., "float" or "double")."""

    ndim_input: int
    """The number of input dimensions."""

    ndim_output: int
    """The number of output dimensions."""

    parameters: ArrayProtocol = ()
    """
    The optimizable parameters of the transform (e.g., translation vector).
    """

    fixed_parameters: ArrayProtocol = ()
    """The fixed parameters of the transform (e.g., center of rotation)."""

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


def _register_type(*names: str) -> tx.Callable:

    def decorator(cls: type) -> type:
        for name in names:
            ITKStruct._REGISTRY[name] = cls
        return cls

    return decorator


# ----------------------------------------------------------------------
#   BASES
# ----------------------------------------------------------------------


class ITKBlockBase(ITKStruct, _xforms.Sequence):
    """What every ITK block shares: its endpoints and its inverse.

    Whatever a block encodes, it maps LPS world coordinates to LPS world
    coordinates in the number of dimensions its file declares. Both
    endpoints are therefore *declared* from `ndim_input` / `ndim_output`
    rather than read back off the chain the way a plain
    [`Sequence`][brainhops.datamodel.transformations.Sequence] reads
    them: reading them off the chain would build the chain, and building
    a warp block's chain decodes its warp data.
    """

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the block maps from."""
        return _make_system(self.ndim_input)

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the block maps to."""
        return _make_system(self.ndim_output)

    def inverse(self, compute: bool = False, **kwargs) -> _xforms.Sequence:
        """The inverse of the block, as a plain sequence."""
        return _inverse_chain(self, compute=compute, **kwargs)


class ITKAffineBase(ITKBlockBase):
    """An ITK block that encodes an affine-like transformation.

    ITK does not store an affine-like block as a single matrix. It stores
    a linear part that acts about a *center of rotation*, optionally
    followed by a translation. That is a chain, so the block itself is a
    [`Sequence`][brainhops.datamodel.transformations.Sequence] whose
    children are named slots:

    | Slot          | Transformation                             |
    | ------------- | ------------------------------------------ |
    | `recenter`    | moves the center of rotation to the origin |
    | `linear`      | the linear part of the block               |
    | `uncenter`    | moves the center of rotation back          |
    | `translation` | the translation, when the block has one    |

    Each slot is derived from `parameters` and `fixed_parameters` on
    first access and cached afterwards, and a slot that a block does not
    use is `None` and is left out of the chain. A block whose linear part
    is itself made of several transformations -- a similarity, which ITK
    parameterizes by a scale and a rotation -- names them individually
    and lists them in `_SLOTS` instead.
    """

    _SLOTS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "recenter",
        "linear",
        "uncenter",
        "translation",
    )
    """The names of the slots that make up the chain, in order."""

    # --- slots --------------------------------------------------------

    @smartproperty(cache=True)
    def center(self) -> tx.Optional[ArrayProtocol]:
        """The center of rotation, or `None` when the block has none.

        ITK stores it in the fixed parameters. A block that has no center
        -- a translation, an identity -- stores an empty vector there.
        """
        return _nonempty(self.fixed_parameters)

    @smartproperty(cache=True)
    def recenter(self) -> tx.Optional[_xforms.Transformation]:
        """Moves the center of rotation to the origin."""
        center = self.center
        if center is None:
            return None
        return _xforms.Translation(center).inverse()

    @smartproperty(cache=True)
    def uncenter(self) -> tx.Optional[_xforms.Transformation]:
        """Moves the center of rotation back to where it was."""
        center = self.center
        if center is None:
            return None
        return _xforms.Translation(center)

    @smartproperty(cache=True)
    def linear(self) -> tx.Optional[_xforms.Transformation]:
        """The linear part of the block, applied about the center."""
        return None

    @smartproperty(cache=True)
    def translation(self) -> tx.Optional[_xforms.Transformation]:
        """The translation applied after the centered linear part."""
        return None

    # --- sequence -----------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations that the block encodes.

        It is assembled from the named slots listed in `_SLOTS`, skipping
        the ones the block does not use, and cached. Assigning to it
        overrides the derived chain.

        It is a tuple rather than a list because the derived chain is
        cached and handed out as is, and a list would let `del block[0]`
        edit the cache in place -- leaving the block reporting a chain
        that its own slots no longer describe.
        """
        chain = (getattr(self, name) for name in self._SLOTS)
        return tuple(child for child in chain if child is not None)


class ITKDisplacementBase(ITKBlockBase):
    """An ITK block that encodes a dense or spline-based warp.

    The warp lives on its own voxel grid, whose geometry the fixed
    parameters carry, while the block maps LPS world coordinates. The
    block is therefore a
    [`Sequence`][brainhops.datamodel.transformations.Sequence] of three
    named slots:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `lps2voxel`    | LPS world coordinates to warp-grid voxels   |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2lps`    | warp-grid voxels back to LPS world          |

    The stored parameters are decoded into `field` on first access and
    cached, so opening a file never touches the warp data: a dask-backed
    or delayed array stays unread until the chain is asked for.

    `order`, `coeff` and `bound` are the spline parameters handed to the
    [`DisplacementField`][brainhops.datamodel.transformations.DisplacementField],
    and a subclass overrides them to describe its own encoding.
    """  # noqa: E501

    order: tx.ClassVar[int] = 1
    """The spline order used to interpolate the field."""

    coeff: tx.ClassVar[bool] = False
    """Whether the field holds spline coefficients rather than values."""

    bound: tx.ClassVar[tx.Union[BoundaryCondition, float]] = (
        BoundaryCondition.nearest
    )
    """The boundary condition used outside of the field of view."""

    interleaved: tx.ClassVar[bool] = True
    """Whether the parameters store one whole vector per grid point.

    ITK flattens the two kinds of warp differently, because in ITK they
    are two different things.

    A dense field's parameters *are* the buffer of an image of vectors:
    one `Vector<T, D>` per voxel, so the `D` components of a voxel sit
    next to each other and the component index varies fastest
    (`interleaved`). A B-spline's parameters are `D` separate scalar
    coefficient images written back to back, so a whole component spans
    a contiguous plane and the component index varies slowest (planar).

    Reading one layout as the other silently transposes the warp rather
    than failing, so each subclass states which it is.
    """

    # --- decoding -----------------------------------------------------

    @lazyproperty
    def _grid(self) -> tx.Tuple[np.ndarray, tx.Tuple[int, ...]]:
        # The compact voxel-to-LPS affine of the warp grid, and its shape.
        return _vox2lps(self.fixed_parameters, self.ndim_input)

    @smartproperty(cache=True)
    def field(self) -> ArrayProtocol:
        """The warp values on their own grid, in voxel units.

        ITK stores them as a flat, C-ordered block of world-space
        displacements, laid out either interleaved or planar -- see
        `interleaved`. Either way they are reordered to
        `(Nx, Ny, Nz, D)` and rotated into voxel units, because a
        [`DisplacementField`][brainhops.datamodel.transformations.DisplacementField]
        adds its values in the units of its own grid.
        """  # noqa: E501
        vox2lps, shape = self._grid
        ndim = self.ndim_input

        # Ensure array-like
        parameters = self.parameters
        parameters = parameters if parameters is not None else np.array([])
        if not hasattr(parameters, "reshape"):
            parameters = get_array_backend(parameters).asarray(parameters)

        # Reorder to (Nx, Ny, Nz, D). `shape` is (Nx, Ny, Nz) while the
        # buffer is C-ordered with x varying fastest, so the spatial axes
        # come out reversed either way and are flipped back; only where
        # the component axis sits differs between the two layouts.
        spatial = range(ndim - 1, -1, -1)
        if self.interleaved:
            # (Nz, Ny, Nx, D) -> (Nx, Ny, Nz, D)
            disp = parameters.reshape(*reversed(shape), ndim)
            disp = disp.transpose(*spatial, ndim)
        else:
            # (D, Nz, Ny, Nx) -> (Nx, Ny, Nz, D)
            disp = parameters.reshape(ndim, *reversed(shape))
            disp = disp.transpose(*(axis + 1 for axis in spatial), 0)

        # Multiply by the world-to-voxel affine to convert from world
        # displacements to voxel displacements. Only the linear part
        # rotates a displacement, and it is the first `ndim` columns of
        # the compact affine.
        lps2vox = _affines.inv(vox2lps)
        backend = get_array_backend(disp)
        rotate = backend.asarray(lps2vox[:, :ndim], dtype=disp.dtype)
        return backend.matmul(rotate, disp[..., None])[..., 0]

    # --- slots --------------------------------------------------------

    @smartproperty(cache=True)
    def lps2voxel(self) -> LPSToVoxel:
        """The affine from LPS world coordinates to warp-grid voxels."""
        vox2lps, _ = self._grid
        return LPSToVoxel(matrix=_affines.inv(vox2lps))

    @smartproperty(cache=True)
    def displacement(self) -> _xforms.DisplacementField:
        """The displacement field, defined on the warp grid."""
        VOX = _systems.VoxelCoordinateSystem()
        return _xforms.DisplacementField(
            field=self.field,
            input=VOX,
            output=VOX,
            order=self.order,
            coeff=self.coeff,
            bound=self.bound,
        )

    @smartproperty(cache=True)
    def voxel2lps(self) -> VoxelToLPS:
        """The affine from warp-grid voxels back to LPS world."""
        vox2lps, _ = self._grid
        return VoxelToLPS(matrix=vox2lps)

    # --- sequence -----------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations that the block encodes.

        It is assembled from the named slots and cached. Assigning to it
        overrides the derived chain.

        It is a tuple rather than a list for the same reason as on
        [`ITKAffineBase`][]: the cached chain is handed out as is, and a
        list would let `del block[0]` edit the cache in place.
        """
        return (self.lps2voxel, self.displacement, self.voxel2lps)


# ----------------------------------------------------------------------
#   BLOCKS
# ----------------------------------------------------------------------


@_register_type("IdentityTransform")
class ITKIdentityStruct(ITKAffineBase):
    """Identity transform with no parameters."""

    type: tx.Literal[_ITKT.IdentityTransform] = _ITKT.IdentityTransform

    parameters: tx.Tuple[tx.Any, ...] = ()

    fixed_parameters: tx.Tuple[tx.Any, ...] = ()

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Identity:
        """The identity."""
        # A one-element chain, rather than an empty one, so that every
        # block is a sequence of at least one transformation and the
        # identity still names the spaces it maps between.
        return _xforms.Identity(input=self.input, output=self.output)


@_register_type("TranslationTransform")
class ITKTranslationStruct(ITKAffineBase):
    """
    Translation transform with parameters for translation in each dimension.
    """

    type: tx.Literal[_ITKT.TranslationTransform] = _ITKT.TranslationTransform

    fixed_parameters: tx.Tuple[tx.Any, ...] = ()

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters)


@_register_type("ScaleTransform")
class ITKScaleStruct(ITKAffineBase):
    """Scale transform with parameters for scaling in each dimension."""

    type: tx.Literal[_ITKT.ScaleTransform] = _ITKT.ScaleTransform

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Scaling:
        """The per-axis scaling."""
        return _xforms.Scaling(self.parameters)


@_register_type("ScaleLogarithmicTransform")
class ITKScaleLogarithmicStruct(ITKAffineBase):
    """
    Scale logarithmic transform with parameters for scaling in each dimension.
    """

    type: tx.Literal[_ITKT.ScaleLogarithmicTransform] = (
        _ITKT.ScaleLogarithmicTransform
    )

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Scaling:
        """The per-axis scaling, whose logarithm is stored."""
        return _xforms.Scaling(np.exp(self.parameters))


@_register_type("Euler2DTransform")
class ITKEuler2DStruct(ITKAffineBase):
    """Euler 2D transform with parameters for rotation and translation."""

    type: tx.Literal[_ITKT.Euler2DTransform] = _ITKT.Euler2DTransform

    ndim_input: tx.Literal[2] = 2
    ndim_output: tx.Literal[2] = 2

    parameters: tx.Tuple[float, float, float]
    """Rotation angle, followed by translation parameters."""

    fixed_parameters: tx.Tuple[float, float]
    """Center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Rotation:
        """The rotation, parameterized by its angle."""
        return _xforms.Rotation(_angle_to_matrix(self.parameters[0]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[1:3])


@_register_type("Euler3DTransform")
class ITKEuler3DStruct(ITKAffineBase):
    """Euler 3D transform with parameters for rotation and translation."""

    type: tx.Literal[_ITKT.Euler3DTransform] = _ITKT.Euler3DTransform

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[float, float, float, float, float, float]
    """Rotation angles (rx, ry, rz), followed by translation parameters."""

    fixed_parameters: tx.Tuple[float, float, float]
    """Center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Rotation:
        """The rotation, parameterized by its Euler angles."""
        return _xforms.Rotation(_euler_to_matrix(self.parameters[:3]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


@_register_type("VersorTransform")
class ITKVersorStruct(ITKAffineBase):
    """Versor transform with parameters for rotation in each dimension."""

    type: tx.Literal[_ITKT.VersorTransform] = _ITKT.VersorTransform

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[float, float, float]
    """Quaternion parameters."""

    fixed_parameters: tx.Tuple[float, float, float]
    """Center of rotation."""

    def __post_init__(self) -> None:
        self._check_same_ndim()

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Rotation:
        """The rotation, parameterized by a versor."""
        return _xforms.Rotation(_versor_to_matrix(self.parameters[:3]))


@_register_type("VersorRigid3DTransform")
class ITKVersorRigid3DStruct(ITKAffineBase):
    """
    Versor rigid 3D transform with parameters for rotation and translation.
    """

    type: tx.Literal[_ITKT.VersorRigid3DTransform] = (
        _ITKT.VersorRigid3DTransform
    )

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[float, float, float, float, float, float]
    """Quaternion parameters followed by translation vector."""

    fixed_parameters: tx.Tuple[float, float, float]
    """Center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Rotation:
        """The rotation, parameterized by a versor."""
        return _xforms.Rotation(_versor_to_matrix(self.parameters[:3]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


@_register_type("Similarity2DTransform")
class ITKSimilarity2DStruct(ITKAffineBase):
    """
    Similarity 2D transform with parameters for rotation, translation,
    and scaling.
    """

    type: tx.Literal[_ITKT.Similarity2DTransform] = _ITKT.Similarity2DTransform

    ndim_input: tx.Literal[2] = 2
    ndim_output: tx.Literal[2] = 2

    parameters: tx.Tuple[float, float, float, float]
    """Scale, angle, and (x, y) translation parameters."""

    fixed_parameters: tx.Tuple[float, float]
    """Center of rotation."""

    # ITK parameterizes the linear part by a scale and an angle, so the
    # two are named -- and chained -- individually.
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
        """The rotation, parameterized by its angle."""
        return _xforms.Rotation(_angle_to_matrix(self.parameters[1]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[2:4])


@_register_type("Similarity3DTransform")
class ITKSimilarity3DStruct(ITKAffineBase):
    """
    Similarity 3D transform with parameters for rotation, translation,
    and scaling.
    """

    type: tx.Literal[_ITKT.Similarity3DTransform] = _ITKT.Similarity3DTransform

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[float, float, float, float, float, float, float]
    """
    (qx, qy, qz) versor parameters, (tx, ty, tz) translation parameters,
    and scale.
    """

    fixed_parameters: tx.Tuple[float, float, float]
    """Center of rotation."""

    # ITK parameterizes the linear part by a scale and a versor, so the
    # two are named -- and chained -- individually.
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
        """The rotation, parameterized by a versor."""
        return _xforms.Rotation(_versor_to_matrix(self.parameters[0:3]))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


@_register_type("ScaleVersor3DTransform")
class ITKScaleVersor3DStruct(ITKAffineBase):
    """
    Scale versor 3D transform with parameters for rotation, translation,
    and scaling.
    """

    type: tx.Literal[_ITKT.ScaleVersor3DTransform] = (
        _ITKT.ScaleVersor3DTransform
    )

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[
        # 9 parameters
        float, float, float, float, float, float, float, float, float
    ]
    """
    (qx, qy, qz) versor parameters, (tx, ty, tz) translation parameters,
    and (sx, sy, sz) scale parameters.
    """

    fixed_parameters: tx.Tuple[float, float, float]
    """Center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Linear:
        """The rotation with the anisotropic scaling folded into it."""
        R = _versor_to_matrix(self.parameters[0:3])
        S = np.diag(np.asarray(self.parameters[6:9]) - 1)
        return _xforms.Linear(R + S)

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector."""
        return _xforms.Translation(self.parameters[3:6])


@_register_type("ScaleSkewVersor3DTransform")
class ITKScaleSkewVersor3DStruct(ITKAffineBase):
    """
    Scale skew versor 3D transform with parameters for rotation, translation,
    scaling, and skewing.
    """

    type: tx.Literal[_ITKT.ScaleSkewVersor3DTransform] = (
        _ITKT.ScaleSkewVersor3DTransform
    )

    ndim_input: tx.Literal[3] = 3
    ndim_output: tx.Literal[3] = 3

    parameters: tx.Tuple[
        # 15 parameters
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
    """
    (qx, qy, qz) versor parameters, (tx, ty, tz) translation parameters,
    (sx, sy, sz) scale parameters, and (kxy, kxz, kyx, kyz, kzx, kzy)
    skew parameters.
    """

    fixed_parameters: tx.Tuple[float, float, float]
    """Center of rotation."""

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Linear:
        """The rotation with the scaling and the skew folded into it."""
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


@_register_type("AffineTransform")
class ITKAffineStruct(ITKAffineBase):
    """
    Affine transform with parameters for linear transformation and translation.
    """

    type: tx.Literal[_ITKT.AffineTransform] = _ITKT.AffineTransform

    def __post_init__(self) -> None:
        self._check_same_ndim()
        ndim = self.ndim_input
        self._check_parameters_length((ndim + 1) * ndim)
        self._check_fixed_parameters_length(ndim)

    @smartproperty(cache=True)
    def linear(self) -> _xforms.Linear:
        """The linear part, stored row-major."""
        Di, Do = self.ndim_input, self.ndim_output
        L = np.array(self.parameters[: Di * Do], dtype=np.float64)
        return _xforms.Linear(L.reshape(Do, Di))

    @smartproperty(cache=True)
    def translation(self) -> _xforms.Translation:
        """The translation vector, stored after the linear part."""
        Do = self.ndim_output
        return _xforms.Translation(
            np.array(self.parameters[-Do:], dtype=np.float64)
        )


@_register_type("DisplacementFieldTransform")
class ITKDisplacementFieldStruct(ITKDisplacementBase):
    """
    Displacement field transform with parameters for a dense deformation map.

    The parameters hold one world-space displacement per voxel of the
    grid that the fixed parameters describe, so the field is sampled,
    not spline-encoded, and is interpolated linearly. They are the raw
    buffer of that image of vectors, so the components are interleaved.
    """

    type: tx.Literal[_ITKT.DisplacementFieldTransform] = (
        _ITKT.DisplacementFieldTransform
    )

    interleaved: tx.ClassVar[bool] = True


@_register_type("BSplineTransform")
class ITKBSplineStruct(ITKDisplacementBase):
    """
    B-spline transform with parameters for a dense deformation map.

    The parameters hold cubic B-spline coefficients on the control-point
    grid that the fixed parameters describe, so the field is evaluated
    -- not interpolated -- and coefficients outside the grid are zero.
    They are one scalar coefficient image per axis, written back to
    back, so the components are planar rather than interleaved.
    """

    type: tx.Literal[_ITKT.BSplineTransform] = _ITKT.BSplineTransform

    order: tx.ClassVar[int] = 3
    coeff: tx.ClassVar[bool] = True
    bound: tx.ClassVar[tx.Union[BoundaryCondition, float]] = (
        BoundaryCondition.zeros
    )
    interleaved: tx.ClassVar[bool] = False


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _inverse_chain(
    struct: _xforms.Sequence, compute: bool = False, **kwargs
) -> _xforms.Sequence:
    """The inverse of an ITK block, as a plain sequence.

    An ITK block derives its children from the parameters that its file
    stores, so the inverse of a block is not itself a block: it is the
    reversed chain of inverted children, and it is returned as a plain
    [`Sequence`][brainhops.datamodel.transformations.Sequence].
    """
    return _xforms.Sequence(
        transformations=[
            child.inverse(compute=compute, **kwargs)
            for child in reversed(struct.transformations or [])
        ],
        input=struct.output,
        output=struct.input,
    )


def _isotropic_scaling(scale: float, ndim: int) -> _xforms.Scaling:
    """An isotropic scaling, written out as one factor per axis.

    ITK stores a similarity's scale as a single number. It is repeated
    per axis rather than passed as a scalar, because a scalar `scale` is
    a zero-dimensional array that the affine converters cannot size.
    """
    return _xforms.Scaling(np.full(ndim, float(scale), dtype=np.float64))


def _nonempty(values: tx.Optional[ArrayProtocol]) -> tx.Optional[tx.Any]:
    """Return `values`, or `None` when it is empty.

    ITK writes an empty parameter vector for a parameter that a block
    does not use, and an empty vector is not a transformation: it is the
    absence of one.
    """
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
    """The voxel-to-LPS affine of a warp grid, and the shape of that grid.

    ITK writes the grid geometry into the fixed parameters as four
    consecutive blocks -- the shape, the origin, the voxel spacing and
    the direction matrix -- sized by the dimensionality of the block, so
    they are read off `ndim` rather than off a 3-D layout.

    The affine is returned in the *compact* `(ndim, ndim + 1)` form, with
    no homogeneous row, which is what
    [`brainhops._core.affines`][] and
    [`Affine`][brainhops.datamodel.transformations.Affine] both take. A
    homogeneous matrix handed to either is read as one dimension too
    many: `affines.inv` would answer with an `(ndim, ndim + 2)` matrix,
    and an `Affine` would claim to map `ndim + 1` coordinates.
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
    """Convert a 2D rotation angle to a rotation matrix."""
    return np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )


#: How far past the unit sphere a versor's vector part may reach and
#: still be read as a unit versor written imprecisely. A half-turn is
#: exactly on the sphere, and a file that stores it in single precision
#: -- or rounds it to a fixed number of decimals -- hands it back a few
#: ulps outside. ITK renormalizes such a vector rather than refusing it,
#: so a file that opens there must open here.
_VERSOR_TOLERANCE = 1e-6


def _versor_to_matrix(q: tx.Sequence[float]) -> np.ndarray:
    """Convert a versor (unit quaternion) to a rotation matrix."""
    qx, qy, qz = q
    norm_sq = qx**2 + qy**2 + qz**2
    if norm_sq > 1.0:
        # The scalar part of a versor is implied by its vector part, so a
        # vector part longer than the unit sphere has no scalar part to
        # complete it. Within the tolerance that is a rounding artifact
        # and the vector is scaled back onto the sphere; beyond it, the
        # value is not a versor and is refused.
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


def _euler_to_matrix(angles: tx.Sequence[float]) -> np.ndarray:
    """Convert Euler angles to a rotation matrix."""
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx
