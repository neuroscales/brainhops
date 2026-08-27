# stdlib
import math
from enum import StrEnum

import numpy as np

# dependencies
import typing_extensions as _tx

from brainhops._core import affines as _affines
from brainhops._core.backends import get_array_backend

# core
from brainhops._core.typing import ArrayProtocol

# externals
from brainhops._ext.struct import Struct
from brainhops.datamodel import systems as _systems

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
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


class ITKStruct(Struct, kw_only=True, convert=True):
    """This object represents a single ITK transform block."""

    _REGISTRY: _tx.ClassVar[_tx.Mapping[str, type]] = {}

    def __new__(cls, **kwargs) -> None:
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

    def _check_same_ndim(self, expected_ndim: int | None = None) -> None:
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


def _register_type(*names: str) -> _tx.Callable:

    def decorator(cls: type) -> type:
        for name in names:
            ITKStruct._REGISTRY[name] = cls
        return cls
    return decorator


@_register_type("IdentityTransform")
class ITKIdentityStruct(ITKStruct):
    """Identity transform with no parameters."""

    type: _tx.Literal[_ITKT.IdentityTransform] = _ITKT.IdentityTransform

    parameters: _tx.Tuple[()] = ()

    fixed_parameters: _tx.Tuple[()] = ()

    def to_transform(self) -> _xforms.Identity:
        """Return a copy of the identity transform."""
        return _xforms.Identity(
            input=_make_system(self.ndim_input),
            output=_make_system(self.ndim_output)
        )


@_register_type("TranslationTransform")
class ITKTranslationStruct(ITKStruct):
    """
    Translation transform with parameters for translation in each dimension.
    """

    type: _tx.Literal[_ITKT.TranslationTransform] = _ITKT.TranslationTransform

    fixed_parameters: _tx.Tuple[()] = ()

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    def to_transform(self) -> _xforms.Translation:
        """Return a translation transform with the specified parameters."""
        return _xforms.Translation(
            input=_make_system(self.ndim_input),
            output=_make_system(self.ndim_output),
            translation=self.parameters
        )


@_register_type("ScaleTransform")
class ITKScaleStruct(ITKStruct):
    """Scale transform with parameters for scaling in each dimension."""

    type: _tx.Literal[_ITKT.ScaleTransform] = _ITKT.ScaleTransform

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    def to_transform(self) -> _xforms.Sequence:
        """Return a scale transform with the specified parameters."""
        return _xforms.Sequence(
            input=_make_system(self.ndim_input),
            output=_make_system(self.ndim_output),
            transformations=[
                _xforms.Translation(self.fixed_parameters).inverse(),
                _xforms.Scaling(self.parameters),
                _xforms.Translation(self.fixed_parameters)
            ]
        )


@_register_type("ScaleLogarithmicTransform")
class ITKScaleLogarithmicStruct(ITKStruct):
    """
    Scale logarithmic transform with parameters for scaling in each dimension.
    """

    type: _tx.Literal[_ITKT.ScaleLogarithmicTransform] \
        = _ITKT.ScaleLogarithmicTransform

    def __post_init__(self) -> None:
        self._check_same_ndim()
        self._check_parameters_length(self.ndim_input)

    def to_transform(self) -> _xforms.Sequence:
        """
        Return a scale logarithmic transform with the specified parameters.
        """
        return _xforms.Sequence(
            input=_make_system(self.ndim_input),
            output=_make_system(self.ndim_output),
            transformations=[
                _xforms.Translation(self.fixed_parameters).inverse(),
                _xforms.Scaling(np.exp(self.parameters)),
                _xforms.Translation(self.fixed_parameters)
            ]
        )


@_register_type("Euler2DTransform")
class ITKEuler2DStruct(ITKStruct):
    """Euler 2D transform with parameters for rotation and translation."""

    type: _tx.Literal[_ITKT.Euler2DTransform] = _ITKT.Euler2DTransform

    ndim_input: _tx.Literal[2] = 2
    ndim_output: _tx.Literal[2] = 2

    parameters: _tx.Tuple[float, float, float]
    """Rotation angle, followed by translation parameters."""

    fixed_parameters: _tx.Tuple[float, float]
    """Center of rotation."""

    def to_transform(self) -> _xforms.Sequence:
        """Return an Euler 2D transform with the specified parameters."""

        angle = self.parameters[0]
        t = self.parameters[1:3]
        c = self.fixed_parameters

        R = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle),  math.cos(angle)]
        ], dtype=np.float64)

        return _xforms.Sequence(
            input=_make_system(2),
            output=_make_system(2),
            transformations=[
                _xforms.Translation(c).inverse(),
                _xforms.Rotation(R),
                _xforms.Translation(c),
                _xforms.Translation(t),
            ]
        )


@_register_type("Euler3DTransform")
class ITKEuler3DStruct(ITKStruct):
    """Euler 3D transform with parameters for rotation and translation."""

    type: _tx.Literal[_ITKT.Euler3DTransform] = _ITKT.Euler3DTransform

    ndim_input: _tx.Literal[3] = 3
    ndim_output: _tx.Literal[3] = 3

    parameters: _tx.Tuple[float, float, float, float, float, float]
    """Rotation angles (rx, ry, rz), followed by translation parameters."""

    fixed_parameters: _tx.Tuple[float, float, float]
    """Center of rotation."""

    def to_transform(self) -> _xforms.Sequence:
        """Return an Euler 3D transform with the specified parameters."""

        R = _euler_to_matrix(self.parameters[:3])
        t = self.parameters[3:6]
        c = self.fixed_parameters

        return _xforms.Sequence(
            input=_make_system(3),
            output=_make_system(3),
            transformations=[
                _xforms.Translation(c).inverse(),
                _xforms.Rotation(R),
                _xforms.Translation(c),
                _xforms.Translation(t),
            ]
        )


@_register_type("VersorTransform")
class ITKVersorStruct(ITKStruct):
    """Versor transform with parameters for rotation in each dimension."""

    type: _tx.Literal[_ITKT.VersorTransform] = _ITKT.VersorTransform

    ndim_input: _tx.Literal[3] = 3
    ndim_output: _tx.Literal[3] = 3

    parameters: _tx.Tuple[float, float, float]
    """Quaternion parameters."""

    fixed_parameters: _tx.Tuple[float, float, float]
    """Center of rotation."""

    def __post_init__(self) -> None:
        self._check_same_ndim()

    def to_transform(self) -> _xforms.Sequence:
        """Return a versor transform with the specified parameters."""

        c = self.fixed_parameters
        R = _versor_to_matrix(self.parameters)

        return _xforms.Sequence(
            input=_make_system(3),
            output=_make_system(3),
            transformations=[
                _xforms.Translation(c).inverse(),
                _xforms.Rotation(R),
                _xforms.Translation(c)
            ]
        )


@_register_type("VersorRigid3DTransform")
class ITKVersorRigid3DStruct(ITKStruct):
    """
    Versor rigid 3D transform with parameters for rotation and translation.
    """

    type: _tx.Literal[_ITKT.VersorRigid3DTransform] = \
        _ITKT.VersorRigid3DTransform

    ndim_input: _tx.Literal[3] = 3
    ndim_output: _tx.Literal[3] = 3

    parameters: _tx.Tuple[float, float, float, float, float, float]
    """Quaternion parameters followed by translation vector."""

    fixed_parameters: _tx.Tuple[float, float, float]
    """Center of rotation."""

    def to_transform(self) -> _xforms.Sequence:
        """Return a versor rigid 3D transform with the specified parameters."""

        q = ITKVersorStruct(
            precision=self.precision,
            parameters=self.parameters[:3],
            fixed_parameters=self.fixed_parameters
        )
        t = self.parameters[3:6]

        xform = q.to_transform()
        xform.transforms.append(_xforms.Translation(t))
        return xform


@_register_type("Similarity2DTransform")
class ITKSimilarity2DStruct(ITKStruct):
    """
    Similarity 2D transform with parameters for rotation, translation,
    and scaling.
    """

    type: _tx.Literal[_ITKT.Similarity2DTransform] = \
        _ITKT.Similarity2DTransform

    ndim_input: _tx.Literal[2] = 2
    ndim_output: _tx.Literal[2] = 2

    parameters: _tx.Tuple[float, float, float, float]
    """Scale, angle, and (x, y) translation parameters."""

    fixed_parameters: _tx.Tuple[float, float]
    """Center of rotation."""

    def to_transform(self) -> _xforms.Sequence:

        scale, angle, tx, ty = self.parameters
        c = self.fixed_parameters

        R = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle),  math.cos(angle)]
        ], dtype=np.float64)

        return _xforms.Sequence(
            input=_make_system(2),
            output=_make_system(2),
            transformations=[
                _xforms.Translation(c).inverse(),
                _xforms.Scaling(scale),
                _xforms.Rotation(R),
                _xforms.Translation(c),
                _xforms.Translation((tx, ty)),
            ]
        )


@_register_type("Similarity3DTransform")
class ITKSimilarity3DStruct(ITKStruct):
    """
    Similarity 3D transform with parameters for rotation, translation,
    and scaling.
    """

    type: _tx.Literal[_ITKT.Similarity3DTransform] = \
        _ITKT.Similarity3DTransform

    ndim_input: _tx.Literal[3] = 3
    ndim_output: _tx.Literal[3] = 3

    parameters: _tx.Tuple[float, float, float, float, float, float, float]
    """
    (qx, qy, qz) versor parameters, (tx, ty, tz) translation parameters,
    and scale.
    """

    fixed_parameters: _tx.Tuple[float, float, float]
    """Center of rotation."""

    def to_transform(self) -> _xforms.Sequence:
        c = self.fixed_parameters
        q = self.parameters[0:3]
        t = self.parameters[3:6]
        s = self.parameters[6]
        R = _versor_to_matrix(q)

        return _xforms.Sequence(
            input=_make_system(3),
            output=_make_system(3),
            transformations=[
                _xforms.Translation(c).inverse(),
                _xforms.Scaling(s),
                _xforms.Rotation(R),
                _xforms.Translation(c),
                _xforms.Translation(t),
            ]
        )


@_register_type("ScaleVersor3DTransform")
class ITKScaleVersor3DStruct(ITKStruct):
    """
    Scale versor 3D transform with parameters for rotation, translation,
    and scaling.
    """

    type: _tx.Literal[_ITKT.ScaleVersor3DTransform] \
        = _ITKT.ScaleVersor3DTransform

    ndim_input: _tx.Literal[3] = 3
    ndim_output: _tx.Literal[3] = 3

    parameters: _tx.Tuple[
        # 9 parameters
        float, float, float,
        float, float, float,
        float, float, float
    ]
    """
    (qx, qy, qz) versor parameters, (tx, ty, tz) translation parameters,
    and (sx, sy, sz) scale parameters.
    """

    fixed_parameters: _tx.Tuple[float, float, float]
    """Center of rotation."""

    def to_transform(self) -> _xforms.Sequence:
        c = self.fixed_parameters
        q = self.parameters[0:3]
        t = self.parameters[3:6]
        s = self.parameters[6:9]
        R = _versor_to_matrix(q)
        S = np.diag(np.asarray(s) - 1)

        return _xforms.Sequence(
            input=_make_system(3),
            output=_make_system(3),
            transformations=[
                _xforms.Translation(c).inverse(),
                _xforms.Linear(R+S),
                _xforms.Translation(c),
                _xforms.Translation(t),
            ]
        )


@_register_type("ScaleSkewVersor3DTransform")
class ITKScaleSkewVersor3DStruct(ITKStruct):
    """
    Scale skew versor 3D transform with parameters for rotation, translation,
    scaling, and skewing.
    """

    type: _tx.Literal[_ITKT.ScaleSkewVersor3DTransform] \
        = _ITKT.ScaleSkewVersor3DTransform

    ndim_input: _tx.Literal[3] = 3
    ndim_output: _tx.Literal[3] = 3

    parameters: _tx.Tuple[
        # 15 parameters
        float, float, float,
        float, float, float,
        float, float, float,
        float, float, float,
        float, float, float,
    ]
    """
    (qx, qy, qz) versor parameters, (tx, ty, tz) translation parameters,
    (sx, sy, sz) scale parameters, and (kxy, kxz, kyx, kyz, kzx, kzy)
    skew parameters.
    """

    fixed_parameters: _tx.Tuple[float, float, float]
    """Center of rotation."""

    def to_transform(self) -> _xforms.Sequence:
        c = self.fixed_parameters
        q = self.parameters[0:3]
        t = self.parameters[3:6]
        s = self.parameters[6:9]
        k = self.parameters[9:15]
        R = _versor_to_matrix(q)
        S = np.diag(np.asarray(s) - 1)
        K = np.array([
            [0, k[0], k[1]],
            [k[2], 0, k[3]],
            [k[4], k[5], 0]
        ], dtype=np.float64)

        return _xforms.Sequence(
            input=_make_system(3),
            output=_make_system(3),
            transformations=[
                _xforms.Translation(c).inverse(),
                _xforms.Linear(R+S+K),
                _xforms.Translation(c),
                _xforms.Translation(t),
            ]
        )


@_register_type("AffineTransform")
class ITKAffineStruct(ITKStruct):
    """
    Affine transform with parameters for linear transformation and translation.
    """

    type: _tx.Literal[_ITKT.AffineTransform] = _ITKT.AffineTransform

    def __post_init__(self) -> None:
        self._check_same_ndim()
        ndim = self.ndim_input
        self._check_parameters_length((ndim + 1) * ndim)
        self._check_fixed_parameters_length(ndim)

    def to_transform(self) -> _xforms.Sequence:
        """Return an affine transform with the specified parameters."""

        Di = self.ndim_input
        Do = self.ndim_output
        L = np.array(self.parameters[:Di * Do], dtype=np.float64)
        L = L.reshape(Do, Di)
        t = np.array(self.parameters[-Do:], dtype=np.float64)
        c = self.fixed_parameters

        return _xforms.Sequence(
            input=_make_system(Di),
            output=_make_system(Do),
            transformations=[
                _xforms.Translation(c).inverse(),
                _xforms.Linear(L),
                _xforms.Translation(c),
                _xforms.Translation(t),
            ]
        )


@_register_type("DisplacementFieldTransform")
class ITKDisplacementFieldStruct(ITKStruct):
    """
    Displacement field transform with parameters for a dense deformation map.
    """

    type: _tx.Literal[_ITKT.DisplacementFieldTransform] = \
        _ITKT.DisplacementFieldTransform

    def to_transform(self) -> _xforms.DisplacementField:

        # Get geometry of the B-spline grid
        # -> Assumig a voxel grid ordered [Nx, Ny, Nz]
        vox2lps, shape = _vox2lps(self.fixed_parameters)

        # Ensure array-like
        parameters = self.parameters
        if not hasattr(parameters, "reshape"):
            parameters = get_array_backend(parameters).asarray(parameters)

        # Reorder from (3, Nz, Ny, Nx) to (Nx, Ny, Nz, 3)
        disp = parameters.reshape(3, *reversed(shape))
        disp = disp.transpose(3, 2, 1, 0)

        # Compute the linear part of the world-to-voxel affine
        lps2vox = _affines.inv(vox2lps)

        # Multiply by the world-to-voxel affine to convert from world
        # displacements to voxel displacements
        backend = get_array_backend(disp)
        rotate = backend.asarray(lps2vox[:3, :3], dtype=disp.dtype)
        disp = backend.matmul(rotate, disp[..., None])[..., 0]

        VOX = _systems.VoxelCoordinateSystem()
        return _xforms.Sequence(
            [
                LPSToVoxel(lps2vox),
                _xforms.DisplacementField(
                    disp,
                    input=VOX,
                    output=VOX,
                ),
                VoxelToLPS(vox2lps),
            ],
        )


@_register_type("BSplineTransform")
class ITKBSplineStruct(ITKStruct):
    """
    B-spline transform with parameters for a dense deformation map.
    """

    type: _tx.Literal[_ITKT.BSplineTransform] = _ITKT.BSplineTransform

    def to_transform(self) -> _xforms.DisplacementField:

        # Get geometry of the B-spline grid
        # -> Assumig a voxel grid ordered [Nx, Ny, Nz]
        vox2lps, shape = _vox2lps(self.fixed_parameters)

        # Ensure array-like
        parameters = self.parameters
        if not hasattr(parameters, "reshape"):
            parameters = get_array_backend(parameters).asarray(parameters)

        # The coefficients have shape [Nx, Ny, Nz, 3] (F-ordered),
        # resulting in a C-ordered shape of [3, Nz, Ny, Nx].
        # We reshape and reorder dimensions to recover [Nx, Ny, Nz, 3].
        coeff = parameters.reshape(3, *reversed(shape))
        coeff = coeff.transpose(3, 2, 1, 0)

        # Compute the linear part of the world-to-voxel affine
        lps2vox = _affines.inv(vox2lps)

        disp = parameters.reshape(3, *reversed(shape))
        disp = disp.transpose(3, 2, 1, 0)

        # Multiply by the world-to-voxel affine to convert from world
        # displacements to voxel displacements
        backend = get_array_backend(disp)
        rotate = backend.asarray(lps2vox[:3, :3], dtype=disp.dtype)
        disp = backend.matmul(rotate, disp[..., None])[..., 0]

        VOX = _systems.VoxelCoordinateSystem()
        return _xforms.Sequence(
            [
                LPSToVoxel(lps2vox),
                _xforms.DisplacementField(
                    disp,
                    input=VOX,
                    output=VOX,
                    order=3,
                    coeff=True,
                    bound="zeros",
                ),
                VoxelToLPS(vox2lps),
            ],
        )


def _vox2lps(fixed_parameters: _tx.Sequence[float]) -> np.ndarray:
    """Compute the node-to-world affine of a B-splines transform."""

    fixed_parameters = np.asarray(fixed_parameters, dtype=np.float64)
    shape = fixed_parameters[0:3]
    origin = fixed_parameters[3:6]
    spacing = fixed_parameters[6:9]
    direction = fixed_parameters[9:18].reshape(3, 3)

    shape = tuple(map(int, map(round, shape)))

    vox2lps = np.eye(4, dtype=np.float64)
    vox2lps[:3, :3] = direction @ np.diag(spacing)
    vox2lps[:3, 3] = origin
    return vox2lps, shape


def _versor_to_matrix(q: _tx.Sequence[float]) -> np.ndarray:
    """Convert a versor (unit quaternion) to a rotation matrix."""
    qx, qy, qz = q
    norm_sq = qx ** 2 + qy ** 2 + qz ** 2
    if norm_sq > 1.0:
        raise ValueError(
            "Versor quaternion vector part has magnitude > 1")
    qw = math.sqrt(max(0.0, 1.0 - norm_sq))
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def _euler_to_matrix(angles: _tx.Sequence[float]) -> np.ndarray:
    """Convert Euler angles to a rotation matrix."""
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx
