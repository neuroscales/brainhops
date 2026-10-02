# A good portion of the code found in this file are lightly modified versions
# of code found on the TIRL GitHub page. All credit goes to the creators.

# stdlib
from numbers import Number

# dependencies
import numpy as np
import typing_extensions as tx

from brainhops._core.enum import StrEnum

# externals
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.base import DataModelBase


# TODO: From my googling it seems the TIRL files do not specify their
# coordinate system. However I am leaving this helper function here in case
# that is wrong
def _make_system(ndim: int) -> _systems.SpatialCoordinateSystem:
    """
    Create a spatial coordinate system with the given number of dimensions.
    """
    return None


# ----------------------------------------------------------------------
#   Enum and registry
# ----------------------------------------------------------------------


class TIRLEnumClass(StrEnum):
    """TIRL transform class name mappings."""

    TranslationTransform = "TxTranslation"
    LinearTransform = "TxLinear"
    ShearTransform = "TxShear"
    AffineTransform = "TxAffine"
    RotationTransform = "TxRotation"
    AxisAngleTransform = "TxAxisAngle"
    Rotation2DTransform = "TxRotation2D"
    EulerAnglesTransform = "TxEularAngles"
    QuaternionTransform = "TxQuaternion"
    ScaleTransform = "TxScale"
    ISOScaleTransform = "TxIsoScale"
    IdentityTransform = "TxIdentity"
    DisplacementTransform = "TxDisplacementField"
    RbfDisplacementTransform = "TxRbfDisplacementField"
    DirectTransform = "TxDirect"
    Parameters = "ParameterVector"
    Domain = "Domain"
    Chain = "Chain"
    Image = "TImage"


_TIRLT = TIRLEnumClass


def _register_type(*names: str) -> tx.Callable:
    def decorator(cls: type) -> type:
        for name in names:
            TIRLStruct._REGISTRY[name] = cls
        return cls

    return decorator


# ----------------------------------------------------------------------
#   Base struct
# ----------------------------------------------------------------------


class TIRLStruct(DataModelBase, kw_only=True, convert=True):
    _REGISTRY: tx.ClassVar[tx.Mapping[str, type]] = {}

    def __new__(cls, **kwargs: dict) -> "TIRLStruct":
        if cls is not TIRLStruct:
            return super().__new__(cls)
        if not hasattr(cls, "_REGISTRY"):
            cls._REGISTRY = {}
        cls = cls._REGISTRY.get(kwargs.get("type"), None)
        return super().__new__(cls)

    type: tx.Optional[TIRLEnumClass] = None
    id: tx.Optional[tx.Any] = None


# ----------------------------------------------------------------------
#   Parameters
# ----------------------------------------------------------------------


@_register_type("ParameterVector")
class TIRLParametersStruct(TIRLStruct):
    type: tx.Literal[_TIRLT.Parameters] = _TIRLT.Parameters

    parameters: tx.Optional[tx.Union[ArrayProtocol, list, tuple]] = None
    lower_bounds: tx.Optional[tx.Union[Number, ArrayProtocol, list]] = None
    upper_bounds: tx.Optional[
        tx.Union[
            Number,
            ArrayProtocol,
            list,
        ]
    ] = None
    locked: tx.Optional[tx.Union[set, ArrayProtocol]] = None
    name: tx.Optional[str] = None
    signature: tx.Optional[tx.Any] = None

    def __post_init__(self) -> None:
        if len(self.parameters) == 0:
            self.parameters = np.array([], dtype="f8")
        elif (len(self.parameters) == 1) and hasattr(
            self.parameters[0], "__iter__"
        ):
            self.parameters = np.asanyarray(self.parameters[0]).ravel()
        else:
            self.parameters = np.array(self.parameters).ravel()

        if isinstance(self.parameters, ArrayProtocol):
            self.parameters = self.parameters.ravel()
        elif isinstance(self.parameters, ArrayProtocol):
            self._parameters = np.ascontiguousarray(self.parameters.ravel())
        elif hasattr(self.parameters, "__iter__"):
            p = np.asarray(self.parameters)
            if np.issubdtype(p.dtype, np.number):
                self._parameters = np.ascontiguousarray(p.ravel())
            else:
                raise TypeError("Transformation parameters must be numeric.")
        else:
            raise TypeError("Transformation parameters must be iterable.")


# ----------------------------------------------------------------------
#   Linear transforms
# ----------------------------------------------------------------------


@_register_type("TxTranslation")
class TIRLTranslationStruct(TIRLStruct):
    type: tx.Literal[_TIRLT.TranslationTransform] = _TIRLT.TranslationTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    def to_transform(self) -> _xforms.Translation:
        ndim = self.parameters.parameters.shape[0]
        return _xforms.Translation(
            input=_make_system(ndim),
            output=_make_system(ndim),
            translation=self.parameters.parameters,
        )


@_register_type("TxLinear")
class TIRLLinearStruct(TIRLStruct):
    type: tx.Literal[_TIRLT.LinearTransform] = _TIRLT.LinearTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    @staticmethod
    def params2matrix(
        parameters: ArrayProtocol, **kwargs: dict
    ) -> ArrayProtocol:
        """
        Builds a linear transformation matrix from a flat parameter vector.
        The base class fills the matrix in C (row-major) order. Subclasses
        override this for their own parameterisation (e.g. rotation angles).
        """
        shape = kwargs.get("shape")
        parameters = np.asarray(parameters, dtype="f8")
        mat = np.eye(*shape, dtype=parameters.dtype)
        mat.flat[:] = parameters
        return mat

    def _read_parameters_and_shape(
        self, param: object, metaparameters: dict
    ) -> tuple[ArrayProtocol, tuple[int]]:
        """
        Normalises the parameter input into a flat array and resolves the
        matrix shape. Accepts a 2D matrix, a 1D ndarray, a numeric iterable,
        or a sequence of scalars.
        """
        mat = param.parameters[0]
        shape = metaparameters.get("shape")

        if (
            isinstance(mat, ArrayProtocol)
            and mat.ndim == 2
            and np.issubdtype(mat.dtype, np.number)
        ):
            shape = mat.shape
            parameters = self.matrix2params(mat, **metaparameters)

        elif (
            isinstance(mat, ArrayProtocol)
            and mat.ndim == 1
            and np.issubdtype(mat.dtype, np.number)
        ):
            parameters = mat.ravel()

        elif hasattr(mat, "__iter__") and all(
            isinstance(v, Number) for v in mat
        ):
            parameters = np.asarray(mat).ravel()

        elif all(isinstance(v, Number) for v in param.parameters):
            parameters = np.asarray(param.parameters).ravel()

        else:
            raise TypeError(f"Unrecognised parameter specification: {param}")

        return parameters, shape

    def to_transform(self) -> _xforms.Linear:
        param, shape = self._read_parameters_and_shape(
            self.parameters, self.metaparameters
        )
        return _xforms.Linear(
            input=_make_system(shape[1]),
            output=_make_system(shape[1]),
            matrix=self.params2matrix(
                parameters=param,
                shape=shape,
                metaparameters=self.metaparameters,
            ),
        )


@_register_type("TxScale")
class TIRLScaleStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.ScaleTransform] = _TIRLT.ScaleTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    def to_transform(self) -> _xforms.Scaling:
        param, shape = self._read_parameters_and_shape(
            self.parameters, self.metaparameters
        )
        return _xforms.Scaling(
            input=param.shape[0],
            output=param.shape[0],
            scale=param,
        )


@_register_type("TxIsoScale")
class TIRLIsoScaleStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.ISOScaleTransform] = _TIRLT.ISOScaleTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    def to_transform(self) -> _xforms.Scaling:
        param, shape = self._read_parameters_and_shape(
            self.parameters, self.metaparameters
        )
        return _xforms.Scaling(
            input=shape[0],
            output=shape[0],
            scale=[param[0]] * shape[0],
        )


@_register_type("TxShear")
class TIRLShearStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.ShearTransform] = _TIRLT.ShearTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    @staticmethod
    def params2matrix(
        parameters: ArrayProtocol, **kwargs: dict
    ) -> ArrayProtocol:
        shape = kwargs.get("shape")
        parameters = np.asarray(parameters, dtype="f8")
        mat = np.eye(*shape, dtype=parameters.dtype)
        indices = kwargs.get("metaparameters", {}).get("indices")
        mat[indices] = parameters
        return mat


@_register_type("TxRotation")
class TIRLRotationStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.RotationTransform] = _TIRLT.RotationTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None


@_register_type("TxRotation2D")
class TIRLRotation2DStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.Rotation2DTransform] = _TIRLT.Rotation2DTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    @staticmethod
    def params2matrix(
        parameters: ArrayProtocol, **kwargs: dict
    ) -> ArrayProtocol:
        """
        Builds a (2, 2) rotation matrix from a single angle (radians).
        If a rotation centre is provided, the result is (2, 3) to account
        for the pre/post translation around that centre.
        """
        dtype = "f8"
        parameters = np.asarray(parameters, dtype=dtype)
        shape = kwargs.get("shape")
        mat = np.eye(*shape, dtype=dtype)
        phi = parameters[0]
        mat[:2, :2] = np.array(
            [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]],
            dtype=dtype,
        )

        centre = kwargs.get("centre", False)
        if centre:
            trans = np.eye(3)
            trans[:-1, -1] = -centre
            rot = np.eye(3)
            rot[:-1, :] = mat
            mat = np.linalg.inv(trans) @ rot @ trans
            mat = mat[:2, :]

        return mat


@_register_type("TxAxisAngle")
class TIRLAxisAngleStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.AxisAngleTransform] = _TIRLT.AxisAngleTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    @staticmethod
    def params2matrix(
        parameters: ArrayProtocol, **kwargs: dict
    ) -> ArrayProtocol:
        """
        Creates (3, 3) rotation matrix from rotation angle and axis as
        parameters using the Rodrigues formula.

        :param parameters:
            Sequence of angle and axis of the rotation:
            (angle, axis_x, axis_y, axis_z)
        :type parameters: ArrayProtocol
        :param kwargs:
            Keyword arguments required to retrieve rotation matrix from
            parameters.
        :type kwargs: Any

        :returns: (3, 3) rotation matrix
        :rtype: ArrayProtocol

        """
        dtype = "f8"
        parameters = np.asarray(parameters, dtype=dtype)
        phi = parameters[0]
        e = parameters[1:]
        cross = np.asarray(
            [[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]], dtype=dtype
        )
        I = np.eye(3, dtype=dtype)
        R = (
            np.cos(phi) * I
            + (1 - np.cos(phi)) * np.outer(e, e)
            + np.sin(phi) * cross
        )

        centre = kwargs.get("centre", False)
        if centre:
            mat = np.eye(4)
            mat[:3, :3] = R
            trans = np.eye(4)
            trans[:-1, -1] = -centre
            R = np.linalg.inv(trans) @ mat @ trans
            R = R[:3, :]

        return R


@_register_type("TxEulerAngles")
class TIRLEulerAnglesStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.EulerAnglesTransform] = _TIRLT.EulerAnglesTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    @staticmethod
    def _Rx(phi: float) -> ArrayProtocol:
        s = np.sin(phi)
        c = np.cos(phi)
        mat = np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]])
        return mat

    @staticmethod
    def _Ry(phi: float) -> ArrayProtocol:
        s = np.sin(phi)
        c = np.cos(phi)
        mat = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return mat

    @staticmethod
    def _Rz(phi: float) -> ArrayProtocol:
        s = np.sin(phi)
        c = np.cos(phi)
        mat = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return mat

    @staticmethod
    def params2matrix(
        parameters: ArrayProtocol, **kwargs: dict
    ) -> ArrayProtocol:
        """
        Creates (3, 3) rotation matrix from rotation angles, given a specific
        order of rotation axes. If the rotation has a centre that is different
        from the origin, the returned matrix is (3, 4).

        :param parameters:
            Rotation angles in radians.
        :type parameters: ArrayProtocol
        :param kwargs:
            Keyword arguments required to retrieve rotation matrix
            from parameters.
        :type kwargs: Any

        :returns: (3, 3) rotation matrix or (3, 4) eccentric rotation matrix
        :rtype: ArrayProtocol

        """
        dtype = "f8"
        order = kwargs.get("order")
        parameters = np.asarray(parameters, dtype=dtype)
        Rax = {
            "x": TIRLEulerAnglesStruct._Rx,
            "y": TIRLEulerAnglesStruct._Ry,
            "z": TIRLEulerAnglesStruct._Rz,
        }
        R = np.eye(3, dtype=dtype)
        for i, ax in enumerate(order):
            R = np.dot(R, Rax[ax](parameters[i]).astype(dtype))

        # Taking the rotation centre into account
        centre = kwargs.get("centre", False)
        if centre:
            mat = np.eye(4)
            mat[:3, :3] = R
            trans = np.eye(4)
            trans[:-1, -1] = -centre
            R = np.linalg.inv(trans) @ mat @ trans
            R = R[:3, :]

        return R


@_register_type("TxQuaternion")
class TIRLQuaternionStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.QuaternionTransform] = _TIRLT.QuaternionTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    @staticmethod
    def params2matrix(
        parameters: ArrayProtocol, **kwargs: dict
    ) -> ArrayProtocol:
        """
        Creates (3, 3) rotation matrix from quaternion.

        :param parameters: quaternion components
        :type parameters: Union[int, float]
        :param kwargs:
            Keyword arguments required to retrieve matrix from parameters.
        :type kwargs: Any

        :returns: (3, 3) rotation matrix
        :rtype: ArrayProtocol

        """
        qw, qx, qy, qz = parameters
        q_vect = np.asarray([qx, qy, qz]).reshape((-1, 1))
        Q = np.asarray([[0, -qz, qy], [qz, 0, -qx], [-qy, qx, 0]])
        I = np.eye(3)
        R = (
            (qw**2 - np.dot(q_vect.T, q_vect)) * I
            + 2 * np.dot(q_vect, q_vect.T)
            + 2 * qw * Q
        )

        # Taking the rotation centre into account
        centre = kwargs.get("centre", False)
        if centre:
            mat = np.eye(4)
            mat[:3, :3] = R
            trans = np.eye(4)
            trans[:-1, -1] = -centre
            R = np.linalg.inv(trans) @ mat @ trans
            R = R[:3, :]

        return R


@_register_type("TxAffine")
class TIRLAffineStruct(TIRLLinearStruct):
    type: tx.Literal[_TIRLT.AffineTransform] = _TIRLT.AffineTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    def to_transform(self) -> _xforms.Affine:
        param, shape = self._read_parameters_and_shape(
            self.parameters, self.metaparameters
        )
        return _xforms.Affine(
            input=_make_system(shape[1]),
            output=_make_system(shape[1]),
            matrix=self.params2matrix(
                parameters=param,
                shape=shape,
                metaparameters=self.metaparameters,
            ),
        )


@_register_type("TxIdentity")
class TIRLIdentityStruct(TIRLStruct):
    type: tx.Literal[_TIRLT.IdentityTransform] = _TIRLT.IdentityTransform

    parameters: tx.Optional[
        tx.Union[TIRLParametersStruct, ArrayProtocol, list, tuple]
    ] = None
    metaparameters: tx.Optional[dict] = None

    def to_transform(self) -> _xforms.Identity:
        return _xforms.Identity(input=None, output=None)


# ----------------------------------------------------------------------
#   Chain and Domain
# ----------------------------------------------------------------------


@_register_type("Chain")
class TIRLChainStruct(TIRLStruct):
    type: tx.Literal[_TIRLT.Chain] = _TIRLT.Chain

    transformations: tx.Optional[tx.List[TIRLStruct]] = None

    def to_transform(self) -> _xforms.Sequence:
        transform_list = [t.to_transform() for t in self.transformations]
        return _xforms.Sequence(
            input=transform_list[0].input,
            output=transform_list[-1].output,
            transformations=transform_list,
        )


@_register_type("TxDirect")
class TIRLDirectStruct(TIRLStruct):
    """
    Represents a direct coordinate assignment (TIRL's TxDirect).
    Maps integer voxel indices to pre-stored physical coordinates via a
    lookup table — used for sparse/non-compact domains where coordinates
    are not on a regular grid.
    """

    type: tx.Literal[_TIRLT.DirectTransform] = _TIRLT.DirectTransform

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None

    def to_transform(self) -> _xforms.CoordinatesField:
        ndim = self.metaparameters.get("ndim")
        coords = self.parameters.parameters.reshape(-1, ndim)
        return _xforms.CoordinatesField(
            input=_make_system(ndim),
            output=_make_system(ndim),
            field=coords,
        )


@_register_type("Domain")
class TIRLDomainStruct(TIRLStruct):
    type: tx.Literal[_TIRLT.Domain] = _TIRLT.Domain

    name: tx.Optional[tx.Any] = None
    shape: tx.Optional[tx.Any] = None
    coordinates: tx.Optional[tx.Any] = None
    dtype: tx.Optional[tx.Any] = None
    internal: tx.Optional[TIRLChainStruct] = None
    external: tx.Optional[TIRLChainStruct] = None
    memlimit: tx.Optional[tx.Any] = None
    storage: tx.Optional[tx.Any] = None

    def make_identity_grid(
        self, shape: tuple, dtype: np.dtype = np.int32
    ) -> ArrayProtocol:
        """Returns a voxel coordinate grid of shape (*spatial_shape, ndim)."""
        # TODO: use dask if loaded
        g_vals = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        identity = np.ones((*shape, len(shape)))
        for i in range(len(g_vals)):
            identity[(*[slice(None)] * len(shape), i)] = g_vals[i]
        return identity

    def _get_seed_coordinates(self) -> ArrayProtocol:
        """
        Returns the seed coordinates for the domain.

        For compact (regular grid) domains, generates a voxel index grid.
        For sparse (non-compact) domains, uses the explicitly stored
        coordinates directly — equivalent to TIRL's TxDirect lookup table
        behaviour.
        """
        ndim = len(self.shape)
        if self.coordinates is not None:
            # Sparse domain: coordinates are stored explicitly
            coords = np.asarray(self.coordinates)
            if coords.ndim == 1:
                coords = coords.reshape(-1, ndim)
            return coords
        else:
            # Compact domain: generate a regular voxel index grid
            return self.make_identity_grid(self.shape)

    def to_transform(self) -> _xforms.Sequence:
        ndim = len(self.shape)
        transformations = [
            _xforms.CoordinatesField(
                input=_make_system(ndim),
                output=_make_system(ndim),
                field=self._get_seed_coordinates(),
            )
        ]
        if self.internal:
            transformations.append(self.internal.to_transform())
        if self.external:
            transformations.append(self.external.to_transform())

        return _xforms.Sequence(
            input=transformations[0].input,
            output=transformations[-1].output,
            transformations=transformations,
        )


# TODO: add support for embedding (will require additional brainhops xform)


# ----------------------------------------------------------------------
#   Nonlinear transform
# ----------------------------------------------------------------------


@_register_type("TxDisplacementField")
class TIRLDisplacementFieldStruct(TIRLStruct):
    type: tx.Literal[_TIRLT.DisplacementTransform] = (
        _TIRLT.DisplacementTransform
    )

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None
    domain: tx.Optional[TIRLDomainStruct] = None
    vectorder: tx.Optional[tx.Any] = None
    interpolator: tx.Optional[dict] = None

    def to_transform(self) -> _xforms.DisplacementField:
        if self.domain is None:
            self.domain = self.metaparameters["domain"]

        domain_shape = tuple(self.domain.shape)
        ndim = len(domain_shape)
        vectorder = (
            tuple(self.vectorder)
            if self.vectorder is not None
            else tuple(range(ndim))
        )
        vectdim = len(vectorder)
        mode = self.metaparameters.get("mode", "abs")

        # Reshape flat parameters into (*spatial_shape, vectdim)
        compact_field = self.parameters.parameters.reshape(
            (*domain_shape, vectdim)
        )

        # Expand into (*spatial_shape, ndim), zeroing axes not in vectorder
        full_field = np.zeros((*domain_shape, ndim), dtype=compact_field.dtype)
        for i, ax in enumerate(vectorder):
            full_field[..., ax] = compact_field[..., i]

        if mode == "rel":
            # Vectors are in voxel space — scale to physical space using
            # the voxel size from the first internal transform (TxScale)
            scale = self.domain.internal.transformations[
                0
            ].parameters.parameters
            full_field = full_field * scale

        seq = self.domain.to_transform()
        return _xforms.DisplacementField(
            input=seq.output,
            output=seq.output,
            field=full_field,
        )


@_register_type("TxRbfDisplacementField")
class TIRLRbfDisplacementFieldStruct(TIRLStruct):
    """
    Represents a sparse RBF displacement field (TIRL's TxRbfDisplacementField).

    Vectors are defined at scattered support points rather than on a regular
    grid. Because your DisplacementField expects a dense grid, this struct
    converts the sparse field to a dense representation by reshaping the
    stored parameters onto a regular grid defined by `dense_shape`.

    If `dense_shape` is not available in metaparameters, it falls back to
    treating the field the same as a regular TIRLDisplacementFieldStruct
    using the domain shape directly.
    """

    type: tx.Literal[_TIRLT.RbfDisplacementTransform] = (
        _TIRLT.RbfDisplacementTransform
    )

    parameters: tx.Optional[TIRLParametersStruct] = None
    metaparameters: tx.Optional[dict] = None
    domain: tx.Optional[TIRLDomainStruct] = None
    vectorder: tx.Optional[tx.Any] = None
    interpolator: tx.Optional[dict] = None

    def to_transform(self) -> _xforms.Sequence:
        if self.domain is None:
            self.domain = self.metaparameters["domain"]

        # dense_shape defines the regular grid the sparse field is rasterised
        # onto
        dense_shape = self.metaparameters.get("dense_shape")
        if dense_shape is None or dense_shape == "auto":
            # Fall back to the domain shape if dense_shape is unavailable
            domain_shape = tuple(self.domain.shape)
        else:
            domain_shape = tuple(dense_shape)

        ndim = len(domain_shape)
        vectorder = (
            tuple(self.vectorder)
            if self.vectorder is not None
            else tuple(range(ndim))
        )
        vectdim = len(vectorder)
        mode = self.metaparameters.get("mode", "abs")

        # Reshape flat parameters into (*domain_shape, vectdim)
        # The sparse field is stored flattened in the same order as the
        # dense grid, so reshaping directly gives the dense representation
        compact_field = self.parameters.parameters.reshape(
            (*domain_shape, vectdim)
        )

        # Expand into (*domain_shape, ndim), zeroing axes not in vectorder
        full_field = np.zeros((*domain_shape, ndim), dtype=compact_field.dtype)
        for i, ax in enumerate(vectorder):
            full_field[..., ax] = compact_field[..., i]

        if mode == "rel":
            # Vectors are in voxel space — scale to physical space using
            # the voxel size from the first internal transform (TxScale)
            scale = self.domain.internal.transformations[
                0
            ].parameters.parameters
            full_field = full_field * scale

        seq = self.domain.to_transform()
        return _xforms.DisplacementField(
            input=seq.output,
            output=seq.output,
            field=full_field,
        )


# ----------------------------------------------------------------------
#   TImage
# ----------------------------------------------------------------------


@_register_type("TImage")
class TIRLImageStruct(TIRLStruct):
    type: tx.Literal[_TIRLT.Image] = _TIRLT.Image

    resmgr: tx.Optional[dict] = None
    header: tx.Optional[tx.Any] = None
    maskmgr: tx.Optional[tx.Any] = None
    name: tx.Optional[str] = None

    def to_transform(self) -> _xforms.Sequence:
        # Use the native (highest) resolution layer only
        return self.resmgr["layers"][0]["domain"].to_transform()
