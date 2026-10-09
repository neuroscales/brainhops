"""
NiftyReg fields and control-point grids stored in NIfTI files.

The encoding, the NiftyReg sources and the limits of the support are described
in [`brainhops.io.transformations.niftyreg`][].
"""

import nibabel as nb
import numpy as np
import typing_extensions as tx

from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
)
from brainhops.io.common.nifti._constants import (
    _NIFTI_INTENT_NAME_NIFTYREG,
    _NIFTI_INTENT_VECTOR,
)
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _nifti_intent,
    _nifti_intent_name,
    _nifti_shape,
    _nifti_vector_field,
    _NiftiObject,
)
from brainhops.io.transformations.base.affines import RASToRAS
from brainhops.io.transformations.base.conversions import no_exact_conversion
from brainhops.io.transformations.base.fields import (
    homogeneous_matrix,
    ras_displacement_chain,
    split_ras_displacement_chain,
    voxel_grid_coordinates,
)
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation

from ._formats import NiftyRegTransformationFormat

# ----------------------------------------------------------------------
#   NIFTYREG CONSTANTS
# ----------------------------------------------------------------------

# NREG_TRANS_TYPE (reg-lib/cpu/Maths.hpp), which NiftyReg stores in
# intent_p1 beside the VECTOR intent and the intent name NREG_TRANS.
DEF_FIELD = 0
"""Dense field of world positions (`reg_transform -def`)."""
DISP_FIELD = 1
"""Dense field of world displacements (`reg_transform -disp`)."""
CUB_SPLINE_GRID = 2
"""Cubic B-spline grid of control-point positions (`reg_f3d -cpp`)."""
DEF_VEL_FIELD = 3
"""Dense stationary velocity field, stored as positions."""
DISP_VEL_FIELD = 4
"""Dense stationary velocity field, stored as displacements."""
SPLINE_VEL_GRID = 5
"""Cubic B-spline grid of a stationary velocity (`reg_f3d -vel`)."""
LIN_SPLINE_GRID = 6
"""Linear B-spline grid of control-point positions."""

_NIFTI_ECODE_IGNORE = 0
"""
Extension code under which NiftyReg stores affines (`NIFTI_ECODE_IGNORE`).
"""

_NDIM = 3
"""Spatial dimension that the readers decode."""

_MAT44_BYTES = 64
"""Size of a `mat44`, made of 16 row-major `float32` values."""


def _niftyreg_type(header: _NiftiObject) -> tx.Optional[int]:
    """
    Return the NiftyReg transformation type stored in `intent_p1`.

    `None` is returned unless the header is a `VECTOR` image named
    `"NREG_TRANS"` with a whole-number `intent_p1`.
    """
    if isinstance(header, nb.Nifti1Image):
        header = header.header
    if _nifti_intent(header) != _NIFTI_INTENT_VECTOR:
        return None
    if _nifti_intent_name(header) != _NIFTI_INTENT_NAME_NIFTYREG:
        return None
    try:
        value = float(header["intent_p1"])
    except Exception:
        return None
    if not np.isfinite(value) or value != round(value):
        return None
    return int(value)


def _vox2world(header: _NiftiObject) -> np.ndarray:
    """
    Return the (4, 4) voxel-to-world affine that NiftyReg uses.

    The sform is used when `sform_code > 0`, and the qform otherwise. When both
    codes are zero, the diagonal of pixel sizes that `nifti1_io` builds is
    returned, rather than the centred fallback of `nibabel`.
    """
    if int(header["sform_code"]) > 0:
        return np.asarray(header.get_sform(), dtype=np.float64)
    if int(header["qform_code"]) > 0:
        return np.asarray(header.get_qform(), dtype=np.float64)
    pixdim = np.asarray(header["pixdim"][1:4], dtype=np.float64)
    pixdim = np.where((pixdim == 0) | ~np.isfinite(pixdim), 1.0, pixdim)
    return np.diag([*pixdim, 1.0])


def _extension_affines(header: _NiftiObject) -> tx.List[np.ndarray]:
    """
    Return the affines that NiftyReg keeps in the header extensions.

    A symmetric registration stores its half affines as raw `mat44` matrices in
    extensions of code `NIFTI_ECODE_IGNORE`. Reading stops at the first
    extension that has another code or is too short.
    """
    affines = []
    for extension in getattr(header, "extensions", None) or ():
        if int(extension.get_code()) != _NIFTI_ECODE_IGNORE:
            break
        content = extension.get_content()
        if not isinstance(content, (bytes, bytearray)):
            break
        if len(content) < _MAT44_BYTES:
            break
        dtype = np.dtype(np.float32).newbyteorder(header.endianness)
        matrix = np.frombuffer(bytes(content[:_MAT44_BYTES]), dtype=dtype)
        affines.append(matrix.astype(np.float64).reshape(4, 4))
    return affines


def _extension(matrix: np.ndarray) -> nb.nifti1.Nifti1Extension:
    """
    Build an extension that holds a `mat44`, padded as NiftyReg writes it.
    """
    content = np.asarray(matrix, dtype="<f4").reshape(16).tobytes()
    return nb.nifti1.Nifti1Extension(_NIFTI_ECODE_IGNORE, content + bytes(8))


# ----------------------------------------------------------------------
#   BASE
# ----------------------------------------------------------------------


class NiftyRegField(NiftyRegTransformationFormat, NiftiBasedTransformation):
    """
    NiftyReg transformation stored in a NIfTI file.

    NiftyReg writes every nonlinear transformation as a `VECTOR` image (1007)
    named `"NREG_TRANS"`, with its type in `intent_p1`. Each concrete reader
    claims, with certainty, the types listed in its `TYPES`. This class is
    abstract and is not registered as a format.
    """

    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset()
    """The `intent_p1` values that the reader decodes."""

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score `Confidence.CERTAIN` for a NiftyReg file of one of the `TYPES`.
        """
        if _niftyreg_type(header) in cls.TYPES:
            return Confidence.CERTAIN
        return Confidence.NO

    @property
    def niftyreg_type(self) -> tx.Optional[int]:
        """The NiftyReg transformation type of the file."""
        return None if self.header is None else _niftyreg_type(self.header)

    @property
    def extension_affines(self) -> tx.List[np.ndarray]:
        """The (4, 4) affines stored in the header extensions."""
        return [] if self.header is None else _extension_affines(self.header)

    def _vox2world(self) -> np.ndarray:
        """Return the (4, 4) voxel-to-world affine of the grid of the file."""
        if self.header is None:
            raise ParserContentError(
                "This field has no NIfTI header to read its grid from."
            )
        return _vox2world(self.header)


class NiftyRegSequence(NiftyRegField, _xforms.ImmutableSequence):
    """
    NiftyReg field, mapping the reference RAS space to the floating RAS space.

    The stored vectors are world positions or displacements in RAS millimetres.
    A
    [`DisplacementField`][brainhops.datamodel.transformations.DisplacementField]
    adds its values in the units of its own grid, so each field is read as the
    chain of [`ras_displacement_chain`][]:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacements, in voxel units           |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    Positions are read as displacements by subtracting the world coordinate of
    each voxel, as NiftyReg does. NiftyReg extends a field beyond its grid with
    the displacement of the nearest edge voxel, which the `nearest` boundary
    reproduces on displacements but not on positions. Only 3-D fields are
    decoded, and this class is not registered as a format.
    """

    degree: tx.ClassVar[int] = 1
    """Spline degree of the interpolation."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """Boundary condition outside the field of view."""

    store: tx.ClassVar[StoreEnum] = StoreEnum.values
    """Whether the field holds spline coefficients rather than values."""

    log: tx.ClassVar[bool] = False
    """
    Whether the field holds a stationary velocity rather than displacements.
    """

    @property
    def steps(self) -> tx.Optional[int]:
        """The squaring steps of a velocity, which this field does not have."""
        return None

    # --- reading ------------------------------------------------------

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """
        Build a field from a `nibabel` image or header.

        Any shape other than (X, Y, Z, 1, 3), in particular a 2-D field, is
        refused from the header alone.
        """
        header = nifti.header if isinstance(nifti, nb.Nifti1Image) else nifti
        shape = _nifti_shape(header)
        if (
            shape is None
            or len(shape) != 5
            or shape[3] != 1
            or shape[4] != _NDIM
        ):
            raise ParserContentError(
                f"A three-dimensional NiftyReg field is stored as a "
                f"(X, Y, Z, 1, 3) array, not as an array of shape {shape}. "
                f"Two-dimensional NiftyReg fields are not supported."
            )
        return super().from_nibabel(nifti, **kwargs)

    # --- copies -------------------------------------------------------

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create a field from an instance of a similar class.

        The chain of the other transformation is carried over rather than
        re-read from its header, which this reader would misread.
        """
        if isinstance(other, _xforms.Sequence) and not isinstance(other, cls):
            kwargs.setdefault("transformations", tuple(other))
        return super().from_instance(other, *args, **kwargs)

    # --- endpoints ----------------------------------------------------
    # The endpoints are declared rather than read off the chain, which would
    # decode the field data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """
        The reference world space that the field maps from, in RAS millimetres.
        """
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """
        The floating world space that the field maps to, in RAS millimetres.
        """
        return _systems.RASmm()

    # --- decoding -----------------------------------------------------

    def _stored_vectors(self) -> ArrayProtocol:
        """Return the stored vectors as an (X, Y, Z, 3) array."""
        data = self.data
        if data is None:
            raise ParserContentError("This field has no data to read.")
        backend = get_array_backend(data)
        data = backend.asarray(data)
        shape = tuple(int(d) for d in data.shape)
        data = _nifti_vector_field(data)
        if data.ndim != _NDIM + 1 or int(data.shape[-1]) != _NDIM:
            raise ParserContentError(
                f"A three-dimensional NiftyReg field is stored as a "
                f"(X, Y, Z, 1, 3) array, not as an array of shape {shape}."
            )
        return data

    def _displacements(self, vox2world: np.ndarray) -> ArrayProtocol:
        """Return the stored vectors as RAS displacements."""
        return self._stored_vectors()

    def _field_chain(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """Return the three slots of the field itself."""
        vox2world = self._vox2world()
        return ras_displacement_chain(
            self._displacements(vox2world),
            vox2world,
            degree=self.degree,
            bound=self.bound,
            store=self.store,
            log=self.log,
            steps=self.steps,
        )

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        The chain is built lazily from the header and the data, and then
        cached. Assigning a chain overrides the derived one, which is how
        fields that do not come from a file are built.
        """
        return self._field_chain()

    # --- slots --------------------------------------------------------

    def _slot(self, index: int) -> tx.Optional[_xforms.Transformation]:
        chain = self.transformations
        # A chain that starts with an affine has its field slots shifted by
        # one.
        return chain[index + len(chain) - 3]

    @property
    def ras2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from RAS world coordinates to the voxels of the field."""
        return self._slot(0)

    @property
    def displacement(self) -> tx.Optional[_xforms.Transformation]:
        """The displacement field, in the voxel units of its grid."""
        return self._slot(1)

    @property
    def voxel2ras(self) -> tx.Optional[_xforms.Transformation]:
        """
        The affine from the voxels of the field back to RAS world coordinates.
        """
        return self._slot(2)

    # --- writing ------------------------------------------------------

    _WHAT: tx.ClassVar[str] = "A NiftyReg field"

    def _split(
        self, chain: tx.Sequence[_xforms.Transformation]
    ) -> tx.Tuple[np.ndarray, ArrayProtocol]:
        """
        Return the grid affine and the RAS displacements of a three-slot chain.
        """
        return split_ras_displacement_chain(
            chain,
            self._WHAT,
            ndim=_NDIM,
            store=self.store,
            degree=self.degree,
            bound=self.bound,
            log=self.log,
        )

    def _write(
        self,
        vectors: ArrayProtocol,
        vox2world: np.ndarray,
        kind: int,
        like: tx.Any = None,
        extensions: tx.Sequence[np.ndarray] = (),
        **overrides,
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build a NiftyReg NIfTI image from an (X, Y, Z, 3) array.

        The image is a `VECTOR` image of shape (X, Y, Z, 1, 3) named
        `"NREG_TRANS"`, with its type in `intent_p1` and the given affines in
        its extensions.
        """
        backend = get_array_backend(vectors)
        vectors = backend.expand_dims(backend.asarray(vectors), axis=3)
        image = _new_nifti(vectors, vox2world)
        header = image.header
        header.set_intent(
            _NIFTI_INTENT_VECTOR, name=_NIFTI_INTENT_NAME_NIFTYREG
        )
        header["intent_p1"] = kind
        for matrix in extensions:
            header.extensions.append(_extension(matrix))
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


# ----------------------------------------------------------------------
#   DENSE FIELDS
# ----------------------------------------------------------------------


@register_format
class NiftyRegDisplacementField(NiftyRegSequence):
    """
    Displacement field written by `reg_transform -disp` (`intent_p1` 1).

    Each voxel of the reference grid holds the displacement, in RAS
    millimetres, from its own world position to the floating position, so the
    field maps `x` to `x + u(x)`. See [`NiftyRegSequence`][].
    """

    HINTS = ("displacement", "disp")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset({DISP_FIELD})
    _WHAT: tx.ClassVar[str] = "A NiftyReg displacement field"

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """Build the NIfTI image that NiftyReg would write for this field."""
        vox2world, vectors = self._split(self.transformations)
        return self._write(vectors, vox2world, DISP_FIELD, like, **overrides)


@register_format
class NiftyRegDeformationField(NiftyRegSequence):
    """
    Deformation field written by `reg_transform -def` (`intent_p1` 0).

    Each voxel of the reference grid holds the floating world position, in RAS
    millimetres. The positions are read as displacements by subtracting the
    world position of each voxel, so the field has the same chain as
    [`NiftyRegDisplacementField`][].
    """

    HINTS = ("deformation", "def")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset({DEF_FIELD})
    _WHAT: tx.ClassVar[str] = "A NiftyReg deformation field"

    def _displacements(self, vox2world: np.ndarray) -> ArrayProtocol:
        positions = self._stored_vectors()
        backend = get_array_backend(positions)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in positions.shape[:_NDIM]), vox2world, backend
        )
        return positions - backend.asarray(grid, dtype=positions.dtype)

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the NIfTI image of this field, with displacements as positions.
        """
        vox2world, vectors = self._split(self.transformations)
        backend = get_array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, backend
        )
        positions = vectors + backend.asarray(grid, dtype=vectors.dtype)
        return self._write(positions, vox2world, DEF_FIELD, like, **overrides)


# ----------------------------------------------------------------------
#   CONTROL-POINT GRIDS
# ----------------------------------------------------------------------


_GRID_DEGREE = {CUB_SPLINE_GRID: 3, LIN_SPLINE_GRID: 1}
"""B-spline degree of each type of grid."""


@register_format
class NiftyRegControlPointGrid(NiftyRegSequence):
    """
    Control-point grid written by `reg_f3d -cpp`.

    The grid is cubic (`intent_p1` 2) or linear (6). Each control point holds a
    floating world position in RAS millimetres. The deformation at a reference
    point is the centred cubic B-spline of these positions, evaluated where the
    point falls on the grid. The grid header places the grid on its own, so no
    reference image is needed.

    The positions are read as the spline coefficients of displacements by
    subtracting the world position of each control point. A cubic B-spline
    reproduces linear functions, so the map is unchanged wherever all the
    acting control points lie in the grid, which covers the whole reference
    image. Beyond the grid, the `nearest` boundary on the coefficients
    reproduces how NiftyReg slides the nearest displacement. A linear grid is
    read the same way at degree 1.

    When the header carries an affine in its extensions, as after a symmetric
    registration, NiftyReg applies that affine before the spline. The chain
    then starts with a [`RASToRAS`][] in the `affine` slot and has four slots.
    """

    HINTS = ("cpp", "f3d")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset(
        {CUB_SPLINE_GRID, LIN_SPLINE_GRID}
    )
    _WHAT: tx.ClassVar[str] = "A NiftyReg control-point grid"

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest

    @property
    def degree(self) -> int:
        """The B-spline degree of the grid: 3 if cubic, 1 if linear."""
        if self.header is not None:
            kind = _niftyreg_type(self.header)
            if kind in _GRID_DEGREE:
                return _GRID_DEGREE[kind]
        chain = getattr(self, "_transformations", None)
        if chain:
            return int(chain[-2].degree)
        return 3

    @property
    def store(self) -> StoreEnum:
        """Whether the grid holds spline coefficients, as above degree 1."""
        return StoreEnum.from_coefficients(self.degree > 1)

    def _displacements(self, vox2world: np.ndarray) -> ArrayProtocol:
        positions = self._stored_vectors()
        backend = get_array_backend(positions)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in positions.shape[:_NDIM]), vox2world, backend
        )
        return positions - backend.asarray(grid, dtype=positions.dtype)

    def _field_chain(self) -> tx.Tuple[_xforms.Transformation, ...]:
        chain = super()._field_chain()
        affines = self.extension_affines
        if affines:
            chain = (RASToRAS(matrix=affines[0][:-1]), *chain)
        return chain

    @property
    def affine(self) -> tx.Optional[_xforms.Transformation]:
        """The affine applied before the spline, or `None`."""
        chain = self.transformations
        return chain[0] if len(chain) == 4 else None

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the NIfTI image of this grid, with coefficients as positions.

        A leading affine is written to the header extension that NiftyReg
        reads.
        """
        chain = tuple(self.transformations or ())
        extensions = []
        if len(chain) == 4:
            affine = homogeneous_matrix(chain[0], self._WHAT, _NDIM)
            extensions.append(affine)
            chain = chain[1:]
        # NiftyReg stores cubic and linear grids, so a field of either degree
        # is
        # written at its own degree, and any other is refitted to cubic.
        degree = int(getattr(chain[1], "degree", 3)) if len(chain) == 3 else 3
        kinds = {degree: kind for kind, degree in _GRID_DEGREE.items()}
        if degree not in kinds:
            degree = 3
        vox2world, vectors = split_ras_displacement_chain(
            chain,
            self._WHAT,
            ndim=_NDIM,
            store=StoreEnum.from_coefficients(degree > 1),
            degree=degree,
            bound=self.bound,
        )
        backend = get_array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, backend
        )
        positions = vectors + backend.asarray(grid, dtype=vectors.dtype)
        return self._write(
            positions, vox2world, kinds[degree], like, extensions, **overrides
        )


# ----------------------------------------------------------------------
#   VELOCITY FIELDS AND GRIDS
# ----------------------------------------------------------------------


class NiftyRegVelocity(NiftyRegSequence):
    """
    Stationary velocity field or grid.

    `reg_f3d -vel` parametrises the deformation by a stationary velocity, whose
    exponential is computed by scaling and squaring with `|intent_p2|` steps. A
    negative `intent_p2` marks a backward field, whose velocity is negated. The
    file is read as the chain of [`NiftyRegSequence`][], with a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.StationaryVelocityField]
    in the `displacement` slot. A velocity grid is squared on its own grid,
    whereas NiftyReg squares on the dense reference grid, so the flow matches
    `reg_transform -def` closely but not exactly.

    The affine that a symmetric registration keeps in the extensions of its
    velocity is not decoded, so building the chain of such a velocity raises
    `NotImplementedError`. A velocity read from a file is written back as read.
    A velocity built from a chain is written with its steps in `intent_p2`, and
    a displacement field is refused, since no logarithm is computed. This class
    is not registered as a format.
    """

    log: tx.ClassVar[bool] = True

    @property
    def squaring_steps(self) -> tx.Optional[int]:
        """The stored squaring steps, negative for a backward field."""
        if self.header is None:
            return None
        return int(round(float(self.header["intent_p2"])))

    @property
    def steps(self) -> tx.Optional[int]:
        """
        The squaring steps of the velocity, or `None` for the default rule.
        """
        steps = self.squaring_steps
        if not steps:
            return None
        return abs(steps)

    def _velocity_vectors(self, vox2world: np.ndarray) -> ArrayProtocol:
        """Return the stored vectors as the RAS velocity that they encode."""
        raise NotImplementedError

    def _displacements(self, vox2world: np.ndarray) -> ArrayProtocol:
        if self.extension_affines:
            raise NotImplementedError(
                f"{type(self).__name__}: this NiftyReg velocity carries an "
                f"affine in its header extensions, which NiftyReg removes "
                f"before integrating it and composes back after; that is not "
                f"decoded. Integrate it with NiftyReg (`reg_transform -ref "
                f"<ref> -def <velocity> <out>.nii.gz`) and read the "
                f"deformation field it writes."
            )
        velocity = self._velocity_vectors(vox2world)
        if (self.squaring_steps or 0) < 0:
            # A backward field integrates the negated velocity.
            velocity = -velocity
        return velocity

    def _read_back(
        self, like: tx.Any = None, **overrides
    ) -> tx.Optional[tx.Union[nb.Nifti1Image, nb.Nifti2Image]]:
        """
        Return the NIfTI image as it was read, or `None`.

        `None` is returned when the velocity was not read from a NiftyReg file
        of its own type, or when its chain was assigned.
        """
        if self.niftyreg_type not in type(self).TYPES:
            return None
        if getattr(self, "_transformations", None) is not None:
            return None
        if self.header is None or self.data is None:
            return None
        header = self.header.copy()
        image_cls = (
            nb.Nifti2Image
            if isinstance(header, nb.Nifti2Header)
            else nb.Nifti1Image
        )
        image = image_cls(self.data, None, header=header)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


def _written_steps(chain: tx.Sequence[_xforms.Transformation]) -> int:
    """
    Return the squaring steps, those of the velocity or of the default rule.
    """
    return int(chain[1]._compute_steps)


@register_format
class NiftyRegVelocityGrid(NiftyRegVelocity):
    """
    Cubic B-spline grid of a stationary velocity (`reg_f3d -vel -cpp`).

    As in [`NiftyRegControlPointGrid`][], the control points hold positions,
    which are read as velocity coefficients. See [`NiftyRegVelocity`][] for the
    integration.
    """

    HINTS = ("velocity", "vel", "cpp")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset({SPLINE_VEL_GRID})
    _WHAT: tx.ClassVar[str] = "A NiftyReg velocity grid"

    degree: tx.ClassVar[int] = 3
    store: tx.ClassVar[StoreEnum] = StoreEnum.coefficients

    def _velocity_vectors(self, vox2world: np.ndarray) -> ArrayProtocol:
        positions = self._stored_vectors()
        backend = get_array_backend(positions)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in positions.shape[:_NDIM]), vox2world, backend
        )
        return positions - backend.asarray(grid, dtype=positions.dtype)

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the NIfTI image of this velocity grid.

        A grid read from a file is written back as read. Otherwise, the
        coefficients are turned into positions, and the squaring steps are
        stored in `intent_p2`.
        """
        image = self._read_back(like, **overrides)
        if image is not None:
            return image
        chain = tuple(self.transformations or ())
        vox2world, vectors = self._split(chain)
        backend = get_array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, backend
        )
        positions = vectors + backend.asarray(grid, dtype=vectors.dtype)
        overrides.setdefault("intent_p2", _written_steps(chain))
        return self._write(
            positions, vox2world, SPLINE_VEL_GRID, like, **overrides
        )


@register_format
class NiftyRegVelocityField(NiftyRegVelocity):
    """
    Dense stationary velocity field, stored as positions or displacements.

    Positions are read as a velocity by subtracting the world position of each
    voxel, as NiftyReg does before integrating. See [`NiftyRegVelocity`][].
    """

    HINTS = ("velocity", "vel")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset(
        {DEF_VEL_FIELD, DISP_VEL_FIELD}
    )
    _WHAT: tx.ClassVar[str] = "A NiftyReg velocity field"

    def _velocity_vectors(self, vox2world: np.ndarray) -> ArrayProtocol:
        vectors = self._stored_vectors()
        if self.niftyreg_type == DISP_VEL_FIELD:
            return vectors
        backend = get_array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, backend
        )
        return vectors - backend.asarray(grid, dtype=vectors.dtype)

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the NIfTI image of this velocity field.

        A field read from a file is written back as read. Otherwise, it is
        written as positions, the type that NiftyReg integrates, with the
        squaring steps in `intent_p2`.
        """
        image = self._read_back(like, **overrides)
        if image is not None:
            return image
        chain = tuple(self.transformations or ())
        vox2world, vectors = self._split(chain)
        backend = get_array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, backend
        )
        positions = vectors + backend.asarray(grid, dtype=vectors.dtype)
        overrides.setdefault("intent_p2", _written_steps(chain))
        return self._write(
            positions, vox2world, DEF_VEL_FIELD, like, **overrides
        )


# ----------------------------------------------------------------------
#   CONVERSIONS
# ----------------------------------------------------------------------


@converter
def _(
    t: _xforms.Sequence, cls: tx.Type[NiftyRegSequence], **kwargs
) -> NiftyRegSequence:
    # NiftyReg stores a field on the grid of its file, in one of several
    # encodings, and nothing converts to it yet (#312). Without this
    # refusal, a chain would be relabelled as the format.
    raise no_exact_conversion(t, cls)


@converter
def _(
    t: NiftyRegSequence, cls: tx.Type[NiftyRegSequence], **kwargs
) -> NiftyRegSequence:
    # Within its own format, a chain is changed by the rules of any chain,
    # which keep its type. Another variant of the format is not converted
    # to yet.
    if not isinstance(t, cls):
        raise no_exact_conversion(t, cls)
    return smart_replace(t, cls, **kwargs)
