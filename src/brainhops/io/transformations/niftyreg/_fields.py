"""
NiftyReg fields and control-point grids, stored in NIfTI files.

The encoding, the NiftyReg sources it is taken from, and what is and is
not supported, are described in the package docstring,
[`brainhops.io.transformations.niftyreg`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import (
    _NIFTI_INTENT_NAME_NIFTYREG,
    _NIFTI_INTENT_VECTOR,
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _nifti_intent,
    _nifti_intent_name,
    _nifti_shape,
    _nifti_vector_field,
    _NiftiObject,
)
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.transformations.base.affines import RASToRAS
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

# `NREG_TRANS_TYPE`, `reg-lib/cpu/Maths.hpp`. NiftyReg stores it in
# `intent_p1` of every transformation it writes, next to the `VECTOR`
# intent code and the intent name `"NREG_TRANS"`.
DEF_FIELD = 0
"""A dense field of world positions (`reg_transform -def`)."""
DISP_FIELD = 1
"""A dense field of world displacements (`reg_transform -disp`)."""
CUB_SPLINE_GRID = 2
"""A cubic B-spline grid of control-point positions (`reg_f3d -cpp`)."""
DEF_VEL_FIELD = 3
"""A dense stationary velocity field, stored as positions."""
DISP_VEL_FIELD = 4
"""A dense stationary velocity field, stored as displacements."""
SPLINE_VEL_GRID = 5
"""A cubic B-spline grid of a stationary velocity (`reg_f3d -vel`)."""
LIN_SPLINE_GRID = 6
"""A linear B-spline grid of control-point positions."""

_NIFTI_ECODE_IGNORE = 0
"""The extension code NiftyReg stores its affine extensions with."""

_NDIM = 3
"""The number of spatial dimensions the readers decode."""

_MAT44_BYTES = 64
"""The size of a `mat44`: sixteen single-precision floats, row-major."""


def _niftyreg_type(header: _NiftiObject) -> tx.Optional[int]:
    """
    The NiftyReg transformation type of a header, from its `intent_p1`.

    `None` unless the header is a NiftyReg transformation: a `VECTOR`
    image named `"NREG_TRANS"` whose `intent_p1` is a whole number.
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
    The `(4, 4)` voxel-to-world affine NiftyReg uses for a header.

    NiftyReg takes the sform when `sform_code > 0`, and the qform
    otherwise (`sto_xyz` / `qto_xyz` throughout `reg-lib`). When the
    qform code is zero too, `nifti1_io` fills the qform with the pixel
    sizes on the diagonal and no offset -- unlike `nibabel`, whose
    fallback affine centres the grid and flips x -- so that is what is
    used here.
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
    The affines NiftyReg keeps in a header's extensions.

    `reg_createSymmetricControlPointGrids` stores the half affines of a
    symmetric registration as raw `mat44` (sixteen row-major floats,
    in the byte order of the machine that wrote them) in extensions of
    code `NIFTI_ECODE_IGNORE`, and `reg_spline_getDeformationField` and
    `reg_defField_getDeformationFieldFromFlowField` read them back from
    the first (and second) extension. Extensions of any other code, or
    too short to hold a matrix, are not NiftyReg's and are left out.
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
    """A `NIFTI_ECODE_IGNORE` extension holding a `mat44`, as NiftyReg
    writes it (`esize = 16 * sizeof(float) + 16`, zero padded)."""
    content = np.asarray(matrix, dtype="<f4").reshape(16).tobytes()
    return nb.nifti1.Nifti1Extension(_NIFTI_ECODE_IGNORE, content + bytes(8))


# ----------------------------------------------------------------------
#   BASE
# ----------------------------------------------------------------------


class NiftyRegField(NiftyRegTransformationFormat, NiftiBasedTransformation):
    """
    A NiftyReg transformation stored in a NIfTI file.

    NiftyReg writes every non-linear transformation as a `VECTOR`
    (1007) image named `"NREG_TRANS"`, and says which kind it is in
    `intent_p1` (`NREG_TRANS_TYPE`). Each concrete reader claims the
    kinds listed in its `TYPES`, with certainty.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch.
    """

    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset()
    """The `intent_p1` values this reader decodes."""

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """`CERTAIN` for a NiftyReg file of one of `TYPES`, else `NO`."""
        if _niftyreg_type(header) in cls.TYPES:
            return Confidence.CERTAIN
        return Confidence.NO

    @property
    def niftyreg_type(self) -> tx.Optional[int]:
        """The NiftyReg transformation type (`intent_p1`) of the file."""
        return None if self.header is None else _niftyreg_type(self.header)

    @property
    def extension_affines(self) -> tx.List[np.ndarray]:
        """The `(4, 4)` affines NiftyReg stored in the header extensions."""
        return [] if self.header is None else _extension_affines(self.header)

    def _vox2world(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-world affine of the file's grid."""
        if self.header is None:
            raise ParserContentError(
                "This field has no NIfTI header to read its grid from."
            )
        return _vox2world(self.header)


class NiftyRegSequence(NiftyRegField, _xforms.ImmutableSequence):
    """
    A NiftyReg field that the data model represents: a chain of
    transformations from reference RAS to floating RAS.

    The stored vectors are world positions or displacements, in NIfTI
    world (RAS) millimetres, sampled on the file's own grid. A
    [`DisplacementField`][brainhops.datamodel.transformations.DisplacementField]
    adds its values in the units of its own grid, so each field is read
    as the chain of [`ras_displacement_chain`][]:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacements, in voxel units           |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    Positions are read as displacements, by subtracting the world
    coordinate of their voxel -- exactly what NiftyReg does
    (`reg_getDisplacementFromDeformation`) -- because NiftyReg extends a
    field beyond its grid by sliding: it keeps the *displacement* of the
    nearest edge voxel (`get_SlidedValues`), which a displacement field
    with the `nearest` boundary condition reproduces, and a field of
    positions would not.

    Only three-dimensional fields are decoded: a 2-D NiftyReg field
    (two components) is refused when read.

    Abstract: it is not decorated with `@register_format`.
    """

    degree: tx.ClassVar[int] = 1
    """The spline degree used to interpolate the field."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition used outside of the field of view."""

    coeff: tx.ClassVar[bool] = False
    """Whether the field holds spline coefficients rather than values."""

    # --- reading ------------------------------------------------------

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """
        Build the field from a `nibabel` header or image.

        A NiftyReg field is `(X, Y, Z, 1, 3)`, with its components in the
        fifth axis (`reg_createDeformationField`,
        `reg_createControlPointGrid`). Anything else -- a 2-D field
        `(X, Y, 1, 1, 2)` in particular -- is refused here, from the
        header alone.
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

    # --- endpoints ----------------------------------------------------
    #
    # Declared rather than read off the chain: reading them off the
    # chain would build it, and building it decodes the field data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The reference world space the field maps from."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The floating world space the field maps to."""
        return _systems.RASmm()

    # --- decoding -----------------------------------------------------

    def _stored_vectors(self) -> ArrayProtocol:
        """The stored vectors, as an `(X, Y, Z, 3)` array."""
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
        """The stored vectors, as RAS displacements."""
        return self._stored_vectors()

    def _field_chain(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The three slots of the field itself."""
        vox2world = self._vox2world()
        return ras_displacement_chain(
            self._displacements(vox2world),
            vox2world,
            degree=self.degree,
            bound=self.bound,
            coeff=self.coeff,
        )

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        It is built from the NIfTI header and data on first access, and
        cached. Assigning to it overrides the derived chain, which is how
        a field that was not read from a file is built.
        """
        return self._field_chain()

    # --- slots --------------------------------------------------------

    def _slot(self, index: int) -> tx.Optional[_xforms.Transformation]:
        chain = self.transformations
        # A chain that starts with an affine (see the control-point grid)
        # has its field slots one further.
        return chain[index + len(chain) - 3]

    @property
    def ras2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from RAS world coordinates to the field's voxels."""
        return self._slot(0)

    @property
    def displacement(self) -> tx.Optional[_xforms.Transformation]:
        """The displacement field, in the voxel units of its grid."""
        return self._slot(1)

    @property
    def voxel2ras(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from the field's voxels back to RAS world."""
        return self._slot(2)

    # --- writing ------------------------------------------------------

    _WHAT: tx.ClassVar[str] = "A NiftyReg field"

    def _split(
        self, chain: tx.Sequence[_xforms.Transformation]
    ) -> tx.Tuple[np.ndarray, ArrayProtocol]:
        """The grid and the RAS displacements of a three-slot chain."""
        return split_ras_displacement_chain(
            chain, self._WHAT, ndim=_NDIM, coeff=self.coeff
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
        Build the NiftyReg NIfTI image of an `(X, Y, Z, 3)` array.

        The vectors are written as NiftyReg writes them: a `VECTOR`
        (1007) image of shape `(X, Y, Z, 1, 3)`, named `"NREG_TRANS"`,
        with the transformation type in `intent_p1`. The grid is stored
        in the sform (with a non-zero code, so NiftyReg reads it) and the
        qform.
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
    A NiftyReg displacement field (`DISP_FIELD`, `intent_p1 = 1`).

    `reg_transform -disp` writes one: on the reference grid, each voxel
    holds the displacement, in world (RAS) millimetres, from its own
    world position to the floating position it maps to. The sign is
    `displacement = deformation - position`
    (`reg_getDisplacementFromDeformation`), so the field maps reference
    RAS to floating RAS as `x -> x + u(x)`.

    It is interpolated linearly, and extended beyond its grid with the
    displacement of the nearest edge voxel, as NiftyReg composes it
    (`reg_defField_compose`). See [`NiftyRegSequence`][] for the chain.
    """

    HINTS = ("displacement", "disp")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset({DISP_FIELD})
    _WHAT: tx.ClassVar[str] = "A NiftyReg displacement field"

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the NIfTI image NiftyReg would write for this field.

        When `like` is given, non-encoding header fields are copied from
        it. Keyword arguments override header fields last.
        """
        vox2world, vectors = self._split(self.transformations)
        return self._write(vectors, vox2world, DISP_FIELD, like, **overrides)


@register_format
class NiftyRegDeformationField(NiftyRegSequence):
    """
    A NiftyReg deformation field (`DEF_FIELD`, `intent_p1 = 0`).

    `reg_transform -def` (and `reg_resample -def`) write one: on the
    reference grid, each voxel holds the floating world (RAS) position,
    in millimetres, that it maps to (`reg_createDeformationField`,
    `reg_spline_getDeformationField`).

    The positions are read as displacements, by subtracting the world
    position of their voxel, and written back by adding it: the field is
    the same chain as a [`NiftyRegDisplacementField`][], interpolated
    linearly and extended with the displacement of the nearest edge
    voxel, which is how NiftyReg composes a deformation field
    (`reg_defField_compose`, `get_SlidedValues`).
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
        Build the NIfTI image NiftyReg would write for this field: the
        displacements are turned back into positions.

        When `like` is given, non-encoding header fields are copied from
        it. Keyword arguments override header fields last.
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
"""The B-spline degree of each kind of control-point grid."""


@register_format
class NiftyRegControlPointGrid(NiftyRegSequence):
    """
    A NiftyReg control-point grid (`CUB_SPLINE_GRID`, `intent_p1 = 2`),
    as written by `reg_f3d -cpp`; also a linear one (`LIN_SPLINE_GRID`,
    `intent_p1 = 6`).

    The grid holds, at each control point, the floating world (RAS)
    *position* of that control point, in millimetres -- not a
    displacement: `reg_createControlPointGrid` initialises it with the
    identity positions and `reg_f3d` optimises them. The deformation at
    a reference world point `x` is the cubic B-spline of those
    positions, evaluated at `x` mapped into the grid's voxels by the
    grid's own header (`reg_cubic_spline_getDeformationField3D`):
    the basis is the centred cubic B-spline in grid units
    (`get_BSplineBasisValues`), so the control points that act on `x`
    are `floor(g) - 1` to `floor(g) + 2`, where `g` is its grid
    coordinate.

    The grid's header places it: `reg_createControlPointGrid` copies the
    reference orientation, scales it to the control-point spacing (the
    pixdims, in mm) and moves its origin one control point before the
    reference origin. So no reference image is needed to read it.

    The positions are read as spline coefficients of displacements, by
    subtracting the world position of each control point. A cubic
    B-spline reproduces linear functions, so the two are the same map
    wherever every control point that acts is inside the grid, which is
    everywhere on the reference image. Beyond the grid, NiftyReg slides
    the displacement of the nearest control point (`get_GridValues`),
    which the `nearest` boundary condition on the coefficients
    reproduces. A linear grid is a field of linearly interpolated
    positions on the control points, read the same way at degree 1.

    The chain is that of [`NiftyRegSequence`][], with degree 3 and
    coefficients. When the header carries an affine in its extensions
    (as the grids of a symmetric registration do), NiftyReg applies it
    to the reference position before the spline
    (`reg_spline_getDeformationField`), so the chain starts with that
    affine, as a [`RASToRAS`][] (`affine` slot), and has four slots.
    """

    HINTS = ("cpp", "f3d")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset(
        {CUB_SPLINE_GRID, LIN_SPLINE_GRID}
    )
    _WHAT: tx.ClassVar[str] = "A NiftyReg control-point grid"

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest

    @property
    def degree(self) -> int:
        """The B-spline degree of the grid: 3, or 1 for a linear grid."""
        if self.header is not None:
            kind = _niftyreg_type(self.header)
            if kind in _GRID_DEGREE:
                return _GRID_DEGREE[kind]
        chain = getattr(self, "_transformations", None)
        if chain:
            return int(chain[-2].degree)
        return 3

    @property
    def coeff(self) -> bool:
        """Whether the grid holds spline coefficients: so it does, at any
        degree above one (at degree one, coefficients are values)."""
        return self.degree > 1

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
        Build the NIfTI image NiftyReg would write for this grid: the
        coefficients are turned back into control-point positions.

        A chain of four slots, whose first is an affine, has that affine
        written to the header extension NiftyReg reads it from. When
        `like` is given, non-encoding header fields are copied from it.
        Keyword arguments override header fields last.
        """
        chain = tuple(self.transformations or ())
        extensions = []
        if len(chain) == 4:
            affine = homogeneous_matrix(chain[0], self._WHAT, _NDIM)
            extensions.append(affine)
            chain = chain[1:]
        degree = int(getattr(chain[1], "degree", 3)) if len(chain) == 3 else 3
        kinds = {degree: kind for kind, degree in _GRID_DEGREE.items()}
        if degree not in kinds:
            raise WriterError(
                f"NiftyReg stores cubic (3) and linear (1) control-point "
                f"grids, not grids of degree {degree}."
            )
        vox2world, vectors = split_ras_displacement_chain(
            chain, self._WHAT, ndim=_NDIM, coeff=degree > 1
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


class NiftyRegVelocity(NiftyRegField):
    """
    A NiftyReg stationary velocity field or grid, read but not decoded.

    `reg_f3d -vel` parametrises the deformation by a stationary
    velocity field, and its output is the deformation's *exponential*:
    the velocity is scaled down by `2 ** n` (`n = |intent_p2|`, negative
    for a backward field) and composed with itself `n` times
    (`reg_defField_getDeformationFieldFromFlowField`). The data model has
    no transformation that exponentiates a field, so the file is read --
    its header, its data, its squaring steps and its affines are all
    available -- but it cannot be used as a transformation: computing,
    converting or inverting it raises `NotImplementedError`.

    To use one, have NiftyReg integrate it into a deformation field
    (`reg_transform -ref <ref> -def <velocity> <out>`) and read that.

    It is written back as it was read.

    Abstract: it is not decorated with `@register_format`.
    """

    @property
    def squaring_steps(self) -> tx.Optional[int]:
        """The number of squaring steps of the exponentiation
        (`intent_p2`; negative for a backward field)."""
        if self.header is None:
            return None
        return int(round(float(self.header["intent_p2"])))

    def _unsupported(self, *args: tx.Any, **kwargs: tx.Any) -> tx.NoReturn:
        raise NotImplementedError(
            f"{type(self).__name__}: a NiftyReg stationary velocity "
            f"{'grid' if self.niftyreg_type == SPLINE_VEL_GRID else 'field'} "
            f"must be exponentiated (scaling and squaring) to give a "
            f"transformation, which the brainhops data model cannot "
            f"represent yet. Integrate it with NiftyReg "
            f"(`reg_transform -ref <ref> -def <velocity> <out>.nii.gz`) and "
            f"read the deformation field it writes."
        )

    compute = _unsupported
    simplify = _unsupported
    inverse = _unsupported
    to = _unsupported
    __call__ = _unsupported

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the NIfTI image of the velocity, as it was read.

        The data and the header, extensions included, are written back
        unchanged. When `like` is given, non-encoding header fields are
        copied from it. Keyword arguments override header fields last.
        """
        if self.header is None or self.data is None:
            raise WriterError(
                "This velocity has no header and data, so there is nothing "
                "to write."
            )
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


@register_format
class NiftyRegVelocityGrid(NiftyRegVelocity):
    """
    A cubic B-spline grid of a stationary velocity (`SPLINE_VEL_GRID`,
    `intent_p1 = 5`), as written by `reg_f3d -vel -cpp`. See
    [`NiftyRegVelocity`][]: it is read, but not decoded.
    """

    HINTS = ("velocity", "vel", "cpp")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset({SPLINE_VEL_GRID})


@register_format
class NiftyRegVelocityField(NiftyRegVelocity):
    """
    A dense stationary velocity field, stored as positions
    (`DEF_VEL_FIELD`, `intent_p1 = 3`) or as displacements
    (`DISP_VEL_FIELD`, `intent_p1 = 4`). See [`NiftyRegVelocity`][]: it
    is read, but not decoded.
    """

    HINTS = ("velocity", "vel")
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset(
        {DEF_VEL_FIELD, DISP_VEL_FIELD}
    )
