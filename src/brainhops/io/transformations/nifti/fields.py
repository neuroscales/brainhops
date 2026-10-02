"""
Fields of RAS coordinates and of RAS displacements, stored in NIfTI files.

Which intent code each one reads and writes, and why, is described in
the package docstring, [`brainhops.io.transformations.nifti`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# core
from brainhops._core import affines as _affines
from brainhops._core.properties import smartproperty
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import (
    _NIFTI_INTENT_DISPVECT,
    _NIFTI_INTENT_NAME_MAPPING,
    _NIFTI_INTENT_VECTOR,
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _nifti_intent,
    _nifti_shape,
    _NiftiObject,
)
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import RASCoordinatesField
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation

_NDIM = 3
"""The number of spatial dimensions the displacement reader supports."""


@register_format
class NiftiRASCoordinatesField(RASCoordinatesField, NiftiBasedTransformation):
    """
    Field of RAS coordinates, stored in a NIfTI file.
    """

    HINTS = ("coordinates",)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a NIfTI header as a field of RAS coordinates.

        `VECTOR` (1007) is the intent code this reader writes, and the
        one SPM writes for its `y_` coordinate maps, so it is claimed
        with certainty. `DISPVECT` (1006) is not claimed at all: the
        standard reserves it for displacements, which
        [`NiftiRASDisplacementField`][] reads. FSL intent codes are left
        to the FSL readers, which decode them.
        """
        intent = _nifti_intent(header)
        if intent == _NIFTI_INTENT_VECTOR:
            return Confidence.CERTAIN
        if intent == _NIFTI_INTENT_DISPVECT:
            return Confidence.NO
        # The intent code is often left unset, so fall back to the
        # shape: a field of RAS coordinates carries a trailing axis of
        # length 3. A plain image has no such axis, and scores zero
        # rather than competing with the affine readers.
        shape = _nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.MAYBE
        return Confidence.NO

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """The field of RAS coordinates."""
        return self.data

    @field.setter
    def field(self, value: tx.Optional[ArrayProtocol]) -> None:
        # `field` is this format's name for the image data, so the two
        # must stay one value. Without a setter the struct's generated
        # `__init__` cannot assign the inherited `field` at all.
        self.data = value

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that encodes this field of RAS coordinates.

        The field array becomes the NIfTI data array, and the header
        carries the `VECTOR` (1007) intent code, with SPM's intent name
        `"Mapping"`, that marks the file as a field of coordinates rather
        than a plain image. `DISPVECT` (1006) is not used: the standard
        reserves it for displacements, and ITK and brainhops read it as
        such. The voxel-to-RAS affine of the
        grid is taken from the source header when the field was read from
        one, and is the identity otherwise.

        The field array keeps its own array backend. A `cupy` or `dask`
        array is passed through rather than coerced into `numpy`. When
        `like` is given, non-encoding header fields are copied from it.
        Keyword arguments override header fields last, so an explicit value
        wins.
        """
        field = self.field
        if field is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        backend = get_array_backend(field)
        field = backend.asarray(field)
        if field.ndim == 4:
            # NIfTI stores a vector field as a five-dimensional array,
            # with the components in the fifth axis and a singleton axis
            # before them. A four-dimensional array would put the
            # components in the time axis, which the reader misreads.
            field = backend.expand_dims(field, axis=3)
        if self.header is not None:
            affine = self.header.get_best_affine()
        else:
            affine = np.eye(4)
        image = _new_nifti(field, affine)
        image.header.set_intent(
            _NIFTI_INTENT_VECTOR, name=_NIFTI_INTENT_NAME_MAPPING
        )
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


@register_format
class NiftiRASDisplacementField(_xforms.Sequence, NiftiBasedTransformation):
    """
    Field of RAS displacements, stored in a NIfTI file.

    This is the `DISPVECT` (1006) field of the NIfTI-1 standard: each
    voxel holds the displacement, in RAS millimetres, of the point at
    its centre, and the field maps RAS to RAS as `x -> x + u(x)`. It is
    how ITK 5.4 and later reads and writes a `DISPVECT` image.

    A `DisplacementField` adds its values in the units of its own grid,
    so the field is the
    [`Sequence`][brainhops.datamodel.transformations.Sequence] of three
    named slots:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    The displacements are interpolated linearly and extended with the
    nearest value outside the grid, as ITK's `DisplacementFieldTransform`
    does by default.
    """

    HINTS = ("displacements",)

    order: tx.ClassVar[int] = 1
    """The spline order used to interpolate the field."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition used outside of the field of view."""

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a NIfTI header as a field of RAS displacements.

        `DISPVECT` (1006) is the code the standard reserves for
        displacements, so a file that carries it is claimed with
        certainty. Nothing else is: a field without it may as well hold
        coordinates, and is left to [`NiftiRASCoordinatesField`][].
        """
        if _nifti_intent(header) == _NIFTI_INTENT_DISPVECT:
            return Confidence.CERTAIN
        return Confidence.NO

    # --- endpoints ----------------------------------------------------
    #
    # Declared rather than read off the chain: reading them off the
    # chain would build it, and building it decodes the field data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the field maps from."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the field maps to."""
        return _systems.RASmm()

    # --- decoding -----------------------------------------------------

    def _vox2ras(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-RAS affine of the field's grid."""
        header = self.header
        if header is None:
            raise ParserContentError(
                "This field has no NIfTI header to read its grid from."
            )
        return np.asarray(header.get_best_affine(), dtype=np.float64)

    def _ras_vectors(self) -> ArrayProtocol:
        """
        The stored displacements, as an `(X, Y, Z, 3)` array in RAS mm.

        The standard layout is five-dimensional, `(X, Y, Z, 1, 3)`, with
        the components in the fifth axis; its singleton axis is dropped.
        """
        data = self.data
        if data is None:
            raise ParserContentError("This field has no data to read.")
        backend = get_array_backend(data)
        data = backend.asarray(data)
        shape = tuple(int(d) for d in data.shape)
        if len(shape) == 5 and shape[3] == 1:
            data = data[:, :, :, 0, :]
        elif len(shape) != 4:
            raise ParserContentError(
                f"A NIfTI displacement field is stored as a (X, Y, Z, 1, 3) "
                f"array, not as an array of shape {shape}."
            )
        if data.shape[-1] != _NDIM:
            raise ParserContentError(
                f"Only three-dimensional displacement fields are "
                f"supported, and this one has {data.shape[-1]} components."
            )
        return data

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        It is built from the NIfTI header and data on first access, and
        cached. Assigning to it overrides the derived chain, which is how
        a field that was not read from a file is built.
        """
        vox2ras = self._vox2ras()
        compact = vox2ras[:_NDIM]
        # The stored vectors are world-space displacements, and a
        # `DisplacementField` adds its values in the units of its own
        # grid, so they are rotated into voxel units. Only the linear
        # part of the world-to-voxel affine acts on a displacement.
        vectors = self._ras_vectors()
        backend = get_array_backend(vectors)
        ras2vox = np.linalg.inv(vox2ras[:_NDIM, :_NDIM])
        rotate = backend.asarray(ras2vox, dtype=vectors.dtype)
        field = backend.matmul(rotate, vectors[..., None])[..., 0]
        voxel = _systems.VoxelCoordinateSystem()
        return (
            RASToVoxel(matrix=_affines.inv(compact)),
            _xforms.DisplacementField(
                field=field,
                input=voxel,
                output=voxel,
                order=self.order,
                bound=self.bound,
            ),
            VoxelToRAS(matrix=compact),
        )

    @property
    def ras2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def displacement(self) -> tx.Optional[_xforms.Transformation]:
        """The displacement field, in the voxel units of its grid."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from the field's voxels back to RAS world."""
        return self.transformations[2]

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that encodes this field of displacements.

        The displacements are rotated from voxel units into RAS
        millimetres and written as a `DISPVECT` (1006) image of shape
        `(X, Y, Z, 1, 3)`, whose voxel-to-RAS affine is the grid's.

        When `like` is given, non-encoding header fields are copied from
        it. Keyword arguments override header fields last.
        """
        chain = tuple(self.transformations or ())
        if len(chain) != 3 or not isinstance(
            chain[1], _xforms.DisplacementField
        ):
            raise WriterError(
                "A NIfTI displacement field is written from a chain of "
                "three transformations: RAS to voxel, a displacement "
                "field, and voxel to RAS."
            )
        displacement = chain[1]
        if displacement.field is None:
            raise WriterError(
                "This field has no displacements, so there is nothing to "
                "write."
            )
        if displacement.coeff:
            raise WriterError(
                "NIfTI stores sampled displacements, and this field holds "
                "spline coefficients. Convert it to values first."
            )
        vox2ras = _homogeneous(chain[2])
        field = displacement.field
        backend = get_array_backend(field)
        field = backend.asarray(field)
        if field.ndim != _NDIM + 1 or field.shape[-1] != _NDIM:
            raise WriterError(
                f"A NIfTI displacement field holds one 3-vector per voxel "
                f"of a 3-D grid, not an array of shape {tuple(field.shape)}."
            )
        rotate = backend.asarray(vox2ras[:_NDIM, :_NDIM], dtype=field.dtype)
        vectors = backend.matmul(rotate, field[..., None])[..., 0]
        # NIfTI stores a vector field as a five-dimensional array, with
        # the components in the fifth axis.
        vectors = backend.expand_dims(vectors, axis=3)
        image = _new_nifti(vectors, vox2ras)
        image.header.set_intent(_NIFTI_INTENT_DISPVECT)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


def _homogeneous(xform: _xforms.Transformation) -> np.ndarray:
    """The `(4, 4)` matrix of a three-dimensional affine-like slot."""
    try:
        matrix = xform.to(_xforms.Affine).homogeneous_matrix
    except Exception as error:
        raise WriterError(
            f"The grid of a NIfTI displacement field must be an affine, "
            f"not a {type(xform).__name__}."
        ) from error
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (_NDIM + 1, _NDIM + 1):
        raise WriterError(
            f"The grid of a NIfTI displacement field must be "
            f"three-dimensional, and its affine has shape "
            f"{matrix.shape[0] - 1}x{matrix.shape[1] - 1}."
        )
    return matrix
