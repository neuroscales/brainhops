"""ITK displacement and coordinate fields, stored as LPS NIfTI vector images.

The encoding, the sources it is taken from, and how these readers are
told apart from the RAS NIfTI fields are described in the package
docstring, [`brainhops.io.transformations.itk.nifti`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

# core
from brainhops._core import affines as _affines
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
    _NIFTI_FSL_INTENTS,
    _NIFTI_INTENT_DISPVECT,
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
from brainhops.io.transformations.base.affines import LPSToVoxel, VoxelToLPS
from brainhops.io.transformations.base.fields import LPSCoordinatesField
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation

_NDIM = 3
"""The number of spatial dimensions these readers support."""

_RAS_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""
The homogeneous matrix that maps RAS to LPS, and LPS to RAS.

It negates the first two axes and is its own inverse. ITK applies it to
the direction and origin of an image when it writes the NIfTI sform and
qform, and to the vector components of a three-component `DISPVECT`
image (see `itkNiftiImageIO.cxx`, `SetNIfTIOrientationFromImageIO` and
`ConvertRASToFromLPS_XYZTC`).
"""

_RAS_LPS_VECTOR = (-1.0, -1.0, 1.0)
"""The same flip, applied to the components of a vector."""

_NIFTI_XFORM_SCANNER_ANAT = 1
"""
The sform and qform code ITK writes.

`SetNIfTIOrientationFromImageIO` sets both codes to
`NIFTI_XFORM_SCANNER_ANAT`, so a field that was not read from a file is
written the same way.
"""


def _is_itk_layout(shape: tx.Optional[tx.Tuple[int, ...]]) -> bool:
    """
    Whether a NIfTI data shape is the layout ITK writes a 3-D field in.

    ITK writes a vector image with `dim[0] = 5`, a singleton time axis
    when the image has fewer than four dimensions, and the components in
    the fifth axis: `(X, Y, Z, 1, 3)` for a three-dimensional field.
    """
    return (
        shape is not None
        and len(shape) == 5
        and shape[3] == 1
        and shape[4] == _NDIM
    )


class ITKNiftiField(_xforms.Sequence, NiftiBasedTransformation):
    """
    A field stored in an ITK NIfTI vector image, from LPS to LPS.

    The vectors of the image are in ITK's LPS physical space, while the
    NIfTI header carries the usual voxel-to-RAS affine. This base reads
    both, and leaves to its subclasses what the vectors mean:
    displacements ([`ITKNiftiDisplacementField`][]) or absolute
    coordinates ([`ITKNiftiCoordinatesField`][]).

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch.
    """

    HINTS = ("itk",)

    # --- endpoints ----------------------------------------------------
    #
    # Declared rather than read off the chain, as the ITK blocks do:
    # reading them off the chain would build the chain, and building it
    # decodes the field data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the field maps from."""
        return _systems.LPSmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the field maps to."""
        return _systems.LPSmm()

    # --- decoding -----------------------------------------------------

    def _vox2lps(self) -> np.ndarray:
        """
        The `(4, 4)` voxel-to-LPS affine of the field's grid.

        The header stores voxel-to-RAS, as every NIfTI does, so the first
        two rows are negated. `get_best_affine` prefers the sform when
        its code is set; ITK writes the same matrix to both forms.
        """
        header = self.header
        if header is None:
            raise ParserContentError(
                "This field has no NIfTI header to read its grid from."
            )
        vox2ras = np.asarray(header.get_best_affine(), dtype=np.float64)
        return _RAS_LPS @ vox2ras

    def _lps_vectors(self) -> ArrayProtocol:
        """
        The stored vectors, as an `(X, Y, Z, 3)` array in LPS.

        The singleton time axis of ITK's five-dimensional layout is
        dropped. A `VECTOR` (1007) file is read as it is, because ITK
        writes its LPS vectors unconverted. A three-component `DISPVECT`
        (1006) file holds RAS vectors, so the first two components are
        negated, as ITK 5.4 and later does when it reads one
        (`ConvertRASDisplacementVectors`, on by default).
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
                f"An ITK field is stored as a (X, Y, Z, 1, 3) array, "
                f"not as an array of shape {shape}."
            )
        if data.shape[-1] != _NDIM:
            raise ParserContentError(
                f"Only three-dimensional ITK fields are supported, and "
                f"this one has {data.shape[-1]} components."
            )
        if _nifti_intent(self.header) == _NIFTI_INTENT_DISPVECT:
            flip = backend.asarray(_RAS_LPS_VECTOR, dtype=data.dtype)
            data = data * flip
        return data

    # --- slots --------------------------------------------------------

    @property
    def lps2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from LPS world coordinates to the field's voxels."""
        return self.transformations[0]

    # --- writing ------------------------------------------------------

    def _xform_codes(self) -> tx.Tuple[int, int]:
        """The sform and qform codes to store."""
        header = self.header
        if header is not None:
            _, scode = header.get_sform(coded=True)
            _, qcode = header.get_qform(coded=True)
            if scode or qcode:
                return int(scode or qcode), int(qcode or scode)
        return _NIFTI_XFORM_SCANNER_ANAT, _NIFTI_XFORM_SCANNER_ANAT

    def _write_vectors(
        self,
        vectors: ArrayProtocol,
        vox2lps: np.ndarray,
        like: tx.Any = None,
        **overrides,
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the ITK NIfTI image of an `(X, Y, Z, 3)` array of LPS vectors.

        The vectors are written unconverted under the `VECTOR` (1007)
        intent, in ITK's `(X, Y, Z, 1, 3)` layout, and the grid's
        voxel-to-LPS affine is turned into the voxel-to-RAS affine that
        NIfTI stores -- which is what ITK writes for a vector image by
        default.
        """
        backend = get_array_backend(vectors)
        vectors = backend.asarray(vectors)
        if vectors.ndim != 4 or vectors.shape[-1] != _NDIM:
            raise WriterError(
                f"An ITK NIfTI field holds one 3-vector per voxel of a 3-D "
                f"grid, not an array of shape {tuple(vectors.shape)}."
            )
        vectors = backend.expand_dims(vectors, axis=3)
        vox2ras = _RAS_LPS @ np.asarray(vox2lps, dtype=np.float64)
        image = _new_nifti(vectors, vox2ras)
        image.header.set_intent(_NIFTI_INTENT_VECTOR)
        scode, qcode = self._xform_codes()
        image.header.set_sform(vox2ras, code=scode)
        image.header.set_qform(vox2ras, code=qcode)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


def _homogeneous(xform: _xforms.Transformation) -> np.ndarray:
    """The `(4, 4)` matrix of a three-dimensional affine-like slot."""
    try:
        matrix = xform.to(_xforms.Affine).homogeneous_matrix
    except Exception as error:
        raise WriterError(
            f"The grid of an ITK NIfTI field must be an affine, not a "
            f"{type(xform).__name__}."
        ) from error
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (_NDIM + 1, _NDIM + 1):
        raise WriterError(
            f"The grid of an ITK NIfTI field must be three-dimensional, "
            f"and its affine has shape {matrix.shape[0] - 1}x"
            f"{matrix.shape[1] - 1}."
        )
    return matrix


@register_format
class ITKNiftiDisplacementField(ITKNiftiField):
    """
    ITK displacement field, stored as a NIfTI vector image.

    This is how ITK, and ANTs through it, store a dense nonlinear warp,
    such as `antsRegistration`'s `<prefix>1Warp.nii.gz`. The field maps
    LPS to LPS, as `x_lps -> x_lps + u_lps(x_lps)`, and is the
    [`Sequence`][brainhops.datamodel.transformations.Sequence] of three
    named slots that the ITK warp blocks use too:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `lps2voxel`    | LPS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2lps`    | the field's voxels back to LPS world        |

    The displacements are interpolated linearly and extended with the
    nearest value outside the grid, which is the default interpolator of
    ITK's `DisplacementFieldTransform`
    (`VectorLinearInterpolateNearestNeighborExtrapolateImageFunction`).

    A `VECTOR` (1007) file with ITK's layout is claimed with certainty,
    but so is it by the RAS NIfTI field reader: the header alone cannot
    tell the two apart, so loading one without a hint is ambiguous. An
    explicit `hint="itk"` decides. Any other file with ITK's layout is
    only claimed weakly, so that a hint can still reach it.
    """

    HINTS = ("displacements",)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a NIfTI header as an ITK displacement field.

        `VECTOR` (1007), in ITK's `(X, Y, Z, 1, 3)` layout, is exactly
        what ITK writes, so it scores `CERTAIN` -- the same as the RAS
        reader gives it, on purpose: nothing in the header says which
        frame the vectors are in, so neither reader may outrank the
        other on content. Another intent code in the same layout is
        still readable but is probably not ITK's, and scores `WEAK`; the
        FSL codes are left to the FSL readers.
        """
        if not _is_itk_layout(_nifti_shape(header)):
            return Confidence.NO
        intent = _nifti_intent(header)
        if intent == _NIFTI_INTENT_VECTOR:
            return Confidence.CERTAIN
        if intent in _NIFTI_FSL_INTENTS:
            return Confidence.NO
        return Confidence.WEAK

    order: tx.ClassVar[int] = 1
    """The spline order used to interpolate the field."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition used outside of the field of view."""

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        It is built from the NIfTI header and data on first access, and
        cached. Assigning to it overrides the derived chain, which is how
        a field that was not read from a file is built.
        """
        vox2lps = self._vox2lps()
        compact = vox2lps[:_NDIM]
        # The stored vectors are world-space displacements, and a
        # `DisplacementField` adds its values in the units of its own
        # grid, so they are rotated into voxel units. Only the linear
        # part of the world-to-voxel affine acts on a displacement.
        vectors = self._lps_vectors()
        backend = get_array_backend(vectors)
        lps2vox = np.linalg.inv(vox2lps[:_NDIM, :_NDIM])
        rotate = backend.asarray(lps2vox, dtype=vectors.dtype)
        field = backend.matmul(rotate, vectors[..., None])[..., 0]
        voxel = _systems.VoxelCoordinateSystem()
        return (
            LPSToVoxel(matrix=_affines.inv(compact)),
            _xforms.DisplacementField(
                field=field,
                input=voxel,
                output=voxel,
                order=self.order,
                bound=self.bound,
            ),
            VoxelToLPS(matrix=compact),
        )

    @property
    def displacement(self) -> tx.Optional[_xforms.Transformation]:
        """The displacement field, in the voxel units of its grid."""
        return self.transformations[1]

    @property
    def voxel2lps(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from the field's voxels back to LPS world."""
        return self.transformations[2]

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that ITK would write for this field.

        The displacements are rotated from voxel units back into LPS
        millimetres and written, unconverted, as a `VECTOR` (1007)
        image of shape `(X, Y, Z, 1, 3)`. The grid's voxel-to-LPS affine
        becomes a voxel-to-RAS sform and qform.

        When `like` is given, non-encoding header fields are copied from
        it. Keyword arguments override header fields last.
        """
        chain = tuple(self.transformations or ())
        if len(chain) != 3 or not isinstance(
            chain[1], _xforms.DisplacementField
        ):
            raise WriterError(
                "An ITK displacement field is written from a chain of "
                "three transformations: LPS to voxel, a displacement "
                "field, and voxel to LPS."
            )
        displacement = chain[1]
        if displacement.field is None:
            raise WriterError(
                "This field has no displacements, so there is nothing to "
                "write."
            )
        if displacement.coeff:
            raise WriterError(
                "ITK stores sampled displacements, and this field holds "
                "spline coefficients. Convert it to values first."
            )
        vox2lps = _homogeneous(chain[2])
        field = displacement.field
        backend = get_array_backend(field)
        field = backend.asarray(field)
        rotate = backend.asarray(vox2lps[:_NDIM, :_NDIM], dtype=field.dtype)
        vectors = backend.matmul(rotate, field[..., None])[..., 0]
        return self._write_vectors(vectors, vox2lps, like, **overrides)


@register_format
class ITKNiftiCoordinatesField(ITKNiftiField):
    """
    Field of LPS coordinates, stored as an ITK NIfTI vector image.

    Each voxel holds the absolute LPS position it maps to. The field maps
    LPS to LPS, as the [`Sequence`][brainhops.datamodel.transformations.Sequence]
    of two named slots:

    | Slot          | Transformation                                 |
    | ------------- | ---------------------------------------------- |
    | `lps2voxel`   | LPS world coordinates to the field's voxels    |
    | `coordinates` | the field of LPS coordinates, on that grid     |

    ITK and ANTs only write displacement fields, and a NIfTI header
    cannot say whether its vectors are displacements or positions, so
    this reader never claims a file from its content. Ask for it with
    `hint="itk.coordinates"`, or use the class directly.
    """  # noqa: E501

    HINTS = ("coordinates",)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Never claim a file from its content.

        A field of positions and a field of displacements share ITK's
        layout and intent code, and the content cannot separate them
        without guessing from the values. Only a hint selects this
        reader; it is still kept in the running by its extension then.
        """
        return Confidence.NO

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        It is built from the NIfTI header and data on first access, and
        cached. Assigning to it overrides the derived chain.
        """
        compact = self._vox2lps()[:_NDIM]
        return (
            LPSToVoxel(matrix=_affines.inv(compact)),
            LPSCoordinatesField(field=self._lps_vectors()),
        )

    @property
    def coordinates(self) -> tx.Optional[_xforms.Transformation]:
        """The field of LPS coordinates, on the field's grid."""
        return self.transformations[1]

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image of this field of LPS coordinates.

        The coordinates are written, unconverted, as a `VECTOR` (1007)
        image of shape `(X, Y, Z, 1, 3)`, and the inverse of the
        `lps2voxel` affine becomes a voxel-to-RAS sform and qform.

        When `like` is given, non-encoding header fields are copied from
        it. Keyword arguments override header fields last.
        """
        chain = tuple(self.transformations or ())
        if len(chain) != 2 or not isinstance(
            chain[1], _xforms.CoordinatesField
        ):
            raise WriterError(
                "An ITK coordinates field is written from a chain of two "
                "transformations: LPS to voxel, and a coordinates field."
            )
        coordinates = chain[1]
        if coordinates.field is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        if coordinates.coeff:
            raise WriterError(
                "ITK stores sampled vectors, and this field holds spline "
                "coefficients. Convert it to values first."
            )
        vox2lps = np.linalg.inv(_homogeneous(chain[0]))
        return self._write_vectors(
            coordinates.field, vox2lps, like, **overrides
        )
