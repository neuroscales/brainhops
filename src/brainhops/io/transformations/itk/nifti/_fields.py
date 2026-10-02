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
from brainhops.datamodel.axes import SpaceAxis
from brainhops.datamodel.enums import BoundaryCondition

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import (
    _NIFTI_INTENT_DISPVECT,
    _NIFTI_INTENT_NAME_MAPPING,
    _NIFTI_INTENT_NAME_NIFTYREG,
    _NIFTI_INTENT_VECTOR,
    _apply_like,
    _apply_overrides,
    _embed_affine,
    _new_nifti,
    _nifti_intent,
    _nifti_intent_name,
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

# locals
from .._systems import _make_system

_NDIMS = (2, 3)
"""
The numbers of spatial dimensions these readers support.

NIfTI stores three spatial axes, and ITK writes a vector image of fewer
dimensions with singleton axes in their place, so a 2-D and a 3-D field
are stored alike and read by the same code. ITK can also write a 4-D
vector image, with its fourth dimension in the time axis, but NIfTI has
no geometry for that axis, so it is not read here.
"""

_NIFTI_NDIM = 3
"""The number of spatial axes, and of geometric dimensions, of a NIfTI."""

_RAS_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""
The homogeneous matrix that maps RAS to LPS, and LPS to RAS.

It negates the first two axes and is its own inverse. ITK applies it to
the direction and origin of an image of any dimension when it writes the
NIfTI sform and qform, and when it reads them back
(`itkNiftiImageIO.cxx`, `SetNIfTIOrientationFromImageIO` and
`SetImageIOOrientationFromNIfTI`), so a 2-D ITK image lives in (L, P).
"""

_NIFTI_XFORM_SCANNER_ANAT = 1
"""
The sform and qform code ITK writes.

`SetNIfTIOrientationFromImageIO` sets both codes to
`NIFTI_XFORM_SCANNER_ANAT`, so a field that was not read from a file is
written the same way.
"""


def _itk_ndim(shape: tx.Optional[tx.Sequence[int]]) -> tx.Optional[int]:
    """
    The number of spatial dimensions of a field stored in ITK's layout.

    ITK writes a vector image with `dim[0] = 5`, singleton axes in place
    of the spatial and time dimensions it does not have, and its
    components in the fifth axis (`WriteImageInformation`): a 3-D field
    is `(X, Y, Z, 1, 3)` and a 2-D one `(X, Y, 1, 1, 2)`. A displacement
    has as many components as the grid has dimensions. `None` when the
    shape is not one of these layouts.
    """
    if shape is None or len(shape) != 5 or int(shape[3]) != 1:
        return None
    ndim = int(shape[4])
    if ndim not in _NDIMS:
        return None
    if any(int(size) != 1 for size in shape[ndim:_NIFTI_NDIM]):
        return None
    return ndim


def _voxel_system(ndim: int) -> _systems.CoordinateSystem:
    """
    The voxel system of an `ndim`-dimensional grid.

    Its axes are spatial and count samples, so the system is a
    `PixelCoordinateSystem` in 2-D and a `VoxelCoordinateSystem` in 3-D.
    """
    axes = [SpaceAxis(name=f"dim{i}", unit="sample") for i in range(ndim)]
    return _systems.CoordinateSystem(axes=axes)


def _homogeneous(xform: _xforms.Transformation) -> np.ndarray:
    """The square homogeneous matrix of an affine-like slot."""
    try:
        matrix = xform.to(_xforms.Affine).homogeneous_matrix
    except Exception as error:
        raise WriterError(
            f"The grid of an ITK NIfTI field must be an affine, not a "
            f"{type(xform).__name__}."
        ) from error
    matrix = np.asarray(matrix, dtype=np.float64)
    ndim = matrix.shape[0] - 1
    if matrix.shape[1] != ndim + 1 or ndim not in _NDIMS:
        raise WriterError(
            f"The grid of an ITK NIfTI field maps between two spaces of "
            f"the same dimension, 2 or 3, and its affine is "
            f"{matrix.shape[0] - 1}x{matrix.shape[1] - 1}."
        )
    return matrix


class ItkNiftiField(_xforms.ImmutableSequence, NiftiBasedTransformation):
    """
    A field stored in an ITK NIfTI vector image, from LPS to LPS.

    The vectors of the image are in ITK's LPS physical space, while the
    NIfTI header carries the usual voxel-to-RAS affine. This base reads
    both, and leaves to its subclasses what the vectors mean:
    displacements ([`ItkNiftiDisplacementField`][]) or absolute
    coordinates ([`ItkNiftiCoordinatesField`][]).

    The same code reads 2-D and 3-D fields: the dimension is read off the
    header, and the endpoints are the ITK spaces of that dimension --
    (L, P) in 2-D and `LPSmm` in 3-D, both in millimetres.

    The chain is made of named slots, so the field is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]:
    editing it in place raises `TypeError`.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch.
    """

    HINTS = ("itk", "ants")

    # --- reading ------------------------------------------------------

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """
        Build the field from a `nibabel` header or image.

        The header must be in ITK's layout, so a file that is not an ITK
        field is refused here, from its header alone, rather than when
        its chain is first built.
        """
        header = nifti.header if isinstance(nifti, nb.Nifti1Image) else nifti
        shape = _nifti_shape(header)
        if _itk_ndim(shape) is None:
            raise ParserContentError(
                f"An ITK field is stored as a (X, Y, Z, 1, 3) or "
                f"(X, Y, 1, 1, 2) array, not as an array of shape {shape}."
            )
        return super().from_nibabel(nifti, **kwargs)

    # --- endpoints ----------------------------------------------------
    #
    # Declared rather than read off the chain, as the ITK blocks do:
    # reading them off the chain would build the chain, and building it
    # decodes the field data. They are the LPS spaces the ITK blocks use,
    # so a warp read from NIfTI and an affine read from a `.tfm` name the
    # same space.

    def _ndim(self) -> tx.Optional[int]:
        """The number of spatial dimensions, without decoding the data."""
        header = self.header
        if header is not None:
            return _itk_ndim(_nifti_shape(header))
        chain = getattr(self, "_transformations", None)
        if chain:
            # The first slot maps LPS to voxels: one row per dimension.
            return int(np.asarray(chain[0].matrix).shape[0])
        return None

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the field maps from."""
        ndim = self._ndim()
        return None if ndim is None else _make_system(ndim)

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the field maps to."""
        ndim = self._ndim()
        return None if ndim is None else _make_system(ndim)

    # --- decoding -----------------------------------------------------

    def _grid(self) -> tx.Tuple[int, np.ndarray]:
        """
        The dimension of the field's grid, and its compact voxel-to-LPS
        affine, of shape `(ndim, ndim + 1)`.

        The header stores voxel-to-RAS, as every NIfTI does, so the first
        two rows are negated. A 2-D grid keeps the rows and columns of x
        and y, as ITK does when it reads a 2-D image. `get_best_affine`
        prefers the sform when its code is set; ITK writes the same
        matrix to both forms.
        """
        header = self.header
        ndim = None if header is None else self._ndim()
        if ndim is None:
            raise ParserContentError(
                "This field has no ITK NIfTI header to read its grid from."
            )
        vox2ras = np.asarray(header.get_best_affine(), dtype=np.float64)
        vox2lps = _RAS_LPS @ vox2ras
        columns = [*range(ndim), _NIFTI_NDIM]
        return ndim, vox2lps[:ndim][:, columns]

    def _lps_vectors(self, ndim: int) -> ArrayProtocol:
        """
        The stored vectors, as an `(*shape, ndim)` array in LPS.

        The singleton axes of ITK's five-dimensional layout are dropped.
        A `VECTOR` (1007) file is read as it is, because ITK writes its
        LPS vectors unconverted. A three-component `DISPVECT` (1006) file
        holds RAS vectors, so its first two components are negated, as
        ITK 5.4 and later does when it reads one
        (`ConvertRASDisplacementVectors`, on by default). ITK leaves a
        two-component `DISPVECT` file unconverted, and so does this.
        """
        data = self.data
        if data is None:
            raise ParserContentError("This field has no data to read.")
        backend = get_array_backend(data)
        data = backend.asarray(data)
        shape = tuple(int(d) for d in data.shape)
        if len(shape) != ndim + 1:
            data = backend.reshape(data, (*shape[:ndim], ndim))
        dispvect = _nifti_intent(self.header) == _NIFTI_INTENT_DISPVECT
        if dispvect and ndim == 3:
            flip = np.diag(_RAS_LPS)[:ndim]
            data = data * backend.asarray(flip, dtype=data.dtype)
        return data

    @staticmethod
    def _spaces(
        ndim: int,
    ) -> tx.Tuple[_systems.CoordinateSystem, _systems.CoordinateSystem]:
        """The LPS space and the voxel space of an `ndim`-D field."""
        return _make_system(ndim), _voxel_system(ndim)

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
        Build the ITK NIfTI image of an `(*shape, ndim)` array of LPS
        vectors, given the grid's homogeneous voxel-to-LPS affine.

        The vectors are written unconverted under the `VECTOR` (1007)
        intent, in ITK's `(X, Y, Z, 1, 3)` or `(X, Y, 1, 1, 2)` layout,
        and the voxel-to-LPS affine becomes the voxel-to-RAS affine that
        NIfTI stores, with the identity on the axes a 2-D grid does not
        have -- which is what ITK writes for a vector image by default.
        """
        backend = get_array_backend(vectors)
        vectors = backend.asarray(vectors)
        ndim = int(vectors.shape[-1])
        if ndim not in _NDIMS or vectors.ndim != ndim + 1:
            raise WriterError(
                f"An ITK NIfTI field holds one vector per voxel of a 2-D "
                f"or 3-D grid, with as many components as the grid has "
                f"dimensions, not an array of shape {tuple(vectors.shape)}."
            )
        if vox2lps.shape != (ndim + 1, ndim + 1):
            raise WriterError(
                f"A {ndim}-D field needs a {ndim}-D grid, and its affine "
                f"is {vox2lps.shape[0] - 1}x{vox2lps.shape[1] - 1}."
            )
        spatial = tuple(int(d) for d in vectors.shape[:ndim])
        singletons = (1,) * (_NIFTI_NDIM - ndim)
        vectors = backend.reshape(vectors, (*spatial, *singletons, 1, ndim))
        vox2ras = _RAS_LPS @ _embed_affine(vox2lps)
        image = _new_nifti(vectors, vox2ras)
        image.header.set_intent(_NIFTI_INTENT_VECTOR)
        scode, qcode = self._xform_codes()
        image.header.set_sform(vox2ras, code=scode)
        image.header.set_qform(vox2ras, code=qcode)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


@register_format
class ItkNiftiDisplacementField(ItkNiftiField):
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

    The field may be 3-D, stored as `(X, Y, Z, 1, 3)` and mapping LPS to
    LPS, or 2-D, stored as `(X, Y, 1, 1, 2)` and mapping (L, P) to
    (L, P).

    The displacements are interpolated linearly and extended with the
    nearest value outside the grid, which is the default interpolator of
    ITK's `DisplacementFieldTransform`
    (`VectorLinearInterpolateNearestNeighborExtrapolateImageFunction`).

    A `VECTOR` (1007) file with ITK's layout is claimed with certainty,
    but so is it by the RAS NIfTI field reader: the header alone cannot
    tell the two apart, so loading one without a hint is ambiguous. An
    explicit `hint="itk"` (or `hint="ants"`) decides, and also reaches a
    file in ITK's layout with any other intent code.
    """

    HINTS = ("displacements",)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a NIfTI header as an ITK displacement field.

        `VECTOR` (1007), in ITK's layout, is exactly what ITK writes, so
        it scores `CERTAIN` -- the same as the RAS reader gives it, on
        purpose: nothing in the header says which frame the vectors are
        in, so neither reader may outrank the other on content. Any other
        intent code is not ITK's default and is not claimed; a hint still
        reaches such a file, through its `.nii`/`.nii.gz` extension.

        A `VECTOR` file whose intent name is `"Mapping"` is not claimed
        either. SPM12 writes its `y_` deformations that way, and so does
        brainhops' RAS coordinates writer, while ITK's `NiftiImageIO`
        never writes an intent name, and ANTs writes through it: the name
        is evidence of a RAS map, not of an ITK one. Nor is one named
        `"NREG_TRANS"`, which NiftyReg writes on its (RAS) fields.
        """
        if _itk_ndim(_nifti_shape(header)) is None:
            return Confidence.NO
        if _nifti_intent(header) != _NIFTI_INTENT_VECTOR:
            return Confidence.NO
        if _nifti_intent_name(header) in (
            _NIFTI_INTENT_NAME_MAPPING,
            _NIFTI_INTENT_NAME_NIFTYREG,
        ):
            return Confidence.NO
        return Confidence.CERTAIN

    degree: tx.ClassVar[int] = 1
    """The spline degree used to interpolate the field."""

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
        ndim, vox2lps = self._grid()
        world, voxel = self._spaces(ndim)
        # The stored vectors are world-space displacements, and a
        # `DisplacementField` adds its values in the units of its own
        # grid, so they are rotated into voxel units. Only the linear
        # part of the world-to-voxel affine acts on a displacement.
        vectors = self._lps_vectors(ndim)
        backend = get_array_backend(vectors)
        lps2vox = np.linalg.inv(vox2lps[:, :ndim])
        rotate = backend.asarray(lps2vox, dtype=vectors.dtype)
        field = backend.matmul(rotate, vectors[..., None])[..., 0]
        return (
            LPSToVoxel(
                matrix=_affines.inv(vox2lps), input=world, output=voxel
            ),
            _xforms.DisplacementField(
                field=field,
                input=voxel,
                output=voxel,
                degree=self.degree,
                bound=self.bound,
            ),
            VoxelToLPS(matrix=vox2lps, input=voxel, output=world),
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
        image of shape `(X, Y, Z, 1, 3)`, or `(X, Y, 1, 1, 2)` in 2-D.
        The grid's voxel-to-LPS affine becomes a voxel-to-RAS sform and
        qform.

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
        ndim = vox2lps.shape[0] - 1
        field = displacement.field
        backend = get_array_backend(field)
        field = backend.asarray(field)
        rotate = backend.asarray(vox2lps[:ndim, :ndim], dtype=field.dtype)
        vectors = backend.matmul(rotate, field[..., None])[..., 0]
        return self._write_vectors(vectors, vox2lps, like, **overrides)


@register_format
class ItkNiftiCoordinatesField(ItkNiftiField):
    """
    Field of LPS coordinates, stored as an ITK NIfTI vector image.

    Each voxel holds the absolute LPS position it maps to. The field maps
    LPS to LPS (or (L, P) to (L, P) in 2-D), as the
    [`Sequence`][brainhops.datamodel.transformations.Sequence] of two
    named slots:

    | Slot          | Transformation                                 |
    | ------------- | ---------------------------------------------- |
    | `lps2voxel`   | LPS world coordinates to the field's voxels    |
    | `coordinates` | the field of LPS coordinates, on that grid     |

    ITK and ANTs only write displacement fields, and a NIfTI header
    cannot say whether its vectors are displacements or positions, so
    this reader never claims a file from its content. Ask for it with
    `hint="itk.coordinates"`, or use the class directly.
    """

    HINTS = ("coordinates",)

    PRIORITY: tx.ClassVar[int] = -1
    """
    Under `hint="itk"`, a file in ITK's layout with another intent than
    `VECTOR` is claimed by neither ITK reader, and both match its
    extension, so the two genuinely tie. ITK and ANTs mean a
    displacement, so this reader yields to that one.
    """

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
        ndim, vox2lps = self._grid()
        world, voxel = self._spaces(ndim)
        return (
            LPSToVoxel(
                matrix=_affines.inv(vox2lps), input=world, output=voxel
            ),
            LPSCoordinatesField(
                field=self._lps_vectors(ndim), input=voxel, output=world
            ),
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
        image of shape `(X, Y, Z, 1, 3)`, or `(X, Y, 1, 1, 2)` in 2-D,
        and the inverse of the `lps2voxel` affine becomes a voxel-to-RAS
        sform and qform.

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
