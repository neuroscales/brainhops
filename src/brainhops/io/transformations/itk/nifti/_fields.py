"""
ITK displacement and coordinates fields stored as LPS NIfTI vector images.

The encoding and the way these files are told apart from RAS NIfTI fields are
described in [`brainhops.io.transformations.itk.nifti`][].
"""

import nibabel as nb
import numpy as np
import typing_extensions as tx

from brainhops._core import affines as _affines
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.axes import SpaceAxis
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.common.nifti._constants import (
    _NIFTI_INTENT_DISPVECT,
    _NIFTI_INTENT_NAME_MAPPING,
    _NIFTI_INTENT_NAME_NIFTYREG,
    _NIFTI_INTENT_VECTOR,
)
from brainhops.io.common.nifti._geometry import _embed_affine
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _header,
    _nifti_intent,
    _nifti_intent_name,
    _nifti_shape,
    _NiftiObject,
)
from brainhops.io.common.nifti._views import (
    _itk_field_to_disk,
    _itk_field_to_model,
)
from brainhops.io.transformations.base.affines import LPSToVoxel, VoxelToLPS
from brainhops.io.transformations.base.fields import LPSCoordinatesField
from brainhops.io.transformations.nifti import NiftiBasedTransformation
from brainhops.io.transformations.nifti._base import _always, _record_image

from .._systems import _make_system

_NDIMS = (2, 3)
"""
Supported spatial dimensions.

ITK pads an image with fewer than three dimensions with singleton axes, so 2-D
and 3-D fields share the same code. A 4-D field is not supported, since NIfTI
has no geometry for a time axis.
"""

_NIFTI_NDIM = 3

_RAS_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""
Homogeneous RAS-to-LPS matrix, which negates the first two axes.

ITK applies it to the direction and origin of an image of any dimension when it
writes or reads the sform and qform (`itkNiftiImageIO.cxx`), so a 2-D ITK image
lives in (L, P).
"""

_NIFTI_XFORM_SCANNER_ANAT = 1
"""
The sform and qform code that ITK writes, used for fields not read from a file.
"""


def _itk_ndim(shape: tx.Optional[tx.Sequence[int]]) -> tx.Optional[int]:
    """
    Return the spatial dimension of a field in the ITK layout, or `None`.

    ITK writes a vector image with five dimensions, padding the missing spatial
    and time axes with singletons and storing the components in the fifth axis.
    A 3-D field has shape (X, Y, Z, 1, 3) and a 2-D field has shape (X, Y, 1,
    1, 2).
    """
    if shape is None or len(shape) != 5 or int(shape[3]) != 1:
        return None
    ndim = int(shape[4])
    if ndim not in _NDIMS:
        return None
    if any(int(size) != 1 for size in shape[ndim:_NIFTI_NDIM]):
        return None
    return ndim


def _check_layout(header: _NiftiObject) -> None:
    """Refuse a header whose array is not in the layout of an ITK field.

    Raises
    ------
    ParserContentError
        If the shape is neither (X, Y, Z, 1, 3) nor (X, Y, 1, 1, 2).
    """
    shape = _nifti_shape(header)
    if _itk_ndim(shape) is None:
        raise ParserContentError(
            f"An ITK field is stored as a (X, Y, Z, 1, 3) or "
            f"(X, Y, 1, 1, 2) array, not as an array of shape {shape}."
        )


def _set_itk_field_data(
    self: "ItkNiftiField", value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the vectors of an ITK field in `raw`, in the ITK layout."""
    self.raw = None if value is None else _itk_field_to_disk(value)
    self.__dict__.pop("_cache_data", None)


def _voxel_system(ndim: int) -> _systems.CoordinateSystem:
    """Return a voxel space whose axes count samples in index units."""
    axes = [SpaceAxis(name=f"dim{i}", unit="index") for i in range(ndim)]
    return _systems.CoordinateSystem(axes=axes)


def _homogeneous(xform: _xforms.Transformation) -> np.ndarray:
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
    Base class of the fields stored in ITK NIfTI vector images.

    The vectors are expressed in the LPS physical space of ITK, while the
    header holds the usual voxel-to-RAS affine. Subclasses decide whether the
    vectors are displacements ([`ItkNiftiDisplacementField`][]) or absolute
    coordinates ([`ItkNiftiCoordinatesField`][]).

    The dimension of the field is read from the header, and its endpoints are
    the ITK spaces of that dimension: (L, P) in 2-D and `LPSmm` in 3-D. The
    field is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of named slots, so editing it in place raises a `TypeError`. This class is
    not registered as a format.
    """

    HINTS = ("itk", "ants")

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        # A file that is not in the ITK layout is refused from its header
        # alone, rather than when the chain is built.
        header = _header(self)
        if header is not None:
            _check_layout(header)

    # --- data ---------------------------------------------------------

    @smartproperty(
        cache=True,
        unset=_always,
        fset=_set_itk_field_data,
        invalidates=("transformations",),
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The stored vectors, as an (X, Y, Z, 3) or (X, Y, 2) array.

        The vectors are decoded from `raw` without the singleton axes of the
        ITK layout, and they are those of the file, before the frame of a
        `DISPVECT` file is converted. Setting the data stores it in `raw` in
        the ITK layout and drops the chain decoded from the previous data.
        """
        if self.raw is None:
            return None
        return _itk_field_to_model(self.raw)

    metadata = smartproperty(
        "metadata", invalidates=("transformations", "input", "output")
    )
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the chain and the endpoints decoded
    from the previous header.
    """

    # --- endpoints ----------------------------------------------------
    # The endpoints are declared rather than read off the chain, which would
    # decode the data. They are the ITK LPS spaces, so a NIfTI warp and a .tfm
    # affine name the same space.

    def _ndim(self) -> tx.Optional[int]:
        """Return the spatial dimension without decoding the data.

        The dimension is read from the header, or else from an assigned
        chain, or else from the shape of `raw`.
        """
        header = _header(self)
        if header is not None:
            return _itk_ndim(_nifti_shape(header))
        chain = getattr(self, "_transformations", None)
        if chain:
            # The first slot maps LPS to voxels, with one row per dimension.
            return int(np.asarray(chain[0].matrix).shape[0])
        if self.raw is not None:
            return _itk_ndim(tuple(self.raw.shape))
        return None

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The ITK physical space that the field maps from."""
        ndim = self._ndim()
        return None if ndim is None else _make_system(ndim)

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The ITK physical space that the field maps to."""
        ndim = self._ndim()
        return None if ndim is None else _make_system(ndim)

    # --- decoding -----------------------------------------------------

    def _grid(self) -> tx.Tuple[int, np.ndarray]:
        """
        Return the grid dimension and the compact voxel-to-LPS affine.

        The affine has shape `(ndim, ndim + 1)`. The header stores a
        voxel-to-RAS affine, so its first two rows are negated. ITK writes the
        same matrix to the sform and the qform. A field without metadata has
        no grid of its own, and its voxel-to-RAS affine is the identity.
        """
        ndim = self._ndim()
        if ndim is None:
            raise ParserContentError(
                "This field has no ITK NIfTI header or array to read its "
                "grid from."
            )
        header = _header(self)
        if header is None:
            vox2ras = np.eye(4)
        else:
            vox2ras = np.asarray(header.get_best_affine(), dtype=np.float64)
        vox2lps = _RAS_LPS @ vox2ras
        columns = [*range(ndim), _NIFTI_NDIM]
        return ndim, vox2lps[:ndim][:, columns]

    def _lps_vectors(self, ndim: int) -> ArrayProtocol:
        """
        Return the stored vectors as an LPS array of shape `(*shape, ndim)`.

        Vectors stored under the `VECTOR` intent (1007) are read as they are,
        because ITK writes LPS vectors unconverted. A 3-component field under
        the `DISPVECT` intent (1006) holds RAS vectors, so its first two
        components are negated, as ITK 5.4 and later do on reading. A
        2-component `DISPVECT` field is left unconverted, as in ITK.
        """
        data = self.data
        if data is None:
            raise ParserContentError("This field has no data to read.")
        backend = get_array_backend(data)
        shape = tuple(int(d) for d in data.shape)
        if len(shape) != ndim + 1:
            data = backend.reshape(data, (*shape[:ndim], ndim))
        dispvect = _nifti_intent(_header(self)) == _NIFTI_INTENT_DISPVECT
        if dispvect and ndim == 3:
            flip = np.diag(_RAS_LPS)[:ndim]
            data = data * backend.asarray(flip, dtype=data.dtype)
        return data

    @staticmethod
    def _spaces(
        ndim: int,
    ) -> tx.Tuple[_systems.CoordinateSystem, _systems.CoordinateSystem]:
        return _make_system(ndim), _voxel_system(ndim)

    # --- slots --------------------------------------------------------

    @property
    def lps2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from LPS world coordinates to the voxels of the field."""
        return self.transformations[0]

    # --- writing ------------------------------------------------------

    def _xform_codes(self) -> tx.Tuple[int, int]:
        """
        Return the sform and qform codes to store.

        The codes come from the header when either is set, and each falls back
        on the other. Otherwise both are `NIFTI_XFORM_SCANNER_ANAT`.
        """
        header = _header(self)
        if header is not None:
            _, scode = header.get_sform(coded=True)
            _, qcode = header.get_qform(coded=True)
            if scode or qcode:
                return int(scode or qcode), int(qcode or scode)
        return _NIFTI_XFORM_SCANNER_ANAT, _NIFTI_XFORM_SCANNER_ANAT

    def _read_back(
        self, like: tx.Any = None, **overrides
    ) -> tx.Optional[tx.Union[nb.Nifti1Image, nb.Nifti2Image]]:
        """
        Return the NIfTI image of the field as it was read, or `None`.

        A field whose chain is decoded from its file is written back as it
        is stored, under a copy of the header of its metadata, so that a
        `DISPVECT` file stays one. `None` is returned when the field has no
        metadata or no array, or when its chain was assigned.
        """
        if self.metadata is None or self.raw is None:
            return None
        if getattr(self, "_transformations", None) is not None:
            return None
        image = _record_image(self.raw, self.metadata, overrides=overrides)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image

    def _write_vectors(
        self,
        vectors: ArrayProtocol,
        vox2lps: np.ndarray,
        like: tx.Any = None,
        **overrides,
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build an ITK NIfTI image from LPS vectors and a voxel-to-LPS affine.

        The vectors are written unconverted under the `VECTOR` intent (1007),
        in the five-dimensional ITK layout. The affine becomes the voxel-to-RAS
        affine of the header, with an identity on the axes that a 2-D grid
        lacks. A copy of the header of the metadata is the base of the new
        header, and its fields that describe the layout and the geometry are
        encoded again.
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
        vox2ras = _RAS_LPS @ _embed_affine(vox2lps)
        vectors = _itk_field_to_disk(vectors)
        image = _record_image(vectors, self.metadata, vox2ras, overrides)
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
    Dense nonlinear ITK or ANTs warp stored as a NIfTI vector image.

    Such a field, for example the `<prefix>1Warp.nii.gz` file of
    `antsRegistration`, maps LPS to LPS as `x -> x + u(x)`, or (L, P) to (L, P)
    in 2-D. It is a [`Sequence`][brainhops.datamodel.transformations.Sequence]
    of three named slots:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `lps2voxel`    | LPS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2lps`    | the field's voxels back to LPS world        |

    The displacements are interpolated linearly and extrapolated with the
    nearest value, as by the default interpolator of ITK's
    `DisplacementFieldTransform`.

    The RAS NIfTI field reader also claims a `VECTOR` file (1007) in the ITK
    layout with certainty, and the header cannot tell the two apart. Such a
    file is therefore read with `hint="itk"` or `hint="ants"`, which also
    reaches a file in the ITK layout with any other intent code.
    """

    HINTS = ("displacements",)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a header as an ITK displacement field.

        A `VECTOR` image in the ITK layout scores `Confidence.CERTAIN`,
        deliberately equal to the RAS reader, since neither frame may outrank
        the other. Other intent codes are left to a hint. The intent name
        `"Mapping"`, which SPM12 and the brainhops coordinates writer use but
        ITK never writes, is not claimed, and neither is the NiftyReg name
        `"NREG_TRANS"`.
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
    """Spline degree of the interpolation."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """Boundary condition outside the field of view."""

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        The chain is built lazily from the header and the data, and then
        cached. Assigning a chain overrides the derived one, which is how
        fields that do not come from a file are built.
        """
        ndim, vox2lps = self._grid()
        world, voxel = self._spaces(ndim)
        # A DisplacementField adds its values in the units of its grid, so the
        # world-space vectors are rotated into voxel units by the linear part
        # of the
        # world-to-voxel affine.
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
        """The displacement field, in voxel units."""
        return self.transformations[1]

    @property
    def voxel2lps(self) -> tx.Optional[_xforms.Transformation]:
        """
        The affine from the voxels of the field back to LPS world coordinates.
        """
        return self.transformations[2]

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that ITK would write for this field.

        The displacements are rotated back to LPS millimetres and written
        unconverted under the `VECTOR` intent. The voxel-to-LPS affine becomes
        the voxel-to-RAS sform and qform. A field read from a file is written
        back as read. Non-encoding header fields are copied from `like`, and
        `overrides` are applied last.
        """
        image = self._read_back(like, **overrides)
        if image is not None:
            return image
        chain = tuple(self.transformations or ())
        if len(chain) != 3 or not isinstance(
            chain[1], _xforms.DisplacementField
        ):
            raise WriterError(
                "An ITK displacement field is written from a chain of "
                "three transformations: LPS to voxel, a displacement "
                "field, and voxel to LPS."
            )
        # ITK stores sampled displacements, so a velocity field is integrated.
        displacement = chain[1].to(log=False, store="values")
        if displacement.data is None:
            raise WriterError(
                "This field has no displacements, so there is nothing to "
                "write."
            )
        vox2lps = _homogeneous(chain[2])
        ndim = vox2lps.shape[0] - 1
        field = displacement.data
        backend = get_array_backend(field)
        field = backend.asarray(field)
        rotate = backend.asarray(vox2lps[:ndim, :ndim], dtype=field.dtype)
        vectors = backend.matmul(rotate, field[..., None])[..., 0]
        return self._write_vectors(vectors, vox2lps, like, **overrides)


@register_format
class ItkNiftiCoordinatesField(ItkNiftiField):
    """
    Field of absolute LPS coordinates stored as an ITK NIfTI vector image.

    Each voxel holds the LPS position that it maps to. The field is a
    [`Sequence`][brainhops.datamodel.transformations.Sequence] of two named
    slots:

    | Slot          | Transformation                                 |
    | ------------- | ---------------------------------------------- |
    | `lps2voxel`   | LPS world coordinates to the field's voxels    |
    | `coordinates` | the field of LPS coordinates, on that grid     |

    ITK and ANTs only write displacement fields, and a header cannot say
    whether it holds displacements or positions. This reader is therefore never
    chosen from the content of a file, only with `hint="itk.coordinates"` or by
    using the class directly.
    """

    HINTS = ("coordinates",)

    PRIORITY: tx.ClassVar[int] = -1
    """
    Priority below the displacement reader.

    Under `hint="itk"`, a file in the ITK layout without the `VECTOR` intent is
    claimed by neither reader, and the tie is resolved in favour of
    displacements, which is what ITK and ANTs write.
    """

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Never claim a file, since only a hint can select this reader."""
        return Confidence.NO

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        The chain is built lazily from the header and the data, and then
        cached. Assigning a chain overrides the derived one.
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
        """The field of LPS coordinates, on the grid of the field."""
        return self.transformations[1]

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image of the coordinates field.

        The coordinates are written unconverted under the `VECTOR` intent, and
        the inverse of the `lps2voxel` affine becomes the voxel-to-RAS sform
        and qform. A field read from a file is written back as read.
        Non-encoding header fields are copied from `like`, and `overrides` are
        applied last.
        """
        image = self._read_back(like, **overrides)
        if image is not None:
            return image
        chain = tuple(self.transformations or ())
        if len(chain) != 2 or not isinstance(
            chain[1], _xforms.CoordinatesField
        ):
            raise WriterError(
                "An ITK coordinates field is written from a chain of two "
                "transformations: LPS to voxel, and a coordinates field."
            )
        # ITK stores sampled vectors.
        coordinates = chain[1].to(store="values")
        if coordinates.data is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        vox2lps = np.linalg.inv(_homogeneous(chain[0]))
        return self._write_vectors(
            coordinates.data, vox2lps, like, **overrides
        )
