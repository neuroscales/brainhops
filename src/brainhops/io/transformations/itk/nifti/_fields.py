"""
ITK displacement and coordinate fields stored as LPS NIfTI vector images.

The encoding and the way these files are told apart from RAS NIfTI fields are
described in [`brainhops.io.transformations.itk.nifti`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, KwOnly, NoRepr, replace

# core
from brainhops._core import affines as _affines
from brainhops._core._internal_api import (
    ArrayProtocol,
    array_backend,
    smartproperty,
)

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.axes import SpaceAxis
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import apply_loss_policy
from brainhops.errors import (
    ParserContentError,
    WriterError,
)

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence
from brainhops.io.common.nifti import NiftiMetadata
from brainhops.io.common.nifti._internal_api import (
    NIFTI_INTENT_DISPVECT,
    NIFTI_INTENT_NAME_ITK_COORDINATES,
    NIFTI_INTENT_NAME_ITK_DISPLACEMENT,
    NIFTI_INTENT_NAME_MAPPING,
    NIFTI_INTENT_NAME_NIFTYREG,
    NIFTI_INTENT_VECTOR,
    NiftiObject,
    apply_like,
    apply_overrides,
    embed_nifti_affine,
    header_of,
    keep_given_fields,
    nifti_intent,
    nifti_intent_name,
    nifti_shape,
    record_image,
    split_options,
    stored_parts,
)
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    overruled_fields,
    requested_fields,
)
from brainhops.io.transformations.base.affines import LPSToVoxel, VoxelToLPS
from brainhops.io.transformations.base.fields import (
    COORDINATE_HINTS,
    DISPLACEMENT_HINTS,
    LPSCoordinateField,
)

# this_format
from brainhops.io.transformations.itk._internal_api import make_system
from brainhops.io.transformations.nifti import NiftiBasedTransformation
from brainhops.io.transformations.nifti._internal_api import (
    consolidated_metadata,
)

from ._views import itk_field_to_disk, itk_field_to_model

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


def _check_layout(header: NiftiObject) -> None:
    """Refuse a header whose array is not in the layout of an ITK field.

    Raises
    ------
    ParserContentError
        If the shape is neither (X, Y, Z, 1, 3) nor (X, Y, 1, 1, 2).
    """
    shape = nifti_shape(header)
    if _itk_ndim(shape) is None:
        raise ParserContentError(
            f"An ITK field is stored as a (X, Y, Z, 1, 3) or "
            f"(X, Y, 1, 1, 2) array, not as an array of shape {shape}."
        )


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


class ItkNiftiField(NiftiBasedTransformation, _xforms.DerivedSequence):
    """
    Base class of the fields stored in ITK NIfTI vector images.

    The vectors are expressed in the LPS physical space of ITK, while the
    header holds the usual voxel-to-RAS affine. Subclasses decide whether the
    vectors are displacements ([`ItkNiftiDisplacementField`][]) or absolute
    coordinates ([`ItkNiftiCoordinateField`][]).

    The dimension of the field is read from the header, and its endpoints are
    the ITK spaces of that dimension: (L, P) in 2-D and `LPSmm` in 3-D. The
    field is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of named slots, so editing it in place raises a `TypeError`. This class is
    not registered as a format.
    """

    HINTS = ("itk", "ants")

    INTENT_NAME: tx.ClassVar[str] = ""
    """The intent name that brainhops writes on the field, so that it reads
    the field back as this class without a hint. ITK ignores it."""

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        # A file that is not in the ITK layout is refused from its header
        # alone, rather than when the chain is built.
        header = header_of(self)
        if header is not None:
            _check_layout(header)

    # --- data ---------------------------------------------------------

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The vectors given to the field, or `None` to decode them from `rawdata`.

    Vectors stored here are pending until [`consolidate`][] or a write
    encodes them into `rawdata`.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The stored vectors, as an (X, Y, Z, 3) or (X, Y, 2) array.

        Vectors that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them. Otherwise
        they are decoded from `rawdata` without the singleton axes of the ITK
        layout, and they are those of the file, before the frame of a
        `DISPVECT` file is converted. Assigning the data drops the chain
        decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return itk_field_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

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

        The pending state comes first, since the header describes the
        file as it was read: the dimension is read from an assigned chain,
        or else from the shape of assigned vectors, or else from the
        header, or else from the shape of `rawdata`.
        """
        chain = self._transformations
        if chain:
            # The first slot maps LPS to voxels, with one row per dimension.
            return int(np.asarray(chain[0].matrix).shape[0])
        if self._data is not None:
            return int(self._data.shape[-1])
        header = header_of(self)
        if header is not None:
            return _itk_ndim(nifti_shape(header))
        if self.rawdata is not None:
            return _itk_ndim(tuple(self.rawdata.shape))
        return None

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The ITK physical space that the field maps from."""
        ndim = self._ndim()
        return None if ndim is None else make_system(ndim)

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The ITK physical space that the field maps to."""
        ndim = self._ndim()
        return None if ndim is None else make_system(ndim)

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
        header = header_of(self)
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
        xp = array_backend(data)
        shape = tuple(int(d) for d in data.shape)
        if len(shape) != ndim + 1:
            data = xp.reshape(data, (*shape[:ndim], ndim))
        dispvect = nifti_intent(header_of(self)) == NIFTI_INTENT_DISPVECT
        if dispvect and ndim == 3:
            flip = np.diag(_RAS_LPS)[:ndim]
            data = data * xp.asarray(flip, dtype=data.dtype)
        return data

    @staticmethod
    def _spaces(
        ndim: int,
    ) -> tx.Tuple[_systems.CoordinateSystem, _systems.CoordinateSystem]:
        return make_system(ndim), _voxel_system(ndim)

    # --- slots --------------------------------------------------------

    @property
    def lps2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from LPS world coordinates to the voxels of the field."""
        return self.transformations[0]

    # --- writing ------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        **options: tx.Any,
    ) -> tx.Self:
        """Encode the pending vectors or chain into `rawdata` and a header.

        A field whose chain is decoded from its file keeps its stored
        vectors, assigned or read, under a copy of the header of its
        metadata. Otherwise, each type encodes its chain as ITK would
        write it. Non-encoding header fields are copied from `like`, and
        `overrides` are applied last. A field that has a header and
        nothing pending, consolidated without options, is left as it is.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the options that name its fields, and its record is
        the base of the new header. The fields given by keyword take
        precedence over `like`, and a `dtype` that is given, or changed
        in the metadata, is the type of the stored array. What the write
        loses or approximates is acted on once, under the error policy
        of the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this field rather than a copy.
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        **options : Any
            Fields of the metadata to override, such as `description` or
            `dtype`, and header fields, which take precedence over the
            derived values and `like`. A header field that holds a field
            of the metadata, such as `descrip`, is refused.

        Returns
        -------
        Self
            The consolidated copy, or this field with `inplace=True`.

        Raises
        ------
        TypeError
            If an option names a header field that holds a field of the
            metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        """
        fields, overrides = split_options(options)
        name = type(self).__name__
        report = ConversionReport(source=name, target=name)
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, NiftiMetadata, fields, report
        )
        if (
            self._data is None
            and self._transformations is None
            and like is None
            and not overrides
            and not requested
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            apply_loss_policy(report, stacklevel=2)
            return self if inplace else replace(self)
        if "dtype" in requested:
            overrides["dtype"] = requested["dtype"]
        # The encoders read the record of the metadata, so they encode
        # over a copy that holds the consolidated metadata.
        source = (
            self
            if metadata is self.metadata
            else replace(self, metadata=metadata)
        )
        image = keep_given_fields(source._encoded(like, **overrides), fields)
        raw, record = stored_parts(image)
        metadata = consolidated_metadata(self, record)
        overruled_fields(requested, metadata, report)
        apply_loss_policy(report, stacklevel=2)
        if not inplace:
            return replace(
                self,
                rawdata=raw,
                metadata=metadata,
                data=None,
                transformations=None,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.data = None
        self.transformations = None
        return self

    def _encoded(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """Encode the field over a copy of its record.

        Each registered type encodes its chain as ITK writes it.
        """
        raise NotImplementedError

    def _xform_codes(self) -> tx.Tuple[int, int]:
        """
        Return the sform and qform codes to store.

        The codes come from the header when either is set, and each falls back
        on the other. Otherwise both are `NIFTI_XFORM_SCANNER_ANAT`.
        """
        header = header_of(self)
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
        Return the NIfTI image of the stored vectors, or `None`.

        A field whose chain is decoded from its file keeps its stored
        vectors, assigned or read, under a copy of the header of its
        metadata, so that a `DISPVECT` file stays one. `None` is returned
        when the field has no metadata or no vectors, or when its chain
        was assigned.
        """
        if self._data is not None:
            stored = itk_field_to_disk(self._data)
        else:
            stored = self.rawdata
        if self.metadata is None or stored is None:
            return None
        if self._transformations is not None:
            return None
        image = record_image(stored, self._record(), overrides=overrides)
        apply_like(image, like)
        apply_overrides(image, overrides)
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
        xp = array_backend(vectors)
        vectors = xp.asarray(vectors)
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
        vox2ras = _RAS_LPS @ embed_nifti_affine(vox2lps)
        vectors = itk_field_to_disk(vectors)
        image = record_image(vectors, self._record(), vox2ras, overrides)
        image.header.set_intent(NIFTI_INTENT_VECTOR, name=self.INTENT_NAME)
        scode, qcode = self._xform_codes()
        image.header.set_sform(vox2ras, code=scode)
        image.header.set_qform(vox2ras, code=qcode)
        apply_like(image, like)
        apply_overrides(image, overrides)
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

    HINTS = DISPLACEMENT_HINTS

    INTENT_NAME: tx.ClassVar[str] = NIFTI_INTENT_NAME_ITK_DISPLACEMENT

    @classmethod
    def score_header(cls, header: NiftiObject) -> float:
        """
        Score a header as an ITK displacement field.

        A `VECTOR` image in the ITK layout scores `Confidence.CERTAIN`,
        deliberately equal to the RAS reader, since neither frame may outrank
        the other. Other intent codes are left to a hint. The intent name
        `"Mapping"`, which SPM12 and the brainhops coordinates writer use but
        ITK never writes, is not claimed, and neither is the NiftyReg name
        `"NREG_TRANS"`.
        """
        if _itk_ndim(nifti_shape(header)) is None:
            return Confidence.NO
        if nifti_intent(header) != NIFTI_INTENT_VECTOR:
            return Confidence.NO
        name = nifti_intent_name(header)
        if name == NIFTI_INTENT_NAME_ITK_DISPLACEMENT:
            # Only this reader claims the name that brainhops writes on it.
            return Confidence.CERTAIN
        if name in (
            NIFTI_INTENT_NAME_MAPPING,
            NIFTI_INTENT_NAME_NIFTYREG,
            NIFTI_INTENT_NAME_ITK_COORDINATES,
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
        xp = array_backend(vectors)
        lps2vox = np.linalg.inv(vox2lps[:, :ndim])
        rotate = xp.asarray(lps2vox, dtype=vectors.dtype)
        field = xp.matmul(rotate, vectors[..., None])[..., 0]
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

    def _encoded(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Encode the field as ITK would write it.

        The displacements are rotated back to LPS millimetres and encoded
        unconverted under the `VECTOR` intent. The voxel-to-LPS affine becomes
        the voxel-to-RAS sform and qform. A field read from a file keeps its
        stored vectors. Non-encoding header fields are copied from `like`, and
        `overrides` are applied last.

        Parameters
        ----------
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        **overrides : Any
            Header fields that take precedence over the derived values and
            `like`.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.
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
        xp = array_backend(field)
        field = xp.asarray(field)
        rotate = xp.asarray(vox2lps[:ndim, :ndim], dtype=field.dtype)
        vectors = xp.matmul(rotate, field[..., None])[..., 0]
        return self._write_vectors(vectors, vox2lps, like, **overrides)


@register_format
class ItkNiftiCoordinateField(ItkNiftiField):
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

    HINTS = COORDINATE_HINTS

    INTENT_NAME: tx.ClassVar[str] = NIFTI_INTENT_NAME_ITK_COORDINATES

    PRIORITY: tx.ClassVar[int] = -1
    """
    Priority below the displacement reader.

    Under `hint="itk"`, a file in the ITK layout without the `VECTOR` intent is
    claimed by neither reader, and the tie is resolved in favour of
    displacements, which is what ITK and ANTs write.
    """

    @classmethod
    def score_header(cls, header: NiftiObject) -> float:
        """
        Claim only the files that brainhops writes as such fields.

        ITK and ANTs never write a field of coordinates, so a file is read
        as one only with a hint, or when it is a `VECTOR` image in the ITK
        layout that carries the intent name `"ITKCoordinates"`, which
        brainhops writes on it.
        """
        if _itk_ndim(nifti_shape(header)) is None:
            return Confidence.NO
        if nifti_intent(header) != NIFTI_INTENT_VECTOR:
            return Confidence.NO
        if nifti_intent_name(header) == NIFTI_INTENT_NAME_ITK_COORDINATES:
            return Confidence.CERTAIN
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
            LPSCoordinateField(
                field=self._lps_vectors(ndim), input=voxel, output=world
            ),
        )

    @property
    def coordinates(self) -> tx.Optional[_xforms.Transformation]:
        """The field of LPS coordinates, on the grid of the field."""
        return self.transformations[1]

    # --- writing ------------------------------------------------------

    def _encoded(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Encode the coordinate field.

        The coordinates are encoded unconverted under the `VECTOR` intent, and
        the inverse of the `lps2voxel` affine becomes the voxel-to-RAS sform
        and qform. A field read from a file keeps its stored coordinates.
        Non-encoding header fields are copied from `like`, and `overrides` are
        applied last.

        Parameters
        ----------
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        **overrides : Any
            Header fields that take precedence over the derived values and
            `like`.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.
        """
        image = self._read_back(like, **overrides)
        if image is not None:
            return image
        chain = tuple(self.transformations or ())
        if len(chain) != 2 or not isinstance(
            chain[1], _xforms.CoordinateField
        ):
            raise WriterError(
                "An ITK coordinate field is written from a chain of two "
                "transformations: LPS to voxel, and a coordinate field."
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
