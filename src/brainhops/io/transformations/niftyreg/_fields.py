"""
NiftyReg fields and control-point grids stored in NIfTI files.

The encoding, the NiftyReg sources and the limits of the support are described
in [`brainhops.io.transformations.niftyreg`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, KwOnly, NoRepr, replace

# core
from brainhops._core.backends import array_backend
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import (
    apply_loss_policy,
)
from brainhops.errors import (
    ConversionError,
    ParserContentError,
    WriterError,
)

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence
from brainhops.io.common.nifti import NiftiMetadata
from brainhops.io.common.nifti._constants import (
    _NIFTI_INTENT_NAME_NIFTYREG,
    _NIFTI_INTENT_VECTOR,
)
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _header,
    _keep_given_fields,
    _nifti_intent,
    _nifti_intent_name,
    _nifti_shape,
    _NiftiObject,
    _record_image,
    _split_options,
    _stored_parts,
)
from brainhops.io.metadata._base import (
    metadata_to_write,
    overruled_fields,
    requested_fields,
)
from brainhops.io.transformations.base._converters import (
    relabel,
    relabelling,
)
from brainhops.io.transformations.base.affines import RASToRAS
from brainhops.io.transformations.base.fields import (
    COORDINATE_HINTS,
    DISPLACEMENT_HINTS,
    homogeneous_matrix,
    ras_displacement_chain,
    split_ras_displacement_chain,
    voxel_grid_coordinates,
)
from brainhops.io.transformations.nifti import NiftiBasedTransformation
from brainhops.io.transformations.nifti._base import _consolidated_metadata
from brainhops.io.transformations.nifti._views import (
    _check_single_field,
    _field_to_disk,
    _field_to_model,
)

# this format
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


def _check_shape(header: _NiftiObject) -> None:
    """Refuse a header whose array is not a 3-D NiftyReg field.

    A NiftyReg field is stored with shape (X, Y, Z, 1, 3). Any other shape,
    in particular that of a 2-D field, is refused.

    Raises
    ------
    ParserContentError
        If the shape is not that of a 3-D NiftyReg field.
    """
    shape = _nifti_shape(header)
    if shape is None or len(shape) != 5 or shape[3] != 1 or shape[4] != _NDIM:
        raise ParserContentError(
            f"A three-dimensional NiftyReg field is stored as a "
            f"(X, Y, Z, 1, 3) array, not as an array of shape {shape}. "
            f"Two-dimensional NiftyReg fields are not supported."
        )


def _extension(matrix: np.ndarray) -> nb.nifti1.Nifti1Extension:
    """
    Build an extension that holds a `mat44`, padded as NiftyReg writes it.
    """
    content = np.asarray(matrix, dtype="<f4").reshape(16).tobytes()
    return nb.nifti1.Nifti1Extension(_NIFTI_ECODE_IGNORE, content + bytes(8))


def _field_index(chain: tx.Sequence[_xforms.Transformation]) -> int:
    """
    Return the position of the field in the chain of a NiftyReg field.

    The field is the middle of its three slots, which an affine may
    precede and another may follow. A chain without a field, which is not
    a chain that a NiftyReg field holds, is read as the three slots alone.
    """
    for i, element in enumerate(chain):
        if isinstance(
            element, (_xforms.DisplacementField, _xforms.CoordinateField)
        ):
            return i
    return 1


def _slot_affines(
    chain: tx.Sequence[_xforms.Transformation], what: str
) -> tx.Tuple[
    tx.Optional[np.ndarray],
    tx.Tuple[_xforms.Transformation, ...],
    tx.Optional[np.ndarray],
]:
    """
    Take the affines that NiftyReg keeps in its extensions out of a chain.

    Returns
    -------
    before : array or None
        The (4, 4) affine applied before the field, or `None`.
    slots : tuple of Transformation
        The three slots of the field.
    after : array or None
        The (4, 4) affine applied after the field, or `None`.
    """
    chain = tuple(chain)
    i = _field_index(chain)
    before = after = None
    if i > 1:
        before = homogeneous_matrix(chain[0], what, _NDIM)
    if len(chain) > i + 2:
        after = homogeneous_matrix(chain[-1], what, _NDIM)
    return before, chain[i - 1 : i + 2], after


def _extensions_of(
    before: tx.Optional[np.ndarray], after: tx.Optional[np.ndarray]
) -> tx.List[np.ndarray]:
    """
    Return the affines to write in the extensions, in the order of NiftyReg.

    NiftyReg applies the affine of the first extension before the field
    and the affine of the second one after it, so an affine applied after
    the field alone is written after an identity.
    """
    if after is not None:
        return [np.eye(4) if before is None else before, after]
    return [] if before is None else [before]


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
    def score_header(cls, header: _NiftiObject) -> float:
        """
        Score `Confidence.CERTAIN` for a NiftyReg file of one of the `TYPES`.
        """
        if _niftyreg_type(header) in cls.TYPES:
            return Confidence.CERTAIN
        return Confidence.NO

    @property
    def niftyreg_type(self) -> tx.Optional[int]:
        """The NiftyReg transformation type that the header records."""
        header = _header(self)
        return None if header is None else _niftyreg_type(header)

    @property
    def extension_affines(self) -> tx.List[np.ndarray]:
        """The (4, 4) affines stored in the extensions of the header."""
        header = _header(self)
        return [] if header is None else _extension_affines(header)

    def _vox2world(self) -> np.ndarray:
        """Return the (4, 4) voxel-to-world affine of the grid of the file.

        A field without metadata has no grid of its own, and its grid is
        placed by the identity.
        """
        header = _header(self)
        if header is None:
            return np.eye(4)
        return _vox2world(header)


class NiftyRegSequence(NiftyRegField, _xforms.DerivedSequence):
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

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        # A field read from a file is refused from its header alone, before
        # its data is read.
        header = _header(self)
        if header is not None:
            _check_shape(header)

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
        """The stored vectors as an (X, Y, Z, 3) array.

        Vectors that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them. Otherwise
        they are decoded from `rawdata`: NiftyReg stores a field with
        shape (X, Y, Z, 1, 3), and the singleton axis is dropped. A file
        whose fourth axis is longer than one holds a field per volume,
        which is not supported yet, so reading its data raises
        `ParserContentError`. The vectors are positions or
        displacements, as the type of the file says. Assigning the data
        drops the chain decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return _field_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the chain decoded from the previous
    header.
    """

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
        if len(data.shape) != _NDIM + 1 or int(data.shape[-1]) != _NDIM:
            shape = tuple(int(d) for d in data.shape)
            raise ParserContentError(
                f"A three-dimensional NiftyReg field is stored as a "
                f"(X, Y, Z, 1, 3) array, and its vectors have the shape "
                f"{shape}."
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
        # The affines of the extensions may come before and after the three
        # slots of the field.
        return chain[_field_index(chain) - 1 + index]

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

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        **options: tx.Any,
    ) -> tx.Self:
        """Encode the pending vectors or chain into `rawdata` and a header.

        A field whose chain is decoded from a NiftyReg file of its own type
        keeps its stored vectors, assigned or read, under a copy of the
        header of its metadata, extensions included. Otherwise, each type
        encodes its chain as NiftyReg would write it. Non-encoding header
        fields are copied from `like`, and `overrides` are applied last. A
        field that has a header and nothing pending, consolidated without
        options, is left as it is.

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
        WriterError
            If the stored array holds a field for each of several
            volumes.
        """
        fields, overrides = _split_options(options)
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
            _check_single_field(self.rawdata, WriterError)
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
        image = _keep_given_fields(source._encoded(like, **overrides), fields)
        raw, record = _stored_parts(image)
        metadata = _consolidated_metadata(self, record)
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

        Each registered type encodes its chain as NiftyReg writes it.
        """
        raise NotImplementedError

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

    def _read_back(
        self, like: tx.Any = None, **overrides
    ) -> tx.Optional[tx.Union[nb.Nifti1Image, nb.Nifti2Image]]:
        """
        Return the NIfTI image of the stored vectors, or `None`.

        A field whose chain is decoded from a NiftyReg file of its own type
        keeps its stored vectors, assigned or read, under a copy of the
        header of its metadata, extensions included. `None` is returned
        when the field was not read from such a file, or when its chain
        was assigned.
        """
        if self._data is not None:
            stored = _field_to_disk(self._data)
        else:
            stored = self.rawdata
        if stored is None or self.niftyreg_type not in type(self).TYPES:
            return None
        if self._transformations is not None:
            return None
        _check_single_field(stored, WriterError)
        image = _record_image(stored, self._record(), overrides=overrides)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image

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
        `"NREG_TRANS"`, with its type in `intent_p1`. A copy of the header of
        the metadata is the base of the new header, and its fields that
        describe the layout and the geometry are encoded again. The given
        affines replace the extensions of that header, because NiftyReg
        reads its affines from the first extensions.

        The vectors are stored in single precision, unless the `dtype` or
        `datatype` override asks for another type. NiftyReg computes its
        deformation fields in single precision, and refuses a grid of
        control points whose type differs from that of the field.
        """
        vectors = _field_to_disk(vectors)
        image = _record_image(vectors, self._record(), vox2world, overrides)
        header = image.header
        header.set_intent(
            _NIFTI_INTENT_VECTOR, name=_NIFTI_INTENT_NAME_NIFTYREG
        )
        header["intent_p1"] = kind
        header.set_data_dtype(np.float32)
        del header.extensions[:]
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

    HINTS = DISPLACEMENT_HINTS
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset({DISP_FIELD})

    PRIORITY: tx.ClassVar[int] = -1
    """Priority below the NIfTI field of RAS displacements, which `io.save`
    writes a field to a `.nii.gz` name as, unless it already is of this
    format. A NiftyReg reader claims only the files that NiftyReg names,
    so the priority does not change what is read."""
    _WHAT: tx.ClassVar[str] = "A NiftyReg displacement field"

    def _encoded(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """Encode the field as NiftyReg would write it.

        A field read from a file keeps its stored vectors. Otherwise, the
        displacements of the chain are encoded.

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

    HINTS = ("deformation", "def", *COORDINATE_HINTS)
    TYPES: tx.ClassVar[tx.FrozenSet[int]] = frozenset({DEF_FIELD})

    PRIORITY: tx.ClassVar[int] = -1
    """Priority below the NIfTI field of RAS displacements, which `io.save`
    writes a field to a `.nii.gz` name as, unless it already is of this
    format. A NiftyReg reader claims only the files that NiftyReg names,
    so the priority does not change what is read."""
    _WHAT: tx.ClassVar[str] = "A NiftyReg deformation field"

    def _displacements(self, vox2world: np.ndarray) -> ArrayProtocol:
        positions = self._stored_vectors()
        xp = array_backend(positions)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in positions.shape[:_NDIM]), vox2world, xp
        )
        return positions - xp.asarray(grid, dtype=positions.dtype)

    def _encoded(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Encode the field, with displacements as positions.

        A field read from a file keeps its stored vectors.

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
        vox2world, vectors = self._split(self.transformations)
        xp = array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, xp
        )
        positions = vectors + xp.asarray(grid, dtype=vectors.dtype)
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

    The header may carry affines in its extensions, as after a symmetric
    registration. NiftyReg applies the affine of the first extension before
    the spline and composes the affine of the second extension after it.
    The chain then starts with a [`RASToRAS`][] in the `affine` slot, and
    ends with another one in the `affine_after` slot when the header has a
    second extension.
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
        kind = self.niftyreg_type
        if kind in _GRID_DEGREE:
            return _GRID_DEGREE[kind]
        chain = self._transformations
        if chain:
            return int(chain[-2].degree)
        return 3

    @property
    def store(self) -> StoreEnum:
        """Whether the grid holds spline coefficients, as above degree 1."""
        return [StoreEnum.values, StoreEnum.coefficients][self.degree > 1]

    def _displacements(self, vox2world: np.ndarray) -> ArrayProtocol:
        positions = self._stored_vectors()
        xp = array_backend(positions)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in positions.shape[:_NDIM]), vox2world, xp
        )
        return positions - xp.asarray(grid, dtype=positions.dtype)

    def _field_chain(self) -> tx.Tuple[_xforms.Transformation, ...]:
        chain = super()._field_chain()
        affines = self.extension_affines
        if affines:
            chain = (RASToRAS(matrix=affines[0][:-1]), *chain)
        if len(affines) > 1:
            chain = (*chain, RASToRAS(matrix=affines[1][:-1]))
        return chain

    @property
    def affine(self) -> tx.Optional[_xforms.Transformation]:
        """The affine applied before the spline, or `None`."""
        chain = self.transformations
        return chain[0] if _field_index(chain) > 1 else None

    @property
    def affine_after(self) -> tx.Optional[_xforms.Transformation]:
        """The affine applied after the spline, or `None`."""
        chain = self.transformations
        return chain[-1] if len(chain) > _field_index(chain) + 2 else None

    def _encoded(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Encode the grid, with coefficients as positions.

        A grid read from a file keeps its stored positions. Otherwise, the
        affines before and after the spline are encoded in the header
        extensions that NiftyReg reads.

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
        before, chain, after = _slot_affines(
            self.transformations or (), self._WHAT
        )
        extensions = _extensions_of(before, after)
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
            store=[StoreEnum.values, StoreEnum.coefficients][degree > 1],
            degree=degree,
            bound=self.bound,
        )
        xp = array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, xp
        )
        positions = vectors + xp.asarray(grid, dtype=vectors.dtype)
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

    NiftyReg composes the affine of the second extension of a velocity after
    its flow, and the chain then ends with a [`RASToRAS`][]. The affine of the
    first extension, which a symmetric registration writes, is not composed:
    NiftyReg subtracts its values from the velocity before integrating it and
    adds them back to the displacement after. That affine is not decoded
    unless it is the identity, so building the chain of such a velocity
    raises `NotImplementedError`. A velocity read from a file is written back
    as read.
    A velocity built from a chain is written with its steps in `intent_p2`, and
    a displacement field is refused, since no logarithm is computed. This class
    is not registered as a format.
    """

    log: tx.ClassVar[bool] = True

    @property
    def squaring_steps(self) -> tx.Optional[int]:
        """The stored squaring steps, negative for a backward field."""
        header = _header(self)
        if header is None:
            return None
        return int(round(float(header["intent_p2"])))

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

    def _field_chain(self) -> tx.Tuple[_xforms.Transformation, ...]:
        chain = super()._field_chain()
        affines = self.extension_affines
        if len(affines) > 1:
            chain = (*chain, RASToRAS(matrix=affines[1][:-1]))
        return chain

    def _displacements(self, vox2world: np.ndarray) -> ArrayProtocol:
        affines = self.extension_affines
        if affines and not np.allclose(affines[0], np.eye(4)):
            raise NotImplementedError(
                f"{type(self).__name__}: this NiftyReg velocity carries an "
                f"affine in its first header extension, whose values NiftyReg "
                f"subtracts before integrating it and adds back after; that "
                f"is not decoded. Integrate it with NiftyReg (`reg_transform "
                f"-ref <ref> -def <velocity> <out>.nii.gz`) and read the "
                f"deformation field it writes."
            )
        velocity = self._velocity_vectors(vox2world)
        if (self.squaring_steps or 0) < 0:
            # A backward field integrates the negated velocity.
            velocity = -velocity
        return velocity

    def _velocity_slots(
        self,
    ) -> tx.Tuple[tx.Tuple[_xforms.Transformation, ...], tx.List[np.ndarray]]:
        """
        Return the three slots of the velocity and the extensions to write.

        Raises
        ------
        ConversionError
            If an affine comes before the velocity, which NiftyReg does not
            compose.
        """
        before, chain, after = _slot_affines(
            self.transformations or (), self._WHAT
        )
        if before is not None and not np.allclose(before, np.eye(4)):
            raise ConversionError(
                f"{self._WHAT} composes an affine after its velocity, and "
                f"not before it, so this chain, which starts with an affine, "
                f"cannot be written."
            )
        return chain, _extensions_of(None, after)


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
        xp = array_backend(positions)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in positions.shape[:_NDIM]), vox2world, xp
        )
        return positions - xp.asarray(grid, dtype=positions.dtype)

    def _encoded(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Encode the velocity grid.

        A grid read from a file keeps its stored positions. Otherwise, the
        coefficients are turned into positions, the squaring steps are
        stored in `intent_p2`, and an affine after the velocity is stored
        in the second header extension.

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
        chain, extensions = self._velocity_slots()
        vox2world, vectors = self._split(chain)
        xp = array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, xp
        )
        positions = vectors + xp.asarray(grid, dtype=vectors.dtype)
        overrides.setdefault("intent_p2", _written_steps(chain))
        return self._write(
            positions,
            vox2world,
            SPLINE_VEL_GRID,
            like,
            extensions,
            **overrides,
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

    PRIORITY: tx.ClassVar[int] = -1
    """Priority below the NIfTI field of RAS displacements, which `io.save`
    writes a field to a `.nii.gz` name as, unless it already is of this
    format. A NiftyReg reader claims only the files that NiftyReg names,
    so the priority does not change what is read."""
    _WHAT: tx.ClassVar[str] = "A NiftyReg velocity field"

    def _velocity_vectors(self, vox2world: np.ndarray) -> ArrayProtocol:
        vectors = self._stored_vectors()
        if self.niftyreg_type == DISP_VEL_FIELD:
            return vectors
        xp = array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, xp
        )
        return vectors - xp.asarray(grid, dtype=vectors.dtype)

    def _encoded(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Encode the velocity field.

        A field read from a file keeps its stored vectors. Otherwise, it is
        encoded as positions, the type that NiftyReg integrates, with the
        squaring steps in `intent_p2` and an affine after the velocity in
        the second header extension.

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
        chain, extensions = self._velocity_slots()
        vox2world, vectors = self._split(chain)
        xp = array_backend(vectors)
        grid = voxel_grid_coordinates(
            tuple(int(s) for s in vectors.shape[:_NDIM]), vox2world, xp
        )
        positions = vectors + xp.asarray(grid, dtype=vectors.dtype)
        overrides.setdefault("intent_p2", _written_steps(chain))
        return self._write(
            positions,
            vox2world,
            DEF_VEL_FIELD,
            like,
            extensions,
            **overrides,
        )


# ----------------------------------------------------------------------
#   CONVERSIONS
# ----------------------------------------------------------------------


# NiftyReg stores a field on the grid of its file, in one of several
# encodings. The dense fields have exact converters (`_converters.py`), and
# no exact conversion into the control-point grids is written yet (#312). A
# chain is relabelled as a grid, which carries the chain over rather than
# letting this reader decode the header of another format.
converter(_xforms.Sequence, NiftyRegSequence)(relabelling)


@converter
def _(
    t: NiftyRegSequence, cls: tx.Type[NiftyRegSequence], **kwargs
) -> NiftyRegSequence:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type. Another variant
    # of the format holds the same chain in another encoding, so it is
    # relabelled, as any other transformation is.
    if not isinstance(t, cls):
        return relabel(t, cls, **kwargs)
    return smart_replace(t, cls, **kwargs)
