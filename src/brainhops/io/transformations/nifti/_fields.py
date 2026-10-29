"""
RAS coordinate fields and RAS displacement fields stored in NIfTI files.

The intent codes and the reasons for them are described in
[`brainhops.io.transformations.nifti`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core._internal_api import InvalidatorInAttribute, smartproperty
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import apply_loss_policy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw
from brainhops.io.common.nifti._internal_api import (
    NIFTI_INTENT_DISPVECT,
    NIFTI_INTENT_NAME_ITK_COORDINATES,
    NIFTI_INTENT_NAME_ITK_DISPLACEMENT,
    NIFTI_INTENT_NAME_NIFTYREG,
    NIFTI_INTENT_VECTOR,
    NiftiObject,
    apply_like,
    apply_overrides,
    header_of,
    keep_given_fields,
    nifti_intent,
    nifti_intent_name,
    nifti_shape,
    record_image,
    set_coordinates_intent,
    split_options,
    stored_parts,
)
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    overruled_fields,
    requested_fields,
)
from brainhops.io.transformations.base.fields import (
    COORDINATE_HINTS,
    DISPLACEMENT_HINTS,
    RASCoordinateField,
    ras_displacement_chain,
    split_ras_displacement_chain,
)

# this format
from ._base import NiftiBasedTransformation, consolidated_metadata
from ._views import check_single_field, field_to_disk, field_to_model

_NDIM = 3
"""Spatial dimension supported by the displacement reader."""


_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


@register_format
class NiftiRASCoordinateField(NiftiBasedTransformation, RASCoordinateField):
    """Field of RAS coordinates stored in a NIfTI file."""

    HINTS = COORDINATE_HINTS

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata")
    """The metadata of the file, which holds its header, or `None`.

    A field built from data has no metadata. Its coordinates are its
    data, so assigning other metadata keeps them.
    """

    @classmethod
    def score_header(cls, header: NiftiObject) -> float:
        """
        Score a header as a field of RAS coordinates.

        A `VECTOR` image (1007), the intent that this reader and SPM write,
        scores `Confidence.CERTAIN`. A `DISPVECT` image (1006) is left to
        [`NiftiRASDisplacementField`][], and a file named `"NREG_TRANS"` to
        [`brainhops.io.transformations.niftyreg`][]. A file named
        `"ITKDisplacement"` or `"ITKCoordinates"`, which brainhops writes
        on ITK fields, is left to the ITK readers. The ITK displacement
        reader ties on any other `VECTOR` image in the ITK layout unless its
        intent name is `"Mapping"`, so a hint decides (see
        [`brainhops.io.transformations.itk.nifti`][]). Any other file whose
        trailing axis has length 3 scores `Confidence.MAYBE`.
        """
        intent = nifti_intent(header)
        if nifti_intent_name(header) in (
            NIFTI_INTENT_NAME_NIFTYREG,
            NIFTI_INTENT_NAME_ITK_DISPLACEMENT,
            NIFTI_INTENT_NAME_ITK_COORDINATES,
        ):
            # Only the NiftyReg readers decode what intent_p1 says the file
            # holds, and only the ITK readers read a field that brainhops
            # wrote as one.
            return Confidence.NO
        if intent == NIFTI_INTENT_VECTOR:
            return Confidence.CERTAIN
        if intent == NIFTI_INTENT_DISPVECT:
            return Confidence.NO
        # The intent is often unset, so fall back on the shape. A plain image
        # scores zero rather than compete with the affine readers.
        shape = nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.MAYBE
        return Confidence.NO

    @smartproperty(
        invalidates=_FORGET_VIEWS,
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The field as an (X, Y, Z, 3) array, which its `field` view reads.

        An array that is assigned is returned as it is, and stays pending
        until [`consolidate`][] or a write encodes it. Otherwise, the data
        is decoded from `rawdata`. NIfTI stores a vector field with
        shape (X, Y, Z, 1, 3), and the data is decoded without the
        singleton axis, since the array would otherwise be sampled as a
        4-D grid of vectors. A file whose fourth axis is longer than one
        holds a field per volume, which is not supported yet, so reading
        its data raises `ParserContentError`.
        """
        if self.rawdata is None:
            return None
        return field_to_model(self.rawdata)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        **options: tx.Any,
    ) -> tx.Self:
        """Encode the pending field into `rawdata` and a new header.

        The header carries the `VECTOR` intent (1007) with the SPM intent
        name `"Mapping"`, which marks a coordinate field, rather than
        `DISPVECT` (1006), which ITK and brainhops read as displacements.
        The field holds no grid of its own, so a copy of the header of the
        metadata, with its voxel-to-RAS affine, is the base of the new
        header, and a field without metadata gets the identity. NIfTI
        stores sampled coordinates, so a field that stores spline
        coefficients is converted to its values, and the copy then stores
        values. The array is kept as it is, so a proxy that was read is
        copied as the file stored it, and a `cupy` or `dask` array is not
        converted to `numpy`. Non-encoding header fields are copied from
        `like`, and `overrides` are applied last.

        A field that has a header and nothing pending, consolidated
        without options, is left as it is.

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
            If the field has no coordinates, or if its array holds a
            field for each of several volumes.
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
            and StoreEnum(self.store) is StoreEnum.values
            and like is None
            and not overrides
            and not requested
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            check_single_field(self.rawdata, WriterError)
            apply_loss_policy(report, stacklevel=2)
            return self if inplace else replace(self)
        if "dtype" in requested:
            overrides["dtype"] = requested["dtype"]
        base = None if metadata is None else metadata.raw
        raw, record = stored_parts(
            _coordinates_encoded(self, like, overrides, base, fields)
        )
        metadata = consolidated_metadata(self, record)
        overruled_fields(requested, metadata, report)
        apply_loss_policy(report, stacklevel=2)
        if not inplace:
            return replace(
                self,
                rawdata=raw,
                metadata=metadata,
                data=None,
                store=StoreEnum.values,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.data = None
        self.store = StoreEnum.values
        return self


@register_format
class NiftiRASDisplacementField(
    NiftiBasedTransformation, _xforms.DerivedSequence
):
    """
    Field of RAS displacements stored in a NIfTI file.

    The file holds a `DISPVECT` field (1006), in which each voxel holds the
    displacement, in RAS millimetres, of the point at its centre. The field
    maps RAS to RAS as `x -> x + u(x)`, which is how ITK 5.4 and later read and
    write `DISPVECT`. A displacement field adds its values in the units of its
    own grid, so this field is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of three named slots:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    The displacements are interpolated linearly and extrapolated with the
    nearest value outside the grid, as by the default
    `DisplacementFieldTransform` of ITK.

    The file may hold a stationary velocity instead, whose flow at time
    one is the map: the standard has no code for one, so it is said with
    the `log` option (`warp.nii.gz|displacements|log:true`, or its alias
    `warp.nii.gz|svf`; in Python, `load(path, log=True)`). The
    `displacement` slot is then a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.\
StationaryVelocityField], integrated with `steps` squaring steps
    (`|svf|steps:6`). The field is written in the encoding the options
    say: a velocity when `log` is set, and otherwise the displacement,
    which a velocity is integrated into.

    Another transformation is converted to this format by its converters
    ([`brainhops.io.transformations.nifti._converters`][]), exactly or not
    at all. Its chain is carried over, rather than re-read from a NIfTI
    header that comes with it and says something else (a NiftyReg file
    holds positions, say). Its encoding is not: `log` and `steps` are this
    format's options, so a velocity converted here is written as its
    displacement unless `log=True` is given -- to the conversion, or to
    `save`.
    """

    HINTS = DISPLACEMENT_HINTS

    degree: tx.ClassVar[int] = 1
    """Spline degree of the interpolation."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """Boundary condition outside the field of view."""

    log: bool = False
    """
    Whether the file holds the stationary velocity whose flow is
    the map, rather than its displacement.
    """

    steps: tx.Optional[int] = None
    """
    The number of squaring steps that integrate a velocity (`log` only).
    If `None`, the default of [`StationaryVelocityField`][].
    """

    _data: NoRepr[tx.Optional[ArrayProtocol]] = None
    """
    The vectors given to the field, or `None` to decode them from `rawdata`.

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
        they are decoded from `rawdata`: NIfTI stores the field with
        shape (X, Y, Z, 1, 3), and the singleton axis is dropped. A file
        whose fourth axis is longer than one holds a field per volume,
        which is not supported yet, so reading its data raises
        `ParserContentError`. The vectors are the RAS displacements of
        the file, or its velocity with `log`. Assigning the data drops
        the chain decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return field_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the chain decoded from the previous
    header.
    """

    # --- reading ------------------------------------------------------
    # The readers hand their keyword arguments to nibabel, so the encoding
    # options are popped first and set on the field read.

    @classmethod
    def from_filename(cls, filename: tx.Any, **kwargs) -> tx.Self:
        """Read a displacement field from the path of a NIfTI file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The encoding options `log` and `steps`, and the options of the
            nibabel readers.

        Returns
        -------
        NiftiRASDisplacementField
            The field, whose array is read lazily.

        Raises
        ------
        TypeError
            If `steps` is given without `log`.
        """
        encoding = _pop_encoding(kwargs)
        obj = super().from_filename(filename, **kwargs)
        return _with_encoding(obj, encoding)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read a displacement field from an open NIfTI stream.

        Parameters
        ----------
        fileobj : BinaryIO
            The stream, which must stay open while the data may be read.
        **kwargs : Any
            The encoding options `log` and `steps`, and the options of the
            nibabel readers.

        Returns
        -------
        NiftiRASDisplacementField
            The field, whose array is read lazily.

        Raises
        ------
        TypeError
            If `steps` is given without `log`.
        """
        encoding = _pop_encoding(kwargs)
        obj = super().from_fileobj(fileobj, **kwargs)
        return _with_encoding(obj, encoding)

    @classmethod
    def score_header(cls, header: NiftiObject) -> float:
        """
        Score a header as a field of RAS displacements.

        Only a `DISPVECT` image (1006) is claimed, with `Confidence.CERTAIN`.
        Other images may hold coordinates and are left to
        [`NiftiRASCoordinateField`][].
        """
        if nifti_intent(header) == NIFTI_INTENT_DISPVECT:
            return Confidence.CERTAIN
        return Confidence.NO

    # --- endpoints ----------------------------------------------------
    # The endpoints are declared rather than read off the chain, which would
    # decode the data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """
        The anatomical space that the field maps from, in RAS millimetres.
        """
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space that the field maps to, in RAS millimetres."""
        return _systems.RASmm()

    # --- decoding -----------------------------------------------------

    def _vox2ras(self) -> np.ndarray:
        """Return the (4, 4) voxel-to-RAS affine of the grid.

        The affine is the best affine of the header, or the identity for a
        field without metadata.
        """
        header = header_of(self)
        if header is None:
            return np.eye(4)
        return np.asarray(header.get_best_affine(), dtype=np.float64)

    def _ras_vectors(self) -> ArrayProtocol:
        """
        Return the stored displacements, in RAS millimetres.

        A field whose array is not a grid of vectors with three components
        is refused.
        """
        data = self.data
        if data is None:
            raise ParserContentError("This field has no data to read.")
        if len(data.shape) != 4:
            shape = tuple(int(d) for d in data.shape)
            raise ParserContentError(
                f"A NIfTI displacement field is stored as a (X, Y, Z, 1, 3) "
                f"array, and its vectors have the shape {shape}."
            )
        if int(data.shape[-1]) != _NDIM:
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

        The chain is built lazily from the header and the data, and then
        cached. Assigning a chain overrides the derived one, which is how
        fields that do not come from a file are built. Since the slots name
        fixed positions, a copy with other slots is made with `replace` rather
        than by editing the chain in place.
        """
        return ras_displacement_chain(
            self._ras_vectors(),
            self._vox2ras(),
            degree=self.degree,
            bound=self.bound,
            log=self.log,
            steps=self.steps,
        )

    @property
    def ras2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from RAS world coordinates to the voxels of the field."""
        return self.transformations[0]

    @property
    def displacement(self) -> tx.Optional[_xforms.Transformation]:
        """
        The displacement field in voxel units, or the velocity with `log`.
        """
        return self.transformations[1]

    @property
    def voxel2ras(self) -> tx.Optional[_xforms.Transformation]:
        """
        The affine from the voxels of the field back to RAS world coordinates.
        """
        return self.transformations[2]

    # --- writing ------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        log: tx.Optional[bool] = None,
        **options: tx.Any,
    ) -> tx.Self:
        """
        Encode the pending vectors or chain into `rawdata` and a new
        header.

        The displacements are encoded in RAS millimetres under the
        `DISPVECT` intent (1006), with shape (X, Y, Z, 1, 3). With `log`,
        which defaults to the option of the field, the velocity is encoded
        instead, and otherwise a velocity is integrated first. The copy
        then has the `log` option that it was encoded with.

        A field whose chain is decoded from its stored vectors, in the
        encoding that it is written in, keeps the vectors under a copy of
        the header of its metadata, whose geometry is kept. A field whose
        chain was assigned is encoded from the chain over that header,
        whose geometry is replaced. Non-encoding header fields are copied
        from `like`, and `overrides` are applied last. A field that has a
        header and nothing pending, consolidated without options, is left
        as it is.

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
        log : bool, optional
            Whether to encode the velocity rather than the displacements.
            It defaults to the `log` option of the field.
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
        fields, overrides = split_options(options)
        name = type(self).__name__
        report = ConversionReport(source=name, target=name)
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, NiftiMetadata, fields, report
        )
        log = self.log if log is None else log
        if (
            self._data is None
            and self._transformations is None
            and log == self.log
            and like is None
            and not overrides
            and not requested
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            check_single_field(self.rawdata, WriterError)
            apply_loss_policy(report, stacklevel=2)
            return self if inplace else replace(self)
        if "dtype" in requested:
            overrides["dtype"] = requested["dtype"]
        base = None if metadata is None else metadata.raw
        raw, record = stored_parts(
            _displacement_encoded(self, like, log, overrides, base, fields)
        )
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
                log=log,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.data = None
        self.transformations = None
        self.log = log
        return self

    def to_nibabel(
        self,
        like: tx.Any = None,
        log: tx.Optional[bool] = None,
        **overrides,
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image of the displacement field.

        The field is consolidated with the options, as by
        [`consolidate`][], into a copy, and the nibabel image holds the
        `rawdata` array and the header of that copy. The field itself is
        not changed.

        Parameters
        ----------
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        log : bool, optional
            Whether to write the velocity rather than the displacements.
            It defaults to the `log` option of the field.
        **overrides : Any
            Header fields that take precedence over the derived values and
            `like`.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.
        """
        return super().to_nibabel(like=like, log=log, **overrides)


def _pop_encoding(kwargs: tx.Dict[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    return {
        name: kwargs.pop(name) for name in ("log", "steps") if name in kwargs
    }


def _with_encoding(
    obj: NiftiRASDisplacementField, encoding: tx.Dict[str, tx.Any]
) -> NiftiRASDisplacementField:
    # steps only makes sense for a velocity, so it requires log.
    if encoding.get("steps") is not None and not encoding.get("log"):
        raise TypeError(
            "steps is the number of squaring steps of a velocity, so it is "
            "given with log=true (or the hint svf)."
        )
    return replace(obj, **encoding) if encoding else obj


def _coordinates_encoded(
    field: NiftiRASCoordinateField,
    like: tx.Any,
    overrides: tx.Mapping[str, tx.Any],
    base: tx.Optional[NiftiRaw],
    fields: tx.Mapping[str, tx.Any],
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Encode a field of RAS coordinates over a copy of a record.

    The record, `base`, is that of the consolidated metadata.

    Raises
    ------
    WriterError
        If the field has no coordinates, or if its array holds a field
        for each of several volumes.
    """
    # NIfTI stores sampled coordinates.
    if StoreEnum(field.store) is not StoreEnum.values:
        values = field.to(store="values").data
        raw = None if values is None else field_to_disk(values)
    elif field._data is not None:
        raw = field_to_disk(field._data)
    else:
        raw = field.rawdata
    if raw is None:
        raise WriterError(
            "This field has no coordinates, so there is nothing to write."
        )
    check_single_field(raw, WriterError)
    record = None if base is None else base.copy()
    image = record_image(raw, record, overrides=overrides)
    set_coordinates_intent(image.header)
    apply_like(image, like)
    keep_given_fields(image, fields)
    apply_overrides(image, overrides)
    return image


def _displacement_encoded(
    field: NiftiRASDisplacementField,
    like: tx.Any,
    log: bool,
    overrides: tx.Mapping[str, tx.Any],
    base: tx.Optional[NiftiRaw],
    fields: tx.Mapping[str, tx.Any],
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Encode a field of RAS displacements over a copy of a record.

    The record, `base`, is that of the consolidated metadata.

    The stored vectors, assigned or read, are kept when the chain was not
    assigned and the encoding does not change. The chain is encoded
    otherwise.
    """
    what = "A NIfTI displacement field"
    if field._data is not None:
        stored = field_to_disk(field._data)
    else:
        stored = field.rawdata
    if (
        stored is not None
        and field._transformations is None
        and log == field.log
    ):
        check_single_field(stored, WriterError)
        record = None if base is None else base.copy()
        image = record_image(stored, record, overrides=overrides)
        if nifti_intent(image.header) != NIFTI_INTENT_DISPVECT:
            image.header.set_intent(NIFTI_INTENT_DISPVECT)
    else:
        vox2ras, vectors = split_ras_displacement_chain(
            field.transformations, what, ndim=_NDIM, log=log
        )
        vectors = field_to_disk(vectors)
        record = None if base is None else base.copy()
        image = record_image(vectors, record, vox2ras, overrides)
        image.header.set_intent(NIFTI_INTENT_DISPVECT)
    apply_like(image, like)
    keep_given_fields(image, fields)
    apply_overrides(image, overrides)
    return image
