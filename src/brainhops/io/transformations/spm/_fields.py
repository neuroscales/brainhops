"""
SPM deformation fields, stored in NIfTI files prefixed with `y_` or `iy_`.
"""

# dependencies
import nibabel as nb
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import Inherit, KwOnly, NoRepr, replace

# internals
from brainhops._core.properties import smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import WriterError
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _keep_given_fields,
    _NiftiObject,
    _record_image,
    _set_coordinates_intent,
    _split_options,
    _stored_parts,
)
from brainhops.io.metadata._base import (
    metadata_to_write,
    overruled_fields,
    requested_fields,
)
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.fields import (
    RASCoordinateField,
    homogeneous_matrix,
)
from brainhops.io.transformations.nifti import (
    NiftiBasedTransformation,
    NiftiRASCoordinateField,
    NiftiRASToVoxel,
)
from brainhops.io.transformations.nifti._base import _consolidated_metadata
from brainhops.io.transformations.nifti._views import (
    _field_to_disk,
    _field_to_model,
)


@register_format
class SpmCoordinateField(NiftiBasedTransformation, _xforms.DerivedSequence):
    """
    Field of RAS coordinates written by SPM.

    The file names of these fields often start with `y_` or `iy_`. The field is
    an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of two named slots, `ras2voxel` and `rasfield`: editing it in place
    raises `TypeError`. It maps RAS to RAS: a point is carried to the
    voxels of the field's grid, where the field holds the RAS position it
    maps to.

    It is written as SPM writes it: the RAS coordinates, as values, in a
    NIfTI file whose voxel-to-RAS affine is the inverse of `ras2voxel`.
    Another transformation is converted to it exactly, or not at all (see
    [`brainhops.io.transformations.spm._converters`][]).
    """

    HINTS = ("spm",)

    PREFIXES: tx.ClassVar[tx.Tuple[str, ...]] = ("y_", "iy_")
    """
    File name prefixes of SPM fields.

    An SPM field is a plain NIfTI field with a naming convention, so the prefix
    is the only thing that distinguishes it from a generic field. A parser that
    constrains the prefix is more specific and wins ties.
    """

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a header as an SPM field.

        The score is that of [`NiftiRASCoordinateField`][], since the content
        of the file cannot distinguish the two readers, and pretending
        otherwise would invent evidence. The `y_` or `iy_` prefix decides the
        tie instead, through
        [`PREFIXES`][brainhops.io.transformations.spm.SpmCoordinateField.PREFIXES].
        """
        return NiftiRASCoordinateField._score_nibabel(header)

    # --- data -----------------------------------------------------------

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The coordinates given to the field, or `None` to decode them.

    Coordinates stored here are pending until [`consolidate`][] or a
    write encodes them into `rawdata`.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The RAS coordinates as an (X, Y, Z, 3) array.

        Coordinates that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them. Otherwise
        they are decoded from `rawdata`: NIfTI stores the field with
        shape (X, Y, Z, 1, 3), and the singleton axis is dropped.
        Assigning the data drops the chain decoded from the previous
        data.
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

    # --- chain --------------------------------------------------------
    # The stored chain is a tuple, as `ImmutableSequence` declares it, so
    # the derived and the assigned chain are both immutable, and in-place
    # edits such as `del field[0]` are refused.

    @smartproperty(cache=True)
    def transformations(
        self,
    ) -> tx.Tuple[RASToVoxel, RASCoordinateField]:
        """The transformations, decoded from the NIfTI file unless stored.

        The decoded chain is made of the RAS-to-voxel affine of the header
        and the field of RAS coordinates of the file. Both hold the metadata
        of this field, and the field holds its `rawdata` array and its pending
        coordinates too.
        """
        return (
            NiftiRASToVoxel(metadata=self.metadata),
            NiftiRASCoordinateField(
                data=self._data, rawdata=self.rawdata, metadata=self.metadata
            ),
        )

    @transformations.setter
    def transformations(
        self,
        value: tx.Optional[
            tx.Tuple[
                tx.Optional[RASToVoxel],
                tx.Optional[RASCoordinateField],
            ]
        ],
    ) -> None:
        # None means that the chain is decoded from the NIfTI file.
        self._transformations = None if value is None else tuple(value)

    @property
    def ras2voxel(self) -> tx.Optional[RASToVoxel]:
        """The transformation from RAS to voxel space."""
        xform = self.transformations[0]
        if xform is None:
            xform = NiftiRASToVoxel(metadata=self.metadata)
        return xform

    @property
    def rasfield(self) -> tx.Optional[RASCoordinateField]:
        """The field of RAS coordinates."""
        xform = self.transformations[1]
        if xform is None:
            xform = NiftiRASCoordinateField(
                data=self._data, rawdata=self.rawdata, metadata=self.metadata
            )
        return xform

    @ras2voxel.setter
    def ras2voxel(self, value: RASToVoxel) -> None:
        self._transformations = (value, self.transformations[1])

    @rasfield.setter
    def rasfield(self, value: RASCoordinateField) -> None:
        self._transformations = (self.transformations[0], value)

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        on_loss: tx.Optional[OnLoss] = None,
        **options: tx.Any,
    ) -> tx.Self:
        """
        Encode the pending coordinates or chain into `rawdata` and a
        header.

        The coordinates of `rasfield`, as values, become the array, with
        the `VECTOR` (1007) intent code and the intent name `"Mapping"`
        that SPM writes. The voxel-to-RAS affine of the grid is the
        inverse of `ras2voxel`.

        A field whose chain is decoded from its stored coordinates keeps
        them under a copy of the header of its metadata, whose geometry is
        kept. A field whose chain was assigned is encoded from the chain
        over that header, whose geometry is replaced. When `like` is given,
        non-encoding header fields are copied from it. Keyword arguments
        override header fields last. A field that has a header and nothing
        pending, consolidated without options, is left as it is.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the options that name its fields, and its record is
        the base of the new header. The fields given by keyword take
        precedence over `like`, and a `dtype` that is given, or changed
        in the metadata, is the type of the stored array. What the write
        loses or approximates is acted on once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this field rather than a copy.
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the header cannot hold. By
            default, the policy in effect.
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
            If the field has no coordinates.
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
            apply_loss_policy(report, on_loss, stacklevel=2)
            return self if inplace else replace(self)
        if "dtype" in requested:
            overrides["dtype"] = requested["dtype"]
        base = None if metadata is None else metadata.raw
        raw, record = _stored_parts(
            _spm_encoded(self, like, overrides, base, fields)
        )
        metadata = _consolidated_metadata(self, record)
        overruled_fields(requested, metadata, report)
        apply_loss_policy(report, on_loss, stacklevel=2)
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


def _spm_encoded(
    field: SpmCoordinateField,
    like: tx.Any,
    overrides: tx.Mapping[str, tx.Any],
    base: tx.Optional[NiftiRaw],
    fields: tx.Mapping[str, tx.Any],
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Encode an SPM field over a copy of a record.

    The record, `base`, is that of the consolidated metadata.

    The stored coordinates, assigned or read, are kept when the chain was
    not assigned, and the chain is encoded otherwise.

    Raises
    ------
    WriterError
        If the field has no coordinates.
    """
    if field._data is not None:
        stored = _field_to_disk(field._data)
    else:
        stored = field.rawdata
    if stored is not None and field._transformations is None:
        record = None if base is None else base.copy()
        image = _record_image(stored, record, overrides=overrides)
    else:
        what = "An SPM deformation field"
        # The inverse of a `NiftiRASToVoxel` read from a header is that
        # header's affine itself, rather than an inverse of its inverse.
        vox2ras = homogeneous_matrix(field.ras2voxel.inverse(), what, ndim=3)
        # SPM stores sampled coordinates.
        coordinates = field.rasfield.values
        if coordinates is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        coordinates = _field_to_disk(coordinates)
        record = None if base is None else base.copy()
        image = _record_image(coordinates, record, vox2ras, overrides)
    _set_coordinates_intent(image.header)
    _apply_like(image, like)
    _keep_given_fields(image, fields)
    _apply_overrides(image, overrides)
    return image
