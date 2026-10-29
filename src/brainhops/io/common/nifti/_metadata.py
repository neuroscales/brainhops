"""The metadata of a NIfTI file."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr, replace

# internals
from brainhops.datamodel.metadata._internal_api import (
    Indexed,
    Resampled,
    takes_error_policy,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileWriter
from brainhops.io.metadata import MetadataFormat, MetadataReader
from brainhops.io.metadata._internal_api import consolidated

# this format
from ._raw import NiftiRaw
from ._vocabulary import clear_spatial_slots, decode, encode


@register_format
class NiftiMetadata(MetadataReader, BinaryFileWriter, MetadataFormat):
    """Metadata of a NIfTI file, which holds the header of the file.

    The metadata of a NIfTI file is read from the header alone, so the
    voxels that follow the header are never read, and a file cut after its
    header can still be read. The header is held as a [`NiftiRaw`][]
    record, which [`NiftiImage`][brainhops.io.images.nifti.NiftiImage] uses
    as the base of the header that it writes. The metadata keeps no open
    file.

    A file is read through the record: [`from_fileobj`][] reads a
    [`NiftiRaw`][] and wraps it, and writing the metadata writes the header
    of the consolidated record without voxels.

    The fields of the vocabulary that the header stores are decoded when
    the metadata is built, and are listed in `SUPPORTED_FIELDS`. The slot
    of each field is given in the table of
    `brainhops.io.common.nifti._vocabulary`. The world space (`space`)
    and an intent that changes how the axes are read belong to the
    geometry, so a change of them is reported as approximated. NIfTI
    has no free-form store, so `extra` is not supported.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nifti",)
    RAW: tx.ClassVar[type] = NiftiRaw
    SUPPORTED_FIELDS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "description",
        "sources",
        "intent",
        "space",
        "display_range",
        "phase_encoding_direction",
        "slice_encoding_direction",
        "slice_timing",
        "repetition_time",
        "dtype",
        "scale_slope",
        "scale_intercept",
    )
    """The fields of the vocabulary that a NIfTI header stores."""

    raw: NoRepr[tx.Optional[NiftiRaw]] = None
    """The header of the file, or `None` for metadata without a record."""

    @classmethod
    def from_raw(
        cls, raw: tx.Optional[NiftiRaw], **kwargs: tx.Any
    ) -> "NiftiMetadata":
        """Build the metadata of a record, decoding its vocabulary.

        Parameters
        ----------
        raw : NiftiRaw, optional
            The record, held as it is given.
        **kwargs : Any
            Other fields, which take precedence over the decoded values.

        Returns
        -------
        NiftiMetadata
            The metadata.
        """
        return cls(raw=raw, **{**decode(raw), **kwargs})

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> "NiftiMetadata":
        """Encode the fields that differ from the header into a copy of it.

        The contract is that of `consolidated`:
        the changes are written into a copy of the record, or into an
        empty NIfTI-1 header, and the result is decoded from it afresh.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this metadata rather than a copy.
        **fields : Any
            Fields to override before the encoding.

        Returns
        -------
        NiftiMetadata
            The consolidated copy, or this metadata with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If a field is reported under the `"raise"` policy.
        """
        return consolidated(
            self,
            encode,
            NiftiRaw,
            inplace=inplace,
            fields=fields,
        )

    def derive(
        self,
        operation: tx.Union[Indexed, Resampled, None] = None,
        *,
        history: tx.Union[str, tx.Sequence[str], None] = None,
    ) -> "NiftiMetadata":
        """Return the metadata of an image computed from this one.

        The fields follow
        [`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive].
        When the operation moves the spatial axes, the record is also
        replaced by a copy without `dim_info` and the slice slots, which
        describe the old voxel axes. The encoding directions that the
        derivation maps differ from that copy, so the next consolidation
        writes them.

        Parameters
        ----------
        operation : Indexed or Resampled, optional
            What the image operation did to the axes of the image.
        history : str or sequence of str, optional
            The steps of the derivation, appended to `history`.

        Returns
        -------
        NiftiMetadata
            New metadata.
        """
        derived = super().derive(operation, history=history)
        if operation is None or not operation.moves_space:
            return derived
        if self.raw is None:
            return derived
        return replace(derived, raw=clear_spatial_slots(self.raw))

    @takes_error_policy
    def to_bytes(
        self,
        **fields: tx.Any,
    ) -> bytes:
        """Return the header of the consolidated record as NIfTI stores it.

        The metadata is consolidated first, with the overrides given, so a
        field that was changed is written. The metadata itself is not
        changed. Metadata without a record is written as an empty NIfTI-1
        header.

        Parameters
        ----------
        **fields : Any
            Fields to override, as in [`consolidate`][].

        Returns
        -------
        bytes
            The header and its extensions.
        """
        raw = self.consolidate(**fields).raw
        return (NiftiRaw() if raw is None else raw).to_bytes()

    @takes_error_policy
    def to_fileobj(
        self,
        file: tx.BinaryIO,
        **fields: tx.Any,
    ) -> None:
        """Write the header of the consolidated record to a stream.

        The metadata is consolidated first, as in [`to_bytes`][], so that
        a file written from the metadata holds what [`to_bytes`][] returns.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        **fields : Any
            Fields to override, as in [`consolidate`][].
        """
        file.write(self.to_bytes(**fields))
