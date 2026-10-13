__all__ = ["MetadataFormat"]

import typing_extensions as tx
from bagof.magic import NoRepr, replace

from brainhops.datamodel.metadata import Metadata
from brainhops.io.base._base import (
    Format,
    _FileBasedModelMixin,
    format_registry,
)
from brainhops.io.base.specs import register_parser


@register_parser(Metadata)
@format_registry(isolated=True)
class MetadataFormat(_FileBasedModelMixin, Metadata, Format, eq=False):
    """Format dispatcher and common base for the metadata of files.

    The metadata of a file is what its header records beside the data. Each
    file format that has a header defines a subclass named after the format,
    such as `NiftiMetadata`, which holds the header in `raw` as the format
    stores it. `MetadataFormat.load` reads the header of a file in any
    registered format, and the same method of a subclass reads a file of
    its own format. In both cases, the data that follows the header is not
    read.

    The header held in `raw` is called the record of the file. The record
    describes the file as it was read, or as the object that holds it was
    last consolidated. A record is never changed in place: an object that
    encodes a new header, with its `consolidate` method or when it is
    written, encodes it into a copy and holds that copy in new metadata.
    Metadata keeps no open file and no reference to the data, so it is
    cheap to copy and it can be pickled.
    The record of one format means nothing to another, so `from_instance`
    leaves `raw` unset when it copies the metadata of another format.

    The metadata registry is separate from the registry of
    [`Format`][], so that [`load`][brainhops.io.load] never returns
    metadata when it reads an image or a transformation. A concrete
    subclass is added to the metadata registry with `@register_format`.
    """

    raw: NoRepr[tx.Any] = None
    """The header of the file as the format stores it, or `None`.

    The type of the record depends on the format, and only the format
    reads it. The record is hidden from the representation of the
    metadata, because a header is usually long.
    """

    @classmethod
    def from_raw(cls, raw: tx.Any, **kwargs) -> tx.Self:
        """Build the metadata that holds a record.

        A reader calls this method once it has parsed the header of a file.
        The record is held as it is given, without a copy.

        Parameters
        ----------
        raw : object
            The header of a file, in the representation of the format.
        **kwargs : Any
            Other fields of the metadata.

        Returns
        -------
        MetadataFormat
            The metadata.
        """
        return cls(raw=raw, **kwargs)

    def consolidate(self, *, inplace: bool = False) -> tx.Self:
        """Encode the fields of the metadata into its record.

        Consolidating metadata makes its record describe every field of
        the metadata, so that the header written from the record holds
        them. The fields that differ from a fresh decode of the record are
        encoded into a copy of the record, and the metadata then holds
        that copy. The record held before is never changed. Today the
        metadata of a format holds only its record, so there is nothing
        to encode: the copy holds the same record, and with
        `inplace=True` the metadata is left as it is.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this metadata rather than a copy.

        Returns
        -------
        Self
            The consolidated copy, or this metadata with `inplace=True`.
        """
        if inplace:
            return self
        return replace(self)
