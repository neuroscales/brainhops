__all__ = ["MetadataFormat"]

import copy

import typing_extensions as tx
from bagof.magic import NoRepr

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

    The header held in `raw` is called the record of the file. A record is
    never changed in place: [`to_raw`][] returns a copy, and a writer
    encodes its changes into that copy. Metadata keeps no open file and no
    reference to the data, so it is cheap to copy and it can be pickled.
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

    def to_raw(self) -> tx.Any:
        """Return a copy of the record that the caller is free to change.

        A writer encodes its changes into the copy, so the record held by
        the metadata stays as it was read. The default implementation makes
        a deep copy, and a format whose record has a cheaper copy overrides
        this method.

        Returns
        -------
        object
            A copy of `raw`, or `None` when no record is held.
        """
        return copy.deepcopy(self.raw)
