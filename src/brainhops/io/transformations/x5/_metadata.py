"""The metadata of an X5 file."""

__all__ = ["X5Metadata"]

# dependencies
import h5py
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.common.hdf5 import Hdf5ReaderWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import X5Raw


@register_format
class X5Metadata(MetadataFormat, Hdf5ReaderWriter):
    """Metadata of an X5 file, which holds the file as an [`X5Raw`][].

    An X5 file has no header that precedes its data. Its record holds the
    root attributes, the chains and every transform of the file, with the
    JSON metadata of each transform. The matrices are read at once, and the
    fields are held as lazy arrays that read the file again by its name
    when they are used, so reading the metadata never reads a field. An
    [`X5Transform`][brainhops.io.transformations.x5.X5Transform] decodes its
    chain of transformations from the record of its metadata.

    The file is read and written by the record: [`from_h5`][] reads an
    [`X5Raw`][] and wraps it, and [`to_h5`][] writes the record, with the
    fields copied from their file. The metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".x5",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("x5", "bids")

    raw: NoRepr[tx.Optional[X5Raw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def to_raw(self) -> tx.Optional[X5Raw]:
        """Return a copy of the record that the caller is free to change.

        The copy shares the arrays of the record, which are never changed
        in place, so that copying a record never reads a field.

        Returns
        -------
        X5Raw or None
            A copy of `raw`, or `None` when no record is held.
        """
        return None if self.raw is None else self.raw.copy()

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score an open HDF5 file by its root `Format` attribute.

        The file is scored by [`X5Raw.sniff_h5`][].
        """
        return X5Raw.sniff_h5(h5file, error=error)

    @classmethod
    def from_h5(cls, h5file: h5py.File, load: bool = False) -> "X5Metadata":
        """Read the metadata of an open X5 file.

        The record is read by [`X5Raw.from_h5`][].

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        load : bool, default=False
            Whether to read the fields into memory at once, rather than
            when they are used.

        Returns
        -------
        X5Metadata
            The metadata.
        """
        return cls.from_raw(X5Raw.from_h5(h5file, load=load))

    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write the record into an empty HDF5 file open for writing.

        Metadata without a record is written as a new, empty record, which
        is an X5 file without transforms.
        """
        raw = X5Raw() if self.raw is None else self.raw
        raw.to_h5(h5file, **kwargs)
