"""The metadata of an Imaris file."""

# dependencies
import h5py
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.common.hdf5 import Hdf5Reader
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import ImarisRaw


@register_format
class ImarisMetadata(Hdf5Reader, MetadataFormat):
    """Metadata of an Imaris file, which holds its groups and parameters.

    An Imaris file is an HDF5 file with no header before its data. Its
    metadata holds an [`ImarisRaw`][] record, which describes every
    resolution level, the numbers of time points and channels, and the
    parameters of the image, without reading the voxels. The metadata
    keeps no open file. Imaris files cannot be written by brainhops, so
    the metadata cannot be written either.

    A file is read through the record: [`from_h5`][] reads an
    [`ImarisRaw`][] and wraps it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ims",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("imaris", "ims")

    raw: NoRepr[tx.Optional[ImarisRaw]] = None
    """The record of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Return the confidence that an open HDF5 file is an Imaris file.

        The file is scored by [`ImarisRaw.sniff_h5`][].

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an Imaris file.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return ImarisRaw.sniff_h5(h5file, error=error)

    @classmethod
    def from_h5(cls, h5file: h5py.File, **kwargs) -> "ImarisMetadata":
        """Read the metadata of an open Imaris file, without its voxels.

        The record is read by [`ImarisRaw.from_h5`][], which receives the
        keyword arguments and refuses them.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        **kwargs : Any
            Options of the reader, which [`ImarisRaw.from_h5`][] refuses.

        Returns
        -------
        ImarisMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(ImarisRaw.from_h5(h5file, **kwargs))
