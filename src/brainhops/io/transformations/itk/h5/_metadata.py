"""The metadata of an ITK HDF5 (`.h5`) file."""

__all__ = ["H5Metadata"]

# dependencies
import h5py
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.common.hdf5 import Hdf5Reader
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import H5Raw


@register_format
class H5Metadata(MetadataFormat, Hdf5Reader):
    """Metadata of an ITK `.h5` file, which holds the file as an [`H5Raw`][].

    An ITK `.h5` file has no header that precedes its data. Its record
    holds the provenance strings of its root, such as the version of ITK
    that wrote it, and every block of the file. The parameters of the
    warps are held as lazy arrays that read the file again by its name
    when they are used, so reading the metadata never reads a warp. An
    [`H5Transform`][brainhops.io.transformations.itk.h5.H5Transform]
    decodes its chain from the record of its metadata.

    The file is read by the record: [`from_h5`][] reads an [`H5Raw`][] and
    wraps it. brainhops does not write ITK `.h5` files, so the metadata
    cannot be written either. The metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".h5", ".hdf5")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("itk", "ants", "h5")

    raw: NoRepr[tx.Optional[H5Raw]] = None
    """The root and the blocks of the file, or `None` without a record."""

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score an open HDF5 file as an ITK transform file.

        The file is scored by [`H5Raw.sniff_h5`][].

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an ITK
            transform file.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return H5Raw.sniff_h5(h5file, error=error)

    @classmethod
    def from_h5(cls, h5file: h5py.File, load: bool = False) -> "H5Metadata":
        """Read the metadata of an open ITK HDF5 file.

        The record is read by [`H5Raw.from_h5`][].

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        load : bool, default=False
            Whether to read the parameters of the warps into memory at
            once, rather than when they are used.

        Returns
        -------
        H5Metadata
            The metadata.
        """
        return cls.from_raw(H5Raw.from_h5(h5file, load=load))
