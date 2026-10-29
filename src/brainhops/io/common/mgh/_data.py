"""The lazy array that reads the voxels of an MGH file.

nibabel parses and writes the headers of MGH files, and the proxy below
reads their voxels. An MGH file stores its voxels big-endian and
contiguously, with the first axis varying fastest, right after its
header of 284 bytes. The voxels are not scaled.
"""

# dependencies
import typing_extensions as tx
from nibabel.arrayproxy import ArrayProxy as _NibabelArrayProxy
from nibabel.freesurfer import mghformat as _mgh

# internals
from brainhops._core._internal_api import (
    ArrayProxy,
    Path,
    StridedProxy,
    is_same_file,
)


class MghProxy(StridedProxy):
    """A lazy array of the voxels of an MGH or MGZ file.

    The proxy reads the voxels as the file stores them. A read gets only
    the bytes that its selection needs (see [`StridedProxy`][]). The
    voxels of a local `.mgh` file are memory-mapped when `mmap` is true,
    and an `.mgz` file is decompressed up to the end of the selection.
    Whether a file is compressed is judged from its content, not from
    its name. The blocks that the proxy gives Dask are slabs of the last
    axis that has more than one value, which is the axis of the frames
    when there are several.

    A reader builds the lazy array of a file with [`from_header`][].
    """

    @classmethod
    def from_header(
        cls,
        source: tx.Any,
        header: _mgh.MGHHeader,
        start: int = 0,
        mmap: bool = True,
    ) -> "MghProxy":
        """Return the lazy array of the voxels that an MGH header describes.

        The shape, the stored type and the offset of the voxels are read
        from the header, as nibabel reads them. A single frame gives a
        three-dimensional array.

        Parameters
        ----------
        source : path-like or file object
            The file, by its path, or a stream open in binary mode.
        header : nibabel.freesurfer.mghformat.MGHHeader
            The header of the file.
        start : int, default=0
            The position of the file in `source`, before decompression.
        mmap : bool, default=True
            Whether a local file that is not compressed is memory-mapped.

        Returns
        -------
        MghProxy
            The lazy array of the voxels.
        """
        return cls.contiguous(
            source,
            header.get_data_dtype(),
            header.get_data_shape(),
            offset=header.get_data_offset(),
            order="F",
            start=start,
            mmap=mmap,
        )

    @classmethod
    def from_nibabel(cls, proxy: tx.Any) -> tx.Any:
        """
        Return the lazy array that reads the same voxels as a nibabel proxy.

        A nibabel array proxy is replaced by the lazy array that
        [`from_header`][] would build for the same file, with the same
        offset, stored type, shape and order. Any other array is returned
        as it is.

        Parameters
        ----------
        proxy : array
            The `dataobj` of a nibabel image.

        Returns
        -------
        array
            The lazy array of the voxels, or `proxy` itself.
        """
        if not isinstance(proxy, _NibabelArrayProxy):
            return proxy
        return cls.contiguous(
            proxy.file_like,
            proxy.dtype,
            proxy.shape,
            offset=proxy.offset,
            order=proxy.order,
            mmap=bool(getattr(proxy, "_mmap", True)),
        )


def reads_mgh(raw: tx.Any, file: Path) -> bool:
    """Tell whether a lazy array reads the voxels of an MGH file.

    The array reads the file when the proxy below its index, permutation
    or scaling is the proxy of that file, by its path.

    Parameters
    ----------
    raw : array
        The `rawdata` array of an object.
    file : Path
        The file that a writer writes.

    Returns
    -------
    bool
        Whether `raw` reads `file`.
    """
    root = raw.root if isinstance(raw, ArrayProxy) else raw
    return (
        isinstance(root, MghProxy)
        and isinstance(root.source, Path)
        and is_same_file(root.source, file)
    )
