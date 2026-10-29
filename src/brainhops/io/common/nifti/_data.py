"""The lazy array that reads the voxels of a NIfTI file.

nibabel parses and writes the headers of NIfTI files, and the proxy
below reads their voxels. A NIfTI file stores its voxels contiguously,
with the first axis varying fastest, from the offset that the header
gives. The values may be scaled by the slope and the intercept of the
header, as nibabel scales them.
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly
from nibabel.arrayproxy import ArrayProxy as _NibabelArrayProxy

# internals
from brainhops._core._internal_api import ArrayProxy, ScaledProxy, StridedProxy

_NiftiHeader = tx.Union[nb.Nifti1Header, nb.Nifti2Header]


class _NiftiProxy(StridedProxy):
    """A lazy array of the stored voxels of a NIfTI file.

    The proxy reads the voxels of a `.nii` or `.nii.gz` file, or of the
    image file of a header and image pair, as the file stores them.
    It keeps the slope and the intercept of the header, but it does not
    apply them. When they change the values, the reader wraps the proxy
    in a [`ScaledProxy`][brainhops._core.proxy.ScaledProxy] that applies
    them, so that the values and their type are those that nibabel
    reads.

    A read gets only the bytes that its selection needs (see
    [`StridedProxy`][]). The voxels of a local file that is not
    compressed are memory-mapped when `mmap` is true, and a file
    compressed with gzip is decompressed up to the end of the selection.
    The blocks that the proxy gives Dask are slabs of the last axis that
    has more than one value, such as the volumes of a series.

    A reader builds the lazy array of a file, scaled or not, with
    [`from_header`][].
    """

    slope: KwOnly[float] = 1.0
    """The slope of the header, or 1 when the header does not scale."""

    inter: KwOnly[float] = 0.0
    """The intercept of the header, or 0 when the header does not
    scale."""

    @classmethod
    def from_header(
        cls,
        source: tx.Any,
        header: _NiftiHeader,
        start: int = 0,
        mmap: bool = True,
    ) -> ArrayProxy:
        """
        Return the lazy array of the voxels that a NIfTI header describes.

        The shape, the stored type, the byte order and the offset of the
        voxels are read from the header. The slope and the intercept are
        read as nibabel reads them. A slope of zero, or a slope that is
        not finite, means that the voxels are not scaled, and the
        intercept is then ignored. When the slope and the intercept
        change the values, the proxy is wrapped in a
        [`ScaledProxy`][brainhops._core.proxy.ScaledProxy] that applies
        them.

        Parameters
        ----------
        source : path-like or file object
            The file that holds the voxels, by its path, or a stream open
            in binary mode. For a header and image pair, it is the image
            file.
        header : nibabel.Nifti1Header or nibabel.Nifti2Header
            The header of the file.
        start : int, default=0
            The position of the file in `source`, before decompression.
        mmap : bool, default=True
            Whether a local file that is not compressed is memory-mapped.

        Returns
        -------
        _NiftiProxy or ScaledProxy
            The lazy array of the voxels.

        Raises
        ------
        nibabel.spatialimages.HeaderDataError
            If the slope is valid but the intercept is not finite.
        """
        slope, inter = header.get_slope_inter()
        proxy = cls.contiguous(
            source,
            header.get_data_dtype(),
            header.get_data_shape(),
            offset=header.get_data_offset(),
            order="F",
            start=start,
            mmap=mmap,
            slope=1.0 if slope is None else slope,
            inter=0.0 if inter is None else inter,
        )
        return proxy.scaled()

    @classmethod
    def from_nibabel(cls, proxy: tx.Any) -> tx.Any:
        """
        Return the lazy array that reads the same voxels as a nibabel proxy.

        A nibabel array proxy is replaced by the lazy array that
        [`from_header`][] would build for the same file, with the same
        offset, stored type, shape, order and scaling. Any other array is
        returned as it is.

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
        stored = cls.contiguous(
            proxy.file_like,
            proxy.dtype,
            proxy.shape,
            offset=proxy.offset,
            order=proxy.order,
            mmap=bool(getattr(proxy, "_mmap", True)),
            slope=float(proxy.slope),
            inter=float(proxy.inter),
        )
        return stored.scaled()

    @property
    def is_scaled(self) -> bool:
        """Whether the slope and the intercept change the values."""
        if self.dtype.fields is not None:
            # nibabel cannot scale structured values, such as RGB
            # triplets, so their scaling is ignored.
            return False
        return (self.slope, self.inter) != (1.0, 0.0)

    def scaled(self) -> ArrayProxy:
        """Return the lazy array of the values once the header scales them.

        The proxy itself is returned when its scaling does not change the
        values. Otherwise, the result is a
        [`ScaledProxy`][brainhops._core.proxy.ScaledProxy]. A stored
        floating-point type is kept, and a stored integer type gives a
        floating-point type that holds its values.
        """
        if not self.is_scaled:
            return self
        return ScaledProxy(self, self.slope, self.inter)


def whole_file(raw: tx.Any) -> tx.Optional[_NiftiProxy]:
    """Return the lazy array of a whole NIfTI file that an array reads.

    A writer copies the stored voxels of such a lazy array as they are.
    The array qualifies when it is the lazy array that the reader built
    for the file. This is the stored proxy itself when the header does
    not scale, or the scaled proxy that applies the scaling of the
    header. An index, a permutation, or a scaling that differs from the
    header is encoded like an array in memory.

    Parameters
    ----------
    raw : array
        The `rawdata` array of an object.

    Returns
    -------
    _NiftiProxy or None
        The proxy of the stored voxels, or `None`.
    """
    scaled = isinstance(raw, ScaledProxy)
    proxy = raw.base if scaled else raw
    if not isinstance(proxy, _NiftiProxy):
        return None
    if scaled != proxy.is_scaled:
        return None
    if not scaled:
        return proxy
    for factor, value in ((raw.slope, proxy.slope), (raw.inter, proxy.inter)):
        if factor is None or factor.size != 1:
            return None
        if float(factor.item()) != value:
            return None
    return proxy


def stored_dtype(raw: tx.Any) -> np.dtype:
    """Return the type in which a file stores an array.

    The lazy array of a whole file is stored in the type of its file,
    whatever the type of its scaled values. Any other array is stored
    in its own type.

    Parameters
    ----------
    raw : array
        The array.

    Returns
    -------
    numpy.dtype
        The stored type.
    """
    whole = whole_file(raw)
    return np.dtype(raw.dtype if whole is None else whole.dtype)
