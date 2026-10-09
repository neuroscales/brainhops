"""Low-level MINC helpers: sniffing and reading voxels."""

# stdlib
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from nibabel import minc1 as _minc1
from nibabel import minc2 as _minc2
from nibabel.externals.netcdf import netcdf_file as _netcdf_file

# internals
from brainhops.datamodel.axes import Axis
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
)

# this format
from ._constants import (
    _AXES,
    _HDF5_SIGNATURE,
    _MINC2_ROOT,
    _NETCDF_MAGICS,
    SPATIAL_DIMENSIONS,
)

_Source = tx.Union[str, bytes]
"""The source of the voxels: a local path, or the file bytes."""


def minc_version(head: bytes) -> tx.Optional[int]:
    """
    Return the MINC version that the leading bytes suggest, or `None`.

    NetCDF content suggests version 1, and HDF5 content version 2, although
    a MINC2 file also needs the `/minc-2.0` group.
    """
    head = bytes(head[:8])
    if head[:4] in _NETCDF_MAGICS:
        return 1
    if head == _HDF5_SIGNATURE:
        return 2
    return None


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _axis(name: str) -> Axis:
    """Return the voxel axis that a MINC dimension stands for."""
    axis_name, axis_type = _AXES.get(name, (name, None))
    return Axis(axis_name, axis_type, unit="index")


def _score(score: float, error: tx.Union[bool, tx.Type[Exception]]) -> float:
    """Return a score, or raise if it is zero and the caller asks to."""
    if score or not error:
        return score
    if error is True:
        error = SnifferContentError
    raise error("Content is not a MINC file of this version.")


def _sniff_minc1(head: bytes) -> float:
    """Score the leading bytes of a NetCDF file as MINC1."""
    named = any(name.encode() in head for name in SPATIAL_DIMENSIONS)
    if named and b"image" in head:
        return Confidence.CERTAIN
    return Confidence.WEAK


def _sniff_minc2(file: tx.Union[str, tx.IO]) -> float:
    """Score an HDF5 file, given by path or stream, as MINC2."""
    import h5py

    try:
        with h5py.File(file, "r") as f:
            found = isinstance(f.get(_MINC2_ROOT), h5py.Group)
    except Exception:
        return Confidence.NO
    return Confidence.CERTAIN if found else Confidence.NO


def _read_minc(
    source: _Source,
    version: int,
    read: tx.Callable[[tx.Any, _minc1.Minc1File, int], tx.Any],
) -> tx.Any:
    """
    Open a MINC file with nibabel, apply `read` to it, and close it.

    The result of `read` must not refer to the file content, because a
    MINC1 file on disk is memory-mapped and the map is closed on return.
    """
    try:
        if version == 1:
            if isinstance(source, str):
                container = _netcdf_file(source, "r", mmap=True)
            else:
                container = _netcdf_file(BytesIO(source), "r", mmap=False)
        else:
            import h5py

            fileish = source if isinstance(source, str) else BytesIO(source)
            container = h5py.File(fileish, "r")
    except Exception as e:
        raise ParserContentError(f"Not a MINC{version} file: {e}") from e
    mfile, failure = None, None
    try:
        try:
            mfile = (_minc1.Minc1File if version == 1 else _minc2.Minc2File)(
                container
            )
        except (
            AttributeError,
            KeyError,
            ValueError,
            _minc1.MincError,
        ) as e:
            # Raise outside this block: a held traceback would keep
            # memory-mapped variables alive past close.
            failure = repr(e)
        if failure is not None:
            raise ParserContentError(
                f"Cannot read this MINC{version} file with nibabel: "
                f"{failure}. Irregularly spaced dimensions, dimensions "
                f"without a variable (such as vector_dimension) and "
                f"volumes without image-min/image-max are not supported."
            )
        return read(container, mfile, version)
    finally:
        mfile = None
        container.close()


def _read_voxels(
    container: tx.Any, mfile: _minc1.Minc1File, version: int
) -> np.ndarray:
    """Read the scaled voxels of an open file, in file (C) order."""
    array = mfile.get_scaled_data()
    # Copy into native byte order (MINC1 is big-endian), so that the result
    # outlives a memory-mapped file.
    return np.array(array, dtype=array.dtype.newbyteorder("="))
