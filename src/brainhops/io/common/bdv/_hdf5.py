"""The HDF5 file of a BigDataViewer dataset: its resolution levels and the
lazy array that reads the voxels of one level of one view.

This module needs h5py, so only the image readers import it. The layout
of the file is described in [`brainhops.io.common.bdv`][].
"""

# stdlib
import os

# dependencies
import h5py
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops._core import path
from brainhops._core._internal_api import StoredProxy, alias
from brainhops.io.base.parsers import ParserContentError, ParserExistsError
from brainhops.io.common.hdf5 import H5ArrayProxy
from brainhops.io.common.hdf5._parsers import read_string

# this format
from ._raw import BdvRaw

_LEGACY_STORED = np.dtype("int16")
"""The type in which older files store 16-bit unsigned voxels."""

_LEGACY_TYPE = np.dtype("uint16")
"""The type of the voxels of a file without a `dataType` attribute."""


class _BdvProxy(StoredProxy):
    """A lazy array of the voxels of one level of one view.

    The proxy holds a
    [`H5ArrayProxy`][brainhops.io.common.hdf5.H5ArrayProxy] of the
    `cells` dataset, which opens the HDF5 file by its name each time
    values are read, and the type of the voxels that the record
    gives. The values are presented in the order of the file, z, y, x,
    and in that type. The 16-bit signed values of an older file are
    reinterpreted as unsigned, bit for bit, as BigDataViewer reads them,
    and values stored in another type than the one that the `dataType`
    attribute declares are converted to it. Indexing the proxy returns
    another lazy array (see [`StoredProxy`][]), and converting that array
    reads only the selected part of the dataset.
    """

    cells: H5ArrayProxy
    """The lazy `cells` dataset."""

    _dtype: np.dtype
    """
    The type of the voxels that the record gives, given as `dtype=`,
    which may differ from the stored type.
    """

    shape = alias("cells.shape", readonly=True)
    """The shape of the voxels, in the order z, y, x of the file."""

    dtype = alias("_dtype", readonly=True)
    """The type of the voxels, which may differ from the stored type."""

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The shape of an HDF5 chunk of the `cells` dataset, or `None`.

        A chunk is the unit that HDF5 reads at once. `None` is given
        for a dataset that is stored without chunks.
        """
        return self.cells.chunks

    def _converted(self, values: tx.Any) -> np.ndarray:
        """Return stored values in the type of the voxels."""
        values = np.asarray(values)
        if values.dtype == self._dtype:
            return values
        if values.dtype == _LEGACY_STORED and self._dtype == _LEGACY_TYPE:
            return values.view(_LEGACY_TYPE)
        return values.astype(self._dtype)

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        # h5py reads the integers and the slices with a positive step that
        # the base class gives.
        return self._converted(self.cells._read(index))


def read_dataset(
    filename: path.FilenameLike,
    setup: tx.Optional[int] = None,
    timepoint: tx.Optional[int] = None,
) -> tx.Tuple[BdvRaw, int, int, tx.List[_BdvProxy]]:
    """Read a view of a dataset: its record and a lazy array of each level.

    The XML file is read into a record, and the HDF5 file that it names
    is opened for the resolution levels of every setup, which the record
    then holds, and for the shape and the type of each level of the view.
    No voxel is read.

    Parameters
    ----------
    filename : path-like
        The path of the XML file.
    setup : int, optional
        The id of the setup of the view, by default the first one.
    timepoint : int, optional
        The id of the time point of the view, by default the first one.

    Returns
    -------
    raw : BdvRaw
        The record, with the resolution levels of the HDF5 file.
    setup : int
        The id of the setup of the view.
    timepoint : int
        The id of the time point of the view.
    levels : list of _BdvProxy
        A lazy array of each level of the view, finest first.

    Raises
    ------
    ParserExistsError
        If the XML file or the HDF5 file does not exist.
    ParserContentError
        If the dataset is not stored in HDF5, if the view does not exist or
        is missing, or if the HDF5 file does not hold it.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    if not path.exists(filename):
        raise ParserExistsError(f"No such file: {filename}")
    with filename.open("rb") as file:
        raw = BdvRaw.from_bytes(file.read())
    if raw.loader != "bdv.hdf5":
        raise ParserContentError(
            f"This BigDataViewer dataset is stored with the image loader "
            f"{raw.loader!r}, and brainhops reads the 'bdv.hdf5' loader "
            f"only."
        )
    setup, timepoint = raw.view(setup, timepoint)
    if (timepoint, setup) in raw.missing:
        raise ParserContentError(
            f"The view of time point {timepoint} and setup {setup} is "
            f"listed as missing in this BigDataViewer dataset."
        )
    hdf5 = raw.hdf5_filename(filename)
    if not path.exists(hdf5):
        raise ParserExistsError(
            f"No such file: {hdf5}, which the BigDataViewer dataset "
            f"{filename} names."
        )
    hdf5 = os.path.abspath(os.fspath(hdf5))
    with h5py.File(hdf5, "r") as h5file:
        pyramids = _pyramids(h5file, raw)
        raw = replace(raw, pyramids=pyramids)
        pyramid = raw.pyramid(setup)
        levels = [
            _level(h5file, hdf5, pyramid, timepoint, level)
            for level in range(len(pyramid.resolutions))
        ]
    return raw, setup, timepoint, levels


def level_count(
    filename: path.FilenameLike, setup: tx.Optional[int] = None
) -> tx.Optional[int]:
    """Return the number of levels of a setup, or `None` when unknown.

    Only the XML file and the resolutions of the setup in the HDF5 file
    are read. Any failure gives `None`, so that a sniffer that calls this
    function never fails.

    Parameters
    ----------
    filename : path-like
        The path of the XML file.
    setup : int, optional
        The id of the setup, by default the first one.

    Returns
    -------
    int or None
        The number of levels.
    """
    try:
        if isinstance(filename, str):
            filename = path.Path(filename)
        with filename.open("rb") as file:
            raw = BdvRaw.from_bytes(file.read())
        setup, _ = raw.view(setup, None)
        hdf5 = raw.hdf5_filename(filename)
        with h5py.File(os.fspath(hdf5), "r") as h5file:
            return len(_resolutions(h5file, setup))
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _pyramids(h5file: h5py.File, raw: BdvRaw) -> tx.Tuple[BdvRaw.Pyramid, ...]:
    """Read the resolution levels of the setups that the file describes.

    The groups follow `bdv/img/hdf5/Util.java` (bigdataviewer-core,
    commit dfe1a66). A setup without a group in the file is left out.
    """
    pyramids = []
    for item in raw.setups:
        group = f"s{item.id:02d}"
        if group not in h5file:
            continue
        subdivisions = h5file.get(group + "/subdivisions")
        data_type = h5file[group].attrs.get("dataType")
        pyramids.append(
            BdvRaw.Pyramid(
                setup=item.id,
                resolutions=_resolutions(h5file, item.id),
                subdivisions=()
                if subdivisions is None
                else tuple(
                    tuple(int(v) for v in row)
                    for row in np.asarray(subdivisions).reshape(-1, 3)
                ),
                data_type=_data_type(data_type),
            )
        )
    return tuple(pyramids)


def _data_type(value: tx.Any) -> tx.Optional[str]:
    """Decode the `dataType` attribute of a setup group, if it exists.

    The attribute is a string, which may be stored as bytes of a fixed
    length that end with null characters, and these are removed. How
    JHDF5, with which BigDataViewer writes the file, pads the string was
    not checked.
    """
    if value is None:
        return None
    return read_string(value).rstrip("\x00")


def _resolutions(h5file: h5py.File, setup: int) -> tx.Tuple[tx.Any, ...]:
    """Read the downsampling factors of each level of a setup.

    Raises
    ------
    ParserContentError
        If the file has no resolutions for the setup.
    """
    name = f"s{setup:02d}/resolutions"
    if name not in h5file:
        raise ParserContentError(
            f"The HDF5 file of this BigDataViewer dataset has no {name} "
            f"dataset."
        )
    values = np.asarray(h5file[name], dtype=float).reshape(-1, 3)
    return tuple(tuple(float(v) for v in row) for row in values)


def _level(
    h5file: h5py.File,
    hdf5: str,
    pyramid: BdvRaw.Pyramid,
    timepoint: int,
    level: int,
) -> _BdvProxy:
    """Return a lazy array of the `cells` of a level of a view.

    Raises
    ------
    ParserContentError
        If the file has no voxels for that level, or if the declared type
        of the voxels is unknown.
    """
    name = f"t{timepoint:05d}/s{pyramid.setup:02d}/{level}/cells"
    if name not in h5file:
        raise ParserContentError(
            f"The HDF5 file of this BigDataViewer dataset has no {name} "
            f"dataset."
        )
    cells = H5ArrayProxy(hdf5, name)
    # Reading the dataset from the open file stores its shape and type,
    # so that the array knows them without opening the file again.
    stored = cells.to_dataset(h5file).dtype
    if pyramid.data_type is None:
        dtype = _LEGACY_TYPE if stored == _LEGACY_STORED else stored
    else:
        try:
            dtype = np.dtype(pyramid.data_type)
        except TypeError as error:
            raise ParserContentError(
                f"The voxels of setup {pyramid.setup} have the unknown type "
                f"{pyramid.data_type!r}."
            ) from error
    return _BdvProxy(cells, dtype)
