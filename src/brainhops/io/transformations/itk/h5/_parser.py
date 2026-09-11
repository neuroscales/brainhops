# stdlib
import os.path as op
from os import PathLike

# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic

# core
from brainhops._core.backends import da
from brainhops._core.typing import ArrayProtocol

# locals
from .._common import ITKStruct, ITKTransformClass

# optional
if tx.TYPE_CHECKING:
    import h5py
else:
    try:
        import h5py
    except ImportError:

        class h5py:
            File = None
            Dataset = None


# typing
_H5Like = tx.Union[
    tx.BinaryIO,
    PathLike,
    str,
    h5py.File,
]


class H5Header(
    Magic,
    convert=True,
    mapping=HIDE_IF_NONE,
    repr=HIDE_IF_NONE,
):
    """Header of a ITK H5 file."""

    HDFVersion: tx.Optional[str] = None
    """
    A string describing the version of the HDF5 library used.
    Ex: "HDF5 library version: 1.10.4"
    """

    ITKVersion: tx.Optional[str] = None
    """
    A string describing the version of the ITK library used.
    Ex: "5.1.0"
    """

    OSName: tx.Optional[str] = None
    """
    A string describing the operating system name.
    Ex: "Linux"
    """

    OSVersion: tx.Optional[str] = None
    """
    A string describing the operating system version.
    Ex: "6.1.0-1007-oem"
    """


class H5TransformParser(
    Magic,
    convert=True,
    mapping=HIDE_IF_NONE,
    repr=HIDE_IF_NONE,
):
    file: tx.Optional[h5py.File] = None
    header: H5Header = Factory(H5Header)
    transform_group: tx.List[ITKStruct] = Factory(list)

    @classmethod
    def sniff(cls, file: _H5Like) -> bool:
        """
        Check if the file is a valid HDF5 transform file.

        Parameters
        ----------
        file : str | PathLike | IO | h5py.File
            Input file.

        Returns
        -------
        bool
            True if the file is a valid HDF5 transform file, False otherwise.
        """
        if not isinstance(file, h5py.File):
            if not h5py.is_hdf5(file):
                return False

            with h5py.File(file, "r") as f:
                return cls.sniff(f)

        return "ITKVersion" in file.keys()

    @classmethod
    def from_file(
        cls, file: _H5Like, keep_open: bool = False, load: bool = True
    ) -> tx.Self:
        """
        Build an object from a file (path, file-like object, or HDF5 file).

        Parameters
        ----------
        file : str | PathLike | IO | h5py.File
            Input file.
        keep_open : bool, optional
            If True, keep the HDF5 file open after loading.
            If False, close the file after loading.
            If `load=False` and `keep_open=False`, the file will be
            re-opened every time displacement fields are accessed.
        load : bool, optional
            If True, load the data into memory. I
            f False, keep the data on disk.
        """
        if not isinstance(file, h5py.File):
            if keep_open:
                f = h5py.File(file, "r")
                return cls.from_file(f, keep_open=keep_open, load=load)
            else:
                with h5py.File(file, "r") as f:
                    return cls.from_file(f, keep_open=keep_open, load=load)

        return cls.from_h5(file, load=load, keep_open=keep_open)

    @classmethod
    def from_h5(
        cls,
        h5file: h5py.File,
        keep_open: bool = False,
        load: bool = True,
    ) -> tx.Self:
        """
        Build an object from an HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Input HDF5 file.
        load : bool, optional
            If True, load the data into memory.
            If False, keep the data on disk.
        keep_open : bool, optional
            If True, keep the HDF5 file open after loading.
            If False, close the file after loading.

        Returns
        -------
        obj
            The parsed object.
        """
        header = H5Header()
        if "/HDFVersion" in h5file:
            header.HDFVersion = _readstr(h5file["/HDFVersion"])
        if "/ITKVersion" in h5file:
            header.ITKVersion = _readstr(h5file["/ITKVersion"])
        if "/OSName" in h5file:
            header.OSName = _readstr(h5file["/OSName"])
        if "/OSVersion" in h5file:
            header.OSVersion = _readstr(h5file["/OSVersion"])

        obj = cls(header=header, file=h5file if keep_open else None)
        nodes = h5file.get("/TransformGroup", [])

        blocks = []
        for node in nodes:
            # Parse transform type
            xtype = _readstr(nodes[node]["TransformType"])
            xtype, prec, ndim_inp, ndim_out = xtype.split("_")
            xtype = ITKTransformClass(xtype)
            ndim_inp, ndim_out = int(ndim_inp), int(ndim_out)

            if xtype == "CompositeTransform":
                # skip composite transforms, they just point to the
                # following transforms.
                continue

            # Read transform parameters

            parameters = np.array([])
            if "TransformParameters" in nodes[node]:
                parameters_key = "TransformParameters"
                parameters = nodes[node]["TransformParameters"]
            elif "TranformParameters" in nodes[node]:
                # legacy spelling error in older ITK versions
                parameters_key = "TranformParameters"
                parameters = nodes[node][parameters_key]

            fixed_parameters = np.array([])
            if "TransformFixedParameters" in nodes[node]:
                fixed_parameters_key = "TransformFixedParameters"
                fixed_parameters = nodes[node]["TransformFixedParameters"]
            elif "TranformFixedParameters" in nodes[node]:
                # legacy spelling error in older ITK versions
                fixed_parameters_key = "TranformFixedParameters"
                fixed_parameters = nodes[node][fixed_parameters_key]

            # Always load fixed parameters (they are never large)
            fixed_parameters = fixed_parameters[()]

            # Do not load parameters if nonlinear (can be large)
            LARGE_TYPES = ("DisplacementFieldTransform", "BSplineTransform")
            if load or xtype not in LARGE_TYPES:
                parameters = parameters[()]
            else:
                if keep_open:
                    fileish = h5file
                else:
                    fileish = h5file.filename
                    fileish = op.abspath(fileish) if fileish else None

                parameters = DelayedH5Array(
                    fileish, f"/TransformGroup/{node}/{parameters_key}"
                )
                if da:
                    parameters = parameters.to_dask(keep_open=keep_open)

            blocks.append(
                ITKStruct(
                    type=xtype,
                    precision=prec,
                    ndim_input=ndim_inp,
                    ndim_output=ndim_out,
                    parameters=parameters,
                    fixed_parameters=fixed_parameters,
                )
            )

        obj.transform_group = blocks

        if not keep_open:
            h5file.close()
        return obj

    def _close(self) -> None:
        if isinstance(self.file, h5py.File):
            self.file.close()

    def __del__(self) -> None:
        self._close()


class DelayedH5Array:
    """
    Class that holds a H5 dataset that can be accessed even if the file
    is closed (i.e., by reopening the file when needed).
    """

    def __init__(self, file: _H5Like, path: str) -> None:
        self.file: _H5Like = file
        self.path: str = path
        self._file: tx.Optional[h5py.File] = None
        self._shape: tx.Optional[tx.Tuple[int]] = None
        self._dtype: tx.Optional[np.dtype] = None
        self._chunks: tx.Optional[tx.Tuple[int]] = None

    def open(self) -> h5py.File:
        self.to_dataset(keep_open=True)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()

    def to_dataset(
        self, file: tx.Optional[_H5Like] = None, keep_open: bool = False
    ) -> h5py.Dataset:
        if file is None:
            return self.to_dataset(self.file, keep_open=keep_open)

        if self._file is not None:
            file = self._file

        if isinstance(file, (str, PathLike)):
            if not keep_open:
                with h5py.File(file, "r") as f:
                    return self.to_dataset(f)
            else:
                file = self._file = h5py.File(file, "r")

        if isinstance(file, h5py.File):
            dataset = file[self.path]
            # cache info
            self._shape = dataset.shape
            self._dtype = dataset.dtype
            self._chunks = dataset.chunks
            return dataset

        raise ValueError("Invalid file type")

    def to_array(self, **kwargs) -> np.ndarray:
        import numpy as np

        is_mine = self._file is None
        dataset = self.to_dataset(keep_open=True)
        array = np.asarray(dataset, **kwargs)
        if is_mine:
            self.close()
        return array

    def to_dask(self, *, keep_open: bool = False, **kwargs) -> ArrayProtocol:
        import dask.array as da

        kwargs.setdefault("chunks", self.chunks or "auto")
        if keep_open:
            array_like = self.to_dataset(keep_open=True)
        else:
            array_like = self
        return da.from_array(array_like, **kwargs)

    def __getitem__(self, index: tx.Any) -> tx.Any:
        is_mine = self._file is None
        dataset = self.to_dataset(keep_open=True)
        chunk = dataset[index]
        if is_mine:
            self.close()
        return chunk

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        return self.to_array(dtype=dtype)

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            self.to_dataset()
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self.to_dataset()
        return self._dtype

    @property
    def chunks(self) -> tuple:
        if self._chunks is None:
            self.to_dataset()
        return self._chunks

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        from math import prod

        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize


def _readstr(dataset: h5py.Dataset) -> str:
    """
    Read a string from a HDF5 dataset.

    Parameters
    ----------
    dataset : h5py.Dataset
        The HDF5 dataset to read.

    Returns
    -------
    str
        The string read from the dataset.
    """
    return dataset[()].item().decode()
