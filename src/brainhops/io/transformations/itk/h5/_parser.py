# stdlib
import os.path as op
from io import BytesIO
from os import PathLike

# dependencies
import h5py
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic

# core
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import da

# io
from brainhops.io.base.parsers import (
    BinaryFileParser,
    Confidence,
    ParserExistsError,
    SnifferContentError,
)

# locals
from .._common import ITKStruct, ITKTransformClass

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
    BinaryFileParser,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parses an ITK binary (`.h5`) transform file into a chain of
    transform blocks."""

    file: tx.Optional[h5py.File] = None
    header: H5Header = Factory(H5Header)
    transform_group: tx.List[ITKStruct] = Factory(list)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score how confident the parser is that an open HDF5 file is
        an ITK transform file."""
        # An ITK transform file records the ITK version at the root.
        if "ITKVersion" in h5file.keys():
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error("HDF5 file is not an ITK transform file")
        return Confidence.NO

    @classmethod
    def sniff_file(
        cls,
        file: _H5Like,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the parser is that a file (path, open
        HDF5 file, or file-like object) is an ITK transform file."""
        if isinstance(file, h5py.File):
            return cls.sniff_h5(file, error=error)

        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            if not file.exists():
                if error:
                    if error is True:
                        error = ParserExistsError
                    raise error(f"No such file: {file}")
                return Confidence.NO
            if not h5py.is_hdf5(str(file)):
                if error:
                    if error is True:
                        error = SnifferContentError
                    raise error(f"Not an HDF5 file: {file}")
                return Confidence.NO
            with h5py.File(str(file), "r") as f:
                return cls.sniff_h5(f, error=error)

        return cls.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the parser is that an open, seekable
        binary file object is an ITK transform file."""
        with preserve_position(file):
            try:
                with h5py.File(file, "r") as f:
                    return cls.sniff_h5(f, error=error)
            except Exception as e:
                if error:
                    if error is True:
                        error = SnifferContentError
                    raise error("Content is not an ITK HDF5 file") from e
                return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the parser is that bytes are an ITK
        transform file."""
        return cls.sniff_fileobj(BytesIO(content), error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        file: _H5Like,
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
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
        if isinstance(file, h5py.File):
            return cls.from_h5(file, load=load, keep_open=keep_open)

        if isinstance(file, str):
            file = path.Path(file)

        # A real path is handed to `h5py` directly rather than as an open
        # stream, so that the file is reopened by name when a displacement
        # field is read lazily (`load=False`), long after this returns.
        if isinstance(file, path.PathLike):
            if not file.exists():
                raise ParserExistsError(f"No such file: {file}")
            if keep_open:
                f = h5py.File(str(file), "r")
                return cls.from_h5(f, keep_open=keep_open, load=load)
            with h5py.File(str(file), "r") as f:
                return cls.from_h5(f, keep_open=keep_open, load=load)

        return cls.from_fileobj(file, keep_open=keep_open, load=load)

    @classmethod
    def from_fileobj(
        cls,
        file: tx.IO,
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
    ) -> tx.Self:
        """Build an object from an open, seekable binary file object."""
        with preserve_position(file):
            f = h5py.File(file, "r")
            return cls.from_h5(f, keep_open=keep_open, load=load)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build an object from bytes in ITK binary transform format."""
        return cls.from_fileobj(BytesIO(content), **kwargs)

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
        """Close the underlying HDF5 file, if one is still open."""
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
        """Open (or reuse) the underlying HDF5 file and return it."""
        self.to_dataset(keep_open=True)
        return self._file

    def close(self) -> None:
        """Close the underlying HDF5 file, if this array opened it."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        """Close the underlying HDF5 file, if this array opened it."""
        self.close()

    def to_dataset(
        self, file: tx.Optional[_H5Like] = None, keep_open: bool = False
    ) -> h5py.Dataset:
        """Return the underlying `h5py.Dataset`, opening the file if
        needed."""

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
        """Read the whole dataset into a `numpy` array."""
        import numpy as np

        is_mine = self._file is None
        dataset = self.to_dataset(keep_open=True)
        array = np.asarray(dataset, **kwargs)
        if is_mine:
            self.close()
        return array

    def to_dask(self, *, keep_open: bool = False, **kwargs) -> ArrayProtocol:
        """Wrap the dataset as a `dask` array that reads chunks lazily."""
        import dask.array as da

        kwargs.setdefault("chunks", self.chunks or "auto")
        if keep_open:
            array_like = self.to_dataset(keep_open=True)
        else:
            array_like = self
        return da.from_array(array_like, **kwargs)

    def __getitem__(self, index: tx.Any) -> tx.Any:
        """Read the indexed chunk of the dataset, opening the file if
        needed."""
        is_mine = self._file is None
        dataset = self.to_dataset(keep_open=True)
        chunk = dataset[index]
        if is_mine:
            self.close()
        return chunk

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        """Read the whole dataset into a `numpy` array."""
        return self.to_array(dtype=dtype)

    @property
    def shape(self) -> tuple:
        """The shape of the dataset."""
        if self._shape is None:
            self.to_dataset()
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the dataset."""
        if self._dtype is None:
            self.to_dataset()
        return self._dtype

    @property
    def chunks(self) -> tuple:
        """The chunk shape of the dataset, or `None` if it is not
        chunked."""
        if self._chunks is None:
            self.to_dataset()
        return self._chunks

    @property
    def ndim(self) -> int:
        """The number of dimensions of the dataset."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The total number of elements in the dataset."""
        from math import prod

        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        """The size of the dataset in bytes."""
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
