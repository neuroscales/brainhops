# stdlib
from os import PathLike

# dependencies
import numpy as np
import typing_extensions as _tx

# externals
from brainhops._ext.struct import Struct, Factory, HIDE_IF_NONE

# locals
from .._common import ITKStruct

# optional
if _tx.TYPE_CHECKING:
    import h5py
else:
    try:
        import h5py
    except ImportError:
        h5py = None

# typing
_H5Like = _tx.Union[
    _tx.BinaryIO,
    PathLike,
    str,
    h5py.File,
]


class H5Header(
    Struct,
    convert=True,
    mapping=HIDE_IF_NONE,
    repr=HIDE_IF_NONE,
):
    """Header of a ITK H5 file."""

    HDFVersion: str | None = None
    """
    A string describing the version of the HDF5 library used.
    Ex: "HDF5 library version: 1.10.4"
    """

    ITKVersion: str | None = None
    """
    A string describing the version of the ITK library used.
    Ex: "5.1.0"
    """

    OSName: str | None = None
    """
    A string describing the operating system name.
    Ex: "Linux"
    """

    OSVersion: str | None = None
    """
    A string describing the operating system version.
    Ex: "6.1.0-1007-oem"
    """


class H5TransformParser(
    Struct,
    convert=True,
    mapping=HIDE_IF_NONE,
    repr=HIDE_IF_NONE,
):

    file: h5py.File | None = None
    header: H5Header = Factory(H5Header)
    transform_group: _tx.List[ITKStruct] = Factory(list)

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
        cls, file: _H5Like,
        keep_open: bool = False,
        load: bool = True
    ) -> _tx.Self:
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
            re-opened everytime displacement fields are accessed.
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
        cls, h5file: h5py.File,
        keep_open: bool = False,
        load: bool = True,
    ) -> _tx.Self:
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
            ndim_inp, ndim_out = int(ndim_inp), int(ndim_out)

            if xtype == "CompositeTransform":
                # skip composite transforms, they just point to the
                # following transforms.
                continue

            # Read transform parameters

            parameters = np.array([])
            if "TransformParameters" in nodes[node]:
                parameters = nodes[node]["TransformParameters"]
            elif "TranformParameters" in nodes[node]:
                # legacy spelling error in older ITK versions
                parameters = nodes[node]["TranformParameters"]

            fixed_parameters = np.array([])
            if "TransformFixedParameters" in nodes[node]:
                fixed_parameters = nodes[node]["TransformFixedParameters"]
            elif "TranformFixedParameters" in nodes[node]:
                # legacy spelling error in older ITK versions
                fixed_parameters = nodes[node]["TranformFixedParameters"]

            # Always load fixed parameters (they are never large)
            fixed_parameters = fixed_parameters[()]

            # Do not load parameters if nonlinear (can be large)
            if load or xtype not in ("DisplacementFieldTransform", "BSplineTransform"):
                parameters = parameters[()]
            else:
                filename = h5file.filename
                parameters = DelayedH5Array(
                    filename, f"/TransformGroup/{node}/TransformParameters"
                )

            blocks.append(
                ITKStruct(
                    type=nodes[node]["TransformType"][()],
                    precision=prec,
                    ndim_in=ndim_inp,
                    ndim_out=ndim_out,
                    parameters=parameters,
                    fixed_parameters=fixed_parameters,
                )
            )

        obj.transform_group = blocks

        if not keep_open:
            h5file.close()
        return obj

    def _close(self):
        if isinstance(self.file, h5py.File):
            self.file.close()

    def __del__(self):
        self._close()


class DelayedH5Array:
    """
    Class that holds a H5 dataset that can be accessed even if the file
    is closed (i.e., by reopening the file when needed).
    """

    def __init__(self, file: _H5Like, path: str):
        self.file = file
        self.path = path
        self._shape = None
        self._dtype = None

    def to_dataset(self, file: _H5Like | None = None):
        if file is None:
            return self.to_dataset(self.file)
        if isinstance(file, (str, PathLike)):
            with h5py.File(file, "r") as f:
                return self.to_dataset(f)
        if isinstance(file, h5py.File):
            dataset = file[self.path]
            # cache info
            self._shape = dataset.shape
            self._dtype = dataset.dtype
            return dataset
        raise ValueError("Invalid file type")

    def to_array(self, **kwargs):
        import numpy as np
        return np.asarray(self.to_dataset(), **kwargs)

    def to_dask(self, **kwargs):
        import dask.array as da
        return da.from_array(self.to_dataset(), **kwargs)

    def __getitem__(self, index):
        return self.to_dataset()[index]

    def __array__(self, dtype=None):
        return self.to_array(dtype=dtype)

    @property
    def shape(self):
        if self._shape is None:
            self.to_dataset()
        return self._shape

    @property
    def dtype(self):
        if self._dtype is None:
            self.to_dataset()
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        from math import prod
        return prod(self.shape)

    @property
    def nbytes(self):
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
