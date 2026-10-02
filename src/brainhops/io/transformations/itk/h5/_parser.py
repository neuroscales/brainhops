# dependencies
import h5py
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic

# io
from brainhops.io.base.hdf5 import (
    DelayedH5Array,
    HDF5Parser,
    delayed_dataset,
    read_string,
)
from brainhops.io.base.parsers import Confidence, SnifferContentError

# locals
from .._common import ITKStruct, ITKTransformClass

__all__ = ["DelayedH5Array", "H5Header", "H5TransformParser"]


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
    HDF5Parser,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parses an ITK binary (`.h5`) transform file into a chain of
    transform blocks.

    Each block is itself a brainhops transformation, so the parsed blocks
    are stored straight into the `transformations` of the sequence that
    this parser is mixed into.
    """

    file: tx.Optional[h5py.File] = None
    header: H5Header = Factory(H5Header)

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

    # --- from ---------------------------------------------------------

    @classmethod
    def from_h5(
        cls,
        h5file: h5py.File,
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
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
                parameters = delayed_dataset(
                    h5file,
                    f"/TransformGroup/{node}/{parameters_key}",
                    keep_open,
                )

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

        obj.transformations = blocks

        if not keep_open:
            h5file.close()
        return obj

    def _close(self) -> None:
        if isinstance(self.file, h5py.File):
            self.file.close()

    def __del__(self) -> None:
        """Close the underlying HDF5 file, if one is still open."""
        self._close()


def _readstr(dataset: h5py.Dataset) -> str:
    """Read a string from a HDF5 dataset."""
    return read_string(dataset)
