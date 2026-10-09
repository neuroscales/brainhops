import h5py
import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Factory, Magic

from brainhops.io.base.parsers import Confidence, SnifferContentError

# io
from brainhops.io.common.hdf5 import DelayedH5Array, Hdf5Reader
from brainhops.io.common.hdf5._delayed import delayed_dataset
from brainhops.io.common.hdf5._parsers import read_string

from .._common import ItkStruct, ItkTransformClass, _application_order

__all__ = ["DelayedH5Array", "H5Header", "H5TransformReader"]


class H5Header(
    Magic,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Provenance strings stored at the root of an ITK HDF5 file."""

    HDFVersion: tx.Optional[str] = None
    """The HDF5 library version, such as `"HDF5 library version: 1.10.4"`."""

    ITKVersion: tx.Optional[str] = None
    """The ITK library version, such as `"5.1.0"`."""

    OSName: tx.Optional[str] = None
    """The operating system name, such as `"Linux"`."""

    OSVersion: tx.Optional[str] = None
    """The operating system version, such as `"6.1.0-1007-oem"`."""


class H5TransformReader(
    Magic,
    Hdf5Reader,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parser for ITK HDF5 (`.h5`) transform files.

    Every block is parsed into a transformation and stored directly in
    `transformations`, with the blocks of a composite in application order.
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
        """Confidence that an open HDF5 file is an ITK transform file."""
        # ITK records its version at the root of the file.
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
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """Parse an open HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            The open HDF5 file.
        keep_open : bool, default=False
            Whether to keep the file open after parsing, instead of closing it.
        load : bool, default=True
            Whether to load warp parameters into memory, instead of reading
            them lazily from disk.
        position : int, optional
            The top-level transform to read: the composite when the file has
            one, otherwise one of its blocks. By default the first one is read,
            with a warning when there are several.

        Returns
        -------
        Self
            The parsed object.
        """
        header = H5Header()
        if "/HDFVersion" in h5file:
            header.HDFVersion = read_string(h5file["/HDFVersion"])
        if "/ITKVersion" in h5file:
            header.ITKVersion = read_string(h5file["/ITKVersion"])
        if "/OSName" in h5file:
            header.OSName = read_string(h5file["/OSName"])
        if "/OSVersion" in h5file:
            header.OSVersion = read_string(h5file["/OSVersion"])

        obj = cls(header=header, file=h5file if keep_open else None)
        nodes = h5file.get("/TransformGroup", {})

        blocks = []
        composites = []
        # ITK reads the groups by number, but h5py lists them by name, which
        # puts `10` before `2`.
        for index, node in enumerate(sorted(nodes, key=_node_number)):
            xtype = read_string(nodes[node]["TransformType"])
            xtype, prec, ndim_inp, ndim_out = xtype.split("_")
            xtype = ItkTransformClass(xtype)
            ndim_inp, ndim_out = int(ndim_inp), int(ndim_out)

            if xtype == "CompositeTransform":
                # A composite header has no parameters; its queue is the next
                # blocks.
                composites.append(index)
                continue

            parameters = np.array([])
            parameters_key = None
            if "TransformParameters" in nodes[node]:
                parameters_key = "TransformParameters"
                parameters = nodes[node]["TransformParameters"]
            elif "TranformParameters" in nodes[node]:
                # Misspelling written by older ITK versions.
                parameters_key = "TranformParameters"
                parameters = nodes[node][parameters_key]

            fixed_parameters = np.array([])
            if "TransformFixedParameters" in nodes[node]:
                fixed_parameters_key = "TransformFixedParameters"
                fixed_parameters = nodes[node]["TransformFixedParameters"]
            elif "TranformFixedParameters" in nodes[node]:
                # Misspelling written by older ITK versions.
                fixed_parameters_key = "TranformFixedParameters"
                fixed_parameters = nodes[node][fixed_parameters_key]

            # Fixed parameters are small and always loaded.
            fixed_parameters = fixed_parameters[()]

            # Warp parameters can be large, so they are read lazily on request.
            # A group without a parameters dataset gives empty parameters,
            # whether or not the data are loaded.
            LARGE_TYPES = ("DisplacementFieldTransform", "BSplineTransform")
            if load or xtype not in LARGE_TYPES or parameters_key is None:
                parameters = parameters[()]
            else:
                parameters = delayed_dataset(
                    h5file,
                    f"/TransformGroup/{node}/{parameters_key}",
                    keep_open,
                )

            blocks.append(
                ItkStruct(
                    type=xtype,
                    precision=prec,
                    ndim_input=ndim_inp,
                    ndim_output=ndim_out,
                    parameters=parameters,
                    fixed_parameters=fixed_parameters,
                )
            )

        obj.transformations = _application_order(blocks, composites, position)

        if not keep_open:
            h5file.close()
        return obj

    def _close(self) -> None:
        if isinstance(self.file, h5py.File):
            self.file.close()

    def __del__(self) -> None:
        """Close the HDF5 file if it is still open."""
        self._close()


def _node_number(name: str) -> tx.Tuple[int, tx.Union[int, str]]:
    """Sort key that puts numeric group names first, by value."""
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)
