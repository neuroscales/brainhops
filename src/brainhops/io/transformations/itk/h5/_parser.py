# dependencies
import h5py
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic

# io
from brainhops.datamodel.metadata._field import MetadataField
from brainhops.io.base.hdf5 import (
    DelayedH5Array,
    Hdf5Parser,
    delayed_dataset,
    read_string,
)
from brainhops.io.base.parsers import Confidence, SnifferContentError
from brainhops.io.metadata._sync import parent_post_init, sync_metadata

# locals
from .._common import ItkStruct, ItkTransformClass, _application_order
from .._metadata import H5Header, ItkH5Metadata

__all__ = ["DelayedH5Array", "H5Header", "H5TransformParser", "read_h5_header"]


def read_h5_header(h5file: h5py.File) -> H5Header:
    """
    Read the root header of an open ITK `.h5` file.

    Parameters
    ----------
    h5file : h5py.File
        The open file.

    Returns
    -------
    H5Header
        The versions recorded at the root of the file.
    """
    header = H5Header()
    for name in ("HDFVersion", "ITKVersion", "OSName", "OSVersion"):
        if f"/{name}" in h5file:
            setattr(header, name, _readstr(h5file[f"/{name}"]))
    return header


class H5TransformParser(
    Magic,
    Hdf5Parser,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parses an ITK binary (`.h5`) transform file into a chain of
    transform blocks.

    The blocks of a `CompositeTransform` (such as ANTs'
    `<prefix>Composite.h5`) are listed in the order they apply to
    points, which is the reverse of their order in the file (ITK applies
    the last block of a composite first).

    Each block is itself a brainhops transformation, so the parsed blocks
    are stored straight into the `transformations` of the sequence that
    this parser is mixed into.
    """

    file: tx.Optional[h5py.File] = None
    header: H5Header = Factory(H5Header)

    metadata: MetadataField[
        ItkH5Metadata,
        Factory(),
        tx.Doc(
            """
            The metadata of the file: the version of ITK that wrote it
            (`generated_by`), with the root header as its record. The
            blocks have none of their own. See
            [`ItkH5Metadata`][brainhops.io.transformations.itk.ItkH5Metadata].
            """
        ),
    ]

    def __post_init__(self, arguments: tx.Any = None) -> None:
        parent_post_init(super(), arguments)
        sync_metadata(self, ItkH5Metadata, self.header, image=self)

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
        position: tx.Optional[int] = None,
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
        position : int, optional
            Which top-level transform of the file to read: the
            composite, if the file starts with a `CompositeTransform`
            header, else one of its blocks. By default, the first one,
            with a warning if the file holds several.

        Returns
        -------
        obj
            The parsed object.
        """
        header = read_h5_header(h5file)

        obj = cls(header=header, file=h5file if keep_open else None)
        nodes = h5file.get("/TransformGroup", {})

        blocks = []
        composites = []
        # ITK names the groups after their position, `0`, `1`, ...,
        # and reads them by number; h5py lists them by name, which
        # would put `10` before `2`.
        for index, node in enumerate(sorted(nodes, key=_node_number)):
            # Parse transform type
            xtype = _readstr(nodes[node]["TransformType"])
            xtype, prec, ndim_inp, ndim_out = xtype.split("_")
            xtype = ItkTransformClass(xtype)
            ndim_inp, ndim_out = int(ndim_inp), int(ndim_out)

            if xtype == "CompositeTransform":
                # A composite header has no parameters of its own: its
                # queue is the blocks that follow it.
                composites.append(index)
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
        """Close the underlying HDF5 file, if one is still open."""
        self._close()


def _node_number(name: str) -> tx.Tuple[int, tx.Union[int, str]]:
    """Sort key of a `/TransformGroup` child: numbers first, by value."""
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def _readstr(dataset: h5py.Dataset) -> str:
    """Read a string from a HDF5 dataset."""
    return read_string(dataset)
