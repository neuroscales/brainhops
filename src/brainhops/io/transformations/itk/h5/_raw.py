"""The record of an ITK HDF5 (`.h5`) file, which holds its root and blocks."""

__all__ = ["H5Header", "H5Raw"]

import os.path as op

import h5py
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, HideIfNone, Magic

from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._internal_api import Raw
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
)

# io
from brainhops.io.common.hdf5 import H5ArrayProxy, Hdf5ReaderWriter
from brainhops.io.common.hdf5._internal_api import read_string
from brainhops.io.transformations.itk._internal_api import (
    TRANSFORM_TYPE_RE,
    check_composite,
    is_composite,
    read_only,
    type_fields,
    type_name,
)

from .._common import ItkPrecision, ItkStruct, ItkTransformClass

TRANSFORM_GROUP = "TransformGroup"

# The warps hold one value per voxel or per control point, so their
# parameters are read when they are used.
_LARGE_TYPES = (
    ItkTransformClass.DisplacementFieldTransform,
    ItkTransformClass.BSplineTransform,
)


class H5Header(Magic, convert=True, repr=HideIfNone):
    """Provenance strings stored at the root of an ITK HDF5 file."""

    HDFVersion: tx.Optional[str] = None
    """The HDF5 library version, such as `"HDF5 library version: 1.10.4"`."""

    ITKVersion: tx.Optional[str] = None
    """The ITK library version, such as `"5.1.0"`."""

    OSName: tx.Optional[str] = None
    """The operating system name, such as `"Linux"`."""

    OSVersion: tx.Optional[str] = None
    """The operating system version, such as `"6.1.0-1007-oem"`."""


class H5Raw(Raw, Hdf5ReaderWriter):
    """Record of an ITK HDF5 (`.h5`) file, which holds its root and blocks.

    An ITK `.h5` file records the versions of the libraries that wrote it
    at its root, which the record holds as an [`H5Header`][], and it
    stores its blocks as the numbered groups of `/TransformGroup`, which
    the record holds in `blocks`, in the order of the file, as
    [`ItkStruct`][brainhops.io.transformations.itk.ItkStruct] blocks. A
    file that holds a composite transformation starts with a
    `CompositeTransform` header, which the record keeps as its first
    block.

    The parameters of a warp, which are large, are held as a
    [`H5ArrayProxy`][brainhops.io.common.hdf5.H5ArrayProxy], which
    opens the file again by its name when the values are needed, so the
    record never keeps an open file. The other parameters are read at
    once and are read-only, because the blocks of a record are also the
    elements of the chain that a transformation decodes, and a record is
    never changed in place.

    The metadata of an `.h5` file,
    [`H5Metadata`][brainhops.io.transformations.itk.h5.H5Metadata], holds a
    record, and an
    [`H5Transform`][brainhops.io.transformations.itk.h5.H5Transform]
    decodes its chain from the record of its metadata. The record is
    written as ITK writes an `.h5` file (see [`to_h5`][]).
    """

    header: H5Header = Factory(H5Header)
    """The provenance strings at the root of the file."""

    blocks: tx.Tuple[ItkStruct, ...] = ()
    """Every block of `/TransformGroup`, in the order of the file.

    A `CompositeTransform` header can only be the first block.
    """

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score an open HDF5 file as an ITK transform file.

        ITK records its version at the root of the file.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an ITK
            transform file.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if "ITKVersion" in h5file.keys():
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error("HDF5 file is not an ITK transform file")
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_h5(cls, h5file: h5py.File, load: bool = False) -> "H5Raw":
        """Read the record of an open ITK HDF5 file.

        The groups of `/TransformGroup` are read in the order of their
        numbers, as ITK reads them. The parameters of a warp are read when
        they are used, from the file at the name by which it was opened,
        unless `load` is true or the file was opened from a stream, which
        has no name.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        load : bool, default=False
            Whether to read the parameters of the warps into memory at
            once, rather than when they are used.

        Returns
        -------
        H5Raw
            The record of the file.

        Raises
        ------
        ParserContentError
            If a composite header is not the first block.
        """
        header = H5Header()
        for name in ("HDFVersion", "ITKVersion", "OSName", "OSVersion"):
            if f"/{name}" in h5file:
                setattr(header, name, read_string(h5file[f"/{name}"]))
        source = None if load else _source(h5file)
        groups = h5file.get(f"/{TRANSFORM_GROUP}", {})
        # ITK reads the groups by number, but h5py lists them by name,
        # which puts `10` before `2`.
        blocks = [
            _read_block(groups[name], source)
            for name in sorted(groups, key=_group_number)
        ]
        check_composite(blocks)
        return cls(header=header, blocks=tuple(blocks))

    # --- to -----------------------------------------------------------

    @takes_error_policy
    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write the record into an empty HDF5 file open for writing.

        The file is written as `itk::HDF5TransformIOTemplate::Write`
        writes it. The strings of the header are datasets at the root of
        the file, and each block is the group of `/TransformGroup` named
        after its number in the file. A group holds the type of its block
        as a string, and its parameters and fixed parameters as vectors
        of numbers. The parameters are stored in the precision of the
        block, and the fixed parameters in double precision. A
        `CompositeTransform` header holds its type only. ITK records its
        own version at the root of the file, and the reader of brainhops
        recognises an ITK file by it, so a header without `ITKVersion`
        is written with a version that names brainhops instead.

        Parameters
        ----------
        h5file : h5py.File
            The file, empty and open for writing.
        **kwargs : Any
            Options of the writer. There are none, so any option is
            refused.

        Raises
        ------
        TypeError
            If an option is given.
        """
        self._h5_writer(**kwargs)(h5file)

    def _h5_writer(self, **kwargs) -> tx.Callable[[h5py.File], None]:
        # The parameters of a warp are read before the file is opened, so
        # that a file written over the one that they are read from is not
        # truncated first.
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.to_h5() takes no options, but was "
                f"given {', '.join(sorted(kwargs))}."
            )
        header = {
            name: getattr(self.header, name)
            for name in ("HDFVersion", "ITKVersion", "OSName", "OSVersion")
        }
        if header["ITKVersion"] is None:
            header["ITKVersion"] = _WRITER
        if header["HDFVersion"] is None:
            header["HDFVersion"] = (
                f"HDF5 library version: {h5py.version.hdf5_version}"
            )
        groups = [_group_content(block) for block in self.blocks]

        def write(h5file: h5py.File) -> None:
            strings = h5py.string_dtype()
            for name, value in header.items():
                if value is not None:
                    h5file.create_dataset(name, data=[value], dtype=strings)
            root = h5file.create_group(TRANSFORM_GROUP)
            for index, content in enumerate(groups):
                group = root.create_group(str(index))
                kind, parameters, fixed_parameters = content
                group.create_dataset(
                    "TransformType", data=[kind], dtype=strings
                )
                if parameters is None:
                    continue
                group.create_dataset("TransformParameters", data=parameters)
                group.create_dataset(
                    "TransformFixedParameters", data=fixed_parameters
                )

        return write


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _read_block(group: h5py.Group, source: tx.Optional[str]) -> ItkStruct:
    """Read the block of one group of `/TransformGroup`.

    A group without a parameters dataset gives empty parameters. Older ITK
    versions misspelled the datasets `TranformParameters` and
    `TranformFixedParameters`, and both spellings are read.
    """
    kind = read_string(group["TransformType"])
    match = TRANSFORM_TYPE_RE.match(kind)
    if not match:
        raise ParserContentError(f"Malformed ITK transform type: {kind!r}.")
    fields = type_fields(match)
    kind = fields["type"]
    parameters = _dataset(group, "TransformParameters", "TranformParameters")
    fixed_parameters = _dataset(
        group, "TransformFixedParameters", "TranformFixedParameters"
    )
    if parameters is None:
        parameters = np.array([])
    elif source is None or kind not in _LARGE_TYPES:
        parameters = parameters[()]
    else:
        parameters = H5ArrayProxy(source, parameters.name)
    if fixed_parameters is None:
        fixed_parameters = np.array([])
    else:
        fixed_parameters = fixed_parameters[()]
    return ItkStruct(
        **fields,
        parameters=read_only(parameters),
        fixed_parameters=read_only(fixed_parameters),
    )


_WRITER = "unknown, written by brainhops"
"""The `ITKVersion` of a file whose record does not name a version."""


def _group_content(
    block: ItkStruct,
) -> tx.Tuple[str, tx.Optional[np.ndarray], tx.Optional[np.ndarray]]:
    """Return the type and the vectors that the group of a block holds.

    The vectors are read into memory, in the precision that ITK writes
    them in. A composite header has no vectors.
    """
    kind = type_name(block)
    if is_composite(block):
        return kind, None, None
    float32 = ItkPrecision(block.precision) == ItkPrecision.Float
    dtype = np.float32 if float32 else np.float64
    parameters = np.asarray(block.parameters, dtype=np.float64).reshape(-1)
    fixed = np.asarray(block.fixed_parameters, dtype=np.float64).reshape(-1)
    return kind, parameters.astype(dtype), fixed


def _dataset(
    group: h5py.Group, name: str, misspelled: str
) -> tx.Optional[h5py.Dataset]:
    """Return a dataset of a group under its name or its old misspelling."""
    for key in (name, misspelled):
        if key in group:
            return group[key]
    return None


def _source(h5file: h5py.File) -> tx.Optional[str]:
    """Return the path by which the datasets of a file can be read again.

    A file opened from a stream has no path, and `None` is returned, so
    that its datasets are read at once. No open file is ever kept.
    """
    if h5file.driver == "fileobj":
        return None
    return op.abspath(h5file.filename)


def _group_number(name: str) -> tx.Tuple[int, tx.Union[int, str]]:
    """Sort key that puts numeric group names first, by value."""
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)
