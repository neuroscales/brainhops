"""Writing MRtrix files: the options of the writer, the entries of the
header that describe the storage, the stored voxels, and the files of
each variant.

The MRtrix image and the MRtrix warps write the same kind of file, so
they share these helpers. Each format encodes its own geometry and data
into a record, and these helpers say how that record stores the voxels
and write it.
"""

# stdlib
import gzip
from collections import OrderedDict
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.path import is_same_file, write_file
from brainhops._core.proxy import ArrayProxy
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base.parsers import WriterError

# this format
from ._codecs import (
    default_layout,
    dtype_to_mrtrix,
    format_layout,
    parse_layout,
)
from ._constants import _FILE, _RESERVED
from ._data import (
    _is_scaled,
    _MrtrixProxy,
    _scaled_dtype,
    encode_data,
    whole_file,
)
from ._raw import MrtrixRaw

# ----------------------------------------------------------------------
#   OPTIONS
# ----------------------------------------------------------------------


_OPTIONS = ("layout", "datatype", "scaling", "keyval")
"""The options of the MRtrix writer."""


def writer_options(kwargs: tx.Mapping[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Return the options of the MRtrix writer, refusing unknown ones.

    Parameters
    ----------
    kwargs : mapping
        The keyword arguments given to a writer.

    Returns
    -------
    dict
        The value of each option, `None` when it is not given.

    Raises
    ------
    TypeError
        If an unknown option is given.
    """
    unknown = [name for name in kwargs if name not in _OPTIONS]
    if unknown:
        raise TypeError(
            f"Unknown MRtrix writer option(s): {', '.join(unknown)}"
        )
    return {name: kwargs.get(name) for name in _OPTIONS}


def merged_keyval(
    held: tx.Optional[MrtrixRaw],
    keyval: tx.Optional[tx.Mapping[str, tx.Optional[str]]],
) -> "OrderedDict[str, str]":
    """Return the free-form keys of the base, with `keyval` merged last.

    A value of `None` in `keyval` removes the key.

    Parameters
    ----------
    held : MrtrixRaw, optional
        The record whose keys are the base, or `None` for no base.
    keyval : mapping, optional
        The keys to merge into those of the base.

    Returns
    -------
    OrderedDict
        The merged keys.

    Raises
    ------
    WriterError
        If `keyval` sets a key that describes the storage or the
        geometry, which the data, the geometry and the other options of
        the writer decide.
    """
    keyval = dict(keyval or {})
    refused = sorted(
        key
        for key in keyval
        if key.lower() in _RESERVED or key.lower() == _FILE
    )
    if refused:
        raise WriterError(
            f"The MRtrix header keys {refused} cannot be set with keyval=: "
            f"they follow the data and the geometry, or the layout=, "
            f"datatype= and scaling= options."
        )
    merged = OrderedDict(held.keyval) if held is not None else OrderedDict()
    for key, value in keyval.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


# ----------------------------------------------------------------------
#   STORAGE
# ----------------------------------------------------------------------


def keeps_storage(
    record: MrtrixRaw,
    layout: tx.Optional[tx.Union[str, tx.Sequence[int]]],
    datatype: tx.Optional[tx.Any],
    scaling: tx.Optional[tx.Sequence[float]],
) -> bool:
    """Tell whether the options leave the voxels as a record stores them.

    An option that is not given, or that names what the record stores,
    leaves the voxels as they are.

    Parameters
    ----------
    record : MrtrixRaw
        The record of the stored voxels.
    layout : str or sequence of int, optional
        The `layout=` option of the writer.
    datatype : str or dtype, optional
        The `datatype=` option of the writer.
    scaling : sequence of float, optional
        The `scaling=` option of the writer.

    Returns
    -------
    bool
        Whether the voxels can be copied as the file stores them.
    """
    if layout is not None and writer_strides(
        layout, None, record.ndim
    ) != tuple(record.layout):
        return False
    if datatype is not None and dtype_to_mrtrix(datatype) != dtype_to_mrtrix(
        record.datatype
    ):
        return False
    if scaling is not None:
        stored = None if record.scaling is None else tuple(record.scaling)
        if tuple(float(s) for s in scaling) != stored:
            return False
    return True


def stores_as_recorded(raw: tx.Any, record: MrtrixRaw) -> bool:
    """Tell whether a record says how the voxels of `raw` are stored.

    A lazy array keeps the record of the file that it reads. When other
    metadata is assigned, the record of the metadata may store the voxels
    otherwise, with another shape, layout, type or scaling, and the lazy
    array must then be encoded again rather than copied under the record.
    An array in memory is encoded by the writer, and always agrees.

    Parameters
    ----------
    raw : array-like
        The voxels of an object, lazy or in memory.
    record : MrtrixRaw
        The record of the metadata of the object.

    Returns
    -------
    bool
        Whether the voxels can be written under the record as they are.
    """
    proxy = whole_file(raw)
    if proxy is None:
        return True
    stored = proxy.record
    if stored is record:
        return True
    if tuple(stored.dim) != tuple(record.dim):
        return False
    scaling = None if record.scaling is None else tuple(record.scaling)
    if scaling is None and stored.scaling is not None:
        return False
    return keeps_storage(stored, record.layout, record.datatype, scaling)


def writer_strides(
    layout: tx.Optional[tx.Union[str, tx.Sequence[int]]],
    held: tx.Optional[MrtrixRaw],
    ndim: int,
) -> tx.Tuple[int, ...]:
    """Return the strides to store, from the option or else from the base.

    The layout of the base is used when it has as many axes, and the
    Fortran order `+0,+1,+2,...` otherwise.

    Parameters
    ----------
    layout : str or sequence of int, optional
        The `layout=` option, in the MRtrix spelling (`"-0,-1,+2"`) or as
        signed one-based integers.
    held : MrtrixRaw, optional
        The record whose layout is the default.
    ndim : int
        The number of axes of the stored voxels.

    Returns
    -------
    tuple of int
        The signed one-based strides.
    """
    if layout is None:
        if held is not None and len(held.layout) == ndim:
            return tuple(held.layout)
        return default_layout(ndim)
    if isinstance(layout, str):
        return parse_layout(layout, ndim)
    return parse_layout(format_layout(layout), ndim)


def file_entry(
    held: tx.Optional[MrtrixRaw], file: tx.Optional[str]
) -> tx.Tuple[str, int]:
    """Return the `file` entry of the header to write.

    The entry of the base is kept when it names the same kind of file, so
    that an untouched header is the same. A new single-file header gets
    its offset once its text is known.

    Parameters
    ----------
    held : MrtrixRaw, optional
        The record whose entry is the base.
    file : str, optional
        The data file of the header: `"."` for a single-file image, a
        file name for a `.mih` header, or `None` to keep the entry of the
        base.

    Returns
    -------
    tuple of str and int
        The name of the data file and the offset of the data in it.
    """
    stored = None if held is None else held.file
    if file is None:
        return stored if stored is not None else (".", 0)
    if file == ".":
        return stored if stored is not None and stored[0] == "." else (".", 0)
    return (file, 0)


def placed_record(
    record: MrtrixRaw, held: tx.Optional[MrtrixRaw]
) -> MrtrixRaw:
    """Return the held record when an encoding gives it, else the new one.

    The held record is kept when the encoded record formats to the same
    text, so that an object that was read and not changed keeps the bytes
    of its header. A new single-file record is given the offset of its
    first voxel.

    Parameters
    ----------
    record : MrtrixRaw
        The encoded record, which this function may change.
    held : MrtrixRaw, optional
        The record that the object holds.

    Returns
    -------
    MrtrixRaw
        The record to hold.
    """
    if held is not None and record == held:
        return held
    if record.file[0] == ".":
        record.file = (".", len(record.embedded()))
    return record


def payload(record: MrtrixRaw, source: ArrayProtocol) -> bytes:
    """Return the bytes of the voxels that a header describes.

    A lazy array is copied as the file stores it. Any other array is
    mapped through the intensity scaling of the header and encoded.

    Parameters
    ----------
    record : MrtrixRaw
        The header of the file.
    source : array
        The voxels, in the axis order of the header.

    Returns
    -------
    bytes
        The stored voxels.
    """
    proxy = whole_file(source)
    if proxy is not None:
        return proxy.read_stored()
    if record.scaling and tuple(record.scaling) != (0.0, 1.0):
        offset, scale = record.scaling
        source = (np.asarray(source) - offset) / scale
    return encode_data(record, source)


def stored_voxels(record: MrtrixRaw, source: ArrayProtocol) -> ArrayProtocol:
    """Return voxels as the file of a record stores them.

    The writer maps the voxels through the intensity scaling of the
    record, rounds floating-point values that are stored as integers, and
    converts them to the stored type. Voxels that this changes are
    encoded into memory in the same way, and a lazy array of those bytes
    is returned, which reads them as a file of the record is read. A lazy
    array of a file, and voxels of the stored type without scaling, are
    returned as they are, so that they are not read.

    Parameters
    ----------
    record : MrtrixRaw
        The header of the file.
    source : array
        The voxels, in the axis order of the header.

    Returns
    -------
    array
        The voxels that a file of the record holds.
    """
    if whole_file(source) is not None:
        return source
    stored = _scaled_dtype(record).newbyteorder("=")
    if not _is_scaled(record) and stored == np.dtype(
        source.dtype
    ).newbyteorder("="):
        return source
    content = BytesIO(payload(record, source))
    return _MrtrixProxy.from_record(content, 0, 0, record, mmap=False)


# ----------------------------------------------------------------------
#   FILES
# ----------------------------------------------------------------------


def reads_from(raw: tx.Any, written: tx.Sequence[tx.Any]) -> bool:
    """Tell whether a lazy array reads one of the files that are written.

    Parameters
    ----------
    raw : array
        The `rawdata` array of an object.
    written : sequence of path
        The files that a writer writes.

    Returns
    -------
    bool
        Whether `raw` is a lazy array of one of these files.
    """
    raw = raw.root if isinstance(raw, ArrayProxy) else raw
    if not isinstance(raw, _MrtrixProxy):
        return False
    return any(is_same_file(raw.source, file) for file in written)


def written_files(filename: path.FilenameLike) -> tx.List[tx.Any]:
    """Return the files that writing to a path writes.

    A `.mih` header names its data, which is written next to it in a
    `.dat` file. Any other name is a single file.

    Parameters
    ----------
    filename : path-like
        The path of the `.mif`, `.mif.gz` or `.mih` file.

    Returns
    -------
    list of path
        The header file, and the data file of a `.mih` header.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    if str(filename).lower().endswith(".mih"):
        base = filename.name[: -len(".mih")]
        return [filename, path.Path(filename).parent / (base + ".dat")]
    return [filename]


def written_header(
    record: MrtrixRaw, raw: tx.Optional[ArrayProtocol], file: str
) -> MrtrixRaw:
    """Return the header that a writer writes for a file.

    The header is the record of a consolidated object, whose `file` entry
    names the data file that the writer writes: `"."` for a single-file
    image, or a file name for a `.mih` header. The entry of the record is
    kept when it names the same kind of file, and a copy of the record
    holds another entry.

    Parameters
    ----------
    record : MrtrixRaw
        The record of the consolidated object.
    raw : array, optional
        The voxels of the consolidated object.
    file : str
        The name of the data file, or `"."` for a single-file image.

    Returns
    -------
    MrtrixRaw
        The header to write.

    Raises
    ------
    WriterError
        If the object has no voxels.
    """
    if raw is None:
        raise WriterError(
            "This object has no data, so there is nothing to write."
        )
    entry = file_entry(record, file)
    if tuple(entry) != tuple(record.file or ()):
        record = record.copy()
        record.file = entry
        if record.file[0] == ".":
            record.file = (".", len(record.embedded()))
    return record


def mrtrix_bytes(record: MrtrixRaw, raw: tx.Optional[ArrayProtocol]) -> bytes:
    """Return the bytes of a single-file, uncompressed `.mif` file.

    Parameters
    ----------
    record : MrtrixRaw
        The record of the consolidated object.
    raw : array, optional
        The voxels of the consolidated object.

    Returns
    -------
    bytes
        The header, padded up to the first voxel, and the voxels.

    Raises
    ------
    WriterError
        If the object has no voxels.
    """
    header = written_header(record, raw, ".")
    return header.to_bytes() + payload(header, raw)


def write_mrtrix(
    filename: path.FilenameLike,
    record: MrtrixRaw,
    raw: tx.Optional[ArrayProtocol],
) -> None:
    """Write a consolidated object to a path, in the variant it names.

    A `.mif` file holds the header and the data, gzipped for `.mif.gz`.
    A `.mih` header names its data, written next to it in a `.dat` file.
    The content is encoded before any file is written, and each local
    file is replaced at once, so an object can be written over the files
    that it was read from.

    Parameters
    ----------
    filename : path-like
        The path of the `.mif`, `.mif.gz` or `.mih` file.
    record : MrtrixRaw
        The record of the consolidated object.
    raw : array, optional
        The voxels of the consolidated object.

    Raises
    ------
    WriterError
        If the object has no voxels.
    """
    files = written_files(filename)
    if len(files) == 2:
        header = written_header(record, raw, files[1].name)
        content = header.to_bytes()
        data = payload(header, raw)
        write_file(files[1], data)
        write_file(files[0], content)
        return
    content = mrtrix_bytes(record, raw)
    if str(files[0]).lower().endswith(".gz"):
        content = gzip.compress(content)
    write_file(files[0], content)
