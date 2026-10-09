"""NRRD data: reading and encoding the sample values."""

# stdlib
import bz2
import gzip
import os
import zlib
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base._utils_files import sibling as _sibling
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserExistsError,
    WriterError,
)

# this format
from ._codecs import _format_float
from ._header import NrrdHeader

# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def _decode_part(
    file: tx.BinaryIO, start: int, header: NrrdHeader, nbytes: int
) -> bytes:
    """Read and decode the values of one data file from `start`."""
    file.seek(start)
    for _ in range(header.line_skip):
        file.readline()
    skip = header.byte_skip
    encoding = header.encoding
    if encoding == "raw":
        if skip == -1:
            file.seek(-nbytes, os.SEEK_END)
        else:
            file.seek(skip, os.SEEK_CUR)
        return file.read(nbytes)
    content = file.read()
    if encoding == "gzip":
        content = _gunzip(content)
    elif encoding == "bzip2":
        content = bz2.decompress(content)
    if skip == -1:
        if encoding not in ("gzip", "bzip2"):
            raise ParserContentError(
                f"A NRRD byte skip of -1 needs a binary encoding, not "
                f"{encoding!r}."
            )
        return content[len(content) - nbytes :]
    content = content[skip:]
    if encoding == "ascii":
        dtype = header.dtype
        tokens = content.decode("ascii").replace(",", " ").split()
        count = nbytes // dtype.itemsize
        values = np.asarray(
            tokens[:count], dtype=float if dtype.kind == "f" else np.int64
        )
        return values.astype(dtype).tobytes()
    if encoding == "hex":
        text = b"".join(content.split()).decode("ascii")
        return bytes.fromhex(text[: 2 * nbytes])
    return content[:nbytes]


def _gunzip(content: bytes) -> bytes:
    """Decompress gzip members, ignoring trailing data."""
    out = []
    while content[:2] == b"\x1f\x8b":
        decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
        out.append(decomp.decompress(content))
        content = decomp.unused_data
    if not out:
        raise ParserContentError("The NRRD data are not gzip-compressed.")
    return b"".join(out)


def read_data(
    header: NrrdHeader,
    file: tx.Any,
    offset: int,
    mmap: bool = True,
) -> np.ndarray:
    """Read the values of a NRRD image as an F-ordered array.

    `file` is the path of the header, against which relative data-file names
    are resolved, or the file content as bytes. `offset` is the position of the
    first byte after the header.

    Raises
    ------
    ParserExistsError
        If a data file does not exist.
    ParserContentError
        If detached data are read without a path, or the data are too short.
    """
    count, nbytes, dtype = header.count, header.nbytes, header.dtype
    if header.data_files:
        sources = []
        for name in header.data_files:
            if os.path.isabs(name):
                datafile = path.Path(name)
            elif isinstance(file, (bytes, bytearray)) or file is None:
                raise ParserContentError(
                    f"The NRRD header names a separate data file "
                    f"({name!r}), which cannot be found from a stream "
                    f"that has no file name. Read it from its path instead."
                )
            else:
                datafile = _sibling(file, name)
            if not path.exists(datafile):
                raise ParserExistsError(
                    f"No such file: {datafile} (a data file named by the "
                    f"NRRD header {file})"
                )
            sources.append((datafile, 0))
    else:
        sources = [(file, offset)]
    if nbytes % len(sources):
        raise ParserContentError(
            f"The {nbytes} bytes of the NRRD data cannot be split between "
            f"{len(sources)} data files."
        )
    part = nbytes // len(sources)

    if (
        mmap
        and len(sources) == 1
        and header.encoding == "raw"
        and not isinstance(sources[0][0], (bytes, bytearray))
        and _local_path(sources[0][0]) is not None
        and nbytes > 0
    ):
        local = _local_path(sources[0][0])
        with open(local, "rb") as f:
            f.seek(sources[0][1])
            for _ in range(header.line_skip):
                f.readline()
            if header.byte_skip == -1:
                start = os.fstat(f.fileno()).st_size - nbytes
            else:
                start = f.tell() + header.byte_skip
            size = os.fstat(f.fileno()).st_size
        if start < 0 or start + nbytes > size:
            raise ParserContentError(
                f"The NRRD data file {local} is too short for "
                f"{count} values of type {dtype}."
            )
        raw = np.memmap(
            local, dtype=np.uint8, mode="r", offset=start, shape=(nbytes,)
        )
        flat = raw.view(dtype)
    else:
        chunks = []
        for source, start in sources:
            if isinstance(source, (bytes, bytearray)):
                chunks.append(
                    _decode_part(BytesIO(source), start, header, part)
                )
            else:
                with _open_path(source) as f:
                    chunks.append(_decode_part(f, start, header, part))
            if len(chunks[-1]) < part:
                raise ParserContentError(
                    f"The NRRD data hold {len(chunks[-1])} bytes, but "
                    f"{part} are needed."
                )
        flat = np.frombuffer(b"".join(chunks), dtype=dtype, count=count)
    return flat.reshape(header.sizes, order="F")


def encode_data(header: NrrdHeader, data: tx.Any) -> bytes:
    """Encode an array as data-file bytes, as the header describes.

    Raises
    ------
    WriterError
        If the shape of the array does not match the header.
    """
    array = np.asarray(data)
    if tuple(array.shape) != tuple(header.sizes):
        raise WriterError(
            f"The data have shape {array.shape}, but the header says "
            f"{header.sizes}."
        )
    dtype = header.dtype
    if dtype.kind in "iu" and array.dtype.kind in "fc":
        array = np.round(np.real(array))
    flat = np.asarray(array, dtype=dtype).ravel(order="F")
    encoding = header.encoding
    if encoding == "ascii":
        width = header.sizes[0] if header.sizes else 1
        if dtype.kind == "f":
            words = [_format_float(v) for v in flat.tolist()]
        else:
            words = [str(v) for v in flat.tolist()]
        rows = [
            " ".join(words[i : i + width]) for i in range(0, len(words), width)
        ]
        return ("\n".join(rows) + "\n").encode("ascii")
    raw = flat.tobytes()
    if encoding == "hex":
        text = raw.hex()
        return (
            "\n".join(text[i : i + 64] for i in range(0, len(text), 64)) + "\n"
        ).encode("ascii")
    if encoding == "gzip":
        return gzip.compress(raw)
    if encoding == "bzip2":
        return bz2.compress(raw)
    return raw
