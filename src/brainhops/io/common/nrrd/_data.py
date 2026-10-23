"""NRRD samples: decoding, encoding and locating them, and the lazy array
that reads them."""

# stdlib
import bz2
import gzip
import os
import zlib

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops._core.properties import alias
from brainhops._core.proxy import StoredProxy
from brainhops._core.streams import preserve_position
from brainhops._core.strided import StridedProxy
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
from ._layout import image_order
from ._raw import NrrdRaw

# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def _decode_part(
    file: tx.BinaryIO, start: int, header: NrrdRaw, nbytes: int
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


def _is_path(file: tx.Any) -> bool:
    """Tell whether a source of samples is a path rather than a stream."""
    return isinstance(file, (str, os.PathLike))


def _decode(
    header: NrrdRaw,
    locations: tx.Sequence[tx.Tuple[tx.Any, int]],
    mmap: bool,
) -> np.ndarray:
    """Read the values of a NRRD image as an F-ordered array of file axes.

    Each location is a path or a stream, with the position where its part
    of the samples begins. The values are split evenly between the
    locations. Raw values in a single local file are memory-mapped when
    `mmap` is true.

    Raises
    ------
    ParserContentError
        If the samples cannot be split between the files, or the files
        are too short.
    """
    count, nbytes, dtype = header.count, header.nbytes, header.dtype
    if nbytes % len(locations):
        raise ParserContentError(
            f"The {nbytes} bytes of the NRRD data cannot be split between "
            f"{len(locations)} data files."
        )
    part = nbytes // len(locations)
    single, start = locations[0]
    local = _local_path(single) if _is_path(single) else None
    if (
        mmap
        and len(locations) == 1
        and header.encoding == "raw"
        and local is not None
        and nbytes > 0
    ):
        with open(local, "rb") as f:
            f.seek(start)
            for _ in range(header.line_skip):
                f.readline()
            size = os.fstat(f.fileno()).st_size
            if header.byte_skip == -1:
                first = size - nbytes
            else:
                first = f.tell() + header.byte_skip
        if first < 0 or first + nbytes > size:
            raise ParserContentError(
                f"The NRRD data file {local} is too short for "
                f"{count} values of type {dtype}."
            )
        raw = np.memmap(
            local, dtype=np.uint8, mode="r", offset=first, shape=(nbytes,)
        )
        flat = raw.view(dtype)
    else:
        chunks = []
        for file, start in locations:
            if _is_path(file):
                with _open_path(file) as f:
                    chunks.append(_decode_part(f, start, header, part))
            else:
                with preserve_position(file):
                    chunks.append(_decode_part(file, start, header, part))
            if len(chunks[-1]) < part:
                raise ParserContentError(
                    f"The NRRD data hold {len(chunks[-1])} bytes, but "
                    f"{part} are needed."
                )
        flat = np.frombuffer(b"".join(chunks), dtype=dtype, count=count)
    return flat.reshape(header.sizes, order="F")


def encode_data(header: NrrdRaw, data: tx.Any) -> bytes:
    """Encode an array of file axes as data-file bytes, as the header says.

    Parameters
    ----------
    header : NrrdRaw
        The record that gives the sizes, the type, the byte order and the
        encoding of the samples.
    data : array_like
        The samples, with one axis for each axis of the file.

    Returns
    -------
    bytes
        The content of the data file, encoded as the header says.

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


def _storage(header: NrrdRaw) -> tx.Tuple[tx.Any, ...]:
    """Return what a header says about how the samples are stored."""
    return (
        header.dtype,
        header.sizes,
        header.encoding,
        header.line_skip,
        header.byte_skip,
        header.data_files,
    )


# ----------------------------------------------------------------------
#   LAZY ARRAY
# ----------------------------------------------------------------------


class _NrrdProxy(StoredProxy):
    """A lazy array of the samples of a NRRD image.

    The proxy holds where the header lies, which is a path or the stream
    that the caller passed, the paths of the data files that a detached
    header names, and the record of the image, which says how the
    samples are stored. It reads them each time its values are needed
    and keeps no file open, but a stream given by the caller must stay
    open while the values may be read. The values are presented in the
    axis order of the image, spatial axes first (see
    [`image_order`][brainhops.io.common.nrrd._layout.image_order]),
    whatever the order of the file.

    Before it reads the samples, the proxy reads the header again, so
    that attached samples are found after the header that the file holds
    now. A header that no longer describes the same samples, for example
    because the image was written over its own file with another type,
    is refused rather than misread.

    Samples in a single file that are raw, or compressed with gzip or
    bzip2, and that no line skip precedes, are read in part (see
    [`StridedProxy`][brainhops._core.strided.StridedProxy]): a local file
    of raw samples is memory-mapped when `mmap` is true, and compressed
    samples are decompressed up to the end of the selection. The blocks
    that the proxy gives Dask are then slabs of the axis that the file
    stores slowest. Samples split between several files, encoded as text,
    or preceded by skipped lines are decoded whole for any read.
    """

    # --- magic attributes ----

    source: tx.Any
    """The path of the header file, or the stream that holds it."""

    start: int
    """The position of the header in a stream."""

    files: tx.Tuple[tx.Any, ...]
    """The paths of the data files, which are empty for attached samples."""

    record: NoRepr[NrrdRaw]
    """The record of the image, which describes how samples are stored."""

    mmap: bool
    """Whether raw samples in a single local file are memory-mapped."""

    # --- properties ----

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the image, read from the record."""
        sizes = self.record.sizes
        return tuple(sizes[i] for i in image_order(self.record))

    dtype = alias("record.dtype", readonly=True)
    """The type of the samples, with the byte order of the file."""

    @property
    def copyable(self) -> bool:
        """Whether the samples lie in one file that can be copied as it is.

        Samples split between several data files are decoded and encoded
        again by a writer, which stores them in one file.
        """
        return len(self.files) <= 1

    @property
    def partial(self) -> bool:
        """Whether the samples can be read in part.

        They can when they lie in one file, are raw or compressed with
        gzip or bzip2, and follow the header without skipped lines, so
        that each sample lies at a known position of the decoded stream.
        A byte skip of -1 counts from the end of the file, which is known
        only for raw samples.
        """
        record = self.record
        if len(self.files) > 1 or record.line_skip:
            return False
        if record.encoding == "raw":
            return True
        return record.encoding in _PARTIAL_ENCODINGS and record.byte_skip >= 0

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape, or `None` for samples that are decoded whole."""
        if not self.partial:
            return None
        return self._strided(self.source, self.start, 0).chunks

    # --- methods ----

    def _locations(self) -> tx.List[tx.Tuple[tx.Any, int]]:
        """Return where the samples lie, after checking the header again.

        Raises
        ------
        ParserContentError
            If the header no longer describes the samples of the record.
        """
        if _is_path(self.source):
            with _open_path(self.source) as f:
                current = NrrdRaw.from_fileobj(f)
        else:
            with preserve_position(self.source):
                self.source.seek(self.start)
                current = NrrdRaw.from_fileobj(self.source)
        if _storage(current) != _storage(self.record):
            raise ParserContentError(
                f"The NRRD header {self.source} no longer describes the "
                f"samples that this image was read with, so they cannot be "
                f"read. The file was probably written again since then."
            )
        if self.files:
            return [(file, 0) for file in self.files]
        return [(self.source, self.start + len(current.stored))]

    def read_stored(self) -> bytes:
        """Return the bytes of the samples as the file stores them.

        The bytes run from the start of the samples, before any line or
        byte that the header skips, to the end of the file, so that a
        copy keeps the skips and the encoding of the file.

        Raises
        ------
        ParserContentError
            If the samples lie in several data files, or the header no
            longer describes them.
        """
        if not self.copyable:
            raise ParserContentError(
                "The NRRD samples lie in several data files, so they cannot "
                "be copied as one file."
            )
        ((file, start),) = self._locations()
        if _is_path(file):
            with _open_path(file) as f:
                f.seek(start)
                return f.read()
        with preserve_position(file):
            file.seek(start)
            return file.read()

    def read_samples(self) -> np.ndarray:
        """Return all the samples in the axis order of the image.

        Raises
        ------
        ParserContentError
            If the files are too short, or the header no longer describes
            the samples.
        """
        values = _decode(self.record, self._locations(), self.mmap)
        return values.transpose(image_order(self.record))

    def _strided(self, file: tx.Any, start: int, skip: int) -> StridedProxy:
        """The strided reader of the samples that lie in `file`."""
        return StridedProxy.contiguous(
            file,
            self.record.dtype,
            self.record.sizes,
            offset=skip,
            order="F",
            axes=image_order(self.record),
            start=start,
            mmap=self.mmap,
        )

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        if not self.partial:
            return self.read_samples()[index]
        ((file, start),) = self._locations()
        skip = self.record.byte_skip
        if skip == -1:
            # The samples end the file, so they start before its end by
            # their own length.
            start, skip = _end_start(file, self.record.nbytes), 0
        try:
            return self._strided(file, start, skip)._read(index)
        except ValueError as error:
            raise ParserContentError(
                f"The NRRD data are shorter than the {self.record.count} "
                f"values of type {self.record.dtype} that the header asks "
                f"for. {error}"
            ) from error


_PARTIAL_ENCODINGS = ("raw", "gzip", "bzip2")
"""The encodings whose samples lie at known positions once decoded."""


def _end_start(file: tx.Any, nbytes: int) -> int:
    """Return the position of the last `nbytes` bytes of a raw file."""
    if _is_path(file):
        with _open_path(file) as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
    else:
        with preserve_position(file):
            file.seek(0, os.SEEK_END)
            size = file.tell()
    if size < nbytes:
        raise ParserContentError(
            f"The NRRD data file {file} holds {size} bytes, fewer than the "
            f"{nbytes} that the header asks for."
        )
    return size - nbytes


def read_nrrd(
    file: tx.Any, mmap: bool = True
) -> tx.Tuple[NrrdRaw, _NrrdProxy]:
    """Read the record and a lazy array of the samples of a NRRD image.

    The file is a path or a stream. The record is read from the header,
    and the samples are not read. The data files that a detached header
    names are found next to the header file, so a stream must then have a
    name, and they are checked to exist. A single local file of raw
    samples is checked to be long enough.

    Parameters
    ----------
    file : path-like or file object
        The path of the header, or a stream open in binary mode.
    mmap : bool, default=True
        Whether to memory-map a single local file of raw samples.

    Returns
    -------
    raw : NrrdRaw
        The record.
    proxy : array
        A lazy array of the samples, with one axis for each axis of the
        file.

    Raises
    ------
    ParserExistsError
        If the path or a data file does not exist.
    ParserContentError
        If the content is not a NRRD image, the data files cannot be
        found from a stream without a name, or a local file is too short.
    """
    if isinstance(file, str):
        file = path.Path(file)
    if _is_path(file):
        if not path.exists(file):
            raise ParserExistsError(f"No such file: {file}")
        with _open_path(file) as f:
            record = NrrdRaw.from_fileobj(f)
        origin, start = file, 0
    else:
        start = file.tell()
        record = NrrdRaw.from_fileobj(file)
        origin = getattr(file, "name", None)
    files = []
    for name in record.data_files:
        if os.path.isabs(name):
            datafile = path.Path(name)
        elif isinstance(origin, (str, bytes, os.PathLike)):
            if isinstance(origin, bytes):
                origin = os.fsdecode(origin)
            datafile = _sibling(origin, name)
        else:
            raise ParserContentError(
                f"The NRRD header names a separate data file ({name!r}), "
                f"which cannot be found from a stream that has no file "
                f"name. Read it from its path instead."
            )
        if not path.exists(datafile):
            raise ParserExistsError(
                f"No such file: {datafile} (a data file named by the NRRD "
                f"header {origin})"
            )
        files.append(datafile)
    _check_size(record, file if not files else None, files)
    return record, _NrrdProxy(file, start, files, record, mmap)


def _check_size(
    record: NrrdRaw, source: tx.Any, files: tx.Sequence[tx.Any]
) -> None:
    """Check that a single local file of raw samples is long enough.

    The size is compared with the samples and the bytes that the header
    skips, but not with the lines that it skips, which have no fixed
    length.

    Raises
    ------
    ParserContentError
        If the file is too short.
    """
    if record.encoding != "raw" or len(files) > 1:
        return
    if files:
        local, offset = _local_path(files[0]), 0
    elif _is_path(source):
        local, offset = _local_path(source), len(record.stored)
    else:
        return
    if local is None:
        return
    needed = offset + max(record.byte_skip, 0) + record.nbytes
    size = os.path.getsize(local)
    if size < needed:
        raise ParserContentError(
            f"The NRRD data file {local} holds {size} bytes, but the header "
            f"asks for at least {needed}."
        )
