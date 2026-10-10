"""The record of an MGH file, and reading and writing whole files."""

# stdlib
import gzip
from io import BytesIO
from urllib.parse import urlsplit

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, Magic
from nibabel.arrayproxy import ArrayProxy
from nibabel.freesurfer import mghformat as _mgh
from nibabel.volumeutils import array_to_file

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed, preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    SnifferContentError,
)
from brainhops.io.common.freesurfer._geometry import (
    FS_DEFAULT_XRAS,
    FS_DEFAULT_YRAS,
    FS_DEFAULT_ZRAS,
    fs_vox2ras,
    fs_vox2tkr,
)
from brainhops.io.common.nifti._files import _is_local

# this format
from ._constants import (
    _GEOMETRY_FIELDS,
    _MGH_TYPES,
    _MRI_PARAMS,
    MGH_FIELDS_SIZE,
    MGH_FOOTER_SIZE,
    MGH_HEADER_SIZE,
)
from ._utils import _read_prefix, _seekable, _valid_prefix

_Geometry = tx.Tuple[
    tx.Tuple[float, float, float],
    tx.Tuple[float, float, float],
    tx.Tuple[float, float, float],
    tx.Tuple[float, float, float],
    tx.Tuple[float, float, float],
]


class MghRaw(Magic, BinaryFileReader, BinaryFileWriter):
    """Header, footer and tags of an MGH file, as the file stores them.

    An MGH file stores a fixed header before its voxels and, after the
    voxels, a footer of acquisition parameters followed by optional tags,
    such as the command history. The record holds everything but the
    voxels: the fixed header and the footer in a nibabel header, and the
    tags as the bytes that the file stores. The fields are kept as they
    are stored, including a `goodRASFlag` that is not positive, and the
    properties of the record decode them as FreeSurfer does.

    A record is held by
    [`MghMetadata`][brainhops.io.common.mgh.MghMetadata], which never
    changes it in place. A writer works on the copy that [`copy`][]
    returns.

    Reading a record never decodes the voxels. In an `.mgh` file, the
    reader seeks over the voxels to the footer. In an `.mgz` file, which
    is compressed as a single gzip stream, reaching the footer means
    decompressing the voxels, which are then discarded. Writing a record
    alone writes the fixed header only, because the footer and the tags
    belong after the voxels.
    """

    header: tx.Annotated[_mgh.MGHHeader, Factory(_mgh.MGHHeader)]
    """The fixed header and the footer, as a nibabel MGH header.

    A new record holds the header of a 1 mm volume with one voxel.
    """

    tags: bytes = b""
    """The bytes that follow the footer, written back as they are."""

    def copy(self) -> "MghRaw":
        """Return a copy of the record that can be changed freely.

        The header is copied, and its geometry is set again on the copy,
        because nibabel replaces the geometry of a header whose
        `goodRASFlag` is 0 when it copies it.

        Returns
        -------
        MghRaw
            The copy.
        """
        header = _keep_stored_geometry(self.header.copy(), self.header)
        return MghRaw(header=header, tags=self.tags)

    # --- decoded fields -----------------------------------------------

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the volume, `(x, y, z)` or `(x, y, z, frames)`."""
        return tuple(int(d) for d in self.header.get_data_shape())

    @property
    def good_ras(self) -> bool:
        """Whether the stored voxel size, cosines and centre are valid.

        The geometry is valid when the `goodRASFlag` of the header is
        positive. Otherwise, FreeSurfer ignores the stored geometry and
        uses 1 mm voxels, the coronal LIA direction cosines and a zero
        centre, and so do the other properties of the record.
        """
        return int(self.header["goodRASFlag"]) > 0

    @property
    def voxel_size(self) -> tx.Tuple[float, float, float]:
        """The voxel size in millimetres."""
        return _geometry(self.header)[0]

    @property
    def vox2ras(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-scanner-RAS matrix (`mri_info --vox2ras`).

        The matrix equals nibabel's `header.get_vox2ras()`, except that the
        FreeSurfer defaults apply when the `goodRASFlag` is not positive.
        """
        return fs_vox2ras(self.header["dims"][:3], *_geometry(self.header))

    @property
    def vox2tkr(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-tkr-RAS matrix (`mri_info --vox2ras-tkr`)."""
        shape = self.header["dims"][:3]
        return fs_vox2tkr(shape, _geometry(self.header)[0])

    @property
    def mri_params(self) -> tx.Dict[str, float]:
        """The acquisition parameters of the footer.

        The repetition, echo and inversion times are in milliseconds, the
        flip angle in radians, and the field of view in millimetres. A zero
        means that the parameter was not recorded.
        """
        return {name: float(self.header[name]) for name in _MRI_PARAMS}

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an MGH header.

        The stream may be compressed with gzip, and its position is
        restored. A file is recognised from its leading fields, which are
        the version 1, four positive dimensions and a known voxel type, and
        not from its name.

        Parameters
        ----------
        file : file object
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no MGH header.
            With `True`, the error is a `SnifferContentError`, and an
            exception class is raised instead when one is given.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            `Confidence.LIKELY` or `Confidence.NO`.

        Raises
        ------
        SnifferContentError
            If the stream holds no MGH header and `error` is `True`.
        """
        try:
            with preserve_position(file):
                prefix = _read_prefix(open_compressed(file))
        except Exception:
            prefix = None
        if _valid_prefix(prefix):
            return Confidence.LIKELY
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Content is not a valid MGH/MGZ file")
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes start with an MGH header.

        The bytes are read as a stream with [`sniff_fileobj`][].
        """
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MghRaw":
        """Read the header, the footer and the tags of an MGH stream.

        The stream may be compressed with gzip, and a stream that cannot
        seek is first read into memory. The voxels are skipped, and a
        stream that ends before the footer, such as a file cut after its
        header, gives a footer of zeros and no tags. The keyword arguments
        are ignored.

        Raises
        ------
        ParserContentError
            If the stream does not start with an MGH header.
        """
        if not _seekable(file):
            file = BytesIO(file.read())
        stream = open_compressed(file)
        start = stream.tell()
        head = stream.read(MGH_FIELDS_SIZE)
        prefix = _read_prefix(BytesIO(head))
        if not _valid_prefix(prefix) or len(head) < MGH_FIELDS_SIZE:
            raise ParserContentError("Content is not a valid MGH/MGZ file")
        _version, *dims, code, _dof, _flag = prefix
        voxels = _MGH_TYPES[code] * int(np.prod(dims))
        stream.seek(start + MGH_HEADER_SIZE + voxels)
        footer = stream.read(MGH_FOOTER_SIZE)
        tags = stream.read() or b""
        return cls(header=_stored_header(head, footer), tags=tags)

    # --- writing ------------------------------------------------------

    def to_bytes(self, **kwargs) -> bytes:
        """Return the fixed header as an MGH file stores it.

        The header fields are followed by the unused bytes that complete
        the fixed header, as zeros. The footer and the tags are not
        written, because a file stores them after the voxels.
        """
        buffer = BytesIO()
        self.header.writehdr_to(buffer)
        return buffer.getvalue().ljust(MGH_HEADER_SIZE, b"\0")

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the fixed header to a path.

        The header is compressed with gzip when the name ends with `.mgz`
        or `.gz`. The name is read from the path of the URL, so that a
        query does not hide the suffix.
        """
        content = self.to_bytes(**kwargs)
        if _compressed_name(filename):
            content = gzip.compress(content)
        with path.Path(filename).open("wb") as file:
            file.write(content)


def _stored_header(fields: bytes, footer: bytes) -> _mgh.MGHHeader:
    """Build a nibabel header that holds the fields as the file stores them.

    nibabel pads a short footer with zeros. It also replaces the geometry
    of a header whose `goodRASFlag` is 0 with its own defaults, so the
    stored geometry is set again from the bytes.
    """
    block = fields[:MGH_FIELDS_SIZE] + footer
    header = _mgh.MGHHeader(block, check=False)
    template = header.template_dtype
    padded = block.ljust(template.itemsize, b"\0")[: template.itemsize]
    stored = np.frombuffer(padded, dtype=template)[0]
    return _keep_stored_geometry(header, stored)


def _keep_stored_geometry(
    header: _mgh.MGHHeader, stored: tx.Any
) -> _mgh.MGHHeader:
    """Set the geometry fields of a header to the values of another one.

    The fields are listed in [`_GEOMETRY_FIELDS`][], and `stored` is a
    header or a structured value that has them. The header is returned.
    """
    for name in _GEOMETRY_FIELDS:
        header[name] = stored[name]
    return header


def _geometry(header: _mgh.MGHHeader) -> _Geometry:
    """Return the voxel size, direction cosines and centre in effect.

    The FreeSurfer defaults replace the stored values when the
    `goodRASFlag` is not positive.
    """
    if int(header["goodRASFlag"]) <= 0:
        return (
            (1.0, 1.0, 1.0),
            FS_DEFAULT_XRAS,
            FS_DEFAULT_YRAS,
            FS_DEFAULT_ZRAS,
            (0.0, 0.0, 0.0),
        )
    mdc = np.asarray(header["Mdc"], dtype=np.float64)
    return (
        tuple(float(d) for d in header["delta"]),
        tuple(mdc[0]),
        tuple(mdc[1]),
        tuple(mdc[2]),
        tuple(float(c) for c in header["Pxyz_c"]),
    )


def _compressed_name(filename: path.FilenameLike) -> bool:
    """Tell whether a file name asks for gzip, by its `.mgz` or `.gz` suffix.

    The name is read from the path of the URL, so that a query does not
    hide the suffix.
    """
    name = urlsplit(str(filename)).path.lower()
    return name.endswith((".mgz", ".gz"))


# ----------------------------------------------------------------------
#   WHOLE FILES
# ----------------------------------------------------------------------

_PROXY_OPTIONS = ("mmap", "keep_file_open")
"""The options of the nibabel array proxy that a reader passes on."""


def read_mgh(file: path.FileLike, **kwargs) -> tx.Tuple[MghRaw, ArrayProxy]:
    """Read the record and a lazy array of the voxels of an MGH file.

    The record is read from the file as it is stored. The voxels are not
    read: nibabel's array proxy reads them when they are accessed. A local
    file whose name matches its content (`.mgz` or `.mgh.gz` when it is
    compressed, and `.mgh` otherwise) is handed to the proxy by name, so
    that nibabel opens the file each time and can memory-map the voxels.
    Any other path is read into memory, because nibabel chooses the codec
    from the name. An open stream is kept by the proxy, so the caller must
    keep it open while the voxels may be read, and a stream that cannot
    seek is read into memory first.

    Parameters
    ----------
    file : path or file object
        The MGH or MGZ file.
    **kwargs : Any
        Options of the nibabel array proxy, which are `mmap` and
        `keep_file_open`. Other options are ignored.

    Returns
    -------
    raw : MghRaw
        The record of the file.
    proxy : nibabel.arrayproxy.ArrayProxy
        The lazy array of the voxels, in Fortran order.
    """
    options = {k: kwargs[k] for k in _PROXY_OPTIONS if k in kwargs}
    if isinstance(file, str):
        file = path.Path(file)
    if not isinstance(file, path.PathLike):
        return _read_mgh_stream(file, options)
    with file.open("rb") as stream:
        if not (_is_local(file) and _named_as_stored(file, stream)):
            return _read_mgh_stream(BytesIO(stream.read()), options)
        raw = MghRaw.from_fileobj(stream)
    return raw, ArrayProxy(str(file), raw.header, **options)


def _read_mgh_stream(
    fileobj: tx.BinaryIO, options: tx.Mapping[str, tx.Any]
) -> tx.Tuple[MghRaw, ArrayProxy]:
    """Read the record and the proxy of an MGH stream.

    The record is read first, and the position of the stream is restored
    so that the proxy reads the voxels from the same start.
    """
    if not _seekable(fileobj):
        fileobj = BytesIO(fileobj.read())
    stream = open_compressed(fileobj)
    with preserve_position(stream):
        raw = MghRaw.from_fileobj(stream)
    return raw, ArrayProxy(stream, raw.header, **options)


def _named_as_stored(file: path.PathLike, stream: tx.BinaryIO) -> bool:
    """Tell whether nibabel would choose the right codec from a file name.

    nibabel decompresses a file whose name ends with `.mgz` or `.gz`, so
    a compressed file must be named `.mgz` or `.mgh.gz`, and a file that
    is not compressed must be named `.mgh`. The position of the stream is
    restored.
    """
    with preserve_position(stream):
        gzipped = stream.read(2) == b"\x1f\x8b"
    name = str(file).lower()
    if gzipped:
        return name.endswith((".mgz", ".mgh.gz"))
    return name.endswith(".mgh")


def encode_mgh(raw: MghRaw, data: ArrayProtocol) -> bytes:
    """Return the content of an MGH file from its record and its voxels.

    The fixed header, the voxels, the footer and the tags are written in
    this order, without compression. The voxels are written in Fortran
    order with the voxel type of the header, and a nibabel proxy is
    written as it is stored. The shape of `data` must match the dimensions
    of the header.

    Parameters
    ----------
    raw : MghRaw
        The record of the file.
    data : array_like
        The voxels, as the file stores them.

    Returns
    -------
    bytes
        The uncompressed content of the file.
    """
    if isinstance(data, ArrayProxy):
        data = data.get_unscaled()
    buffer = BytesIO()
    raw.header.writehdr_to(buffer)
    dtype = raw.header.get_data_dtype()
    array_to_file(data, buffer, dtype, offset=MGH_HEADER_SIZE)
    raw.header.writeftr_to(buffer)
    buffer.write(raw.tags)
    return buffer.getvalue()
