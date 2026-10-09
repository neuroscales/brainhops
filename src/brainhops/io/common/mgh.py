"""
Reading and writing of MGH and MGZ files.

MGH is the big-endian volume format of FreeSurfer, and MGZ is the same
format gzipped. A file holds a fixed header (version, dimensions, voxel
type, `goodRASFlag`, and the geometry described in
[`brainhops.io.common.freesurfer`][]), the voxels in Fortran order, an
optional footer of acquisition parameters, and optional trailing tags
such as the command history.

The header and voxels are read and written with nibabel. The trailing
tags and the `goodRASFlag`, which nibabel drops or resets to 1, are
read from the raw bytes so that a file round-trips.

!!! warning "goodRASFlag"
    When the flag is not positive, FreeSurfer ignores the stored
    geometry and uses 1 mm voxels, coronal LIA cosines and a zero
    centre. nibabel's default cosines are LSP instead, which disagrees
    with FreeSurfer and with nibabel's own tkr matrix. This module
    follows FreeSurfer.
"""

__all__ = ["MghParser", "MGH_HEADER_SIZE", "MGH_FOOTER_SIZE"]

import gzip
import struct
from io import BytesIO

import numpy as np
import typing_extensions as tx
from nibabel.freesurfer import mghformat as _mgh

from brainhops._core import path
from brainhops._core.streams import open_compressed
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    WriterNotImplementedError,
    preserve_position,
)
from brainhops.io.common.freesurfer import (
    FS_DEFAULT_XRAS,
    FS_DEFAULT_YRAS,
    FS_DEFAULT_ZRAS,
    FreesurferFormat,
    fs_vox2ras,
    fs_vox2tkr,
)
from brainhops.io.common.nifti import (
    _accepted,
    _image_from_stream,
    _image_to_stream,
    _is_local,
)

MGH_HEADER_SIZE = 284
"""The size in bytes of the fixed MGH header, after which the voxels start."""

MGH_FOOTER_SIZE = 20
"""The size in bytes of the footer, five big-endian floats."""

# The leading header fields, enough to recognise a file: the version, the
# four dimensions, the type and dof (int32), and goodRASFlag (int16).
_PREFIX = struct.Struct(">7ih")

# MGH voxel type codes: UCHAR, INT, FLOAT, SHORT.
_MGH_TYPES = {0: 1, 1: 4, 3: 4, 4: 2}

# The four MGH axes, in storage (Fortran) order.
_MGH_AXES = [
    Axis("x", "space"),
    Axis("y", "space"),
    Axis("z", "space"),
    Axis("t", "time"),
]

_MRI_PARAMS = ("tr", "flip_angle", "te", "ti", "fov")
"""The footer fields, as nibabel names them."""

_MghObject = tx.Union[_mgh.MGHHeader, _mgh.MGHImage]


def _read_prefix(fileobj: tx.BinaryIO) -> tx.Optional[tuple]:
    """
    Read the leading header fields of a decompressed stream, or `None` if
    the stream is too short.
    """
    raw = fileobj.read(_PREFIX.size)
    if len(raw) < _PREFIX.size:
        return None
    return _PREFIX.unpack(raw)


def _valid_prefix(prefix: tx.Optional[tuple]) -> bool:
    """Tell whether the leading fields describe an MGH volume."""
    if prefix is None:
        return False
    version, *dims, dtype, _dof, _flag = prefix
    return version == 1 and all(d > 0 for d in dims) and dtype in _MGH_TYPES


class MghParser(DataModelBase, FreesurferFormat, BinaryFileParserWriter):
    """
    The base class of objects encoded as MGH or MGZ files.

    As in [`NiftiParser`][brainhops.io.common.nifti.NiftiParser], the
    object holds a nibabel image or header, together with the raw
    `goodRASFlag` and the trailing tags. It exposes the voxels, their
    coordinate system, and the voxel-to-RAS matrices that FreeSurfer
    derives from the header.
    """

    HINTS = ("mgh", "mgz")

    image: tx.Annotated[
        tx.Optional[_mgh.MGHImage],
        tx.Doc(
            """
            The `nibabel` MGH image associated with this object, or `None`
            when the object was built from a header or from data alone.
            """
        ),
    ] = None

    _header: tx.Annotated[
        tx.Optional[_mgh.MGHHeader],
        tx.Doc(
            """
            The `nibabel` MGH header associated with this object.

            If it is not given, the header of `image` is used.
            """
        ),
    ] = None

    _good_ras: tx.Annotated[
        tx.Optional[bool],
        tx.Doc(
            """
            Whether the header's `goodRASFlag` was positive, i.e. whether
            the stored voxel size, direction cosines and centre are valid.

            `nibabel` resets the flag to 1 when it reads a file, so it is
            read from the raw bytes. `None` means it is not known (an
            object built from a `nibabel` image), and the header is then
            trusted.
            """
        ),
    ] = None

    _tags: tx.Annotated[
        tx.Optional[bytes],
        tx.Doc(
            """
            The raw trailing tags that follow the footer, verbatim.

            If they are not given, they are read lazily from the file the
            `image` was loaded from.
            """
        ),
    ] = None

    @property
    def header(self) -> tx.Optional[_mgh.MGHHeader]:
        """
        The nibabel MGH header: the one set explicitly, or else that of the
        image.
        """
        if getattr(self, "_header", None) is not None:
            return self._header
        if self.image is not None:
            return self.image.header
        return None

    @header.setter
    def header(self, value: tx.Optional[_mgh.MGHHeader]) -> None:
        self._header = value

    @property
    def tags(self) -> bytes:
        """
        The raw bytes of the trailing tags, written back verbatim.

        For an object loaded from a path, the tags are read lazily from that
        file.
        """
        if getattr(self, "_tags", None) is not None:
            return self._tags
        tags = b""
        filename = None
        if self.image is not None and self.header is not None:
            holder = self.image.file_map.get("image")
            filename = getattr(holder, "filename", None)
        if filename:
            with open(filename, "rb") as f:
                stream = open_compressed(f)
                tags = _read_tags(stream, self.header)
        self._tags = tags
        return tags

    @tags.setter
    def tags(self, value: tx.Optional[bytes]) -> None:
        self._tags = None if value is None else bytes(value)

    @property
    def mri_params(self) -> tx.Dict[str, float]:
        """
        The acquisition parameters of the footer, a zero meaning not recorded.

        The repetition, echo and inversion times are in milliseconds, the flip
        angle in radians, and the field of view in millimetres.
        """
        header = self.header
        if header is None:
            return {name: 0.0 for name in _MRI_PARAMS}
        return {name: float(header[name]) for name in _MRI_PARAMS}

    @property
    def shape(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The shape of the volume, `(x, y, z)` or `(x, y, z, frames)`."""
        header = self.header
        if header is None:
            data = getattr(self, "_data", None)
            return None if data is None else tuple(data.shape)
        return tuple(int(d) for d in header.get_data_shape())

    def _geometry(self) -> tx.Tuple[tuple, tuple, tuple, tuple, tuple]:
        """
        Return the voxel size, direction cosines and centre in effect.

        The FreeSurfer defaults replace the stored values when the
        `goodRASFlag` is not positive.
        """
        header = self.header
        if self._good_ras is False:
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

    @property
    def voxel_size(self) -> tx.Optional[tx.Tuple[float, float, float]]:
        """The voxel size in millimetres."""
        if self.header is None:
            return None
        return self._geometry()[0]

    @property
    def vox2ras(self) -> tx.Optional[np.ndarray]:
        """
        The `(4, 4)` voxel-to-scanner-RAS matrix (`mri_info --vox2ras`).

        The matrix equals nibabel's `header.get_vox2ras()`, except that the
        FreeSurfer defaults apply when the `goodRASFlag` is not positive.
        """
        if self.header is None:
            return None
        shape = self.header["dims"][:3]
        return fs_vox2ras(shape, *self._geometry())

    @property
    def vox2tkr(self) -> tx.Optional[np.ndarray]:
        """The `(4, 4)` voxel-to-tkr-RAS matrix (`mri_info --vox2ras-tkr`)."""
        if self.header is None:
            return None
        shape = self.header["dims"][:3]
        return fs_vox2tkr(shape, self._geometry()[0])

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The voxels, read lazily from the image and cached, unless set
        explicitly.
        """
        if getattr(self, "_data", None) is not None:
            return self._data
        if self.image is not None:
            self._data = get_array_backend().asarray(self.image.dataobj)
            return self._data
        return None

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._data = value

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """
        The voxel coordinate system, derived from the header unless set
        explicitly.
        """
        if getattr(self, "_system", None) is not None:
            return self._system
        if self.header is None:
            return None
        ndim = len(self.header.get_data_shape())
        return CoordinateSystem(name="voxel", axes=_MGH_AXES[:ndim], order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """
        Build an object from an MGH or MGZ file.

        A local file whose name matches its content (`.mgz` or `.mgh.gz` when
        gzipped, `.mgh` otherwise) is handed to nibabel by path, so that its
        voxels are memory-mapped and read lazily. nibabel chooses the codec
        from the name, so any other file is read into memory and parsed as a
        stream; the in-memory stream outlives the file, which is closed on
        return.
        """
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, path.PathLike):
            if not path.exists(file):
                raise ParserExistsError(f"No such file: {file}")
            name = str(file)
            with file.open("rb") as f:
                magic = f.read(2)
                f.seek(0)
                gzipped = magic == b"\x1f\x8b"
                lower = name.lower()
                known = (
                    lower.endswith((".mgz", ".mgh.gz"))
                    if gzipped
                    else (lower.endswith(".mgh"))
                )
                if not (_is_local(file) and known):
                    return cls.from_fileobj(BytesIO(f.read()), **kwargs)
                prefix = _read_prefix(open_compressed(f))
            read = _mgh_from_filename
            try:
                image = read(name, **_accepted(read, kwargs))
            except FileNotFoundError:
                with file.open("rb") as f:
                    return cls.from_fileobj(BytesIO(f.read()), **kwargs)
            return cls(
                image=image, header=image.header, good_ras=_good_ras(prefix)
            )
        return super().from_file(file, **kwargs)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        """
        Build an object from an open MGH or MGZ file.

        As for NIfTI, the voxels are read lazily from the stream, which the
        caller must keep open, while the header, the `goodRASFlag` and the tags
        are read immediately. A stream that cannot seek is first read into
        memory.
        """
        if not _seekable(fileobj):
            fileobj = BytesIO(fileobj.read())
        with preserve_position(fileobj):
            stream = open_compressed(fileobj)
            start = stream.tell()
            prefix = _read_prefix(stream)
            stream.seek(start)
            image = _image_from_stream(_mgh.MGHImage, stream, **kwargs)
            tags = _read_tags(stream, image.header)
        return cls(
            image=image,
            header=image.header,
            good_ras=_good_ras(prefix),
            tags=tags,
        )

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> tx.Self:
        """Build an object from the bytes of an MGH or MGZ file."""
        return cls.from_fileobj(BytesIO(data), **kwargs)

    @classmethod
    def from_nibabel(cls, mgh: _MghObject, **kwargs) -> tx.Self:
        """Build an object from a loaded nibabel MGH header or image."""
        if isinstance(mgh, _mgh.MGHHeader):
            return cls(header=mgh, **kwargs)
        if isinstance(mgh, _mgh.MGHImage):
            return cls(image=mgh, header=mgh.header, **kwargs)
        raise TypeError(f"Expected an MGH image or header, got {type(mgh)}")

    def to_nibabel(self, **kwargs) -> _mgh.MGHImage:
        """
        Build the nibabel image that encodes the object.

        Each concrete format overrides this method, on which the other writing
        methods rely.
        """
        raise WriterNotImplementedError(
            f"{type(self).__name__} does not know how to write itself to MGH."
        )

    def to_bytes(self, compress: bool = False, **kwargs) -> bytes:
        """
        Return the MGH encoding: header, voxels, footer and tags.

        With `compress`, the result is gzipped into MGZ.
        """
        stream = BytesIO()
        image = self.to_nibabel(**kwargs)
        _image_to_stream(image, stream)
        stream.write(self.tags or b"")
        content = stream.getvalue()
        if compress:
            content = gzip.compress(content)
        return content

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write the file, gzipped when its name ends with `.mgz` or `.gz` unless
        `compress` says otherwise.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if "compress" not in kwargs:
            kwargs["compress"] = (
                str(filename).lower().endswith((".mgz", ".gz"))
            )
        with filename.open("wb") as f:
            f.write(self.to_bytes(**kwargs))

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the MGH (or, with `compress`, MGZ) encoding to a file."""
        file.write(self.to_bytes(**kwargs))

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Return the confidence that an open file holds an MGH header.

        The file may be gzipped. It is recognised from its leading fields (the
        version 1, four positive dimensions and a known voxel type), not from
        its name.
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
        """Return the confidence that bytes hold an MGH header."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)


def _good_ras(prefix: tx.Optional[tuple]) -> tx.Optional[bool]:
    """
    Tell whether the `goodRASFlag` of the leading fields is positive, or
    `None` if they could not be read.
    """
    return None if prefix is None else prefix[-1] > 0


def _mgh_from_filename(
    filename: str,
    *,
    mmap: tx.Union[bool, str] = True,
    keep_file_open: tx.Optional[bool] = None,
) -> _mgh.MGHImage:
    """
    Load an image like nibabel's `MGHImage.from_filename`, but close the
    file the header is read from, which nibabel leaks.

    The voxels stay lazy, because the array proxy opens the file itself.
    """
    klass = _mgh.MGHImage
    if mmap not in (True, False, "c", "r"):
        raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
    file_map = klass.filespec_to_file_map(filename)
    holder = file_map["image"]
    with holder.get_prepare_fileobj("rb") as f:
        header = klass.header_class.from_fileobj(f)
    data = klass.ImageArrayProxy(
        holder.file_like,
        header.copy(),
        mmap=mmap,
        keep_file_open=keep_file_open,
    )
    return klass(data, header.get_affine(), header, file_map=file_map)


def _seekable(fileobj: tx.IO) -> bool:
    """Tell whether a stream can seek (no answer means no)."""
    try:
        return bool(fileobj.seekable())
    except Exception:
        return False


def _read_tags(stream: tx.BinaryIO, header: _mgh.MGHHeader) -> bytes:
    """
    Read the raw bytes after the footer of a decompressed stream.

    Offsets are counted from the start of the stream, as nibabel counts
    them.
    """
    offset = int(header.get_footer_offset()) + MGH_FOOTER_SIZE
    try:
        stream.seek(offset)
    except Exception:
        return b""
    return stream.read() or b""
