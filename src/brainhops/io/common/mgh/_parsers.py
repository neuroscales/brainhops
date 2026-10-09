"""The MGH/MGZ parser."""

# stdlib
import gzip
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from nibabel.freesurfer import mghformat as _mgh

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    WriterNotImplementedError,
    preserve_position,
)
from brainhops.io.common.freesurfer import FreesurferFormat
from brainhops.io.common.freesurfer._geometry import (
    FS_DEFAULT_XRAS,
    FS_DEFAULT_YRAS,
    FS_DEFAULT_ZRAS,
    fs_vox2ras,
    fs_vox2tkr,
)
from brainhops.io.common.nifti._files import (
    _accepted,
    _image_from_stream,
    _image_to_stream,
    _is_local,
)

# this format
from ._constants import (
    _MGH_AXES,
    _MRI_PARAMS,
)
from ._utils import (
    _good_ras,
    _mgh_from_filename,
    _read_prefix,
    _read_tags,
    _seekable,
    _valid_prefix,
)

_MghObject = tx.Union[_mgh.MGHHeader, _mgh.MGHImage]


class MghParser(
    DataModelBase, FreesurferFormat, BinaryFileReader, BinaryFileWriter
):
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
