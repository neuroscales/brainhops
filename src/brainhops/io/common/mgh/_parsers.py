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
    BinaryFileParserWriter,
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


class MghParser(DataModelBase, FreesurferFormat, BinaryFileParserWriter):
    """
    Base class for objects that are encoded by an MGH or MGZ file.

    It holds the `nibabel` image or header (`image` and `header`, as in
    [`NiftiParser`][brainhops.io.common.nifti.NiftiParser]), the raw
    `goodRASFlag` and the trailing tags, and exposes the voxels, the voxel
    coordinate system and the voxel-to-RAS matrices FreeSurfer derives
    from the header.
    """

    HINTS = ("mgh", "mgz")

    # --- MGH API ------------------------------------------------------

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
        """The `nibabel` MGH header: the one set explicitly, or else the
        header of `image`, or `None`."""
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
        The raw bytes of the trailing tags (empty when there are none).

        They are kept verbatim and written back after the footer, so that
        they survive a round trip. They are read lazily from the source
        file when the object was loaded from a path.
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
        The MRI acquisition parameters of the footer.

        `tr`, `te` and `ti` are in milliseconds, `flip_angle` in radians
        and `fov` in millimetres. A value of zero means "not recorded".
        """
        header = self.header
        if header is None:
            return {name: 0.0 for name in _MRI_PARAMS}
        return {name: float(header[name]) for name in _MRI_PARAMS}

    # --- geometry -----------------------------------------------------

    @property
    def shape(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The shape of the volume, `(x, y, z)` or `(x, y, z, frames)`."""
        header = self.header
        if header is None:
            data = getattr(self, "_data", None)
            return None if data is None else tuple(data.shape)
        return tuple(int(d) for d in header.get_data_shape())

    def _geometry(self) -> tx.Tuple[tuple, tuple, tuple, tuple, tuple]:
        """The voxel size, direction cosines and centre in effect.

        FreeSurfer's defaults replace the stored values when
        `goodRASFlag` was not positive.
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
        """The voxel size in millimetres, `None` without a header."""
        if self.header is None:
            return None
        return self._geometry()[0]

    @property
    def vox2ras(self) -> tx.Optional[np.ndarray]:
        """
        The `(4, 4)` voxel-to-scanner-RAS matrix (`mri_info --vox2ras`).

        Equal to `nibabel`'s `header.get_vox2ras()`, except when
        `goodRASFlag` was not positive, where FreeSurfer's defaults are
        used.
        """
        if self.header is None:
            return None
        shape = self.header["dims"][:3]
        return fs_vox2ras(shape, *self._geometry())

    @property
    def vox2tkr(self) -> tx.Optional[np.ndarray]:
        """
        The `(4, 4)` voxel-to-tkr (surface) RAS matrix
        (`mri_info --vox2ras-tkr`, `header.get_vox2ras_tkr()`).
        """
        if self.header is None:
            return None
        shape = self.header["dims"][:3]
        return fs_vox2tkr(shape, self._geometry()[0])

    # --- datamodel ----------------------------------------------------

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The voxels, `(x, y, z)` or `(x, y, z, frames)`, read lazily
        from `image` and cached, unless set explicitly."""
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
        """The voxel coordinate system `(x, y, z[, t])`, in F order,
        derived from `header` unless set explicitly."""
        if getattr(self, "_system", None) is not None:
            return self._system
        if self.header is None:
            return None
        ndim = len(self.header.get_data_shape())
        return CoordinateSystem(name="voxel", axes=_MGH_AXES[:ndim], order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    # --- BinaryFileParser API -----------------------------------------

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """
        Build the object from an MGH or MGZ file.

        A local file whose name matches its content (`.mgz` or `.mgh.gz`
        when gzipped, `.mgh` when not) is handed to `nibabel` by
        path, through `from_filename`, so that the voxels are
        memory-mapped and read lazily.

        `nibabel` cannot open any other file by name: a remote one (which
        has no local path), or one whose name it would take for another
        codec (it picks the codec from the name). Such a file is read
        into memory and handed to `nibabel` as a stream (see
        [`from_fileobj`][]): the stream `nibabel` reads the voxels from
        lazily must outlive the file, which is closed on return.
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
        Build the object from an open MGH or MGZ file object.

        The image is read with `nibabel`'s stream API, gzipped or not
        (the compression is sniffed from the magic bytes, as a stream has
        no name). As for NIfTI, the voxels are read lazily from the
        stream, so the caller keeps it open for as long as they may be
        read; the header, `goodRASFlag` and trailing tags are read right
        away.

        A stream that cannot seek is read into memory first: the leading
        bytes must be read twice (for `goodRASFlag`, then by `nibabel`),
        and its compression cannot be sniffed otherwise.
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
        """Build the object from MGH bytes (or gzipped MGZ bytes)."""
        return cls.from_fileobj(BytesIO(data), **kwargs)

    @classmethod
    def from_nibabel(cls, mgh: _MghObject, **kwargs) -> tx.Self:
        """Build the object from an already-loaded `nibabel` MGH header
        or image."""
        if isinstance(mgh, _mgh.MGHHeader):
            return cls(header=mgh, **kwargs)
        if isinstance(mgh, _mgh.MGHImage):
            return cls(image=mgh, header=mgh.header, **kwargs)
        raise TypeError(f"Expected an MGH image or header, got {type(mgh)}")

    # --- FileParserWriter API -----------------------------------------

    def to_nibabel(self, **kwargs) -> _mgh.MGHImage:
        """Build the `nibabel` image that encodes this object.

        Each concrete MGH format overrides this method; the other writer
        methods are defined in terms of it."""
        raise WriterNotImplementedError(
            f"{type(self).__name__} does not know how to write itself to MGH."
        )

    def to_bytes(self, compress: bool = False, **kwargs) -> bytes:
        """
        Return the MGH encoding of the object: header, voxels, footer and
        trailing tags. With `compress=True`, return the gzipped (MGZ)
        encoding instead.
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
        """Write the object to a file, gzipped (MGZ) when its name ends
        with `.mgz` or `.gz`, unless `compress` says otherwise."""
        if isinstance(filename, str):
            filename = path.Path(filename)
        if "compress" not in kwargs:
            kwargs["compress"] = (
                str(filename).lower().endswith((".mgz", ".gz"))
            )
        with filename.open("wb") as f:
            f.write(self.to_bytes(**kwargs))

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the MGH encoding of the object (MGZ with
        `compress=True`) to an open file object."""
        file.write(self.to_bytes(**kwargs))

    # --- BinaryFileSniffer API ----------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Score how confident the class is that an open file object holds
        an MGH header, gzipped or not.

        The content is recognised from its leading fields -- a version of
        1, four positive dimensions and a known voxel type -- not from the
        file name.
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
        """Score how confident the class is that bytes hold an MGH
        header, gzipped or not."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)
