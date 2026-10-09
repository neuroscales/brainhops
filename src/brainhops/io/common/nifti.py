"""Reading and writing machinery shared by all NIfTI-based formats.

The image formats and the transformation formats stored as NIfTI files are both
built on [`NiftiParser`][].
"""

__all__ = ["NiftiParser"]

import gzip
import inspect
import warnings
from io import BytesIO
from urllib.parse import urlsplit

import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops._core import path
from brainhops._core.streams import open_compressed
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Transformation
from brainhops.datamodel.units import (
    is_indexunit,
    is_physicalunit,
    is_spaceunit,
    is_timeunit,
)
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    UnrepresentableTransformationError,
    WriterError,
    WriterNotImplementedError,
    preserve_position,
)
from brainhops.io.common._geometry import (
    AxisLayout,
    arrange_voxel_to_ras,
    declared_axes,
    embed_affine,
)
from brainhops.io.common._nifti_units import nifti_unit_meters, unit_to_nifti

_NiftiObject = tx.Union[nb.Nifti1Header, nb.Nifti1Image]


# The array axes are axes of voxel space, so they count samples. A reader that
# builds a physical space gives the axes its own unit.
_INDEX = "index"
_NIFTI_AXES = [
    Axis("x", "space", unit=_INDEX),
    Axis("y", "space", unit=_INDEX),
    Axis("z", "space", unit=_INDEX),
    Axis("t", "time", unit=_INDEX),
    Axis("c", "channel", unit=_INDEX),
    Axis("dim5", unit=_INDEX),
    Axis("dim6", unit=_INDEX),
]
_FLAT_AXES = {
    # The number of points, vertices, triangles and so on.
    0: Axis("n", unit=_INDEX),
    1: Axis("x", unit=_INDEX),
    2: Axis("y", unit=_INDEX),
    3: Axis("z", unit=_INDEX),
}
_FLAT_AXES_CHANNEL = {**_FLAT_AXES, 4: Axis("c", "channel", unit=_INDEX)}
_FLAT_AXES_TIME = {**_FLAT_AXES, 4: Axis("t", "time", unit=_INDEX)}
_AXES_DISP = {4: Axis("c", "displacement", unit=_INDEX)}
_NIFTI_SPECIFIC_AXES = {
    1004: {5: Axis("k", "channel", unit=_INDEX)},  # GENMATRIX
    1006: _AXES_DISP,  # DISPVECT
    1008: _FLAT_AXES_CHANNEL,  # POINTSET
    1009: _FLAT_AXES_CHANNEL,  # TRIANGLE
    # GIFTI intents
    2001: _FLAT_AXES_TIME,  # TIME_SERIES
    2002: _FLAT_AXES_CHANNEL,  # NODE_INDEX
    2003: _FLAT_AXES_CHANNEL,  # RGB_VECTOR
    2004: _FLAT_AXES_CHANNEL,  # RGBA_VECTOR
    2005: _FLAT_AXES_CHANNEL,  # SHAPE
    # FSL intents
    2006: _AXES_DISP,  # FSL_FNIRT_DISPLACEMENT_FIELD
    2007: _AXES_DISP,  # FSL_CUBIC_SPLINE_COEFFICIENTS
    2008: _AXES_DISP,  # FSL_DCT_COEFFICIENTS
    2009: _AXES_DISP,  # FSL_QUADRATIC_SPLINE_COEFFICIENTS
}


_NIFTI_FIELD_INTENTS = frozenset(
    {
        1006,  # DISPVECT
        1007,  # VECTOR
        2006,  # FSL_FNIRT_DISPLACEMENT_FIELD
        2007,  # FSL_CUBIC_SPLINE_COEFFICIENTS
        2008,  # FSL_DCT_COEFFICIENTS
        2009,  # FSL_QUADRATIC_SPLINE_COEFFICIENTS
    }
)
"""Intent codes that mark a file as holding a deformation field.

A NIfTI file can hold an image, affine maps or a field, so the container alone
does not say which object is wanted. The intent code does, which lets each
sniffer score itself instead of relying on an arbitrary precedence.
"""

_NIFTI_FSL_INTENTS = frozenset({2006, 2007, 2008, 2009})
"""FSL field intent codes, left to the FSL readers because the generic field
reader does not decode FSL storage conventions.
"""

_NIFTI_INTENT_NONE = 0
"""Intent code of a plain image."""

_NIFTI_INTENT_DISPVECT = 1006
"""Intent code of a displacement field.

ITK 5.4 and later read a three-component `DISPVECT` image as RAS displacements
in millimetres. brainhops reads it in the same way and writes it only for
displacement fields.
"""

_NIFTI_INTENT_VECTOR = 1007
"""Intent code of a generic vector image.

brainhops writes coordinate fields in RAS with this code, as SPM does for its
`y_` deformations. ITK writes it for every vector image, including its LPS
displacement fields, so the code says nothing about the frame; the intent name
[`_NIFTI_INTENT_NAME_MAPPING`][] marks RAS coordinate maps.
"""

_NIFTI_INTENT_NAME_NIFTYREG = "NREG_TRANS"
"""Intent name of every NiftyReg transformation.

NiftyReg stores its transformations as `VECTOR` images with this name and tells
them apart by `intent_p1`. Only the NiftyReg readers decode such files, so the
generic vector readers decline them.
"""

_NIFTI_INTENT_NAME_MAPPING = "Mapping"
"""Intent name of SPM coordinate fields (`y_` files).

brainhops writes it with `VECTOR` on RAS coordinate fields. ITK never writes an
intent name, so the name also distinguishes such a map from an ITK or ANTs
field in LPS.
"""


_NIFTI_XCODES = {
    0: "unknown",
    1: "scanner",
    2: "aligned",
    3: "talairach",
    4: "mni",
    5: "template",
}
"""World-space name of each xform code.

The reader names each affine map after its code, and the writer reads that name
back to choose the stored code.
"""

_NIFTI_XFORM_CODE_BY_NAME = {
    name: code for code, name in _NIFTI_XCODES.items()
}
"""Xform code of each world-space name."""

_QFORM_NAME = "qform"
"""Name that the reader gives to the world space of the qform."""

_NIFTI_DEFAULT_XFORM_CODE = 2
"""Xform code (`aligned`) stored for a world space without a known reference,
because NIfTI ignores forms with a zero code.
"""

_NIFTI1_MAX_DIM = 2**15 - 1
"""Largest extent that NIfTI-1 stores; larger images are written as NIfTI-2."""


def _nifti_intent(header: "_NiftiObject") -> tx.Optional[int]:
    """Return the intent code of a header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return int(header["intent_code"])
    except Exception:
        return None


def _nifti_intent_name(header: "_NiftiObject") -> tx.Optional[str]:
    """Return the intent name of a header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return str(header.get_intent()[2])
    except Exception:
        return None


def _nifti_vector_field(data: ArrayProtocol) -> ArrayProtocol:
    """Drop the singleton time axis of a five-dimensional array.

    NIfTI stores a vector field with shape `(X, Y, Z, 1, 3)`, whereas readers
    expect `(X, Y, Z, 3)`. Any other array is returned unchanged. Unlike this
    function, [`NiftiParser.data`][] keeps the axis so that it matches the
    header.
    """
    shape = tuple(int(d) for d in data.shape)
    if len(shape) == 5 and shape[3] == 1:
        data = data[:, :, :, 0, :]
    return data


def _nifti_shape(header: "_NiftiObject") -> tx.Optional[tx.Tuple[int, ...]]:
    """Return the data shape of a header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return tuple(int(d) for d in header.get_data_shape())
    except Exception:
        return None


class NiftiParser(DataModelBase, BinaryFileParserWriter):
    """Base class for objects stored as NIfTI files."""

    HINTS = ("nifti",)

    image: tx.Annotated[
        tx.Optional[nb.Nifti1Image],
        tx.Doc(
            """
            The NIfTI image associated with this object.

            If no image is provided, but a header is, the `image` will
            be `None`, and the object will be constructed from the header
            only.
            """
        ),
    ] = None

    _header: tx.Annotated[
        tx.Optional[nb.Nifti1Header],
        tx.Doc(
            """
            The NIfTI header associated with this object.

            If the object was created from a NIfTI image, this will be
            pointing to the header of that image, unless a different
            header was explicitly provided to the constructor.

            If the object was created from a NIfTI header, this will be
            pointing to that header.
            """
        ),
    ] = None

    @property
    def header(self) -> tx.Optional[nb.Nifti1Header]:
        """The NIfTI header of the object.

        A header set explicitly takes precedence over the header of the image.

        !!! example
            ```python
            import nibabel as nb
            image1 = nb.load("image1.nii")
            image2 = nb.load("image2.nii")
            NiftiParser(image1).header                        # `image1.header`
            NiftiParser(header=image2.header).header          # `image2.header`
            NiftiParser(image1, header=image2.header).header  # `image2.header`
            obj = NiftiParser(image1)
            obj.header = image2.header
            obj.header                                        # `image2.header`
            ```
        """
        if getattr(self, "_header", None) is not None:
            return self._header
        if self.image is not None:
            return self.image.header
        return None

    @header.setter
    def header(self, value: tx.Optional[nb.Nifti1Header]) -> None:
        self._header = value

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, read lazily from the image and cached.

        Axes left unnamed by [`_nifti_to_axes`][] are dropped.
        """
        if getattr(self, "_data", None) is not None:
            return self._data

        if self.image is not None:
            data = get_array_backend().asarray(self.image.dataobj)

            slicer = [
                slice(None) if axis.name is not None else 0
                for axis in _nifti_to_axes(self.header)
            ]
            self._data = data[tuple(slicer)]
            return self._data

        return None

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._data = value

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system, built from the header unless set.

        Axes left unnamed by [`_nifti_to_axes`][] are dropped. Without a
        header, the system is `None`.
        """
        if getattr(self, "_system", None) is not None:
            return self._system

        if self.header is None:
            return None

        axes = [
            axis
            for axis in _nifti_to_axes(self.header)
            if axis.name is not None
        ]
        # In F order, the first axis changes fastest.
        return CoordinateSystem(axes=axes, name="voxel", order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Read an object from a NIfTI path or file object.

        A local path is handed to nibabel by name, so that nibabel owns the
        file handle and can memory-map the voxels. A remote path is opened
        through the path backend instead. The keyword arguments are passed to
        nibabel.

        Raises
        ------
        ParserExistsError
            If the path does not exist.
        """
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, path.PathLike):
            if not file.exists():
                raise ParserExistsError(f"No such file: {file}")
            return cls.from_nibabel(_load_nifti(file, **kwargs))
        return super().from_file(file, **kwargs)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read an object from an open NIfTI file object.

        The image data are read when the stream allows it; otherwise only the
        header is read.
        """
        with preserve_position(fileobj):
            try:
                obj = _nifti_from_stream(fileobj, **kwargs)
            except Exception:
                f = open_compressed(fileobj)
                read = nb.Nifti1Header.from_fileobj
                obj = read(f, **_accepted(read, kwargs))
        return cls.from_nibabel(obj)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> tx.Self:
        """Read an object from the bytes of a NIfTI file."""
        return cls.from_fileobj(BytesIO(data), **kwargs)

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """Build an object from a loaded nibabel header or image.

        Raises
        ------
        TypeError
            If `nifti` is neither a header nor an image.
        """
        if isinstance(nifti, nb.Nifti1Header):
            return cls(header=nifti, **kwargs)
        if isinstance(nifti, nb.Nifti1Image):
            return cls(image=nifti, header=nifti.header, **kwargs)
        raise TypeError(f"Expected a NIfTI image or header, got {type(nifti)}")

    def to_nibabel(self, **kwargs) -> nb.Nifti1Image:
        """Return the nibabel image that encodes this object.

        Each concrete format overrides this method, on which the other writer
        methods rely. The base implementation raises
        `WriterNotImplementedError`.
        """
        cls = type(self)
        raise WriterNotImplementedError(
            f"{cls.__name__} does not know how to write itself to NIfTI."
        )

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the object to a path or a file object.

        A path is compressed when its name ends with `.gz`. A file object
        receives an uncompressed NIfTI file.
        """
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, path.PathLike):
            _save_nifti(self.to_nibabel(**kwargs), file)
            return
        return super().to_file(file, **kwargs)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the uncompressed NIfTI encoding of the object."""
        return self.to_nibabel(**kwargs).to_bytes()

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the uncompressed NIfTI encoding of the object to a stream."""
        file.write(self.to_bytes(**kwargs))

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        *,
        version: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a NIfTI header.

        Both NIfTI-1 and NIfTI-2 are tried unless `version` is given.
        """

        if version is None:
            best = Confidence.NO
            for version in (1, 2):
                best = max(
                    best,
                    cls.sniff_fileobj(
                        file, error=False, version=version, **kwargs
                    ),
                )
            if best:
                return best
            if error:
                if error is True:
                    error = SnifferContentError
                raise error("Content is not a valid NIfTI-1 or NIfTI-2 file")
            return Confidence.NO

        NiftiHeader = {1: nb.Nifti1Header, 2: nb.Nifti2Header}[version]
        kwargs["error"] = error
        base_error = None
        try:
            with preserve_position(file):
                f = open_compressed(file)
                nbkwargs = {}
                if "endianness" in kwargs:
                    nbkwargs["endianness"] = kwargs.pop("endianness")
                if "check" in kwargs:
                    nbkwargs["check"] = kwargs.pop("check")
                else:
                    nbkwargs["check"] = False
                # Check the magic number first: nibabel parses anything and
                # warns about the garbage in a file that is not NIfTI.
                start = _tell(f)
                head = f.read(_NIFTI_HEADER_SIZES[version])
                if not _has_nifti_magic(head, version):
                    raise ValueError(f"No NIfTI-{version} magic number")
                if start is not None:
                    f.seek(start)
                else:
                    f = BytesIO(head)
                # Real reads still show the warnings of nibabel.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obj = NiftiHeader.from_fileobj(f, **nbkwargs)
                    result = cls.sniff_nibabel(obj, **kwargs)
        except Exception as e:
            base_error = e
            result = Confidence.NO

        if result:
            return result

        if error:
            if error is True:
                error = SnifferContentError
            error = error(f"Content is not a valid NIfTI-{version} file")
            if base_error is not None:
                raise error from base_error
            else:
                raise error
        return Confidence.NO

    @classmethod
    def sniff_nibabel(
        cls,
        nifti: _NiftiObject,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a nibabel object matches this format.

        The header size is checked first, and a valid header is then scored
        with [`_score_nibabel`][]. A mismatch returns `False`.
        """
        if isinstance(nifti, nb.Nifti1Image):
            return cls.sniff_nibabel(nifti.header, error=error, **kwargs)
        if isinstance(nifti, nb.Nifti2Header):
            result = nifti["sizeof_hdr"] == 540
        elif isinstance(nifti, nb.Nifti1Header):
            result = nifti["sizeof_hdr"] == 348
        else:
            result = False
        if result:
            return cls._score_nibabel(nifti)
        if error:
            if error is True:
                error = SnifferContentError
            raise error(
                f"Magic number does not match NIfTI header: "
                f"{nifti['sizeof_hdr']}"
            )
        return False

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Return how well a valid NIfTI header matches this class.

        The header has already passed the magic check, so the score expresses
        how likely the file holds the kind of object that this class builds. A
        displacement field and a plain image share the container and are told
        apart by the intent code, or else by the data shape.

        !!! note "Why this is a separate method"
            [`sniff_nibabel`][] keeps the validation of the container and
            delegates only the part that depends on the format. Overriding
            `sniff_nibabel` would make each format re-implement the magic
            check.

        Returns
        -------
        float
            A score in `[0, 1]`; the base implementation returns
            `Confidence.MAYBE`.
        """
        return Confidence.MAYBE

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a NIfTI header."""
        kwargs["error"] = error
        return cls.sniff_fileobj(BytesIO(data), **kwargs)


# ----------------------------------------------------------------------
#   READING AND WRITING THROUGH A PATH
# ----------------------------------------------------------------------
# nibabel opens names with the builtin `open`, so it would look up a remote
# name such as `s3://...` locally. These helpers are the only place that hands
# nibabel a path: a local file by name, so that the voxels are memory-mapped,
# and a remote file as an open stream.

# Protocols that bagof.paths reports for local files.
_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})


def _is_local(file: path.FilenameLike) -> bool:
    """Return whether nibabel can open a path by name."""
    return path.Path(file).protocol.lower() in _LOCAL_PROTOCOLS


_NIFTI_IMAGES = {1: nb.Nifti1Image, 2: nb.Nifti2Image}
_NIFTI_HEADERS = {1: nb.Nifti1Header, 2: nb.Nifti2Header}


def _tell(fileobj: tx.IO) -> tx.Optional[int]:
    """Return the stream position, or `None` if the stream cannot seek."""
    try:
        return fileobj.tell() if fileobj.seekable() else None
    except Exception:
        return None


# The NIfTI-1 magic sits near the end of the header, and the NIfTI-2 magic
# right after `sizeof_hdr`, followed by a line-ending check.
_NIFTI_HEADER_SIZES = {1: 348, 2: 540}
_NIFTI_MAGICS = {
    1: (344, (b"n+1\0", b"ni1\0")),
    2: (4, (b"n+2\0\r\n\032\n", b"ni2\0\r\n\032\n")),
}


def _has_nifti_magic(head: bytes, version: int) -> bool:
    """Return whether a decompressed header starts with the header size and
    magic of a NIfTI version.
    """
    size = _NIFTI_HEADER_SIZES[version]
    if len(head) < size:
        return False
    sizes = {int.from_bytes(head[:4], order) for order in ("little", "big")}
    if size not in sizes:
        return False
    offset, magics = _NIFTI_MAGICS[version]
    return any(head[offset : offset + len(m)] == m for m in magics)


def _nifti_version(fileobj: tx.BinaryIO) -> int:
    """Return the NIfTI version of a possibly compressed stream.

    The stream position is preserved. A stream that cannot seek is taken to be
    NIfTI-1.
    """
    start = _tell(fileobj)
    if start is None:
        return 1
    head = open_compressed(fileobj).read(4)
    fileobj.seek(start)
    # `sizeof_hdr` in either byte order.
    sizes = {int.from_bytes(head, order) for order in ("little", "big")}
    return 2 if 540 in sizes else 1


def _accepted(func: tx.Callable, kwargs: tx.Mapping[str, tx.Any]) -> dict:
    """Return the keyword arguments that `func` accepts.

    Files opened by name accept `mmap` and `keep_file_open`, whereas streams do
    not, so each nibabel call receives only the options it knows.
    """
    Parameter = inspect.Parameter
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is Parameter.VAR_KEYWORD for p in parameters):
        return dict(kwargs)
    named = (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    names = {p.name for p in parameters if p.kind in named}
    return {key: value for key, value in kwargs.items() if key in names}


def _image_from_stream(
    image_class: type, fileobj: tx.BinaryIO, **kwargs
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Read a NIfTI image from an uncompressed stream.

    Before nibabel 5.0, which added `from_stream`, the stream is passed in a
    file map.
    """
    if hasattr(image_class, "from_stream"):
        read = image_class.from_stream
        return read(fileobj, **_accepted(read, kwargs))
    file_map = image_class.make_file_map({"image": fileobj, "header": fileobj})
    read = image_class.from_file_map
    return read(file_map, **_accepted(read, kwargs))


def _image_to_stream(
    image: tx.Union[nb.Nifti1Image, nb.Nifti2Image], fileobj: tx.BinaryIO
) -> None:
    """Write a NIfTI image to an uncompressed stream.

    Before nibabel 5.0, which added `to_stream`, the stream is passed in a file
    map.
    """
    if hasattr(image, "to_stream"):
        image.to_stream(fileobj)
        return
    image.to_file_map(
        image.make_file_map({"image": fileobj, "header": fileobj})
    )


def _nifti_from_stream(
    fileobj: tx.BinaryIO, **kwargs
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Read a NIfTI-1 or NIfTI-2 image from a possibly compressed stream.

    The voxels are read lazily, so the stream must stay open while they may be
    accessed. A stream cannot be memory-mapped. On failure, the stream position
    is restored.
    """
    image_class = _NIFTI_IMAGES[_nifti_version(fileobj)]
    start = _tell(fileobj)
    try:
        return _image_from_stream(
            image_class, open_compressed(fileobj), **kwargs
        )
    except Exception:
        if start is not None:
            fileobj.seek(start)
        raise


def _load_nifti(
    file: path.FilenameLike, **kwargs
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Load a NIfTI image from a local or remote path.

    A local file is opened by nibabel and memory-mapped. A remote file is read
    into memory, because the voxels would otherwise be read after the stream is
    closed.
    """
    if _is_local(file):
        filename = str(path.Path(file))
        with open(filename, "rb") as f:
            image_class = _NIFTI_IMAGES[_nifti_version(f)]
        read = image_class.from_filename
        return read(filename, **_accepted(read, kwargs))
    with path.Path(file).open("rb") as f:
        buffer = BytesIO(f.read())
    return _nifti_from_stream(buffer, **kwargs)


def _load_nifti_header(
    file: path.FilenameLike,
) -> tx.Union[nb.Nifti1Header, nb.Nifti2Header]:
    """Read the header of a NIfTI file without reading the voxels."""
    if _is_local(file):
        return _load_nifti(file).header
    with path.Path(file).open("rb") as f:
        header_class = _NIFTI_HEADERS[_nifti_version(f)]
        return header_class.from_fileobj(open_compressed(f))


def _save_nifti(
    image: tx.Union[nb.Nifti1Image, nb.Nifti2Image], file: path.FilenameLike
) -> None:
    """Write a NIfTI image to a local or remote path.

    The file is compressed when its name ends with `.gz`.
    """
    if _is_local(file):
        image.to_filename(str(path.Path(file)))
        return
    # The name is taken from the URL, because the backend may not know it and a
    # query is not part of it. `open_compressed` only reads, so gzip compresses
    # here.
    compress = urlsplit(str(file)).path.lower().endswith(".gz")
    with path.Path(file).open("wb") as f:
        if compress:
            with gzip.GzipFile(fileobj=f, mode="wb") as gz:
                _image_to_stream(image, gz)
        else:
            _image_to_stream(image, f)


def _nifti_to_axes(header: nb.Nifti1Header) -> tx.List[Axis]:
    """Return the voxel axes described by a header.

    The intent code can change the names and types of the axes. Every axis
    counts samples, so its unit is the index unit.
    """

    ndim = len(header.get_data_shape())

    axes = _NIFTI_AXES[:ndim]

    # Indexing the header gives an unhashable 0-d array, so the intent is read
    # with `_nifti_intent`.
    intent = _nifti_intent(header)
    if intent in _NIFTI_SPECIFIC_AXES:
        axes_map = _NIFTI_SPECIFIC_AXES[intent]
        axes = [axes_map.get(i, axis) for i, axis in enumerate(axes)]

    # Copy the shared templates so that no system can mutate them.
    return [replace(axis) for axis in axes]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _new_nifti(
    data: ArrayProtocol, affine: tx.Optional[np.ndarray]
) -> _NiftiObject:
    """Build a nibabel image, as NIfTI-2 if an extent exceeds NIfTI-1.

    The array is stored as given, so lazy and device arrays are not
    materialised.

    Raises
    ------
    WriterError
        If NIfTI cannot store the data type of the array.
    """
    shape = tuple(int(d) for d in getattr(data, "shape", ()) or ())
    image_cls = nb.Nifti1Image
    if any(d > _NIFTI1_MAX_DIM for d in shape):
        image_cls = nb.Nifti2Image
    try:
        return image_cls(data, affine)
    except ValueError as error:
        dtype = getattr(data, "dtype", "unknown")
        raise WriterError(
            f"NIfTI cannot store an array of type {dtype}: {error}"
        ) from error


def _embed_affine(matrix: np.ndarray) -> np.ndarray:
    """Embed a voxel-to-world matrix in the `(4, 4)` matrix that NIfTI stores,
    with [`embed_affine`][].
    """
    return embed_affine(matrix, "NIfTI")


def _voxel_to_ras(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> np.ndarray:
    """Return the `(4, 4)` voxel-to-RAS matrix of a transformation."""
    return _voxel_to_ras_and_others(xform, voxel_axes)[0]


_NIFTI_POLICY = dict(fill_space=True, fill_time=True, time_slot=3)
"""Where NIfTI stores the array axes.

The policy is passed to [`arrange_voxel_to_ras`][]. The spatial axes come
first, then time, then the vector components or channels, then any other axis.
A slice with other axes gets a `z` of size one, and an image with channels but
no time gets a time axis of size one.
"""


def _nifti_geometry(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> tx.Tuple[
    np.ndarray,
    tx.List[tx.Optional[tx.Tuple[float, float]]],
    bool,
    tx.Optional[AxisLayout],
]:
    """Compute the NIfTI geometry of a voxel-to-world transformation.

    The transformation is reduced to an affine map, and the axes of each side
    are placed by what the spaces declare, in the order of [`_NIFTI_POLICY`][].
    The affine part applies to the spatial axes, and each later axis gets a
    scale and an offset. A time axis that the transformation leaves unchanged
    and that still counts frames is not mapped to time, so its repetition time
    is written as missing.

    Returns
    -------
    matrix : numpy.ndarray
        The `(4, 4)` voxel-to-RAS matrix.
    others : list
        The scale and offset of each later axis, or `None`.
    timed : bool
        Whether the first later axis is time.
    layout : AxisLayout or None
        The layout in which the data must be stored.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no NIfTI form, for example because it maps
        time to another type of axis.
    """
    arranged = arrange_voxel_to_ras(
        xform, voxel_axes, "NIfTI", **_NIFTI_POLICY
    )
    others = list(arranged.others)

    # The fourth axis is time unless a space says otherwise, and both sides
    # must agree.
    timed = [
        groups[3] == "time"
        for groups in (arranged.voxel_groups, arranged.world_groups)
        if groups is not None and len(groups) > 3
    ]
    if len(set(timed)) > 1:
        raise UnrepresentableTransformationError(
            "This transformation maps the time axis of one space to an "
            "axis of another type, so it cannot be written as NIfTI "
            "geometry, which maps time to time."
        )
    timed = timed[0] if timed else True

    if (
        timed
        and others
        and others[0] == (1.0, 0.0)
        and (arranged.filled_time or _is_frame_index(arranged.world, 3))
    ):
        others = [None, *others[1:]]
    return arranged.matrix, others, timed, arranged.layout


def _voxel_to_ras_and_others(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[Axis]] = None
) -> tx.Tuple[np.ndarray, tx.List[tx.Optional[tx.Tuple[float, float]]], bool]:
    """Return the first three results of [`_nifti_geometry`][]."""
    return _nifti_geometry(xform, voxel_axes)[:3]


def _declared_axes(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> tx.Optional[tx.List[Axis]]:
    """Return the axes of a system if they say where NIfTI stores each."""
    return declared_axes(system, ndim)


def _voxel_axes(
    transformation: Transformation, data: ArrayProtocol
) -> tx.Optional[tx.List[Axis]]:
    """Return the declared axes of the input space of a transformation, which
    indexes the data, or `None`.
    """
    ndim = len(getattr(data, "shape", ()) or ())
    return _declared_axes(getattr(transformation, "input", None), ndim)


def _is_frame_index(system: tx.Optional[CoordinateSystem], k: int) -> bool:
    """Return whether axis `k` of a system is a time axis counting frames."""
    axes = list(getattr(system, "axes", None) or [])
    if k >= len(axes) or axes[k] is Ellipsis:
        return False
    axis = axes[k]
    return getattr(axis, "type", None) == "time" and is_indexunit(
        getattr(axis, "unit", None)
    )


def _set_other_axes(
    image: _NiftiObject,
    others: tx.Sequence[tx.Optional[tx.Tuple[float, float]]],
    timed: bool = True,
) -> None:
    """Store the scale and offset of the axes after the spatial ones.

    Scales become spacings, and the offset of a time axis becomes `toffset`. A
    time axis that is not mapped to time gets a spacing of zero, which NIfTI
    reads as missing.

    Raises
    ------
    UnrepresentableTransformationError
        If a spacing is negative or an axis other than time has an offset.
    WriterError
        If the transformation maps more axes than the data have.
    """
    if not others:
        return
    header = image.header
    zooms = list(header.get_zooms())
    if 3 + len(others) > len(zooms):
        raise WriterError(
            f"The voxel-to-world transformation maps {3 + len(others)} "
            f"axes, but the data has {len(zooms)}."
        )
    for k, other in enumerate(others):
        if other is None:
            zooms[3 + k] = 0.0
            continue
        scale, offset = other
        if scale < 0:
            raise UnrepresentableTransformationError(
                f"NIfTI stores the spacing of axis {3 + k} as a positive "
                f"number, so a map that reverses it ({scale}) cannot be "
                f"written."
            )
        if (k or not timed) and offset != 0:
            raise UnrepresentableTransformationError(
                f"NIfTI stores an origin for the time axis only, so a map "
                f"that shifts axis {3 + k} ({offset}) cannot be written."
            )
        zooms[3 + k] = scale
    header.set_zooms(zooms)
    if timed and others[0] is not None:
        header["toffset"] = others[0][1]


def _reference_code(system: tx.Optional[CoordinateSystem]) -> int:
    """Return the non-zero xform code of a world space.

    A space named after a known reference gets its code; any other space is
    stored as `aligned`.
    """
    name = getattr(system, "name", None)
    code = _NIFTI_XFORM_CODE_BY_NAME.get(name, _NIFTI_DEFAULT_XFORM_CODE)
    return code or _NIFTI_DEFAULT_XFORM_CODE


def _sform_and_qform(
    transformations: tx.Sequence[Transformation],
    sform: np.ndarray,
    scode: int,
    voxel_axes: tx.Optional[tx.List[Axis]] = None,
    layout: tx.Optional[AxisLayout] = None,
) -> tx.Tuple[np.ndarray, int, tx.Optional[CoordinateSystem]]:
    """Choose the qform matrix, code and world space for an sform.

    The first transformation, other than the preferred last one, whose world
    space is named after a known reference is chosen. Otherwise the rigid
    `qform` transformation is used under the sform code, and failing that the
    sform itself. A transformation that places the data axes differently from
    `layout` is skipped. The world space is returned so that the caller can
    read its unit.
    """
    preferred = transformations[-1] if transformations else None

    def _placed(xform: Transformation) -> tx.Optional[np.ndarray]:
        matrix, _, _, placed = _nifti_geometry(xform, voxel_axes)
        if _layout_key(placed) != _layout_key(layout):
            return None
        return matrix

    for xform in transformations:
        if xform is preferred:
            continue
        output = getattr(xform, "output", None)
        name = getattr(output, "name", None)
        if _NIFTI_XFORM_CODE_BY_NAME.get(name):
            matrix = _placed(xform)
            if matrix is not None:
                return matrix, _NIFTI_XFORM_CODE_BY_NAME[name], output

    for xform in transformations:
        output = getattr(xform, "output", None)
        if getattr(output, "name", None) == _QFORM_NAME:
            matrix = _placed(xform)
            if matrix is not None:
                return matrix, scode, output

    return sform, scode, getattr(preferred, "output", None)


def _layout_key(
    layout: tx.Optional[AxisLayout],
) -> tx.Optional[tx.Tuple[tx.Tuple[int, ...], tx.Tuple[int, ...]]]:
    """Return the order and inserted axes of a layout, or `None` if trivial."""
    if layout is None or layout.trivial:
        return None
    return tuple(layout.order), tuple(layout.inserted)


def _space_unit_meters(
    system: tx.Optional[CoordinateSystem],
) -> tx.Optional[float]:
    """Return the size in metres of the first spatial unit of a world space, or
    `None` if it has no physical spatial unit.
    """
    for axis in get_axes(system):
        unit = getattr(axis, "unit", None)
        if is_physicalunit(unit) and is_spaceunit(unit):
            return float(unit.scale)
    return None


def _xyzt_labels(
    system: tx.Optional[CoordinateSystem],
) -> tx.Tuple[str, str]:
    """Return the NIfTI spatial and temporal unit labels of a space.

    The labels come from the first axes with a physical spatial and temporal
    unit, through [`unit_to_nifti`][]. A spatial unit that NIfTI cannot store
    gets the nearest label, and a missing unit gives `"unknown"`.
    """
    space = time = None
    # The `...` of an open space carries no unit.
    for axis in get_axes(system):
        unit = getattr(axis, "unit", None)
        if not is_physicalunit(unit):
            continue
        if space is None and is_spaceunit(unit):
            space = unit_to_nifti(unit, "space", nearest=True)
        elif time is None and is_timeunit(unit):
            time = unit_to_nifti(unit, "time")
    return space or "unknown", time or "unknown"


def _unit_scale(system: tx.Optional[CoordinateSystem], label: str) -> float:
    """Return the factor that converts a world space to the unit of a NIfTI
    label, or 1 if either unit is unknown.
    """
    stored = nifti_unit_meters(label)
    if stored is None:
        return 1.0
    meters = _space_unit_meters(system)
    if meters is None:
        return 1.0
    return meters / stored


def _scale_spatial(matrix: np.ndarray, factor: float) -> np.ndarray:
    """Scale the spatial rows of a `(4, 4)` matrix by `factor`."""
    if factor == 1.0:
        return matrix
    scaled = np.array(matrix, dtype=float, copy=True)
    scaled[:3, :] *= factor
    return scaled


def _like_header(like: tx.Any) -> tx.Optional[nb.Nifti1Header]:
    """Return the header to copy from a `like` template, or `None`.

    The template is a nibabel header or image, an object with a header, or the
    path of a NIfTI file.
    """
    if like is None:
        return None
    if isinstance(like, (nb.Nifti1Header, nb.Nifti2Header)):
        return like
    if isinstance(like, (nb.Nifti1Image, nb.Nifti2Image)):
        return like.header
    header = getattr(like, "header", None)
    if header is not None:
        return header
    if isinstance(like, (str, path.PathLike)):
        return _load_nifti_header(like)
    return None


def _apply_like(image: _NiftiObject, like: tx.Any) -> _NiftiObject:
    """Copy the description and intent of a template onto an image.

    The intent is copied only if the image has none. Fields that change the
    geometry or how the voxels are read back are never copied. The image is
    returned for chaining.
    """
    header = _like_header(like)
    if header is None:
        return image
    target = image.header
    try:
        target["descrip"] = header["descrip"]
    except (KeyError, ValueError):
        pass
    if int(target["intent_code"]) == 0:
        for field in (
            "intent_code",
            "intent_name",
            "intent_p1",
            "intent_p2",
            "intent_p3",
        ):
            try:
                target[field] = header[field]
            except (KeyError, ValueError):
                pass
    return image


def _apply_overrides(
    image: _NiftiObject, overrides: tx.Mapping
) -> _NiftiObject:
    """Apply header overrides and return the image.

    The keys `dtype`, `intent` and `descrip` set the stored data type, the
    intent and the description; any other key sets the header field of that
    name.
    """
    overrides = dict(overrides or {})
    dtype = overrides.pop("dtype", None)
    intent = overrides.pop("intent", None)
    descrip = overrides.pop("descrip", None)
    if dtype is not None:
        image.header.set_data_dtype(dtype)
    if intent is not None:
        image.header.set_intent(intent)
    if descrip is not None:
        if isinstance(descrip, str):
            descrip = descrip.encode("utf-8", "replace")
        image.header["descrip"] = descrip
    for field, value in overrides.items():
        image.header[field] = value
    return image


def _image_with_geometry(
    data: ArrayProtocol,
    transformation: Transformation,
    transformations: tx.Sequence[Transformation],
    like: tx.Any = None,
    overrides: tx.Optional[tx.Mapping] = None,
) -> _NiftiObject:
    """Build a NIfTI image from data and their voxel-to-world geometry.

    The preferred transformation provides the sform, the spacing of the later
    axes and the units, and the qform comes from [`_sform_and_qform`][]. A
    spatial unit that NIfTI cannot store is converted, and each matrix is
    scaled accordingly.

    The axes are written in NIfTI order, so data whose voxel space declares
    another order are transposed (lazily for lazy arrays). Singleton axes are
    inserted where NIfTI needs them: `(x,y,t)->(X,Y,1,T)`,
    `(x,y,z,c)->(X,Y,Z,1,C)` and `(x,y,c)->(X,Y,1,1,C)`. Header fields are then
    copied from `like`, and the `overrides` are applied last.
    """
    preferred_output = getattr(transformation, "output", None)
    space, time = _xyzt_labels(preferred_output)

    voxel_axes = _voxel_axes(transformation, data)
    sform_raw, others, timed, layout = _nifti_geometry(
        transformation, voxel_axes
    )
    scode = _reference_code(preferred_output)
    qform_raw, qcode, qform_output = _sform_and_qform(
        transformations, sform_raw, scode, voxel_axes, layout
    )
    if layout is not None:
        data = layout.apply(data)

    sform = _scale_spatial(sform_raw, _unit_scale(preferred_output, space))
    qform = _scale_spatial(qform_raw, _unit_scale(qform_output, space))

    image = _new_nifti(data, sform)
    _set_other_axes(image, others, timed)
    _apply_like(image, like)
    image.header.set_sform(sform, code=scode)
    image.header.set_qform(qform, code=qcode)
    image.header.set_xyzt_units(space, time)
    _apply_overrides(image, overrides)
    return image
