"""The shared NIfTI-reading and NIfTI-writing machinery behind every
NIfTI-based image and transformation format."""

__all__ = ["NiftiMetadata", "NiftiParser"]

# stdlib
import gzip
import inspect
import warnings
from io import BytesIO
from urllib.parse import urlsplit

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, replace

from brainhops._core import path
from brainhops._core.streams import open_compressed

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.metadata import (
    MetadataField,
    apply_loss_policy,
    preferred_storage,
)
from brainhops.datamodel.systems import (
    CoordinateSystem,
    _axes_or_unknown,
)
from brainhops.datamodel.transformations import Transformation
from brainhops.datamodel.units import (
    is_physicalunit,
    is_spaceunit,
    is_timeunit,
)
from brainhops.io.base._geometry import (
    embed_affine,
    ras_conversion,
    reduce_to_affine,
)
from brainhops.io.base._nifti_metadata import (
    NiftiMetadata,
    _shape,
    copy_record,
    set_time_step,
    time_step,
)
from brainhops.io.base._nifti_units import nifti_unit_meters, unit_to_nifti
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    WriterError,
    WriterNotImplementedError,
    preserve_position,
)
from brainhops.io.metadata._sync import sync_metadata

# typing
_NiftiObject = tx.Union[nb.Nifti1Header, nb.Nifti1Image]


# The axes of a NIfTI array, by position. They are the axes of its voxel
# space, so they count samples (`IndexUnit`): reversing one shifts its
# origin by one less than its extent. A reader that builds a physical space
# from them gives them its own unit.
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
    # number of points / vertices / triangles / ...
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
    # --- GIFTI ---
    2001: _FLAT_AXES_TIME,  # TIME_SERIES
    2002: _FLAT_AXES_CHANNEL,  # NODE_INDEX
    2003: _FLAT_AXES_CHANNEL,  # RGB_VECTOR
    2004: _FLAT_AXES_CHANNEL,  # RGBA_VECTOR
    2005: _FLAT_AXES_CHANNEL,  # SHAPE
    # --- FSL ---
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
"""
Intent codes that mark a NIfTI file as holding a deformation field.

A NIfTI file is legitimately an image *and* a set of affines *and*,
sometimes, a field -- so the container alone cannot say which object the
caller wants. The intent code can, and is what lets sniffers score
themselves instead of relying on an arbitrary precedence between kinds.
"""

_NIFTI_FSL_INTENTS = frozenset({2006, 2007, 2008, 2009})
"""FSL-specific field intent codes, decoded by the FSL readers.

A file with one of these codes is left to the FSL readers rather than
claimed by the generic field reader, which does not decode FSL's storage
conventions.
"""

_NIFTI_INTENT_NONE = 0
"""Intent code of a plain image: no specialized interpretation."""

_NIFTI_INTENT_DISPVECT = 1006
"""
Intent code of a field of displacement vectors.

The NIfTI-1 standard reserves it "specifically for displacements", and
ITK 5.4 and later reads a three-component `DISPVECT` image as RAS
displacements in millimetres. brainhops reads it the same way, and
writes it only for displacement fields.
"""

_NIFTI_INTENT_VECTOR = 1007
"""
Intent code of a generic vector image.

The NIfTI-1 standard reserves it "for any other type of vector" than a
displacement. brainhops writes its fields of RAS coordinates with it,
as SPM writes its `y_` deformations (coordinate maps), and ITK writes it
for every vector image unless told otherwise, so it is also the code of
ITK's (LPS) displacement fields. It says nothing about the frame its
vectors are in; the intent name `"Mapping"` (see below) marks the RAS
coordinate maps.
"""

_NIFTI_INTENT_NAME_NIFTYREG = "NREG_TRANS"
"""
The intent name NiftyReg gives every transformation it writes.

NiftyReg stores its deformation and displacement fields and its
control-point grids as `VECTOR` (1007) images named `"NREG_TRANS"`, and
tells them apart with `intent_p1` (`reg-lib/cpu/Maths.hpp`,
`NREG_TRANS_TYPE`). The name is evidence of a NiftyReg file, which only
the NiftyReg readers decode, so the generic `VECTOR` readers decline it.
"""

_NIFTI_INTENT_NAME_MAPPING = "Mapping"
"""
The intent name SPM gives a field of coordinates (`y_` files).

brainhops writes it next to `VECTOR` on a field of RAS coordinates, so
the file says what its vectors are, not only that they are vectors. ITK's
`NiftiImageIO` never writes an intent name, so neither ITK nor ANTs
files carry it, and it also tells such a map from an ITK (LPS) field.
"""


_NIFTI_XCODES = {
    0: "unknown",
    1: "scanner",
    2: "aligned",
    3: "talairach",
    4: "mni",
    5: "template",
}
"""
The world space each NIfTI xform code names.

A qform or sform code labels the world space its matrix maps voxels into.
The reader names each affine after its code, and the writer reads that
name back to choose the code to store.
"""

_NIFTI_XFORM_CODE_BY_NAME = {
    name: code for code, name in _NIFTI_XCODES.items()
}
"""The xform code for a world-space name, the reverse of `_NIFTI_XCODES`."""

_QFORM_NAME = "qform"
"""The name the reader gives the rigid voxel-to-RAS affine of the qform."""

_NIFTI_DEFAULT_XFORM_CODE = 2
"""
The xform code stored when the world space names no known reference.

NIfTI ignores a form whose code is zero, so a form that carries real
geometry is stored with a non-zero code. The value `2` is NIfTI's
"aligned" code.
"""

_NIFTI1_MAX_DIM = 2**15 - 1
"""
The largest array dimension NIfTI-1 can store.

NIfTI-1 records each dimension in a signed 16-bit field. An array with a
larger extent along any axis is written as NIfTI-2 instead.
"""


def _nifti_intent(header: "_NiftiObject") -> tx.Optional[int]:
    """The intent code of a NIfTI header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return int(header["intent_code"])
    except Exception:
        return None


def _nifti_intent_name(header: "_NiftiObject") -> tx.Optional[str]:
    """The intent name of a NIfTI header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return str(header.get_intent()[2])
    except Exception:
        return None


def _nifti_vector_field(data: ArrayProtocol) -> ArrayProtocol:
    """
    Drop the singleton time axis of a NIfTI vector field.

    NIfTI stores a 3-D vector field as `(X, Y, Z, 1, 3)`, with the
    components in the fifth axis and a singleton in place of time. Every
    reader of a field wants it as `(X, Y, Z, 3)`, or it would sample it
    as a 4-D grid of 3-vectors, so the singleton is dropped here, in one
    place. Any other array -- a 4-D `(X, Y, Z, 3)` field, a 5-D one with
    several time points -- is returned as it is, for the caller to
    accept or refuse.

    The shared `NiftiParser.data` keeps the axis on purpose: it is the
    array that matches the header, and is what writers that copy a file
    back store.
    """
    shape = tuple(int(d) for d in data.shape)
    if len(shape) == 5 and shape[3] == 1:
        data = data[:, :, :, 0, :]
    return data


def _nifti_shape(header: "_NiftiObject") -> tx.Optional[tx.Tuple[int, ...]]:
    """The data shape of a NIfTI header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return tuple(int(d) for d in header.get_data_shape())
    except Exception:
        return None


# The `metadata` field of every NIfTI-based class. It is declared again
# on `NiftiBasedTransformation`: there, `Transformation` comes before
# `NiftiParser` in the MRO, and its generic declaration would win.
NiftiMetadataField = MetadataField[
    NiftiMetadata,
    Factory(),
    tx.Doc(
        """
        The metadata of the file: the common vocabulary decoded from the
        header (description, display range, slice timing, ...), with the
        header itself as its raw record (`metadata.raw`). A field set
        here is written over the header on save; see
        [`NiftiMetadata`][brainhops.io.images.nifti.NiftiMetadata].
        """
    ),
]


class NiftiParser(DataModelBase, BinaryFileParserWriter):
    """
    Base class for objects that are encoded by a NIfTI file.

    This class is a base for `NiftiBasedImage` and `NiftiBasedTransformation`.
    """

    HINTS = ("nifti",)

    # --- NIfTI API ----------------------------------------------------

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

    metadata: NiftiMetadataField

    @property
    def header(self) -> tx.Optional[nb.Nifti1Header]:
        """
        The NIfTI header associated with this object.

        If a header was explicitly set by the user (at construction or
        later), this will be pointing to that header.

        Otherwise, if the object was created from a NIfTI header, this
        will be pointing to that header.

        Otherwise, if the object was created from a NIfTI image, this
        will be pointing to the header of that image.

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
        # A new record: the metadata is read from it again.
        if value is not None:
            self.metadata = NiftiMetadata.from_raw(value, image=self)

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        self._sync_metadata()

    def _sync_metadata(self) -> None:
        """
        Read the metadata from the header, when it is not its record yet.

        An object built from a header (or a `nibabel` image) holds the
        metadata decoded from it, with the header as its record. A
        `metadata` given along with it (explicitly, or carried over by
        `replace(image, header=...)`) keeps the fields that changed in
        it, over the decoded ones, and they count as changes on write
        (see `FileBasedMetadata.update_from_raw`).
        """
        if self.header is not None:
            sync_metadata(self, NiftiMetadata, self.header, image=self)

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, read lazily from `image` and cached, unless
        it has been set explicitly.

        The axes that the intent code marks as irrelevant, such as a
        singleton axis before a vector's components, are dropped.
        """
        if getattr(self, "_data", None) is not None:
            return self._data

        if self.image is not None:
            # Get the nibabel array
            data = get_array_backend().asarray(self.image.dataobj)

            # Drop irrelevant axes as specified by the intent code.
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
        """The voxel coordinate system, derived from `header`, unless it
        has been set explicitly.

        The axes that the intent code marks as irrelevant are dropped.
        `None` when there is no header to derive it from.
        """
        if getattr(self, "_system", None) is not None:
            return self._system

        if self.header is None:
            return None

        # Drop irrelevant axes, as specified by the intent code. The axes
        # already count samples, as the axes of a voxel space do.
        axes = [
            axis
            for axis in _nifti_to_axes(self.header)
            if axis.name is not None
        ]
        # A NIfTI array is F-ordered: the first axis changes fastest.
        return CoordinateSystem(axes=axes, name="voxel", order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    # --- BinaryFileParser API -----------------------------------------

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """
        Build the object from a NIfTI file.

        A local path is handed to `nibabel` by name, so that it owns the
        file handle and can memory-map the voxels: its array proxy reads
        them lazily, long after the call returns. A remote path is opened
        through its own backend instead, since `nibabel` would take its
        name for a local file. See `_load_nifti`.
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
        """Build the object from an open NIfTI file object, image data
        included when the stream allows reading it."""
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
        """Build the object from bytes in NIfTI format."""
        return cls.from_fileobj(BytesIO(data), **kwargs)

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """Build the object from an already-loaded `nibabel` header or
        image."""
        if isinstance(nifti, nb.Nifti1Header):
            return cls(header=nifti, **kwargs)
        if isinstance(nifti, nb.Nifti1Image):
            return cls(image=nifti, header=nifti.header, **kwargs)
        raise TypeError(f"Expected a NIfTI image or header, got {type(nifti)}")

    # --- FileParserWriter API -----------------------------------------

    def to_nibabel(self, **kwargs) -> nb.Nifti1Image:
        """
        Build the `nibabel` image that encodes this object.

        Each concrete NIfTI format overrides this method to describe how
        its own contents map onto a NIfTI image. The other writer methods
        are defined in terms of this one.
        """
        cls = type(self)
        raise WriterNotImplementedError(
            f"{cls.__name__} does not know how to write itself to NIfTI."
        )

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """
        Write the object to a NIfTI file.

        A path is written gzipped when its name ends in `.gz`: a local
        path is handed to `nibabel` by name, and a remote one is opened
        through its own backend. See `_save_nifti`. A file-like object is
        written the uncompressed NIfTI bytes.
        """
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, path.PathLike):
            _save_nifti(self.to_nibabel(**kwargs), file)
            return
        return super().to_file(file, **kwargs)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the uncompressed NIfTI-1 encoding of the object."""
        return self.to_nibabel(**kwargs).to_bytes()

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the uncompressed NIfTI-1 encoding of the object to an
        open file object."""
        file.write(self.to_bytes(**kwargs))

    # --- BinaryFileSniffer API ----------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        *,
        version: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score how confident the class is that an open file object
        holds a NIfTI-1 or NIfTI-2 header, or a header of the given
        `version` when one is passed."""

        # --- If nifti version not provided, try both NIfTI-1 and NIfTI-2
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

        # --- Version hint is provided, use appropriate nibabel class
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
                # Check the magic before asking `nibabel`, which parses
                # anything it is given -- and warns about the garbage
                # it finds in a file that is not a NIfTI at all.
                start = _tell(f)
                head = f.read(_NIFTI_HEADER_SIZES[version])
                if not _has_nifti_magic(head, version):
                    raise ValueError(f"No NIfTI-{version} magic number")
                if start is not None:
                    f.seek(start)
                else:
                    f = BytesIO(head)
                # A probe must not leak warnings: actual reads still
                # surface what `nibabel` has to say.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obj = NiftiHeader.from_fileobj(f, **nbkwargs)
                    result = cls.sniff_nibabel(obj, **kwargs)
        except Exception as e:
            base_error = e
            result = Confidence.NO

        if result:
            return result

        # Cannot parse this content -> return False or error
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
        """Score how confident the class is that an already-loaded
        `nibabel` header or image matches this format.

        The header's magic number is checked first. A header that passes
        is then scored for how well it matches this particular format,
        as opposed to another kind of NIfTI-based format.
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
            # The magic number only says "this is a NIfTI". How *well* it
            # matches this particular class is for the subclass to say.
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
        """
        How well a valid NIfTI header matches *this* class.

        Called once the magic number has been checked, so the answer is
        never "not a NIfTI" -- it is "how likely is this NIfTI to be the
        kind of object I build". A displacement field and a plain image
        live in the same container and can only be told apart by the
        intent code, or failing that the data shape.

        !!! note "Why this is a separate, overridable method"
            It is the one piece of sniffing that differs per format, so
            it is a hook rather than inline code: `sniff_nibabel` keeps
            the part every NIfTI format shares -- validating the
            container -- and delegates the rest. Overriding
            `sniff_nibabel` directly would make each format re-implement
            the magic-number check, and get it subtly wrong.

            It is private because it is an extension point for formats in
            this package, not something callers invoke: ask `sniff` which
            format matches, or a concrete format how confident it is.

        Returns
        -------
        score : float
            Confidence in `[0, 1]`; `Confidence.MAYBE` by default,
            meaning "readable, nothing more".
        """
        return Confidence.MAYBE

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that bytes hold a NIfTI-1 or
        NIfTI-2 header."""
        kwargs["error"] = error
        return cls.sniff_fileobj(BytesIO(data), **kwargs)


# ----------------------------------------------------------------------
#   READING AND WRITING THROUGH A PATH
# ----------------------------------------------------------------------
#
# `nibabel` opens a file it is handed by name with the built-in `open`,
# so it takes any name for a local file: given `s3://bucket/x.nii` or
# `memory://x.nii`, it looks for a local file of that name. The helpers
# below are the one place in `io` that hands `nibabel` a path. A local
# path goes to `nibabel` by name, so the voxels are memory-mapped as
# before. A remote path is opened through its own backend, and `nibabel`
# is handed the open file.

# The protocols of a path on the local file system, as `bagof.paths`
# reports them: none, `file://` and `local://`.
_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})


def _is_local(file: path.FilenameLike) -> bool:
    """Whether a path names a file that `nibabel` can open by name."""
    return path.Path(file).protocol.lower() in _LOCAL_PROTOCOLS


# A NIfTI image or header class, by NIfTI version.
_NIFTI_IMAGES = {1: nb.Nifti1Image, 2: nb.Nifti2Image}
_NIFTI_HEADERS = {1: nb.Nifti1Header, 2: nb.Nifti2Header}


def _tell(fileobj: tx.IO) -> tx.Optional[int]:
    """The position of a stream, or `None` if it cannot seek back."""
    try:
        return fileobj.tell() if fileobj.seekable() else None
    except Exception:
        return None


# The size of the header, and where and what its magic string is, by
# NIfTI version. NIfTI-1 keeps it near the end of its header, NIfTI-2
# right after `sizeof_hdr`, followed by a DOS/Unix line-ending check.
_NIFTI_HEADER_SIZES = {1: 348, 2: 540}
_NIFTI_MAGICS = {
    1: (344, (b"n+1\0", b"ni1\0")),
    2: (4, (b"n+2\0\r\n\032\n", b"ni2\0\r\n\032\n")),
}


def _has_nifti_magic(head: bytes, version: int) -> bool:
    """
    Whether the first bytes of a (decompressed) file hold the header
    size and magic string of a NIfTI header of the given version.

    This is cheap and exact, so a file that is not a NIfTI is turned
    away before `nibabel` is asked to make sense of it.
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
    """
    The NIfTI version of an open, possibly gzipped, file object: 2 if
    its header size is that of NIfTI-2, else 1.

    The stream is left where it was. One that cannot seek back is not
    peeked at, and taken for NIfTI-1.
    """
    start = _tell(fileobj)
    if start is None:
        return 1
    head = open_compressed(fileobj).read(4)
    fileobj.seek(start)
    # `sizeof_hdr`, in either byte order.
    sizes = {int.from_bytes(head, order) for order in ("little", "big")}
    return 2 if 540 in sizes else 1


def _accepted(func: tx.Callable, kwargs: tx.Mapping[str, tx.Any]) -> dict:
    """
    The keyword arguments, of `kwargs`, that `func` accepts: all of them
    if it takes `**kwargs`.

    The options for reading a NIfTI file differ with how it is read: a
    file `nibabel` opens by name takes `mmap` and `keep_file_open`, a
    stream does not. Each `nibabel` call is handed the ones it knows.
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
    """
    Build a NIfTI image from an open, uncompressed file object.

    `from_stream` arrived in nibabel 5.0; before, the stream goes in a
    file map, as `nibabel` 4's own `from_bytes` does.
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
    """
    Write a NIfTI image to an open file object, uncompressed.

    `to_stream` arrived in nibabel 5.0; before, the stream goes in a file
    map, as `nibabel` 4's own `to_bytes` does.
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
    """
    Build a NIfTI-1 or NIfTI-2 image from an open, possibly gzipped, file
    object.

    A stream has no name for `nibabel` to tell a `.gz` from, so the
    compression is sniffed from its magic bytes. The image's array proxy
    reads the voxels from `fileobj` lazily, so the caller keeps it open
    for as long as they may be read. On failure, the stream is put back
    where it was.

    `nibabel` cannot memory-map a stream: given `mmap`, it falls back to
    reading the voxels, and `keep_file_open` has no effect on an open
    file object.
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
    """
    Load a NIfTI image from a path, local or remote.

    A local path is handed to `nibabel`'s `from_filename`, which
    memory-maps the voxels and reads them only when asked. A remote path
    is opened through its own backend (universal-pathlib or
    cloudpathlib, through `bagof.paths`) and read into memory: the array
    proxy reads long after this returns, when the remote stream would be
    closed, and reading the voxels fetches them all anyway. Each is
    handed the `kwargs` it accepts; see `_nifti_from_stream`.
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
    """Read the header of a NIfTI file at a path, local or remote,
    without reading its voxels."""
    if _is_local(file):
        return _load_nifti(file).header
    with path.Path(file).open("rb") as f:
        header_class = _NIFTI_HEADERS[_nifti_version(f)]
        return header_class.from_fileobj(open_compressed(f))


def _save_nifti(
    image: tx.Union[nb.Nifti1Image, nb.Nifti2Image], file: path.FilenameLike
) -> None:
    """
    Write a NIfTI image to a path, local or remote, gzipped when its name
    ends in `.gz`.

    A local path is handed to `nibabel`'s `to_filename`, which picks the
    compression from the extension. A remote path is opened through its
    own backend, and the image written to the stream.
    """
    if _is_local(file):
        image.to_filename(str(path.Path(file)))
        return
    # The name is read from the URL's text: a backend may not know it,
    # and a query (`?token=...`) is not part of it. A stream is written
    # what it is given, so compression is ours to add. `open_compressed`
    # only reads (`indexed_gzip` cannot write), so this is `gzip`'s.
    compress = urlsplit(str(file)).path.lower().endswith(".gz")
    with path.Path(file).open("wb") as f:
        if compress:
            with gzip.GzipFile(fileobj=f, mode="wb") as gz:
                _image_to_stream(image, gz)
        else:
            _image_to_stream(image, f)


def _nifti_to_axes(header: nb.Nifti1Header) -> tx.List[Axis]:
    """
    Compute the axes of a NIfTI file, based on its header.

    This function takes into account the intent code of the NIfTI file
    to change the names and types of the axes, if necessary.

    Axes that are deemed irrelevant by the intent code are given a name
    of `None`. Every axis is an axis of the voxel space, so its unit is
    the index unit.
    """

    ndim = len(header.get_data_shape())

    # Get names and types of existing axes
    axes = _NIFTI_AXES[:ndim]

    # Apply specific knowledge from intent code.
    # `header[...]` hands back a 0-d numpy array, which is unhashable and
    # cannot be looked up in a dict, so go through `_nifti_intent`.
    intent = _nifti_intent(header)
    if intent in _NIFTI_SPECIFIC_AXES:
        axes_map = _NIFTI_SPECIFIC_AXES[intent]
        axes = [axes_map.get(i, axis) for i, axis in enumerate(axes)]

    # The tables above are module-level templates: hand out copies, so a
    # system built from them never holds -- and never changes -- the
    # shared ones.
    return [replace(axis) for axis in axes]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _new_nifti(
    data: ArrayProtocol,
    affine: tx.Optional[np.ndarray],
    dtype: tx.Any = None,
) -> _NiftiObject:
    """
    Build a `nibabel` image, reporting an unwritable dtype as a writer error.

    The array is stored as it is given. Any object that follows the array
    protocol is accepted, including a `cupy` or `dask` array, so a lazy or
    device array is not materialized into `numpy` here.

    NIfTI-1 records each dimension in a signed 16-bit field. An array whose
    extent along any axis exceeds that limit is written as NIfTI-2, which
    stores dimensions in 64-bit fields. A smaller array is written as
    NIfTI-1.

    NIfTI-1 cannot store some array types, such as 64-bit integers.
    `nibabel` raises a bare `ValueError` for one, which is re-raised as a
    `WriterError` that names the dtype. `dtype`, when given, is the type
    to store the array as instead (`nibabel` casts, or scales, on write).
    """
    shape = tuple(int(d) for d in getattr(data, "shape", ()) or ())
    image_cls = nb.Nifti1Image
    if any(d > _NIFTI1_MAX_DIM for d in shape):
        image_cls = nb.Nifti2Image
    try:
        if dtype is not None and np.dtype(dtype) != getattr(
            data, "dtype", None
        ):
            return image_cls(data, affine, dtype=np.dtype(dtype))
        return image_cls(data, affine)
    except ValueError as error:
        dtype = getattr(data, "dtype", "unknown")
        raise WriterError(
            f"NIfTI cannot store an array of type {dtype}: {error}"
        ) from error


def _embed_affine(matrix: np.ndarray) -> np.ndarray:
    """
    Embed a homogeneous voxel-to-world matrix in the `(4, 4)` NIfTI stores.

    See [`embed_affine`][brainhops.io.base._geometry.embed_affine].
    """
    return embed_affine(matrix, "NIfTI")


def _voxel_to_ras(xform: Transformation) -> np.ndarray:
    """
    Compute the `(4, 4)` voxel-to-RAS matrix of a transformation.

    The transformation must map voxel coordinates to a world space. An
    affine transformation is used directly. A transformation of any other
    kind that reduces to an affine, such as a `Scaling` or a `Sequence` of
    affines, is converted first. A two-dimensional affine is embedded in a
    `(4, 4)` matrix, which is the shape NIfTI stores. The world space is
    turned into RAS from the anatomical orientation of its axes.

    A transformation that has no affine representation, such as a
    displacement field, cannot be written as NIfTI geometry, and raises
    `UnrepresentableTransformationError`. A spatial transformation of more
    than three dimensions raises `WriterError`.
    """
    affine = reduce_to_affine(xform, "NIfTI")
    matrix = affine.homogeneous_matrix
    if matrix is None:
        matrix = np.eye(4)
    matrix = np.asarray(matrix, dtype=float)
    world_ndim = matrix.shape[0] - 1
    matrix = _embed_affine(matrix)

    world = _closed_world(getattr(affine, "output", None), world_ndim)
    conversion = ras_conversion(world)
    return conversion @ matrix


def _closed_world(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> tx.Optional[CoordinateSystem]:
    """
    The world space, closed to the `ndim` axes the affine maps into.

    NIfTI cannot store an open world space, one whose axes hold `...`, so
    it is closed from the shape of the voxel-to-world matrix. The axes that
    `...` stands for carry no orientation, as any axis NIfTI knows nothing
    about. A world space that states more axes than the matrix has rows
    raises `WriterError`.
    """
    if system is None or system.ndim is not None:
        return system
    try:
        return system.expand(ndim)
    except ValueError as error:
        raise WriterError(
            f"The world space of this transformation states more axes than "
            f"the {ndim} its voxel-to-world matrix maps into, so it cannot "
            f"be written as NIfTI geometry."
        ) from error


def _reference_code(system: tx.Optional[CoordinateSystem]) -> int:
    """
    The NIfTI xform code for a world space, which is never zero.

    The code names the world reference the matrix maps into, one of
    scanner, aligned, talairach, mni or template. It is taken from the
    world space's name only when that name is one of those references. An
    orientation name, an unnamed space, or an unrecognized reference yields
    the "aligned" code, so a form that carries real geometry is never
    stored with a zero code, which `nibabel` would ignore.
    """
    name = getattr(system, "name", None)
    code = _NIFTI_XFORM_CODE_BY_NAME.get(name, _NIFTI_DEFAULT_XFORM_CODE)
    return code or _NIFTI_DEFAULT_XFORM_CODE


def _sform_and_qform(
    transformations: tx.Sequence[Transformation],
    sform: np.ndarray,
    scode: int,
) -> tx.Tuple[np.ndarray, int, tx.Optional[CoordinateSystem]]:
    """
    Choose the qform matrix, code and world space to store with an sform.

    The matrix and the code always describe the same world space. The
    first transformation other than the preferred one whose world space is
    named after a known reference provides both. Failing that, the rigid
    edge the reader names "qform" provides the matrix, under the sform's
    code. Failing that, the qform is the sform, which `nibabel` reduces to
    its rigid part, under the sform's code.

    The world space that supplies the matrix is returned alongside it, so
    the caller can read the qform's own spatial unit rather than assuming
    it matches the sform's.
    """
    preferred = transformations[-1] if transformations else None

    for xform in transformations:
        if xform is preferred:
            continue
        output = getattr(xform, "output", None)
        name = getattr(output, "name", None)
        if _NIFTI_XFORM_CODE_BY_NAME.get(name):
            return (
                _voxel_to_ras(xform),
                _NIFTI_XFORM_CODE_BY_NAME[name],
                output,
            )

    for xform in transformations:
        output = getattr(xform, "output", None)
        if getattr(output, "name", None) == _QFORM_NAME:
            return _voxel_to_ras(xform), scode, output

    return sform, scode, getattr(preferred, "output", None)


def _space_unit_meters(
    system: tx.Optional[CoordinateSystem],
) -> tx.Optional[float]:
    """
    The size in meters of a world space's first spatial unit, or `None`.

    A space with no physical spatial unit -- every axis unspecified, or
    counting samples -- returns `None`.

    Only the axes the space states can carry a unit, so the `...` of an open
    space reads as it would once closed: as axes with no unit.
    """
    for axis in _axes_or_unknown(system):
        unit = getattr(axis, "unit", None)
        if is_physicalunit(unit) and is_spaceunit(unit):
            return float(unit.scale)
    return None


def _xyzt_labels(
    system: tx.Optional[CoordinateSystem],
) -> tx.Tuple[str, str]:
    """
    The NIfTI spatial and temporal labels for a world space.

    The first axis with a physical spatial unit gives the spatial label,
    and the first with a physical temporal unit gives the temporal label;
    both go through [`unit_to_nifti`][brainhops.io.base._nifti_units.
    unit_to_nifti], whose policies apply. A spatial unit NIfTI cannot store
    is given the nearest label it can (the affine is rescaled to match, see
    [`_unit_scale`][]). An axis whose unit is unspecified, or counts
    samples, says nothing about either label, which stays `"unknown"`. The
    labels are read from one world space, the preferred transformation's
    output, so a different edge does not change them.
    """
    space = time = None
    # The `...` of an open space carries no unit, as it would once closed.
    for axis in _axes_or_unknown(system):
        unit = getattr(axis, "unit", None)
        if not is_physicalunit(unit):
            continue
        if space is None and is_spaceunit(unit):
            space = unit_to_nifti(unit, "space", nearest=True)
        elif time is None and is_timeunit(unit):
            time = unit_to_nifti(unit, "time")
    return space or "unknown", time or "unknown"


def _unit_scale(system: tx.Optional[CoordinateSystem], label: str) -> float:
    """
    The factor that rescales a world space's affine into a NIfTI unit.

    A world space measured in one spatial unit is stored under the NIfTI
    label `label`, which is a different unit. The affine is multiplied by
    this factor so the stored geometry keeps the same physical size. A
    space whose unit is unknown, or a label NIfTI cannot store, leaves the
    factor at `1`.
    """
    stored = nifti_unit_meters(label)
    if stored is None:
        return 1.0
    meters = _space_unit_meters(system)
    if meters is None:
        return 1.0
    return meters / stored


def _scale_spatial(matrix: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale the spatial rows of a `(4, 4)` voxel-to-world matrix.

    The three spatial rows carry the world coordinates, so multiplying them
    by the factor converts those coordinates into another spatial unit. The
    bottom row is left unchanged.
    """
    if factor == 1.0:
        return matrix
    scaled = np.array(matrix, dtype=float, copy=True)
    scaled[:3, :] *= factor
    return scaled


def _like_header(like: tx.Any) -> tx.Optional[nb.Nifti1Header]:
    """
    Resolve a `like` template to the NIfTI header to copy fields from.

    The template may be a path to a NIfTI file, a `nibabel` image or
    header, or an object built by this package that carries a header. An
    object with no readable header resolves to `None`.
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
    """
    Copy non-encoding header fields from a template onto an image.

    The geometry of the written image always comes from the object being
    written, so the sform, the qform and their codes are never copied. The
    description is taken from the template. The intent is taken from the
    template only when the image has none of its own, so a field's own
    intent is preserved.

    Fields that change how the voxels are read back are never copied. The
    data scaling (`scl_slope`, `scl_inter`) rescales every value, and the
    stored data type reinterprets the bytes, so both are left as the image
    computed them from its own array.

    The image is returned so calls can be chained.
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


def _apply_metadata(
    image: _NiftiObject,
    obj: tx.Any,
    like: tx.Any = None,
    overrides: tx.Optional[tx.Dict[str, tx.Any]] = None,
    *,
    intent: bool = True,
    record: bool = True,
    data_type: bool = False,
    data: tx.Any = None,
) -> _NiftiObject:
    """
    Write the metadata of `obj` into the header of a freshly built image.

    In order, lowest precedence first:

    1. what is safe to keep from the record of the file that was read
       (`obj.metadata.raw`; see `copy_record`). `intent` says whether a
       record's intent may be kept: the transformation writers set their
       own. A writer whose header is already a copy of the record passes
       `record=False`;
    2. the `like` template, exactly as before (description, and the
       intent when none is set);
    3. the common fields that changed since the read (all of them for an
       object built in memory or converted from another format);
    4. with `data_type` (the image writer, which passes the values it
       writes as `data`), the stored data type and intensity scaling:
       the metadata's `data_type`, `scale_slope` and `scale_intercept`
       when the values fit them (see `preferred_storage`). A `dtype`
       override still wins. The writer has already stored the values
       accordingly; this step sets the header and reports.

    Caller overrides are applied afterwards. What cannot be written is
    reported, and the report handed to the loss policy: `on_loss`, popped
    from `overrides` when given there, or the policy in effect.
    """
    on_loss = None
    if overrides is not None:
        on_loss = overrides.pop("on_loss", None)
    metadata = getattr(obj, "metadata", None)
    if metadata is not None:
        # Another format's metadata (on a class that did not narrow its
        # field) is converted; what NIfTI cannot hold is in `report`.
        metadata, report = NiftiMetadata.writable(metadata)
        if record:
            copy_record(image.header, metadata.raw, intent=intent)
    _apply_like(image, like)
    if metadata is not None:
        metadata.update_raw(image.header, image=obj, on_loss=report)
        if data_type and (overrides or {}).get("dtype") is None:
            if data is None:
                data = image.dataobj
            dtype, slope, intercept = preferred_storage(
                metadata, data, on_loss=report
            )
            try:
                image.header.set_data_dtype(dtype)
            except Exception:
                report.approximated["data_type"] = (
                    f"NIfTI cannot store {dtype.name}"
                )
            if slope is not None:
                image.header.set_slope_inter(slope, intercept)
        elif not data_type:
            # A transformation is stored unscaled.
            changed = metadata._changed_fields()
            for name in ("scale_slope", "scale_intercept"):
                if changed.get(name) is not None:
                    report.lost[name] = changed[name]
        apply_loss_policy(report, on_loss, stacklevel=4)
    return image


def _apply_overrides(
    image: _NiftiObject, overrides: tx.Mapping
) -> _NiftiObject:
    """
    Apply caller-supplied header overrides onto an image.

    The overrides are applied last, after the derived geometry and after
    any `like` template, so an explicit value always wins. `dtype` sets the
    stored data type, `intent` sets the intent code, and `descrip` sets the
    description. Any other name is written straight to the header field of
    that name.

    The array's own data type is kept unless `dtype` is given, so the
    override is opt in. The image is returned so calls can be chained.
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
    owner: tx.Any = None,
) -> _NiftiObject:
    """
    Build a NIfTI image from data and its voxel-to-world geometry.

    The preferred transformation becomes the sform, and its world space
    supplies the sform code. The qform is the rigid edge among the
    transformations when present, and the rigid part of the sform
    otherwise.

    The spatial and temporal units are read from the preferred
    transformation's output space. A spatial unit NIfTI cannot store is
    converted to the nearest one it can, and each form's affine is scaled
    from its own world space's unit, so the stored geometry keeps its
    physical size.

    Non-encoding header fields are taken from the metadata of `owner`
    (the object being written) and from `like` when it is given (see
    `_apply_metadata`), and caller `overrides` are applied last so an
    explicit value wins.
    """
    preferred_output = getattr(transformation, "output", None)
    space, time = _xyzt_labels(preferred_output)

    sform_raw = _voxel_to_ras(transformation)
    scode = _reference_code(preferred_output)
    qform_raw, qcode, qform_output = _sform_and_qform(
        transformations, sform_raw, scode
    )

    sform = _scale_spatial(sform_raw, _unit_scale(preferred_output, space))
    qform = _scale_spatial(qform_raw, _unit_scale(qform_output, space))

    overrides = dict(overrides or {})
    # The stored type and scaling: `dtype=`, else the metadata's when the
    # values fit them (reported, if need be, by `_apply_metadata`).
    dtype, slope, intercept = preferred_storage(
        getattr(owner, "metadata", None),
        data,
        overrides.get("dtype"),
        on_loss="ignore",
    )
    stored = data
    if slope is not None:
        values = np.asarray(data, dtype=np.float64)
        stored = np.round((values - intercept) / slope).astype(dtype)
    image = _new_nifti(stored, sform, dtype)
    image.header.set_sform(sform, code=scode)
    image.header.set_qform(qform, code=qcode)
    image.header.set_xyzt_units(space, time)
    # The time step is geometry: the data model's, never the record's.
    step = time_step(transformations)
    if step is not None and len(_shape(image.header)) >= 4:
        set_time_step(image.header, step)
    _apply_metadata(image, owner, like, overrides, data_type=True, data=data)
    _apply_overrides(image, overrides)
    return image
