__all__ = ["NiftiParser"]

# stdlib
from io import BytesIO
from math import log10

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

from brainhops._core import path
from brainhops._core.streams import open_compressed

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    ConversionError,
    Sequence,
    Transformation,
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

# typing
_NiftiObject = tx.Union[nb.Nifti1Header, nb.Nifti1Image]


_NIFTI_AXES = [
    Axis("x", "space"),
    Axis("y", "space"),
    Axis("z", "space"),
    Axis("t", "time"),
    Axis("c", "channel"),
    Axis("dim5"),
    Axis("dim6"),
]
_FLAT_AXES = {
    0: Axis("n"),  # number of points / vertices / triangles / ...
    1: Axis("x"),
    2: Axis("y"),
    3: Axis("z"),
}
_FLAT_AXES_CHANNEL = {**_FLAT_AXES, 4: Axis("c", "channel")}
_FLAT_AXES_TIME = {**_FLAT_AXES, 4: Axis("t", "time")}
_AXES_DISP = {4: Axis("c", "displacement")}
_NIFTI_SPECIFIC_AXES = {
    1004: {5: Axis("k", "channel")},  # GENMATRIX
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

_NIFTI_INTENT_NONE = 0
"""Intent code of a plain image: no specialized interpretation."""

_NIFTI_INTENT_DISPVECT = 1006
"""Intent code that marks a NIfTI file as a displacement or vector field."""


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

_NIFTI_SPACE_UNITS = {
    "meter": "meter",
    "millimeter": "mm",
    "micrometer": "micron",
    "micron": "micron",
}
"""The NIfTI spatial-unit label for a space unit's name."""

_NIFTI_TIME_UNITS = {
    "second": "sec",
    "millisecond": "msec",
    "microsecond": "usec",
}
"""The NIfTI time-unit label for a time unit's name."""

_NIFTI_SPACE_UNIT_METERS = {
    "meter": 1.0,
    "mm": 1e-3,
    "micron": 1e-6,
}
"""
The size in meters of each spatial unit NIfTI can store.

NIfTI records a spatial unit as one of `meter`, `mm` or `micron`. A unit
outside that set is written by converting it to the nearest of these three
and scaling the affine, so the stored geometry keeps the same physical
size.
"""

_RAS_FROM_ORIENTATION = {
    "left-to-right": (0, 1.0),
    "right-to-left": (0, -1.0),
    "posterior-to-anterior": (1, 1.0),
    "anterior-to-posterior": (1, -1.0),
    "inferior-to-superior": (2, 1.0),
    "superior-to-inferior": (2, -1.0),
}
"""
The RAS axis and sign that an anatomical orientation points along.

Each key is the value of an anatomical orientation carried by an axis. The
first element of the pair is the index of the RAS axis the orientation runs
along, and the second is its sign. This drives the conversion of a
voxel-to-world affine into voxel-to-RAS from the axes themselves, rather
than from the world space's name.
"""

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


def _nifti_shape(header: "_NiftiObject") -> tx.Optional[tx.Tuple[int, ...]]:
    """The data shape of a NIfTI header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return tuple(int(d) for d in header.get_data_shape())
    except Exception:
        return None


class NiftiParser(DataModelBase, BinaryFileParserWriter):
    """
    Base class for objects that are encoded by a NIfTI file.

    This class is a base for `NiftiBasedImage` and `NiftiBasedTransformation`.
    """

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

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
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
        if getattr(self, "_system", None) is not None:
            return self._system

        if self.header is None:
            return None

        # Drop irrelevant axes, as specified by the intent code.
        axes = [
            axis
            for axis in _nifti_to_axes(self.header)
            if axis.name is not None
        ]

        return CoordinateSystem(axes=axes, name="voxel")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    # --- BinaryFileParser API -----------------------------------------

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """
        Build the object from a NIfTI file.

        For a real path, `nibabel` is handed the path rather than an open
        stream, so that it owns the file handle. Its array proxy reads
        the voxels lazily, long after the call returns, and would find a
        closed file if we opened the stream ourselves.
        """
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, path.PathLike):
            if not file.exists():
                raise ParserExistsError(f"No such file: {file}")
            return cls.from_nibabel(nb.load(str(file), **kwargs))
        return super().from_file(file, **kwargs)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        # `from_file_map` wants `FileHolder`s, not raw file objects; handed
        # a bare stream it raises, and the header-only fallback below used
        # to swallow that -- so the image data was never read at all.
        # `ImageOpener` transparently handles gzipped streams.
        with preserve_position(fileobj):
            f = open_compressed(fileobj)
            try:
                holder = nb.FileHolder(fileobj=f)
                obj = nb.Nifti1Image.from_file_map(
                    {"header": holder, "image": holder}, **kwargs
                )
            except Exception:
                f = open_compressed(fileobj)
                obj = nb.Nifti1Header.from_fileobj(f, **kwargs)
        return cls.from_nibabel(obj)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> tx.Self:
        return cls.from_fileobj(BytesIO(data), **kwargs)

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
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

        For a real path, `nibabel` is handed the path rather than an open
        stream, so that it chooses gzip compression from the `.nii.gz`
        extension. A file-like object is written the uncompressed NIfTI-1
        bytes.
        """
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, path.PathLike):
            nb.save(self.to_nibabel(**kwargs), str(file))
            return
        return super().to_file(file, **kwargs)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the uncompressed NIfTI-1 encoding of the object."""
        return self.to_nibabel(**kwargs).to_bytes()

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
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
        kwargs["error"] = error
        return cls.sniff_fileobj(BytesIO(data), **kwargs)


def _nifti_to_axes(header: nb.Nifti1Header) -> tx.List[Axis]:
    """
    Compute the axes of a NIfTI file, based on its header.

    This function takes into account the intent code of the NIfTI file
    to change the names and types of the axes, if necessary.

    Axes that are deemed irrelevant by the intent code are given a name
    of `None`.
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

    return axes


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _new_nifti(
    data: ArrayProtocol, affine: tx.Optional[np.ndarray]
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
    `WriterError` that names the dtype.
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
    """
    Embed a homogeneous voxel-to-world matrix in the `(4, 4)` NIfTI stores.

    NIfTI stores a three-dimensional voxel-to-world affine. A
    two-dimensional map yields a `(3, 3)` homogeneous matrix, whose
    rotation and translation are placed in a `(4, 4)` matrix whose extra
    axis is the identity. A three-dimensional map is already `(4, 4)` and
    is returned unchanged.

    A spatial map of more than three dimensions has no NIfTI geometry to be
    written into, and raises `WriterError`.
    """
    out_dim = matrix.shape[0] - 1
    in_dim = matrix.shape[1] - 1
    if out_dim > 3 or in_dim > 3:
        raise WriterError(
            f"NIfTI stores a three-dimensional voxel-to-world affine, so a "
            f"{out_dim}D-to-{in_dim}D transformation cannot be written. "
            f"Reduce the transformation to three spatial dimensions before "
            f"writing it to NIfTI."
        )
    embedded = np.eye(4)
    embedded[:out_dim, :in_dim] = matrix[:out_dim, :in_dim]
    embedded[:out_dim, 3] = matrix[:out_dim, in_dim]
    return embedded


def _ras_conversion(system: tx.Optional[CoordinateSystem]) -> np.ndarray:
    """
    The `(4, 4)` matrix that maps a world space's coordinates into RAS.

    The matrix is built from the anatomical orientation carried by each
    axis, not from the world space's name. An LPS space becomes a flip of
    the first two axes, an RSA space becomes a permutation, and a space
    already in RAS becomes the identity.

    The conversion is derived only when all three leading axes carry a
    recognized anatomical orientation. When any of them does not, the
    identity is returned, so a space with no orientation is stored as it
    is.
    """
    axes = list(getattr(system, "axes", None) or [])[:3]
    mapping = []
    for axis in axes:
        value = getattr(getattr(axis, "orientation", None), "value", None)
        if value not in _RAS_FROM_ORIENTATION:
            return np.eye(4)
        mapping.append(_RAS_FROM_ORIENTATION[value])
    if len(mapping) != 3:
        return np.eye(4)
    conversion = np.zeros((4, 4))
    conversion[3, 3] = 1.0
    for column, (row, sign) in enumerate(mapping):
        conversion[row, column] = sign
    return conversion


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
    reduced = xform.compute() if isinstance(xform, Sequence) else xform
    error = None
    affine = reduced
    if not isinstance(affine, Affine):
        try:
            affine = reduced.to(Affine)
        except ConversionError as exc:
            error = exc
    if not isinstance(affine, Affine):
        # A field returns itself from a conversion to `Affine`, and a
        # `Sequence` of a non-affine reduces to one, so the result has to
        # be checked rather than trusted.
        raise UnrepresentableTransformationError(
            f"A {type(xform).__name__} cannot be written as NIfTI geometry: "
            f"NIfTI stores an affine voxel-to-world matrix, and this "
            f"transformation has no affine representation."
        ) from error

    matrix = affine.homogeneous_matrix
    if matrix is None:
        matrix = np.eye(4)
    matrix = np.asarray(matrix, dtype=float)
    matrix = _embed_affine(matrix)

    conversion = _ras_conversion(getattr(affine, "output", None))
    return conversion @ matrix


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


def _nifti_space_label(name: str, unit: tx.Any) -> tx.Optional[str]:
    """
    The NIfTI spatial label a unit is stored under, or `None`.

    A unit NIfTI can store directly returns its own label. A spatial unit
    outside NIfTI's set returns the label of the nearest of NIfTI's three
    spatial units, measured in log space. A unit that is not spatial, or
    whose size is unknown, returns `None`.
    """
    if name in _NIFTI_SPACE_UNITS:
        return _NIFTI_SPACE_UNITS[name]
    meters = getattr(unit, "scale", None)
    if getattr(unit, "type", None) != "space":
        return None
    if not isinstance(meters, (int, float)) or meters <= 0:
        return None
    label, _ = min(
        _NIFTI_SPACE_UNIT_METERS.items(),
        key=lambda item: abs(log10(meters) - log10(item[1])),
    )
    return label


def _space_unit_meters(
    system: tx.Optional[CoordinateSystem],
) -> tx.Optional[float]:
    """
    The size in meters of a world space's first spatial unit, or `None`.

    A NIfTI spatial label carried directly on an axis resolves to its own
    size in meters. Any other spatial unit resolves to its `scale`, which
    the unit reports in meters. A space with no usable spatial unit returns
    `None`.
    """
    for axis in getattr(system, "axes", None) or ():
        unit = getattr(axis, "unit", None)
        name = getattr(unit, "name", None)
        if not isinstance(name, str):
            continue
        for label, meters in _NIFTI_SPACE_UNIT_METERS.items():
            if _NIFTI_SPACE_UNITS.get(name) == label:
                return meters
        if getattr(unit, "type", None) == "space":
            meters = getattr(unit, "scale", None)
            if isinstance(meters, (int, float)) and meters > 0:
                return meters
    return None


def _xyzt_labels(
    system: tx.Optional[CoordinateSystem],
) -> tx.Tuple[str, str]:
    """
    The NIfTI spatial and temporal labels for a world space.

    The first spatial axis that carries a usable unit gives the spatial
    label, and the first temporal axis that carries a representable unit
    gives the temporal label. An axis with no usable unit leaves the label
    "unknown". The labels are read from one world space, the preferred
    transformation's output, so a different edge does not change them.
    """
    space = "unknown"
    time = "unknown"
    for axis in getattr(system, "axes", None) or ():
        unit = getattr(axis, "unit", None)
        name = getattr(unit, "name", None)
        if not isinstance(name, str):
            continue
        if space == "unknown":
            label = _nifti_space_label(name, unit)
            if label is not None:
                space = label
                continue
        if time == "unknown" and name in _NIFTI_TIME_UNITS:
            time = _NIFTI_TIME_UNITS[name]
    return space, time


def _unit_scale(system: tx.Optional[CoordinateSystem], label: str) -> float:
    """
    The factor that rescales a world space's affine into a NIfTI unit.

    A world space measured in one spatial unit is stored under the NIfTI
    label `label`, which is a different unit. The affine is multiplied by
    this factor so the stored geometry keeps the same physical size. A
    space whose unit is unknown, or a label NIfTI cannot store, leaves the
    factor at `1`.
    """
    if label not in _NIFTI_SPACE_UNIT_METERS:
        return 1.0
    meters = _space_unit_meters(system)
    if meters is None:
        return 1.0
    return meters / _NIFTI_SPACE_UNIT_METERS[label]


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
        return nb.load(str(like)).header
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

    Non-encoding header fields are taken from `like` when it is given, and
    caller `overrides` are applied last so an explicit value wins.
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

    image = _new_nifti(data, sform)
    _apply_like(image, like)
    image.header.set_sform(sform, code=scode)
    image.header.set_qform(qform, code=qcode)
    image.header.set_xyzt_units(space, time)
    _apply_overrides(image, overrides)
    return image
