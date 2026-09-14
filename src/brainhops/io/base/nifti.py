__all__ = ["NiftiParser"]

# stdlib
from io import BytesIO

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
    data: np.ndarray, affine: tx.Optional[np.ndarray]
) -> nb.Nifti1Image:
    """
    Build a `nibabel` image, reporting an unwritable dtype as a writer error.

    NIfTI-1 cannot store some array types, such as 64-bit integers.
    `nibabel` raises a bare `ValueError` for one, which is re-raised as a
    `WriterError` that names the dtype.
    """
    array = np.asarray(data)
    try:
        return nb.Nifti1Image(array, affine)
    except ValueError as error:
        raise WriterError(
            f"NIfTI cannot store an array of type {array.dtype}: {error}"
        ) from error


def _voxel_to_ras(xform: Transformation) -> np.ndarray:
    """
    Compute the `(4, 4)` voxel-to-RAS matrix of a transformation.

    The transformation must map voxel coordinates to a world space. An
    affine transformation is used directly. A transformation of any other
    kind that reduces to an affine, such as a `Scaling` or a `Sequence` of
    affines, is converted first. A two-dimensional affine is embedded in a
    `(4, 4)` matrix, which is the shape NIfTI stores. A world space named
    "LPS" is flipped to RAS, which is the convention NIfTI stores.

    A transformation that has no affine representation, such as a
    displacement field, cannot be written as NIfTI geometry, and raises
    `UnrepresentableTransformationError`.
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

    if matrix.shape != (4, 4):
        # A 2D image yields a `(3, 3)` matrix; embed its rotation and
        # translation in a `(4, 4)` matrix whose extra axis is the identity.
        ndim = min(matrix.shape[0] - 1, 3)
        embedded = np.eye(4)
        embedded[:ndim, :ndim] = matrix[:ndim, :ndim]
        embedded[:ndim, 3] = matrix[:ndim, matrix.shape[1] - 1]
        matrix = embedded

    if getattr(affine.output, "name", None) == "LPS":
        # NIfTI stores voxel-to-RAS, so an LPS world is flipped on its
        # first two axes to become RAS.
        matrix = np.diag([-1.0, -1.0, 1.0, 1.0]) @ matrix

    return matrix


def _sform_and_qform(
    transformations: tx.Sequence[Transformation],
    sform: np.ndarray,
    scode: int,
) -> tx.Tuple[np.ndarray, int]:
    """
    Choose the qform matrix and code to store alongside an sform.

    The matrix and the code always describe the same world space. The
    first transformation other than the preferred one whose world space is
    named after an xform code provides both. Failing that, the rigid edge
    the reader names "qform" provides the matrix, under the sform's code.
    Failing that, the qform is the sform, which `nibabel` reduces to its
    rigid part, under the sform's code.
    """
    preferred = transformations[-1] if transformations else None

    for xform in transformations:
        if xform is preferred:
            continue
        name = getattr(getattr(xform, "output", None), "name", None)
        if name in _NIFTI_XFORM_CODE_BY_NAME:
            return _voxel_to_ras(xform), _NIFTI_XFORM_CODE_BY_NAME[name]

    for xform in transformations:
        name = getattr(getattr(xform, "output", None), "name", None)
        if name == _QFORM_NAME:
            return _voxel_to_ras(xform), scode

    return sform, scode


def _xyzt_units(
    transformations: tx.Sequence[Transformation],
) -> tx.Tuple[str, str]:
    """
    Read the spatial and temporal NIfTI unit labels off the axes.

    The first spatial axis that carries a recognized unit gives the
    spatial label, and the first temporal axis gives the temporal label.
    An axis with no recognized unit leaves the label "unknown".
    """
    space = "unknown"
    time = "unknown"
    for xform in transformations:
        for system in (
            getattr(xform, "input", None),
            getattr(xform, "output", None),
        ):
            for axis in getattr(system, "axes", None) or ():
                unit = getattr(axis, "unit", None)
                name = getattr(unit, "name", None)
                if not isinstance(name, str):
                    continue
                if space == "unknown" and name in _NIFTI_SPACE_UNITS:
                    space = _NIFTI_SPACE_UNITS[name]
                elif time == "unknown" and name in _NIFTI_TIME_UNITS:
                    time = _NIFTI_TIME_UNITS[name]
    return space, time


def _image_with_geometry(
    data: np.ndarray,
    transformation: Transformation,
    transformations: tx.Sequence[Transformation],
) -> nb.Nifti1Image:
    """
    Build a NIfTI image from data and its voxel-to-world geometry.

    The preferred transformation becomes the sform, and its world space's
    name becomes the sform code. The qform is the rigid edge among the
    transformations when present, and the rigid part of the sform
    otherwise. The spatial and temporal units are read off the axes.
    """
    sform = _voxel_to_ras(transformation)
    scode = _NIFTI_XFORM_CODE_BY_NAME.get(
        getattr(transformation.output, "name", None), 2
    )
    qform, qcode = _sform_and_qform(transformations, sform, scode)
    image = _new_nifti(data, sform)
    image.header.set_sform(sform, code=scode)
    image.header.set_qform(qform, code=qcode)
    image.header.set_xyzt_units(*_xyzt_units(transformations))
    return image
