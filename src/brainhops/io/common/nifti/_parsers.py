"""The NIfTI parser."""

# stdlib
import warnings
from io import BytesIO

# dependencies
import nibabel as nb
import typing_extensions as tx

from brainhops._core import path
from brainhops._core.streams import open_compressed

# internals
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

# this format
from ._files import (
    _NIFTI_HEADER_SIZES,
    _accepted,
    _has_nifti_magic,
    _load_nifti,
    _nifti_from_stream,
    _save_nifti,
    _tell,
)
from ._header import (
    _nifti_to_axes,
    _NiftiObject,
)


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
