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
    BinaryFileReader,
    BinaryFileWriter,
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


class NiftiReaderWriter(DataModelBase, BinaryFileReader, BinaryFileWriter):
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
            # `image1.header`
            NiftiReaderWriter(image1).header
            # `image2.header`
            NiftiReaderWriter(header=image2.header).header
            # `image2.header`
            NiftiReaderWriter(image1, header=image2.header).header
            obj = NiftiReaderWriter(image1)
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
        nifti: object,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a nibabel object matches this format.

        Only NIfTI-1 and NIfTI-2 images and headers can match. The
        size that the header records for itself (the `sizeof_hdr` field)
        is checked first, and a valid header is then scored with
        [`_score_nibabel`][]. Any other object, including other nibabel
        images and headers, gives `Confidence.NO`, or the requested error
        when `error` is set.

        Parameters
        ----------
        nifti : object
            The object to test, usually a nibabel image or header.
        error : bool or type[Exception], default=False
            Whether to raise an error when the object does not match. With
            `True`, the error is a `SnifferContentError`; an exception
            class is raised instead when one is given.
        **kwargs : dict
            Ignored.

        Returns
        -------
        float
            A score in `[0, 1]`.

        Raises
        ------
        SnifferContentError
            If the object does not match and `error` is `True`.
        """
        if isinstance(nifti, nb.Nifti1Image):
            return cls.sniff_nibabel(nifti.header, error=error, **kwargs)
        if isinstance(nifti, nb.Nifti2Header):
            expected = 540
        elif isinstance(nifti, nb.Nifti1Header):
            expected = 348
        else:
            expected = None
        if expected is not None:
            size = nifti["sizeof_hdr"]
            if size == expected:
                return cls._score_nibabel(nifti)
            message = f"Invalid NIfTI header size: {size} != {expected}"
        else:
            message = f"Not a NIfTI header: {type(nifti).__name__}"
        if error:
            if error is True:
                error = SnifferContentError
            raise error(message)
        return Confidence.NO

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
