"""An affine stored in an MNI transform file of one linear transform."""

__all__ = ["XfmAffine"]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import InvalidatorInAttribute, smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.metadata._base import (
    metadata_options,
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)

# this format
from ._formats import XfmFormat
from ._metadata import XfmMetadata
from ._raw import XfmRaw, _decode, _read_bytes, sniff_transforms

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------
# The matrix of a linear transform is small and lives in the record, so
# the data of an affine read from a file is a view of the record. A
# matrix that is assigned is held by the data model, in its private
# field, as for any affine, and it is pending until the affine is
# consolidated, which encodes a new record.


def _record(xform: "XfmAffine") -> tx.Optional[XfmRaw]:
    """Return the record of the metadata of an affine, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _matrix_to_model(matrix: tx.Any) -> np.ndarray:
    """Return the data of an affine from the matrix that a record stores.

    The record stores the three rows of the `(3, 4)` matrix, which are
    the data of the affine.
    """
    return np.asarray(matrix, dtype=np.float64)


def _matrix_to_disk(data: tx.Optional[ArrayProtocol]) -> tx.Tuple:
    """Return the matrix that a record stores for the data of an affine.

    The record stores the three rows of the `(3, 4)` matrix as tuples of
    floats. Data of `None`, which the data model reads as the identity,
    is stored as the identity.
    """
    if data is None:
        data = np.eye(4)[:-1]
    data = np.asarray(data, dtype=np.float64)
    return tuple(tuple(float(v) for v in row) for row in data)


def _linear(record: XfmRaw) -> XfmRaw.Linear:
    """Return the single linear transform of a record.

    Raises
    ------
    ParserContentError
        If the record holds another transform, or several.
    """
    blocks = record.transforms
    if len(blocks) == 1 and isinstance(blocks[0], XfmRaw.Linear):
        return blocks[0]
    kinds = ", ".join(type(block).__name__ for block in blocks) or "nothing"
    raise ParserContentError(
        f"An XfmAffine holds a file of one linear transform, and this file "
        f"holds {kinds}. Read it as an XfmTransform, which holds a chain "
        f"of transforms."
    )


def _decoded(block: XfmRaw.Linear) -> np.ndarray:
    """Return the read-only `(3, 4)` matrix that a linear transform applies.

    The matrix of an inverted transform is inverted, as libminc inverts
    it when it reads `Invert_Flag = True;`. The matrix is cached, but the
    writer reads the record and not the cached matrix, so an edit in
    place would be lost. A new matrix is assigned as data instead.
    """
    matrix = _matrix_to_model(block.matrix)
    if block.inverted:
        homogeneous = np.eye(4)
        homogeneous[:3] = matrix
        matrix = np.linalg.inv(homogeneous)[:3]
    matrix = np.array(matrix, dtype=np.float64)
    matrix.flags.writeable = False
    return matrix


# ----------------------------------------------------------------------
#   FORMAT
# ----------------------------------------------------------------------


# Text adapters supply byte decoding and encoding before the generic bases.
@register_format
class XfmAffine(
    XfmFormat,
    AffineTransformationFormat,
    TextFileReader,
    TextFileWriter,
    _xforms.Affine,
    TransformationFormat,
):
    """An affine stored in an MNI transform file (`.xfm`).

    This is the format of an `.xfm` file that holds a single linear
    transform, such as the `talairach.xfm` of a FreeSurfer subject. The
    affine maps RAS millimetre world coordinates of a source image to
    those of a target image: `talairach.xfm` maps the scanner RAS of the
    subject to MNI305 RAS. A file that concatenates several transforms,
    or that holds a grid transform, is read by
    [`XfmTransform`][brainhops.io.transformations.minc.XfmTransform].

    The file is short and is read in one pass, so the whole file is held
    as an [`XfmRaw`][] record by the [`XfmMetadata`][] of the affine. The
    matrix lives in that record, and the data of the affine is read from
    it. A linear transform with `Invert_Flag = True;` is read as the
    inverse of its matrix, as libminc reads it. The record describes the
    file as it was read, or as the affine was last consolidated. A matrix
    that is assigned, and an `input` or `output` that is set, are
    pending: they win over the record until [`consolidate`][] encodes
    them into a new record. An affine without a record, such as one built
    from data alone, holds its matrix as any affine does and has no
    metadata.

    !!! note "What is written"
        An affine that holds a record, and whose matrix, `input` and
        `output` were not set, is written as that record, byte for byte,
        with its comments. Any other affine is consolidated, and written
        as one linear transform, without `Invert_Flag`, after the
        comments of its record. Its `input` and `output` must be `RASmm`,
        and writing any other system raises
        [`UnrepresentableTransformationError`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xfm",)

    _metadata: NoRepr[tx.Optional[XfmMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    An affine built from data alone has no metadata, and setting its data
    does not create any. Assigning other metadata drops the matrix decoded
    from the old record. A matrix that was assigned is pending and is
    kept, and it wins over the matrix of the new record until the affine
    is consolidated.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The RAS world of the source, unless set explicitly."""
        return _systems.RASmm()

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The RAS world of the target, unless set explicitly."""
        return _systems.RASmm()

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(3, 4)` affine matrix, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record. Otherwise the
        matrix is decoded from the linear transform of the record, and is
        read-only, or `None` without a record.

        Raises
        ------
        ParserContentError
            If the record does not hold exactly one linear transform.
        """
        record = _record(self)
        if record is None:
            return None
        return _decoded(_linear(record))

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the text of a file as a file of one linear transform.

        A file that libminc reads, and that holds one linear transform, is
        certainly of this format. Any other text scores `Confidence.NO`,
        so that
        [`XfmTransform`][brainhops.io.transformations.minc.XfmTransform]
        reads the other files, and reports what is wrong with a file that
        libminc would not read.

        Parameters
        ----------
        text : str
            The text of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not a file of one
            linear transform.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        transforms = sniff_transforms(text)
        single = transforms is not None and len(transforms) == 1
        if single and isinstance(transforms[0], XfmRaw.Linear):
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error(
                "Not an MNI transform file of one linear transform, as "
                "libminc reads it."
            )
        return Confidence.NO

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file as a file of one linear transform.

        The lines are joined and scored by [`sniff_text`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines are not a file of one
            linear transform.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_text("\n".join(lines), error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of a file as a file of one linear transform.

        The bytes are decoded as [`XfmRaw.from_bytes`][] decodes them, and
        scored by [`sniff_text`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not a file of one
            linear transform.
        **kwargs : Any
            Options of the reader. `encoding` is used, and the others are
            ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        text = _decode(content, kwargs.get("encoding", "utf-8"))
        return cls.sniff_text(text, error=error)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a file as a file of one linear transform.

        The file is read as bytes and scored by [`sniff_bytes`][].

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a file of one
            linear transform.
        **kwargs : Any
            Options of the reader, passed to [`sniff_bytes`][].

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        content = _read_bytes(filename)
        if content is None:
            return super().sniff_filename(filename, error=error, **kwargs)
        return cls.sniff_bytes(content, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: XfmRaw, **kwargs) -> tx.Self:
        """Build an affine from an already parsed [`XfmRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : XfmRaw
            The record of the file, which holds one linear transform.
        **kwargs : Any
            Field values, such as `input`, `output` or `matrix`, which go
            to the constructor and override the record.

        Returns
        -------
        Self
            The affine.

        Raises
        ------
        ParserContentError
            If the record does not hold exactly one linear transform.
        """
        _linear(raw)
        return cls(metadata=XfmMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build an affine from the text of an MNI transform file.

        The text is read by [`XfmRaw.from_text`][].

        Parameters
        ----------
        text : str
            The text of the file.
        **kwargs : Any
            Field values, such as `input`, `output` or `matrix`, which go
            to the constructor and override the file.

        Returns
        -------
        Self
            The affine.

        Raises
        ------
        ParserContentError
            If libminc would not read the text, or if the file does not
            hold exactly one linear transform.
        """
        return cls.from_raw(XfmRaw.from_text(text), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build an affine from the lines of an MNI transform file.

        The lines are read by [`XfmRaw.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The affine.
        """
        return cls.from_raw(XfmRaw.from_lines(lines), **kwargs)

    @classmethod
    def from_bytes(
        cls, content: bytes, encoding: str = "utf-8", **kwargs
    ) -> tx.Self:
        """Build an affine from the bytes of an MNI transform file.

        The bytes are read by [`XfmRaw.from_bytes`][], which keeps the line
        endings and the bytes that do not decode.

        Parameters
        ----------
        content : bytes
            The content of the file.
        encoding : str, default="utf-8"
            The encoding of the text.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The affine.
        """
        return cls.from_raw(XfmRaw.from_bytes(content, encoding), **kwargs)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, encoding: str = "utf-8", **kwargs
    ) -> tx.Self:
        """Read an affine from an MNI transform file.

        The file is read as bytes by [`XfmRaw.from_filename`][].

        Parameters
        ----------
        filename : path-like
            The path of the file.
        encoding : str, default="utf-8"
            The encoding of the text.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The affine.
        """
        raw = XfmRaw.from_filename(filename, encoding=encoding)
        return cls.from_raw(raw, **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build an affine from a record, a file or a data model.

        An [`XfmRaw`][] is read with [`from_raw`][], and any other value as
        the bases read it.

        Parameters
        ----------
        other : object
            A record, a file, its content, a transformation, a mapping, or
            an instance of a similar class.
        *args : Any
            Constructor arguments. A record and a file take keyword
            arguments only.
        **kwargs : Any
            Options of the reader, and field values otherwise.

        Returns
        -------
        Self
            The new affine.
        """
        if isinstance(other, XfmRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        An affine that holds a record, and whose matrix, `input` and
        `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise, a new record is built, which holds
        one linear transform with `matrix`, and the comments of the
        current record. An affine without a record is encoded in the same
        way. The new record is held by new metadata, and the matrix is
        then decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a MINC
        transform file holds no field of the vocabulary yet, so a field
        that is set is reported as lost, once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this affine rather than a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        Self
            The consolidated copy, or this affine with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If the systems are not `RASmm`, or if the matrix is not that
            of a 3-D affine.
        """
        report = ConversionReport(source="XfmAffine", target="XfmAffine")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, XfmMetadata, fields, report
        )
        apply_loss_policy(report, on_loss, stacklevel=2)
        if not requested:
            metadata = self.metadata
        record = _record(self)
        overridden = (
            self._data is not None
            or self._input is not None
            or self._output is not None
        )
        if record is not None and not overridden:
            if not inplace:
                return replace(self, metadata=metadata)
            if requested:
                self.metadata = metadata
            return self
        record = _encode(self, record)
        if metadata is None:
            metadata = XfmMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self, metadata=metadata, data=None, input=None, output=None
            )
        self.metadata = metadata
        self.data = None
        self.input = None
        self.output = None
        return self

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the affine to a path or a file object.

        The affine is consolidated into a copy, whose record is written,
        so an affine that cannot be encoded leaves the file untouched, and
        the affine itself is not changed.

        Parameters
        ----------
        file : path or file object
            The path or the open file.
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][]. `on_loss` and the fields
            of the metadata are given to [`consolidate`][].

        Raises
        ------
        UnrepresentableTransformationError
            If the affine cannot be encoded.
        """
        record, options = _written(self, kwargs)
        record.to_file(file, **options)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the affine to a file, replacing a local file at once.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][]. `on_loss` and the fields
            of the metadata are given to [`consolidate`][].

        Raises
        ------
        UnrepresentableTransformationError
            If the affine cannot be encoded.
        """
        record, options = _written(self, kwargs)
        record.to_filename(filename, **options)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the affine to an open file, in text or binary mode.

        Parameters
        ----------
        file : IO
            The open file.
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][]. `on_loss` and the fields
            of the metadata are given to [`consolidate`][].

        Raises
        ------
        UnrepresentableTransformationError
            If the affine cannot be encoded.
        """
        record, options = _written(self, kwargs)
        record.to_fileobj(file, **options)

    def to_text(self, **kwargs) -> str:
        """Return the text of the MNI transform file that encodes the affine.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which refuses them.
            `on_loss` and the fields of the metadata are given to
            [`consolidate`][].

        Returns
        -------
        str
            The text of the file.

        Raises
        ------
        UnrepresentableTransformationError
            If the affine cannot be encoded.
        """
        record, options = _written(self, kwargs)
        return record.to_text(**options)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the MNI transform file that encodes the affine.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which refuses them.
            `on_loss` and the fields of the metadata are given to
            [`consolidate`][].

        Yields
        ------
        str
            The lines of the file, without line endings.

        Raises
        ------
        UnrepresentableTransformationError
            If the affine cannot be encoded.
        """
        record, options = _written(self, kwargs)
        return record.to_lines(**options)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the MNI transform file that encodes the affine.

        Parameters
        ----------
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][]. `on_loss` and the fields
            of the metadata are given to [`consolidate`][].

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        UnrepresentableTransformationError
            If the affine cannot be encoded.
        """
        record, options = _written(self, kwargs)
        return record.to_bytes(**options)


# ----------------------------------------------------------------------
#   ENCODING AN AFFINE AS A RECORD
# ----------------------------------------------------------------------


def _system_name(system: tx.Optional[_systems.CoordinateSystem]) -> str:
    """Return the name of a coordinate system for an error message."""
    name = getattr(system, "name", None)
    if name:
        return name
    if system is not None:
        return type(system).__name__
    return "an unspecified system"


def _encode(xform: XfmAffine, old: tx.Optional[XfmRaw]) -> XfmRaw:
    """Encode the matrix of an affine as a new record.

    See [`XfmAffine.consolidate`][].

    Raises
    ------
    UnrepresentableTransformationError
        If the systems are not `RASmm`, or if the matrix is not that of a
        3-D affine.
    """
    src, dst = xform.input, xform.output
    if not (
        isinstance(src, _systems.RASmm) and isinstance(dst, _systems.RASmm)
    ):
        raise UnrepresentableTransformationError(
            f"An MNI transform file maps RAS to RAS world coordinates, and "
            f"this affine maps {_system_name(src)} to {_system_name(dst)}. "
            f"Set its input and output to RASmm to say that it is one."
        )
    matrix = xform.matrix
    shape = None if matrix is None else tuple(np.shape(matrix))
    if shape is not None and shape != (3, 4):
        raise UnrepresentableTransformationError(
            f"An MNI transform file stores a 3-D affine, of shape (3, 4), "
            f"but the matrix has shape {shape}."
        )
    linear = XfmRaw.Linear(matrix=_matrix_to_disk(matrix))
    comments = () if old is None else old.comments
    return XfmRaw(comments=comments, transforms=(linear,))


def _written(
    xform: XfmAffine, kwargs: tx.Mapping[str, tx.Any]
) -> tx.Tuple[XfmRaw, tx.Dict[str, tx.Any]]:
    """Return the record that a file written from an affine holds.

    The record is that of the copy consolidated with `on_loss` and the
    fields of the metadata among `kwargs`, and the other options, for
    the writer of the record, are returned with it.

    Raises
    ------
    UnrepresentableTransformationError
        If the affine cannot be encoded.
    """
    metadata, options = metadata_options(kwargs)
    return _record(xform.consolidate(**metadata)), options
