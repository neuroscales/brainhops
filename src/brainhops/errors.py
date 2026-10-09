# dependencies
import typing_extensions as tx

# typing
if tx.TYPE_CHECKING:
    from .datamodel._transformations.base import Transformation


# --- transformations --------------------------------------------------


class AdaptationError(TypeError):
    """Raised when one coordinate system cannot be adapted to another.

    Adaptation reorders, rescales, and flips the axes that two coordinate
    systems share, so it succeeds only when every axis of one system
    corresponds to an axis of the other. This error is raised when an axis
    that must be matched has no correspondence, or when two matched axes
    carry incompatible units. Its message names the two systems and the
    axes that could not be reconciled.
    """


class CompositionError(TypeError):
    """Raised when two transformations cannot be composed."""


class DomainError(ValueError):
    """Raised when a transformation lies outside the domain of an operator.

    The square and the square root of a transformation (see
    [`Transformation.square`][] and [`Transformation.sqrt`][]) are defined
    only for a transformation that maps a space to itself. The principal
    square root, and the principal logarithm that `.to(log=True)` takes of
    a matrix, also need the linear part to have no eigenvalue on the
    closed negative real axis: a singular matrix, a reflection and a
    rotation by a half turn have neither. Rather than return a complex, a
    non-principal or an approximate result, the operation raises this
    error, whose message names it and the reason.

    A transformation the operator is defined for, but that brainhops does
    not know how to compute it for, raises `NotImplementedError` instead.
    """


class RestrictionError(TypeError):
    """Raised when a transformation cannot be restricted to a block of axes.

    Either no restriction rule applies to its type, or the block cannot be
    cut out of it soundly (a chain inside which the block is coupled to the
    other axes, or a transform with no affine reading). The factor pass
    leaves a chain unfactored when this is raised, rather than drop a
    piece it cannot restrict.
    """


class ConversionError(TypeError):
    """Raised when a transformation cannot be converted to another type."""


class LossyConversionError(ConversionError):
    """Raised when a conversion would discard information.

    This error is raised by [`Transformation.to`][] when a conversion is
    only possible at the cost of losing information, and the conversion
    was not explicitly allowed to be lossy. The `result` attribute holds
    the transformation that the conversion would have produced.
    """

    def __init__(
        self,
        *args,
        result: tx.Optional["Transformation"] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.result = result


# --- IO helpers --------------------------------------------------------


class AxisError(ValueError):
    """Raised when a list of axes cannot be read as a vector field.

    A vector field names its grid axes together with exactly one vector
    axis, of type `displacement` or `coordinate`. A list that names no
    vector axis, more than one, or one of each type, is refused with this
    error.
    """


# ---- sniff -----------------------------------------------------------


class SnifferError(Exception):
    """Base class for sniffer errors."""

    pass


class SnifferTypeError(SnifferError, TypeError):
    """Raised when a sniffer encounters an unexpected type."""

    pass


class SnifferExistsError(SnifferError, FileNotFoundError):
    """Raised when a sniffer encounters an unexpected content."""

    pass


class SnifferContentError(SnifferError, TypeError):
    """Raised when a sniffer encounters an unexpected content."""

    pass


class SnifferNotImplementedError(SnifferError, NotImplementedError):
    """Raised when a sniffer function is not implemented."""

    pass


# ---- from ------------------------------------------------------------


class ParserError(Exception):
    """Base class for parser errors."""

    pass


class ParserTypeError(ParserError, TypeError):
    """Raised when a parser encounters an unexpected type."""

    pass


class ParserExistsError(ParserError, FileNotFoundError):
    """Raised when a parser encounters an unexpected content."""

    pass


class ParserContentError(ParserError, TypeError):
    """Raised when a parser encounters an unexpected content."""

    pass


class ParserNotImplementedError(ParserError, NotImplementedError):
    """Raised when a parser function is not implemented."""

    pass


class AmbiguousFormatError(ParserError):
    """
    Raised when several parsers claim the same content, equally well.

    Reaching this means two parsers agree on the confidence score, the
    extension they matched, the constraints they declare and their
    explicit priority, yet build different objects. There is no
    meaningful way to choose between them, and picking one at random
    would silently return the wrong kind of object.

    The message is written for the user who hit it: it names each
    candidate format, and the `hint=` value (or the format's own `load`)
    that reads the content as that format.

    If the formats should be able to tell such content apart, the fix
    belongs in the parsers, not in the caller: give one of them a
    sniffer that can tell the two apart (a magic number, an intent code,
    a filename constraint), or set an explicit `PRIORITY`.
    """

    pass


# ---- to ------------------------------------------------------------


class WriterError(ParserError):
    """Base class for writer errors."""

    pass


class WriterNotImplementedError(WriterError, NotImplementedError):
    """Raised when a writer function is not implemented."""

    pass


class UnrepresentableTransformationError(WriterError):
    """
    Raised when a transformation cannot be encoded in the target format.

    A format that stores only affine geometry, such as NIfTI, cannot hold
    an arbitrary transformation. When the object being written carries one
    that the format has no way to represent, the writer raises this error
    rather than resampling the data or discarding the transformation. The
    message names the transformation that could not be written.
    """

    pass
