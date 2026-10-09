import typing_extensions as tx

if tx.TYPE_CHECKING:
    from .datamodel._transformations.base import Transformation


# --- transformations --------------------------------------------------


class AdaptationError(TypeError):
    """A coordinate system cannot be adapted to another.

    Adaptation reorders, rescales and flips the axes that two coordinate
    systems share. It fails when an axis that must be matched has no
    counterpart in the other system, or when two matched axes have
    incompatible units. The error message names both systems and the axes
    that could not be reconciled.
    """


class CompositionError(TypeError):
    """Two transformations cannot be composed."""


class DomainError(ValueError):
    """A transformation lies outside the domain of an operator.

    The square and the square root of a transformation (see
    [`Transformation.square`][] and [`Transformation.sqrt`][]) are defined
    only for a transformation that maps a space to itself. The principal
    square root, and the principal logarithm that `.to(log=True)` computes,
    also require the linear part to have no eigenvalue on the closed negative
    real axis, a condition that excludes singular matrices, reflections and
    rotations by a half turn. Rather than return a complex, non-principal or
    approximate result, the operation raises this error.

    When the operator is defined for a transformation but brainhops does not
    know how to compute it, the operation raises `NotImplementedError`
    instead.
    """


class RestrictionError(TypeError):
    """A transformation cannot be restricted to a block of axes.

    Restricting a transformation extracts the part of it that maps a subset
    of the input axes to a subset of the output axes. The restriction fails
    when no restriction rule exists for the type of the transformation, or
    when the block cannot be cut out soundly, for example in a chain where
    the block is coupled to other axes. When the factoring pass, which splits
    a chain into groups of axes that transform independently, meets this
    error, it leaves the chain unfactored rather than drop a piece.
    """


class ConversionError(TypeError):
    """A transformation cannot be converted into another type."""


class LossyConversionError(ConversionError):
    """A conversion would discard information.

    [`Transformation.to`][] raises this error when a conversion is lossy and
    the loss was not allowed. The `result` attribute holds the transformation
    that the conversion would have produced, or `None`.
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
    """A list of axes cannot be read as a vector field.

    The axes of a vector field must include exactly one vector axis, whose
    type is `displacement` or `coordinate`. The other axes are grid axes.
    """


# ---- sniff -----------------------------------------------------------


class SnifferError(Exception):
    """The base class of sniffer errors."""

    pass


class SnifferTypeError(SnifferError, TypeError):
    """A sniffer received an input of an unexpected type."""

    pass


class SnifferExistsError(SnifferError, FileNotFoundError):
    """A sniffer received an input that does not exist."""

    pass


class SnifferContentError(SnifferError, TypeError):
    """A sniffer received an input with unexpected content."""

    pass


class SnifferNotImplementedError(SnifferError, NotImplementedError):
    """A sniffer function is not implemented."""

    pass


# ---- from ------------------------------------------------------------


class ParserError(Exception):
    """The base class of parser errors."""

    pass


class ParserTypeError(ParserError, TypeError):
    """A parser received an input of an unexpected type."""

    pass


class ParserExistsError(ParserError, FileNotFoundError):
    """A parser received an input that does not exist."""

    pass


class ParserContentError(ParserError, TypeError):
    """A parser received an input with unexpected content."""

    pass


class ParserNotImplementedError(ParserError, NotImplementedError):
    """A parser function is not implemented."""

    pass


class AmbiguousFormatError(ParserError):
    """Several parsers claim the same content equally well.

    The parsers tie on every criterion (confidence, extension, constraints and
    priority) yet build different objects, so any choice could silently return
    the wrong kind of object. The message names each candidate format and the
    `hint=` value that selects it. The fix belongs in the parsers: give one a
    distinguishing sniffer (a magic number, intent code or filename
    constraint), or set an explicit `PRIORITY`.
    """

    pass


# ---- to ------------------------------------------------------------


class WriterError(ParserError):
    """The base class of writer errors, which is a kind of parser error."""

    pass


class WriterNotImplementedError(WriterError, NotImplementedError):
    """A writer function is not implemented."""

    pass


class UnrepresentableTransformationError(WriterError):
    """A transformation cannot be encoded in the target format.

    Affine-only formats such as NIfTI cannot store arbitrary transformations,
    and the writer raises this error rather than resample the data or drop the
    transformation.
    """

    pass
