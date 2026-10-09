import typing_extensions as tx

if tx.TYPE_CHECKING:
    from .datamodel._transformations.base import Transformation


class AdaptationError(TypeError):
    """A coordinate system cannot be adapted to another.

    Adaptation reorders, rescales and flips shared axes. It fails when a
    required axis has no match in the other system, or when matched axes have
    incompatible units. The message names both systems and the axes concerned.
    """


class CompositionError(TypeError):
    """Two transformations cannot be composed."""


class DomainError(ValueError):
    """A transformation lies outside the domain of an operator.

    [`Transformation.square`][] and [`Transformation.sqrt`][] require a
    transformation from a space to itself. The principal square root and
    logarithm (as in `.to(log=True)`) also require a linear part without
    eigenvalues on the closed negative real axis, which excludes singular
    matrices, reflections and half-turns. The operation raises this error
    rather than return a complex, non-principal or approximate result. An
    operator that brainhops cannot compute raises NotImplementedError instead.
    """


class RestrictionError(TypeError):
    """A transformation cannot be restricted to a block of axes.

    Either no restriction rule exists for its type, or the block cannot be cut
    out soundly, as in a chain where the block is coupled to other axes.
    Factoring leaves such a chain unfactored rather than drop a piece.
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


class AxisError(ValueError):
    """A list of axes cannot be read as a vector field.

    A vector field has grid axes and exactly one vector axis, of type
    `displacement` or `coordinate`.
    """


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


class WriterError(ParserError):
    """The base class of writer errors, itself a parser error."""

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
