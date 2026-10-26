import warnings

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
    """An object cannot be converted into another class.

    A transformation raises it when no converter turns it into the
    requested class. Any other object of the data model raises it when
    it is copied into a class that is not related to its own, because
    fields with the same name may then mean different things.
    """


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


class IncompatibleSystemError(ConversionError):
    """Error raised when a class refuses the endpoint it is given.

    Some classes are bound to coordinate systems. For example,
    `VoxelToRAS` maps voxel coordinates to RAS millimetres. Such a class
    accepts only endpoints that are compatible with its systems, and
    raises this error for any other endpoint. Building
    `VoxelToRAS(input=LPSmm())` raises it, and so does
    `t.to(VoxelToRAS)` when `t` maps between LPS spaces.

    The error is a [`ConversionError`][] because `t.to(VoxelToRAS)`
    rebuilds `t` as the bound class, and that conversion cannot be made
    without changing what `t` means.
    """


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


OnError: tx.TypeAlias = tx.Union[
    tx.Literal["raise", "warn", "ignore"],
    tx.Type[Exception],
    tx.Type[Warning],
    Exception,
    Warning,
    None,
    tx.Any,
]

_ExceptionOrWarning: tx.TypeAlias = tx.Union[
    tx.Type[Exception],
    tx.Type[Warning],
]
_DefaultError: tx.TypeAlias = tx.Union[
    _ExceptionOrWarning,
    tx.Sequence[_ExceptionOrWarning],
]


def raise_or_warn(
    msg: str,
    on_error: OnError = "raise",
    default: _DefaultError = (RuntimeError, UserWarning),
) -> tx.Any:
    """
    Raise an error or warn based on the specified error handling strategy.

    Parameters
    ----------
    msg : str
        The error or warning message.
    on_error : OnError, optional
        The error handling strategy. It can be one of the following:
        - "raise": Raise an exception (default).
        - "warn": Issue a warning.
        - "ignore": Do nothing.
        - Exception class or instance: Raise the specified exception.
        - Warning class or instance: Issue the specified warning.
        - Anything else: return as is
    default : _DefaultError, optional
        The default exception or warning to use if `on_error` is not specified.

    Returns
    -------
    Any
        If `on_error` is not an exception or warning, return the value
        of `on_error`.

    Raises
    ------
    Exception
        If `on_error` is "raise" or an exception class/instance, raise
        the specified exception with the provided message.
    """
    if on_error == "ignore":
        return
    if on_error == "raise":
        on_error = _get_default_exception(default)
    if on_error == "warn":
        on_error = _get_default_warning(default)
    if isinstance(on_error, type):
        if issubclass(on_error, Warning):
            return warnings.warn(msg, on_error, stacklevel=2)
        elif issubclass(on_error, Exception):
            on_error = on_error(msg)
    if isinstance(on_error, Exception):
        raise on_error
    return on_error


def _get_default_exception(default: _DefaultError) -> tx.Type[Exception]:
    if isinstance(default, type):
        return default
    if isinstance(default, (list, tuple)):
        for cls in default:
            if issubclass(cls, Exception):
                return cls
    return RuntimeError


def _get_default_warning(default: _DefaultError) -> tx.Type[Warning]:
    if isinstance(default, type):
        return default
    if isinstance(default, (list, tuple)):
        for cls in default:
            if issubclass(cls, Warning):
                return cls
    return UserWarning
