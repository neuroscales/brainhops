__all__ = [
    "AdaptationError",
    "AmbiguousFormatError",
    "AxisError",
    "CompositionError",
    "ConversionError",
    "ConversionWarning",
    "DomainError",
    "ErrorAction",
    "ErrorPolicy",
    "IncompatibleSystemError",
    "LossyConversionError",
    "ParserContentError",
    "ParserError",
    "ParserExistsError",
    "ParserNotImplementedError",
    "ParserTypeError",
    "RestrictionError",
    "SnifferContentError",
    "SnifferError",
    "SnifferExistsError",
    "SnifferNotImplementedError",
    "SnifferTypeError",
    "UNSET",
    "UnrepresentableTransformationError",
    "WriterError",
    "WriterNotImplementedError",
]

import enum

import typing_extensions as tx

if tx.TYPE_CHECKING:
    # imported for the annotations, and for the cross-references of
    # mkdocstrings
    from .datamodel._transformations.base import Transformation  # noqa: F401
    from .datamodel.base import DataModelBase

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

    [`Transformation.to`][] raises this error when a conversion is lossy
    and the loss was not allowed. The `result` attribute holds the
    object that the conversion would have produced, or `None`. An error
    policy that accepts the loss, such as `on_loss="ignore"`, returns
    that object instead of raising (see [`ErrorPolicy`][]). A loss of
    metadata is reported by a subclass,
    [`MetadataLossError`][brainhops.datamodel.metadata.MetadataLossError].
    """

    def __init__(
        self,
        *args,
        result: tx.Optional["DataModelBase"] = None,
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


# --- error policy ------------------------------------------------------


class ConversionWarning(UserWarning):
    """The warning that an error policy issues for its `"warn"` action.

    A conversion that is given `on_error="warn"` turns the error into
    this warning and returns its result (see [`ErrorPolicy`][]). A loss
    of metadata is reported by a subclass,
    [`MetadataLossWarning`][brainhops.datamodel.metadata.MetadataLossWarning],
    which also carries the report of what was lost.
    """


class _Unset(enum.Enum):
    """The type of [`UNSET`][], which has one member.

    | Member  | Value     |
    | ------- | --------- |
    | `UNSET` | `"UNSET"` |
    """

    UNSET = "UNSET"

    def __repr__(self) -> str:
        return self.value


UNSET: tx.Final = _Unset.UNSET
"""
The default of `on_error=`, which means that no policy was given.

A conversion called without a policy raises every error. A loss of
metadata then follows the policy in effect, which
[`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy]
sets. This value differs from `None`, which is a policy that returns
`None` in place of the result.
"""

ErrorAction: tx.TypeAlias = tx.Any
"""
What an error policy does with one error.

| Value                  | Effect                                      |
| ---------------------- | ------------------------------------------- |
| `"raise"`              | Raise the error again.                      |
| `"warn"`               | Warn, then return the result.               |
| `"ignore"`             | Return the result without warning.          |
| a `ConversionReport`   | Record the error, then return the result.   |
| an exception class     | Raise it, from the original error.          |
| an exception instance  | Raise it, from the original error.          |
| any other value        | Return the value.                           |

The result is the lossy result that a [`LossyConversionError`][]
carries in its `result` attribute, and `None` for any other error. A
string is always read as an action, so a policy cannot return the
string `"raise"`, `"warn"` or `"ignore"`, and any other string is
refused. The report is a
[`ConversionReport`][brainhops.datamodel.metadata.ConversionReport].
"""

ErrorPolicy: tx.TypeAlias = tx.Union[
    ErrorAction,
    tx.Mapping[tx.Type[BaseException], ErrorAction],
]
"""
What a conversion does when it fails or loses information.

A policy is the value that the `on_error=` keyword of a conversion
takes. It is either one action (see [`ErrorAction`][]) or a mapping
from exception classes to actions, such as
`{LossyConversionError: "warn", ConversionError: None}`.

Each call that takes a policy declares its scope, which is the tuple
of the exception classes that the call raises on purpose when it
cannot convert. The scope of a conversion is `(ConversionError,)`,
which covers a [`LossyConversionError`][] and a loss of metadata. A
single action `v` is read as the mapping `{cls: v for cls in scope}`,
so that `on_error=None` returns `None` when the conversion fails, and
an error outside the scope, such as an `OSError` or a bug, is raised
unchanged. A mapping is taken as written, and may name a class outside
the scope, as `{Exception: None}` does to catch every error. The value
`True` or `False` is refused as a single action, because it reads as a
yes or a no rather than as an action.

When a conversion raises an error, the policy walks the classes of the
error in method resolution order and applies the action of the first
class that has an entry. An error whose classes have no entry is
raised. For example, a [`LossyConversionError`][] matches its own entry
before that of [`ConversionError`][], which it subclasses.

The `on_loss=` keyword of a conversion is the half of the policy that
concerns losses. It takes the same values as `on_error=`, except a
mapping, and sets the entry of [`LossyConversionError`][], which also
covers a loss of metadata, so that the policy that a call applies is
`{LossyConversionError: on_loss, **on_error}`.
Because that class is more specific than the scope of a single value
of `on_error=`, `on_loss="raise", on_error="ignore"` raises on a loss
and ignores the other conversion errors. Without `on_loss=`, a loss
falls under `on_error=`, so `on_error="ignore"` alone also returns the
lossy result of a conversion that loses information.

The strings `"raise"`, `"warn"` and `"ignore"` are reserved for the
actions, so a policy cannot return one of them as a value.
"""
