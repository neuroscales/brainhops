import enum
import warnings
from collections.abc import Mapping

import typing_extensions as tx

from brainhops._core.deprecation import external_stacklevel

if tx.TYPE_CHECKING:
    # imported for the cross-references of mkdocstrings
    from .datamodel._transformations.base import Transformation  # noqa: F401

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
        *args: tx.Any,
        result: tx.Any = None,
        **kwargs: tx.Any,
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
`{LossyConversionError: on_loss, **on_error}` (see [`error_policy`][]).
Because that class is more specific than the scope of a single value
of `on_error=`, `on_loss="raise", on_error="ignore"` raises on a loss
and ignores the other conversion errors. Without `on_loss=`, a loss
falls under `on_error=`, so `on_error="ignore"` alone also returns the
lossy result of a conversion that loses information.

The strings `"raise"`, `"warn"` and `"ignore"` are reserved for the
actions, so a policy cannot return one of them as a value.
"""


def error_policy(
    on_error: ErrorPolicy = UNSET,
    *,
    on_loss: ErrorAction = UNSET,
    scope: tx.Tuple[tx.Type[BaseException], ...] = (ConversionError,),
) -> tx.Dict[tx.Type[BaseException], ErrorAction]:
    """Return the policy that a conversion applies, as a mapping.

    A conversion takes a policy as its `on_error=` keyword, and the
    action for a loss as its `on_loss=` keyword. This function merges
    the two into one mapping from exception classes to actions, which
    [`policy_action`][] and [`apply_error_policy`][] read. A single
    action applies to each class of `scope` (see [`ErrorPolicy`][]).
    The entry that `on_loss` sets comes first, so that an entry for
    [`LossyConversionError`][] in `on_error` replaces it. Because that
    class is more specific than [`ConversionError`][],
    `on_loss="ignore"` with `on_error="raise"` accepts a loss and
    raises any other error.

    Parameters
    ----------
    on_error : ErrorPolicy, optional
        One action, which applies to every class of `scope`, or a
        mapping from exception classes to actions, which is taken as
        written. By default, no class has an entry, so every error is
        raised.
    on_loss : ErrorAction, optional
        The action for a [`LossyConversionError`][], with the same
        values as a single action of `on_error`. By default, it adds no
        entry, and a loss falls under `on_error`. The value `None` is
        an action, which returns `None` in place of a lossy result.
    scope : tuple of type, default=(ConversionError,)
        The exception classes that the conversion raises on purpose,
        to which a single action applies.

    Returns
    -------
    dict
        The action of each exception class that has an entry.

    Raises
    ------
    TypeError
        If `on_error` or `on_loss` is `True` or `False`, if a key of
        `on_error` is not an exception class, or if `on_loss` is a
        mapping.
    ValueError
        If an action is a string other than `"raise"`, `"warn"` and
        `"ignore"`.

    Examples
    --------
    ```python
    error_policy("raise", on_loss="ignore")
    # {LossyConversionError: 'ignore', ConversionError: 'raise'}
    error_policy({Exception: None})
    # {Exception: None}
    ```
    """
    for name, value in (("on_error", on_error), ("on_loss", on_loss)):
        if isinstance(value, bool):
            raise TypeError(
                f"{name}= takes an action or a mapping, not {value!r}. "
                f"Pass {name}='raise' to raise, {name}='ignore' to return "
                f"the result, or a value to return in its place, such as "
                f"{name}=None."
            )
    if on_error is UNSET:
        entries = {}
    elif isinstance(on_error, Mapping):
        entries = {}
        for cls, action in on_error.items():
            if not (isinstance(cls, type) and issubclass(cls, BaseException)):
                raise TypeError(
                    f"The keys of an error policy are exception classes, "
                    f"not {cls!r}."
                )
            entries[cls] = _checked_action(action)
    else:
        action = _checked_action(on_error)
        entries = {cls: action for cls in scope}
    if on_loss is UNSET:
        return entries
    if isinstance(on_loss, Mapping):
        raise TypeError(
            "on_loss= takes one action, which applies to a loss, not a "
            "mapping. Pass a mapping as on_error= instead."
        )
    return {LossyConversionError: _checked_action(on_loss), **entries}


def policy_action(
    policy: tx.Mapping[tx.Type[BaseException], ErrorAction],
    cls: tx.Type[BaseException],
) -> ErrorAction:
    """Return the action that a policy takes for a class of errors.

    The classes of `cls` are walked in method resolution order, and the
    action of the first one that has an entry in `policy` is returned.

    Parameters
    ----------
    policy : Mapping[type, ErrorAction]
        A policy as a mapping, as [`error_policy`][] returns it.
    cls : type of BaseException
        The class of the error.

    Returns
    -------
    ErrorAction
        The action, or [`UNSET`][] when no class of `cls` has an entry,
        in which case the error is raised.
    """
    for base in cls.__mro__:
        if base in policy:
            return policy[base]
    return UNSET


def apply_error_policy(
    error: BaseException,
    on_error: ErrorPolicy = UNSET,
    *,
    on_loss: ErrorAction = UNSET,
    scope: tx.Tuple[tx.Type[BaseException], ...] = (ConversionError,),
    stacklevel: int = 2,
) -> tx.Any:
    """Act on an error that a conversion raised, according to a policy.

    A conversion calls this function when it catches an error. The
    action is chosen by the class of the error (see [`ErrorPolicy`][]),
    and its effect is listed in [`ErrorAction`][]. An action that does
    not raise returns the value that the conversion returns in place of
    its result.

    Parameters
    ----------
    error : BaseException
        The error that the conversion raised.
    on_error : ErrorPolicy, optional
        The policy. By default, the error is raised again.
    on_loss : ErrorAction, optional
        The action for a [`LossyConversionError`][], merged with
        `on_error` as [`error_policy`][] merges it.
    scope : tuple of type, default=(ConversionError,)
        The exception classes that the conversion raises on purpose,
        to which a single action applies.
    stacklevel : int, default=2
        The lowest stack level of the warning, counted from the caller
        of this function. The warning points at the first frame at or
        above it that is outside brainhops, which is the code that
        called the conversion.

    Returns
    -------
    Any
        The result that the error carries, `None`, or the value that the
        policy returns.

    Raises
    ------
    BaseException
        The error itself, or the exception that the policy names.
    """
    policy = error_policy(on_error, on_loss=on_loss, scope=scope)
    action = policy_action(policy, type(error))
    if action is UNSET or _is_action(action, "raise"):
        raise error
    result = error.result if isinstance(error, LossyConversionError) else None
    if _is_action(action, "warn"):
        warnings.warn(
            _as_warning(error), stacklevel=external_stacklevel(stacklevel + 1)
        )
        return result
    if _is_action(action, "ignore"):
        return result
    if isinstance(action, BaseException) or (
        isinstance(action, type) and issubclass(action, BaseException)
    ):
        raise action from error
    if _is_report(action):
        action.record(error)
        return result
    return action


# --- error policy, private ---------------------------------------------

_ACTIONS = ("raise", "warn", "ignore")


def _is_action(action: tx.Any, name: str) -> bool:
    # `action == name` would compare an array element by element.
    return isinstance(action, str) and action == name


def _checked_action(action: tx.Any) -> tx.Any:
    if isinstance(action, str) and action not in _ACTIONS:
        raise ValueError(
            f"A string in an error policy is one of the actions "
            f"{_ACTIONS}, not {action!r}."
        )
    return action


def _is_report(action: tx.Any) -> bool:
    # The report is a data model, which imports this module.
    from brainhops.datamodel.metadata._report import ConversionReport

    return isinstance(action, ConversionReport)


def _as_warning(error: BaseException) -> Warning:
    # A loss of metadata is reported by a warning that carries the report.
    from brainhops.datamodel.metadata._report import (
        MetadataLossError,
        MetadataLossWarning,
    )

    if isinstance(error, MetadataLossError):
        return MetadataLossWarning(error.report)
    text = str(error) or "The conversion loses information."
    if not isinstance(error, ConversionError):
        text = f"{type(error).__name__}: {text}"
    return ConversionWarning(text)
