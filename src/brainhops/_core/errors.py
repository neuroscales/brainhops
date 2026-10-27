"""The functions that read an error policy and act on one error.

A conversion takes an error policy as its `on_error=` keyword and the
action for a loss as its `on_loss=` keyword (see
[`ErrorPolicy`][brainhops.errors.ErrorPolicy]). The functions of this
module merge the two keywords into one mapping, find the action that
the mapping takes for a class of errors, and carry that action out on
an error that a conversion raised. The types of the policy and the
exception classes stay public in `brainhops.errors`, and these
functions are internal: a conversion applies a policy by calling
[`call_under_policy`][brainhops.datamodel.metadata._report.call_under_policy],
which uses them.
"""

__all__ = ["apply_error_policy", "error_policy", "policy_action"]

import warnings
from collections.abc import Mapping

import typing_extensions as tx

from brainhops._core.deprecation import external_stacklevel
from brainhops.errors import (
    UNSET,
    ConversionError,
    ConversionWarning,
    ErrorAction,
    ErrorPolicy,
    LossyConversionError,
)


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
    action applies to each class of `scope` (see
    [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). The entry that
    `on_loss` sets comes first, so that an entry for
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
    action is chosen by the class of the error (see
    [`ErrorPolicy`][brainhops.errors.ErrorPolicy]), and its effect is
    listed in [`ErrorAction`][brainhops.errors.ErrorAction]. An action
    that does not raise returns the value that the conversion returns in
    place of its result.

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


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------

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
