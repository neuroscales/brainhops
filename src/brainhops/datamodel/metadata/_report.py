"""Loss reports, and the loss policy."""

__all__ = [
    "ConversionReport",
    "LossPolicy",
    "MetadataLossError",
    "MetadataLossWarning",
    "OnLoss",
    "apply_loss_policy",
    "call_under_policy",
    "metadata_loss_policy",
    "takes_error_policy",
]

# stdlib
import contextlib
import contextvars
import enum
import functools
import inspect
import warnings

# externals
import typing_extensions as tx
from bagof.magic import Factory

# internals
from brainhops._core.deprecation import pop_renamed_keyword
from brainhops.errors import (
    UNSET,
    ConversionWarning,
    ErrorAction,
    ErrorPolicy,
    LossyConversionError,
    apply_error_policy,
    error_policy,
    policy_action,
)

from ..base import DataModelBase


class ConversionReport(DataModelBase):
    """
    The record of what a conversion or a consolidation could not carry
    over.

    Consolidating the metadata of a file format, which every write of
    that format does, fills a report with what the format could not
    store: a field whose value was dropped is listed in `lost`, with the
    value, and a field whose value was stored, but not exactly, is
    listed in `approximated`, with a short description of the change
    (such as `"truncated to 80 bytes"`). A conversion into the metadata
    of a format fills a report with the fields that the format declares
    no slot for, which is a lower bound of what a write loses.

    A report is also an action of an error policy (see
    [`ErrorPolicy`][brainhops.errors.ErrorPolicy]), so that it can
    record what any conversion loses. A conversion given
    `lossy=report`, or a policy that maps an exception class to the
    report, records each error that the policy catches with
    [`record`][ConversionReport.record] and returns its result: a loss
    of metadata adds its fields to `lost` and `approximated`, and any
    other error, such as the
    [`LossyConversionError`][brainhops.errors.LossyConversionError] of a
    transformation, is appended to `errors`.

    The loss policy decides what happens to a report that is not empty
    (see [`metadata_loss_policy`][]).
    """

    source: tx.Optional[str] = None
    """The format converted from."""

    target: tx.Optional[str] = None
    """The format converted to."""

    lost: tx.Dict[str, tx.Any] = Factory()
    """The value that was dropped, for each field name."""

    approximated: tx.Dict[str, str] = Factory()
    """What changed in the stored value, for each field name."""

    errors: tx.List[Exception] = Factory()
    """The errors recorded by an error policy, other than losses of
    metadata."""

    @property
    def lossy(self) -> bool:
        """
        Whether the report lists a field as lost or approximated, or
        records a lossy conversion.
        """
        return bool(
            self.lost
            or self.approximated
            or any(isinstance(e, LossyConversionError) for e in self.errors)
        )

    def merge(self, other: tx.Self) -> tx.Self:
        """
        Add the entries of another report to this report, in place.

        Parameters
        ----------
        other : ConversionReport
            The report whose entries are added. An entry of `other`
            replaces the entry of this report for the same field, and
            the errors of `other` are appended to those of this report.

        Returns
        -------
        ConversionReport
            This report.
        """
        self.lost.update(other.lost)
        self.approximated.update(other.approximated)
        self.errors.extend(e for e in other.errors if e not in self.errors)
        return self

    def record(self, error: Exception) -> tx.Self:
        """
        Add an error that an error policy caught to this report.

        A [`MetadataLossError`][] adds the entries of its own report, as
        [`merge`][ConversionReport.merge] adds them, and gives this
        report its `source` and `target` when it has none yet. Any other
        error is appended to `errors`.

        Parameters
        ----------
        error : Exception
            The error to record.

        Returns
        -------
        ConversionReport
            This report.
        """
        if isinstance(error, MetadataLossError):
            if error.report is not self:
                self.source = self.source or error.report.source
                self.target = self.target or error.report.target
                self.merge(error.report)
        else:
            self.errors.append(error)
        return self

    def raise_if_lossy(self) -> None:
        """
        Raise an error when anything was lost or approximated.

        Raises
        ------
        MetadataLossError
            If the report lists a field as lost or approximated.
        """
        if self.lossy:
            raise MetadataLossError(self)

    def __str__(self) -> str:
        where = f"{self.source or '?'} -> {self.target or '?'}"
        kind = "Conversion" if self.errors else "Metadata conversion"
        if not self.lossy and not self.errors:
            return f"{kind} {where}: nothing lost."
        parts = []
        if self.lost:
            items = ", ".join(f"{k}={short(v)}" for k, v in self.lost.items())
            parts.append(f"lost {items}")
        if self.approximated:
            items = ", ".join(
                f"{k} ({v})" for k, v in self.approximated.items()
            )
            parts.append(f"approximated {items}")
        if self.errors:
            items = ", ".join(
                f"{type(e).__name__}({str(e)!r})"
                if str(e)
                else type(e).__name__
                for e in self.errors
            )
            parts.append(f"recorded {items}")
        return f"{kind} {where}: " + "; ".join(parts) + "."


class MetadataLossWarning(ConversionWarning):
    """
    The warning issued when some metadata could not be carried over.

    The `report` attribute holds the [`ConversionReport`][] that says what
    was lost or approximated.
    """

    def __init__(self, report: ConversionReport) -> None:
        """
        Parameters
        ----------
        report : ConversionReport
            What was lost or approximated.
        """
        super().__init__(str(report))
        self.report = report


class MetadataLossError(LossyConversionError):
    """
    The error raised, under the `"raise"` policy, when some metadata could
    not be carried over.

    The `report` attribute holds the [`ConversionReport`][] that says what
    was lost or approximated. The error is a
    [`LossyConversionError`][brainhops.errors.LossyConversionError], so
    that `lossy=` and `on_error=` treat a loss of metadata as they treat
    any other loss (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]).
    It is therefore also a `TypeError`. The field converters of the data
    model turn a `TypeError` into a conversion error of their own, but
    they let a
    [`LossyConversionError`][brainhops.errors.LossyConversionError]
    through unchanged, so that a refused loss surfaces as itself.

    The metadata methods raise the error before they build their result,
    so its `result` attribute is usually `None`.
    """

    def __init__(
        self, report: ConversionReport, *, result: tx.Any = None
    ) -> None:
        """
        Parameters
        ----------
        report : ConversionReport
            What was lost or approximated.
        result : Any, optional
            The object that the conversion would have returned, when it
            is known.
        """
        super().__init__(str(report), result=result)
        self.report = report


LossPolicy: tx.TypeAlias = tx.Literal["ignore", "warn", "raise"]
"""
A loss policy: ignore a loss, warn about it, or raise. Each of the three
values is also an action of an error policy (see
[`ErrorPolicy`][brainhops.errors.ErrorPolicy]).
"""

OnLoss: tx.TypeAlias = tx.Union[LossPolicy, ConversionReport]
"""
What the deprecated `on_loss=` argument takes: a [`LossPolicy`][], or a
[`ConversionReport`][] to fill with what is lost, instead of warning or
raising. The `on_error=` argument, which replaces it, takes an
[`ErrorPolicy`][brainhops.errors.ErrorPolicy].
"""


@contextlib.contextmanager
def metadata_loss_policy(policy: OnLoss) -> tx.Iterator[None]:
    """
    Set the loss policy of the conversions and writes made in a block.

    The policy in effect governs every conversion and every
    consolidation that takes no `on_error=` argument, or one that has no
    entry for a loss of metadata, in particular the implicit conversions
    that field converters trigger. Outside of any block, the policy is
    `"warn"`. A report given as the policy collects what the block
    loses, and nothing is warned or raised, so that a caller such as
    `io.save`, which converts an object and then writes it, acts once on
    one report.

    Parameters
    ----------
    policy : {"ignore", "warn", "raise"} or ConversionReport
        The policy: ignore a loss, warn about it, or raise
        [`MetadataLossError`][], or a report to fill.

    Yields
    ------
    None

    Raises
    ------
    ValueError
        If `policy` is not one of the three policies.

    Examples
    --------
    ```python
    with metadata_loss_policy("raise"):
        # Raises if MGH has no slot for a field that is set.
        mgh = metadata.to(MghMetadata)
    ```
    """
    if not isinstance(policy, ConversionReport):
        policy = _check_policy(policy)
    token = _POLICY.set(policy)
    try:
        yield
    finally:
        _POLICY.reset(token)


def apply_loss_policy(
    report: ConversionReport,
    on_error: ErrorPolicy = UNSET,
    *,
    stacklevel: int = 2,
) -> ConversionReport:
    """
    Act on a report, according to the action that a policy takes for a
    loss of metadata.

    The action is the one that `on_error` takes for a
    [`MetadataLossError`][] (see
    [`ErrorPolicy`][brainhops.errors.ErrorPolicy]), and, when `on_error`
    has no entry for it, the policy in effect (see
    [`metadata_loss_policy`][]). Under the `"ignore"` action nothing
    happens, under `"warn"` a [`MetadataLossWarning`][] is issued, and
    any other action raises a [`MetadataLossError`][], which the method
    that called this function hands to the policy (see
    [`takes_error_policy`][]). A report that lists nothing as lost or
    approximated is silent under every action.

    When the action is itself a report, the entries of `report` are
    added to it, whether anything was lost or not, and nothing is warned
    or raised: the caller acts on the report it passed. Its `source` and
    `target` are set from `report` when it has none yet.

    Parameters
    ----------
    report : ConversionReport
        The report to act on.
    on_error : ErrorPolicy, optional
        The policy to apply. By default, or with `None`, the policy in
        effect.
    stacklevel : int, default=2
        The stack level of the warning, counted from the caller of this
        function and without the frame that [`takes_error_policy`][]
        adds around it.

    Returns
    -------
    ConversionReport
        `report` itself.

    Raises
    ------
    MetadataLossError
        If the action is neither `"ignore"`, `"warn"` nor a report, and
        the report is lossy.
    ValueError
        If `on_error` is a string other than an action.
    """
    action = UNSET
    if on_error is not UNSET and on_error is not None:
        action = policy_action(error_policy(on_error), MetadataLossError)
    if action is UNSET:
        action = _POLICY.get()
    if isinstance(action, ConversionReport):
        if action is not report:
            action.source = action.source or report.source
            action.target = action.target or report.target
            action.merge(report)
        return report
    if not report.lossy or _is(action, "ignore"):
        return report
    if _is(action, "warn"):
        # One frame more than asked, for the wrapper of the method.
        warnings.warn(MetadataLossWarning(report), stacklevel=stacklevel + 2)
        return report
    raise MetadataLossError(report)


def takes_error_policy(method: tx.Callable) -> tx.Callable:
    """
    Make a metadata method apply the error policy that it is given.

    A method of the metadata, such as `consolidate` or `from_instance`,
    takes an error policy as its `on_error=` keyword (see
    [`ErrorPolicy`][brainhops.errors.ErrorPolicy]), and some also take
    `lossy=`. The body of the method acts on what it loses by calling
    [`apply_loss_policy`][], which warns and returns as the method goes
    on, and this decorator applies the rest of the policy around the
    call.

    The decorator merges `on_error=` and `lossy=` into one policy, and
    passes to the method only the action for a loss of metadata, as the
    mapping `{MetadataLossError: action}`. An action that the body can
    take where the loss happens, which is `"ignore"`, `"warn"`,
    `"raise"` or a report, is passed as it is. Any other action, such as
    a value to return, is passed as `"raise"`, so that the body raises a
    [`MetadataLossError`][], which the decorator then hands to the
    policy. Every other error that the method raises is handed to the
    policy as well. Because the method receives a mapping that only
    concerns losses, it can pass its `on_error` on to another decorated
    method without widening the policy. A call with neither keyword
    calls the method as it is, and a loss then follows the policy in
    effect.

    The decorator also accepts the deprecated `on_loss=` keyword, with a
    `DeprecationWarning`. The value `v` of `on_loss=` is read as
    `on_error={MetadataLossError: v}`, which keeps its old meaning: it
    applies to a loss of metadata and to no other error.

    Parameters
    ----------
    method : callable
        The method. It takes `on_error=`, as a keyword parameter or
        through `**kwargs`, and it takes `lossy=` only if its signature
        names it.

    Returns
    -------
    callable
        The decorated method.
    """
    owner = method.__qualname__
    takes_lossy = "lossy" in inspect.signature(method).parameters

    @functools.wraps(method)
    def wrapper(*args: tx.Any, **kwargs: tx.Any) -> tx.Any:
        loss = pop_renamed_keyword(kwargs, "on_loss", "on_error", owner, UNSET)
        on_error = kwargs.pop("on_error", UNSET)
        lossy = kwargs.pop("lossy", None) if takes_lossy else None
        if loss is not UNSET and loss is not None:
            on_error = {MetadataLossError: loss}
        policy = error_policy(on_error, lossy=lossy)
        if not policy:
            return method(*args, **kwargs)
        action = _loss_action(policy)
        if action is not UNSET:
            kwargs["on_error"] = {MetadataLossError: action}
        try:
            return method(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            return apply_error_policy(error, policy, stacklevel=2)

    return wrapper


def call_under_policy(
    call: tx.Callable[[], tx.Any],
    on_error: ErrorPolicy = UNSET,
    *,
    lossy: ErrorAction = None,
    stacklevel: int = 2,
) -> tx.Any:
    """
    Call a conversion, and apply an error policy to what it raises.

    A conversion that has no `on_error=` keyword to pass on, such as the
    converters of transformations, runs under the policy of its caller.
    The action that the policy takes for a loss of metadata becomes the
    policy in effect (see [`metadata_loss_policy`][]) while `call` runs,
    so that the conversions of metadata that `call` triggers follow it,
    as [`takes_error_policy`][] arranges for a method. An error that
    `call` raises is then handed to the policy (see
    [`apply_error_policy`][brainhops.errors.apply_error_policy]). A
    policy that has no entry calls `call` as it is.

    Parameters
    ----------
    call : callable
        The conversion, called without arguments.
    on_error : ErrorPolicy, optional
        The policy. By default, every error is raised, and a loss of
        metadata follows the policy in effect.
    lossy : ErrorAction, optional
        The action for a loss, merged with `on_error` as
        [`error_policy`][brainhops.errors.error_policy] merges it.
    stacklevel : int, default=2
        The stack level of a warning, counted from the caller of this
        function.

    Returns
    -------
    Any
        What `call` returns, or what the policy returns in its place.
    """
    policy = error_policy(on_error, lossy=lossy)
    if not policy:
        return call()
    action = _loss_action(policy)
    scope = (
        contextlib.nullcontext()
        if action is UNSET
        else metadata_loss_policy(action)
    )
    with scope:
        try:
            return call()
        except Exception as error:  # noqa: BLE001
            return apply_error_policy(error, policy, stacklevel=stacklevel + 1)


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def short(value: tx.Any, width: int = 40) -> str:
    """
    A short text for a value in a report.

    Parameters
    ----------
    value : object
        The value. A known term reads as its string.
    width : int, optional
        The maximum number of characters.

    Returns
    -------
    str
        The `repr` of the value, cut with `...` when it is too long.
    """
    if isinstance(value, enum.Enum):
        # A known term reads as the term (`'scanner'`).
        value = value.value
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


_POLICIES = ("ignore", "warn", "raise")


def _check_policy(policy: str) -> str:
    if policy not in _POLICIES:
        raise ValueError(
            f"A metadata loss policy is one of {_POLICIES}, not {policy!r}."
        )
    return policy


def _is(action: tx.Any, name: str) -> bool:
    # `action == name` would compare an array element by element.
    return isinstance(action, str) and action == name


def _loss_action(policy: tx.Mapping[type, tx.Any]) -> tx.Any:
    # The action of `policy` for a loss of metadata, as the body of a
    # conversion can take it where the loss happens: one of the three
    # loss policies or a report. Any other action is taken as `"raise"`,
    # and the caller applies the policy to the error. `UNSET` when the
    # policy has no entry for a loss.
    action = policy_action(policy, MetadataLossError)
    if action is UNSET or isinstance(action, ConversionReport):
        return action
    if isinstance(action, str) and action in _POLICIES:
        return action
    return "raise"


# Not annotated: `ContextVar[...]` is not subscriptable on Python 3.8, and
# the type is inferred from the default anyway.
_POLICY = contextvars.ContextVar(
    "brainhops_metadata_loss_policy", default="warn"
)
