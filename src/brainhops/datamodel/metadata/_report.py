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
from brainhops._core.deprecation import external_stacklevel
from brainhops._core.errors import (
    UNSET,
    apply_error_policy,
    error_policy,
    policy_action,
)
from brainhops.errors import (
    ConversionError,
    ConversionWarning,
    ErrorAction,
    ErrorPolicy,
    LossyConversionError,
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
    `on_loss=report`, or a policy that maps an exception class to the
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
    that `on_loss=` and `on_error=` treat a loss of metadata as they treat
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
        self,
        report: ConversionReport,
        *,
        result: tx.Optional[DataModelBase] = None,
    ) -> None:
        """
        Parameters
        ----------
        report : ConversionReport
            What was lost or approximated.
        result : DataModelBase, optional
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
The values that the `on_loss=` argument of the metadata methods took
before it became the loss half of an error policy: a [`LossPolicy`][],
or a [`ConversionReport`][] to fill with what is lost, instead of
warning or raising. These values are still actions of an error policy,
and `on_loss=` now takes any action (see
[`ErrorPolicy`][brainhops.errors.ErrorPolicy]).
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
    *,
    stacklevel: int = 2,
) -> ConversionReport:
    """
    Act on a report, according to the loss policy in effect.

    The body of a conversion or of a consolidation calls this function
    with the report of what it could not carry over. The action is the
    policy in effect (see [`metadata_loss_policy`][]), which the call
    that received `on_error=` and `on_loss=` sets while its body runs
    (see [`call_under_policy`][]). Under the `"ignore"` action nothing
    happens, under `"warn"` a [`MetadataLossWarning`][] is issued, and
    under `"raise"` a [`MetadataLossError`][] is raised, which that call
    hands to its policy. A report that lists nothing as lost or
    approximated is silent under every action.

    When the action is itself a report, the entries of `report` are
    added to it, whether anything was lost or not, and nothing is warned
    or raised: the caller acts on the report it passed. Its `source` and
    `target` are set from `report` when it has none yet.

    Parameters
    ----------
    report : ConversionReport
        The report to act on.
    stacklevel : int, default=2
        The lowest stack level of the warning, counted from the caller
        of this function. The warning points at the first frame at or
        above it that is outside brainhops, which is the code that
        called the conversion.

    Returns
    -------
    ConversionReport
        `report` itself.

    Raises
    ------
    MetadataLossError
        If the action is `"raise"` and the report is lossy.
    """
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
        # Two frames more than asked, for the call that applies the
        # policy around the body.
        warnings.warn(
            MetadataLossWarning(report),
            stacklevel=external_stacklevel(stacklevel + 2),
        )
        return report
    raise MetadataLossError(report)


_CONVERSION_ERRORS: tx.Tuple[tx.Type[BaseException], ...] = (ConversionError,)


def call_under_policy(
    call: tx.Callable[[], tx.Any],
    on_error: ErrorPolicy = UNSET,
    *,
    on_loss: ErrorAction = UNSET,
    scope: tx.Tuple[tx.Type[BaseException], ...] = _CONVERSION_ERRORS,
    stacklevel: int = 2,
) -> tx.Any:
    """
    Call a conversion, and apply an error policy to what it raises.

    Every method that takes `on_error=` and `on_loss=` applies them
    with this function, and passes no policy on to the calls that its
    body makes. The function merges the two keywords into one policy,
    in which a single action of `on_error` applies to the classes of
    `scope` (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). While
    `call` runs, the action that the policy takes for a loss of
    metadata becomes the policy in effect (see
    [`metadata_loss_policy`][]), so that every consolidation and every
    conversion of metadata that `call` makes follows it, and acts on
    what it loses with [`apply_loss_policy`][]. An action that the body
    can take where the loss happens, which is `"ignore"`, `"warn"`,
    `"raise"` or a report, is set as it is. Any other action, such as a
    value to return or an exception to raise, is set as `"raise"`, so
    that the body raises a [`MetadataLossError`][]. An error that `call`
    raises is then handed to the policy, which raises it unchanged
    unless the policy has an entry for it, and the value that the
    policy returns takes the place of the result.

    A call with no policy calls `call` as it is, under the policy in
    effect. A method called without `on_error=` and `on_loss=` from the
    body of another one therefore follows the policy of the outer call,
    and only the outermost call returns a value in place of a result.

    Parameters
    ----------
    call : callable
        The conversion, called without arguments.
    on_error : ErrorPolicy, optional
        The policy. By default, every error is raised, and a loss of
        metadata follows the policy in effect.
    on_loss : ErrorAction, optional
        The action for a loss. The policy that the call applies is
        `{LossyConversionError: on_loss, **on_error}`, so that an entry
        for a loss in `on_error` takes precedence over `on_loss`.
    scope : tuple of type, default=(ConversionError,)
        The exception classes that `call` raises on purpose, to which a
        single action of `on_error` applies.
    stacklevel : int, default=2
        The lowest stack level of a warning, counted from the caller of
        this function. The warning points at the first frame at or
        above it that is outside brainhops.

    Returns
    -------
    Any
        What `call` returns, or what the policy returns in its place.
    """
    policy = error_policy(on_error, on_loss=on_loss, scope=scope)
    if not policy:
        return call()
    with _loss_context(policy):
        try:
            return call()
        except Exception as error:  # noqa: BLE001
            return apply_error_policy(error, policy, stacklevel=stacklevel + 1)


def takes_error_policy(method: tx.Callable) -> tx.Callable:
    """
    Give a method the keywords `on_error=` and `on_loss=` of a policy.

    The writers of the file formats use this decorator. It is written
    above each entry point of a writer, such as `to_bytes` or `save`,
    both in their common base, `WriterMixin`, and in each format that
    defines or overrides one. The wrapper removes `on_error=` and
    `on_loss=` from the keyword arguments and calls the method with
    [`call_under_policy`][], so that the body of the method never
    receives a policy and passes none on. A single action of
    `on_error=` applies to a `ConversionError`, which covers a loss of
    metadata, and never to an `OSError`. The method itself declares
    neither keyword and does not document them.

    An override that calls the method of its base class through
    `super()` passes no policy, so the inner call runs under the policy
    of the outer one, and only the outer call returns a value in place
    of a result. A method that is already decorated is returned as it
    is, so that it is never wrapped twice.

    A generator function, such as a `to_lines` method, runs its body
    only while its result is iterated. The wrapper of such a method
    returns a generator that sets the policy in effect for a loss while
    it asks the body for each item, and not while the caller handles
    that item. An error that the body raises during the iteration is
    handed to the policy. When the policy does not raise, the iteration
    ends there, and the value that the policy returns is dropped,
    because the generator has no result that the value could replace.
    The policy itself is checked when the method is called.

    Parameters
    ----------
    method : callable
        The method.

    Returns
    -------
    callable
        The decorated method.
    """
    if getattr(method, "takes_error_policy", False):
        return method
    if inspect.isgeneratorfunction(method):

        @functools.wraps(method)
        def wrapper(*args: tx.Any, **kwargs: tx.Any) -> tx.Any:
            on_error = kwargs.pop("on_error", UNSET)
            on_loss = kwargs.pop("on_loss", UNSET)
            policy = error_policy(on_error, on_loss=on_loss)
            items = method(*args, **kwargs)
            if not policy:
                return items
            return _iterate_under_policy(items, policy)

    else:

        @functools.wraps(method)
        def wrapper(*args: tx.Any, **kwargs: tx.Any) -> tx.Any:
            on_error = kwargs.pop("on_error", UNSET)
            on_loss = kwargs.pop("on_loss", UNSET)
            return call_under_policy(
                lambda: method(*args, **kwargs),
                on_error,
                on_loss=on_loss,
                stacklevel=3,
            )

    # Marks the method, so that it is not decorated twice, and so that
    # a test can check that every entry point of a writer is decorated.
    wrapper.takes_error_policy = True
    return wrapper


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


def _loss_context(
    policy: tx.Mapping[type, tx.Any],
) -> tx.ContextManager[None]:
    # The block in which the action of `policy` for a loss of metadata
    # is the policy in effect, or a block that changes nothing when the
    # policy has no entry for a loss.
    action = _loss_action(policy)
    if action is UNSET:
        return contextlib.nullcontext()
    return metadata_loss_policy(action)


def _iterate_under_policy(
    items: tx.Iterator[tx.Any], policy: tx.Mapping[type, tx.Any]
) -> tx.Iterator[tx.Any]:
    # The items of a generator, each one computed under `policy`. The
    # policy in effect is set only while the generator runs, so that it
    # does not leak into the code that consumes the items, and an error
    # of the generator ends the iteration once the policy has acted on
    # it.
    while True:
        with _loss_context(policy):
            try:
                item = next(items)
            except StopIteration:
                return
            except Exception as error:  # noqa: BLE001
                apply_error_policy(error, policy, stacklevel=2)
                return
        yield item


# Not annotated: `ContextVar[...]` is not subscriptable on Python 3.8, and
# the type is inferred from the default anyway.
_POLICY = contextvars.ContextVar(
    "brainhops_metadata_loss_policy", default="warn"
)
