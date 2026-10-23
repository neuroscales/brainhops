"""Loss reports, and the loss policy."""

__all__ = [
    "ConversionReport",
    "LossPolicy",
    "MetadataLossError",
    "MetadataLossWarning",
    "OnLoss",
    "apply_loss_policy",
    "metadata_loss_policy",
]

# stdlib
import contextlib
import contextvars
import enum
import warnings

# externals
import typing_extensions as tx
from bagof.magic import Factory

# internals
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

    @property
    def lossy(self) -> bool:
        """Whether the report lists a field as lost or approximated."""
        return bool(self.lost or self.approximated)

    def merge(self, other: tx.Self) -> tx.Self:
        """
        Add the entries of another report to this report, in place.

        Parameters
        ----------
        other : ConversionReport
            The report whose entries are added. An entry of `other`
            replaces the entry of this report for the same field.

        Returns
        -------
        ConversionReport
            This report.
        """
        self.lost.update(other.lost)
        self.approximated.update(other.approximated)
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
        if not self.lossy:
            return f"Metadata conversion {where}: nothing lost."
        parts = []
        if self.lost:
            items = ", ".join(f"{k}={short(v)}" for k, v in self.lost.items())
            parts.append(f"lost {items}")
        if self.approximated:
            items = ", ".join(
                f"{k} ({v})" for k, v in self.approximated.items()
            )
            parts.append(f"approximated {items}")
        return f"Metadata conversion {where}: " + "; ".join(parts) + "."


class MetadataLossWarning(UserWarning):
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


class MetadataLossError(Exception):
    """
    The error raised, under the `"raise"` policy, when some metadata could
    not be carried over.

    The `report` attribute holds the [`ConversionReport`][] that says what
    was lost or approximated. The error is deliberately neither a
    `TypeError` nor a `ValueError`, because field converters turn those
    two into conversion errors, and a refused loss must surface as itself.
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


LossPolicy: tx.TypeAlias = tx.Literal["ignore", "warn", "raise"]
"""A loss policy: ignore a loss, warn about it, or raise."""

OnLoss: tx.TypeAlias = tx.Union[LossPolicy, ConversionReport]
"""
What an `on_loss=` argument takes: a [`LossPolicy`][], or a
[`ConversionReport`][] to fill with what is lost, instead of warning or
raising.
"""


@contextlib.contextmanager
def metadata_loss_policy(policy: OnLoss) -> tx.Iterator[None]:
    """
    Set the loss policy of the conversions and writes made in a block.

    The policy in effect governs every conversion and every
    consolidation that takes no `on_loss=` argument, in particular the
    implicit conversions that field converters trigger. Outside of any
    block, the policy is `"warn"`. A report given as the policy collects
    what the block loses, and nothing is warned or raised, so that a
    caller such as `io.save`, which converts an object and then writes
    it, acts once on one report.

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
    on_loss: tx.Optional[OnLoss] = None,
    *,
    stacklevel: int = 2,
) -> ConversionReport:
    """
    Act on a report, according to a loss policy.

    Under the `"ignore"` policy nothing happens, under `"warn"` a
    [`MetadataLossWarning`][] is issued, and under `"raise"` a
    [`MetadataLossError`][] is raised. A report that lists nothing as lost
    or approximated is silent under every policy.

    When `on_loss` is itself a report, the entries of `report` are added
    to it, whether anything was lost or not, and nothing is warned or
    raised: the caller acts on the report it passed. Its `source` and
    `target` are set from `report` when it has none yet.

    Parameters
    ----------
    report : ConversionReport
        The report to act on.
    on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
        The policy to apply, or a report to fill. By default, the policy
        in effect (see [`metadata_loss_policy`][]).
    stacklevel : int, optional
        The stack level of the warning, counted from the caller of this
        function.

    Returns
    -------
    ConversionReport
        `report` itself.

    Raises
    ------
    MetadataLossError
        If the policy is `"raise"` and the report is lossy.
    ValueError
        If `on_loss` is not a policy nor a report.
    """
    if on_loss is None:
        on_loss = _POLICY.get()
    if isinstance(on_loss, ConversionReport):
        if on_loss is not report:
            on_loss.source = on_loss.source or report.source
            on_loss.target = on_loss.target or report.target
            on_loss.merge(report)
        return report
    policy = _check_policy(on_loss)
    if not report.lossy or policy == "ignore":
        return report
    if policy == "raise":
        raise MetadataLossError(report)
    warnings.warn(MetadataLossWarning(report), stacklevel=stacklevel + 1)
    return report


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


# Not annotated: `ContextVar[...]` is not subscriptable on Python 3.8, and
# the type is inferred from the default anyway.
_POLICY = contextvars.ContextVar(
    "brainhops_metadata_loss_policy", default="warn"
)
