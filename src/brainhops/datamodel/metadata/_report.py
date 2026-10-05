"""Loss reports, and the loss policy."""

__all__ = [
    "ConversionReport",
    "LossPolicy",
    "MetadataLossError",
    "MetadataLossWarning",
    "OnLoss",
    "apply_loss_policy",
    "collect_loss_reports",
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
    The record of what a conversion or a write could not carry over.

    A conversion between metadata classes, or the write of metadata into
    a file, fills a report as it goes. A field whose value was dropped is
    listed in `lost`, with the value. A field whose value was stored, but
    not exactly, is listed in `approximated`, with a short description of
    the change (such as `"truncated to 80 bytes"`).

    The loss policy decides what happens to a report that is not empty
    (see [`metadata_loss_policy`][]).
    """

    source: tx.Annotated[
        tx.Optional[str], tx.Doc("The format converted from.")
    ] = None
    target: tx.Annotated[
        tx.Optional[str], tx.Doc("The format converted to.")
    ] = None
    lost: tx.Annotated[
        tx.Dict[str, tx.Any],
        tx.Doc("Field name -> the value that was dropped."),
        Factory(),
    ]
    approximated: tx.Annotated[
        tx.Dict[str, str],
        tx.Doc("Field name -> what changed in the stored value."),
        Factory(),
    ]

    @property
    def lossy(self) -> bool:
        """Whether the report lists a field as lost or approximated."""
        return bool(self.lost or self.approximated)

    def merge(self, other: "ConversionReport") -> tx.Self:
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

    @classmethod
    def merged(cls, reports: tx.Sequence["ConversionReport"]) -> tx.Self:
        """
        Merge several reports, in order, into a new report.

        The merged report goes from the source of the first report to the
        target of the last one, and holds the entries of all of them.
        When two reports have an entry for the same field, the later
        entry wins.

        Parameters
        ----------
        reports : sequence of ConversionReport
            The reports to merge, in the order the steps happened.

        Returns
        -------
        ConversionReport
            The merged report.
        """
        merged = cls(
            source=reports[0].source if reports else None,
            target=reports[-1].target if reports else None,
        )
        for report in reports:
            merged.merge(report)
        return merged

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


LossPolicy = tx.Literal["ignore", "warn", "raise"]
"""A loss policy: ignore a loss, warn about it, or raise."""

OnLoss = tx.Union[LossPolicy, ConversionReport]
"""
What an `on_loss=` argument takes: a [`LossPolicy`][], or a
[`ConversionReport`][] to fill with what is lost, instead of warning or
raising.
"""


@contextlib.contextmanager
def metadata_loss_policy(policy: LossPolicy) -> tx.Iterator[None]:
    """
    Set the loss policy of the conversions and writes made in a block.

    The policy in effect governs every conversion that takes no
    `on_loss=` argument, in particular the implicit conversions that field
    converters trigger: assigning a `NiftiMetadata` to a field typed
    `Metadata`, or saving an MGH image as NIfTI. Outside of any block, the
    policy is `"warn"`.

    Parameters
    ----------
    policy : {"ignore", "warn", "raise"}
        The policy: ignore a loss, warn about it, or raise
        [`MetadataLossError`][].

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
        nifti = NiftiImage.from_other(mgh)  # raises if anything is lost
    ```
    """
    token = _POLICY.set(_check_policy(policy))
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
    or approximated is silent under every policy. Inside a
    [`collect_loss_reports`][] block, a report that would be warned about
    is collected instead.

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
    if isinstance(on_loss, ConversionReport):
        if on_loss is not report:
            on_loss.source = on_loss.source or report.source
            on_loss.target = on_loss.target or report.target
            on_loss.merge(report)
        return report
    policy = _check_policy(on_loss or _POLICY.get())
    if not report.lossy or policy == "ignore":
        return report
    if policy == "raise":
        raise MetadataLossError(report)
    collected = _COLLECTED.get()
    if collected is not None:
        # Inside `collect_loss_reports`: the caller acts on it.
        collected.append(report)
        return report
    warnings.warn(MetadataLossWarning(report), stacklevel=stacklevel + 1)
    return report


@contextlib.contextmanager
def collect_loss_reports() -> tx.Iterator[tx.List[ConversionReport]]:
    """
    Collect the reports that the conversions and writes made in a block
    would warn about, instead of warning about them.

    Under the `"raise"` policy, a loss still raises where it happens.
    `io.save` uses this context manager to warn once for a save that
    first converts the object into the format of the file and then
    writes it.

    Yields
    ------
    list of ConversionReport
        The list that the collected reports are appended to.

    Examples
    --------
    ```python
    with collect_loss_reports() as reports:
        nifti = NiftiImage.from_other(mgh)
        nifti.save("out.nii.gz")
    if reports:
        apply_loss_policy(ConversionReport.merged(reports), "warn")
    ```
    """
    reports: tx.List[ConversionReport] = []
    token = _COLLECTED.set(reports)
    try:
        yield reports
    finally:
        _COLLECTED.reset(token)


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


# The reports collected by `collect_loss_reports`, when one is active (a
# list of `ConversionReport`), else `None`.
_COLLECTED = contextvars.ContextVar(
    "brainhops_metadata_loss_reports", default=None
)
