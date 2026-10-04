"""Loss reports, and the loss policy."""

__all__ = [
    "ConversionReport",
    "LossPolicy",
    "MetadataLossError",
    "MetadataLossWarning",
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
    What a conversion or a write could not carry over.

    `lost` maps a field to the value that was dropped; `approximated`
    maps a field to a short description of what changed (`"truncated to
    80 bytes"`); `passed_through` lists the keys that were moved into a
    free-form store instead of a dedicated slot.
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
        Factory(dict),
    ]
    approximated: tx.Annotated[
        tx.Dict[str, str],
        tx.Doc("Field name -> what changed in the stored value."),
        Factory(dict),
    ]
    passed_through: tx.Annotated[
        tx.Tuple[str, ...],
        tx.Doc(
            "Fields a key/value format moved into its free-form store "
            "(`extra`) instead of losing them (see `Metadata._import`)."
        ),
    ] = ()

    @property
    def lossy(self) -> bool:
        """Whether anything was lost or approximated."""
        return bool(self.lost or self.approximated)

    def merge(self, other: "ConversionReport") -> "ConversionReport":
        """Add the entries of another report to this one, in place."""
        self.lost.update(other.lost)
        self.approximated.update(other.approximated)
        self.passed_through = tuple(
            dict.fromkeys(self.passed_through + other.passed_through)
        )
        return self

    @classmethod
    def merged(
        cls, reports: tx.Sequence["ConversionReport"]
    ) -> "ConversionReport":
        """
        One report of several, in order: from the source of the first to
        the target of the last, with the entries of all (a later entry
        wins over an earlier one for the same field).
        """
        merged = cls(
            source=reports[0].source if reports else None,
            target=reports[-1].target if reports else None,
        )
        for report in reports:
            merged.merge(report)
        return merged

    def raise_if_lossy(self) -> None:
        """Raise [`MetadataLossError`][] if anything was lost or
        approximated."""
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
    """Some metadata could not be carried over. `report` says what."""

    def __init__(self, report: ConversionReport) -> None:
        super().__init__(str(report))
        self.report = report


class MetadataLossError(Exception):
    """
    Some metadata could not be carried over, under the `"raise"` policy.
    `report` says what.

    It is deliberately neither a `TypeError` nor a `ValueError`: those
    are what field converters turn into conversion errors, and a refused
    loss must surface as itself.
    """

    def __init__(self, report: ConversionReport) -> None:
        super().__init__(str(report))
        self.report = report


LossPolicy = tx.Literal["ignore", "warn", "raise"]


@contextlib.contextmanager
def metadata_loss_policy(policy: LossPolicy) -> tx.Iterator[None]:
    """
    Set the loss policy for the conversions and writes in this block.

    This is what governs the implicit conversions that `bagof`'s field
    converters trigger (assigning a `NiftiMetadata` to a field typed
    `Metadata`, saving an MGH image as NIfTI, ...), which take no
    `on_loss=` argument.

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
    on_loss: tx.Optional[LossPolicy] = None,
    *,
    stacklevel: int = 2,
) -> ConversionReport:
    """
    Act on a report: do nothing, warn once, or raise.

    `on_loss` defaults to the policy in effect (see
    [`metadata_loss_policy`][]). A report with nothing lost or
    approximated is always silent. The report is returned.
    """
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
    Collect, instead of warning them, the reports that the conversions
    and writes in this block would warn about. Under the `"raise"`
    policy a loss still raises where it happens.

    `io.save` uses it to warn once for a save that converts the object
    into the format of the file and then writes it:

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


_POLICY: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "brainhops_metadata_loss_policy", default="warn"
)


_Reports = tx.Optional[tx.List[ConversionReport]]


# The reports collected by `collect_loss_reports`, when one is active.
_COLLECTED: "contextvars.ContextVar[_Reports]" = contextvars.ContextVar(
    "brainhops_metadata_loss_reports", default=None
)
