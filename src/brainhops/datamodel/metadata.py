"""
Non-spatial metadata, shared across file formats.

Every file format keeps descriptive metadata (a description, a repetition
time, slice timing, provenance, ...) under its own names and types. This
module gives them one representation, in three layers
(`docs/design/format-metadata.md`, M2):

1. a **common vocabulary**: one [`Magic`][bagof.magic.Magic] field per
   concept, named after its BIDS key in snake case and stored in BIDS
   units (seconds, degrees, tesla), on [`FormatMetadata`][];
2. a **raw record**, `raw`: the format's own faithful record (a `nibabel`
   header, an MRtrix header, ...), used by read-then-write in the same
   format and never copied across formats;
3. **extras**, `extra`: a free-form `str -> Any` namespace for what the
   vocabulary does not cover, copied into any free-form store a target
   format has.

**Three values per field.** A vocabulary field holds a value, `None`
("unknown") or [`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED]
("this format has no slot for it"). A format says which fields it can
store with the `supports=` class keyword; every other field defaults to
`UNSUPPORTED` and is refused at construction. Converting into a format
reports the values it cannot hold in a [`ConversionReport`][], and the
loss policy (`"ignore"`, `"warn"`, `"raise"`) decides what happens to the
report.

**Writing a format: the per-format hook API.** A format subclasses
`FormatMetadata` once, next to its parser:

```python
class MyMetadata(
    FormatMetadata,
    on={"format": "my"},                  # polymorphic discriminant
    supports=("description", "history"),  # everything else UNSUPPORTED
    derived=(),                           # fields the geometry owns
):
    format: tx.Literal["my"] = "my"
    raw: tx.Annotated[tx.Optional[MyRecord], NoRepr(), NoEq()] = None

    @classmethod
    def _default_raw(cls) -> MyRecord: ...

    @classmethod
    def _decode(cls, raw, *, image=None) -> dict: ...

    def _encode(self, raw, changed, *, image=None, report) -> MyRecord: ...

    def _geometry(self, image) -> dict: ...

    def _check_record(self, image) -> MyRecord: ...

    @classmethod
    def _import(cls, other, values, *, report) -> None: ...

    def _derive_raw(self, raw, *, grid_changed, volumes) -> MyRecord: ...
```

The hooks are all optional, and all private:

- `supports=` (class keyword): the vocabulary fields (and `"extra"`) the
  format can store, or [`ALL`][brainhops.datamodel.metadata.ALL].
  Omitted, a subclass keeps its parent's capabilities. The others
  default to `UNSUPPORTED`; `unsupported_fields` lists them on the class.
- `derived=` (class keyword): supported fields that are, for this format,
  a view of geometry the data model owns (NIfTI `repetition_time` is
  the time step, `pixdim[4]`). They are decoded on read. On write, the
  value the data model gives (`_geometry`) is what the writer stores:
  a changed field that disagrees with it is reported under
  `report.approximated` and never reaches `_encode`, one that agrees is
  dropped silently, and only a field the data model says nothing about
  is left to `_encode`. `derived_fields` lists them.
- `_default_raw() -> raw`: a fresh, empty record, for an object built in
  memory (or converted from another format). Defaults to `None`.
- `_decode(raw, *, image=None) -> dict`: record to common fields, on
  read. Returns vocabulary field names (and `"extra"`) to values; `None`
  values may be left out, and a value for a field the format does not
  support is a bug (`from_raw` raises `TypeError`). `image` is the data
  model object (image or transformation) the record belongs to, for
  fields that need it. A field whose decoding would read a lazy part of
  the record (the MGH tags, after the whole compressed volume) may be
  returned as [`Lazy`][brainhops.datamodel.metadata.Lazy]`(load)`: it is
  decoded on first access (or assignment), and joins the snapshot then.
- `_encode(raw, changed, *, image=None, report) -> raw`: common fields
  to record, on write. `raw` is the record to write over (already a
  copy, or the writer's own fresh record) and `changed` holds only the
  fields that differ from the read-time snapshot, so an untouched field
  keeps the record's value. A `None` in `changed` *clears* the slot.
  `changed["extra"]` is a per-key diff whose `None` values remove a key.
  Value-dependent loss goes in `report` (`report.lost[name] = value`,
  `report.approximated[name] = reason`). Returns the record to write.
- `_geometry(image) -> dict`: the values the data model gives for the
  derived fields (NIfTI: `{"repetition_time": <time step>}`), `None`
  where it says nothing. Defaults to `{}`.
- `_check_record(image) -> raw`: the record
  [`check_writable`][brainhops.datamodel.metadata.FormatMetadata.check_writable]
  encodes over: what the writer would pass to `write_raw` (NIfTI: the
  record, reshaped to the data of `image`), so that a value-dependent
  check reads the same state as a real write. Defaults to a copy of
  the record, or a default one.
- `_import(other, values, *, report) -> None`: called by `from_instance`
  with the source object, the values about to be passed to the
  constructor, and the report. A format may *recover* a loss here, for
  example by moving a lost vocabulary value into `values["extra"]` and
  removing it from `report.lost`.
- `_derive_raw(raw, *, grid_changed, volumes) -> raw`: called by
  [`derive`][brainhops.datamodel.metadata.FormatMetadata.derive] to
  scrub record content that is tied to the grid or to the volumes but is
  outside the vocabulary. It must not modify `raw` in place.

A reader builds the object with
[`from_raw`][brainhops.datamodel.metadata.FormatMetadata.from_raw],
which decodes the record and keeps the read-time snapshot; a parser
given a record and a `metadata` that is not that record's (explicit, or
carried by `replace()`) uses
[`with_record`][brainhops.datamodel.metadata.FormatMetadata.with_record],
which decodes the new record and keeps the changes. A writer builds its
record, calls
[`write_raw`][brainhops.datamodel.metadata.FormatMetadata.write_raw] with
it and a report (and `force=` for a writer keyword that must win over
the record, such as MGH `tr=`), and hands the report to
[`apply_loss_policy`][brainhops.datamodel.metadata.apply_loss_policy].
`io.save` wraps a conversion and the write that follows in
[`one_loss_warning`][brainhops.datamodel.metadata.one_loss_warning], so
that one save warns once.

**The `metadata` field of a format class.** `Image` and `Transformation`
declare `metadata: Optional[Metadata]` (keyword-only, out of `repr` and
`==`). A format narrows it to its own class, with a default factory,
which is what makes a change of format convert (and report). Both are
written with
[`metadata_annotation`][brainhops.datamodel.metadata.metadata_annotation],
whose converter also converts on a class that does not convert its
fields (a plain `Magic` parser), and *copies* a metadata object that is
already of the field's class
([`copy`][brainhops.datamodel.metadata.FormatMetadata.copy]: the record
is shared, the snapshot and `extra` are not): two objects never hold
the same metadata, so editing the result of `replace()` or `from_other`
never edits the original. `bagof` takes a field from the *first* base
that has it, so the narrowed declaration must sit on the class itself
or on its first base: a transformation format whose first base is a
data model transformation must declare it again (see
`NiftiBasedTransformation`). `DataModelBase.from_other` passes the
`metadata` of a data model it hands to a constructor (an x5 chain made
into a NIfTI field), so the field converts it and reports the loss.

**Unused so far.** Some of the surface has no producer in the
prototype formats yet, and is kept for the formats the design memo
plans: `ConversionReport.passed_through` and the `_import` hook (for a
key/value format, MRtrix or NRRD, that moves what it has no slot for
into its free-form store), and `ConversionReport.merge` outside
`one_loss_warning`.
"""

__all__ = [
    "ALL",
    "UNSUPPORTED",
    "Unsupported",
    "Maybe",
    "Bids",
    "Scope",
    "FILE",
    "ACQUISITION",
    "GRID",
    "VOLUME",
    "GeneratedBy",
    "Channel",
    "FormatMetadata",
    "Metadata",
    "OpaqueMetadata",
    "ConversionReport",
    "MetadataLossWarning",
    "MetadataLossError",
    "metadata_loss_policy",
    "get_metadata_loss_policy",
    "apply_loss_policy",
    "convert",
    "collect_loss_reports",
    "one_loss_warning",
    "Lazy",
    "metadata_annotation",
]

# stdlib
import contextlib
import contextvars
import copy
import datetime
import math
import warnings

# externals
import typing_extensions as tx
from bagof.converters import Converter
from bagof.magic import ConvertTo, Factory, Field, KwOnly, NoEq, NoRepr, fields

# internals
from .base import DataModelBase

# ----------------------------------------------------------------------
#   SENTINEL
# ----------------------------------------------------------------------


class Unsupported:
    """
    The type of [`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED].

    A field of a format that cannot store it holds `UNSUPPORTED`, which
    is distinct from `None` ("nobody set it"). There is one instance:
    it is falsy, never equal to anything but itself, and survives
    copying and pickling as the same object.
    """

    __slots__ = ()
    _instance: tx.ClassVar[tx.Optional["Unsupported"]] = None

    def __new__(cls) -> "Unsupported":
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNSUPPORTED"

    def __reduce__(self) -> str:
        # A string makes pickle (and `copy`) look the singleton up by name.
        return "UNSUPPORTED"


UNSUPPORTED = Unsupported()
"""The format cannot store this field. Singleton, falsy, not `None`."""

_T = tx.TypeVar("_T")

Maybe = tx.Union[_T, None, Unsupported]
"""
A vocabulary value: a value, `None` (unknown) or `UNSUPPORTED`.

`Maybe[float]` is `Union[float, None, Unsupported]`.
"""

ALL = "all"
"""`supports=ALL` declares that a format can store every field."""


class Lazy:
    """
    A value that `_decode` returns for a field whose decoding would read
    a lazy part of the record (the trailing tags of an MGZ, which sit
    after the whole compressed volume).

    `load` is called, with no argument, the first time the field is read
    (an attribute access, `repr`, `==`, `changed_fields()`, a copy into
    another format...). Its value then joins the read-time snapshot, as
    if it had been decoded with the rest. A `load` that reads a file
    should be picklable (a module-level function or a `functools.partial`
    of one), so that metadata still waiting for it can be pickled.
    """

    __slots__ = ("load",)

    def __init__(self, load: tx.Callable[[], tx.Any]) -> None:
        self.load = load

    def __repr__(self) -> str:
        return f"Lazy({self.load!r})"


# ----------------------------------------------------------------------
#   FIELD ANNOTATIONS
# ----------------------------------------------------------------------

FILE = "file"
"""Scope of a field about the file itself (kept by `derive`)."""

ACQUISITION = "acquisition"
"""Scope of a field invariant under resampling (kept by `derive`)."""

GRID = "grid"
"""Scope of a field tied to the voxel grid (cleared when it changes)."""

VOLUME = "volume"
"""Scope of a field with one entry per volume or channel."""

_SCOPES = (FILE, ACQUISITION, GRID, VOLUME)


class Bids(Field):
    """
    The BIDS sidecar key of a vocabulary field.

    Used as an annotation, `tx.Annotated[Maybe[float],
    Bids("RepetitionTime")]`; the key lands in the field's `metadata`
    under `"bids"`, where the sidecar codec reads it.
    """

    def __init__(self, key: str) -> None:
        super().__init__(metadata={"bids": key})


class Scope(Field):
    """
    The propagation scope of a vocabulary field: one of `"file"`,
    `"acquisition"`, `"grid"` or `"volume"` (see `derive`).

    Used as an annotation; the scope lands in the field's `metadata`
    under `"scope"`.
    """

    def __init__(self, scope: str) -> None:
        if scope not in _SCOPES:
            raise ValueError(f"A scope is one of {_SCOPES}, not {scope!r}.")
        super().__init__(metadata={"scope": scope})


# ----------------------------------------------------------------------
#   STRUCTURED VALUES
# ----------------------------------------------------------------------


class GeneratedBy(DataModelBase):
    """One entry of BIDS `GeneratedBy`: a program that made the data."""

    name: tx.Annotated[str, tx.Doc("Name of the program (BIDS `Name`).")]
    version: tx.Annotated[
        tx.Optional[str], tx.Doc("Its version (BIDS `Version`).")
    ] = None
    description: tx.Annotated[
        tx.Optional[str], tx.Doc("What it did (BIDS `Description`).")
    ] = None
    code_url: tx.Annotated[
        tx.Optional[str], tx.Doc("Where its code lives (BIDS `CodeURL`).")
    ] = None


class Channel(DataModelBase):
    """The description of one channel (or volume) of an image."""

    name: tx.Annotated[tx.Optional[str], tx.Doc("The channel label.")] = None
    color: tx.Annotated[
        tx.Optional[str], tx.Doc("Display color, as an RGBA hex string.")
    ] = None
    display_range: tx.Annotated[
        tx.Optional[tx.Tuple[float, float]],
        tx.Doc("Display window `(min, max)`."),
    ] = None
    unit: tx.Annotated[
        tx.Optional[str], tx.Doc("Unit of the channel's values.")
    ] = None


# ----------------------------------------------------------------------
#   LOSS REPORTING
# ----------------------------------------------------------------------


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
        tx.Doc("Keys moved to a free-form store (`extra`, keyval, ...)."),
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
            items = ", ".join(f"{k}={_short(v)}" for k, v in self.lost.items())
            parts.append(f"lost {items}")
        if self.approximated:
            items = ", ".join(
                f"{k} ({v})" for k, v in self.approximated.items()
            )
            parts.append(f"approximated {items}")
        return f"Metadata conversion {where}: " + "; ".join(parts) + "."


def _short(value: tx.Any, width: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


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


_POLICIES = ("ignore", "warn", "raise")

LossPolicy = tx.Literal["ignore", "warn", "raise"]

_POLICY: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "brainhops_metadata_loss_policy", default="warn"
)


def _check_policy(policy: str) -> str:
    if policy not in _POLICIES:
        raise ValueError(
            f"A metadata loss policy is one of {_POLICIES}, not {policy!r}."
        )
    return policy


def get_metadata_loss_policy() -> str:
    """The loss policy in effect (`"warn"` unless set)."""
    return _POLICY.get()


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
    policy = _check_policy(on_loss or get_metadata_loss_policy())
    if not report.lossy or policy == "ignore":
        return report
    if policy == "raise":
        raise MetadataLossError(report)
    collected = _COLLECTED.get()
    if collected is not None:
        # Inside `one_loss_warning`: warned once, merged, at its end.
        collected.append(report)
        return report
    warnings.warn(MetadataLossWarning(report), stacklevel=stacklevel + 1)
    return report


_Reports = tx.Optional[tx.List[ConversionReport]]

# The reports collected by `collect_loss_reports`, when one is active.
_COLLECTED: "contextvars.ContextVar[_Reports]" = contextvars.ContextVar(
    "brainhops_metadata_loss_reports", default=None
)


@contextlib.contextmanager
def collect_loss_reports() -> tx.Iterator[tx.List[ConversionReport]]:
    """
    Collect, instead of warning them, the reports that the conversions
    and writes in this block would warn about. Under the `"raise"`
    policy a loss still raises where it happens.
    """
    reports: tx.List[ConversionReport] = []
    token = _COLLECTED.set(reports)
    try:
        yield reports
    finally:
        _COLLECTED.reset(token)


@contextlib.contextmanager
def one_loss_warning(*, stacklevel: int = 2) -> tx.Iterator[None]:
    """
    Merge the warnings of the conversions and writes in this block into
    one [`MetadataLossWarning`][], issued when the block ends.

    `io.save` uses it, so that a save that converts the object into the
    format of the file (saving an MGH image as NIfTI) and then writes it
    warns once, with one report. Only warnings are merged: under the
    `"raise"` policy a loss raises where it happens, and nothing is
    warned if the block raises. `stacklevel=2` points the warning at
    the caller of the function that holds the block.
    """
    with collect_loss_reports() as reports:
        yield
    if reports:
        merged = ConversionReport(
            source=reports[0].source, target=reports[-1].target
        )
        for report in reports:
            merged.merge(report)
        apply_loss_policy(merged, "warn", stacklevel=stacklevel + 2)


# ----------------------------------------------------------------------
#   CAPABILITIES: supports= / derived=
# ----------------------------------------------------------------------

# The vocabulary, filled in once `FormatMetadata` exists: field name ->
# (annotation as declared, default when supported). `extra` is in it, so
# that a format can opt out of free-form keys with `supports=`.
_VOCABULARY: tx.Dict[str, tx.Tuple[tx.Any, tx.Any]] = {}

# Fields of `FormatMetadata` that are not vocabulary.
_NOT_VOCABULARY = ("format", "raw", "_decoded")

# What a default is when the factory builds it.
_FACTORY = object()


def _unsupported_hint(name: str) -> tx.Any:
    """The annotation of a vocabulary field a format cannot store: the
    declared one, without its default factory."""
    for field in fields(FormatMetadata):
        if field.name == name:
            extras = []
            if field.doc:
                extras.append(tx.Doc(field.doc))
            if isinstance(field.metadata, dict) and field.metadata:
                extras.append(Field(metadata=dict(field.metadata)))
            if not extras:
                return field.type
            return tx.Annotated[(field.type, *extras)]
    raise KeyError(name)  # pragma: no cover


class _FormatMetadataMeta(type(DataModelBase)):
    """
    Reads the `supports=` and `derived=` class keywords.

    `bagof` builds a class's fields before `__init_subclass__` runs, and
    does not forward class keywords to it, so the keywords are read here,
    before the class body reaches `bagof`: every vocabulary field that a
    class does not support is redeclared in its namespace with the
    default `UNSUPPORTED`, which is exactly what writing `x: T =
    UNSUPPORTED` in the class body does.
    """

    def __new__(
        metacls,
        name: str,
        bases: tx.Tuple[type, ...],
        namespace: tx.Dict[str, tx.Any],
        supports: tx.Optional[tx.Union[str, tx.Iterable[str]]] = None,
        derived: tx.Optional[tx.Iterable[str]] = None,
        **kwargs: tx.Any,
    ) -> type:
        own_repr = "__repr__" in namespace
        if supports is not None:
            _declare_supports(name, namespace, supports)
        cls = super().__new__(metacls, name, bases, namespace, **kwargs)
        if "__magic_discard__" in name:
            return cls
        _wrap_setattr(cls)
        if not own_repr:
            # `bagof` writes a `__repr__` for every class; this one also
            # hides `UNSUPPORTED` and an empty `extra`, or a format that
            # stores three fields would print thirty-five.
            cls.__repr__ = _metadata_repr
        if not _VOCABULARY:
            return cls
        unsupported = frozenset(
            field.name
            for field in fields(cls)
            if field.name in _VOCABULARY and field.default is UNSUPPORTED
        )
        cls.unsupported_fields = unsupported
        if derived is not None:
            derived = frozenset(derived)
            wrong = derived - (frozenset(_VOCABULARY) - unsupported)
            if wrong:
                raise TypeError(
                    f"{name} declares derived={sorted(wrong)}, which are "
                    f"not vocabulary fields it supports."
                )
            cls.derived_fields = derived
        else:
            cls.derived_fields = frozenset(
                getattr(cls, "derived_fields", frozenset()) - unsupported
            )
        return cls


def _wrap_setattr(cls: type) -> None:
    """
    Make assigning a field whose decoding was deferred (see `Lazy`)
    decode it first, so that the read-time snapshot holds the value the
    assignment replaces (`history = None` then clears the record's).
    `bagof` writes a `__setattr__` for every class, so it is wrapped on
    each one.
    """
    inner = cls.__dict__.get("__setattr__")
    if inner is None or getattr(inner, "_loads_pending", False):
        return

    def __setattr__(self: tx.Any, name: str, value: tx.Any) -> None:
        pending = self.__dict__.get("_pending")
        if pending and name in pending:
            self._load_pending(name)
        inner(self, name, value)

    __setattr__._loads_pending = True  # type: ignore[attr-defined]
    cls.__setattr__ = __setattr__


def _declare_supports(
    name: str,
    namespace: tx.Dict[str, tx.Any],
    supports: tx.Union[str, tx.Iterable[str]],
) -> tx.FrozenSet[str]:
    if not _VOCABULARY:
        raise TypeError(
            f"{name}: supports= is only for subclasses of FormatMetadata."
        )
    if isinstance(supports, str):
        if supports != ALL:
            raise TypeError(
                f"{name}: supports= takes a sequence of field names or "
                f"ALL, not {supports!r}."
            )
        supported = frozenset(_VOCABULARY)
    else:
        supported = frozenset(supports)
    unknown = supported - frozenset(_VOCABULARY)
    if unknown:
        raise TypeError(
            f"{name}: supports= names {sorted(unknown)}, which are not "
            f"vocabulary fields; expected some of {sorted(_VOCABULARY)}."
        )
    annotations = dict(namespace.get("__annotations__", {}))
    for field_name, (hint, default) in _VOCABULARY.items():
        if field_name in annotations or field_name in namespace:
            # Written out in the class body: the body wins.
            continue
        if field_name in supported:
            # Redeclared as supported, in case a parent did not support it.
            annotations[field_name] = hint
            if default is not _FACTORY:
                namespace[field_name] = default
        else:
            annotations[field_name] = _unsupported_hint(field_name)
            namespace[field_name] = UNSUPPORTED
    namespace["__annotations__"] = annotations
    return supported


def _metadata_repr(self: tx.Any) -> str:
    parts = []
    for field in fields(type(self)):
        if not field.repr or field.var:
            continue
        value = getattr(self, field.name, None)
        if value is None or value is UNSUPPORTED:
            continue
        if field.name == "extra" and not value:
            continue
        parts.append(f"{field.public_name}={value!r}")
    return f"{type(self).__name__}({', '.join(parts)})"


class _hybridmethod:
    """A method that also works on the class, binding the class."""

    def __init__(self, func: tx.Callable) -> None:
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj: tx.Any, owner: type) -> tx.Callable:
        target = owner if obj is None else obj
        return self.func.__get__(target, owner)


def _agrees(value: tx.Any, given: tx.Any) -> bool:
    """Whether a value agrees with what the data model gives (numbers
    within single-precision rounding)."""
    if isinstance(value, (int, float)) and isinstance(given, (int, float)):
        return math.isclose(value, given, rel_tol=1e-6, abs_tol=1e-9)
    return not _differs(value, given)


def _differs(a: tx.Any, b: tx.Any) -> bool:
    """Whether two field values differ, without trusting `!=` on arrays."""
    if a is b:
        return False
    try:
        return bool(a != b)
    except Exception:
        return True


# ----------------------------------------------------------------------
#   FORMAT METADATA
# ----------------------------------------------------------------------


class FormatMetadata(
    DataModelBase,
    metaclass=_FormatMetadataMeta,
    polymorphic=True,
    kw_only=True,
):
    """
    Common metadata vocabulary, plus one format's faithful record.

    Subclasses are selected on `format`: `FormatMetadata(format="nifti",
    ...)` builds a `NiftiMetadata` (once `brainhops.io` is imported).
    Each vocabulary field holds a value, `None` (unknown) or
    `UNSUPPORTED` (this format has no slot for it); see the module
    documentation for the per-format hooks.
    """

    # --- class attributes ---------------------------------------------

    unsupported_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc("The vocabulary fields this format cannot store."),
    ] = frozenset()

    derived_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc(
            "The vocabulary fields that this format derives from the "
            "geometry of the data model (read-only on write)."
        ),
    ] = frozenset()

    vocabulary_fields: tx.Annotated[
        tx.ClassVar[tx.Tuple[str, ...]],
        tx.Doc("The names of the common vocabulary fields."),
    ] = ()

    # --- format and record --------------------------------------------

    format: tx.Annotated[
        str,
        tx.Doc("The format this metadata belongs to; selects the subclass."),
    ] = "generic"

    raw: tx.Annotated[
        tx.Any,
        tx.Doc(
            """
            The format's own record (a `nibabel` header, ...). Edit it
            only for what the vocabulary does not cover: an untouched
            common field keeps the record's value on write, a common
            field you set wins over it. Never copied across formats.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    extra: tx.Annotated[
        Maybe[tx.Dict[str, tx.Any]],
        tx.Doc(
            "Free-form keys the vocabulary does not cover, copied into "
            "any free-form store a format has."
        ),
        Scope(FILE),
        Factory(dict),
    ]

    # --- 4.1 core and provenance --------------------------------------

    name: tx.Annotated[
        Maybe[str], tx.Doc("A short name for the data."), Scope(FILE)
    ] = None

    description: tx.Annotated[
        Maybe[str],
        tx.Doc("A free-text description."),
        Bids("Description"),
        Scope(FILE),
    ] = None

    history: tx.Annotated[
        Maybe[tx.Tuple[str, ...]],
        tx.Doc("The commands that produced the data, oldest first."),
        Scope(FILE),
    ] = None

    generated_by: tx.Annotated[
        Maybe[tx.Tuple[GeneratedBy, ...]],
        tx.Doc("The programs that produced the data."),
        Bids("GeneratedBy"),
        Scope(FILE),
    ] = None

    creation_time: tx.Annotated[
        Maybe[datetime.datetime],
        tx.Doc("When the file was created."),
        Scope(FILE),
    ] = None

    sources: tx.Annotated[
        Maybe[tx.Tuple[str, ...]],
        tx.Doc("Files the data was derived from (BIDS provenance)."),
        Bids("Sources"),
        Scope(FILE),
    ] = None

    space: tx.Annotated[
        Maybe[str],
        tx.Doc(
            "The label of the world space (`'MNI152NLin6Asym'`, "
            "`'scanner'`, ...); the space itself is geometry."
        ),
        Bids("SpatialReference"),
        Scope(FILE),
    ] = None

    intent: tx.Annotated[
        Maybe[str],
        tx.Doc("What the values are, as a NIfTI intent name."),
        Scope(FILE),
    ] = None

    # --- 4.2 MRI acquisition ------------------------------------------

    repetition_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Repetition time, in seconds."),
        Bids("RepetitionTime"),
        Scope(ACQUISITION),
    ] = None

    echo_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Echo time, in seconds."),
        Bids("EchoTime"),
        Scope(ACQUISITION),
    ] = None

    inversion_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Inversion time, in seconds."),
        Bids("InversionTime"),
        Scope(ACQUISITION),
    ] = None

    flip_angle: tx.Annotated[
        Maybe[float],
        tx.Doc("Flip angle, in degrees."),
        Bids("FlipAngle"),
        Scope(ACQUISITION),
    ] = None

    magnetic_field_strength: tx.Annotated[
        Maybe[float],
        tx.Doc("Nominal field strength, in tesla."),
        Bids("MagneticFieldStrength"),
        Scope(ACQUISITION),
    ] = None

    manufacturer: tx.Annotated[
        Maybe[str],
        tx.Doc("Manufacturer of the equipment."),
        Bids("Manufacturer"),
        Scope(ACQUISITION),
    ] = None

    manufacturers_model_name: tx.Annotated[
        Maybe[str],
        tx.Doc("Model name of the equipment."),
        Bids("ManufacturersModelName"),
        Scope(ACQUISITION),
    ] = None

    institution_name: tx.Annotated[
        Maybe[str],
        tx.Doc("Institution responsible for the equipment."),
        Bids("InstitutionName"),
        Scope(ACQUISITION),
    ] = None

    acquisition_time: tx.Annotated[
        Maybe[datetime.datetime],
        tx.Doc("When the acquisition started."),
        Bids("AcquisitionTime"),
        Scope(ACQUISITION),
    ] = None

    phase_encoding_direction: tx.Annotated[
        Maybe[str],
        tx.Doc("Phase-encoding axis, `'i'`, `'j-'`, ... (voxel axes)."),
        Bids("PhaseEncodingDirection"),
        Scope(GRID),
    ] = None

    total_readout_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Total readout time, in seconds."),
        Bids("TotalReadoutTime"),
        Scope(ACQUISITION),
    ] = None

    effective_echo_spacing: tx.Annotated[
        Maybe[float],
        tx.Doc("Effective echo spacing, in seconds."),
        Bids("EffectiveEchoSpacing"),
        Scope(ACQUISITION),
    ] = None

    slice_encoding_direction: tx.Annotated[
        Maybe[str],
        tx.Doc("Slice axis, `'i'`, `'j'`, `'k'`, ... (voxel axes)."),
        Bids("SliceEncodingDirection"),
        Scope(GRID),
    ] = None

    slice_timing: tx.Annotated[
        Maybe[tx.Tuple[float, ...]],
        tx.Doc("Acquisition time of each slice, in seconds."),
        Bids("SliceTiming"),
        Scope(GRID),
    ] = None

    multiband_acceleration_factor: tx.Annotated[
        Maybe[int],
        tx.Doc("Multiband acceleration factor."),
        Bids("MultibandAccelerationFactor"),
        Scope(ACQUISITION),
    ] = None

    # --- 4.3 diffusion ------------------------------------------------

    diffusion_bvalues: tx.Annotated[
        Maybe[tx.Tuple[float, ...]],
        tx.Doc("One b-value per volume, in s/mm^2."),
        Scope(VOLUME),
    ] = None

    diffusion_bvectors: tx.Annotated[
        Maybe[tx.Tuple[tx.Tuple[float, float, float], ...]],
        tx.Doc(
            "One unit gradient direction per volume, in world (RAS) "
            "coordinates."
        ),
        Scope(VOLUME),
    ] = None

    # --- 4.4 display and channels -------------------------------------

    display_range: tx.Annotated[
        Maybe[tx.Tuple[float, float]],
        tx.Doc("Display window `(min, max)`."),
        Scope(VOLUME),
    ] = None

    channels: tx.Annotated[
        Maybe[tx.Tuple[Channel, ...]],
        tx.Doc("One description per channel."),
        Scope(VOLUME),
    ] = None

    data_unit: tx.Annotated[
        Maybe[str], tx.Doc("Unit of the data values."), Scope(VOLUME)
    ] = None

    # --- 4.5 microscopy -----------------------------------------------

    objective_magnification: tx.Annotated[
        Maybe[float],
        tx.Doc("Nominal magnification of the objective."),
        Scope(ACQUISITION),
    ] = None

    objective_numerical_aperture: tx.Annotated[
        Maybe[float],
        tx.Doc("Numerical aperture of the objective."),
        Scope(ACQUISITION),
    ] = None

    illumination_type: tx.Annotated[
        Maybe[str], tx.Doc("Illumination type."), Scope(ACQUISITION)
    ] = None

    contrast_method: tx.Annotated[
        Maybe[str], tx.Doc("Contrast method."), Scope(ACQUISITION)
    ] = None

    # --- 4.6 transformation-specific ----------------------------------

    moving: tx.Annotated[
        Maybe[str],
        tx.Doc("The moving image of a registration (file reference)."),
        Scope(FILE),
    ] = None

    fixed: tx.Annotated[
        Maybe[str],
        tx.Doc("The fixed image of a registration (file reference)."),
        Scope(FILE),
    ] = None

    input_space: tx.Annotated[
        Maybe[str],
        tx.Doc("Label of the space a transformation maps from."),
        Scope(FILE),
    ] = None

    output_space: tx.Annotated[
        Maybe[str],
        tx.Doc("Label of the space a transformation maps to."),
        Scope(FILE),
    ] = None

    # --- snapshot -----------------------------------------------------

    _decoded: tx.Annotated[
        tx.Dict[str, tx.Any],
        tx.Doc(
            """
            The read-time snapshot: what the reader decoded from `raw`.
            A common field is written over the record only when it
            differs from it. Filled by the reader; never set by hand.
            """
        ),
        Factory(dict),
        NoRepr(),
        NoEq(),
    ]

    # --- construction -------------------------------------------------

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        cls = type(self)
        for name in cls.unsupported_fields:
            value = getattr(self, name, None)
            if value is None:
                setattr(self, name, UNSUPPORTED)
            elif value is not UNSUPPORTED:
                raise ValueError(
                    f"{cls.__name__} cannot store {name!r} (it is "
                    f"UNSUPPORTED by this format), so {value!r} is "
                    f"refused."
                )

    def __getattribute__(self, name: str) -> tx.Any:
        # A field whose decoding was deferred (see `Lazy`) is unset on
        # the instance (its class default would be read instead): it is
        # decoded now, once.
        if name[:1] != "_":
            state = object.__getattribute__(self, "__dict__")
            if name not in state:
                pending = state.get("_pending")
                if pending and name in pending:
                    return self._load_pending(name)
        return object.__getattribute__(self, name)

    def _load_pending(self, name: str) -> tx.Any:
        value = self.__dict__["_pending"].pop(name)()
        if value is UNSUPPORTED:
            value = None
        setattr(self, name, value)
        value = self.__dict__.get(name)
        if value is not None:
            self._decoded[name] = copy.deepcopy(value)
        return value

    def __getstate__(self) -> tx.Dict[str, tx.Any]:
        # `_source` (what a format whose record is rebuilt on each read,
        # Zarr, read it from: a store node) is a handle, not state, and
        # is not pickled nor deep-copied; `copy()` keeps it.
        state = dict(self.__dict__)
        state.pop("_source", None)
        return state

    @classmethod
    def from_raw(
        cls, raw: tx.Any, *, image: tx.Any = None, **values: tx.Any
    ) -> tx.Self:
        """
        Build the metadata of a record that was just read.

        The record is decoded into the common fields (`_decode`), and
        what was decoded is kept as the read-time snapshot, so that an
        untouched field keeps the record's value when it is written back.
        A field decoded as [`Lazy`][brainhops.datamodel.metadata.Lazy] is
        decoded on first access instead. Keyword arguments set fields
        over the decoded values (they then count as changes).

        Raises
        ------
        TypeError
            If `_decode` returned a value for a field this format does
            not support: a bug of the format, which would otherwise drop
            the value without a report.
        """
        decoded: tx.Dict[str, tx.Any] = {}
        lazy: tx.Dict[str, tx.Callable[[], tx.Any]] = {}
        for key, value in cls._decode(raw, image=image).items():
            if value is None or value is UNSUPPORTED:
                continue
            if key in cls.unsupported_fields:
                raise TypeError(
                    f"{cls.__name__}._decode returned {key}={value!r}, "
                    f"but {cls.__name__} does not support {key!r}."
                )
            if isinstance(value, Lazy):
                lazy[key] = value.load
            else:
                decoded[key] = value
        obj = cls(raw=raw, **decoded)
        # Snapshot the *converted* values, so that a decoded list held as
        # a tuple does not count as a change.
        obj._decoded = {
            key: copy.deepcopy(getattr(obj, key)) for key in decoded
        }
        if lazy:
            for key in lazy:
                # Unset, so that the first read decodes it.
                obj.__dict__.pop(key, None)
            obj.__dict__["_pending"] = lazy
        for key, value in values.items():
            setattr(obj, key, value)
        return obj

    def copy(self) -> tx.Self:
        """
        A copy of this metadata, sharing its record.

        The record (`raw`) is shared, as `replace()` shares it; the
        snapshot and `extra` are copied, so editing the copy never edits
        this object. This is what an image or a transformation holds
        when it is given metadata that another object holds already
        (`replace()`, `from_other`, `metadata=`). A field still waiting
        to be decoded (see `Lazy`) stays so in both.
        """
        new = copy.copy(self)
        if "_source" in self.__dict__:
            new.__dict__["_source"] = self.__dict__["_source"]
        new.__dict__["_decoded"] = dict(self._decoded)
        extra = self.__dict__.get("extra")
        if isinstance(extra, dict):
            new.__dict__["extra"] = dict(extra)
        pending = self.__dict__.get("_pending")
        if pending:
            new.__dict__["_pending"] = dict(pending)
        return new

    def with_record(self, raw: tx.Any, *, image: tx.Any = None) -> tx.Self:
        """
        The metadata of another record, keeping the changes made here.

        `raw` is decoded (`from_raw`), and every field that changed since
        this object was read (`changed_fields()`: all the fields that
        are set, for an object built in memory) is set over the decoded
        values, as a change. `extra` is merged key by key. This is what
        an object given a new record holds (`replace(image,
        header=...)`), and what `metadata=` given along with a record
        becomes.
        """
        changed = {
            key: value
            for key, value in self.changed_fields().items()
            if key not in type(self).unsupported_fields
        }
        extra = changed.pop("extra", None)
        obj = type(self).from_raw(raw, image=image)
        for key, value in changed.items():
            setattr(obj, key, value)
        if extra:
            merged = dict(obj.extra or {})
            for key, value in extra.items():
                if value is None:
                    merged.pop(key, None)
                else:
                    merged[key] = value
            obj.extra = merged
        return obj

    # --- capabilities -------------------------------------------------

    @_hybridmethod
    def supports(self_or_cls, name: str) -> bool:
        """
        Whether this format can store the vocabulary field `name`.

        On the class, this reads the class declaration (`supports=`); on
        an instance, it reads the instance, for formats whose capability
        depends on the instance.
        """
        if name not in _VOCABULARY:
            raise KeyError(f"{name!r} is not a vocabulary field.")
        if isinstance(self_or_cls, type):
            return name not in self_or_cls.unsupported_fields
        return getattr(self_or_cls, name) is not UNSUPPORTED

    # --- the change-detecting write -----------------------------------

    def changed_fields(self) -> tx.Dict[str, tx.Any]:
        """
        The common fields that differ from the read-time snapshot.

        An object read from a file and left untouched has none. An object
        built in memory has an empty snapshot, so every field that is not
        `None` is a change. A field set to `None` after a read is a change
        to `None` (the record's slot is cleared on write). `extra` is
        compared key by key: its entry is the per-key diff, where `None`
        removes a key.
        """
        snapshot = self._decoded or {}
        changed: tx.Dict[str, tx.Any] = {}
        for name in _VOCABULARY:
            value = getattr(self, name, None)
            if value is UNSUPPORTED:
                continue
            if name == "extra":
                diff = _extra_diff(snapshot.get("extra"), value)
                if diff:
                    changed["extra"] = diff
            elif _differs(value, snapshot.get(name)):
                changed[name] = value
        return changed

    def write_raw(
        self,
        raw: tx.Any = None,
        *,
        image: tx.Any = None,
        report: tx.Optional[ConversionReport] = None,
        force: tx.Collection[str] = (),
    ) -> tx.Any:
        """
        Encode the common fields over a record, and return the record.

        `raw` is the record to write over: a writer passes its own fresh
        record, already filled with what it keeps from `self.raw`. When
        it is `None`, a copy of `self.raw` (or a default record) is used.
        Only the fields that changed since the read are encoded (see
        [`changed_fields`][brainhops.datamodel.metadata.FormatMetadata.changed_fields]),
        and the fields named in `force`, whether they changed or not (a
        writer keyword that must win over the record, such as MGH `tr=`;
        a `None` there clears the slot).

        A derived field (`derived=`) for which the data model gives a
        value (`_geometry`) is not encoded: the data model's value is
        what the writer stores, and a changed value that disagrees with
        it is reported as approximated. A field this format does not
        support but that was assigned after construction is recorded as
        lost in `report`, as are the value-dependent losses `_encode`
        finds.
        """
        if report is None:
            report = ConversionReport(source=self.format, target=self.format)
        unsupported = type(self).unsupported_fields
        for name in unsupported:
            value = getattr(self, name, None)
            if value is not None and value is not UNSUPPORTED:
                report.lost[name] = value
        if raw is None:
            raw = self._raw_or_default()
        changed = {
            key: value
            for key, value in self.changed_fields().items()
            if key not in unsupported
        }
        for name in force:
            if name in _VOCABULARY and name != "extra":
                if name not in unsupported:
                    changed[name] = getattr(self, name)
        self._check_derived(changed, image, report)
        return self._encode(raw, changed, image=image, report=report)

    def _check_derived(
        self,
        changed: tx.Dict[str, tx.Any],
        image: tx.Any,
        report: ConversionReport,
    ) -> None:
        """Take out of `changed` the derived fields the data model gives,
        and report those that disagree with it."""
        names = sorted(type(self).derived_fields & changed.keys())
        if image is None or not names:
            return
        geometry = self._geometry(image)
        for name in names:
            given = geometry.get(name)
            if given is None:
                continue
            value = changed.pop(name)
            if value is not None and not _agrees(value, given):
                report.approximated[name] = (
                    f"derived from the data model ({_short(given)})"
                )

    def check_writable(self, *, image: tx.Any = None) -> ConversionReport:
        """
        What a write of this object would lose, without writing it.

        Class-level declarations are the lower bound of loss; this runs
        the encoder on a scratch record (the one the format's writer
        would start from, see `_check_record`), so value-dependent
        losses (an over-long description, an irregular slice timing) are
        included. Pass the image (or transformation) for the fields that
        need it.
        """
        report = ConversionReport(source=self.format, target=self.format)
        self.write_raw(self._check_record(image), image=image, report=report)
        return report

    def _raw_or_default(self) -> tx.Any:
        if self.raw is not None:
            return copy.deepcopy(self.raw)
        return self._default_raw()

    # --- per-format hooks ---------------------------------------------

    @classmethod
    def _default_raw(cls) -> tx.Any:
        """A fresh, empty record for this format."""
        return None

    @classmethod
    def _decode(cls, raw: tx.Any, *, image: tx.Any = None) -> tx.Dict:
        """Record to common fields. Default: nothing to decode."""
        return {}

    def _encode(
        self,
        raw: tx.Any,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> tx.Any:
        """Common fields to record. Default: the record is unchanged."""
        return raw

    def _geometry(self, image: tx.Any) -> tx.Dict[str, tx.Any]:
        """
        The values the data model gives for the derived fields (NIfTI:
        the time step of the image as `repetition_time`). A field left
        out, or `None`, is one the data model says nothing about: its
        changed value is then left to `_encode`. Default: nothing.
        """
        return {}

    def _check_record(self, image: tx.Any) -> tx.Any:
        """
        The record `check_writable` encodes over: what the writer of
        this format would pass to `write_raw` for `image`. Default: a
        copy of the record, or a default one.
        """
        return self._raw_or_default()

    @classmethod
    def _import(
        cls,
        other: "FormatMetadata",
        values: tx.Dict[str, tx.Any],
        *,
        report: ConversionReport,
    ) -> None:
        """Recover losses of a conversion. Default: nothing."""

    def _derive_raw(
        self,
        raw: tx.Any,
        *,
        grid_changed: bool,
        volumes: tx.Optional[tx.Sequence[int]],
    ) -> tx.Any:
        """Scrub grid- or volume-bound record content. Default: none."""
        return raw

    # --- conversion ---------------------------------------------------

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Convert the metadata of another format into this one.

        The record and the snapshot are never copied across formats
        (they are when `other` is already of this class). Each vocabulary
        value is copied, except where this format declares the field
        unsupported: the value is then lost, and reported. `UNSUPPORTED`
        on the source side reads as `None`. The report is handed to the
        loss policy in effect (see [`metadata_loss_policy`][]); use
        [`convert`][brainhops.datamodel.metadata.convert] to get it back.
        """
        if not isinstance(other, FormatMetadata):
            return super().from_instance(other, *args, **kwargs)
        obj, report = cls._convert_from(other, args, kwargs)
        apply_loss_policy(report, stacklevel=3)
        return obj

    @classmethod
    def _convert_from(
        cls,
        other: "FormatMetadata",
        args: tx.Tuple[tx.Any, ...] = (),
        kwargs: tx.Optional[tx.Dict[str, tx.Any]] = None,
    ) -> tx.Tuple["FormatMetadata", ConversionReport]:
        """`from_instance`, returning the report instead of acting on it."""
        kwargs = dict(kwargs or {})
        target = cls
        same = isinstance(other, cls)
        if same:
            # A copy: keep the most specific class.
            target = type(other)
        report = ConversionReport(
            source=_format_name(other), target=_format_name(target)
        )
        values: tx.Dict[str, tx.Any] = {}
        if same:
            values["raw"] = other.raw
            values["decoded"] = dict(other._decoded)
        unsupported = target.unsupported_fields
        for name in _VOCABULARY:
            value = getattr(other, name, None)
            if value is None or value is UNSUPPORTED:
                continue
            if name == "extra":
                if not value:
                    continue
                value = dict(value)
            if name in unsupported:
                report.lost[name] = value
                continue
            values[name] = value
        target._import(other, values, report=report)
        values.update(kwargs)
        return target(*args, **values), report

    # --- propagation --------------------------------------------------

    def derive(
        self,
        *,
        grid_changed: bool = False,
        volumes: tx.Optional[tx.Sequence[int]] = None,
        volumes_changed: bool = False,
        step: tx.Optional[str] = None,
    ) -> tx.Self:
        """
        The metadata of an object derived from this one (resampled,
        cropped, a selection of volumes...). Always a new object.

        - `file` fields are kept, except `creation_time` (cleared) and
          `history`, to which `step` is appended; `generated_by` gains a
          brainhops entry once.
        - `acquisition` fields are kept.
        - `grid` fields are cleared when `grid_changed`.
        - `volume` fields are indexed by `volumes` (the selected volume
          indices) when given, and cleared when `volumes_changed` and no
          selection is known.
        - `extra` is kept verbatim; `raw` and the snapshot are kept, so
          a cleared field is cleared in the record on write, and
          `_derive_raw` scrubs what the vocabulary does not cover.
        """
        values: tx.Dict[str, tx.Any] = {}
        for name in _VOCABULARY:
            value = getattr(self, name, None)
            if value is UNSUPPORTED or name in self.unsupported_fields:
                continue
            scope = _field_scope(name)
            if name == "creation_time":
                value = None
            elif name == "history" and step is not None:
                value = tuple(value or ()) + (step,)
            elif name == "generated_by":
                value = _with_brainhops(value)
            elif name == "extra":
                value = dict(value or {})
            elif scope == GRID and grid_changed:
                value = None
            elif scope == VOLUME and value is not None:
                value = _select_volumes(name, value, volumes, volumes_changed)
            values[name] = value
        values["raw"] = self._derive_raw(
            self.raw, grid_changed=grid_changed, volumes=volumes
        )
        values["decoded"] = dict(self._decoded)
        return type(self)(**values)


# The vocabulary, read off the class that declares it.
for _field in fields(FormatMetadata):
    if _field.name in _NOT_VOCABULARY or _field.var:
        continue
    _VOCABULARY[_field.name] = (
        FormatMetadata.__annotations__[_field.name],
        _FACTORY if _field.name == "extra" else _field.default,
    )
del _field
FormatMetadata.vocabulary_fields = tuple(
    name for name in _VOCABULARY if name != "extra"
)


def _field_metadata(name: str) -> tx.Dict[str, tx.Any]:
    for field in fields(FormatMetadata):
        if field.name == name:
            meta = field.metadata
            return dict(meta) if isinstance(meta, dict) else {}
    raise KeyError(name)


def _field_scope(name: str) -> str:
    """The propagation scope of a vocabulary field."""
    return _field_metadata(name).get("scope", FILE)


def _bids_key(name: str) -> tx.Optional[str]:
    """The BIDS key of a vocabulary field, or `None` if BIDS has none."""
    return _field_metadata(name).get("bids")


def _extra_diff(
    before: tx.Optional[tx.Mapping], after: tx.Any
) -> tx.Dict[str, tx.Any]:
    before = dict(before or {})
    after = dict(after or {})
    diff = {
        key: value
        for key, value in after.items()
        if key not in before or _differs(value, before[key])
    }
    for key in before:
        if key not in after:
            diff[key] = None
    return diff


def _with_brainhops(
    generated_by: tx.Optional[tx.Tuple[GeneratedBy, ...]],
) -> tx.Tuple[GeneratedBy, ...]:
    entries = tuple(generated_by or ())
    if any(getattr(g, "name", None) == "brainhops" for g in entries):
        return entries
    try:
        from brainhops import __version__ as version
    except ImportError:  # pragma: no cover
        version = None
    return entries + (GeneratedBy(name="brainhops", version=version),)


def _select_volumes(
    name: str,
    value: tx.Any,
    volumes: tx.Optional[tx.Sequence[int]],
    volumes_changed: bool,
) -> tx.Any:
    if name == "display_range" or name == "data_unit":
        # One value for every volume: a selection keeps it.
        return value
    if volumes is not None:
        try:
            return tuple(value[i] for i in volumes)
        except (IndexError, TypeError):
            return None
    return None if volumes_changed else value


def _format_name(obj: tx.Any) -> str:
    if isinstance(obj, type):
        for field in fields(obj):
            if field.name == "format":
                default = field.default
                if isinstance(default, str):
                    return default
        return obj.__name__
    return str(getattr(obj, "format", type(obj).__name__))


class Metadata(FormatMetadata, on={"format": "generic"}, supports=ALL):
    """
    Format-agnostic metadata: every field supported, no record.

    This is what in-memory images and transformations carry, the hub
    through which formats convert (`NiftiMetadata -> Metadata ->
    MghMetadata` loses exactly what `NiftiMetadata -> MghMetadata`
    loses), and what the BIDS sidecar codec reads and writes.
    """

    format: tx.Annotated[
        tx.Literal["generic"], tx.Doc("Always `'generic'`.")
    ] = "generic"

    raw: tx.Annotated[
        None, tx.Doc("Always `None`: no record."), NoRepr(), NoEq()
    ] = None

    @classmethod
    def from_bids(cls, sidecar: tx.Any) -> "Metadata":
        """
        Read a BIDS JSON sidecar: a mapping, a JSON string, or a path.

        Keys that name a vocabulary field (through its BIDS key) fill
        that field; every other key lands in `extra`.
        """
        from brainhops.io.metadata.bids import from_bids

        return from_bids(sidecar)

    def to_bids(
        self, *, on_loss: tx.Optional[LossPolicy] = None
    ) -> tx.Dict[str, tx.Any]:
        """
        The BIDS JSON sidecar (a JSON-serialisable `dict`) of this
        metadata. The diffusion fields are not sidecar keys, and are
        reported as lost.
        """
        from brainhops.io.metadata.bids import to_bids

        return to_bids(self, on_loss=on_loss)


class OpaqueMetadata(FormatMetadata, on={"format": "opaque"}, supports=()):
    """
    The metadata of a format that stores none: every field unsupported.

    Base class of the formats that store a bare matrix (FLIRT `.mat`,
    matrix text, ITK `.tfm`). A read-then-save loses nothing, because
    nothing is there.
    """

    format: tx.Annotated[
        tx.Literal["opaque"], tx.Doc("Always `'opaque'`.")
    ] = "opaque"

    raw: tx.Annotated[
        None, tx.Doc("Always `None`: no record."), NoRepr(), NoEq()
    ] = None


class _Copied:
    """
    The converter of a `metadata` field: converts what it is given into
    the field's type, as `bagof` would, and copies a metadata object
    that already is of that type (`FormatMetadata.copy`), so that two
    images or transformations never hold the same one.
    """

    def __init__(self, hint: tx.Any) -> None:
        self.hint = hint
        self._convert: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None

    def __call__(self, value: tx.Any) -> tx.Any:
        if self._convert is None:
            self._convert = Converter.get(self.hint)
        out = self._convert(value)
        if out is value and isinstance(out, FormatMetadata):
            out = out.copy()
        return out


def metadata_annotation(
    hint: tx.Any, doc: str, *, default: tx.Any = None
) -> tx.Any:
    """
    The annotation of a `metadata` field: `hint` (a `FormatMetadata`
    subclass, or `Optional` of one), keyword-only, out of `repr` and
    `==`, converted on assignment (even on a class that does not convert
    its fields) and copied rather than shared.

    `default` is a `FormatMetadata` subclass to build the default from
    (a format's narrowed field), or `None` for a `None` default (the data
    model roots). A narrowed field must be declared on the class itself
    or on its first base (see the module documentation).
    """
    extras: tx.List[tx.Any] = [
        tx.Doc(doc),
        ConvertTo(_Copied(hint)),
        KwOnly(),
        NoRepr(),
        NoEq(),
    ]
    if default is not None:
        extras.append(Factory(default))
    return tx.Annotated[(hint, *extras)]


def _metadata_class(target: tx.Any) -> tx.Type[FormatMetadata]:
    if isinstance(target, type) and issubclass(target, FormatMetadata):
        return target
    if isinstance(target, str):
        stack = [FormatMetadata]
        while stack:
            klass = stack.pop()
            if _format_name(klass) == target and klass is not FormatMetadata:
                return klass
            stack.extend(klass.__subclasses__())
        raise ValueError(f"No metadata class for the format {target!r}.")
    raise TypeError(
        f"Expected a FormatMetadata subclass or a format name, got {target!r}."
    )


def convert(
    source: FormatMetadata,
    target: tx.Union[str, tx.Type[FormatMetadata]],
    *,
    on_loss: tx.Optional[LossPolicy] = None,
    **values: tx.Any,
) -> tx.Tuple[FormatMetadata, ConversionReport]:
    """
    Convert metadata into another format, and say what was lost.

    Parameters
    ----------
    source : FormatMetadata
        The metadata to convert.
    target : type or str
        The `FormatMetadata` subclass to convert to, or its format name
        (`"generic"`, `"nifti"`, ...).
    on_loss : {"ignore", "warn", "raise"}, optional
        What to do if anything is lost. Defaults to the policy in effect
        (see [`metadata_loss_policy`][]).
    **values
        Fields to set on the result.

    Returns
    -------
    target : FormatMetadata
        The converted metadata.
    report : ConversionReport
        What was lost or approximated.
    """
    klass = _metadata_class(target)
    obj, report = klass._convert_from(source, (), values)
    apply_loss_policy(report, on_loss, stacklevel=2)
    return obj, report
