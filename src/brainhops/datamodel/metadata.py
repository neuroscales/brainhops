"""
Non-spatial metadata, shared across file formats.

Every file format keeps descriptive metadata (a description, a repetition
time, slice timing, provenance, ...) under its own names and types. This
module gives them one representation (`docs/design/format-metadata.md`,
M2), as a class hierarchy that mirrors the images'
(`Image` -> `FileBasedImage` -> `NiftiImage`):

- [`Metadata`][brainhops.datamodel.metadata.Metadata]: the **common
  vocabulary**, one [`Magic`][bagof.magic.Magic] field per concept, named
  after its BIDS key in snake case and stored in BIDS units (seconds,
  degrees, tesla), plus **extras**, `extra`, a free-form `str -> Any`
  namespace for what the vocabulary does not cover, copied into any
  free-form store a target format has. It is what in-memory images and
  transformations carry, and the hub through which formats convert. The
  vocabulary is declared in six groups (mixins):
  [`ProvenanceMetadata`][brainhops.datamodel.metadata.ProvenanceMetadata],
  [`MRIMetadata`][brainhops.datamodel.metadata.MRIMetadata],
  [`DiffusionMetadata`][brainhops.datamodel.metadata.DiffusionMetadata],
  [`DisplayMetadata`][brainhops.datamodel.metadata.DisplayMetadata],
  [`MicroscopyMetadata`][brainhops.datamodel.metadata.MicroscopyMetadata]
  and
  [`TransformMetadata`][brainhops.datamodel.metadata.TransformMetadata];
  `Metadata` inherits them all.
- [`FileBasedMetadata`][brainhops.datamodel.metadata.FileBasedMetadata]
  adds the **raw record**, `raw`: the format's own faithful record (a
  `nibabel` header, an MRtrix header, ...), used by read-then-write in
  the same format and never copied across formats, and the read-time
  snapshot of what was decoded from it.
- One `<Fmt>Metadata` per format, a `FileBasedMetadata` subclass that
  lives next to its parser under `brainhops.io`.

**Three values per field.** A vocabulary field holds a value, `None`
("unknown") or [`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED]
("this format has no slot for it"). A format says which fields it can
store with the `supports=` class keyword; every other field defaults to
`UNSUPPORTED` and is refused at construction. Converting into a format
(`metadata.to(NiftiMetadata)`) reports the values it cannot hold in a
[`ConversionReport`][], and the loss policy (`"ignore"`, `"warn"`,
`"raise"`) decides what happens to the report.

**Writing a format: the per-format hook API.** A format subclasses
`FileBasedMetadata` once, next to its parser:

```python
class MyMetadata(
    FileBasedMetadata,
    on={"format": "my"},                  # polymorphic discriminant
    supports=(ProvenanceMetadata, "echo_time"),  # everything else UNSUPPORTED
    derived=(),                           # fields the geometry owns
    lazy=("history",),                    # fields decoded on first access
):
    format: tx.Literal["my"] = "my"
    raw: tx.Annotated[tx.Optional[MyRaw], NoRepr(), NoEq()] = None

    @classmethod
    def _default_raw(cls) -> MyRaw: ...

    @classmethod
    def _decode(cls, raw, *, image=None) -> dict: ...

    def _encode(self, raw, changed, *, image=None, report) -> MyRaw: ...

    def _geometry(self, image) -> dict: ...

    def _check_raw(self, image) -> MyRaw: ...

    @classmethod
    def _import(cls, other, values, *, report) -> None: ...

    def _derive_raw(self, raw, *, grid_changed, volumes) -> MyRaw: ...
```

The hooks are all optional, and all private:

| Hook | Used by | Gives |
|---|---|---|
| `supports=` | the class | `supported_fields` (`unsupported_fields`) |
| `derived=` | the class | `derived_fields` |
| `lazy=` | the class | `lazy_fields`, one `LazyField` each |
| `_default_raw()` | in-memory objects | a fresh raw record |
| `_decode(raw, *, image)` | `from_raw` | the common fields of a raw record |
| `_encode(raw, changed, *, image, report)` | `update_raw` | the raw record |
| `_geometry(image)` | `update_raw` | the data model's derived values |
| `_check_raw(image)` | `check_writable` | the raw a writer starts from |
| `_import(other, values, *, report)` | `to`, `from_other` | recovered losses |
| `_derive_raw(raw, *, grid_changed, volumes)` | `derive` | a scrubbed raw |

- `supports=` (class keyword): the vocabulary fields (and `"extra"`) the
  format can store, as names, group classes (all the fields of the
  group) or [`ALL`][brainhops.datamodel.metadata.ALL]. Omitted, a
  subclass keeps its parent's capabilities. The others default to
  `UNSUPPORTED`. `supported_fields` lists what it stores, and
  `unsupported_fields`, derived from it, the rest.
- `derived=` (class keyword): supported fields that are, for this format,
  a view of geometry the data model owns (NIfTI `repetition_time` is
  the time step, `pixdim[4]`). They are decoded on read. On write, the
  value the data model gives (`_geometry`) is what the writer stores:
  a changed field that disagrees with it is reported under
  `report.approximated` and never reaches `_encode`, one that agrees is
  dropped silently, and only a field the data model says nothing about
  is left to `_encode`. `derived_fields` lists them.
- `lazy=` (class keyword): supported fields whose decoding would read a
  lazy part of the raw record (the MGH tags, after the whole compressed
  volume). Each gets a
  [`LazyField`][brainhops._core.properties.LazyField] descriptor, and
  `_decode` may return
  [`Lazy`][brainhops._core.properties.Lazy]`(load)` for it: the field
  is decoded on first access (or assignment), and joins the snapshot
  then. Any other attribute access is plain.
- `_default_raw() -> raw`: a fresh, empty raw record, for an object
  built in memory (or converted from another format). Defaults to
  `None`.
- `_decode(raw, *, image=None) -> dict`: raw record to common fields, on
  read. Returns vocabulary field names (and `"extra"`) to values; `None`
  values may be left out. `image` is the data model object (image or
  transformation) the raw record belongs to, for fields that need it.
- `_encode(raw, changed, *, image=None, report) -> raw`: common fields
  to raw record, on write. `raw` is the record to write over (already a
  copy, or the writer's own fresh record) and `changed` holds only the
  fields that differ from the read-time snapshot, so an untouched field
  keeps the record's value. A `None` in `changed` *clears* the slot.
  `changed["extra"]` is a per-key diff whose `None` values remove a key.
  Value-dependent loss goes in `report` (`report.lost[name] = value`,
  `report.approximated[name] = reason`). Returns the raw record to write.
- `_geometry(image) -> dict`: the values the data model gives for the
  derived fields (NIfTI: `{"repetition_time": <time step>}`), `None`
  where it says nothing. Defaults to `{}`.
- `_check_raw(image) -> raw`: the raw record
  [`check_writable`][brainhops.datamodel.metadata.FileBasedMetadata.check_writable]
  encodes over: what the writer would pass to `update_raw` (NIfTI: the
  record, reshaped to the data of `image`), so that a value-dependent
  check reads the same state as a real write. Defaults to a copy of the
  record, or a default one.
- `_import(other, values, *, report) -> None`: called on a conversion
  with the source object, the values about to be passed to the
  constructor, and the report. A format may *recover* a loss here, for
  example by moving a lost vocabulary value into `values["extra"]` and
  removing it from `report.lost`.
- `_derive_raw(raw, *, grid_changed, volumes) -> raw`: called by
  [`derive`][brainhops.datamodel.metadata.FileBasedMetadata.derive] to
  scrub raw content that is tied to the grid or to the volumes but is
  outside the vocabulary. It must not modify `raw` in place.

A `_decode` that returns a value for a field its class does not
support (or a `Lazy` for a field not in `lazy=`) is a bug of the format
class, and `from_raw` raises `TypeError` rather than drop it unreported.

**One word for the raw record.** A reader builds the object with
[`from_raw`][brainhops.datamodel.metadata.FileBasedMetadata.from_raw],
which decodes the raw record and keeps the read-time snapshot; a parser
given a raw record and a `metadata` that is not that record's (explicit,
or carried by `replace()`) uses
[`update_from_raw`][brainhops.datamodel.metadata.FileBasedMetadata.update_from_raw],
which decodes the new raw record and keeps the changes. A writer builds
its raw record, calls
[`update_raw`][brainhops.datamodel.metadata.FileBasedMetadata.update_raw]
with it and a report (and `force=` for a writer keyword that must win
over the record, such as MGH `tr=`), and hands the report to
[`apply_loss_policy`][brainhops.datamodel.metadata.apply_loss_policy].
`io.save` wraps a conversion and the write that follows in
[`one_loss_warning`][brainhops.datamodel.metadata.one_loss_warning], so
that one save warns once.

**The `metadata` field of a format class.** `Image` and `Transformation`
declare `metadata: Optional[Metadata]` (keyword-only, out of `repr` and
`==`). A format narrows it to its own class, with a default factory,
which is what makes a change of format convert (and report). Both are
written with
[`MetadataField`][brainhops.datamodel.metadata.MetadataField]`[hint,
*annotations]`, whose converter also converts on a class that does not
convert its fields (a plain `Magic` parser), converts a metadata object
of another class into the field's class (a `NiftiMetadata` given to a
field typed `Metadata` becomes generic `Metadata`), and *copies* a
metadata object that already is of the field's class
([`copy`][brainhops.datamodel.metadata.FileBasedMetadata.copy]: the raw
record is shared, the snapshot and `extra` are not): two objects never
hold the same metadata, so editing the result of `replace()` or
`from_other` never edits the original. `bagof` takes a field from the
*first* base that has it, so the narrowed declaration must sit on the
class itself or on its first base: a transformation format whose first
base is a data model transformation must declare it again (see
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
    "VOCABULARY",
    "GROUPS",
    "GeneratedBy",
    "Channel",
    "EncodingDirection",
    "ProvenanceMetadata",
    "MRIMetadata",
    "DiffusionMetadata",
    "DisplayMetadata",
    "MicroscopyMetadata",
    "TransformMetadata",
    "Metadata",
    "FileBasedMetadata",
    "OpaqueMetadata",
    "MetadataField",
    "ConversionReport",
    "MetadataLossWarning",
    "MetadataLossError",
    "metadata_loss_policy",
    "get_metadata_loss_policy",
    "apply_loss_policy",
    "collect_loss_reports",
    "one_loss_warning",
    "preferred_dtype",
    "Lazy",
    "LazyField",
]

# stdlib
import contextlib
import contextvars
import copy
import datetime
import enum
import inspect
import itertools
import math
import warnings

# externals
import numpy as np
import typing_extensions as tx
from bagof.converters import Converter
from bagof.magic import (
    ConvertTo,
    Factory,
    Field,
    KwOnly,
    Magic,
    NoEq,
    NoRepr,
    fields,
)

# internals
from brainhops._core.compat import own_annotations
from brainhops._core.properties import Lazy, LazyField

from .base import DataModelBase
from .enums import (
    ContrastMethod,
    IlluminationType,
    Intent,
    Manufacturer,
    Space,
)
from .units import Unit

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
#   CONVERTERS
# ----------------------------------------------------------------------


def _passes(value: tx.Any) -> bool:
    """`None` and `UNSUPPORTED` go through every vocabulary converter."""
    return value is None or value is UNSUPPORTED


class _Term:
    """
    The converter of a free-text field with a list of known terms (an
    enum): a known term becomes the enum member, any other string stays
    a string. `bagof` does not do this for a `Union[Enum, str]` by
    itself (a string already satisfies the union).
    """

    def __init__(self, enum: type) -> None:
        self.enum = enum

    def __call__(self, value: tx.Any) -> tx.Any:
        if _passes(value) or isinstance(value, self.enum):
            return value
        if isinstance(value, str):
            try:
                return self.enum(value)
            except ValueError:
                return str(value)
        raise TypeError(
            f"Expected a {self.enum.__name__} or a str, not "
            f"{type(value).__name__}."
        )


def _unit(value: tx.Any) -> tx.Any:
    """A unit name the units module parses, as a `Unit`; a name it
    cannot parse stays a string, so that a file's own spelling survives."""
    if _passes(value) or isinstance(value, Unit):
        return value
    if isinstance(value, str):
        try:
            return Unit(value)
        except ValueError:
            return value
    raise TypeError(f"Expected a Unit or a str, not {type(value).__name__}.")


def _dtype(value: tx.Any) -> tx.Any:
    """A numpy data type, in native byte order: the byte order is
    storage encoding, never metadata (M1)."""
    if _passes(value):
        return value
    return np.dtype(value).newbyteorder("=")


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


_AXES = "ijk"


def _bids_vector(value: str) -> tx.Tuple[float, ...]:
    """The unit vector of a BIDS direction (`"i"`, `"j-"`, `"k"`)."""
    axis = value[:-1] if value.endswith("-") else value
    if len(axis) != 1 or axis not in _AXES:
        raise ValueError(
            f"A BIDS direction is one of 'i', 'j', 'k', optionally "
            f"followed by '-', not {value!r}."
        )
    vector = [0.0] * len(_AXES)
    vector[_AXES.index(axis)] = -1.0 if value.endswith("-") else 1.0
    return tuple(vector)


def _vector(value: tx.Any) -> tx.Tuple[float, ...]:
    if isinstance(value, str):
        return _bids_vector(value)
    return tuple(float(v) for v in np.ravel(np.asarray(value, dtype=float)))


class EncodingDirection(DataModelBase):
    """
    The direction of an encoding axis (phase or slice encoding): a unit
    vector in a named coordinate system.

    `space=None` means the image's own voxel (array) axes, the frame of
    BIDS `PhaseEncodingDirection`: `EncodingDirection("j-")` is
    `EncodingDirection((0, -1, 0))`, and compares equal to `"j-"`. A
    direction in voxel axes that is aligned with one of them reads and
    writes as a BIDS string (`to_bids()`); any other one (an oblique
    direction, after a resampling) is kept exactly, and the formats that
    can only store an axis report it as lost.
    """

    vector: tx.Annotated[
        tx.Tuple[float, ...],
        tx.Doc(
            "The direction, a unit vector (normalised on construction); "
            "a BIDS string (`'j-'`) is accepted."
        ),
        ConvertTo(_vector),
    ]
    space: tx.Annotated[
        tx.Optional[tx.Union[Space, str]],
        tx.Doc(
            "The coordinate system of `vector`: `None` for the image's "
            "voxel axes, or the label of a world space."
        ),
        ConvertTo(_Term(Space)),
    ] = None

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        vector = np.asarray(self.vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        if not vector.size or not norm or not math.isfinite(norm):
            raise ValueError(
                f"A direction is a non-zero vector, not {self.vector!r}."
            )
        if not math.isclose(norm, 1.0, rel_tol=1e-12):
            self.vector = tuple(float(v) for v in vector / norm)

    @classmethod
    def from_bids(cls, value: str) -> "EncodingDirection":
        """The direction of a BIDS string (`"i"`, `"j-"`, `"k"`), in
        voxel axes."""
        return cls(_bids_vector(value))

    def axis(self) -> tx.Optional[tx.Tuple[int, int]]:
        """`(index, sign)` of the voxel axis this direction is aligned
        with, or `None` (another space, or an oblique direction)."""
        if self.space is not None:
            return None
        vector = np.asarray(self.vector, dtype=float)
        index = int(np.argmax(np.abs(vector)))
        if not math.isclose(abs(vector[index]), 1.0, abs_tol=1e-6):
            return None
        return index, (1 if vector[index] > 0 else -1)

    def to_bids(self) -> tx.Optional[str]:
        """The BIDS string of this direction (`"j-"`), or `None` when it
        is not aligned with one of the first three voxel axes."""
        axis = self.axis()
        if axis is None or axis[0] >= len(_AXES):
            return None
        index, sign = axis
        return _AXES[index] + ("-" if sign < 0 else "")

    def transform(self, linear: tx.Any) -> "EncodingDirection":
        """The direction after a linear map of its space (`linear`, a
        matrix from the old axes to the new ones)."""
        matrix = np.asarray(linear, dtype=float)
        vector = matrix @ np.asarray(self.vector, dtype=float)
        return EncodingDirection(tuple(vector), space=self.space)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = EncodingDirection(other)
            except ValueError:
                return False
        if not isinstance(other, EncodingDirection):
            return NotImplemented
        if self.space != other.space or len(self.vector) != len(other.vector):
            return False
        return bool(np.allclose(self.vector, other.vector, atol=1e-9))

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        bids = self.to_bids()
        if bids is not None:
            return f"EncodingDirection({bids!r})"
        vector = tuple(round(v, 6) for v in self.vector)
        if self.space is None:
            return f"EncodingDirection({vector!r})"
        return f"EncodingDirection({vector!r}, space={str(self.space)!r})"


def _direction(value: tx.Any) -> tx.Any:
    """The converter of an encoding direction field: a BIDS string, a
    vector, a mapping (`vector`/`space`, or the JSON `Vector`/`Space`)."""
    if _passes(value) or isinstance(value, EncodingDirection):
        return value
    if isinstance(value, tx.Mapping):
        vector = value.get("vector", value.get("Vector"))
        space = value.get("space", value.get("Space"))
        return EncodingDirection(vector, space=space)
    return EncodingDirection(value)


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
    if isinstance(value, enum.Enum):
        # A known term reads as the term (`'scanner'`).
        value = value.value
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
#   VOCABULARY GROUPS
# ----------------------------------------------------------------------


class _VocabularyGroup(Magic, kw_only=True, convert=True):
    """
    Base of the vocabulary groups: `Magic` mixins that declare fields and
    nothing else. They convert their fields (`convert=True`: a mixin's
    fields keep the options of the class that declares them), and are
    not meant to be instantiated: [`Metadata`][] inherits them all.
    """


class ProvenanceMetadata(_VocabularyGroup):
    """
    Vocabulary group: what the data is and where it comes from (`file`
    scope). Not meant to be instantiated; see [`Metadata`][].
    """

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
        Maybe[tx.Union[Space, str]],
        tx.Doc(
            "The label of the world space (`'MNI152NLin6Asym'`, "
            "`'scanner'`, ...; a known one is a `Space`); the space itself "
            "is geometry."
        ),
        Bids("SpatialReference"),
        Scope(FILE),
        ConvertTo(_Term(Space)),
    ] = None

    intent: tx.Annotated[
        Maybe[tx.Union[Intent, str]],
        tx.Doc(
            "What the values are, as a NIfTI intent name (a known one is "
            "an `Intent`)."
        ),
        Scope(FILE),
        ConvertTo(_Term(Intent)),
    ] = None


class MRIMetadata(_VocabularyGroup):
    """
    Vocabulary group: MRI acquisition parameters (`acquisition` scope,
    except the encoding directions and the slice timing, `grid`). Not
    meant to be instantiated; see [`Metadata`][].
    """

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
        Maybe[tx.Union[Manufacturer, str]],
        tx.Doc(
            "Manufacturer of the equipment (a known one is a `Manufacturer`)."
        ),
        Bids("Manufacturer"),
        Scope(ACQUISITION),
        ConvertTo(_Term(Manufacturer)),
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
        Maybe[EncodingDirection],
        tx.Doc(
            "Phase-encoding direction: a unit vector, by default in voxel "
            "axes (BIDS `'j-'` is accepted and compares equal)."
        ),
        Bids("PhaseEncodingDirection"),
        Scope(GRID),
        ConvertTo(_direction),
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
        Maybe[EncodingDirection],
        tx.Doc(
            "Slice-encoding direction: a unit vector, by default in voxel "
            "axes (BIDS `'k'` is accepted and compares equal)."
        ),
        Bids("SliceEncodingDirection"),
        Scope(GRID),
        ConvertTo(_direction),
    ] = None

    slice_timing: tx.Annotated[
        Maybe[tx.Tuple[float, ...]],
        tx.Doc(
            "Acquisition time of each slice, in seconds, along the slice "
            "encoding direction."
        ),
        Bids("SliceTiming"),
        Scope(GRID),
    ] = None

    multiband_acceleration_factor: tx.Annotated[
        Maybe[int],
        tx.Doc("Multiband acceleration factor."),
        Bids("MultibandAccelerationFactor"),
        Scope(ACQUISITION),
    ] = None


class DiffusionMetadata(_VocabularyGroup):
    """
    Vocabulary group: the diffusion gradient table (`volume` scope). BIDS
    stores it in `.bval`/`.bvec` files, not in the sidecar. Not meant to
    be instantiated; see [`Metadata`][].
    """

    bvalues: tx.Annotated[
        Maybe[tx.Tuple[float, ...]],
        tx.Doc("One b-value per volume, in s/mm^2."),
        Scope(VOLUME),
    ] = None

    bvectors: tx.Annotated[
        Maybe[tx.Tuple[tx.Tuple[float, float, float], ...]],
        tx.Doc(
            "One unit gradient direction per volume, in world (RAS) "
            "coordinates."
        ),
        Scope(VOLUME),
    ] = None


class DisplayMetadata(_VocabularyGroup):
    """
    Vocabulary group: how the values are shown and what they are
    (`volume` scope, except `data_type`, `grid`). Not meant to be
    instantiated; see [`Metadata`][].
    """

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
        Maybe[tx.Union[Unit, str]],
        tx.Doc(
            "Unit of the data values: a `Unit` whenever the units module "
            "parses the name (`'a.u.'`, `'mm/s'`, `'HU'`), and the name "
            "itself, as a string, when it does not, so that a file with "
            "an odd unit still reads. Formats write it as its symbol "
            "(`Unit.symbol`), which parses back to the same unit."
        ),
        Scope(VOLUME),
        ConvertTo(_unit),
    ] = None

    data_type: tx.Annotated[
        Maybe[np.dtype],
        tx.Doc(
            "The element type of the data as stored (in native byte "
            "order), which may differ from the type of the loaded array "
            "(a scaled integer file loads as floats). Writers store the "
            "data as this type when the array's values are of its kind, "
            "and a `dtype=` writer option wins over it."
        ),
        Scope(GRID),
        ConvertTo(_dtype),
    ] = None


class MicroscopyMetadata(_VocabularyGroup):
    """
    Vocabulary group: microscopy acquisition (`acquisition` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

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
        Maybe[tx.Union[IlluminationType, str]],
        tx.Doc("Illumination type (a known one is an `IlluminationType`)."),
        Scope(ACQUISITION),
        ConvertTo(_Term(IlluminationType)),
    ] = None

    contrast_method: tx.Annotated[
        Maybe[tx.Union[ContrastMethod, str]],
        tx.Doc("Contrast method (a known one is a `ContrastMethod`)."),
        Scope(ACQUISITION),
        ConvertTo(_Term(ContrastMethod)),
    ] = None


class TransformMetadata(_VocabularyGroup):
    """
    Vocabulary group: what a transformation relates (`file` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

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
        Maybe[tx.Union[Space, str]],
        tx.Doc("Label of the space a transformation maps from."),
        Scope(FILE),
        ConvertTo(_Term(Space)),
    ] = None

    output_space: tx.Annotated[
        Maybe[tx.Union[Space, str]],
        tx.Doc("Label of the space a transformation maps to."),
        Scope(FILE),
        ConvertTo(_Term(Space)),
    ] = None


_GROUP_CLASSES = (
    ProvenanceMetadata,
    MRIMetadata,
    DiffusionMetadata,
    DisplayMetadata,
    MicroscopyMetadata,
    TransformMetadata,
)

GROUPS: tx.Dict[type, tx.Tuple[str, ...]] = {
    group: tuple(own_annotations(group)) for group in _GROUP_CLASSES
}
"""Each vocabulary group class, and the names of its fields."""

VOCABULARY: tx.Tuple[str, ...] = tuple(
    itertools.chain.from_iterable(GROUPS.values())
)
"""
The names of the common vocabulary fields, group by group. `extra` is not
one (it is the free-form store next to them). `bagof` lists the fields of
a class with several bases in reverse MRO order, so this is the order to
iterate in, never `fields(Metadata)`.
"""

# The vocabulary, filled in once `Metadata` exists: field name ->
# (annotation as declared, default when supported). `extra` is in it, so
# that a format can opt out of free-form keys with `supports=`.
_VOCABULARY: tx.Dict[str, tx.Tuple[tx.Any, tx.Any]] = {}

# What a default is when the factory builds it.
_FACTORY = object()


# ----------------------------------------------------------------------
#   CAPABILITIES: supports= / derived= / lazy=
# ----------------------------------------------------------------------


def _unsupported_hint(name: str) -> tx.Any:
    """The annotation of a vocabulary field a format cannot store: the
    declared one, without its default factory nor its converter."""
    for field in fields(Metadata):
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


class _MetadataMeta(type(DataModelBase)):
    """
    Reads the `supports=`, `derived=` and `lazy=` class keywords.

    `bagof` builds a class's fields before `__init_subclass__` runs, and
    does not forward class keywords to it, so the keywords are read here,
    before the class body reaches `bagof`: every vocabulary field that a
    class does not support is redeclared in its namespace with the
    default `UNSUPPORTED`, which is exactly what writing `x: T =
    UNSUPPORTED` in the class body does. The lazy fields get their
    descriptor once the class is built.
    """

    def __new__(
        metacls,
        name: str,
        bases: tx.Tuple[type, ...],
        namespace: tx.Dict[str, tx.Any],
        supports: tx.Optional[tx.Union[str, tx.Iterable[tx.Any]]] = None,
        derived: tx.Optional[tx.Iterable[str]] = None,
        lazy: tx.Optional[tx.Iterable[str]] = None,
        **kwargs: tx.Any,
    ) -> type:
        own_repr = "__repr__" in namespace
        if supports is not None:
            _declare_supports(name, namespace, supports)
        cls = super().__new__(metacls, name, bases, namespace, **kwargs)
        if "__magic_discard__" in name:
            return cls
        if not own_repr:
            # `bagof` writes a `__repr__` for every class; this one also
            # hides `UNSUPPORTED` and an empty `extra`, or a format that
            # stores three fields would print forty.
            cls.__repr__ = _metadata_repr
        if _VOCABULARY:
            _set_capabilities(cls, derived, lazy)
        elif derived is not None or lazy is not None:
            raise TypeError(f"{name}: derived= and lazy= need a vocabulary.")
        return cls


def _set_capabilities(
    cls: type,
    derived: tx.Optional[tx.Iterable[str]] = None,
    lazy: tx.Optional[tx.Iterable[str]] = None,
) -> None:
    name = cls.__name__
    supported = frozenset(
        field.name
        for field in fields(cls)
        if field.name in _VOCABULARY and field.default is not UNSUPPORTED
    )
    cls.supported_fields = supported
    cls.unsupported_fields = frozenset(_VOCABULARY) - supported
    if derived is not None:
        derived = frozenset(derived)
        wrong = derived - supported
        if wrong:
            raise TypeError(
                f"{name} declares derived={sorted(wrong)}, which are "
                f"not vocabulary fields it supports."
            )
        cls.derived_fields = derived
    else:
        cls.derived_fields = frozenset(
            getattr(cls, "derived_fields", frozenset()) & supported
        )
    if lazy is not None:
        lazy = frozenset(lazy)
        wrong = lazy - (supported - {"extra"})
        if wrong:
            raise TypeError(
                f"{name} declares lazy={sorted(wrong)}, which are not "
                f"vocabulary fields it supports."
            )
        cls.lazy_fields = lazy
    else:
        cls.lazy_fields = frozenset(
            getattr(cls, "lazy_fields", frozenset()) & supported
        )
    for field in fields(cls):
        if field.name not in cls.lazy_fields:
            continue
        # Installed on every class that has a lazy field: a subclass
        # that redeclares the field (`supports=`) hides its parent's.
        if not isinstance(inspect.getattr_static(cls, field.name), LazyField):
            setattr(
                cls,
                field.name,
                LazyField(
                    field.name,
                    default=field.default,
                    prepare=_unsupported_as_none,
                    on_load=_join_snapshot,
                ),
            )


def _unsupported_as_none(value: tx.Any) -> tx.Any:
    return None if value is UNSUPPORTED else value


def _join_snapshot(obj: tx.Any, name: str, value: tx.Any) -> None:
    """A lazy field, once loaded, joins the read-time snapshot, as if it
    had been decoded with the rest."""
    snapshot = obj.__dict__.get("_decoded")
    if value is not None and snapshot is not None:
        setattr(snapshot, name, copy.deepcopy(value))


def _declare_supports(
    name: str,
    namespace: tx.Dict[str, tx.Any],
    supports: tx.Union[str, tx.Iterable[tx.Any]],
) -> tx.FrozenSet[str]:
    if not _VOCABULARY:
        raise TypeError(f"{name}: supports= is only for Metadata subclasses.")
    if isinstance(supports, str):
        if supports != ALL:
            raise TypeError(
                f"{name}: supports= takes a sequence of field names or "
                f"vocabulary groups, or ALL, not {supports!r}."
            )
        supported = frozenset(_VOCABULARY)
    else:
        names: tx.Set[str] = set()
        for item in supports:
            if isinstance(item, type):
                if item not in GROUPS:
                    raise TypeError(
                        f"{name}: supports= names {item.__name__}, which "
                        f"is not a vocabulary group; expected one of "
                        f"{[g.__name__ for g in GROUPS]}."
                    )
                names.update(GROUPS[item])
            else:
                names.add(item)
        supported = frozenset(names)
    unknown = supported - frozenset(_VOCABULARY)
    if unknown:
        raise TypeError(
            f"{name}: supports= names {sorted(unknown)}, which are not "
            f"vocabulary fields; expected some of {sorted(_VOCABULARY)}."
        )
    annotations = own_annotations(namespace)
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
    # On Python 3.14+ the body's annotations come as a lazy annotate
    # function; it is replaced by the plain dict, as up to 3.13.
    for key in ("__annotate__", "__annotate_func__"):
        namespace.pop(key, None)
    namespace["__annotations__"] = annotations
    return supported


def _metadata_repr(self: tx.Any) -> str:
    parts = []
    own = [
        field
        for field in fields(type(self))
        if field.name not in _VOCABULARY or field.name == "extra"
    ]
    # `format` and a format's own fields first, then `extra`, then the
    # vocabulary in its declared order (not `bagof`'s reverse MRO).
    own.sort(key=lambda field: field.name == "extra")
    names = [field.name for field in own if field.repr and not field.var]
    for name in names + [n for n in VOCABULARY if n in _VOCABULARY]:
        value = getattr(self, name, None)
        if value is None or value is UNSUPPORTED:
            continue
        if name == "extra" and not value:
            continue
        parts.append(f"{name.lstrip('_')}={value!r}")
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
#   METADATA
# ----------------------------------------------------------------------


class Metadata(
    DataModelBase,
    ProvenanceMetadata,
    MRIMetadata,
    DiffusionMetadata,
    DisplayMetadata,
    MicroscopyMetadata,
    TransformMetadata,
    metaclass=_MetadataMeta,
    polymorphic=True,
    kw_only=True,
):
    """
    Format-agnostic metadata: the common vocabulary, and `extra`.

    This is what in-memory images and transformations carry, the hub
    through which formats convert (`NiftiMetadata -> Metadata ->
    MghMetadata` loses exactly what `NiftiMetadata -> MghMetadata`
    loses), and what the BIDS sidecar codec reads and writes. Every
    field is supported, and there is no raw record.

    It is also the root of the metadata classes, selected on `format`:
    `Metadata(format="nifti", ...)` builds a `NiftiMetadata` (once
    `brainhops.io` is imported), and an unknown format builds a
    `Metadata`. The vocabulary is declared by six groups (mixins), which
    `Metadata` inherits: [`ProvenanceMetadata`][],
    [`MRIMetadata`][], [`DiffusionMetadata`][], [`DisplayMetadata`][],
    [`MicroscopyMetadata`][] and [`TransformMetadata`][]. Each field holds
    a value, `None` (unknown) or `UNSUPPORTED` (a format has no slot for
    it); see the module documentation for the formats' hooks.
    """

    # --- class attributes ---------------------------------------------

    supported_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc(
            "The vocabulary fields (and `'extra'`) this class can store, "
            "from its `supports=` declaration."
        ),
    ] = frozenset()

    unsupported_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc(
            "The vocabulary fields this class cannot store: the "
            "complement of `supported_fields`."
        ),
    ] = frozenset()

    derived_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc(
            "The vocabulary fields that this format derives from the "
            "geometry of the data model (read-only on write)."
        ),
    ] = frozenset()

    lazy_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc("The vocabulary fields decoded on first access."),
    ] = frozenset()

    # --- format and extras --------------------------------------------

    format: tx.Annotated[
        str,
        tx.Doc("The format this metadata belongs to; selects the subclass."),
    ] = "generic"

    extra: tx.Annotated[
        Maybe[tx.Dict[str, tx.Any]],
        tx.Doc(
            "Free-form keys the vocabulary does not cover, copied into "
            "any free-form store a format has."
        ),
        Scope(FILE),
        Factory(dict),
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

    def copy(self) -> tx.Self:
        """
        A copy of this metadata, which can be edited without editing this
        object (`extra` is copied too). This is what an image or a
        transformation holds when it is given metadata that another
        object holds already (`replace()`, `from_other`, `metadata=`).
        """
        new = copy.copy(self)
        extra = self.__dict__.get("extra")
        if isinstance(extra, dict):
            new.__dict__["extra"] = dict(extra)
        return new

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
            return name in self_or_cls.supported_fields
        return getattr(self_or_cls, name) is not UNSUPPORTED

    # --- conversion ---------------------------------------------------

    def to(
        self,
        cls: tx.Union[None, str, tx.Type["Metadata"]] = None,
        *,
        on_loss: tx.Optional[LossPolicy] = None,
        report: tx.Optional[ConversionReport] = None,
        **values: tx.Any,
    ) -> "Metadata":
        """
        Convert this metadata into another class, as images and
        transformations convert with `to()`.

        Parameters
        ----------
        cls : type or str, optional
            The `Metadata` subclass to convert to, or its format name
            (`"generic"`, `"nifti"`, ...). `None` keeps the class: a copy
            (a same-format copy shares the raw record).
        on_loss : {"ignore", "warn", "raise"}, optional
            What to do if anything is lost. Defaults to the policy in
            effect (see [`metadata_loss_policy`][]), unless `report` is
            given.
        report : ConversionReport, optional
            Filled with what was lost or approximated. When it is given
            (and `on_loss` is not), the caller acts on it, and no loss
            policy is applied.
        **values
            Fields to set on the result.

        Returns
        -------
        Metadata
            The converted metadata.
        """
        target = type(self) if cls is None else _metadata_class(cls)
        obj, found = target._convert_from(self, (), values)
        if report is not None:
            report.source = report.source or found.source
            report.target = report.target or found.target
            report.merge(found)
        if report is None or on_loss is not None:
            apply_loss_policy(found, on_loss, stacklevel=2)
        return obj

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Convert the metadata of another format into this one.

        The raw record and the snapshot are never copied across formats
        (they are when `other` is already of this class). Each
        vocabulary value is copied, except where this format declares the
        field unsupported: the value is then lost, and reported.
        `UNSUPPORTED` on the source side reads as `None`. The report is
        handed to the loss policy in effect (see
        [`metadata_loss_policy`][]); use
        [`to`][brainhops.datamodel.metadata.Metadata.to]`(cls,
        report=...)` to get it back.
        """
        if not isinstance(other, Metadata):
            return super().from_instance(other, *args, **kwargs)
        obj, report = cls._convert_from(other, args, kwargs)
        apply_loss_policy(report, stacklevel=3)
        return obj

    @classmethod
    def _convert_from(
        cls,
        other: "Metadata",
        args: tx.Tuple[tx.Any, ...] = (),
        kwargs: tx.Optional[tx.Dict[str, tx.Any]] = None,
    ) -> tx.Tuple["Metadata", ConversionReport]:
        """`from_instance`, returning the report instead of acting on it."""
        kwargs = dict(kwargs or {})
        same = _fits(other, cls)
        # A copy keeps the most specific class.
        target = type(other) if same else cls
        report = ConversionReport(
            source=_format_name(other), target=_format_name(target)
        )
        values: tx.Dict[str, tx.Any] = {}
        if same and isinstance(other, FileBasedMetadata):
            values["raw"] = other.raw
            values["decoded"] = _copy_snapshot(other._decoded)
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

    @classmethod
    def _import(
        cls,
        other: "Metadata",
        values: tx.Dict[str, tx.Any],
        *,
        report: ConversionReport,
    ) -> None:
        """Recover losses of a conversion. Default: nothing."""

    # --- propagation --------------------------------------------------

    def derive(
        self,
        *,
        grid_changed: bool = False,
        grid_map: tx.Any = None,
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
        - `grid` fields are cleared when `grid_changed`, with one
          exception: given `grid_map`, the linear part of the map from
          the old voxel axes to the new ones, an encoding direction in
          voxel axes is mapped through it (`normalize(grid_map @ v)`)
          instead. A direction in a named world space is kept.
        - `volume` fields are indexed by `volumes` (the selected volume
          indices) when given, and cleared when `volumes_changed` and no
          selection is known.
        - `extra` is kept verbatim.
        """
        values = self._derive_values(
            grid_changed=grid_changed,
            grid_map=grid_map,
            volumes=volumes,
            volumes_changed=volumes_changed,
            step=step,
        )
        return type(self)(**values)

    def _derive_values(
        self,
        *,
        grid_changed: bool,
        grid_map: tx.Any,
        volumes: tx.Optional[tx.Sequence[int]],
        volumes_changed: bool,
        step: tx.Optional[str],
    ) -> tx.Dict[str, tx.Any]:
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
                value = _map_direction(value, grid_map)
            elif scope == VOLUME and value is not None:
                value = _select_volumes(name, value, volumes, volumes_changed)
            values[name] = value
        return values

    # --- BIDS ---------------------------------------------------------

    @classmethod
    def from_bids(cls, sidecar: tx.Any) -> "Metadata":
        """
        Read a BIDS JSON sidecar: a mapping, a JSON string, or a path.

        Keys that name a vocabulary field (through its BIDS key) fill
        that field; every other key lands in `extra`. The result is
        generic `Metadata`.
        """
        from brainhops.io.metadata.bids import from_bids

        return from_bids(sidecar)

    def to_bids(
        self, *, on_loss: tx.Optional[LossPolicy] = None
    ) -> tx.Dict[str, tx.Any]:
        """
        The BIDS JSON sidecar (a JSON-serialisable `dict`) of this
        metadata. The diffusion fields are not sidecar keys, and are
        reported as lost, as is an encoding direction BIDS cannot write
        (one that is not along a voxel axis).
        """
        from brainhops.io.metadata.bids import to_bids

        return to_bids(self, on_loss=on_loss)


# The vocabulary, read off the class that declares it.
for _name in ("extra",) + VOCABULARY:
    _field = next(f for f in fields(Metadata) if f.name == _name)
    _hint = next(
        own_annotations(c)[_name]
        for c in Metadata.__mro__
        if _name in own_annotations(c)
    )
    _VOCABULARY[_name] = (
        _hint,
        _FACTORY if _name == "extra" else _field.default,
    )
del _name, _field, _hint
_set_capabilities(Metadata)


class FileBasedMetadata(Metadata):
    """
    The metadata of a file format: the common vocabulary, plus the
    format's own raw record and the read-time snapshot.

    This is the base of every `<Fmt>Metadata` (as `FileBasedImage` is of
    every format's image class): a format declares what it can store
    with `supports=`, and decodes and encodes its raw record with the
    hooks described in the module documentation.
    """

    raw: tx.Annotated[
        tx.Any,
        tx.Doc(
            """
            The format's own raw record (a `nibabel` header, ...). Edit
            it only for what the vocabulary does not cover: an untouched
            common field keeps the record's value on write, a common
            field you set wins over it. Never copied across formats.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    _decoded: tx.Annotated[
        tx.Optional[Metadata],
        tx.Doc(
            """
            The read-time snapshot: what the reader decoded from `raw`,
            as generic `Metadata` (`None` for an object built in memory).
            A common field is written over the raw record only when it
            differs from it. Filled by the reader; never set by hand.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    def __getstate__(self) -> tx.Dict[str, tx.Any]:
        # `_source` (what a format whose raw record is rebuilt on each
        # read, Zarr, read it from: a store node) is a handle, not state,
        # and is not pickled nor deep-copied; `copy()` keeps it.
        state = dict(self.__dict__)
        state.pop("_source", None)
        return state

    # --- reading ------------------------------------------------------

    @classmethod
    def from_raw(
        cls, raw: tx.Any, *, image: tx.Any = None, **values: tx.Any
    ) -> tx.Self:
        """
        Build the metadata of a raw record that was just read.

        The raw record is decoded into the common fields (`_decode`), and
        what was decoded is kept as the read-time snapshot, so that an
        untouched field keeps the record's value when it is written back.
        A lazy field (`lazy=`) decoded as
        [`Lazy`][brainhops._core.properties.Lazy] is decoded on first
        access instead. Keyword arguments set fields over the decoded
        values (they then count as changes).

        Raises
        ------
        TypeError
            If `_decode` returned a value for a field this class does not
            support, or a `Lazy` for a field not declared `lazy=`. The
            raw record is the format's own, so this is never about the
            data: it is a bug of the format class, whose `_decode` and
            `supports=` (or `lazy=`) disagree, and which would otherwise
            drop the value without a report.
        """
        decoded: tx.Dict[str, tx.Any] = {}
        pending: tx.Dict[str, Lazy] = {}
        for key, value in cls._decode(raw, image=image).items():
            if value is None or value is UNSUPPORTED:
                continue
            if key not in cls.supported_fields:
                raise TypeError(
                    f"{cls.__name__}._decode returned {key}={value!r}, "
                    f"but {cls.__name__} does not support {key!r}."
                )
            if isinstance(value, Lazy):
                if key not in cls.lazy_fields:
                    raise TypeError(
                        f"{cls.__name__}._decode returned a Lazy {key!r}, "
                        f"which is not one of its lazy= fields."
                    )
                pending[key] = value
            else:
                decoded[key] = value
        obj = cls(raw=raw, **decoded)
        # Snapshot the *converted* values, so that a decoded list held as
        # a tuple does not count as a change.
        obj._decoded = Metadata(
            **{key: copy.deepcopy(getattr(obj, key)) for key in decoded}
        )
        for key, value in pending.items():
            # Injected as is (no conversion): the first read loads it.
            obj.__dict__[key] = value
        for key, value in values.items():
            setattr(obj, key, value)
        return obj

    def copy(self) -> tx.Self:
        """
        A copy of this metadata, sharing its raw record.

        The raw record (`raw`) is shared, as `replace()` shares it; the
        snapshot and `extra` are copied, so editing the copy never edits
        this object. A field still waiting to be decoded (see `lazy=`)
        stays so in both.
        """
        new = super().copy()
        if "_source" in self.__dict__:
            new.__dict__["_source"] = self.__dict__["_source"]
        new.__dict__["_decoded"] = _copy_snapshot(self._decoded)
        return new

    def update_from_raw(self, raw: tx.Any, *, image: tx.Any = None) -> tx.Self:
        """
        The metadata of another raw record, keeping the changes made here.

        `raw` is decoded (`from_raw`), and every field that changed since
        this object was read (`changed_fields()`: all the fields that
        are set, for an object built in memory) is set over the decoded
        values, as a change. `extra` is merged key by key. This is what
        an object given a new raw record holds (`replace(image,
        header=...)`), and what `metadata=` given along with a raw record
        becomes.
        """
        changed = {
            key: value
            for key, value in self.changed_fields().items()
            if key in type(self).supported_fields
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

    # --- the change-detecting write -----------------------------------

    def changed_fields(self) -> tx.Dict[str, tx.Any]:
        """
        The common fields that differ from the read-time snapshot.

        An object read from a file and left untouched has none. An object
        built in memory has no snapshot, so every field that is not
        `None` is a change. A field set to `None` after a read is a change
        to `None` (the record's slot is cleared on write). `extra` is
        compared key by key: its entry is the per-key diff, where `None`
        removes a key.
        """
        snapshot = self._decoded
        changed: tx.Dict[str, tx.Any] = {}
        for name in _VOCABULARY:
            value = getattr(self, name, None)
            if value is UNSUPPORTED:
                continue
            before = getattr(snapshot, name, None)
            if name == "extra":
                diff = _extra_diff(before, value)
                if diff:
                    changed["extra"] = diff
            elif _differs(value, before):
                changed[name] = value
        return changed

    def update_raw(
        self,
        raw: tx.Any = None,
        *,
        image: tx.Any = None,
        report: tx.Optional[ConversionReport] = None,
        force: tx.Collection[str] = (),
    ) -> tx.Any:
        """
        Encode the common fields over a raw record, and return it.

        `raw` is the record to write over: a writer passes its own fresh
        record, already filled with what it keeps from `self.raw`. When
        it is `None`, a copy of `self.raw` (or a default record) is used.
        Only the fields that changed since the read are encoded (see
        [`changed_fields`][brainhops.datamodel.metadata.FileBasedMetadata.changed_fields]),
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
        the encoder on a scratch raw record (the one the format's writer
        would start from, see `_check_raw`), so value-dependent losses
        (an over-long description, an irregular slice timing) are
        included. Pass the image (or transformation) for the fields that
        need it.
        """
        report = ConversionReport(source=self.format, target=self.format)
        self.update_raw(self._check_raw(image), image=image, report=report)
        return report

    def _raw_or_default(self) -> tx.Any:
        if self.raw is not None:
            return copy.deepcopy(self.raw)
        return self._default_raw()

    # --- per-format hooks ---------------------------------------------

    @classmethod
    def _default_raw(cls) -> tx.Any:
        """A fresh, empty raw record for this format."""
        return None

    @classmethod
    def _decode(cls, raw: tx.Any, *, image: tx.Any = None) -> tx.Dict:
        """Raw record to common fields. Default: nothing to decode."""
        return {}

    def _encode(
        self,
        raw: tx.Any,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> tx.Any:
        """Common fields to raw record. Default: the record is unchanged."""
        return raw

    def _geometry(self, image: tx.Any) -> tx.Dict[str, tx.Any]:
        """
        The values the data model gives for the derived fields (NIfTI:
        the time step of the image as `repetition_time`). A field left
        out, or `None`, is one the data model says nothing about: its
        changed value is then left to `_encode`. Default: nothing.
        """
        return {}

    def _check_raw(self, image: tx.Any) -> tx.Any:
        """
        The raw record `check_writable` encodes over: what the writer of
        this format would pass to `update_raw` for `image`. Default: a
        copy of the record, or a default one.
        """
        return self._raw_or_default()

    def _derive_raw(
        self,
        raw: tx.Any,
        *,
        grid_changed: bool,
        volumes: tx.Optional[tx.Sequence[int]],
    ) -> tx.Any:
        """Scrub grid- or volume-bound raw content. Default: none."""
        return raw

    # --- propagation --------------------------------------------------

    def derive(
        self,
        *,
        grid_changed: bool = False,
        grid_map: tx.Any = None,
        volumes: tx.Optional[tx.Sequence[int]] = None,
        volumes_changed: bool = False,
        step: tx.Optional[str] = None,
    ) -> tx.Self:
        """
        As [`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive];
        in addition, the raw record and the snapshot are kept, so that a
        cleared field is cleared in the raw record on write, and
        `_derive_raw` scrubs what the vocabulary does not cover.
        """
        values = self._derive_values(
            grid_changed=grid_changed,
            grid_map=grid_map,
            volumes=volumes,
            volumes_changed=volumes_changed,
            step=step,
        )
        values["raw"] = self._derive_raw(
            self.raw, grid_changed=grid_changed, volumes=volumes
        )
        values["decoded"] = _copy_snapshot(self._decoded)
        return type(self)(**values)


class OpaqueMetadata(FileBasedMetadata, on={"format": "opaque"}, supports=()):
    """
    The metadata of a format that stores none, not even in memory: every
    field is unsupported, and there is no raw record.

    Base class of the formats that store a bare matrix and are given
    nothing else (matrix text, ITK `.tfm`/`.mat`). A read-then-save
    loses nothing, because nothing is there. A format that keeps
    anything, if only in memory (FLIRT `moving`/`fixed`), is a
    `FileBasedMetadata` with a `supports=` list instead.
    """

    format: tx.Annotated[
        tx.Literal["opaque"], tx.Doc("Always `'opaque'`.")
    ] = "opaque"

    raw: tx.Annotated[
        None, tx.Doc("Always `None`: no raw record."), NoRepr(), NoEq()
    ] = None


def _copy_snapshot(snapshot: tx.Optional[Metadata]) -> tx.Optional[Metadata]:
    return None if snapshot is None else snapshot.copy()


def _fits(value: tx.Any, cls: type) -> bool:
    """Whether a metadata object is one of `cls` already: of the class
    itself, or of a subclass of a format class (generic `Metadata` holds
    generic metadata only)."""
    if type(value) is cls:
        return True
    return cls is not Metadata and isinstance(value, cls)


def _field_metadata(name: str) -> tx.Dict[str, tx.Any]:
    for field in fields(Metadata):
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


def _map_direction(value: tx.Any, grid_map: tx.Any) -> tx.Any:
    """A `grid` field after a grid change: an encoding direction in voxel
    axes goes through `grid_map` when it is given (and fits), one in a
    world space is kept; anything else is cleared."""
    if not isinstance(value, EncodingDirection):
        return None
    if value.space is not None:
        return value
    if grid_map is None:
        return None
    matrix = np.asarray(grid_map, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(value.vector):
        return None
    try:
        return value.transform(matrix)
    except ValueError:
        return None


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


def _metadata_class(target: tx.Any) -> tx.Type[Metadata]:
    if isinstance(target, type) and issubclass(target, Metadata):
        return target
    if isinstance(target, str):
        if target == "generic":
            return Metadata
        stack = list(Metadata.__subclasses__())
        while stack:
            klass = stack.pop()
            if _format_name(klass) == target and klass is not (
                FileBasedMetadata
            ):
                return klass
            stack.extend(klass.__subclasses__())
        raise ValueError(f"No metadata class for the format {target!r}.")
    raise TypeError(
        f"Expected a Metadata subclass or a format name, got {target!r}."
    )


# ----------------------------------------------------------------------
#   THE `metadata` FIELD
# ----------------------------------------------------------------------


def _target_class(hint: tx.Any) -> tx.Optional[type]:
    """The `Metadata` class of a field hint (`Optional[X]` gives `X`)."""
    if isinstance(hint, type) and issubclass(hint, Metadata):
        return hint
    for arg in tx.get_args(hint):
        found = _target_class(arg)
        if found is not None:
            return found
    return None


class _EnsureCopy:
    """
    The converter of a `metadata` field: converts what it is given into
    the field's class, as `bagof` would (and a metadata object of another
    class, even a subclass of a generic field's `Metadata`, by
    conversion, which reports the loss), and copies a metadata object
    that already is of that class (`Metadata.copy`), so that two images
    or transformations never hold the same one.
    """

    def __init__(self, hint: tx.Any) -> None:
        self.hint = hint
        self._convert: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None
        self._target: tx.Optional[type] = None

    def __call__(self, value: tx.Any) -> tx.Any:
        if self._convert is None:
            self._convert = Converter.get(self.hint)
            self._target = _target_class(self.hint)
        target = self._target
        if (
            isinstance(value, Metadata)
            and target is not None
            and not _fits(value, target)
        ):
            return target.from_instance(value)
        out = self._convert(value)
        if out is value and isinstance(out, Metadata):
            out = out.copy()
        return out


class MetadataField:
    """
    The annotation of a `metadata` field:
    `MetadataField[hint, *annotations]` is
    `Annotated[hint, ConvertTo(...), KwOnly(), NoRepr(), NoEq(),
    *annotations]`.

    The field is keyword-only, out of `repr` and `==`, converted on
    assignment (even on a class that does not convert its fields) and
    copied rather than shared. `hint` is a `Metadata` class, or
    `Optional` of one; the annotations add the rest:

    ```python
    metadata: MetadataField[tx.Optional[Metadata], tx.Doc("...")] = None
    metadata: MetadataField[
        NiftiMetadata, Factory(NiftiMetadata), tx.Doc("...")
    ]
    ```

    A narrowed field must be declared on the class itself or on its
    first base (see the module documentation).
    """

    def __class_getitem__(cls, params: tx.Any) -> tx.Any:
        if not isinstance(params, tuple):
            params = (params,)
        hint, *extras = params
        return tx.Annotated[
            (
                hint,
                ConvertTo(_EnsureCopy(hint)),
                KwOnly(),
                NoRepr(),
                NoEq(),
                *extras,
            )
        ]


# ----------------------------------------------------------------------
#   WRITERS
# ----------------------------------------------------------------------


def _same_kind(source: np.dtype, target: np.dtype) -> bool:
    """Whether values of `source` may be stored as `target` without
    changing their kind: integers (booleans included) as integers,
    floats as floats, complex numbers as complex numbers."""

    def kind(dtype: np.dtype) -> str:
        return "i" if dtype.kind in "biu" else dtype.kind

    return kind(source) == kind(target) and target.kind != "b"


def preferred_dtype(
    metadata: tx.Any,
    array_dtype: tx.Any,
    dtype: tx.Any = None,
    *,
    report: tx.Optional[ConversionReport] = None,
) -> np.dtype:
    """
    The element type a writer stores an array as.

    In order: an explicit `dtype` (the writer option); the `data_type` of
    the metadata (the type the file had when it was read, or the one
    set), when the array's values are of its kind, so that a label map
    read as `uint8` is written as `uint8` again but a resampled, floating
    point version of it is not quantised; the array's own type. A
    `data_type` set (or converted) by hand that is not used is reported
    in `report` as approximated; one that was only read is dropped
    silently (the data changed kind since the read).
    """
    array_dtype = np.dtype(array_dtype)
    if dtype is not None:
        return np.dtype(dtype)
    wanted = getattr(metadata, "data_type", None)
    if wanted is None or wanted is UNSUPPORTED:
        return array_dtype
    wanted = np.dtype(wanted)
    if _same_kind(array_dtype, wanted):
        return wanted
    if report is not None:
        changed = (
            metadata.changed_fields()
            if isinstance(metadata, FileBasedMetadata)
            else {"data_type": wanted}
        )
        if "data_type" in changed:
            report.approximated["data_type"] = (
                f"stored as {array_dtype.name}: the values are not "
                f"{wanted.name} values"
            )
    return array_dtype
