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
    "LossPolicy",
    "preferred_dtype",
    "Lazy",
    "LazyField",
]

from types import FunctionType as _FunctionType

from brainhops._core.properties import Lazy, LazyField

from ._base import Metadata
from ._base import _bids_key as _bids_key  # for the BIDS codec
from ._field import MetadataField
from ._filebased import FileBasedMetadata, OpaqueMetadata, preferred_dtype
from ._report import (
    ConversionReport,
    LossPolicy,
    MetadataLossError,
    MetadataLossWarning,
    apply_loss_policy,
    collect_loss_reports,
    get_metadata_loss_policy,
    metadata_loss_policy,
    one_loss_warning,
)
from ._sentinel import ALL, UNSUPPORTED, Maybe, Unsupported
from ._terms import Channel, EncodingDirection, GeneratedBy
from ._vocabulary import (
    ACQUISITION,
    FILE,
    GRID,
    GROUPS,
    VOCABULARY,
    VOLUME,
    Bids,
    DiffusionMetadata,
    DisplayMetadata,
    MicroscopyMetadata,
    MRIMetadata,
    ProvenanceMetadata,
    Scope,
    TransformMetadata,
)

# The public names read as members of this package (in reprs, tracebacks
# and pickles), not of the private module that defines them.
for _name in __all__:
    _obj = globals()[_name]
    if isinstance(_obj, (type, _FunctionType)) and _obj.__module__.startswith(
        __name__ + "._"
    ):
        _obj.__module__ = __name__
del _name, _obj
