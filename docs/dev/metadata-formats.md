# Writing the metadata of a format

This page is for whoever adds a file format to brainhops, or the
metadata of one that has none yet. What the metadata *is*, and how a user
reads, edits and converts it, is in the user guide
([Metadata](../start/metadata.md)); why it is built this way is in the
design memo (`docs/design/format-metadata.md`).

A format's metadata is one class, `<Fmt>Metadata`, a subclass of
[`FileBasedMetadata`][brainhops.datamodel.metadata.FileBasedMetadata]
that lives next to the format's parser. It says which vocabulary fields
the format can store, and how to decode them from the format's own
**raw record** (a `nibabel` header, a dict of attributes, ...) and encode
them back. The framework does the rest: the read-time snapshot, change
detection, conversion between formats, loss reports, propagation
(`derive`, and the image operations), the `metadata` field of images
and transformations.

`brainhops/io/base/_nifti_metadata.py` is the worked example. A format's
metadata module reads in one order: its docstring (what each field is
stored as, and what is lossy), its constants, the raw-record type when
there is one (`MghRaw`, `ZarrRaw`: it must exist before the annotation
of `raw` names it), the metadata class with its hooks in the order of
the table below, its private codec helpers (decode side, then encode
side), and last the helpers its image classes import.

## Where the names live

The package `brainhops.datamodel.metadata` exports what a user needs
(`Metadata`, `FileBasedMetadata`, `UNSUPPORTED`, `Scope`, the value
classes, the report and the loss policy). A format author imports the
rest from the private modules that define it:

| Module | Names |
|---|---|
| `brainhops.datamodel.metadata._vocabulary` | the groups (`ProvenanceVocabulary`, ..., `StorageVocabulary`, the base `Vocabulary`), the annotations `Bids`, `Scoped` and `Along`, and the tables `VOCABULARY`, `GROUPS`, `BIDS_KEYS`, `SCOPES`, `ALONG` |
| `brainhops.datamodel.metadata._filebased` | `OpaqueMetadata` |
| `brainhops.datamodel.metadata._field` | `MetadataField` |
| `brainhops.datamodel.metadata._report` | `apply_loss_policy`, `collect_loss_reports`, `OnLoss`, `LossPolicy` |
| `brainhops.datamodel.metadata._dtype` | `preferred_dtype`, `preferred_storage` |
| `brainhops.datamodel.metadata._sentinel` | `ALL`, `Maybe`, `Unsupported` |
| `brainhops._core.fields` | `Lazy`, `LazyField` |
| `brainhops.io.base._metadata_parser` | `MetadataParser`, `Hdf5MetadataParser` |
| `brainhops.io.metadata._json` | the JSON codec of key/value stores |
| `brainhops.io.metadata._sync` | `sync_metadata` |

## The class

```python
@register_format
class MyMetadata(
    MetadataParser,  # reads the raw record of a file (see below)
    FileBasedMetadata,
    on={"format": "my"},  # polymorphic discriminant
    supports=(
        ProvenanceVocabulary,
        "echo_time",
    ),  # everything else UNSUPPORTED
    lazy=("history",),  # fields decoded on first access
):
    raw: tx.Annotated[tx.Optional[MyRaw], NoRepr(), NoEq()] = None

    EXTENSIONS = (".my",)
    HINTS = ("my",)

    @classmethod
    def sniff_fileobj(cls, file, error=False, **kwargs) -> float: ...

    @classmethod
    def _read_raw(cls, file, **kwargs) -> MyRaw: ...

    def _write_raw(self, raw, file) -> None: ...  # a record of its own

    @classmethod
    def _default_raw(cls) -> MyRaw: ...

    @classmethod
    def _decode(cls, raw, *, image=None) -> dict: ...

    def _encode(self, raw, changed, *, image=None, report) -> MyRaw: ...

    def _geometry(self, image) -> dict: ...

    def _check_raw(self, image) -> MyRaw: ...

    def _derive_raw(self, raw, *, changed) -> MyRaw: ...

    @classmethod
    def _import(
        cls, other, values, *, report
    ) -> None: ...  # key/value formats
```

`format` is the discriminant of the polymorphic root, and `on=` declares
it: `bagof` narrows the field to the literal `"my"`, with that default,
so the class does not declare `format` itself. `Metadata(format="my",
...)` builds a `MyMetadata`, and `"my"` is the name `metadata.to("my")`
takes. The one exception is a format that subclasses another format
(`ItkMetadata` under `OpaqueMetadata`): its field is narrowed to the
value of its parent, so it declares its own.

`raw` declares the type of the raw record, and stays out of `repr` and
`==`. The type matters beyond documentation. `raw` and the read-time
snapshot are fields of `Metadata`, so generic metadata carries the record
of the metadata it was converted from, and a conversion gives the record
back to a format only when the record is an instance of the type that the
format declares (`FileBasedMetadata._accepts_raw`). Every format must
therefore declare a type of its own: wrap a plain `dict` or `tuple` in a
small class (`ZarrRaw`, `X5Raw`), never share a type with another format,
and do not declare a subclass of the type of another format.
`tests/test_io_metadata_matrix.py` checks it. A format with no record
declares `None`. A read alias under the familiar name of the record
(`header`, `tags`, `node`) is a plain property over `raw`.

The parser comes before `FileBasedMetadata` among the bases, so that
its `load` wins by the MRO: a format without a parser inherits
`FileBasedMetadata.load`, which refuses. The format registers itself
into the registry of `MetadataParser` with
`@register_format`, which is what `Metadata.load(path)` dispatches over:
its `EXTENSIONS`, `HINTS` and sniffers work as those of an image parser.
`_read_raw(file)` reads the raw record from a path or an open binary
file, and nothing else (a NIfTI header, never the voxels); the parser
then builds the metadata with `from_raw`. A format stored in HDF5 derives
from `Hdf5MetadataParser` instead, and implements `_sniff_h5(h5file)` and
`_read_raw_h5(h5file)`. A format whose record is an object of its own on
disk (the attributes of a Zarr array) implements `_write_raw(raw, file)`,
and `metadata.to_file(path)` writes `to_raw()` there; every other format
refuses, since its record is written along with the data. The reader of
the image or transformation shares the code that reads the record with
`_read_raw` (`_load_nifti_header`, `read_mgh_raw`, `read_h5_header`). A
format whose files hold no metadata (FLIRT, ITK `.tfm`) is not a
`MetadataParser`.

## Class keywords

- `supports=`: the vocabulary fields (and `"extra"`) the format can
  store, as names, group classes (all the fields of the group) or
  [`ALL`][brainhops.datamodel.metadata._sentinel.ALL]. Omitted, a subclass keeps
  its parent's capabilities. Every other field defaults to
  [`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED] and is
  refused at construction. `supported_fields` lists what the class
  stores, `unsupported_fields` the rest. A new vocabulary field is
  unsupported by every format until one opts in.
- `lazy=`: supported fields whose decoding would read a lazy part of the
  raw record (the MGH tags, after the whole compressed volume). Each
  gets a [`LazyField`][brainhops._core.fields.LazyField]
  descriptor, and `_decode` may return
  [`Lazy`][brainhops._core.fields.Lazy]`(load)` for it: the field is
  decoded on first access (or assignment), and joins the snapshot then.
  Any other attribute access is plain.

## Hooks

All the hooks are optional, and all private.

| Hook | Called by | Gives |
|---|---|---|
| `_default_raw()` | an object built in memory | a fresh raw record |
| `_decode(raw, *, image)` | `from_raw` | the vocabulary values of a raw record |
| `_encode(raw, changed, *, image, report)` | `update_raw` | the raw record to write |
| `_geometry(image)` | `update_raw` | the fields the data model owns, and their values |
| `_check_raw(image)` | `check_writable` | the raw record a writer starts from |
| `_derive_raw(raw, *, changed)` | `derive`, `_select`, `_reslice` | a scrubbed copy of the raw record |
| `_import(other, values, *, report)` | `to`, `from_other` | recovered losses (key/value formats) |

- `_default_raw() -> raw`: a fresh, empty raw record, for an object
  built in memory (or converted from another format). Defaults to
  `None`.
- `_decode(raw, *, image=None) -> dict`: raw record to vocabulary, on
  read. Returns field names (and `"extra"`) to values; `None` values may
  be left out. `image` is the image or transformation the raw record
  belongs to, for fields that need it. A value for a field the class
  does not support (or a `Lazy` for a field not in `lazy=`) is a bug of
  the format class, and `from_raw` raises `TypeError` rather than drop
  it unreported.
- `_encode(raw, changed, *, image=None, report) -> raw`: vocabulary to
  raw record, on write. `raw` is the record to write over (already a
  copy, or the writer's own fresh record) and `changed` holds only the
  fields that differ from the read-time snapshot, so an untouched field
  keeps the record's value. A `None` in `changed` *clears* the slot.
  `changed["extra"]` is a per-key diff whose `None` values remove a key.
  Value-dependent loss goes in `report` (`report.lost[name] = value`,
  `report.approximated[name] = reason`). Returns the raw record to write.
- `_geometry(image) -> dict`: the fields that are, for this format, a
  view of geometry the data model owns, and the values it gives for them
  (NIfTI: `{"repetition_time": <time step>}`, `pixdim[4]`). This is the
  single source of truth of the *derived* fields: they are decoded on
  read like any other, but on write the data model's value is what the
  writer stores. A changed field that disagrees with it is reported
  under `report.approximated` and never reaches `_encode`, one that
  agrees is dropped silently, and only a field the data model says
  nothing about (left out, or `None`) is left to `_encode`. Defaults to
  `{}`.
- `_check_raw(image) -> raw`: the raw record
  [`check_writable`][brainhops.datamodel.metadata.FileBasedMetadata.check_writable]
  encodes over: what the writer would pass to `update_raw` (NIfTI: the
  record, reshaped to the data of `image`), so that a value-dependent
  check reads the same state as a real write. Defaults to a copy of the
  record, or a default one.
- `_derive_raw(raw, *, changed) -> raw`: called by
  [`derive`][brainhops.datamodel.metadata.Metadata.derive] and by the
  hooks the image operations call (`_select(axis, positions)` for
  `image[index]` along a time or channel axis, `_reslice(linear)` for a
  change of the spatial axes) for the raw record of the derived object.
  `changed` maps the `AxisType` of each changed axis to what changed it:
  the kept positions for `_select`, the linear map of the voxel axes (or
  `None`) for `_reslice`; it is empty for `derive`. A format only tests
  which types are in it. The default is a deep copy of `raw`, so that a
  derived object never shares its record. A format whose record holds
  content tied to some axes, but outside the vocabulary, removes it from
  the copy: NIfTI clears its slice slots and `dim_info` when
  `AxisType.space` is in `changed`. The hook never modifies `raw` in
  place.
- `_import(other, values, *, report) -> None`: a hook for the key/value
  formats (MRtrix, NRRD), called on a conversion with the source object,
  the values about to be passed to the constructor, and the report. A
  format may *recover* a loss here, for example by moving a lost
  vocabulary value into `values["extra"]`, removing it from
  `report.lost` and listing it in `report.passed_through`. Defaults to
  nothing; no format in the package needs it yet. Three uses:
    - *MRtrix keeps BIDS-named keys.* An MGH image converted to MRtrix
      has `echo_time` and `flip_angle`, for which `.mif` has no
      dedicated key (it has `PhaseEncodingDirection`,
      `TotalReadoutTime`, `dw_scheme`, ...), so `MrtrixMetadata`
      declares them unsupported. But `keyval` holds any `key: value`:
      `_import` moves them into `values["extra"]` as `EchoTime` and
      `FlipAngle` (what `mrconvert -json_import` writes), and the report
      lists them as passed through instead of lost.
    - *NRRD `keyvalue`.* NRRD has no field for the slice timing or the
      phase encoding direction of a NIfTI image; `_import` writes them
      as `key:=value` pairs (`SliceTiming:=0 0.5 1 1.5`), so a NIfTI ->
      NRRD -> NIfTI round trip keeps them.
    - *A field out of a lost `extra`.* A format with dedicated slots and
      no free-form store (MGH) loses the `extra` of an MRtrix source as
      a whole; `_import` can take `extra["EchoTime"]` back into
      `values["echo_time"]` before the rest of `extra` is reported lost.

  The MRtrix version is six lines:

  ```python
  @classmethod
  def _import(cls, other, values, *, report):
      extra = dict(values.get("extra") or {})
      for name in [n for n in report.lost if n != "extra"]:
          extra[BIDS_KEYS[name]] = report.lost.pop(name)
          report.passed_through += (name,)
      values["extra"] = extra
  ```

## Scopes and axes

Every vocabulary field declares how it propagates to a derived image,
with an annotation: `Scoped(Scope.FILE)`, `Scoped(Scope.ACQUISITION)`,
`Scoped(Scope.SPATIAL)`, or, for a field with one entry per index along
a non-spatial axis, `Along(AxisType.time)` or `Along(AxisType.channel)`
(the `AXIS` scope). These live in
`brainhops.datamodel.metadata._vocabulary`, with the groups.

A writer of an arbitrary image maps the axes of its coordinate system to
the slots of the format by type, not by name: NIfTI stores the time axis
as its fourth dimension and the channel axis as its fifth, and OME-Zarr
names them `t` and `c`. A per-axis field goes with the axis its `Along`
names. When the image has no axis of that type, or when the length of
the field differs from the length of the axis, `_encode` reports the
field as lost rather than writing entries that describe nothing.

## Reading and writing

**One word for the raw record.** Every name says "raw". A reader builds
the metadata with
[`from_raw`][brainhops.datamodel.metadata.FileBasedMetadata.from_raw],
which decodes the raw record and keeps the read-time snapshot. A parser
given a raw record and a `metadata` that is not that record's (explicit,
or carried by `replace()`) uses
[`update_from_raw`][brainhops.datamodel.metadata.FileBasedMetadata.update_from_raw],
which decodes the new raw record and keeps the changes. A parser does
both in one call from its `__post_init__`,
`sync_metadata(self, MyMetadata, raw, image=self)`
(`brainhops/io/metadata/_sync.py`): `same=` tells whether the metadata
holds the parser's record already (default: by identity), `raw` may be
a function that reads the record, called only when needed (Zarr, MGH),
and `force=True` decodes it afresh.

A writer starts from
`metadata, report = MyMetadata.writable(obj.metadata)`
([`writable`][brainhops.datamodel.metadata.FileBasedMetadata.writable]:
the metadata converted to its class when it is of another, and a report
that holds the conversion's losses, with no policy applied yet), builds
its raw record, calls
[`update_raw`][brainhops.datamodel.metadata.FileBasedMetadata.update_raw]
with it and that report, as `on_loss=report` (a report given as
`on_loss` is filled, never warned about; `force=` names a writer keyword
that must win over the record, such as MGH `tr=`), and hands the report to
[`apply_loss_policy`][brainhops.datamodel.metadata._report.apply_loss_policy]:
one write, one report, one warning.
`io.save` collects the reports of a conversion and of the write that
follows ([`collect_loss_reports`][brainhops.datamodel.metadata._report.collect_loss_reports])
and warns once, with
[`ConversionReport.merged`][brainhops.datamodel.metadata.ConversionReport.merged].

**JSON and key/value stores.** A format whose store is a JSON object or
a set of key/value pairs (x5 node `Metadata`, Zarr attributes, and the
MRtrix and NRRD headers to come) does not write its own codec:
`brainhops/io/metadata/_json.py` holds the one BIDS sidecars use.
`decode_object(obj, names)` splits an object into the values of the
vocabulary fields `names` (read from their sidecar keys, BIDS keys or
`CamelCase` names) and the other keys, which are `extra`;
`encode_changes(obj, changed)` writes the changed fields back under
their keys (`None` removes one), and `encode_extra(obj, diff, report=,
reserved=)` applies the `extra` diff, reporting the keys the format
keeps for itself as lost. `X5Metadata` is the shortest example. A
key/value format that can hold what it has no slot for pairs this with
the `_import` hook.

## The `metadata` field of a format class

`Image` and `Transformation` declare `metadata: Optional[Metadata]`
(keyword-only, out of `repr` and `==`). A format narrows it to its own
class, with a default factory, which is what makes a change of format
convert (and report). Both are written with
[`MetadataField`][brainhops.datamodel.metadata._field.MetadataField]`[hint,
*annotations]`, whose converter also converts on a class that does not
convert its fields (a plain `Magic` parser), converts a metadata object
of another class into the field's class (a `NiftiMetadata` given to a
field typed `Metadata` becomes generic `Metadata`), and *copies* a
metadata object that already is of the field's class
([`copy`][brainhops.datamodel.metadata.Metadata.copy]: the raw
record is shared, the snapshot and `extra` are not): two objects never
hold the same metadata, so editing the result of `replace()` or
`from_other` never edits the original.

!!! warning "The first base wins"
    `bagof` takes a field from the *first* base that has it, so the
    narrowed declaration must sit on the class itself or on its first
    base: a transformation format whose first base is a data model
    transformation must declare it again (see
    `NiftiBasedTransformation`).

`DataModelBase.from_other` passes the `metadata` of a data model it hands
to a constructor (an x5 chain made into a NIfTI field), so the field
converts it and reports the loss.

## Checklist

1. Write `<Fmt>Metadata` next to the parser: `on=`, `supports=`, `raw`
   with a record type of its own, then the hooks it needs.
2. When the file holds metadata, make it a `MetadataParser`
   (`@register_format`, `EXTENSIONS`, `HINTS`, a sniffer and
   `_read_raw`), so that `Metadata.load` reads it.
3. In the parser's `__post_init__`, read the metadata from the raw
   record (`sync_metadata`); in the writer, encode it (`update_raw`) and
   apply the loss policy.
4. Narrow the `metadata` field of the image or transformation class
   with `MetadataField`.
5. Add the class to the matrix test, `tests/test_io_metadata_matrix.py`:
   a format class that is not in its table fails it.
6. Describe what the format stores in the "Formats" section of the
   user guide.
