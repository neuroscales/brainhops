# Writing the metadata of a format

This page is for whoever adds a file format to brainhops, or the
metadata of one that has none yet. What the metadata *is*, and how a user
reads, edits and converts it, is in the user guide
(`docs/start/metadata.md`); why it is built this way is in the
design memo (`docs/design/format-metadata.md`).

A format's metadata is one class, `<Fmt>Metadata`, a subclass of
[`FileBasedMetadata`][brainhops.io.metadata.FileBasedMetadata]
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
there is one (`MghRaw`, `ZarrRaw`: it must exist before the class names
it as the type argument of its base), the metadata class (its public
methods: the read aliases, the parser's `from_*` and sniffers, a public
override such as `check_writable`; then the two hooks), its private
codec helpers (decode side, then encode side), and last the helpers its
image classes import.

## Where the names live

The package `brainhops.datamodel.metadata` exports what a user needs
(`Metadata`, `UNSUPPORTED`, `Scope`, the value classes, the report and
the loss policy), and the vocabulary groups that `supports=` names
(`ProvenanceVocabulary`, `MRIVocabulary`, `DiffusionVocabulary`,
`DisplayVocabulary`, `StorageVocabulary`, `MicroscopyVocabulary`,
`TransformVocabulary`, and their base `Vocabulary`). The base of a
format's metadata lives in `brainhops.io`, as `FileBasedImage` does,
since it derives from the dispatcher of the formats: `from
brainhops.io.metadata import FileBasedMetadata, OpaqueMetadata`. The
data model never imports it. A format author imports
the rest from the private modules that define it:

| Module | Names |
|---|---|
| `brainhops.datamodel.metadata._vocabulary` | the annotations `Bids`, `Scoped` and `Along`, and the tables `VOCABULARY`, `GROUPS`, `BIDS_KEYS`, `SCOPES`, `ALONG` |
| `brainhops.datamodel.metadata._field` | `MetadataField` |
| `brainhops.datamodel.metadata._report` | `apply_loss_policy`, `collect_loss_reports`, `OnLoss`, `LossPolicy` |
| `brainhops.datamodel.metadata._dtype` | `preferred_dtype`, `preferred_storage` |
| `brainhops.datamodel.metadata._sentinel` | `ALL`, `Maybe`, `Unsupported` |
| `brainhops.io.base._metadata_parser` | `MetadataParser`, `Hdf5MetadataParser` |
| `brainhops.io.metadata._json` | the JSON codec of key/value stores |
| `brainhops.io.metadata._sync` | `sync_metadata` |

## The class

```python
@register_format
class MyMetadata(
    MetadataParser,  # reads the raw record of a file (see below)
    FileBasedMetadata[MyRaw],  # the type of the raw record
    on={"format": "my"},  # polymorphic discriminant
    supports=(
        ProvenanceVocabulary,
        "echo_time",
    ),  # everything else UNSUPPORTED
):
    EXTENSIONS = (".my",)
    HINTS = ("my",)

    @classmethod
    def sniff_fileobj(cls, file, error=False, **kwargs) -> float: ...

    @classmethod
    def from_fileobj(cls, file, **kwargs) -> "MyMetadata":
        return cls.from_raw(read_my_record(file))

    def to_file(self, file, **kwargs) -> None: ...  # a record of its own

    @classmethod
    def _decode_raw(cls, raw, *, image=None) -> dict: ...

    def _encode_raw(self, raw, changed, *, image=None, report) -> MyRaw: ...
```

`format` is the discriminant of the polymorphic root, and `on=` declares
it: `bagof` narrows the field to the literal `"my"`, with that default,
so the class does not declare `format` itself. `Metadata(format="my",
...)` builds a `MyMetadata`, and `"my"` is the name `metadata.to("my")`
takes. The one exception is a format that subclasses another format
(`ItkMetadata` under `OpaqueMetadata`): its field is narrowed to the
value of its parent, so it declares its own.

The type argument of `FileBasedMetadata` declares the type of the raw
record, `raw`, which stays out of `repr` and `==`; the class does not
declare `raw` again (what the record holds goes in its docstring). A
new, empty record is that type called without arguments (`MyRaw()`), for
an object built in memory or converted from another format. Do not
derive a shared base of several formats from `FileBasedMetadata`: its
field would win by the MRO over the type argument of each format (the
Zarr formats share `_ZarrMetadataParser`, a `MetadataParser` only). The
type matters beyond documentation. `raw` and the read-time snapshot are
fields of `Metadata`, so generic metadata carries the record of the
metadata it was converted from, and a conversion gives the record back
to a format only when the record is an instance of the type that the
format declares (`_raw_class`, which `FileBasedMetadata` sets to
`type(None)` until a format declares its own). Every format must
therefore declare a type of its own: wrap a plain `dict` or `tuple` in a
small class (`ZarrRaw`, `X5Raw`), never share a type with another
format, and do not declare a subclass of the type of another format.
`tests/test_io_metadata_matrix.py` checks it. A format with no record
declares `FileBasedMetadata[None]`. A read alias under the familiar name
of the record (`header`, `tags`, `node`) is a plain property over `raw`.

`FileBasedMetadata` is the dispatcher of the formats whose files hold
metadata: `FileBasedMetadata.load(path, hint=...)` picks among the
classes registered into it with `@register_format`, by their
`EXTENSIONS`, `HINTS` and sniffers, as `brainhops.io.load` picks an
image parser. It is not a `FileBasedObject`, so `brainhops.io.load`
never returns metadata. The parser of a format, a `MetadataParser`,
comes first among its bases, and owns no registry. It is a `FileParser`:
the format implements `from_fileobj(file)`, which reads the raw record
of an open binary file, and nothing else (a NIfTI header, never the
voxels), then builds the metadata with `from_raw`; `from_filename` opens
a path in binary mode and hands it over, and `from_bytes` wraps the
bytes in a stream. A format that reads a path otherwise overrides
`from_filename` too (MGH, whose tags are read lazily from a path). A
format stored in HDF5 derives from `Hdf5MetadataParser` instead, and
implements `sniff_h5(h5file)` and `from_h5(h5file, **kwargs)`, as an
`Hdf5Parser` format does; a Zarr format implements `sniff_node(node)`
and `from_node(node)`, as `ZarrImage` does. A format whose record is an
object of its own on disk (the attributes of a Zarr array) overrides
`to_file(file)` to write `to_raw()` there; every other format refuses,
since its record is written along with the data. The reader of the image
or transformation shares the code that reads the record with the parser
(`_load_nifti_header`, `read_mgh_raw`, `read_h5_header`). A format whose
files hold no metadata (FLIRT, ITK `.tfm`) is not a `MetadataParser`,
and overrides `load` to refuse (`OpaqueMetadata.load`, from which
`ItkMetadata` inherits, and `FlirtMetadata.load`).

## Class keyword

- `supports=`: the vocabulary fields (and `"extra"`) the format can
  store, as names, group classes (all the fields of the group) or
  [`ALL`][brainhops.datamodel.metadata._sentinel.ALL]. Omitted, a subclass keeps
  its parent's capabilities. Every other field defaults to
  [`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED] and is
  refused at construction. `supported_fields` lists what the class
  stores, `unsupported_fields` the rest. A new vocabulary field is
  unsupported by every format until one opts in.

## Hooks

A format implements two hooks, both private and both optional:

| Hook | Called by | Gives |
|---|---|---|
| `_decode_raw(raw, *, image)` | `from_raw` | the vocabulary values of a raw record |
| `_encode_raw(raw, changed, *, image, report)` | `update_raw` | the raw record to write |

- `_decode_raw(raw, *, image=None) -> dict`: raw record to vocabulary,
  on read. Returns field names (and `"extra"`) to values; `None` values
  may be left out. `image` is the image or transformation the raw record
  belongs to, for fields that need it. A value for a field the class
  does not support is a bug of the format class, and `from_raw` raises
  `TypeError` rather than drop it unreported.
- `_encode_raw(raw, changed, *, image=None, report) -> raw`: vocabulary
  to raw record, on write. `raw` is the record to write over (already a
  copy, or the writer's own fresh record) and `changed` holds only the
  fields that differ from the read-time snapshot, so an untouched field
  keeps the record's value. A `None` in `changed` *clears* the slot.
  `changed["extra"]` is a per-key diff whose `None` values remove a key.
  Value-dependent loss goes in `report` (`report.lost[name] = value`,
  `report.approximated[name] = reason`). Returns the raw record to write.

**Fields the data model owns.** Some formats store, in the same slot, a
vocabulary field and a piece of the geometry of the data model: NIfTI
stores the repetition time as the time step of the image (`pixdim[4]`),
and a Zarr array is stored with the data type of its values. The writer
stores what the data model says, whatever the metadata says, so
`_encode_raw` does not write such a field when the data model has a
value for it (NIfTI writes `repetition_time` only for an image without a
time step; the Zarr formats never write `data_type`). The writer then
calls the public
[`check_raw`][brainhops.io.metadata.FileBasedMetadata.check_raw]
on its finished record: it decodes the record as a reader would, and
reports as approximated each changed field whose value the record does
not hold. So a field the data model owns needs no declaration.

**Public overrides.** What else a format needs, it does by overriding a
public method and calling `super()`:

- [`check_writable(*, image=None, raw=None)`][brainhops.io.metadata.FileBasedMetadata.check_writable]
  runs `update_raw` and `check_raw` on a scratch record, a copy of `raw`
  by default. A format whose writer starts from another record builds it
  and passes it on (NIfTI: the record, with the shape and the time step
  of `image`), so that a value-dependent check reads the same state as a
  real write.
- `_select(axis, positions, *, history=None)` and
  `_reslice(linear, *, history=None)`, the hooks of the image operations
  on [`Metadata`][brainhops.datamodel.metadata.Metadata] (`image[index]`
  along a time or channel axis, a change of the spatial axes), give the
  metadata of the derived image, with a deep copy of `raw`. A format
  whose record holds content tied to some axes, but outside the
  vocabulary, overrides them, calls `super()`, and removes that content
  from the copy: NIfTI's `_reslice` clears its slice slots and
  `dim_info`.

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
the field differs from the length of the axis, `_encode_raw` reports the
field as lost rather than writing entries that describe nothing.

## Reading and writing

**One word for the raw record.** Every name says "raw". A reader builds
the metadata with
[`from_raw`][brainhops.io.metadata.FileBasedMetadata.from_raw],
which decodes the raw record and keeps the read-time snapshot. A parser
given a raw record and a `metadata` that is not that record's (explicit,
or carried by `replace()`) uses
[`update_from_raw`][brainhops.io.metadata.FileBasedMetadata.update_from_raw],
which decodes the new raw record and keeps the changes. A parser does
both in one call from its `__post_init__`,
`sync_metadata(self, MyMetadata, raw, image=self)`
(`brainhops/io/metadata/_sync.py`): `same=` tells whether the metadata
holds the parser's record already (default: by identity), `raw` may be
a function that reads the record, called only when needed (Zarr, MGH),
and `force=True` decodes it afresh.

A writer starts from
`metadata, report = MyMetadata.writable(obj.metadata)`
([`writable`][brainhops.io.metadata.FileBasedMetadata.writable]:
the metadata converted to its class when it is of another, and a report
that holds the conversion's losses, with no policy applied yet), builds
its raw record, calls
[`update_raw`][brainhops.io.metadata.FileBasedMetadata.update_raw]
with it and that report, as `on_loss=report` (a report given as
`on_loss` is filled, never warned about; `force=` names a writer keyword
that must win over the record, such as MGH `tr=`), sets what it takes
from the data model, calls
[`check_raw`][brainhops.io.metadata.FileBasedMetadata.check_raw]
on the finished record with the same report, and hands the report to
[`apply_loss_policy`][brainhops.datamodel.metadata._report.apply_loss_policy]:
one write, one report, one warning.
`io.save` collects the reports of a conversion and of the write that
follows ([`collect_loss_reports`][brainhops.datamodel.metadata._report.collect_loss_reports])
and warns once, with
[`ConversionReport.merged`][brainhops.datamodel.metadata.ConversionReport.merged].

**JSON and key/value stores.** A format whose store is a JSON object or
a set of key/value pairs (x5 node `Metadata`, Zarr attributes, and the
MRtrix and NRRD headers to come) does not write its own codec:
`brainhops/io/metadata/_json.py` holds the one BIDS sidecars use (the
sidecars themselves are read and written by
`brainhops.io.metadata.bids`: `from_bids`, `to_bids`, and the
`BidsSidecar` reader that `FileBasedMetadata.load` picks for a `.json`
file; the data model does no input or output). `decode_object(obj,
names)` splits an object into the values of the vocabulary fields
`names` (read from their sidecar keys, BIDS keys or `CamelCase` names)
and the other keys, which are `extra`; `encode_changes(obj, changed,
report=)` writes the changed fields back under their keys (`None`
removes one, and a value JSON cannot hold is reported as lost), and
`encode_extra(obj, diff, report=, reserved=)` applies the `extra` diff,
reporting the keys the format keeps for itself as lost. `X5Metadata` is
the shortest example.

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

1. Write `<Fmt>Metadata` next to the parser: `on=`, `supports=`,
   `FileBasedMetadata[<record type of its own>]`, then `_decode_raw` and
   `_encode_raw`.
2. When the file holds metadata, put a `MetadataParser` first among its
   bases (`@register_format`, `EXTENSIONS`, `HINTS`, a sniffer and
   `from_fileobj`), so that `FileBasedMetadata.load` reads it;
   otherwise, override `load` to refuse.
3. In the parser's `__post_init__`, read the metadata from the raw
   record (`sync_metadata`); in the writer, encode it (`update_raw`),
   check the finished record (`check_raw`) and apply the loss policy.
4. Narrow the `metadata` field of the image or transformation class
   with `MetadataField`.
5. Add the class to the matrix test, `tests/test_io_metadata_matrix.py`:
   a format class that is not in its table fails it.
6. Describe what the format stores in the "Formats" section of the
   user guide.
