# Writing the metadata of a format

This page is for whoever adds the common vocabulary to the metadata of
a file format. The vocabulary is the set of fields of
[`Metadata`][brainhops.datamodel.metadata.Metadata], such as
`repetition_time`, `description` or `slice_timing`, that every format
shares so that metadata can be carried from one format to another. The
design is recorded in the memos `docs/design/format-records.md` (#415)
and `docs/design/format-metadata.md` (#233).

The metadata of a format is the class `<X>Metadata`, a subclass of
[`MetadataFormat`][brainhops.io.metadata.MetadataFormat] that lives next
to the record of the format, in `io/common/<x>/_metadata.py` or in the
package of the format. It holds the record of the file, `<X>Raw`, in its
`raw` field, and it holds the vocabulary in the fields that it inherits
from `Metadata`. A format that stores a vocabulary adds four things to
its class: the constant `SUPPORTED_FIELDS`, a `from_raw` that decodes
the record, a `consolidate` that encodes the fields that changed, and
two module-level functions that do the work, `_decode` and `_encode`.
NIfTI (`io/common/nifti/_metadata.py` and `_vocabulary.py`), MGH
(`io/common/mgh/_metadata.py`) and X5
(`io/transformations/x5/_metadata.py`) are the worked examples.

## Where the names live

The package `brainhops.datamodel.metadata` exports what a user needs:
`Metadata`, the vocabulary groups and their base `Vocabulary`, `Scope`,
the value classes `GeneratedBy`, `Channel` and `EncodingDirection`, the
report `ConversionReport`, and the loss policy. The package
`brainhops.io.metadata` exports `MetadataFormat`, the bases
`MetadataReader` and `MetadataWriter`, and the BIDS sidecar.
A format author imports the rest from the private modules that define
it.

| Module | Names |
|---|---|
| `brainhops.datamodel.metadata._vocabulary` | the annotations `Bids`, `Scoped` and `Along`, and the tables `VOCABULARY`, `FIELDS`, `GROUPS`, `BIDS_KEYS`, `SCOPES` and `ALONG` |
| `brainhops.datamodel.metadata._derive` | `Indexed` and `Resampled`, which an image operation builds for `derive` |
| `brainhops.datamodel.metadata._report` | `apply_loss_policy`, `OnLoss` and `LossPolicy` |
| `brainhops.io.metadata._base` | `consolidated`, `lost_fields` and `unsupported_values`, and, for the format objects, `metadata_to_write`, `requested_fields`, `overruled_fields` and `metadata_options` |
| `brainhops.io.metadata._json` | the codec of the JSON stores, which BIDS and X5 share |
| `brainhops.io.metadata._storage` | `preferred_storage` and `stored_values` |

## The class

```python
@register_format
class MyMetadata(MetadataReader, BinaryFileWriter, MetadataFormat):
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".my",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("my",)
    RAW: tx.ClassVar[type] = MyRaw
    SUPPORTED_FIELDS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "description",
        "echo_time",
    )

    raw: NoRepr[tx.Optional[MyRaw]] = None

    @classmethod
    def from_raw(cls, raw, **kwargs):
        return cls(raw=raw, **{**_decode(raw), **kwargs})

    def consolidate(self, *, inplace=False, on_loss=None, **fields):
        return consolidated(
            self,
            _encode,
            MyRaw,
            inplace=inplace,
            on_loss=on_loss,
            fields=fields,
        )
```

The record class `MyRaw` already scores and reads a file, so the
metadata names it in `RAW` and lists
[`MetadataReader`][brainhops.io.metadata.MetadataReader] among its bases.
Every `sniff_*` and `from_*` method of the metadata then passes its
input to the method of the same name of `MyRaw`, and the metadata opens
files in the mode of `MyRaw`. A format whose metadata writes its record
as it is lists
[`MetadataWriter`][brainhops.io.metadata.MetadataWriter] in the same
way. The example keeps `BinaryFileWriter` instead, because its writers
consolidate the metadata first, as described below.

`SUPPORTED_FIELDS` lists the fields of the vocabulary that the record
of the format can store, and `"extra"` when the format has a free-form
store. It is plain data that one function reads,
`unsupported_values`, so that a conversion into the format can report
the fields that a write will drop. The list also documents the format
on its API page. Put a table of the slot of each field in the
docstring of the class.

`from_raw` decodes the vocabulary at once, when the metadata is built.
A field that the format has no slot for, or that the record leaves
empty, holds `None`. Readers call `from_raw` once per read, and the
format objects call it when they consolidate, so the decoding must be
cheap and must not read the data of the file.

`consolidate` hands the encoder of the format to `consolidated`, with a
function that builds an empty record for metadata that has none, or
`None` when the format never makes up a record, as X5 does, because its
vocabulary lives in a transform. `consolidated` follows the contract of
the record memo: it applies the overrides given by name, compares each
field with a fresh decode of the record, returns a copy that shares
the record when nothing differs, encodes the changes into a copy of the
record otherwise, reports what is lost and approximated, and decodes the
result afresh, so that a second consolidation changes nothing.

The writers of the metadata, `to_bytes`, `to_filename` or `to_h5`,
consolidate first and write the record of the copy, and they take the
same overrides and `on_loss=` as `consolidate`. The metadata itself is
never changed by a write.

An image or a transformation of the format consolidates its metadata
first, as step 2 of the precedence on write, whether or not the format
stores a vocabulary. Its `consolidate` takes `on_loss=` and the fields
of the metadata by keyword. It asks `requested_fields` which fields the
write is asked to store, consolidates the metadata with
`metadata_to_write` and the report of the write, encodes its geometry
and its data over the record of the result, and acts on the report once
with `apply_loss_policy`. A format whose model owns a slot, such as the
time step of a NIfTI image, reports a field that it encoded over with
`overruled_fields`. A writer that writes the record of a consolidated
copy splits its options with `metadata_options`, so that `on_loss` and
the fields go to `consolidate` and the other options to the writer of
the record.

## The codec

`_decode(raw)` returns a dictionary of field values for a record, or
an empty dictionary for `None`. The values may be in any form that the
fields convert, such as a BIDS string for an encoding direction.

`_encode(raw, changed, report)` receives a copy of the record, which it
may change in place, and the fields that differ from the record. It
writes each field that the record can take, takes its name out of
`changed`, and returns the record. A field whose value has no slot is
left in `changed`, and `consolidated` reports it as lost, so a field
outside `SUPPORTED_FIELDS` needs no code. A value that the slot holds
only approximately, such as a description longer than the slot, is
written and reported in `report.approximated`. `consolidated` also
reports as approximated a field that the new record does not decode to
the value given, within single precision, unless the encoder reported
it already.

`None` clears a slot. What clearing means is the format's choice: an
empty string for a description, a zero for a parameter of the MGH
footer, a removed key for a JSON store.

A slot that belongs to the geometry or to the axes, which the image or
the transformation owns, is not written by the metadata. NIfTI reports
a changed `space`, and an intent that changes how the axes are read, as
approximated.

## Derived metadata

[`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive] gives
the metadata of an image computed from another one. It keeps the class
and the record of the metadata, and applies the rule of the scope of
each field. A field that the derivation clears differs from the record,
so the next consolidation clears its slot. A format overrides `derive`,
and calls `super()`, only when its record holds something tied to the
axes that the vocabulary does not cover: NIfTI then replaces the record
by a copy without `dim_info` and its slice slots when the spatial axes
move.

## Checklist

- `SUPPORTED_FIELDS` lists every field that `_decode` can return and
  that `_encode` takes. `tests/test_io_metadata_matrix.py` checks that
  the losses of a fully populated metadata are exactly the other fields.
- The exemplar of the format in `tests/test_io_conformance.py` gives a
  field and a value in `vocabulary`, for checks 21 to 25.
- The tests of the format decode each slot, encode each field into a
  copy, and probe each value that a slot holds approximately or not at
  all.
