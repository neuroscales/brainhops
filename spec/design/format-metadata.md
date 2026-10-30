# Design: format metadata and cross-format metadata conversion

Status: decided, not implemented yet. The fifth pass of the
format-record work (#415) implements it. This memo answers #233. It
records how brainhops represents the descriptive metadata of files,
which every format keeps under its own names and types, so that it can
be carried from one format to another. It builds on the record design
of `format-records.md`, where each format class holds a record of the
header of its file, and on `DataModelBase` (#97, `bagof.magic`).
Decisions are tagged `M1` to `M13` and collected at the end, followed
by the open questions. The `on_loss=` keyword that this memo names is
now the half of an error policy that concerns losses, beside
`on_error=` (see `error-policy.md`, #590).

On `main` today, `Metadata` has no field, and the metadata of a format
holds only its record: `MetadataFormat.consolidate` takes no option and
returns a copy that holds the same record. The code that this memo
describes is in two open pull requests, #310 and #287, which form pass
5a.

The fifth pass is split in three parts (decision Q4 of the maintainer).
Pass 5a (#310) adds the vocabulary to `Metadata`, the consolidation of
the metadata of a format, the BIDS sidecar, and the vocabulary of
NIfTI, MGH and X5; the wiring of the vocabulary into images,
transformations and their writers (#287) completes it. Pass 5b brings
the vocabulary of the key/value and FreeSurfer formats (LTA, AFNI,
MRtrix, NRRD, MINC), and pass 5c that of Zarr, OME-Zarr, FLIRT and
ITK-H5, over the records that the fourth pass gives them. The fourth
pass has given records to Zarr (#460), FLIRT and ITK-H5 (#452), and
OME-Zarr still waits for its record. Many formats have moved to records
or been added since the split was decided, and the part of the fifth
pass that takes each of them is open (section 11 and open question 13).
The section "Addendum: the record design" at the end records the
decisions taken when the design was adapted to the records, and it
holds where an older paragraph disagrees.

The problem, in one sentence: every file format keeps its non-spatial
metadata under a different name and type (`header`, `keyval`, `tags`,
`info`, `ome_xml`, `imagej_metadata`, `attributes`, `parameter_map`,
`struct`, ...), nothing was shared across formats, and `from_instance`
dropped all of it when a file was converted (NIfTI to MGH kept the data
model only, through `_foreign_format_fields` in `io/base/_base.py`).

## 1. What existed before

Facts, from the inventory of the branch that opened #233:

- No metadata slot on the data model (`Image`, `SingleScaleImage`,
  `MultiScaleImage`, `Transformation`); the x5 docs said so explicitly.
- Each parser mixin held a raw structure as the source of truth for
  geometry (nibabel headers, `MrtrixHeader`, `AfniHeader`, `NrrdHeader`,
  `X5Header`/`X5Node`, `LtaStruct`, `M3zStruct`, `ItkStruct`, elastix
  `parameter_map`, TIFF/Pillow/OpenSlide flat fields). MINC kept only
  `dimensions`.
- Same-format round trip was good for MRtrix, AFNI, NRRD, MGH, x5, LTA,
  M3Z, elastix; poor for NIfTI (fresh header unless `like=`; `slice_*`,
  `cal_*`, `aux_file`, extensions dropped), OME-TIFF (only `Name`,
  `Channel.Name`), OME-Zarr (no `omero`, no name), Pillow (only
  `icc_profile`, `exif`).
- Six formats have free-form key/value stores (MRtrix `keyval`, NRRD
  `keyvalue`, AFNI `attributes`, x5 `Metadata`/`attrs`, TIFF
  `imagej_metadata`/`tags`, Pillow `info`); none was carried across.
- Units disagree: MGH TR/TE/TI in ms and flip angle in radians, AFNI
  codes the TR unit, BIDS uses seconds and degrees.
- `Transformation.metadata_fields` already exists and means "the
  meta-attributes that define the encoding" (`order`, `bound`, `coeff`).

The record design of #415 has since replaced the parser mixins with a
record per format (`<X>Raw`), held by the metadata of the format
(`<X>Metadata`), which every format object holds. In this memo, `raw`
always names that record, the field `metadata.raw`. The arrays that a
format object stores, as they are laid out in the file, are held in its
field `rawdata` (#467), which the metadata never holds.

## 2. Goals, non-goals, and what stays out

Goals:

1. One `metadata` field, of one base type, on every image and
   transformation, which replaces all the format-specific attributes
   above.
2. A small common vocabulary with BIDS names and units, as `Magic`
   fields, so that the data model carries it across formats and `bagof`
   gives it conversion, repr, docs and equality.
3. A faithful record per format, so that read-then-save in the same
   format loses nothing it did not lose before (and NIfTI loses less).
4. Explicit, reported loss: each format declares the fields it can
   store, and a consolidation, which every write does, says what was
   lost.

Non-goals (v1): parsing DICOM; a units-aware quantity type for metadata
values (the vocabulary fixes one unit per field and stores plain
numbers; a quantity type can replace that later without renaming
fields).

What stays out of metadata (M1). Anything the data model already
represents is not metadata and never appears in the vocabulary:

- geometry: affines, voxel sizes, origins, direction cosines, the time
  axis origin (NIfTI `toffset`, AFNI `TAXIS_FLOATS[0]`) and step (NIfTI
  `pixdim[4]`), LTA/M3Z `src`/`dst` volume geometries;
- axes, their kinds and names, spatial and temporal units (`xyzt_units`,
  NRRD `space units`/`kinds`, OME `*Unit`, TIFF resolution tags);
- transformations and coordinate systems, including the transform
  *kind* (ITK `type`, LTA `type`, M3Z `type`, x5 `Type`/`SubType`,
  elastix `Transform`), which the transformation class itself encodes;
- storage encoding: byte order, layout, compression, data file names.
  These are writer options or record content, not descriptive metadata.
  The element type on disk and the intensity scaling are the exception
  (`dtype`, `scale_slope` and `scale_intercept`, section 4.4): they are
  user-visible choices that may differ from the loaded array (a scaled
  `int16` file loads as floats), so they are metadata; the byte order
  is not.

The record still contains these (a nibabel header has `pixdim`), and
the writer still encodes them from the data model. The vocabulary never
exposes the geometry.

## 3. Representation and layering (M2)

Answer to issue question 1: both, in one object. The metadata of a file
has three layers.

1. The common vocabulary: `Magic` fields, declared in seven groups
   (section 4) that the root class `Metadata` inherits, with BIDS names
   and units. Every metadata class has them.
2. The record `raw`: one field of the metadata of a format, whose type is
   the record of the format (`NiftiRaw`, `MghRaw`, `X5Raw`, the JSON
   object of a sidecar). Read-then-write in the same format goes through
   it. It is never copied across formats.
3. The extras `extra: Dict[str, Any]`: a free-form namespace for keys
   the vocabulary does not cover. It is copied across formats, into
   whatever free-form store the target has (MRtrix `keyval`, NRRD
   `keyvalue`, AFNI `attributes`, x5 `Metadata`, ImageJ metadata, PNG
   text chunks), and reported as lost where there is none (NIfTI, MGH,
   LTA, ITK, FLIRT).

This mirrors nibabel's `Nifti1Header` (typed accessors over a structured
record), with the difference that the accessors are fields, decoded
when the metadata is built.

The classes follow the layers of the record memo.

| Class | Package | Role |
|---|---|---|
| `Metadata` | `brainhops.datamodel.metadata` | The vocabulary, `extra`, `to` and `derive`. It holds no record, does no input or output, supports every field, and compares by value. |
| `MetadataFormat` | `brainhops.io.metadata` | The dispatcher of the metadata formats, and their base: `raw`, `SUPPORTED_FIELDS`, `from_raw`, `consolidate`. It compares by identity. |
| `<X>Metadata` | next to `<X>Raw` | The metadata of one format: its constant `SUPPORTED_FIELDS`, and a `from_raw` and a `consolidate` that call the module-level functions `_decode` and `_encode` of the format (for NIfTI, `decode` and `encode` in `_vocabulary.py`). |
| `BidsSidecar` | `brainhops.io.metadata` | The metadata of a BIDS JSON sidecar, a file of metadata only, whose record is the JSON object. |

```python
# brainhops/datamodel/metadata/_base.py
class Metadata(
    DataModelBase,
    TransformVocabulary,
    MicroscopyVocabulary,
    StorageVocabulary,
    DisplayVocabulary,
    DiffusionVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    kw_only=True,
    reverse=True,
    repr=HideIfNone,
):
    extra: tx.Dict[str, tx.Any] = Factory(dict)

    def to(self, cls=None, *, on_loss=None, **values) -> "Metadata": ...
    def derive(self, operation=None, *, history=None) -> tx.Self: ...


# brainhops/io/common/nifti/_metadata.py
@register_format
class NiftiMetadata(MetadataReader, BinaryFileWriter, MetadataFormat):
    SUPPORTED_FIELDS = ("description", "sources", "intent", ...)
    raw: NoRepr[tx.Optional[NiftiRaw]] = None

    @classmethod
    def from_raw(cls, raw, **kwargs):
        return cls(raw=raw, **{**decode(raw), **kwargs})

    def consolidate(self, *, inplace=False, on_loss=None, **fields):
        return consolidated(
            self,
            encode,
            NiftiRaw,
            inplace=inplace,
            on_loss=on_loss,
            fields=fields,
        )
```

The groups are listed in the reverse of their order because `bagof`
lists the fields of a class before those it inherits (`reverse=True`),
so that `repr` shows `extra`, then the vocabulary in its declared
order. The package `brainhops.datamodel.metadata` exports 17 names:
`Metadata`, `Vocabulary` and the seven groups, `Scope`, `GeneratedBy`,
`Channel`, `EncodingDirection`, `ConversionReport`,
`MetadataLossWarning`, `MetadataLossError` and `metadata_loss_policy`.
The names that a format author needs are imported from its private
modules (`_vocabulary`, `_terms`, `_report`, `_derive`), which the
format author's guide (`docs/dev/metadata-formats.md`, added by #310)
lists. No module of the data model imports `brainhops.io`.

## 4. Vocabulary (M3)

Answer to issue question 2. Names are the BIDS keys in snake_case
(`RepetitionTime` becomes `repetition_time`); the BIDS spelling is kept
as field metadata (`tx.Annotated[..., Bids("RepetitionTime")]`) for the
sidecar codec. Units are the BIDS units: seconds, degrees, tesla, mm,
s/mm². Where BIDS has no key, the name is chosen to read like one.

Every field is `Optional[T]`, where `None` means that the value is
unknown or that the format has no slot for it (section 5), and carries
a `Scope` used by propagation (section 9): `FILE` (about the file or
the data as a whole), `ACQUISITION` (invariant under resampling),
`SPATIAL` (tied to the spatial sampling), `AXIS` (one entry per
index along a non-spatial axis, which `Along(AxisType...)` names) and
`VALUES` (a description of the values themselves, which a change of
frame invalidates).

Groups. Each subsection below is a `Magic` base class that declares its
fields, and nothing else: `ProvenanceVocabulary` (4.1), `MRIVocabulary`
(4.2), `DiffusionVocabulary` (4.3), `DisplayVocabulary` and
`StorageVocabulary` (4.4), `MicroscopyVocabulary` (4.5),
`TransformVocabulary` (4.6). `Metadata` inherits all seven, and a field
is defined in exactly one group, never on `Metadata` itself. The groups
are an organisational device: they give one place per topic for the
docs and a per-group rule in the codecs (BIDS keeps
`DiffusionVocabulary` in `.bval`/`.bvec` files, not in the sidecar).
They do not narrow types. `VOCABULARY` (the field names, group by
group), `FIELDS` (`extra`, then the vocabulary), `GROUPS`, `BIDS_KEYS`,
`SCOPES` and `ALONG` are module constants of `_vocabulary`.

Known terms. A free-text field with a list of known terms is typed
`Optional[Union[<Enum>, str]]`: a known term is held as the enum member
(a `StrEnum`, so it is still a string and compares equal to its value),
any other string stays a string, so the vocabulary is documented
without being closed. The enums live in `brainhops.datamodel.enums`:
`SpaceEnum`, `IntentEnum`, `Manufacturer`, `IlluminationType`,
`ContrastMethod`. Truly free text (`name`, `description`, `history`,
file references) stays `str`. The converter of `bagof` turns a string
that names a member into that member and keeps any other string, so
those fields need no converter of their own. Inherited fields come out
of `fields()` in reverse MRO order, so every loop over the vocabulary iterates `VOCABULARY`, never
`fields(Metadata)`.

### 4.1 Core and provenance (`file`)

| Field | Type / unit | BIDS | Native sources |
|---|---|---|---|
| `name` | str | — | OME `Name`, TIFF `DocumentName`, OME-Zarr multiscale `name` |
| `description` | str | — | NIfTI `descrip`, TIFF `ImageDescription` (plain), NRRD `content`, OpenSlide `openslide.comment`, JP2 `comments`, x5 `Metadata.description` |
| `history` | tuple[str] | — | MRtrix `command_history`, AFNI `HISTORY_NOTE`, MGH `tags` (cmdline), MRtrix `.txt` comments, LTA comments |
| `generated_by` | tuple[GeneratedBy(name, version, description)] | `GeneratedBy` | TIFF `Software`, MRtrix `mrtrix_version`, ITK h5 `ITKVersion`, x5 `Format`/`Version`, AFNI `TYPESTRING` |
| `creation_time` | datetime | — | TIFF `DateTime`, AFNI `IDCODE_DATE`, EXIF |
| `sources` | tuple[str] | `Sources` | BIDS-style provenance inputs only: NIfTI `aux_file`, elastix `InitialTransformParameterFileName`, NRRD/MRtrix none |
| `space` | `SpaceEnum` or str | `SpatialReference` | NIfTI `sform_code` name, AFNI `TEMPLATE_SPACE`, NRRD `space`, x5 `Domain.Coordinates` label |
| `intent` | `IntentEnum` or str (the NIfTI intent names) | — | NIfTI `intent_code/name`, AFNI `BRICK_STATSYM`, NRRD `kinds` (non-spatial) |

`space` is a label (`"MNI152NLin6Asym"`, `"scanner"`, `"orig"`); the
coordinate system itself stays in the data model. `SpaceEnum` lists the
NIfTI code names (`scanner`, `aligned`, `talairach`, `mni`,
`template`), the BIDS standard templates and the BIDS non-standard
spaces; `IntentEnum` lists the NIfTI intent names as nibabel spells
them. For NIfTI, `space` and an intent that retypes the axes belong to
the geometry (section 6.2). The LTA/M3Z image references are
`moving`/`fixed` (4.6), not `sources`.

### 4.2 MRI acquisition (`acquisition`, except where noted)

| Field | Unit | BIDS | Native sources |
|---|---|---|---|
| `repetition_time` | s | `RepetitionTime` | MGH `tr` (ms), AFNI `TAXIS_FLOATS[1]`+unit code, MRtrix keyval, NIfTI `pixdim[4]` (geometry-derived, 6.2) |
| `echo_time` | s | `EchoTime` | MGH `te` (ms), MRtrix/NRRD keyval |
| `inversion_time` | s | `InversionTime` | MGH `ti` (ms) |
| `flip_angle` | deg | `FlipAngle` | MGH `flip_angle` (rad) |
| `magnetic_field_strength` | T | `MagneticFieldStrength` | keyval only |
| `manufacturer` | `Manufacturer` or str | same | TIFF `Make`, OpenSlide `vendor`, OME Instrument |
| `manufacturers_model_name`, `institution_name` | str | same | TIFF `Model`, OME Instrument |
| `acquisition_time` | datetime | `AcquisitionTime` | OME `AcquisitionDate`, EXIF |
| `phase_encoding_direction` (`SPATIAL`) | `EncodingDirection` | same | MRtrix keyval, NIfTI `dim_info` |
| `total_readout_time`, `effective_echo_spacing` | s | same | keyval |
| `slice_encoding_direction` (`SPATIAL`) | `EncodingDirection` | same | NIfTI `dim_info`, MRtrix keyval |
| `slice_timing` (`SPATIAL`) | tuple[float] s | `SliceTiming` | NIfTI `slice_code/start/end/duration` (expanded), AFNI `TAXIS_OFFSETS`, MRtrix keyval |
| `multiband_acceleration_factor` | int | same | MRtrix keyval |

The TR is in the vocabulary even though the time step is geometry: the
two coincide for a plain fMRI series but not for a multi-echo stack or
a sparse acquisition, and MGH/AFNI store the TR without any time axis.
Formats that only have a time step treat `repetition_time` as
geometry-derived (6.2).

Encoding directions. Three representations were compared. (a) The
BIDS strings `i`/`j`/`k` with an optional `-`: trivial codecs, but tied
to the on-disk axis order, broken by any in-memory permutation or flip,
and unable to say anything after a resampling that is not axis-aligned.
(b) The axis names of the voxel system of the image (`"-y"`): this
survives a permutation if the axis objects are kept, but it is still
bound to an axis. (c) An oriented unit vector in a coordinate system:
this survives any linear map, and the codecs snap it back to an axis.
(b) is (c) with an axis-aligned vector in voxel space, so (c) is
chosen: `EncodingDirection(vector, space=None)`, where `space=None`
means the voxel axes of the image, which is the BIDS frame, and a label
(a `SpaceEnum` or a string) or a brainhops `CoordinateSystem` means a
world space. A BIDS string is accepted wherever a direction is
(`metadata.phase_encoding_direction = "j-"` stores `(0, -1, 0)`), and
`to_bids()` gives the string back, or `None` when it is along no voxel
axis. Equality is that of `bagof`, field by field, and a coordinate
system is never equal to a label, not even its own name. The vector is
normalised on construction and its components within `1e-9` of 0 or
+-1 are snapped, so that `(0, 0, 2)`, `"k"` and a `"k"` mapped through
a permutation are the same tuple. BIDS, MRtrix and NIfTI `dim_info`
store an axis: they round-trip an axis-aligned direction exactly (NIfTI
drops the polarity, approximated) and report any other one as lost. A
JSON store (x5) writes it as `{"Vector": [...], "Space": ...}`, with a
coordinate system written as its name, and reads it back as a label; a
direction in a coordinate system without a name is lost there.
`EncodingDirection.transform` maps a direction by a matrix or by a
transformation that reduces to an affine. The slice timing stays a
tuple indexed along the slice encoding direction.

### 4.3 Diffusion (`AXIS`, along the time axis)

| Field | Type / unit | Native sources |
|---|---|---|
| `bvalues` | (N,) float, s/mm² | MRtrix `dw_scheme` col 4, NRRD `DWMRI_b-value` x gradient norms, BIDS `.bval` |
| `bvectors` | (N,3) float, unit vectors in **world (RAS) coordinates** | MRtrix `dw_scheme` cols 1-3 (already world), NRRD gradients via `measurement frame`, BIDS `.bvec` via the voxel-to-world rotation |

The names carry no `diffusion_` prefix: no format we read has b-values
or b-vectors of another kind (NRRD `DWMRI_b-value`, MRtrix `dw_scheme`
and BIDS `.bval`/`.bvec` are all diffusion), and the group says the
rest. The entries run along the time axis of brainhops, which is NIfTI
dimension 4.

The frame convention is the lossy spot in diffusion metadata; the
codecs rotate into and out of world coordinates using the affine of the
image, which is a step of the wiring of the format objects.

### 4.4 Display, channels and storage

| Field | Type | Native sources |
|---|---|---|
| `display_range` | (min, max) float | NIfTI `cal_min/max`, AFNI `BRICK_STATS` (first volume) |
| `channels` | tuple[Channel(name, color, display_range, unit)] | AFNI `BRICK_LABS`, NRRD `labels`, ImageJ `Labels`/`LUTs`/`Ranges`, OME `Channel.Name/Color`, OME-Zarr `omero.channels` |
| `data_unit` | `Unit` or str | NRRD `sample units`, OME channel unit, MINC `units` |
| `dtype` | `numpy.dtype` (native order) | NIfTI `datatype`, MGH `type`, MRtrix/NRRD `datatype`/`type`, Zarr array `dtype`, ITK `precision` |
| `scale_slope`, `scale_intercept` | float | NIfTI `scl_slope`, `scl_inter` |

`display_range`, `data_unit` and the storage fields are in the
`VALUES` scope, and `channels` in the `AXIS` scope, along the channel
axis. `Channel.color` is an RGBA hex string, held in upper case without
`#`, with an opaque alpha added to an RGB string; LUT arrays stay in
`extra`.

`data_unit` is a `Unit` whenever `brainhops.datamodel.units` (backed by
`pint` since #291) parses the name: compound units (`"mm/s"`),
intensity, angle and field units, and the arbitrary unit (`"a.u."`,
`"au"`) all parse. Only a name that does not parse (`"mm2/s"`) stays a
string, so that a file with an odd unit still reads. The type therefore
stays `Optional[Union[Unit, str]]` with a lenient converter. Units
compare as units (`Unit("a.u.") == Unit("au")`), not as strings. A
codec writes a `Unit` as its symbol, which parses back to the same
unit, and an unparsed name as it was read.

`dtype` is the element type of the data as stored, which may differ
from the type of the loaded array (a scaled `int16` file loads as
floats). It was called `data_type` before the record design, and is
named `dtype` now because it is the same thing as the `dtype=` option
of the writers (section 6.2). Its byte order is storage encoding (M1),
so it is held in native order. `scale_slope` and `scale_intercept`
describe the intensity scaling of the file: a value is
`stored * scale_slope + scale_intercept`.

### 4.5 Microscopy (`acquisition`)

The group is deliberately small:
`objective_magnification` (float; OpenSlide `objective-power`, OME
`Objective.NominalMagnification`), `objective_numerical_aperture`,
`illumination_type` and `contrast_method` (`IlluminationType` and
`ContrastMethod`, the OME enumerations, or any other string).
Everything else goes to `extra` until two formats agree on it. The
Imaris (#437) and BigDataViewer (#441) readers now exist, and the
parameters of their records (the `DataSetInfo` groups of an Imaris
file, the setups of a BigDataViewer dataset) have not been compared
with this group yet. Whether they add fields to it is open, and so is
the same question for CZI, LIF and ND2, which brainhops does not read.

### 4.6 Transformation-specific (`file`)

| Field | Type | Native sources |
|---|---|---|
| `moving` / `fixed` | str (file reference) | LTA `src/dst filename`, M3Z `image/atlas fname`, FLIRT `src`/`ref` (user-given, never stored), ANTs convention |
| `input_space` / `output_space` | `SpaceEnum` or str label | x5 `Domain.Coordinates`, LTA (derived from `type`), OME-Zarr coordinate system names |

`moving`/`fixed` are deliberately not `source`/`target`: the data model
warns that the `input`/`output` of brainhops are the inverse of the
imaging direction, and `moving`/`fixed` are unambiguous about which
image is which. `history`, `generated_by`, `description` and `sources`
are shared with images.

### 4.7 Not in the vocabulary, and why

Field of view (derived from geometry), AFNI per-volume stats (record),
NIfTI `intent_p1..p3` (record; FNIRT and NiftyReg read them from the
record), byte order and compression (encoding), elastix resampler keys
(record, written back by `transformation_to_map`), EXIF/ICC blobs
(`extra["exif"]`, `extra["icc_profile"]`), OpenSlide
`bounds`/`background_color`/associated images (`extra`), JP2 boxes
(record). The vocabulary has 39 fields; adding one needs two formats
that carry it natively, or one format plus a BIDS key, and it goes into
one group.

## 5. Supported fields (M5)

A format must be able to say that it cannot store a field, and the
library must be able to list what each format stores. Two ways were
weighed: a sentinel value, `UNSUPPORTED`, held by the fields that a
format has no slot for, or `None` with a declaration of the supported
fields on each class. The sentinel was dropped (decision Q2 of the
maintainer, answer b).

- A field that the format has no slot for decodes as `None`, as an
  unknown value does. Every field holds `Optional[T]`, so the types, the
  converters and `repr` need no special case.
- Each `<X>Metadata` class declares the fields that it supports as one
  plain constant, `SUPPORTED_FIELDS`, a tuple of field names, with
  `"extra"` when the format has a free-form store. The constant is
  documented on the API page of the format, next to a table of the slot
  of each field in the docstring of the class.
- One function reads the constant: `unsupported_values(cls, metadata)`
  in `brainhops.io.metadata._base`, which returns the fields that are
  set in `metadata` and that `cls` does not list.
- A conversion into a format, `MghMetadata.from_instance(md)` or
  `md.to(MghMetadata)`, copies every field by name, including those that
  the format cannot store, and reports with `unsupported_values` the
  fields that a write will drop. This is a lower bound of the loss: a
  value may not fit a slot that exists (an 81-byte description in
  NIfTI), which only consolidation finds.
- `consolidate` reports the exact losses: the encoder of the format
  takes the fields that it writes, and `lost_fields` reports the fields
  that are left. A field outside `SUPPORTED_FIELDS` needs no code.
- A field can be assigned whatever the format supports. Assignment is
  not validated, and the next consolidation drops and reports what the
  format cannot store.
- There is no metaclass, no `supports=` class keyword, no
  `supported_fields` class variable and no `format` discriminant.
  Generic `Metadata` supports every field.

`tests/test_io_metadata_matrix.py` ties the constant to the codecs: a
fully populated metadata, converted into each format and consolidated
into a fresh record, loses exactly the fields that the constant does
not list, no more and no less.

## 6. Precedence: record versus fields (M6)

Answer to issue question 5. Two models were weighed:

- Views. The record is the only state; common fields are properties
  that decode and encode it (nibabel style). This is faithful by
  construction, but properties are not `Magic` fields: no `from_other`,
  no `replace`, no repr, no `eq`, and every format writes about 35
  property pairs.
- Overlay. Common fields and the record are both stored. On read the
  codec decodes the record into the common fields once. On write the
  record is the base and the common fields are encoded over it.

Decision: overlay, with eager decoding and a comparison with a fresh
decode at consolidation (decision Q1 of the maintainer).

- `<X>Metadata.from_raw(raw)` decodes the vocabulary at once, with the
  module-level `_decode(raw)` of the format, and holds the record as it
  is given. Every record of a migrated format is already parsed in
  memory (`NiftiRaw.header`, `MghRaw.tags`, `X5Raw.nodes`), so the cost
  is small; reading the metadata of an MGZ file decompresses it to the
  tags, which the record memo accepts.
- `consolidate(*, inplace=False, on_loss=None, **fields)` is the only
  encoder. It replaces the fields given by name (an unknown name raises
  `TypeError`), builds a fresh decode of the record with `from_raw`, and
  treats as changes the fields that differ from it. With no change, the
  copy shares the record, and metadata without a record and without a
  field set stays without a record. Otherwise the changes are encoded
  into a copy of the record, or into a new, empty record of the format,
  by the module-level `_encode(record, changed, report)` of the format,
  which takes out of `changed` each field that it writes. The fields
  that are left are reported as lost, a field that the new record holds
  only approximately is reported as approximated, and the copy is
  decoded afresh from the new record, so a second consolidation changes
  nothing and a truncated description reads back truncated.
- A record is never changed in place, so metadata that shares a record
  never sees another one change it.
- `on_loss=` takes a policy (`"ignore"`, `"warn"`, `"raise"`) or a
  `ConversionReport` to fill. It is the only option of `consolidate`.

The rule a user has to know follows: a field that differs from the
record wins, and a field set to `None` clears its slot in the record.
What clearing means is per format: NIfTI `descrip = ""`, `slice_code =
0` (with `slice_start/end/duration`), `cal_min = cal_max = 0`; a
keyval key is removed; a zero in the MGH footer. The record is edited
for what the vocabulary does not cover: edit a copy of the record
(`record.copy()`), and build the metadata from it with `from_raw`, so
that its fields describe the edited record. Editing the record of
metadata in place is not supported; its fields would then differ from
the record, and the next consolidation would encode them over the edit.

A first version kept a read-time snapshot of the decoded values and
compared the fields with it. The snapshot was state that every copy,
conversion and derivation had to carry, and the comparison with a fresh
decode replaces it with no state at all. The methods `update_raw`,
`update_from_raw`, `to_raw`, `check_raw`, `check_writable` and
`writable` were removed with it.

### 6.1 Naming: `raw`, not `struct`

The record is held in the field `raw`, hidden from `repr`, and its
types are named `<X>Raw` (`NiftiRaw`, `MghRaw`, `X5Raw`, `LtaRaw`).
The historical accessors (`header`, `keyval`, `struct`, `tags`, `node`)
are not kept as aliases; the record memo removed them.

### 6.2 Fields that the data model owns

Some fields are, for one format, a view of something that the image or
the transformation owns. The metadata alone encodes them into the
record, and the format object, which encodes its geometry over the
record of its metadata (step 3 of the precedence on write of the record
memo), decides them when it is written.

- NIfTI `repetition_time` is the time step `pixdim[4]`. The metadata
  writes it into a header with a time axis, and reports it as lost
  otherwise. When an image is written, its time step is the geometry's,
  and a `repetition_time` that disagrees with it is reported as
  approximated.
- NIfTI `space` is the name of the codes of the sform and the qform,
  which the geometry decides, so the metadata reports a changed `space`
  as approximated. An intent that retypes the axes (a displacement
  field or a vector) is the geometry's in the same way; another intent
  is written into `intent_code`.
- `dtype` decides the type of the stored values (decision Q5). The
  precedence is: an override given to `save` or to `consolidate`, then
  `metadata.dtype`, then the type of the data. A format object forwards
  the keyword arguments of `save` and `consolidate` to the
  consolidation of its metadata, so any field of the metadata can be
  overridden on save, and the writer option `dtype=` is the metadata
  field `dtype`; there is no separate option. A label map read as
  `uint8` and computed on as `int64` is therefore written as `uint8`
  again, as nibabel does with `header.get_data_dtype()`. Where a writer
  option of a format has the name of a field of the metadata, it means
  the same thing. A format that cannot store a type reports it as
  approximated (MGH stores `uint8`, `int16`, `int32` and `float32`).
- `scale_slope` and `scale_intercept` are written into the NIfTI header
  by the metadata. `preferred_storage` (`brainhops.io.metadata._storage`)
  keeps the scaling when the values fit it, and otherwise stores the
  values without a scaling and reports the scaling as approximated.
- The `labels` property of an image (#634) says whether the image is a
  label map. It is stored on the image, because a table of label names
  and colours has no place in the vocabulary, and because a value given
  by the caller must win over the metadata. When no value is stored,
  the property is derived from `intent`: the intents `label` and
  `neuroname` give `True`, another intent gives `False`, and no intent
  gives `None`. A multi-scale image without a value of its own takes
  the value of its finest level. When a NIfTI image whose `labels` is
  true is written without a label intent, the intent `label` is written.
  A label map changes two defaults, and an explicit argument still
  wins. `reslice` uses nearest interpolation (degree 0) instead of
  linear interpolation, and the scales of a pyramid, built by
  `with_scales`, by `to(..., nscales=)` or on save, use the mode
  `"mode"` instead of `"mean"`. This is implemented.

The forwarding of the save options and the use of `metadata.dtype` by
the writers are part of the wiring of pass 5a (#287). The metadata half,
`MetadataFormat.consolidate(**fields)`, is implemented in #310. On
`main` today, the stored type is a writer option of each image format,
which `consolidate` and `save` take: NIfTI and MGH take it as the header
override `dtype`, and AFNI, MRtrix and NRRD as `datatype=`. The option
`dtype` of NIfTI and MGH becomes the metadata field. Whether the option
`datatype=` of AFNI, MRtrix and NRRD is renamed `dtype` in pass 5b, so
that it also becomes the metadata field, is open (open question 14).

## 7. Conversion and loss reporting (M7)

Answer to issue question 3. Conversion is the data model's own path:
`TargetMetadata.from_any(source)` copies metadata of any class with
`from_instance`, and `source.to(Target)` is sugar over it, symmetric to
images and transformations. `MetadataFormat.from_instance`:

1. copies every field of the vocabulary and `extra` by name, and copies
   `extra` as a new dictionary;
2. leaves `raw` unset when `other` is of another format, because the
   record of one format means nothing to another
   (`_foreign_format_fields`), and copies it within the format;
3. reports the fields that the target does not list in
   `SUPPORTED_FIELDS` as lost, a lower bound of the loss of a write.

Loss also depends on the value. A slot exists, but a value may not fit
it: NIfTI `description` holds 80 bytes; NIfTI `slice_timing` is
encodable only when it matches one of the `slice_code` patterns
(sequential, alternating, their reverses); MGH holds one scalar TR; an
AFNI per-brick list is written only when it has as many entries as
volumes. These are known only when the value is encoded, so the encoder
reports them in the report of the consolidation:
`report.approximated["description"] = "truncated to 80 bytes
(descrip)"`, or a field left in `changed`, which is reported as lost.

```python
class ConversionReport(DataModelBase):
    source: tx.Optional[str] = None
    target: tx.Optional[str] = None
    lost: tx.Dict[str, tx.Any] = Factory()  # field -> value dropped
    approximated: tx.Dict[str, str] = Factory()  # field -> what changed

    @property
    def lossy(self) -> bool: ...
    def merge(self, other) -> tx.Self: ...
    def raise_if_lossy(self) -> None: ...
```

Policy, from least to most strict: `"ignore"`, `"warn"` (the default:
one `MetadataLossWarning` per conversion or consolidation, carrying the
report), `"raise"` (`MetadataLossError`, which derives from
`Exception`, not `ValueError`, so that a field converter does not turn
it into a conversion error). The policy is a keyword, `on_loss=`, on
`to`, `from_instance`, `consolidate`, the writers of the metadata and
`to_bids`, and a context manager, `metadata_loss_policy("raise")`, for
the implicit conversions that field converters trigger. A report given
as `on_loss=` is filled instead of warning or raising, so a caller that
converts and then consolidates can merge the two reports and warn once.
`apply_loss_policy(report, on_loss)` is the one place where a report is
acted on. A known term reads as its string in a report.

Free-form pass-through (M8). `extra` is `str -> Any`. Key/value formats
whose values are text (MRtrix, NRRD) write non-string values as JSON
and read back strings unchanged. Typed stores (AFNI attributes, x5
`Metadata`, ImageJ) keep Python types. Keys are copied verbatim; a
format with reserved keys refuses them and reports them as lost. A JSON
store replaces its free-form keys with `extra` as a whole when `extra`
changed.

BIDS JSON sidecar (M8). `BidsSidecar` is a registered metadata format
(`EXTENSIONS = (".json",)`), whose record is the JSON object of the
sidecar, so `MetadataFormat.load("bold.json")` reads it. Vocabulary
fields are stored under their `Bids(...)` key, or under their name in
CamelCase when BIDS has none (`display_range` as `"DisplayRange"`), so
that a sidecar written by brainhops reads back whole; `generated_by` is
the BIDS list of objects, `channels` a list of objects with CamelCase
keys, times are ISO strings, a known term its string, `data_unit` its
unit symbol and `dtype` its numpy name. Unknown keys are `extra`. The
`DiffusionVocabulary` group is not in the sidecar, and an encoding
direction along no voxel axis has no BIDS string; both are lost.
`from_bids(source)` reads a mapping, a JSON string, a path or an open
file into generic `Metadata`, and `to_bids(metadata, *, on_loss=None)`
returns the JSON object, with one report for the whole write. Wiring
sidecars into `io.load`/`io.save` (`sidecar=True`) is open question 8.

## 8. Where `bagof.magic` is used, and where it is not (M4)

- The vocabulary fields are `Magic` fields, declared on seven `Magic`
  groups (which say `convert=True` and `repr=HideIfNone` themselves,
  since a field keeps the options of the class that declares it) and
  inherited by `Metadata`, a `DataModelBase` with `kw_only=True`. This
  gives `from_dict`, `from_instance`, `replace`, documented fields, and
  a repr that hides the `None`s.
- Conversion is `from_any`/`from_instance`, explicit code on
  `Metadata` and on `MetadataFormat`, which also accepts metadata of a
  sibling format in `from_any`.
- Equality: generic `Metadata` compares by value. The metadata of a
  format is a `MetadataFormat`, which is declared with `eq=False` and
  compares by identity, as format objects do.
- Not used: a metaclass, class keywords, polymorphic construction on a
  `format` field, descriptors for lazy fields, or a registry of
  propagation handlers. Each was tried in the prototype and removed:
  the maintainer asked for explicit code in each class, explicit
  dispatch, and no generic hooks or class-variable indirection. The one
  class variable that is read, `SUPPORTED_FIELDS`, is plain data with
  one reader.

## 9. Propagation (M9)

Answer to issue question 4. Propagation is driven by the scope of each
field. The image owns the operation: the metadata does not know the
coordinate system, the shape or the transformation of the image, so the
image describes what it did with a record, and hands it to
`Metadata.derive`, which returns new metadata of the same class.

```python
def derive(self, operation=None, *, history=None) -> tx.Self: ...


class Indexed(DataModelBase, frozen=True):  # image[index]
    index: tuple
    shape: tx.Tuple[int, ...]
    system: tx.Optional[CoordinateSystem] = None

    def positions(self, axis: AxisType) -> tx.Optional[np.ndarray]: ...

    moves_space: bool  # a property
    voxel_map: tx.Optional[np.ndarray]  # a property


class Resampled(DataModelBase, frozen=True):  # image.reslice(geometry)
    transformation: Transformation  # new voxels -> old voxels
    geometry: tx.Optional[Geometry] = None


class Reframed(DataModelBase, frozen=True):  # image.reframe(frame)
    axes: tx.Tuple[AxisType, ...] = ()  # types of the re-expressed axes
```

`Indexed`, `Resampled` and `Reframed` live in the private module
`brainhops.datamodel.metadata._derive`, which the image operations
import. `Indexed.positions(axis)` is `np.arange(size)[component]` for
the first axis of that type (`None` when the axis was dropped or the
selection is not 1-D), and `voxel_map` is the linear part of the
old-to-new voxel map (`None` when it is unknown or not affine). The
rules are one explicit loop over the vocabulary, in `derive`:

- `FILE` and `ACQUISITION` fields are kept.
- `SPATIAL` fields (`slice_timing`, `slice_encoding_direction`,
  `phase_encoding_direction`) are cleared when the spatial axes move,
  with one exception: an encoding direction in voxel axes is pushed
  through `voxel_map` (`v' = normalize(L @ v)`), so a permutation or a
  flip remaps `"j-"` and an oblique resampling keeps an oblique
  direction. The map of a 4-D image has more axes than the direction,
  which lies in its first three; the direction is kept only when it
  still lies there. Without a map the direction is cleared; a direction
  in a world space does not move with the voxels.
- `AXIS` fields (`channels`, `bvalues`, `bvectors`) keep, after an
  index, their entries at `positions` along the axis type that their
  `Along` names, so a slice, integers or a boolean mask all work. They
  are kept when the image has no such axis, or when the index leaves it
  untouched, and cleared when the index drops it or keeps a position
  beyond the field. After a change of frame, they are cleared when an
  axis of the type that their `Along` names was re-expressed.
- `VALUES` fields (`data_unit`, `display_range`, `dtype`,
  `scale_slope`, `scale_intercept`) are kept, except after a change of
  frame, which clears them (see section 5 of
  `spec/design/vector-and-tensor-images.md`).
- `extra` is kept, as a new dictionary.
- Provenance: `creation_time` is cleared, the steps of `history` are
  appended to `history`, and `generated_by` gains a brainhops entry,
  once. `MetadataFormat.derive` keeps `history` and `generated_by` as
  they were when the format has no slot for them, so that a derived
  image is not reported to lose a provenance that only the derivation
  added.

Derived metadata keeps its class and its record. The record is never
changed, so it is shared, not copied. A field that the derivation
cleared differs from the record, so the next consolidation clears its
slot: NIfTI `slice_code` goes to 0 after a resampling. A format whose
record holds content tied to the axes outside the vocabulary overrides
`derive` and calls `super()`: when the spatial axes move,
`NiftiMetadata.derive` replaces the record by a copy without `dim_info`
and the slice slots (`clear_spatial_slots`), and the encoding
directions that the derivation mapped are written again at the next
consolidation. AFNI `TAXIS_OFFSETS` and the `_PER_GRID`/`_PER_BRICK`
attributes, and the NRRD `_PER_AXIS` fields, will do the same in pass
5b.

Where it is called (the wiring of #287):

- Data-model operations. `SingleScaleImage.__getitem__` derives with
  `Indexed(index, self.shape, grid.input)` and the step `"getitem"`.
  `reslice` derives with `Resampled(new2old, geometry)`, where
  `new2old` is the voxel-to-voxel map without the grid, and the step
  `"reslice"`. `__call__` keeps the metadata as it is. A
  `MultiScaleImage` reslices through one of its levels, so it inherits
  the rule.
- Multiscale. The metadata of a pyramid is one object on the
  `MultiScaleImage` (OME-Zarr `omero` is per multiscale, TIFF tags are
  per file). The record design has since decided how a pyramid format
  holds it (section "Pyramids" of the record memo, tension 15): one
  record describes the whole container, and the pyramid and each of
  its levels hold the same metadata object, by identity, with the index
  `level` of the level in the record. A first version of this memo gave
  each level derived metadata instead: the first level got `derive()`,
  and level `i` got `derive(Resampled(...))` with
  `level_0.transformation.inverse() @ level_i.transformation`, so that a
  direction in voxel axes would survive the coarser sampling. How the
  vocabulary of a pyramid reaches its levels, under the shared metadata
  object of the record design, is open (open question 12).
- Transformations: composition does not merge. A `Sequence` built in
  memory has no metadata, and each block keeps its own. A writer of a
  single-block format (LTA, FLIRT, an x5 node) takes the metadata of the
  block it writes. On read, a single-block file puts its metadata on the
  block too: the X5 converters pass the metadata of a single-node file
  to the block that they build, with `metadata=`.

## 10. The `metadata` field (M10) and the `metadata_fields` clash (M11)

The data model gets one optional field on each root:

```python
class Image(DataModelBase, kw_only=True):
    metadata: NoRepr[tx.Optional[Metadata]] = None


class Transformation(DataModelBase, kw_only=True):
    metadata: NoRepr[tx.Optional[Metadata]] = None
```

Both roots are keyword-only and compare by identity, so the field lands
last in every signature and needs no `NoEq`. Putting it on the roots,
rather than only on the formats, lets `from_instance` copy it by name
as a shared field, and lets an in-memory `Affine` or a resampled image
carry provenance. A format takes the field over under the private name
`_metadata`, with `Inherit("metadata")`, behind a public `metadata`
property, as it takes `data` and `transformations` over (the record
memo). A conversion of an image into another format converts its
metadata through `MetadataFormat.from_any`, and reports what the target
cannot store. On `main` today, the roots have no `metadata` field: each
format declares a private `_metadata` field of its own type and a
public `metadata` property, and the wiring of #287 adds the field to the
roots and the takeover to the formats.

Shared, not copied (decision Q3 of the maintainer). Copies of an image
or a transformation within a format share the metadata object
(`replace(image).metadata is image.metadata`), and the object stays
mutable, so a change made through one copy is seen by the others. To
change the metadata of one copy only, assign it new metadata:
`image.metadata = replace(image.metadata, description="smoothed")`.
nibabel copies the header instead, through `from_header` and
`header.copy()`, when it builds an image. A first version of this memo
copied the metadata in a field converter (`MetadataField`), and the
record design removed the converter, since a record is never changed in
place and sharing is therefore safe. A conversion and a derivation
build new metadata anyway.

Assigned values win over the metadata (#440, implemented on `main`).
The transformations of an image are not tied to its metadata. The
property `transformations` returns the list that the user assigned if
there is one, and otherwise the list decoded from the record, which it
caches. Assigning metadata clears only that cache of decoded values,
and never a list that the user assigned, which a later consolidation
encodes over the new record. The same holds for the transformations of
chains, for the matrix of the NIfTI and LTA affines, held as `data`, for
the ITK endpoints, for the X5 nodes, and for the levels and placements
of a pyramid (rules of the record memo, conformance checks 20 and P11).
The wiring of #287 follows the same rule.

Name clash. `Transformation.metadata_fields` means the meta-attributes
that define the encoding (`order`, `bound`, `coeff`). It is renamed
`encoding_fields`, without an alias, in the wiring of #287.

## 11. Format sketches

Pass 5a implements NIfTI, MGH, X5 and BIDS; the table below records the
plan for the other formats, and the part of the fifth pass that takes
each of them. Each format writes `SUPPORTED_FIELDS`, the module-level
`_decode` and `_encode`, a `from_raw` and a `consolidate` that call
them, and an override of `derive` only to scrub its record. The NIfTI
transformations (the NIfTI affines and fields, and the SPM, FNIRT,
NiftyReg and AFNI fields stored as NIfTI) hold `NiftiMetadata`, so they
get its vocabulary in pass 5a.

NIfTI (`io/common/nifti/_metadata.py`, with the slot helpers in
`_vocabulary.py`) stores `description` (`descrip`, 80 bytes), `sources`
(`aux_file`, one name of 24 bytes), `intent`, `space`, `display_range`
(`cal_min`, `cal_max`), the encoding directions (`dim_info`, an axis
without polarity), `slice_timing` (the `slice_*` slots),
`repetition_time` (`pixdim[4]`), `dtype` and the scaling. NIfTI has no
free-form store, so `extra` is not supported (open question 7). Header
floats are single precision and are decoded as the shortest decimal
(`0.3`, not `0.30000001`).

MGH (`io/common/mgh/_metadata.py`, with the tags in `_tags.py`) stores
the footer (`tr`, `te`, `ti` in milliseconds and `flip_angle` in
radians, a zero meaning "not recorded"), `history` in the `TAG_CMDLINE`
tags (decoded only when the whole tag stream parses; writing it
replaces the command lines and keeps every other tag), and `dtype`.
nibabel reads the footer as part of `MGHHeader`, but not the tags, which
`MghRaw.tags` holds as bytes. The metadata of an MGH file writes the
fixed header alone, so the footer and the tags reach a file only through
the image.

X5 (`io/transformations/x5/_metadata.py`) stores every field in the JSON
`Metadata` of a transform, under its sidecar key, and the other keys of
the object are `extra`. The transform that holds the vocabulary of the
file is the single transform of its first chain, or the single
transform of a file without chains; a file whose first chain composes
several transforms has no vocabulary of its own, and a field written to
it is lost. `Domain.Coordinates` is not decoded, because nitransforms
writes the kind of coordinates there (`"cartesian"`), not a space.

MRtrix and NRRD (pass 5b) write whatever has no dedicated key under its
BIDS name in `keyval` and `keyvalue`, which is what `mrconvert
-json_import` produces; `dw_scheme` becomes `bvalues`/`bvectors` in the
world frame, and NRRD uses the `DWMRI_*` convention and the
`measurement frame` rotation. LTA (pass 5b) takes `moving`/`fixed` from
the `src`/`dst` filenames and `description`/`history` from comments,
which its parser must then keep.

| Format | Record | `SUPPORTED_FIELDS` (highlights) | `extra` store / notes | Pass |
|---|---|---|---|---|
| MGH | `MghRaw(header, tags)` | TR/TE/TI/flip (ms→s, rad→deg; one scalar TR, else `approximated`), `history` (cmdline tag), `dtype` (`type`) | none | 5a |
| AFNI | `AfniRaw` | `history`, `channels` (`BRICK_LABS`), `repetition_time` (unit code), `slice_timing`, `space`, `creation_time` | remaining attributes; `_GENERATED`/`_PER_BRICK`/`_PER_GRID` rules move into `_encode` (count mismatch → `lost`) and `derive` | 5b |
| MINC (read-only) | `MincRaw` | `history`, acquisition/patient/study variable attributes | the rest of them | 5b |
| OME-Zarr images | none yet (step 4b-ii of pass 4); the prototype had `OmeZarrRaw` (abczarr `Multiscale`, `omero`, attributes) | `name`, `channels`, `display_range`, `dtype` (derived from the arrays) | the other group attributes | 5c |
| plain Zarr | `ZarrRaw` (the abczarr document of the array, with its attributes) | generic `Metadata` serialised into the attributes under `"brainhops"` (prototype) | same | 5c |
| ITK h5 (read-only) | `H5Raw` (the root strings in `H5Header`, the blocks) | `generated_by` (`ITKVersion`) | none | 5c |
| FLIRT (read-only) | `FlirtRaw` (the matrix) | none: the images `moving` and `reference` are options of the transformation, never stored, so `moving`/`fixed` are `lost` on write | none | 5c |
| TIFF/OME/ImageJ | `TiffRaw` (one series: `tags`, `ome_xml`, `imagej`) | `name`, `description`, `generated_by`, `creation_time`, `manufacturer*`, `channels` (by dialect) | ImageJ extras or plain tags; OME-XML round-trips whole, `_encode` patches `Name`/`Channel` only | open |
| Pillow | `PillowRaw` (one frame, with its `info`) | `description`, `creation_time` | text chunks, `exif`, `icc_profile` | open |
| OpenSlide (read-only) | `OpenSlideRaw` (`properties`, levels) | `description`, `manufacturer`, `objective_magnification` | `properties` | open |
| JPEG 2000 | `Jpeg2000Raw` | `description` (`comments`) | XML and UUID boxes | open |
| ITK tfm, ITK mat | `TfmRaw`, `MatRaw` (the blocks) | none | none | open |
| elastix | `ElastixRaw` (the parameter map, and the records of the initial transforms) | `sources` (`InitialTransformParameterFileName`) | the non-transform keys (the "base map" rule) | open |
| plain matrices | `MatrixAffineRaw` | none | none | open |
| M3Z | `M3zRaw` | `moving`/`fixed` (`image`/`atlas` fnames) | byte-exact otherwise | open |
| image stacks | `StackRaw` (the members, with the header of each) | not designed | not designed | open |
| N5 | `N5Raw` (`attributes.json`) | not designed | not designed | open |
| Imaris | `ImarisRaw` (the `DataSetInfo` parameters) | not designed (section 4.5) | not designed | open |
| Neuroglancer precomputed | `PrecomputedRaw` (the `info` file) | not designed | not designed | open |
| BigDataViewer | `BdvRaw` (the XML file) | not designed (section 4.5) | not designed | open |
| FreeSurfer `register.dat` | `RegisterDatRaw` (the subject, the matrix) | not designed | not designed | open |
| Convert3D affine | `C3dAffineRaw` | not designed | not designed | open |
| TrakEM2 | `TrakEM2Raw` (the XML file) | not designed | not designed | open |
| AFNI, MRtrix, NiftyReg and MINC affines | `AfniAffineRaw`, `MrtrixLinearRaw`, `NiftyRegAffineRaw`, `XfmRaw` | not designed | not designed | open |
| OME-Zarr fields, OME-NGFF 0.6 transformations (#433) | none yet | not designed | not designed | open |

The Zarr, OME-Zarr, FLIRT and ITK-H5 classes of the prototype are kept
out of pass 5a and wait for pass 5c; their code is in the history of
#310 (2390b07). The fourth pass has given records to Zarr (#460), FLIRT
and ITK-H5 (#452), so the classes of the prototype must be rebuilt over
those records, and the OME-Zarr images wait for their record. The
pyramids of TIFF, OpenSlide, JPEG 2000, N5, Imaris, precomputed volumes
and BigDataViewer, and the formats added since the fifth pass was split,
are not named by the split, so the part of the fifth pass that takes
them is open (open question 13). The rows marked "not designed" record
only what the record of the format holds; which of it is vocabulary is
decided with that part.

Future formats (GIFTI/CIFTI, TRK/TCK/TRX, CZI/LIF/ND2, Bruker, MRC,
FITS, MetaImage, BrainVoyager, Interfile, EEG coordsystem, NetCDF) each
add one class and, if they bring a concept two of them share, one
vocabulary field.

## 12. Backward compatibility and migration (M12)

Historical names are not kept as aliases, as the record memo decided:
the attributes `header`, `keyval`, `struct`, `tags` and `node`, the
parser mixins, and `Transformation.metadata_fields` are removed. This
breaks the public API, which is accepted, and the breaking changes are
announced with item 13 of the tensions of the record memo.

Constructor and writer keywords (M13). The format-specific metadata
inputs become fields of the metadata, which `save` and `consolidate`
take as overrides (section 6.2), without deprecation aliases. Since
the record design, these inputs are options of `consolidate` and of the
writers, and a few have moved into the record:

| Before | Where | Becomes |
|---|---|---|
| `like=` | `NiftiImage` and `MghImage`, `consolidate`, `to_nibabel` and `save` | stays a writer option, under the fields that changed |
| `descrip=`, `intent=`, other header names | `NiftiImage`, `**overrides` of `consolidate` and `to_nibabel` | `description=`, `intent=`; header names that the vocabulary does not cover stay overrides of the record |
| `tr=`, `te=`, `ti=`, `flip_angle=`, `fov=` | `MghImage`, `**overrides` of `consolidate` and `to_nibabel` | `repetition_time=` (s) and the other fields; `fov` stays in the record |
| the MGH tags | `MghRaw.tags`, in the record | `history=` / `raw.tags` |
| `keyval=` | `MrtrixImage` and the MRtrix warps, `consolidate` | `extra=` (BIDS keys land in vocabulary fields) |
| `keyvalue=` | `NrrdImage`, `consolidate` | `extra=` |
| `attributes=`, `view=` | `AfniImage` and the AFNI warps, `consolidate` | `extra=` / `raw.attributes`; `view` stays a writer option (encoding) |
| `datatype=` | `AfniImage`, `MrtrixImage`, `NrrdImage` and the AFNI warps, `consolidate` | open (open question 14) |
| the `info` dictionary | `PillowRaw.info`, in the record | `extra=` |
| `name` | `build_ome(...)`, a private helper of OME-Zarr | `name=` |

Passes. Pass 5a is #310 (the vocabulary, the consolidation of the
metadata of a format, the BIDS sidecar, NIfTI, MGH and X5) and #287
(the wiring into images, transformations, their copies, their
operations and their writers). Pass 5b brings the key/value and
FreeSurfer formats, and pass 5c Zarr, OME-Zarr, FLIRT and ITK-H5. The
part that takes each other format is open (section 11).

Tests. The test files below are added by #310 and #287.
`tests/test_datamodel_metadata.py` covers the vocabulary, the terms,
the loss policy and `derive`; `tests/test_io_metadata_base.py`
covers the contract of consolidation on a synthetic format, the
conversions and the BIDS sidecar; each format has a file
(`test_io_metadata_nifti.py`, `_mgh.py`, `_x5.py`) that decodes each
slot, encodes each field into a copy, and probes each value that a slot
holds approximately or not at all; `tests/test_io_metadata_matrix.py`
is the capability oracle of section 5. The conformance test of the
record memo gains five checks, numbered after check 20, which tests
that assigned values survive new metadata (#440):

21. `from_raw(raw).consolidate()` shares `raw`, so nothing is pending
    after a read, and `replace(md, **vocabulary).consolidate()` holds
    the value, decoded afresh.
22. A changed vocabulary field survives a save and a load of the
    metadata, and the other fields read back as before.
23. A field that the format has no slot for is reported as lost by
    `consolidate` and is absent from the copy, and `on_loss="raise"`
    raises `MetadataLossError`.
24. A conversion between formats with `from_instance` carries the
    vocabulary and resets `raw` (#287).
25. `img[index]` and `img.reslice()` keep the class and the record of
    the metadata; a `SPATIAL` field is cleared, an `AXIS` field
    selected, and the `FILE` fields kept (#287).

## Decisions

- M1. Geometry, axes, spatial and temporal units, transformations, the
  transform kind and storage encoding (byte order, layout, compression)
  are never metadata. The element type on disk (`dtype`) and the
  intensity scaling (`scale_slope`, `scale_intercept`) are: they are
  user-visible choices that may differ from the loaded array.
- M2. Three layers (`Magic` vocabulary fields, a format-private `raw`
  record, an `extra` dict) in the classes of the record memo: `Metadata`
  (the vocabulary and `extra`, no record, no input or output, equality
  by value) and `MetadataFormat` (the dispatcher and base of the
  formats, `raw`, `SUPPORTED_FIELDS`, `from_raw`, `consolidate`,
  equality by identity), with one `<X>Metadata` per format.
- M3. The vocabulary is BIDS names in snake_case with BIDS units, 39
  fields declared in seven groups (`ProvenanceVocabulary`,
  `MRIVocabulary`, `DiffusionVocabulary`, `DisplayVocabulary`,
  `StorageVocabulary`, `MicroscopyVocabulary`, `TransformVocabulary`)
  that `Metadata` inherits, each field tagged with a scope. A free-text
  field with known terms is `Union[<Enum>, str]`; `data_unit` is a
  `Unit` when the units module parses it; encoding directions are
  `EncodingDirection` vectors; the storage field is `dtype`.
- M4. `Magic` is used for the vocabulary fields and for
  `from_instance` conversion, in explicit code on `Metadata` and
  `MetadataFormat`. There is no metaclass, no class keyword, no
  polymorphic `format` field and no registry.
- M5. A field that a format has no slot for decodes as `None`. Each
  `<X>Metadata` declares its supported fields in one constant,
  `SUPPORTED_FIELDS`, which one function reads
  (`unsupported_values`): a conversion reports a lower bound of the
  loss from it, and `consolidate` reports the exact loss.
- M6. Overlay with eager decoding: `from_raw` decodes the vocabulary,
  and `consolidate(*, inplace=False, on_loss=None, **fields)` encodes
  the fields that differ from a fresh decode of the record into a copy
  of the record, reports what no slot took, and decodes the copy
  afresh. `None` clears a slot. Fields that the data model owns are
  decided by the format object, and `dtype` follows the precedence of
  section 6.2: an override on save or consolidate, then
  `metadata.dtype`, then the type of the data.
- M7. `ConversionReport`; a conversion reports the lower bound from
  `SUPPORTED_FIELDS`, and a consolidation, which every write does,
  reports the exact loss; `on_loss` is `ignore`, `warn` (the default)
  or `raise`, or a report to fill; `metadata.to(cls, *, on_loss)` is the
  explicit conversion.
- M8. `extra` passes unknown keys through to any free-form store. The
  BIDS JSON sidecar is a metadata format, `BidsSidecar`, with the
  functions `from_bids` and `to_bids`.
- M9. Propagation is driven by scope and owned by the image operations,
  which describe what they did with `Indexed` or `Resampled`; `derive`
  applies the rules in one explicit loop, keeps the class and the
  record, and a format scrubs its record in an explicit override.
  Composition does not merge, and a single-block file puts its metadata
  on the block too. A pyramid format and its levels share one metadata
  object (the record memo); how its vocabulary reaches the levels is
  open.
- M10. `metadata: Optional[Metadata]` on `Image` and `Transformation`,
  taken over by each format with `Inherit("metadata")`. Copies within a
  format share the metadata object, which stays mutable; assigning
  `replace(md, ...)` changes one copy only. Assigning metadata clears the
  cache of the decoded transformations, never a list that was assigned
  (#440).
- M11. `Transformation.metadata_fields` becomes `encoding_fields`,
  without an alias.
- M12. Three parts of the fifth pass (5a, 5b, 5c), no deprecated
  aliases, the matrix test as the capability oracle, and the
  conformance checks 21 to 25.
- M13. Writer and constructor keywords (`tr=`/`te=`..., `keyval=`,
  `keyvalue=`, `attributes=`, `info=`, `tags=`) become fields of the
  metadata, which `save` and `consolidate` take as overrides, without
  aliases.

## Addendum: the record design (#415)

The format-record memo, `format-records.md`, replaced the parser mixins
with a record per format, and the fifth pass adapted this memo to it.
The decisions below are written into the sections above; this list
records which decisions of the prototype they replace.

- M2. `FileBasedMetadata`, `OpaqueMetadata` and `MetadataParser` are
  replaced by `MetadataFormat` and the stateless adapter bases of the
  record memo. The record is a field of the metadata of a format, never
  of `Metadata`.
- M4 and M5. The metaclass `MetadataMeta`, the class keyword
  `supports=`, `supported_fields`, `unsupported_fields`, `ALL`, the
  `UNSUPPORTED` sentinel, `Maybe` and the `format` discriminant are
  removed (maintainer decisions Q2, answer b): a field without a slot
  decodes as `None`, and `SUPPORTED_FIELDS` declares the supported
  fields.
- M6. The read-time snapshot, `update_raw`, `update_from_raw`, `to_raw`,
  `check_raw`, `check_writable`, `writable` and the hooks
  `_decode_raw`/`_encode_raw` are replaced by eager decoding in
  `from_raw` (Q1) and the comparison with a fresh decode in
  `consolidate`, with module-level `_decode` and `_encode` in each
  format. `data_type` is renamed `dtype` (Q5), and `consolidate` takes
  any field as an override. `save` forwards its keyword arguments to
  `consolidate`, so the writer option `dtype=` is the metadata field
  and not a separate option.
- M7. `collect_loss_reports` and `ConversionReport.merged` are removed;
  a caller passes a report as `on_loss=` and merges.
- M9. The registry of propagation handlers (`propagates`, `propagate`,
  `propagate_raw`) and the `Operation` base are removed; `derive` is one
  loop, and derived metadata shares the record instead of a deep copy.
- M10. `MetadataField` and its copying converter are removed: metadata
  is shared by copies and mutable (Q3).
- M11, M12 and M13. No deprecation aliases.
- Section 6.1. The historical accessors are removed, and `struct`
  became `raw` everywhere.

## Open questions for the maintainer

1. Where the field lives (M10). Closed: on the roots of the data model,
   taken over by the formats with `Inherit`.
2. Default loss policy (M7). `warn` with one warning per conversion or
   consolidation, or silent `ignore` with a report available on
   request. Recommendation: `warn`; a converter that drops the slice
   timing of a user without a word is the behaviour this design leaves.
3. Change-detecting write (M6). Closed: the snapshot is replaced by the
   comparison with a fresh decode.
4. Diffusion frame (M3). World/RAS unit vectors (MRtrix convention),
   with NRRD and BIDS codecs rotating via the affine; or voxel-axis
   vectors (BIDS/FSL convention). Recommendation: world; it is the only
   frame that survives a reorientation without touching metadata.
5. Polymorphic `format` discriminant (M4). Closed: dropped.
6. `intent` vocabulary (M3). Closed: the NIfTI intent names are the
   canonical set, as `IntentEnum` (`Union[IntentEnum, str]`, so an
   unknown name is kept).
7. NIfTI extras. Unsupported (reported as lost), or written into a
   NIfTI extension (a JSON extension with a brainhops-chosen ecode) that
   only brainhops reads. Recommendation: unsupported in v1; an extension
   is a later opt-in writer option, because other tools would carry an
   opaque blob.
8. Sidecar integration (M8). Codec only (`BidsSidecar`, `from_bids`,
   `to_bids`), or also `io.load(path, sidecar=True)` /
   `io.save(..., sidecar=True)` reading and writing `<stem>.json` next
   to the image. Recommendation: the codec in pass 5a, and `sidecar=`
   where it is used.
9. Rename `metadata_fields` (M11). Closed: `encoding_fields`, without
   an alias.
10. Open PRs. Closed: moot after the record design.
11. Deprecation window (M12). Closed: no aliases.
12. Levels and the pyramid (M9). Open. The record design gives a
    pyramid format one metadata object, which the pyramid and each of
    its levels share by identity. This memo first gave each level
    metadata derived through a `Resampled` record, so that a direction
    in voxel axes follows the coarser sampling. Which of the two holds
    for the vocabulary, or whether a level derives its fields on access
    from the shared object, is not decided. It concerns every pyramid
    format (TIFF, OpenSlide, OME-Zarr, JPEG 2000, N5, Imaris,
    precomputed volumes, BigDataViewer), and only `SPATIAL` and `AXIS`
    fields depend on the answer.
13. The part of the fifth pass that takes each format that the split
    does not name. Open. The split names NIfTI, MGH, X5 and BIDS (5a),
    LTA, AFNI, MRtrix, NRRD and MINC (5b), and Zarr, OME-Zarr, FLIRT and
    ITK-H5 (5c). It does not name M3Z, the ITK text and MATLAB formats,
    elastix, the plain matrices, the AFNI, MRtrix, NiftyReg and MINC
    affines, `register.dat`, Convert3D, TrakEM2, TIFF, Pillow,
    OpenSlide, JPEG 2000, N5, Imaris, precomputed volumes,
    BigDataViewer, image stacks, the OME-Zarr fields or the OME-NGFF
    transformations (section 11).
14. The option `datatype=` of AFNI, MRtrix and NRRD (M13). Open. The
    field `dtype` of the metadata replaces the option `dtype` of NIfTI
    and MGH (decision Q5), but these three formats name the same option
    `datatype=`. It is not decided whether pass 5b renames it `dtype`,
    so that it becomes the metadata field, or keeps it as a writer
    option.
