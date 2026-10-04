# Design: format metadata and cross-format metadata conversion

Status: design, with a phase-1 prototype (draft PR #287). Answers #233.
Builds on the io model (`FileBasedImage`, `FileBasedTransformation`,
`from_other`/`from_instance` in `io/base/_base.py`), on `DataModelBase`
(#97, `bagof.magic`), and on the units memo (`units-polymorphism.md`) for
where `Magic` polymorphism does and does not fit. Decisions are tagged
`M1`..`M13` and collected at the end, followed by the open questions.

> **Prototype note.** The prototype on this branch covers the framework
> (`brainhops/datamodel/metadata.py`, whose module docstring documents
> the per-format hooks), the BIDS sidecar codec
> (`brainhops/io/metadata/bids.py`), the `metadata` field on `Image` and
> `Transformation`, and the formats of batches 2, 4 and 5 that exercise
> the design most: `NiftiMetadata` (`NiftiImage` and every NIfTI-based
> transformation), `MghMetadata`, `X5Metadata`, the ITK and FLIRT
> classes, and `ZarrMetadata`/`OmeZarrMetadata`. The user guide is
> `docs/start/metadata.md`. Deviations are recorded in notes like this
> one, at the section they concern. Two reviews of the prototype changed
> decisions. The first: M6 (a geometry hook for derived fields, and a
> lazy part of the raw record), M9/M10 (metadata is copied, not aliased,
> and a single-block file puts its metadata on the block too). The second
> (the inline review of PR #287): the class hierarchy is `Metadata` ->
> `FileBasedMetadata` -> `<Fmt>Metadata` (M2), the vocabulary is
> declared in six group mixins (M3), `supports=` gives
> `supported_fields` (M5), lazy fields are descriptors (M6), one
> vocabulary for the raw record (`from_raw`, `update_from_raw`,
> `update_raw`; M6), conversion is `metadata.to(cls)` (M7), the field is
> `MetadataField[...]` (M10), free-text fields with known terms are
> enums, encoding directions are vectors, and the element type is
> metadata (`data_type`, M1).

The problem, in one sentence: every file format keeps its non-spatial
metadata under a different name and type (`header`, `keyval`, `tags`,
`info`, `ome_xml`, `imagej_metadata`, `attributes`, `parameter_map`,
`struct`, ...), nothing is shared across formats, and `from_instance`
deliberately drops all of it when a file is converted (NIfTI to MGH keeps
the data model only; `_foreign_format_fields`, `_base.py:617`).

## 1. What exists today

Facts, from the inventory of the current branch:

- No metadata slot on the data model (`Image`, `SingleScaleImage`,
  `MultiScaleImage`, `Transformation`); the x5 docs say so explicitly.
- Each parser mixin holds a raw structure as the source of truth for
  geometry (nibabel headers, `MrtrixHeader`, `AfniHeader`, `NrrdHeader`,
  `X5Header`/`X5Node`, `LtaStruct`, `M3zStruct`, `ItkStruct`, elastix
  `parameter_map`, TIFF/Pillow/OpenSlide flat fields). MINC keeps only
  `dimensions`.
- Same-format round trip is good for MRtrix, AFNI, NRRD, MGH, x5, LTA,
  M3Z, elastix; poor for NIfTI (fresh header unless `like=`; `slice_*`,
  `cal_*`, `aux_file`, extensions dropped), OME-TIFF (only `Name`,
  `Channel.Name`), OME-Zarr (no `omero`, no name), Pillow (only
  `icc_profile`, `exif`).
- Six formats have free-form key/value stores (MRtrix `keyval`, NRRD
  `keyvalue`, AFNI `attributes`, x5 `Metadata`/`attrs`, TIFF
  `imagej_metadata`/`tags`, Pillow `info`); none is carried across.
- Units disagree: MGH TR/TE/TI in ms and flip angle in radians, AFNI
  codes the TR unit, BIDS uses seconds and degrees.
- `Transformation.metadata_fields` already exists and means "the
  meta-attributes that define the encoding" (`order`, `bound`, `coeff`).

## 2. Goals, non-goals, and what stays out

Goals:

1. One `metadata` field, of one base type, on every file-based image and
   transformation, replacing all the format-specific attributes above.
2. A small common vocabulary with BIDS names and units, as `Magic`
   fields, so that the data model's `from_other` carries it across
   formats and `bagof` gives it conversion, repr, docs and equality.
3. A faithful raw structure per format, so that read-then-save in the
   same format loses nothing it does not lose today (and NIfTI loses
   less).
4. Explicit, reported loss: a format can declare that it *cannot* hold a
   piece of information, and converters and writers say what was lost.

Non-goals (v1): parsing DICOM; a units-aware quantity type for metadata
values (the vocabulary fixes one unit per field and stores plain
numbers; a quantity type can replace that later without renaming
fields); automatic propagation through every data-model operation
(section 9 defines the rules; wiring them into `resample` and friends is
a follow-up).

**What stays OUT of metadata (M1).** Anything the data model already
represents is not metadata and never appears in the vocabulary:

- geometry: affines, voxel sizes, origins, direction cosines, the time
  axis origin (NIfTI `toffset`, AFNI `TAXIS_FLOATS[0]`) and step (NIfTI
  `pixdim[4]`), LTA/M3Z `src`/`dst` volume geometries;
- axes, their kinds and names, spatial and temporal units (`xyzt_units`,
  NRRD `space units`/`kinds`, OME `*Unit`, TIFF resolution tags);
- transformations and coordinate systems, including the transform
  *kind* (ITK `type`, LTA `type`, M3Z `type`, x5 `Type`/`SubType`,
  elastix `Transform`), which the transformation class itself encodes;
- storage encoding: byte order, intensity scaling (`scl_slope`,
  `BRICK_FLOAT_FACS`, MRtrix `scaling`, MINC `image-min/max`), layout,
  compression, data file names. These are writer options or raw-record
  content, not descriptive metadata. The *element type* on disk is the
  exception (`data_type`, section 4.4): it is a user-visible choice that
  may differ from the type of the loaded array (a scaled `int16` file
  loads as floats), so it is metadata; its byte order is not.

The raw structure still *contains* these (a nibabel header has `pixdim`),
and the writer still overrides them from the data model, exactly as
today. The vocabulary simply never exposes them.

## 3. Representation and layering (M2)

Answer to issue question 1: both, in one object, whose class hierarchy
mirrors the images' (`Image` -> `FileBasedImage` -> `NiftiImage`). A
file format's metadata has three layers:

1. **Common vocabulary**: `Magic` fields, declared in six group mixins
   (section 4) that the root class `Metadata` inherits, with BIDS names
   and units. Every metadata class has them.
2. **Raw record** `raw`: one format-private field whose type is the
   format's faithful record (the nibabel `Nifti1Header`,
   `MrtrixHeader`, `LtaStruct`, ...). Read-then-write in the same format
   goes through it. It is never copied across formats. It is declared by
   `FileBasedMetadata`, with the read-time snapshot (section 6).
3. **Extras** `extra: Dict[str, Any]`: a free-form namespace for keys
   the vocabulary does not cover. It *is* copied across formats, into
   whatever free-form store the target has (MRtrix `keyval`, NRRD
   `keyvalue`, AFNI `attributes`, x5 `Metadata`, ImageJ metadata, PNG
   text chunks), and reported as lost where there is none (NIfTI, MGH,
   LTA, ITK, FLIRT).

This mirrors nibabel's `Nifti1Header` (typed accessors over a structured
record) and brainhops' own images (a data model plus a parser mixin that
keeps the raw file state).

**Layering.** `Image` and `Transformation` type their field as the
generic `Metadata`, so everything format-agnostic lives in the data
model, and only format subclasses live under io:

- `brainhops/datamodel/metadata.py`: `Unsupported`/`UNSUPPORTED`,
  `Maybe`, the `Bids`/`Scope` annotations, the six vocabulary groups,
  `Metadata` (the root and the generic, lossless metadata, with
  `extra`, `to`, `derive` and the BIDS codec entry points),
  `FileBasedMetadata` (`raw`, the snapshot, `from_raw`,
  `update_from_raw`, `update_raw`, `check_writable` and the hooks),
  `OpaqueMetadata` (nothing supported), `MetadataField`,
  `ConversionReport` and the loss policies.
- `brainhops/_core/properties.py`: `Lazy` and the `LazyField`
  descriptor (section 6), next to `lazyproperty`.
- `brainhops/datamodel/enums.py`: the enums of the known terms (section
  4).
- `brainhops/io/metadata/`: the BIDS sidecar codec (it reads files).
- `brainhops/io/images/<fmt>/_metadata.py`,
  `brainhops/io/transformations/<fmt>/_metadata.py`: one
  `<Fmt>Metadata(FileBasedMetadata)` per format, next to its parser.

```python
# brainhops/datamodel/metadata.py
class Metadata(
    DataModelBase,
    ProvenanceMetadata,
    MRIMetadata,
    DiffusionMetadata,
    DisplayMetadata,
    MicroscopyMetadata,
    TransformMetadata,
    polymorphic=True,
    kw_only=True,
):
    """Common vocabulary + extras; the root, and the generic metadata."""

    format: str = "generic"  # discriminant, a real field (M4)
    extra: Maybe[tx.Dict[str, tx.Any]] = Factory(dict)
    # supported_fields / unsupported_fields / lazy_fields:
    # ClassVars computed from the class keywords (M5, 6, 6.2)

    def to(self, cls=None, *, on_loss=None, report=None, **values): ...  # M7
    def derive(
        self, *, grid_changed=False, grid_map=None, volumes=None, step=None
    ) -> tx.Self: ...  # M9


class FileBasedMetadata(Metadata):  # mirrors FileBasedImage
    raw: NoRepr[NoEq[tx.Any]] = None  # format-private record (M6)
    # read-time snapshot, field name -> decoded value
    _snapshot: NoRepr[NoEq[tx.Dict[str, tx.Any]]] = Factory(dict)

    @classmethod
    def from_raw(cls, raw, *, image=None, **values) -> tx.Self: ...
    def update_from_raw(self, raw, *, image=None) -> tx.Self: ...
    def update_raw(self, raw=None, *, image=None, report=None, force=()): ...
    def check_writable(self, *, image=None) -> "ConversionReport": ...  # M7


class OpaqueMetadata(FileBasedMetadata, supports=()):  # nothing, anywhere (M5)
    format: tx.Literal["opaque"] = "opaque"
    raw: None = None
```

`Metadata` is the root *and* the generic instance: it is what in-memory
objects carry and what the BIDS sidecar codec reads and writes (no
`raw`, nothing unsupported, so `Nifti -> Metadata -> Mrtrix` loses
exactly what `Nifti -> Mrtrix` loses), and it selects a format class on
`format` (`Metadata(format="nifti")` builds a `NiftiMetadata`; an
unknown format builds a `Metadata`). `FileBasedMetadata` is what a
format subclasses: everything that needs a raw record lives there, so
the generic metadata has none, not even a `None` one. `OpaqueMetadata`
is the convenience base of the formats that keep nothing at all, not
even in memory (ITK `.tfm`/`.mat`, matrix text): every field is
unsupported and a read-then-save loses nothing because nothing is there.
A format that keeps something, if only in memory, is a
`FileBasedMetadata` with a `supports=` list: FLIRT's `.mat` stores no
metadata, but the reader is given the moving and reference images, whose
paths it keeps as `moving`/`fixed` (lost, and reported, on write), so
`FlirtMetadata` is not opaque.

> **Prototype note.** `raw` is excluded from `==` as well as from `repr`
> (`NoRepr[NoEq[tx.Any]]`, or `Annotated[..., NoRepr(), NoEq()]`): two
> metadata objects are equal when their vocabulary and `extra` are. The
> snapshot is the field `_snapshot` (constructor keyword `snapshot=`),
> a `dict` of the decoded values (section 6). A first prototype had a base
> `FormatMetadata` with `raw` and the snapshot, and `Metadata` as one of
> its subclasses; the review of PR #287 inverted that, as above, so
> that the generic metadata carries no raw record and an unknown format
> falls back to the root without a catch-all `on=` predicate.

## 4. Vocabulary (M3)

Answer to issue question 2. Names are the BIDS keys in snake_case
(`RepetitionTime` becomes `repetition_time`); the BIDS spelling is kept
as field metadata (`tx.Annotated[..., Bids("RepetitionTime")]`) for the
sidecar codec. Units are the BIDS units: seconds, degrees, tesla, mm,
s/mm². Where BIDS has no key, the name is chosen to read like one.

Every field is `Maybe[T] = Union[T, None, Unsupported]` (section 5) and
carries a `Scope` tag used by propagation (section 9): `file` (about this
file), `acquisition` (invariant under resampling), `grid` (tied to the
voxel grid), `volume` (one entry per volume/channel).

**Groups.** Each subsection below is a `Magic` mixin that declares its
fields, and nothing else: `ProvenanceMetadata` (4.1), `MRIMetadata`
(4.2), `DiffusionMetadata` (4.3), `DisplayMetadata` (4.4),
`MicroscopyMetadata` (4.5), `TransformMetadata` (4.6). `Metadata`
inherits all six (the "all metadata" class), and a field is defined in
exactly one group, never on `Metadata` itself. The groups are an
organisational device, and the memo says so plainly. They buy one place
per topic for the docs (each group is an API entry with its fields'
`Doc`/`Bids`/`Scope`), `supports=(ProvenanceMetadata, "echo_time")` in a
format declaration, and a per-group rule in the codecs (BIDS keeps
`DiffusionMetadata` in `.bval`/`.bvec` files, not in the sidecar). They
do not shrink a format class (every format still exposes every field, so
that each can answer `UNSUPPORTED`), and they do not narrow types (`def
f(m: DiffusionMetadata)` accepts any metadata). `VOCABULARY` (the field
names, group by group) and `GROUPS` (group class -> its field names) are
module constants.

**Known terms.** A free-text field with a list of known terms is typed
`Maybe[Union[<Enum>, str]]`: a known term is held as the enum member (a
`StrEnum`, so it is still a string and compares equal to its value), any
other string stays a string, so the vocabulary is documented without
being closed. The enums live in `brainhops.datamodel.enums`: `Space`,
`Intent`, `Manufacturer`, `IlluminationType`, `ContrastMethod`. Truly
free text (`name`, `description`, `history`, file references) stays
`str`.

> **Prototype note.** `bagof` collects fields from `Magic` mixins, with
> two constraints: a mixin must be a `Magic` subclass, and it must say
> `convert=True` itself (a field keeps the options of the class that
> declares it; the groups share a small base for that). Inherited fields
> come out of `fields()` in *reverse* MRO order, so every loop over the
> vocabulary (repr, `changed_fields`, `derive`, conversion, the codecs)
> iterates `VOCABULARY`, never `fields(Metadata)`. `bagof` does not
> coerce a string into the enum member of a `Union[Enum, str]` (a string
> already satisfies the union), so those fields carry a small converter.

### 4.1 Core and provenance (`file`)

| Field | Type / unit | BIDS | Native sources |
|---|---|---|---|
| `name` | str | — | OME `Name`, TIFF `DocumentName`, OME-Zarr multiscale `name` |
| `description` | str | — | NIfTI `descrip`, TIFF `ImageDescription` (plain), NRRD `content`, OpenSlide `openslide.comment`, JP2 `comments`, x5 `Metadata.description` |
| `history` | tuple[str] | — | MRtrix `command_history`, AFNI `HISTORY_NOTE`, MGH `tags` (cmdline), MRtrix `.txt` comments, LTA comments |
| `generated_by` | tuple[GeneratedBy(name, version, description)] | `GeneratedBy` | TIFF `Software`, MRtrix `mrtrix_version`, ITK h5 `ITKVersion`, x5 `Format`/`Version`, AFNI `TYPESTRING` |
| `creation_time` | datetime | — | TIFF `DateTime`, AFNI `IDCODE_DATE`, EXIF |
| `sources` | tuple[str] | `Sources` | BIDS-style provenance inputs only: NIfTI `aux_file`, elastix `InitialTransformParameterFileName`, NRRD/MRtrix none |
| `space` | `Space` or str | `SpatialReference` | NIfTI `sform_code` name, AFNI `TEMPLATE_SPACE`, NRRD `space`, x5 `Domain.Coordinates` label |
| `intent` | `Intent` or str (the NIfTI intent names) | — | NIfTI `intent_code/name`, AFNI `BRICK_STATSYM`, NRRD `kinds` (non-spatial) |

`space` is a *label* (`"MNI152NLin6Asym"`, `"scanner"`, `"orig"`); the
coordinate system itself stays in the data model. `Space` lists the
NIfTI code names (`scanner`, `aligned`, `talairach`, `mni`, `template`),
the BIDS standard templates and the BIDS non-standard spaces; `Intent`
lists the NIfTI intent names as `nibabel` spells them. `intent` is
descriptive; the NIfTI reader still reads `intent_code` from the raw
record to type axes, and the writer derives the code from the axes first
and from `intent` second (a *geometry-derived* field for NIfTI, section
6.2). The LTA/M3Z image references are `moving`/`fixed` (4.6), not
`sources`.

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
| `phase_encoding_direction` (`grid`) | `EncodingDirection` | same | MRtrix keyval, NIfTI `dim_info` |
| `total_readout_time`, `effective_echo_spacing` | s | same | keyval |
| `slice_encoding_direction` (`grid`) | `EncodingDirection` | same | NIfTI `dim_info`, MRtrix keyval |
| `slice_timing` (`grid`) | tuple[float] s | `SliceTiming` | NIfTI `slice_code/start/end/duration` (expanded), AFNI `TAXIS_OFFSETS`, MRtrix keyval |
| `multiband_acceleration_factor` | int | same | MRtrix keyval |

The TR is in the vocabulary even though the time step is geometry: the
two coincide for a plain fMRI series but not for a multi-echo stack or a
sparse acquisition, and MGH/AFNI store the TR without any time axis.
Formats that only have a time step treat `repetition_time` as
geometry-derived (6.2).

**Encoding directions.** Three representations were compared. (a) The
BIDS strings `i`/`j`/`k` with an optional `-`: trivial codecs, but tied
to the on-disk axis order, broken by any in-memory permutation or flip,
and unable to say anything after a resampling that is not axis-aligned.
(b) The axis names of the image's voxel system (`"-y"`): survives a
permutation if the axis objects are kept, still axis-bound. (c) An
oriented unit vector in a named coordinate system: survives any linear
map, and the codecs snap it back to an axis. (b) is (c) with an
axis-aligned vector in voxel space, so (c) is chosen:
`EncodingDirection(vector, space=None)`, where `space=None` means the
image's voxel (array) axes, the BIDS frame, and a label (a `Space`)
means a world space. A BIDS string is accepted wherever a direction is
(`metadata.phase_encoding_direction = "j-"` stores `(0, -1, 0)`), a
direction compares equal to its BIDS string, and `to_bids()` gives the
string back, or `None` when it is along no voxel axis. BIDS, MRtrix and
NIfTI `dim_info` store an axis: they round-trip an axis-aligned
direction exactly (NIfTI drops the polarity, approximated, as before)
and report any other one as lost. A JSON store (x5, plain Zarr) writes
it as `{"Vector": [...], "Space": ...}` and reads it back. The slice
timing stays a tuple indexed along the slice encoding direction.

### 4.3 Diffusion (`volume`)

| Field | Type / unit | Native sources |
|---|---|---|
| `bvalues` | (N,) float, s/mm² | MRtrix `dw_scheme` col 4, NRRD `DWMRI_b-value` x gradient norms, BIDS `.bval` |
| `bvectors` | (N,3) float, unit vectors in **world (RAS) coordinates** | MRtrix `dw_scheme` cols 1-3 (already world), NRRD gradients via `measurement frame`, BIDS `.bvec` via the voxel-to-world rotation |

The names carry no `diffusion_` prefix: no format we read has b-values or
b-vectors of another kind (NRRD `DWMRI_b-value`, MRtrix `dw_scheme` and
BIDS `.bval`/`.bvec` are all diffusion), and the group says the rest.

The frame convention is the lossy spot in diffusion metadata; the
codecs rotate into and out of world coordinates using the image's
affine, which is why `_decode`/`_encode` take the data model object too.

### 4.4 Display and channels (`volume`)

| Field | Type | Native sources |
|---|---|---|
| `display_range` | (min, max) float | NIfTI `cal_min/max`, AFNI `BRICK_STATS` (first volume) |
| `channels` | tuple[Channel(name, color, display_range, unit)] | AFNI `BRICK_LABS`, NRRD `labels`, ImageJ `Labels`/`LUTs`/`Ranges`, OME `Channel.Name/Color`, OME-Zarr `omero.channels` |
| `data_unit` | `Unit` or str | NRRD `sample units`, OME channel unit, MINC `units` |
| `data_type` (`grid`) | `numpy.dtype` (native order) | NIfTI `datatype`, MGH `type`, MRtrix/NRRD `datatype`/`type`, Zarr array `dtype`, ITK `precision` |

`Channel.color` is an RGBA hex string; LUT arrays stay in `extra`.

`data_unit` is a `Unit` whenever `brainhops.datamodel.units` (backed by
`pint` since #291) parses the name: compound units (`"mm/s"`),
intensity, angle and field units, and the arbitrary unit (`"a.u."`,
`"au"`) all parse. Only a name that does not parse (`"mm2/s"`) stays a
string, so that a file with an odd unit still reads. The type therefore
stays `Maybe[Union[Unit, str]]` with a lenient converter (`Unit(name)`,
the string on `ValueError`) rather than `Maybe[Unit]`: a strict field
would refuse real files, and declaring `Unit` while holding a string
would lie to type checkers. Units compare as units
(`Unit("a.u.") == Unit("au")`), not as strings. A codec writes a `Unit`
as its symbol (`Unit("a.u.").symbol == "a.u."`, `"mm / s"`, `"ms"`),
which parses back to the same unit, and an unparsed name as it was read.

`data_type` is the element type of the data as stored, which may differ
from the type of the loaded array (a scaled `int16` file loads as
floats). It is the type a reader saw, and the type a writer prefers
(6.2). Its byte order is storage encoding (M1), so it is held in native
order; endianness stays in the raw record. BIDS has no key for it; the
name follows NIfTI/MRtrix "datatype". It is `grid`-scoped, so that
`derive(grid_changed=True)` clears it (a resampled label map is not
forced back to `uint8`), while a crop or a volume selection keeps it. It
is not geometry-derived (the data model does not own it), except where
the format stores no element type but the array's own (Zarr, OME-Zarr).

> **Prototype note.** `data_type` is wired for NIfTI (`datatype`), MGH
> (`type`; a type it cannot store is approximated by the nearest) and
> Zarr/OME-Zarr (derived from the arrays). ITK `precision` stays in the
> raw record (`ItkStruct.precision`) for now; MRtrix and NRRD come with
> their batch.

### 4.5 Microscopy (`acquisition`)

Deliberately small until CZI/LIF/ND2/Imaris/BDV readers exist:
`objective_magnification` (float; OpenSlide `objective-power`, OME
`Objective.NominalMagnification`), `objective_numerical_aperture`,
`illumination_type` and `contrast_method` (`IlluminationType` and
`ContrastMethod`, the OME enumerations, or any other string).
Everything else goes to `extra` until two formats agree on it.

### 4.6 Transformation-specific (`file`)

| Field | Type | Native sources |
|---|---|---|
| `moving` / `fixed` | str (file reference) | LTA `src/dst filename`, M3Z `image/atlas fname`, FLIRT `src`/`ref` (user-given, never stored), ANTs convention |
| `input_space` / `output_space` | `Space` or str label | x5 `Domain.Coordinates`, LTA (derived from `type`), OME-Zarr coordinate system names |

`moving`/`fixed` are deliberately *not* `source`/`target`: the data model
warns that brainhops' `input`/`output` are the inverse of the imaging
direction, and `moving`/`fixed` are unambiguous about which image is
which. `history`, `generated_by`, `description` and `sources` are shared
with images.

### 4.7 Not in the vocabulary, and why

Field of view (derived from geometry), intensity scaling and AFNI
per-volume stats (raw), NIfTI `intent_p1..p3` (raw; FNIRT/NiftyReg read
them from the raw record), byte order and compression (encoding),
elastix resampler keys (raw, written back by `transformation_to_map`),
EXIF/ICC blobs (`extra["exif"]`, `extra["icc_profile"]`), OpenSlide
`bounds`/`background_color`/associated images (`extra`), JP2 boxes (raw).
37 fields in all; adding one needs two formats that carry it natively,
or one format plus a BIDS key, and it goes into one group. A first
draft kept the element type out (ITK `precision` as "encoding"); it is
`data_type` now (4.4).

## 5. The `UNSUPPORTED` sentinel (M5)

A format class must be able to say "this format cannot store this", and
that must be different from "nobody set it".

```python
# brainhops/datamodel/metadata.py
class Unsupported:
    """The format cannot store this field. Singleton, falsy, not None."""

    __slots__ = ()

    def __bool__(self):
        return False

    def __repr__(self):
        return "UNSUPPORTED"

    def __reduce__(self):
        return "UNSUPPORTED"  # pickles to the singleton


UNSUPPORTED = Unsupported()
T = tx.TypeVar("T")
Maybe = tx.Union[T, None, Unsupported]
```

Three-valued semantics, per field:

| Value | Meaning | Converter (`from_instance`) | Writer |
|---|---|---|---|
| `None` | unknown / absent | copies nothing | writes nothing |
| a value | known | copies it, or reports it lost | writes it, or reports it lost/approximated (section 7) |
| `UNSUPPORTED` | this format has no slot for it | source side: treated as `None`; target side: incoming value is **lost** and reported | nothing to write |

Rules:

- **The sentinel is a value**, so the vocabulary can *take* it: a field
  of a format that has no slot for it holds `UNSUPPORTED` as its
  default, and `repr` shows it (it is not `None`), which is what one
  wants when inspecting a format class.
- **Declaration is class-level and compact.** Redeclaring ~30 defaults
  per format is not acceptable (LTA, ITK, FLIRT, MGH and Pillow support
  a handful of fields each). The class keyword `supports=` lists the
  vocabulary fields a format can store; the class machinery (a
  metaclass, see the note below) turns every other vocabulary field into
  one with default
  `UNSUPPORTED` (by rewriting the field default before `Magic` builds
  the class). `supports` is chosen over `unsupported=` because the
  short list is the honest one for most formats, and because a
  vocabulary field added later is unsupported everywhere until a format
  opts in, which is the safe default and what makes the vocabulary easy
  to extend. It takes field names, vocabulary groups (all their fields:
  `supports=(ProvenanceMetadata, "echo_time")`), or `ALL`
  (`MrtrixMetadata`, `X5Metadata`); omitted, a subclass keeps its
  parent's (`Metadata`, the root, supports everything). The
  declaration gives `supported_fields`, a `ClassVar` frozenset of the
  fields (and `"extra"`) the class stores, which lets tests, docs and
  the codecs list a format's capabilities without an instance;
  `unsupported_fields` is derived from it (the rest of the vocabulary),
  and kept because every write loop needs the complement. A subclass may
  still redeclare a single field with `= UNSUPPORTED` by hand; the
  keyword is sugar, not a second mechanism.
- **Per-instance is allowed where the capability really is
  per-instance.** `TiffMetadata` supports `channels` only in the OME and
  ImageJ dialects; its `__post_init__` sets `channels = UNSUPPORTED`
  for a plain TIFF when it is `None` (through `object.__setattr__`, as
  `kinds.py:559` does). `replace()` goes through `__init__`, so
  `__post_init__` runs again and the rule holds after a `replace(...,
  dialect="plain")`. `Cls.supports(name)` (a classmethod) and
  `supported_fields` read the declaration; an instance's capability is
  `meta.name is UNSUPPORTED`.
- **`UNSUPPORTED` never travels.** `from_instance` maps a source
  `UNSUPPORTED` to `None` on the target (the target may well support
  it). Only the target's own declaration produces loss.
- **Setting an unsupported field.** `from_dict` and the constructor
  refuse a non-`None` value for a field the class declares unsupported
  (same spirit as `pin_discriminant="pin+narrow"`: a class does not
  build an instance that contradicts itself; this catches typos in a
  sidecar). Attribute assignment is not validated (`Magic` does not
  validate on set), so `lta.metadata.slice_timing = (0.0, 0.5)` is kept
  on the instance and `check_writable()` reports it as lost at write
  time. Validation therefore lives in two places that already exist:
  construction and the writer.
- **Interaction with `bagof` converters.** `Maybe[T]` is a `Union`; the
  converter for a field must let an `Unsupported` instance through
  untouched. If `bagof` does not do that for free, the framework PR
  registers one identity converter for `Unsupported` with
  `register_converter`, as `DataModelConverter` already does.

> **Prototype note.** Three deviations here.
> (1) `supports=` and `lazy=` are read by a metaclass,
> `_MetadataMeta(type(DataModelBase))` (an adaptor to the classmethods
> `Metadata._declare` and `Metadata._finish`), not by `__init_subclass__`:
> `bagof` builds the fields *before* `__init_subclass__` runs and does not
> forward class keywords to it. The metaclass redeclares each unsupported
> field in the class namespace (annotation + `= UNSUPPORTED`) before
> `MetaMagic` sees it, which is exactly the hand-written spelling, so the
> keyword stays sugar. It runs only for `Metadata` subclasses (no
> metaclass is added to any io class). The vocabulary is a module
> constant (`VOCABULARY`), not a class attribute: it is not
> class-specific.
> (2) `repr` *hides* `UNSUPPORTED` (and an empty `extra`), like `None`:
> with the full vocabulary a NIfTI object printed 28 `UNSUPPORTED`
> entries for 9 real ones. `supported_fields` and the classmethod
> `supports(name)` show capabilities (an earlier `supports` that also
> worked on an instance went: `meta.name is UNSUPPORTED` says it).
> (3) `Maybe[T]` needs no converter registration: `bagof`'s union
> converter lets an `Unsupported` instance through untouched.

## 6. Precedence: raw record versus common fields (M6)

Answer to issue question 5. Two models were weighed:

- *Views.* The record is the only state; common fields are properties
  decoding and encoding it (nibabel style). Faithful by construction,
  but properties are not `Magic` fields: no `from_other`, no `replace`,
  no repr, no `eq`, and every format writes ~35 property pairs. It also
  needs a record for the generic `Metadata`, which has none.
- *Overlay.* Common fields and the record are both stored. On read the
  codec decodes the record into the common fields once. On write the
  record is the base and the common fields are encoded over it.

**Decision: overlay, with a change-detecting write.** On read, the
codec decodes the record into the common fields and keeps a deep copy
of what it decoded as the *snapshot*. On write, a common field is
encoded over the record only when it differs from the snapshot:

```python
_snapshot: NoRepr[NoEq[tx.Dict[str, tx.Any]]] = Factory(dict)  # name -> value


def update_raw(self, raw=None, *, image=None, report=None, force=()):
    raw = self._raw_or_default() if raw is None else raw
    changed = {
        k: v for k, v in self._vocab_items() if v != self._snapshot.get(k)
    }
    return self._encode(raw, changed, image=image, report=report)
```

The record is never re-decoded at write time: the snapshot is the
reference, and it says what the user was shown. An object built in
memory or converted from another format has an empty snapshot (and a
default record), so every non-`None` field counts as a change, which is
what a fresh record needs. Four cases follow, and they are the rule the
user has to know: *untouched means "keep the record's"; a common field
you set wins; a common field you set to `None` is cleared in the
record; edit the record only for what the vocabulary does not cover*.

1. Nothing touched after read: `changed` is empty, the record is written
   as read (faithful round trip, no encode at all).
2. The user edits the record (`metadata.raw["descrip"] = ...`): the
   common field still equals its snapshot, so it is not a change and
   the record edit survives.
3. The user sets the common field: it differs from the snapshot and
   wins, whatever the record holds.
4. The user sets a common field to `None` after a read: it differs from
   the snapshot, so `changed` carries `None` and `_encode` *clears* the
   slot in the record. What clearing means is per format: NIfTI
   `descrip = ""`, `slice_code = 0` (with `slice_start/end/duration`),
   `cal_min = cal_max = 0`; a keyval/keyvalue/attribute key is removed;
   LTA comments are dropped. Without this a user could never erase a
   value, and `derive()` could not invalidate one (section 9). It is
   the same rule as `extra[key] = None` removing a key (today's MRtrix
   and NRRD rule), so there is no second sentinel.

`extra` is compared key by key against its snapshot in the same way.

**One word for the record: raw.** The helpers are public, on
`FileBasedMetadata`, and all say "raw":

| Method | What it does |
|---|---|
| `from_raw(raw, *, image=None, **values)` | build from a raw record just read: decode, then snapshot the *converted* values (a decoded list held as a tuple is not a change) |
| `update_from_raw(raw, *, image=None)` | the metadata of a new raw record, keeping this object's changes |
| `update_raw(raw=None, *, image=None, report=None, force=())` | encode the changes into a raw record, and return it |
| `changed_fields()` | the diff above (`extra` as a per-key diff whose `None` removes a key) |
| `check_writable(*, image=None)` | what a write would lose, from `update_raw` over `_check_raw(image)` |

`update_raw` reports assigned-but-unsupported fields as lost, then calls
`_encode` with the changes. A writer passes its own fresh raw record to
it (NIfTI builds a new header from the data model, copies the safe slots
of `raw` onto it, then encodes the changes), so `_raw_or_default()` is
only the fallback. `check_writable` encodes over what a format's
`_check_raw(image)` hook returns, the raw record its writer would start
from (NIfTI: the header reshaped to the data), so that a value-dependent
check (slice timing against the shape, the time step) reads the same
state as a real write. A writer keyword that must win over the raw
record whatever the snapshot says (MGH `tr=`) is passed as
`update_raw(..., force=("repetition_time",))`: still no second sentinel.
A parser given a raw record and a `metadata` that is not that record's
(an explicit `metadata=`, or one carried by `replace(image,
header=...)`) holds `metadata.update_from_raw(raw)`: the new raw record
decoded, with the fields that changed in the metadata set over it as
changes. Whether the metadata's raw record *is* the parser's is an
identity test (the NIfTI header, the MGH header, the x5 `(header,
node)` pair); a Zarr raw record is rebuilt on each read, so it carries
the node it was read from (`ZarrRaw.node`, `OmeZarrRaw.node`, a handle
that a deep copy or a pickle drops), and the test is `metadata.raw.node
is node`. The base class knows nothing of any format: an earlier
prototype kept the node in `metadata.__dict__["_source"]`, special-cased
by `FileBasedMetadata.copy` and `__getstate__`. The per-format raw types
are named for it too (`MghRaw`, `ZarrRaw`, `OmeZarrRaw`).

`from_raw` raises `TypeError` when `_decode` returns a value for a field
its class does not support (or a `Lazy` for a field that is not lazy).
The raw record is always the format's own, so this never fires on data:
it fires when a format's `_decode` and its `supports=` disagree, a bug
of the format class that would otherwise drop the value without a
report.

**Snapshot lifetime.** `_snapshot` is a real `Magic` field (private
name, like `Transformation._input`), excluded from `repr` and `eq` and
never set by users, holding a `dict` of field name to decoded value
(empty for an object built in memory): it is what the reader decoded,
a frozen view of the raw record, and a missing key means "not decoded,
or decoded as `None`". It survives everything the raw record
survives:
`replace()` and `copy` carry it (a `replace(description="x")` therefore
changes exactly one field), pickling keeps it, and same-format
`from_instance` copies it next to `raw`. When an image or a
transformation is copied (`replace(image, data=...)`, same-class
`from_other`), its metadata is *copied*, not aliased (M10): the copy
shares `raw` and gets its own snapshot and `extra`, so the snapshot
still describes the shared record, and editing the copy's fields never
edits the original's. Cross-format `from_instance`
resets it to `{}` next to the reset `raw`, and `derive()` keeps it
next to the kept `raw` (section 9). It is filled once, by the reader,
with `copy.deepcopy` of the decoded values (the `extra` dict and the
tuples in it are mutable or shared; the nibabel header is in `raw`, not
in the snapshot, and needs no copy), and by a lazy field when it is
loaded. Losing it is therefore never
expected; if it ever is empty with a non-default record (a hand-built
instance), the behaviour degrades to the plain overlay, which is still
correct, only less faithful to record edits.

Codec hooks on the subclass, both private:

```python
@classmethod
def _decode(cls, raw, *, image=None) -> dict: ...  # record -> common fields
def _encode(
    self, raw, changed: dict, *, image=None, report
) -> raw: ...  # None in `changed` clears
```

`image=` (or the transformation) is passed for the fields that need the
data model: diffusion b-vector frames, slice timing expansion from
`slice_code` (needs the slice axis length), AFNI per-brick checks, and
the geometry-derived fields below. Decoding is eager on read (raw
records are small; a read metadata object is complete when `repr`-ed),
with one exception: **a lazy part of the raw record**. Where part of the
raw record is expensive to reach and few users need it (the MGH tags
follow the whole volume, so reading them decompresses an MGZ to its
end), the raw record holds a loader for that part, the format declares
the fields that need it with `lazy=("history",)`, and `_decode` returns
them as `Lazy(load)`. Each lazy field is a `LazyField` descriptor
(`brainhops/_core/properties.py`, next to `lazyproperty`): the pending
`Lazy` sits in the instance `__dict__`, and the descriptor decodes it on
first access (attribute, `repr`, `==`, `changed_fields()`, a
conversion) or before an assignment, through `setattr` (so the field
converts it), then puts it in the snapshot, so the change-detecting
write is unchanged. No other attribute access is intercepted. A load
that never touches them stays as cheap as before the metadata existed.
The raw record otherwise stays lazy only where it is today (the NIfTI
header is parsed when the image is).

> **Prototype note.** A first prototype overrode `__getattribute__` on
> the base class and wrapped every `__setattr__`, keeping a private
> `_pending` dict; the review asked for lazy properties instead. The
> metaclass installs the descriptor after `bagof` built the class (it
> keeps its own field table, so the constructor, the defaults, `fields()`
> and `copy` are unaffected), and again on a subclass whose `supports=`
> redeclares the field. `copy()` keeps a pending field pending (the
> loader is shared).

### 6.1 Naming: `raw`, not `struct`

The maintainer described the raw record as "(semi-)private", as nibabel's
`Nifti1Header` wraps its structured array. `struct` (the LTA name)
sounds like a first-class field; `_raw` would hide it from `fields()`
and `replace()`, which the same-format copy needs. The field is `raw`
(`repr=False`), documented as "the format's record; edit it only for
what the vocabulary does not cover", and each format keeps its
historical public name as a read accessor (`NiftiMetadata.header`,
`MrtrixMetadata.keyval`, `LtaMetadata.struct`) returning `raw` or the
relevant part of it. One name in the framework, a familiar alias per
format, and the same word in the method names (`from_raw`,
`update_from_raw`, `update_raw`, `_check_raw`, `_derive_raw`) and the
raw types (`MghRaw`, `OmeZarrRaw`).

### 6.2 Geometry-derived fields

Some vocabulary fields are, *for one format*, a view of geometry the
data model owns: NIfTI `repetition_time` (`pixdim[4]`), NIfTI `intent`
(the code that types the vector axis), OME-Zarr `channels` when they
come from the `c` axis. Such a field is neither unsupported nor freely
writable in that format. It is a supported field for which the format's
**geometry hook** gives a value, and:

- `_decode` fills it from the record (so it reads naturally);
- on write, the data model wins, through the geometry hook,
  `_geometry(image) -> {field: value}`: the values the data model gives
  for the fields that are views of its geometry (NIfTI: the time step of the image, the scale
  of its time axis, as `repetition_time`). The writer stores the data
  model's value (NIfTI writes the time step as `pixdim[4]`), and
  `update_raw` takes a changed derived field out of `changed` before
  `_encode` sees it, recording it under `report.approximated[name] =
  "derived from the data model (...)"` when it differs (NIfTI write with
  `repetition_time=2.0` on a 1.5 s time axis says so instead of silently
  writing 1.5), silently when they agree. Only a derived field the data
  model says nothing about (an image with no physical time axis) reaches
  `_encode`, which may store it (NIfTI: as the time step) or report it.
  The comparison is never against the record, which the writer has
  rebuilt from the data model anyway;
- `from_instance` *into* such a format copies the value anyway, so a
  later `check_writable()` can compare it with the geometry; whether
  the value is then approximated is decided at write, where the
  geometry is known.

> **Prototype note.** A first prototype also declared these fields with
> a class keyword, `derived=(...)`, exposed as `derived_fields`. It
> decided nothing that `_geometry` did not: a field the hook gives no
> value for reached `_encode` anyway, so NIfTI's `intent` and `space`
> entries were documentation only, and possibly wrong. The keyword
> went; `_geometry` is the single source of truth (a field is derived
> when the hook gives a value for it).

**The element type (`data_type`).** A writer stores the data with this
precedence: an explicit `dtype=` writer option, then `metadata.data_type`
when the array's values are of its kind, then the array's own type. "Of
its kind" means integers (booleans included) as an integer type, floats
as a float type: a label map read as `uint8` and computed on as `int64`
is written as `uint8` again, but a smoothed, floating point version of
it is not quantised, and an integer image is not turned into floats. A
`data_type` the writer does not use is reported as approximated when it
was set (or converted) by hand, and dropped silently when it was only
read (the data changed kind since). A format that cannot store it
writes the nearest type it can (MGH: `uint8`, `int16`, `int32`,
`float32`), reported as approximated. The rule is
`brainhops.datamodel.metadata.preferred_dtype`, which the NIfTI and MGH
writers call.

> **Prototype note.** The plan was "`dtype=` > `data_type` > the array",
> unconditionally. That would quantise a resampled label map as long as
> data-model operations do not call `derive()` yet (section 9), and would
> write an integer image read from a scaled file back as floats or the
> other way round; the "same kind" condition keeps the common case (an
> unchanged type round-trips) without either surprise.

> **Prototype note.** For NIfTI, `space` follows the geometry too (the
> sform code is the world space's name, which the writer takes from the
> data model), but through `_encode`, which compares it with the codes
> the writer set. `intent` is
> written from the field only when the writer set no intent and the
> intent does not retype the axes. `NiftiMetadata._geometry` gives only
> `repetition_time` (the first scaling whose output has a time axis with
> a physical unit, as the NIfTI and MGH readers build it); `intent` and
> `space` are still compared with the header the writer built from the
> data model, which is the same thing for them. A first prototype
> compared `repetition_time` with the fresh header (whose `pixdim[4]` is
> always 1) and wrote no time step, so a 4-D file's TR came back as 1.0
> with a clean report; the hook is what fixed it.

## 7. Conversion and loss reporting (M7)

Answer to issue question 3. Conversion is the data model's own path:
`TargetMetadata.from_other(source)` → `from_instance`, as for images,
and, symmetric to images and transformations, `source.to(Target)`
(section 7, below). `Metadata.from_instance` does, in order:

1. Reset `raw` and the snapshot to the target's defaults (never copied
   across formats; copied as-is when `other` is already of the target
   format class, exactly like `_foreign_format_fields`). Generic
   `Metadata` holds generic metadata only: a `NiftiMetadata` converted
   to `Metadata` loses its raw record, although it is a subclass.
2. For each vocabulary field: source `UNSUPPORTED` → `None`; target
   declared unsupported and source value not `None` → recorded in the
   report under `lost`; else copied.
3. `extra`: copied when the target supports it, else recorded under
   `lost["extra"]`.
4. `cls._import(other, report)`: a format hook that may *recover* a
   loss. `MrtrixMetadata._import` and `NrrdMetadata._import` move lost
   vocabulary values into `extra` under their BIDS key, so nothing is
   lost for key/value formats.

**Loss also depends on the value.** Class-level support says a slot
exists, not that every value fits it: NIfTI `description` holds 80
bytes; NIfTI `slice_timing` is encodable only when it matches one of
the `slice_code` patterns (sequential, alternating, their reverses);
MGH holds one scalar TR; an AFNI per-brick list is written only when it
has as many entries as volumes; `channels` colors do not fit NRRD
labels. These are only known at write, with the value and the data
model in hand, so `_encode` receives the report and fills it per value:
`report.approximated["description"] = "truncated to 80 bytes"`,
`report.lost["slice_timing"] = (...)` with the reason. Class-level
declarations are therefore the *lower bound* of loss; `check_writable()`
gives the exact figure for a given instance and image.

```python
class ConversionReport(Magic):
    source: str
    target: str
    lost: tx.Dict[str, tx.Any] = Factory(dict)  # field -> value dropped
    approximated: tx.Dict[str, str] = Factory(dict)  # field -> what changed
    passed_through: tx.Tuple[str, ...] = ()  # extras moved to a store

    def raise_if_lossy(self): ...
    def __str__(self): ...  # one readable paragraph
```

**`to()`.** `metadata.to(cls=None, *, on_loss=None, report=None,
**values)` is the explicit conversion, symmetric to
`Transformation.to`: `cls` is a `Metadata` subclass or a format name
(`"nifti"`, `"generic"`), `None` keeps the class (a copy with `**values`
set, sharing the raw record within a format). Given a `report`, it
fills it and leaves the loss policy to the caller (as `update_raw`
does); otherwise, or when `on_loss` is given too, the policy applies.
`from_instance` stays the `bagof` hook; `to` is sugar over it.

> **Prototype note.** As specified, with a `lossy` property and
> `merge()`. `MetadataLossError` derives from `Exception`, not
> `ValueError`: `DataModelConverter` turns a `TypeError`/`ValueError`
> raised during an implicit conversion into a `bagof` conversion error,
> and a refused loss must surface as itself. The policy is a context
> variable, and `apply_loss_policy(report, on_loss=None)` is the one
> place a report is acted on (writers call it). A first prototype had a
> module-level `convert(source, target, *, on_loss) -> (target, report)`
> (exported as `convert_metadata`); `to()` replaced it. A known term
> reads as its string in a report (`space='scanner'`, not the enum
> repr). `_import` receives the constructor values,
> `_import(other, values, *, report)`, so that a recovered loss can be
> written into `values["extra"]`; the classmethod cannot otherwise reach
> the object being built. `on_loss=` reaches `io.save` as a writer option
> (`NiftiImage.save(path, on_loss="raise")`), popped before the header
> overrides. **One warning per save:** `io.save` of an object that is not
> of the file's format converts it first (the field converter reports),
> then writes (the writer reports); it runs both under its own `on_loss`
> inside `collect_loss_reports`, which collects the reports the policy
> would warn about, and warns once with `ConversionReport.merged(reports)`.
> Two explicit calls (`from_other`, then `save`) still warn once each.
> The policy surface is one keyword (`on_loss=`), one function
> (`apply_loss_policy`), one context manager (`metadata_loss_policy`)
> and one collector (`collect_loss_reports`); an earlier
> `one_loss_warning` context manager and `get_metadata_loss_policy`
> were folded into `io.save`.

Policy, from least to most strict: `"ignore"`, `"warn"` (default: one
`MetadataLossWarning` per conversion or write carrying the report, not
one per field), `"raise"` (`MetadataLossError`). It is a keyword on
`io.save(obj, file, on_loss=...)`, on the explicit
`source.to(Target, on_loss=...)` (or `report=...` to get the report
back), and a context manager `metadata_loss_policy("raise")`
for the implicit conversions that `bagof`'s field converter triggers
(assigning a `MrtrixMetadata` to a field typed `NiftiMetadata`).

**Free-form pass-through (M8).** `extra` is `str -> Any`. Key/value
formats whose values are text (MRtrix, NRRD) write non-string values as
JSON and read back strings unchanged (no JSON parsing on read). Typed
stores (AFNI attributes, x5 `Metadata`, ImageJ) keep Python types. Keys
are copied verbatim; a format with reserved keys (`_RESERVED` in MRtrix,
NRRD standard fields, AFNI `_GENERATED`) silently skips them, as today.

**BIDS JSON sidecar (M8, codec).** `Metadata.from_bids(dict | path)` and
`Metadata.to_bids() -> dict` in `brainhops/io/metadata/bids.py`:
vocabulary fields through their `Bids(...)` name (units already match),
`generated_by` as the BIDS list of dicts, unknown keys to and from
`extra`. The `DiffusionMetadata` group (`bvalues`/`bvectors`) is not
sidecar keys; a separate `to_bvals_bvecs(image)` rotates them into voxel
axes. Wiring sidecars into `io.load`/`io.save` (`sidecar=True`) is open
question 8.

> **Prototype note.** A vocabulary field with no BIDS key is written
> under its name in CamelCase (`display_range` as `"DisplayRange"`), so
> that a sidecar written by brainhops reads back whole; `channels` is a
> list of objects with CamelCase keys, times are ISO strings. `to_bids`
> takes `on_loss=` and reports the diffusion fields as lost, and an
> encoding direction along no voxel axis. A known term is written as its
> string, `data_unit` as its unit symbol, `data_type` as its `numpy` name.
> `from_bids` also takes a JSON string or an open file.

## 8. Where `bagof.magic` is used, and where it is not (M4)

- **Common-vocabulary fields are `Magic` fields**, declared on six
  `Magic` mixins (the groups, which say `convert=True` themselves) and
  inherited by `Metadata`, a `DataModelBase` with `convert=True`,
  `mapping=False`, `repr=HIDE_IF_NONE`, `doc=True`. This is what gives
  `from_dict`/`from_other`/`replace`, documented fields, and a repr that
  hides the ~30 `None`s. `bagof` handles the multiple inheritance; its
  one quirk is the reverse-MRO field order (section 4).
- **`format` is a polymorphic discriminant, and a real field.** `on=`
  matches an init field (`ItkStruct.type`, `itk/_common.py:80-102`), so
  `format` is declared as a narrowly typed field with a default, not a
  `ClassVar`: `format: tx.Literal["nifti"] = "nifti"` on
  `NiftiMetadata(FileBasedMetadata, on={"format": "nifti"})`. This is the
  same shape as `OrientedAxis`/`AnatomicalAxis`: `DataModelBase` sets
  `pin_discriminant="pin+narrow"`, which composes with `polymorphic=True`
  (that is exactly the combination the `DataModelBase` docstring
  describes), so `NiftiMetadata(format="mrtrix")` raises and
  `Metadata(format="mrtrix", echo_time=0.03)` builds a
  `MrtrixMetadata`. An unknown `format` falls back to the root, the
  generic `Metadata`. The value is the `FileSniffer.HINTS` string where a hint
  exists. Cheap to drop if unused (open question 5).
- **Conversion is `from_other`/`from_instance`**, overridden once on the
  base (section 7). The implicit conversion when a `MrtrixMetadata` is
  assigned to a field typed `NiftiMetadata` comes from `convert=True`
  plus the `DataModelConverter`, which already routes through
  `from_other`.
- **A metaclass** (`_MetadataMeta`) passes `supports=`/`lazy=` (M5, 6)
  to `Metadata`: `bagof` refuses class keywords it
  does not know and builds the fields before `__init_subclass__` runs.
  It is an adaptor only: it calls the classmethod `Metadata._declare`
  (which completes the namespace) before `bagof` builds the class, and
  `cls._finish()` (which sets the capabilities and the lazy
  descriptors) after; the logic is on the class. `Metadata` is built
  with `bagof`'s `repr=False`, which its subclasses inherit, and has one
  hand-written `__repr__`.
- **Descriptors** for the lazy fields (`LazyField`), installed on the
  class after `bagof` built it.
- **Not `Magic`:** the sentinel (a plain singleton, like `_ABSENT` in
  `datamodel/base.py`), nibabel headers (wrapped as they are inside
  `raw`), and the raw records that already exist as frozen `Magic`
  (`NrrdHeader`, `AfniHeader`, `X5Header`, `LtaStruct`, `M3zStruct`,
  `TiffMetadata` in `tiff/_utils.py`, which must be renamed
  `TiffStruct` to free the name). No `Magic` polymorphism is used to
  pick a record type: the format class names it directly.

As in the units memo, polymorphism is used only where there is a field
to match on (`format`), and nothing is dispatched on a parsed name.

## 9. Propagation (M9)

Answer to issue question 4. Propagation is driven by the field `Scope`
tags, through one method:

```python
def derive(
    self, *, grid_changed=False, grid_map=None, volumes=None, step=None
) -> tx.Self:
    """Metadata for an object derived from this one. Always a new object."""
```

> **Prototype note.** `derive` gains `volumes_changed=False`, the flag
> that "the volume count changed and no selection is known" needs.
> `display_range` and `data_unit` are one value for all volumes, so a
> selection keeps them. The only caller so far is the OME-Zarr reader
> (each level gets `derive(grid_changed=level > 0)`, and keeps the
> pyramid's `data_type`: its levels share it). Writers do not call
> it on a conversion yet (the first item of the list below): a converted
> object carries the converted values with no snapshot, which
> writes the same thing for the fields the prototype formats hold; the
> data-model operations are a follow-up PR.

- `file`-scoped fields are kept, except `creation_time` (cleared) and
  `history`, to which `step` (a short string such as
  `"brainhops resample ..."`) is appended. `generated_by` gains a
  brainhops entry once.
- `acquisition`-scoped fields are kept.
- `grid`-scoped fields (`slice_timing`, `slice_encoding_direction`,
  `phase_encoding_direction`, `data_type`) are cleared when
  `grid_changed`, with one exception: given `grid_map`, the linear part
  of the map from the old voxel axes to the new ones, an encoding
  direction in voxel axes is pushed through it (`v' = normalize(grid_map
  @ v)`, section 4.2) instead, so a permutation or flip remaps `"j-"`
  and an oblique resampling keeps an oblique direction; a direction in
  a named world space does not move with the grid. The slice timing is
  still cleared. Without `grid_map`, the directions are cleared, as
  before; the image-level call sites pass it once they have the
  affine.
- `volume`-scoped fields (`channels`, `diffusion_*`, per-volume
  `display_range`) are indexed by `volumes` (the selected volume
  indices) or cleared when the volume count changed and no selection is
  known. This is today's AFNI `_PER_BRICK` and NRRD `_PER_AXIS` rule,
  made generic.
- `extra` is kept verbatim; nothing in it is understood.
- `raw` and the snapshot are **kept**, so a same-format read, resample,
  save keeps extensions, `aux_file` and the rest of the record. The
  fields `derive` clears are set to `None` on the new object, which
  differs from the snapshot, so the write *clears* them in the record
  (case 4 of section 6): NIfTI `slice_code` goes to 0 after a
  resampling, the keyval `SliceTiming` is removed. Record content that
  is grid- or volume-bound but outside the vocabulary (NIfTI
  `slice_start/end`, AFNI `TAXIS_OFFSETS` and the `_PER_GRID`/
  `_PER_BRICK` attributes, NRRD `_PER_AXIS` fields) is scrubbed by a
  per-format hook, `_derive_raw(raw, *, grid_changed, volumes)`, which
  is where today's AFNI and NRRD rules move. Without both, case 1 would
  write stale slice timing from an untouched record. The hook's default
  is a deep copy of the record (a derived object never shares it, as
  M10 copies rather than aliases), so a format overrides it only to
  scrub.

Where it is called:

- **Writers**, when the source object is of another format
  (`from_instance`), with nothing changed: a pure conversion.
- **Multiscale.** The pyramid's metadata is one object on the
  `MultiScaleImage` (OME-Zarr `omero` is per multiscale, TIFF tags are
  per file). Each level is a `SingleScaleImage` and so has a `metadata`
  field too; it holds a *derived copy* (`derive(grid_changed=level >
  0)`), built when the level is materialised, never the parent object by
  identity. Sharing by identity would let `levels[2].metadata.description
  = ...` silently edit the pyramid, and would make a level's
  `slice_timing` wrong. The price is that editing a level's metadata
  does not reach the pyramid, which is the right direction: the pyramid
  is the file. Writers of a multiscale format read the pyramid's object
  only.
- **Transformations**: composition does not merge. A `Sequence` built in
  memory has `metadata=None`; each block keeps its own. A writer of a
  single-block format (LTA, FLIRT, x5 node) takes the metadata of the
  block it writes; `moving`/`fixed` of a chain are those of its ends when
  they agree with the chain's endpoints and `None` otherwise. The
  converse holds on read: **a single-block file puts its metadata on the
  block too**. A reader that maps a file to a `Sequence` (x5) holds the
  metadata on the sequence (it is the file's), and, when the file is one
  block (one x5 node), gives that block a copy as well, so that the
  natural call on the block (`NiftiRASDisplacementField.from_other(
  x5.transformations[0])`) carries it, converted and reported, without a
  `metadata=` argument. The blocks of a chain of several carry nothing.
  A call on the sequence itself is covered by `DataModelBase.from_other`,
  which passes the `metadata` of a data model it hands to a constructor
  (section 10).
- **Data-model operations** (`resample`, channel selection, cropping)
  call `derive` in a follow-up PR; until then the field is copied by
  `replace()` (a copy, not the same object: section 10), which is no
  worse than today.

## 10. Datamodel field (M10) and the `metadata_fields` clash (M11)

The data model gets one optional field on each root:

```python
class Image(DataModelBase):
    metadata: tx.Optional[Metadata] = Field(None, repr=False, eq=False)


class Transformation(DataModelBase):
    metadata: tx.Optional[Metadata] = Field(None, repr=False, eq=False)
```

(`eq=False` assuming `bagof.magic.Field` has a compare flag as
`dataclasses.field` does; otherwise metadata is excluded from `__eq__`
by the class `eq` hook. To verify in the framework PR.) In the
prototype the field is written
`MetadataField[tx.Optional[Metadata], tx.Doc("...")] = None`
(`MetadataField[hint, *annotations]` is `Annotated[hint,
ConvertTo(_EnsureCopy(hint)), KwOnly(), NoRepr(), NoEq(),
*annotations]`), and a format narrows it as
`MetadataField[NiftiMetadata, Factory(NiftiMetadata), tx.Doc("...")]`. Putting it on
the data model rather than only on the file-based mixins is what lets
`from_instance` copy it by name as a *shared* field (not a foreign
format field), and lets an in-memory `Affine` or a resampled image carry
provenance. `Metadata` lives in `brainhops.datamodel.metadata`, so this
import points inward. Each format narrows the type and makes it
mandatory:

```python
class NiftiImage(NiftiParser, WritableFileBasedImage, SingleScaleImage):
    metadata: NiftiMetadata = Factory(NiftiMetadata, repr=False)
```

The narrowing is what triggers conversion (`convert=True`), and
conversion is what reports loss, so `MrtrixImage.from_other(nifti_image)`
warns about the fields NIfTI carried that MRtrix cannot, with no code in
`MrtrixImage` itself.

**Copied, not aliased.** The field converter also *copies* a metadata
object that already is of the field's class: two images or
transformations never hold the same metadata object. `replace(image,
data=...)`, a same-class `from_other`, and `metadata=m` each give the new
object its own copy, which shares `raw` (as the header was shared before
metadata existed) but has its own snapshot and `extra`. Users are told
to edit `image.metadata` in place, so sharing would make "resample, then
edit the copy's description" edit the original too. A conversion builds
a new object anyway. When an object of another family is handed to a
constructor (`NiftiRASDisplacementField.from_other(x5_sequence)`, which
`from_other` passes positionally, since a `Sequence` is not in the
field's family), `DataModelBase.from_other` passes its `metadata` too,
when the class has the field and the caller gave none, so the field
converts and reports it. This is the only place the data model base
knows of the field, and it changes nothing else of `from_other`.

> **Prototype note.** `NoEq()`/`NoRepr()` exist in `bagof.magic`, and the
> root field is also `KwOnly()`, so it never shifts a positional argument
> (`SingleScaleImage(data)`, `Affine(matrix, input, output)`): it lands
> last in every signature. `_foreign_format_fields` needed no change:
> `Image`/`Transformation` declare the field, so it is shared.
> **Pitfall for format classes:** `bagof` takes an inherited field from
> the *first* base that has it (each base carries its whole field table),
> so the narrowed declaration must be on the class itself or on its first
> base. `NiftiImage(NiftiParser, ...)` gets it from `NiftiParser`, but in
> `NiftiRASCoordinatesField(RASCoordinatesField, NiftiBasedTransformation)`
> the first base carries `Transformation`'s generic field. The NIfTI
> transformations therefore declare `metadata: NiftiMetadataField` again
> (`NiftiBasedTransformation`, `NiftiRASCoordinatesField`,
> `NiftiRASDisplacementField`, `NiftiRASToVoxel`, `NiftiVoxelToRAS`,
> `ItkNiftiField`, `SpmCoordinatesField`), a test checks every registered
> NIfTI format, and the writer converts a foreign metadata object anyway.
> The `metadata_fields` -> `encoding_fields` rename is not done in the
> prototype: nothing needs it yet. Every `metadata` field, the roots'
> and the narrowed ones, is written with `MetadataField[...]`, whose
> converter (`_EnsureCopy`) is an explicit `ConvertTo`: it applies on the
> plain-`Magic` parsers too (x5, FLIRT), so a metadata assigned after
> construction is converted on every format, not only on the classes
> with `convert=True`. Since a format class is now a subclass of
> `Metadata`, the converter also converts (and reports) a format's
> metadata given to a field typed `Metadata`, which `bagof` would keep
> as it is. A first prototype used a function,
> `metadata_annotation(hint, doc, default=...)`, and named the converter
> `_Copied`.

**Name clash.** `Transformation.metadata_fields` means "meta-attributes
that define the encoding" (`order`, `bound`, `coeff`). It is renamed
`encoding_fields`, with a `metadata_fields` class property that returns
`encoding_fields` and warns, kept for one minor release.

## 11. Format sketches

Base class and sentinel are in sections 3 and 5. A format writes
`supports=`, `_decode`, `_encode`, optionally `_geometry` and
`_import`.

**NIfTI (image format, nibabel record).**

```python
class NiftiMetadata(
    FileBasedMetadata, on={"format": "nifti"},
    supports=("description", "intent", "space", "display_range", "slice_timing",
              "slice_encoding_direction", "phase_encoding_direction", "sources",
              "generated_by", "repetition_time", "data_type"),
):  # `_geometry` gives `repetition_time` (pixdim[4], the time step)
    format: tx.Literal["nifti"] = "nifti"
    raw: NoRepr[NoEq[tx.Optional[nb.Nifti1Header]]] = None
    # extra, channels, history, echo_time, ... are UNSUPPORTED (no store; OQ 7)

    @property
    def header(self): return self.raw

    @classmethod
    def _decode(cls, h, *, image=None):
        return dict(description=_str(h["descrip"]) or None, intent=_intent_name(h),
                    space=_sform_space(h), display_range=_cal(h),
                    slice_timing=_expand_slices(h, image), ...)

    def _encode(self, h, changed, *, image=None, report):
        if "description" in changed:
            s = changed["description"].encode()
            if len(s) > 79: report.approximated["description"] = "truncated to 80 bytes"
            h["descrip"] = s[:79]
        if "slice_timing" in changed:
            code = _slice_code_for(changed["slice_timing"], image)
            if code is None: report.lost["slice_timing"] = changed["slice_timing"]
            else: h["slice_code"], h["slice_duration"], ... = code
        ...; return h
```

`NiftiImage.to_nibabel(like=None, **overrides)` keeps its signature:
`like=` becomes `NiftiMetadata.from_other(like)` merged under the
instance's own metadata, and `**overrides` keep patching the header
after `_encode`. The default write now starts from `metadata.raw` when
there is one, which is the NIfTI round-trip improvement: `slice_*`,
`cal_*`, `aux_file`, extensions survive a read-then-save. Geometry,
`xyzt_units`, dtype and `scl_*` are still overridden from the data model
and writer options. `NiftiBasedTransformation` reuses `NiftiMetadata`
unchanged; FNIRT/NiftyReg/SPM keep reading `intent_p*` from `raw`.

> **Prototype note (NIfTI).** `NiftiMetadata` lives in
> `brainhops/io/base/_nifti_metadata.py`, next to the shared NIfTI parser
> (`io/base/nifti.py`), and is re-exported from `io.images.nifti` and
> `io.transformations.nifti`. `supports=` drops `generated_by` (NIfTI has
> no slot for it) and `space` is derived (6.2). `NiftiParser` syncs the
> metadata from the header in `__post_init__` (and when `header` is
> assigned): an explicit `metadata=` without a record keeps its values over
> the decoded ones. The writers build their header as before, then
> `_apply_metadata` applies, lowest precedence first: the safe slots of
> the record (`descrip`, `aux_file`, `cal_*`, `dim_info`, `slice_*` when
> the slice axis kept its length, a non-structural intent for images, and
> the extensions unless the writer added its own), then `like=` exactly as
> before, then the changed common fields, then `**overrides`. So `like=`
> sits above the record but below an explicit edit, which keeps the
> existing `like=` behaviour. A slice timing that matches no NIfTI order
> is lost *and* clears the record's slice fields (the user's value
> replaced them). Header floats are single precision and are decoded as
> the shortest decimal (`0.3`, not `0.30000001`). The image writer stores
> the data model's time step as `pixdim[4]` (in the header's time unit,
> seconds when it has none), so a 4-D image keeps its `repetition_time`
> through a read and a save (6.2); an image with no physical time axis
> gets the field's value there instead. `NiftiParser` reads the header
> again when it is not the metadata's raw record (`replace(image,
> header=h)`), keeping the changed fields (`update_from_raw`).
> `data_type` is the header's `datatype` (`get_data_dtype()`), which the
> image writer uses as in 6.2 (and builds the `nibabel` image with, so
> that an `int64` array of a `uint8` label map can be written); an
> encoding direction is written into `dim_info` as its axis (polarity
> approximated, an oblique one lost).

**MRtrix (key/value format).**

```python
class MrtrixMetadata(FileBasedMetadata, on={"format": "mrtrix"}, supports=ALL):
    format: tx.Literal["mrtrix"] = "mrtrix"
    raw: MrtrixHeader = Factory(MrtrixHeader, repr=False)
    _BIDS_IN_KEYVAL = (
        "EchoTime",
        "RepetitionTime",
        "FlipAngle",
        "PhaseEncodingDirection",
        "TotalReadoutTime",
        "SliceEncodingDirection",
        "SliceTiming",
        "MultibandAccelerationFactor",
        ...,
    )

    @property
    def keyval(self):
        return self.raw.keyval

    @classmethod
    def _decode(cls, h, *, image=None):
        kv = dict(h.keyval)
        out = dict(
            history=_lines(kv.pop("command_history", None)),
            description=kv.pop("comments", None),
            generated_by=_mrtrix_version(kv.pop("mrtrix_version", None)),
        )
        out.update(_bids_from_strings(kv, cls._BIDS_IN_KEYVAL))  # pops them
        if "dw_scheme" in kv:
            out.update(_dw_scheme(kv.pop("dw_scheme")))  # already world frame
        out["extra"] = kv  # everything else
        return out

    def _encode(self, h, changed, *, image=None, report):
        kv = dict(h.keyval)
        kv.update(_merge_extra(changed.get("extra")))  # None removes
        kv.update(_bids_to_strings(changed))
        ...
        return replace(h, keyval=kv)

    @classmethod
    def _import(
        cls, other, report
    ):  # nothing is lost: unknown vocab -> keyval[Bids name]
        ...
```

Whatever has no dedicated key is written under its BIDS name in
`keyval`, which is what `mrconvert -json_import` produces anyway. The
open MRtrix warp PR's `_merge_keyval`/`_writer_layout` move into
`_encode`; `MrtrixWarp` and `MrtrixLinearTransform` (`# key: value`
comments) share this class with `MrtrixImage`. NRRD is the same shape
with `keyvalue`, the `DWMRI_*` convention and the `measurement frame`
rotation.

**LTA (transformation format, text record).**

```python
class LtaMetadata(
    FileBasedMetadata,
    on={"format": "lta"},
    supports=(
        "moving",
        "fixed",
        "input_space",
        "output_space",
        "description",
        "history",
    ),
):
    format: tx.Literal["lta"] = "lta"
    raw: LtaStruct = Factory(LtaStruct, repr=False)

    @property
    def struct(self):
        return self.raw

    @classmethod
    def _decode(cls, s, *, transformation=None):
        return dict(
            moving=s.src.filename or None,
            fixed=s.dst.filename or None,
            input_space=_lta_space(s.type)[0],
            output_space=_lta_space(s.type)[1],
            description=_first_comment(s),
            history=_comments(s),
        )

    def _encode(self, s, changed, *, transformation=None, report):
        src = (
            replace(s.src, filename=changed["moving"])
            if "moving" in changed
            else s.src
        )
        dst = (
            replace(s.dst, filename=changed["fixed"])
            if "fixed" in changed
            else s.dst
        )
        return replace(s, src=src, dst=dst, comments=_comments_from(self, s))
```

The parser must stop stripping `#` comments so that `history` and
`description` round-trip (`LtaStruct` gains a `comments` tuple). The
`src`/`dst` geometries stay in `raw`. `X5Metadata` is the same shape
with `raw = (X5Header, X5Node)`, `description`/`history`/`generated_by`
in the node's `Metadata` JSON, `extra` in `attrs`, and `supports=ALL`
for a JSON-capable node.

> **Prototype note (x5).** `X5Metadata` lives in
> `io/transformations/x5/_metadata.py`, with `supports=ALL`. Every
> vocabulary field is stored in the node's JSON `Metadata` under its BIDS
> sidecar key, or its name in `CamelCase` when BIDS has none (the
> sidecar codec's own naming, so an x5 node's JSON *is* a sidecar);
> `extra` is the other JSON keys only. Node `attrs` stay in the record
> and are written back as read, not mirrored into `extra`.
> `input_space`/`output_space` are the JSON keys `InputSpace`/
> `OutputSpace`: `Domain.Coordinates` is not decoded, because the two
> implementations write the kind of coordinates there (`"cartesian"`),
> not a space label. **Where the metadata sits:** the reader maps a
> file to one `X5Transform` (a `Sequence`) whose blocks are data model
> transformations, so the metadata is on the `X5Transform`, and it is
> that of *the node it reads* when it reads a single node (a `position`,
> a one-node chain, or a file with no chain). For a chain of several
> nodes, composition does not merge: the record is `(header, None)`,
> nothing is decoded, every node keeps its JSON and writes it back, and
> a field set on the chain is reported lost on save. The writer encodes
> the changed fields into the node they were read from; a transformation
> whose chain was reassigned (or built in memory) and encodes a single
> node gets all its fields written into that node. The block decoded
> from the node the metadata is that of gets a copy of it (as generic
> `Metadata`, the block's field type) when it is decoded, so
> `from_other(block)` carries it (section 9); a chain's blocks carry
> nothing. An untouched read
> writes every node as read, so the JSON string round-trips unchanged.
> `save(on_loss=)` is popped by the writer. The parser syncs in
> `__post_init__` through `io/metadata/_sync.py` (`sync_metadata`,
> which keeps a metadata whose raw record is the parser's and otherwise
> uses `update_from_raw`; every parser uses it: NIfTI, MGH, Zarr,
> OME-Zarr, x5, FLIRT, ITK `.h5`); the narrowed field is
> `MetadataField[X5Metadata, Factory(X5Metadata), ...]`, whose converter
> converts on these plain-`Magic` parsers too.
>
> **Prototype note (ITK, FLIRT).** `ItkMetadata` (`format="itk"`) is an
> `OpaqueMetadata` subclass for `.tfm` and `.mat`; the `.mat` writer
> reports any field set after construction (`on_loss=` popped). The `.h5`
> class is `ItkH5Metadata` (`format="itk-h5"`): `/ITKVersion` is
> `generated_by = (GeneratedBy("ITK", version),)`, and an encode keeps
> only the `ITK` entry (others lost); there is no `.h5` writer yet. ITK
> blocks keep `metadata=None`. `FlirtMetadata` (`format="flirt"`) is
> *not* opaque: it is a `FileBasedMetadata` with `supports=("moving",
> "fixed")` and no raw record (`raw: None`). They are the paths of the
> `moving`/`reference` images the reader was given (`get_filename()` of a
> `nibabel` image, or of a NIfTI image's `image`), decoded at
> construction, kept by a copy or a conversion, and always reported lost
> by `check_writable` (there is no FLIRT writer). A first prototype made
> it an `OpaqueMetadata` subclass, which contradicted "opaque"; the
> review moved it. ITK's `precision` stays in the raw record for now
> (4.4).

**Everything else.**

| Format | `raw` | `supports=` (highlights) | `extra` store / notes |
|---|---|---|---|
| MGH | `MghRaw(header, tags)` | TR/TE/TI/flip (ms→s, rad→deg; one scalar TR, else `approximated`), `history` (cmdline tag, lazy), `data_type` (`type`) | none |
| AFNI | `AfniHeader` | `history`, `channels` (`BRICK_LABS`), `repetition_time` (unit code), `slice_timing`, `space`, `creation_time` | remaining attributes; `_GENERATED`/`_PER_BRICK`/`_PER_GRID` rules move into `_encode` (count mismatch → `lost`) and `_derive_raw` |
| TIFF/OME/ImageJ | `TiffStruct` (renamed) + `ome_xml`/ImageJ dict | `name`, `description`, `generated_by`, `creation_time`, `manufacturer*`, `channels` (per instance by dialect) | ImageJ extras or plain tags; OME-XML round-trips whole, `_encode` patches `Name`/`Channel` only |
| Pillow | `dict(info)` | `description`, `creation_time` | text chunks, `exif`, `icc_profile` |
| OpenSlide (read-only) | — | `description`, `manufacturer`, `objective_magnification` | `properties` |
| JP2 (open PR) | `Jpeg2000Header` | `description` (`comments`) | boxes |
| OME-Zarr | `OmeZarrRaw` (abczarr `Multiscale`, `omero`, attributes) | `name`, `channels`, `display_range`, `data_type` (derived from the arrays) | the other group attributes |
| plain Zarr | — | generic `Metadata` serialised into `.zattrs["brainhops"]` | same |
| MINC (read-only) | none kept | `history`, acquisition/patient/study variable attributes | the rest of them |
| ITK tfm/mat | — | `OpaqueMetadata` subclass (nothing, anywhere) | none |
| ITK h5 | small h5 header | `generated_by` (`ITKVersion`) | none |
| Elastix | `parameter_map` | `sources` (`InitialTransformParameterFileName`) | the non-transform keys (today's "base map" rule) |
| FLIRT | — | `FileBasedMetadata` with `moving`/`fixed` (reader needs them, kept in memory, never stored, always `lost` on write) | none |
| matrix text | — | `OpaqueMetadata` | none |
| M3Z | `M3zStruct` | `moving`/`fixed` (`image`/`atlas` fnames) | byte-exact otherwise |

> **Prototype note (MGH).** `MghMetadata` lives in
> `brainhops/io/base/_mgh_metadata.py`, next to the MGH parser, and is
> re-exported from `io.images.freesurfer`. Its raw record is
> `MghRaw(header, tags)`: `nibabel` has no separate footer class (the
> footer fields `tr`, `flip_angle`, `te`, `ti`, `fov` are part of
> `MGHHeader`, whose `hf_dtype` is the header and the footer), so the
> header is the footer's raw record too; the code here is for the tag
> stream after the footer, which `nibabel` neither reads nor writes.
> Zero in a footer slot reads as `None`; the flip angle is decoded
> as the shortest decimal of degrees that stores the same single-precision
> radians (`9.0`, not `9.0000004`). `repetition_time` is not derived: MGH
> stores the TR as a scalar, whatever the time axis. `history` comes from
> the `TAG_CMDLINE` tags (id 3, `int64` length, NUL-terminated), and only
> when the whole tag stream parses (`TAG_OLD_MGH_XFORM` has an `int32`
> length, the other legacy tags none); writing it replaces the command-line
> tags in place and keeps every other tag, and over tags that do not parse
> it is reported as lost. `MghParser` syncs the metadata in `__post_init__`
> and when `header` or `tags` are assigned, but the tags stay lazy (6, a
> lazy part of the raw record): `MghRaw` holds a loader for them,
> `history` is a lazy field (`lazy=("history",)`), and a load reads the
> header and footer only (a first
> prototype read the tags on load, a third of the load time of a 38 MB
> MGZ). The record and the image share one read of the tags. The writer
> keeps the footer of the record, then `like=`, then the changed fields;
> `tr=`/`te=`/`ti=`/`flip_angle=` (still in ms and radians) are set on a
> copy and passed as `update_raw(force=...)` (so they win even over
> `like=` and over a value equal to the one read, and a zero clears the
> slot), while `fov=` and other header names still patch the header
> last. As for NIfTI, `like=` sits *under* a changed field but *over* an
> unchanged one, which keeps the record's value only where `like=` has
> none. No deprecation warning yet. `data_type` is the header's `type`;
> the writer uses it as in 6.2, and a type MGH cannot store becomes the
> nearest one (approximated).

> **Prototype note (Zarr).** `OmeZarrMetadata` and `ZarrMetadata` live in
> `brainhops/io/images/zarr/_metadata.py`. The OME-Zarr raw record is
> `OmeZarrRaw(multiscale, omero, attrs)`: the typed 0.6 `Multiscale`,
> but `omero` as plain JSON, read from the group attributes, because the
> typed `abczarr` model turns nested free-form keys into `True` (`rdefs`
> does not survive `Omero.from_json(...).to_json()`); it is written back
> as JSON into the envelope of the written version. `channels` are not
> derived from the `c` axis: they are what `omero` says, and a
> `display_range` with no channels writes one white channel per entry of
> the channel axis. A window is required per channel; when nothing gives
> one, its `start`/`end` are the range of the values of the smallest
> level (cheap to read; `min`/`max` are the range of the data type), and
> it is reported as approximated under `channels` (a first prototype
> wrote the range of the data type silently, which renders a 12-bit
> `uint16` image black). A Zarr record is rebuilt from the attributes on
> each read, so the metadata remembers the node it was read from (a
> handle, not pickled), and a parser reads the attributes again when
> its node is not that one. `data_unit` is
> **unsupported** (OME-Zarr has no intensity unit, and inventing an
> `omero` key was not worth it); a `Channel.unit` and a non-opaque alpha
> are reported as approximated. The writer keeps the multiscale `name`,
> `type` and downsampling `metadata` of the record. Levels get
> `derive(grid_changed=level > 0)` when the pyramid is read; a pyramid
> built in memory keeps the levels it was given. For plain Zarr the
> generic option was taken: `ZarrMetadata` (format `"zarr"`) stores the
> vocabulary as a BIDS sidecar (the codec of `to_bids`) under the array
> attribute `"brainhops"`, `extra` is the other attributes, and the
> diffusion fields are unsupported (they are not sidecar keys:
> `supports=` names the four other groups and `"extra"`). In both,
> `data_type` is derived from the arrays (a Zarr array stores its own
> type, and the writer stores the array as it is), so a `data_type`
> that disagrees with the array is reported as approximated, never
> written; the levels of a pyramid keep the pyramid's.

Future formats (GIFTI/CIFTI, TRK/TCK/TRX, CZI/LIF/ND2, Bruker, MRC,
FITS, MetaImage, BrainVoyager, Interfile, EEG coordsystem, NetCDF) each
add one class and, if they bring a concept two of them share, one
vocabulary field.

## 12. Backward compatibility and migration (M12)

**Attribute aliases.** Every attribute being replaced stays as a
deprecated property on the format class for one minor release,
delegating to the metadata object: `NiftiImage.header -> metadata.raw`,
`MrtrixImage.keyval -> metadata.raw.keyval`, `AfniImage.attributes ->
metadata.raw.attributes`, `MghImage.tags`/`mri_params`,
`TiffImage.tags/ome_xml/imagej_metadata`, `PillowImage.info`,
`OpenSlideImage.properties`, `X5Transform.header/nodes`,
`LtaTransform.struct`, `M3zTransform.struct`,
`ElastixTransform.parameter_map`, `Jpeg2000Image.header`. The
NiftiParser `header` property is the one alias kept permanently, without
a warning (it is how nibabel users think).

**Constructor and writer keywords (M13).** These are format-specific
metadata inputs too, and each becomes `metadata=<FileBasedMetadata>` built
from the old value, with a `DeprecationWarning` naming the replacement:

| Today | Where | Becomes |
|---|---|---|
| `like=` | `NiftiImage.to_nibabel`/`save` | `metadata=NiftiMetadata.from_other(like)` merged under the instance's own (kept as sugar, one release) |
| `descrip=`, `intent=`, other header names | `NiftiImage.to_nibabel(**overrides)` | `metadata.description`/`.intent`; raw header names stay as `**overrides` (they patch `raw`) |
| `tr=`, `te=`, `ti=`, `flip_angle=`, `fov=` | `MghImage` writer | `metadata.repetition_time` (s) etc.; `fov` → `raw` |
| `tags=` | `MghImage` | `metadata.history` / `raw.tags` |
| `keyval=` | `MrtrixImage._mrtrix_header`, warps PR | `metadata.extra` (BIDS keys land in vocabulary fields) |
| `keyvalue=` | `NrrdImage` writer | `metadata.extra` |
| `attributes=`, `view=` | `AfniImage`, AFNI xforms PR | `metadata.extra` / `raw.attributes`; `view` stays a writer option (encoding) |
| `info=` | `PillowImage` | `metadata.extra` |
| `name=` | `build_ome(...)` | `metadata.name` |

**PR plan, in order.** Sizes are rough line counts including tests.

1. *Framework* (~1,500): `brainhops/datamodel/metadata.py`
   (`Metadata`, `FileBasedMetadata`, `OpaqueMetadata`, the vocabulary
   groups, `UNSUPPORTED`/`Maybe`, `supports=`/`lazy=`, the
   vocabulary with `Bids`/`Scope` annotations, `ConversionReport` +
   policies, `to`, `derive`);
   `brainhops/io/metadata/bids.py` (sidecar codec); the `metadata` field
   on `Image`/`Transformation`; the `encoding_fields` rename;
   `_FileBasedModelMixin.from_instance` taught that `metadata` is
   shared. No format touched. The three open PRs (JP2, MRtrix warps,
   AFNI xforms) merge before or alongside this with their current
   attributes; they are migrated in their batches below.
2. *NIfTI* (~1,200): `NiftiMetadata`, `NiftiImage`, all
   `NiftiBasedTransformation` formats (plain, FNIRT, NiftyReg, SPM, ITK
   NIfTI), `like=` and `header` aliases. Biggest batch, because NIfTI
   round-trip changes behaviour (more is kept).
3. *Key/value formats* (~900): MRtrix image + warps + linear, NRRD, AFNI
   image + xforms.
4. *FreeSurfer* (~500): MGH, LTA (with comments), M3Z.
5. *Transform records* (~500): x5, ITK tfm/mat/h5, elastix, FLIRT,
   matrix text, OME-Zarr fields.
6. *Rasters and microscopy* (~1,000): TIFF/OME/ImageJ (`TiffStruct`
   rename), Pillow, OpenSlide, JP2, OME-Zarr images (per-level
   `derive`), plain Zarr.
7. *MINC and cleanup* (~300): MINC vocabulary, deprecated-alias removal
   one minor release after PR 6, `derive` wired into `resample` and
   channel selection.

**Tests.** The framework PR adds: sentinel semantics (identity, falsy,
pickling, `Maybe` conversion, constructor refusal); `supports=` →
`supported_fields`, `supports()` and per-instance `UNSUPPORTED`; `from_instance`
loss accounting on two synthetic formats; policies (`ignore`/`warn`/
`raise`, one warning per conversion); the change-detecting write
(cases 1-4 of section 6, including clearing); `derive` for each scope and for a level;
sidecar import/export with unknown keys. Each format PR adds: (a)
decode assertions on an existing fixture; (b) same-format round trip
comparing `raw` (byte-exact where the format already promises it: M3Z,
x5, NRRD, AFNI; field-wise elsewhere), with and without a record edit;
(c) the format's row and column in one parametrized cross-format matrix
test that converts a fully populated, *encodable* `Metadata` fixture
(short description, sequential slice timing, matching volume counts)
into the format and back and asserts `report.lost == {fields the class
declares unsupported}`, no more, no less, so a wrong `supports=` list
fails CI; (d) a value-dependent loss test per format with such a rule
(81-byte description, alternating-irregular slice timing, mismatched
per-brick list) asserting the exact `lost`/`approximated` entries;
(e) deprecated aliases and keywords warn and still work.

> **Prototype note.** The matrix test (c) is
> `tests/test_io_metadata_matrix.py`, over every prototype format: the
> conversion of the fully populated fixture loses exactly
> `unsupported_fields` (the complement of `supported_fields`), the way
> back to `Metadata` loses nothing, and
> where a fresh record can hold the values (NIfTI, MGH, x5, Zarr, ITK
> `.h5`; OME-Zarr through a real save) they are written and read back,
> derived fields aside. A format class that is not in its table fails
> it. The writer keywords of M13 (`like=`, MGH `tr=`...) keep working
> without a `DeprecationWarning` in the prototype.

## Decisions

- **M1** Geometry, axes, spatial/temporal units, transformations, the
  transform kind and storage encoding (byte order, scaling, layout,
  compression) are never metadata. The element type on disk is
  (`data_type`): it is a user-visible choice that may differ from the
  loaded array.
- **M2** Three layers (`Magic` vocabulary fields, a format-private
  `raw`, an `extra` dict) in a hierarchy that mirrors the images':
  `Metadata` (the vocabulary and `extra`; the root, polymorphic on
  `format`, and the generic, lossless hub) -> `FileBasedMetadata`
  (`raw`, the read-time snapshot, the read/write hooks) ->
  `<Fmt>Metadata`. `OpaqueMetadata` is a format that keeps nothing,
  not even in memory (ITK `.tfm`/`.mat`, matrix text); a format that
  keeps anything (FLIRT `moving`/`fixed`, in memory) is a
  `FileBasedMetadata` with a `supports=` list. The format-agnostic part
  lives in `brainhops.datamodel.metadata`; format classes and the
  sidecar codec under `io`.
- **M3** Vocabulary = BIDS names in snake_case with BIDS units, 37
  fields declared in six group mixins (`ProvenanceMetadata`,
  `MRIMetadata`, `DiffusionMetadata`, `DisplayMetadata`,
  `MicroscopyMetadata`, `TransformMetadata`) that `Metadata` inherits,
  each field tagged with a propagation scope; `VOCABULARY` and `GROUPS`
  are module constants. A free-text field with known terms is
  `Union[<Enum>, str]` (`Space`, `Intent`, `Manufacturer`,
  `IlluminationType`, `ContrastMethod`); `data_unit` is a `Unit` when
  the units module parses it, the name otherwise, written as its
  symbol; encoding directions are `EncodingDirection` vectors (voxel
  axes by default, BIDS strings accepted); `bvalues`/`bvectors`. LTA/M3Z
  image references are `moving`/`fixed`; `sources` is BIDS provenance.
- **M4** `Magic` is used for the vocabulary fields (on mixins), for
  `from_other`/`from_instance` conversion, and for polymorphic
  construction on a real `format` field (`pin+narrow`); a metaclass
  adaptor for `supports=`/`lazy=`; not for the sentinel or the raw
  record types.
- **M5** `UNSUPPORTED` singleton; `Maybe[T]`; declared compactly with
  `supports=(...)` (names or groups; everything else unsupported), which
  gives `supported_fields` (`unsupported_fields` is derived from it);
  per-instance via `__post_init__` where the capability is per-instance;
  refused at construction, reported at write when assigned; never
  copied across.
- **M6** Overlay with change detection: a common field wins only when
  it differs from its read-time snapshot (a `Magic` field holding a
  `dict` of the decoded values, which survives `replace`/`derive`); `None` clears
  the slot; raw record edits survive. The raw record field is `raw`,
  with a per-format read alias, and every name says "raw" (`from_raw`,
  `update_from_raw`, `update_raw`, `MghRaw`). Geometry-derived fields
  are read-only for that format: a geometry hook
  (`_geometry(image)`) gives the data model's value, which the writer
  stores, and a changed value that disagrees with it is reported as
  `approximated` (never compared with the raw record). Decoding is
  eager, except for a lazy part of the raw record (MGH tags), whose
  fields (`lazy=`) are `LazyField` descriptors decoded on first access.
  A writer stores the data as `data_type` when the values are of its
  kind; `dtype=` wins.
- **M7** `ConversionReport`; class-level support is the lower bound,
  `_encode` adds value-dependent `lost`/`approximated`; `on_loss` =
  `ignore`/`warn` (default)/`raise`; one aggregated warning; `_import`
  hook for recovery; `metadata.to(cls, *, on_loss, report)` is the
  explicit conversion, symmetric to images and transformations.
- **M8** `extra` passes unknown keys through to any free-form store;
  BIDS JSON sidecar is a codec on `Metadata`.
- **M9** `derive(grid_changed, grid_map, volumes, step)` driven by
  scope (an encoding direction goes through `grid_map`); a multiscale
  level holds a derived copy, never the pyramid's object; composition
  does not merge; a single-block file puts its metadata on the block as
  well as on the file object.
- **M10** `metadata: Optional[Metadata]` on `Image` and `Transformation`,
  narrowed and made mandatory by each format class, all written
  `MetadataField[hint, *annotations]`. The field copies (sharing `raw`),
  never aliases: `replace()`, same-class `from_other` and `metadata=`
  give the new object its own metadata; a format's metadata given to a
  generic field is converted.
- **M11** `Transformation.metadata_fields` becomes `encoding_fields`,
  alias kept one minor release.
- **M12** Seven PRs, framework first, deprecated aliases for one minor
  release, cross-format matrix test as the capability oracle plus
  value-dependent loss tests.
- **M13** Writer/constructor keywords (`like=`, `tr=`/`te=`...,
  `keyval=`, `keyvalue=`, `attributes=`, `info=`, `tags=`) become
  `metadata=` with deprecation aliases.

## Open questions for the maintainer

1. **Where the field lives (M10).** On the datamodel roots (`Image`,
   `Transformation`, optional, `None` by default) so in-memory objects
   carry provenance and `from_instance` treats it as shared; or only on
   the file-based mixins, keeping the datamodel metadata-free as the x5
   docs currently promise. Recommendation: datamodel roots, excluded
   from equality.
2. **Default loss policy (M7).** `warn` with one aggregated warning, or
   silent `ignore` with a report available on request. Recommendation:
   `warn`; a converter that drops a user's slice timing without a word
   is the current behaviour we are trying to leave.
3. **Change-detecting write (M6).** Keep the read-time snapshot so that
   record edits survive, or the plain overlay (common fields always
   win, record edits to covered keys are overridden). Recommendation:
   snapshot; it is one dict and removes the only surprising case.
4. **Diffusion frame (M3).** World/RAS unit vectors (MRtrix convention),
   with NRRD and BIDS codecs rotating via the affine; or voxel-axis
   vectors (BIDS/FSL convention). Recommendation: world; it is the only
   frame that survives a reorientation without touching metadata.
5. **Polymorphic `format` discriminant (M4).** Keep `Metadata(
   format="nifti", ...)` dispatch, or make subclasses plain and
   construct them by name only. Recommendation: keep; it costs one line
   per class and matches `ItkStruct`.
6. **`intent` vocabulary (M3).** *Closed:* the NIfTI intent names are
   the canonical set, as the `Intent` enum (`Union[Intent, str]`, so an
   unknown name is kept); nothing else has a richer set and AFNI/NRRD
   map onto it.
7. **NIfTI extras.** Unsupported (reported as lost), or written into a
   NIfTI extension (a JSON extension with a brainhops-chosen ecode) that
   only brainhops reads. Recommendation: unsupported in v1; an extension
   is a later opt-in writer option, because other tools would carry an
   opaque blob.
8. **Sidecar integration (M8).** Codec only (`Metadata.from_bids`/
   `to_bids`), or also `io.load(path, sidecar=True)` / `io.save(...,
   sidecar=True)` reading and writing `<stem>.json` next to the image.
   Recommendation: codec in the framework PR, `sidecar=` in the NIfTI PR
   where it is actually used.
9. **Rename `metadata_fields` (M11).** `encoding_fields` with a warning
   alias, or leave the clash and document it. Recommendation: rename;
   the alias is three lines.
10. **Open PRs.** Merge JP2, MRtrix warps and AFNI xforms as they are and
    migrate them in batches 3 and 6, or hold them until the framework
    lands. Recommendation: merge now; each migration is small and the
    framework PR touches no format.
11. **Deprecation window (M12).** One minor release for the aliases, or
    two. Recommendation: one, since the library is pre-1.0 and the
    aliases are mechanical.
12. **Levels and the pyramid (M9).** Derived copy per level (recommended)
    versus `None` on levels and metadata only on the pyramid, which is
    simpler but makes a level saved alone lose everything.
