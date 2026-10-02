# Design: format metadata and cross-format metadata conversion

Status: design only, no code. Answers #233. Builds on the io model
(`FileBasedImage`, `FileBasedTransformation`, `from_other`/`from_instance`
in `io/base/_base.py`), on `DataModelBase` (#97, `bagof.magic`), and on
the units memo (`units-polymorphism.md`) for where `Magic` polymorphism
does and does not fit. Decisions are tagged `M1`..`M12` and collected at
the end, followed by the open questions.

The problem, in one sentence: every file format keeps its non-spatial
metadata under a different name and type (`header`, `keyval`, `tags`,
`info`, `ome_xml`, `imagej_metadata`, `attributes`, `parameter_map`,
`struct`, ...), nothing is shared across formats, and `from_instance`
deliberately drops all of it when a file is converted (NIfTI to MGH keeps
the data model only; `_foreign_format_fields`, `_base.py:617`).

## 1. What exists today

Facts, from the inventory of the current branch:

- There is no metadata slot on the data model. `Image`,
  `SingleScaleImage`, `MultiScaleImage` and `Transformation` have none,
  and the x5 docs say so explicitly ("The datamodel holds no metadata").
- Each parser mixin holds a raw structure as the source of truth for
  geometry (nibabel headers for NIfTI and MGH, `MrtrixHeader`,
  `AfniHeader`, `NrrdHeader`, `X5Header`/`X5Node`, `LtaStruct`,
  `M3zStruct`, `ItkStruct`, elastix `parameter_map`, TIFF/Pillow/OpenSlide
  flat fields). MINC keeps nothing but `dimensions`.
- Round trip within a format is good for most formats (MRtrix, AFNI,
  NRRD, MGH, x5, LTA, M3Z, elastix) and poor for NIfTI (fresh header
  unless `like=` is given; `slice_*`, `cal_*`, `aux_file`, extensions
  dropped), OME-TIFF (only `Name` and `Channel.Name`), OME-Zarr (no
  `omero`, no multiscale name) and Pillow (only `icc_profile`, `exif`).
- Free-form key/value stores exist in six formats (MRtrix `keyval`, NRRD
  `keyvalue`, AFNI `attributes`, x5 `Metadata`/`attrs`, TIFF
  `imagej_metadata`/`tags`, Pillow `info`) and are never carried across.
- Units disagree: MGH stores TR/TE/TI in ms and the flip angle in
  radians, AFNI codes the TR unit (ms/s/Hz), BIDS uses seconds and
  degrees.
- `Transformation.metadata_fields` already exists and means something
  else ("the meta-attributes that define the encoding": `order`,
  `bound`, `coeff`).

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
- storage encoding: dtype, byte order, intensity scaling (`scl_slope`,
  `BRICK_FLOAT_FACS`, MRtrix `scaling`, MINC `image-min/max`), layout,
  compression, data file names. These are writer options or raw-struct
  content, not descriptive metadata.

The raw structure still *contains* these (a nibabel header has `pixdim`),
and the writer still overrides them from the data model, exactly as
today. The vocabulary simply never exposes them.

## 3. Representation (M2): typed vocabulary + faithful raw struct + extras

Answer to issue question 1: both, in one object. A `FormatMetadata`
instance has three layers:

1. **Common vocabulary**: `Magic` fields on the base class, grouped as in
   section 4, with BIDS names and units. Every subclass has them.
2. **Raw structure** `struct`: one format-private field whose type is
   the format's faithful record (the nibabel `Nifti1Header`,
   `MrtrixHeader`, `LtaStruct`, ...). Read-then-write in the same format
   goes through it. It is never copied across formats.
3. **Extras** `extra: Dict[str, Any]`: a free-form namespace for keys
   the vocabulary does not cover. It *is* copied across formats, into
   whatever free-form store the target has (MRtrix `keyval`, NRRD
   `keyvalue`, AFNI `attributes`, x5 `Metadata`, ImageJ metadata, PNG
   text chunks), and reported as lost where there is none (NIfTI, MGH,
   LTA, ITK, FLIRT).

This mirrors nibabel's `Nifti1Header` (typed accessors over a structured
record) and brainhops' own images (a data model plus a parser mixin that
keeps the raw file state).

```python
# brainhops/io/metadata/_base.py
class FormatMetadata(DataModelBase, polymorphic=True):
    """Common vocabulary + one format's faithful structure."""

    format: tx.ClassVar[str] = "generic"       # discriminant, see M4
    struct: tx.Any = None                     # format-private, repr=False
    extra: Maybe[tx.Dict[str, tx.Any]] = Factory(dict)

    # --- common vocabulary (section 4), e.g. ---
    description: Maybe[str] = None
    repetition_time: Maybe[float] = None      # seconds (BIDS RepetitionTime)
    ...

    @classmethod
    def from_instance(cls, other, *args, **kwargs) -> tx.Self: ...   # M6
    def check_writable(self) -> "ConversionReport": ...               # M5
    def derive(self, **changes) -> tx.Self: ...                       # M9
```

A plain `Metadata` subclass (`format="generic"`, `struct=None`, nothing
unsupported) is what in-memory objects carry and what the BIDS sidecar
codec reads and writes.

## 4. Vocabulary (M3)

Answer to issue question 2. Names are the BIDS keys in snake_case
(`RepetitionTime` becomes `repetition_time`); the BIDS spelling is kept
as field metadata (`tx.Annotated[..., Bids("RepetitionTime")]`) for the
sidecar codec. Units are the BIDS units: seconds, degrees, tesla, mm,
s/mm². Where BIDS has no key, the name is chosen to read like one.

Every field is `Maybe[T] = Union[T, None, Unsupported]` (section 5) and
carries a `scope` tag used by propagation (section 9): `file` (about this
file), `acquisition` (invariant under resampling), `grid` (tied to the
voxel grid), `volume` (one entry per volume/channel).

### 4.1 Core and provenance (`file`)

| Field | Type / unit | BIDS | Native sources |
|---|---|---|---|
| `name` | str | — | OME `Name`, TIFF `DocumentName`, OME-Zarr multiscale `name` |
| `description` | str | — | NIfTI `descrip`, TIFF `ImageDescription` (plain), NRRD `content`, OpenSlide `openslide.comment`, JP2 `comments`, x5 `Metadata.description` |
| `history` | tuple[str] | — | MRtrix `command_history`, AFNI `HISTORY_NOTE`, MGH `tags` (cmdline), MRtrix `.txt` comments |
| `generated_by` | tuple[GeneratedBy(name, version, description)] | `GeneratedBy` | TIFF `Software`, MRtrix `mrtrix_version`, ITK h5 `ITKVersion`, x5 `Format`/`Version`, AFNI `TYPESTRING` |
| `creation_time` | datetime | — | TIFF `DateTime`, AFNI `IDCODE_DATE`, EXIF |
| `sources` | tuple[str] | `Sources` | NIfTI `aux_file`, LTA `src/dst filename`, M3Z `image/atlas fname`, elastix `InitialTransformParameterFileName` |
| `space` | str | `SpatialReference` | NIfTI `sform_code` name, AFNI `TEMPLATE_SPACE`, NRRD `space`, x5 `Domain.Coordinates` label |
| `intent` | str (NIfTI intent names as the canonical vocabulary) | — | NIfTI `intent_code/name`, AFNI `BRICK_STATSYM`, NRRD `kinds` (non-spatial) |

`space` is a *label* (`"MNI152NLin6Asym"`, `"scanner"`, `"orig"`); the
coordinate system itself stays in the data model. `intent` is
descriptive; the NIfTI reader still reads `intent_code` from the struct
to type axes, and the writer derives the code from the axes first and
from `intent` second (the one documented exception to the precedence
rule of section 6).

### 4.2 MRI acquisition (`acquisition`, except where noted)

| Field | Unit | BIDS | Native sources |
|---|---|---|---|
| `repetition_time` | s | `RepetitionTime` | MGH `tr` (ms), AFNI `TAXIS_FLOATS[1]`+unit code, MRtrix keyval, NIfTI `pixdim[4]` only when the time axis says so |
| `echo_time` | s | `EchoTime` | MGH `te` (ms), MRtrix/NRRD keyval |
| `inversion_time` | s | `InversionTime` | MGH `ti` (ms) |
| `flip_angle` | deg | `FlipAngle` | MGH `flip_angle` (rad) |
| `magnetic_field_strength` | T | `MagneticFieldStrength` | keyval only |
| `manufacturer`, `manufacturers_model_name`, `institution_name` | str | same | TIFF `Make`/`Model`, OpenSlide `vendor`, OME Instrument |
| `acquisition_time` | datetime | `AcquisitionTime` | OME `AcquisitionDate`, EXIF |
| `phase_encoding_direction` (`grid`) | `"i"`..`"k-"` | same | MRtrix keyval, NIfTI `dim_info` |
| `total_readout_time`, `effective_echo_spacing` | s | same | keyval |
| `slice_encoding_direction` (`grid`) | `"i"`..`"k-"` | same | NIfTI `dim_info`, MRtrix keyval |
| `slice_timing` (`grid`) | tuple[float] s | `SliceTiming` | NIfTI `slice_code/start/end/duration` (expanded), AFNI `TAXIS_OFFSETS`, MRtrix keyval |
| `multiband_acceleration_factor` | int | same | MRtrix keyval |

The TR is in the vocabulary even though the time step is geometry: the
two coincide for a plain fMRI series but not for, say, a multi-echo
stack or a sparse acquisition, and MGH/AFNI store the TR without any
time axis. The writer of a format that *only* has a time step (NIfTI
`pixdim[4]`) takes it from the data model, never from `repetition_time`.

### 4.3 Diffusion (`volume`)

| Field | Type / unit | Native sources |
|---|---|---|
| `diffusion_bvalues` | (N,) float, s/mm² | MRtrix `dw_scheme` col 4, NRRD `DWMRI_b-value` x gradient norms, BIDS `.bval` |
| `diffusion_bvectors` | (N,3) float, unit vectors in **world (RAS) coordinates** | MRtrix `dw_scheme` cols 1-3 (already world), NRRD gradients via `measurement frame`, BIDS `.bvec` via the voxel-to-world rotation |

The frame convention is the lossy spot in diffusion metadata; the
converters of each format rotate into and out of world coordinates using
the image's affine, which is why these codecs need the data model, not
just the struct (section 6.3).

### 4.4 Display and channels (`volume`)

| Field | Type | Native sources |
|---|---|---|
| `display_range` | (min, max) float | NIfTI `cal_min/max`, AFNI `BRICK_STATS` (per volume, first) |
| `channels` | tuple[Channel(name, color, display_range, unit)] | AFNI `BRICK_LABS`, NRRD `labels`, ImageJ `Labels`/`LUTs`/`Ranges`, OME `Channel.Name/Color`, OME-Zarr `omero.channels` |
| `data_unit` | str | NRRD `sample units`, OME-Zarr `omero`/OME channel unit, MINC `units` |

`Channel.color` is an RGBA hex string; LUT arrays stay in `extra`.

### 4.5 Microscopy (`acquisition`)

Deliberately small until CZI/LIF/ND2/Imaris/BDV readers exist:
`objective_magnification` (float; OpenSlide `objective-power`, OME
`Objective.NominalMagnification`), `objective_numerical_aperture`,
`illumination_type`/`contrast_method` as free strings. Everything else
goes to `extra` until two formats agree on it.

### 4.6 Transformation-specific (`file`)

| Field | Type | Native sources |
|---|---|---|
| `moving` / `fixed` | str (file reference) | LTA `src/dst filename`, M3Z `image/atlas fname`, FLIRT `src`/`ref` (given by the user, not stored), elastix (not stored), ANTs convention |
| `input_space` / `output_space` | str label | x5 `Domain.Coordinates`, LTA (derived from `type`), OME-Zarr coordinate system names |

`moving`/`fixed` are deliberately *not* `source`/`target`: the data model
warns that brainhops' `input`/`output` are the inverse of the imaging
direction, and `moving`/`fixed` are unambiguous about which image is
which. `history`, `generated_by`, `description` and `sources` are shared
with images.

### 4.7 Not in the vocabulary, and why

Field of view (derived from geometry), intensity scaling and display
`cal_*` for per-volume AFNI stats (raw), NIfTI `intent_p1..p3` (raw;
FNIRT/NiftyReg read them from the struct), ITK `precision` (encoding),
elastix resampler keys (raw, written back by `transformation_to_map`),
EXIF/ICC blobs (`extra["exif"]`, `extra["icc_profile"]`), OpenSlide
`bounds`/`background_color`/associated images (`extra`), JP2 boxes
(struct). Around 35 fields in all; adding one needs two formats that
carry it natively, or one format plus a BIDS key.

## 5. The `UNSUPPORTED` sentinel (M5)

A format class must be able to say "this format cannot store this", and
that must be different from "nobody set it".

```python
# brainhops/io/metadata/_sentinel.py
class Unsupported:
    """The format cannot store this field. Singleton, falsy, not None."""
    __slots__ = ()
    def __bool__(self): return False
    def __repr__(self): return "UNSUPPORTED"
    def __reduce__(self): return "UNSUPPORTED"      # pickles to the singleton

UNSUPPORTED = Unsupported()
T = tx.TypeVar("T")
Maybe = tx.Union[T, None, Unsupported]
```

Three-valued semantics, per field:

| Value | Meaning | Converter (`from_instance`) | Writer |
|---|---|---|---|
| `None` | unknown / absent | copies nothing | writes nothing |
| a value | known | copies it, or reports it lost | writes it |
| `UNSUPPORTED` | this format has no slot for it | source side: treated as `None`; target side: incoming value is **lost** and reported | nothing to write |

Rules:

- **Declaration is class-level, by default value.** A subclass marks a
  field unsupported by redeclaring it with the sentinel as default:
  `slice_timing: Maybe[tuple] = UNSUPPORTED` in `LtaMetadata`. The base
  class exposes `FormatMetadata.unsupported_fields` (a `ClassVar`
  computed once from the field defaults) so tests, docs and the sidecar
  codec can list a format's capabilities without an instance.
- **Per-instance is allowed where the capability really is
  per-instance.** `TiffMetadata` supports `channels` only in the OME and
  ImageJ dialects; its `__post_init__` sets `channels = UNSUPPORTED`
  for a plain TIFF. `supports(name)` on an instance reads the instance;
  `unsupported_fields` on a class reads the defaults.
- **`UNSUPPORTED` never travels.** `from_instance` maps a source
  `UNSUPPORTED` to `None` on the target (the target may well support
  it). Only the target's own declaration produces loss.
- **A user may overwrite an unsupported field.** Nothing stops
  `lta.metadata.slice_timing = (0.0, 0.5)`; the value is kept on the
  instance, and `check_writable()` reports it as lost at write time
  (section 7). This keeps validation at one place (the writer) and
  keeps `replace()` and `from_dict` simple. `from_dict` on a class that
  declares a field unsupported *does* refuse a non-`None` value for it
  (same spirit as `pin_discriminant="pin+narrow"`: a class does not
  build an instance that contradicts itself) — see open question 4.
- **Generic `Metadata` has no unsupported field**, so it is the lossless
  hub: `NiftiMetadata -> Metadata -> MrtrixMetadata` loses exactly what
  `NiftiMetadata -> MrtrixMetadata` loses.
- **Interaction with `bagof` converters.** `Maybe[T]` is a `Union`; the
  converter for a field must let an `Unsupported` instance through
  untouched. If `bagof` does not do that for free, the framework PR
  registers one converter for `Unsupported` (identity) with
  `register_converter`, as `DataModelConverter` already does for the
  data model.

## 6. Precedence: raw struct versus common fields (M6)

Answer to issue question 5. Two models were weighed:

- *Views.* The struct is the only state; common fields are properties
  decoding and encoding it (nibabel style). Faithful by construction,
  but properties are not `Magic` fields: no `from_other`, no `replace`,
  no repr, no `eq`, and every format writes ~35 property pairs. It also
  needs a struct for the generic `Metadata`, which has none.
- *Overlay.* Common fields and the struct are both stored. On read the
  codec decodes the struct into the common fields once. On write the
  struct is the base and every common field that is not `None` is
  encoded over it. A common field is therefore "the last word".

**Decision: overlay.** The rule the user has to know is one sentence:
*set a common field to change it; edit `struct` only for what the
vocabulary does not cover.* Editing `struct["descrip"]` after a read has
no effect on write because `description` was decoded from it and wins;
`metadata.description = None` hands the field back to the struct.
`metadata.extra` behaves the same way with respect to the format's
free-form store: it is merged over the struct's store on write, and a
key set to `None` removes it (today's MRtrix and NRRD rule, kept).

Codec hooks on the subclass, both private:

```python
@classmethod
def _decode(cls, struct, *, image=None) -> dict:      # struct -> common fields
def _encode(self, struct, *, image=None) -> struct    # common fields over struct
```

`image=` (or the transformation) is passed for the few fields that need
the data model: diffusion b-vector frames, slice timing expansion from
`slice_code` (needs the slice axis length), AFNI per-brick checks.

Decoding is eager on read (headers are small; it also means a read
metadata object is complete when `repr`-ed). The struct stays lazy only
where it is lazy today (the NIfTI header is parsed when the image is).

## 7. Conversion and loss reporting (M7)

Answer to issue question 3. Conversion is the data model's own path:
`TargetMetadata.from_other(source)` → `from_instance`, as for images.
`FormatMetadata.from_instance` does, in order:

1. Reset `struct` to the target's default (never copied across formats;
   copied as-is when `isinstance(other, cls)`, exactly like
   `_foreign_format_fields`).
2. For each vocabulary field: source `UNSUPPORTED` → `None`; target
   declared unsupported and source value not `None` → recorded in the
   report under `lost`; else copied.
3. `extra`: copied when the target supports it, else recorded under
   `lost["extra"]`.
4. `cls._import(other, report)`: a format hook that may *recover* a
   loss. `MrtrixMetadata._import` and `NrrdMetadata._import` move lost
   vocabulary values into `extra` under their BIDS key (so an MRtrix file
   keeps `EchoTime` even though its vocabulary slot and its keyval are
   the same store); `NiftiMetadata._import` truncates `description` to
   80 bytes and records it under `approximated`.

```python
class ConversionReport(Magic):
    source: str; target: str
    lost: tx.Dict[str, tx.Any] = Factory(dict)          # field -> value dropped
    approximated: tx.Dict[str, str] = Factory(dict)     # field -> what changed
    passed_through: tx.Tuple[str, ...] = ()             # extras moved to a store
    def raise_if_lossy(self): ...
    def __str__(self): ...                              # one readable paragraph
```

Policy, from least to most strict: `"ignore"`, `"warn"` (default: one
`MetadataLossWarning` per conversion carrying the report, not one per
field), `"raise"` (`MetadataLossError`). It is a keyword on
`io.save(obj, file, on_loss=...)`, on the explicit
`brainhops.io.metadata.convert(source, Target, on_loss=...) ->
(target, report)`, and a context manager `metadata_loss_policy("raise")`
for the implicit conversions that `bagof`'s field converter triggers
(assigning a `MrtrixMetadata` to a field typed `NiftiMetadata`). The
same report type is returned by `check_writable()`, which a writer calls
first to report user-set values on unsupported fields and extras without
a store.

**Free-form pass-through (M8).** `extra` is `str -> Any`. Key/value
formats whose values are text (MRtrix, NRRD) write non-string values as
JSON and read back strings unchanged (they do not try to parse JSON on
read; a value that was JSON-encoded on write comes back as a string
unless the sidecar codec or the user decodes it). Typed stores (AFNI
attributes, x5 `Metadata`, ImageJ) keep Python types. Keys are copied
verbatim; a format with reserved keys (`_RESERVED` in MRtrix, NRRD
standard fields, AFNI `_GENERATED`) silently skips them, as today.

**BIDS JSON sidecar (M8, codec).** `Metadata.from_bids(dict | path)` and
`Metadata.to_bids() -> dict`: vocabulary fields through their `Bids(...)`
name (units already match), `generated_by` as the BIDS list of dicts,
unknown keys to and from `extra`. `diffusion_bvalues/bvectors` are not
sidecar keys; `to_bids` leaves them out and a separate
`to_bvals_bvecs(image)` rotates them into voxel axes. Wiring sidecars
into `io.load`/`io.save` (`sidecar=True`, path derived from the image
path) is open question 9.

## 8. Where `bagof.magic` is used, and where it is not (M4)

- **Common-vocabulary fields are `Magic` fields** on `FormatMetadata`,
  which is a `DataModelBase` and so inherits `convert=True`,
  `mapping=False`, `repr=HIDE_IF_NONE`, `doc=True`. This is what gives
  `from_dict`/`from_other`/`replace`, documented fields, and a repr that
  hides the ~30 `None`s. `UNSUPPORTED` is shown by `repr` (it is not
  `None`), which is what one wants when inspecting a format's class.
- **`format` is a polymorphic discriminant.** `FormatMetadata` is
  `polymorphic=True`; each subclass registers `on={"format": "nifti"}`
  like the ITK blocks do with `type`. `FormatMetadata(format="mrtrix",
  echo_time=0.03)` builds a `MrtrixMetadata`; an unknown format falls
  back to the generic `Metadata`. The value is the same one
  `FileSniffer.HINTS` uses where a hint exists. This is cheap to drop
  if it proves unused (open question 6).
- **Conversion is `from_other`/`from_instance`**, overridden once on the
  base (section 7), the same door the io classes already use. The
  implicit conversion when a `MrtrixMetadata` is assigned to a field
  typed `NiftiMetadata` comes from `convert=True` plus the
  `DataModelConverter`, which already routes through `from_other`.
- **Not `Magic`:** the sentinel (a plain singleton, like `_ABSENT` in
  `datamodel/base.py`), nibabel headers (wrapped as they are inside
  `struct`), and the raw structs that already exist as frozen `Magic`
  (`NrrdHeader`, `AfniHeader`, `X5Header`, `LtaStruct`, `M3zStruct`,
  `TiffMetadata` in `tiff/_utils.py`, which must be renamed
  `TiffStruct` to free the name). No `Magic` polymorphism is used to
  pick a struct type: the format class names its struct type directly.

As in the units memo, polymorphism is used only where there is a field
to match on (`format`), and nothing is dispatched on a parsed name.

## 9. Propagation (M9)

Answer to issue question 4. Propagation is driven by the field `scope`
tags, through one method:

```python
def derive(self, *, grid_changed=False, volumes=None, step=None) -> tx.Self:
    """Metadata for an object derived from this one."""
```

- `file`-scoped fields are kept, except `creation_time` (cleared) and
  `history`, to which `step` (a short string such as
  `"brainhops resample ..."`) is appended. `generated_by` gains a
  brainhops entry once.
- `acquisition`-scoped fields are kept.
- `grid`-scoped fields (`slice_timing`, `slice_encoding_direction`,
  `phase_encoding_direction`) are cleared when `grid_changed`. A pure
  axis permutation or flip is not a grid change for the PE direction,
  and the permutation that `Permutation`/`Projection` carry (compute-api
  memo, A.9) remaps `"i"`/`"j-"`; everything else that touches the grid
  clears them.
- `volume`-scoped fields (`channels`, `diffusion_*`, per-volume
  `display_range`) are indexed by `volumes` (the selected volume
  indices) or cleared when the volume count changed and no selection is
  known. This is today's AFNI `_PER_BRICK` and NRRD `_PER_AXIS` rule,
  made generic.
- `extra` is kept verbatim; nothing in it is understood.

Where it is called:

- **Writers**, when the source object is of another format
  (`from_instance`), with nothing changed: a pure conversion.
- **Multiscale**: one metadata object per pyramid, on the
  `MultiScaleImage`; levels share it (OME-Zarr `omero` is per
  multiscale, TIFF tags are per file). Selecting a level is
  `grid_changed=True` for the single-scale view, which clears slice
  timing, which is right.
- **Transformations**: composition does not merge. A `Sequence` built in
  memory has `metadata=None`; each block keeps its own. A writer of a
  single-block format (LTA, FLIRT, x5 node) takes the metadata of the
  block it writes; `moving`/`fixed` of a chain are those of its ends when
  they agree with the chain's endpoints and `None` otherwise.
- **Data-model operations** (`resample`, channel selection, cropping)
  call `derive` in a follow-up PR; until then the field is simply copied
  by `replace()`, which is no worse than today.

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
by the existing `eq` hook. To verify in the framework PR.) Putting it on
the data model rather than only on the file-based mixins is what lets
`from_instance` copy it by name as a *shared* field (not a foreign
format field), and lets an in-memory `Affine` or a resampled image carry
provenance. Each format narrows the type and makes it mandatory:

```python
class NiftiImage(NiftiParser, WritableFileBasedImage, SingleScaleImage):
    metadata: NiftiMetadata = Factory(NiftiMetadata, repr=False)
```

The narrowing is what triggers conversion (`convert=True`), and
conversion is what reports loss, so `MrtrixImage.from_other(nifti_image)`
warns about the fields NIfTI carried that MRtrix cannot, with no code in
`MrtrixImage` itself.

**Name clash.** `Transformation.metadata_fields` means "meta-attributes
that define the encoding" (`order`, `bound`, `coeff`). It is renamed
`encoding_fields`, with a `metadata_fields` class property that returns
`encoding_fields` and warns, kept for one minor release. The new
instance field `metadata` and the class var `encoding_fields` then read
as what they are.

## 11. Format sketches

Base class and sentinel are in sections 3 and 5. The hooks a format
writes are `_decode`, `_encode`, optionally `_import`, plus the
`UNSUPPORTED` defaults.

**NIfTI (image format, nibabel struct).**

```python
class NiftiMetadata(FormatMetadata, on={"format": "nifti"}):
    format = "nifti"
    struct: tx.Optional[nb.Nifti1Header] = Field(None, repr=False)
    extra: Maybe[dict] = UNSUPPORTED               # no free-form store (OQ 8)
    channels: Maybe[tuple] = UNSUPPORTED
    history: Maybe[tuple] = UNSUPPORTED            # 80-byte descrip is not a log
    echo_time = inversion_time = flip_angle = UNSUPPORTED   # and the other acquisition fields

    @classmethod
    def _decode(cls, h, *, image=None):
        return dict(
            description=_str(h["descrip"]) or None,
            intent=_intent_name(h), space=_sform_space(h),
            display_range=_cal(h), slice_timing=_expand_slices(h, image),
            phase_encoding_direction=_dim_info_pe(h), sources=_aux(h),
            generated_by=_from_extensions(h), ...)

    def _encode(self, h, *, image=None):
        if self.description is not None: h["descrip"] = self.description.encode()[:79]
        ...; return h
```

`NiftiImage.to_nibabel(like=None, **overrides)` keeps its signature:
`like=` becomes `NiftiMetadata.from_other(like)` merged under the
instance's own metadata, and `**overrides` keep patching the header
after `_encode`. The default write now starts from `metadata.struct`
when there is one, which is the NIfTI round-trip improvement: `slice_*`,
`cal_*`, `aux_file`, extensions survive a read-then-save. Geometry,
`xyzt_units`, dtype and `scl_*` are still overridden from the data model
and writer options. `NiftiBasedTransformation` reuses `NiftiMetadata`
unchanged; FNIRT/NiftyReg/SPM keep reading `intent_p*` from `struct`.

**MRtrix (key/value format).**

```python
class MrtrixMetadata(FormatMetadata, on={"format": "mrtrix"}):
    format = "mrtrix"
    struct: MrtrixHeader = Factory(MrtrixHeader, repr=False)
    _BIDS_IN_KEYVAL = ("EchoTime", "RepetitionTime", "FlipAngle",
                       "PhaseEncodingDirection", "TotalReadoutTime",
                       "SliceEncodingDirection", "SliceTiming",
                       "MultibandAccelerationFactor", ...)

    @classmethod
    def _decode(cls, h, *, image=None):
        kv = dict(h.keyval)
        out = dict(history=_lines(kv.pop("command_history", None)),
                   description=kv.pop("comments", None),
                   generated_by=_mrtrix_version(kv.pop("mrtrix_version", None)))
        out.update(_bids_from_strings(kv, cls._BIDS_IN_KEYVAL))   # pops them
        if "dw_scheme" in kv:
            out.update(_dw_scheme(kv.pop("dw_scheme")))             # already world frame
        out["extra"] = kv                                            # everything else
        return out

    def _encode(self, h, *, image=None):
        kv = dict(h.keyval); kv.update(_merge_extra(self.extra))   # None removes
        kv.update(_bids_to_strings(self)); ...; h.keyval = kv; return h

    @classmethod
    def _import(cls, other, report):
        for name, value in list(report.lost.items()):           # nothing is lost:
            ...                                                 # unknown vocab -> keyval[Bids name]
```

Nothing in the vocabulary is unsupported for MRtrix: whatever has no
dedicated key is written under its BIDS name in `keyval`, which is what
`mrconvert -json_import` produces anyway. The open MRtrix warp PR's
`_merge_keyval`/`_writer_layout` move into `_encode`; `MrtrixWarp` and
`MrtrixLinearTransform` (`# key: value` comments) share this class with
`MrtrixImage`. NRRD is the same shape with `keyvalue` and the
`DWMRI_*` convention plus `measurement frame` rotation.

**LTA (transformation format, text struct).**

```python
class LtaMetadata(FormatMetadata, on={"format": "lta"}):
    format = "lta"
    struct: LtaStruct = Factory(LtaStruct, repr=False)
    extra = channels = slice_timing = diffusion_bvalues = ... = UNSUPPORTED
    # i.e. everything except: moving, fixed, sources, input_space, output_space,
    # description (written as a leading '# ' comment), history (comments)

    @classmethod
    def _decode(cls, s, *, transformation=None):
        return dict(moving=s.src.filename or None, fixed=s.dst.filename or None,
                    input_space=_lta_space(s.type)[0], output_space=_lta_space(s.type)[1],
                    description=_first_comment(s), history=_comments(s))

    def _encode(self, s, *, transformation=None):
        return replace(s, src=replace(s.src, filename=self.moving or s.src.filename),
                          dst=replace(s.dst, filename=self.fixed or s.dst.filename),
                          comments=_comments_from(self))
```

The parser must stop stripping `#` comments so that `history` and
`description` round-trip (`LtaStruct` gains a `comments` tuple). The
`src`/`dst` geometries stay in `struct`. `X5Metadata` is the same shape
with `struct = (X5Header, X5Node)`, `description`/`history`/
`generated_by` in the node's `Metadata` JSON, `extra` in `attrs`, and
nothing unsupported for a JSON-capable node.

**Everything else, one line each.** MGH: `struct = MghStruct(header,
tags)`; TR/TE/TI/flip converted ms→s and rad→deg; `history` from the
cmdline tag; everything else unsupported. AFNI: `struct = AfniHeader`;
`history`, `channels` (`BRICK_LABS`), `repetition_time` (unit code),
`slice_timing`, `space` (`TEMPLATE_SPACE`), `creation_time`, `extra` =
the remaining attributes; `_GENERATED`/`_PER_BRICK`/`_PER_GRID` rules
move into `_encode` and `derive`. TIFF: `struct = TiffStruct` (renamed)
+ raw `ome_xml`/ImageJ dict; `name`, `description`, `generated_by`,
`creation_time`, `manufacturer*`, `channels` (dialect-dependent, per
instance), `extra` = ImageJ extras or plain tags; OME-XML round-trips
whole via the struct, with `_encode` patching only `Name`/`Channel`.
Pillow: `struct = dict(info)`; `extra` = text chunks, `exif`,
`icc_profile`; nearly everything else unsupported. OpenSlide (read-only):
`description`, `manufacturer`, `objective_magnification`, `extra` =
`properties`. JP2 (open PR): `struct = Jpeg2000Header`; `description`
from `comments`, `extra` from boxes. OME-Zarr: `struct` = the abczarr
`Multiscale`; `name`, `channels`, `data_unit` from `omero` — the
round-trip fix the inventory flags. Plain Zarr: generic `Metadata` in
`.zattrs["brainhops"]` (extras only). MINC (read-only): `history`,
`acquisition`/`patient`/`study` variable attributes into the vocabulary
and `extra`; no struct kept. ITK tfm/mat/h5: `struct` = the small h5
header or `None`; `generated_by` from `ITKVersion`; everything else
unsupported, which is honest. Elastix: `struct = parameter_map`;
`sources` = `InitialTransformParameterFileName`; `extra` = the
non-transform keys (today's "base map" rule). FLIRT and plain matrix
text: generic `Metadata` with *every* field unsupported except `moving`/
`fixed` (which FLIRT needs anyway and never stores); a read-then-save
loses nothing because nothing is there. M3Z: `struct = M3zStruct`;
`moving`/`fixed` from `image`/`atlas` fnames; byte-exact otherwise.
Future formats (GIFTI/CIFTI, TRK/TCK/TRX, CZI/LIF/ND2, Bruker, MRC,
FITS, MetaImage, BrainVoyager, Interfile, EEG coordsystem, NetCDF) each
add one class and, if they bring a concept two of them share, one
vocabulary field.

## 12. Backward compatibility and migration (M12)

**Aliases.** Every attribute being replaced stays as a deprecated
property on the format class for one minor release, delegating to the
metadata object: `NiftiImage.header -> metadata.struct`,
`MrtrixImage.keyval -> metadata.struct.keyval`, `AfniImage.attributes ->
metadata.struct.attributes`, `MghImage.tags`, `TiffImage.tags/ome_xml/
imagej_metadata`, `PillowImage.info`, `X5Transform.header/nodes`,
`LtaTransform.struct`, `ElastixTransform.parameter_map`. The constructor
keywords `keyval=`, `attributes=`, `tags=`, `info=` and the NIfTI `like=`
keep working: each becomes `metadata=<FormatMetadata>` built from the
old value, with a `DeprecationWarning` naming the replacement. The
NiftiParser `header` property is the one alias worth keeping permanently
as a convenience (it is how nibabel users think), without a warning.

**PR plan, in order.** Sizes are rough line counts including tests.

1. *Framework* (~1,500): `brainhops/io/metadata/` with `FormatMetadata`,
   `Metadata`, `UNSUPPORTED`/`Maybe`, the vocabulary with `Bids`/`scope`
   annotations, `ConversionReport` + policies, `derive`, the BIDS sidecar
   codec; the `metadata` field on `Image`/`Transformation`; the
   `encoding_fields` rename; `_FileBasedModelMixin.from_instance` taught
   that `metadata` is shared. No format touched. The three open PRs
   (JP2, MRtrix warps, AFNI xforms) merge before or alongside this with
   their current attributes; they are migrated in their batches below.
2. *NIfTI* (~1,200): `NiftiMetadata`, `NiftiImage`, all
   `NiftiBasedTransformation` formats (plain, FNIRT, NiftyReg, SPM, ITK
   NIfTI), `like=` and `header` aliases. Biggest batch, because NIfTI
   round-trip changes behaviour (more is kept).
3. *Key/value formats* (~900): MRtrix image + warps + linear, NRRD, AFNI
   image + xforms.
4. *FreeSurfer* (~500): MGH, LTA (with comments), M3Z.
5. *Transform structs* (~500): x5, ITK tfm/mat/h5, elastix, FLIRT,
   matrix text, OME-Zarr fields.
6. *Rasters and microscopy* (~1,000): TIFF/OME/ImageJ (`TiffStruct`
   rename), Pillow, OpenSlide, JP2, OME-Zarr images, plain Zarr.
7. *MINC and cleanup* (~300): MINC vocabulary, deprecated-alias removal
   one minor release after PR 6, `derive` wired into `resample` and
   channel selection.

**Tests.** The framework PR adds: sentinel semantics (identity, falsy,
pickling, `Maybe` conversion); `unsupported_fields` from class defaults
and per-instance `supports`; `from_instance` loss accounting on two
synthetic formats; policies (`ignore`/`warn`/`raise`, one warning per
conversion); `derive` for each scope; sidecar import/export with unknown
keys. Each format PR adds: (a) decode assertions on an existing fixture;
(b) same-format round trip comparing `struct` (byte-exact where the
format already promises it: M3Z, x5, NRRD, AFNI; field-wise elsewhere);
(c) the format's row and column in one parametrized cross-format matrix
test that converts a fully populated `Metadata` into the format and back
and asserts `report.lost == {fields the class declares unsupported}`,
no more, no less, so an unsupported declaration that is wrong fails CI;
(d) deprecated aliases warn and still work.

## Decisions

- **M1** Geometry, axes, spatial/temporal units, transformations, the
  transform kind and storage encoding are never metadata.
- **M2** One `FormatMetadata` per format: `Magic` vocabulary fields + a
  format-private `struct` + an `extra` dict; a generic `Metadata` with
  no struct and nothing unsupported.
- **M3** Vocabulary = BIDS names in snake_case with BIDS units, ~35
  fields in six groups, each tagged with a propagation scope.
- **M4** `Magic` is used for the vocabulary fields, for `from_other`/
  `from_instance` conversion, and for polymorphic construction on
  `format`; not for the sentinel or for picking struct types.
- **M5** `UNSUPPORTED` singleton; `Maybe[T] = Union[T, None,
  Unsupported]`; class-level declaration by default value, per-instance
  where the capability is per-instance; never copied across formats.
- **M6** Overlay precedence: the struct is the base, non-`None` common
  fields win on write; `_decode`/`_encode` hooks; eager decode.
- **M7** `ConversionReport`; `on_loss` = `ignore`/`warn` (default)/
  `raise`; one aggregated warning; `_import` hook for recovery.
- **M8** `extra` passes unknown keys through to any free-form store;
  BIDS JSON sidecar is a codec on `Metadata`.
- **M9** `derive(grid_changed, volumes, step)` driven by field scope;
  one metadata per multiscale pyramid; composition does not merge.
- **M10** `metadata: Optional[Metadata]` on `Image` and `Transformation`,
  narrowed and made mandatory by each format class.
- **M11** `Transformation.metadata_fields` becomes `encoding_fields`,
  alias kept one minor release.
- **M12** Seven PRs, framework first, deprecated aliases for one minor
  release, cross-format matrix test as the capability oracle.

## Open questions for the maintainer

1. **Overlay versus views (M6).** Overlay (common fields win, struct is
   the fallback) is recommended; it keeps the vocabulary as real `Magic`
   fields. The cost is that editing the struct for a covered key after a
   read is silently overridden. Alternative: views over the struct,
   nibabel style, at the price of losing `from_other`/`replace`/`eq` on
   the vocabulary. Recommendation: overlay.
2. **Where the field lives (M10).** On the datamodel roots (`Image`,
   `Transformation`, optional, `None` by default) so in-memory objects
   carry provenance and `from_instance` treats it as shared; or only on
   the file-based mixins, keeping the datamodel metadata-free as the x5
   docs currently promise. Recommendation: datamodel roots, excluded
   from equality.
3. **Default loss policy (M7).** `warn` with one aggregated warning, or
   silent `ignore` with a report available on request. Recommendation:
   `warn`; a converter that drops a user's slice timing without a word
   is the current behaviour we are trying to leave.
4. **How hard is the sentinel (M5).** Should `from_dict`/the constructor
   *refuse* a value for a field the class declares unsupported (like
   `pin+narrow` refuses a contradicting discriminant), or accept it and
   let the writer report it? Recommendation: refuse in `from_dict`/
   constructor, accept on attribute assignment, report at write; this
   catches typos in sidecars while leaving an escape hatch.
5. **Diffusion frame (M3).** World/RAS unit vectors (MRtrix convention),
   with NRRD and BIDS codecs rotating via the affine; or voxel-axis
   vectors (BIDS/FSL convention). Recommendation: world; it is the only
   frame that survives a reorientation without touching metadata.
6. **Polymorphic `format` discriminant (M4).** Keep `FormatMetadata(
   format="nifti", ...)` dispatch, or make subclasses plain and
   construct them by name only. Recommendation: keep; it costs one line
   per class and matches `ItkStruct`.
7. **`intent` vocabulary (M3).** NIfTI intent names as the canonical
   set (`"Z-score"`, `"Label"`, `"Displacement vector"`), or a smaller
   brainhops enum. Recommendation: NIfTI names in v1; nothing else has a
   richer set and AFNI/NRRD map onto it.
8. **NIfTI extras.** Drop `extra` for NIfTI (declared unsupported,
   reported as lost), or write it into a NIfTI extension (a JSON
   extension with a brainhops-chosen ecode) that only brainhops reads.
   Recommendation: unsupported in v1; an extension is a later opt-in
   writer option, because other tools would carry an opaque blob.
9. **Sidecar integration (M8).** Codec only (`Metadata.from_bids`/
   `to_bids`), or also `io.load(path, sidecar=True)` / `io.save(...,
   sidecar=True)` reading and writing `<stem>.json` next to the image.
   Recommendation: codec in the framework PR, `sidecar=` in the NIfTI PR
   where it is actually used.
10. **Rename `metadata_fields` (M11).** `encoding_fields` with a
    warning alias, or leave the clash and document it. Recommendation:
    rename; the alias is three lines.
11. **Open PRs.** Merge JP2, MRtrix warps and AFNI xforms as they are and
    migrate them in batches 3 and 6, or hold them until the framework
    lands. Recommendation: merge now; each migration is small and the
    framework PR touches no format.
12. **Deprecation window (M12).** One minor release for the aliases, or
    two. Recommendation: one, since the library is pre-1.0 and the
    aliases are mechanical.
