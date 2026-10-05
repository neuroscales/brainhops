# Metadata

Images and transformations carry more than voxels and coordinates: a
description, a repetition time, the slice timing of an fMRI run, the
programs that produced a file... Every file format stores these under its
own names and units. brainhops reads them into one representation, so that
they survive a change of format, and says so when they cannot.

Every image and transformation has a `metadata` attribute. The metadata
classes mirror the image classes (`Image`, `FileBasedImage`,
`NiftiImage`):

- objects built in memory hold a format-agnostic
  [`Metadata`][brainhops.datamodel.metadata.Metadata] (or `None`);
- objects read from a file hold the metadata of their format, such as
  [`NiftiMetadata`][brainhops.io.images.nifti.NiftiMetadata], a
  [`FileBasedMetadata`][brainhops.datamodel.metadata.FileBasedMetadata]
  whose `raw` attribute is the format's own raw record (the `nibabel`
  header, for NIfTI).

All metadata classes share one vocabulary of fields, named after the
[BIDS](https://bids-specification.readthedocs.io) keys in snake case and
stored in BIDS units: `repetition_time` is `RepetitionTime`, in seconds.
A field holds a value, `None` (unknown), or
[`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED] (this format has
no place to store it). The vocabulary comes in seven groups, each a class
that `Metadata` inherits:
[`ProvenanceVocabulary`][brainhops.datamodel.metadata.ProvenanceVocabulary]
(description, history, space, ...),
[`MRIVocabulary`][brainhops.datamodel.metadata.MRIVocabulary] (repetition
time, slice timing, ...),
[`DiffusionVocabulary`][brainhops.datamodel.metadata.DiffusionVocabulary]
(`bvalues`, `bvectors`),
[`DisplayVocabulary`][brainhops.datamodel.metadata.DisplayVocabulary]
(display range, channels, unit of the values),
[`StorageVocabulary`][brainhops.datamodel.metadata.StorageVocabulary]
(type and intensity scaling of the stored values),
[`MicroscopyVocabulary`][brainhops.datamodel.metadata.MicroscopyVocabulary]
and
[`TransformVocabulary`][brainhops.datamodel.metadata.TransformVocabulary]
(`moving`, `fixed`, ...).

The examples on this page run as they are. They write their files to a
temporary directory:

```python
>>> import os, tempfile, warnings
>>> import numpy as np
>>> import nibabel as nb
>>> from brainhops import io
>>> from brainhops.datamodel.metadata import Metadata, UNSUPPORTED
>>> tmp = tempfile.mkdtemp()

```

## Reading metadata

Let us write a small fMRI run with `nibabel`, with a description, a display
range and a slice timing in its header:

```python
>>> nii = nb.Nifti1Image(np.zeros((4, 5, 6, 10), "float32"), np.eye(4))
>>> nii.header["descrip"] = b"resting state, run 1"
>>> nii.header["cal_min"], nii.header["cal_max"] = 0, 1000
>>> nii.header.set_xyzt_units("mm", "sec")
>>> nii.header.set_dim_info(phase=1, slice=2)
>>> nii.header.set_slice_duration(0.5)
>>> nii.header["slice_code"] = 1  # sequential, increasing
>>> nb.save(nii, os.path.join(tmp, "bold.nii.gz"))

```

The header is decoded into the common fields when the file is read:

```python
>>> bold = io.load(os.path.join(tmp, "bold.nii.gz"))
>>> bold.metadata.description
'resting state, run 1'
>>> bold.metadata.slice_timing
(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
>>> bold.metadata.display_range, bold.metadata.data_type
((0.0, 1000.0), dtype('float32'))

```

`data_type` is the type of the data in the file, which may differ from
the type of the loaded array. A file that stores integers with an
intensity scaling (`scl_slope` and `scl_inter`) loads as floating-point
values, and its metadata holds the scaling as `scale_slope` and
`scale_intercept`.

The header itself is the raw record of the metadata. It is still
available as `bold.header`, and it is the same object:

```python
>>> bold.metadata.raw is bold.header
True

```

A field NIfTI has no place for is `UNSUPPORTED`. The class says which:

```python
>>> bold.metadata.echo_time
UNSUPPORTED
>>> from brainhops.io.images.nifti import NiftiMetadata
>>> NiftiMetadata.supports("echo_time"), NiftiMetadata.supports("description")
(False, True)
>>> sorted(NiftiMetadata.supported_fields)  # doctest: +NORMALIZE_WHITESPACE
['data_type', 'description', 'display_range', 'intent',
 'phase_encoding_direction', 'repetition_time', 'scale_intercept',
 'scale_slope', 'slice_encoding_direction', 'slice_timing', 'sources',
 'space']

```

## Editing and saving

Reading and saving again keeps the header: the description, the slice
timing, the display range, the auxiliary file and the extensions are
written back as they were read. Geometry and units are always taken from
the image itself. The data is stored as `data_type` when its values are
of that kind (integers as an integer type, floats as a float type), so a
label map read as `uint8` is saved as `uint8`, but a resampled, floating
point version of it is not rounded. A scaled integer file is saved with
its scaling when the values still fit it, so that it is written back as
it was read; a resampled version is saved unscaled, as floats. A
`dtype=` option of `save` wins over both.

A field you set is written over the header; a field you set to `None` is
cleared in it. A field you leave alone keeps the header's value, so an
edit made directly to the header (for what the vocabulary does not cover)
survives (`aux_file` is the record of `sources`, which was not touched):

```python
>>> bold.metadata.description = "resting state, run 1, denoised"
>>> bold.metadata.display_range = None
>>> bold.metadata.raw["aux_file"] = b"sub-01_T1w.nii"
>>> bold.save(os.path.join(tmp, "denoised.nii.gz"))
>>> header = nb.load(os.path.join(tmp, "denoised.nii.gz")).header
>>> header["descrip"].item(), float(header["cal_max"]), int(header["slice_code"])
(b'resting state, run 1, denoised', 0.0, 1)
>>> header["aux_file"].item()
b'sub-01_T1w.nii'

```

## Format-agnostic metadata

[`Metadata`][brainhops.datamodel.metadata.Metadata] supports every field
and has no raw record. It is what in-memory objects carry, and the hub
through which formats convert. `to()` converts metadata into another
class, as images and transformations convert; given a
[`ConversionReport`][brainhops.datamodel.metadata.ConversionReport], it
fills it with what was lost:

```python
>>> from brainhops.datamodel.metadata import ConversionReport
>>> report = ConversionReport()
>>> generic = bold.metadata.to(Metadata, on_loss=report)
>>> generic.description, generic.slice_timing
('resting state, run 1, denoised', (0.0, 0.5, 1.0, 1.5, 2.0, 2.5))
>>> hasattr(generic, "raw")  # the raw record never leaves its format
False
>>> report.lossy
False

```

Converting back to NIfTI gives the same fields, without the raw record:

```python
>>> report = ConversionReport()
>>> back = generic.to(NiftiMetadata, on_loss=report)
>>> back == bold.metadata, report.lossy
(True, False)

```

The target may also be named by its format (`generic.to("nifti")`), and
`to()` with no class makes a copy.

Every class converts with `from_other` too, as the data model does:

```python
>>> NiftiMetadata.from_other(generic) == back
True

```

## Loss reports

Converting into a format that cannot hold a field reports the field as
lost. The report says what was dropped, and what was stored only
approximately:

```python
>>> scan = Metadata(description="T1w", echo_time=0.0029, flip_angle=8.0)
>>> report = ConversionReport()
>>> nifti = scan.to(NiftiMetadata, on_loss=report)
>>> report.lost
{'echo_time': 0.0029, 'flip_angle': 8.0}
>>> print(report)
Metadata conversion generic -> nifti: lost echo_time=0.0029, flip_angle=8.0.

```

Some losses depend on the value rather than on the field. NIfTI stores a
description of 80 bytes at most, and a slice timing only when it follows
one of its slice orders. `check_writable()` runs the writer without
writing, and says what a save would lose:

```python
>>> from brainhops.io.images.nifti import NiftiImage
>>> image = NiftiImage(data=np.zeros((4, 5, 6), "float32"))
>>> image.metadata.description = "x" * 81
>>> image.metadata.check_writable(image=image).approximated
{'description': 'truncated to 80 bytes (descrip)'}

```

What happens to a report is the *loss policy*: `"ignore"`, `"warn"` (the
default: one `MetadataLossWarning` per conversion or save, carrying the
report) or `"raise"` (a `MetadataLossError`). It is the `on_loss=` option
of `to()` and of `save`, which also takes a `ConversionReport`: the report
is then filled, and nothing is warned or raised (as in the examples
above). A policy and a report are exclusive: a report means "I will look
at it myself".

```python
>>> from brainhops.datamodel.metadata import MetadataLossError
>>> try:
...     image.save(os.path.join(tmp, "long.nii"), on_loss="raise")
... except MetadataLossError as error:
...     print(error)
Metadata conversion nifti -> nifti: approximated description (truncated to 80 bytes (descrip)).
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     image.save(os.path.join(tmp, "long.nii"))
>>> [type(w.message).__name__ for w in caught]
['MetadataLossWarning']
>>> caught[0].message.report.approximated
{'description': 'truncated to 80 bytes (descrip)'}
>>> report = ConversionReport()
>>> image.save(os.path.join(tmp, "long.nii"), on_loss=report)
>>> report.approximated
{'description': 'truncated to 80 bytes (descrip)'}

```

Conversions also happen implicitly, when an image changes format or is
given metadata of another class. They take no `on_loss=`; the
`metadata_loss_policy` context manager sets the policy for them:

```python
>>> from brainhops.datamodel.metadata import metadata_loss_policy
>>> with metadata_loss_policy("ignore"):
...     nifti = NiftiMetadata.from_other(scan)
>>> nifti.description
'T1w'

```

## Changing the format of an image

An image's metadata goes with it when it changes class. A generic image
holds generic metadata; making a NIfTI image of it converts the metadata
(and reports what NIfTI cannot hold), and saving it writes what was kept:

```python
>>> from brainhops.datamodel.images import SingleScaleImage
>>> generic_image = SingleScaleImage(
...     np.zeros((4, 5, 6), "float32"),
...     metadata=Metadata(description="from memory", flip_angle=8.0),
... )
>>> with metadata_loss_policy("ignore"):
...     io.save(generic_image, os.path.join(tmp, "memory.nii"))
>>> io.load(os.path.join(tmp, "memory.nii")).metadata.description
'from memory'

```

And the other way, a NIfTI image read as a plain image holds generic
metadata:

```python
>>> SingleScaleImage.from_instance(bold).metadata.description
'resting state, run 1, denoised'

```

An image holds its own metadata: one made from another (`replace()`,
`from_other`, `metadata=`) gets a copy, which shares the record but can
be edited on its own:

```python
>>> from bagof.magic import replace
>>> resampled = replace(bold, data=np.ones((4, 5, 6, 10), "float32"))
>>> resampled.metadata.description = "resampled"
>>> bold.metadata.description
'resting state, run 1, denoised'

```

## BIDS sidecars

A BIDS JSON sidecar reads into `Metadata`, and `Metadata` writes one.
Keys that name a field fill it; other keys are kept in `extra`:

```python
>>> sidecar = {
...     "RepetitionTime": 2.0,
...     "EchoTime": 0.03,
...     "SliceTiming": [0.0, 1.0, 0.5, 1.5],
...     "TaskName": "rest",
... }
>>> meta = Metadata.from_bids(sidecar)
>>> meta.repetition_time, meta.extra
(2.0, {'TaskName': 'rest'})
>>> meta.to_bids() == sidecar
True

```

`from_bids` also reads a path to a `.json` file, or a JSON string. A
sidecar converts into a format like any other metadata. Here, NIfTI has
no place for the echo time, nor for free-form keys such as the task name:

```python
>>> report = ConversionReport()
>>> nifti = meta.to(NiftiMetadata, on_loss=report)
>>> sorted(report.lost)
['echo_time', 'extra']

```

And NIfTI metadata becomes a sidecar (a field BIDS has no key for, such
as the data type, is written under its name in `CamelCase`):

```python
>>> sidecar = bold.metadata.to(Metadata).to_bids()
>>> sidecar["SliceTiming"], sidecar["DataType"]
([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 'float32')

```

## Terms, units and directions

A free-text field with a list of known terms holds an enum member when its
value is one of them, and the string otherwise:
[`SpaceEnum`][brainhops.datamodel.enums.SpaceEnum] for `space`, `input_space` and
`output_space` (the NIfTI space names and the BIDS templates),
[`IntentEnum`][brainhops.datamodel.enums.IntentEnum] (the NIfTI intent names),
[`Manufacturer`][brainhops.datamodel.enums.Manufacturer],
[`IlluminationType`][brainhops.datamodel.enums.IlluminationType] and
[`ContrastMethod`][brainhops.datamodel.enums.ContrastMethod]. A member is
a string too:

```python
>>> term = Metadata(space="MNI152NLin6Asym", intent="my own intent")
>>> term.space, term.space == "MNI152NLin6Asym"
(<SpaceEnum.MNI152NLin6Asym: 'MNI152NLin6Asym'>, True)
>>> term.intent
'my own intent'

```

`data_unit` is a [`Unit`][brainhops.datamodel.units.Unit] whenever the
units module parses its name, and the name itself otherwise, so that a
file with an odd unit still reads. Units compare as units, and a format
writes one as its symbol, which reads back as the same unit:

```python
>>> from brainhops.datamodel.units import Unit
>>> unit = Metadata(data_unit="a.u.").data_unit
>>> unit, unit == Unit("au"), unit.symbol
('arbitrary_unit', True, 'a.u.')
>>> Metadata(data_unit="mm/s").to_bids()
{'DataUnit': 'mm / s'}
>>> Metadata(data_unit="mm2/s").data_unit
'mm2/s'

```

An encoding direction (`phase_encoding_direction`,
`slice_encoding_direction`) is an
[`EncodingDirection`][brainhops.datamodel.metadata.EncodingDirection]: a
unit vector, by default in the voxel axes of the image, where the BIDS
string `"j-"` stands for `(0, -1, 0)` (`to_bids()` gives the string
back). A resampling that maps the voxel axes maps the direction too,
even when it is no longer along an axis (see the next section):

```python
>>> bold.metadata.slice_encoding_direction
EncodingDirection('k')
>>> bold.metadata.slice_encoding_direction.to_bids()
'k'
>>> swap = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
>>> Metadata(phase_encoding_direction="j-").derive(
...     changed={"space": None}, spatial_map=swap
... ).phase_encoding_direction
EncodingDirection('i-')

```

A format that can only store an axis (NIfTI `dim_info`, BIDS) reports a
direction that is along none as lost.

## Derived images

An image computed from another one (resampled, cropped, or a selection
of its volumes) gets metadata derived from the metadata of its source,
with `derive`. What a field becomes depends on its
[`Scope`][brainhops.datamodel.metadata.Scope], which says what the field
is tied to:

- a field about the file or the data as a whole (`FILE`), such as the
  description, is kept, and the provenance records the derivation;
- a field that describes the acquisition (`ACQUISITION`), such as the
  echo time, is kept;
- a field tied to the spatial sampling (`SPATIAL`), such as the slice
  timing, is cleared when the spatial axes change, except an encoding
  direction, which follows a known map of the voxel axes;
- a field with one entry per index along a non-spatial axis (`AXIS`) is
  indexed when that axis changes. The b-values and b-vectors run along
  the time axis, and the channel descriptions along the channel axis.

The caller says which axes changed, by
[`AxisType`][brainhops.datamodel.enums.AxisType], and gives the indices
that were kept along them, or `None` when the change is not a selection:

```python
>>> dwi = Metadata(
...     bvalues=(0.0, 1000.0, 1000.0, 2000.0),
...     slice_timing=(0.0, 0.5, 1.0),
...     echo_time=0.08,
... )
>>> first = dwi.derive(changed={"time": [0, 1, 2]}, step="select")
>>> first.bvalues, first.slice_timing, first.history
((0.0, 1000.0, 1000.0), (0.0, 0.5, 1.0), ('select',))
>>> resliced = dwi.derive(changed={"space": None})
>>> resliced.bvalues, resliced.slice_timing, resliced.echo_time
((0.0, 1000.0, 1000.0, 2000.0), None, 0.08)

```

The metadata of a file format also keeps its raw record, from which the
format removes what the changed axes invalidate, so that a cleared field
is cleared in the file as well.

When an arbitrary image is written, a format maps the axes of the image
to its own slots by type: NIfTI stores the time axis as its fourth
dimension and the channel axis as its fifth, OME-Zarr names them `t` and
`c`. A per-axis field therefore goes with the axis it runs along.

## Formats

Each format stores a different part of the vocabulary. This section lists,
per format, what it reads and writes, and where. To add the metadata of a
format, see [Writing the metadata of a format](../dev/metadata-formats.md).

### NIfTI

[`NiftiMetadata`][brainhops.io.images.nifti.NiftiMetadata] is the
metadata of `NiftiImage` and of every NIfTI-based transformation (NIfTI
fields and affines, FSL FNIRT, ITK NIfTI fields, NiftyReg, SPM). Its record
is the `nibabel` header, also available as `metadata.header`.

| Field | Header | Notes |
|---|---|---|
| `description` | `descrip` | 80 bytes, longer is truncated (approximated) |
| `display_range` | `cal_min`, `cal_max` | |
| `sources` | `aux_file` | the first source only, 24 bytes |
| `slice_encoding_direction`, `phase_encoding_direction` | `dim_info` | no polarity: `"j-"` is stored as `"j"` (approximated) |
| `slice_timing` | `slice_code`, `slice_start`, `slice_end`, `slice_duration` | the NIfTI slice orders only, else lost |
| `repetition_time` | `pixdim[4]` | derived from the time axis |
| `intent` | `intent_code` | derived from the axes; set when they set none |
| `space` | `sform_code`, `qform_code` | derived from the world space name |

A *derived* field is read from the header, but on save the image's own
geometry wins, and a value that disagrees with it is reported as
approximated. The repetition time is the time step of the image, so a
4-D image keeps it through a read and a save; an image whose geometry has
no time step (one built in memory) stores the field's value there. NIfTI
has no free-form store, so `extra` is unsupported.

Transformations keep their metadata through a read and a save as images
do, and the readers that need them (FSL FNIRT, NiftyReg) still read
`intent_p1`..`intent_p3` from the header:

```python
>>> warp = nb.Nifti1Image(np.zeros((3, 4, 5, 1, 3), "float32"), np.eye(4))
>>> warp.header.set_intent("vector", name="Mapping")
>>> warp.header["descrip"] = b"a coordinates field"
>>> nb.save(warp, os.path.join(tmp, "y_warp.nii.gz"))
>>> field = io.load(os.path.join(tmp, "y_warp.nii.gz"), hint="coordinates")
>>> field.metadata.description, field.metadata.intent
('a coordinates field', <IntentEnum.vector: 'vector'>)

```

### MGH / MGZ

[`MghMetadata`][brainhops.io.images.freesurfer.mgh.MghMetadata] is the
metadata of `MghImage`. Its record is the `nibabel` header, whose footer
holds the MRI acquisition parameters, and the raw trailing tags.
FreeSurfer stores times in milliseconds and the flip angle in radians; the
metadata holds them in BIDS units.

| Field | Record | Notes |
|---|---|---|
| `repetition_time` | footer `tr` | ms -> s; zero means unknown |
| `echo_time` | footer `te` | ms -> s |
| `inversion_time` | footer `ti` | ms -> s |
| `flip_angle` | footer `flip_angle` | rad -> deg |
| `history` | the command-line tags | only when the tags parse |

The field of view stays in the record, and MGH has no free-form store,
so `extra` is unsupported. Let us write a T1-weighted scan, as
`mri_convert` would, with its acquisition parameters and the command that
made it:

```python
>>> import gzip, struct
>>> from nibabel.freesurfer.mghformat import MGHImage
>>> mgh = MGHImage(np.zeros((4, 5, 6), "float32"), np.eye(4))
>>> mgh.header["tr"], mgh.header["te"], mgh.header["ti"] = 2300, 2.98, 900
>>> mgh.header["flip_angle"] = np.deg2rad(9)
>>> nb.save(mgh, os.path.join(tmp, "T1.mgz"))
>>> command = b"mri_convert T1.nii.gz T1.mgz\0"
>>> tag = struct.pack(">iq", 3, len(command)) + command  # TAG_CMDLINE
>>> with open(os.path.join(tmp, "T1.mgz"), "rb") as f:
...     content = gzip.decompress(f.read()) + tag
>>> with open(os.path.join(tmp, "T1.mgz"), "wb") as f:
...     _ = f.write(gzip.compress(content))
>>> t1 = io.load(os.path.join(tmp, "T1.mgz"))
>>> t1.metadata.repetition_time, t1.metadata.echo_time, t1.metadata.flip_angle
(2.3, 0.00298, 9.0)
>>> t1.metadata.history
('mri_convert T1.nii.gz T1.mgz',)

```

A field you set is written back in FreeSurfer's units, and the tags that
are not command lines are kept. The writer's `tr=`, `te=`, `ti=` and
`flip_angle=` keywords (in milliseconds and radians, as before) set the
same fields:

```python
>>> t1.metadata.echo_time = 0.0035
>>> t1.metadata.history += ("recon-all -s bert",)
>>> t1.save(os.path.join(tmp, "T1_edited.mgz"))
>>> edited = io.load(os.path.join(tmp, "T1_edited.mgz"))
>>> round(edited.mri_params["te"], 3), edited.metadata.history
(3.5, ('mri_convert T1.nii.gz T1.mgz', 'recon-all -s bert'))

```

NIfTI has no place for the echo time, the inversion time, the flip angle
or the history, so saving the scan as NIfTI loses them, and the report says
so (a 3-D NIfTI image has no time axis to hold the repetition time
either). The save converts the image to NIfTI, then writes it; it warns
once, with one report for both:

```python
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     io.save(edited, os.path.join(tmp, "T1.nii.gz"))
>>> len(caught)
1
>>> sorted(caught[0].message.report.lost)
['echo_time', 'flip_angle', 'history', 'inversion_time', 'repetition_time']

```

The tags follow the whole volume, so reading them decompresses an MGZ file
to its end. They are read, and `history` decoded, only when it is first
used: loading an MGZ file reads the header and the footer only.

Going through format-agnostic metadata keeps them all, and a BIDS sidecar
holds them in BIDS units, ready to sit next to the NIfTI file:

```python
>>> report = ConversionReport()
>>> generic = edited.metadata.to(Metadata, on_loss=report)
>>> report.lossy
False
>>> sidecar = generic.to_bids()
>>> sidecar["RepetitionTime"], sidecar["EchoTime"], sidecar["FlipAngle"]
(2.3, 0.0035, 9.0)
>>> sidecar["InversionTime"], sidecar["History"]
(0.9, ['mri_convert T1.nii.gz T1.mgz', 'recon-all -s bert'])

```

### Zarr and OME-Zarr

[`OmeZarrMetadata`][brainhops.io.images.zarr.OmeZarrMetadata] is the
metadata of an OME-Zarr pyramid (`OmeZarrImage`). Its record is the
multiscale (as `abczarr` reads it), the `omero` rendering settings and the
other attributes of the group. There is one metadata object per pyramid:
each level holds a copy derived from it, and the pyramid's is the one
written.

| Field | Record | Notes |
|---|---|---|
| `name` | multiscale `name` | |
| `channels` | `omero.channels` | `label`, `color`, `window` |
| `display_range` | `omero.channels[*].window` | when every channel shares it |
| `extra` | the other group attributes | |
| `data_type` | the data type of the arrays | derived from the data |

OME-Zarr has no unit for the values, so `data_unit` is unsupported.
`data_type` is the type of the arrays: a writer stores the array as it
is, and reports a `data_type` that disagrees with it as approximated. Every
channel needs a display window: one that nothing gives (no display range)
is written as the range of the values of the smallest level, and reported
as approximated. A
pyramid of a two-channel stain, with its channel names and colors:

```python
>>> from brainhops.datamodel.axes import ChannelAxis, SpaceAxis
>>> from brainhops.datamodel.metadata import Channel
>>> from brainhops.io.images.zarr import OmeZarrImage, OmeZarrMetadata
>>> stain = OmeZarrImage(
...     images=[SingleScaleImage(np.zeros((8, 8, 4, 2), "uint16"))],
...     axes=[SpaceAxis("x"), SpaceAxis("y"), SpaceAxis("z"), ChannelAxis("c")],
...     metadata=OmeZarrMetadata(
...         name="slide 3",
...         channels=(
...             Channel(name="DAPI", color="0000FF", display_range=(0, 900)),
...             Channel(name="GFP", color="00FF00", display_range=(0, 500)),
...         ),
...     ),
... )
>>> stain.save(os.path.join(tmp, "stain.ome.zarr"))
>>> stain = io.load(os.path.join(tmp, "stain.ome.zarr"))
>>> stain.metadata.name, [c.name for c in stain.metadata.channels]
('slide 3', ['DAPI', 'GFP'])
>>> stain.metadata.channels[0]
Channel(name='DAPI', color='0000FFFF', display_range=(0.0, 900.0))

```

The channel names go through format-agnostic metadata as they are, and
NIfTI, which has no channel names, reports them:

```python
>>> report = ConversionReport()
>>> generic = stain.metadata.to(Metadata, on_loss=report)
>>> [c.name for c in generic.channels], report.lossy
(['DAPI', 'GFP'], False)
>>> report = ConversionReport()
>>> _ = stain.metadata.to(NiftiMetadata, on_loss=report)
>>> sorted(report.lost)
['channels', 'name']

```

A plain Zarr array (`ZarrImage`) has no metadata convention, only
attributes:
[`ZarrMetadata`][brainhops.io.images.zarr.ZarrMetadata] stores the
vocabulary as a BIDS sidecar under the attribute `"brainhops"`, so every
field but the diffusion ones survives (`data_type` is the array's, as for
OME-Zarr), and `extra` maps to the other attributes. The attributes
are its raw record, also available as `metadata.attributes`.

### Transformations: x5, ITK and FLIRT

| Format | Metadata class | What it stores |
|---|---|---|
| x5 (`.x5`) | [`X5Metadata`][brainhops.io.transformations.x5.X5Metadata] | every field, in the JSON `Metadata` of the node |
| ITK `.h5` | [`ItkH5Metadata`][brainhops.io.transformations.itk.ItkH5Metadata] | `generated_by`, from `/ITKVersion` |
| ITK `.tfm`, `.mat` | [`ItkMetadata`][brainhops.io.transformations.itk.ItkMetadata] | nothing |
| FSL FLIRT `.mat` | [`FlirtMetadata`][brainhops.io.transformations.fsl.flirt.FlirtMetadata] | nothing in the file; `moving` and `fixed` in memory |

An x5 node stores its metadata as a JSON object: each field under its BIDS
key (`Description`, `GeneratedBy`), or under its name in `CamelCase` when
BIDS has none (`History`, `Moving`, `Fixed`, `InputSpace`, `OutputSpace`).
The other keys are `extra`. The raw record of the metadata is the pair
`(header, node)`. Here is a displacement field written as nitransforms
writes one:

```python
>>> import h5py, json
>>> with h5py.File(os.path.join(tmp, "warp.x5"), "w") as f:
...     f.attrs["Format"], f.attrs["Version"] = "X5", np.uint16(1)
...     node = f.create_group("TransformGroup/0")
...     node.attrs["Type"], node.attrs["SubType"] = "nonlinear", "densefield"
...     node.attrs["Representation"] = "displacements"
...     node.attrs["Metadata"] = json.dumps({
...         "Description": "sub-01 T1w to MNI",
...         "GeneratedBy": [{"Name": "fMRIPrep", "Version": "24.1.0"}],
...         "InputSpace": "T1w",
...         "OutputSpace": "MNI152NLin2009cAsym",
...         "WrittenBy": "NiTransforms 25.1.0",
...     })
...     _ = node.create_dataset("Transform", data=np.zeros((3, 4, 5, 3)))
...     domain = node.create_group("Domain")
...     _ = domain.create_dataset("Size", data=[3, 4, 5])
...     _ = domain.create_dataset("Mapping", data=np.eye(4))
>>> x5 = io.load(os.path.join(tmp, "warp.x5"))
>>> x5.metadata.description, x5.metadata.output_space
('sub-01 T1w to MNI', <SpaceEnum.MNI152NLin2009cAsym: 'MNI152NLin2009cAsym'>)
>>> x5.metadata.extra
{'WrittenBy': 'NiTransforms 25.1.0'}
>>> x5.metadata.node is x5.nodes[0]
True

```

A file read and saved again keeps the JSON of every node as it was; an
edited field is written into it. The metadata is that of the node the
transformation was read from: a chain of several nodes has none of its
own, and each node keeps its JSON (composition does not merge).

x5 metadata becomes a BIDS sidecar through `Metadata`, losing nothing:

```python
>>> x5.metadata.to(Metadata).to_bids()["OutputSpace"]
'MNI152NLin2009cAsym'

```

A NIfTI displacement field holds only a description of all this. Making
one from the x5 file (or from its field) converts its metadata, and
reports the rest:

```python
>>> from brainhops.io.transformations.nifti import NiftiRASDisplacementField
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     warp = NiftiRASDisplacementField.from_other(x5)
>>> warp.metadata.description
'sub-01 T1w to MNI'
>>> sorted(caught[0].message.report.lost)
['extra', 'generated_by', 'input_space', 'output_space']

```

The field of a file of one node carries the metadata of the node too
(a copy), so `NiftiRASDisplacementField.from_other(x5.transformations[0])`
converts it the same way. The nodes of a chain of several keep their own
JSON, and their fields carry nothing.

ITK `.tfm` and `.mat` files store a bare chain of parameters, and no
metadata at all, so everything is lost:

```python
>>> from brainhops.io.transformations.itk import ItkMetadata
>>> report = ConversionReport()
>>> itk = x5.metadata.to(ItkMetadata, on_loss=report)
>>> print(report)  # doctest: +ELLIPSIS
Metadata conversion x5 -> itk: lost extra=..., description='sub-01 T1w to MNI', generated_by=..., input_space='T1w', output_space='MNI152NLin2009cAsym'.
>>> itk.description
UNSUPPORTED

```

An ITK `.h5` file records the version of ITK that wrote it, read as
`generated_by`. The blocks of an ITK chain (or composite) have no metadata
of their own.

A FLIRT matrix stores nothing either, but its metadata is not empty: when
the moving and reference images given to the reader were read from
files, their paths are `moving` and `fixed`. They are kept in memory (a
copy or a conversion carries them); a `.mat` file has no place for them,
so a write would lose them.
