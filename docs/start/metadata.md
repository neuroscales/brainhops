# Metadata

Images and transformations carry more than voxels and coordinates: a
description, a repetition time, the slice timing of an fMRI run, the
programs that produced a file. Every file format stores these under its
own names and units. brainhops reads them into one representation, so
that they survive a change of format, and it reports what a format
cannot hold.

Every image and every transformation has a `metadata` attribute. An
object built in memory holds generic
[`Metadata`][brainhops.datamodel.metadata.Metadata], or `None`. An
object read from a file holds the metadata of its format, such as
[`NiftiMetadata`][brainhops.io.images.nifti.NiftiMetadata], which is a
[`MetadataFormat`][brainhops.io.metadata.MetadataFormat]. The metadata
of a format also holds the header of the file in its `raw` attribute,
which is called the record of the file.

All metadata classes share one vocabulary of fields, named after the
[BIDS](https://bids-specification.readthedocs.io) keys in snake case and
stored in BIDS units: `repetition_time` is `RepetitionTime`, in seconds.
A field holds a value, or `None` when the value is unknown or when the
format has no slot for the field. The vocabulary comes in seven groups,
each a class that `Metadata` inherits:
[`ProvenanceVocabulary`][brainhops.datamodel.metadata.ProvenanceVocabulary]
(description, history, space),
[`MRIVocabulary`][brainhops.datamodel.metadata.MRIVocabulary]
(repetition time, slice timing),
[`DiffusionVocabulary`][brainhops.datamodel.metadata.DiffusionVocabulary]
(`bvalues`, `bvectors`),
[`DisplayVocabulary`][brainhops.datamodel.metadata.DisplayVocabulary]
(display range, channels, unit of the values),
[`StorageVocabulary`][brainhops.datamodel.metadata.StorageVocabulary]
(type and intensity scaling of the stored values),
[`MicroscopyVocabulary`][brainhops.datamodel.metadata.MicroscopyVocabulary]
and
[`TransformVocabulary`][brainhops.datamodel.metadata.TransformVocabulary]
(`moving`, `fixed`). The field `extra` holds the keys that the
vocabulary does not cover.

The examples on this page run as they are. They write their files to a
temporary directory:

```python
>>> import os, tempfile, warnings
>>> import numpy as np
>>> import nibabel as nb
>>> from bagof.magic import replace
>>> from brainhops import io
>>> from brainhops.datamodel.metadata import ConversionReport, Metadata
>>> tmp = tempfile.mkdtemp()

```

## Reading metadata

Let us write a small fMRI run with nibabel, with a description, a
display range and a slice timing in its header:

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

The header is decoded into the fields of the vocabulary when the file
is read:

```python
>>> bold = io.load(os.path.join(tmp, "bold.nii.gz"))
>>> bold.metadata.description
'resting state, run 1'
>>> bold.metadata.slice_timing
(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
>>> bold.metadata.display_range, bold.metadata.dtype
((0.0, 1000.0), dtype('float32'))

```

`dtype` is the type of the values in the file, which may differ from
the type of the loaded array. A file that stores integers with an
intensity scaling (`scl_slope` and `scl_inter`) loads as floating-point
values, and its metadata holds the scaling as `scale_slope` and
`scale_intercept`.

The header itself is the record of the metadata, held by a
[`NiftiRaw`][brainhops.io.images.nifti.NiftiRaw]. A field that NIfTI
has no slot for is `None`, and the constant `SUPPORTED_FIELDS` of each
metadata class lists the fields that its record can store:

```python
>>> from brainhops.io.images.nifti import NiftiMetadata
>>> type(bold.metadata.raw.header).__name__
'Nifti1Header'
>>> bold.metadata.echo_time is None
True
>>> "echo_time" in NiftiMetadata.SUPPORTED_FIELDS
False

```

### Reading the metadata alone

[`MetadataFormat.load`][brainhops.io.metadata.MetadataFormat.load] reads
the metadata of a file without its data. The format is found as
`io.load` finds it, from the name and the content of the file, and the
result is the metadata of that format, with its record:

```python
>>> from brainhops.io.metadata import MetadataFormat
>>> meta = MetadataFormat.load(os.path.join(tmp, "bold.nii.gz"))
>>> type(meta).__name__, meta.description
('NiftiMetadata', 'resting state, run 1')

```

The class of a format reads a file as one of its own
(`NiftiMetadata.load(path)`), and `hint=` names the format when the file
cannot tell, as for `io.load`.

## Editing and saving

Every write consolidates the metadata first: the fields that differ from
a fresh decode of the record are encoded into a copy of it, and the
geometry and the data of the image are encoded over that copy. A field
left alone keeps the value of the record, so the slots that the
vocabulary does not cover, such as the extensions of a NIfTI header, are
written back as they were read. The record held by the image is never
changed.

The copies of an image share its metadata object, and a change made in
place is seen by all of them. To change the metadata of one image only,
assign it new metadata:

```python
>>> denoised = replace(bold, data=np.ones((4, 5, 6, 10), "float32"))
>>> denoised.metadata = replace(
...     denoised.metadata,
...     description="resting state, run 1, denoised",
...     display_range=None,
... )
>>> bold.metadata.description
'resting state, run 1'
>>> denoised.save(os.path.join(tmp, "denoised.nii.gz"))
>>> header = nb.load(os.path.join(tmp, "denoised.nii.gz")).header
>>> header["descrip"].item(), float(header["cal_max"])
(b'resting state, run 1, denoised', 0.0)
>>> int(header["slice_code"])
1

```

Assigning other metadata drops the transformations that the image
decoded from the previous record, so that they are decoded from the new
one, and keeps the transformations that were assigned.

The keyword arguments of `save` and `consolidate` override the fields
of the metadata by name, so any field can be given when an image is
written. A writer option with the name of a field is that field: the
`dtype=` of a NIfTI or MGH writer is the field `dtype`. The values are
stored with the type given as `dtype=`, or else with `metadata.dtype`,
or else with the type of the data, as nibabel stores them with the type
of its header. A label map read as `uint8` and computed on as `int64` is
therefore written as `uint8` again, and floating-point values are scaled
into an integer type:

```python
>>> bold.save(os.path.join(tmp, "options.nii.gz"), description="run 1")
>>> io.load(os.path.join(tmp, "options.nii.gz")).metadata.description
'run 1'
>>> bold.save(os.path.join(tmp, "int16.nii.gz"), dtype="int16")
>>> nb.load(os.path.join(tmp, "int16.nii.gz")).get_data_dtype() == "int16"
True

```

A header field that holds a field of the vocabulary is refused as an
option, with the name of the field to give instead, so that a value is
never written twice under two names:

```python
>>> bold.save(os.path.join(tmp, "refused.nii.gz"), descrip="run 1")
Traceback (most recent call last):
...
TypeError: These header fields hold metadata, and cannot be overridden by name: descrip= (give description= instead).

```

## Generic metadata

[`Metadata`][brainhops.datamodel.metadata.Metadata] supports every field
and holds no record. It is what objects built in memory carry, and the
hub through which formats convert. `to()` converts metadata into another
class, field by field, as images and transformations convert:

```python
>>> generic = bold.metadata.to(Metadata)
>>> generic.description, generic.slice_timing
('resting state, run 1', (0.0, 0.5, 1.0, 1.5, 2.0, 2.5))

```

The record of a format means nothing to another format, so a conversion
copies it only into the same class. Metadata converted into a format
holds no record, and its values ride along until the next consolidation
encodes them into a new record of that format. Two generic metadata
objects are equal when their fields are equal, while the metadata of a
format compares by identity.

## Loss reports

A field that a format has no slot for is dropped when the metadata is
consolidated, so when it is written, and the drop is reported. A field
that the record holds only approximately, such as a description longer
than the 80 bytes of a NIfTI header, is reported as approximated. A
conversion into a format reports only a lower bound of what a write will
lose, from `SUPPORTED_FIELDS`:

```python
>>> scan = Metadata(description="T1w", echo_time=0.0029, flip_angle=8.0)
>>> report = ConversionReport()
>>> nifti = scan.to(NiftiMetadata, lossy=report)
>>> report.lost
{'echo_time': 0.0029, 'flip_angle': 8.0}
>>> nifti.echo_time
0.0029
>>> report = ConversionReport()
>>> nifti.consolidate(on_error=report).echo_time is None
True
>>> report.lost
{'echo_time': 0.0029, 'flip_angle': 8.0}

```

What happens to a report is decided by an error policy (see
[`ErrorPolicy`][brainhops.errors.ErrorPolicy]). A loss of metadata is a
`MetadataLossError`, which is a kind of `LossyConversionError`, and the
policy acts on it as on any other loss: `"ignore"` drops it, `"warn"`
issues one `MetadataLossWarning` that carries the report, `"raise"`
raises the error, and a `ConversionReport` is filled instead of warning
or raising. The `to()` method of metadata takes the action for a loss
as its `lossy=` option. The `to()` method, `consolidate` and `save` take
a whole policy as their `on_error=` option, which also decides what
happens to their other conversion errors. A loss that no option covers
follows the policy in effect, which is `"warn"` unless the
[`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy]
context manager sets another one. That context manager also sets the
policy of the conversions that take no option, such as those of the
fields of an image.

`io.save` converts an object into the format that the file name calls
for, and then writes it. The conversion and the write fill one report,
and the policy acts on it once:

```python
>>> from brainhops.datamodel.images import SingleScaleImage
>>> from brainhops.datamodel.metadata import MetadataLossError
>>> image = SingleScaleImage(np.zeros((4, 5, 6), "float32"), metadata=scan)
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     io.save(image, os.path.join(tmp, "memory.nii"))
>>> len(caught), sorted(caught[0].message.report.lost)
(1, ['echo_time', 'flip_angle'])
>>> io.load(os.path.join(tmp, "memory.nii")).metadata.description
'T1w'
>>> try:
...     io.save(image, os.path.join(tmp, "raise.nii"), on_error="raise")
... except MetadataLossError as error:
...     print(sorted(error.report.lost))
['echo_time', 'flip_angle']

```

## BIDS sidecars

The functions [`from_bids`][brainhops.io.metadata.from_bids] and
[`to_bids`][brainhops.io.metadata.to_bids] read a BIDS JSON sidecar into
generic metadata, and write metadata as one. A key that names a field
fills it, and the other keys are kept in `extra`:

```python
>>> from brainhops.io.metadata import from_bids, to_bids
>>> sidecar = {
...     "RepetitionTime": 2.0,
...     "EchoTime": 0.03,
...     "SliceTiming": [0.0, 1.0, 0.5, 1.5],
...     "TaskName": "rest",
... }
>>> meta = from_bids(sidecar)
>>> meta.repetition_time, meta.extra
(2.0, {'TaskName': 'rest'})
>>> to_bids(meta) == sidecar
True

```

`from_bids` also reads a path to a `.json` file, or a JSON string. A
sidecar file is also the metadata format
[`BidsSidecar`][brainhops.io.metadata.BidsSidecar], which
`MetadataFormat.load` reads.

## Terms, units and directions

A free-text field with a list of known terms holds an enum member when
its value is one of them, and the string otherwise:
[`SpaceEnum`][brainhops.datamodel.enums.SpaceEnum] for `space`,
`input_space` and `output_space`,
[`IntentEnum`][brainhops.datamodel.enums.IntentEnum] for the NIfTI
intent names,
[`Manufacturer`][brainhops.datamodel.enums.Manufacturer],
[`IlluminationType`][brainhops.datamodel.enums.IlluminationType] and
[`ContrastMethod`][brainhops.datamodel.enums.ContrastMethod]. A member
is a string too:

```python
>>> term = Metadata(space="MNI152NLin6Asym", intent="my own intent")
>>> term.space, term.space == "MNI152NLin6Asym"
(<SpaceEnum.MNI152NLin6Asym: 'MNI152NLin6Asym'>, True)
>>> term.intent
'my own intent'

```

An encoding direction (`phase_encoding_direction`,
`slice_encoding_direction`) is an
[`EncodingDirection`][brainhops.datamodel.metadata.EncodingDirection]: a
unit vector, by default in the voxel axes of the image, where the BIDS
string `"j-"` stands for `(0, -1, 0)`. An operation that maps the voxel
axes, such as a flip or a resampling, maps the direction too:

```python
>>> bold.metadata.phase_encoding_direction
EncodingDirection('j')
>>> bold[:, ::-1].metadata.phase_encoding_direction
EncodingDirection('j-')

```

## Derived images

An image computed from another one, by an index, a resampling or a
change of the frame of its vectors and tensors, gets metadata derived
from the metadata of its source, with
[`derive`][brainhops.datamodel.metadata.Metadata.derive]. What a field
becomes depends on its [`Scope`][brainhops.datamodel.metadata.Scope]:

- a field about the file or the data as a whole (`FILE`), such as the
  description, is kept;
- a field that describes the acquisition (`ACQUISITION`), such as the
  echo time, is kept;
- a field tied to the spatial sampling (`SPATIAL`), such as the slice
  timing, is cleared when the spatial axes move, except an encoding
  direction, which follows the map of the voxel axes;
- a field with one entry per position along a non-spatial axis (`AXIS`),
  such as the b-values along the time axis, keeps the entries of the
  positions that an index keeps, and is cleared when the values along
  that axis are re-expressed in another frame;
- a field that describes the values themselves (`VALUES`), such as the
  unit or the display range, is kept, except when the values are
  re-expressed in another frame, which clears it.

The derivation is recorded: `creation_time` is cleared, the step is
appended to `history`, and `generated_by` gains a brainhops entry. A
format that has no slot for these two fields keeps them as they were.

```python
>>> dwi = SingleScaleImage.from_instance(bold)
>>> dwi.metadata = replace(
...     dwi.metadata.to(Metadata),
...     bvalues=(0.0,) + (1000.0,) * 4 + (2000.0,) * 5,
...     echo_time=0.08,
... )
>>> first = dwi[..., :3]
>>> first.metadata.bvalues, first.metadata.slice_timing[:2]
((0.0, 1000.0, 1000.0), (0.0, 0.5))
>>> flipped = dwi[:, ::-1]
>>> flipped.metadata.slice_timing, flipped.metadata.echo_time
(None, 0.08)
>>> flipped.metadata.history
('getitem',)

```

The derived metadata keeps its class and its record. A derived NIfTI
image therefore writes the extensions and the other slots of the header
that it was read with, and a field that the derivation cleared differs
from the record, so the next consolidation clears its slot:

```python
>>> type(bold[:, ::-1].metadata).__name__
'NiftiMetadata'
>>> bold[..., :2].metadata.raw is bold.metadata.raw
True

```

`reslice()` derives the metadata in the same way, through the map from
the new voxels to the old ones, and `image(transformation)`, which moves
no voxel, shares the metadata of the image.

## Formats

Each format stores a different part of the vocabulary. This section
lists, per format, what it reads and writes, and where. To add the
metadata of a format, see
[Writing the metadata of a format](../dev/metadata-formats.md).

### NIfTI

[`NiftiMetadata`][brainhops.io.images.nifti.NiftiMetadata] is the
metadata of `NiftiImage` and of every transformation stored in a NIfTI
file (the NIfTI affines and fields, FSL FNIRT, ITK NIfTI fields,
NiftyReg and SPM). Its record is a `NiftiRaw`, which holds the nibabel
header.

| Field | Header | Notes |
|---|---|---|
| `description` | `descrip` | 80 bytes, longer is truncated (approximated) |
| `sources` | `aux_file` | the first source only, 24 bytes |
| `intent` | `intent_code` | an intent that changes how the axes are read is approximated |
| `space` | `sform_code`, `qform_code` | decided by the geometry (approximated) |
| `display_range` | `cal_min`, `cal_max` | |
| `phase_encoding_direction`, `slice_encoding_direction` | `dim_info` | an axis, without polarity |
| `slice_timing` | `slice_code` and the other `slice_*` | the NIfTI slice orders only |
| `repetition_time` | `pixdim[4]` | the time step of the image |
| `dtype` | `datatype`, `bitpix` | |
| `scale_slope`, `scale_intercept` | `scl_slope`, `scl_inter` | |

The image owns its geometry, so a field that the geometry encodes over,
such as a repetition time that disagrees with the time step of a 4-D
image, is reported as approximated. The header fields `descrip`,
`aux_file`, `cal_min`, `cal_max`, `scl_slope`, `scl_inter`, the `slice_*`
fields and `dim_info` are refused as writer options, and `intent=` is
the field, which takes a NIfTI intent name. NIfTI has no free-form
store, so `extra` is not supported.

### MGH and MGZ

[`MghMetadata`][brainhops.io.images.freesurfer.mgh.MghMetadata] is the
metadata of `MghImage`. Its record holds the fixed header, the footer
of the MRI parameters, and the tags. FreeSurfer stores times in
milliseconds and the flip angle in radians, and the metadata holds them
in BIDS units, seconds and degrees.

| Field | Record | Notes |
|---|---|---|
| `repetition_time` | footer `tr` | ms to s, zero reads as unknown |
| `echo_time` | footer `te` | ms to s |
| `inversion_time` | footer `ti` | ms to s |
| `flip_angle` | footer `flip_angle` | rad to deg |
| `history` | the command-line tags | only when the tags parse |
| `dtype` | header `type` | uint8, int16, int32 or float32 |

The writer options `tr=`, `te=` and `ti=` are refused, with the field
to give instead, and `flip_angle=` is the field, in degrees. The
parameters given by keyword take precedence over those of a template
`like`, and a TR stated by a 4-D image takes precedence over both:

```python
>>> from brainhops.io.images.freesurfer.mgh import MghImage
>>> t1 = MghImage(data=np.zeros((4, 5, 6), "float32"))
>>> t1.save(os.path.join(tmp, "T1.mgz"), echo_time=0.003, flip_angle=9.0)
>>> t1 = io.load(os.path.join(tmp, "T1.mgz"))
>>> t1.metadata.echo_time, t1.metadata.flip_angle
(0.003, 9.0)

```

### X5

[`X5Metadata`][brainhops.io.transformations.x5.X5Metadata] stores every
field in the JSON `Metadata` of a transform of the file, each under its
BIDS key, or under its name in `CamelCase` when BIDS has none (`History`,
`Moving`, `InputSpace`). The other keys are `extra`. The transform that
holds the vocabulary of the file is the single transform of its first
chain, or the single transform of a file without chains, so a chain of
several transforms has no vocabulary of its own, and each transform
keeps its own JSON. That transform gives its transformation a generic
copy of the vocabulary, so converting the transformation into another
format carries it:

```python
>>> from brainhops.datamodel.systems import RASmm
>>> from brainhops.datamodel.transformations import Affine
>>> from brainhops.io.transformations.nifti import NiftiVoxelToRAS
>>> from brainhops.io.transformations.x5 import X5Transform
>>> affine = Affine(np.eye(4)[:-1], input=RASmm(), output=RASmm())
>>> X5Transform(
...     transformations=[affine], metadata=Metadata(description="reg")
... ).save(os.path.join(tmp, "reg.x5"))
>>> x5 = io.load(os.path.join(tmp, "reg.x5"))
>>> x5.metadata.description, x5.transformations[0].metadata.description
('reg', 'reg')

```

The converters into the NIfTI transformations pass the metadata of the
transformation that they convert, which the format converts into
`NiftiMetadata`:

```python
>>> Affine(np.eye(4)[:-1], metadata=Metadata(description="reg")).to(
...     NiftiVoxelToRAS
... ).metadata.description
'reg'

```

### Other formats

The records of the other formats, such as LTA, AFNI, MRtrix, NRRD,
MINC, Zarr, TIFF, FLIRT and the ITK formats, hold no field of the
vocabulary yet. Their metadata still holds the fields of the vocabulary,
and a field that is set is reported as lost when the object is
consolidated or written. The formats that are not migrated to records
yet, such as the OME-Zarr images, hold the metadata of their data model
and write none of it, and `io.save` reports its fields as lost when it
converts an object into one of them.
