# Metadata

Images and transformations carry more than voxels and coordinates: a
description, a repetition time, the slice timing of an fMRI run, the
programs that produced a file... Every file format stores these under its
own names and units. brainhops reads them into one representation, so that
they survive a change of format, and says so when they cannot.

Every image and transformation has a `metadata` attribute:

- objects built in memory hold a format-agnostic
  [`Metadata`][brainhops.datamodel.metadata.Metadata] (or `None`);
- objects read from a file hold the metadata of their format, such as
  [`NiftiMetadata`][brainhops.io.images.nifti.NiftiMetadata],
  whose `raw` attribute is the format's own record (the `nibabel` header,
  for NIfTI).

All metadata classes share one vocabulary of fields, named after the
[BIDS](https://bids-specification.readthedocs.io) keys in snake case and
stored in BIDS units: `repetition_time` is `RepetitionTime`, in seconds.
A field holds a value, `None` (unknown), or
[`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED] (this format has
no place to store it).

The examples on this page run as they are. They write their files to a
temporary directory:

```python
>>> import os, tempfile, warnings
>>> import numpy as np
>>> import nibabel as nb
>>> from brainhops import io
>>> from brainhops.datamodel import Metadata, UNSUPPORTED
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
>>> bold.metadata.slice_encoding_direction, bold.metadata.display_range
('k', (0.0, 1000.0))

```

The header itself is the record of the metadata. It is still available as
`bold.header`, and it is the same object:

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
>>> sorted(NiftiMetadata.unsupported_fields)[:4]
['acquisition_time', 'channels', 'contrast_method', 'creation_time']

```

## Editing and saving

Reading and saving again keeps the header: the description, the slice
timing, the display range, the auxiliary file and the extensions are
written back as they were read. Geometry, units, data type and intensity
scaling are always taken from the image itself.

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

`changed_fields()` lists what will be written over the record:

```python
>>> sorted(bold.metadata.changed_fields())
['description', 'display_range']

```

## Format-agnostic metadata

[`Metadata`][brainhops.datamodel.metadata.Metadata] supports every field
and has no record. It is what in-memory objects carry, and the hub through
which formats convert.
[`convert`][brainhops.datamodel.metadata.convert] (also available as
`brainhops.datamodel.convert_metadata`) converts metadata into another
class, and returns a report of what was lost:

```python
>>> from brainhops.datamodel.metadata import convert
>>> generic, report = convert(bold.metadata, Metadata)
>>> generic.description, generic.slice_timing
('resting state, run 1, denoised', (0.0, 0.5, 1.0, 1.5, 2.0, 2.5))
>>> generic.raw is None  # the record never leaves its format
True
>>> report.lossy
False

```

Converting back to NIfTI gives the same fields, without the record:

```python
>>> back, report = convert(generic, NiftiMetadata)
>>> back == bold.metadata, report.lossy
(True, False)

```

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
>>> nifti, report = convert(scan, NiftiMetadata, on_loss="ignore")
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
of `convert` and of `save`:

```python
>>> image.save(os.path.join(tmp, "long.nii"), on_loss="raise")
Traceback (most recent call last):
    ...
brainhops.datamodel.metadata.MetadataLossError: Metadata conversion nifti -> nifti: approximated description (truncated to 80 bytes (descrip)).
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     image.save(os.path.join(tmp, "long.nii"))
>>> [type(w.message).__name__ for w in caught]
['MetadataLossWarning']
>>> caught[0].message.report.approximated
{'description': 'truncated to 80 bytes (descrip)'}

```

Conversions also happen implicitly, when an image changes format or is
given metadata of another class. They take no `on_loss=`; the
`metadata_loss_policy` context manager sets the policy for them:

```python
>>> from brainhops.datamodel import metadata_loss_policy
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
>>> nifti, report = convert(meta, NiftiMetadata, on_loss="ignore")
>>> sorted(report.lost)
['echo_time', 'extra']

```

And NIfTI metadata becomes a sidecar through `Metadata`:

```python
>>> convert(bold.metadata, Metadata)[0].to_bids()["SliceTiming"]
[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

```

## Formats

Each format stores a different part of the vocabulary. This section lists,
per format, what it reads and writes, and where.

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
approximated. NIfTI has no free-form store, so `extra` is unsupported.

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
('a coordinates field', 'vector')

```

<!--
  Later phases append one subsection per format here (MGH, Zarr, ITK,
  x5, FLIRT, ...), with the same shape: what the format stores, a table
  of fields and native slots, and a runnable example, including a
  conversion from NIfTI metadata into that format's metadata.
-->
