# Transformations stored as images

This memo is agreed and is not implemented. No issue tracks it yet.

Many transformations are stored in an image file. A displacement field
in a NIfTI file, an AFNI warp and an MRtrix warp are images whose
voxels hold vectors, and whose header holds the grid of the field. This
memo decides that the format class of such a transformation keeps its
stored state in an image object of the matching image format, and
delegates reading, consolidation and writing to that image. It builds
on [Format records and their layering](format-records.md), whose rules
on pending data and consolidation it does not change.

## The problem

The low-level code is already shared. The records, the metadata and
the lazy arrays of each file format live in `io/common/<format>/`, and
both the image formats and the transformation formats import them. What
is repeated is the layer above, which manages the state of a format
object. `NiftiImage` (`io/images/nifti/_image.py`) and
`NiftiRASCoordinateField` (`io/transformations/nifti/_fields.py`) show
the repetition clearly.

- Both declare a `rawdata` field, take the `metadata` field of their
  model over as `_metadata`, and keep assigned data in a private `_data`
  field until it is consolidated.
- Their `_consolidate` methods follow the same steps. They split the
  options into metadata fields and header overrides, consolidate the
  metadata, return early when nothing is pending, encode the header and
  the array, decode new metadata from the new record, report the
  overruled fields, apply the loss policy, and finally either change the
  object or return a changed copy.
- Both encode a voxel-to-RAS grid into the header and handle the `like=`
  template and the header overrides.

The same pattern is repeated in the NIfTI-based transformations of
NiftyReg, ITK, SPM and FSL FNIRT, in the MRtrix warps and in the AFNI
warps. Some formats already go through an image class, but each in its
own way. `AfniBrikWarp.from_fileobj` reads the dataset with
`AfniImage.from_fileobj` and keeps only its `rawdata` and its metadata
(`io/transformations/afni/_warp.py:314`). `AfniBrikWarp` and
`AfniNiftiWarp` have a `to_image()` method. FNIRT imports the views of
the NIfTI image format, and the grid transform of MINC reads its volume
as a `MincImage` (`io/transformations/minc/_grid.py:186`).

## Decision

### Two tiers of state

A transformation format that is stored as an image holds its state in
two tiers, as an image format already does.

The stored tier is a private field, `_image`, which holds an image of
the image format that the transformation is stored in, such as a
`NiftiImage`. The image describes the file as it was read, or as the
transformation was last consolidated. The properties `rawdata` and
`metadata` of the transformation read the `rawdata` and the metadata of
that image, and assigning them replaces the image with a copy that holds
the new value. Every transformation format therefore keeps `rawdata` and
`metadata`, whether or not it stores them in an image, so the interface
is the same across formats.

The model tier holds what the user assigned, in the private fields of
the data model, such as `_data` and `_transformations`. It is pending
until the transformation is consolidated, and it takes precedence over
the stored tier. `None` in one of these fields means that the property
of the same name decodes its value from the image, as described in the
section on data of [Format records](format-records.md#data).

The `data` of a transformation is not the `data` of its image. It is a
view of the stored array, built by the functions `to_model` and
`to_disk` of the format. For example, the view of a NIfTI field removes
the singleton axis of the `(X, Y, Z, 1, 3)` array that the file stores.
The `transformations` of a transformation are the slots of its chain,
while the `transformations` of an image are its voxel-to-world
transformations. Neither `data` nor `transformations` is therefore ever
passed through to the image.

### Constructor

The constructor keeps the arguments that it takes today. The fields of
the data model come first, such as `data` for a coordinate field and
`transformations` for a field stored as a chain. The keywords `image`,
`rawdata` and `metadata` are optional and keyword-only. The constructor
applies the keywords in the following order, so that a later keyword
overrides an earlier one.

1. `image` becomes the stored image.
2. `rawdata` and `metadata` replace the matching parts of the stored
   image. Without `image`, the constructor builds the stored image from
   them, for example with `NiftiImage(rawdata=..., metadata=...)`.
3. The fields of the data model are stored in their private fields, as
   values that the user defined.

Building a transformation with these keywords therefore gives the same
object as building it with `image` alone and then assigning `rawdata`,
`metadata` and the fields of the model in that order. `bagof.replace`,
which the consolidation code calls with `rawdata=` and `metadata=`,
keeps working without a special case.

Each format names the image class that it stores, and an image of
another class raises `TypeError`. An image that has pending data or
transformations is consolidated when it is given, so that the stored
tier always describes a file.

### Factories

A class method `from_image(image, **options)` builds a transformation
from an image. It is the same as `cls(image=image, **options)`. The
other factories, `from_fileobj`, `from_filename` and `from_bytes`, read
the file with the image class and then call `from_image`.

The method `to_image()` returns the image that holds what the
transformation would store. It consolidates the transformation and
returns the stored image of the copy. The stored image is not exposed
as a public property, because it describes the file as it was read and
may disagree with the pending state.

### Consolidation

Consolidation is unchanged in its meaning. It encodes the pending state
in memory and never writes a file, so that the stored objects and the
views of the transformation agree again. A transformation that is
stored as an image consolidates in three steps.

1. It builds an image from its pending state. The data is
   `to_disk(data)`, the grid of the chain is the voxel-to-world
   transformation of the image, and the codes that mark the format, such
   as the NIfTI intent code and intent name, are header overrides.
2. It calls `consolidate` on that image with the options of the call,
   such as `like=`, `dtype=` and header overrides. The precedence on
   write, the loss report and the choice between NIfTI-1 and NIfTI-2
   are therefore implemented in one place.
3. It returns a copy that holds the consolidated image and no pending
   state, or changes itself with `inplace=True`.

A transformation that has an image and nothing pending, consolidated
without options, keeps its image, so the transformation is still saved
byte for byte.

### Scope

The decision applies to the transformation formats whose file is an
image of a format that brainhops reads as an image:

| Transformation formats                                   | Stored image  |
|----------------------------------------------------------|---------------|
| `NiftiRASCoordinateField`, `NiftiRASDisplacementField`   | `NiftiImage`  |
| `ItkNiftiDisplacementField`, `ItkNiftiCoordinateField`   | `NiftiImage`  |
| the fields of NiftyReg (`NiftyRegField` and subclasses)  | `NiftiImage`  |
| `SpmCoordinateField`                                     | `NiftiImage`  |
| `FnirtWarpField`                                         | `NiftiImage`  |
| `AfniNiftiWarp`                                          | `NiftiImage`  |
| `AfniBrikWarp`                                           | `AfniImage`   |
| `MrtrixCoordinateField`, `MrtrixDisplacementField`, `MrtrixWarpFull` | `MrtrixImage` |

The other transformation formats keep plain `rawdata` and `metadata`
fields. Their files are a header without an image array, such as the
NIfTI affines in `io/transformations/nifti/_affines.py`, or a text or
structured file, such as an AFNI `.1D` matrix, MINC `.xfm`, LTA, ITK
`.txt`, FSL `.mat`, elastix, X5 and the BigDataViewer XML. An
image without voxels would serve no purpose for them, since
`NiftiImage` refuses to consolidate an image without data.

The steps of `_consolidate` that every format repeats can also be moved
into a template method of a shared base class. That change is
independent of this memo, and it would help the image formats and the
formats that keep plain fields as well.

## Points to settle

`MrtrixWarpFull` has a public field named `from_image`
(`io/transformations/mrtrix/_warps.py:1025`), which is the `-from`
option of MRtrix and also an option of its reader. A class method
of the same name cannot exist beside it, so the field must be renamed
before `from_image` is added. No release has users yet, so the rename
needs no deprecation.

`AfniNiftiWarp` is stored as a NIfTI file, but its `to_image()` returns
an `AfniImage`, which is the dataset that AFNI itself writes. Under
this memo, `to_image()` returns the stored image, a `NiftiImage`. The
conversion to an AFNI dataset needs another name, or should become a
conversion between formats.

The grid of a field and the transformations of its image are decoded
from the same header. The transformation should take its grid from the
decoder that the image uses, rather than decode the sform and the qform
again.

## Order of the work

1. Rename the `from_image` field of `MrtrixWarpFull`.
2. Move the two NIfTI fields to a stored `NiftiImage`, add `from_image`
   and `to_image`, and check that a file read and saved again is still
   copied byte for byte.
3. Move the AFNI warps, and settle the name of the conversion of
   `AfniNiftiWarp` to an AFNI dataset.
4. Move the other formats of the table, if the first two steps make the
   code shorter without awkward forwarding.
