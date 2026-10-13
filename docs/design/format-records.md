# Format records and their layering

This memo is agreed and is being implemented. It is tracked in #415,
and the work is done in the passes described at the end. The first
two passes are written and under review, and the third is in progress.
The design of pending data and `consolidate()` is applied across the
whole stack. The plain matrix formats and the NiftyReg affine are moved
on the model of LTA (items 300 to 306).

A file format in brainhops is a class that reads and writes one kind of
file, such as a NIfTI image or an LTA transformation. This memo decides
how such a class keeps what it parsed from the file. Each format class
holds a record of the file's header and a reference to the file's data,
instead of inheriting a parser that holds that state. The memo also
fixes how the record relates to the metadata of #233, and lists the
problems that are known so far.

## The problem

About 45 concrete format classes inherit a parser class that holds
state. Roughly 22 of them are transformations and 23 are images, and
they are built on 13 parser bases. Because the parser is a base class,
its fields and methods become part of the format class. They appear in
its constructor, its `repr`, its `replace` method and its attributes.

NIfTI showed the problem most clearly before the first pass, which
deleted the classes and workarounds described in this paragraph and the
next two. `NiftiReaderWriter` (`io/common/nifti/_parsers.py:46`) was a
base of `NiftiImage` and of about 16 transformation classes. It declared
`image` and `_header` as constructor fields, and it cached the voxel
array in `_data`. The name
`_data` is also where the transformation data models store their data,
for example `TransformationField._data`
(`datamodel/_transformations/concrete.py:271`) and `Affine._data`
(`concrete.py:706`). The two meanings of the same attribute collided in
#342, and three workarounds were added because of it.

1. `NiftiRASCoordinatesField` uses a custom `data` setter,
   `_store_through_the_parser`, which writes through the parser
   (`io/transformations/nifti/fields.py:65`).
2. The NIfTI affines `NiftiRASToVoxel` and `NiftiVoxelToRAS`
   (`io/transformations/nifti/affines.py:82` and `:151`) move a matrix
   given as `data=` into an `_explicit_matrix` attribute.
3. FNIRT declares a `_WritableNifti` base
   (`io/transformations/fsl/fnirt/_base.py:64`) whose only purpose is to
   keep the constructor's positional parameters in a usable order.

The order of the base classes is also constrained. `NiftiImage`
(`io/images/nifti/_image.py:37`) had to list `NiftiReaderWriter` before
`SingleScaleImage`, because the fields are collected in reverse method
resolution order and the parser's fields have defaults. LTA and elastix
still pass `reverse=False` to their class declarations for a similar
reason.

Other parsers leak their state in the same way. `MghReaderWriter` adds
four fields and eight properties. `X5TransformReaderWriter` adds
`header`, `nodes`, `chain`, `position` and an open file.
`FlirtMatrixReader` forces `Affine._data` to become a class variable
(`io/transformations/fsl/flirt/_xform.py:52`). `ZarrReader` adds `node`,
and the AFNI, MRtrix, NRRD and MINC parsers add `_header` and
`dataobj`. Parsers that only contribute methods, such as `ArrayReader`,
the ITK readers and `Hdf5Reader`, do not cause these problems.

Finally, copying between formats needs special care. When
`from_instance` builds an MGH image from a NIfTI image, the NIfTI header
must not be copied into a field of the same name that means something
else. `_foreign_format_fields` (`io/base/_base.py:567`) walks the method
resolution order to find the fields that only parsers declare, and
resets them to their defaults. This function exists only because parser
state lives on the format class.

`LtaTransformation` (`io/transformations/freesurfer/lta/_xforms.py`)
already avoids most of these problems. It holds an `LtaStruct` in a
field and delegates reading and writing to it. It is the reference for
the design below.

## The decided design

### Layers

The design separates four layers. No format class inherits a parser
that holds state.

| Layer           | Classes                                       | Role                                                  |
|-----------------|-----------------------------------------------|-------------------------------------------------------|
| Model API       | `Metadata`, `Image`, `Transformation`         | In-memory objects. `Image` and `Transformation` hold a `Metadata`. |
| Dispatchers     | `MetadataFormat`, `ImageFormat`, `TransformationFormat` | Find the file format to read or write.      |
| Format metadata | `<X>Metadata(MetadataFormat)`                 | Holds `<X>Raw`, the header record, and calls its methods for input and output. |
| Format object   | `<X>Image(ImageFormat, SingleScaleImage)`, `<X>Transformation` | Holds a `<X>Metadata` and a `raw` data object. |

A dispatcher is a class that does not read a format itself but chooses
among the formats registered under it, as `format_registry` does today
(`io/base/_base.py:31`). `ImageFormat` and `TransformationFormat`
already exist. `Metadata` and `MetadataFormat` are new, and
`MetadataFormat` is an isolated registry, so that its formats are not
offered when an image or a transformation is loaded.

The record `<X>Raw` is a small class that reads and writes the header
of one format. It knows where the data lies in the file, but it
usually does not read the data.

### Inheritance and delegation

The format classes still inherit the stateless adapter bases of
`io/base/parsers.py`: `FileReader`, `FileWriter`, `TextFileReader`,
`TextFileWriter`, `BinaryFileReader` and `BinaryFileWriter`. These
bases hold no fields. They derive every `from_*` and `to_*` method from
the few that a class implements, so that a class that implements
`from_bytes` also gets `from_file` and `load`.

Each class implements only its lowest entry points, and implements them
by calling the layer just below it. The format object calls
`<X>Metadata`, which calls `<X>Raw`. The format object adds its own code
to build the data object from the source and the record. NIfTI also
overrides `from_file` and `from_fileobj`, because nibabel needs a file
name or a file object to build its lazy proxy. These overrides must
respect the guards in `io/base/parsers.py` that stop `from_bytes` and
`from_fileobj` from calling each other forever. `from_bytes` falls back
to `from_fileobj` only when a class overrides `from_fileobj` without
marking it as a passthrough (`io/base/parsers.py:335` and `:358`), so a
class whose bytes are read through a stream must give `from_fileobj` a
real implementation.

For NIfTI, the first pass gives the classes the following bases.

| Class                      | Bases                                                     | Lowest entry points                               |
|----------------------------|-----------------------------------------------------------|---------------------------------------------------|
| `NiftiRaw`                 | `Magic`, `BinaryFileReader`, `BinaryFileWriter`           | `sniff_fileobj`, `from_fileobj` and `to_fileobj`, which read and write the header only |
| `NiftiMetadata`            | `MetadataFormat`, `BinaryFileReader`, `BinaryFileWriter`  | the same three methods, which call `NiftiRaw`     |
| `NiftiImage`               | `ImageFormat`, `SingleScaleImage`, `BinaryFileReader`, `BinaryFileWriter` | `sniff_fileobj`, `from_file`, `from_fileobj`, `to_file` and `to_bytes` |
| `NiftiBasedTransformation` | `TransformationFormat`, `BinaryFileReader`, `BinaryFileWriter` | the same as `NiftiImage`, written once for all NIfTI transformations |

### Data

On a format object, `raw` is the only attribute that stores data. It
holds the data as it is laid out in the file. After a read, `raw` is
usually a lazy proxy, and once data has been assigned it is an array.
The proxy types are nibabel's `ArrayProxy`, the existing
`DelayedH5Array` for HDF5, Zarr arrays, and a new proxy for raw binary
blocks that knows their offset and compression.

The public `data` attribute is a cached view of `raw`. Each format
defines a pair of pure functions, `to_model` and `to_disk`, with
`data = to_model(raw)` and `raw = to_disk(data)`. Reading `data` decodes
`raw` once and caches the result.

Assigned data is pending. A format takes the `data` field of its model
over under the private name `_data`, with
`_data: ... = Inherit("data", default=None)`, and its `data` property is
a `smartproperty(cache=True)`. An array stored in `_data` is returned as
it is, and `None` means that `data` is decoded from `raw`. Assigning
`data` stores the array in `_data` and clears the cache, and it never
changes `raw`. The data model of a transformation already stores its
data in `_data`, and a format whose model is a chain, such as a NIfTI
displacement field, declares a private `_data` field of its own.
`raw` and the record therefore describe the file as it was read, or as
the object was last consolidated, until `consolidate()` or a write
encodes the pending state. This replaces the rule of #302, which stored
an assigned array in `raw` at once.

The transformations of an image are compared at consolidation instead
of being pending by assignment. An image takes the `transformations`
field of its model over as `_transformations`, which holds an assigned
list only, and `None` means that no list was assigned. The property is
a `smartproperty(cache=True)`: it decodes the record on first access
and caches the list apart from `_transformations`, so a change made to
the list in place, such as an appended transformation, acts on the
cached list. The list that the image holds is the assigned list, or
else the cached list (`held_transformations` in
`io/common/_geometry.py`). `consolidate()` and every write compare that
list with a fresh decode of the record, in the data model and exactly:
each transformation must have
the same type, the same input and output spaces, and the same affine
matrix (`same_transformations` in `io/common/_geometry.py`). The
encoded bytes are never compared. When the lists are the same, or when
the image holds no list, the geometry of the record is kept as it
is stored, byte for byte, and otherwise it is encoded again from the
list. An image that was read and whose list was only read is therefore
saved byte for byte, and assigning an equal list keeps the record. The
encoders read the list without storing it, so that a write leaves the
image as it was. Assigning other metadata drops the cached list and
keeps an assigned one, as the rules below require. This applies to the
images that take `transformations` over: the toy image of the tests,
NIfTI, MGH, AFNI, MRtrix, NRRD, MINC and image stacks. MINC cannot be
written, so it keeps the list that it holds.

Within a format, `replace` and `from_instance` copy `_data` and
`_transformations` as they are stored, so a copy of an untouched object
keeps the lazy proxy and decodes nothing, and a copy of a changed
object keeps its pending state. From another class, `from_instance`
reads the decoded `data` and `transformations` of the other object,
because its `raw` and its record mean nothing to this format.

Only exact and invertible changes may happen between `raw` and `data`.
These are permuting axes, removing or adding singleton axes (such as
the time axis of a NIfTI displacement field), reshaping and changing the
memory order. Intensity scaling is not one of them, because the proxy
and the writer options already handle it.

A small amount of data may also live in the header record, for example
the matrix of an affine, or the content of a file that is cheaper to
parse in one pass. On write, the data of the model always takes
precedence over the copy in the record.

### Consolidation

`consolidate(*, inplace=False, **options)` is the only public method
that encodes a format object, and every format object has it: images,
transformations and metadata. It returns a copy whose `raw` and record
are encoded from the pending state, which is the assigned data, the
transformations that differ from those of the record and the fields of
the model, and from the writer options of the format, such as `like=`,
`dtype=` and header overrides. The options of each class are those that its writer takes.
The copy holds new metadata that holds the new record, and its `_data`
and `_transformations` are `None`. With `inplace=True`, the object
itself is changed, and it is returned. `inplace` and every option are
keyword-only.

A copy consolidated with an option that changes how the data is stored,
such as `dtype=` for NIfTI and MGH or `datatype=` for AFNI, MRtrix and
NRRD, holds what a file written with that option holds. Its `raw` holds
the values as the writer converts them, with the rounding and the
clipping of the writer, and its record holds the header that the writer
writes. NIfTI writes the encoded image to bytes and reads it again, so
that `raw` is a lazy array of the stored voxels in memory and the record
holds the scaling that nibabel chose (`_stored_parts` in
`io/common/nifti/_header.py`). MGH, AFNI and NRRD encode the voxels with
their writer and decode them with their reader. MRtrix encodes them with
its writer, and `raw` is then a lazy array of those bytes, which applies
the intensity scaling as a file is read. Data of the stored type, and a
lazy array of a file, are kept as they are, so that nothing is read.

An object that has a record and nothing pending, consolidated without
options, gives a copy that shares `raw` and the metadata, so nothing is
copied and the object is still saved byte for byte. Consolidating a
consolidated object changes nothing. Once an object is consolidated,
its geometry is decoded from the new record, so it has the precision of
the file, such as the single precision of a NIfTI sform or the
quaternion of a qform.

`MetadataFormat.consolidate(*, inplace=False)` has the same form, but
it takes no options today. It is to encode the fields of the metadata
that differ from a fresh decode of the record into a copy of the
record, which the fifth pass implements. Today metadata holds only its
record, so the method returns a copy that holds the same record, or the
metadata itself with `inplace=True`.

The methods `to_raw()` and `to_metadata()` are removed from every class,
including `MetadataFormat`. The encoding code of each format stays
explicit, in private helpers that `consolidate` calls. The reading
methods, `from_raw` and `from_any` given a record, are kept.

A write, through `save`, `to_bytes`, `to_fileobj` or `to_filename`,
consolidates the object into a copy with the writer options and writes
that copy, so the object itself is not changed. The single exception is
a save over the file that the object's own lazy array reads (#426). The
object is then consolidated in place with the same options and written,
and its `raw` then reads the new file, so that the object holds what
`load(path)` gives. For NIfTI, the record is also read again from the
new file, because nibabel sets fields such as the offset of the voxels
when it writes, so the object equals `load(path)`, record included.

### Precedence on write

Consolidation, and therefore every write, follows four steps.

1. The data to store is `to_disk(_data)` when data is pending, and
   `raw` otherwise.
2. The base record is a copy of the record of the metadata. In the
   fifth pass, only the metadata fields that differ from a fresh decode
   of the record are encoded over it.
3. The geometry is encoded from the model over that record. The image
   or the transformation owns its geometry and its data, the metadata
   owns its vocabulary, and the record supplies everything else.
4. The writer writes the header of the consolidated copy, then its
   data.

If `raw` is still the proxy that the reader built, the data is copied
from the source file without being decoded, so an object that was read
and not changed is saved byte for byte. Otherwise the array in `raw` is
written.

### Rules

A raw record is never changed in place. Encoding copies the record
first, so a record can be shared between objects safely.

A record is never made up to hold data. An object that was read from a
file has a record, and its data may live there. An object that was
built without a record holds its data in the model, and its format
writes a record only when it is saved. A record made up for an object
built from scratch would claim header fields that the object never
had.

`raw` and the record describe the file as it was read, or as the object
was last consolidated. Assigning `data`, `transformations` or another
field of the model leaves them as they are until `consolidate()` or a
write, where the transformations of an image are compared with those
of the record. There is no `pending` property. An untouched object, consolidated
without options, shares `raw` and its metadata with its copy, so that
`obj.consolidate().metadata is obj.metadata` and
`obj.consolidate().raw is obj.raw`. After `consolidate()`, nothing is
pending, and a second `consolidate()` changes nothing.

Assigning other metadata drops what an object decodes from its record
and keeps what was assigned to it. This was decided by the maintainer on
2026-10-13, for every format. A value that was never assigned therefore
follows the new metadata, whether it was read before or not, and an
assigned value is pending and is put over the new record by
`consolidate()`. The decoded value is a cached `smartproperty`, apart
from the private field that holds an assigned value, and `metadata` is a
`smartproperty` that names it in `invalidates=`, which deletes the cache
and never the assigned value. This holds for the transformations of
images and of chains, for the matrix of the NIfTI and LTA affines, held
as `data`, and for the views derived from it. The ITK endpoints, which
follow from the dimension of the header unless a chain is assigned, and
the X5 nodes, which index the old record, are dropped in the same way.
Conformance check 20 tests the rule. A change made in place to a decoded
list of transformations lives in the cache, so it is lost when other
metadata is assigned, and a copy made with `replace` or `from_instance`
decodes the list again instead of copying the cache.

For chain formats (X5, M3Z), assigned `input` and `output` systems count
as overrides. `consolidate()` encodes them, and it checks that the file
can express them, raising `UnrepresentableTransformationError`
otherwise.

Each class states its rule for overrides in `consolidate()` explicitly,
and a test probes it by assigning each public attribute and checking
whether the consolidated copy still holds the same metadata. An
attribute that is neither honored on write nor refused is the bug that
recurred in the LTA data, the X5 systems and the M3Z systems. Where
nothing was assigned, a class keeps the held record after a cheap
check, and does not encode it again.

The views that a class decodes from the record are read-only by
construction, and do not rely on the flags of the arrays in the record.

Each image class gives an explicit `shape` read from the pending data,
or else from `raw`, without reading the voxels; `ndim` follows from it.

A lazy array answers an empty selection without reading the file, so
that Dask's metadata probe reads nothing.

Header fields that the data or `dtype=` decide (dimensions, voxel type)
cannot be overridden; the writer refuses them with `WriterError`.

A save never removes or overwrites a file that a live proxy reads. When
it replaces that file, the object is consolidated in place with the
writer options, and its `raw` reads the new file (#426).

A conformance check that is vacuous for a format is logged in the comment
of the exemplar of that format and in the list of tensions, instead of
getting a generic hook.

`<X>Metadata` holds no file handle. The format object builds the data
proxy from the source and from the record, which says where the data
is. Metadata therefore stays cheap to copy and can be pickled, which
matters because operations copy it from their input to their output.

A format that needs almost nothing gets a degenerate stack. Its
metadata is `OpaqueMetadata`, which has no record, or its record is
just the parsed array, as for ITK text transforms and plain matrix
files.

Historical names are not kept as aliases. The attributes `header`,
`struct`, `keyval` and `image`, and the parser mixins, are removed.
This breaks the public API, which is accepted.

### Pyramids

A pyramid, or multiscale image, stores the same image at several
resolutions, which are its levels. Every container of a pyramid that
brainhops reads stores one document that describes the whole pyramid and
each of its levels. This document is the group document of an OME-Zarr
store, the IFDs and the OME-XML of a TIFF file, the properties of an
OpenSlide slide, the boxes of a JPEG 2000 codestream, the `info` file of
a precomputed volume, the attributes of an N5 group, the `DataSetInfo`
groups of an Imaris file, or the XML file of a BigDataViewer dataset.
OME-Zarr also gives each level an array document of its own, inside the
same store. No container stores a level that can be read without the
document of its pyramid.

A pyramid format therefore has one record, `<X>Raw`, for its whole
container. The record holds the metadata of the pyramid and a tuple
`levels`, which holds the storage description of each level, such as its
shape, its data type and its chunks, but never its data. One
`<X>Metadata` holds the record, and the pyramid and each of its levels
hold the same metadata object, by identity. A level is an object of the
single-scale format of the same package, such as `OmeZarrLevel` or
`TiffImage`. It holds the lazy array of its level in `raw`, the shared
metadata, and the index `level` of its level in the record. The level
classes are public, and each one reads a single level of a container
with the reader option `level`. The fourth pass first planned one record
per level, and this design replaces that plan.

The pyramid holds the tuple of the arrays of its levels in `raw`, finest
first, so that `raw` stays the only attribute that stores data and an
untouched copy shares it. Unlike a list, a tuple cannot be changed in
place. The pyramid takes the `images` and `transformations` fields of
`MultiScaleImage` over as `_images` and `_transformations`, which hold
assigned lists only. Its `images` property is a `smartproperty` with a
cache, which decodes one level object for each array of `raw`, all
sharing the metadata, and caches the list. Its `transformations`
property decodes the placements of the pyramid, which map the space of
its levels to the world, in the same way, and is empty for a container
that stores no placement. The `metadata` property names both of them in
`invalidates=`, so other metadata drops the cached levels and placements
and keeps the assigned ones, as the rules above require. The level
objects that are dropped keep the old metadata. Within the format,
`replace` and `from_instance` copy `raw`, the metadata and the assigned
lists, but not the cache, so a copy of an untouched pyramid decodes
nothing. From another class, `from_instance` reads the `images` and the
`transformations` of the other object. Each pyramid gives an explicit
`shape`, read from the levels that it holds or else from the first array
of `raw`, without reading the voxels. `MultiScaleImage.data` stays a
read-only property that returns the data of the finest level.

The helpers of `io/common/_pyramids.py` tell which levels of a pyramid
are still the levels of its file. `held_images` returns the assigned
list of levels, or else the cached one, or `None` when the levels were
neither assigned nor read. `own_level` tells whether a held level is an
object of the level class that holds the metadata of the pyramid at the
same index, whose `raw` is the array of that index in the `raw` of the
pyramid, with no pending data, and with transformations that were never
read or are the same as a fresh decode. `own_levels` applies that test
to every held level. Each format calls them explicitly, with its level
class and its function that decodes the transformations of a level.

A pyramid that has a record, whose levels are those of its file, whose
placements were never read or are the same as a fresh decode, and which
is consolidated without options, gives a copy that shares `raw` and the
metadata. Otherwise each level gives a stored array and a storage
description. A level of the file gives them as they are stored. A level
with pending data gives `to_disk` of that data, and a level of another
class, or one that holds other metadata, gives `to_disk` of its decoded
data. The transformations of a level are compared with a fresh decode
and encoded only when they differ, and so are the placements. The new
record is a copy of the held record with new levels, and the
consolidated copy holds the new arrays in `raw`, new metadata, and no
assigned list, so its levels are decoded from the new record.

Zarr and TIFF write every level as it is given, and no ratio between the
shapes of the levels is checked, for any format. A format whose writer
derives the coarser levels from the finest one, such as JPEG 2000,
writes from the finest level only, and assumes that the coarser levels
that the user assigned are downsampled versions of it. It never refuses
them. A format that cannot write, such as OpenSlide, consolidates only
its data, as MINC does (item 101). The pending data of each level then
moves into the tuple `raw`, the metadata and the transformations are
kept, and a level of another class is refused with `WriterError`,
because no record can describe it. The maintainer decided these rules on
2026-10-13.

OME-NGFF 0.6 lets a dataset of a multiscale hold only a scale, an
identity, or a scale followed by a translation. The OME-Zarr writer
encodes a level transformation that is an axis-aligned scale with an
optional translation, and refuses any other with
`UnrepresentableTransformationError`, whose message says to move the
rotation or the shear into the placements of the pyramid, which version
0.6 stores as common transformations. Factoring a level transformation
into a common part and a part for each level is tracked in #447. A
container that stores no placement, such as TIFF or JPEG 2000, composes
an assigned preferred placement into the transformation of each level
that it writes, and its class docstring says so.

The reader options `level`, `series`, `setup`, `timepoint` and `frame`
select what is read. They are kept as fields and never change the record
(item 49). Options that describe the geometry, such as `pixel_size=`,
`unit=` and `origin=`, become assigned transformations, so a file read
with them is written with that geometry, and not byte for byte.

A level is a single-scale image, so a level that is consolidated or
written alone narrows the record to its own level, at index 0, and keeps
the rest of the record. `TiffImage.load(path, level=1).save(other)`
therefore writes a file of one level, and the pyramid class is the way
to keep the other levels.

An untouched pyramid in a local directory is saved as a copy, file for
file, of its directory. A pyramid in which one level changed writes
again only the files of that level and the documents that changed, so
every other file keeps its bytes. A record keeps the bytes of each
document as read, and the writer writes them again when the encoded
document equals the parsed bytes, under a comparison that tells `1` from
`1.0`. A local directory is written with `replace_directory`
(`io/base/_utils_files.py`), which fills a new directory beside the
target and then moves it into place, so a failed write leaves the old
directory as it was. A save over the directory that the proxies of the
object read (#426) consolidates the object in place, copies the files of
the untouched levels from the old directory into the new one, moves the
new directory into place, and reads the proxies and the record again
from it. A remote store cannot be renamed, so it is written in place,
level by level, and the save is not atomic. A remote save over the store
that the object reads is refused with `WriterError`, so that the proxies
never read a store that is half written. New Zarr stores are written in
version 3, and a store that was read keeps its version.

The maintainer decided this design on 2026-10-13. Pass 4a implements
the helpers, `replace_directory`, the toy pyramid of the conformance
tests and the checks P1 to P12. Zarr and OME-Zarr, TIFF, OpenSlide and
Pillow move to the design in the later steps of the fourth pass, which
also move the format fields of TIFF and Pillow into their records and
make `OmeZarrField` read its levels lazily. The pyramids of JPEG 2000
(#273), precomputed (#436), Imaris (#437), N5 (#438) and BigDataViewer
(#441) were merged before pass 4a. They stay among the formats that are
not migrated, which the checks P1 to P12 never run on, and move to this
design after pass 4a.

### Public surface

Each format package is split into private submodules, such as
`_raw.py`, `_metadata.py`, `_image.py`, `_affines.py`, `_fields.py` and
`_views.py`. The package `__init__` exports only the public names. The
`<X>Raw` and `<X>Metadata` classes are public. The proxy types are
private, except those that are already public, such as
`DelayedH5Array`.

`Metadata` is exported by `brainhops.datamodel`, and `MetadataFormat` by
the new package `brainhops.io.metadata`. After the first pass,
`brainhops.io.common.nifti` exports `NiftiRaw`, `NiftiMetadata` and
`NiftiUnitWarning`. The NIfTI image and transformation packages export
their format classes and also export `NiftiRaw` and `NiftiMetadata`
again, so that a user finds them next to the classes that use them.
The modules of the NIfTI transformation package are private, and the
SPM package exports `SpmCoordinatesField` from its private `_fields`
module instead of the public module `spm.y`.

## Relation to #233 and the #306 memo

The metadata design of #233 is written up in the memo of #306 and
partly implemented in #310 and #287. That memo gives each file-based
metadata object a private `raw` record. The record is decoded into
common fields on read and encoded back on write, and a stored snapshot
of the decoded fields tells which ones were changed.

This memo replaces three parts of that design. The stored snapshot is
replaced by the comparison with a fresh decode of the record, described
in the first step of the write precedence above. Formats no longer
inherit their parser, which also removes the base-order problem the
memo describes, where NIfTI transformations had to declare their
`metadata` field again because their first base carried the generic
one. The historical accessors that the memo planned to keep, such as
`header`, `keyval` and `struct`, are removed instead.

This memo keeps the name `raw` and the `<X>Raw` types that the memo
chose in its section 6.1. `LtaStruct` therefore becomes `LtaRaw`, and
`from_struct` becomes `from_raw`, while `to_struct` is removed in favor
of `consolidate()` (see Consolidation). It also keeps the vocabulary of
common metadata fields and the `to` conversion.

The vocabulary itself arrives in the fifth pass, after the formats have
moved to records. That pass takes the vocabulary from #310 and #287 and
adds it to `Metadata`. Each format gets module-level functions that
decode and encode the vocabulary, and `consolidate()` encodes only the
fields that differ from a fresh decode. The same pass adds the BIDS
codec, puts metadata on the roots of the data model, propagates metadata
through operations, and updates section 6.1 of the memo.

## Implementation passes

The work is done in five passes. Each pass is reviewed before the next
one starts.

The first pass adds the core classes and moves NIfTI. It is made of
three stacked pull requests. The first adds `Metadata` and the
`MetadataFormat` dispatcher, whose `from_raw` builds an object from a
record and whose `consolidate` returns a copy of the metadata. It also
adds the mechanism of pending data and of `consolidate()`, and a
conformance test, which checks every migrated format against the rules
of this memo and lists the formats that are not migrated yet. The
second adds `NiftiRaw`, `NiftiMetadata` and the views of NIfTI data,
and moves `NiftiImage` to them. The third moves the NIfTI
transformations, including the SPM, NiftyReg, ITK and FNIRT fields, and
deletes `NiftiReaderWriter`, `_explicit_matrix`,
`_store_through_the_parser` and `_WritableNifti`. It also renames the
`header=` option of the NIfTI converters to `metadata=`.

The conformance test runs the following checks on every migrated
format.

1. The constructor fields of the format are the fields of its model,
   `raw`, `metadata` and the options the format declares. None of them
   is named `image`, `header`, `struct`, `_header`, `dataobj` or
   `node`.
2. Loading is lazy. After a load, `raw` is a proxy and no decoded data
   is cached yet.
3. Setting `data` leaves the stored data and the record as they are,
   clears the cached data and the views derived from it, and is
   pending. `consolidate()` stores `to_disk(value)` in a copy.
4. `to_model(to_disk(x))` equals `x`.
5. An object that is loaded and saved without changes gives the same
   bytes, after decompression.
6. `<X>Metadata.load` never reads the data, which is tested on a file
   truncated after its header.
7. The metadata can be pickled.
8. An object built from `data=` alone holds its data pending, and it
   can be consolidated and saved.
9. `replace` and `from_instance` keep `raw`, `metadata` and the pending
   fields within a format, so a copy of an untouched object stays lazy,
   and they reset `raw` and `metadata` when copying from another format.
10. An edit of `metadata.raw` survives a save, and a changed geometry
    takes precedence over the stale header.
11. Each format package exports exactly its intended public names, no
    name that starts with an underscore, and only private submodules.
12. The adapter contract holds. A format whose bytes are read through a
    stream really overrides `from_fileobj`, a binary format reads in
    `"rb"` mode, and a concrete format is not a dispatcher.
13. The transformations of an image are decoded from the record on first
    access and stored, unless they are assigned, and copies within the
    format keep them.
14. Copies between classes carry the decoded transformations.
15. Consolidating an untouched object gives a copy that shares `raw`
    and the metadata, and decodes nothing.
16. Consolidating is idempotent, nothing is pending afterwards, and the
    copy holds exactly what a file written from the object holds. With
    an option that changes how the data is stored, given by
    `Exemplar.storage`, the copy holds the data, their type and the
    header of the file exactly. A format that cannot write consolidates
    only its data.
17. A save writes the consolidated copy and leaves the object as it was.
18. A save over the file that the `raw` of the object reads consolidates
    the object in place, and the object then holds what `load(path)`
    gives (#426), its header included.
19. The transformations of an image are compared at consolidation. A
    list that was only read keeps the record and the bytes, an equal
    list that is assigned keeps the record, a list changed in place is
    written, and a copy of a read image stays lazy.
20. Assigning other metadata keeps what was assigned and drops what was
    decoded, whether it was read before or not. Transformations, or the
    data of a format whose data is its matrix, that were not assigned
    follow the new metadata, as in an object built with it, and a saved
    file holds them with an edit of the new record. Those that were
    assigned survive the new metadata, and a consolidated copy and a
    saved file hold them with the edit of its record.

The checks above run on the level class of each pyramid format, which
reads one level with the option `level`. A level class whose container is
a directory has no bytes, so its metadata is compared through a summary,
and its saves are compared file for file. Pass 4a adds the following
checks, P1 to P12, which run on each pyramid format and on the toy
pyramid of the test. A format that cannot write consolidates only its
data in P4, refuses a level of another class in P5, and refuses to write
in the other checks that write, while P7 and P9 do not apply to it.

- P1. The constructor fields of the pyramid are `images`,
  `transformations`, `raw`, `metadata` and the options that it declares,
  and none of them is named `node`, `_slide` or `_source`. `images` and
  `transformations` are cached properties, and `metadata` invalidates
  both.
- P2. Loading is lazy. After a load, `raw` is a tuple of proxies of the
  level class, and the levels are not decoded. Reading `images` caches a
  list of objects of the level class, each with its index in `level`, its
  array of `raw` and the metadata of the pyramid, and no data is decoded.
- P3. A level of a loaded pyramid holds the data, the geometry and the
  record of the same level read alone with the option `level`.
- P4. Data assigned to a level is pending. The arrays and the record of
  the pyramid do not change, and `consolidate()` gives a copy whose array
  of that level is `to_disk` of the data, whose other arrays are the same
  objects, and in which nothing is pending, while the pyramid keeps its
  pending state.
- P5. An assigned list of levels, in which one level is an image of
  another class, is pending and is written.
- P6. Consolidating an untouched pyramid, or one whose levels and
  placements were only read, gives a copy that shares `raw` and the
  metadata and decodes nothing, and consolidating is idempotent.
- P7. The transformations of a level, and the placements of the pyramid,
  are compared at consolidation. A list that was only read keeps the
  record, and so does an equal list that is assigned, while a changed
  list is written. A container that stores no placement writes the same
  world geometry into its levels.
- P8. An untouched pyramid is saved identically, file for file. In a
  directory, the files that change when one level changes never change
  when another level changes.
- P9. A save over the container that the proxies read consolidates the
  pyramid in place (#426). Its `raw` is then a new tuple of proxies of the
  new container, and its record is that of a fresh load.
- P10. `replace` and `from_instance` within the format share `raw` and
  the metadata and decode nothing, and `from_instance` from another class
  carries the levels of that object, which are written.
- P11. Assigning other metadata drops the cached levels and placements
  and keeps the assigned ones, as check 20 tests for single images.
- P12. The metadata of a container reads no data, which is tested on a
  copy of the container that holds no data.

The second pass moves LTA first, so that the reference format matches
this design. `LtaStruct` becomes `LtaRaw`, its field becomes a
keyword-only `raw`, and `reverse=False` is dropped. The pass then moves
X5, whose record holds the header and the nodes while `chain` and
`position` stay options of the format, and M3Z, whose arrays stay in
the record because the file is parsed in one pass.

The move of LTA, called pass 2a, is pull request #421. The move of X5,
called pass 2b, is pull request #422. The move of M3Z, called pass 2c,
is under review. ITK-H5 was deferred, because it shares `ItkStruct` with
the ITK text and MATLAB formats and has no writer, and it moves with them
in the fourth pass.

The third pass moves MGH, AFNI, MRtrix, NRRD and MINC. MGH gets a
record with its header and its tags, and uses nibabel's proxy. For the
others, the private `_header` becomes the record, `dataobj` becomes
`raw`, and the existing transpositions become `to_model` and `to_disk`.
The transpositions that depend on the record live in the lazy array
instead, and `to_model` and `to_disk` are the identity for them. The
intensity scaling of MRtrix stays out of the view. For these image
formats, the pass also decides which header fields the record keeps
and which ones the geometry encoding replaces, and whether `replace` and
`from_instance` keep the proxy. Its tests compare compressed variants of
a file after decompression.

The move of MGH, called pass 3a, is pull request #424. The move of AFNI
and MRtrix, called pass 3b, is pull request #425. The move of NRRD and
MINC, called pass 3c, is in progress.

The elastix parameter files move in the same way as M3Z, since a
parameter file is parsed in one pass. Their record also holds the records
of the files of their chain of initial transforms, as items 320 to 329
describe.

The fourth pass moves FLIRT and the formats of pyramids. A pyramid has
one record for its whole container, which the pyramid and each of its
levels hold, and in which a level is found by its index `level`, as the
section on pyramids describes. Pass 4a adds the helpers, the extensions
of the conformance test and the toy pyramid. The later steps move
Pillow, then Zarr and OME-Zarr, TIFF and OpenSlide, and then update the
open pull requests of pyramid formats. The class variable workaround of
FLIRT is removed once it is decided whether `moving` and `reference`
belong to the format or to the record.

The move of FLIRT and of the three ITK formats that store parameters, the
text `.tfm`, MATLAB `.mat` and HDF5 `.h5` files, is in progress. The
images `moving` and `reference` of FLIRT belong to the format: they are
options of the transformation that are never stored, as the geometries of
a `register.dat` file are. The record of FLIRT holds its matrix
(`FlirtRaw`), and the record of each ITK format holds the blocks of its
file in the order of the file (`TfmRaw`, `MatRaw` and `H5Raw`), with the
root strings of an HDF5 file and the bytes of a MATLAB file. Only the
MATLAB format writes. The formats that do not write encode a pending
chain or matrix into a new record in memory (item 311).

The fifth pass brings the #233 vocabulary, as described in the previous
section.

## Open tensions and workarounds

The work has found the following tensions and workarounds so far.
This list is updated as the work finds new ones, and each entry records
its status.

1. The constructor assigns fields in their declared order, so the
   default `raw=None` overwrites a value given as `data=`. A
   `__post_init__` applies `data=` again, and for the NIfTI
   transformations it is written once, in `NiftiBasedTransformation`
   (`io/transformations/nifti/_base.py:84`). When both `raw=` and
   `data=` are given, `data=` wins. Status: resolved by pending data.
   The constructor stores `data=` in `_data`, which the default of
   `raw` no longer overwrites, so `store_data_argument` and the
   `__post_init__` methods that only called it are deleted. When both
   `raw=` and `data=` are given, `data` is read first, and `raw` is kept
   until the object is consolidated.
2. A `smartproperty` whose setter was added later with `@data.setter`
   lost its `invalidates=` option, because the invalidator was wrapped
   around the setter only when the property was built. Status: resolved
   in pass 1a, where `smartproperty` returns a subclass of `property`
   whose `setter` keeps the invalidator.
3. nibabel applies `scl_slope` when it writes, so copying an untouched
   file byte for byte requires writing the unscaled data of the proxy
   with the slope and intercept of the record. Status: open.
4. Using the record as the base header means choosing which header
   fields are reset and which are kept. Fields such as `cal_*`,
   `dim_info`, `slice_*` and the extensions now survive a save, but
   `dim_info` and `slice_*` become stale when the writer transposes the
   axes. The intent of the record now also takes precedence over
   `like=`, because `_apply_like` copies the intent only when it is 0.
   Status: open.
5. A NIfTI affine has two sources, a matrix in `raw` or the `sform` of
   the record, and the inverse of an affine that comes from the header
   must also come from the header. Status: decided in pass 1c, where a
   matrix in `raw` takes precedence over the `sform`, and the inverse of
   an affine read from the header still reads the header.
6. Reading `data` turns the proxy into an array with the array backend,
   which copies with NumPy but stays lazy with Dask. An edit made in
   place on the decoded array would then be lost, because an untouched
   `raw` is written from the proxy. Status: decided in pass 1b, where
   the proxy is always written and the decoded NumPy arrays are
   read-only, so that values are changed by assigning `data`.
7. In the first pass, `metadata` is a field of the format classes,
   while #287 puts it on the roots of the data model. The name also
   clashes with `Transformation.metadata_fields`, which may be renamed.
   Status: open.
8. Views that are decoded from `metadata` and cached, such as
   `transformations`, would become stale when `metadata` is assigned
   again. Each format holds a private `_metadata` field behind a public
   `metadata` `smartproperty` with `invalidates=` naming those views,
   because bagof does not allow a field and a property with the same
   name in one class. Status: resolved in pass 1c for the NIfTI
   transformations, and for every format by the maintainer decision of
   2026-10-13, described in the rules, which also covers the images.
   Their decoded list now lives in the cache of `transformations`, apart
   from an assigned list, so the invalidation drops only the decoded
   one.
9. A proxy built over a stream that the caller passed in must keep that
   stream alive for as long as the proxy is used. For the same reason,
   a format whose `raw` is a proxy must override `from_filename` or
   `from_file`, because the adapter's `from_filename` closes the stream
   when it returns. X5 has no name to reopen when its file is read from
   a stream or from bytes, so it reads all its fields at once in that
   case, while NIfTI keeps the stream of the caller. Status: the first
   part is a rule for every pass, and the second part is accepted in
   the pass 2b review (146c69b), because there is no name to reopen.
10. `_holds` (`io/base/_save.py:220`) refuses a format that does not
    take every constructor field of its model. A format must therefore
    keep `data` in its constructor even though `data` becomes a view,
    and `MultiScaleImage.data` is a read-only property for which a rule
    is set in the fourth pass. Status: open for `data`, and decided for
    pyramids in pass 4a. `MultiScaleImage.data` stays a read-only property
    that returns the data of the finest level, and `_holds` is unaffected,
    because a pyramid format takes `images` and `transformations` over
    under the same public names.
11. Reader options fall into two groups. Some describe the object, such
    as `moving`, `reference`, `log` and `steps`, and others describe
    the reading, such as `mmap` and `keep_file_open`. Until the two
    groups are split, the NIfTI transformations remove the first group
    from the options before they call nibabel, and set those values on
    the object after it is read. Status: open.
12. The NiftyReg writer replaces the extensions of the record instead
    of appending to them (`io/transformations/niftyreg/_fields.py:473`),
    so that a field that is saved again does not accumulate copies of
    its extension affines. Status: decided in pass 1c.
13. Several public names and behaviors break and must be announced.
    The names are `NiftiReaderWriter`, which pass 1c removes,
    `sniff_nibabel`, the `image` and `header` attributes and arguments,
    the `header=` argument of `from_nibabel`, the `header=` option of
    the NIfTI converters, which becomes `metadata=`, `LtaStruct`, the
    modules `nifti.affines`, `nifti.base`, `nifti.fields` and `spm.y`,
    which become private, and the positional parameters recorded in the
    constructor signature test. The behaviors are `NiftiImage.system`,
    which becomes read-only, the `data` of a NiftyReg field, whose shape
    loses its singleton axis and becomes `(X, Y, Z, 3)`, and a
    displacement field loaded with `log=True`, which is written back as
    a velocity unless it is saved with `log=False`. Pass 2b adds the X5
    changes. The fields of an X5 file are read lazily by default
    (`load=False`), `keep_open=` raises `TypeError`, `chain` and
    `position` become keyword-only, the `header`, `nodes` and `file`
    attributes are removed in favor of `metadata.raw`,
    `X5TransformReaderWriter` is removed, and `to_struct` is removed.
    The design of pending data adds that `to_raw()` and `to_metadata()`
    are removed from every class in favor of `consolidate()`, that
    assigned data no longer changes `raw`, that a write no longer
    changes the object except over its own file, and that an untouched
    object is written with its record as it is. Status: open.
14. The design needs `bagof-magic` 0.3.dev2 or later, which is not on
    PyPI yet, so the checks before each pull request need it installed
    from another source. Status: open.
15. The levels of a `MultiScaleImage` and the metadata of a pyramid
    had no record. Status: decided by the maintainer on 2026-10-13, as
    the section on pyramids describes. A pyramid has one record for its
    container, which each of its levels holds. Pass 4a implements it for
    the toy pyramid of the conformance test, and the formats move to it in
    the later steps of the fourth pass.
16. `_foreign_format_fields` counted the `MetadataFormat` dispatcher as
    a format that owns `raw`, so `raw` was reset when it should have
    been copied. Status: resolved in pass 1a, where dispatchers are no
    longer counted as owners.
17. Copying an object within its format with `from_instance`, as
    `io.save` does, read `data` and so turned the proxy into an array.
    Status: resolved by pending data. Within a format, `from_instance`
    copies `_data` as it is stored, so copies stay lazy and an untouched
    object is still saved byte for byte, and the special case of
    `_FileBasedModelMixin` that passed `data=None` is deleted. From
    another class, it reads the decoded `data` of a format that takes
    `data` over. The data model of a transformation copies the decoded
    view of another instance (item 35), so `LtaTransformation` and the
    NIfTI affines and coordinates field pass their stored `_data` within
    the format, in explicit code in each class.
18. `replace` reads `data` through the property, so a copy made with
    `replace` turns the proxy into an array and is no longer saved byte
    for byte. The setters of the NIfTI transformations empty the
    private `_data` of the model
    (`io/transformations/nifti/_affines.py:55` and `_fields.py:87`), so
    that `replace` passes `data=None` and the copy keeps `raw`. Status:
    resolved by pending data for every format, since `replace` copies
    `_data`, which is `None` for an untouched object, and the setters
    that emptied `_data` are deleted.
19. The `repr` of an image decoded the whole proxy to show `data`.
    Status: resolved in pass 1a, where `SingleScaleImage.data` is
    hidden from `repr`, which changes the public `repr`.
20. A cached `smartproperty` that is set through a hand-written setter
    keeps its own cache, unless the setter deletes it or names it in
    `invalidates`. Doing this automatically would break the setters
    that fill the cache on purpose (`concrete.py:396-421`), so every
    NIfTI setter declares `unset=_always` and removes `_cache_data` by
    hand, since `_FORGET_VIEWS` clears only the attributes listed in
    `derived_fields`. Status: open.
21. The test that compares the registry with the table of formats sees
    only registered classes, so a concrete format that is not
    registered escapes it. `NiftiRASToVoxel` is not registered, so a
    separate test checks that it is saved byte for byte. Status: open,
    and partly addressed in pass 1c.
22. `MetadataFormat` also inherits `_FileBasedModelMixin`, which is
    needed for `from_any(file)` and for the reset of fields when
    copying from another format. The base list of the plan did not
    include it. Status: open.
23. The conformance check that reads only the header assumes that the
    header is a prefix of the file. Formats with a separate header
    file, such as detached NRRD and MINC, need a hook in their exemplar
    in the third pass. HDF5 has no header before its data, so X5 needed
    `cut` and `header` hooks for this check in pass 2b. Status: accepted
    in pass 2b for HDF5 (146c69b), where no check is weakened, because a
    file that the `cut` hook truncates raises `OSError` on any read of a
    field, and open for NRRD and MINC.
24. An untouched save is identical byte for byte only when the geometry
    is encoded back exactly. A `qform` or `sform` whose code is 0 used
    to be written back with code 2, and the padding before `vox_offset`
    was lost. Status: decided in pass 1c for the fields, which treat the
    geometry as unchanged while `_transformations` is unset and then
    write `raw` under the record with its forms kept. Resolved by
    consolidation for every NIfTI format, since an untouched object is
    written with its record as it is. A geometry that is assigned, or a
    writer option, encodes the forms again.
25. Encoding a model in millimeters over a record whose spatial unit
    is unknown could replace that unit. Status: decided in pass 1b,
    where the unknown unit of the record is kept.
26. The fields of a nibabel header are views into the header, so the
    stored scaling changed when the structural fields were reset.
    Status: resolved in pass 1b, where the scaling is read with
    `.item()` before the reset.
27. `_new_nifti` builds the nibabel image twice, so that nibabel
    validates the data type and applies its policy on 64-bit integers.
    The array is not copied. Status: accepted.
28. Code in the FSL formats and in `_like_header` reached the record
    through attributes, as in `metadata.raw.header`. Status: resolved
    in pass 1c, where the helper `_header(obj)`
    (`io/common/nifti/_header.py:174`) replaces these lookups in the
    FSL, ITK, NiftyReg and FNIRT formats. The helper still falls back on
    the `header` attribute for nibabel images and MGH images, which is
    accepted until the third pass.
29. The base adapter has no fallback from `sniff_bytes` to
    `sniff_fileobj`, so each class writes `sniff_bytes` once. Status:
    accepted.
30. `NiftiMetadata.save("x.nii.gz")` wrote an uncompressed file.
    Status: resolved in pass 1b, where `NiftiRaw.to_filename`
    compresses the file when its name ends in `.gz`. Whether
    `FileWriter.to_filename` should do the same for every format is an
    open question.
31. `NiftiImage.transformations` is returned as the same object each
    time, so appending to the list in place persists. It used to return
    a copy. Status: resolved. The list is stored in `_transformations`
    and compared at consolidation with the list that the record decodes
    to, so a transformation appended in place is written, for every
    image format.
32. `read_nifti` parses the header twice when it reads a local path,
    once for the record and once for the nibabel image. Status:
    accepted.
33. `_reset_structural_fields` also resets `xyzt_units`, so a field
    written from a new chain of transformations declares unknown units,
    as it did before pass 1c when the header was built from scratch.
    Status: accepted.
34. The NiftyReg and ITK fields check the shape and the layout of their
    header in `__post_init__`, so a wrong header is refused when the
    object is built instead of in `from_nibabel`. Status: decided in
    pass 1c.
35. `_copy_map` reads `other.data` only when no data is given
    (`datamodel/_transformations/concrete.py:1217-1234`), and the
    laziness of `from_instance` depends on that. Status: accepted.
36. `FnirtWarpField.transformations` keeps a hand-written cache,
    `_chain_cache`, that it looks up by the identity of the objects
    (`io/transformations/fsl/fnirt/_base.py:408-443`), and the bases of
    the class list the model last. X5 follows the same pattern in pass
    2b. It keeps a cache from nodes to transformations by hand in
    `__dict__["_decoded_nodes"]` (`io/transformations/x5/_xform.py`),
    because the writer recognizes the decoded nodes by their identity
    after the `transformations` setter has dropped
    `_cache_transformations`. An element is written as its node only if
    this transformation decoded it from its own record. An element taken
    from another object, or a copy made with `replace`, is encoded again
    and loses the extra content of the node, which is its metadata, its
    inverse and its Jacobian. A cleaner design would need an
    `X5Affine(Affine)` subclass that carries its node, and that is left
    to the maintainer. Status: accepted for X5 in the pass 2b review
    (146c69b), which follows FNIRT, and open for FNIRT, whose cache is
    older than pass 1c. The key of the FNIRT cache now includes the
    pending `_data`.
37. In the conformance tests, check 9 does not convert NiftyReg, ITK
    and FNIRT fields to another format, because they have no converter
    until #312, and it compares `raw` and `data` only for models that
    have `data`. Status: accepted.
38. Private helpers such as `_always`, `_record_image`,
    `_set_field_data` and `_set_coordinates_intent` were imported by
    other packages from `io/transformations/nifti`. Status: resolved in
    pass 1c, where they move to `io/common/nifti/_header.py` and
    `_core/properties.py`.
39. Setting the data of a fresh LTA transformation used to build a
    record for it, and that record broke the untouched VOX and PHYS
    files that have no geometry blocks. Status: resolved in pass 2a.
    Without a record, the model holds its matrix, and `consolidate`
    keeps the record only when `_data`, `_input` and `_output` are all
    unset. The rules above now forbid making up a record to hold data.
40. Two contracts existed for `to_raw`. A format object returned the
    record that it held, and `MetadataFormat.to_raw()` returned a copy.
    Status: resolved, since `to_raw` is removed and `consolidate()` is
    the single public encoding method, with one contract for every
    class.
41. Setting `data` on an LTA transformation that has a record replaced
    its metadata object, so that
    `LtaTransformation(metadata=md, matrix=m).metadata is not md`.
    Status: resolved by pending data. The matrix is pending in `_data`
    and the metadata is kept, and `consolidate()` replaces the metadata
    with new metadata that holds a new record, as for every format.
42. Without a record, the views of an LTA transformation read an
    anonymous volume block that was built again on each access. Status:
    resolved in pass 2a.
43. Copying one LTA view into another shares the record and reads it
    again in the coordinate systems of the target, because the views
    are views of one file. Status: accepted and documented.
44. The LTA writer keeps only 16 significant digits. The problem is
    older than this work and is filed as #420. Status: open.
45. The conformance tests needed new hooks. `Exemplar.stored` reads data
    that lives in the record. The stream contract of check 12 applies
    to binary formats only, and a text format is checked for the read
    mode `"rt"`. Formats that hold a chain of transformations needed
    `view`, `data` and `set_data` hooks. Status: accepted in passes 2a
    and 2b.
46. The X5 record holds the lazy `DelayedH5Array` proxies, so the data
    lives in the metadata rather than on the format object
    (`io/transformations/x5/_raw.py`). The nodes and their arrays form
    one record, and HDF5 has no split between a header and its data. A
    proxy holds only a file name, so the metadata can still be pickled,
    but copied metadata refers to the source file by its name. Status:
    accepted in the pass 2b review (146c69b).
47. The shared HDF5 adapter still read the `keep_open` option
    (`io/common/hdf5/_parsers.py`), because ITK-H5 was not migrated yet.
    The `TypeError` that X5 raises for `keep_open=` comes from its
    constructor. Status: resolved by the move of ITK-H5, where the
    adapter always closes the file that it opens, and ITK-H5 raises the
    same `TypeError`.
48. An untouched HDF5 file is saved byte for byte only when h5py wrote
    it with the attribute and creation order of brainhops and with
    contiguous datasets, which includes a fixture in the layout of
    nitransforms. Chunked or compressed datasets and other attribute
    orders are written back as contiguous datasets with the same
    decoded content. Files written by fslpy 0.1.0 are rewritten in the
    current layout. h5py writes different bytes to a `BytesIO` than to
    a file, so the checks compare files with files and buffers with
    buffers. A file that was opened with h5py in `r+` mode is rewritten
    differently, while a contiguous file that h5py wrote is identical.
    Status: accepted in the pass 2b review (146c69b).
49. The X5 options `chain` and `position` select what is decoded, but
    they do not count as overrides, so a file read with `chain=1` is
    written back whole. An assigned `input` or `output` does count.
    Read options never change the record, as for the `log=` option of
    NIfTI, and writing only the selection would silently drop nodes.
    The idiom to write only the selected chain is
    `xform.transformations = list(xform)`. Status: decided in the pass
    2b review (146c69b), where options do not count as overrides.
50. `_read_only_field` (`io/transformations/x5/_blocks.py:362`) makes
    read-only the vectors in voxel units that are computed afresh and
    are not a view of the record. An edit made in place would still be
    lost, because the writer writes the node that it recognizes by
    identity. Status: accepted in the pass 2b review (146c69b), which
    is consistent with the read-only views of NIfTI and LTA.
51. Saving a plain `Affine`, or a lazy NIfTI displacement field, to an
    `.x5` file with `io.save` either has no converter and raises
    `ConversionError`, or writes an empty chain. The problem is older
    than this work and out of its scope. Status: open.
52. `X5Raw.copy()` shared the metadata dictionaries of the nodes between
    the copy and the source. Status: resolved in the pass 2b review
    (146c69b), where the dictionaries are copied deeply.
53. The legacy deformation fields written by fslpy were read eagerly.
    Status: resolved in the pass 2b review (146c69b), where they are
    read lazily or the eager read is documented.
54. `X5Raw.from_raw` did not check `chain` and `position` when it was
    called, unlike `from_h5`. Status: resolved in the pass 2b review
    (146c69b), where it checks them eagerly.
55. Two minor costs of X5 are logged without a fix. The `selection`
    warns on every read of a multi-node file that is read without a
    chain, although in practice it warns once because the decoded result
    is cached, and the writer reads each lazy field whole to copy it.
    Status: accepted.
56. `M3zMorph` has no `raw` field. The positions live in the record, and
    the field is a read-only view of `record.positions`
    (`io/transformations/freesurfer/m3z/_xform.py`). Status: accepted in
    pass 2c, in the same category as the matrix of LTA.
57. A copy of an M3Z record (`M3zRaw.copy()`) shares its arrays instead
    of copying them, and pickling the metadata serializes the whole morph
    (`io/transformations/freesurfer/m3z/_metadata.py`). Dropping the
    arrays would make the record unwritable, and sharing them keeps
    copies free. Status: accepted in pass 2c.
58. Assigning a chain to an M3Z transformation does not build a record
    until the file is written, because the encoding needs the writer
    options and may refuse. X5 behaves in the same way. Status: accepted
    in pass 2c.
59. `M3zMorph.consolidate(spacing=, image_shape=, atlas_shape=)` takes
    writer options, unlike the `consolidate` of LTA and X5, because a
    chain cannot say the shape of an image. Status: accepted in pass 2c.
60. Assigned `input` and `output` systems were not counted as overrides
    by M3Z, so a changed system was silently ignored on write. Status:
    resolved in pass 2c (730ef86), as a rule for every chain format, and
    the writer checks that the file can express the systems, which are
    RASmm for X5 and M3Z, and refuses otherwise.
61. `M3zMetadata().save` raises `WriterError` for metadata without a
    record, while LTA and X5 write an empty record. FreeSurfer cannot
    read a morph that has no nodes. Status: accepted in pass 2c.
62. The read-only protection is a flag on the arrays that the reader
    builds. Arrays that are unpickled come back writable, and the arrays
    of an `M3zRaw` that a user builds are not protected, while the
    decoded field is always a read-only view. Status: accepted in pass
    2c. The lesson is to make the decoded views read-only by
    construction, and not to trust the flags of the record.
63. Only `M3zRaw` implements `from_bytes`. The other classes go through
    the `BytesIO` adapter, which has no measurable copy. Status:
    accepted in pass 2c.
64. The helper `_decompressed` of the conformance harness recognizes a
    compressed file by the gzip magic number and not by a `.gz` suffix.
    Status: accepted in pass 2c. In pass 4a, the checks compare saves
    with `_tree`, which decompresses each file of a directory in the same
    way.
65. `M3zMorph.from_instance` given a sequence of transformations
    relabels any chain, and the writer refuses it later. X5 behaves in
    the same way (item 51), and the problem is older than this work.
    Status: open.
66. Check 6 of the conformance test is vacuous for the formats that are
    parsed in one pass, which are LTA, M3Z and `register.dat`, because
    the truncated copy is the whole file. Status: accepted in pass 2c.
67. The record of an MGH image keeps the geometry of a file whose
    `goodRASFlag` is 0 as it is stored, although the reader ignores it.
    nibabel overwrites that geometry when it builds or copies a header,
    so `_stored_header` and `copy` restore it
    (`io/common/mgh/_raw.py`). The geometry methods of nibabel on the
    record therefore see the ignored geometry, which is the path that
    the FSL formats use (item 77), and `MghRaw.vox2ras` is the right
    source. Status: accepted in the pass 3a review (4411bec).
68. `MghMetadata.save` writes only the fixed header of 284 bytes, because
    the footer and the tags lie after the voxels
    (`io/common/mgh/_raw.py`). Check 6 of the conformance test is
    therefore weak for MGH. Status: accepted in the pass 3a review
    (4411bec).
69. The geometry of an MGH image counts as unchanged when its spatial
    dimensions equal those of the record and the `vox2scanner` matrix of
    the model equals `record.vox2ras` exactly
    (`io/images/freesurfer/mgh/_image.py`). The test cannot be whether
    `transformations` was assigned, because `from_instance` and
    `replace` assign the decoded list. Re-encoding the geometry does not
    drift: none of 300 random oblique geometries changed when it was
    encoded again. Status: accepted in the pass 3a review (4411bec).
70. `MghImage.consolidate` keeps the held metadata after it compares the
    record with an encoded copy, and not only after it checks that
    nothing was assigned (`io/images/freesurfer/mgh/_image.py`). AFNI,
    MRtrix and NRRD compare in the same way. Status: accepted in the
    pass 3a review (4411bec).
71. An untouched MGH file is saved byte for byte only for canonical
    layouts, where bytes 90 to 283 of the header are zero. A file
    without a footer gains a footer of 20 zero bytes
    (`io/common/mgh/_raw.py`). Status: accepted in the pass 3a review
    (4411bec).
72. A proxy that is built over a stream passed by the caller keeps that
    stream open (`io/common/mgh/_raw.py`), as NIfTI does (item 9).
    Status: accepted in the pass 3a review (4411bec).
73. The FSL helper `_shape(obj)` (`io/transformations/fsl/_affines.py`)
    reads `obj.shape`, and for an MGH image that decoded the voxels,
    because `shape` and `ndim` came from the data model
    (`datamodel/images.py`). This was a regression from
    `MghReaderWriter`, and `NiftiImage` has had the same gap since pass
    1b. Status: resolved in the pass 3a review (4411bec), where
    `MghImage` and `NiftiImage` each give an explicit `shape` read from
    `raw.shape`.
74. `io/common/mgh/_views.py` repeats about ten lines of
    `_read` from the NIfTI views, as explicit code in each format.
    Status: accepted in the pass 3a review (4411bec).
75. `MghImage.to_nibabel` drops the tags. The geometry of a flag-0 file
    is encoded again as the default LIA orientation with flag 1
    (`io/images/freesurfer/mgh/_image.py`). Status: accepted in the pass
    3a review (4411bec).
76. The writer options `dims=` and `type=` of MGH wrote a header that
    disagreed with the voxels. Status: resolved in the pass 3a review
    (4411bec), as a rule for every image format: a header field that the
    data or `dtype=` decide cannot be overridden, and the writer refuses
    it with `WriterError`.
77. The FSL helper `_best_affine`, for an `MghImage`, goes through
    `MGHHeader.get_best_affine()`. For a flag-0 file that gives the
    stored geometry that the reader ignores, which was already wrong
    before pass 3a. The fix is to prefer `record.vox2ras` when the
    record is an `MghRaw`. Status: open.
78. `_frame_tr` ignores a changed time scaling in `transformations[0]`,
    so a knob is neither honored nor refused, which breaks the rule on
    overrides. The problem is older than this work. Status: open.
79. The option `like=` is silently ignored when its value is not an MGH
    image. The problem is older than this work. Status: open.
80. MGH imports `_is_local` from the private module
    `io/common/nifti/_files.py`. The function should move to
    `io/common/_files.py`. Status: open.
81. `replace` on an `MghImage` read `data` and so materialized the
    proxy, as for `NiftiImage` (item 18). Status: resolved by pending
    data, as in item 18.
82. `Metadata.__eq__` is based on identity, not on content. This is a
    matter of the base class and not of MGH. Status: open.
83. The axis changes that depend on the record live in the lazy arrays
    of AFNI and MRtrix, and not in `to_model` and `to_disk`, which are
    the identity. For AFNI this is the squeeze of a single sub-brick
    (`io/common/afni/_data.py`). For MRtrix it is the permutation and
    the flips of the layout (`io/common/mrtrix/_data.py`). The
    transposition needs the record, and the lazy array is the object
    that holds it. The third-pass paragraph above and the rules are
    amended to say so. Status: accepted in the pass 3b review (c7e2a4e).
84. The AFNI and MRtrix records keep the text or the bytes that they were
    read from (`AfniRaw.text` and `MrtrixRaw.stored`), and the writer
    reuses them only after it parses them again and finds the same
    attributes (`io/common/afni/_raw.py` and `io/common/mrtrix/_raw.py`).
    An attribute whose value is NaN never compared equal to itself, so
    the stored text was discarded and the header was rewritten. Status:
    accepted in the pass 3b review (c7e2a4e), and the NaN case is
    resolved there, where the comparison of the records in
    `_format_attributes` treats NaN as equal to NaN.
85. The conformance harness has a new field, `Exemplar.header_file`, for
    the formats that keep their header in a separate file
    (`tests/test_io_conformance.py`). AFNI uses it, because its header is
    the `.HEAD` file and its voxels are in the `.BRIK` file. `from_bytes`
    of AFNI is refused, because one buffer cannot hold two files. The
    field also answers item 23 for the formats of this pass. Status:
    accepted in the pass 3b review (c7e2a4e), and open for NRRD and MINC
    (item 23).
86. `replace` on an `AfniImage` or a `MrtrixImage` reads `data` and so
    materializes the lazy array, as for `NiftiImage` (item 18) and
    `MghImage` (item 81). The copy then recomputes `BRICK_STATS` and
    receives a new `IDCODE` for AFNI, and a scaled MRtrix image is stored
    as floating point. The problem is shared by every image format and is
    to be fixed once in the base, as in item 18, and not once per format.
    Status: resolved by pending data, as in item 18, for every format at
    once.
87. Saving an AFNI image with a different compression of the BRIK file
    removed the old BRIK file, which the lazy array of the loaded image
    still read, so the image lost its data (`_BrikProxy` in
    `io/common/afni/_data.py`, and `io/images/afni/_image.py`). This was
    a loss of data, and the rule that a save never removes a file that a
    live proxy reads comes from it. Status: resolved in the pass 3b
    review (c7e2a4e), where `raw` points to the new BRIK file after the
    old one is removed. Such a save is now a save over the dataset of
    the image (#426), so the image is consolidated in place first.
88. The `__getitem__` of the proxies of AFNI and MRtrix decoded the whole
    file when Dask asked for an empty selection to find the metadata of
    the array, so a Dask array read the file twice. Status: resolved in
    the pass 3b review (c7e2a4e), where an empty selection is answered
    without reading, and the tests no longer raise the warnings of
    `array_equal` on Dask arrays.
89. Two older knobs are neither honored nor refused. A TR given through a
    `Scaling` in `transformations` is ignored by AFNI when the number of
    sub-bricks does not change (`io/images/afni/_image.py`), and the
    voxel sizes of MRtrix beyond the third axis are taken from the
    record and not from the model (`io/images/mrtrix/_image.py`). Both
    break the rule on overrides, in the same way as the time scaling of
    `_frame_tr` for MGH (item 78). The problems are older than this work.
    Status: open.
90. The layout of an MRtrix image is never `None`, even when it is the
    identity, so the writer asks the layout whether it is trivial
    (`layout.trivial`) instead of comparing it with `None`
    (`io/images/mrtrix/_image.py`). Status: accepted in the pass 3b
    review (c7e2a4e).
91. A `.mih` file cannot be a variant of the untouched save, because its
    header names its data file and so cannot be moved to another name
    without a change. Status: accepted in the pass 3b review (c7e2a4e).
92. AFNI keeps the `IDCODE` of an untouched file, and writes a new one
    only for an image that changed. The option `attributes=` of AFNI
    refuses the attributes that describe the storage, and the option
    `keyval=` of MRtrix refuses the keys for the dimensions, the voxel
    sizes, the layout, the data type, the transformation, the scaling and
    the data file. These values are decided by the data or by the model,
    in the same way as item 76. Status: accepted in the pass 3b review
    (c7e2a4e).
93. `attributes=` of AFNI first accepted the attributes of the geometry,
    which the writer encodes from the model, and applied them after the
    encoding, so the header could contradict the model. Status: resolved in the pass 3b review (c7e2a4e),
    where `attributes=` refuses them with `WriterError`.
94. An attribute of AFNI with a string that is not Latin-1 failed with an
    encoding error that is not a `WriterError`. Status: resolved in the
    pass 3b review (c7e2a4e), where the writer raises `WriterError`.
95. Pass 3b removes the names `AfniReaderWriter`, `AfniHeader`,
    `MrtrixReaderWriter` and `MrtrixHeader`, and the attributes `header`,
    `_header` and `dataobj`, without aliases. Loading a compressed
    AFNI or MRtrix file no longer reads its voxels. They are to be
    announced with the other changes of item 13. Status: open, as for
    item 13.
96. For MRtrix, a `#` inside the value of a key truncates the value,
    because it starts a comment. The problem is older than this work.
    Status: open.
97. An AFNI image that is read from a `.BRIK.gz` file and saved to a
    `.HEAD` name writes an uncompressed BRIK file. Status: open.
98. The helpers `_declined` and `_sniffed_raw` are repeated in
    `io/common/afni/_raw.py` and `io/common/mrtrix/_raw.py`. Status:
    open, and a shared helper in `io/base/parsers.py` is a candidate.
99. `_views.py` now exists in three copies, for MGH, AFNI and MRtrix, in
    line with the rule of explicit code in each format (item 74).
    Status: accepted in the pass 3b review (c7e2a4e).
100. After `consolidate()`, the geometry is decoded from the new record,
     so it has the precision of the file: a NIfTI sform in single
     precision, a qform as a quaternion, and the 16 significant digits
     of LTA (item 44). An assigned geometry is therefore not returned
     exactly by the consolidated copy. Keeping the assigned list in
     `_transformations` would keep it exact, but something would then
     stay pending after a consolidation. Status: accepted.
101. MINC files cannot be written, so `MincImage.consolidate()` only
     moves pending data into `raw` (`io/images/minc/_image.py`). The
     metadata is kept, and so are the transformations that the image
     holds, read or assigned, because no record can be encoded. The conformance harness checks
     this data-only consolidation instead of checks 16 and 17. Status:
     accepted.
102. A consolidated NIfTI affine has no `raw` array, because its matrix
     lives in the header that the consolidation encodes
     (`io/transformations/nifti/_affines.py`). The harness reads the
     matrix of the header through `Exemplar.stored`, as it reads the
     matrix of LTA. Status: accepted.
103. The formats whose data model is a chain (the NIfTI displacement
     field, SPM, NiftyReg, ITK and FNIRT) declare a private `_data`
     field, so that their `data` can be pending, and their constructor
     therefore takes `data=`. Check 1 allows `data` for them. Status:
     accepted, and to be announced with item 13.
104. A save over the file that the lazy array of an object reads
     consolidates the object in place, and then points `raw` at the new
     file. The consolidation alone is not enough, because the lazy array
     of the old file describes how that file stored the data, such as
     its offset, its type and its scaling, which the new file may store
     differently, either because a header changed length or because a
     writer option such as `dtype=` changed the storage. Status:
     accepted.
105. An AFNI dataset that is encoded again receives a new `IDCODE`, so
     two consolidations of the same image give different headers. Check
     17 therefore compares the files that a save and a save of the
     consolidated copy write by reading them back, and not by their
     bytes. Status: accepted.
106. An X5 file is written with h5py in mode `"w"`, which truncates the
     file, while the lazy fields of an X5 record read the file by its
     name, so an X5 transformation saved over its own file loses its
     fields. X5 keeps its fields in its record and has no `raw`, so
     check 18 does not cover it. The problem is older than this work.
     Status: open.
107. A writer option that changes how the data is stored, such as
     `dtype=` for NIfTI or `datatype=` for AFNI, was recorded in the
     header of the consolidated copy, while `raw` held the values before
     they were converted to that type, and the NIfTI header had no
     scaling yet. The copy did not hold what its file holds. Status:
     resolved. The copy holds the values as the writer converts them,
     and the record holds the header that the writer writes, as the
     section on consolidation describes. Check 16 compares them exactly.
108. `ZarrImage`, `OmeZarrLevel` and the AFNI image of the core take
     `data` over as `_data`, so that `replace` keeps their lazy source,
     and the permutation of an OME-Zarr level is a field (`perm`), so
     that a copy still transposes the array that it reads. Status:
     resolved in the first pass of pending data.
109. An untouched object is written with its record as it is, so a NIfTI
     qform whose code is 0, or the `DISPVECT` intent of a legacy field
     of coordinates read through a hint, is kept on an untouched save.
     The transformations are compared at consolidation, so assigning an
     equal list, such as `img.transformations = img.transformations`,
     keeps the record too. Writing the same object with its geometry
     encoded again needs a different list or a writer option. Status:
     accepted.
110. Some readers of the record do not see what is pending. The FSL
     helpers `_best_affine` and `_pixdim`
     (`io/transformations/fsl/_affines.py`) read `metadata.raw` of a NIfTI
     image, so after `img.transformations = ...` they still give the
     geometry of the file. `NiftiImage.system` is decoded from the
     header, so it ignores pending data of another number of dimensions.
     Status: open.
111. FNIRT cannot encode a chain. An assigned chain was neither written
     nor refused: `consolidate` treated the field as untouched, and a
     save wrote the file without it. Status: resolved. `consolidate`
     encodes when a chain is pending, and the encoder raises
     `UnrepresentableTransformationError`
     (`io/transformations/fsl/fnirt/_base.py`).
112. A NIfTI image or transformation saved over its own file pointed
     `raw` at the new file but kept the record of the consolidation,
     whose header differs from the one written. Status: resolved. The
     object reads the record and the lazy array of the new file, and
     check 18 compares its header with that of `load(path)` for every
     format.
113. With the exact comparison of check 16, an X5 field read lazily was
     decoded in double precision, while its consolidated copy was decoded
     in single precision. Dask ignores the `dtype` of `asarray`, so the
     matrix that rotates the vectors into voxel units stayed in double
     precision (`io/transformations/base/fields.py`). Status: resolved,
     where the matrix is cast by NumPy.
114. The formats whose data model is a chain (the NIfTI displacement
     field, SPM, NiftyReg, ITK, FNIRT, X5 and M3Z) keep their
     transformations pending by assignment. Their decoded chain is an
     immutable tuple, so it cannot be changed in place, and comparing a
     chain with a fresh decode would compare whole fields of vectors.
     Assigning an equal chain therefore still encodes it again. Status:
     open, for the maintainer to decide whether these formats compare
     their chain at consolidation too.
115. JPEG 2000 has a pyramid before the fourth pass decides the levels of
     a multiscale image (item 15). `Jpeg2000MultiScaleImage` holds a lazy
     array of the full resolution in `raw`, takes `images` and
     `transformations` over as `OmeZarrImage` does, and decodes its levels
     from `raw` and the record as `Jpeg2000Image` objects that share its
     metadata (`io/images/jpeg2000/_multiscale.py`). It is consolidated
     through its full-resolution level alone. As the maintainer decided,
     the coarser levels are taken to be downsampled versions of the first
     one, which the codec derives again: whatever was assigned to them,
     data or transformations, is ignored rather than refused, and the
     ratios of their shapes are not checked, here as for any format. The
     pyramid is listed in `NOT_MIGRATED`, and its full-resolution level
     is an exemplar. Status: accepted for the coarser levels, and open
     until the fourth pass.
116. A raw JPEG 2000 codestream has no resolution box, so the pixel size
     of an image written as a `.j2k` file is lost, as for the Pillow
     formats that store no resolution. The checks of geometry would be
     vacuous for such a file, which is therefore not a variant of the
     conformance harness. Status: open.
117. The JPEG 2000 record keeps the bytes of the file up to its first tile
     (`Jpeg2000Raw.stored`), as NRRD does, so that the metadata can be
     written. An untouched full-resolution image copies its file. When
     only the comments, the resolutions, the XML and UUID boxes or the
     container change, the header is rewritten around the coded image,
     which is copied (`io/images/jpeg2000/_boxes.py`), and a record that
     changes a field that the coded image decides is refused with
     `WriterError`. Any other change encodes the pixels again with
     Pillow, and the comments of the record, including the version
     comment of the encoder of the source, are written over those of the
     new encoder. Status: accepted.
118. A JPEG 2000 image read with `level=r` for `r > 0` is not the file
     that it reads, so it is encoded again whenever it is consolidated,
     even untouched. The level is the level that `raw` reads, and not a
     field, so that the consolidated copy, whose `raw` reads the new
     file, holds level 0. The options `pixel_size=` and `unit=` of the
     readers become assigned transformations, so a file read with them
     is written with that pixel size, and not byte for byte (item 11).
     Status: accepted.
119. Pillow decodes a whole JPEG 2000 level at once, so a Dask array
     built over the lazy array of a large level, which Dask cuts into
     several chunks, decodes the level once per chunk. Status: open.
120. Pillow reads signed JPEG 2000 samples but cannot encode them, so an
     image of signed samples can be written only while its coded image is
     copied. Status: open, and older than this work.
121. An image stack (#193) is an image whose file is a folder of
     numbered two-dimensional files (`io/images/stack`). The folder has
     no header, so its record, `StackRaw`, lists the files and holds the
     header of each one, with the stacking axis and the geometry of the
     stack. TIFF and Pillow images have no record on the main branch
     yet, so the header of a member is the structure that their readers
     already parse: the `TiffMetadata` of the TIFF backend, and a
     `PillowHeader` that the Pillow backend reads without decoding the
     pixels. Status: open. The members are to hold `TiffRaw` and
     `PillowRaw` once the TIFF (#459) and Pillow (#458) records are on
     the main branch.
122. Image stacks are written as folders of numbered files, through the
     TIFF and Pillow writers of single files. `StackImage.consolidate()`
     is the only encoding, and a save writes a consolidated copy and
     leaves the image unchanged. An untouched stack copies the bytes of
     its files, so a folder that is saved again is identical file for
     file. A slice without a file is not written. A folder is written
     beside its target and moved in its place by `_replace_folder`
     (`io/images/stack/_files.py`), which follows `replace_directory` of
     the pyramid core (#451) and replaces an existing folder only if it
     holds nothing but images, and nothing but the files that the image
     reads when it reads some of them. Files with numbered names, such as
     `out_0.tif` for the path `out.tif`, are all written under hidden
     names before any of them is renamed. Status: implemented. The
     helper is to be replaced with `replace_directory` once #451 is on
     the main branch.
123. The conformance harness has a field `Exemplar.folder` for a format
     whose object is a folder. Check 12 then checks that the format and
     its metadata refuse bytes and streams, since neither can hold a
     folder. Check 5 compares the folders file by file, and writes the
     copy of the generic writer with `from_instance` and the format
     itself, since no extension names a folder for `io.save` to choose
     from. The metadata of a stack is written only with its image, so
     checks 7 and 10 compare the records of metadata that cannot write.
     The stack has no writer option that changes its stored data, so
     check 16a is skipped. Check 6 cuts each TIFF file of the folder
     before its pixels, which tifffile writes after the tags. Status:
     accepted in the review of #431.
124. The size of the pixels, the spacing of the slices and their unit
     are fields of `StackRaw`, decoded from the files where they record
     them, and the reader options `pixel_size`, `spacing` and `unit` set
     them on load, as the review of #431 asks "for now". An untouched
     stack is copied byte for byte, so a size that an option set is not
     written into its files. A consolidated record holds only what the
     files record: a TIFF file records the pixel size when it has a unit
     of length, and the magnitude of the spacing, as the `PhysicalSizeZ`
     of a volume of one slice, when it has no channels, and a Pillow file
     records no size. A translation, a rotation and a flip within a file
     are not recorded. Status: open, until the TIFF and Pillow records
     decide where these sizes live.
125. `StackImage` and `StackMetadata` claim a local folder that is not a
     Zarr store and whose image files follow one numbering, with the
     confidence `LIKELY`, judged by the names alone. They never claim a
     file, a stream or bytes, so they take nothing from the TIFF and
     Pillow readers. Status: accepted in the review of #431.
126. A number that no file carries is a missing slice. The record does
     not store the missing numbers, which follow from the numbers of its
     members (`StackRaw.missing`), so a record cannot contradict itself.
     The lazy array fills a missing slice with the reader option `fill`,
     NaN or zero by default, without reading a file. `fill` describes
     how the pixels are read, so it is held by the lazy array and is not
     a field. Assigned data covers every slice, so a stack consolidated
     with assigned data has no missing slice, and the values that filled
     a missing slice are written as a file. Status: open, for the
     maintainer to decide whether assigned data should keep the missing
     slices of the record.
127. The stacking axis is the field `StackRaw.axis`, which the reader
     option `axis` sets and the writer option `axis` changes. Names that
     carry several numbers, one per axis, are refused. Supporting them
     would give each member one number per stacking axis and the record
     one axis per number, which the fields `StackMember.number` and
     `StackRaw.axis` leave room for. Status: open, for a later pull
     request.
128. `StackImage` declares no extension, so `io.save` writes an image
     stack to `volume.tif` as one multi-page TIFF file, and cannot
     choose the stack for a path without an extension. A folder is
     written with `StackImage.save`, or with
     `StackImage.from_instance(image).save` for another image. Declaring
     the TIFF and Pillow extensions would let `io.save(stack, "out.tif")`
     write numbered files, but would also make the stack a candidate for
     every TIFF and Pillow file that is loaded. Status: open, for the
     maintainer to decide.
140. MNI transform files (#222) are parsed in one pass, so the matrices
     of their linear transforms live in the record, as for LTA. The
     displacements of a grid transform live in a separate MINC volume,
     which `XfmTransform` reads into `raw` as a tuple of lazy
     `MincImage` objects, one per grid transform
     (`io/transformations/minc/_chain.py`). A file without a grid
     transform has no `raw`, so check 18 does not apply to the exemplar,
     and check 6 is vacuous, as for LTA and M3Z (item 66). Status:
     accepted, pending review.
141. brainhops cannot write MINC volumes, so an `XfmTransform` writes a
     grid transform only as the volume that it was read from, which it
     recognizes by the identity of the decoded element, as X5 does (item
     36), and which it copies beside a file written elsewhere. A new
     displacement field raises `UnrepresentableTransformationError`.
     Status: open, until a MINC writer exists.
142. libminc writes a MINC1 displacement volume without a variable for
     `vector_dimension` (`volume_io/Volumes/output_mnc.c`), and both the
     MINC reader of brainhops and nibabel refuse such a file, so only
     MINC2 displacement volumes, whose HDF5 layer gives every dimension a
     dataset, are read. Status: open. Reading these volumes would need
     `io/common/minc` to read the voxels of MINC1 without nibabel's
     `Minc1File`.
143. The record of an MNI transform file keeps the text that it was read
     from, and writes it while it still parses to the fields of the
     record, as AFNI and MRtrix do (item 84). Comments, the layout of the
     numbers, line endings and bytes that do not decode as UTF-8 are
     therefore kept by an untouched save. A record that is encoded again
     keeps only the comments at the top of the file. Status: accepted,
     pending review.
144. libminc interpolates a grid transform with cubic interpolating
     splines, lowers the degree near the edges, and applies no
     displacement beyond half a voxel outside the grid, and it inverts a
     grid transform by an iteration. brainhops uses cubic B-splines of the
     values with a zero boundary and the inverse of the data model. The
     two agree at the nodes of the grid only. Status: accepted and
     documented in `XfmGridTransform`.
150. A `register.dat` file holds no volume geometry, so the geometries
     of the reference and the movable volume are options of the reader,
     `reference=` and `moving=`, which are kept on the transformation and
     never written (`io/transformations/freesurfer/regdat/_xforms.py`).
     The view `RegisterDatTransformationRASToRAS` reads no matrix without
     them, and the conformance checks load files without options, so the
     view is not one of their variants and is tested in
     `tests/test_io_regdat.py` instead. Status: accepted.
151. The tkregister RAS space of a volume is one class,
     `FreesurferTkrSystem` (`io/common/freesurfer/_tkr.py`), which MGH
     images, `register.dat` and LTA files share. It is not `RASmm`, so
     the formats that check the class of their systems, such as LTA and
     MINC, do not write a tkregister space as scanner RAS. The data model
     compares systems by their axes only, and the axes of a tkregister
     space are those of RAS in millimetres, so a converter that bridges
     or checks endpoints by their axes, such as those of the MRtrix
     linear transform and of the NIfTI voxel-to-RAS affine, still accepts
     a tkregister space as scanner RAS. The tkr space of a 4-D MGH image keeps a system of
     four axes named `"tkr"`, as its scanner space keeps four axes named
     `"scanner"`. Status: open, for the maintainer to decide whether a
     system may be told apart from RAS by more than its axes.
152. The geometries `reference` and `moving` of a `register.dat`
     transformation (item 150) are `Geometry` values of the data model.
     An image gives its `geometry`, and an LTA geometry block is
     converted, without its file name. The geometry must place a 3-D
     volume in an anatomical space in millimetres by an affine, and any
     other geometry is refused when it is given. A `Geometry` compares
     equal only to itself, so a tkregister system that holds a geometry
     equals only a system that holds the same object, and the systems
     that an LTA file decodes hold no geometry: the record keeps it.
     Status: accepted, pending review.
153. A `register.dat` transformation that knows both geometries converts
     exactly to an LTA file of type `LINEAR_RAS_TO_RAS`, the type that
     `lta_convert --inreg` writes, and an LTA file that knows both
     geometries converts exactly to a `register.dat`
     (`io/transformations/freesurfer/regdat/_converters.py`). Both keep
     the map, so the reference volume is the source of the LTA file.
     FreeSurfer keeps the roles of the volumes instead, and its LTA file
     maps the movable volume to the reference, so the file written here
     is the one that `lta_convert --inreg ... --invert` writes. `LtaRaw`
     has no field for the `subject` and `fscale` lines that FreeSurfer
     appends to an LTA file, so the subject of a `register.dat` is lost
     through LTA. Status: accepted, pending review.
154. LTA reads type 14, `REGISTER_DAT`, which `ltaReadFileEx` in
     `utils/transform.cpp` of FreeSurfer reads as any other type and
     `LTAchangeType` converts. The record stores the matrix from the
     tkregister space of `dst` to that of `src`, so the transformation
     reads its inverse, and a matrix assigned between two tkregister
     spaces is written with this type and inverted. FreeSurfer refuses
     to change the type of such a file without valid geometries, whereas
     brainhops writes it with invalid geometry blocks when no geometry is
     known. Status: accepted, pending review.
155. `register.dat` and the plain matrix files both claim `.dat`. Their
     sniffers do not accept each other's content, so reading picks the
     right one in both directions. Writing has no content to sniff, and
     `io.save` prefers `RegisterDatTransformation` for a plain `Affine`
     saved to a `.dat` name, because it is the more specific format. An
     affine between `RASmm` spaces is then refused, since a
     `register.dat` file needs the geometries to store it, where the
     plain matrix format wrote it before `register.dat` was added.
     Status: open, for the maintainer to decide whether `io.save` should
     fall back to another candidate when the chosen one refuses.
160. A Neuroglancer precomputed volume holds several scales, so it is read
     as a pyramid, `PrecomputedMultiScaleImage`, whose levels are
     `PrecomputedImage` objects (`io/images/precomputed`). Each level
     follows this design: it holds the `info` file of the volume as its
     record and the lazy array of its scale in `raw`, and it is an
     exemplar of the conformance tests. The pyramid holds the same
     metadata as its levels and has no `raw` of its own, and its
     `consolidate` consolidates its levels. The rules for the levels and
     the record of a pyramid are decided in the Zarr pass (item 15), so
     the pyramid is listed with the other pyramids among the formats that
     are not migrated. Status: open.
161. Precomputed volumes are read and not written, as MINC (item 101).
     A volume is a directory of files, local or remote, while the atomic
     save of `io/base/_utils_files.py`, the save over the file that a
     proxy reads (#426) and the untouched copy that is identical byte
     for byte are defined for files. A writer would also encode a pyramid,
     whose rules wait for the Zarr pass (item 160), and TensorStore,
     which brainhops would write with, writes its own `info` file, with
     other members, such as `jpeg_quality`, and other formatting. Status:
     decided for the reader, and open for a writer.
162. The specification of precomputed volumes does not say where the
     centre of a voxel lies. The viewer rounds a position down to find
     its voxel and subtracts half a voxel before it interpolates, so the
     corner of a voxel sits at an integer coordinate, and the
     transformation of a scale moves by `(voxel_offset + 0.5) *
     resolution` nanometres (`io/images/precomputed/_image.py`, with the
     sources in `io/images/precomputed/__init__.py`). Status: decided.
163. The conformance exemplar of `PrecomputedImage` names a volume by its
     `info` file, with the suffix `/info`, because checks 6 and 12 read
     and copy the file that a case names, and a volume is a directory.
     `Exemplar.header_file` is the identity, so that the bytes of the
     `info` file are refused as a volume and read as its metadata. The
     checks write the volume as `volume.md` describes it, without
     TensorStore, and the tests of the format read volumes named by their
     directory. Status: accepted.
164. The readable encodings are those that TensorStore 0.1.45 decodes:
     `raw`, `jpeg` and `compressed_segmentation`, sharded or not. The
     list is fixed in `io/images/precomputed/_data.py`, so a scale in
     `png`, `compresso`, `jxl` or `crackle` is refused when it is
     loaded, even with a later TensorStore that decodes one of them.
     TensorStore reads a missing chunk as zeros. Status: open, until the
     list follows the version of TensorStore.
165. A remote volume is read with the key-value stores of TensorStore,
     which read `gs`, `http` and `https` URLs in version 0.1.45, and not
     `s3`. The `precomputed://` prefix of Neuroglancer is not accepted,
     because `bagof.paths.Path` merges the slashes of a nested URL. The
     tests read a remote volume through a `file://` URL and use no
     network. Status: accepted.
170. The Convert3D matrix format of Greedy, `c3d_affine_tool` and
     ITK-SNAP (`io/transformations/c3d`, #197) is read in one pass, as
     LTA is, and its record `C3dAffineRaw` holds the matrix, the
     characters that end each row and the text of the file. The text is
     written back only while it still parses to the matrix and the
     layout of the record, as in item 84, so an untouched file keeps its
     bytes, including its six-digit numbers. Check 6 of the conformance
     test is vacuous, because the file holds no data beside its record,
     and check 10 edits `row_end`, the only field of the record outside
     the matrix. Status: open, for review.
171. `C3dAffine` repeats four short helpers of LTA (`_record`,
     `_read_only`, `_matrix_to_model` and `_matrix_to_disk` in
     `io/transformations/c3d/_affine.py` and
     `io/transformations/freesurfer/lta/_xforms.py`), because the LTA
     helpers are private to their package. They could move to
     `io/transformations/base` once a third one-pass affine format needs
     them. Status: open.
172. A Convert3D matrix holds no marker, so `C3dAffine` never claims a
     file from its content and is selected by the hints `c3d`, `greedy`
     and `itksnap`. Without a hint, a `.mat` file of sixteen numbers is
     still read as a FLIRT matrix, whose meaning differs, and a `.txt`
     file as a plain matrix. The format declares the `.mat` extension,
     so that `io.save` and the metadata dispatcher can choose it, and
     `io.save` of an affine between world spaces to a `.mat` file, which
     raised `WriterError` before, now writes a Convert3D matrix. Status:
     open, for the maintainer to decide whether `.mat` should select this
     writer.
173. `RASToRAS` checks its systems only when it is built
     (`io/transformations/base/affines.py`), so an `input` or an `output`
     that is assigned later is not refused there. `C3dAffine.consolidate`
     checks both systems again with the private `_check_endpoint` and
     refuses a system that is not RAS with
     `UnrepresentableTransformationError`. `NiftyRegAffine` has no such
     check. Status: open.
180. Imaris files are read only, as MINC files are (item 101). An
     untouched Imaris file could not be saved byte for byte: its `Data`
     datasets are always chunked and compressed, often with the LZ4
     filter that h5py reads only with a plugin, and h5py rewrites only
     contiguous datasets identically (item 48). A valid file also needs
     every resolution level, binned as Imaris bins them, and the
     histograms of every level, which belong to the writer of a pyramid
     whose record is not designed yet (item 15). `ImarisImage.consolidate`
     therefore moves pending data into `raw` only
     (`io/images/imaris/_image.py`). Status: open, for the maintainer to
     confirm.
181. An Imaris pyramid holds no record of its own.
     `ImarisMultiScaleImage` is listed among the formats that are not
     migrated, and its levels are `ImarisImage` objects, which follow the
     design and share one `ImarisMetadata`. Its record describes the
     whole file, every level included, and the level that an image holds
     is the reader option `level`, which the image keeps as a field, as
     X5 keeps `chain` and `position` (item 49). The Zarr pass plans one
     record per resolution level instead. Status: open, until the record
     of a pyramid is decided (item 15).
182. The time points and the channels of an Imaris file are axes of the
     data, which always have five axes, `(x, y, z, t, c)`, even for a
     single time point and a single channel. Brainhops orders space, time
     and channels in this way for NIfTI, TIFF hyperstacks and OME-Zarr,
     Imaris itself treats every file as five-dimensional, and the lazy
     array reads only the datasets that a selection covers, so options
     that select a time point or a channel would duplicate an index of
     the data. Such options would also leave the record describing data
     that the image does not hold. The time axis stays an index, although
     `/DataSetInfo/TimeInfo` records a date for each time point, because
     the dates need not be evenly spaced. Status: open, for the
     maintainer to confirm.
183. Every level of an Imaris pyramid is placed over the whole extent of
     the file, as TIFF pyramids are, so its voxel size is the extent
     divided by its size. ImarisWriter bins two voxels into one and drops
     the last voxel of an odd axis (`bpMultiresolutionImsImage.cxx`), so
     its coarse levels are placed by a factor of exactly two, which
     imaris-ims-file-reader also uses. The two placements differ by less
     than one coarse voxel along an odd axis, and how Imaris itself places
     the levels is not documented. Status: open.
184. The record of an Imaris file holds the size, the stored shape and
     the type of each level, read from its first time point and channel,
     but not the shape of its chunks. The copy of a file that the
     conformance check of metadata reads stores its voxels in an external
     file, which has no chunks, and the record must be the same for both
     files. The lazy array reads the chunk shape from the file when Dask
     asks for it (`io/images/imaris/_data.py`). Status: accepted.
185. The adapter `Hdf5Reader` does not pass reader options to `sniff_h5`,
     so the Imaris classes override `sniff_filename`, `sniff_fileobj` and
     `sniff_bytes` to see the option `level`, which makes the pyramid
     decline and the single image certain. An open `h5py.File` is still
     sniffed without the option. Passing the options to `sniff_h5` would
     break the X5 and ITK sniffers, which take none. Status: accepted.
190. Before the maintainer decision of 2026-10-13, an image stored the
     list that it decoded in `_transformations`, the field of an assigned
     list, and its setter of `metadata` set that field to `None`, so other
     metadata dropped an assigned list too. The decoded list now lives in
     the cache of `transformations`, and the writers and `consolidate`
     read the assigned list, or else the cached one, through
     `held_transformations` (`io/common/_geometry.py`), which reads the
     cache with `cached_value` (`_core/properties.py`). The images of
     the formats merged later (N5, Imaris, precomputed, JPEG 2000,
     BigDataViewer and image stacks) follow the same rule, and the JPEG
     2000 pyramid caches the levels and the placements that it decodes in
     the same way. The chains of the ITK text, MATLAB and HDF5 formats
     (`TfmTransform`, `MatTransform` and `H5Transform`), which are
     mutable lists as the transformations of an image are, follow it too
     (item 312). Status: resolved.
191. A change made in place to a decoded list of transformations lives in
     the cache, so it is lost when other metadata is assigned, and a copy
     made with `replace` or `from_instance` decodes the list again instead
     of copying the change. A write of the image itself, including
     `io.save`, still writes the change. Assigning the changed list keeps
     it everywhere. The images that cannot be written (N5, Imaris,
     precomputed and BigDataViewer) have no record to encode the change into, so their
     `consolidate` gives the copy the changed list as an assigned one.
     Status: open, for the maintainer to decide whether copies within a
     format should carry the cache.
192. Check 20 cannot tell an old geometry from a new one for some
     exemplars. The transformation of `NiftiRASCoordinatesField` and of
     the FNIRT exemplar, which has no moving image, is its data, which
     does not depend on the header, and the MINC files of the harness
     have one geometry. The part of the check on assigned values still
     applies to them. Status: accepted.
193. The NRRD encoder kept the spacings and the positions of the base
     record when the preferred transformation was the unit scaling that a
     file without geometry decodes to, so the file was written with the
     geometry of the base. A test of this work found it. Status:
     resolved, where `_geometry_fields` (`io/images/nrrd/_image.py`)
     drops those fields of the spatial axes whenever there is no
     geometry to write. Check 20 likewise found that an MRtrix image or
     warp with nothing pending copied its lazy voxels under the record of
     other metadata that stores them with another type, which wrote a
     file too short for its header. Its `consolidate` now encodes the
     voxels again unless the record stores them as the file that they
     come from (`stores_as_recorded`, `io/common/mrtrix/_files.py`).

The items from 200 on were found while adding the BigDataViewer formats
(`io/common/bdv`, `io/images/bdv` and `io/transformations/bdv`), which
read a dataset made of an XML file and an HDF5 file, and write the
registrations of its views back into the XML file.

200. The global space of a BigDataViewer dataset has no unit of its own.
     A dataset exported by the Fiji plugin stores the voxel size as its
     only transform (`ExportImagePlusPlugIn`), so its global space is in
     the unit of the voxel size, but BigStitcher stores the voxel size
     divided by the smallest voxel size of the dataset as `calibration`
     (`DatasetCreationUtils.createViewRegistrations`), so its global
     space is in units of that smallest size. BigDataViewer measures the
     global space through the voxel size along x and the scale of the
     transform along x (`ScaleBarOverlayRenderer`). The axes of the
     `"global"` system therefore have no unit, and each level of an image
     also has an affine to a `"physical"` system, scaled by the voxel size
     in its unit (`io/images/bdv/_image.py`). Status: open, for the
     maintainer to decide whether the unit should be derived as the scale
     bar of BigDataViewer derives it.
201. BigDataViewer images cannot be written, as MINC images cannot
     (item 101). A view is one dataset of a chunked and compressed HDF5
     file that holds every setup, time point and level of the
     acquisition, and the XML file describes the whole acquisition, so
     writing one image would mean rewriting or editing in place a file
     that h5py does not write back byte for byte (item 48). The
     registration of a view is written by `BdvViewRegistration`, into
     the same XML file. The `transformations` of an image follows item
     190, and its `consolidate` gives a copy a decoded list changed in
     place, as item 191 says. Status: decided in this pass.
202. The record of an image holds the XML file and also what the HDF5
     file says about the resolution levels of each setup
     (`BdvRaw.pyramids`), which the image reader adds, because the
     transformations of a level are decoded from both. `BdvMetadata.load`
     reads the XML file alone, so its `pyramids` are `None`, and the
     metadata of an image and the metadata of its XML file differ in this
     field while they write the same bytes. Status: open.
203. The XML file is read and written as bytes, by binary adapters,
     because BigDataViewer writes it with JDOM, whose default line
     separator is CRLF (`LineSeparator.DEFAULT`), and a text adapter
     reads with universal newlines and would drop the carriage returns.
     Status: accepted.
204. A chain that is assigned to a registration is written by replacing
     the bytes of the `<ViewRegistration>` element of its view, at the
     offsets that expat reports, so the rest of the file is kept byte for
     byte. An element of the chain that the registration decoded from its
     own record is written as its `<ViewTransform>` element, with its
     `<Name>` and the text of its values, and is recognized by its
     identity, as X5 recognizes its nodes (item 36). An element taken
     from another object, or from a copy made with `replace`, is written
     again without a name. The data model has no name for a
     transformation, so a cleaner design would need a subclass of
     `Affine` that carries the name of its transform. The systems
     between the elements of the chain are unknown and are `None`.
     Status: open, as item 36.
205. A dataset stored in HDF5 is both an image and a set of
     registrations. `BdvViewRegistration` scores `MAYBE`, below the
     image readers, so that `io.load` reads the dataset as an image whose
     transformation is the registration, as it reads a NIfTI file as an
     image rather than as an affine (`NiftiVoxelToRAS` scores `WEAK`).
     `transformations.load` reads the registration, and a dataset stored
     with another image loader, such as N5, is read as a registration by
     `io.load`. No change of the data model is needed. Status: decided in
     this pass.
206. A view is selected by the reader options `setup` and `timepoint`,
     which are ids, and both default to the first id, which is setup 0
     and time point 0 in the datasets that BigDataViewer and BigStitcher
     write. A reader stores the ids that it resolved. The options select
     what is read and do not count as overrides, as for X5 (item 49), so
     a registration is always written back with every view. Status:
     decided in this pass.
207. A file written before the `dataType` attribute existed stores 16-bit
     unsigned voxels as signed integers, which are read back bit for bit
     as unsigned values, as BigDataViewer reads them. BigDataViewer reads
     any other stored type of such a file through an HDF5 conversion to
     signed 16-bit integers, and brainhops reads it in its stored type
     instead. Values stored in another type than the one that `dataType`
     declares are converted with NumPy, while BigDataViewer converts them
     with HDF5, whose rounding and clipping were not checked. Status:
     open.
208. `BdvMultiScaleImage` is listed in `NOT_MIGRATED`, because a pyramid
     has no record yet (item 15). Its levels are `BdvImage` objects,
     which follow the design and share one metadata object. Status: open,
     as item 15.
209. The image readers read the whole XML file to sniff it, and open the
     HDF5 file to count the levels of the setup, so that a view with a
     single level is read as a `BdvImage`, as the Imaris readers do.
     Status: accepted.
210. A generic `<ViewTransform>`, whose content a Java class reads, is
     kept and written back as it is, but decoding the chain of its view
     raises `ParserContentError`, so a chain assigned to that view
     cannot keep it. Status: accepted.
211. A registration saved under another name is the whole XML file, whose
     `<hdf5>` path may be relative, so the copy names an HDF5 file beside
     it that may not exist, as a `.mih` header names its data file
     (item 91). Status: accepted.
212. `BdvMetadata` without a record refuses to write, because a dataset
     needs at least a setup and a time point, as `M3zMetadata` refuses
     (item 61). A registration without a record writes a new file with
     a single setup, a single time point and no image loader. Status:
     accepted.
220. FNIRT coefficient files (#204) placed their knots from the stored
     voxels of the reference, as fslpy does, and decided the offset of
     the knots from the knot spacing after it was rescaled to the given
     reference, so a spacing that rescaled to one lost its offset. FSL
     holds a reference whose voxel-to-world matrix has a positive
     determinant in the radiological order, and writes the coefficients
     in that order, so its knots follow the scaled-mm positions of the
     reference, and the offset depends on the stored spacing
     (`src/newimage/generalio.cc`, `src/warpfns/fnirt_file_writer.cpp`
     and `src/basisfield/fsl_splines.h` of FSL 5.0.9, at commit 513f6a0
     of https://github.com/neurodebian/fsl). Status: resolved, where the
     knots are placed from the scaled-mm positions
     (`io/transformations/fsl/fnirt/_base.py`). The reader now differs
     from fslpy 3.29.1 for such references, and the sources of FSL 6
     could not be reached to check that it still behaves as FSL 5.0.9.
221. FNIRT reads and writes its deformation fields and its coefficient
     files with one class, which the intent of the record selects. A field
     built from an array alone has no record and is written as a
     deformation field, because a record is never made up. The
     conformance harness therefore has a field `Exemplar.fixture`, which
     writes the coefficient file that the variant
     `FnirtWarpField-coefficients` reads, and check 8 of that variant
     checks a deformation field (`tests/test_io_conformance.py`). Status:
     open, for the maintainer to accept the field or to ask for a
     separate class.
222. No converter turns a coefficient file into a FNIRT deformation
     field, as `fnirtfileutils` does. Both kinds are the same class, so a
     converter between classes does not apply, and the deformation field
     would need a header made up from the reference image. Evaluating the
     chain on the grid of the reference gives the same field in the data
     model. Status: open.
223. FSL checks that the shape of the coefficients is the one that its
     spline gives for the shape of the reference stored in the qform, and
     it evaluates a coefficient file only on that reference. The reader
     checks neither, and places a reference of another shape or resolution
     by its scaled-mm positions, as fslpy does. Status: open.
250. An N5 pyramid, a group whose datasets are `s0`, `s1`, ..., is read
     as `N5MultiScaleImage`, whose levels are `N5Image` objects
     (`io/images/n5`). Each level follows this design: it holds the
     attributes of the group and of its datasets as its record and the
     lazy array of its dataset in `raw`, and it is an exemplar of the
     conformance tests. The pyramid holds the same metadata as its levels
     and has no `raw` of its own, and its `consolidate` consolidates its
     levels. The rules for the levels and the record of a pyramid are
     decided in the Zarr pass (item 15), so the pyramid is listed among
     the formats that are not migrated. Status: open.
251. N5 datasets are read and not written, as MINC (item 101). A dataset
     is a directory of files, while the atomic save of
     `io/base/_utils_files.py`, the save over the file that a proxy reads
     (#426) and the untouched copy that is identical byte for byte are
     defined for files. TensorStore, which brainhops would write with,
     writes every block at its full size, while the Java tools truncate
     the blocks at the upper edges, and it writes its own
     `attributes.json`, so an untouched copy would not keep its bytes. A
     writer would also encode a pyramid, whose rules wait for the Zarr
     pass (item 250). Status: decided for the reader, and open for a
     writer.
252. The conformance exemplar of `N5Image` names a dataset by its
     `attributes.json` file, because checks 6 and 12 read and copy the
     file that a case names, and `Exemplar.header_file` is the identity.
     An N5 dataset is sparse, so a copy without blocks would read as
     zeros and check 6 would not show that the metadata reads no block.
     The copy therefore holds a block that cannot be decoded. The checks
     write the datasets as the specification describes them, without
     TensorStore. Status: accepted.
253. The readable compressions are those that TensorStore 0.1.45
     decodes: `raw`, `gzip`, `bzip2`, `xz` and `blosc`. The list is fixed
     in `io/images/n5/_data.py`, so a dataset compressed with `lz4` or
     `zstd`, or described by the `compressionType` of version 0 of the
     specification, is refused when it is loaded, although its record is
     read. Status: open, until the list follows the version of
     TensorStore.
254. The specification of N5 defines no spatial attribute, and its tools
     disagree on some points (`io/images/n5/__init__.py`). brainhops
     reads the attributes of a dataset that is read on its own as
     n5-universe does, and completes a level of a pyramid from its group
     as Paintera and the deprecated layout of the n5-viewer do. It reads
     a bare `pixelResolution` without a unit, as n5-universe does, where
     the older n5-viewer assumed micrometres. It reads `resolution` and
     `offset` in the order of `dimensions`, although Paintera can reverse
     them for files whose tools wrote them in the reverse order. It does
     not infer the factors `2**k` of a level from its name `s<k>`, which
     n5-universe does. The axes are named `x`, `y`, `z`, `t` and `c` by
     position, since the `axes` attribute of n5-universe is not read.
     Status: decided, for the maintainer to confirm.
255. A dataset that is read on its own, and the same dataset read as a
     level of its pyramid, can have different transformations, because
     the attributes of the group apply only to the levels of a pyramid.
     Status: accepted.
256. A spatial attribute of N5 that breaks the conventions, such as a
     `pixelResolution` of the wrong length or a unit that is not a
     length, is ignored with a new public warning, `N5AttributeWarning`,
     when the transformations are decoded, instead of making the dataset
     unreadable, since the attributes are not part of the specification.
     Status: open, for the maintainer to choose between this and a
     refusal.
257. The lazy array of N5 translates NumPy indices for TensorStore,
     which reads integers and slice bounds as absolute positions, in its
     own helper `_basic_index` (`io/images/n5/_data.py`). Another reader
     backed by TensorStore needs the same translation, which should then
     move to a shared private module of `io/common`. Status: open.
258. Only local N5 containers are read. A pyramid is found by reading
     `s0`, `s1`, ... up to the first missing level, without listing the
     directory, so a level after a gap is not read, and the channel
     groups `c0`, `c1`, ... of the n5-viewer are read one at a time.
     Status: open.
280. A TrakEM2 affine (`io/transformations/trakem2/_xform.py`) is
     written only into the project that it was read from, by replacing
     the value of the `transform` attribute of its patch, as the
     BigDataViewer registration of #441 replaces one element. An affine
     without a record, such as one built from data or converted from
     another format, is refused by `consolidate` with
     `UnrepresentableTransformationError`, because a project cannot be
     made from an affine alone: a patch needs an image file, a layer and
     a project. The conformance harness has a new field,
     `Exemplar.template`, which writes a project by hand, and checks 8 and
     9 check the refusal for such a format. Status: open, for the
     maintainer to decide whether a format may refuse to write an object
     without a record.
281. A TrakEM2 patch is placed by a 2D affine in the plane of its layer,
     and the layer has a z and a thickness. The affine is read as a 2D
     `Affine` of shape `(2, 3)`, and the z, the thickness and the
     calibration stay in the record of the patch. A 3D placement would
     need a convention for the third axis of a patch, which has none:
     either a unit axis at the z of the layer, or an axis scaled by the
     thickness of the layer, and in either case whether z is the top or
     the middle of the section and whether it is calibrated by
     `pixelWidth`, as TrakEM2 does, or by `pixelDepth`. The datamodel has
     no embedding of a 2D transformation in 3D that would carry it.
     Status: open.
282. The coordinate transform of a TrakEM2 patch is not composed with
     its affine. TrakEM2 shifts the image between the two by the corner of
     the bounding box of a mesh of the image
     (`Patch.getFullCoordinateTransform`), which depends on the mesh
     resolution and on a truncation to integers, and brainhops does not
     compute it. The `input` of an affine whose patch has a coordinate
     transform is therefore `None`. Only an affine model and a similarity
     model written inside the patch are decoded, by
     `coordinate_transform()`, and lists, other models and transforms
     kept in separate `.ct` files are refused when decoded and kept in
     the file as they are. Status: accepted.
283. The TrakEM2 reader parses XML with expat to know the byte spans of
     its elements and attributes (`io/transformations/trakem2/_xml.py`),
     as the BigDataViewer reader of #441 does in
     `io/common/bdv/_xml.py`. The two helpers overlap, and are to be
     merged into one module under `io/common/` once both are on `main`.
     Status: open.
284. The similarity model of TrakEM2 reads its four values in single
     precision with Java's `Float.parseFloat`, and brainhops reads them
     with `numpy.float32`, which rounds the decimal text through a double
     and may differ from Java in the last bit for rare values. Status:
     accepted.
285. Check 6 of the conformance test is vacuous for TrakEM2, because the
     metadata holds the whole file, as for LTA and M3Z (item 66), and
     check 18 does not apply, because the format has no `raw`. Status:
     accepted.

The items from 290 on were found in pass 4a, which adds the helpers
of pyramids (`io/common/_pyramids.py`), `replace_directory`, the toy
pyramid of the conformance test and the checks P1 to P12.

290. A level that is consolidated or written alone narrows the record to
     its own level, at index 0, and keeps the rest of the record, such as
     the placement of its pyramid. A level of a pyramid of several levels
     is therefore never untouched in the sense of check 15, even when
     nothing was assigned to it, because its record describes more than
     the file that it writes. Status: decided by the maintainer on
     2026-10-13, and implemented for the toy level of the conformance test
     in pass 4a.
291. A remote store cannot be renamed, so a pyramid in a remote store is
     written in place, level by level, and the save is not atomic. A
     remote save over the store that the object reads is refused with
     `WriterError`. The conformance checks of atomic saves and of #426
     therefore run on local paths only. Status: decided by the maintainer
     on 2026-10-13, and to be implemented with Zarr in pass 4b.
292. tifffile encodes again everything that it writes, so an untouched
     TIFF file can be saved byte for byte only by copying the bytes of its
     source. A TIFF file whose dialect or levels changed is encoded again.
     Status: decided, to be implemented in pass 4c.
293. OpenSlide cannot write, so its pyramids and levels consolidate only
     their data. Its 22 classes are to leave the list of formats that are
     not migrated through the exemplars of the generic TIFF vendor, which
     share their bases with the other vendors. The maintainer has not
     answered this point, so the default stands. Status: open, for pass
     4d.
294. The toy pyramid of the conformance test is a directory that holds a
     JSON document and one toy file per level, and its level class is the
     toy level. They exercise the rules of directory stores without any
     optional dependency. The toy container stores its preferred placement
     only, so the branches of the checks for a container that composes the
     placements into its levels, for a writer that derives its levels, and
     for a format that cannot write are first run by TIFF, JPEG 2000 and
     OpenSlide. The generic loader reads a toy pyramid as a pyramid, whose
     sniffer is more confident, and reads it as a level when the option
     `level` is given, which the pyramid refuses. Status: accepted in pass
     4a.
295. A level changed in place, as in `pyramid.images[1].data = value`,
     lives in the cache of `images`, as a list of transformations changed
     in place does (item 191). A copy made with `replace` or
     `from_instance` within the format decodes the levels again, so it
     loses that change, while a write of the pyramid itself, including
     `io.save`, writes it. Assigning the list of levels keeps the change
     everywhere. Status: open, together with item 191.
296. `replace_directory` (`io/base/_utils_files.py`) cannot exchange two
     directories in one step. It moves the old directory aside and then
     the new one into place, so a reader that opens the target between
     the two moves finds no directory, but never a half-written one. The
     old directory is moved back when the second move fails. The new
     directory has a hidden name that ends like the name of the target,
     as the new file of `replace_file` has, so that a library that chooses
     an encoding from the suffix of a name chooses the same. Status:
     accepted in pass 4a.
297. The conformance harness has a new field, `Exemplar.container`, which
     says whether an object is stored in a file or in a directory. The
     metadata of a directory writes no bytes, so checks 7 and 10 compare
     its summary through `Exemplar.header`, and check 12 checks that the
     format refuses bytes. Every check that compares saves uses `_tree`,
     which reads a directory file by file. `PyramidExemplar` reads the
     metadata class, the suffix, the proxies, the edit of a record and the
     hooks `cut` and `header` from the exemplar of the level class, so it
     declares no `container`, `cut` or `storage` of its own. Image
     stacks (#431) added `Exemplar.folder` for a folder of numbered files
     that no extension names, which checks 5 and 12 treat apart, so the
     two fields are kept side by side, and checks 7 and 10 compare the
     summary of any metadata that writes no bytes of its own. Status:
     accepted in pass 4a.
298. Check P8 cannot tell which files of a directory belong to which level
     without a hook in the exemplar. It checks instead that the files
     that change when one level changes never change when another level
     changes. A save that writes again the files of every level with the
     same bytes passes, because only the bytes are compared. Status:
     accepted in pass 4a.
299. The pyramid formats merged before pass 4a log tensions that their
     move to the design of pyramids closes. Items 160 and 250 note that
     the precomputed and N5 pyramids have no `raw` of their own, items
     161 and 251 note that the rules of atomic and untouched saves were
     defined for files only, and items 181 and 208 note that the Imaris
     and BigDataViewer pyramids have no record. The JPEG 2000 pyramid
     already keeps the cache model of this design, with metadata that
     drops its decoded `images` and `transformations`, and its writer no
     longer refuses coarse data that was assigned, as the maintainer
     decided on 2026-10-13 for every format that derives its levels. The
     design of pyramids also answers items 130 and 131 of #433, because
     each record keeps the bytes of its documents, and a level
     transformation that is not diagonal is refused (#447). Status: open
     until those pyramids move to the design.
300. The seven plain matrix formats (`io/transformations/matrix`) share
     one record, `MatrixAffineRaw`, because a text container and a binary
     container differ only by the text that a text file keeps and the
     name of the array that `.npz` and `.mat` files select. Each container
     has its own registered metadata class, such as `TxtMatrixMetadata`,
     which reads the container with its array reader and builds the
     record, and each format class calls its own metadata explicitly.
     The record holds the stored matrix and the conventions that the
     caller stated (`ndim`, `vector`, `direction`, `index_base`, the
     voxel-to-world matrices of the `source` and `target` images, and the
     coordinate systems), which replace the former attributes
     `raw_matrix`, `variable`, `vector`, `direction` and `index_base`.
     `consolidate` stores an assigned matrix in the conventions and the
     layout of the record (`encode` in
     `io/transformations/matrix/_conventions.py`), and refuses a
     translation in a file that stores a linear map and a 1-based endpoint
     that is no longer a voxel space.
     Status: open, for review.
301. The plain matrix formats read files but do not write them, because
     the writer was ruled out of scope in #242, so the checks of writing
     make sure that they refuse to write, as for MINC (item 101), and an
     untouched save is not checked. The checks write their files with
     NumPy, SciPy and h5py. The data of these formats lives in their
     record, so a consolidation that cannot be written still encodes a
     new record, and the harness accepts new metadata in that case
     (`_check_data_only` in `tests/test_io_conformance.py`). Check 6 is
     vacuous, as in item 66, and check 10 edits `variable`. The text
     exemplar writes `.dat` files, and its variant `.1D` files, which
     `io.save` writes as a `register.dat` file and an AFNI affine (item
     306), as it writes a `.txt` file as a NiftyReg affine. For a future
     writer, rewriting the same
     content gives the same bytes with `np.save`, `np.savez`,
     `scipy.io.savemat` in format 4 and h5py, but not with
     `scipy.io.savemat` in format 5, whose header holds the time of
     writing, and a `.npz` or `.mat` file of several arrays would have to
     keep its other arrays. Status: open, for the maintainer to decide
     whether these formats should write.
302. The plain matrix formats give the coordinate systems of their record
     to the transformation as its own `input` and `output`, instead of
     decoding them as properties as LTA does, and a system counts as
     assigned only when it is not the very object of the record
     (`_overridden` in `io/transformations/matrix/_xform.py`). The data
     model copies the private `_input` and `_output` fields, so systems
     decoded from a record are lost when a transformation is copied into
     another class. LTA already loses them in this way:
     `LtaTransformation.to(Affine).input` is `None` for a file of the RAS
     type, on `main` as well. Status: open, for the maintainer to decide
     whether the data model should copy the decoded systems of another
     class, which would let the plain matrix formats decode them too.
303. `NiftyRegAffine` holds a `NiftyRegAffineRaw` record that keeps the
     text of the file, as `C3dAffineRaw` does in item 170, so an untouched
     file keeps its `%.7g` numbers, and a matrix that is encoded is
     written with the shortest digits that read back exactly. The record
     holds nothing beside the matrix and its text, and a changed matrix
     replaces the text, so check 10 is vacuous. The metadata of a
     NiftyReg affine loses a `.txt` file to the metadata of a plain
     matrix, as the transformation does, so check 6 selects it with the
     new field `Exemplar.hint`. `consolidate` refuses an `input` or an
     `output` that was assigned and is not RAS, which answers item 173
     for NiftyReg. Status: open, for review.
304. The four helpers that LTA and the Convert3D format repeated (item
     171) live in `io/transformations/base/_records.py` as `record_of`,
     `read_only`, `matrix_to_model` and `matrix_to_disk`, and LTA and
     NiftyReg call them, and so does `register.dat` (#432), which called
     the private helpers of LTA. The Convert3D format of #435 still
     repeats all four, unchanged, in `io/transformations/c3d/_affine.py`,
     and can switch to them in a later pull request. The MNI transform,
     BigDataViewer and TrakEM2 formats repeat `_record`, but their
     helpers `_matrix_to_model` and `_matrix_to_disk` store three rows,
     or a flat row of twelve or six numbers, rather than a homogeneous
     matrix, so they are not the same helpers. Status: open.
305. The readers refuse the options that they do not take with
     `TypeError`. A plain matrix file of a single array refuses
     `variable` and `key`, which it used to ignore, and `variable` and
     `key` that name different arrays are refused. A NiftyReg affine
     refuses `variable`, `key` and the conventions of a plain matrix. A
     text file that is read from a path or a stream receives `encoding`
     after it was decoded, so `encoding` is refused there, as for the
     Convert3D format, and honoured only when bytes are decoded. Status:
     open, for the maintainer to decide whether the adapters of
     `io/base/parsers.py` should open a text file with the given
     encoding.
306. The Convert3D format of #435 declares the `.mat` extension,
     `register.dat` (#432) the `.dat` extension and the AFNI affine the
     `.1D` extension, and each converts an affine whose systems are
     unknown, so `io.save` of a plain matrix built from data writes a
     file of that format instead of raising `WriterError`. The checks of
     writing of a format that cannot write therefore accept a save by
     another format, and check only that the format is not a writer and
     that the file does not read back as the format
     (`_refuses_writing` in `tests/test_io_conformance.py`). A
     Convert3D matrix holds no marker, so the metadata of a NiftyReg
     affine, which scores a text file of a homogeneous matrix from its
     content, wins over the metadata of Convert3D, which a `.mat` name
     alone selects, and check 6 selects the Convert3D metadata with
     `Exemplar.hint`, as it does for NiftyReg (item 303). Status: open,
     for the maintainer to decide whether a `.mat` text file should
     dispatch to the Convert3D metadata without a hint.
310. The data of a FLIRT transform is its FLIRT matrix, `flirt_matrix`,
     which lives in the record and which an assigned matrix overrides
     until it is consolidated. The affine of the data model maps the RAS
     space of the reference to that of the moving image, and it is
     computed from the FLIRT matrix and the two images, so it cannot be
     assigned, and the fields `_data` and `_matrix` of `Affine` are still
     forgotten (`io/transformations/fsl/flirt/_xform.py`). Making the
     affine the pending data would need both images to encode a record,
     and a FLIRT matrix could then not be built without them. The images
     are options of the format, which are never stored, as the
     geometries of a `register.dat` file are. The conformance harness
     reads the pending state of FLIRT through `Exemplar.pending`. Status:
     open, for the maintainer to decide whether the affine should be the
     pending data.
311. FLIRT, ITK text and ITK HDF5 files are not written by brainhops,
     but their records are plain structures, so `consolidate()` encodes a
     pending matrix or chain into a new record in memory, unlike MINC,
     which consolidates only its data (item 101). The conformance harness
     marks them with `Exemplar.records`, and runs check 16 without a
     file. Checks 5, 8, 9 and 10 check that they refuse to write, and
     check 6 is vacuous for the files that are read in one pass (FLIRT,
     ITK text and MATLAB), as for LTA (item 66). The records of the text
     formats do not keep the text of their files, because nothing writes
     it. The records of FLIRT and ITK text files hold nothing but their
     data, so an assigned value replaces an edit of the record, and
     check 20 does not check that the edit is kept under it for the
     exemplars marked with `Exemplar.bare`. Status: open, for the maintainer to decide whether these
     formats should write, which would make the untouched round trips
     byte for byte (with the exemption of item 48 for HDF5).
312. The blocks of an ITK record are `ItkStruct` objects
     (`io/transformations/itk/_common.py`), which are also
     transformations, and the chain that an ITK format
     decodes holds the blocks of its record, without a copy. Their
     parameters are read-only arrays, or lazy arrays for the warps of an
     HDF5 file, but a field of a block can still be assigned, which would
     change the record in place. An ITK format is a
     `MutableSequence`, whose chain could be edited in place and was then
     written, so its chain follows the rule of the images (item 190): it
     is decoded on first access and cached apart from an assigned chain,
     `consolidate` reads it through `held_transformations`, and it is
     compared there with a fresh decode of the record, block by block
     and by identity (`io/transformations/itk/_chains.py`). Assigning
     other metadata or another `position` drops the decoded chain, with
     a change made to it in place (item 191), and keeps an assigned
     chain. The other chain formats keep their chain pending by
     assignment (item 114). Status: open.
313. An untouched ITK MATLAB file is written byte for byte because the
     record keeps the bytes of the file (`MatRaw.content`), which are
     written while they still decode to the blocks and the byte order of
     the record. Encoding the blocks again gives the same bytes for the
     files that SimpleITK writes, but not for a file whose variables are
     marked as stored row by row, whose fixed parameters have another name
     or are in single precision, as `itk::MatlabTransformIOTemplate`
     writes them for a transform in single precision
     (`itkMatlabTransformIO.cxx` and `vnl_matlab_write.cxx`, ITK v5.4.0).
     The option `position` only selects what is read, so a file with
     several blocks is written whole, as for X5 (item 49). A chain that is
     encoded again keeps the byte order of the record, and the options
     `byteorder` and `precision` encode every block of the record again
     when they differ from it. Status: open.
314. The move of FLIRT and ITK removes the names `FlirtMatrixReader`,
     `TfmTransformReader`, `MatTransformReaderWriter` and
     `H5TransformReader`, and the attributes `header` and `file` of
     `H5Transform`, in favor of `metadata.raw`. The option `keep_open` of
     `H5Transform` raises `TypeError`, and its warps are read lazily by
     default (`load=False`). The images of a FLIRT transform are
     keyword-only, and its FLIRT matrix is read from the record. A FLIRT
     transform copied within its format no longer needs its images, and
     assigning an image drops the views derived from the old one. An
     untouched ITK MATLAB file read with `position=` is written whole,
     and a chain that is encoded again keeps the byte order of its file.
     `MetadataFormat.load` reads a `.mat` file of sixteen numbers as
     `FlirtMetadata` unless the hint `c3d` is given, as the dispatcher of
     the transformations reads it as a FLIRT matrix (item 172), so check
     6 dispatches the metadata of the Convert3D exemplar with the hint of
     `Exemplar.hint` (item 306). These changes are to be announced with
     item 13. Status: open, as for item 13.
320. An elastix parameter file may name an initial transform in another
     file, and that file may name another. The record of each file of the
     chain is read when the file that names it is read, and it is held in
     the record of that file, in `ElastixRaw.initial`
     (`io/transformations/elastix/_raw.py`). The record of the last file
     therefore holds the whole chain, and so does its metadata. The files
     are small text files, and reading them at once keeps the error of a
     missing or looping file where the file is read, as before. The name in
     the parameter map is kept as the file stores it. Status: proposed in
     the elastix pass.
321. A transformation whose chain was not assigned is written to a path
     with the files of its chain. Each file whose name is relative is
     written beside the file that names it, under that name, unless it
     already holds the same bytes, and a file that holds other bytes is
     never replaced: the writer raises `WriterError`, and so it does when a
     file of the chain would be written over the new file
     (`_place_initial` in `io/transformations/elastix/_xform.py`). A name
     that is absolute is left as it is. The MINC transform files of #434
     place their grid volumes in the same way, but replace a file that
     holds other bytes. elastix looks for a relative name in the working
     directory first, so a file of the same name there would be read
     instead of the placed one. Status: open, for the maintainer to choose
     one rule for both formats.
322. The reader option `initial` of elastix selects what is read and
     decoded. `initial=False` reads no file of the chain, and a file name
     reads that file in place of the one that the map names. A
     transformation given as `initial` is held in the private field
     `_initial`, which copies keep, and it is not written, as for the X5
     options of item 49. A consolidation that encodes an assigned chain
     resets it, since the new record has no initial transform. Status:
     proposed in the elastix pass.
323. The elastix record keeps the text that it was read from and writes
     it back while it still parses to the parameter map, as AFNI does
     (item 84). A file is read as bytes and decoded, so that its line
     endings and comments survive. A record read from lines keeps no text,
     because the line endings are not known, and a stream that the caller
     opened in text mode has already translated them. Status: proposed in
     the elastix pass.
324. One metadata class, `ElastixMetadata`, reads both syntaxes of
     elastix. The syntax is a field of the record, which a file name gives
     as elastix does (`.toml` or not), and which the content gives when
     there is no name. Each format class passes its syntax explicitly to
     the reader and to `consolidate`. A record in the other syntax, as
     after a conversion between the two formats, is copied into a record
     of the syntax of the class at consolidation, without its text, and
     the converters between the two formats keep the metadata of an
     untouched transformation. Status: proposed in the elastix pass.
325. The transform parameters of an elastix record are a read-only array,
     and the blocks decoded from the record hold read-only views of it
     (`_read_only` in `io/transformations/elastix/_blocks.py`), so an edit
     in place cannot change the record. SciPy refuses a read-only array, so
     `AffineLogTransform` copies the matrix before its exponential. Status:
     proposed in the elastix pass.
326. The elastix pass removes the `parameter_map` field in favor of
     `metadata.raw.parameters`, the `reverse=False` of the base class, the
     class methods `_read_map` and `_format_map`, and the module function
     `initial_filename` of the private `_xform` module, which moves to
     `_raw`. The writers no longer accept unknown options, and an
     untouched file keeps its bytes. These changes are to be announced
     with item 13. Status: open, as for item 13.
327. The block of an elastix transformation is a view of its record, and
     not of its chain, so assigning a chain does not clear it, and
     `xform.block` still describes the file while the chain is pending. In
     the conformance tests, check 6 is vacuous, as for LTA and M3Z (item
     66), and checks 18 and 19 do not apply, since the format has no `raw`
     and is not an image. The chains of several files are checked in
     `tests/test_io_elastix.py`, because the harness builds objects from
     data alone. Status: proposed in the elastix pass.
328. The elastix formats import private names of other packages: the ITK
     block classes and `_make_system` from `io/transformations/itk`, and
     `_compatible` from `io/transformations/base/affines.py`, which checks
     an assigned system against LPS. ITK is moved to records at the same
     time, so these imports may need to follow it. Status: open.
329. An elastix transformation without a record and without a chain
     raises `UnrepresentableTransformationError` when it is consolidated,
     as it did when it was written, and `ElastixMetadata` without a record
     raises `WriterError`, as M3Z does (item 61). LTA and X5 write an empty
     record instead. Status: proposed in the elastix pass.
