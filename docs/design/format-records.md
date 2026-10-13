# Format records and their layering

This memo is agreed and is being implemented. It is tracked in #415,
and the work is done in the passes described at the end. The first
two passes are written and under review, and the third is in progress.
The design of pending data and `consolidate()` is applied across the
whole stack.

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
NIfTI, MGH, AFNI, MRtrix, NRRD and MINC. MINC cannot be written, so it
keeps the list that it holds.

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

The second pass moves LTA first, so that the reference format matches
this design. `LtaStruct` becomes `LtaRaw`, its field becomes a
keyword-only `raw`, and `reverse=False` is dropped. The pass then moves
X5, whose record holds the header and the nodes while `chain` and
`position` stay options of the format, and M3Z, whose arrays stay in
the record because the file is parsed in one pass.

The move of LTA, called pass 2a, is pull request #421. The move of X5,
called pass 2b, is pull request #422. The move of M3Z, called pass 2c,
is under review. ITK-H5 is deferred. It shares
`ItkStruct` with the ITK text and MAT formats and has no writer, so it
moves with them.

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

The fourth pass moves Zarr, with one record per resolution level, and
FLIRT. The class variable workaround of FLIRT is removed once it is
decided whether `moving` and `reference` belong to the format or to the
record.

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
    is set in the fourth pass. Status: open.
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
    have no record yet. They are decided in the Zarr pass. Status:
    open.
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
47. The shared HDF5 adapter still reads the `keep_open` option
    (`io/common/hdf5/_parsers.py`), because ITK-H5 is not migrated yet.
    The `TypeError` that X5 raises for `keep_open=` comes from its
    constructor. Status: accepted until ITK-H5 is migrated.
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
    Status: accepted in pass 2c.
65. `M3zMorph.from_instance` given a sequence of transformations
    relabels any chain, and the writer refuses it later. X5 behaves in
    the same way (item 51), and the problem is older than this work.
    Status: open.
66. Check 6 of the conformance test is vacuous for the formats that are
    parsed in one pass, which are LTA and M3Z, because the truncated
    copy is the whole file. Status: accepted in pass 2c.
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
     the formats merged later (N5, Imaris, precomputed and JPEG 2000)
     follow the same rule, and the JPEG 2000 pyramid caches the levels
     and the placements that it decodes in the same way. Status:
     resolved.
191. A change made in place to a decoded list of transformations lives in
     the cache, so it is lost when other metadata is assigned, and a copy
     made with `replace` or `from_instance` decodes the list again instead
     of copying the change. A write of the image itself, including
     `io.save`, still writes the change. Assigning the changed list keeps
     it everywhere. The images that cannot be written (N5, Imaris and
     precomputed) have no record to encode the change into, so their
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
