# Format records and their layering

This memo is agreed and is being implemented. It is tracked in #415,
and the work is done in the passes described at the end. The first
pass, which moves the NIfTI formats, is written and under review.

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
`raw` once and caches the result. Setting `data` stores
`to_disk(value)` in `raw` and clears the cache. This is the rule of
#302, which keeps a single stored array and decodes views from it.

Only exact and invertible changes may happen between `raw` and `data`.
These are permuting axes, removing or adding singleton axes (such as
the time axis of a NIfTI displacement field), reshaping and changing the
memory order. Intensity scaling is not one of them, because the proxy
and the writer options already handle it.

A small amount of data may also live in the header record, for example
the matrix of an affine, or the content of a file that is cheaper to
parse in one pass. On write, the data of the model always takes
precedence over the copy in the record.

### Precedence on write

Writing follows three steps.

1. The base record comes from `metadata.to_raw()`. Only the metadata
   fields that differ from a fresh decode of the record are encoded
   over it.
2. The geometry is encoded from the model over that record. The image
   or the transformation owns its geometry and its data, the metadata
   owns its vocabulary, and the record supplies everything else.
3. The header is written, then the data.

If `raw` is still the proxy that the reader built, the data is copied
from the source file without being decoded, so an object that was read
and not changed is saved byte for byte. Otherwise the array in `raw` is
written.

### Rules

A raw record is never changed in place. Encoding copies the record
first, so a record can be shared between objects safely.

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
`from_struct` and `to_struct` become `from_raw` and `to_raw`. It also
keeps the vocabulary of common metadata fields and the `to` conversion.

The vocabulary itself arrives in the fifth pass, after the formats have
moved to records. That pass takes the vocabulary from #310 and #287 and
adds it to `Metadata`. Each format gets module-level functions that
decode and encode the vocabulary, and `to_raw` encodes only the fields
that differ from a fresh decode. The same pass adds the BIDS codec,
puts metadata on the roots of the data model, propagates metadata
through operations, and updates section 6.1 of the memo.

## Implementation passes

The work is done in five passes. Each pass is reviewed before the next
one starts.

The first pass adds the core classes and moves NIfTI. It is made of
three stacked pull requests. The first adds `Metadata` and the
`MetadataFormat` dispatcher, whose `from_raw` builds an object from a
record and whose `to_raw` returns a copy of the record. It also adds a
conformance test, which checks every migrated format against the rules
of this memo and lists the formats that are not migrated yet. The
second adds `NiftiRaw`, `NiftiMetadata` and the views of NIfTI data,
and moves `NiftiImage` to them. The third moves the NIfTI
transformations, including the SPM, NiftyReg, ITK and FNIRT fields, and
deletes `NiftiReaderWriter`, `_explicit_matrix`,
`_store_through_the_parser` and `_WritableNifti`. It also renames the
`header=` option of the NIfTI converters to `metadata=`.

The conformance test runs twelve checks on every migrated format.

1. The constructor fields of the format are the fields of its model,
   `raw`, `metadata` and the options the format declares. None of them
   is named `image`, `header`, `struct`, `_header`, `dataobj` or
   `node`.
2. Loading is lazy. After a load, `raw` is a proxy and no decoded data
   is cached yet.
3. Setting `data` stores `to_disk(value)` in `raw`, and clears the
   cached data and the views derived from it.
4. `to_model(to_disk(x))` equals `x`.
5. An object that is loaded and saved without changes gives the same
   bytes, after decompression.
6. `<X>Metadata.load` never reads the data, which is tested on a file
   truncated after its header.
7. The metadata can be pickled.
8. An object built from `data=` alone can be saved.
9. `replace` and `from_instance` keep `raw` and `metadata` within a
   format, and reset both when copying from another format.
10. An edit of `metadata.raw` survives a save, and a changed geometry
    takes precedence over the stale header.
11. Each format package exports exactly its intended public names, no
    name that starts with an underscore, and only private submodules.
12. The adapter contract holds. A format whose bytes are read through a
    stream really overrides `from_fileobj`, a binary format reads in
    `"rb"` mode, and a concrete format is not a dispatcher.

The second pass moves LTA first, so that the reference format matches
this design. `LtaStruct` becomes `LtaRaw`, its field becomes a
keyword-only `raw`, and `reverse=False` is dropped. The pass then moves
X5, whose record holds the header and the nodes while `chain` and
`position` stay options of the format, and M3Z, whose arrays stay in
the record because the file is parsed in one pass.

The third pass moves MGH, AFNI, MRtrix, NRRD and MINC. MGH gets a
record with its header and its tags, and uses nibabel's proxy. For the
others, the private `_header` becomes the record, `dataobj` becomes
`raw`, and the existing transpositions become `to_model` and `to_disk`.
The intensity scaling of MRtrix stays out of the view.

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
   `data=` are given, `data=` wins. Status: open.
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
   `transformations`, become stale when `metadata` is assigned again.
   The fix is a private `_metadata` field behind a public `metadata`
   `smartproperty` with `invalidates=("transformations",)`, because
   bagof does not allow a field and a property with the same name in
   one class. Status: resolved in pass 1c, where every NIfTI
   transformation class applies this fix.
9. A proxy built over a stream that the caller passed in must keep that
   stream alive for as long as the proxy is used. For the same reason,
   a format whose `raw` is a proxy must override `from_filename` or
   `from_file`, because the adapter's `from_filename` closes the stream
   when it returns. Status: open, and the second part is a rule for
   every pass.
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
    a velocity unless it is saved with `log=False`. Status: open.
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
    Status: resolved in pass 1a, where `from_instance` keeps `raw` and
    leaves `data` unset within a format, so that copies stay lazy and
    an untouched object is still saved byte for byte.
18. `replace` reads `data` through the property, so a copy made with
    `replace` turns the proxy into an array and is no longer saved byte
    for byte. The setters of the NIfTI transformations empty the
    private `_data` of the model
    (`io/transformations/nifti/_affines.py:55` and `_fields.py:87`), so
    that `replace` passes `data=None` and the copy keeps `raw`. Status: resolved in pass 1c for the
    transformations, and open for `NiftiImage`, which has no `_data`.
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
    in the third pass. Status: open.
24. An untouched save is identical byte for byte only when the geometry
    is encoded back exactly. A `qform` or `sform` whose code is 0 used
    to be written back with code 2, and the padding before `vox_offset`
    was lost. Status: decided in pass 1c for the fields, which treat the
    geometry as unchanged while `_transformations` is unset and then
    write `raw` under the record with its forms kept, and open for the
    affines, which write the `sform` and the `qform` under the single
    code that `_xform_code` chooses
    (`io/transformations/nifti/_affines.py:185`).
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
31. `NiftiImage.transformations` is cached and returned as the same
    object each time, so appending to the list in place persists. It
    used to return a copy. Status: decided in pass 1b.
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
    the class list the model last. Status: open, and older than pass
    1c.
37. In the conformance tests, check 9 does not convert NiftyReg, ITK
    and FNIRT fields to another format, because they have no converter
    until #312, and it compares `raw` and `data` only for models that
    have `data`. Status: accepted.
38. Private helpers such as `_always`, `_record_image`,
    `_set_field_data` and `_set_coordinates_intent` were imported by
    other packages from `io/transformations/nifti`. Status: resolved in
    pass 1c, where they move to `io/common/nifti/_header.py` and
    `_core/properties.py`.
