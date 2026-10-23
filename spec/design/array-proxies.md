# Lazy arrays of file data

This memo decides how brainhops reads the voxels of a file only when
they are needed, and how the classes that do so share their code. It
also explains how the same design lets brainhops read NIfTI, MGH and
MINC files without nibabel.

The first two layers below are implemented in
`brainhops._core.proxy`, and the fifteen proxies of brainhops
derive from the base class. A first form of the third layer is
implemented in `brainhops._core.strided` (see "What is implemented of
the third layer" below), and the MRtrix, NRRD and AFNI proxies use it.
The full third layer, the fourth layer, and the reading of NIfTI, MGH
and MINC without nibabel, are not implemented yet. The steps are listed
at the end of the memo.

## What a proxy is for

When brainhops reads an image or a transformation field, it parses the
header at once but does not read the voxels. It keeps instead a small
object, called a proxy, that knows where the voxels lie and how they are
stored. The proxy behaves like a read-only array: it has a `shape` and a
`dtype`, and NumPy converts it with `np.asarray`. The proxy is stored as
the `rawdata` of the format object (see [Format
records](format-records.md)). The `data` property turns it into a NumPy
array or a Dask array, depending on the array backend.

A proxy keeps no file open between reads. It opens the file each time
its values are needed, so that it can be pickled, sent to Dask workers
and read from several threads, and so that the file can be written over
while the format object exists. When the caller passed a stream rather
than a path, the proxy keeps the stream and its starting position, and
the caller must keep the stream open.

## What the code had before the base class

brainhops defines fifteen proxy classes, and NIfTI and MGH use the
`ArrayProxy` of nibabel. They fall in two groups.

The first group holds formats whose voxels are an array of fixed-size
values stored at an offset of a file that may be compressed. The
proxies of this group decode the bytes themselves or ask nibabel to.

| Format    | Class                  | Partial reads | Scaling         |
|-----------|------------------------|---------------|-----------------|
| MRtrix    | `_MrtrixProxy`         | yes, except `Bit` | yes, global |
| NRRD      | `_NrrdProxy`           | yes, for raw, gzip and bzip2 in one file | no |
| AFNI      | `_BrikProxy`           | yes, for sub-bricks of one type | yes, per volume |
| MINC      | `_MincProxy` (nibabel) | yes, for a local file that is not compressed | yes, per slice |
| NIfTI     | nibabel `ArrayProxy`   | yes           | yes, global     |
| MGH       | nibabel `ArrayProxy`   | yes           | no              |

The second group wraps a library that already reads chunks or tiles:
Zarr (`_ZarrProxy`), N5 (`_N5Proxy`), Neuroglancer precomputed
(`_PrecomputedProxy`), HDF5 (`DelayedH5Array`, `_BdvProxy`,
`_ImarisProxy`), OpenSlide (`_SlideArray`), TIFF (`_TiffProxy`), image
stacks (`_StackProxy`), JPEG 2000 (`_Jpeg2000Proxy`) and Pillow
(`_PillowProxy`). Most of these proxies read only the part of the file
that an index selects. TIFF does so only when it memory-maps the file,
and JPEG 2000 and Pillow always decode the whole image.

None of these classes shared a base class or a protocol, and each one
is recognized by `isinstance` checks against its own class. They
repeated the same code:

- the `shape`, `ndim` and `dtype` properties, read from the record;
- an `__array__` method that ignores its `copy` argument;
- a shortcut, copied in thirteen classes, that answers an empty
  selection without reading the file, because Dask asks for one to
  learn the type of an array;
- four variants of a `_basic_index` function, which turns an index made
  of integers, slices and an ellipsis into explicit bounds;
- a context manager that opens a path or reuses a stream, and a test
  of whether the file is compressed;
- a method that returns the bytes of the file as they are stored, so
  that a writer can copy them without decoding them (`read_stored`,
  `read_bytes`, `copy_to`);
- a function that tells whether a proxy reads a given file, so that a
  writer does not empty a file before the proxy has read it
  (`reads_brik`, `reads_file`, `reads_from`, `reads_store`).

Each of the format views (`_views.py`) also turned the proxy into an
array in its own way: some call `asarray` of the backend, some call
`da.from_array` with the chunks of the proxy, and others call a method
such as `to_dask`, `view` or `read`.

## Problems

The proxies of the first group that brainhops wrote decode the whole
file each time their values are read, and then select the values in
memory. Dask
splits an array into chunks and reads the proxy once for each chunk, so
the file is read once per chunk. The base class does not change this,
because the proxies of the first group still implement `_read` by
decoding the whole file, and the third layer is what fixes it. A float32
image of shape `(256, 256, 256, 4)`, which takes 256 MiB, is split into
eight chunks by the default settings of Dask 2026.8, so it is
decompressed eight times when it is stored with gzip. A local file that
is not compressed is memory-mapped, which makes the repeated read cheap,
but the MRtrix and AFNI proxies then apply the intensity scaling to the
whole volume before they select a chunk, which costs as much as reading
it. Only the nibabel proxy and the proxies of the second group avoid the
problem.

A writer recognizes a proxy that it can copy without decoding by
checking for its own class. A NIfTI image whose `rawdata` is an MRtrix
proxy is therefore always decoded and encoded again, even when the two
files store the same bytes.

There are two models of the same thing in the first group: the
`ArrayProxy` of nibabel for NIfTI and MGH, and hand-written proxies for
the other formats. The second model does not read partially, and the
first one cannot be used without nibabel.

## Decision

The design has four layers. A consumer depends only on the first one,
and each format uses the highest layer that fits its storage.

### First layer: a protocol

`ArrayProxyProtocol` is a runtime-checkable `typing.Protocol` that
states what a consumer may rely on. A NumPy array, a Dask array and a
nibabel `ArrayProxy` satisfy it, which keeps the transition gradual.

```python
import numpy as np
import typing_extensions as tx


@tx.runtime_checkable
class ArrayProxyProtocol(tx.Protocol):
    @property
    def shape(self) -> tx.Tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def ndim(self) -> int: ...

    def __getitem__(self, index: tx.Any) -> "ArrayProxyProtocol": ...

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray: ...
```

A proxy is always lazy. Indexing it, as in `proxy[index]`, returns
another proxy that reads only the selected values, and only
`np.asarray(proxy)` reads the file. Indexing a NumPy array or a nibabel
`ArrayProxy` returns a NumPy array instead, which also satisfies the
protocol, so a consumer must call `np.asarray` on the result of an index
whenever it needs the values. Dask accepts lazy indexing: the function
that reads a chunk calls `np.asarray` on whatever indexing returns, and
each chunk then reads only its own part of the file.

A proxy indexes lazily with integers, slices of any step, the ellipsis,
`None`, and an array of integers along a single axis. Several integer
arrays, and boolean masks, select points rather than a box, and a
proxy cannot describe that selection without reading. Indexing with
them raises an `IndexError` that tells the caller to write
`np.asarray(proxy)[index]`, so that the read is explicit.

### Second layer: a base class and three kinds of proxy

`ArrayProxy` is an abstract class that implements everything the
proxies repeated: `ndim`, `size`, `nbytes`, `len()`, `T`, and the
conversions `to_numpy`, `to_cupy`, `to_backend` and `to_dask`. Its
subclasses fall in three kinds, and each kind does one thing.

- A `StoredProxy` presents an array as its storage holds it. The proxy
  of each file format is one. A subclass defines `shape`, `dtype`, and
  one method, `_read(box)`, which reads a box of the stored array. The
  box has one entry for each stored axis: an integer, or a slice whose
  start, stop and step are integers with a positive step. A subclass
  may also define `chunks`, the block shape that the storage prefers,
  such as the chunks of a Zarr array or one slice of a stack.
- An `IndexedProxy` presents a part of a stored proxy. Indexing a
  stored proxy returns one, and it reads nothing, so the shortcut for
  an empty selection is no longer needed. It holds the stored proxy as
  `base` and the index as `selection`, a NumPy index normalized to one
  entry for each stored axis: an integer, which drops the axis, a
  slice, or an array of integers. A `NewAxis` entry stands for each
  axis that `None` added, with the positions of its single value that
  a later index keeps, removes or repeats.
- A `ScaledProxy` presents the values of another proxy multiplied by a
  factor and shifted by an offset, which broadcast against the shape,
  as in `slope * stored + inter`. It covers the single factor of
  MRtrix and NIfTI, and the factor of each AFNI sub-brick.

A fourth class, `PermutedProxy`, is a stored proxy that presents the
axes of another stored proxy in another order. Transposing a stored
proxy returns one, and transposing it again composes the two
permutations, so permuted proxies never nest.

The kinds are kept in a fixed order, which every operation preserves:
a scaled proxy holds an indexed proxy or a stored one, and an indexed
proxy holds a stored proxy.

- Indexing an indexed proxy composes the new index with its selection,
  so a chain of indexes gives what NumPy gives for the same chain, and
  the result holds the same stored proxy. An index that selects every
  value in order returns the stored proxy itself.
- Transposing an indexed proxy moves the entries of its selection with
  their axes and transposes the stored proxy, so that the selection
  stays in the order of the stored axes.
- Indexing or transposing a scaled proxy does the same to its inner
  proxy and to the factors. An axis of length one of a factor stays of
  length one, so a factor never grows to the shape of the values, and
  only the values that are read are scaled.
- NumPy moves the axis of an array index to the front of the result
  when the array and the integers of the index are not next to each
  other in the index as written, and every proxy does the same.

Reading follows from this order. An indexed proxy calls the `_read`
method of its stored proxy with the smallest box with a positive step
that holds the selected values, and then applies the negative steps,
the arrays of positions and the new axes in memory, so a stored proxy
never sees them. A scaled proxy reads its inner proxy and scales the
values. `__array__` honors `dtype`, and it raises a `ValueError` when
`copy=False` is asked for, since the values cannot be read without
making them.

The order also answers the two questions that writers ask.

- A writer that copies the stored bytes of a proxy as they are checks
  the class of the proxy: an instance of the proxy of its format is the
  whole stored array, since an index or a transpose of it is an
  indexed or a permuted proxy. The MRtrix and AFNI writers also accept
  the scaled proxy that their reader builds, when its factors are those
  of the header (`whole_file` and `whole_brik`). The test
  `tests/test_io_proxy_views.py` writes a flipped index of a file in
  each of these formats.
- A writer that must not empty a file that a proxy still reads asks
  `stored_proxy(array)`, which returns the stored proxy below any
  scaling, index or permutation, and compares its file with the target.

Every indexed proxy reads through the stored proxy that a reader built,
so a handle that the stored proxy opens on its first read, such as an
open store, is opened once and shared by every index of it.
`to_dask()` calls `da.from_array` with the `chunks` of the proxy, which
an indexed proxy carries through its selection, with `asarray=True`,
with `fancy=False`, and with a `meta` argument, so that Dask does not
index the proxy to learn the type.

Status: implemented, in `brainhops._core.proxy`.

Two members are not implemented yet. `reads(file)` will tell whether
the proxy reads a given file, and replace the functions that do so
today (`reads_brik`, `reads_from`, `reads_store` and others). `stored`
will return a description of the bytes that a writer may copy as they
are, or `None` for an indexed proxy, and replace the `isinstance`
checks of the writers.

The format views only index and transpose the proxy, so the `dataobj`
of an object read from a file is a proxy in the layout of its `data`,
and `data` reads it with the backend setting (see
[Array backends](array-backends.md)). The NIfTI and MGH views wrap the
proxy of nibabel in `brainhops.io.base._utils_proxies.NibabelProxy`.
The voxels of these files are stored contiguously, so that proxy
prefers blocks of whole slices along the last spatial axis, with as
many slices as fit in the block size of Dask (the option
`array.chunk-size` of its configuration). This is implemented.
The views of TIFF and OpenSlide keep their own functions, because
tifffile can give a memory map or a Dask array itself, and an
OpenSlide level is kept as a Dask array when it is read lazily.

### Third layer: strided data behind a file

This layer will live in `brainhops._core`, next to the base class.
It knows nothing about brain images, and it is made of three parts.

A `ByteSource` says how to open a fresh stream of the decoded bytes. It
holds a path, a URL that fsspec can open, a stream with its starting
position, or bytes in memory, and it holds a codec. Each read opens the
source again and wires the decompressor: gzip through indexed_gzip when
it is installed, so that seeking is cheap, and bzip2, xz or zstd
otherwise. A source also tells whether it is a local file that is not
compressed, in which case the data can be memory-mapped, and it
implements `reads(file)`. Text encodings, such as the ASCII and hex
encodings of NRRD, and packed bits, such as the `Bit` type of MRtrix,
are codecs too. Such a codec cannot seek, so it decodes the whole file
into memory, and the rest of the layer reads from those bytes as from
any other stream.

A `Layout` describes where each value lies in the decoded stream: an
offset in bytes, a shape, a dtype with its byte order, and one stride in
bytes for each axis. Strides may be negative. A layout covers C order,
Fortran order, the axis permutation and flips of an MRtrix `layout` and
the axis order of NRRD, without copying anything. Indexing or
transposing a layout returns another layout, computed by arithmetic
alone, so `StridedProxy` composes an index by changing its layout. A
layout also turns a selection into the list of byte ranges to read.
Ranges that are close are merged, so that one large read replaces many
small ones. This replaces `nibabel.fileslice`, and the functions that
write through a proxy will be built on it (#479).

`StridedProxy` is the `ArrayProxy` that pairs a source with a layout.
Its `_read` memory-maps a local file that is not compressed and returns
a view of it. For any other source, it reads the merged ranges. With a
codec that cannot seek cheaply, such as bzip2, it reads the stream once
from the first range to the last one and keeps only what the ranges
select. Its `stored` description is the source and the byte range of
the data, which a writer copies when the target stores the same layout.

A `StridedProxy` takes a `cache` option, which is false by default.
The cache holds only what the proxy computes to decode a compressed or
encoded file, and never the values of a file that can be read in
place. A local file that is not compressed is memory-mapped, and the
operating system already caches its pages.

- For a codec that cannot seek, such as bzip2, xz, the text encodings
  of NRRD or the packed bits of MRtrix, the cache holds the decoded
  bytes, so that Dask does not decode the file once for each chunk.
- For gzip read through indexed_gzip, the cache holds the index of
  seek points that indexed_gzip builds, which is small. Without it,
  each read builds the index again, which costs a decompression of the
  file up to the values read.

Because only compressed and encoded files are cached, and such files
cannot be written in part (#479), a write through a proxy never makes
a cache stale. A file written again by other code is detected by its
size and its modification time, which the proxy checks before it uses
the cache. The cache belongs to the source, so a proxy derived from
another one by indexing shares the cache of its parent. The cache is
dropped when the proxy is pickled, so each Dask worker process fills
its own, and threads share one cache through a lock. The option breaks
the rule that a proxy keeps nothing between reads, which is why it is
off by default.

#### Remote sources

A remote file, for example on S3, reaches a reader as a path object of
bagof-paths whose `open` method goes through fsspec. A `ByteSource`
keeps that path and opens it for each read, as it does a local path.
The stream that fsspec returns can seek, and each read after a seek
becomes an HTTP request for a range of bytes, so `StridedProxy` reads
the merged ranges of a selection without downloading the whole file. A
remote file cannot be memory-mapped. A compressed remote file is read
through the same decompressors, and gzip benefits most from the cached
index, because each decompression downloads the file again.

The proxy does not cache remote bytes itself. fsspec already provides
caches, either on the stream (the `cache_type` option) or as a link of
a chain such as `simplecache::s3://bucket/image.nii`, which keeps a
local copy of the file. A `ByteSource` checks that a remote file has
not changed through the entity tag or the modification time that
`fs.info` returns, when the file system gives one.

Credentials should come from the environment or the configuration of
the file system, such as an AWS profile. Options passed to the path,
such as a key and a secret, are pickled with the proxy and sent to
every Dask worker.

Two generic proxies complete the layer. `ScaledProxy`, described with
the second layer, applies a slope and an intercept that broadcast
against the shape, which covers the single factor of NIfTI and MRtrix,
the factor of each AFNI sub-brick and the slice-wise scaling of MINC.
Because it scales only what it reads, a chunk costs only its own
values. It is implemented, and MRtrix and AFNI use it.
`ConcatenatedProxy` joins proxies along one axis. It covers NRRD data
split into several files, AFNI sub-bricks of different types, and image
stacks. Indexing a `ConcatenatedProxy` indexes only the proxies that the
selection reaches.

### Fourth layer: chunked stores keep their libraries

Formats whose library already reads chunks or tiles keep that library.
Their proxies derive from `ArrayProxy`, implement `_read` with the
library, and set `chunks` to the native block shape. They lose their
copies of the shortcuts and of `_basic_index`, but their reading code
does not change. The TensorStore proxies of N5 and precomputed share a
small base class, because they differ only in the driver name.
`DelayedH5Array` becomes the shared HDF5 proxy of BigDataViewer, Imaris,
ITK and X5. Pillow and JPEG 2000 keep reading the whole image, because
their codecs cannot decode a part of it.

The native block shapes are implemented: the proxies of Zarr, N5,
precomputed, BigDataViewer, Imaris, OpenSlide and image stacks, and
`DelayedH5Array`, give them as `chunks`. The shared base class of
N5 and precomputed is not implemented. The readers of transformation
fields stored in HDF5 (ITK and X5) and in OME-Zarr convert their lazy
arrays with the backend setting, so that under `"dask"` a field is a
Dask array with these blocks, and the writer of OME-Zarr fields writes
it block by block into chunks of the same shape.

## How each format would read its data

| Format        | Proxy                                           |
|---------------|-------------------------------------------------|
| NIfTI         | `ScaledProxy(StridedProxy(...))`                |
| MGH           | `StridedProxy`                                  |
| MRtrix        | `ScaledProxy(StridedProxy(...))`, `Bit` codec   |
| NRRD          | `StridedProxy`, or `ConcatenatedProxy` of files |
| AFNI          | `ScaledProxy(ConcatenatedProxy(...))`           |
| MINC1         | `ScaledProxy(StridedProxy(...))`                |
| MINC2         | `ScaledProxy` over the HDF5 proxy               |
| Image stacks  | `ConcatenatedProxy` of Pillow or TIFF proxies   |
| Others        | their library, on `ArrayProxy`                  |

MINC1 files are NetCDF classic files, whose variables are stored
contiguously at an offset that the header gives, so they fit the third
layer. The header can be parsed with `scipy.io.netcdf_file`, and SciPy
is already a dependency.

A few formats need the offset to be found when the file is opened
rather than when the header is parsed. NRRD may skip lines, whose
length is not known in advance, or count its offset from the end of the
file. A `ByteSource` therefore accepts a function that computes the
offset from the open stream. The NRRD proxy also checks, before it
reads, that the header still describes the same data, and the same
function can do that check.

## Removing nibabel

Once NIfTI, MGH and MINC read their data with the third layer, nibabel
is used only to parse and write headers. The remaining work is the
following.

- NIfTI-1 and NIfTI-2: the fixed header is a NumPy structured dtype
  of 348 or 540 bytes, read in either byte order. The code also
  needs the conversion between the quaternion of the `qform` and a
  matrix, the rules that choose between the `qform` and the `sform`,
  the tables of data types, intents and units, and the chain of
  extensions. The source holds
  about 110 references to `nb.Nifti1Image`, `nb.Nifti1Header` and their
  NIfTI-2 counterparts, most of them in the writers of transformation
  fields, and they would use `NiftiRaw` instead.
- MGH: a big-endian header of 284 bytes and an optional footer of
  tagged fields.
- MINC1: the NetCDF header through SciPy, and the attributes
  `image-min` and `image-max`. MINC2 already needs h5py.

nibabel would remain a test dependency, so that the tests compare every
header and every array with what nibabel reads. It could also remain an
optional dependency for a conversion to nibabel images.

## What is implemented of the third layer

`StridedProxy` in `brainhops._core.strided` reads an array of
fixed-size values at an offset of a file that may be compressed. Where
each value lies is given as the layout above describes it, by the
position of the first value and one stride in bytes for each axis,
which may be negative, so the proxy holds a layout in its own fields
rather than in a separate `Layout` class. Its constructor
`StridedProxy.contiguous` builds the strides of an array stored in C or
Fortran order and presented through an axis permutation and flips,
which are the layouts of MRtrix, NRRD and AFNI. The proxy presents the
stored values, and the MRtrix and AFNI readers wrap it in a
`ScaledProxy` when their header scales them. A read gets the range of
bytes from the first value of the selected box to its last one, rather
than a merged list of byte ranges, and a local file that is not
compressed is memory-mapped. The blocks that it gives Dask are slabs of
the axis with the largest stride, as many as fit in the option
`array.chunk-size` of Dask, so that each block reads one contiguous
range of bytes. It has no cache, so a compressed file is decompressed
from its start for each block that Dask reads, which indexed_gzip makes
cheap when it is installed. Its constructor is built from its fields,
which convert their arguments to their types, and the MRtrix and AFNI
proxies add the record of their file as a field and build themselves
from it with `from_record`. Status: implemented.

`ArrayProxy` is a `Magic` class of bagof without fields, and every proxy
of brainhops declares its state as fields, so that its constructor is
built from them. A value that a proxy computes rather than receives is
a property, such as a shape or a type read from the record, and a cache
that it fills on the first read, such as an open store or file, is a
private field that is not an argument of the constructor and is not
shown. The metaclass of `Magic` is that of an abstract base class, so
the members that a subclass must define stay abstract. A field cannot
implement an abstract property, so a proxy that receives its shape or
its type keeps it in a private field, given by its public name, and
returns it from the property: the `_shape` and `_dtype` of
`StridedProxy` are given as `shape=` and `dtype=`. A proxy is compared
by identity. The fields of a subclass convert their arguments to their
types, and its representation is the one that `Magic` builds, which
shows the fields that do not hold their defaults. An indexed proxy
shows its stored proxy and its selection. Status: implemented.

The cases that do not fit a single block of fixed-size values are still
decoded whole: packed `Bit` data of MRtrix, NRRD samples split between
several files, encoded as text or preceded by skipped lines, and AFNI
sub-bricks of several types. MINC keeps nibabel, which is now given the
selection, so a local MINC1 or MINC2 file that is not compressed is read
in part.

## Order of the work

1. Add the protocol and the base class, and move every proxy of
   brainhops onto the base class. This step is implemented. It removes
   the duplicated code and makes indexing lazy for every proxy. The
   proxies of the first group still decode the whole file for any read.
2. Write `ByteSource`, `Layout` and `StridedProxy` in `brainhops._core`,
   without nibabel. A first `StridedProxy` is implemented, without
   `ByteSource`, `Layout`, the merged byte ranges and the cache.
3. Move MRtrix, NRRD and AFNI onto the third layer. This step is
   implemented for the cases that the first `StridedProxy` covers.
4. Parse MGH headers without nibabel, which is the smallest header and
   tests the path from header to `StridedProxy`.
5. Parse NIfTI headers without nibabel, then MINC1.
6. Replace the `isinstance` checks of the writers by `stored` and
   `reads(file)`, and make nibabel optional.

## Later work

Remote sources will read their byte ranges with
`fsspec.AbstractFileSystem.cat_ranges`, which fetches several ranges
in one call, rather than with seeks on one stream. This comes after the
layers above are in place.

The proxies stay synchronous, and no asynchronous interface is planned.
The file systems of fsspec for HTTP and cloud storage are asynchronous
inside, and `cat_ranges` already fetches the ranges of one read
concurrently. Dask already reads several chunks at once from its
threads. An asynchronous interface would add a second way to call every
proxy for a small gain, and it can be reconsidered if a consumer that
is itself asynchronous, such as a viewer, needs it.

Writing through a proxy is left out of this memo, because no writer
needs it yet. It is tracked in #479, which records that compressed
files and remote files will not be written in part.
