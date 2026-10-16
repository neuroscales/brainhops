# Lazy arrays of file data

This memo is a proposal and nothing in it is implemented yet. It
decides how brainhops reads the voxels of a file only when they are
needed, and how the classes that do so share their code. It also
explains how the same design lets brainhops read NIfTI, MGH and MINC
files without nibabel.

## What a proxy is for

When brainhops reads an image or a transformation field, it parses the
header at once but does not read the voxels. It keeps instead a small
object, called a proxy, that knows where the voxels lie and how they
are stored. The proxy behaves like a read-only array: it has a `shape`
and a `dtype`, NumPy converts it with `np.asarray`, and indexing it
returns the selected values. The proxy is stored as the `rawdata` of
the format object (see [Format records](format-records.md)). The `data`
property turns it into a NumPy array or a Dask array, depending on the
array backend.

A proxy keeps no file open between reads. It opens the file each time
its values are needed, so that it can be pickled, sent to Dask workers
and read from several threads, and so that the file can be written over
while the format object exists. When the caller passed a stream rather
than a path, the proxy keeps the stream and its starting position, and
the caller must keep the stream open.

## What the code has today

brainhops defines fifteen proxy classes, and NIfTI and MGH use the
`ArrayProxy` of nibabel. They fall in two groups.

The first group holds formats whose voxels are an array of fixed-size
values stored at an offset of a file that may be compressed. The
proxies of this group decode the bytes themselves or ask nibabel to.

| Format    | Class                  | Partial reads | Scaling         |
|-----------|------------------------|---------------|-----------------|
| MRtrix    | `_MrtrixProxy`         | no            | yes, global     |
| NRRD      | `_NrrdProxy`           | no            | no              |
| AFNI      | `_BrikProxy`           | no            | yes, per volume |
| MINC      | `_MincProxy` (nibabel) | no            | yes, per slice  |
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

None of these classes shares a base class or a protocol, and each one
is recognized by `isinstance` checks against its own class. They
nevertheless repeat the same code:

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

Each of the format views (`_views.py`) also turns the proxy into an
array in its own way: some call `asarray` of the backend, some call
`da.from_array` with the chunks of the proxy, and others call a method
such as `to_dask`, `view` or `read`.

The module `brainhops._ext.npfileobj` was started to give lazy access
to an array stored contiguously in any file-like object, including a
remote one. It holds index arithmetic (`indexing.py`) and the writing
counterpart of `nibabel.fileslice` (`readwrite.py`), but no proxy, and
nothing in brainhops uses it. Its writing functions import private
helpers of nibabel.

## Problems

The proxies of the first group that brainhops wrote read the whole file
each time they are indexed, and then select the values in memory. Dask
splits an array into chunks and indexes the proxy once for each chunk,
so the file is read once per chunk. A float32 image of shape
`(256, 256, 256, 4)`, which takes 256 MiB, is split into eight chunks by
the default settings of Dask 2026.8, so it is decompressed eight times
when it is stored with gzip. A local file that is not compressed is
memory-mapped, which makes the repeated read cheap, but the MRtrix and
AFNI proxies then apply the intensity scaling to the whole volume
before they select a chunk, which costs as much as reading it. Only the
nibabel proxy and the proxies of the second group avoid the problem.

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

    def __getitem__(self, index: tx.Any) -> np.ndarray: ...

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray: ...
```

Indexing a proxy reads the values and returns a NumPy array, as
indexing the nibabel proxy does. Dask and every other consumer of
array-like objects expect this.

### Second layer: a base class

`ArrayProxy` is an abstract class that implements everything the
proxies repeat today. A subclass defines `shape`, `dtype` and one
method, `_read(index)`, which receives an index that is already
normalized: a tuple with one integer or one slice with a positive step
for each axis. The base class provides the rest.

- `__getitem__` normalizes the index with the functions of
  `npfileobj.indexing`, answers an empty selection without reading,
  and reads the bounding box of any other index with `_read`. It then
  applies negative steps, new axes and array indices in memory, so
  that a subclass never sees them.
- `__array__` reads `self[...]` and honors `dtype` and `copy`.
- `chunks` is the block shape that the storage prefers, such as the
  chunks of a Zarr array or one slice of a stack. It is `None` when
  the storage has no preference.
- `to_dask()` calls `da.from_array` with these chunks and with a
  `meta` argument, so that Dask does not have to read an empty
  selection to learn the type.
- `lazy[index]` and `transpose(perm)` return a new proxy that reads
  only the selected values in the new axis order. The default
  implementation composes the index and the permutation with
  `compose_index` and `split_operation` of `npfileobj.indexing`.
  The third layer replaces the composition by arithmetic on the
  layout.
- `reads(file)` tells whether the proxy reads a given file, and
  replaces the four functions that do so today. It returns `False`
  unless the subclass knows its file.
- `stored` returns a description of the bytes that a writer may copy
  as they are, or `None` when there are none (see below).

The format views then share one function, which returns the proxy as a
read-only NumPy array or as `to_dask()` depending on the backend.

### Third layer: strided data behind a file

This layer is what `npfileobj` was started for, and it lives there
because it knows nothing about brain images. It is made of three
parts.

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
offset in bytes, a shape, a dtype with its byte order, and one stride
in bytes for each axis. Strides may be negative. A layout covers C
order, Fortran order, the axis permutation and flips of an MRtrix
`layout` and the axis order of NRRD, without copying anything. Indexing
or transposing a layout returns another layout, computed by arithmetic
alone, which is what makes `lazy[index]` free. A layout also turns a
selection into the list of byte ranges to read. Ranges that are close
are merged, so that one large read replaces many small ones. This
replaces `nibabel.fileslice`, and the writing functions of
`readwrite.py` are rewritten on it.

`StridedProxy` is the `ArrayProxy` that pairs a source with a layout.
Its `_read` memory-maps a local file that is not compressed and returns
a view of it. For any other source, it reads the merged ranges. With a
codec that cannot seek cheaply, such as bzip2, it reads the stream once
from the first range to the last one and keeps only what the ranges
select. Its `stored` description is the source and the byte range of
the data, which a writer copies when the target stores the same layout.

Two generic proxies complete the layer. `ScaledProxy` applies a slope
and an intercept that broadcast against the shape, which covers the
single factor of NIfTI and MRtrix, the factor of each AFNI sub-brick
and the slice-wise scaling of MINC. Because it scales only what it
reads, a chunk costs only its own values. `ConcatenatedProxy` joins
proxies along one axis. It covers NRRD data split into several files,
AFNI sub-bricks of different types, and image stacks.

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

## Order of the work

1. Add the protocol and the base class, and move the proxies of the
   second group onto the base class. This removes the duplicated code
   and changes no behavior.
2. Write `ByteSource`, `Layout` and `StridedProxy` in `npfileobj`
   without nibabel, and rewrite `readwrite.py` on them.
3. Move MRtrix, NRRD and AFNI onto the third layer. This fixes the
   repeated whole-file reads, and the existing tests cover the result.
4. Parse MGH headers without nibabel, which is the smallest header and
   tests the path from header to `StridedProxy`.
5. Parse NIfTI headers without nibabel, then MINC1.
6. Replace the `isinstance` checks of the writers by `stored` and
   `reads(file)`, and make nibabel optional.

## Open questions

- Should a `StridedProxy` keep a small cache of decoded bytes for
  codecs that cannot seek, so that Dask does not decompress a bzip2
  file once per chunk? A cache would break the rule that a proxy keeps
  nothing between reads.
- Should remote sources read their byte ranges with
  `fsspec.AbstractFileSystem.cat_ranges`, which can fetch them in
  parallel, rather than with seeks on one stream?
- Writing through a proxy, which `readwrite.py` prepares for, is left
  out of this memo, because no writer needs it yet.
