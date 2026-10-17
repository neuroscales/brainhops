# Array backends of images and transformations

This memo decides which array package holds the values of an image or
a transformation, how a user moves an object from one package to
another, and how a computation chooses the package of its result. It is
agreed and tracked in #486, and all its steps are implemented: the
backend settings in `brainhops.backends` (#490), the conversion methods
of lazy arrays (#491), those of images and transformations (#492), the
conversion of `data` at each read (#493), the rules of computations
(#494) and `dataobj` (#514). The paths that involve CuPy are written
from a reading of the code and are tested only where CuPy is installed,
which excludes the continuous integration.

## What a backend is for

An array backend is the package that holds the values of an array.
brainhops supports three of them. NumPy holds the values in memory and
computes them at once. CuPy holds them on a GPU. Dask splits an array
into chunks and computes them only when its result is asked for, which
lets brainhops handle images that do not fit in memory.

A user meets backends when reading `image.data`, when reslicing an
image or composing transformations, and when an image is too large for
NumPy or should run on a GPU. Images and transformation fields hold
large arrays of voxels. Affine transformations, scalings and the like
hold small parameters, such as a matrix of a few rows.

## What the code did before this design

The module `brainhops.backends` keeps one global setting, which
`set_backend` and the `backend()` context manager change. At import,
the setting is the first installed backend in the order Dask, CuPy,
NumPy (`backends.py:23` and `backends.py:41`), so it is Dask whenever
Dask is installed. `get_array_backend(x)` returns the package of an
array, and it returns the package of the global setting when `x` is
`None` or is not an array, such as a lazy array of a file.

An image built from an array (`SingleScaleImage`) keeps that array as
`data`, without converting it. An image read from a file keeps instead
a lazy array of the file, called a proxy, as `rawdata` (see
[Lazy arrays of file data](array-proxies.md)). Its `data` property
decodes `rawdata` the first time it is read and caches the result
(`io/images/afni/_image.py` for example). The format views converted
the proxy with the global setting at that time, so reading `data`
either read the whole file with NumPy or wrapped the proxy in a Dask
array.

Transformation fields keep their array as it is given. Affine
transformations, linear maps, scalings, translations and permutations
declare their parameters as NumPy arrays (`npmatrix` and `npvector`,
for example `datamodel/_transformations/concrete.py:771`), and the
declaration converts any array to NumPy when it is stored, a Dask array
included.

Most computations follow their inputs. The affine helpers call
`get_array_backend(A)` on their input (`_core/affines.py`), and the
spline code calls `best_backend(input, coords)` (`_core/bsplines.py`),
which prefers Dask, then CuPy, then NumPy. A few places create arrays
with the global setting instead. `CartesianField.field`
(`concrete.py:678`) is one of them: it builds the coordinate grid of a
reslice with the global setting.

No method converts an image or a transformation to another backend.
The free function `backends.to_backend(x, backend)` converts one array.

## Problems

The backend of a result depends on what is installed, not on the
inputs. With Dask installed, reslicing a NumPy image of shape
`(8, 9, 10)` by a rotation returns a Dask array, because the grid that
`CartesianField.field` builds is a Dask array and `best_backend` then
prefers Dask.

Reading `data` on an image read from a file does work that the reader
cannot see. With NumPy, it reads the whole file. With Dask, it builds a
Dask array. The result is cached with the setting of the first access,
so a later change of the setting has no effect on that image.

There is no way to move an image or a transformation to another
backend, and CuPy cannot be used with affine transformations. The NumPy
declaration of their parameters converts any array with NumPy, and
CuPy refuses to be converted to NumPy implicitly.

## Decision

### What an object holds

An object holds two arrays. `rawdata` is the array of its file as it
is stored, usually a proxy, and the writers rely on it. `_data` is an
array that a user assigned as `data`, or `None` when nothing was
assigned. Two properties are derived from them.

`dataobj` is the array of the object without reading it. It is the
assigned array when there is one. Otherwise it is the lazy array of the
file in the layout of `data`. `rawdata` and `data` can differ in
layout, because a reader may reverse or permute the axes, drop an axis
of length 1, or move the vector axis of a field to the end. Every
format turns `rawdata` into `data` with such a view only, while the
decoding of values happens inside the proxy, so `dataobj` applies the
view to `rawdata` and reads nothing. The NIfTI and MGH formats wrap the
proxy of nibabel in a proxy of brainhops for that purpose. `shape`,
`dtype` and `geometry` read `dataobj`, so describing an object never
reads its file, and a user can read a part of a large file in the
layout of `data`.

`data` is the array of values. It always supports NumPy arithmetic, so
that an expression such as `image.data + 1` works. An assigned array
is returned converted with the setting in effect. Otherwise, the first
read converts `dataobj` with the setting in effect and caches the
result: under the default setting, a small file is read with NumPy and
a large one is kept lazy as a Dask array, and under
`with backend("dask"):` the file is kept lazy whatever its size. Each
later read returns the cached array converted with the setting in
effect at that read, and the converted array is not cached. A user who
wants the object to hold another backend for good calls an `as_*`
method, or chooses the backend before the first read.

### Conversion methods

Every class that holds data has two families of methods. The names
follow pandas and xarray, where `to_numpy()` returns the bare array and
`as_numpy()` returns an object of the same type that holds NumPy data.

- `to_numpy()`, `to_cupy()`, `to_dask()` and `to_backend(backend=None)`
  return the data of the object as a bare array of that backend. The
  object does not change.
- `as_numpy()`, `as_cupy()`, `as_dask()` and
  `as_backend(backend=None, *, inplace=False)` return an object of the
  same class whose data is an array of that backend. With
  `inplace=True`, the object itself holds the converted data and is
  returned, as `consolidate(inplace=True)` does. On an image read from
  a file, `image.as_numpy(inplace=True)` therefore reads the file once
  and keeps its values.

A proxy has the `to_*` methods as well, so that code can call them on
any array that it receives. It has no `as_*` methods, because a proxy
is an array rather than an object that holds one.

An object that holds other objects converts them with their own `as_*`
methods. A multi-scale image converts every level, and its `to_*`
methods return the data of the finest level, as `data` does. A sequence
converts its members, an inverse converts its forward, and a subspace
or a bijection converts the transformations inside it. A sequence holds
transformations rather than data, so its `to_*` methods raise a
`TypeError`. A file format of a field, such as a displacement field
read from a NIfTI file, derives its members from the field that it
stores in `data`, so it converts that field alone, and its members are
derived again from it.

Each of these classes overrides `as_backend`. By default, `to_backend`
converts `data`, and `as_backend` returns the object as it is, which is
right for a transformation that holds only small parameters.

Names that start with `with_` are kept for methods that replace a field
by a value that the caller gives, such as a future `with_data(array)`.
Methods named after the package alone, as `Tensor.numpy()` in PyTorch,
are not used, because `image.numpy` reads like an attribute or a
module.

### Backend settings

A backend setting is an argument of `to_backend` and `as_backend`, or
the value of the `backend()` context manager and of `set_backend`. It
combines two choices: whether arrays are computed at once or lazily,
and whether they live in memory or on a GPU.

| Where     | At once | Lazily      | Chosen by size |
|-----------|---------|-------------|----------------|
| In memory | `numpy` | `dask`      | `auto`         |
| On a GPU  | `cupy`  | `dask+cupy` | `auto+cupy`    |

The settings apply as follows.

- `None`, the default, keeps every array that is already loaded in its
  backend, and resolves every proxy with `"auto"`.
- `"auto"` chooses by size for every array, loaded or not. An array of
  at most a threshold of bytes is a NumPy array, and a larger one is a
  Dask array, or a NumPy array when Dask is not installed. Wrapping a
  large NumPy array in Dask copies nothing, while turning a small Dask
  array into NumPy computes it. `"auto+cupy"` chooses in the same way
  between CuPy and Dask over CuPy, and never moves an array back to the
  host.
- `"numpy"`, `"cupy"`, `"dask"` and `"dask+cupy"` convert every array to
  that backend. `"dask+cupy"` is a Dask array whose chunks are CuPy
  arrays. The Dask versions of the `ndimage` functions choose the
  `ndimage` package of each block, move the coordinates of a block to
  the package of its window of the input, and declare CuPy blocks to
  Dask when the input or the coordinates are on a GPU.

The global setting becomes `None` instead of the first installed
backend, so that installing Dask no longer changes any result.

### Small parameters

The settings apply to voxel arrays, the data of images and of
transformation fields. Small parameters, such as the matrix of an
affine transformation, are NumPy arrays where they are stored, as
their declarations already require, and they are built with NumPy when
the code creates them, as the matrix of a slice of an image is. A
computation moves them to the device of the voxel data that it works
on, at the moment it combines them with the voxels. With CuPy, the matrix is
copied to the GPU for the call, which costs little next to the work on
the voxels. With Dask, it enters the graph as a NumPy array. The code
that checks, factors and inverts small parameters keeps using NumPy,
which is also faster than a GPU for a matrix of a few rows.

`to_cupy()` and `to_dask()` still convert the parameters of an affine
transformation when they are called, and `as_cupy()` and `as_dask()`
return the transformation unchanged, so that code can convert a list of
objects without testing their type.

### Computations

A computation, such as reslicing an image or composing transformations,
follows these rules.

1. It reads the `data` of its inputs, which follows the setting in
   effect. An input read from a file is therefore read as any first
   read of `data` is, and its cache keeps the array of that read. A
   computation does not try to leave its inputs unchanged, because it
   would have to read `data` differently from a user.
2. It computes with the package that `best_backend` chooses among the
   arrays that it combines: Dask if one of them is a Dask array, CuPy if
   one is a CuPy array, and NumPy otherwise. With the setting `None`,
   this is the package of the inputs. With any other setting, the
   inputs are converted to that setting, and so is the result.
3. Before it combines arrays, it moves them to that package with
   `match_backends`. When one of the arrays is on a GPU, a NumPy array
   becomes a CuPy array, and a Dask array of NumPy chunks becomes a
   Dask array of CuPy chunks. A NumPy array that meets a Dask array in
   memory is left as it is, because Dask accepts NumPy operands. The
   arrays that the computation creates, such as a coordinate grid, and
   the small parameters, such as a matrix, are moved in this way. A
   grid that is combined with a field is built in the package of the
   field.

The `values` and `coefficients` of a field follow the setting as its
`data` does, so the three are always arrays of the same package. The
inversion of a displacement field is only implemented with NumPy, so a
field on a GPU is copied to memory, a Dask field is computed, and the
inverse is converted back to the package of the field.

For example, `with backend("cupy"): moved = image.reslice(target)`
copies the voxels of `image` to the GPU, reslices them there and
returns an image that holds a CuPy array, while `image` keeps its
NumPy array.

## Order of the work

1. Parse the backend settings in `brainhops.backends`, including the
   threshold of `"auto"`, `"dask+cupy"` and `"auto+cupy"`, and make
   `None` the default. This step is implemented. The threshold is
   64 MiB by default, and both `set_auto_threshold` and
   `backend(..., threshold=...)` change it.
2. Add `to_numpy`, `to_cupy` and `to_backend` to `ArrayProxy`, and let
   its `to_dask` place its chunks on a GPU. This step is implemented,
   and `to_dask(cupy=True)` gives the chunks of `"dask+cupy"`.
3. Add the `to_*` and `as_*` methods to the classes of images and
   transformations. This step is implemented (#492).
4. Make `data` convert its cached array with the setting in effect at
   each access (#493). This step is implemented. Images, fields and
   the formats of fields declare their `data` with `dataproperty`, and
   the conversion methods read the array that is held.
5. Apply the rules of computations (#494). This step is implemented.
   `match_backends` moves the arrays that a computation combines to
   one package, the sampling, the spline filters, the separable
   reslice and the composition of fields call it, small parameters are
   built with NumPy, and the views of a field follow the setting.
6. Add `dataobj`, the lazy array of the file in the layout of `data`
   (#514). This step is implemented. Each format defines `dataobj`
   rather than `data`, its views no longer read anything, NIfTI and MGH
   wrap the nibabel proxy in `NibabelProxy`, and the view of the ITK
   fields indexes its axes of length 1 rather than reshape them. The
   TIFF proxy gives the memory map or the Dask array of tifffile when
   it is converted, so that a compressed level is decoded once.

## Open questions

- Readers could take a `backend=` argument, but
  `io.load(path).as_dask()` already says the same thing, so the memo
  proposes no such argument.
