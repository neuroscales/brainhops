# Array backends of images and transformations

This memo decides which array package holds the values of an image or
a transformation, how a user moves an object from one package to
another, and how a computation chooses the package of its result. It is
agreed and tracked in #486. Its first three steps are implemented:
the backend settings in `brainhops.backends` (#490), the conversion
methods of lazy arrays (#491), and those of images and transformations
(#492). The other steps are not implemented yet.

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

## What the code does today

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
(`io/images/afni/_image.py:289` for example). The format views convert
the proxy with the global setting at that time
(`io/base/_utils_proxies.py`), so reading `data` either reads the
whole file with NumPy or wraps the proxy in a Dask array.

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

An object built by the readers of `brainhops.io` holds either a NumPy
array or a proxy, and `data` returns what the object holds, without
converting it. A proxy is lazy: indexing or transposing it returns
another proxy and reads nothing (see
[Lazy arrays of file data](array-proxies.md)). The format views mostly
transpose or select axes, so they return a proxy as well, and `data`
no longer needs a cache. The values are read only when the user asks
for them, as described below, or when a computation needs them.

An object that a user builds keeps the array that it was given, as it
does today.

This choice changes the code that computes directly on `data`. An
expression such as `image.data + 1` fails when `data` is a proxy, so
that code, inside brainhops and in the tests, must first call
`to_numpy()` or `np.asarray`.

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
  arrays. The Dask versions of the `ndimage` functions already choose
  the `ndimage` package of each block (`_core/dask_ndimage.py:269` and
  `:372`), so they should work on CuPy chunks, which has not been
  tested.

The global setting becomes `None` instead of the first installed
backend, so that installing Dask no longer changes any result.

### Small parameters

The settings apply to voxel arrays, the data of images and of
transformation fields. Small parameters, such as the matrix of an
affine transformation, stay NumPy arrays where they are stored, as
their declarations already require. A computation moves them to the
device of the voxel data that it works on. With CuPy, the matrix is
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

1. Every proxy among its inputs is resolved with the setting in effect.
   The input object does not change.
2. With the setting `None`, the result takes the backend of highest
   priority among the inputs, as `best_backend` does today. With any
   other setting, the inputs are converted to that setting and so is
   the result.
3. An array that the computation creates, such as a coordinate grid,
   takes the backend of the result, and never the global setting. This
   rule fixes the reslice of a NumPy image that returns a Dask array.

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
4. Make `data` return what the object holds, and update the code and
   the tests that compute on it.
5. Apply the rules of computations: resolve the proxies of the inputs,
   create arrays in the backend of the result, and move small
   parameters to the device of the voxel data.

## Open questions

- On a multi-scale image, `to_*` could return the finest level, as
  `data` does, while `as_*` converts every level.
- Readers could take a `backend=` argument, but
  `io.load(path).as_dask()` already says the same thing, so the memo
  proposes no such argument.
