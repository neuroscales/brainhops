# Getting started with the Python interface

## Load data from files

brainhops has readers and writers for many image and transformation
formats. The function `load` guesses what kind of content a file holds and
returns an object of the appropriate class.

Some formats can hold several kinds of content without recording which
one they hold. A NIfTI file, for example, may contain an image, a
displacement field or a coordinate field. When the file does not say, the
`hint` argument names the format to use.

```python
from brainhops import io

src = io.load("source.nii.gz")  # -> NiftiImage
dst = io.load("dest.nii.gz")  # -> NiftiImage
aff = io.load("affine.lta")  # -> LtaTransformation
dsp = io.load(  # -> NiftiRASDisplacementField
    "disp.nii.gz", hint="displacements"
)
wrp = io.load("warp.nii.gz", hint="spm")  # -> SpmCoordinateField
```

The format classes can also be used directly:

```python
from brainhops.io.images.nifti import NiftiImage
from brainhops.io.transformations.freesurfer.lta import LtaTransformation
from brainhops.io.transformations.nifti import NiftiRASDisplacementField
from brainhops.io.transformations.spm import SpmCoordinateField

src = NiftiImage.load("source.nii.gz")
dst = NiftiImage.load("dest.nii.gz")
aff = LtaTransformation.load("affine.lta")
dsp = NiftiRASDisplacementField.load("disp.nii.gz")
wrp = SpmCoordinateField.load("warp.nii.gz")
```

When the kind of object is known, the loaders of `io.images` and
`io.transformations` only consider formats of that kind, which is both
faster and less ambiguous:

```python
src = io.images.load("source.nii.gz")
dst = io.images.load("dest.nii.gz")
aff = io.transformations.load("affine.lta")
dsp = io.transformations.load("disp.nii.gz", hint="displacements")
wrp = io.transformations.load("warp.nii.gz", hint="spm")
```

!!! tip "Lazy loading"
    Reading a file does not read its voxels. The `data` of an image read
    from a file is a lazy array of the file, which reads its values only
    when a computation needs them. Its shape and the type of its values
    come from the header, and indexing it reads only the part that the
    index selects. A computation runs in the backend that the setting
    chooses. Under the default setting, a small array is read into a
    NumPy array, and an array larger than 64 MiB becomes a
    `dask.array.Array` that reads its values block by block. Another
    backend can be selected within a context manager, or as the setting
    of the whole session, and `to_backend` converts the data on request.

    === "context manager"

        ```python
        from brainhops.backends import backend

        src = io.images.load("source.nii.gz")
        shape = src.data.shape  # the file is not read
        with backend("cupy"):
            total = src.data[..., 0].sum()  # the first volume, on the GPU
        ```

    === "session setting"

        ```python
        from brainhops.backends import set_backend

        set_backend("cupy")
        data = io.images.load("source.nii.gz").to_backend()  # a CuPy array
        ```

## Save data to files

The function `save` writes an object in the format that the file name
calls for. An object read from a file can be written back in its own
format, or in any other format that holds the same kind of object. Writing
Zarr requires the `zarr` extra. A plain Zarr array stores no geometry, so
an image that has one is refused there rather than written without it,
until its transformations are cleared.

```python
img = io.images.load("source.nii.gz")
io.save(img, "copy.nii.gz")  # -> NIfTI
img.transformations = []
io.save(img, "copy.zarr")  # -> Zarr, without the geometry
```

An image computed in memory is written the same way, since NIfTI and Zarr
both hold a plain image. A transformation is converted to the format the
name asks for, with the same converters as `t.to(Format)`. Its `input`
and `output` are read, and bridged to the format's: an affine from voxels
to LPS is written as the voxel-to-RAS affine of a NIfTI file, with its
first two axes flipped, and an affine whose systems are not known is
taken to map the format's. A field is written as the values at the nodes
of its grid, as a field of displacements or of coordinates, whichever the
format stores. The two are the same map inside the grid and differ only
in how they are extended outside it, which each format decides:

```python
io.save(affine, "affine.nii.gz")  # -> NiftiVoxelToRAS
io.save(warp, "warp.nii.gz")  # -> NiftiRASDisplacementField
io.save(deformation, "y_deformation.nii")  # -> SpmCoordinateField
```

Several formats may claim the same name, as NiftyReg, MRtrix and elastix
all claim `.txt`. The object is converted to each of them, and the format
that keeps the most of it is chosen: a field of the same kind, and the
same coordinate systems, so that a field between LPS spaces is written as
an ITK field rather than as a NIfTI field between RAS spaces. A format
that cannot hold the object gives way to the next. An error that is not
such a refusal, such as a file that cannot be written, stops the save at
once.

A `hint` names the format, as it does for `load`, and only the formats
that answer to it are chosen from. A format named by its hint is written
whatever the extension of the name, which lets a format be written under
a name that no format claims, such as a plain `.json` name:

```python
io.save(affine, "transform.json", hint="neuroglancer.affine")
```

A format that holds a transformation only approximately is used only
when `lossy=True` is given. A field interpolated with cubic splines, for
example, is written in a NIfTI file as the values at its nodes, which are
read back with linear interpolation. Some formats need images that the
transformation does not hold, such as the reference and moving images of
an FSL warp, which are given as options of the conversion. A format that
cannot hold the transformation at all is refused, with the reason it
gives: an affine between two world spaces is not a voxel-to-RAS affine.

```python
io.save(bspline, "warp.nii.gz", lossy=True)
io.save(
    warp,
    "fnirt.nii.gz",
    hint="fnirt",
    conversion={"reference": ref, "moving": moving},
)
```

The same conversion can be asked for explicitly, or the format can be
built directly:

```python
import numpy as np
from brainhops.datamodel.transformations import Affine
from brainhops.io.transformations.nifti import NiftiVoxelToRAS

affine = Affine(np.eye(3, 4))
NiftiVoxelToRAS.from_any(affine).save("affine.nii")
affine.to(NiftiVoxelToRAS)  # the same conversion
```

Every family with an affine form, such as a `Scaling` or a
`Translation`, is converted in the same way. A class that is bound to
coordinate systems, such as `VoxelToRAS`, accepts only endpoints that
are compatible with those systems. A transformation between other
systems is therefore refused with a `ConversionError`, rather than given
the class with its parameters unchanged. `t.to(Format)`,
`Format.from_any(t)` and `Format.from_instance(t)` give the same result,
or raise the same error.

Every affine file format converts any affine between worlds, and a
chain of affines, into its own coordinate systems. An affine between
LPS spaces is flipped into RAS for an LTA or an MNI `.xfm` file, and an
affine between RAS spaces into LPS for an ITK file. An affine whose
systems are not known is taken to map the systems of the format. A
format that places its matrix with the geometry of two images, such as
a FLIRT matrix or a `register.dat` file, takes the images as the options
`reference=`, the image of the input, and `moving=`, the image of the
output, and refuses the conversion without them.

```python
from brainhops.datamodel.systems import LPSmm
from brainhops.io.transformations.fsl.flirt import FlirtTransform

matrix = np.eye(3, 4)
affine = Affine(matrix, input=LPSmm(), output=LPSmm())
io.save(affine, "affine.lta")  # an LTA file between RAS spaces
io.save(io.load("affine.lta"), "copy.lta")  # -> the same file
flirt = affine.to(FlirtTransform, reference=src, moving=dst)
flirt.save("affine.mat")
```

## Images

The images `src` and `dst` are `NiftiImage` objects. `NiftiImage` is a
`SingleScaleImage`, which is itself an `Image`. An image combines an array
with the transformations that place its voxels in world space.

| Name              | Type                   | Description                                                                                                                                         |
| ----------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`            | `ArrayProtocol`        | The content of the image, F-ordered (e.g. {x, y, z, t, c, ...}). A file is a lazy array that reads its values when a computation needs them.         |
| `transformations` | `list[Transformation]` | Transformations from the voxel space to different world spaces. The last transformation in the list is the preferred one.                         |
| `transformation`  | `Transformation`       | The preferred voxel-to-world transformation, which is the last element of `transformations`.                                                       |
| `geometry`        | `Geometry`             | The preferred transformation, preceded by a `CartesianField` whose shape matches the shape of the data. It defines the grid on which the image lives. |

## Apply a transformation to an image

An image can be called on a transformation whose output space is the
preferred world space of the image. The call moves the image by that
transformation without resampling anything. It returns a new image whose
preferred transformation is the inverse of the given transformation
composed with the original preferred transformation. Representing the
moved image in this way delays all computation until the image is
resampled, which keeps the operation general and cheap. The two following
lines are equivalent:

```python
from brainhops.datamodel.images import SingleScaleImage

mov = src(aff)
mov = SingleScaleImage(
    data=src.data,
    transformations=[*src.transformations, aff.inverse() @ src.transformation],
)
```

The moved image is computed by `reslice`, which resamples it onto a grid.
The grid is given by an image or by its geometry:

```python
mov = src.reslice(dst)  # -> geometry = dst.geometry
mov = src.reslice(dst.geometry)  # the same
mov = src(aff).reslice(dst)  # move src by aff, then resample onto dst
mov = src(aff)(dsp).reslice(dst)  # apply aff first, then dsp
```

By default, the output keeps the input dtype. A floating output dtype
retains interpolated fractions even if the input stores integers:

```python
mov = src(aff).reslice(dst, dtype="float32")
io.save(mov, "resampled.nii.gz")
```

The loaded image is unchanged. An explicit dtype promotes the input and
output dtypes to choose floating working precision, with a minimum of
float32. Integer-only combinations use float64. Integer output keeps the
existing rounding and saturation rules, and Dask arrays remain lazy.

Without an argument, `reslice` resamples the image onto its own grid.
Inferring the output grid from the transformations themselves, for example
from a chain that starts with a `Geometry` or with a displacement field
defined on a grid, is planned but not implemented yet.

## Transformations

The basic transformations are mostly modelled on the
[OME-NGFF specification](https://ngff.openmicroscopy.org/specifications/dev/index.html),
with some additional flexibility:

- The input and output spaces of a transformation are stored in the
  transformation itself, rather than named and looked up in a separate
  dictionary of coordinate systems.
- Spaces may be partially specified, for example without a name or without
  named axes, or not specified at all. Consecutive transformations
  therefore do not need to declare matching spaces: brainhops bridges them
  by matching axes by type, orientation or name. Bridging is less robust
  than a sequence whose spaces match, so critical applications should make
  sure that they do, but it usually works.
- Transformations that act on a subset of the axes do not need to be
  wrapped explicitly. When the dimensionality of two transformations
  differs, brainhops matches their axes partially and wraps the smaller one
  in a `SubspaceTransformation`.
- Additional transformations are available, such as the exponential
  parameterisations of affine subgroups (`AffineExponential`,
  `RotationExponential` and others) and stationary velocity fields.

### Operators

A transformation that maps a space to itself has an inverse, a square, and
a principal square root. Each of them is a method, and each is lazy in the
same way as `inverse`: nothing is computed until the result is applied,
computed or converted. A typed result stays in its family, so the square
root of a `Rotation` is a `Rotation`.

```python
half = xform.sqrt()  # the half-transformation: half @ half maps like xform
twice = xform.square()  # xform @ xform
expr = a.inverse() @ b.sqrt()
result = expr.compute()
```

A transformation outside the domain of an operator, such as a reflection
under `sqrt`, raises `DomainError` rather than returning a complex or
non-principal result.

The exponential and the logarithm are not operators but an encoding. The
`log` flag says that `data` holds the tangent of the map about the
identity, and `.to(log=...)` converts between the two encodings.

```python
velocity = DisplacementField(data=v, log=True)  # a StationaryVelocityField
warp = velocity.field  # the displacement of its flow, by scaling and squaring
plain = velocity.to(log=False)  # the same map, as a DisplacementField
tangent = Affine(matrix=m).to(log=True)  # an AffineExponential: logm(m)
half = velocity.sqrt()  # exact: the velocity, halved
```

The inverse, square root and square of a tangent are exact. A velocity
stored in a file is read with `io.transformations.load(path, log=True)`,
or with the `svf` hint on the command line (`warp.nii.gz|svf`). See
[Tangents: the `log` flag](../api/datamodel/transformations.md#tangents-the-log-flag)
for the full model.

### Restriction to some axes

`restrict` keeps a part of a transformation, such as the spatial part of
a map of space and time. It reads the transformation from some of its
input axes to some of its output axes, which are given by position, by
name or by any other axis query, such as an axis class. The restriction
is lazy, like `inverse`: it places a `Projection` before the
transformation, which creates the inputs that are removed, and another
after it, which drops the outputs that are removed, and `compute`
reduces the result.

A removed input takes the value that `at` gives it, so that a
restriction can select one volume of a time series that has one affine
per volume. A removed input that `at` does not name is free, which
means that the kept outputs must not read it, and `compute` raises
`RestrictionError` when one of them does.

```python
from brainhops.datamodel.axes import SpaceAxis

spatial = xform.restrict(SpaceAxis, SpaceAxis)
volume = xform.restrict(SpaceAxis, SpaceAxis, at={"t": 4})
matrix = volume.compute().matrix
```

## Comparing transformations and images

Transformations and images compare and hash by identity: `a == b` is the
same as `a is b`, and `==` never raises.

```python
from brainhops.datamodel.transformations import Affine

a = Affine(matrix)
b = Affine(matrix)
a == a  # -> True
a == b  # -> False: two distinct objects, even with the same matrix
{a, b}  # -> a set of two transformations
```

As a consequence, transformations and images can be put in a `set` or used
as dictionary keys, and list operations such as `in`, `index` and `remove`
find them by identity. A distinct object with the same parameters is a
different element.

!!! note "Testing whether two transformations are the same map"
    "The same" has several possible meanings: the same object, the same
    map, or the same parameters in the same coordinate systems. Since no
    single meaning is right for every use, `==` does not pick one. Whether
    two transformations describe the same map is tested by checking that
    one composed with the inverse of the other is the identity, and their
    coordinate systems are compared explicitly:

    ```python
    from brainhops.datamodel.transformations import is_identity

    is_identity((a.inverse() @ b).compute(), compute=True)  # -> True
    ```

    Likewise, the data of two images is compared explicitly, for example
    with `numpy.array_equal(img1, img2)`.
