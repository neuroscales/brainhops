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
wrp = io.load("warp.nii.gz", hint="spm")  # -> SpmCoordinatesField
```

The format classes can also be used directly:

```python
from brainhops.io.images.nifti import NiftiImage
from brainhops.io.transformations.freesurfer.lta import LtaTransformation
from brainhops.io.transformations.nifti import NiftiRASDisplacementField
from brainhops.io.transformations.spm.y import SpmCoordinatesField

src = NiftiImage.load("source.nii.gz")
dst = NiftiImage.load("dest.nii.gz")
aff = LtaTransformation.load("affine.lta")
dsp = NiftiRASDisplacementField.load("disp.nii.gz")
wrp = SpmCoordinatesField.load("warp.nii.gz")
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
    By default, multidimensional arrays other than affine matrices are not
    loaded into memory. They are mapped lazily to a `dask.array.Array`
    instead. Another array backend can be selected when a file is loaded,
    within a context manager, or as the default for the whole session.

    === "`load` argument"

        ```python
        src = io.images.load("source.nii.gz", backend="cupy")
        ```

    === "context manager"

        ```python
        from brainhops.backends import backend

        with backend("cupy"):
            src = io.images.load("source.nii.gz")
        ```

    === "default backend"

        ```python
        from brainhops.backends import set_backend

        set_backend("cupy")
        src = io.images.load("source.nii.gz")
        ```

## Save data to files

The function `save` writes an object in the format that the file name
calls for. An object read from a file can be written back in its own
format, or in any other format that holds the same kind of object. Writing
Zarr requires the `zarr` extra.

```python
img = io.images.load("source.nii.gz")
io.save(img, "copy.nii.gz")  # -> NIfTI
io.save(img, "copy.zarr")  # -> Zarr
```

An image computed in memory is written the same way, since NIfTI and Zarr
both hold a plain image. A transformation is converted to the format the
name asks for, with the same converters as `t.to(Format)`, but only when
the format holds the very same map. Its `input` and `output` are read,
and bridged to the format's: an affine from voxels to LPS is written as
the voxel-to-RAS affine of a NIfTI file, with its first two axes
flipped, and an affine whose systems are not known is taken to map the
format's. A field of displacements is written as a NIfTI displacement
field, between its grid's world-to-voxel affine and its inverse, and a
field of coordinates as a NIfTI (or, with a `y_` prefix, an SPM) field of
coordinates:

```python
io.save(affine, "affine.nii.gz")  # -> NiftiVoxelToRAS
io.save(warp, "warp.nii.gz")  # -> NiftiRASDisplacementField
io.save(deformation, "y_deformation.nii")  # -> SpmCoordinatesField
```

Nothing is approximated to fit a format. An affine between two world
spaces is not a voxel-to-RAS affine, and a field interpolated with cubic
splines is not the linearly interpolated values a NIfTI file holds, so
both are refused, with the reason each format gives. The same conversion
can be asked for explicitly, or the format can be built directly:

```python
import numpy as np
from brainhops.datamodel.transformations import Affine
from brainhops.io.transformations.nifti import NiftiVoxelToRAS

affine = Affine(np.eye(3, 4))
NiftiVoxelToRAS.from_any(affine).save("affine.nii")
affine.to(NiftiVoxelToRAS)  # the same conversion
```

Every family with an affine form, such as a `Scaling` or a `Translation`,
is converted in the same way. A format that has no exact conversion yet
refuses `t.to(Format)` with a `ConversionError` rather than relabelling
`t` as the format.

An LTA file states which coordinate systems its affine maps between. A
general `Affine` is therefore written to an LTA file only if its `input`
and `output` say so as well: both must be `RASmm` (or both `RSAmm`), or
both must be the voxel or physical system of an LTA volume. An affine read
from an LTA file is written back as it was read.

```python
from brainhops.datamodel.systems import RASmm

matrix = np.eye(3, 4)
io.save(Affine(matrix, input=RASmm(), output=RASmm()), "affine.lta")
io.save(io.load("affine.lta"), "copy.lta")  # -> the same file
```

## Images

The images `src` and `dst` are `NiftiImage` objects. `NiftiImage` is a
`SingleScaleImage`, which is itself an `Image`. An image combines an array
with the transformations that place its voxels in world space.

| Name              | Type                   | Description                                                                                                                                         |
| ----------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`            | `ArrayProtocol`        | The content of the image, F-ordered (e.g. {x, y, z, t, c, ...}). A file is mapped to a `dask.array.Array` by default.                                |
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
