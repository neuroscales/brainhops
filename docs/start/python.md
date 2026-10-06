# Getting started with the Python interface

## Load data from files

Brainhops implements readers and writers for many image and transformation
formats. By default, `load` tries to guess the type of content that is
stored in a file, and returns the appropriate object. Some file formats
are used to store different types of content, without internal metadata
that specifies the content type. An example is NIfTI, which can contain
arrays that should be interpreted as images (e.g. MRIs), or arrays that
should be interpreted as displacement or coordinates fields. In such cases,
a `hint` can be provided:

```python
from brainhops import io

src = io.load("source.nii.gz")  # -> Nifti1Image
dst = io.load("dest.nii.gz")  # -> Nifti1Image
aff = io.load("affine.lta")  # -> LtaTransformation
dsp = io.load("disp.nii.gz", hint="voxdisp")  # -> NiftiVoxelDisplacementField
wrp = io.load("warp.nii.gz", hint="spmy")  # -> SpmCoordinatesField
```

Alternatively, the appropriate classes could have been used:

```python
src = io.Nifti1Image.load("source.nii.gz")
dst = io.Nifti1Image.load("dest.nii.gz")
aff = io.LtaTransformation.load("affine.lta")
dsp = io.NiftiVoxelDisplacementField.load("disp.nii.gz")
wrp = io.SpmCoordinatesField.load("warp.nii.gz")
```

or loaders specific to subtypes of objects:

```python
src = io.images.load("source.nii.gz")
dst = io.images.load("dest.nii.gz")
aff = io.transformations.load("affine.lta")
dsp = io.transformations.load("disp.nii.gz", hint="voxdisp")
wrp = io.transformations.load("warp.nii.gz", hint="spmy")
```

!!! tip "Lazy loading"
    By default, multidimensional arrays (other than affine matrices)
    are not loaded in memory on `load`, but instead are mapped lazily
    into a `dask.Array`. This behavior can be altered by choosing
    a different array backend, either on load or using a context manager:

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

`save` writes an object in the format its file name calls for. An object
read from a file can be written back, or written in another format that
holds the same kind of object:

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
can be asked for explicitly:

```python
from brainhops.io.transformations.nifti import NiftiVoxelToRAS

NiftiVoxelToRAS.from_other(affine).save("affine.nii")
affine.to(NiftiVoxelToRAS)  # the same conversion
```

An LTA file says which coordinate systems its affine maps between, so a
general `Affine` is written to one when its `input` and `output` say it
too: both `RASmm` (or both `RSAmm`), or both the voxel or physical
system of an LTA volume. An affine read from an LTA file is written back
as it was read:

```python
from brainhops.datamodel.systems import RASmm

io.save(Affine(matrix, input=RASmm(), output=RASmm()), "affine.lta")
io.save(io.load("affine.lta"), "copy.lta")  # -> the same file
```

## Images Are Transformed Arrays

The source and destination images are `NiftiImage` objects, which
inherit from `Image`, which itself inherits from `TransformedArray`.
They have the attributes:

| Name | Type | Description |
| ---- | ---- | ----------- |
| `data` | `da.Array` | The content of the image, F-ordered (e.g. {x, y, z, t, c, ...}) |
| `transformations` | `list[Transformation]` | Transformations that can be applied to the voxel grid. The output space of the last transformation in the list is the preferred model space.
| `transformation` | `Transformation` | A transformation from the voxel space to the preferred model space. This is a property that gets automatically computed on the fly.
| `geometry` | `Transformation` | The preferred transformation, concatenated with a `CartesianField` object, whose shape matches the shape of the data.

## Apply a transformation to an image

An `Image` can be called on a `Transformation` that maps from any space
to its preferred space. It returns another image, whose `transformation`
attribute is the composition of the original `transformation` attribute
and the inverse of the transformation. While this may seem counter-intuitive,
this is the most general way of "delaying" the application of a transformation.
This means that the two following blocks of statements are (almost) equivalent:

```python
mov = src(xform)
mov = Image(data=src.data, transformation=xform.inverse() @ src.transformation)
```

The transformed image can then be computed by calling:

```python
mov = src.reslice(dst.geometry)  # -> geometry = dst.geometry
mov = src(dsp).reslice(dsp.geometry)  # -> geometry = dsp.geometry
```

While it is not implemented yet, it might be useful to automatically
detect transformations that start with a `Geometry` object, allowing
`reslice` to be called without an argument:

```python
mov = src(dst.geometry).reslice()
mov = src(wrp @ dst.geometry).reslice()
mov = src(disp).reslice()  # -> geometry == CartesianField(vox_disp.shape)
mov = src(ras2ras).reslice()  # -> raise Exception("Cannot guess geometry")
```

We may even guess the geometry of transformations that start with a
displacement of coordinate field (but that may only be the case for
certain formats, not general fields, and is not well specified in our
data model yet). For example:


```python
mov = mov.reslice(geometry)
```

Note that different behaviours are obtained, depending on whether the
chain of transformation ends with a `CartesianField`, a `CoordinatesField`
or another type of transformation:

```python
mov = src(disp).reslice()  # -> assumes that `disp` has a geometry
mov = src(ras2ras).reslice()  # -> raise Exception("Cannot guess geometry")
```

## Transformations

Basic transformations in `brainhops` are mostly modeled on the
[OME-NGFF](https://ngff.openmicroscopy.org/specifications/dev/index.html)
specification, with additional flexibility:

- Input and output spaces are entirely contained in each transform, rather
  than saving a unique coordinate system name and having to query
  this system from a dictionary.
- Input and output spaces can be partially specified (e.g. no coordinate
  system name, no named axes, etc.) or not specified at all! Consequently,
  the output and input spaces of two sequential transforms do not need
  to exactly match. When they do not, `brainhops` does its best to bridge
  the two transformations in a smart way (by matching axes across the two
  systems based on their type, orientation and/or name). This is (obviously)
  not as robust as ensuring a matching sequence of transformations, so if
  your application is critical, please do so. That said, in most neuroimaging
  applications, our matching algorithm operates reasonably.
- "By Dimension" wrappers are not mandatory. Similarly to the previous
  point, if the number of dimensions in the output and input spaces of
  two sequential transformations differ, `brainhops` will partially
  match axes and generate the appropriate `ByDimension` wrapper.
- Additional transformations are available. For example, non-matrix
  representations of some affine subgroups (quaternions, lie algebra, ...)
  are implemented in `brainhops`.

### Operators

A transformation that maps a space to itself has an inverse, a square and a
principal square root. Each is a method, and each is lazy, like the
inverse: the result is computed when it is applied, computed or converted,
and a typed result stays an instance of the family it belongs to (the
square root of a `Rotation` is a `Rotation`).

```python
half = xform.sqrt()  # the half-transformation: half @ half == xform
twice = xform.square()  # xform @ xform
expr = a.inverse() @ b.sqrt()
result = expr.compute()
```

A transformation outside an operator's domain, such as a reflection under
`sqrt`, raises a `DomainError` rather than returning a complex or
non-principal result.

The exponential and the logarithm are not operators but an encoding: the
`log` flag says that `data` holds the tangent of the map about the
identity, and `.to(log=...)` converts between the two.

```python
velocity = DisplacementField(data=v, log=True)  # a StationaryVelocityField
warp = velocity.field  # the displacement of its flow, by scaling and squaring
plain = velocity.to(log=False)  # the same map, as a DisplacementField
tangent = Affine(matrix=m).to(log=True)  # an AffineExponential: logm(m)
half = velocity.sqrt()  # exact: the velocity, halved
```

The inverse, square root and square of a tangent are exact. A velocity
stored in a file is read with the `svf` hint (`warp.nii.gz|svf`) or with
`io.transformations.load(path, log=True)`. See [Tangents: the `log`
flag](../api/datamodel/transformations.md#tangents-the-log-flag).

## Comparing transformations and images

Transformations and images compare, and hash, **by identity**, not by
value: `a == b` is the same as `a is b`, and `==` never raises.

```python
from brainhops.datamodel.transformations import Affine

a = Affine(matrix)
b = Affine(matrix)
a == a  # -> True
a == b  # -> False: two distinct objects, even with the same matrix
{a, b}  # -> a set of two transformations
```

This means that a transformation (or an image) can be put in a `set`, used
as a dictionary key, or looked up in a list with `in`, `index` or `remove`,
and is always found by identity: a distinct object with the same parameters
is a different element.

!!! note "Testing whether two transformations are the same map"
    Whether two transformations are "the same" -- the same object, the same
    map, or the same parameters in the same coordinate systems -- has no
    single answer, so `==` does not pick one. To test whether two
    transformations map coordinates the same way, check that one composed
    with the inverse of the other is the identity, and compare their
    coordinate systems explicitly:

    ```python
    from brainhops.datamodel.transformations import is_identity

    is_identity((a.inverse() @ b).compute(), compute=True)  # -> True
    ```

    Likewise, compare the data of two images explicitly
    (e.g., `numpy.array_equal(img1, img2)`).
