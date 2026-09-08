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
aff = io.load("affine.lta")  # -> LTATransformationRAS2RAS
dsp = io.load("disp.nii.gz", hint="voxdisp")  # -> NiftiVoxelDisplacementField
wrp = io.load("warp.nii.gz", hint="spmy")  # -> SPMCoordinatesField
```

Alternatively, the appropriate classes could have been used:

```python
src = io.Nifti1Image.load("source.nii.gz")
dst = io.Nifti1Image.load("dest.nii.gz")
aff = io.LTATransformationRAS2RAS.load("affine.lta")
dsp = io.NiftiVoxelDisplacementField.load("disp.nii.gz")
wrp = io.SPMCoordinatesField.load("warp.nii.gz")
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
