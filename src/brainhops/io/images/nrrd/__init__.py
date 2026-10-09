"""NRRD images.

NRRD ("Nearly Raw Raster Data", <https://teem.sourceforge.net/nrrd/>) is
the native format of 3D Slicer and teem, and is also handled by ITK. Files
are parsed by [`brainhops.io.common.nrrd`][brainhops.io.common.nrrd] with
numpy alone.

| Class                   | Extension | Hints              |
| ----------------------- | --------- | ------------------ |
| [`AttachedNrrdImage`][] | `.nrrd`   | `attached`         |
| [`DetachedNrrdImage`][] | `.nhdr`   | `nhdr`, `detached` |

Both classes answer to the hint `nrrd`, and their own hints can also be
qualified, as in `nrrd.attached` or `nrrd.nhdr`. Both derive from
[`NrrdImage`][] and read either kind of file, since the content decides:
a header that names a `data file` is detached. When an image is written,
the file name decides which kind is produced.

```python
import brainhops.io as io

image = io.load("dwi.nhdr")            # a DetachedNrrdImage
image.data                             # [x, y, z, c], F order, a view
image.transformation                   # index -> LPS mm (Affine)
image.header.keyvalue["DWMRI_b-value"] # key/value pairs, as text
image.save("dwi.nrrd")                 # attached, gzip by default
io.save(image, "dwi.nii.gz")           # or any other image format
```

The first NRRD axis varies fastest, so the data are in Fortran order. The
image axes are a transposed view of the stored values (`dataobj`), with
spatial axes (`x`, `y`, `z`) first, then time (`t`), channels (`c`) and
other axes (`dim<i>`), each group in file order. An axis is spatial if it
has a `space direction`. In a header without directions, an axis is
spatial if its kind is `domain` or `space`, or, when there are no `kinds`
at all, if it is one of the first three. Vector, colour, complex,
quaternion and matrix kinds give channel axes.

An image with a world space carries a `Scaling` to `"physical"` by the
length of each space direction, and the preferred `Affine` to the world,
whose columns are the space directions and whose translation is the
`space origin`. The `space origin` is the centre of the first sample,
whatever the axis `centers`. The world follows `space`: RAS, LAS and LPS
(which Slicer and ITK write) are anatomically oriented, while
`scanner-xyz` and the 3D handed spaces are not, and the `-time` variants
add a time axis. Without a world space, a single transformation to
`"physical"` is built from `spacings`, `axis mins` and `axis maxs`.

When an image is written, the preferred transformation becomes the
`space directions` and `space origin`, in the `space` of the source
header, the anatomical space that the transformation maps to, or RAS. The
`encoding`, `endian`, `datatype`, `space`, `keyvalue` and `data_file`
options control the output. Key/value pairs and other descriptive fields
of the source header, such as `content` and the `measurement frame`, are
written back.
"""

__all__ = ["AttachedNrrdImage", "DetachedNrrdImage", "NrrdImage"]

from ._image import AttachedNrrdImage, DetachedNrrdImage, NrrdImage
