"""NRRD images.

NRRD ("Nearly Raw Raster Data", <https://teem.sourceforge.net/nrrd/>) is
the native format of 3D Slicer and teem, and is also handled by ITK. Files
are parsed by [`brainhops.io.common.nrrd`][brainhops.io.common.nrrd] with
numpy alone.

| Class                   | Extension | Hints              |
| ----------------------- | --------- | ------------------ |
| [`AttachedNrrdImage`][] | `.nrrd`   | `attached`         |
| [`DetachedNrrdImage`][] | `.nhdr`   | `nhdr`, `detached` |

Both classes answer to the format hint `nrrd`, the format name that can
be attached to a path to choose its reader. Their own hints can also be
qualified, as in `nrrd.attached` or `nrrd.nhdr`. Both classes derive from
[`NrrdImage`][] and read either kind of file, because the content decides
the kind: a header that names a `data file` is detached. When an image
is written, the file name decides which kind is produced.

```python
import brainhops.io as io

image = io.load("dwi.nhdr")            # a DetachedNrrdImage
image.data                             # [x, y, z, c], F order
image.transformation                   # index -> LPS mm (Affine)
image.metadata.raw.keyvalue["DWMRI_b-value"]  # key/value pairs, as text
image.save("dwi.nrrd")                 # attached, in the source encoding
io.save(image, "dwi.nii.gz")           # or any other image format
```

An image holds the header of its file as [`NrrdMetadata`][], whose
record is an [`NrrdRaw`][], and its samples in `raw`, a lazy array that
reads them only when `data` is first needed. The first NRRD axis varies
fastest, so the data are in Fortran order. The lazy array presents the
axes of the file with spatial axes (`x`, `y`, `z`) first, then time
(`t`), channels (`c`) and other axes (`dim<i>`), each group in file
order. An axis is spatial if it
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

When an image is written, the record of its metadata is the base of the
new header. The preferred transformation becomes the `space directions`
and `space origin`, unless it is the geometry that the record already
stores. These fields are expressed in the `space` of the source header,
or else in the anatomical space that the transformation maps to, and in
RAS by default. The `encoding`, `endian`, `datatype`, `space`,
`keyvalue` and `data_file` options control the output. Key/value pairs
and other descriptive fields of the source header, such as `content` and
the `measurement frame`, are written back. An image that is read and
written again without changes keeps its bytes, and both files of a
detached image are kept when it is written under the same name.
"""

__all__ = [
    "AttachedNrrdImage",
    "DetachedNrrdImage",
    "NrrdImage",
    "NrrdMetadata",
    "NrrdRaw",
]

from brainhops.io.common.nrrd import NrrdMetadata, NrrdRaw

from ._image import AttachedNrrdImage, DetachedNrrdImage, NrrdImage
