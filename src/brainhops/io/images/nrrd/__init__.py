"""
Readers and writers for images stored in NRRD files.

NRRD ("Nearly Raw Raster Data", <https://teem.sourceforge.net/nrrd/>) is
the native format of 3D Slicer and teem, and is read and written by ITK.
It is parsed by [`brainhops.io.common.nrrd`][brainhops.io.common.nrrd], with
no dependency beyond numpy:

| Class                     | Extension | Hints                       |
| ------------------------- | --------- | --------------------------- |
| [`AttachedNrrdImage`][]   | `.nrrd`   | `nrrd`, `nrrd.attached`     |
| [`DetachedNrrdImage`][]   | `.nhdr`   | `nrrd`, `nhdr`, `nrrd.nhdr` |

Both derive from [`NrrdImage`][], read either kind of file (the content
decides which class does: a header that names a `data file` is
detached), and write the kind the file name asks for.

```python
import brainhops.io as io

image = io.load("dwi.nhdr")            # a DetachedNrrdImage
image.data                             # [x, y, z, c], F order, a view
image.transformation                   # index -> LPS mm (Affine)
image.header.keyvalue["DWMRI_b-value"] # key/value pairs, as text
image.save("dwi.nrrd")                 # attached, gzip by default
io.save(image, "dwi.nii.gz")           # or any other image format
```

**Axes.** NRRD stores its first axis fastest, so the values are read in
F order. The image axes are the spatial axes first (`x`, `y`, `z`), then
the time (`t`), channel (`c`) and other (`dim<i>`) axes, each group in the
order of the file; this is a transposed *view* of the stored values
(`dataobj`). An axis is spatial when it has a `space direction` (or, in a
header without them, when its kind is `domain` or `space`, or, when the
header has no `kinds`, when it is one of the first three). A `time` axis
is temporal. An axis of kind `vector`, `list`, `point`,
`covariant-vector`, `normal`, `N-vector`, `RGB-color` (and the other
colours), `complex`, `quaternion` or a matrix kind is a channel axis. Any
other (`scalar`, `stub`, `none`) is an untyped axis.

**Coordinate systems.** The transformations are, in order:

1. index -> `"physical"`: a `Scaling` by the length of each space
   direction (and the `spacings` of the other axes);
2. index -> world: the `Affine` whose columns are the space directions,
   and whose translation is the `space origin`.

The world space follows `space`: `right-anterior-superior` (`"RAS"`),
`left-anterior-superior` (`"LAS"`) and `left-posterior-superior`
(`"LPS"`, what 3D Slicer and ITK write) have anatomically oriented axes,
so they convert to one another (and to NIfTI's RAS) from the axes alone;
`scanner-xyz` and `3D-right-handed` / `3D-left-handed` have unoriented
ones; the `-time` variants add a time axis. Their unit is the `space
units`, or millimetres (but for the `3D-*-handed` spaces). A header with
only a `space dimension` has an unoriented `"world"`. A header with no
world space has a single index -> `"physical"` map from the `spacings`,
`axis mins` / `axis maxs` and `units` of its axes.

**Index space and centering.** An integer index is the centre of a
sample, as everywhere in brainhops. NRRD's `space origin` is the position
of the centre of the first sample, whatever the axis `centers`, so the
world `Affine` needs no shift. The `centers` only matter for the `axis
mins` / `axis maxs` fallback: a `cell`-centred axis (the default, as in
teem) spans `size` samples from the edge of the first to the edge of the
last, so its first sample is centred at `min + spacing / 2`; a
`node`-centred axis has samples at `min` and `max` exactly.

**Writing.** The preferred transformation becomes `space directions` and
`space origin`. When its world is anatomical, it is written in the
`space` of the source header, else in the one it maps to (RAS, LAS or
LPS), else in RAS; the `space` writer option chooses one. An unoriented
world is written with a `space dimension`, but a scaling onto unoriented
axes of an image whose source had no world space goes back to `spacings`
and `axis mins`. An image read from NRRD whose shape has not changed is
written in the axis order, and with the `kinds`, of its file; any other in
its own order, with kinds `domain`, `time`, `vector` and `none`. The
writer options are `encoding` (default: the source's, else `gzip`),
`endian`, `datatype`, `space`, `keyvalue` (merged into the source's; a
value of `None` removes a key) and `data_file` (the data file of a
detached header; default: the header's name with `.raw`, `.raw.gz`,
`.raw.bz2`, `.txt` or `.hex`). The source header's key/value pairs,
`content`, `measurement frame` (converted to the new `space`), `sample
units`, `old min` / `old max`, and its `centers`, `labels`, `thicknesses`
and the `units`, `spacings` and `axis mins` / `maxs` of the non-spatial
axes, are written back.
"""

__all__ = ["AttachedNrrdImage", "DetachedNrrdImage", "NrrdImage"]

from ._image import AttachedNrrdImage, DetachedNrrdImage, NrrdImage
