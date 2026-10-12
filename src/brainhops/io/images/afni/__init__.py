"""AFNI datasets.

[`AfniImage`][brainhops.io.images.afni.AfniImage] reads and writes native
AFNI datasets with numpy alone. A dataset is a text header,
`prefix+view.HEAD`, together with a file of voxel values,
`prefix+view.BRIK`, which may be compressed. Either file, or the bare
dataset name, can be loaded. The header format is described in
[`brainhops.io.common.afni`][brainhops.io.common.afni].

```python
import brainhops.io as io

image = io.load("epi+orig.HEAD")       # an AfniImage
image.data                             # [x, y, z, sub-brick], scaled
image.transformation                   # voxel -> "orig", LPS mm (Affine)
record = image.metadata.raw            # the header, as an AfniRaw
record["HISTORY_NOTE"]                 # any attribute of the header
record.labels                          # sub-brick labels (BRICK_LABS)
image.save("copy+orig.BRIK.gz")        # .HEAD + gzipped .BRIK
io.save(image, "epi.nii.gz")           # or any other image format
```

!!! note "One class for every view"
    The view (`+orig`, `+acpc` or `+tlrc`) is a property of the dataset
    rather than a different file format, so one class reads and writes
    every view and names the world space after it.

The array is indexed `[x, y, z]`, or `[x, y, z, sub-brick]` in Fortran
order. The fourth axis is `t` if the dataset has `TAXIS_NUMS`, and `brick`
otherwise. After a read, `raw` is a lazy array of the BRIK, which reads
the voxels only when `data` is first accessed and applies the scale
factors of `BRICK_FLOAT_FACS` to them.

The AFNI world is LPS in millimeters, and an image carries three
transformations, the last of which is preferred:

1. voxel to `physical`, a scaling by `|DELTA|` and by the TR of a time
   series;
2. voxel to `<view>-cardinal`, the cardinal grid defined by
   `ORIENT_SPECIFIC`, `ORIGIN` and `DELTA`;
3. voxel to `<view>`, the possibly oblique geometry of
   `IJK_TO_DICOM_REAL`, which AFNI exports to NIfTI.

When an image is written, the preferred transformation becomes
`IJK_TO_DICOM_REAL` and its closest cardinal grid defines the other
geometry attributes, unless the geometry of the source header is
unchanged. The `view`, `datatype` and `attributes` options control the
view, the stored type and extra header attributes. Valid attributes of
the source header are written back, and a dataset that is read and
written again without changes keeps the bytes of both of its files.
"""

__all__ = ["AfniImage", "AfniMetadata", "AfniRaw"]

from brainhops.io.common.afni import AfniMetadata, AfniRaw

from ._image import AfniImage
