"""Readers and writers of Neuroglancer coordinate transforms.

Neuroglancer, a viewer of large volumes in the browser, places each data
source of a layer with an affine coordinate transform, a JSON object
written in the state of the viewer as the `transform` of the layer
`source`. This package reads and writes that object on its own, so that
an affine computed by brainhops can be pasted into the state of a
viewer, or read back from it (#227).

The JSON is short and is read in one pass. Its whole content is held as
a [`NeuroglancerTransformRaw`][] record by
[`NeuroglancerTransformMetadata`][], and a [`NeuroglancerTransform`][]
holds that metadata. The matrix of the affine lives in the record, so
the transformation decodes its matrix and its coordinate systems from
the record, and a transformation that is read and written again without
changes is written as its file stored it.

## The conventions of Neuroglancer

The conventions were read in the source of Neuroglancer at
google/neuroglancer@abbdfc5ea97b81c0c0bf04a414f5c59f4d84d27f.

- `src/coordinate_transform.ts`, `coordinateTransformSpecificationFromJson`
  and `coordinateTransformSpecificationToJson`: the transform is an object
  with the keys `outputDimensions`, `inputDimensions`, `matrix` and
  `sourceRank`. Its rank is the number of output dimensions, the input
  dimensions must be as many, and `matrix` is a list of `rank` rows of
  `rank + 1` numbers, the last of which is the translation. A missing
  `matrix` is the identity, and a missing `sourceRank` equals the rank.
- `coordinateSpaceFromJson` in the same file: a dimension is
  `[scale, unit]`, with a finite positive scale, or a coordinate array
  `{"coordinates": [...], "labels": [...]}`, which has no unit and a scale
  of 1. `src/util/si_units.ts` lists the units: `""` (no unit) and `m`,
  `s`, `Hz` and `rad/s` with an SI prefix from `y` to `Y`, including `h`,
  `da`, `d`, `c` and `u` for micro. A prefixed unit is folded into the
  scale by `scaleByExp10`, so `[4, "nm"]` is the same as `[4e-09, "m"]`.
- `isValidDimensionName` in the same file: a name starts with a letter,
  continues with letters, digits and underscores, and may end with `'` (a
  local dimension) or `^` (a channel dimension). An input dimension may
  also be named by a non-negative integer.
- `transformBoundingBox` in the same file, and `scaleTransformSubmatrix`
  in `src/render_coordinate_transform.ts`: the linear coefficients are
  multiplied by the input scale and divided by the output scale, and the
  translation is not. The linear part therefore maps physical input
  coordinates to physical output coordinates, and the translation is in
  multiples of the output scale. `docs/concepts/coordinate_spaces.rst`
  says the same: "the translation coefficients are in the units specified
  for the output (layer) dimension". The matrix maps the source to the
  layer, that is the input to the output.
- `set spec` of `WatchableCoordinateSpaceTransform` in the same file: a
  transform without `inputDimensions` takes its input scales and units
  from the data source, or from `outputDimensions` when the source has
  none of its own. The file alone cannot say what a data source gives, so
  the input is decoded with the units of the output.

Where a voxel lies is not part of the transform. A data source declares
the box of its voxels in its own coordinate space, and the NIfTI source
(`getDataSource` in `src/datasource/nifti/frontend.ts`) declares the box
`[0, size)`, so the voxel of index `i` covers `[i, i + 1)` and has its
centre at `i + 0.5`, which `computeCombinedBounds` in
`src/coordinate_transform.ts` infers from the integer bounds. The
transform maps that space as it is, and adds no half voxel. An affine
whose input is a voxel system of the data model, whose voxel centres are
at integers, would need the scale of the voxels and that half voxel,
which only the data source knows, so such an affine is not encoded.

These conventions were read in the source and were not checked in a
running viewer. The other data sources of Neuroglancer were not read, so
that every source declares integer bounds is not verified.

## The units of the data model

A dimension of the file is decoded as an axis whose unit is the unit of
the dimension with the SI prefix that Neuroglancer would display for its
scale (`pickSiPrefix` in `src/util/si_units.ts`), which is a multiple of
three. A coordinate of the axis is a physical coordinate in that unit, so
the scale of an input dimension changes no coefficient of the decoded
matrix: it says how the data source is scaled, and it stays in the
record. A dimension without a unit is decoded as an axis without a unit.
The coefficients are converted with decimal arithmetic on the shortest
representation of each number, so that a matrix whose units match is
decoded and encoded without any change.

!!! example "Exporting an affine to Neuroglancer"
    ```python
    from brainhops.datamodel.systems import RASmm
    from brainhops.datamodel.transformations import Affine
    from brainhops.io.transformations.neuroglancer import (
        NeuroglancerTransform,
    )

    affine = Affine([[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30]],
                    input=RASmm(), output=RASmm())
    text = NeuroglancerTransform.from_any(affine).to_text()
    ```
"""

__all__ = [
    "NeuroglancerTransform",
    "NeuroglancerTransformMetadata",
    "NeuroglancerTransformRaw",
]

from ._metadata import NeuroglancerTransformMetadata
from ._raw import NeuroglancerTransformRaw
from ._xform import NeuroglancerTransform

# Importing the private converters registers them with the data model,
# so that a conversion into this format, such as `t.to(Format)`, is exact
# or fails with a reason.
from . import _converters  # noqa: E402, F401  isort: skip
