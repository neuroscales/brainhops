"""Zarr and OME-Zarr images.

A plain Zarr array is read as a [`ZarrImage`][] without a world geometry,
and an OME-Zarr group, which stores a pyramid of resolution levels with
its metadata, is read as an [`OmeZarrImage`][]. The metadata is parsed by
abczarr and normalized to OME-NGFF 0.6, the version that adopted RFC-5,
whatever version the store uses.

## Coordinate systems

OME-NGFF 0.6 names its coordinate systems and gives each of them a list
of axes, with a name, a type such as `space` or `time`, and a unit such
as `micrometer`. Every named system of a multiscale becomes a brainhops
[`CoordinateSystem`][brainhops.datamodel.systems.CoordinateSystem] of the
same name, whose axes are sorted into the brainhops order: the spatial
axes first, as x, y and z, then time, channels and the other axes. Each
system is sorted on its own, so two systems that list the same axes in
different orders end up with the same brainhops order. The levels map
their voxels onto the intrinsic system, which every level names as its
output, and the transformations declared on the whole multiscale carry
the intrinsic system to the other systems. They become the placements in
`transformations`, whose last entry is the preferred one.

## Transformations

A transformation maps points of its input system to points of its
output system, in OME-NGFF 0.6 as in brainhops (ngff-spec, branch `0.6`
at 89da1da, `index.md`, section "coordinateTransformations metadata").
Its parameters follow the order in which its systems list their axes,
which for an image is the order of the array dimensions, and they are
reordered when they are read (same file, sections "Additional details"
and "Matrix transformations"). The types are read as follows.

| OME-NGFF 0.6                   | brainhops                       |
| ------------------------------ | ------------------------------- |
| `identity`                     | `Identity`                      |
| `scale`                        | `Scaling`                       |
| `translation`                  | `Translation`                   |
| `rotation`                     | `Rotation`                      |
| `affine`                       | `Affine`                        |
| `mapAxis`                      | `Permutation`                   |
| `projectAxis`                  | `Projection`                    |
| `sequence`                     | `Sequence`                      |
| `displacements`, `coordinates` | the field of the group it names |
| `bijection`                    | `Bijection`                     |
| `byDimension`                  | `Affine`, or subspace parts     |

When the two systems of a transformation sort their axes differently, a
type that cannot express the reordering, such as a scale, is followed by
a `Permutation`, so that the result stays exact. A `byDimension` whose
parts are not all affine becomes a `Sequence` of
`SubspaceTransformation` objects. Version 0.6 also allows a
transformation of the multiscale to end in the intrinsic system instead
of starting there, and such a transformation is read as its inverse.

A field transformation, `displacements` or `coordinates`, names a
multiscale group that holds the vectors, with an axis of type
`displacement` or `coordinate` before the spatial axes. Component `i` of
a vector belongs to axis `i` of the output system as OME lists it, so the
components are reordered with the axes of that system. Drafts of version
0.6 named the array of the field instead of a group, and such a field is
read too.

## Voxel centres

An integer index is the centre of a voxel. OME-NGFF 0.6 states this
convention in its section "Coordinate convention", and brainhops uses
the same one, as NIfTI and ITK do, so the transformations are read and
written without a shift of half a voxel. Versions 0.4 and 0.5 do not
state a convention. They are read with the same one, which agrees with
the note of version 0.6 that a level downsampled by a factor `f` is
translated by `(f - 1) / 2` base voxels, and with ngff-zarr (5187d94),
whose module `ngff_transform_to_itk_transform` states that both OME and
ITK put the centre of a voxel at its integer index. Whether every writer
of versions 0.4 and 0.5 follows it is not verified.

## Limitations

The draft type `inverseOf` was removed before version 0.6 and is
refused. A `byDimension` is read only when its parts are affine, or when
each part writes the axes that it reads. A transformation of the
multiscale that refers to a system of another group, such as a labels
image, is skipped, and only the first multiscale of a group is read.

The writer encodes the geometry from the levels and the placements, and
does not keep the metadata as it was read, so a pyramid that is saved
unchanged does not keep the bytes of its metadata. The writer also puts
the geometry of each level in its dataset as it is. Version 0.6 only
allows a `scale`, or a `scale` followed by a `translation`, there, so a
level that is rotated or sheared is written as an `affine` that the
official schema of version 0.6 rejects.
"""

__all__ = ["ZarrImage", "OmeZarrImage", "OmeImageError"]

from ._image import ZarrImage
from ._multiscale import OmeZarrImage
from ._ome import OmeImageError
