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

OME-NGFF 0.6 names a system by any non-empty string, unique among the
systems of its multiscale, and refers to a system of another group by
the path of that group and the name (ngff-spec tag `0.6` at 89da1da,
`index.md`, sections "coordinateSystems metadata" and "Constraints").
brainhops tells two systems apart by their name and axes, so the systems
of a pyramid that is not in the root of its hierarchy are named after the
path of its group, followed by `//` and their OME name, as in
`labels/cells//physical`. A Zarr node name is never empty and never holds
a `/`, so `//` never occurs in a group path, and the name splits back
unambiguously. The systems of the root group keep their OME names, and
one whose name holds `//` is preceded by `//`. A Zarr group does not know
its path in its hierarchy, so the reader that opened the root gives it to
[`OmeZarrImage`][] as `group`. The writer writes the systems of that
group under their OME names.

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
of starting there, and such a transformation is read as its inverse,
unless it drops axes and so has no inverse, in which case it is refused.

A field transformation, `displacements` or `coordinates`, names a
multiscale group that holds the vectors, with an axis of type
`displacement` or `coordinate` before the spatial axes. Component `i` of
a vector belongs to axis `i` of the output system as OME lists it, so the
components are reordered with the axes of that system. Drafts of version
0.6 named the array of the field instead of a group, and such a field is
read too.

## Vectors and tensors

Some images hold, at each voxel, the components of a vector or a tensor
along one axis, rather than independent channels. The coordinate system
in which these components are expressed is called their frame (see
[`SingleScaleImage.frames`][brainhops.datamodel.images.SingleScaleImage.frames]).
The reader records the frame of each level without converting the
values, and the writer re-expresses the values of each level in the
frame that the store states for it. The types of these axes are read
and written as follows.

| OME-NGFF axis type | brainhops axis      | Frame of the components     |
| ------------------ | ------------------- | --------------------------- |
| `displacement`     | `DisplacementAxis`  | the intrinsic system        |
| `coordinate`       | `CoordinateAxis`    | a foreign coordinate system |
| `vector`           | `VectorAxis`        | the intrinsic system        |
| `tensor`           | `TensorAxis`        | the intrinsic system        |
| `point`            | `PointAxis`         | the intrinsic system        |

The intrinsic system is the system onto which every level maps, and
the components follow its axes in the order in which the store lists
them, such as z, y and x. RFC-3 of OME-NGFF, which lets an image have
any axes, orders the components of a vector in this way (rule 7 of its
changes to the section "multiscales metadata"). It names no type for
such an axis, and OME-NGFF 0.6 allows custom types, so `vector`,
`tensor` and `point` are custom types of brainhops. A placement of the
pyramid carries the intrinsic system to a world, so the components are
not in that world. Each level has its own frame, the map of its voxels
followed by the order of the store, so a vector has the same components
at every level, and a level whose voxels are twice as large has a frame
that scales twice as much.

OME-NGFF 0.6 states that the `i`th entry along the axis of a
displacement field refers to the `i`th output axis of the
transformation that uses the field, and that the field maps its array
onto the input system of that transformation (ngff-spec tag `0.6`,
`index.md`, section "coordinates and displacements"). A field read as
an image does not know that transformation. A displacement has as many
output axes as input axes, because it is added to the input point, so
the reader takes the output axes to be the axes of the system of the
field, in its order. The reader of fields,
[`OmeZarrField`][brainhops.io.transformations.zarr.OmeZarrField], makes
the same assumption when it converts the vectors to voxels. A
coordinate field holds positions in that output system, whose number of
axes may differ and which the field does not name, so its frame is a
foreign [`CoordinateSystem`][brainhops.datamodel.systems.CoordinateSystem]
without a name, which every level shares.

The valence of a vector, and the valence and the layout of a tensor,
have no field in OME-NGFF, so the writer stores them in custom fields
of the axis, which the schema of version 0.6 allows:

| Field     | Axis type | Value                                  |
| --------- | --------- | -------------------------------------- |
| `valence` | `vector`  | `"covector"`                           |
| `valence` | `tensor`  | the valences joined by commas          |
| `layout`  | `tensor`  | the name of a layout, as `"symmetric"` |

A field is written only when it differs from the default: a vector of
valence `"vector"`, a tensor of two indices of valence `"vector"`, and
no layout, with which a tensor reads its layout from the number of its
entries. A tensor whose layout is an
explicit tuple of components has no such field, because abczarr keeps
only scalar values in the custom fields of an axis.

The writer applies a change of frame through affine maps silently,
since it is exact, as for vectors in the voxels of a level. Values in
the frame that a level read from its store are written as they are when
the geometry of the level is written unchanged, and this includes the
coordinates of a coordinate field. The writer refuses, with a
[`LossyConversionError`][brainhops.errors.LossyConversionError], values
in a foreign frame, coordinates of points in another frame than that of
the store, since re-expressing points needs the inverse of a map, a
tensor with an explicit layout, and a change of frame through a map
that is not affine. The argument `lossy=True` writes these values as
they are, or approximately for a map that is not affine.

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
image, is skipped, since labels and scenes are not read yet, and only
the first multiscale of a group is read. A `scale` or a `translation`
whose values are stored in an array named by `path` is refused, since
the schema of version 0.6 requires them inline, as those of versions
0.4 and 0.5 do. Only the drafts 0.6.dev1 and 0.6.dev2 allowed it, and
they are still read. An `affine` or a `rotation` may name an array of
parameters.

The writer encodes the geometry from the levels and the placements, and
does not keep the metadata as it was read, so a pyramid that is saved
unchanged does not keep the bytes of its metadata. It writes
every placement that reduces to an affine as a `scale`, a
`translation`, a `rotation`, an `affine` or a `sequence` of them, so
that a placement read as an inverse is written in the forward direction.
It refuses the other placements, such as a projection, a bijection or a
field, with a [`WriterError`][brainhops.io.base.parsers.WriterError].
Version 0.6 only allows a `scale`, an `identity`, or a `scale` followed
by a `translation` in the dataset of a level. The writer writes the
geometry of a level that only scales and translates each axis in that
form, whatever its spelling. The geometry of a single level that also
rotates or shears becomes the placement of a pyramid that has none, and
the level then keeps a unit scale. Any other level that rotates or
shears is refused with a
[`WriterError`][brainhops.io.base.parsers.WriterError]. Splitting its
geometry into a diagonal part per level and a placement shared by all
levels is not implemented. A flip is never written as
a negative `scale`, which the schema of version 0.6 refuses, but as an
`affine`. The flip that every level shares goes before the placements of
the pyramid, so that each level keeps a positive scale, while levels that
flip different axes are refused.
"""

# The tensions of spec/design/format-records.md number the limitations
# above as items 130 to 135. The split of the geometry of a level is
# issue #447.

__all__ = [
    "OmeImageError",
    "OmeZarrImage",
    "ZarrImage",
    "ZarrMetadata",
    "ZarrRaw",
]

from brainhops.io.common.zarr import ZarrMetadata, ZarrRaw

from ._image import ZarrImage
from ._multiscale import OmeZarrImage
from ._ome import OmeImageError
