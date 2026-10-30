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

## Building a pyramid

The writer writes the levels that a pyramid holds, and nothing more. An
image that has a single resolution gets its coarser levels when it is
converted to `OmeZarrImage`, which happens when it is saved under a name
that ends with `.ome.zarr`. By default, the conversion builds as many
levels as the rules of
[`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales]
allow, by averaging blocks of 2 voxels along each spatial axis until the
axes have 32 voxels or fewer. The keyword `nscales=` takes the values of
the argument of `with_scales`, as in `image.to(OmeZarrImage, nscales=4)`
or `image.to(OmeZarrImage, nscales=[2, 4])`, and `nscales=1` keeps the
single level. The keywords `factor=`, `mode=`, `align=`, `min_size=` and
`strict=` are those of `with_scales`, which `OmeZarrImage` inherits from
[`MultiScaleImage`][brainhops.datamodel.images.MultiScaleImage], and
which builds the levels of a pyramid afterwards. The axes are those that
the writer stores, so they are selected by the names of `axes`.

With the default alignment `"exact"`, a level is placed by the rule of
version 0.6 for a level downsampled by a factor `f`, which puts its
voxel `i` at the base voxel `f * i + (f - 1) / 2`. The alignments
`"edges"` and `"centers"` place it by the actual scale of each axis, as
[`brainhops._core.pyramids`][] explains, and every level is still
written as a scale and a translation.
[`OmeZarrField`][brainhops.io.transformations.zarr.OmeZarrField] builds
the levels of a field in the same way, and builds its metadata again
when it is written.

Every level is written with the chunks of the finest level, which shrink
to the shape of a level that is smaller. These chunks are those given to
`save`, or else those of the store or of the Dask array that the finest
level reads, or else those that the writer chooses for the finest level.
A coarser level that is a Dask array is rechunked to them before it is
written, and the shards, when the levels are sharded, follow the same
rule.

## Limitations

The draft type `inverseOf` was removed before version 0.6 and is
refused. A `byDimension` is read only when its parts are affine, or when
each part writes the axes that it reads. A transformation of the
multiscale that refers to a system of another group, such as a labels
image, is skipped, since labels and scenes are not read yet (items 132
and 133 of the tensions in `spec/design/format-records.md`), and only the
first multiscale of a group is read. A `scale` or a `translation` whose
values are stored in an array named by `path` is refused, since the
schema of version 0.6 requires them inline, as those of versions 0.4 and
0.5 do. Only the drafts 0.6.dev1 and 0.6.dev2 allowed it, and they are
still read. An `affine` or a `rotation` may name an array of parameters
(item 134).

The writer encodes the geometry from the levels and the placements, and
does not keep the metadata as it was read, so a pyramid that is saved
unchanged does not keep the bytes of its metadata (item 130). It writes
every placement that reduces to an affine as a `scale`, a
`translation`, a `rotation`, an `affine` or a `sequence` of them, so
that a placement read as an inverse is written in the forward direction.
It refuses the other placements, such as a projection, a bijection or a
field, with a [`WriterError`][brainhops.io.base.parsers.WriterError].
Version 0.6 only allows a `scale`, an `identity`, or a `scale` followed
by a `translation` in the dataset of a level. The writer writes the
geometry of a level that only scales and translates each axis in that
form, whatever its spelling. When a level also rotates, shears or flips
its axes, the writer splits the geometry of every level into a positive
scale and a translation, which the dataset of the level holds, and a
placement that every level shares, which goes before the placements of
the pyramid (issue #447, item 131). OME-NGFF 0.6 makes the intrinsic
system physical, so the dataset of a level holds the size of its voxels
and the position of its voxels in physical units. The size of a voxel
along an axis is the length of the column of that axis in the linear
part of the geometry, which is the distance between two neighbouring
voxels along the axis, even when the axes are sheared. A polar or a QR
decomposition would not give that distance. The shared placement holds
the rest of the geometry of the finest level: its linear part with
columns of unit length, which only rotates, shears or flips the axes,
and the position of its first voxel. It is therefore a map between two
physical spaces. The translation of a coarser level is the offset of
its first voxel from that of the finest level, which is `(f - 1) / 2`
voxels of the finest level for a level downsampled by a factor `f`. A
single level is split in the same way, so that a rotated image with
voxels of 0.5 mm is written with a scale of 0.5 and a rotation.

The split exists when every level is rotated, sheared and flipped in the
same way, so that the levels differ only by a positive scale and a
translation along each axis. The columns of their matrices are compared
with a relative tolerance of 1e-6, which accepts geometries stored in
single precision. Levels whose geometries differ in any other way, such
as levels that flip different axes, are refused with a
[`WriterError`][brainhops.io.base.parsers.WriterError]. A flip is never
written as a negative `scale`, which the schema of version 0.6 refuses,
but as an `affine` (item 135). A pyramid that is read back keeps the
geometry of each level, but not necessarily the way in which it was
split between the level and the placements.
"""

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
