"""
Reading and writing of BigDataViewer datasets.

BigDataViewer is the Fiji viewer of large, multi-view and multi-scale
images, and BigStitcher, the Fiji plugin that stitches and registers
light-sheet acquisitions, stores its results in the same dataset. A
dataset is an XML file, usually `dataset.xml`, and an HDF5 file that
holds the voxels. This package reads and writes the XML file, as a
[`BdvRaw`][] record held by [`BdvMetadata`][]. The images of a dataset
are read by [`brainhops.io.images.bdv`][], and the registration of a view
is read and written by [`brainhops.io.transformations.bdv`][].

The conventions below were checked against the code of
bigdataviewer-core (commit dfe1a66), spimdata (commit 84728ff),
imglib2-realtransform (commit ca9039d), bigdataviewer_fiji (commit
c154fb2), multiview-reconstruction, the code of BigStitcher (commit
60243ba), n5 (commit 7ba6b14) and jdom (commit c6065f6).

## The XML file

```text
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="bdv.hdf5">
      <hdf5 type="relative">dataset.h5</hdf5>
    </ImageLoader>
    <ViewSetups>
      <ViewSetup>
        <id>0</id>
        <name>0</name>
        <size>1024 1024 200</size>          x, y, z
        <voxelSize>
          <unit>µm</unit>
          <size>0.4 0.4 2.0</size>          x, y, z
        </voxelSize>
      </ViewSetup>
    </ViewSetups>
    <Timepoints type="range">               or "list", or "pattern"
      <first>0</first>
      <last>0</last>
    </Timepoints>
    <MissingViews/>                         optional
  </SequenceDescription>
  <ViewRegistrations>
    <ViewRegistration timepoint="0" setup="0">
      <ViewTransform type="affine">
        <Name>calibration</Name>            optional
        <affine>1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 5.0 0.0</affine>
      </ViewTransform>
    </ViewRegistration>
  </ViewRegistrations>
</SpimData>
```

A view is one setup (a channel, an angle, an illumination or a tile) at
one time point. The tags are those of `XmlKeys.java` in spimdata. A path
is relative only when its `type` attribute is `"relative"`
(`XmlHelpers.loadPath`): `<BasePath>` is then relative to the directory
of the XML file, and `<hdf5>` to the base path. BigDataViewer writes the
file with JDOM's pretty format, which indents by two spaces and ends its
lines with CRLF.

## Registrations

Each `<ViewTransform>` of type `affine` holds twelve values, the first
three rows of a 3D affine matrix, row after row
(`XmlHelpers.getAffineTransform3D` and `AffineTransform3D.set` in
imglib2-realtransform). A transform of type `generic` names a Java class
that reads its own content, and brainhops keeps it without reading it.

The transforms of a view map the coordinates of its voxels to the global
coordinates of the dataset. `ViewRegistration.updateModel` concatenates
them in the order of the file, and `AffineTransform3D.concatenate(b)`
sets `a = a * b`, so the map is `T0 * T1 * ... * Tn`: the last transform
of the file is applied first, and the first one last. BigStitcher adds a
transform at the front of the list with `preconcatenateTransform`, so its
first transform, named `calibration`, ends up last.

Voxel coordinates are in the order x, y, z, and an integer coordinate is
the center of a voxel, which is the convention of ImgLib2. The default
transform of a coarser level (`MipmapTransforms.getMipmapTransformDefault`)
confirms it: it places voxel 0 of a level downsampled by `f` at `(f - 1)
/ 2` in full-resolution voxels, at the center of the block that it
averages.

## Units and the global space

The voxel size and its unit are given by each setup. The global space
has no unit of its own. A dataset exported by the Fiji plugin
"Export Current Image as XML/HDF5" has a single transform, the voxel
size (`ExportImagePlusPlugIn`), so its global space is in the unit of
the voxel size. BigStitcher instead stores as `calibration` the voxel
size divided by the smallest voxel size of the dataset
(`DatasetCreationUtils.createViewRegistrations`), so its global space is
in units of that smallest voxel size. BigDataViewer itself measures the
global space through the voxel size of the source along x and the scale
of its transform along x (`ScaleBarOverlayRenderer`). Brainhops therefore
gives the global space no unit.

## The HDF5 file

```text
/s00/resolutions        (levels, 3) float64: downsampling of each level,
                        x, y, z
/s00/subdivisions       (levels, 3) int32: chunk shape of each level,
                        x, y, z
/s00 attrs: dataType    "uint8", "int16", "float32", ...; absent in older
                        files
/t00000/s00/0/cells     the voxels of time point 0, setup 0, level 0,
                        stored z, y, x
```

The group names follow `bdv/img/hdf5/Util.java`: `s%02d` for a setup and
`t%05d/s%02d/%d/cells` for a level of a view. The arrays of shapes and
chunks are reversed from the order x, y, z of ImgLib2 into the order of
HDF5 (`Util.reorder`). A file without the `dataType` attribute stores
16-bit unsigned voxels as signed 16-bit integers, which BigDataViewer
reads back with the same bits as unsigned values
(`Hdf5ImageLoader.createSetupImgLoader`). A dataset split into partitions
has a main file whose `cells` datasets are external links to the
partition files (`WriteSequenceToHdf5.writeHdf5PartitionLinkFile`), and
BigDataViewer, like brainhops, reads the main file only.
"""

__all__ = [
    "BdvMetadata",
    "BdvRaw",
]

from ._metadata import BdvMetadata
from ._raw import BdvRaw
