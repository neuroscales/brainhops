r"""Imaris images.

Imaris (`.ims`) is the file format of the Imaris and ImarisViewer
software of Oxford Instruments, common in light-sheet and confocal
microscopy. An Imaris file is an HDF5 file that stores an image at
several resolutions, with time points and channels. The classes of this
module only read files, with `h5py` (`pip install brainhops[imaris]`),
and answer to the format hints `imaris` and `ims`.

| Class                   | Reads                                       |
| ----------------------- | ------------------------------------------- |
| `ImarisMultiScaleImage` | every resolution level of a file, as a pyramid |
| `ImarisImage`           | one resolution level, the finest by default |

```python
import brainhops.io as io

pyramid = io.images.load("cells.ims")       # ImarisMultiScaleImage
pyramid.images[0].data.shape                # (x, y, z, t, c)
pyramid.images[0].transformation            # voxel -> physical, in um
level = io.images.load("cells.ims", level=2)  # ImarisImage
level.metadata.raw.info["Channel 0"]["Name"]  # a parameter of the file
```

A file with a single resolution level is read as an [`ImarisImage`][].
The data of an image always have five axes, `(x, y, z, t, c)`, as the
other brainhops images order space, time and channels, and they are read
lazily, one dataset of the file per channel and time point. An image holds the
groups and parameters of its file as [`ImarisMetadata`][], whose record
is an [`ImarisRaw`][]. The layout and the conventions of the format, and
where they come from, are described in the section on the format below.

Brainhops cannot write Imaris files, so an image is saved to another
format, such as NIfTI or OME-Zarr. A valid Imaris file needs every
resolution level, computed by the binning of Imaris, and the histograms
of every level, and the reference writer is ImarisWriter.

## The format

Imaris was formerly sold by Bitplane. An Imaris file stores an image at
several resolutions so that a viewer can read only the blocks and the
resolution that it shows. The layout below is the one of the
published description of the format, "Imaris5.5 File Format Description
(IMS)", which Oxford Instruments ships as `doc/Imaris5FileFormat.pdf` in
[ImarisWriter](https://github.com/imaris/ImarisWriter) (commit
`b128e6e7d1a147261e9d5caf24ebc6b5c9c63779`). Where the description is
silent, the writer of the same repository (`writer/bpWriterHDF5.cxx`)
and two readers settle the convention: imaris-ims-file-reader
(`imaris_ims_file_reader/ims.py`, commit
`51d8c394ce9a8eeb35421addb9a525cb032ea1fd`) and the `ImarisHDFReader` of
Bio-Formats (commit `7daafcb9b33557a06fbd79d51cedc154ecdd5fce`).

```
/DataSet/ResolutionLevel R/TimePoint T/Channel C   ImageSizeX, Y and Z
/DataSet/ResolutionLevel R/TimePoint T/Channel C/Data   voxels (z, y, x)
/DataSetInfo/Image        ExtMin0..2, ExtMax0..2, Unit
/DataSetInfo/Channel C    Name, Color, ...
/DataSetInfo/TimeInfo     TimePoint1, TimePoint2, ...
```

Each channel of each time point of each resolution level is a separate
three-dimensional `Data` dataset, whose axes are z, y and x, slowest
first: ImarisWriter creates it with the z size first (`WriteDataBlock`
in `writer/bpWriterHDF5.cxx`), and both readers index it as `(z, y, x)`.
The dataset is chunked, and its shape is a whole number of chunks, so it
is padded beyond the image. The size of the image is recorded in the
`ImageSizeX`, `ImageSizeY` and `ImageSizeZ` attributes of the `Channel`
group, and the voxels beyond it are not part of the image. Level 0 is the
finest, and each coarser level halves, rounding down, the axes that are
not much smaller than the others (section 5.1 of the description).

Every parameter is stored as an attribute that holds an array of single
characters, such as `[b"1", b"2", b"8"]` for 128, as the description says
and as ImarisWriter writes them (`WriteAttribute`). The image is placed
in space by its extent, the box from `ExtMin0`, `ExtMin1` and `ExtMin2`
to `ExtMax0`, `ExtMax1` and `ExtMax2` along x, y and z. The description
says that the box lies on the outer border of the voxels on its edge, not
at their centers, so the voxel size is the size of the box divided by the
number of voxels, and the first voxel is centered half a voxel inside the
box. Bio-Formats computes the voxel size in the same way. Every level
spans the same box. The unit of the extent is the `Unit` parameter,
`"m"`, `"mm"`, `"um"` or `"nm"`. ImarisWriter writes `"um"` when it is
not given, and Bio-Formats reads the extent in micrometers, so a file
without a unit is read in micrometers.

The voxels of the description are 8-bit or 16-bit unsigned integers, and
ImarisWriter also writes 32-bit unsigned integers and 32-bit floats. The
`Data` datasets may be compressed with gzip or, since Imaris 9.6, with
LZ4, which h5py reads only when the LZ4 filter plugin of HDF5 is
installed.

!!! warning "Not verified"
    The description gives no text encoding, so the parameters are
    decoded as UTF-8, or as Latin-1 when they are not valid UTF-8.
    ImarisWriter bins two voxels of a level into one voxel of the next
    level and drops the last voxel of an odd axis, while the record holds
    one extent for every level. Brainhops places every level over the
    whole extent, as for the other pyramids, which differs from the
    binning by less than one coarse voxel along an odd axis. How Imaris
    itself places the coarse levels is not documented.
"""

__all__ = [
    "ImarisImage",
    "ImarisLevel",
    "ImarisMetadata",
    "ImarisMultiScaleImage",
    "ImarisRaw",
]

from ._image import ImarisImage
from ._metadata import ImarisMetadata
from ._multiscale import ImarisMultiScaleImage
from ._raw import ImarisLevel, ImarisRaw
