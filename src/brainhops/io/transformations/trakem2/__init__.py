"""Readers and writers of the affines of TrakEM2 patches.

TrakEM2 is the Fiji plugin for serial-section electron microscopy. It
saves a project as an XML file, in which a layer set holds layers, one
for each section, and each layer holds patches, which are the images of
the section. Each patch is placed in the plane of its layer by a 2D
affine, which brainhops reads as a
[`TrakEM2PatchAffine`][]. The whole XML file is held as a
[`TrakEM2Raw`][] record by [`TrakEM2Metadata`][], and a patch whose
affine is changed is written back by replacing the value of its
`transform` attribute only, so the rest of the file is kept byte for
byte.

The conventions below were read in the source of TrakEM2 at commit
cdd8b8f (github.com/trakem2/TrakEM2), of trakem2-transform at commit
5a92a82 (github.com/trakem2/trakem2-transform) and of mpicbg at commit
fc319fc (github.com/axtimwalde/mpicbg).

- The `transform` attribute of a `<t2_patch>` is written as
  `matrix(a,b,c,d,e,f)` from `java.awt.geom.AffineTransform.getMatrix`,
  so its values are `m00 m10 m01 m11 m02 m12`, and the affine maps
  `(x, y)` to `(m00 x + m01 y + m02, m10 x + m11 y + m12)`
  (`Displayable.exportXML` and the `Displayable` constructor in
  `ini/trakem2/display/Displayable.java`).
- The affine maps the local coordinates of the patch to the coordinates
  of its layer. Local coordinates are pixels of the image, with integers
  at the centers of the pixels: TrakEM2 shifts the image by half a pixel
  before AWT draws it, because AWT puts integers at the corners
  (`Patch.paint` in `ini/trakem2/display/Patch.java`).
- The coordinates of a layer are in pixels of the layer set. The z and
  the thickness of a `<t2_layer>` are in the same pixels, and TrakEM2
  multiplies x by `pixelWidth`, y by `pixelHeight` and z by `pixelWidth`,
  not `pixelDepth`, of the `<t2_calibration>` of the layer set to give
  calibrated coordinates in its `unit` (`Layer.getCalibratedZ` in
  `ini/trakem2/display/Layer.java` and the distances in
  `ini/trakem2/display/Tree.java`). The origins of the calibration are
  written but not used there.
- A patch may also have a coordinate transform, written inside it as an
  `<ict_transform>`, `<iict_transform>`, `<ict_transform_list>` or
  `<iict_transform_list>` element with a Java `class` and its `data`
  (`TMLHandler.makeCoordinateTransform` in
  `ini/trakem2/persistence/TMLHandler.java`), or kept in a separate file
  named by its `ct_id`. TrakEM2 applies it to the pixels first, then
  shifts the result by the corner of the bounding box of a mesh, then
  applies the affine (`Patch.getFullCoordinateTransform`).

The affine is read as a 2D affine, from the pixels of the patch to the
plane of its layer. It is not placed in 3D at the z of the layer, because
a patch has no third axis, and the scale and the origin of that axis,
which a 3D affine would need, are not defined by TrakEM2. The z, the
thickness and the calibration are kept in the record of each patch.
Only an affine or a similarity model written inside the patch is decoded
as its coordinate transform, and it is not composed with the affine,
because brainhops does not compute the bounding box of the mesh. The
other models, the lists and the transforms in separate files are kept in
the file as they are, and are refused when they are decoded.

It is not verified whether TrakEM2 projects written by versions older
than commit cdd8b8f use other layouts. A title that holds `&` or `<`,
which TrakEM2 writes without escaping, makes the file malformed XML, and
such a file is refused.
"""

__all__ = [
    "TrakEM2Metadata",
    "TrakEM2PatchAffine",
    "TrakEM2Raw",
]

from ._metadata import TrakEM2Metadata
from ._raw import TrakEM2Raw
from ._xform import TrakEM2PatchAffine

# Importing the private converters registers them with the data model,
# so that a conversion into this format, such as `t.to(Format)`, finds
# them.
from . import _converters  # noqa: E402, F401  isort: skip
