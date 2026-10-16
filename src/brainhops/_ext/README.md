This folder holds modules that are more general than brainhops but are
used by it. Each of them is meant to become a standalone package one day,
and a dependency of brainhops. They are kept here while their interfaces
still change too often to be published separately.

`invfield` is a compact implementation of John Ashburner's inversion of
displacement fields, in two and three dimensions. It treats the field as
a mesh in which each cell defines an affine transformation, and inverts
that piecewise-affine map exactly. It was written independently from the
published papers and is not a port of the SPM code. brainhops uses it to
invert displacement and coordinate fields.
