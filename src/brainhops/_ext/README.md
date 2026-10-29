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

`interpol` is a port of torch-interpol 0.3.0, by Yael Balbastre, from
PyTorch to NumPy and CuPy, and keeps its MIT license in the file
`interpol/LICENSE`. It resamples arrays with B-splines of order 0 to 7,
with one order and one boundary condition per axis, and with batch axes
in the input and in the grid that are processed without a loop in
Python. It also computes spline coefficients that interpolate the
values exactly for every boundary condition, which corrects the zero
and replicate boundaries of the original prefilter. It runs with the
package of its input, so that arrays on a GPU stay on the GPU. brainhops
does not use it yet.
