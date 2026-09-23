"""Regenerate the deterministic ITK fixtures in this directory.

Run it with SimpleITK installed, from anywhere:

    python tests/data/generate_itk_fixtures.py

It writes the transform files that pin ITK's *layout* and *conventions*,
together with a `<name>_expected.npy` of the values SimpleITK itself
reports for each one. The tests assert against those stored arrays, so
the suite needs SimpleITK only to regenerate the fixtures, never to run.

The values are deliberately not random. A displacement field decoded in
the wrong memory layout, or Euler angles composed in the wrong order,
still produce a well-formed array of plausible numbers -- the error only
shows against values whose every entry says where it came from. Each
warp fixture therefore carries the ramp

    value = 1000 * component + 100 * x + 10 * y + z

so a transposed axis or a mistaken component stride is legible at a
glance, and each warp grid is left with unit spacing and an identity
direction so that the world-space displacements SimpleITK reports are
already in the grid's own voxel units.
"""

# stdlib
from pathlib import Path

# dependencies
import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).parent

#: The grid shape, in ITK's own (x, y, z) order. Small enough that the
#: `.tfm` files stay a few hundred bytes, big enough that no two axes
#: have the same length -- an axis swap between equal-length axes is
#: invisible.
SHAPE = (2, 3, 4)


def ramp(x: int, y: int, z: int) -> list:
    """The decodable value of each component at one grid point."""
    return [1000 * c + 100 * x + 10 * y + z for c in range(3)]


def write_displacement_field() -> None:
    """A dense displacement field whose every value names its voxel.

    ITK hands back the parameters of a `DisplacementFieldTransform` as
    the raw buffer of an image of vectors, so the components of a voxel
    are adjacent. The expected array is taken from SimpleITK's own view
    of the image, not from the parameter vector, so it is independent of
    how that buffer is laid out.
    """
    image = sitk.Image(SHAPE, sitk.sitkVectorFloat64, 3)
    for z in range(SHAPE[2]):
        for y in range(SHAPE[1]):
            for x in range(SHAPE[0]):
                image[x, y, z] = ramp(x, y, z)

    transform = sitk.DisplacementFieldTransform(sitk.Image(image))
    sitk.WriteTransform(transform, str(HERE / "itk_displacement_ramp3d.tfm"))

    # `GetArrayFromImage` is (z, y, x, component); the field is stored
    # (x, y, z, component).
    expected = sitk.GetArrayFromImage(image).transpose(2, 1, 0, 3)
    np.save(HERE / "itk_displacement_ramp3d_expected.npy", expected)


def write_bspline() -> None:
    """A B-spline whose coefficients name their control point.

    Unlike a dense field, a B-spline's parameters are one scalar
    coefficient image per axis, written back to back. The expected array
    is stacked from the coefficient images SimpleITK exposes, so it too
    is independent of the parameter layout.
    """
    # A cubic spline pads its mesh with three control points per axis,
    # so a (1, 2, 3) mesh gives this (4, 5, 6) control grid -- again no
    # two axes the same length, and still tiny.
    grid = (4, 5, 6)
    images = []
    for component in range(3):
        image = sitk.Image(grid, sitk.sitkFloat64)
        for z in range(grid[2]):
            for y in range(grid[1]):
                for x in range(grid[0]):
                    image[x, y, z] = ramp(x, y, z)[component]
        images.append(image)

    transform = sitk.BSplineTransform(images, 3)
    sitk.WriteTransform(transform, str(HERE / "itk_bspline_ramp3d.tfm"))

    expected = np.stack(
        [
            sitk.GetArrayFromImage(image).transpose(2, 1, 0)
            for image in transform.GetCoefficientImages()
        ],
        axis=-1,
    )
    np.save(HERE / "itk_bspline_ramp3d_expected.npy", expected)


def write_euler3d(compute_zyx: bool, name: str) -> None:
    """A rigid Euler 3-D transform, in one of ITK's two angle orders.

    The three angles differ from one another and the center is away from
    the origin, so neither the composition order nor the folding of the
    center can cancel out. ITK >= 5 writes the `ComputeZYX` flag as a
    fourth fixed parameter, which is what makes both files four-long.

    The expected array is the compact `(3, 4)` affine that ITK itself
    describes: its rotation from `GetMatrix`, and its offset read off
    `TransformPoint` at the origin, which is where the center folds in.
    """
    transform = sitk.Euler3DTransform()
    transform.SetComputeZYX(compute_zyx)
    transform.SetCenter((4.0, 5.0, 6.0))
    transform.SetRotation(0.1, 0.2, 0.3)
    transform.SetTranslation((1.0, 2.0, 3.0))
    sitk.WriteTransform(transform, str(HERE / f"{name}.tfm"))

    expected = np.zeros((3, 4), dtype=np.float64)
    expected[:, :3] = np.asarray(transform.GetMatrix()).reshape(3, 3)
    expected[:, 3] = transform.TransformPoint((0.0, 0.0, 0.0))
    np.save(HERE / f"{name}_expected.npy", expected)


def main() -> None:
    write_displacement_field()
    write_bspline()
    write_euler3d(compute_zyx=False, name="itk_euler3d")
    write_euler3d(compute_zyx=True, name="itk_euler3d_zyx")


if __name__ == "__main__":
    main()
