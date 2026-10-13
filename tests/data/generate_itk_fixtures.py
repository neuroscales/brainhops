"""Regenerate the deterministic ITK fixtures stored next to this script.

The script needs SimpleITK, which the tests themselves do not require.
It can be run from any directory::

    python tests/data/generate_itk_fixtures.py

Each fixture is a transform file and a `<name>_expected.npy` file with the
values reported by SimpleITK. The values are not random, because a wrong
memory layout or angle order would still give plausible random arrays.
The warp fixtures carry the ramp::

    value = 1000 * component + 100 * x + 10 * y + z

so that a transposed axis or a wrong component stride is visible at a
glance. The warp grids have unit spacing and an identity direction.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).parent

# The shape is in ITK (x, y, z) order, and its distinct lengths make axis
# swaps visible.
SHAPE = (2, 3, 4)


def ramp(x: int, y: int, z: int) -> list:
    """Return the ramp value of each component at a grid point."""
    return [1000 * c + 100 * x + 10 * y + z for c in range(3)]


def write_displacement_field() -> None:
    """Write a dense displacement field whose values name their voxel.

    The expected array comes from the image view of SimpleITK, independently
    of the parameter layout.
    """
    image = sitk.Image(SHAPE, sitk.sitkVectorFloat64, 3)
    for z in range(SHAPE[2]):
        for y in range(SHAPE[1]):
            for x in range(SHAPE[0]):
                image[x, y, z] = ramp(x, y, z)

    transform = sitk.DisplacementFieldTransform(sitk.Image(image))
    sitk.WriteTransform(transform, str(HERE / "itk_displacement_ramp3d.tfm"))

    # `GetArrayFromImage` returns (z, y, x, component); the field is stored as
    # (x, y, z, component).
    expected = sitk.GetArrayFromImage(image).transpose(2, 1, 0, 3)
    np.save(HERE / "itk_displacement_ramp3d_expected.npy", expected)


def write_bspline() -> None:
    """Write a B-spline whose coefficients name their control point.

    The expected array comes from the coefficient images of SimpleITK,
    independently of the parameter layout.
    """
    # A cubic spline pads a (1, 2, 3) mesh to a (4, 5, 6) grid.
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
    """Write a rigid 3-D Euler transform in either ITK angle order.

    The angles are distinct and the center lies away from the origin, so that
    errors in composition order or center folding cannot cancel out. The
    expected array is the compact (3, 4) affine.
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


def write_generic_affine(ndim: int) -> None:
    """Write an affine in the binary MATLAB format written by ANTs.

    This is the `<prefix>0GenericAffine.mat` file of `antsRegistration`. All
    matrix entries differ and the center lies away from the origin, so that a
    transposed matrix or a dropped center is detected.
    """
    matrix = np.arange(1, ndim * ndim + 1, dtype=np.float64) / 10
    matrix = matrix.reshape(ndim, ndim) + np.eye(ndim)
    transform = sitk.AffineTransform(ndim)
    transform.SetMatrix(matrix.ravel().tolist())
    transform.SetTranslation([1.0, 2.0, 3.0][:ndim])
    transform.SetCenter([4.0, 5.0, 6.0][:ndim])
    name = f"itk_affine{ndim}d_0GenericAffine"
    sitk.WriteTransform(transform, str(HERE / f"{name}.mat"))

    expected = np.zeros((ndim, ndim + 1), dtype=np.float64)
    expected[:, :ndim] = np.asarray(transform.GetMatrix()).reshape(ndim, ndim)
    expected[:, ndim] = transform.TransformPoint([0.0] * ndim)
    np.save(HERE / f"{name}_expected.npy", expected)


def main() -> None:
    write_displacement_field()
    write_bspline()
    write_euler3d(compute_zyx=False, name="itk_euler3d")
    write_euler3d(compute_zyx=True, name="itk_euler3d_zyx")
    write_generic_affine(3)
    write_generic_affine(2)


if __name__ == "__main__":
    main()
