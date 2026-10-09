"""Regenerate the elastix fixtures stored next to this script.

The script needs itk-elastix, which the tests themselves do not require.
It can be run from any directory::

    python tests/data/elastix/generate_elastix_fixtures.py

Each case is a parameter file written by elastix itself and an archive
`<name>_expected.npz` with the transformix result: `vox2lps`, the (D, D+1)
voxel-to-LPS affine of the deformation field, and `disp`, the (X, Y[, Z], D)
displacement in LPS millimeters. The parameters are deliberately awkward:
an asymmetric fixed direction, centers of rotation away from the origin,
and B-spline grids with a nonzero `GridIndex` and an oblique
`GridDirection`.
"""

import math
import os
import tempfile
from pathlib import Path

import itk
import numpy as np

HERE = Path(__file__).parent


def rotation3d(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ rx @ ry


def rotation2d(a: float) -> np.ndarray:
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


# Oblique, non-symmetric and flipping the last axis, so that a transposed
#
# direction is detected.
DIRECTION = {
    2: rotation2d(0.3) @ np.diag([1, -1]),
    3: rotation3d(0.2, -0.1, 0.4) @ np.diag([1, 1, -1]),
}


def column_major(matrix: np.ndarray) -> list:
    """Flatten a matrix column by column, as elastix stores `Direction`."""
    return [float(v) for v in np.asarray(matrix).T.ravel()]


def fmt(values: list) -> list:
    return [
        repr(float(v)) if not float(v).is_integer() else str(int(v))
        for v in values
    ]


def fixed_map(ndim: int) -> dict:
    """Return the fixed-image part of a parameter map."""
    size = [4, 5, 6][:ndim]
    spacing = [1.5, 2.0, 2.5][:ndim]
    origin = [-3.0, 4.0, 5.5][:ndim]
    return {
        "FixedImageDimension": [str(ndim)],
        "MovingImageDimension": [str(ndim)],
        "FixedInternalImagePixelType": ["float"],
        "MovingInternalImagePixelType": ["float"],
        "Size": [str(s) for s in size],
        "Index": ["0"] * ndim,
        "Spacing": fmt(spacing),
        "Origin": fmt(origin),
        "Direction": fmt(column_major(DIRECTION[ndim])),
        "UseDirectionCosines": ["true"],
        "HowToCombineTransforms": ["Compose"],
        "InitialTransformParameterFileName": ["NoInitialTransform"],
        "ResampleInterpolator": ["FinalBSplineInterpolator"],
        "FinalBSplineInterpolationOrder": ["3"],
        "Resampler": ["DefaultResampler"],
        "DefaultPixelValue": ["0"],
        "ResultImageFormat": ["nii"],
        "ResultImagePixelType": ["float"],
        "CompressResultImage": ["false"],
    }


def transform_map(ndim: int, name: str, params: list, **extra) -> dict:
    pmap = fixed_map(ndim)
    pmap["Transform"] = [name]
    pmap["NumberOfParameters"] = [str(len(params))]
    pmap["TransformParameters"] = fmt(params)
    for key, value in extra.items():
        pmap[key] = value if isinstance(value, list) else [value]
    return pmap


def bspline_map(
    ndim: int, degree: int, seed: int, grid_size: tuple = (10, 11, 12)
) -> dict:
    """Return a B-spline map whose valid region covers the fixed grid.

    Elastix evaluates a B-spline only where the whole spline support lies
    inside the control grid, so the fixed grid is placed well inside it.
    """
    rng = np.random.default_rng(seed)
    grid_size = list(grid_size)[:ndim]
    grid_index = [1, -2, 0][:ndim]
    grid_spacing = [2.0, 2.5, 3.0][:ndim]
    direction = rotation2d(-0.2) if ndim == 2 else rotation3d(0.1, 0.2, -0.3)
    # Centre the control grid on the center of the fixed grid.
    fixed = fixed_map(ndim)
    size = np.array([float(v) for v in fixed["Size"]])
    spacing = np.array([float(v) for v in fixed["Spacing"]])
    origin = np.array([float(v) for v in fixed["Origin"]])
    center = origin + DIRECTION[ndim] @ (spacing * (size - 1) / 2)
    mid = np.array(grid_index) + (np.array(grid_size) - 1) / 2
    grid_origin = center - direction @ (np.array(grid_spacing) * mid)
    ncoeff = ndim * int(np.prod(grid_size))
    params = rng.uniform(-2.0, 2.0, ncoeff).round(3).tolist()
    return transform_map(
        ndim,
        "BSplineTransform",
        params,
        GridSize=[str(v) for v in grid_size],
        GridIndex=[str(v) for v in grid_index],
        GridSpacing=fmt(grid_spacing),
        GridOrigin=fmt(grid_origin.round(6).tolist()),
        GridDirection=fmt(column_major(direction)),
        BSplineTransformSplineOrder=str(degree),
        UseCyclicTransform="false",
    )


def cases() -> dict:
    """Return all single-file cases, keyed by name."""
    c3 = ["4.5", "-2", "7.25"]
    c2 = ["4.5", "-2"]
    return {
        "translation3d": transform_map(
            3, "TranslationTransform", [1.5, -2.0, 3.25]
        ),
        "euler2d": transform_map(
            2, "EulerTransform", [0.3, 1.5, -2.0], CenterOfRotationPoint=c2
        ),
        "euler3d": transform_map(
            3,
            "EulerTransform",
            [0.1, -0.2, 0.3, 1.5, -2.0, 3.25],
            CenterOfRotationPoint=c3,
            ComputeZYX="false",
        ),
        "euler3d_zyx": transform_map(
            3,
            "EulerTransform",
            [0.1, -0.2, 0.3, 1.5, -2.0, 3.25],
            CenterOfRotationPoint=c3,
            ComputeZYX="true",
        ),
        "similarity2d": transform_map(
            2,
            "SimilarityTransform",
            [1.2, 0.3, 1.5, -2.0],
            CenterOfRotationPoint=c2,
        ),
        "similarity3d": transform_map(
            3,
            "SimilarityTransform",
            [0.1, -0.2, 0.15, 1.5, -2.0, 3.25, 0.9],
            CenterOfRotationPoint=c3,
        ),
        "affine2d": transform_map(
            2,
            "AffineTransform",
            [1.1, 0.2, -0.1, 0.9, 1.5, -2.0],
            CenterOfRotationPoint=c2,
        ),
        "affine3d": transform_map(
            3,
            "AffineTransform",
            [1.1, 0.2, -0.1, 0.05, 0.9, 0.15, -0.2, 0.1, 1.2, 1.5, -2.0, 3.25],
            CenterOfRotationPoint=c3,
        ),
        "affinelog3d": transform_map(
            3,
            "AffineLogTransform",
            [
                0.1,
                0.2,
                -0.1,
                -0.15,
                -0.05,
                0.1,
                0.2,
                -0.1,
                0.15,
                1.5,
                -2.0,
                3.25,
            ],
            CenterOfRotationPoint=c3,
        ),
        "affinedti2d": transform_map(
            2,
            "AffineDTITransform",
            [0.3, 0.1, -0.2, 1.1, 0.9, 1.5, -2.0],
            CenterOfRotationPoint=c2,
        ),
        "affinedti3d": transform_map(
            3,
            "AffineDTITransform",
            [0.1, -0.2, 0.3, 0.1, -0.05, 0.2, 1.1, 0.9, 1.05, 1.5, -2.0, 3.25],
            CenterOfRotationPoint=c3,
        ),
        "bspline3d": bspline_map(3, 3, seed=0),
        "bspline3d_order1": bspline_map(3, 1, seed=1),
        "bspline3d_order2": bspline_map(3, 2, seed=2),
        "bspline2d": bspline_map(2, 3, seed=3),
    }


def transformix(maps: list, ndim: int) -> dict:
    """Run transformix on a chain of maps, applied first to last."""
    parameter_object = itk.ParameterObject.New()
    for pmap in maps:
        parameter_object.AddParameterMap(pmap)
    moving = itk.image_from_array(np.zeros((6,) * ndim, np.float32))
    field = itk.transformix_deformation_field(moving, parameter_object)
    disp = itk.array_from_image(field).astype(np.float64)
    disp = disp.transpose(*reversed(range(ndim)), ndim)
    direction = np.asarray(field.GetDirection())
    spacing = np.asarray(field.GetSpacing())
    origin = np.asarray(field.GetOrigin())
    vox2lps = np.concatenate(
        [direction @ np.diag(spacing), origin[:, None]], axis=1
    )
    return {"vox2lps": vox2lps, "disp": disp}


def write(name: str, maps: list, files: dict, suffix: str = ".txt") -> None:
    """Write the parameter files and expected transformix output of a case."""
    ndim = int(maps[-1]["FixedImageDimension"][0])
    for filename, pmap in files.items():
        _write_map(pmap, HERE / filename)
    np.savez(HERE / f"{name}_expected.npz", **transformix(maps, ndim))


def _object(pmap: dict) -> "itk.ParameterObject":
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(pmap)
    return parameter_object


def _write_map(pmap: dict, filename: Path) -> None:
    # The file extension selects the text or TOML syntax.
    _object(pmap).WriteParameterFile(str(filename))


def main() -> None:
    # Transformix writes its deformation field into the working directory.
    os.chdir(tempfile.mkdtemp())
    for name, pmap in cases().items():
        write(name, [pmap], {f"{name}.txt": pmap})

    # The same Euler file in the elastix TOML syntax.
    euler = cases()["euler3d"]
    write("euler3d_toml", [euler], {"euler3d.toml": euler})

    # An affine, then a B-spline naming it as initial transform by a path
    #
    # relative to the B-spline file.
    affine = cases()["affine3d"]
    # Enlarge the B-spline grid so that it still covers the moved points.
    bspline = bspline_map(3, 3, seed=4, grid_size=(16, 14, 18))
    bspline["InitialTransformParameterFileName"] = [
        "chain_TransformParameters.0.txt"
    ]
    write(
        "chain",
        [affine, bspline],
        {
            "chain_TransformParameters.0.txt": affine,
            "chain_TransformParameters.1.txt": bspline,
        },
    )

    # Three-link chain using the deprecated key spelling
    #
    # `InitialTransformParametersFileName`.
    translation = cases()["translation3d"]
    euler = dict(cases()["euler3d"])
    euler["InitialTransformParameterFileName"] = [
        "chain3_TransformParameters.0.txt"
    ]
    similarity = dict(cases()["similarity3d"])
    del similarity["InitialTransformParameterFileName"]
    similarity["InitialTransformParametersFileName"] = [
        "chain3_TransformParameters.1.txt"
    ]
    write(
        "chain3",
        [translation, euler, similarity],
        {
            "chain3_TransformParameters.0.txt": translation,
            "chain3_TransformParameters.1.txt": euler,
            "chain3_TransformParameters.2.txt": similarity,
        },
    )


if __name__ == "__main__":
    main()
