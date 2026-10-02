"""Regenerate the elastix fixtures in this directory.

Run it with ITK-Elastix installed (`pip install itk-elastix`), from
anywhere:

    python tests/data/elastix/generate_elastix_fixtures.py

For every case it writes a transform parameter file *as elastix itself
writes it* (through `ParameterObject.WriteParameterFile`, so the keys
come out sorted, and the values formatted, the way elastix formats
them), and a `<name>_expected.npz` holding what transformix computes for
that file:

- `vox2lps`: the `(D, D + 1)` voxel-to-LPS affine of the deformation
  field that transformix returns, read off the ITK image itself -- the
  fixed-image grid, as elastix understands `Size`, `Origin`, `Spacing`
  and `Direction`;
- `disp`: the deformation field, `(X, Y[, Z], D)` (x first), in LPS
  millimetres: transformix maps the fixed-image point `p` to `p + disp`.

The tests assert against those stored arrays, so the suite needs
ITK-Elastix only to regenerate the fixtures, never to run.

The parameters are deliberately awkward: a fixed-image direction that is
not symmetric (so that reading elastix's column-major `Direction` as
row-major gives a different, wrong grid), centers of rotation away from
the origin, and B-spline grids with a non-zero `GridIndex` and an
oblique `GridDirection`.
"""

# stdlib
import math
import os
import tempfile
from pathlib import Path

# dependencies
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


#: An oblique, non-symmetric direction matrix (with a flip of the last
#: axis, which is still a valid ITK direction).
DIRECTION = {
    2: rotation2d(0.3) @ np.diag([1, -1]),
    3: rotation3d(0.2, -0.1, 0.4) @ np.diag([1, 1, -1]),
}


def column_major(matrix: np.ndarray) -> list:
    """elastix writes a direction matrix column by column."""
    return [float(v) for v in np.asarray(matrix).T.ravel()]


def fmt(values: list) -> list:
    return [
        repr(float(v)) if not float(v).is_integer() else str(int(v))
        for v in values
    ]


def fixed_map(ndim: int) -> dict:
    """The fixed-image part of a transform parameter map."""
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
    ndim: int, order: int, seed: int, grid_size: tuple = (10, 11, 12)
) -> dict:
    """A B-spline whose valid region covers the fixed grid.

    elastix evaluates a B-spline only where the whole support of the
    spline is inside the control-point grid, and returns the input point
    elsewhere. The fixed grid is placed well inside that region.
    """
    rng = np.random.default_rng(seed)
    grid_size = list(grid_size)[:ndim]
    grid_index = [1, -2, 0][:ndim]
    grid_spacing = [2.0, 2.5, 3.0][:ndim]
    direction = rotation2d(-0.2) if ndim == 2 else rotation3d(0.1, 0.2, -0.3)
    # Center the control grid on the center of the fixed grid.
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
        BSplineTransformSplineOrder=str(order),
        UseCyclicTransform="false",
    )


def cases() -> dict:
    """Every single-file case, by name."""
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
    """What transformix computes for a chain of maps (first applies
    first)."""
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
    """Write the files of a case, and transformix's answer for it."""
    ndim = int(maps[-1]["FixedImageDimension"][0])
    for filename, pmap in files.items():
        _write_map(pmap, HERE / filename)
    np.savez(HERE / f"{name}_expected.npz", **transformix(maps, ndim))


def _object(pmap: dict) -> "itk.ParameterObject":
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(pmap)
    return parameter_object


def _write_map(pmap: dict, filename: Path) -> None:
    # `WriteParameterFile(filename)` writes every map of the object, and
    # picks the text or the TOML syntax from the extension.
    _object(pmap).WriteParameterFile(str(filename))


def main() -> None:
    # transformix writes its deformation field into the working directory.
    os.chdir(tempfile.mkdtemp())
    for name, pmap in cases().items():
        write(name, [pmap], {f"{name}.txt": pmap})

    # The same Euler file in elastix's TOML syntax.
    euler = cases()["euler3d"]
    write("euler3d_toml", [euler], {"euler3d.toml": euler})

    # A chain: an affine, then a B-spline that names it as its initial
    # transform, by a path relative to its own directory -- which is
    # where elastix looks when the path does not exist from the working
    # directory.
    affine = cases()["affine3d"]
    # The affine moves the fixed grid, so the B-spline grid is made large
    # enough for its valid region to cover the moved points too.
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

    # A three-link chain through the deprecated spelling of the key.
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
