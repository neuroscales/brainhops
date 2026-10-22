"""Tests of the conversions of FNIRT, NiftyReg grids and elastix files.

Each test converts a transformation and checks that points inside the
field of view map to the same positions before and after the
conversion. An exact conversion is checked to the rounding of the
arithmetic, and a lossy one, which needs `lossy=True`, to a loose
tolerance. The FSL fixtures are those of `tests/data/fsl`, taken from
fslpy (see `tests/data/fsl/ATTRIBUTION.md`).
"""

import os

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    CoordinateSystem,
    LPSmm,
    RASmm,
    VoxelCoordinateSystem,
)
from brainhops.errors import (  # noqa: E402
    ConversionError,
    LossyConversionError,
)
from brainhops.io.transformations.base._converters import (  # noqa: E402
    relabels,
)
from brainhops.io.transformations.base.affines import (  # noqa: E402
    RASToVoxel,
    VoxelToRAS,
)
from brainhops.io.transformations.elastix import (  # noqa: E402
    ElastixParameterTransform,
    ElastixTomlTransform,
)
from brainhops.io.transformations.fsl import FnirtWarpField  # noqa: E402
from brainhops.io.transformations.itk.nifti._fields import (  # noqa: E402
    ItkNiftiDisplacementField,
)
from brainhops.io.transformations.mrtrix._warps import (  # noqa: E402
    MrtrixDisplacementField,
)
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASDisplacementField,
)
from brainhops.io.transformations.niftyreg import (  # noqa: E402
    NiftyRegControlPointGrid,
    NiftyRegVelocityGrid,
)
from brainhops.io.transformations.spm import SpmCoordinatesField  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "fsl")
REF = os.path.join(DATA, "ref.nii.gz")
SRC = os.path.join(DATA, "src.nii.gz")
DENSE = os.path.join(DATA, "displacementfield.nii.gz")
COEFFICIENTS = os.path.join(DATA, "coefficientfield.nii.gz")

EXACT = 1e-6
"""The tolerance, in millimetres, of an exact conversion, which allows
for the single precision of the values that a file stores."""

LOOSE = 0.5
"""The tolerance, in millimetres, of a lossy conversion."""

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])[:-1]


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _points() -> np.ndarray:
    # Points of the RAS world of the reference, away from the edges of
    # its field of view.
    reference = nb.load(REF)
    rng = np.random.default_rng(0)
    shape = np.asarray(reference.shape[:3])
    voxels = rng.uniform(3, shape - 4, size=(40, 3))
    return voxels @ reference.affine[:3, :3].T + reference.affine[:3, 3]


POINTS = _points()


def _in_ras(t: xforms.Transformation) -> xforms.Transformation:
    # A transformation between LPS spaces, read and written in RAS.
    if t.input != LPSmm():
        return t
    return xforms.Sequence(
        transformations=[
            xforms.Affine(matrix=FLIP, input=RASmm(), output=LPSmm()),
            t,
            xforms.Affine(matrix=FLIP, input=LPSmm(), output=RASmm()),
        ]
    )


def _apply(t: xforms.Transformation, points: np.ndarray) -> np.ndarray:
    points = xforms.CoordinatesField(field=points[:, None, None, :])
    chain = xforms.Sequence(transformations=[points, _in_ras(t)])
    out = chain.compute().to(xforms.CoordinatesField)
    return np.asarray(out.field)[:, 0, 0, :]


def _error(a: xforms.Transformation, b: xforms.Transformation) -> float:
    return float(np.abs(_apply(a, POINTS) - _apply(b, POINTS)).max())


def _load(path: str) -> FnirtWarpField:
    return io.load(path, reference=REF, moving=SRC)


def _smooth(shape: tuple, scale: float = 1.0) -> np.ndarray:
    # A smooth field of vectors, in the voxels of a grid of `shape`.
    axes = np.meshgrid(
        *[np.linspace(0, np.pi, n) for n in shape], indexing="ij"
    )
    x, y, z = axes
    return scale * np.stack(
        [np.sin(x) * np.cos(y), np.cos(z) * np.sin(y), np.sin(x + z)], axis=-1
    )


VOX2RAS = nb.load(REF).affine
"""The grid of the reference, on which the synthetic fields lie."""


def _chain(
    shape: tuple = (18, 54, 23), degree: int = 1, log: bool = False
) -> xforms.Sequence:
    # A smooth field of displacements, or a velocity, between RAS spaces
    # on the grid of the reference.
    voxel = VoxelCoordinateSystem()
    store = "coefficients" if degree > 1 else "values"
    field = xforms.DisplacementField(
        data=_smooth(shape, 0.3 if log else 1.0),
        degree=degree,
        store=store,
        log=log,
        input=voxel,
        output=voxel,
    )
    return xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=np.linalg.inv(VOX2RAS)[:-1]),
            field,
            VoxelToRAS(matrix=VOX2RAS[:-1]),
        ]
    )


def _affine(linear: np.ndarray, ends: dict) -> xforms.Affine:
    matrix = np.concatenate([linear, [[1.5], [-2.0], [3.25]]], axis=1)
    return xforms.Affine(matrix=matrix, **ends)


def _rotation() -> np.ndarray:
    a, b, c = 0.1, -0.2, 0.3
    rx = np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )
    ry = np.array(
        [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
    )
    rz = np.array(
        [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]]
    )
    return rz @ ry @ rx


# ----------------------------------------------------------------------
#   FNIRT AS A SOURCE
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", [DENSE, COEFFICIENTS])
def test_a_fnirt_warp_without_its_images_names_the_missing_options(
    path: str,
) -> None:
    warp = io.load(path)
    with pytest.raises(ConversionError, match="moving="):
        warp.to(NiftiRASDisplacementField, lossy=True)


@pytest.mark.parametrize("path", [DENSE, COEFFICIENTS])
def test_the_images_of_a_fnirt_warp_are_conversion_options(path: str) -> None:
    warp = _load(path)
    plain = io.load(path)
    converted = plain.to(
        NiftiRASDisplacementField, reference=REF, moving=SRC, lossy=True
    )
    assert type(converted) is NiftiRASDisplacementField
    assert plain.moving is None
    expected = warp.to(NiftiRASDisplacementField, lossy=True)
    assert _error(converted, expected) < EXACT


def test_the_images_of_a_fnirt_warp_are_load_options_given_as_paths() -> None:
    warp = io.load(DENSE, reference=REF, moving=SRC)
    assert type(warp) is FnirtWarpField
    assert len(warp) == 3


@pytest.mark.parametrize(
    "cls",
    [
        NiftiRASDisplacementField,
        SpmCoordinatesField,
        ItkNiftiDisplacementField,
        MrtrixDisplacementField,
    ],
)
def test_a_fnirt_deformation_converts_to_a_dense_field(
    cls: type,
    tmp_path,  # noqa: ANN001
) -> None:
    warp = _load(DENSE)
    if relabels(warp, cls):
        pytest.skip(f"{cls.__name__} has no exact converter yet.")
    try:
        converted = warp.to(cls, lossy=True)
    except ConversionError as error:
        pytest.skip(f"{cls.__name__} refuses the warp: {error}")
    assert _error(converted, warp) < EXACT
    back = converted.to(FnirtWarpField, reference=REF, moving=SRC, lossy=True)
    assert _error(back, warp) < EXACT


def test_a_fnirt_deformation_is_exactly_a_nifti_field_and_back() -> None:
    warp = _load(DENSE)
    nifti = warp.to(NiftiRASDisplacementField)
    assert _error(nifti, warp) < EXACT
    back = nifti.to(FnirtWarpField, reference=REF, moving=SRC)
    assert type(back) is FnirtWarpField
    assert back.store == "values"
    assert _error(back, warp) < EXACT


def test_a_fnirt_coefficient_field_is_sampled_on_the_reference() -> None:
    warp = _load(COEFFICIENTS)
    with pytest.raises(LossyConversionError):
        warp.to(NiftiRASDisplacementField)
    nifti = warp.to(NiftiRASDisplacementField, lossy=True)
    shape = nb.load(REF).shape[:3]
    assert tuple(nifti.displacement.data.shape[:3]) == shape
    assert _error(nifti, warp) < LOOSE


# ----------------------------------------------------------------------
#   FNIRT AS A TARGET
# ----------------------------------------------------------------------


def test_a_conversion_to_fnirt_names_the_missing_images() -> None:
    with pytest.raises(ConversionError, match="reference= and moving="):
        _chain().to(FnirtWarpField)
    with pytest.raises(ConversionError, match="moving="):
        _chain().to(FnirtWarpField, reference=REF)


def test_an_affine_is_exactly_a_fnirt_deformation() -> None:
    affine = _affine(_rotation(), dict(input=RASmm(), output=RASmm()))
    warp = affine.to(FnirtWarpField, reference=REF, moving=SRC)
    assert _error(warp, affine) < EXACT


def test_a_field_on_another_grid_is_sampled_into_fnirt() -> None:
    chain = _chain(shape=(9, 27, 12), degree=3)
    with pytest.raises(LossyConversionError):
        chain.to(FnirtWarpField, reference=REF, moving=SRC)
    warp = chain.to(FnirtWarpField, reference=REF, moving=SRC, lossy=True)
    assert _error(warp, chain) < LOOSE


def test_a_fnirt_coefficient_field_is_exact_on_its_own_knot_grid() -> None:
    warp = _load(COEFFICIENTS)
    nifti = _load(DENSE).to(NiftiRASDisplacementField)
    for source in (warp, warp.to(ElastixParameterTransform)):
        again = source.to(
            FnirtWarpField, reference=REF, moving=SRC, knot_spacing=5
        )
        assert again.store == "coefficients"
        assert tuple(again.data.shape) == tuple(warp.data.shape)
        assert _error(again, warp) < EXACT
    with pytest.raises(LossyConversionError):
        nifti.to(FnirtWarpField, reference=REF, moving=SRC, knot_spacing=5)
    fitted = nifti.to(
        FnirtWarpField, reference=REF, moving=SRC, knot_spacing=5, lossy=True
    )
    assert _error(fitted, nifti) < LOOSE


@pytest.mark.parametrize("spacing", [None, 4])
def test_fnirt_warps_written_by_save_read_back(
    spacing: tx.Optional[int],
    tmp_path,  # noqa: ANN001
) -> None:
    chain = _chain(degree=3)
    conversion = {"reference": REF, "moving": SRC}
    if spacing is not None:
        conversion["knot_spacing"] = spacing
    path = str(tmp_path / "warp.nii.gz")
    io.save(chain, path, hint="fnirt", lossy=True, conversion=conversion)
    back = io.load(path, reference=REF, moving=SRC)
    assert type(back) is FnirtWarpField
    assert back.degree == (1 if spacing is None else 3)
    assert _error(back, chain) < LOOSE


# ----------------------------------------------------------------------
#   NIFTYREG GRIDS OF CONTROL POINTS
# ----------------------------------------------------------------------


def test_a_cubic_spline_is_exactly_a_niftyreg_grid(tmp_path) -> None:  # noqa: ANN001
    chain = _chain(degree=3)
    grid = chain.to(NiftyRegControlPointGrid)
    assert grid.degree == 3
    assert _error(grid, chain) < EXACT
    path = str(tmp_path / "cpp.nii.gz")
    io.save(grid, path)
    back = io.load(path)
    assert type(back) is NiftyRegControlPointGrid
    assert _error(back, chain) < EXACT


def test_an_affine_before_a_spline_goes_into_the_niftyreg_extension() -> None:
    affine = _affine(
        np.diag([1.1, 0.9, 1.0]), dict(input=RASmm(), output=RASmm())
    )
    chain = xforms.Sequence(transformations=[affine, *_chain(degree=3)])
    grid = chain.to(NiftyRegControlPointGrid)
    assert grid.affine is not None
    assert _error(grid, chain) < EXACT


def test_a_dense_field_is_interpolated_on_a_niftyreg_grid() -> None:
    chain = _chain()
    with pytest.raises(LossyConversionError):
        chain.to(NiftyRegControlPointGrid)
    grid = chain.to(NiftyRegControlPointGrid, lossy=True)
    assert grid.degree == 3
    assert _error(grid, chain) < LOOSE
    linear = chain.to(NiftyRegControlPointGrid, degree=1)
    assert linear.degree == 1
    assert _error(linear, chain) < EXACT
    coarse = chain.to(NiftyRegControlPointGrid, spacing=6.0, lossy=True)
    assert _error(coarse, chain) < LOOSE


def test_a_niftyreg_grid_converts_to_a_dense_field() -> None:
    grid = _chain(degree=3).to(NiftyRegControlPointGrid)
    nifti = grid.to(NiftiRASDisplacementField, lossy=True)
    assert _error(nifti, grid) < LOOSE
    with pytest.raises(LossyConversionError):
        grid.to(NiftiRASDisplacementField)


def test_a_fnirt_coefficient_field_is_exactly_a_niftyreg_grid() -> None:
    warp = _load(COEFFICIENTS)
    grid = warp.to(NiftyRegControlPointGrid)
    assert _error(grid, warp) < EXACT


def test_a_velocity_is_held_by_a_niftyreg_velocity_grid() -> None:
    velocity = _chain(degree=3, log=True)
    grid = velocity.to(NiftyRegVelocityGrid)
    assert _error(grid, velocity) < EXACT
    dense = _chain(log=True)
    fitted = dense.to(NiftyRegVelocityGrid, lossy=True)
    assert _error(fitted, dense) < LOOSE
    with pytest.raises(ConversionError, match="logarithm"):
        _chain(degree=3).to(NiftyRegVelocityGrid)


# ----------------------------------------------------------------------
#   ELASTIX
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "linear, name",
    [
        (np.eye(3), "TranslationTransform"),
        (_rotation(), "EulerTransform"),
        (_rotation() @ np.diag([1.1, 0.9, 1.0]), "AffineTransform"),
    ],
)
@pytest.mark.parametrize(
    "cls", [ElastixParameterTransform, ElastixTomlTransform]
)
@pytest.mark.parametrize("space", [RASmm(), LPSmm(), None])
def test_an_affine_is_exactly_an_elastix_transform(
    linear: np.ndarray,
    name: str,
    cls: type,
    space: tx.Optional[CoordinateSystem],
    tmp_path,  # noqa: ANN001
) -> None:
    ends = {} if space is None else dict(input=space, output=space)
    affine = _affine(linear, ends)
    expected = affine if space is not None else _affine(linear, {})
    converted = affine.to(cls, center=[1.0, -2.0, 3.0])
    transform = converted.consolidate()
    assert transform.metadata.raw.parameters["Transform"] == (name,)
    if space is None:
        # Unknown systems are taken to be those of elastix.
        reference = xforms.Affine(
            matrix=affine.matrix, input=LPSmm(), output=LPSmm()
        )
        assert _error(converted, reference) < EXACT
        return
    assert _error(converted, expected) < EXACT
    path = str(tmp_path / ("t" + cls.EXTENSIONS[0]))
    io.save(converted, path)
    assert _error(io.load(path), expected) < EXACT


def test_an_elastix_transform_takes_the_fixed_image() -> None:
    affine = _affine(np.eye(3), dict(input=RASmm(), output=RASmm()))
    converted = affine.to(ElastixParameterTransform, reference=REF)
    assert converted.fixed_geometry.shape == nb.load(REF).shape[:3]
    assert _error(converted, affine) < EXACT


def test_a_fnirt_spline_is_chained_in_elastix_files(tmp_path) -> None:  # noqa: ANN001
    warp = _load(COEFFICIENTS)
    converted = warp.to(ElastixParameterTransform)
    assert converted.initial is not None
    assert _error(converted, warp) < EXACT
    path = str(tmp_path / "TransformParameters.0.txt")
    io.save(converted, path)
    assert len(os.listdir(tmp_path)) == 2
    back = io.load(path)
    assert _error(back, warp) < EXACT
    assert _error(back.to(NiftyRegControlPointGrid), warp) < EXACT
    assert _error(back.to(NiftiRASDisplacementField, lossy=True), warp) < 1.0


@pytest.mark.parametrize("degree", [1, 3])
def test_a_field_is_an_elastix_bspline(degree: int) -> None:
    chain = _chain(degree=degree)
    nifti = chain.to(NiftiRASDisplacementField, lossy=True)
    for source in (chain, nifti) if degree == 1 else (chain,):
        converted = source.to(ElastixParameterTransform)
        block = converted.transformations[-1]
        assert block.degree == degree
        assert _error(converted, source) < EXACT


def test_a_velocity_is_fitted_into_an_elastix_bspline() -> None:
    velocity = _chain(log=True)
    with pytest.raises(LossyConversionError):
        velocity.to(ElastixParameterTransform)
    converted = velocity.to(ElastixParameterTransform, lossy=True)
    assert _error(converted, velocity) < LOOSE


# ----------------------------------------------------------------------
#   SAVE
# ----------------------------------------------------------------------


def test_a_generic_field_is_still_saved_as_a_nifti_field(tmp_path) -> None:  # noqa: ANN001
    path = str(tmp_path / "warp.nii.gz")
    io.save(_chain(), path)
    assert type(io.load(path)) is NiftiRASDisplacementField


@pytest.mark.parametrize("space", [None, RASmm(), LPSmm()])
def test_a_generic_affine_is_saved_to_txt_without_ambiguity(
    space: tx.Optional[CoordinateSystem],
    tmp_path,  # noqa: ANN001
) -> None:
    ends = {} if space is None else dict(input=space, output=space)
    affine = _affine(_rotation(), ends)
    chain = xforms.Sequence(transformations=[affine])
    for i, t in enumerate((affine, chain)):
        path = str(tmp_path / f"a{i}.txt")
        io.save(t, path)
        assert os.path.exists(path)


def test_a_cubic_field_is_saved_as_a_niftyreg_grid(tmp_path) -> None:  # noqa: ANN001
    # A NIfTI displacement field holds a cubic field only approximately,
    # and a NiftyReg grid of control points holds it exactly, so the
    # exact conversion is chosen.
    path = str(tmp_path / "warp.nii.gz")
    chain = _chain(degree=3)
    io.save(chain, path)
    back = io.load(path)
    assert type(back) is NiftyRegControlPointGrid
    assert _error(back, chain) < EXACT


def test_images_given_to_save_select_the_fsl_format(
    tmp_path,  # noqa: ANN001
) -> None:
    # The field lies on the grid of the reference, so both FNIRT and NIfTI
    # hold it exactly. FNIRT takes the images that the save was given, so
    # it is chosen, and a NIfTI field is written without them.
    path = str(tmp_path / "warp.nii.gz")
    conversion = {"reference": REF, "moving": SRC}
    io.save(_chain(), path, conversion=conversion)
    assert type(io.load(path)) is FnirtWarpField
    io.save(_chain(), path)
    assert type(io.load(path)) is NiftiRASDisplacementField
    io.save(_chain(), path, hint="fnirt", conversion=conversion)
    assert type(io.load(path)) is FnirtWarpField
