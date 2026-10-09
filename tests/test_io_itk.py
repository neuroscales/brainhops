import warnings
from pathlib import Path

import numpy as np
import pytest
from bagof.magic import replace

from brainhops import io
from brainhops._core import affines
from brainhops._core.dependencies import HAS_H5PY
from brainhops.datamodel import transformations as xforms
from brainhops.io.transformations import itk

data_dir = Path(__file__).parent / "data"

FILES_H5 = list(data_dir.glob("*.h5"))
FILES_TFM = list(data_dir.glob("*.tfm"))

needs_h5py = pytest.mark.skipif(not HAS_H5PY, reason="needs h5py")

# Without h5py, the HDF5 reader is unregistered and dispatch fails
# differently, so these cases are skipped rather than the whole run.
PARAMS_H5 = [pytest.param(f, marks=needs_h5py) for f in FILES_H5]

TfmTransform = io.transformations.itk.tfm.TfmTransform


@pytest.mark.parametrize("filename", FILES_H5)
@pytest.mark.parametrize("load", [True, False])
@pytest.mark.parametrize("keep_open", [True, False])
def test_read_h5(filename: str, load: bool, keep_open: bool) -> None:
    pytest.importorskip("h5py")
    H5Transform = io.transformations.itk.h5.H5Transform
    transform = H5Transform.from_file(filename, load=load, keep_open=keep_open)
    transforms = transform.transformations  # noqa: F841


@pytest.mark.parametrize("filename", FILES_TFM)
def test_read_tfm(filename: str) -> None:
    transform = TfmTransform.from_file(filename)
    transforms = transform.transformations  # noqa: F841


# ----------------------------------------------------------------------
#   REGISTRY / DISPATCH
# ----------------------------------------------------------------------
# The ITK readers register themselves, so the generic loaders find them.


@pytest.mark.parametrize("filename", FILES_TFM)
def test_tfm_is_dispatched(filename: str) -> None:
    assert io.transformations.sniff(filename) is TfmTransform
    assert io.sniff(filename) is TfmTransform
    assert type(io.transformations.load(filename)) is TfmTransform
    assert type(io.load(filename)) is TfmTransform


@pytest.mark.parametrize("filename", FILES_H5)
def test_h5_is_dispatched(filename: str) -> None:
    pytest.importorskip("h5py")
    H5Transform = io.transformations.itk.h5.H5Transform
    assert io.transformations.sniff(filename) is H5Transform
    assert io.sniff(filename) is H5Transform
    assert type(io.transformations.load(filename)) is H5Transform
    assert type(io.load(filename)) is H5Transform


def test_tfm_header_only_is_read_as_empty(tmp_path) -> None:  # noqa: ANN001
    """A .tfm file with only the version header is read as empty."""
    header_only = tmp_path / "header_only.tfm"
    header_only.write_text("# Insight Transform File V1.0\n")

    assert TfmTransform.sniff_line("") == 0.0
    assert list(TfmTransform.from_file(header_only).transformations) == []
    # The .tfm extension still routes to the ITK reader.
    assert io.transformations.sniff(header_only) is TfmTransform
    assert type(io.transformations.load(header_only)) is TfmTransform


# ----------------------------------------------------------------------
#   BLOCKS AS TRANSFORMATIONS
# ----------------------------------------------------------------------
# Each block of an ITK file is a structured sequence with named lazy slots.


@pytest.mark.parametrize("filename", FILES_TFM + PARAMS_H5)
def test_blocks_are_transformations(filename: str) -> None:
    transform = io.transformations.load(filename)
    assert isinstance(transform, xforms.Sequence)
    for block in transform.transformations:
        assert isinstance(block, itk.ItkStruct)
        assert isinstance(block, xforms.Sequence)
        # A block is a non-empty chain of transformations.
        assert len(block) >= 1
        assert all(isinstance(t, xforms.Transformation) for t in block)


def test_affine_block_exposes_named_cached_slots() -> None:
    block = TfmTransform.from_file(data_dir / "itk_affine3d.tfm")[0]
    assert isinstance(block, itk.ItkAffineBase)

    assert np.allclose(block.center, block.fixed_parameters)
    assert isinstance(block.linear, xforms.Linear)
    assert isinstance(block.translation, xforms.Translation)
    assert list(block) == [
        block.recenter,
        block.linear,
        block.uncenter,
        block.translation,
    ]

    # Each slot is computed once and cached.
    assert block.linear is block.linear
    assert block.transformations is block.transformations

    # An assigned chain overrides the derived one until cleared.
    block.transformations = [xforms.Identity()]
    assert len(block) == 1
    block.transformations = None
    assert len(block) == 4


def _versor_rigid_3d(
    versor: tuple, translation: tuple, center: tuple
) -> itk.ItkStruct:
    return itk.ItkStruct(
        type=itk.ItkTransformClass.VersorRigid3DTransform,
        precision="double",
        ndim_input=3,
        ndim_output=3,
        parameters=tuple(versor) + tuple(translation),
        fixed_parameters=tuple(center),
    )


def test_versor_rigid_3d_applies_its_translation() -> None:
    """The rotation acts about the center, then the translation applies."""
    block = _versor_rigid_3d((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    matrix = block.compute().to(xforms.Affine, lossy=True).matrix
    assert np.allclose(np.asarray(matrix)[:, :3], np.eye(3))
    assert np.allclose(np.asarray(matrix)[:, 3], [1.0, 2.0, 3.0])


def test_versor_rigid_3d_folds_a_real_rotation_about_its_center() -> None:
    """A quarter turn about an off-origin center gives [R | c + t - Rc]."""
    angle = np.pi / 2
    versor = (0.0, 0.0, np.sin(angle / 2))
    translation = np.array([1.0, 2.0, 3.0])
    center = np.array([4.0, 5.0, 6.0])
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    block = _versor_rigid_3d(versor, translation, center)
    matrix = np.asarray(block.compute().to(xforms.Affine, lossy=True).matrix)
    assert np.allclose(matrix[:, :3], rotation)
    assert np.allclose(matrix[:, 3], center + translation - rotation @ center)


def test_versor_tolerates_a_rounded_unit_versor() -> None:
    """A versor that rounding puts just past the unit sphere still loads."""
    block = _versor_rigid_3d(
        (1.0000000002, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    matrix = np.asarray(block.compute().to(xforms.Affine, lossy=True).matrix)
    assert np.allclose(matrix[:, :3], np.diag([1.0, -1.0, -1.0]))

    too_long = _versor_rigid_3d(
        (1.1, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="magnitude > 1"):
        too_long.compute()


def test_displacement_blocks_are_lps_to_lps_chains() -> None:
    pytest.importorskip("h5py")
    for name, degree, store in [
        ("itk_displacement3d.h5", 1, "values"),
        ("itk_bspline3d.h5", 3, "coefficients"),
    ]:
        block = io.transformations.load(data_dir / name)[-1]
        assert isinstance(block, itk.ItkDisplacementBase)
        assert list(block) == [
            block.lps2voxel,
            block.displacement,
            block.voxel2lps,
        ]
        assert block.degree == degree
        assert block.store == store
        # The array is what the field stores: values or B-spline coefficients.
        assert block.displacement.data is block.field
        assert block.field.shape[-1] == 3


# ----------------------------------------------------------------------
#   WARP MEMORY LAYOUT
# ----------------------------------------------------------------------
# ITK flattens the two warp kinds differently. The fixtures hold a ramp
# whose entries name their voxel and component, so that a transposed stride
# cannot hide, and tests/data/generate_itk_fixtures.py stores the arrays
# that SimpleITK reports beside them.


@pytest.mark.parametrize(
    "name, interleaved",
    [
        ("itk_displacement_ramp3d", True),
        ("itk_bspline_ramp3d", False),
    ],
)
def test_warp_field_is_decoded_in_itks_own_layout(
    name: str, interleaved: bool
) -> None:
    """Displacement fields interleave components; B-splines stack them."""
    block = io.transformations.load(data_dir / f"{name}.tfm")[-1]
    assert isinstance(block, itk.ItkDisplacementBase)

    expected = np.load(data_dir / f"{name}_expected.npy")
    field = np.asarray(block.field)
    assert field.shape == expected.shape
    np.testing.assert_allclose(field, expected)
    assert block.interleaved is interleaved

    # The other layout gives a different result.
    parameters = np.asarray(block.parameters)
    shape = expected.shape[:-1]
    ndim = len(shape)
    other = (
        parameters.reshape(ndim, *reversed(shape)).transpose(3, 2, 1, 0)
        if interleaved
        else parameters.reshape(*reversed(shape), ndim).transpose(2, 1, 0, 3)
    )
    assert not np.allclose(other, expected)


def test_warp_fixtures_still_match_simpleitk() -> None:
    """The stored warp expectations still equal what SimpleITK reports."""
    sitk = pytest.importorskip("SimpleITK")

    transform = sitk.ReadTransform(
        str(data_dir / "itk_displacement_ramp3d.tfm")
    )
    field = sitk.DisplacementFieldTransform(transform).GetDisplacementField()
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(field).transpose(2, 1, 0, 3),
        np.load(data_dir / "itk_displacement_ramp3d_expected.npy"),
    )

    transform = sitk.ReadTransform(str(data_dir / "itk_bspline_ramp3d.tfm"))
    images = sitk.BSplineTransform(transform).GetCoefficientImages()
    np.testing.assert_allclose(
        np.stack(
            [sitk.GetArrayFromImage(i).transpose(2, 1, 0) for i in images],
            axis=-1,
        ),
        np.load(data_dir / "itk_bspline_ramp3d_expected.npy"),
    )


# ----------------------------------------------------------------------
#   EULER ANGLES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["itk_euler3d", "itk_euler3d_zyx"])
def test_euler_3d_composes_its_angles_the_way_itk_does(name: str) -> None:
    """The angles compose as ZXY, or as ZYX when ComputeZYX is set."""
    block = TfmTransform.from_file(data_dir / f"{name}.tfm")[0]
    assert block.type == itk.ItkTransformClass.Euler3DTransform

    matrix = np.asarray(block.compute().to(xforms.Affine, lossy=True).matrix)
    np.testing.assert_allclose(
        matrix, np.load(data_dir / f"{name}_expected.npy"), atol=1e-12
    )


def test_euler_3d_reads_the_modern_four_fixed_parameters() -> None:
    """The fourth fixed parameter of ITK 5 is a flag, not a coordinate."""
    plain = TfmTransform.from_file(data_dir / "itk_euler3d.tfm")[0]
    zyx = TfmTransform.from_file(data_dir / "itk_euler3d_zyx.tfm")[0]

    assert len(plain.fixed_parameters) == 4
    assert plain.compute_zyx is False
    assert zyx.compute_zyx is True

    for block in (plain, zyx):
        center = np.asarray(block.center)
        assert center.shape == (3,)
        np.testing.assert_allclose(center, [4.0, 5.0, 6.0])

    # Files older than ITK 5 hold the center alone and compose as ZXY.
    legacy = itk.ItkStruct(
        type=itk.ItkTransformClass.Euler3DTransform,
        precision="double",
        ndim_input=3,
        ndim_output=3,
        parameters=plain.parameters,
        fixed_parameters=(4.0, 5.0, 6.0),
    )
    assert legacy.compute_zyx is False
    np.testing.assert_allclose(
        np.asarray(legacy.compute().to(xforms.Affine, lossy=True).matrix),
        np.load(data_dir / "itk_euler3d_expected.npy"),
        atol=1e-12,
    )


def test_euler_3d_matches_simpleitk() -> None:
    """The stored Euler expectations are still what ITK reports."""
    sitk = pytest.importorskip("SimpleITK")

    for name in ("itk_euler3d", "itk_euler3d_zyx"):
        transform = sitk.Euler3DTransform(
            sitk.ReadTransform(str(data_dir / f"{name}.tfm"))
        )
        expected = np.load(data_dir / f"{name}_expected.npy")
        np.testing.assert_allclose(
            expected[:, :3], np.asarray(transform.GetMatrix()).reshape(3, 3)
        )
        np.testing.assert_allclose(
            expected[:, 3], transform.TransformPoint((0.0, 0.0, 0.0))
        )


def test_transform_group_is_gone() -> None:
    """Blocks live in `transformations`, with no second list."""
    transform = TfmTransform.from_file(data_dir / "itk_affine3d.tfm")
    assert not hasattr(transform, "transform_group")


# ----------------------------------------------------------------------
#   WARP BLOCKS COMPOSE AND INVERT
# ----------------------------------------------------------------------
# Both affines of a warp block are compact, so that the chain composes.


@pytest.mark.parametrize("name", ["itk_displacement3d.h5", "itk_bspline3d.h5"])
def test_warp_block_affines_are_compact(name: str) -> None:
    pytest.importorskip("h5py")
    block = io.transformations.load(data_dir / name)[-1]
    ndim = block.ndim_input
    vox2lps = np.asarray(block.voxel2lps.matrix)
    lps2vox = np.asarray(block.lps2voxel.matrix)
    assert vox2lps.shape == (ndim, ndim + 1)
    assert lps2vox.shape == (ndim, ndim + 1)

    # The voxel-to-LPS matrix is direction @ diag(spacing), and the origin
    # is its offset.
    origin = np.asarray(block.fixed_parameters)[ndim : 2 * ndim]
    spacing = np.asarray(block.fixed_parameters)[2 * ndim : 3 * ndim]
    direction = np.asarray(block.fixed_parameters)[
        3 * ndim : 3 * ndim + ndim * ndim
    ].reshape(ndim, ndim)
    np.testing.assert_allclose(vox2lps[:, :ndim], direction @ np.diag(spacing))
    np.testing.assert_allclose(vox2lps[:, ndim], origin)

    # The two affines are mutual inverses.
    round_trip = xforms.Sequence(
        transformations=[block.lps2voxel, block.voxel2lps]
    ).compute()
    np.testing.assert_allclose(
        np.asarray(round_trip.matrix),
        np.eye(ndim, ndim + 1),
        atol=1e-10,
    )


@pytest.mark.parametrize("name", ["itk_displacement3d.h5", "itk_bspline3d.h5"])
def test_warp_block_computes(name: str) -> None:
    pytest.importorskip("h5py")
    block = io.transformations.load(data_dir / name)[-1]
    vox2lps, _ = block._grid

    result = block.compute()

    # The world-to-voxel affine comes first, followed by the field in grid
    # units.
    assert isinstance(result, xforms.Sequence)
    assert isinstance(result[0], xforms.Affine)
    np.testing.assert_allclose(
        np.asarray(result[0].matrix), affines.inv(vox2lps)
    )
    assert isinstance(result[-1], xforms.DisplacementField)
    assert np.asarray(result[-1].field).shape == np.asarray(block.field).shape
    assert result[-1].degree == block.degree
    assert result[-1].store == block.store
    assert result.input == block.input
    assert result.output == block.output


def test_composite_warp_computes() -> None:
    """A warp block composes with the surrounding affine blocks."""
    pytest.importorskip("h5py")
    transform = io.load(data_dir / "itk_composite_displacement3d.h5")
    result = transform.compute()
    assert isinstance(result, xforms.Sequence)
    # The warp block applies first, starting with its LPS-to-voxel affine.
    assert isinstance(result[0], xforms.Affine)
    ndim = transform[-1].ndim_input
    assert np.asarray(result[0].matrix).shape == (ndim, ndim + 1)
    assert isinstance(result[-1], xforms.DisplacementField)


@pytest.mark.parametrize("name", ["itk_displacement3d.h5", "itk_bspline3d.h5"])
def test_warp_block_inverts(name: str) -> None:
    """Every child has an inverse, so the block has one."""
    pytest.importorskip("h5py")
    block = io.transformations.load(data_dir / name)[-1]
    inverse = block.inverse()
    assert isinstance(inverse, xforms.Sequence)
    assert len(inverse) == len(block)
    assert inverse.input == block.output
    assert inverse.output == block.input
    # The inverse chain is reversed.
    assert isinstance(inverse[0], xforms.Inverse)
    assert inverse[0].forward is block.voxel2lps
    assert inverse[-1].forward is block.lps2voxel


def test_warp_block_grid_is_read_at_its_own_dimensionality() -> None:
    """The grid geometry follows ndim_input rather than a 3-D layout."""
    # The shape, origin, spacing and a 2x2 direction make 2 + 2 + 2 + 4
    # fixed parameters.
    block = itk.ItkStruct(
        type=itk.ItkTransformClass.DisplacementFieldTransform,
        precision="double",
        ndim_input=2,
        ndim_output=2,
        parameters=np.zeros(2 * 3 * 4),
        fixed_parameters=np.array(
            [3.0, 4.0, 10.0, 20.0, 2.0, 5.0, 0.0, -1.0, 1.0, 0.0]
        ),
    )
    vox2lps, shape = block._grid
    assert shape == (3, 4)
    np.testing.assert_allclose(vox2lps, [[0.0, -5.0, 10.0], [2.0, 0.0, 20.0]])
    assert np.asarray(block.voxel2lps.matrix).shape == (2, 3)
    assert np.asarray(block.field).shape == (3, 4, 2)
    assert block.input == block.output
    assert len(block.input.axes) == 2


# ----------------------------------------------------------------------
#   THE DERIVED CHAIN
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["itk_affine3d.tfm", "itk_displacement3d.h5"])
def test_block_chain_is_an_immutable_tuple(name: str) -> None:
    """The cached chain is handed out as is, so it cannot be a list."""
    if name.endswith(".h5"):
        pytest.importorskip("h5py")
    block = io.transformations.load(data_dir / name)[-1]
    assert isinstance(block.transformations, tuple)
    length = len(block)
    with pytest.raises(TypeError):
        del block[0]
    assert len(block) == length


def test_assigning_an_empty_chain_takes_effect() -> None:
    """An empty chain is a chain, not the absence of one."""
    block = TfmTransform.from_file(data_dir / "itk_affine3d.tfm")[0]
    assert len(block) == 4
    block.transformations = []
    assert len(block) == 0
    block.transformations = None
    assert len(block) == 4


def test_replace_rebuilds_the_chain_from_the_new_parameters() -> None:
    """`replace` does not freeze a chain derived from the old parameters."""
    block = TfmTransform.from_file(data_dir / "itk_affine3d.tfm")[0]
    assert len(block) == 4

    parameters = np.asarray(block.parameters).copy()
    parameters[-3:] = [100.0, 200.0, 300.0]
    copy = replace(block, parameters=parameters)

    np.testing.assert_allclose(
        np.asarray(copy.translation.translation), [100.0, 200.0, 300.0]
    )
    np.testing.assert_allclose(
        np.asarray(copy.transformations[-1].translation),
        [100.0, 200.0, 300.0],
    )
    # The original is untouched and shares no chain object.
    np.testing.assert_allclose(
        np.asarray(block.transformations[-1].translation), [10.0, 5.0, 2.0]
    )
    assert copy.transformations is not block.transformations


def test_warp_block_endpoints_do_not_decode_the_field() -> None:
    """The endpoints of a warp block come from its header, not its field."""
    pytest.importorskip("h5py")
    block = io.transformations.load(data_dir / "itk_displacement3d.h5")[-1]
    assert isinstance(block, itk.ItkDisplacementBase)

    assert block.input == block.output
    assert not hasattr(block, "_cache_field")

    # Asking for the chain decodes the field.
    assert len(block) == 3
    assert hasattr(block, "_cache_field")


# ----------------------------------------------------------------------
#   SIMILARITY BLOCKS
# ----------------------------------------------------------------------
# ITK parameterizes a similarity by one scale factor, exposed as a
# length-one `Scaling` whose endpoints state the number of axes.


def _similarity_2d(
    scale: float, angle: float, translation: tuple, center: tuple
) -> itk.ItkStruct:
    return itk.ItkStruct(
        type=itk.ItkTransformClass.Similarity2DTransform,
        precision="double",
        ndim_input=2,
        ndim_output=2,
        parameters=(scale, angle) + tuple(translation),
        fixed_parameters=tuple(center),
    )


def _similarity_3d(
    scale: float, versor: tuple, translation: tuple, center: tuple
) -> itk.ItkStruct:
    return itk.ItkStruct(
        type=itk.ItkTransformClass.Similarity3DTransform,
        precision="double",
        ndim_input=3,
        ndim_output=3,
        parameters=tuple(versor) + tuple(translation) + (scale,),
        fixed_parameters=tuple(center),
    )


def _versor_matrix(versor: tuple) -> np.ndarray:
    """Return the rotation matrix of the vector part of a unit quaternion."""
    x, y, z = versor
    w = np.sqrt(1.0 - (x * x + y * y + z * z))
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_similarity_scale_is_a_one_element_vector(ndim: int) -> None:
    """The scale factor is a one-element vector, not repeated per axis."""
    if ndim == 2:
        block = _similarity_2d(1.5, 0.3, (4.0, -2.0), (10.0, 20.0))
    else:
        block = _similarity_3d(
            1.5, (0.1, 0.2, 0.3), (4.0, -2.0, 7.0), (10.0, 20.0, 30.0)
        )

    scaling = block.scaling
    assert isinstance(scaling, xforms.Scaling)
    np.testing.assert_allclose(np.asarray(scaling.scale), [1.5])

    # The endpoints let a single element broadcast over every axis.
    assert scaling.input == block.input
    assert scaling.output == block.output


def test_similarity_2d_composes_the_expected_affine() -> None:
    """A 2-D block collapses to [sR | c + t - sRc]."""
    scale, angle = 1.5, 0.3
    translation = np.array([4.0, -2.0])
    center = np.array([10.0, 20.0])
    linear = scale * np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    block = _similarity_2d(scale, angle, translation, center)
    matrix = np.asarray(block.compute().to(xforms.Affine, lossy=True).matrix)
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(matrix[:, :2], linear)
    np.testing.assert_allclose(
        matrix[:, 2], center + translation - linear @ center
    )


def test_similarity_3d_composes_the_expected_affine() -> None:
    """A 3-D block also collapses to [sR | c + t - sRc]."""
    scale = 1.5
    versor = (0.1, 0.2, 0.3)
    translation = np.array([4.0, -2.0, 7.0])
    center = np.array([10.0, 20.0, 30.0])
    linear = scale * _versor_matrix(versor)

    block = _similarity_3d(scale, versor, translation, center)
    matrix = np.asarray(block.compute().to(xforms.Affine, lossy=True).matrix)
    assert matrix.shape == (3, 4)
    np.testing.assert_allclose(matrix[:, :3], linear)
    np.testing.assert_allclose(
        matrix[:, 3], center + translation - linear @ center
    )


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_itk_systems_are_lps_millimetres_in_every_dimension(ndim: int) -> None:
    # Oriented axes leave the unit open, so every dimension states it.
    from brainhops.datamodel.units import Unit
    from brainhops.io.transformations.itk._systems import _make_system

    system = _make_system(ndim)
    assert len(system.axes) == ndim
    assert all(axis.unit is Unit("mm") for axis in system.axes)
    expected = [
        "right-to-left",
        "anterior-to-posterior",
        "inferior-to-superior",
    ]
    values = [getattr(axis.orientation, "value", None) for axis in system.axes]
    assert values[:3] == expected[:ndim]


# ----------------------------------------------------------------------
#   COMPOSITE ORDER
# ----------------------------------------------------------------------
# A file [Composite, T0, T1] maps x to T0(T1(x)), since ITK applies the
# queue back to front, while a brainhops sequence lists [T1, T0].

# The two blocks do not commute: one scales and the other translates
# by SHIFT.
SHIFT = [10.0, -20.0, 30.0]
SCALE = [2.0, 3.0, 4.0]
COMPOSITE = [
    ("CompositeTransform", [], []),
    ("TranslationTransform", SHIFT, []),
    ("ScaleTransform", SCALE, [0.0, 0.0, 0.0]),
]
POINTS = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0], [0.0, 0.0, 0.0]])


def _itk_order(points: np.ndarray) -> np.ndarray:
    """Compute T0(T1(x)) by hand, scaling first and translating second."""
    return points * SCALE + SHIFT


def _apply(transform: xforms.Transformation, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(
        transform.compute().to(xforms.Affine, lossy=True).matrix
    )
    return points @ matrix[:, :-1].T + matrix[:, -1]


def _write_tfm(path: Path, blocks: list) -> Path:
    lines = ["#Insight Transform File V1.0"]
    for index, (name, parameters, fixed) in enumerate(blocks):
        lines.append(f"#Transform {index}")
        lines.append(f"Transform: {name}_double_3_3")
        if name != "CompositeTransform":
            lines.append("Parameters: " + " ".join(map(str, parameters)))
            lines.append("FixedParameters: " + " ".join(map(str, fixed)))
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_h5(path: Path, blocks: list) -> Path:
    """Write blocks in the itk::HDF5TransformIO layout, one group per block."""
    h5py = pytest.importorskip("h5py")
    string = h5py.string_dtype("ascii")
    with h5py.File(path, "w") as f:
        f["ITKVersion"] = np.array([b"5.4.0"], dtype=string)
        group = f.create_group("TransformGroup")
        for index, (name, parameters, fixed) in enumerate(blocks):
            node = group.create_group(str(index))
            node["TransformType"] = np.array(
                [f"{name}_double_3_3".encode()], dtype=string
            )
            if name != "CompositeTransform":
                node["TransformParameters"] = np.asarray(
                    parameters, dtype="f8"
                )
                node["TransformFixedParameters"] = np.asarray(
                    fixed, dtype="f8"
                )
    return path


WRITERS = {"tfm": _write_tfm, "h5": _write_h5}


@pytest.mark.parametrize("ext", sorted(WRITERS))
def test_composite_blocks_apply_in_itks_order(tmp_path, ext: str) -> None:  # noqa: ANN001
    path = WRITERS[ext](tmp_path / f"composite.{ext}", COMPOSITE)
    transform = io.transformations.load(path)
    assert [block.type for block in transform] == [
        itk.ItkTransformClass.ScaleTransform,
        itk.ItkTransformClass.TranslationTransform,
    ]
    np.testing.assert_allclose(_apply(transform, POINTS), _itk_order(POINTS))


@pytest.mark.parametrize("ext", sorted(WRITERS))
def test_single_block_files_are_unchanged(tmp_path, ext: str) -> None:  # noqa: ANN001
    path = WRITERS[ext](tmp_path / f"single.{ext}", COMPOSITE[1:2])
    (block,) = io.transformations.load(path)
    assert block.type == itk.ItkTransformClass.TranslationTransform
    np.testing.assert_allclose(_apply(block, POINTS), POINTS + SHIFT)


@pytest.mark.parametrize("ext", sorted(WRITERS))
def test_a_plain_list_reads_its_first_transform(tmp_path, ext: str) -> None:  # noqa: ANN001
    """Without a composite header, the first block is read with a warning."""
    path = WRITERS[ext](tmp_path / f"list.{ext}", COMPOSITE[1:])
    with pytest.warns(UserWarning, match="holds 2 transforms"):
        (block,) = io.transformations.load(path)
    assert block.type == itk.ItkTransformClass.TranslationTransform


@pytest.mark.parametrize("ext", sorted(WRITERS))
def test_a_plain_list_reads_the_given_position(tmp_path, ext: str) -> None:  # noqa: ANN001
    path = WRITERS[ext](tmp_path / f"list.{ext}", COMPOSITE[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = io.transformations.load(path, position=0)
        (block,) = io.transformations.load(path, position=1)
    assert [b.type for b in first] == [
        itk.ItkTransformClass.TranslationTransform
    ]
    assert block.type == itk.ItkTransformClass.ScaleTransform
    np.testing.assert_allclose(_apply(block, POINTS), POINTS * SCALE)


@pytest.mark.parametrize("ext", sorted(WRITERS))
def test_a_position_out_of_range_is_refused(tmp_path, ext: str) -> None:  # noqa: ANN001
    from brainhops.io.base.parsers import ParserContentError

    path = WRITERS[ext](tmp_path / f"list.{ext}", COMPOSITE[1:])
    format = itk.tfm.TfmTransform if ext == "tfm" else itk.h5.H5Transform
    with pytest.raises(ParserContentError, match="no transform 2"):
        format.from_file(path, position=2)


@pytest.mark.parametrize("ext", sorted(WRITERS))
def test_a_composite_file_holds_one_transform(tmp_path, ext: str) -> None:  # noqa: ANN001
    from brainhops.io.base.parsers import ParserContentError

    path = WRITERS[ext](tmp_path / f"composite.{ext}", COMPOSITE)
    format = itk.tfm.TfmTransform if ext == "tfm" else itk.h5.H5Transform
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transform = format.from_file(path, position=0)
    assert [block.type for block in transform] == [
        itk.ItkTransformClass.ScaleTransform,
        itk.ItkTransformClass.TranslationTransform,
    ]
    with pytest.raises(ParserContentError, match="has 1 transform"):
        format.from_file(path, position=1)


@pytest.mark.parametrize("ext", sorted(WRITERS))
def test_a_composite_header_must_come_first(tmp_path, ext: str) -> None:  # noqa: ANN001
    from brainhops.io.base.parsers import ParserContentError

    blocks = [COMPOSITE[1], COMPOSITE[0], COMPOSITE[2]]
    path = WRITERS[ext](tmp_path / f"misplaced.{ext}", blocks)
    format = itk.tfm.TfmTransform if ext == "tfm" else itk.h5.H5Transform
    with pytest.raises(ParserContentError, match="first block"):
        format.from_file(path)


def test_h5_blocks_are_read_by_number(tmp_path) -> None:  # noqa: ANN001
    """HDF5 groups are read by number, so 10 comes after 2."""
    shifts = [[float(i), 0.0, 0.0] for i in range(12)]
    blocks = [COMPOSITE[0]] + [
        ("TranslationTransform", shift, []) for shift in shifts
    ]
    transform = io.transformations.load(_write_h5(tmp_path / "x.h5", blocks))
    read = [list(np.asarray(block.parameters)) for block in transform]
    assert read == shifts[::-1]


@pytest.mark.parametrize("ext", ["tfm", "h5"])
def test_composite_fixture_matches_hand_computed_itk_order(ext: str) -> None:
    """ITK scales first, then applies the affine about its center."""
    if ext == "h5":
        pytest.importorskip("h5py")
    transform = io.transformations.load(
        data_dir / f"itk_composite_affine3d.{ext}"
    )
    linear = np.array([[0.9, 0.1, 0.0], [-0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
    center = np.full(3, 50.0)
    translation = np.array([10.0, 5.0, 2.0])
    scaled = POINTS * [1.2, 0.8, 1.0]
    expected = (scaled - center) @ linear.T + center + translation
    np.testing.assert_allclose(_apply(transform, POINTS), expected)
    # A single point computed by hand.
    np.testing.assert_allclose(
        _apply(transform, np.array([[1.0, 2.0, 3.0]])),
        [[11.24, 16.32, 5.0]],
    )


@pytest.mark.parametrize("ext", ["tfm", "h5"])
def test_composite_order_matches_simpleitk(tmp_path, ext: str) -> None:  # noqa: ANN001
    sitk = pytest.importorskip("SimpleITK")
    paths = [
        WRITERS[ext](tmp_path / f"composite.{ext}", COMPOSITE),
        data_dir / f"itk_composite_affine3d.{ext}",
    ]
    for path in paths:
        reference = sitk.ReadTransform(str(path))
        expected = np.array(
            [reference.TransformPoint(tuple(map(float, p))) for p in POINTS]
        )
        transform = io.transformations.load(path)
        np.testing.assert_allclose(_apply(transform, POINTS), expected)


def test_composite_order_matches_nitransforms(tmp_path) -> None:  # noqa: ANN001
    """nitransforms reverses an ITK .h5 composite into the same chain."""
    pytest.importorskip("h5py")
    manip = pytest.importorskip("nitransforms.manip")
    # nitransforms reads only affine and displacement blocks.
    shift = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, *SHIFT]
    scale = [SCALE[0], 0, 0, 0, SCALE[1], 0, 0, 0, SCALE[2], 0, 0, 0]
    path = _write_h5(
        tmp_path / "composite.h5",
        [
            COMPOSITE[0],
            ("AffineTransform", shift, [0.0, 0.0, 0.0]),
            ("AffineTransform", scale, [0.0, 0.0, 0.0]),
        ],
    )
    chain = manip.TransformChain.from_filename(str(path), fmt="itk")
    lps_to_ras = np.array([-1.0, -1.0, 1.0])
    expected = (
        np.asarray(chain.map(POINTS * lps_to_ras), dtype=float) * lps_to_ras
    )
    transform = io.transformations.load(path)
    np.testing.assert_allclose(_apply(transform, POINTS), expected)
    np.testing.assert_allclose(expected, _itk_order(POINTS))


def test_a_block_compares_by_identity() -> None:
    # Identity comparison holds although the struct base comes first.
    block = _versor_rigid_3d((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (0, 0, 0))
    assert isinstance(block, xforms.Transformation)
    assert block == block
    assert block != _versor_rigid_3d((0, 0, 0), (1, 2, 3), (0, 0, 0))
    assert len({block, block}) == 1
    assert itk.ItkStruct.__eq__ is object.__eq__
