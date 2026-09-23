# stdlib
from pathlib import Path

# dependencies
import numpy as np
import pytest

# internals
from bagof.magic import replace

from brainhops import io
from brainhops._core import affines
from brainhops.datamodel import transformations as xforms
from brainhops.io.transformations import itk

data_dir = Path(__file__).parent / "data"

FILES_H5 = list(data_dir.glob("*.h5"))
FILES_TFM = list(data_dir.glob("*.tfm"))

TFMTransform = io.transformations.itk.tfm.TFMTransform


@pytest.mark.parametrize("filename", FILES_H5)
@pytest.mark.parametrize("load", [True, False])
@pytest.mark.parametrize("keep_open", [True, False])
def test_read_h5(filename: str, load: bool, keep_open: bool) -> None:
    pytest.importorskip("h5py")
    H5Transform = io.transformations.itk.h5.H5Transform
    transform = H5Transform.from_file(filename, load=load, keep_open=keep_open)
    # trigger conversion
    transforms = transform.transformations  # noqa: F841


@pytest.mark.parametrize("filename", FILES_TFM)
def test_read_tfm(filename: str) -> None:
    transform = TFMTransform.from_file(filename)
    # trigger conversion
    transforms = transform.transformations  # noqa: F841


# ----------------------------------------------------------------------
#   REGISTRY / DISPATCH
# ----------------------------------------------------------------------
#
# The ITK readers register themselves with the shared parser registry, so
# the generic `io.load` and the scoped `io.transformations.load` both
# resolve an ITK file to its reader without being told the format.


@pytest.mark.parametrize("filename", FILES_TFM)
def test_tfm_is_dispatched(filename: str) -> None:
    assert io.transformations.sniff(filename) is TFMTransform
    assert io.sniff(filename) is TFMTransform
    assert type(io.transformations.load(filename)) is TFMTransform
    assert type(io.load(filename)) is TFMTransform


@pytest.mark.parametrize("filename", FILES_H5)
def test_h5_is_dispatched(filename: str) -> None:
    pytest.importorskip("h5py")
    H5Transform = io.transformations.itk.h5.H5Transform
    assert io.transformations.sniff(filename) is H5Transform
    assert io.sniff(filename) is H5Transform
    assert type(io.transformations.load(filename)) is H5Transform
    assert type(io.load(filename)) is H5Transform


def test_tfm_header_only_is_read_as_empty(tmp_path) -> None:  # noqa: ANN001
    """
    A `.tfm` file that carries only the version header has no transform
    block. `peekable_lines` drops the header comment and then yields its
    end sentinel, which is not a string, so `sniff_line` must return a
    numeric confidence rather than raise or claim the file.
    """
    header_only = tmp_path / "header_only.tfm"
    header_only.write_text("# Insight Transform File V1.0\n")

    assert TFMTransform.sniff_line("") == 0.0
    assert list(TFMTransform.from_file(header_only).transformations) == []
    # The `.tfm` extension still routes it to the ITK reader, which reads
    # it as an empty transform.
    assert io.transformations.sniff(header_only) is TFMTransform
    assert type(io.transformations.load(header_only)) is TFMTransform


# ----------------------------------------------------------------------
#   BLOCKS AS TRANSFORMATIONS
# ----------------------------------------------------------------------
#
# An ITK file is a chain of transform blocks, and each block is itself a
# brainhops transformation: a structured `Sequence` whose children are
# named, lazily evaluated slots. Nothing is converted after parsing.


@pytest.mark.parametrize("filename", FILES_TFM + FILES_H5)
def test_blocks_are_transformations(filename: str) -> None:
    transform = io.transformations.load(filename)
    assert isinstance(transform, xforms.Sequence)
    for block in transform.transformations:
        assert isinstance(block, itk.ITKStruct)
        assert isinstance(block, xforms.Sequence)
        # A block is a non-empty chain, and every child is a
        # transformation in its own right.
        assert len(block) >= 1
        assert all(isinstance(t, xforms.Transformation) for t in block)


def test_affine_block_exposes_named_cached_slots() -> None:
    block = TFMTransform.from_file(data_dir / "itk_affine3d.tfm")[0]
    assert isinstance(block, itk.ITKAffineBase)

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

    # Assigning a chain overrides the derived one; clearing it restores
    # the derived one.
    block.transformations = [xforms.Identity()]
    assert len(block) == 1
    block.transformations = None
    assert len(block) == 4


def _versor_rigid_3d(
    versor: tuple, translation: tuple, center: tuple
) -> itk.ITKStruct:
    return itk.ITKStruct(
        type=itk.ITKTransformClass.VersorRigid3DTransform,
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
    """The block collapses to `[R | c + t - R.c]`.

    The zero versor only exercises `R = I`, which hides every mistake in
    how the center of rotation is folded in. This uses a quarter turn
    about z -- versor `(0, 0, sin(pi/4))` -- and a center away from the
    origin, so the linear part and the offset are both non-trivial.
    """
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
    """A versor rounded just past the unit sphere still loads.

    ITK renormalizes a versor whose vector part overshoots by a rounding
    error, so a file that writes a half-turn as `1.0000000002` opens
    there. It must open here too, and give the same half-turn. A versor
    that is genuinely too long is still refused.
    """
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
    for name, order, coeff in [
        ("itk_displacement3d.h5", 1, False),
        ("itk_bspline3d.h5", 3, True),
    ]:
        block = io.transformations.load(data_dir / name)[-1]
        assert isinstance(block, itk.ITKDisplacementBase)
        assert list(block) == [
            block.lps2voxel,
            block.displacement,
            block.voxel2lps,
        ]
        assert block.order == order
        assert block.coeff == coeff
        assert block.displacement.field is block.field
        assert block.field.shape[-1] == 3


# ----------------------------------------------------------------------
#   WARP MEMORY LAYOUT
# ----------------------------------------------------------------------
#
# ITK flattens its two kinds of warp differently, and the difference is
# invisible in random data: either layout reshapes without complaint and
# gives a plausible field. The fixtures below therefore carry a ramp
# whose every entry names its own voxel and component --
# `1000 * component + 100 * x + 10 * y + z` -- so a transposed axis or a
# mistaken component stride cannot hide. Their grids have unit spacing
# and an identity direction, so the world-space values SimpleITK reports
# are already in the grid's own voxel units and the expected arrays can
# be compared to `block.field` directly.
#
# The expected arrays are stored beside the fixtures, taken from
# SimpleITK's own view of the images -- never from our decoder, which is
# the thing under test. `tests/data/generate_itk_fixtures.py` regenerates
# both, and is the only place SimpleITK is needed.


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
    """Each kind of warp is read in the layout ITK writes it in.

    A `DisplacementFieldTransform`'s parameters are the raw buffer of an
    image of vectors, so the component index varies fastest. A
    `BSplineTransform`'s are one scalar coefficient image per axis,
    written back to back, so it varies slowest. Reading either as the
    other transposes the warp silently.
    """
    block = io.transformations.load(data_dir / f"{name}.tfm")[-1]
    assert isinstance(block, itk.ITKDisplacementBase)

    expected = np.load(data_dir / f"{name}_expected.npy")
    field = np.asarray(block.field)
    assert field.shape == expected.shape
    np.testing.assert_allclose(field, expected)
    assert block.interleaved is interleaved

    # ... and the other layout really would have given something else,
    # so the assertion above is not satisfied by both.
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
    """The stored expectations are still what ITK itself reports.

    The tests above run from the committed arrays so the suite does not
    need SimpleITK. This re-derives them when it happens to be installed,
    so a stale fixture cannot quietly outlive the ITK behavior it pins.
    """
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
    """Both of ITK's angle orders give the matrix ITK gives.

    `Euler3DTransform` composes its three axis rotations as `Rz @ Rx @ Ry`
    unless its `ComputeZYX` flag is set, in which case it composes
    `Rz @ Ry @ Rx`. The angles alone do not say which, so a reader that
    assumes one order silently returns a valid but wrong rotation for
    every file written with the other.

    The expected matrix is ITK's own: its rotation from `GetMatrix`, and
    its offset -- which is where the center of rotation folds in -- read
    off `TransformPoint` at the origin.
    """
    block = TFMTransform.from_file(data_dir / f"{name}.tfm")[0]
    assert block.type == itk.ITKTransformClass.Euler3DTransform

    matrix = np.asarray(block.compute().to(xforms.Affine, lossy=True).matrix)
    np.testing.assert_allclose(
        matrix, np.load(data_dir / f"{name}_expected.npy"), atol=1e-12
    )


def test_euler_3d_reads_the_modern_four_fixed_parameters() -> None:
    """ITK >= 5 writes the `ComputeZYX` flag as a fourth fixed parameter.

    A reader that insists on exactly three refuses every Euler 3-D file
    current ITK writes. The fourth entry is a flag, not a coordinate, so
    it must also stay out of the center of rotation -- a four-long center
    would make the block claim a fourth axis.
    """
    plain = TFMTransform.from_file(data_dir / "itk_euler3d.tfm")[0]
    zyx = TFMTransform.from_file(data_dir / "itk_euler3d_zyx.tfm")[0]

    assert len(plain.fixed_parameters) == 4
    assert plain.compute_zyx is False
    assert zyx.compute_zyx is True

    for block in (plain, zyx):
        center = np.asarray(block.center)
        assert center.shape == (3,)
        np.testing.assert_allclose(center, [4.0, 5.0, 6.0])

    # A pre-5 file writes the center alone, and is read as ZXY -- the
    # only order that existed before the flag did.
    legacy = itk.ITKStruct(
        type=itk.ITKTransformClass.Euler3DTransform,
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
    """The stored Euler expectations are still what ITK itself reports."""
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
    """Blocks live in `transformations`, so there is no second list."""
    transform = TFMTransform.from_file(data_dir / "itk_affine3d.tfm")
    assert not hasattr(transform, "transform_group")


# ----------------------------------------------------------------------
#   WARP BLOCKS COMPOSE AND INVERT
# ----------------------------------------------------------------------
#
# A warp block's two affines are built from the grid geometry its fixed
# parameters carry. They must be compact `(ndim, ndim + 1)` matrices: a
# homogeneous one is read as a transformation of `ndim + 1` coordinates
# and the chain no longer composes. Nothing below asserts on shapes
# alone -- each case computes or inverts the chain, which is what the
# mis-shaped matrices broke.


@pytest.mark.parametrize("name", ["itk_displacement3d.h5", "itk_bspline3d.h5"])
def test_warp_block_affines_are_compact(name: str) -> None:
    pytest.importorskip("h5py")
    block = io.transformations.load(data_dir / name)[-1]
    ndim = block.ndim_input
    vox2lps = np.asarray(block.voxel2lps.matrix)
    lps2vox = np.asarray(block.lps2voxel.matrix)
    assert vox2lps.shape == (ndim, ndim + 1)
    assert lps2vox.shape == (ndim, ndim + 1)

    # The voxel-to-LPS affine is the grid geometry the fixed parameters
    # describe: `direction @ diag(spacing)` beside the origin.
    origin = np.asarray(block.fixed_parameters)[ndim : 2 * ndim]
    spacing = np.asarray(block.fixed_parameters)[2 * ndim : 3 * ndim]
    direction = np.asarray(block.fixed_parameters)[
        3 * ndim : 3 * ndim + ndim * ndim
    ].reshape(ndim, ndim)
    np.testing.assert_allclose(vox2lps[:, :ndim], direction @ np.diag(spacing))
    np.testing.assert_allclose(vox2lps[:, ndim], origin)

    # ... and the two affines undo one another.
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

    # The leading affine is the world-to-voxel map of the warp grid, and
    # the field follows it in the grid's own units.
    assert isinstance(result, xforms.Sequence)
    assert isinstance(result[0], xforms.Affine)
    np.testing.assert_allclose(
        np.asarray(result[0].matrix), affines.inv(vox2lps)
    )
    assert isinstance(result[-1], xforms.DisplacementField)
    assert np.asarray(result[-1].field).shape == np.asarray(block.field).shape
    assert result[-1].order == block.order
    assert result[-1].coeff == block.coeff
    assert result.input == block.input
    assert result.output == block.output


def test_composite_warp_computes() -> None:
    """A warp block composes with the affine blocks around it."""
    pytest.importorskip("h5py")
    transform = io.load(data_dir / "itk_composite_displacement3d.h5")
    result = transform.compute()
    assert isinstance(result, xforms.Sequence)
    # The affines on either side fold into one, leaving affine + field.
    assert isinstance(result[0], xforms.Affine)
    ndim = transform[-1].ndim_input
    assert np.asarray(result[0].matrix).shape == (ndim, ndim + 1)
    assert isinstance(result[-1], xforms.DisplacementField)


@pytest.mark.parametrize("name", ["itk_displacement3d.h5", "itk_bspline3d.h5"])
def test_warp_block_inverts(name: str) -> None:
    """Every child of a warp block has an inverse, so the block does."""
    pytest.importorskip("h5py")
    block = io.transformations.load(data_dir / name)[-1]
    inverse = block.inverse()
    assert isinstance(inverse, xforms.Sequence)
    assert len(inverse) == len(block)
    assert inverse.input == block.output
    assert inverse.output == block.input
    # The chain reads back the other way round.
    assert isinstance(inverse[0], xforms.Inverse)
    assert inverse[0].forward is block.voxel2lps
    assert inverse[-1].forward is block.lps2voxel


def test_warp_block_grid_is_read_at_its_own_dimensionality() -> None:
    """The grid geometry is read off `ndim_input`, not off a 3-D layout."""
    # A 2-D grid writes 2 + 2 + 2 + 4 fixed parameters: shape, origin,
    # spacing, then the 2x2 direction matrix.
    block = itk.ITKStruct(
        type=itk.ITKTransformClass.DisplacementFieldTransform,
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
    """An empty chain is a chain, not 'no chain given'."""
    block = TFMTransform.from_file(data_dir / "itk_affine3d.tfm")[0]
    assert len(block) == 4
    block.transformations = []
    assert len(block) == 0
    block.transformations = None
    assert len(block) == 4


def test_replace_rebuilds_the_chain_from_the_new_parameters() -> None:
    """`replace` must not freeze the chain derived from the old ones."""
    block = TFMTransform.from_file(data_dir / "itk_affine3d.tfm")[0]
    assert len(block) == 4  # build and cache the derived chain

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
    # The original is untouched, and the two do not share a chain object.
    np.testing.assert_allclose(
        np.asarray(block.transformations[-1].translation), [10.0, 5.0, 2.0]
    )
    assert copy.transformations is not block.transformations


def test_warp_block_endpoints_do_not_decode_the_field() -> None:
    """Reading a block's endpoints must not touch the warp data.

    A warp block declares its endpoints from the dimensions its file
    states. Deriving them the way a plain `Sequence` does would build the
    chain, and building the chain decodes the field -- a full read of a
    delayed array for a question the header already answers.
    """
    pytest.importorskip("h5py")
    block = io.transformations.load(data_dir / "itk_displacement3d.h5")[-1]
    assert isinstance(block, itk.ITKDisplacementBase)

    assert block.input == block.output
    assert not hasattr(block, "_cache_field")

    # And asking for the chain does decode it.
    assert len(block) == 3
    assert hasattr(block, "_cache_field")


# ----------------------------------------------------------------------
#   SIMILARITY BLOCKS
# ----------------------------------------------------------------------
#
# ITK parameterizes a similarity by a single scale factor, which the
# block exposes as a `Scaling`. A scaling is parameterized by a vector,
# so the ITK scalar is exposed as a vector of length one -- not repeated
# per axis -- and the slot carries the block's endpoints so that the one
# element broadcasts over the right number of axes.


def _similarity_2d(
    scale: float, angle: float, translation: tuple, center: tuple
) -> itk.ITKStruct:
    return itk.ITKStruct(
        type=itk.ITKTransformClass.Similarity2DTransform,
        precision="double",
        ndim_input=2,
        ndim_output=2,
        parameters=(scale, angle) + tuple(translation),
        fixed_parameters=tuple(center),
    )


def _similarity_3d(
    scale: float, versor: tuple, translation: tuple, center: tuple
) -> itk.ITKStruct:
    return itk.ITKStruct(
        type=itk.ITKTransformClass.Similarity3DTransform,
        precision="double",
        ndim_input=3,
        ndim_output=3,
        parameters=tuple(versor) + tuple(translation) + (scale,),
        fixed_parameters=tuple(center),
    )


def _versor_matrix(versor: tuple) -> np.ndarray:
    """A rotation matrix from the vector part of a unit quaternion."""
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
    """The ITK scalar is exposed as a vector of length one.

    A `Scaling` is parameterized by a vector of factors, and ITK stores
    the isotropic case as one number. Writing that number out once per
    axis would state a dimensionality the ITK parameter does not carry,
    so it is exposed as it is: a single factor that broadcasts.
    """
    if ndim == 2:
        block = _similarity_2d(1.5, 0.3, (4.0, -2.0), (10.0, 20.0))
    else:
        block = _similarity_3d(
            1.5, (0.1, 0.2, 0.3), (4.0, -2.0, 7.0), (10.0, 20.0, 30.0)
        )

    scaling = block.scaling
    assert isinstance(scaling, xforms.Scaling)
    np.testing.assert_allclose(np.asarray(scaling.scale), [1.5])

    # One element only broadcasts to the right number of axes when
    # something says how many there are, so the slot is given the
    # block's own space at both ends.
    assert scaling.input == block.input
    assert scaling.output == block.output


def test_similarity_2d_composes_the_expected_affine() -> None:
    """The block collapses to `[sR | c + t - sR.c]`.

    This is the composition the one-element scale has to survive: the
    factor multiplies every axis of the rotation, and the center of
    rotation is folded in with the scaled linear part.
    """
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
    """The 3-D block collapses to `[sR | c + t - sR.c]` as well."""
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
