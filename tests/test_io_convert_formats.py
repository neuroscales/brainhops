"""
General transformations converted into the NIfTI and SPM formats.

A converter returns an instance of its format that is exactly the map it
was given, or raises a `ConversionError` that says what the format cannot
hold. The endpoints are read from the transformation, and bridged to the
format's: an affine or a field in LPS is flipped into RAS. These tests
check the maps -- matrices exactly, fields at points on and off their
nodes, and outside their grid -- before and after a round trip through
`save` and `load`, and that `t.to(Format)`, `Format.from_instance(t)`,
`Format.from_any(t)` and `io.save` are one conversion.
"""

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    LPSmm,
    RASmm,
    VoxelCoordinateSystem,
)
from brainhops.errors import ConversionError  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    WriterError,
)
from brainhops.io.transformations.base import (  # noqa: E402
    WritableFileBasedTransformation,
)
from brainhops.io.transformations.base.affines import (  # noqa: E402
    LPSToVoxel,
    RASToVoxel,
    VoxelToLPS,
    VoxelToRAS,
)
from brainhops.io.transformations.fsl.fnirt import (  # noqa: E402
    FnirtWarpField,
)
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiRASDisplacementField,
    NiftiRASToVoxel,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.spm.y import (  # noqa: E402
    SpmCoordinatesField,
)

SHAPE = (4, 5, 6)
"""Grid shape: small, and no two axes of the same length."""

# A voxel-to-world affine with a permutation, a flip, anisotropic spacing
# and an offset. Its entries are exact in float32, as a NIfTI header
# stores them, so a round trip through a file keeps it exactly.
VOX2WORLD = np.array(
    [
        [0.0, -2.0, 0.0, 10.0],
        [1.5, 0.0, 0.0, -20.0],
        [0.0, 0.0, 2.5, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

LPS2RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""LPS and RAS differ by the sign of their first two axes."""

VOXEL = VoxelCoordinateSystem()


def _displacements(seed: int = 0) -> np.ndarray:
    """A smooth-free `(X, Y, Z, 3)` field, in the voxels of its grid."""
    return np.random.default_rng(seed).normal(size=(*SHAPE, 3))


def _voxels() -> np.ndarray:
    """Points of the grid: on its nodes, between them, and outside it."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [0.5, 1.25, 2.75],
            [2.3, 3.7, 4.1],
            [1.9, 0.1, 5.0],
            [-1.5, 2.0, 3.0],
            [4.5, 6.0, -2.0],
        ]
    )


def _world(vox2world: np.ndarray = VOX2WORLD) -> np.ndarray:
    """The points of `_voxels`, in the world of the grid."""
    return _voxels() @ vox2world[:3, :3].T + vox2world[:3, 3]


def _apply(xform: xforms.Transformation, points: np.ndarray) -> np.ndarray:
    """Map points through a transformation."""
    points = xforms.CoordinatesField(field=points[:, None, None, :])
    out = xforms.Sequence(transformations=[points, xform]).compute()
    field = out.to(xforms.CoordinatesField).field
    return np.asarray(field)[:, 0, 0, :]


def _flip(points: np.ndarray) -> np.ndarray:
    """LPS points in RAS, or the reverse."""
    return points * np.array([-1.0, -1.0, 1.0])


def _lps_displacement_chain(degree: int = 1) -> xforms.Sequence:
    """An LPS-to-LPS field of displacements, as ITK holds one."""
    vox2lps = LPS2RAS @ VOX2WORLD
    return xforms.Sequence(
        transformations=[
            LPSToVoxel(matrix=np.linalg.inv(vox2lps)[:-1]),
            xforms.DisplacementField(
                field=_displacements(),
                input=VOXEL,
                output=VOXEL,
                degree=degree,
            ),
            VoxelToLPS(matrix=vox2lps[:-1]),
        ]
    )


def _ras_displacement_chain() -> xforms.Sequence:
    """A RAS-to-RAS field of displacements."""
    return xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=np.linalg.inv(VOX2WORLD)[:-1]),
            xforms.DisplacementField(
                field=_displacements(), input=VOXEL, output=VOXEL
            ),
            VoxelToRAS(matrix=VOX2WORLD[:-1]),
        ]
    )


def _coordinates() -> np.ndarray:
    """A field of world coordinates on the grid."""
    i, j, k = np.meshgrid(*map(np.arange, SHAPE), indexing="ij")
    grid = np.stack([i, j, k], axis=-1).astype(float) + _displacements(1)
    return grid @ VOX2WORLD[:3, :3].T + VOX2WORLD[:3, 3]


def _lps_spm_chain() -> xforms.Sequence:
    """An LPS-to-LPS map through a field of LPS coordinates, as SPM's."""
    vox2lps = LPS2RAS @ VOX2WORLD
    return xforms.Sequence(
        transformations=[
            LPSToVoxel(matrix=np.linalg.inv(vox2lps)[:-1]),
            xforms.CoordinatesField(
                field=_flip(_coordinates()), input=VOXEL, output=LPSmm()
            ),
        ]
    )


def _conversions(
    t: xforms.Transformation, cls: type
) -> tx.List[xforms.Transformation]:
    """`t` converted to `cls` in every way there is to ask for it."""
    return [t.to(cls), cls.from_instance(t), cls.from_any(t)]


# ----------------------------------------------------------------------
#   AFFINES
# ----------------------------------------------------------------------


def test_a_voxel_to_ras_affine_is_held_as_it_is() -> None:
    affine = xforms.Affine(VOX2WORLD[:-1], input=VOXEL, output=RASmm())
    for nifti in _conversions(affine, NiftiVoxelToRAS):
        assert type(nifti) is NiftiVoxelToRAS
        np.testing.assert_array_equal(nifti.matrix, VOX2WORLD[:-1])


def test_an_affine_of_unknown_systems_maps_the_format_ones() -> None:
    nifti = xforms.Affine(VOX2WORLD[:-1]).to(NiftiVoxelToRAS)
    np.testing.assert_array_equal(nifti.matrix, VOX2WORLD[:-1])
    assert nifti.output == RASmm()


def test_an_affine_to_lps_is_bridged_into_ras(tmp_path) -> None:  # noqa: ANN001
    affine = VoxelToLPS(matrix=VOX2WORLD[:-1])
    expected = (LPS2RAS @ VOX2WORLD)[:-1]
    for nifti in _conversions(affine, NiftiVoxelToRAS):
        np.testing.assert_array_equal(nifti.matrix, expected)
        assert nifti.output == RASmm()
    io.save(affine, tmp_path / "affine.nii.gz")
    back = io.transformations.load(tmp_path / "affine.nii.gz")
    assert isinstance(back, NiftiVoxelToRAS)
    np.testing.assert_array_equal(back.matrix, expected)
    points = _voxels()
    np.testing.assert_array_equal(
        _apply(back, points), _flip(_apply(affine, points))
    )


def test_an_affine_from_lps_to_voxels_is_bridged() -> None:
    affine = LPSToVoxel(matrix=np.linalg.inv(VOX2WORLD)[:-1])
    nifti = affine.to(NiftiRASToVoxel)
    assert type(nifti) is NiftiRASToVoxel
    np.testing.assert_array_equal(
        nifti.matrix, (np.linalg.inv(VOX2WORLD) @ LPS2RAS)[:-1]
    )


def test_a_chain_of_affines_is_composed() -> None:
    scale = np.diag([2.0, 2.0, 2.0, 1.0])
    chain = xforms.Sequence(
        transformations=[
            VoxelToLPS(matrix=VOX2WORLD[:-1]),
            xforms.Affine(scale[:-1], input=LPSmm(), output=LPSmm()),
        ]
    )
    nifti = chain.to(NiftiVoxelToRAS)
    np.testing.assert_array_equal(
        nifti.matrix, (LPS2RAS @ scale @ VOX2WORLD)[:-1]
    )


@pytest.mark.parametrize(
    "affine",
    [
        xforms.Affine(VOX2WORLD[:-1], input=RASmm(), output=RASmm()),
        xforms.Affine(VOX2WORLD[:-1], input=VOXEL, output=VOXEL),
    ],
    ids=["world-to-world", "voxel-to-voxel"],
)
def test_an_affine_between_other_spaces_is_refused(
    affine: xforms.Affine,
) -> None:
    with pytest.raises(ConversionError, match="cannot be held exactly"):
        affine.to(NiftiVoxelToRAS)
    with pytest.raises(ConversionError, match="cannot be held exactly"):
        NiftiVoxelToRAS.from_any(affine)
    assert affine.to(NiftiVoxelToRAS, error=False) is False


def test_a_chain_with_a_field_is_not_an_affine() -> None:
    field = xforms.DisplacementField(
        field=_displacements(), input=VOXEL, output=VOXEL
    )
    chain = xforms.Sequence(
        transformations=[field, VoxelToRAS(matrix=VOX2WORLD[:-1])]
    )
    with pytest.raises(ConversionError, match="not an affine"):
        chain.to(NiftiVoxelToRAS)


# ----------------------------------------------------------------------
#   FIELDS OF DISPLACEMENTS
# ----------------------------------------------------------------------


def test_a_displacement_field_of_unknown_systems_is_held() -> None:
    field = xforms.DisplacementField(field=_displacements())
    points = _voxels()
    for nifti in _conversions(field, NiftiRASDisplacementField):
        assert type(nifti) is NiftiRASDisplacementField
        np.testing.assert_allclose(
            _apply(nifti, points), _apply(field, points), rtol=0, atol=1e-12
        )


def test_a_ras_displacement_chain_is_held(tmp_path) -> None:  # noqa: ANN001
    chain = _ras_displacement_chain()
    nifti = chain.to(NiftiRASDisplacementField)
    points = _world()
    expected = _apply(chain, points)
    np.testing.assert_allclose(_apply(nifti, points), expected, atol=1e-12)
    io.save(chain, tmp_path / "warp.nii.gz")
    back = io.transformations.load(tmp_path / "warp.nii.gz")
    assert isinstance(back, NiftiRASDisplacementField)
    np.testing.assert_allclose(_apply(back, points), expected, atol=1e-10)


def test_an_lps_displacement_chain_is_bridged_into_ras(tmp_path) -> None:  # noqa: ANN001
    # An ITK-like field: displacements of LPS points, in LPS.
    chain = _lps_displacement_chain()
    nifti = NiftiRASDisplacementField.from_any(chain)
    lps = _flip(_world())
    expected = _flip(_apply(chain, lps))
    np.testing.assert_allclose(_apply(nifti, _flip(lps)), expected, atol=1e-12)
    io.save(chain, tmp_path / "warp.nii.gz")
    back = io.transformations.load(tmp_path / "warp.nii.gz")
    assert isinstance(back, NiftiRASDisplacementField)
    np.testing.assert_allclose(_apply(back, _flip(lps)), expected, atol=1e-10)


def test_a_velocity_is_held_and_written_as_its_displacement(tmp_path) -> None:  # noqa: ANN001
    chain = list(_ras_displacement_chain())
    chain[1] = xforms.DisplacementField(
        data=0.1 * _displacements(), input=VOXEL, output=VOXEL, log=True
    )
    chain = xforms.Sequence(transformations=chain)
    nifti = chain.to(NiftiRASDisplacementField)
    assert not nifti.log
    points = _world()
    expected = _apply(chain, points)
    np.testing.assert_allclose(_apply(nifti, points), expected, atol=1e-12)
    io.save(chain, tmp_path / "warp.nii.gz")
    back = io.transformations.load(tmp_path / "warp.nii.gz")
    np.testing.assert_allclose(_apply(back, points), expected, atol=1e-10)


def test_a_cubic_displacement_field_is_refused() -> None:
    # NIfTI's values are read back with linear interpolation, which is
    # another map between the nodes than a cubic spline's.
    with pytest.raises(ConversionError, match="degree 3"):
        _lps_displacement_chain(degree=3).to(NiftiRASDisplacementField)


def test_ends_that_do_not_undo_each_other_are_refused() -> None:
    chain = list(_ras_displacement_chain())
    chain[2] = VoxelToRAS(matrix=(2 * VOX2WORLD)[:-1])
    chain = xforms.Sequence(transformations=chain)
    with pytest.raises(ConversionError, match="do not undo each other"):
        chain.to(NiftiRASDisplacementField)


def test_two_fields_are_not_composed_into_one() -> None:
    field = xforms.DisplacementField(field=_displacements())
    chain = xforms.Sequence(transformations=[field, field])
    with pytest.raises(ConversionError, match="holds 2 fields"):
        chain.to(NiftiRASDisplacementField)


def test_a_field_of_coordinates_is_not_one_of_displacements() -> None:
    # They extend differently outside their grid.
    coordinates = xforms.CoordinatesField(field=_coordinates())
    with pytest.raises(ConversionError, match="holds displacements"):
        coordinates.to(NiftiRASDisplacementField)
    with pytest.raises(ConversionError, match="holds coordinates"):
        xforms.DisplacementField(field=_displacements()).to(
            NiftiRASCoordinatesField
        )


# ----------------------------------------------------------------------
#   FIELDS OF COORDINATES
# ----------------------------------------------------------------------


def test_a_field_of_lps_coordinates_is_bridged_into_ras(tmp_path) -> None:  # noqa: ANN001
    field = xforms.CoordinatesField(
        field=_flip(_coordinates()), input=VOXEL, output=LPSmm()
    )
    points = _voxels()
    expected = _flip(_apply(field, points))
    for nifti in _conversions(field, NiftiRASCoordinatesField):
        assert type(nifti) is NiftiRASCoordinatesField
        np.testing.assert_array_equal(np.asarray(nifti.field), _coordinates())
        np.testing.assert_allclose(_apply(nifti, points), expected, atol=1e-12)
    io.save(field, tmp_path / "coordinates.nii.gz")
    back = io.transformations.load(tmp_path / "coordinates.nii.gz")
    assert isinstance(back, NiftiRASCoordinatesField)
    np.testing.assert_allclose(_apply(back, points), expected, atol=1e-10)


def test_an_affine_after_the_coordinates_is_applied_to_them() -> None:
    field = xforms.CoordinatesField(
        field=_coordinates(), input=VOXEL, output=RASmm()
    )
    scale = xforms.Affine(
        np.diag([2.0, 3.0, 4.0, 1.0])[:-1], input=RASmm(), output=RASmm()
    )
    chain = xforms.Sequence(transformations=[field, scale])
    nifti = chain.to(NiftiRASCoordinatesField)
    points = _voxels()
    np.testing.assert_allclose(
        _apply(nifti, points), _apply(chain, points), atol=1e-12
    )


def test_an_affine_before_the_coordinates_is_refused() -> None:
    # It would resample the field onto other voxels.
    shift = xforms.Affine(
        np.eye(4)[:-1] + [[0, 0, 0, 1]] * 3, input=VOXEL, output=VOXEL
    )
    field = xforms.CoordinatesField(field=_coordinates(), input=VOXEL)
    chain = xforms.Sequence(transformations=[shift, field])
    with pytest.raises(ConversionError, match="voxels of the file"):
        chain.to(NiftiRASCoordinatesField)


# ----------------------------------------------------------------------
#   SPM DEFORMATIONS
# ----------------------------------------------------------------------


def test_an_lps_deformation_is_written_as_spm(tmp_path) -> None:  # noqa: ANN001
    chain = _lps_spm_chain()
    lps = _flip(_world())
    expected = _flip(_apply(chain, lps))
    for spm in _conversions(chain, SpmCoordinatesField):
        assert type(spm) is SpmCoordinatesField
        np.testing.assert_allclose(
            _apply(spm, _flip(lps)), expected, atol=1e-10
        )
    io.save(chain, tmp_path / "y_deformation.nii")
    back = io.transformations.load(tmp_path / "y_deformation.nii")
    assert isinstance(back, SpmCoordinatesField)
    np.testing.assert_allclose(_apply(back, _flip(lps)), expected, atol=1e-10)
    np.testing.assert_array_equal(back.header.get_best_affine(), VOX2WORLD)


def test_an_spm_file_round_trips(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "y_source.nii"
    image = nb.Nifti1Image(_coordinates()[:, :, :, None, :], VOX2WORLD)
    image.header.set_intent(1007, name="Mapping")
    nb.save(image, str(path))
    spm = io.transformations.load(path)
    assert isinstance(spm, SpmCoordinatesField)
    io.save(spm, tmp_path / "y_copy.nii")
    back = io.transformations.load(tmp_path / "y_copy.nii")
    assert isinstance(back, SpmCoordinatesField)
    np.testing.assert_array_equal(back.header.get_best_affine(), VOX2WORLD)
    np.testing.assert_array_equal(
        np.asarray(back.rasfield.field), _coordinates()
    )
    assert back.header.get_intent() == ("vector", (), "Mapping")


def test_a_displacement_field_is_not_an_spm_deformation() -> None:
    with pytest.raises(ConversionError, match="holds coordinates"):
        _ras_displacement_chain().to(SpmCoordinatesField)


# ----------------------------------------------------------------------
#   SAVE
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj, name, cls",
    [
        (
            xforms.Affine(VOX2WORLD[:-1], input=VOXEL, output=RASmm()),
            "affine.nii.gz",
            NiftiVoxelToRAS,
        ),
        (
            xforms.DisplacementField(field=_displacements()),
            "warp.nii.gz",
            NiftiRASDisplacementField,
        ),
        (
            xforms.CoordinatesField(field=_coordinates()),
            "coordinates.nii.gz",
            NiftiRASCoordinatesField,
        ),
        (_ras_displacement_chain(), "warp.nii.gz", NiftiRASDisplacementField),
        (_lps_spm_chain(), "y_deformation.nii.gz", SpmCoordinatesField),
    ],
    ids=["affine", "displacements", "coordinates", "chain", "spm"],
)
def test_a_general_transformation_is_saved_in_the_format_it_converts_to(
    tmp_path,  # noqa: ANN001
    obj: xforms.Transformation,
    name: str,
    cls: type,
) -> None:
    io.save(obj, tmp_path / name)
    back = io.transformations.load(tmp_path / name)
    assert type(back) is cls
    converted = obj.to(cls)
    for points in (_voxels(), _world()):
        np.testing.assert_allclose(
            _apply(back, points), _apply(converted, points), atol=1e-10
        )


def test_save_gives_each_format_reason_to_refuse(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(WriterError) as info:
        io.save(_lps_displacement_chain(degree=3), tmp_path / "warp.nii.gz")
    message = str(info.value)
    assert "NiftiRASDisplacementField: ConversionError" in message
    assert "degree 3" in message


# ----------------------------------------------------------------------
#   FNIRT WRITING
# ----------------------------------------------------------------------


def test_fnirt_is_offered_for_writing() -> None:
    assert FnirtWarpField in WritableFileBasedTransformation._REGISTRY


def test_a_fnirt_warp_is_written(tmp_path) -> None:  # noqa: ANN001
    image = nb.Nifti1Image(np.zeros((*SHAPE, 3), dtype="float32"), VOX2WORLD)
    image.header.set_intent(2006)
    warp = FnirtWarpField.from_nibabel(image)
    path = tmp_path / "warp.nii.gz"
    warp.save(path)
    written = nb.load(path)
    assert int(written.header["intent_code"]) == 2006
    np.testing.assert_array_equal(written.get_fdata(), image.get_fdata())
    # It is written as a NIfTI displacement field instead.
    warp.moving = image
    NiftiRASDisplacementField.from_any(warp).save(path)
    back = io.transformations.load(path)
    assert isinstance(back, NiftiRASDisplacementField)
    points = _world()
    np.testing.assert_allclose(
        _apply(back, points), _apply(warp, points), atol=1e-6
    )


# ----------------------------------------------------------------------
#   OPTIONS, AND FIELDS WITHOUT DATA
# ----------------------------------------------------------------------


def test_a_conversion_does_not_relabel_the_endpoints() -> None:
    # `input=LPSmm()` would label the unflipped matrix as LPS.
    affine = xforms.Affine(VOX2WORLD[:-1])
    with pytest.raises(ConversionError, match="input= cannot be overridden"):
        NiftiVoxelToRAS.from_any(affine, input=LPSmm())
    with pytest.raises(ConversionError, match="input= cannot be overridden"):
        affine.to(NiftiVoxelToRAS, input=LPSmm())


def test_a_conversion_does_not_replace_the_map() -> None:
    affine = xforms.Affine(VOX2WORLD[:-1])
    with pytest.raises(ConversionError, match="matrix= cannot be overridden"):
        NiftiVoxelToRAS.from_any(affine, matrix=np.eye(4)[:-1])
    chain = _ras_displacement_chain()
    with pytest.raises(ConversionError, match="transformations= cannot"):
        chain.to(NiftiRASDisplacementField, transformations=())


def test_the_format_options_are_passed_on() -> None:
    header = nb.Nifti1Header()
    header.set_sform(np.eye(4), code=4)
    affine = xforms.Affine(VOX2WORLD[:-1])
    nifti = NiftiVoxelToRAS.from_any(affine, header=header)
    np.testing.assert_array_equal(nifti.matrix, VOX2WORLD[:-1])
    assert nifti.to_nibabel().header.get_sform(coded=True)[1] == 4


def test_log_is_refused_for_a_field_that_holds_a_displacement() -> None:
    field = xforms.DisplacementField(field=_displacements())
    with pytest.raises(ConversionError, match="log=True writes the velocity"):
        NiftiRASDisplacementField.from_any(field, log=True)
    with pytest.raises(ConversionError, match="steps= is the number"):
        NiftiRASDisplacementField.from_any(field, steps=4)


def test_log_writes_a_velocity_as_it_is(tmp_path) -> None:  # noqa: ANN001
    velocity = xforms.DisplacementField(data=0.1 * _displacements(), log=True)
    nifti = NiftiRASDisplacementField.from_any(velocity, log=True)
    assert nifti.log
    nifti.save(tmp_path / "velocity.nii.gz")
    written = nb.load(tmp_path / "velocity.nii.gz").get_fdata()[:, :, :, 0]
    np.testing.assert_allclose(written, 0.1 * _displacements(), atol=1e-12)


@pytest.mark.parametrize(
    "cls", [NiftiRASCoordinatesField, SpmCoordinatesField]
)
def test_a_field_of_coordinates_without_data_is_held_without_data(
    cls: type,
) -> None:
    # As a displacement field without data is: both are the identity.
    converted = xforms.CoordinatesField().to(cls)
    assert type(converted) is cls
    empty = xforms.DisplacementField().to(NiftiRASDisplacementField)
    assert empty.displacement.data is None


def test_no_coordinates_cannot_carry_an_affine() -> None:
    scale = xforms.Affine(
        np.diag([2.0, 2.0, 2.0, 1.0])[:-1], input=RASmm(), output=RASmm()
    )
    chain = xforms.Sequence(
        transformations=[
            xforms.CoordinatesField(input=VOXEL, output=RASmm()),
            scale,
        ]
    )
    with pytest.raises(ConversionError, match="holds no coordinates"):
        chain.to(NiftiRASCoordinatesField)


def test_a_nearly_singular_grid_is_refused_by_spm() -> None:
    chain = xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=np.diag([1.0, 1.0, 1e-14, 1.0])[:-1]),
            xforms.CoordinatesField(
                field=_coordinates(), input=VOXEL, output=RASmm()
            ),
        ]
    )
    with pytest.raises(ConversionError, match="nearly so"):
        chain.to(SpmCoordinatesField)


@pytest.mark.parametrize(
    "cls",
    [
        NiftiVoxelToRAS,
        NiftiRASToVoxel,
        NiftiRASDisplacementField,
        NiftiRASCoordinatesField,
        SpmCoordinatesField,
    ],
)
def test_the_converted_formats_still_read_files(tmp_path, cls: type) -> None:  # noqa: ANN001
    path = tmp_path / "y_field.nii"
    image = nb.Nifti1Image(_coordinates()[:, :, :, None, :], VOX2WORLD)
    image.header.set_intent(1007, name="Mapping")
    nb.save(image, str(path))
    assert type(cls.from_any(path)) is cls


# ----------------------------------------------------------------------
#   BACK INTO THE DATA MODEL
# ----------------------------------------------------------------------
# The NIfTI formats derive their map from a header, and keep the image of
# the parser under `_data`, so a conversion must copy the map rather than
# what is stored there (#342).


def _header() -> tx.Any:
    header = nb.Nifti1Header()
    header.set_data_shape(SHAPE)
    header.set_sform(VOX2WORLD, code=1)
    return header


# Each NIfTI affine, the class of the data model it belongs to, and its map.
NIFTI_AFFINES = [
    (NiftiVoxelToRAS, VoxelToRAS, VOX2WORLD[:-1]),
    (NiftiRASToVoxel, RASToVoxel, np.linalg.inv(VOX2WORLD)[:-1]),
]


@pytest.mark.parametrize("nifti, base, matrix", NIFTI_AFFINES)
def test_a_nifti_affine_read_from_a_header_converts_back(
    nifti: type, base: type, matrix: np.ndarray
) -> None:
    xform = nifti(header=_header())
    for cls in (xforms.Affine, base):
        affine = xform.to(cls)
        assert type(affine) is cls
        np.testing.assert_allclose(affine.matrix, matrix)


@pytest.mark.parametrize("nifti, base, matrix", NIFTI_AFFINES)
def test_a_nifti_affine_with_a_set_matrix_converts_back(
    nifti: type, base: type, matrix: np.ndarray
) -> None:
    xform = nifti(matrix=matrix)
    for cls in (xforms.Affine, base):
        affine = xform.to(cls)
        assert type(affine) is cls
        np.testing.assert_array_equal(affine.matrix, matrix)


def test_a_loaded_nifti_affine_converts_back_into_an_affine(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.nii.gz"
    nb.save(nb.Nifti1Image(np.zeros(SHAPE, "float32"), VOX2WORLD), str(path))
    nifti = NiftiVoxelToRAS.load(path)
    np.testing.assert_array_equal(
        nifti.to(xforms.Affine).matrix, VOX2WORLD[:-1]
    )
    np.testing.assert_array_equal(
        nifti.inverse().to(xforms.Affine).matrix,
        np.linalg.inv(VOX2WORLD)[:-1],
    )
    # Back into the format, the affine is written and read unchanged.
    io.save(nifti.to(xforms.Affine).to(NiftiVoxelToRAS), tmp_path / "x.nii")
    back = io.transformations.load(tmp_path / "x.nii")
    np.testing.assert_array_equal(back.matrix, VOX2WORLD[:-1])


def test_a_loaded_nifti_field_of_coordinates_converts_back(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "y_field.nii"
    image = nb.Nifti1Image(_coordinates()[:, :, :, None, :], VOX2WORLD)
    image.header.set_intent(1007, name="Mapping")
    nb.save(image, str(path))
    nifti = NiftiRASCoordinatesField.load(path)
    field = nifti.to(xforms.CoordinatesField)
    assert type(field) is xforms.CoordinatesField
    np.testing.assert_allclose(field.field, nifti.field)
