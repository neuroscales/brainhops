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

from brainhops.io.base.parsers import FileWriter

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    LPSmm,
    RASmm,
    VoxelCoordinateSystem,
)
from brainhops.errors import (  # noqa: E402
    ConversionError,
    LossyConversionError,
)
from brainhops.io.base.parsers import (  # noqa: E402
    WriterError,
)
from brainhops.io.transformations.base import (  # noqa: E402
    TransformationFormat,
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
    NiftiMetadata,
    NiftiRASCoordinateField,
    NiftiRASDisplacementField,
    NiftiRASToVoxel,
    NiftiRaw,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.spm import (  # noqa: E402
    SpmCoordinateField,
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
    points = xforms.CoordinateField(field=points[:, None, None, :])
    out = xforms.Sequence(transformations=[points, xform]).compute()
    field = out.to(xforms.CoordinateField).field
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
            xforms.CoordinateField(
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
    assert affine.to(NiftiVoxelToRAS, on_error=None) is None


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
    io.save(nifti, tmp_path / "warp.nii.gz")
    back = io.transformations.load(tmp_path / "warp.nii.gz")
    assert isinstance(back, NiftiRASDisplacementField)
    np.testing.assert_allclose(_apply(back, _flip(lps)), expected, atol=1e-10)


def test_an_lps_displacement_chain_is_saved_as_an_itk_field(tmp_path) -> None:  # noqa: ANN001
    # `io.save` keeps the LPS endpoints of the chain in an ITK field, which
    # is read back with its hint, since its header cannot say ITK.
    from brainhops.io.transformations.itk.nifti import (
        ItkNiftiDisplacementField,
    )

    chain = _lps_displacement_chain()
    lps = _flip(_world())
    io.save(chain, tmp_path / "warp.nii.gz")
    back = io.transformations.load(tmp_path / "warp.nii.gz", hint="itk")
    assert isinstance(back, ItkNiftiDisplacementField)
    np.testing.assert_allclose(
        _apply(back, lps), _apply(chain, lps), atol=1e-10
    )


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


def test_a_cubic_displacement_field_is_lossy() -> None:
    # NIfTI's values are read back with linear interpolation, which is
    # another map between the nodes than a cubic spline's.
    chain = _lps_displacement_chain(degree=3)
    with pytest.raises(LossyConversionError, match="degree 3"):
        chain.to(NiftiRASDisplacementField)
    nifti = chain.to(NiftiRASDisplacementField, lossy=True)
    # The values at the nodes are kept.
    lps = LPS2RAS @ VOX2WORLD
    nodes = _voxels()[:3] @ lps[:3, :3].T + lps[:3, 3]
    np.testing.assert_allclose(
        _flip(_apply(nifti, _flip(nodes))), _apply(chain, nodes), atol=1e-10
    )


def test_ends_that_do_not_undo_each_other_are_carried_into_the_field() -> None:
    chain = list(_ras_displacement_chain())
    chain[2] = VoxelToRAS(matrix=(2 * VOX2WORLD)[:-1])
    chain = xforms.Sequence(transformations=chain)
    nifti = chain.to(NiftiRASDisplacementField)
    # The last two points lie outside the grid, where the two fields are
    # extended differently.
    points = _world()[:-2]
    np.testing.assert_allclose(
        _apply(nifti, points), _apply(chain, points), atol=1e-10
    )


def test_two_fields_are_not_composed_into_one() -> None:
    field = xforms.DisplacementField(field=_displacements())
    chain = xforms.Sequence(transformations=[field, field])
    with pytest.raises(ConversionError, match="holds 2 fields"):
        chain.to(NiftiRASDisplacementField)


def test_coordinates_and_displacements_convert_into_each_other() -> None:
    # They are the same map inside their grid, and only extend differently
    # outside it, which is not a loss.
    coordinates = xforms.CoordinateField(
        field=_coordinates(), input=VOXEL, output=RASmm()
    )
    chain = xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=np.linalg.inv(VOX2WORLD)[:-1]),
            coordinates,
        ]
    )
    displacement = chain.to(NiftiRASDisplacementField)
    points = _world()[:-2]
    np.testing.assert_allclose(
        _apply(displacement, points), _apply(chain, points), atol=1e-10
    )
    nifti = _ras_displacement_chain().to(SpmCoordinateField)
    np.testing.assert_allclose(
        _apply(nifti, points),
        _apply(_ras_displacement_chain(), points),
        atol=1e-10,
    )


# ----------------------------------------------------------------------
#   FIELDS OF COORDINATES
# ----------------------------------------------------------------------


def test_a_field_of_lps_coordinates_is_bridged_into_ras(tmp_path) -> None:  # noqa: ANN001
    field = xforms.CoordinateField(
        field=_flip(_coordinates()), input=VOXEL, output=LPSmm()
    )
    points = _voxels()
    expected = _flip(_apply(field, points))
    for nifti in _conversions(field, NiftiRASCoordinateField):
        assert type(nifti) is NiftiRASCoordinateField
        np.testing.assert_array_equal(np.asarray(nifti.field), _coordinates())
        np.testing.assert_allclose(_apply(nifti, points), expected, atol=1e-12)
    io.save(field, tmp_path / "coordinates.nii.gz")
    back = io.transformations.load(tmp_path / "coordinates.nii.gz")
    assert isinstance(back, NiftiRASCoordinateField)
    np.testing.assert_allclose(_apply(back, points), expected, atol=1e-10)


def test_an_affine_after_the_coordinates_is_applied_to_them() -> None:
    field = xforms.CoordinateField(
        field=_coordinates(), input=VOXEL, output=RASmm()
    )
    scale = xforms.Affine(
        np.diag([2.0, 3.0, 4.0, 1.0])[:-1], input=RASmm(), output=RASmm()
    )
    chain = xforms.Sequence(transformations=[field, scale])
    nifti = chain.to(NiftiRASCoordinateField)
    points = _voxels()
    np.testing.assert_allclose(
        _apply(nifti, points), _apply(chain, points), atol=1e-12
    )


def test_an_affine_before_the_coordinates_is_refused() -> None:
    # It would resample the field onto other voxels.
    shift = xforms.Affine(
        np.eye(4)[:-1] + [[0, 0, 0, 1]] * 3, input=VOXEL, output=VOXEL
    )
    field = xforms.CoordinateField(field=_coordinates(), input=VOXEL)
    chain = xforms.Sequence(transformations=[shift, field])
    with pytest.raises(ConversionError, match="voxels of the file"):
        chain.to(NiftiRASCoordinateField)


# ----------------------------------------------------------------------
#   SPM DEFORMATIONS
# ----------------------------------------------------------------------


def test_an_lps_deformation_is_written_as_spm(tmp_path) -> None:  # noqa: ANN001
    chain = _lps_spm_chain()
    lps = _flip(_world())
    expected = _flip(_apply(chain, lps))
    for spm in _conversions(chain, SpmCoordinateField):
        assert type(spm) is SpmCoordinateField
        np.testing.assert_allclose(
            _apply(spm, _flip(lps)), expected, atol=1e-10
        )
    io.save(chain, tmp_path / "y_deformation.nii")
    back = io.transformations.load(tmp_path / "y_deformation.nii")
    assert isinstance(back, SpmCoordinateField)
    np.testing.assert_allclose(_apply(back, _flip(lps)), expected, atol=1e-10)
    np.testing.assert_array_equal(
        back.metadata.raw.header.get_best_affine(), VOX2WORLD
    )


def test_an_spm_file_round_trips(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "y_source.nii"
    image = nb.Nifti1Image(_coordinates()[:, :, :, None, :], VOX2WORLD)
    image.header.set_intent(1007, name="Mapping")
    nb.save(image, str(path))
    spm = io.transformations.load(path)
    assert isinstance(spm, SpmCoordinateField)
    io.save(spm, tmp_path / "y_copy.nii")
    back = io.transformations.load(tmp_path / "y_copy.nii")
    assert isinstance(back, SpmCoordinateField)
    np.testing.assert_array_equal(
        back.metadata.raw.header.get_best_affine(), VOX2WORLD
    )
    np.testing.assert_array_equal(
        np.asarray(back.rasfield.field), _coordinates()
    )
    assert back.metadata.raw.header.get_intent() == ("vector", (), "Mapping")


def test_a_map_from_ras_is_not_a_nifti_field_of_coordinates() -> None:
    # A NIfTI field of coordinates maps the voxels of its grid.
    with pytest.raises(ConversionError, match="SPM deformation"):
        _ras_displacement_chain().to(NiftiRASCoordinateField)


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
            xforms.CoordinateField(field=_coordinates()),
            "coordinates.nii.gz",
            NiftiRASCoordinateField,
        ),
        (_ras_displacement_chain(), "warp.nii.gz", NiftiRASDisplacementField),
        (_lps_spm_chain(), "y_deformation.nii.gz", SpmCoordinateField),
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
        # A quadratic field, which a NiftyReg grid of control points,
        # cubic or linear, does not hold exactly either.
        io.save(_lps_displacement_chain(degree=2), tmp_path / "warp.nii.gz")
    message = str(info.value)
    assert "NiftiRASDisplacementField: LossyConversionError" in message
    assert "degree 2" in message


# ----------------------------------------------------------------------
#   FNIRT WRITING
# ----------------------------------------------------------------------


def test_fnirt_is_offered_for_writing() -> None:
    assert issubclass(FnirtWarpField, FileWriter)
    assert FnirtWarpField in TransformationFormat._REGISTRY


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
    header["descrip"] = b"kept"
    metadata = NiftiMetadata.from_raw(NiftiRaw(header=header))
    affine = xforms.Affine(VOX2WORLD[:-1])
    nifti = NiftiVoxelToRAS.from_any(affine, metadata=metadata)
    np.testing.assert_array_equal(nifti.matrix, VOX2WORLD[:-1])
    written = nifti.to_nibabel().header
    assert written.get_sform(coded=True)[1] == 4
    assert written["descrip"].item() == b"kept"


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


@pytest.mark.parametrize("cls", [NiftiRASCoordinateField, SpmCoordinateField])
def test_a_field_of_coordinates_without_data_is_held_without_data(
    cls: type,
) -> None:
    # As a displacement field without data is: both are the identity.
    converted = xforms.CoordinateField().to(cls)
    assert type(converted) is cls
    empty = xforms.DisplacementField().to(NiftiRASDisplacementField)
    assert empty.displacement.data is None


def test_no_coordinates_cannot_carry_an_affine() -> None:
    scale = xforms.Affine(
        np.diag([2.0, 2.0, 2.0, 1.0])[:-1], input=RASmm(), output=RASmm()
    )
    chain = xforms.Sequence(
        transformations=[
            xforms.CoordinateField(input=VOXEL, output=RASmm()),
            scale,
        ]
    )
    with pytest.raises(ConversionError, match="holds no coordinates"):
        chain.to(NiftiRASCoordinateField)


def test_a_nearly_singular_grid_is_refused_by_spm() -> None:
    chain = xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=np.diag([1.0, 1.0, 1e-14, 1.0])[:-1]),
            xforms.CoordinateField(
                field=_coordinates(), input=VOXEL, output=RASmm()
            ),
        ]
    )
    with pytest.raises(ConversionError, match="nearly so"):
        chain.to(SpmCoordinateField)


@pytest.mark.parametrize(
    "cls",
    [
        NiftiVoxelToRAS,
        NiftiRASToVoxel,
        NiftiRASDisplacementField,
        NiftiRASCoordinateField,
        SpmCoordinateField,
    ],
)
def test_the_converted_formats_still_read_files(tmp_path, cls: type) -> None:  # noqa: ANN001
    path = tmp_path / "y_field.nii"
    image = nb.Nifti1Image(_coordinates()[:, :, :, None, :], VOX2WORLD)
    image.header.set_intent(1007, name="Mapping")
    nb.save(image, str(path))
    assert type(cls.from_any(path)) is cls
