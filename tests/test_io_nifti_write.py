"""Tests for writing images and transformations to NIfTI.

A file that was read can be written again without drift in the data, the
forms or their codes. A transformation that NIfTI cannot represent is
refused instead of being resampled silently.
"""

import warnings

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from bagof.magic import replace  # noqa: E402

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    CoordinateSystem,
    LPSCoordinateSystem,
    RASCoordinateSystem,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    DisplacementField,
    Scaling,
    Sequence,
)
from brainhops.datamodel.units import is_indexunit  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiVoxelToRAS,
)

# A rotation, anisotropic zoom and translation without shear, so that the
# qform is exact.
AFFINE = np.array(
    [
        [0.0, -1.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 2.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _write_image(tmp_path, name, data, scode=2, qcode=1):  # noqa: ANN001, ANN202
    img = nb.Nifti1Image(data, AFFINE)
    img.header.set_sform(AFFINE, code=scode)
    img.header.set_qform(AFFINE, code=qcode)
    target = tmp_path / name
    nb.save(img, str(target))
    return target


# ----------------------------------------------------------------------
#   IMAGES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_an_image_round_trips_through_save(tmp_path, suffix) -> None:  # noqa: ANN001
    data = np.arange(4 * 5 * 6, dtype="float32").reshape(4, 5, 6)
    source = _write_image(tmp_path, "source.nii", data)

    loaded = io.images.load(source)
    assert isinstance(loaded, NiftiImage)

    target = tmp_path / ("out" + suffix)
    loaded.save(target)
    assert target.exists()

    reloaded = io.images.load(target)
    assert np.array_equal(np.asarray(reloaded), data)

    before, after = nb.load(str(source)).header, nb.load(str(target)).header
    assert np.allclose(before.get_sform(), after.get_sform())
    assert np.allclose(before.get_qform(), after.get_qform())
    assert int(before["sform_code"]) == int(after["sform_code"])
    assert int(before["qform_code"]) == int(after["qform_code"])
    assert np.allclose(
        nb.load(str(source)).affine, nb.load(str(target)).affine
    )


def test_the_sform_code_follows_the_world_space(tmp_path) -> None:  # noqa: ANN001
    """A scanner-coded sform is written back with the scanner code."""
    data = np.zeros((3, 4, 5), dtype="float32")
    source = _write_image(tmp_path, "scanner.nii", data, scode=1, qcode=1)

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    assert int(nb.load(str(target)).header["sform_code"]) == 1


def test_a_qform_is_written_even_without_a_rigid_edge(tmp_path) -> None:  # noqa: ANN001
    """With only an sform, the rigid part is still written as the qform."""
    data = np.zeros((3, 4, 5), dtype="float32")
    img = nb.Nifti1Image(data, AFFINE)
    img.header.set_sform(AFFINE, code=2)
    img.header.set_qform(None, code=0)
    source = tmp_path / "sform_only.nii"
    nb.save(img, str(source))

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    header = nb.load(str(target)).header
    assert int(header["qform_code"]) != 0
    assert np.allclose(header.get_qform(), AFFINE, atol=1e-5)


def test_a_scaling_geometry_is_representable(tmp_path) -> None:  # noqa: ANN001
    """A non-affine transformation that reduces to an affine is written."""
    image = NiftiImage(
        data=np.zeros((4, 5, 6), dtype="float32"),
        transformations=[Scaling(scale=[2.0, 3.0, 4.0])],
    )
    target = tmp_path / "scaled.nii"
    image.save(target)

    assert np.allclose(np.diag(nb.load(str(target)).affine), [2, 3, 4, 1])


def test_a_sequence_of_affines_writes_the_composed_sform(tmp_path) -> None:  # noqa: ANN001
    first = Affine(
        matrix=np.array([[2.0, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3]])
    )
    second = Affine(
        matrix=np.array([[0.0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[Sequence([first, second])],
    )
    target = tmp_path / "seq.nii"
    image.save(target)

    first_h = np.vstack([first.matrix, [0, 0, 0, 1]])
    second_h = np.vstack([second.matrix, [0, 0, 0, 1]])
    assert np.allclose(nb.load(str(target)).affine, second_h @ first_h)


def test_the_qform_matrix_and_code_come_from_one_edge(tmp_path) -> None:  # noqa: ANN001
    """The preferred mni edge gives the sform, the scanner edge the qform."""
    ras, voxel = RASCoordinateSystem(), VoxelCoordinateSystem()
    scanner = Affine(
        matrix=np.eye(3, 4),
        input=voxel,
        output=replace(ras, name="scanner"),
    )
    mni = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:3],
        input=voxel,
        output=replace(ras, name="mni"),
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[scanner, mni],
    )
    target = tmp_path / "coded.nii"
    image.save(target)

    header = nb.load(str(target)).header
    assert int(header["sform_code"]) == 4  # mni
    assert int(header["qform_code"]) == 1  # scanner
    assert np.allclose(header.get_qform(), np.eye(4))


def test_a_2d_nifti_round_trips(tmp_path) -> None:  # noqa: ANN001
    """A 2-D image has a (3, 3) affine, embedded in the NIfTI (4, 4) one."""
    data = np.arange(4 * 5, dtype="float32").reshape(4, 5)
    img = nb.Nifti1Image(data, np.diag([2.0, 3.0, 1.0, 1.0]))
    source = tmp_path / "plane.nii"
    nb.save(img, str(source))

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    reloaded = io.images.load(target)
    assert np.array_equal(np.asarray(reloaded), data)
    assert np.allclose(
        nb.load(str(source)).affine, nb.load(str(target)).affine
    )


def test_the_units_survive_the_round_trip(tmp_path) -> None:  # noqa: ANN001
    img = nb.Nifti1Image(np.zeros((4, 5, 6, 2), dtype="float32"), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    source = tmp_path / "units.nii"
    nb.save(img, str(source))

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    assert nb.load(str(target)).header.get_xyzt_units() == ("mm", "sec")
    reloaded = io.images.load(target).transformation.output
    assert [str(axis.unit) for axis in reloaded.axes] == [
        "millimeter",
        "millimeter",
        "millimeter",
        "second",
    ]


def test_int64_data_is_reported_as_a_writer_error(tmp_path) -> None:  # noqa: ANN001
    """NIfTI-1 cannot store int64."""
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="int64"),
        transformations=[Affine(matrix=np.eye(3, 4))],
    )
    with pytest.raises(WriterError):
        image.save(tmp_path / "big.nii")


def test_an_image_without_data_cannot_be_written(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(WriterError):
        NiftiImage().save(tmp_path / "empty.nii")


def test_an_image_built_from_data_alone_is_written(tmp_path) -> None:  # noqa: ANN001
    """An image built from data alone is written with an identity geometry."""
    data = np.arange(24, dtype="float32").reshape(2, 3, 4)
    image = NiftiImage(data=data)
    assert image.transformations == []

    target = tmp_path / "bare.nii"
    image.save(target)

    written = nb.load(str(target))
    assert np.array_equal(np.asarray(written.dataobj), data)
    assert np.allclose(written.affine, np.eye(4))


# ----------------------------------------------------------------------
#   TRANSFORMATIONS
# ----------------------------------------------------------------------


def test_an_affine_round_trips_through_save(tmp_path) -> None:  # noqa: ANN001
    data = np.zeros((4, 5, 6), dtype="float32")
    source = _write_image(tmp_path, "source.nii", data)

    affine = NiftiVoxelToRAS.from_file(source)
    target = tmp_path / "affine.nii"
    affine.save(target)

    reloaded = NiftiVoxelToRAS.from_file(target)
    assert np.allclose(affine.matrix, reloaded.matrix)
    # The affine is written over a minimal placeholder volume.
    assert nb.load(str(target)).shape == (1, 1, 1)


def test_a_field_round_trips_with_its_intent_code(tmp_path) -> None:  # noqa: ANN001
    """A coordinates field is written as VECTOR and reads back as such."""
    field = np.zeros((4, 5, 6, 1, 3), dtype="float32")
    field[..., 0] = 1.0
    img = nb.Nifti1Image(field, np.eye(4))
    img.header.set_intent(1007, name="Mapping")
    source = tmp_path / "field.nii"
    nb.save(img, str(source))

    loaded = io.transformations.load(source)
    assert isinstance(loaded, NiftiRASCoordinatesField)

    target = tmp_path / "out.nii"
    loaded.save(target)

    reloaded = io.transformations.load(target)
    assert isinstance(reloaded, NiftiRASCoordinatesField)
    assert np.array_equal(np.asarray(reloaded.field), field[:, :, :, 0])
    header = nb.load(str(target)).header
    assert int(header["intent_code"]) == 1007
    assert header.get_intent()[2] == "Mapping"


def test_a_4d_field_is_written_5d(tmp_path) -> None:  # noqa: ANN001
    """(X, Y, Z, C) is written as (X, Y, Z, 1, C), so C is not time."""
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 1.0
    source = NiftiRASCoordinatesField(field=field)

    target = tmp_path / "field.nii"
    source.save(target)

    assert nb.load(str(target)).shape == (4, 5, 6, 1, 3)
    assert isinstance(
        io.transformations.load(target), NiftiRASCoordinatesField
    )


# ----------------------------------------------------------------------
#   UNREPRESENTABLE TRANSFORMATIONS
# ----------------------------------------------------------------------


def test_a_nonaffine_geometry_cannot_be_written(tmp_path) -> None:  # noqa: ANN001
    """An image whose preferred transformation is a field is refused."""
    image = NiftiImage(
        data=np.zeros((4, 5, 6), dtype="float32"),
        transformations=[
            DisplacementField(field=np.zeros((4, 5, 6, 3), dtype="float32"))
        ],
    )
    with pytest.raises(UnrepresentableTransformationError) as info:
        image.save(tmp_path / "bad.nii")
    assert "DisplacementField" in str(info.value)


def test_the_writer_is_registered_for_the_image_kind() -> None:
    from brainhops.io.images.base import WritableFileBasedImage

    assert NiftiImage in WritableFileBasedImage._REGISTRY
    assert issubclass(NiftiImage, SingleScaleImage)


# ----------------------------------------------------------------------
#   REVIEW FOLLOW-UPS (#44)
# ----------------------------------------------------------------------


def test_a_dask_field_is_not_coerced_to_numpy() -> None:
    """A dask field stays lazy instead of being converted to numpy."""
    da = pytest.importorskip("dask.array")
    field = da.zeros((4, 5, 6, 3), dtype="float32")

    image = NiftiRASCoordinatesField(field=field)
    nifti = image.to_nibabel()

    assert not isinstance(nifti.dataobj, np.ndarray)
    assert nifti.shape == (4, 5, 6, 1, 3)


def test_a_non_nifti_unit_is_scaled_to_a_valid_one(tmp_path) -> None:  # noqa: ANN001
    """A cm affine is scaled to mm, preserving the physical size."""
    ras_cm = replace(
        RASCoordinateSystem(),
        axes=[
            replace(axis, unit="centimeter")
            for axis in RASCoordinateSystem().axes
        ],
    )
    affine = Affine(
        matrix=np.eye(3, 4),
        input=VoxelCoordinateSystem(),
        output=ras_cm,
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[affine],
    )
    target = tmp_path / "cm.nii"
    image.save(target)

    header = nb.load(str(target)).header
    assert header.get_xyzt_units()[0] == "mm"
    assert np.allclose(np.diag(nb.load(str(target)).affine), [10, 10, 10, 1])


def test_the_sform_code_is_derived_from_orientation_not_name(tmp_path) -> None:  # noqa: ANN001
    """An anonymous RAS space still gets a non-zero sform code."""
    ras = CoordinateSystem(name=None, axes=list(RASCoordinateSystem().axes))
    affine = Affine(
        matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3],
        input=VoxelCoordinateSystem(),
        output=ras,
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[affine],
    )
    target = tmp_path / "anon.nii"
    image.save(target)

    header = nb.load(str(target)).header
    assert int(header["sform_code"]) != 0
    assert np.allclose(np.diag(nb.load(str(target)).affine), [2, 3, 4, 1])


def test_the_ras_flip_follows_orientation_not_the_name(tmp_path) -> None:  # noqa: ANN001
    """An anonymous LPS space is flipped to RAS from its orientations."""
    lps = CoordinateSystem(name=None, axes=list(LPSCoordinateSystem().axes))
    affine = Affine(
        matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3],
        input=VoxelCoordinateSystem(),
        output=lps,
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[affine],
    )
    target = tmp_path / "lps.nii"
    image.save(target)

    affine_out = nb.load(str(target)).affine
    assert np.allclose(np.diag(affine_out), [-2, -3, 4, 1])


def test_a_space_and_time_coupled_geometry_is_rejected(tmp_path) -> None:  # noqa: ANN001
    """NIfTI stores space and time apart, so coupling them is refused."""
    for row, column in ((0, 3), (3, 0)):
        matrix = np.eye(4, 5)
        matrix[row, column] = 0.5
        image = NiftiImage(
            data=np.zeros((2, 2, 2, 2), dtype="float32"),
            transformations=[Affine(matrix=matrix)],
        )
        with pytest.raises(UnrepresentableTransformationError) as info:
            image.save(tmp_path / "four.nii")
        assert "mixes the spatial axes" in str(info.value)


def test_a_time_axis_that_is_not_a_spacing_is_rejected(tmp_path) -> None:  # noqa: ANN001
    # A reversed time axis has no positive spacing.
    image = NiftiImage(
        data=np.zeros((2, 2, 2, 2), dtype="float32"),
        transformations=[Affine(matrix=np.diag([1.0, 1, 1, -2, 1])[:4])],
    )
    with pytest.raises(UnrepresentableTransformationError):
        image.save(tmp_path / "four.nii")


def test_a_space_and_time_geometry_survives_the_round_trip(tmp_path) -> None:  # noqa: ANN001
    """Forms, codes, spacings, TR, time offset and units survive."""
    sform = np.array(
        [[0, -2.0, 0, 10], [1.5, 0, 0, -3], [0, 0, 2.5, 4], [0, 0, 0, 1]]
    )
    nii = nb.Nifti1Image(np.random.rand(4, 5, 6, 4).astype("f4"), sform)
    nii.header.set_xyzt_units("mm", "msec")
    nii.header.set_zooms((2.0, 1.5, 2.5, 750.0))
    nii.header["toffset"] = 125.0
    nii.header.set_qform(np.diag([1.5, 2.0, 2.5, 1.0]), code=1)
    nii.header.set_sform(sform, code=4)
    source = tmp_path / "bold.nii"
    nb.save(nii, str(source))

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    before, after = nb.load(str(source)).header, nb.load(str(target)).header
    assert np.allclose(before.get_sform(), after.get_sform())
    assert np.allclose(before.get_qform(), after.get_qform())
    assert int(after["sform_code"]) == 4
    assert int(after["qform_code"]) == 1
    assert np.allclose(before.get_zooms(), after.get_zooms())
    assert float(after["toffset"]) == 125.0
    assert after.get_xyzt_units() == ("mm", "msec")


def test_a_space_and_time_geometry_is_written_to_mrtrix(tmp_path) -> None:  # noqa: ANN001
    # Another format that stores a spatial affine keeps the spatial block.
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, -1] = [1.0, 2.0, 3.0]
    nii = nb.Nifti1Image(np.zeros((4, 5, 6, 2), dtype="f4"), affine)
    nb.save(nii, str(tmp_path / "bold.nii"))

    img = io.images.load(tmp_path / "bold.nii")
    io.save(SingleScaleImage.from_instance(img), tmp_path / "bold.mif")

    matrix = io.images.load(tmp_path / "bold.mif").transformation.to(Affine)
    assert np.allclose(
        np.asarray(matrix.matrix)[:3, [0, 1, 2, -1]], affine[:3]
    )


def _xyzt_system(order, unit_space, unit_time):  # noqa: ANN001, ANN202
    from brainhops.datamodel.axes import SpaceAxis, TimeAxis

    axes = {n: SpaceAxis(name=n, unit=unit_space) for n in "xyz"}
    axes["t"] = TimeAxis(name="t", unit=unit_time)
    return CoordinateSystem(axes=[axes[n] for n in order])


# The x, y, z, t vox-to-world matrix holds a spatial affine, a TR of 2 s
# and a time offset of 0.5 s.
XYZT = np.array(
    [
        [0.0, -2.0, 0.0, 0.0, 10.0],
        [1.5, 0.0, 0.0, 0.0, -3.0],
        [0.0, 0.0, 2.5, 0.0, 4.0],
        [0.0, 0.0, 0.0, 2.0, 0.5],
    ]
)


def _reordered(voxel_order, world_order):  # noqa: ANN001, ANN202
    # Return XYZT with its voxel columns and world rows in the given orders.
    columns = ["xyzt".index(n) for n in voxel_order] + [4]
    rows = ["xyzt".index(n) for n in world_order]
    return XYZT[rows][:, columns]


@pytest.mark.parametrize("backend", ["numpy", "dask"])
@pytest.mark.parametrize(
    "voxel_order, world_order",
    [("txyz", "xyzt"), ("xtyz", "xyzt"), ("txyz", "txyz"), ("xyzt", "tzyx")],
)
def test_the_axes_are_written_in_nifti_order(  # noqa: D103
    tmp_path,  # noqa: ANN001
    backend: str,
    voxel_order: str,
    world_order: str,
) -> None:
    # Axes are placed by their declared types, and the data is transposed.
    shape = dict(zip("xyzt", (4, 5, 6, 3)))
    values = np.random.rand(*(shape[n] for n in "xyzt")).astype("f4")
    data = values.transpose(["xyzt".index(n) for n in voxel_order])
    if backend == "dask":
        da = pytest.importorskip("dask.array")
        data = da.from_array(data, chunks=2)
    voxel = _xyzt_system(voxel_order, "index", "index")
    world = _xyzt_system(world_order, "mm", "s")
    xform = Affine(
        matrix=_reordered(voxel_order, world_order), input=voxel, output=world
    )
    image = NiftiImage(data=data, transformations=[xform])
    target = tmp_path / "bold.nii"
    image.save(target)

    nii = nb.load(str(target))
    assert np.array_equal(np.asarray(nii.dataobj), values)
    assert np.allclose(nii.header.get_sform()[:3], XYZT[:3, [0, 1, 2, 4]])
    assert np.allclose(nii.header.get_zooms(), (1.5, 2.0, 2.5, 2.0))
    assert float(nii.header["toffset"]) == 0.5
    assert nii.header.get_xyzt_units() == ("mm", "sec")

    reloaded = io.images.load(target)
    assert np.allclose(reloaded.transformation.to(Affine).matrix, XYZT)
    assert [a.name for a in reloaded.transformation.input.axes] == list("xyzt")


def test_a_world_that_declares_no_axes_follows_the_voxel_axes(  # noqa: D103
    tmp_path,  # noqa: ANN001
) -> None:
    # A world space without axes follows the voxel axes.
    data = np.random.rand(3, 4, 5, 6).astype("f4")
    voxel = _xyzt_system("txyz", "index", "index")
    image = NiftiImage(
        data=data,
        transformations=[Scaling(scale=[2.0, 1.5, 2.5, 3.0], input=voxel)],
    )
    image.save(tmp_path / "bold.nii")

    nii = nb.load(str(tmp_path / "bold.nii"))
    assert np.array_equal(np.asarray(nii.dataobj), data.transpose(1, 2, 3, 0))
    assert np.allclose(nii.header.get_zooms(), (1.5, 2.5, 3.0, 2.0))
    assert np.allclose(np.diag(nii.header.get_sform()), (1.5, 2.5, 3.0, 1.0))


def test_a_coupled_space_and_time_is_rejected_by_declared_axes(  # noqa: D103
    tmp_path,  # noqa: ANN001
) -> None:
    # The coupling is found by declared axis type, not by position.
    matrix = _reordered("txyz", "xyzt")
    matrix[0, 0] = 0.5  # world x reads voxel t
    image = NiftiImage(
        data=np.zeros((3, 4, 5, 6), dtype="f4"),
        transformations=[
            Affine(
                matrix=matrix,
                input=_xyzt_system("txyz", "index", "index"),
                output=_xyzt_system("xyzt", "mm", "s"),
            )
        ],
    )
    with pytest.raises(UnrepresentableTransformationError) as info:
        image.save(tmp_path / "bold.nii")
    assert "mixes the spatial axes" in str(info.value)


def test_axes_of_no_type_fill_the_spatial_slots(tmp_path) -> None:  # noqa: ANN001, D103
    # Untyped axes fill the spatial slots, and time is moved fourth.
    from brainhops.datamodel.axes import Axis, TimeAxis

    data = np.random.rand(3, 4, 5, 6).astype("f4")
    voxel = CoordinateSystem(
        axes=[TimeAxis(name="t", unit="index")]
        + [Axis(name=n, unit="index") for n in "ijk"]
    )
    image = NiftiImage(
        data=data,
        transformations=[Scaling(scale=[2.0, 1.5, 2.5, 3.0], input=voxel)],
    )
    image.save(tmp_path / "bold.nii")

    nii = nb.load(str(tmp_path / "bold.nii"))
    assert np.array_equal(np.asarray(nii.dataobj), data.transpose(1, 2, 3, 0))
    assert np.allclose(nii.header.get_zooms(), (1.5, 2.5, 3.0, 2.0))


@pytest.mark.parametrize(
    "voxel, message",
    [
        (["t", "x", "y", "t"], "one time axis"),
        (["t"], "after 3 spatial axes"),
        (["x", "y", "z", "x"], "at most three spatial axes"),
    ],
)
def test_axes_that_nifti_cannot_place_are_rejected(  # noqa: D103
    tmp_path,  # noqa: ANN001
    voxel: list,
    message: str,
) -> None:
    from brainhops.datamodel.axes import SpaceAxis, TimeAxis

    axes = [
        TimeAxis(name="t", unit="index")
        if n == "t"
        else SpaceAxis(name=n, unit="index")
        for n in voxel
    ]
    image = NiftiImage(
        data=np.zeros((2,) * len(voxel), dtype="f4"),
        transformations=[
            Scaling(
                scale=[1.0] * len(voxel), input=CoordinateSystem(axes=axes)
            )
        ],
    )
    with pytest.raises(UnrepresentableTransformationError) as info:
        image.save(tmp_path / "bad.nii")
    assert message in str(info.value)


def _typed(spec, space="index", time="index"):  # noqa: ANN001, ANN202
    """Typed axes: x, y, z or ijkab are spatial, t is time, c is a channel."""
    from brainhops.datamodel.axes import Axis, SpaceAxis, TimeAxis

    axes = []
    for name in spec:
        if name == "t":
            axes.append(TimeAxis(name="t", unit=time))
        elif name == "c":
            axes.append(Axis(name="c", type="channel", unit="index"))
        else:
            axes.append(SpaceAxis(name=name, unit=space))
    return CoordinateSystem(axes=axes)


def _lazy(values, backend):  # noqa: ANN001, ANN202
    if backend == "dask":
        da = pytest.importorskip("dask.array")
        return da.from_array(values, chunks=2)
    return values


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_a_slice_time_series_is_given_a_z_axis(  # noqa: D103
    tmp_path,  # noqa: ANN001
    backend: str,
) -> None:
    # (x, y, t) is written as (X, Y, 1, T) with an inserted identity z.
    values = np.random.rand(4, 5, 3).astype("f4")
    matrix = np.array([[2.0, 0, 0, 1.0], [0, 3.0, 0, 2.0], [0, 0, 1.5, 0.5]])
    xform = Affine(
        matrix=matrix, input=_typed("xyt"), output=_typed("xyt", "mm", "s")
    )
    image = NiftiImage(data=_lazy(values, backend), transformations=[xform])
    image.save(tmp_path / "slice.nii")

    nii = nb.load(str(tmp_path / "slice.nii"))
    assert nii.shape == (4, 5, 1, 3)
    assert np.array_equal(np.asarray(nii.dataobj)[:, :, 0], values)
    expected = np.diag([2.0, 3.0, 1.0, 1.0])
    expected[:2, 3] = 1.0, 2.0
    assert np.allclose(nii.affine, expected)
    assert np.allclose(nii.header.get_zooms(), (2.0, 3.0, 1.0, 1.5))
    assert float(nii.header["toffset"]) == 0.5

    reloaded = io.images.load(tmp_path / "slice.nii")
    assert [(a.name, a.type) for a in reloaded.system.axes] == [
        ("x", "space"),
        ("y", "space"),
        ("z", "space"),
        ("t", "time"),
    ]
    assert np.array_equal(np.asarray(reloaded.data)[:, :, 0], values)
    expected = np.eye(5)
    expected[[0, 1, 3], :] = 0
    expected[0, [0, 4]] = 2.0, 1.0
    expected[1, [1, 4]] = 3.0, 2.0
    expected[3, [3, 4]] = 1.5, 0.5
    assert np.allclose(
        reloaded.transformation.to(Affine).homogeneous_matrix, expected
    )


@pytest.mark.parametrize("backend", ["numpy", "dask"])
@pytest.mark.parametrize(
    "spec, shape, stored",
    [
        ("xyzc", (4, 5, 6, 3), (4, 5, 6, 1, 3)),
        ("cxyz", (3, 4, 5, 6), (4, 5, 6, 1, 3)),
        ("xyc", (4, 5, 3), (4, 5, 1, 1, 3)),
    ],
)
def test_channels_are_stored_after_a_singleton_time_axis(  # noqa: D103
    tmp_path,  # noqa: ANN001
    backend: str,
    spec: str,
    shape: tuple,
    stored: tuple,
) -> None:
    # Channels go to dim[5] and read back as channels; time has no TR.
    values = np.random.rand(*shape).astype("f4")
    scale = [{"x": 2.0, "y": 3.0, "z": 4.0, "c": 1.0}[n] for n in spec]
    image = NiftiImage(
        data=_lazy(values, backend),
        transformations=[Scaling(scale=scale, input=_typed(spec))],
    )
    image.save(tmp_path / "channels.nii")

    nii = nb.load(str(tmp_path / "channels.nii"))
    assert nii.shape == stored
    natural = values.transpose(
        sorted(range(len(spec)), key=lambda i: "xyzc".index(spec[i]))
    )
    stored_values = np.asarray(nii.dataobj)
    assert np.array_equal(stored_values.reshape(natural.shape), natural)
    assert nii.header.get_zooms()[3] == 0.0
    assert np.allclose(np.diag(nii.affine)[:2], (2.0, 3.0))

    reloaded = io.images.load(tmp_path / "channels.nii")
    axes = reloaded.system.axes
    assert (axes[3].name, axes[3].type) == ("t", "time")
    assert (axes[4].name, axes[4].type) == ("c", "channel")
    assert is_indexunit(axes[4].unit)
    assert np.array_equal(np.asarray(reloaded.data), np.asarray(nii.dataobj))

    # Written again, the file is the same.
    reloaded.save(tmp_path / "again.nii")
    again = nb.load(str(tmp_path / "again.nii"))
    assert again.shape == stored
    assert np.allclose(again.affine, nii.affine)
    assert np.allclose(again.header.get_zooms(), nii.header.get_zooms())
    assert np.array_equal(np.asarray(again.dataobj), np.asarray(nii.dataobj))


def test_a_slice_placed_in_space_completes_its_normal(tmp_path) -> None:  # noqa: ANN001
    # The inserted z axis runs along the plane normal, with unit spacing.
    matrix = np.array(
        [
            [0.0, 2.0, 0.0, 10.0],  # world x <- voxel y
            [0.0, 0.0, 0.0, 20.0],  # world y: slice position
            [3.0, 0.0, 0.0, 30.0],  # world z <- voxel x
            [0.0, 0.0, 1.5, 0.0],  # world t <- voxel t
        ]
    )
    xform = Affine(
        matrix=matrix, input=_typed("xyt"), output=_typed("xyzt", "mm", "s")
    )
    image = NiftiImage(
        data=np.zeros((4, 5, 3), dtype="f4"), transformations=[xform]
    )
    image.save(tmp_path / "coronal.nii")

    nii = nb.load(str(tmp_path / "coronal.nii"))
    assert nii.shape == (4, 5, 1, 3)
    # (0, 0, 3) x (2, 0, 0) = (0, 6, 0), so the normal is +y.
    expected = np.array(
        [
            [0.0, 2.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, 20.0],
            [3.0, 0.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(nii.affine, expected)
    assert np.allclose(nii.header.get_zooms(), (3.0, 2.0, 1.0, 1.5))

    # A slice alone is completed in the same way.
    flat = Affine(matrix=matrix[:3][:, [0, 1, 3]])
    NiftiImage(data=np.zeros((4, 5), dtype="f4"), transformations=[flat]).save(
        tmp_path / "flat.nii"
    )
    assert np.allclose(nb.load(str(tmp_path / "flat.nii")).affine, expected)


def test_an_lps_slice_is_turned_into_ras(tmp_path) -> None:  # noqa: ANN001
    # The z axis inserted next to an LPS plane is the third RAS axis.
    from brainhops.datamodel.axes import Axis, TimeAxis
    from brainhops.datamodel.orientations import Orientation

    def oriented(name: str, value: str) -> Axis:
        orientation = Orientation(type="anatomical", value=value)
        return Axis(name, "space", unit="mm", orientation=orientation)

    world = CoordinateSystem(
        axes=[
            oriented("x", "right-to-left"),
            oriented("y", "anterior-to-posterior"),
            TimeAxis(name="t", unit="s"),
        ]
    )
    xform = Affine(
        matrix=np.diag([2.0, 3.0, 1.5, 1.0])[:3],
        input=_typed("xyt"),
        output=world,
    )
    NiftiImage(
        data=np.zeros((4, 5, 3), dtype="f4"), transformations=[xform]
    ).save(tmp_path / "lps.nii")
    nii = nb.load(str(tmp_path / "lps.nii"))
    assert np.allclose(nii.affine, np.diag([-2.0, -3.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    "spec, stored",
    [
        ("zyx", "xyz"),
        ("yxz", "xyz"),
        ("zx", "xz"),
        ("kij", "kij"),  # not named x, y, z: the declared order is kept
        ("jxz", "jxz"),  # not all named x, y, z: the declared order is kept
    ],
)
def test_spatial_axes_are_ordered_by_name_when_named_xyz(  # noqa: D103
    tmp_path,  # noqa: ANN001
    spec: str,
    stored: str,
) -> None:
    # The per-axis zooms reveal the storage order.
    sizes = {"x": 2, "y": 3, "z": 4, "i": 5, "j": 6, "k": 7}
    factor = {"x": 1.5, "y": 2.5, "z": 3.5, "i": 4.5, "j": 5.5, "k": 6.5}
    values = np.random.rand(*(sizes[n] for n in spec), 2).astype("f4")
    image = NiftiImage(
        data=values,
        transformations=[
            Scaling(
                scale=[factor[n] for n in spec] + [2.0],
                input=_typed(spec + "t"),
            )
        ],
    )
    image.save(tmp_path / "named.nii")

    nii = nb.load(str(tmp_path / "named.nii"))
    zooms = [factor[n] for n in stored] + [1.0] * (3 - len(stored)) + [2.0]
    assert np.allclose(nii.header.get_zooms(), zooms)
    order = [spec.index(n) for n in stored] + [len(spec)]
    stored_values = values.transpose(order)
    assert np.array_equal(
        np.asarray(nii.dataobj).reshape(stored_values.shape), stored_values
    )


def test_a_large_image_is_written_as_nifti2() -> None:
    """Sizes beyond the 16-bit dim fields of NIfTI-1 are written as NIfTI-2."""
    small = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[Affine(matrix=np.eye(3, 4))],
    )
    assert isinstance(small.to_nibabel(), nb.Nifti1Image)
    assert not isinstance(small.to_nibabel(), nb.Nifti2Image)

    large = NiftiImage(
        data=np.zeros((40000, 1, 1), dtype="float32"),
        transformations=[Affine(matrix=np.eye(3, 4))],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = large.to_nibabel()
    assert isinstance(image, nb.Nifti2Image)


def test_like_copies_non_geometry_but_not_geometry(tmp_path) -> None:  # noqa: ANN001
    """`like` supplies the description and intent, never the geometry."""
    template = nb.Nifti1Header()
    template.set_sform(np.diag([9.0, 9.0, 9.0, 1.0]), code=2)
    template["descrip"] = b"from the template"
    template.set_intent(1011)  # NIFTI_INTENT_ESTIMATE

    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[Affine(matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3])],
    )

    built = image.to_nibabel(like=template)
    assert built.header["descrip"].tobytes().startswith(b"from the template")
    assert int(built.header["intent_code"]) == 1011

    # The geometry is the object's, not diag(9, 9, 9) of the template.
    target = tmp_path / "out.nii"
    image.save(target, like=template)
    assert np.allclose(np.diag(nb.load(str(target)).affine), [2, 3, 4, 1])
    reloaded = nb.load(str(target)).header
    assert reloaded["descrip"].tobytes().startswith(b"from the template")
    assert int(reloaded["intent_code"]) == 1011


def test_like_does_not_rescale_the_data(tmp_path) -> None:  # noqa: ANN001
    """The scl_slope and scl_inter of the template are not copied."""
    template = nb.Nifti1Header()
    template["scl_slope"] = 2.0
    template["scl_inter"] = 1.0

    image = NiftiImage(
        data=np.ones((3, 4, 5), dtype="float32"),
        transformations=[Affine(matrix=np.eye(3, 4))],
    )
    target = tmp_path / "ones.nii"
    image.save(target, like=template)

    assert np.allclose(nb.load(str(target)).get_fdata(), 1.0)


def test_the_unit_scale_comes_from_the_preferred_transform(tmp_path) -> None:  # noqa: ANN001
    """The unit scale comes from the preferred transformation only."""
    ras_cm = replace(
        RASCoordinateSystem(),
        axes=[
            replace(axis, unit="centimeter")
            for axis in RASCoordinateSystem().axes
        ],
    )
    scanner_mm = Affine(
        matrix=np.eye(3, 4),
        input=VoxelCoordinateSystem(),
        output=replace(RASCoordinateSystem(), name="scanner"),
    )
    preferred_cm = Affine(
        matrix=np.eye(3, 4),
        input=VoxelCoordinateSystem(),
        output=ras_cm,
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[scanner_mm, preferred_cm],
    )
    target = tmp_path / "mixed.nii"
    image.save(target)

    header = nb.load(str(target)).header
    assert header.get_xyzt_units()[0] == "mm"
    assert np.allclose(np.diag(nb.load(str(target)).affine), [10, 10, 10, 1])


def test_the_flip_handles_an_arbitrary_orientation(tmp_path) -> None:  # noqa: ANN001
    """Any orientation triple, here ARS, is reoriented to RAS."""
    from brainhops.datamodel import axes as _axes

    ars = CoordinateSystem(name=None, axes=[_axes.A(), _axes.R(), _axes.S()])
    affine = Affine(
        matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3],
        input=VoxelCoordinateSystem(),
        output=ars,
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[affine],
    )
    target = tmp_path / "ars.nii"
    image.save(target)

    expected = np.array(
        [
            [0.0, 3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(nb.load(str(target)).affine, expected)


def test_a_keyword_override_wins_over_the_object_and_like(tmp_path) -> None:  # noqa: ANN001
    """A keyword override is applied after both the object and `like`."""
    template = nb.Nifti1Header()
    template.set_intent(1011)  # NIFTI_INTENT_ESTIMATE

    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[Affine(matrix=np.eye(3, 4))],
    )
    target = tmp_path / "override.nii"
    image.save(target, like=template, dtype="int16", intent=2001)

    header = nb.load(str(target)).header
    assert header.get_data_dtype() == np.dtype("int16")
    assert int(header["intent_code"]) == 2001


def test_nifti_axes_are_copies_of_the_module_templates() -> None:
    """A system read from a header never holds the shared axis templates."""
    from brainhops.io.common.nifti._constants import _NIFTI_AXES
    from brainhops.io.common.nifti._header import _nifti_to_axes

    header = nb.Nifti1Image(np.zeros((2, 3, 4, 5)), np.eye(4)).header
    for axis, template in zip(_nifti_to_axes(header), _NIFTI_AXES):
        assert axis == template
        assert axis is not template


@pytest.mark.parametrize(
    "shape, intent",
    [((2, 3, 4), None), ((2, 3, 4, 5), None), ((2, 3, 4, 1, 3), 1006)],
)
def test_nifti_axes_count_samples(shape: tuple, intent: object) -> None:
    from brainhops.datamodel.units import IndexUnit
    from brainhops.io.common.nifti._header import _nifti_to_axes

    image = nb.Nifti1Image(np.zeros(shape, dtype="float32"), np.eye(4))
    if intent:
        image.header.set_intent(intent)
    axes = _nifti_to_axes(image.header)
    assert len(axes) == len(shape)
    assert all(isinstance(axis.unit, IndexUnit) for axis in axes)
