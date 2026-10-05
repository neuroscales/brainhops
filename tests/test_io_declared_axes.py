"""
Tests for placing the axes of an image by their declared types, when it
is written to MGH and to MRtrix.

Like the NIfTI writer, each places the spatial axes first (in `x, y, z`
order when they are so named), then time, then the other axes, transposes
the data to match, stores the repetition time where the format has a
place for it, and refuses what it cannot represent.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.axes import Axis, SpaceAxis, TimeAxis  # noqa: E402
from brainhops.datamodel.systems import CoordinateSystem  # noqa: E402
from brainhops.datamodel.transformations import Affine, Scaling  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    UnrepresentableTransformationError,
)
from brainhops.io.images.freesurfer import MghImage  # noqa: E402
from brainhops.io.images.mrtrix import MrtrixImage  # noqa: E402

# A voxel-to-world map in `x, y, z, t` order: a spatial affine, and a
# repetition time of 2 s.
XYZT = np.array(
    [
        [0.0, -2.0, 0.0, 0.0, 10.0],
        [1.5, 0.0, 0.0, 0.0, -3.0],
        [0.0, 0.0, 2.5, 0.0, 4.0],
        [0.0, 0.0, 0.0, 2.0, 0.0],
    ]
)
SHAPE = dict(zip("xyzt", (4, 5, 6, 3)))


def _system(spec, space="index", time="index"):  # noqa: ANN001, ANN202
    axes = []
    for name in spec:
        if name == "t":
            axes.append(TimeAxis(name="t", unit=time))
        elif name == "c":
            axes.append(Axis(name="c", type="channel", unit="index"))
        else:
            axes.append(SpaceAxis(name=name, unit=space))
    return CoordinateSystem(axes=axes)


def _image(cls, voxel_order, backend, time="s", matrix=XYZT):  # noqa: ANN001, ANN202
    """An image of `x, y, z, t`, its axes listed in `voxel_order`, and
    its values in `x, y, z, t` order."""
    values = np.random.rand(*(SHAPE[n] for n in "xyzt")).astype("f4")
    data = values.transpose(["xyzt".index(n) for n in voxel_order])
    if backend == "dask":
        da = pytest.importorskip("dask.array")
        data = da.from_array(data, chunks=2)
    columns = ["xyzt".index(n) for n in voxel_order] + [4]
    xform = Affine(
        matrix=matrix[:, columns],
        input=_system(voxel_order),
        output=_system("xyzt", "mm", time),
    )
    return cls(data=data, transformations=[xform]), values


def _vox2ras(matrix: np.ndarray = XYZT) -> np.ndarray:
    affine = np.eye(4)
    affine[:3] = matrix[:3, [0, 1, 2, 4]]
    return affine


# ----------------------------------------------------------------------
#   MGH
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask"])
@pytest.mark.parametrize("voxel_order", ["xyzt", "txyz", "zytx"])
def test_mgh_places_the_frames_by_type(  # noqa: D103
    tmp_path,  # noqa: ANN001
    backend: str,
    voxel_order: str,
) -> None:
    image, values = _image(MghImage, voxel_order, backend)
    target = tmp_path / "bold.mgz"
    image.save(target)

    mgh = nb.load(str(target))
    assert np.array_equal(np.asarray(mgh.dataobj), values)
    assert np.allclose(mgh.affine, _vox2ras(), atol=1e-5)
    # The repetition time is stored in milliseconds.
    assert float(mgh.header["tr"]) == pytest.approx(2000.0)

    reloaded = io.images.load(target)
    assert [a.name for a in reloaded.system.axes] == list("xyzt")
    assert np.allclose(reloaded.transformations[0].scale[-1], 2000.0)
    assert np.allclose(reloaded.vox2ras, _vox2ras(), atol=1e-5)

    # Read back and written again, it is the same file.
    again = tmp_path / "again.mgz"
    reloaded.save(again)
    assert float(nb.load(str(again)).header["tr"]) == pytest.approx(2000.0)
    assert np.array_equal(np.asarray(nb.load(str(again)).dataobj), values)


def test_mgh_reads_a_time_unit(tmp_path) -> None:  # noqa: ANN001
    image, _ = _image(MghImage, "xyzt", "numpy", time="ms")
    image.save(tmp_path / "bold.mgh")
    assert float(nb.load(str(tmp_path / "bold.mgh")).header["tr"]) == 2.0


def test_mgh_gives_a_slice_with_frames_a_z_axis(tmp_path) -> None:  # noqa: ANN001
    values = np.random.rand(4, 5, 3).astype("f4")
    xform = Scaling(scale=[1.0, 2.0, 3.0], input=_system("txy"))
    MghImage(data=values.transpose(2, 0, 1), transformations=[xform]).save(
        tmp_path / "slice.mgh"
    )
    mgh = nb.load(str(tmp_path / "slice.mgh"))
    assert mgh.shape == (4, 5, 1, 3)
    assert np.array_equal(np.asarray(mgh.dataobj)[:, :, 0], values)
    assert np.allclose(mgh.header.get_zooms()[:3], (2.0, 3.0, 1.0))
    # Frames that still count frames state no repetition time.
    assert float(mgh.header["tr"]) == 0.0


def test_mgh_stores_channels_as_frames(tmp_path) -> None:  # noqa: ANN001
    values = np.random.rand(4, 5, 6, 3).astype("f4")
    xform = Scaling(scale=[1.0, 2.0, 3.0, 4.0], input=_system("cxyz"))
    MghImage(data=values.transpose(3, 0, 1, 2), transformations=[xform]).save(
        tmp_path / "rgb.mgh"
    )
    mgh = nb.load(str(tmp_path / "rgb.mgh"))
    assert np.array_equal(np.asarray(mgh.dataobj), values)
    assert float(mgh.header["tr"]) == 0.0


@pytest.mark.parametrize(
    "spec, scale, message",
    [
        ("xyztc", [1.0] * 5, "1 axis besides the spatial ones"),
        ("xyzc", [1.0, 1.0, 1.0, 2.0], "frames that are not time"),
    ],
)
def test_mgh_refuses_what_it_cannot_store(  # noqa: D103
    spec: str,
    scale: list,
    message: str,
) -> None:
    image = MghImage(
        data=np.zeros((2,) * len(spec), dtype="f4"),
        transformations=[Scaling(scale=scale, input=_system(spec))],
    )
    with pytest.raises(UnrepresentableTransformationError) as info:
        image.to_bytes()
    assert message in str(info.value)


def test_mgh_refuses_a_time_origin() -> None:  # noqa: D103
    matrix = XYZT.copy()
    matrix[3, 4] = 0.5
    image, _ = _image(MghImage, "xyzt", "numpy", matrix=matrix)
    with pytest.raises(UnrepresentableTransformationError) as info:
        image.to_bytes()
    assert "no origin for the time axis" in str(info.value)


# ----------------------------------------------------------------------
#   MRtrix
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask"])
@pytest.mark.parametrize("voxel_order", ["xyzt", "txyz", "zytx"])
def test_mrtrix_places_the_volumes_by_type(  # noqa: D103
    tmp_path,  # noqa: ANN001
    backend: str,
    voxel_order: str,
) -> None:
    image, values = _image(MrtrixImage, voxel_order, backend)
    target = tmp_path / "bold.mif"
    image.save(target)

    reloaded = MrtrixImage.load(target)
    assert np.array_equal(np.asarray(reloaded.data), values)
    # The time axis is spaced by the repetition time, in seconds.
    assert reloaded.header.vox == pytest.approx((1.5, 2.0, 2.5, 2.0))
    assert np.allclose(
        reloaded.transformation.homogeneous_matrix[:, [0, 1, 2, -1]],
        _vox2ras(),
    )

    # Read back and written again, it is the same file.
    again = tmp_path / "again.mif"
    reloaded.save(again)
    assert again.read_bytes() == target.read_bytes()


def test_mrtrix_states_times_in_seconds(tmp_path) -> None:  # noqa: ANN001
    image, _ = _image(MrtrixImage, "txyz", "numpy", time="ms")
    image.save(tmp_path / "bold.mif")
    assert MrtrixImage.load(tmp_path / "bold.mif").header.vox[3] == 0.002


def test_mrtrix_orders_the_axes_after_the_spatial_ones(tmp_path) -> None:  # noqa: ANN001
    # Time first, then the channels, whatever their declared order; a
    # slice is given a `z` axis, so they are not read as spatial.
    values = np.random.rand(4, 5, 3, 2).astype("f4")  # x, y, t, c
    xform = Scaling(scale=[1.0, 2.0, 3.0, 4.0], input=_system("cxty"))
    image = MrtrixImage(
        data=values.transpose(3, 0, 2, 1), transformations=[xform]
    )
    image.save(tmp_path / "slice.mif")

    reloaded = MrtrixImage.load(tmp_path / "slice.mif")
    assert reloaded.header.dim == (4, 5, 1, 3, 2)
    assert np.array_equal(np.asarray(reloaded.data)[:, :, 0], values)
    assert reloaded.header.vox == pytest.approx((2.0, 4.0, 1.0, 3.0, 1.0))


def test_mrtrix_refuses_an_origin_on_the_volumes() -> None:  # noqa: D103
    matrix = XYZT.copy()
    matrix[3, 4] = 0.5
    image, _ = _image(MrtrixImage, "xyzt", "numpy", matrix=matrix)
    with pytest.raises(UnrepresentableTransformationError) as info:
        image.to_bytes()
    assert "no origin" in str(info.value)


def test_a_volume_series_from_mrtrix_stays_four_dimensional_in_nifti(  # noqa: D103
    tmp_path,  # noqa: ANN001
) -> None:
    # The MRtrix reader gives the volume axis no type, and maps the spatial
    # axes only: the data is written as it is, its volumes in `dim[4]`.
    from brainhops.datamodel.images import SingleScaleImage

    values = np.random.rand(4, 5, 6, 7).astype("f4")
    MrtrixImage(data=values).save(tmp_path / "dwi.mif")
    image = io.images.load(tmp_path / "dwi.mif")
    io.save(SingleScaleImage.from_instance(image), tmp_path / "dwi.nii")
    nii = nb.load(str(tmp_path / "dwi.nii"))
    assert nii.shape == (4, 5, 6, 7)
    assert np.array_equal(np.asarray(nii.dataobj), values)


# ----------------------------------------------------------------------
#   The directions of inserted spatial axes
# ----------------------------------------------------------------------


def test_complete_basis() -> None:  # noqa: D103
    from brainhops.io.base._geometry import complete_basis

    # Two in-plane directions: the unit normal, right-handed.
    columns = np.array([[0.0, 2.0], [0.0, 0.0], [3.0, 0.0]])
    assert np.allclose(complete_basis(columns), [[0.0], [1.0], [0.0]])
    # One direction, or two that do not span a plane: the world axes, less
    # what the columns span.
    assert np.allclose(
        complete_basis(np.array([[2.0], [0.0], [0.0]])),
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )
    parallel = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]])
    assert np.allclose(complete_basis(parallel), [[1.0], [0.0], [0.0]])
