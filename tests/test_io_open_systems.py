"""Tests for open coordinate systems at the I/O boundary.

No file format can store `...`. A writer closes an open system from the
shape of what it writes, filling the axes that `...` stands for as the
format fills an axis it knows nothing about. A reader produces closed
systems, and a closed system survives a round trip unchanged.
"""

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.axes import (  # noqa: E402
    Axis,
    L,
    P,
    S,
    SpaceAxis,
)
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    CoordinateSystem,
    LPSCoordinateSystem,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import Affine  # noqa: E402
from brainhops.io.base.nifti import _voxel_to_ras  # noqa: E402
from brainhops.io.base.parsers import WriterError  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402

CS = CoordinateSystem
LPS_FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])


def _affine(output: tx.Optional[CS]) -> Affine:
    return Affine(
        matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3],
        input=VoxelCoordinateSystem(),
        output=output,
    )


# ----------------------------------------------------------------------
#   NIFTI
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "axes", [[L(), P(), S(), ...], [..., L(), P(), S()], [L(), ..., P(), S()]]
)
def test_nifti_closes_an_open_world_space_from_the_matrix(axes: list) -> None:
    # Wherever `...` sits, the 3-row matrix closes it to the three LPS
    # axes, whose orientation gives the flip to RAS.
    closed = _voxel_to_ras(_affine(LPSCoordinateSystem()))
    assert np.allclose(_voxel_to_ras(_affine(CS(axes=axes))), closed)
    assert np.allclose(closed, LPS_FLIP @ np.diag([2.0, 3.0, 4.0, 1.0]))


@pytest.mark.parametrize("output", [None, CS(), CS(axes=[...])])
def test_nifti_reads_an_unknown_world_space_as_unoriented(
    output: tx.Optional[CS],
) -> None:
    # The axes `...` stands for carry no orientation, as any axis NIfTI
    # knows nothing about, so no flip is applied.
    assert np.allclose(
        _voxel_to_ras(_affine(output)), np.diag([2.0, 3.0, 4.0, 1.0])
    )


def test_nifti_refuses_a_world_space_larger_than_the_matrix() -> None:
    with pytest.raises(WriterError, match="states more axes"):
        _voxel_to_ras(_affine(CS(axes=[L(), P(), S(), Axis(name="t"), ...])))


def test_nifti_writes_an_open_world_space_as_its_closed_form(
    tmp_path,  # noqa: ANN001
) -> None:
    data = np.zeros((3, 4, 5), dtype="float32")
    for name, output in (
        ("open.nii", CS(axes=[L(), P(), ...])),
        ("closed.nii", CS(axes=[L(), P(), Axis()])),
    ):
        NiftiImage(data=data, transformations=[_affine(output)]).save(
            tmp_path / name
        )
    open_header = nb.load(str(tmp_path / "open.nii")).header
    closed_header = nb.load(str(tmp_path / "closed.nii")).header
    assert np.allclose(open_header.get_sform(), closed_header.get_sform())
    assert np.allclose(open_header.get_qform(), closed_header.get_qform())


def test_nifti_round_trip_keeps_closed_systems(tmp_path) -> None:  # noqa: ANN001
    # A file read, written and read again carries the same systems, and
    # every system it carries is closed.
    affine = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, -20.0],
            [0.0, 0.0, 2.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image = nb.Nifti1Image(np.zeros((4, 5, 6), dtype="float32"), affine)
    image.header.set_sform(affine, code=2)
    image.header.set_qform(affine, code=1)
    nb.save(image, str(tmp_path / "source.nii"))

    first = io.images.load(tmp_path / "source.nii")
    first.save(tmp_path / "copy.nii")
    second = io.images.load(tmp_path / "copy.nii")

    def systems(loaded: tx.Any) -> list:
        out = []
        for t in loaded.transformations:
            out += [t.input, t.output]
        return out

    assert systems(first) == systems(second)
    for system in systems(first) + systems(second):
        assert system is not None and system.ndim is not None


# ----------------------------------------------------------------------
#   OME-ZARR
# ----------------------------------------------------------------------


def _pyramid(axes: list) -> tx.Any:
    from brainhops.io.images.zarr import OmeZarrImage

    data = np.zeros((4, 5, 6), dtype="float32")
    affine = Affine(matrix=np.diag([1.0, 2.0, 3.0, 1.0])[:3])
    image = SingleScaleImage(data=data, transformations=[affine])
    return OmeZarrImage(images=[image], axes=axes)


def _voxel_axes(path: str) -> list:
    loaded = io.images.load(path)
    return list(loaded.images[0].transformations[0].input.axes)


@pytest.mark.parametrize(
    "axes",
    [
        [SpaceAxis(name="x"), ...],
        [..., SpaceAxis(name="x")],
    ],
)
def test_zarr_closes_open_axes_from_the_data(
    tmp_path,  # noqa: ANN001
    axes: list,
) -> None:
    # The axes `...` stands for are written as any axis with no
    # description is: the same file as from the closed list.
    expanded = CS(axes=axes).expand(3).axes
    _pyramid(axes).save(str(tmp_path / "open.zarr"))
    _pyramid(expanded).save(str(tmp_path / "closed.zarr"))
    read_open = _voxel_axes(str(tmp_path / "open.zarr"))
    read_closed = _voxel_axes(str(tmp_path / "closed.zarr"))
    assert read_open == read_closed
    assert all(axis is not ... for axis in read_open)


def test_zarr_refuses_more_axes_than_the_data_has(tmp_path) -> None:  # noqa: ANN001
    axes = [SpaceAxis(name=n) for n in "xyzt"] + [...]
    with pytest.raises(WriterError, match="4 explicit axes to 3"):
        _pyramid(axes).save(str(tmp_path / "bad.zarr"))


def test_zarr_round_trip_keeps_closed_systems(tmp_path) -> None:  # noqa: ANN001
    axes = [SpaceAxis(name=n) for n in "xyz"]
    _pyramid(axes).save(str(tmp_path / "a.zarr"))
    first = io.images.load(str(tmp_path / "a.zarr"))
    first.save(str(tmp_path / "b.zarr"))
    second = io.images.load(str(tmp_path / "b.zarr"))
    for a, b in zip(first.images, second.images):
        for t, u in zip(a.transformations, b.transformations):
            assert t.input == u.input and t.output == u.output
            assert t.input.ndim == 3 and t.output.ndim == 3
