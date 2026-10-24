"""Tests of the transformations read from a NIfTI image with a flat intent.

Under a flat intent, such as a time series, a shape or a point set, the
axes named x, y and z are indices rather than spatial axes. These tests
check that such an image can be read and that its axes keep their types.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.transformations import Scaling  # noqa: E402

TIME_SERIES = 2001
SHAPE = 2005
POINTSET = 1008


def _write(tmp_path, shape: tuple, intent: int):  # noqa: ANN001, ANN202
    """Write a zero image with an intent code and return its path."""
    image = nb.Nifti1Image(np.zeros(shape, "float32"), np.eye(4))
    image.header.set_intent(intent)
    path = tmp_path / "flat.nii.gz"
    nb.save(image, path)
    return path


@pytest.mark.parametrize("intent", [TIME_SERIES, SHAPE, POINTSET])
@pytest.mark.parametrize("shape", [(4, 5, 6), (4, 5, 6, 10)])
def test_a_flat_intent_reads_without_a_world_space(
    tmp_path,  # noqa: ANN001
    intent: int,
    shape: tuple,
) -> None:
    """A flat intent gives only the voxel-to-physical scaling."""
    xforms = io.load(_write(tmp_path, shape, intent)).transformations
    assert len(xforms) == 1
    assert isinstance(xforms[0], Scaling)
    axes = xforms[0].output.axes
    assert [axis.name for axis in axes] == ["n", "x", "y", "z"][: len(shape)]
    assert all(axis.type is None for axis in axes)
    assert all(axis.orientation is None for axis in axes)


def test_spatial_axes_still_receive_an_orientation(tmp_path) -> None:  # noqa: ANN001
    """An image without an intent keeps its oriented RAS axes."""
    xforms = io.load(_write(tmp_path, (4, 5, 6, 10), 0)).transformations
    axes = xforms[-1].output.axes
    assert [axis.type for axis in axes] == ["space", "space", "space", "time"]
    assert axes[0].orientation is not None
    assert axes[3].orientation is None
