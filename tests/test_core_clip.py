"""Clipping limits the sampled voxel box without changing extrapolation."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from brainhops._core.backends import backend
from brainhops._core.bsplines import pull, pull_axes
from brainhops.datamodel._transformations.compute import separable as sep
from brainhops.datamodel.transformations import Affine, CartesianField


@pytest.mark.parametrize("margin", [None, -0.5, 0.0, 0.5])
@pytest.mark.parametrize("bound", ["nearest", "mirror", 7.0])
@pytest.mark.parametrize("degree", [0, 1, 3])
def test_clip_keeps_closed_endpoints_and_leaves_stencils_unchanged(
    margin: object, bound: object, degree: int
) -> None:
    values = np.array([10, 20, 40, 80], dtype=np.float32)
    x = np.array([-1, -0.5, -0.25, 0, 0.5, 2.5, 3, 3.25, 3.5, 4])
    mode = bound if isinstance(bound, str) else "grid-constant"
    cval = 0 if isinstance(bound, str) else bound
    expected = map_coordinates(
        values, x[None], order=degree, mode=mode, cval=cval
    )
    if margin is not None:
        expected[(x < -margin) | (x > 3 + margin)] = 0
    got = pull(values, x[:, None], degree, bound, False, clip=margin)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == values.dtype


def test_pull_axes_clips_only_sampled_axes_and_broadcasts_over_batch() -> None:
    values = np.full((2, 4, 3), 10, dtype=np.int16)
    coords = np.array([[-0.5], [0], [3.5], [3.75]])
    got = pull_axes(values, coords, [1], 1, "nearest", False, clip=0.5)
    assert got.shape == (2, 3, 4)
    np.testing.assert_array_equal(
        got, np.broadcast_to([10, 10, 10, 0], got.shape)
    )
    assert got.dtype == values.dtype


@pytest.mark.parametrize("clip", [float("nan"), float("inf"), -float("inf")])
def test_clip_rejects_nonfinite_margins(clip: float) -> None:
    with pytest.raises(ValueError, match="finite real numbers"):
        pull(np.ones(3), np.zeros((1, 1)), 1, "nearest", False, clip=clip)


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize(
    "margin", [-0.5, 0.0, 0.5, (0.25, 0.5), [(0, 0.5), (0.5, -0.5), (1, 0)]]
)
@pytest.mark.parametrize("bound", ["nearest", "mirror", 7.0])
@pytest.mark.parametrize("degree", [0, 1, 3])
def test_separable_masks_equal_monolithic_including_nonzero_background(
    coupled: bool,
    margin: object,
    bound: object,
    degree: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.random.default_rng(4).normal(size=(4, 5, 6))
    matrix = np.array([[0.7, 0, 0, -0.5], [0, 1, 0, -1], [0, 0, 0.8, -0.25]])
    if coupled:
        matrix[0, 1] = 0.2
    seq = Affine(matrix=matrix) @ CartesianField(shape=(8, 7, 9))
    opts = dict(degree=degree, bound=bound, coeff=False, clip=margin)
    with backend("numpy"):
        expected = pull(values, seq.compute().field, **opts)
        assert sep._plan(values.shape, seq, **opts) is not None

        def no_monolithic(*args, **kwargs) -> None:
            raise AssertionError("Clipping must preserve separable execution")

        monkeypatch.setattr(sep, "pull", no_monolithic)
        got = sep.pull_separable(values, seq, **opts)
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_clipped_dask_reslice_stays_lazy() -> None:
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    seq = Affine(matrix=np.array([[0.5, 0, -1], [0, 1, 0]])) @ CartesianField(
        shape=(12, 5)
    )
    executed = []
    with backend("dask"):
        with Callback(posttask=lambda *args: executed.append(True)):
            lazy = sep.pull_separable(
                da.from_array(values, chunks=(2, 3)),
                seq,
                degree=1,
                bound="nearest",
                coeff=False,
                clip=0.5,
            )
            assert isinstance(lazy, da.Array)
            assert not executed
    with backend("numpy"):
        expected = pull(
            values, seq.compute().field, 1, "nearest", False, clip=0.5
        )
    np.testing.assert_allclose(lazy.compute(), expected)


def test_core_dask_clipping_stays_lazy_and_handles_batched_coordinates() -> (
    None
):
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    coords = np.array([[-0.75], [-0.5], [0], [3.5], [3.75]])
    executed = []
    with Callback(posttask=lambda *args: executed.append(True)):
        lazy = pull(
            da.from_array(values, chunks=(1, 2)),
            da.from_array(coords, chunks=(2, 1)),
            1,
            "nearest",
            False,
            clip=0.5,
        )
        assert not executed
    expected = pull(values, coords, 1, "nearest", False, clip=0.5)
    np.testing.assert_array_equal(lazy.compute(), expected)


@pytest.mark.parametrize(
    "clip_arg", ["0.5", "(0.5, 0.5)", "[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]"]
)
def test_reader_images_and_cli_use_the_same_clip_margin(
    tmp_path,  # noqa: ANN001
    clip_arg: str,
) -> None:
    nb = pytest.importorskip("nibabel")
    from brainhops import io
    from brainhops.cli import main
    from brainhops.datamodel.images import MultiScaleImage

    source_path = tmp_path / "source.nii.gz"
    reference_path = tmp_path / "reference.nii.gz"
    output_path = tmp_path / "resliced.nii.gz"
    nb.save(
        nb.Nifti1Image(np.full((4, 5, 6), 10, np.float32), np.eye(4)),
        source_path,
    )
    affine = np.diag([0.25, 1, 1, 1])
    affine[:3, 3] = [-0.75, 1, 1]
    nb.save(
        nb.Nifti1Image(np.zeros((21, 2, 2), np.float32), affine),
        reference_path,
    )
    with backend("numpy"):
        source, reference = (
            io.images.load(source_path),
            io.images.load(reference_path),
        )
        result = source.reslice(reference, bound="nearest", clip=0.5)
        expected = np.full((21, 2, 2), 10, np.float32)
        x = -0.75 + np.arange(21) * 0.25
        expected[(x < -0.5) | (x > 3.5)] = 0
        np.testing.assert_array_equal(result.data, expected)
        pyramid = MultiScaleImage(images=[source])
        level = pyramid.reslice(
            reference, bound="nearest", clip=[(0.5, 0.5)] * 3
        )
        np.testing.assert_array_equal(level.data, expected)
        assert (
            main(
                [
                    "reslice",
                    str(source_path),
                    "--reference",
                    str(reference_path),
                    "--output",
                    str(output_path),
                    "--bound",
                    "nearest",
                    "--clip",
                    clip_arg,
                ]
            )
            == 0
        )
    np.testing.assert_array_equal(nb.load(output_path).get_fdata(), expected)


@pytest.mark.parametrize(
    "clip", [[], [(0, 0)], [0, 1, 2], [(0, 0, 0), (0, 0)], "bad", (None, 0)]
)
def test_malformed_clip_metadata_is_rejected(clip: object) -> None:
    with pytest.raises(ValueError, match="clip"):
        pull(np.ones((3, 4)), np.zeros((2, 2)), 1, "nearest", False, clip=clip)


def test_asymmetric_pairs_apply_in_sampled_axes_order() -> None:
    values = np.full((3, 2, 5), 10.0)
    # axes=[2, 0]: the first pair applies to length 5, the second to length 3.
    coords = np.array([[-0.5, 0], [4.25, 0], [1, -0.25], [1, 2.5], [1, 2.75]])
    got = pull_axes(
        values, coords, [2, 0], 1, "nearest", False, clip=[(0.5, 0), (0, 0.5)]
    )
    np.testing.assert_array_equal(got, [[10, 0, 0, 10, 0]] * 2)


def test_per_axis_margins_follow_input_axes_after_permutation() -> None:
    values = np.full((3, 4, 5), 10.0)
    matrix = np.array([[0, 0, 0.5, -0.5], [0.5, 0, 0, 0], [0, 1, 0, -1]])
    seq = Affine(matrix=matrix) @ CartesianField(shape=(9, 8, 8))
    clip = [(0, 0.5), (0.5, -0.5), (1, 0)]
    with backend("numpy"):
        coords = np.asarray(seq.compute().field)
        valid = (
            (coords[..., 0] >= 0)
            & (coords[..., 0] <= 2.5)
            & (coords[..., 1] >= -0.5)
            & (coords[..., 1] <= 2.5)
            & (coords[..., 2] >= -1)
            & (coords[..., 2] <= 4)
        )
        got = sep.pull_separable(
            values, seq, degree=1, bound="nearest", coeff=False, clip=clip
        )
    np.testing.assert_array_equal(got, np.where(valid, 10, 0))
