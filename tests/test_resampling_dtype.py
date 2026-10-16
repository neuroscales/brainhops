"""Output dtypes preserve interpolation before the final conversion."""

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from brainhops import io
from brainhops._core.bsplines import pull, pull_axes
from brainhops.backends import backend
from brainhops.cli import main
from brainhops.datamodel._transformations.compute import separable as sep
from brainhops.datamodel.images import MultiScaleImage
from brainhops.datamodel.transformations import Affine, CartesianField


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.uint8])
@pytest.mark.parametrize("degree", [0, 1, 3])
def test_pull_dtype_matches_scipy(dtype: object, degree: int) -> None:
    values = np.array([[-12, 3, 257, 20], [2, 9, 103, 5]], dtype=np.int16)
    coords = np.array([[-0.25], [0.5], [1.25], [2.75], [3.5]])
    got = pull(values, coords, degree, "nearest", False, dtype=dtype)
    expected = np.stack(
        [
            map_coordinates(
                row, coords.T, order=degree, mode="nearest", output=dtype
            )
            for row in values
        ]
    )
    assert got.dtype == dtype
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_pull_axes_dtype_preserves_fractions_and_clipping() -> None:
    values = np.arange(24, dtype=np.int16).reshape(2, 4, 3)
    coords = np.array([[0.25], [1.5], [4]])
    got = pull_axes(
        values, coords, [1], 1, "nearest", False, clip=0.5, dtype="float32"
    )
    expected = np.stack(
        [values[:, 0] + 0.75, values[:, 1] + 1.5, np.zeros((2, 3))],
        axis=-1,
    )
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)


def test_integer_output_does_not_quantize_the_input_first() -> None:
    got = pull(
        np.array([0.1, 1.1]),
        np.array([[0.45]]),
        1,
        "nearest",
        False,
        dtype="int16",
    )
    np.testing.assert_array_equal(got, [1])


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_separable_and_fallback_keep_output_dtype(
    coupled: bool, dtype: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.arange(120, dtype=np.int16).reshape(4, 5, 6)
    matrix = np.array([[1, 0, 0, 0.25], [0, 1, 0, 0.3], [0, 0, 1, 0.4]])
    if coupled:
        matrix[0, 1] = 0.1
    seq = Affine(matrix=matrix) @ CartesianField(shape=(3, 4, 5))
    options = dict(degree=1, bound="nearest", coeff=False, dtype=dtype)
    with backend("numpy"):
        expected = pull(values, seq.compute().field, **options)
        actual = sep.pull_separable(values, seq, **options)
        monkeypatch.setattr(sep, "_plan", lambda *args, **kwargs: None)
        fallback = sep.pull_separable(values, seq, **options)
    assert actual.dtype == fallback.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_array_equal(fallback, expected)


def test_native_loaded_image_and_pyramid_support_float_output(
    tmp_path: Path,
) -> None:
    nib = pytest.importorskip("nibabel")
    values = np.arange(120, dtype=np.int16).reshape(4, 5, 6)
    source_path, ref_path = tmp_path / "source.nii", tmp_path / "ref.nii"
    reference_affine = np.eye(4)
    reference_affine[0, 3] = 0.25
    nib.save(nib.Nifti1Image(values, np.eye(4)), source_path)
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5)), reference_affine), ref_path)
    with backend("numpy"):
        source = io.images.load(source_path, hint="nifti")
        reference = io.images.load(ref_path, hint="nifti")
        original = np.asarray(source.data).copy()
        output = source.reslice(reference, dtype="float32")
        pyramid = source.to(MultiScaleImage)
        pyramid_output = pyramid.reslice(reference, dtype="float32")
    expected = values[:3, :4, :5] + 7.5
    assert output.dtype == pyramid_output.dtype == np.float32
    assert output.metadata.dtype == np.float32
    np.testing.assert_array_equal(output.data, expected)
    np.testing.assert_array_equal(pyramid_output.data, expected)
    assert source.dtype == np.int16
    assert source.metadata.dtype == np.int16
    np.testing.assert_array_equal(source.data, original)


def test_dask_output_dtype_stays_lazy() -> None:
    dask = pytest.importorskip("dask")
    da = pytest.importorskip("dask.array")
    events = []

    @dask.delayed
    def read() -> np.ndarray:
        events.append("read")
        return np.array([0, 3, 9, 15], dtype=np.int16)

    values = da.from_delayed(read(), shape=(4,), dtype=np.int16)
    coords = da.from_array(np.array([[0.25], [1.5]]), chunks=(1, 1))
    lazy = pull(values, coords, 1, "nearest", False, dtype="float32")
    assert isinstance(lazy, da.Array) and lazy.dtype == np.float32
    assert events == []
    np.testing.assert_array_equal(
        lazy.compute(scheduler="synchronous"), [0.75, 6]
    )
    assert events


def test_cli_can_save_float_output_from_integer_input(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    source, reference, output = [
        tmp_path / name for name in ("source.nii", "ref.nii", "out.nii")
    ]
    affine = np.eye(4)
    affine[0, 3] = 0.25
    values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    nib.save(nib.Nifti1Image(values, np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.zeros((1, 2, 3)), affine), reference)
    assert (
        main(
            [
                "reslice",
                str(source),
                "--reference",
                str(reference),
                "--output",
                str(output),
                "--dtype",
                "float32",
            ]
        )
        == 0
    )
    saved = nib.load(output)
    assert saved.get_data_dtype() == np.float32
    np.testing.assert_array_equal(saved.get_fdata(), values[:1, :2, :3] + 3)


def test_native_dask_image_reslice_keeps_dtype_selection_lazy(
    tmp_path: Path,
) -> None:
    nib = pytest.importorskip("nibabel")
    da = pytest.importorskip("dask.array")
    callbacks = pytest.importorskip("dask.callbacks")
    values = np.arange(120, dtype=np.int16).reshape(4, 5, 6)
    path = tmp_path / "image.nii"
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)
    tasks = []
    callback = callbacks.Callback(pretask=lambda *args: tasks.append(True))
    with backend("dask"), callback:
        image = io.images.load(path, hint="nifti")
        output = image.reslice(dtype="float32")
        assert isinstance(output.data, da.Array)
        assert output.dtype == np.float32 and image.dtype == np.int16
        assert not tasks
    np.testing.assert_array_equal(output.data.compute(), values)


@pytest.mark.parametrize("dtype", [np.int16, np.float32])
def test_none_dtype_keeps_existing_results(dtype: object) -> None:
    values = np.arange(12, dtype=dtype)
    coords = np.array([[0.5], [2.25], [11.25]])
    got = pull(values, coords, 1, "reflect", False, dtype=None, clip=None)
    expected = map_coordinates(values, coords.T, order=1, mode="reflect")
    assert got.dtype == values.dtype
    np.testing.assert_array_equal(got, expected)
