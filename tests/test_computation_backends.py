"""Tests of the backend in which computations run and return results.

A computation chooses its package with `best_backend`, and
`to_promoted_backend` moves the other arrays to that package. The CuPy tests
are skipped where CuPy is not installed, and the Dask tests check the
same paths in memory.
"""

import numpy as np
import pytest

from brainhops._core import bsplines
from brainhops._core.backends import (
    backend,
    promote_backends,
    to_backend_of,
    to_promoted_backend,
)
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Affine, DisplacementField

da = pytest.importorskip("dask.array")
Callback = pytest.importorskip("dask.callbacks").Callback

RNG = np.random.default_rng(0)
VOLUME = RNG.random((12, 13, 14)).astype(np.float32)
FIELD = RNG.random((12, 13, 14, 3)) * 0.3
ANGLE = 0.3
ROTATION = np.array(
    [
        [np.cos(ANGLE), -np.sin(ANGLE), 0.0, 1.5],
        [np.sin(ANGLE), np.cos(ANGLE), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)


class _Computations(Callback):
    # Count the Dask graphs that run while the context is open.

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def _start(self, dsk) -> None:  # noqa: ANN001
        self.count += 1


def _target() -> SingleScaleImage:
    # A grid rotated with respect to the voxels of VOLUME, so that the
    # reslice samples every voxel with coordinates.
    return SingleScaleImage(np.zeros_like(VOLUME), [Affine(ROTATION)])


def test_arrays_in_memory_are_not_moved() -> None:
    # Without an array on a GPU, the arrays are returned as they are,
    # because Dask accepts NumPy operands.
    matrix = np.eye(3)
    lazy = da.from_array(FIELD, chunks=6)
    moved = to_promoted_backend(lazy, matrix, None, 2.0)
    assert moved[0] is lazy
    assert moved[1] is matrix
    assert moved[2:] == (None, 2.0)
    assert to_backend_of(matrix, lazy) is matrix
    assert to_backend_of(lazy, matrix) is lazy


def test_the_promoted_backend_follows_the_inputs() -> None:
    # Without a setting, a computation is lazy when one of its inputs is
    # lazy. An explicit setting is kept, and no array gives no setting.
    lazy = da.from_array(FIELD, chunks=6)
    assert promote_backends(VOLUME, None) == "numpy"
    assert promote_backends(VOLUME, lazy) == "dask"
    assert promote_backends() is None
    with backend("numpy"):
        assert promote_backends(lazy) == "numpy"


def test_coordinates_are_created_in_the_promoted_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reslice of a Dask image creates the coordinates at which it
    # samples the voxels as a Dask array, rather than with NumPy and then
    # moving them. The small weight matrices of `spline_matrix` are also
    # sampled with `pull`, from NumPy arrays, and are not counted.
    seen = []
    original = bsplines.pull

    def pull(input, coords, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if isinstance(input, da.Array):
            seen.append(type(coords))
        return original(input, coords, *args, **kwargs)

    monkeypatch.setattr(bsplines, "pull", pull)
    SingleScaleImage(da.from_array(VOLUME, chunks=6)).reslice(_target())
    assert seen
    assert all(issubclass(kind, da.Array) for kind in seen)


def test_a_lazy_reslice_computes_nothing() -> None:
    # A Dask image is resliced into a Dask image, without running any
    # graph, and its values are those of the NumPy reslice.
    expected = SingleScaleImage(VOLUME).reslice(_target()).data
    image = SingleScaleImage(da.from_array(VOLUME, chunks=6))
    with _Computations() as computations:
        moved = image.reslice(_target())
    assert isinstance(moved.dataobj, da.Array)
    assert computations.count == 0
    np.testing.assert_allclose(np.asarray(moved.data), expected, atol=1e-5)


def test_a_reslice_follows_an_explicit_setting() -> None:
    # Under the Dask setting, a NumPy image is resliced lazily.
    image = SingleScaleImage(VOLUME)
    with backend("dask"):
        moved = image.reslice(_target())
    assert isinstance(moved.dataobj, da.Array)
    assert image.dataobj is VOLUME


def test_field_views_follow_the_setting() -> None:
    # The values and coefficients of a field follow the setting, whereas
    # its data is the array that it stores.
    field = DisplacementField(FIELD)
    with backend("dask"):
        assert field.data is FIELD
        assert isinstance(field.field, da.Array)
        assert isinstance(field.coefficients, da.Array)
    assert isinstance(field.field, np.ndarray)


def test_a_dask_field_is_inverted_into_a_dask_field() -> None:
    # The inversion is computed with NumPy, and its result is converted
    # back to the package of the field.
    expected = DisplacementField(FIELD).inverse().data
    lazy = DisplacementField(da.from_array(FIELD, chunks=6))
    inverse = lazy.inverse().data
    assert isinstance(inverse, da.Array)
    np.testing.assert_allclose(np.asarray(inverse), expected)


def test_a_composition_follows_the_setting() -> None:
    # Composing an affine with a field under the Dask setting gives a lazy
    # field with the values of the NumPy composition.
    expected = (Affine(ROTATION) @ DisplacementField(FIELD)).compute().field
    with backend("dask"):
        composed = (Affine(ROTATION) @ DisplacementField(FIELD)).compute()
        field = composed.field
    assert isinstance(field, da.Array)
    np.testing.assert_allclose(np.asarray(field), expected)


def test_slicing_builds_its_matrix_with_numpy() -> None:
    # The affine that maps a slice to the original grid is a small
    # parameter, which is built with NumPy under any setting, rather than
    # built with Dask and then computed.
    image = SingleScaleImage(VOLUME, [Affine(ROTATION)])
    with _Computations() as computations, backend("dask"):
        part = image[1:3]
    assert computations.count == 0
    (chain,) = part.transformations
    assert all(isinstance(t.matrix, np.ndarray) for t in chain.transformations)


# --- CuPy -------------------------------------------------------------


def test_cupy_arrays_attract_the_others() -> None:
    # With an array on a GPU, NumPy arrays become CuPy arrays and Dask
    # arrays get CuPy chunks.
    cp = pytest.importorskip("cupy")
    on_gpu = cp.asarray(VOLUME)
    moved = to_promoted_backend(on_gpu, np.eye(3), da.from_array(VOLUME))
    assert moved[0] is on_gpu
    assert isinstance(moved[1], cp.ndarray)
    assert isinstance(moved[2]._meta, cp.ndarray)


def test_only_the_moved_array_goes_to_the_gpu() -> None:
    # The one-way form moves the array to the device of the reference
    # and leaves the reference where it is.
    cp = pytest.importorskip("cupy")
    on_gpu = cp.asarray(VOLUME)
    assert isinstance(to_backend_of(VOLUME, on_gpu), cp.ndarray)
    assert isinstance(to_backend_of(on_gpu, VOLUME), np.ndarray)


def test_a_cupy_image_is_resliced_on_the_gpu() -> None:
    # The coordinates and the matrices meet the voxels on the GPU.
    cp = pytest.importorskip("cupy")
    expected = SingleScaleImage(VOLUME).reslice(_target()).data
    moved = SingleScaleImage(VOLUME).as_cupy().reslice(_target())
    assert isinstance(moved.dataobj, cp.ndarray)
    np.testing.assert_allclose(moved.dataobj.get(), expected, atol=1e-5)


def test_a_cupy_field_is_inverted_on_the_gpu() -> None:
    # The inversion runs with CuPy and gives the NumPy values.
    cp = pytest.importorskip("cupy")
    expected = DisplacementField(FIELD).inverse().data
    inverse = DisplacementField(cp.asarray(FIELD)).inverse().data
    assert isinstance(inverse, cp.ndarray)
    np.testing.assert_allclose(inverse.get(), expected)


def test_fields_are_composed_on_the_gpu() -> None:
    # Two fields stored with NumPy are composed under the CuPy setting.
    cp = pytest.importorskip("cupy")
    first = DisplacementField(FIELD)
    second = DisplacementField(FIELD[::-1].copy())
    expected = (first @ second).compute().field
    with backend("cupy"):
        field = (first @ second).compute().field
    assert isinstance(field, cp.ndarray)
    np.testing.assert_allclose(field.get(), expected)
