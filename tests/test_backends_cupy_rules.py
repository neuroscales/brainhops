"""Computations with CuPy, checked against a strict stand-in for CuPy.

The stand-in of `fake_cupy` refuses NumPy arrays and lazy arrays as
operands, as CuPy does, so that these tests find the computations that
would fail on a GPU. A computation that involves a lazy array runs in
CuPy under the CuPy setting, and next to a CuPy array under the default
setting.
"""

import fake_cupy
import numpy as np
import pytest
from test_computation_backends import FIELD, ROTATION, VOLUME, rotated_target
from test_core_proxy import RecordingProxy

from brainhops._core.backends import backend
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Affine, DisplacementField


@pytest.fixture
def cp(monkeypatch: pytest.MonkeyPatch) -> object:
    return fake_cupy.install(monkeypatch)


def _proxy(values: np.ndarray) -> RecordingProxy:
    return RecordingProxy(np.asarray(values))


def _on_gpu(cp: object, result: object, expected: np.ndarray) -> None:
    assert isinstance(result, cp.ndarray), type(result)
    np.testing.assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
#   COMPUTATIONS ON PROXIES
# ----------------------------------------------------------------------

COMPUTATIONS = [
    lambda x: x + 1,
    lambda x: 2 * x,
    lambda x: x * np.ones(x.shape),
    lambda x: np.exp(x / 100),
    lambda x: np.mean(x, axis=0),
    lambda x: np.concatenate([x, x]),
    lambda x: x.sum(),
    lambda x: x.reshape(-1).astype(np.float32),
    lambda x: x == 3,
]


@pytest.mark.parametrize("compute", COMPUTATIONS)
def test_a_computation_on_a_proxy_follows_the_setting(cp, compute) -> None:  # noqa: ANN001
    values = np.arange(24.0).reshape(4, 6)
    with backend("cupy"):
        result = compute(_proxy(values))
    _on_gpu(cp, result, compute(values))


@pytest.mark.parametrize("left", [True, False])
def test_a_proxy_joins_a_cupy_operand(cp, left: bool) -> None:  # noqa: ANN001
    values = np.arange(24.0).reshape(4, 6)
    proxy, gpu = _proxy(values), cp.asarray(values + 1)
    result = proxy * gpu if left else gpu * proxy
    _on_gpu(cp, result, values * (values + 1))
    _on_gpu(cp, np.add(gpu, proxy), values + values + 1)
    _on_gpu(cp, np.stack([gpu, proxy]), np.stack([values + 1, values]))


def test_conversions_follow_the_setting(cp) -> None:  # noqa: ANN001
    image = SingleScaleImage(_proxy(VOLUME))
    with backend("cupy"):
        _on_gpu(cp, image.to_backend(), VOLUME)
    _on_gpu(cp, image.to_cupy(), VOLUME)
    _on_gpu(cp, image.as_cupy().data, VOLUME)


# ----------------------------------------------------------------------
#   COMPUTATIONS OF BRAINHOPS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("lazy", [False, True])
def test_a_reslice_follows_the_setting(cp, lazy: bool) -> None:  # noqa: ANN001
    expected = SingleScaleImage(VOLUME).reslice(rotated_target()).data
    data = _proxy(VOLUME) if lazy else VOLUME
    with backend("cupy"):
        moved = SingleScaleImage(data).reslice(rotated_target())
    _on_gpu(cp, moved.data, expected)


def test_a_cupy_image_is_resliced_on_the_gpu(cp) -> None:  # noqa: ANN001
    expected = SingleScaleImage(VOLUME).reslice(rotated_target()).data
    moved = SingleScaleImage(cp.asarray(VOLUME)).reslice(rotated_target())
    _on_gpu(cp, moved.data, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_a_composition_follows_the_setting(cp, lazy: bool) -> None:  # noqa: ANN001
    expected = (Affine(ROTATION) @ DisplacementField(FIELD)).compute().field
    data = _proxy(FIELD) if lazy else FIELD
    with backend("cupy"):
        field = (Affine(ROTATION) @ DisplacementField(data)).compute().field
    _on_gpu(cp, field, expected)


def test_fields_are_composed_on_the_gpu(cp) -> None:  # noqa: ANN001
    first = DisplacementField(FIELD)
    second = DisplacementField(_proxy(FIELD[::-1].copy()))
    expected = (first @ DisplacementField(FIELD[::-1].copy())).compute()
    with backend("cupy"):
        field = (first @ second).compute().field
    _on_gpu(cp, field, expected.field)


def test_the_coefficients_of_a_lazy_field_follow_the_setting(cp) -> None:  # noqa: ANN001
    expected = DisplacementField(FIELD).coefficients
    with backend("cupy"):
        coefficients = DisplacementField(_proxy(FIELD)).coefficients
    _on_gpu(cp, coefficients, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_a_field_is_inverted_in_the_setting(cp, lazy: bool) -> None:  # noqa: ANN001
    expected = DisplacementField(FIELD).inverse().data
    data = _proxy(FIELD) if lazy else FIELD
    with backend("cupy"):
        inverse = DisplacementField(data).inverse().data
        _on_gpu(cp, inverse, expected)


def test_a_cupy_field_is_inverted_on_the_gpu(cp) -> None:  # noqa: ANN001
    expected = DisplacementField(FIELD).inverse().data
    inverse = DisplacementField(cp.asarray(FIELD)).inverse().data
    _on_gpu(cp, inverse, expected)


def _warped(field: object) -> SingleScaleImage:
    # A grid whose voxels are displaced by a field.
    return SingleScaleImage(np.zeros_like(VOLUME), [DisplacementField(field)])


@pytest.mark.parametrize("lazy_image", [False, True])
@pytest.mark.parametrize("lazy_field", [False, True])
def test_a_reslice_through_a_field_follows_the_setting(
    cp,  # noqa: ANN001
    lazy_image: bool,
    lazy_field: bool,
) -> None:
    expected = SingleScaleImage(VOLUME).reslice(_warped(FIELD)).data
    image = SingleScaleImage(_proxy(VOLUME) if lazy_image else VOLUME)
    target = _warped(_proxy(FIELD) if lazy_field else FIELD)
    with backend("cupy"):
        _on_gpu(cp, image.reslice(target).data, expected)


def test_a_lazy_field_meets_a_cupy_image(cp) -> None:  # noqa: ANN001
    # Under the default setting, the lazy field joins the image on the GPU.
    expected = SingleScaleImage(VOLUME).reslice(_warped(FIELD)).data
    image = SingleScaleImage(cp.asarray(VOLUME))
    moved = image.reslice(_warped(_proxy(FIELD)))
    _on_gpu(cp, moved.data, expected)


def test_a_cupy_field_is_composed_with_a_lazy_field(cp) -> None:  # noqa: ANN001
    # Under the default setting, the lazy field joins the field on the GPU.
    second = FIELD[::-1].copy()
    expected = (DisplacementField(FIELD) @ DisplacementField(second)).compute()
    first = DisplacementField(cp.asarray(FIELD))
    field = (first @ DisplacementField(_proxy(second))).compute().field
    _on_gpu(cp, field, expected.field)
    field = (
        (
            DisplacementField(_proxy(FIELD))
            @ DisplacementField(cp.asarray(second))
        )
        .compute()
        .field
    )
    _on_gpu(cp, field, expected.field)


def test_the_coefficients_of_a_cupy_field_stay_on_the_gpu(cp) -> None:  # noqa: ANN001
    expected = DisplacementField(FIELD).coefficients
    _on_gpu(cp, DisplacementField(cp.asarray(FIELD)).coefficients, expected)


def test_a_proxy_is_read_before_it_meets_the_gpu(cp) -> None:  # noqa: ANN001
    # Under the default setting, a lazy array is read and moved to the
    # GPU of the arrays that it is combined with.
    from brainhops._core.backends import to_backend_of, to_promoted_backend

    gpu = cp.asarray(VOLUME)
    moved = to_promoted_backend(gpu, _proxy(VOLUME), np.eye(3))
    assert all(isinstance(x, cp.ndarray) for x in moved)
    _on_gpu(cp, to_backend_of(_proxy(VOLUME), gpu), VOLUME)
    converted = to_promoted_backend(gpu, _proxy(VOLUME), dtype=np.float64)
    assert all(x.dtype == np.float64 for x in converted)


def test_to_device_reads_a_proxy_onto_the_device(cp) -> None:  # noqa: ANN001
    from brainhops._core.backends import to_device

    _on_gpu(cp, to_device(_proxy(VOLUME), "cuda"), VOLUME)
    on_cpu = to_device(_proxy(VOLUME), "cpu")
    assert isinstance(on_cpu, np.ndarray)
    np.testing.assert_array_equal(on_cpu, VOLUME)
    assert to_device(None, "cuda") is None
