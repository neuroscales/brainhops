"""Tests of the geometry and reslicing of 4-D (space and time) NIfTI images.

The affine applies to the spatial axes, while time is mapped by its own
spacing and origin.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")
ndi = pytest.importorskip("scipy.ndimage")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel._transformations.compute import (  # noqa: E402
    separable,
)
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)

SHAPE = (6, 7, 5, 4)


def _save(path: object, data: np.ndarray, affine: np.ndarray) -> str:
    nii = nb.Nifti1Image(data, affine)
    nii.header.set_xyzt_units("mm", "sec")
    nb.save(nii, str(path))
    return str(path)


def _save_timed(
    path: object, affine: np.ndarray, tr: float, toffset: float
) -> str:
    nii = nb.Nifti1Image(np.zeros(SHAPE, dtype="float32"), affine)
    nii.header.set_xyzt_units("mm", "sec")
    nii.header.set_zooms(nii.header.get_zooms()[:3] + (tr,))
    nii.header["toffset"] = toffset
    nb.save(nii, str(path))
    return str(path)


def test_a_4d_geometry_is_a_spatial_and_a_temporal_subspace(
    tmp_path,  # noqa: ANN001
) -> None:
    """The affine maps (x, y, z), and TR then toffset map the frame index."""
    affine = _rotation(0.4)
    affine[:3, -1] += [3.0, -2.0, 1.0]
    img = io.load(_save_timed(tmp_path / "bold.nii", affine, 2.5, 0.75))

    xform = img.transformation
    assert isinstance(xform, Sequence)
    spatial, temporal = xform.transformations
    assert isinstance(spatial, SubspaceTransformation)
    assert isinstance(temporal, SubspaceTransformation)
    assert list(spatial.input_axes) == list(spatial.output_axes) == [0, 1, 2]
    assert list(temporal.input_axes) == list(temporal.output_axes) == [3]
    assert np.allclose(spatial.transformation.matrix, affine[:3])

    scaling, translation = temporal.transformation.transformations
    assert isinstance(scaling, Scaling) and isinstance(
        translation, Translation
    )
    assert np.allclose(scaling.scale, [2.5])
    assert np.allclose(translation.translation, [0.75])
    # frame index, then time since the first frame, then world time
    assert str(scaling.input.axes[0].unit) == "index"
    assert str(scaling.output.axes[0].unit) == "second"
    assert translation.input.axes == scaling.output.axes
    assert str(translation.output.axes[0].unit) == "second"

    # Between the steps, space is in world units and time is still frames.
    middle = spatial.output
    assert middle.axes[:3] == xform.output.axes[:3]
    assert middle.axes[3] == xform.input.axes[3]
    assert temporal.input is middle
    assert str(xform.output.axes[3].unit) == "second"

    # As one affine, the product is block-diagonal.
    matrix = np.asarray(xform.to(Affine).matrix)
    expected = np.zeros((4, 5))
    expected[:3, [0, 1, 2, 4]] = affine[:3]
    expected[3, 3:] = [2.5, 0.75]
    assert np.allclose(matrix, expected)


def test_a_3d_geometry_is_still_a_plain_affine(tmp_path) -> None:  # noqa: ANN001
    nii = nb.Nifti1Image(np.zeros(SHAPE[:3], dtype="float32"), _rotation(1))
    nb.save(nii, str(tmp_path / "t1.nii"))
    img = io.load(str(tmp_path / "t1.nii"))
    assert type(img.transformation) is Affine
    assert np.allclose(img.transformation.matrix, _rotation(1)[:3])


def test_the_geometry_is_decoded_once_per_header(tmp_path) -> None:  # noqa: ANN001
    """The same header gives the same decoded transformations."""
    img = io.load(_save_timed(tmp_path / "bold.nii", np.eye(4), 2.0, 0.0))
    first = img.transformations
    assert img.transformations is not first
    assert all(a is b for a, b in zip(img.transformations, first))
    assert img.geometry.transformation is img.transformation

    img.header = img.header.copy()
    assert all(a is not b for a, b in zip(img.transformations, first))


def _rotation(angle: float) -> np.ndarray:
    """A rotation about z around the centre of the volume."""
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.eye(4)
    rotation[:2, :2] = [[c, -s], [s, c]]
    center = np.eye(4)
    center[:3, -1] = -(np.asarray(SHAPE[:3]) - 1) / 2
    return np.linalg.inv(center) @ rotation @ center


def _expected(
    data: np.ndarray, source: np.ndarray, target: np.ndarray
) -> tuple:
    """Reslice each frame with scipy and report which voxels sample inside."""
    ijk = np.stack(
        np.meshgrid(*[np.arange(n) for n in SHAPE[:3]], indexing="ij"), -1
    )
    vox2vox = np.linalg.inv(source) @ target
    coords = ijk @ vox2vox[:3, :3].T + vox2vox[:3, -1]
    coords = np.moveaxis(coords, -1, 0)
    expected = np.stack(
        [
            ndi.map_coordinates(data[..., t], coords, order=1)
            for t in range(SHAPE[-1])
        ],
        -1,
    )
    inside = np.all(
        [(coords[d] >= 0) & (coords[d] <= SHAPE[d] - 1) for d in range(3)],
        axis=0,
    )
    return expected, inside


def test_a_4d_nifti_resliced_onto_its_own_grid_is_unchanged(
    tmp_path,  # noqa: ANN001
) -> None:
    data = np.arange(np.prod(SHAPE), dtype="float32").reshape(SHAPE)
    path = _save(tmp_path / "bold.nii.gz", data, np.eye(4))

    resliced = io.load(path).reslice()

    assert resliced.shape == SHAPE
    assert np.array_equal(np.asarray(resliced), data)


def test_a_4d_nifti_with_numpy_data_resliced_onto_its_own_geometry(
    tmp_path,  # noqa: ANN001
) -> None:
    """Reslicing an image onto its own geometry returns the same data."""
    data = np.arange(np.prod(SHAPE), dtype="float32").reshape(SHAPE)
    path = _save(tmp_path / "bold.nii.gz", data, _rotation(0.3))
    img = io.load(path)

    resliced = SingleScaleImage(
        data=np.asarray(img.data), transformations=img.transformations
    ).reslice(img.geometry)

    assert np.array_equal(np.asarray(resliced), data)


@pytest.mark.parametrize("backend", ["dask", "numpy"])
def test_a_4d_nifti_is_resliced_through_its_spatial_affine(
    tmp_path,  # noqa: ANN001
    backend: str,
) -> None:
    """The rotation couples x and y, so it is not 1-D separable."""
    rng = np.random.default_rng(0)
    data = rng.random(SHAPE).astype("float32")
    source = _rotation(np.pi / 7)
    source[:3, :3] *= [1.5, 1.0, 2.0]
    target = np.diag([1.25, 1.25, 2.0, 1.0])
    img = io.load(_save(tmp_path / "source.nii.gz", data, source))
    geometry = io.load(_save(tmp_path / "target.nii.gz", data, target))
    if backend == "numpy":
        img = SingleScaleImage(
            data=np.asarray(img.data), transformations=img.transformations
        )
    else:
        pytest.importorskip("dask.array")
        assert type(img.data).__module__.startswith("dask")

    resliced = img.reslice(geometry.geometry)

    expected, inside = _expected(data, source, target)
    got = np.asarray(resliced)
    assert got.shape == SHAPE
    assert inside.any() and not inside.all()
    assert np.allclose(got[inside], expected[inside], atol=1e-5)
    # The output has the target geometry, with time spacing 1 and origin 0.
    matrix = np.asarray(resliced.transformation.to(Affine).matrix)
    assert matrix.shape == (4, 5)
    assert np.allclose(matrix[:3, [0, 1, 2, 4]], target[:3])
    assert np.array_equal(matrix[:3, 3], np.zeros(3))
    assert np.array_equal(matrix[3], [0, 0, 0, 1, 0])


def _plans(monkeypatch) -> list:  # noqa: ANN001
    """Record the plan of each separable reslice, keyed by grid axes."""
    plans = []
    plan = separable._plan

    def spy(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
        steps = plan(*args, **kwargs)
        plans.append(
            None
            if steps is None
            else {tuple(step["G"]): step["kind"] for step in steps}
        )
        return steps

    monkeypatch.setattr(separable, "_plan", spy)
    return plans


def test_a_4d_reslice_is_separable_with_time_on_its_own(
    tmp_path,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
) -> None:
    """Space and time are planned as separate groups, never as one pull."""
    rng = np.random.default_rng(1)
    data = rng.random(SHAPE).astype("float32")
    source = _rotation(np.pi / 5)
    target = np.diag([1.0, 1.0, 1.5, 1.0])
    img = io.load(_save(tmp_path / "source.nii.gz", data, source))
    geometry = io.load(_save(tmp_path / "target.nii.gz", data, target))
    img = SingleScaleImage(
        data=np.asarray(img.data), transformations=img.transformations
    )
    plans = _plans(monkeypatch)

    def monolithic(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("the monolithic pull was used")

    monkeypatch.setattr(separable, "pull", monolithic)

    resliced = img.reslice(geometry.geometry)

    assert plans == [{(0, 1): "pull", (2,): "matrix", (3,): "gather"}]
    expected, inside = _expected(data, source, target)
    got = np.asarray(resliced)
    assert np.allclose(got[inside], expected[inside], atol=1e-5)


def test_a_4d_identity_reslice_returns_a_view(
    tmp_path,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
) -> None:
    """The geometry cancels by identity, and numpy returns a view."""
    data = np.arange(np.prod(SHAPE), dtype="float32").reshape(SHAPE)
    img = io.load(_save(tmp_path / "bold.nii.gz", data, _rotation(0.3)))
    assert img.transformation is img.transformation
    numpy = SingleScaleImage(
        data=np.asarray(img.data), transformations=img.transformations
    )
    plans = _plans(monkeypatch)

    for resliced in (numpy.reslice(), numpy.reslice(img.geometry)):
        assert np.shares_memory(np.asarray(resliced.data), numpy.data)
        assert np.array_equal(np.asarray(resliced), data)
    gathers = {(d,): "gather" for d in range(4)}
    assert plans == [gathers, gathers]


def _save_untimed(
    path: object, data: np.ndarray, tr: float, forms: bool = True
) -> str:
    """Write a 4-D NIfTI file whose TR is missing (0 or NaN)."""
    nii = nb.Nifti1Image(data, _rotation(0.3))
    nii.header.set_xyzt_units("mm", "sec")
    nii.header["pixdim"][4] = tr
    if not forms:
        nii.header.set_sform(None, code=0)
        nii.header.set_qform(None, code=0)
    nb.save(nii, str(path))
    return str(path)


@pytest.mark.parametrize("tr", [0.0, np.nan, np.inf])
def test_a_4d_geometry_without_a_repetition_time_is_spatial_only(
    tmp_path,  # noqa: ANN001
    tr: float,
) -> None:
    """Without a TR, only space is mapped and time counts frames."""
    data = np.zeros(SHAPE, dtype="float32")
    img = io.load(_save_untimed(tmp_path / "bold.nii", data, tr))

    xform = img.transformation
    assert isinstance(xform, Sequence)
    (spatial,) = xform.transformations
    assert isinstance(spatial, SubspaceTransformation)
    assert list(spatial.input_axes) == list(spatial.output_axes) == [0, 1, 2]
    assert np.allclose(spatial.transformation.matrix, _rotation(0.3)[:3])
    assert spatial.output is xform.output
    for transformation in img.transformations:
        assert str(transformation.output.axes[3].unit) == "index"
        assert transformation.output.axes[3].type == "time"
    physical = img.transformations[0]
    assert np.allclose(physical.scale, [1.0, 1.0, 1.0, 1.0])

    matrix = np.asarray(xform.to(Affine).matrix)
    expected = np.zeros((4, 5))
    expected[:3, [0, 1, 2, 4]] = _rotation(0.3)[:3]
    expected[3, 3] = 1.0
    assert np.allclose(matrix, expected)


def test_a_4d_identity_reslice_without_a_repetition_time_is_a_view(
    tmp_path,  # noqa: ANN001
) -> None:
    data = np.arange(np.prod(SHAPE), dtype="float32").reshape(SHAPE)
    img = io.load(_save_untimed(tmp_path / "bold.nii.gz", data, 0.0))

    assert np.array_equal(np.asarray(img.reslice()), data)
    numpy = SingleScaleImage(
        data=np.asarray(img.data), transformations=img.transformations
    )
    for resliced in (numpy.reslice(), numpy.reslice(img.geometry)):
        assert np.shares_memory(np.asarray(resliced.data), numpy.data)
        assert np.array_equal(np.asarray(resliced), data)


@pytest.mark.parametrize("forms", [True, False])
def test_a_4d_image_without_a_repetition_time_round_trips(
    tmp_path,  # noqa: ANN001
    forms: bool,
) -> None:
    """Saving keeps the TR missing and leaves toffset alone."""
    data = np.arange(np.prod(SHAPE), dtype="float32").reshape(SHAPE)
    # With neither form, the preferred transformation is the scaling.
    path = _save_untimed(tmp_path / "bold.nii", data, 0.0, forms=forms)

    img = io.load(path)
    img.save(tmp_path / "copy.nii")

    header = nb.load(str(tmp_path / "copy.nii")).header
    assert header["pixdim"][4] == 0
    assert header["toffset"] == 0
    assert np.allclose(
        header.get_zooms()[:3], nb.load(path).header.get_zooms()[:3]
    )
    if forms:
        assert np.allclose(header.get_best_affine(), _rotation(0.3))
    copy = io.load(str(tmp_path / "copy.nii"))
    (spatial,) = copy.transformation.transformations
    assert list(spatial.input_axes) == [0, 1, 2]
    assert np.array_equal(np.asarray(copy), data)
