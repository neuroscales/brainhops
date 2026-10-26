"""Tests of the to_* and as_* methods of images and transformations."""

import os
import warnings

import numpy as np
import pytest

import brainhops.io as io
from brainhops._core.backends import backend
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    CartesianField,
    DisplacementField,
    Identity,
    Sequence,
    SubspaceTransformation,
)

da = pytest.importorskip("dask.array")
nb = pytest.importorskip("nibabel")

VOLUME = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
FIELD = np.linspace(0, 0.1, 4 * 5 * 6 * 3).reshape(4, 5, 6, 3)
MATRIX = np.eye(3, 4)


def test_to_methods_return_bare_arrays() -> None:
    # The to_* methods return the data and leave the object unchanged.
    image = SingleScaleImage(VOLUME)
    assert isinstance(image.to_dask(), da.Array)
    assert isinstance(image.to_numpy(), np.ndarray)
    assert image.data is VOLUME
    assert image.to_backend() is VOLUME
    assert isinstance(Affine(MATRIX).to_dask(), da.Array)
    assert Identity().to_numpy() is None


def test_as_methods_return_converted_objects() -> None:
    # The as_* methods return an object of the same class with its data
    # converted, and leave the object unchanged.
    image = SingleScaleImage(VOLUME, [Affine(MATRIX)])
    lazy = image.as_dask()
    assert type(lazy) is SingleScaleImage
    assert isinstance(lazy.data, da.Array)
    assert image.data is VOLUME
    back = lazy.as_numpy()
    np.testing.assert_array_equal(back.data, VOLUME)


def test_inplace_converts_the_object_itself() -> None:
    # With inplace=True, the object holds the converted data and is
    # returned.
    image = SingleScaleImage(VOLUME)
    assert image.as_dask(inplace=True) is image
    assert isinstance(image.data, da.Array)
    field = DisplacementField(FIELD)
    assert field.as_dask(inplace=True) is field
    assert isinstance(field.data, da.Array)


def test_small_parameters_stay_numpy() -> None:
    # An affine transformation holds no voxels, so as_* returns it
    # unchanged, and the matrices of an image stay NumPy arrays.
    affine = Affine(MATRIX)
    assert affine.as_dask() is affine
    image = SingleScaleImage(VOLUME, [affine]).as_dask()
    assert image.transformations[0] is affine


def test_composites_convert_their_members() -> None:
    # A sequence converts its fields and keeps its affine members, and an
    # inverse converts its forward.
    affine = Affine(MATRIX)
    sequence = Sequence([affine, DisplacementField(FIELD)]).as_dask()
    assert sequence.transformations[0] is affine
    assert isinstance(sequence.transformations[1].data, da.Array)
    inverse = DisplacementField(FIELD).inverse().as_dask()
    assert isinstance(inverse.forward.data, da.Array)
    with pytest.raises(TypeError):
        Sequence([affine]).to_numpy()


def test_meta_transformations_convert_their_inner_ones() -> None:
    # A subspace converts its inner transformation, and a bijection both
    # of its sides.
    field = DisplacementField(FIELD)
    subspace = SubspaceTransformation(field, [0, 1, 2], [0, 1, 2])
    assert isinstance(subspace.as_dask().transformation.data, da.Array)
    bijection = Bijection(field, DisplacementField(FIELD)).as_dask()
    assert isinstance(bijection.forward.data, da.Array)
    assert isinstance(bijection.backward.data, da.Array)
    affine = SubspaceTransformation(Affine(MATRIX))
    assert affine.as_dask() is affine


def test_a_grid_is_kept() -> None:
    # A grid stores only its shape, so as_* returns it unchanged.
    grid = CartesianField(shape=(2, 3))
    assert grid.as_dask() is grid


def test_inplace_converts_the_members_of_a_sequence() -> None:
    # With inplace=True, the sequence itself holds the converted members.
    sequence = Sequence([DisplacementField(FIELD)])
    assert sequence.as_dask(inplace=True) is sequence
    assert isinstance(sequence.transformations[0].data, da.Array)


def test_a_pyramid_converts_every_level() -> None:
    # as_* converts every level, and to_* gives the finest one.
    pyramid = MultiScaleImage(
        [SingleScaleImage(VOLUME), SingleScaleImage(VOLUME[::2, ::2, ::2])]
    )
    lazy = pyramid.as_dask()
    assert all(isinstance(level.data, da.Array) for level in lazy.images)
    np.testing.assert_array_equal(pyramid.to_numpy(), VOLUME)


def test_as_backend_follows_the_setting() -> None:
    # Without an argument, as_backend follows the current setting, and
    # "native" keeps loaded arrays.
    image = SingleScaleImage(VOLUME)
    assert image.as_backend().data is VOLUME
    with backend("dask"):
        converted = image.as_backend()
        kept = image.as_backend("native")
    assert isinstance(converted.data, da.Array)
    assert kept.data is VOLUME


def test_objects_read_from_files(tmp_path) -> None:  # noqa: ANN001
    # An image and a displacement field read from NIfTI files convert, and
    # are written back with their values.
    nb.save(nb.Nifti1Image(VOLUME, np.eye(4)), tmp_path / "image.nii.gz")
    image = io.images.load(tmp_path / "image.nii.gz")
    lazy = image.as_dask()
    assert type(lazy) is type(image)
    assert isinstance(lazy.data, da.Array)
    io.save(lazy, tmp_path / "copy.nii.gz")
    copy = io.images.load(tmp_path / "copy.nii.gz")
    np.testing.assert_array_equal(np.asarray(copy.data), VOLUME)

    values = FIELD.astype(np.float32).reshape(4, 5, 6, 1, 3)
    header = nb.Nifti1Image(values, np.eye(4))
    header.header.set_intent("vector")
    nb.save(header, tmp_path / "disp.nii.gz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = io.transformations.load(
            tmp_path / "disp.nii.gz", hint="displacements"
        )
        assert isinstance(field.to_dask(), da.Array)
        lazy = field.as_dask()
        assert isinstance(lazy.data, da.Array)
        io.save(lazy, tmp_path / "out.nii.gz")
        back = io.transformations.load(
            os.fspath(tmp_path / "out.nii.gz"), hint="displacements"
        )
    np.testing.assert_allclose(np.asarray(back.data), np.asarray(field.data))


def test_data_follows_the_setting_at_each_read(tmp_path) -> None:  # noqa: ANN001
    # data converts what the object holds with the setting in effect, and
    # the object keeps holding the same array.
    image = SingleScaleImage(VOLUME)
    with backend("dask"):
        assert isinstance(image.data, da.Array)
    assert image.data is VOLUME
    field = DisplacementField(FIELD)
    with backend("dask"):
        assert isinstance(field.data, da.Array)
    assert field.data is FIELD

    # An image read from a file caches its data on first access, and a
    # later setting still applies to each read.
    nb.save(nb.Nifti1Image(VOLUME, np.eye(4)), tmp_path / "image.nii.gz")
    image = io.images.load(tmp_path / "image.nii.gz")
    cached = image.data
    assert isinstance(cached, np.ndarray)
    with backend("dask"):
        lazy = image.data
    assert isinstance(lazy, da.Array)
    np.testing.assert_array_equal(lazy.compute(), cached)
    assert image.data is cached


def test_conversions_read_what_the_object_holds() -> None:
    # Under a setting, as_* and to_* convert the stored array once, so
    # that as_dask stores a Dask array and to_numpy gives the stored
    # values.
    image = SingleScaleImage(VOLUME)
    with backend("dask"):
        lazy = image.as_dask()
        values = image.to_numpy()
        assert np.asarray(image) is VOLUME
        assert image.shape == VOLUME.shape
    assert isinstance(lazy.data, da.Array)
    assert values is VOLUME


def test_nested_conversions_forward_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from brainhops.datamodel import _backends

    original = _backends.to_backend
    devices = []

    def tracked(data, target=None, *, device=None):  # noqa: ANN001, ANN202
        # A read of `data` converts with the setting and names no target,
        # so only the conversions that name one are recorded.
        if target is not None:
            devices.append(device)
        return original(data, target, device=device)

    # The conversion methods call the function that their module imports.
    monkeypatch.setattr(_backends, "to_backend", tracked)
    field = DisplacementField(FIELD)
    wrapped = SubspaceTransformation(field, [0, 1, 2], [0, 1, 2])
    sequence = Sequence([wrapped])
    lazy = sequence.as_dask(device="cpu")
    assert isinstance(lazy[0].transformation.dataobj, da.Array)
    assert devices and all(device == "cpu" for device in devices)


def test_explicit_cpu_conversion_keeps_dask_lazy() -> None:
    from dask.callbacks import Callback

    from brainhops import backends

    lazy = da.from_array(VOLUME, chunks=2)
    executed = []
    with Callback(posttask=lambda *args: executed.append(True)):
        assert backends.to_device(lazy, "cpu") is lazy
        assert SingleScaleImage(lazy).to_dask(device="cpu") is lazy
    assert not executed


class _CountingArray:
    """An array that is not NumPy, CuPy or Dask, and counts its reads."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim
        self.reads = 0

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        self.reads += 1
        return np.asarray(self.values, dtype=dtype)


def test_an_assigned_lazy_array_is_loaded_once() -> None:
    # `dataobj` keeps the array as it was assigned, and `data` loads it at
    # the first read and returns the same loaded array afterwards.
    lazy = _CountingArray(VOLUME)
    image = SingleScaleImage(lazy)
    assert image.dataobj is lazy
    assert lazy.reads == 0
    first = image.data
    assert isinstance(first, np.ndarray)
    assert image.data is first
    assert lazy.reads == 1
    np.testing.assert_array_equal(first, VOLUME)


def test_a_loaded_array_is_data_itself() -> None:
    # A NumPy array is stored and read as it is, so a change made to it
    # in place is kept.
    values = VOLUME.copy()
    image = SingleScaleImage(values)
    assert image.data is values
    assert image.dataobj is values


def test_assigning_dataobj_is_assigning_data() -> None:
    # Both store the array and clear the loaded array that `data` caches.
    image = SingleScaleImage(_CountingArray(VOLUME))
    image.data  # noqa: B018
    lazy = _CountingArray(VOLUME + 1)
    image.dataobj = lazy
    assert image.dataobj is lazy
    np.testing.assert_array_equal(image.data, VOLUME + 1)
    image.data = VOLUME
    assert image.dataobj is VOLUME
    assert image.data is VOLUME
