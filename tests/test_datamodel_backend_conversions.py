"""Tests of the to_* and as_* methods of images and transformations."""

import os
import warnings

import numpy as np
import pytest

import brainhops.io as io
from brainhops.backends import backend
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
    # None keeps loaded arrays.
    image = SingleScaleImage(VOLUME)
    assert image.as_backend().data is VOLUME
    with backend("dask"):
        assert isinstance(image.as_backend().data, da.Array)
        assert image.as_backend(None).data is VOLUME


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
