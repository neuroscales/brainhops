"""Tests for the `labels` property of images.

The property says whether an image is a label map. It is unknown,
false, true, or a table of names, and it is derived from the `intent`
metadata field unless a value is given.
"""

import numpy as np
import pytest

from brainhops.datamodel.images import (
    Label,
    MultiScaleImage,
    SingleScaleImage,
)
from brainhops.datamodel.metadata import Metadata

DATA = np.zeros((2, 3, 4), dtype="int16")
TABLE = {1: Label("left", (255, 0, 0, 255)), 2: Label("right")}


@pytest.mark.parametrize("value", [None, False, True, TABLE])
def test_a_given_value_is_returned(value) -> None:  # noqa: ANN001
    assert SingleScaleImage(DATA, labels=value).labels == value


@pytest.mark.parametrize(
    ("intent", "expected"),
    [("label", True), ("neuroname", True), ("t test", False), (None, None)],
)
def test_the_value_is_derived_from_the_intent(intent, expected) -> None:  # noqa: ANN001
    image = SingleScaleImage(DATA, metadata=Metadata(intent=intent))
    assert image.labels is expected


def test_a_given_value_wins_over_the_intent() -> None:
    metadata = Metadata(intent="label")
    image = SingleScaleImage(DATA, metadata=metadata, labels=False)
    assert image.labels is False
    image = SingleScaleImage(DATA, metadata=Metadata(intent="t test"))
    image.labels = TABLE
    assert image.labels == TABLE
    image.labels = None
    assert image.labels is False


def test_the_value_survives_indexing_and_lazy_transformations() -> None:
    image = SingleScaleImage(DATA, labels=TABLE)
    assert image[0].labels == TABLE
    assert image(image.transformation).labels == TABLE


def test_a_pyramid_inherits_from_its_finest_level() -> None:
    fine = SingleScaleImage(DATA, labels=TABLE)
    coarse = SingleScaleImage(DATA[::2], labels=False)
    assert MultiScaleImage([fine, coarse]).labels == TABLE
    assert MultiScaleImage([fine], labels=False).labels is False
    assert MultiScaleImage([SingleScaleImage(DATA)]).labels is None


def test_the_label_type_compares_and_prints() -> None:
    assert Label("a") == Label("a", None)
    assert Label("a", (1, 2, 3, 4)) != Label("a")
    expected = "Label(name='a', color=(1, 2, 3, 4))"
    assert repr(Label("a", (1, 2, 3, 4))) == expected


def test_a_nifti_label_map_round_trips(tmp_path) -> None:  # noqa: ANN001
    nb = pytest.importorskip("nibabel")
    import brainhops.io as io
    from brainhops.io.images.nifti import NiftiImage

    img = nb.Nifti1Image(DATA, np.eye(4))
    img.header.set_intent("label")
    nb.save(img, str(tmp_path / "seg.nii"))
    assert io.images.load(tmp_path / "seg.nii").labels is True

    target = tmp_path / "out.nii"
    image = NiftiImage.from_instance(SingleScaleImage(DATA, labels=True))
    image.save(target)
    assert int(nb.load(str(target)).header["intent_code"]) == 1002
    assert io.images.load(target).labels is True


# --- defaults for label maps ---------------------------------------


def _labelled(labels):  # noqa: ANN001, ANN202
    data = np.zeros((8, 8, 8), dtype="int16")
    data[:4] = 3
    data[4:] = 7
    return SingleScaleImage(data, labels=labels)


def test_reslice_of_a_label_map_is_nearest_by_default() -> None:
    from brainhops.datamodel.transformations import Translation

    image = _labelled(True)
    shift = Translation([0.5, 0.0, 0.0]) @ image.transformation
    resliced = image.reslice(shift)
    assert set(np.unique(np.asarray(resliced))) <= {3, 7}
    assert resliced.labels is True
    linear = image.reslice(shift, degree=1, dtype="float32")
    assert 5.0 in set(np.unique(np.asarray(linear)))


@pytest.mark.parametrize("labels", [False, None])
def test_reslice_of_another_image_is_linear_by_default(labels) -> None:  # noqa: ANN001
    from brainhops.datamodel.transformations import Translation

    image = _labelled(labels)
    shift = Translation([0.5, 0.0, 0.0]) @ image.transformation
    resliced = image.reslice(shift, dtype="float32")
    assert 5.0 in set(np.unique(np.asarray(resliced)))


def test_a_label_pyramid_uses_the_mode_by_default() -> None:
    # Along the first axis, three voxels of 7 follow one voxel of 3, so
    # each coarse voxel of factor 2 covers either 7 and 7 or 7 and 3.
    rows = np.where(np.arange(8) % 4 == 3, 3, 7).astype("int16")
    data = np.broadcast_to(rows[:, None, None], (8, 8, 8)).copy()
    base = SingleScaleImage(data, labels=True)
    pyramid = base.to(MultiScaleImage, nscales=2, min_size=1)
    coarse = np.asarray(pyramid.images[1].data)
    assert coarse.dtype == data.dtype
    assert set(np.unique(coarse)) <= {3, 7}
    assert pyramid.labels is True
    mean = base.to(MultiScaleImage, nscales=2, min_size=1, mode="mean")
    assert 5 in set(np.unique(np.asarray(mean.images[1].data)))
