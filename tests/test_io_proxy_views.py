"""Tests of writing an image whose lazy array is an index of its file.

Indexing the lazy array of a file returns an indexed proxy, which
selects a part of the file. A writer that copies the stored bytes of a
lazy array as they are must not copy those of an indexed proxy, since
they hold the whole file.
"""

import warnings

import numpy as np
import pytest
from bagof.magic import replace

import brainhops.io as io
from brainhops._core.backends import backend
from brainhops._core.proxy import ArrayProxy
from brainhops.datamodel.images import SingleScaleImage

VOLUME = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
PLANE = np.arange(5 * 6, dtype=np.uint8).reshape(5, 6) * 7

CASES = [
    ("a.mif", VOLUME, None),
    ("a.nrrd", VOLUME, None),
    ("a+orig.HEAD", VOLUME, None),
    ("a.tif", VOLUME, "tifffile"),
    ("a.zarr", VOLUME, "abczarr"),
    ("a.png", PLANE, "PIL"),
    ("a.jp2", PLANE, "PIL"),
]


@pytest.mark.parametrize("name, data, module", CASES)
def test_a_view_is_written_as_its_values(
    tmp_path,  # noqa: ANN001
    name: str,
    data: np.ndarray,
    module: str,
) -> None:
    # A flipped view has the shape of the file but other values, so a
    # writer that copied the stored bytes would write the original file.
    if module is not None:
        pytest.importorskip(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.save(SingleScaleImage(data=data), tmp_path / name)
        image = io.images.load(tmp_path / name)
        assert isinstance(image.rawdata, ArrayProxy)
        view = image.rawdata[::-1]
        assert view.root is image.rawdata.root
        assert view is not image.rawdata
        flipped = replace(image, rawdata=view)
        io.save(flipped, tmp_path / ("b" + name[1:]))
        with backend("numpy"):
            expected = np.asarray(flipped.data)
            original = np.asarray(image.data)
            written = np.asarray(
                io.images.load(tmp_path / ("b" + name[1:])).data
            )
    assert not np.array_equal(expected, original)
    np.testing.assert_array_equal(written, expected)
