# stdlib
from pathlib import Path

# dependencies
import pytest

# internals
from brainhops.io.transformations.itk.h5 import H5Transform
from brainhops.io.transformations.itk.tfm import TFMTransform

data_dir = Path(__file__).parent / "data"

FILES_H5 = list(data_dir.glob("*.h5"))
FILES_TFM = list(data_dir.glob("*.tfm"))


@pytest.mark.parametrize("filename", FILES_H5)
@pytest.mark.parametrize("load", [True, False])
@pytest.mark.parametrize("keep_open", [True, False])
def test_read_h5(filename, load, keep_open):
    transform = H5Transform.from_file(filename, load=load, keep_open=keep_open)
    transforms = H5Transform.transformations  # trigger conversion


@pytest.mark.parametrize("filename", FILES_TFM)
def test_read_tfm(filename):
    transform = TFMTransform.from_file(filename)
    transforms = TFMTransform.transformations  # trigger conversion
