# stdlib
from pathlib import Path

# dependencies
import pytest

# internals
import brainhops.io as io
from brainhops.io.transformations.itk.tfm import TFMTransform

try:
    from brainhops.io.transformations.itk.h5 import H5Transform
except ImportError:  # h5py is optional
    H5Transform = None

data_dir = Path(__file__).parent / "data"

FILES_H5 = list(data_dir.glob("*.h5"))
FILES_TFM = list(data_dir.glob("*.tfm"))

requires_h5py = pytest.mark.skipif(
    H5Transform is None, reason="h5py is not installed"
)


@requires_h5py
@pytest.mark.parametrize("filename", FILES_H5)
@pytest.mark.parametrize("load", [True, False])
@pytest.mark.parametrize("keep_open", [True, False])
def test_read_h5(filename: str, load: bool, keep_open: bool) -> None:
    transform = H5Transform.from_file(filename, load=load, keep_open=keep_open)
    # trigger conversion
    transforms = transform.transformations  # noqa: F841


@pytest.mark.parametrize("filename", FILES_TFM)
def test_read_tfm(filename: str) -> None:
    transform = TFMTransform.from_file(filename)
    # trigger conversion
    transforms = transform.transformations  # noqa: F841


# ----------------------------------------------------------------------
#   REGISTRY / DISPATCH
# ----------------------------------------------------------------------
#
# The ITK readers register themselves with the shared parser registry, so
# the generic `io.load` and the scoped `io.transformations.load` both
# resolve an ITK file to its reader without being told the format.


@pytest.mark.parametrize("filename", FILES_TFM)
def test_tfm_is_dispatched(filename: str) -> None:
    assert io.transformations.sniff(filename) is TFMTransform
    assert io.sniff(filename) is TFMTransform
    assert type(io.transformations.load(filename)) is TFMTransform
    assert type(io.load(filename)) is TFMTransform


@requires_h5py
@pytest.mark.parametrize("filename", FILES_H5)
def test_h5_is_dispatched(filename: str) -> None:
    assert io.transformations.sniff(filename) is H5Transform
    assert io.sniff(filename) is H5Transform
    assert type(io.transformations.load(filename)) is H5Transform
    assert type(io.load(filename)) is H5Transform


def test_tfm_header_only_is_read_as_empty(tmp_path) -> None:  # noqa: ANN001
    """
    A `.tfm` file that carries only the version header has no transform
    block. `peekable_lines` drops the header comment and then yields its
    end sentinel, which is not a string, so `sniff_line` must return a
    numeric confidence rather than raise or claim the file.
    """
    header_only = tmp_path / "header_only.tfm"
    header_only.write_text("# Insight Transform File V1.0\n")

    assert TFMTransform.sniff_line("") == 0.0
    assert list(TFMTransform.from_file(header_only).transform_group) == []
    # The `.tfm` extension still routes it to the ITK reader, which reads
    # it as an empty transform.
    assert io.transformations.sniff(header_only) is TFMTransform
    assert type(io.transformations.load(header_only)) is TFMTransform
