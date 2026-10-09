"""Tests of the indexing helpers and the segment writer of npfileobj."""

import io

import numpy as np
import pytest

from brainhops._ext.npfileobj import indexing
from brainhops._ext.npfileobj.indexing import (
    compose_index,
    expand_index,
    is_fullslice,
    oob_slice,
)
from brainhops._ext.npfileobj.readwrite import write_segments


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the indexing module behave as if PyTorch were not installed."""
    monkeypatch.setattr(indexing, "torch", None)


def test_oob_slice_newaxis_is_a_full_slice() -> None:
    # An out-of-bounds slice that adds a new axis counts as a full
    # slice, and one that does not add an axis does not.
    assert is_fullslice(oob_slice(newaxis=True), 1)
    assert not is_fullslice(oob_slice(), 1)


def test_compose_new_axis_parent_with_oob_child() -> None:
    # The parent dimension is a new axis.
    result = compose_index((None,), (oob_slice(),), (3,))
    assert len(result) == 2
    assert isinstance(result[0], oob_slice)
    assert result[0].newaxis
    assert result[1] == slice(None)


def test_compose_slice_parent_with_oob_child() -> None:
    # The parent dimension is a slice.
    result = compose_index((slice(None),), (oob_slice(),), (3,))
    assert len(result) == 1
    assert isinstance(result[0], oob_slice)
    assert not result[0].newaxis


@pytest.mark.usefixtures("no_torch")
@pytest.mark.parametrize(
    "scalar", [0, -3, np.array(0), np.array(-3, dtype=np.int32)]
)
def test_expand_index_accepts_integers_without_torch(scalar: object) -> None:
    # Without PyTorch, an integer index, whether a Python integer or a
    # NumPy scalar, becomes a plain non-negative Python integer.
    result = expand_index((scalar,), (3,))
    assert result == (0,)
    assert type(result[0]) is int


@pytest.mark.usefixtures("no_torch")
def test_index_helpers_with_integers_without_torch() -> None:
    # The helpers that call `expand_index` also accept integers.
    assert is_fullslice((0, slice(None)), (1, 4)) == (True, True)
    assert compose_index((slice(1, None), 2), (0,), (3, 4)) == (1, 2)


@pytest.mark.usefixtures("no_torch")
def test_expand_index_rejects_non_scalar_arrays_without_torch() -> None:
    with pytest.raises(TypeError):
        expand_index((np.array([0, 1]),), (3,))


class _ShortWrite(io.BytesIO):
    """File object that writes at most two bytes per call."""

    def write(self, b: bytes) -> int:
        return super().write(b[:2])


def test_write_segments_single_segment_message() -> None:
    # A short write to a single segment raises an error whose message
    # gives the expected and the written number of bytes.
    with pytest.raises(ValueError) as excinfo:
        write_segments(_ShortWrite(), [(0, 4)], b"abcd")
    assert excinfo.value.args == ("Expected to write 4 bytes but wrote 2.",)
