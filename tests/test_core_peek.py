"""Tests of the lookahead iterator."""

from brainhops._core.peek import EMPTY, peekable


def test_a_plain_peekable_yields_the_items_unchanged() -> None:
    """The default `preproc` hook returns each item as it is."""
    assert list(peekable(iter([1, 2, 3]))) == [1, 2, 3]


def test_peek_does_not_consume_the_item() -> None:
    it = peekable([1, 2])
    assert it.peek() == 1
    assert it.next() == 1
    assert it.peek() == 2
    assert next(it) == 2
    assert it.peek() is EMPTY
