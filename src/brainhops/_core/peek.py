from collections.abc import Iterator

import typing_extensions as tx

T = tx.TypeVar("T")


class EMPTY_TYPE:
    def __new__(cls) -> object:
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return "<empty>"

    def __repr__(self) -> str:
        return "EMPTY()"


EMPTY = EMPTY_TYPE()


class peekable(Iterator, tx.Generic[T]):
    """Iterator with a lookahead of one item.

    The `peek` method returns the next item without consuming it, and the
    `next` method consumes it. Subclasses may override two hooks: `preproc`
    transforms each item, and `is_valid` rejects items that are then
    skipped. Both hooks can be bypassed with the `preproc` and `valid`
    arguments of `peek`, `next` and `iter`. When the iterator is exhausted,
    `peek` returns the [`EMPTY`][] sentinel and `next` raises
    `StopIteration`.
    """

    EMPTY = EMPTY_TYPE()

    def __init__(self, iterable: tx.Iterable[T]) -> None:
        if not hasattr(iterable, "__next__"):
            iterable = iter(iterable)
        self._iterator: tx.Iterator[T] = iterable
        self._peeked: tx.Union[T, EMPTY_TYPE] = EMPTY

    def peek(
        self, preproc: bool = True, valid: bool = True
    ) -> tx.Union[T, EMPTY_TYPE]:
        if self._peeked is EMPTY:
            self._peeked = self._next(preproc=preproc, valid=valid)
        return self._peeked

    def next(self, preproc: bool = True, valid: bool = True) -> T:
        if self._peeked is not EMPTY:
            item, self._peeked = self._peeked, EMPTY
        else:
            item = self._next(preproc=preproc, valid=valid)
        if item is EMPTY:
            raise StopIteration
        return item

    def iter(self, preproc: bool = True, valid: bool = True) -> tx.Iterator[T]:
        while True:
            try:
                yield self.next(preproc=preproc, valid=valid)
            except StopIteration:
                return

    __next__ = next
    __iter__ = iter

    @classmethod
    def is_valid(cls, item: T) -> bool:
        return True

    @classmethod
    def preproc(cls, item: T) -> T:
        """Return the item unchanged, unless a subclass overrides it."""
        return item

    def _next(
        self, preproc: bool = True, valid: bool = True
    ) -> tx.Union[T, EMPTY_TYPE]:
        while True:
            item = next(self._iterator, EMPTY)
            if item is EMPTY:
                return EMPTY
            if preproc:
                item = self.preproc(item)
            if valid and not self.is_valid(item):
                continue
            return item


class peekable_lines(peekable[str]):
    """Peekable iterator over the lines of a text.

    Each line is preprocessed by removing its line terminator, cutting it at
    the comment marker and stripping the surrounding whitespace. The comment
    marker is `#` by default and is disabled by `comment=None`. Lines that
    are empty after preprocessing are skipped.
    """

    def __init__(
        self, lines: tx.Iterable[str], comment: tx.Optional[str] = "#"
    ) -> None:
        super().__init__(lines)
        self.comment = comment

    @classmethod
    def is_valid(cls, item: tx.Optional[str]) -> bool:
        return bool(item)

    def preproc(self, line: tx.Optional[str]) -> tx.Optional[str]:
        if line is None:
            return None
        line = line.rstrip("\r\n")
        if self.comment:
            line = line.split(self.comment, 1)[0]
        line = line.strip()
        return line or None
