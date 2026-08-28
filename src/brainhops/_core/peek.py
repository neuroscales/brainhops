# stdlib
from collections.abc import Iterator

# dependencies
import typing_extensions as _tx

# typing
T = _tx.TypeVar('T')


class EMPTY_TYPE:

    def __new__(cls) -> object:
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return '<empty>'

    def __repr__(self) -> str:
        return 'EMPTY()'


EMPTY = EMPTY_TYPE()


class peekable(Iterator, _tx.Generic[T]):
    """A peekable iterator."""

    EMPTY = EMPTY_TYPE()

    def __init__(self, iterable: _tx.Iterable[T]) -> None:
        if not hasattr(iterable, '__next__'):
            # make an iterator (with a state)
            iterable = iter(iterable)
        self._iterator: _tx.Iterator[T] = iterable
        self._peeked: T | EMPTY_TYPE = EMPTY

    def peek(self, preproc: bool = True, valid: bool = True) -> _tx.Optional[T]:
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

    def iter(self, preproc: bool = True, valid: bool = True) -> _tx.Iterator[T]:
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
        return T

    def _next(self, preproc: bool = True, valid: bool = True) -> T | EMPTY_TYPE:
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
    """A peekable iterator over lines of text."""

    def __init__(self, lines: _tx.Iterable[str], comment: str | None = "#") -> None:
        super().__init__(lines)
        self.comment = comment

    @classmethod
    def is_valid(cls, item: _tx.Optional[str]) -> bool:
        return bool(item)

    def preproc(self, line: _tx.Optional[str]) -> _tx.Optional[str]:
        if line is None:
            return None
        line = line.rstrip("\r\n")
        if self.comment:
            line = line.split(self.comment, 1)[0]
        line = line.strip()
        return line or None
