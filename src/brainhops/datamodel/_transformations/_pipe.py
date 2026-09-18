import typing_extensions as tx

if tx.TYPE_CHECKING:
    from .base import Transformation


class _Pipe:
    # This is an idea to implement a pipe-like syntax `a |p> b`.
    # It's not very pythonic!

    def __init__(
        self,
        input: tx.Optional[Transformation] = None,
        side: tx.Optional[tx.Literal["L", "R"]] = None,
    ) -> None:
        self.input = input
        self.side = side

    def __ror__(
        self, other: Transformation
    ) -> tx.Union[tx.Self, Transformation]:
        if self.side == "R":
            return self.input(other)
        elif self.side is None:
            return _Pipe(input=other, side="L")
        else:
            raise SyntaxError("Invalid pipe syntax")

    def __gt__(
        self, other: Transformation
    ) -> tx.Union[tx.Self, Transformation]:
        if self.side == "L":
            return other(self.input)
        elif self.side is None:
            return _Pipe(input=other, side="R")
        else:
            raise SyntaxError("Invalid pipe syntax")


p = _Pipe()
