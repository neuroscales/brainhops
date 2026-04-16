import typing_extensions as _tx

from brainhops.struct import Frozen, NoInit, NoRepr, Repr


T = _tx.TypeVar("T")
Const = _tx.Annotated[T, Frozen(), NoInit()]
HiddenConst = NoRepr[Const[T]]