import typing_extensions as _tx

from brainhops.struct import Frozen, NoInit, NoRepr


T = _tx.TypeVar("T")
Const = _tx.Annotated[T, Frozen(), NoInit()]
ConstHidden = NoRepr[Const[T]]