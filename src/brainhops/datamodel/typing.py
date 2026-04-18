import typing_extensions as _tx

from brainhops.struct import Frozen, NoInit, NoRepr
from brainhops.struct.converters import numpy_typing as npt


T = _tx.TypeVar("T")
Const = _tx.Annotated[T, Frozen(), NoInit()]
HiddenConst = _tx.Annotated[Const[T], NoRepr()]
ArrayProtocol = npt.ArrayProtocol
