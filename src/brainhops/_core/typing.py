"""Type hints shared across the package.

!!! warning "Internal module"
    `brainhops._core.typing` is an internal module. It is documented only
    to explain the type hints that appear in the public interface, and it
    should not be imported or relied upon. It may change or disappear
    without notice.
"""

__all__ = [
    "Const",
    "HiddenConst",
    "ArrayProtocol",
    "ArrayLike",
    "safe_get_origin",
    "npscalar",
    "npvector",
    "npmatrix",
    "art",
    "npt",
    "cpt",
    "dkt",
    "is_instance_or_subclass",
]
import typing_extensions as tx
from bagof.core.magic import safe_get_origin
from bagof.hints import array as art
from bagof.hints import cupy as cpt
from bagof.hints import dask as dkt
from bagof.hints import numpy as npt
from bagof.hints.array import ArrayLike, ArrayProtocol
from bagof.hints.numpy import dtype, ndarray
from bagof.magic import Frozen, NoInit, NoRepr

T = tx.TypeVar("T")


def is_instance_or_subclass(
    obj: tx.Any, cls: tx.Union[type, tx.Tuple[type, ...]]
) -> bool:
    """Return whether an object is an instance or a subclass of a class.

    Classes are tested with `issubclass` and other objects with
    `isinstance`, so the function returns `False` for unrelated classes and
    for objects such as `None`.
    """
    if isinstance(obj, type):
        return issubclass(obj, cls)
    return isinstance(obj, cls)


Deactivated = tx.ClassVar
"""Marker that deactivates an inherited field.

A field redeclared with this marker in a subclass becomes inert. The
field disappears from the signature of `__init__` and from the generated
`__repr__`, but it is still listed by `fields(cls)`, with `init=False`
and `var=False`.
"""

Derived = tx.ClassVar
"""Marker for an inherited field that a subclass derives from other data.

The marker behaves like [`Deactivated`][]. It is used when a field that is
stored by a base class becomes virtual in a subclass.
"""

Const = tx.Annotated[T, Frozen(), NoInit()]
"""Field that is frozen and cannot be set through the constructor."""

HiddenConst = tx.Annotated[Const[T], NoRepr()]
"""Constant field that is also hidden from the generated `__repr__`."""

npscalar = tx.Union[T, ndarray[tx.Tuple[()], dtype[T]]]
"""Scalar, given either as a plain value or as a 0-d array."""

npvector = ndarray[tx.Tuple[int], dtype[T]]
"""One-dimensional array."""

npmatrix = ndarray[tx.Tuple[int, int], dtype[T]]
"""Two-dimensional array."""
