
__all__ = ["SlotsBase", "rebuild_cls", "slots"]
import copy as copy_
import inspect
import types as _t
import typing_extensions as _tx

from .constants import MISSING


def _get_origin(type: _tx.Any) -> _tx.Any:
    origin = _tx.get_origin(type)
    if origin is None:
        return type
    return origin


def _update_func_cell_for__class__(
    f: _tx.Optional[_tx.Callable], oldcls: type, newcls: type
) -> bool:
    # Returns True if we update a cell, else False.
    if f is None:
        # f will be None in the case of a property where not all of
        # fget, fset, and fdel are used.  Nothing to do in that case.
        return False
    try:
        idx = f.__code__.co_freevars.index("__class__")
    except ValueError:
        # This function doesn't reference __class__, so nothing to do.
        return False
    # Fix the cell to point to the new class, if it's already pointing
    # at the old class.  I'm not convinced that the "is oldcls" test
    # is needed, but other than performance can't hurt.
    closure = f.__closure__[idx]
    if closure.cell_contents is oldcls:
        closure.cell_contents = newcls
        return True
    return False


def rebuild_cls(
    cls: type,
    type_func: _tx.Callable[[str, tuple[type, ...], dict], type] = type,
) -> type:
    namespace = dict(cls.__dict__)

    # Instert attributes that are not in the class dict, but handled
    # by the low level Cpython.
    namespace.setdefault("__qualname__", cls.__qualname__)

    # Remove __dict__ and __weakref__ from the class dict.
    namespace.pop('__dict__', None)
    namespace.pop('__weakref__', None)
    # qualname  = namespace.pop('__qualname__', None)

    # Make a new class with the same name, bases, and namespace.
    newcls = type_func(cls.__name__, cls.__bases__, namespace)

    # Restore the original qualname.
    # if qualname is not None:
    #     newcls.__qualname__ = qualname

    # Fix up any closures which reference __class__.  This is used to
    # fix zero argument super so that it points to the correct class
    # (the newly created one, which we're returning) and not the
    # original class.  We can break out of this loop as soon as we
    # make an update, since all closures for a class will share a
    # given cell.
    for member in newcls.__dict__.values():
        # If this is a wrapped function, unwrap it.
        member = inspect.unwrap(member)

        if isinstance(member, _t.FunctionType):
            if _update_func_cell_for__class__(member, cls, newcls):
                break
        elif isinstance(member, property):
            if (_update_func_cell_for__class__(member.fget, cls, newcls)
                or _update_func_cell_for__class__(member.fset, cls, newcls)
                or _update_func_cell_for__class__(member.fdel, cls, newcls)):
                break

    return newcls


@_tx.overload
def slots(cls: type, *_slots: _tx.Tuple[str]) -> type: ...

@_tx.overload
def slots(*_slots: _tx.Tuple[str]) -> _tx.Callable[[type], type]: ...


def slots(*aslots, **kwslots):

    if aslots and isinstance(aslots[0], type):
        cls, aslots = aslots[0], aslots[1:]
    else:
        return (lambda cls: slots(cls, *aslots, **kwslots))

    if kwslots:
        for slot in aslots:
            kwslots[slot] = None
        _slots = kwslots
    else:
        _slots = aslots

    def add_slots(name: str, bases: tuple[type, ...], namespace: dict) -> type:
        namespace['__slots__'] = _slots
        return type(name, bases, namespace)

    return rebuild_cls(cls, add_slots)


@slots
class SlotsBase:

    def __init__(self, **kwargs) -> None:
        for slot in self._slots():
            setattr(self, slot, kwargs.get(slot, MISSING))

    def __repr__(self) -> str:
        repr_slots = (slot for slot in self._slots()
                      if getattr(self, slot, MISSING) is not MISSING)
        params = (f"{slot}={getattr(self, slot)!r}" for slot in repr_slots)
        params = ", ".join(params)
        return f"{self.__class__.__name__}({params})"

    def __getattr__(self, name: str) -> _tx.Any:
        if name in self._slots():
            return MISSING
        raise AttributeError(
            f"{self.__class__.__name__!r} object has no attribute {name!r}"
        )

    @classmethod
    def _slots(cls) -> _tx.Iterator[str]:
        seen = dict()
        for cls in reversed(cls.__mro__):
            for slot in getattr(cls, '__slots__', ()):
                if slot not in seen:
                    seen[slot] = True
                    yield slot

    def update(self, options: _tx.Self) -> None:
        for slot in self._slots():
            if getattr(options, slot, MISSING) is not MISSING:
                setattr(self, slot, getattr(options, slot, MISSING))

    def setdefault(self, options: _tx.Self) -> None:
        for slot in self._slots():
            if getattr(self, slot, MISSING) is MISSING:
                setattr(self, slot, getattr(options, slot, MISSING))

    def copy(self) -> _tx.Self:
        return copy_.copy(self)

    def deepcopy(self) -> _tx.Self:
        return copy_.deepcopy(self)
