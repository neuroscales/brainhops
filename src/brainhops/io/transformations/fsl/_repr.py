# dependencies
import typing_extensions as tx


def stored_repr(obj: tx.Any, names: tx.Sequence[str]) -> str:
    """A repr built from an object's stored fields.

    Each name in `names` is read with `getattr`, and a field whose value
    is `None` is left out. Reading these fields does not resolve the
    lazily computed transformation chain, so the repr of an incompletely
    specified transform does not raise.
    """
    parts = []
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            parts.append(f"{name}={value!r}")
    return f"{type(obj).__name__}({', '.join(parts)})"
