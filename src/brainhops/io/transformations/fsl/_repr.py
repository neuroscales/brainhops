import typing_extensions as tx


def stored_repr(obj: tx.Any, names: tx.Sequence[str]) -> str:
    """Build a repr from the stored fields of an object.

    Each name in `names` is read with `getattr`, and fields whose value is
    `None` are omitted. Reading the fields does not resolve the lazily
    computed chain of transformations, so the repr of an incompletely
    specified transform does not raise.
    """
    parts = []
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            parts.append(f"{name}={value!r}")
    return f"{type(obj).__name__}({', '.join(parts)})"
