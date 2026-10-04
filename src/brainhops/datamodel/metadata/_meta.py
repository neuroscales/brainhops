"""The metaclass that passes the class keywords of `Metadata` to it."""

# externals
import typing_extensions as tx

# internals
from ..base import DataModelBase


class _MetadataMeta(type(DataModelBase)):
    """
    Passes the `supports=` and `lazy=` class keywords of a
    `Metadata` subclass to `Metadata._declare` and `Metadata._finish`.

    A class hook cannot read them: `bagof` refuses class keywords it does
    not know, and builds the fields of a class before
    `__init_subclass__` runs. So the keywords are popped here, the
    namespace is completed (`_declare`) before `bagof` reads it, and the
    class is finished (`_finish`) once built. The logic lives on
    `Metadata`; this is only the adaptor.
    """

    def __new__(
        metacls,
        name: str,
        bases: tx.Tuple[type, ...],
        namespace: tx.Dict[str, tx.Any],
        supports: tx.Any = None,
        lazy: tx.Optional[tx.Iterable[str]] = None,
        **kwargs: tx.Any,
    ) -> type:
        parent = next((b for b in bases if isinstance(b, metacls)), None)
        if parent is None:
            # `Metadata` itself, whose capabilities are in its body.
            if (supports, lazy) != (None, None):
                raise TypeError(
                    f"{name}: supports= and lazy= are for Metadata subclasses."
                )
            return super().__new__(metacls, name, bases, namespace, **kwargs)
        if supports is not None:
            parent._declare(name, namespace, supports)
        cls = super().__new__(metacls, name, bases, namespace, **kwargs)
        if "__magic_discard__" not in name:
            # Not one of the transient classes `bagof` builds.
            cls._finish(lazy)
        return cls
