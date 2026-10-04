"""The metaclass that passes the class keywords of `Metadata` to it."""

# externals
import typing_extensions as tx
from bagof.magic._options import Options

# internals
from ..base import DataModelBase

# The class keywords `bagof` reads (it ignores any other one silently).
_BAGOF_KEYWORDS = frozenset(Options._DEFAULTS) | {"on", "priority"}


class _MetadataMeta(type(DataModelBase)):
    """
    Passes the `supports=` and `lazy=` class keywords of a
    `Metadata` subclass to `Metadata._declare` and `Metadata._finish`.

    A class hook cannot read them: `bagof` builds the fields of a class
    before `__init_subclass__` runs, and ignores the class keywords it
    does not know. So the keywords are popped here, the namespace is
    completed (`_declare`) before `bagof` reads it, and the class is
    finished (`_finish`) once built; any other keyword `bagof` does not
    read is refused, rather than ignored. The logic lives on `Metadata`;
    this is only the adaptor.
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
        unknown = sorted(set(kwargs) - _BAGOF_KEYWORDS)
        if unknown:
            hint = (
                " (`derived=` was removed: a format's `_geometry` says "
                "which fields the data model gives)"
                if "derived" in unknown
                else ""
            )
            raise TypeError(
                f"{name}: unknown class keyword(s) {unknown}; a metadata "
                f"class takes supports=, lazy= and the bagof options{hint}."
            )
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
