"""
The metaclass of `Metadata`: it reads the `supports=` class keyword of a
subclass, and sets its capabilities.
"""

# externals
import typing_extensions as tx
from bagof.magic import Factory, fields
from bagof.magic._options import Options

# internals
from brainhops._core.compat import own_annotations

from ..base import DataModelBase
from ._sentinel import ALL, UNSUPPORTED
from ._vocabulary import FIELDS, GROUPS

# The class keywords `bagof` reads (it ignores any other one silently).
_BAGOF_KEYWORDS = frozenset(Options._DEFAULTS) | {"on", "priority"}


class MetadataMeta(type(DataModelBase)):
    """
    Reads the `supports=` class keyword of a `Metadata` subclass.

    A class hook cannot read it: `bagof` builds the fields of a class
    before `__init_subclass__` runs, and ignores the class keywords it
    does not know. So the keyword is popped here, the namespace is
    completed (`_declare`) before `bagof` reads it, and the class is
    finished (`_finish`) once built; any other keyword `bagof` does not
    read is refused, rather than ignored. No metadata class overrides
    either step, so they live here rather than on `Metadata`.
    """

    def __new__(
        metacls,
        name: str,
        bases: tx.Tuple[type, ...],
        namespace: tx.Dict[str, tx.Any],
        supports: tx.Any = None,
        **kwargs: tx.Any,
    ) -> type:
        unknown = sorted(set(kwargs) - _BAGOF_KEYWORDS)
        if unknown:
            raise TypeError(
                f"{name}: unknown class keyword(s) {unknown}; a metadata "
                f"class takes supports= and the bagof options."
            )
        parent = next((b for b in bases if isinstance(b, metacls)), None)
        if parent is None:
            # `Metadata` itself, whose capabilities are in its body.
            if supports is not None:
                raise TypeError(
                    f"{name}: supports= is for Metadata subclasses."
                )
            return super().__new__(metacls, name, bases, namespace, **kwargs)
        if supports is not None:
            # `Metadata`: the last class of the MRO built by this metaclass.
            root = [b for b in parent.__mro__ if isinstance(b, metacls)][-1]
            metacls._declare(root, name, namespace, supports)
        cls = super().__new__(metacls, name, bases, namespace, **kwargs)
        metacls._read_raw_class(cls)
        if "__magic_discard__" not in name:
            # Not one of the transient classes `bagof` builds.
            cls._finish()
        return cls

    @staticmethod
    def _declare(
        root: type,
        name: str,
        namespace: tx.Dict[str, tx.Any],
        supports: tx.Union[str, tx.Iterable[tx.Any]],
    ) -> None:
        """
        Read `supports=` into the namespace of the class `name`, before
        `bagof` builds its fields: a supported field is declared again
        (in case a parent did not support it), and any other one is
        declared with the default `UNSUPPORTED`, which is exactly what
        writing `x: T = UNSUPPORTED` in the class body does. A field
        written out in the body is left as it is. `root` is `Metadata`,
        whose declarations are copied.
        """
        supported = _supported_names(name, supports)
        annotations = own_annotations(namespace)
        for field in FIELDS:
            if field in annotations or field in namespace:
                continue
            hint = _declared_hint(root, field)
            if field in supported:
                annotations[field] = hint
                if field != "extra":  # `extra` has a factory
                    namespace[field] = None
            else:
                annotations[field] = _without_factory(hint)
                namespace[field] = UNSUPPORTED
        # On Python 3.14+ the body's annotations come as a lazy annotate
        # function; it is replaced by the plain dict, as up to 3.13.
        for key in ("__annotate__", "__annotate_func__"):
            namespace.pop(key, None)
        namespace["__annotations__"] = annotations

    @staticmethod
    def _read_raw_class(cls: type) -> None:
        """
        Set `_raw_class` from the type argument of a generic base, as in
        `class NiftiMetadata(FileBasedMetadata[nb.Nifti1Header])`:
        `type(None)` for `[None]`. A class without such a base inherits
        the value of its parent.
        """
        for base in cls.__dict__.get("__orig_bases__", ()):
            origin = tx.get_origin(base)
            args = tx.get_args(base)
            if not isinstance(origin, MetadataMeta) or len(args) != 1:
                continue
            (arg,) = args
            if isinstance(arg, tx.TypeVar):
                continue
            cls._raw_class = type(None) if arg is None else arg

    def _finish(cls) -> None:
        """
        Set the capabilities of a class once `bagof` has built it: its
        `supported_fields` (the fields whose default is not
        `UNSUPPORTED`) and `unsupported_fields`.
        """
        supported = frozenset(
            field.name
            for field in fields(cls)
            if field.name in FIELDS and field.default is not UNSUPPORTED
        )
        cls.supported_fields = supported
        cls.unsupported_fields = frozenset(FIELDS) - supported


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _supported_names(
    name: str, supports: tx.Union[str, tx.Iterable[tx.Any]]
) -> tx.FrozenSet[str]:
    """The field names a `supports=` declaration stands for."""
    if isinstance(supports, str):
        if supports != ALL:
            raise TypeError(
                f"{name}: supports= takes a sequence of field names or "
                f"vocabulary groups, or ALL, not {supports!r}."
            )
        return frozenset(FIELDS)
    names: tx.Set[str] = set()
    for item in supports:
        if not isinstance(item, type):
            names.add(item)
        elif item in GROUPS:
            names.update(GROUPS[item])
        else:
            raise TypeError(
                f"{name}: supports= names {item.__name__}, which is not a "
                f"vocabulary group; expected one of "
                f"{[g.__name__ for g in GROUPS]}."
            )
    unknown = names - frozenset(FIELDS)
    if unknown:
        raise TypeError(
            f"{name}: supports= names {sorted(unknown)}, which are not "
            f"vocabulary fields; expected some of {sorted(FIELDS)}."
        )
    return frozenset(names)


def _declared_hint(root: type, name: str) -> tx.Any:
    """The annotation of a field as `Metadata` (`root`, or one of its
    groups) declares it."""
    return next(
        own_annotations(klass)[name]
        for klass in root.__mro__
        if name in own_annotations(klass)
    )


def _without_factory(hint: tx.Any) -> tx.Any:
    """An annotation without its default factory (that of `extra`), for a
    field whose default becomes `UNSUPPORTED`. Its converters stay: they
    all let `UNSUPPORTED` through."""
    if tx.get_origin(hint) is not tx.Annotated:
        return hint
    base, *extras = tx.get_args(hint)
    kept = [extra for extra in extras if not isinstance(extra, Factory)]
    return tx.Annotated[(base, *kept)] if kept else base
