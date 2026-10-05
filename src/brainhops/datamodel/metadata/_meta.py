"""
The metaclass of `Metadata`: it reads the `supports=` and `lazy=` class
keywords of a subclass, and sets its capabilities.
"""

# stdlib
import copy
import inspect

# externals
import typing_extensions as tx
from bagof.magic import Factory, fields
from bagof.magic._options import Options

# internals
from brainhops._core.compat import own_annotations
from brainhops._core.fields import LazyField

from ..base import DataModelBase
from ._sentinel import ALL, UNSUPPORTED
from ._vocabulary import FIELDS, GROUPS

# The class keywords `bagof` reads (it ignores any other one silently).
_BAGOF_KEYWORDS = frozenset(Options._DEFAULTS) | {"on", "priority"}


class MetadataMeta(type(DataModelBase)):
    """
    Reads the `supports=` and `lazy=` class keywords of a `Metadata`
    subclass.

    A class hook cannot read them: `bagof` builds the fields of a class
    before `__init_subclass__` runs, and ignores the class keywords it
    does not know. So the keywords are popped here, the namespace is
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
            # `Metadata`: the last class of the MRO built by this metaclass.
            root = [b for b in parent.__mro__ if isinstance(b, metacls)][-1]
            metacls._declare(root, name, namespace, supports)
        cls = super().__new__(metacls, name, bases, namespace, **kwargs)
        if "__magic_discard__" not in name:
            # Not one of the transient classes `bagof` builds.
            cls._finish(lazy)
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

    def _finish(cls, lazy: tx.Optional[tx.Iterable[str]] = None) -> None:
        """
        Set the capabilities of a class once `bagof` has built it: its
        `supported_fields` (the fields whose default is not
        `UNSUPPORTED`), `unsupported_fields` and `lazy_fields` (`lazy=`,
        or the inherited ones it still supports), and a `LazyField`
        descriptor per lazy field.
        """
        supported = frozenset(
            field.name
            for field in fields(cls)
            if field.name in FIELDS and field.default is not UNSUPPORTED
        )
        cls.supported_fields = supported
        cls.unsupported_fields = frozenset(FIELDS) - supported
        if lazy is None:
            cls.lazy_fields = cls.lazy_fields & supported
        else:
            cls.lazy_fields = frozenset(lazy)
            wrong = cls.lazy_fields - (supported - {"extra"})
            if wrong:
                raise TypeError(
                    f"{cls.__name__} declares lazy={sorted(wrong)}, which "
                    f"are not vocabulary fields it supports."
                )
        # The descriptors are installed once the class is built, for two
        # reasons. Nothing of the field is lost: `bagof` keeps the `Field`
        # in the field table of the class (`fields(cls)`), and only the
        # class attribute is replaced. That attribute is the plain default
        # that `bagof` set, and the descriptor returns the same default.
        # And a descriptor placed in the namespace before the build would
        # be read by `bagof` as the default value of the field. The
        # descriptor is not a field, by design: it only changes how the
        # attribute is read.
        for field in fields(cls):
            # Installed on every class that has a lazy field: a subclass
            # that redeclares the field (`supports=`) hides its parent's.
            if field.name in cls.lazy_fields and not isinstance(
                inspect.getattr_static(cls, field.name), LazyField
            ):
                descriptor = LazyField(
                    field.name,
                    default=field.default,
                    prepare=_unsupported_as_none,
                    on_load=_join_snapshot,
                )
                setattr(cls, field.name, descriptor)


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


def _unsupported_as_none(value: tx.Any) -> tx.Any:
    return None if value is UNSUPPORTED else value


def _join_snapshot(obj: tx.Any, name: str, value: tx.Any) -> None:
    """A lazy field, once loaded, joins the read-time snapshot, as if it
    had been decoded with the rest."""
    snapshot = obj.__dict__.get("_snapshot")
    if value is not None and snapshot is not None:
        snapshot[name] = copy.deepcopy(value)
