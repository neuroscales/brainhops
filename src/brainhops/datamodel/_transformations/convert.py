# dependencies
import typing_extensions as tx

# internals
from .errors import ConversionError
from .registries import Dispatcher

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation as _Transformation

Transformation: tx.TypeAlias = "_Transformation"
TransformationType: tx.TypeAlias = tx.Type[Transformation]
Key: tx.TypeAlias = tx.Tuple[TransformationType, TransformationType]
Converter: tx.TypeAlias = tx.Callable[[Transformation], Transformation]
FROM = tx.TypeVar("FROM", bound=Transformation, default=Transformation)
TO = tx.TypeVar("TO", bound=Transformation, default=Transformation)


class Convert(Dispatcher[Key, Converter]):
    """
    A registry of converters that convert one transformation type to another.

    The registry is a [`Dispatcher`][] that maps a pair of transformation
    types to a converter function. When being called with a pair of
    arguments, it dispatches them to the most appropriate converter function.
    """

    @classmethod
    def key_from_func(cls, func: Converter) -> Key:
        hints = tx.get_type_hints(func)
        inp, *_, out = hints.values()
        return inp, out

    @classmethod
    def key_from_args(_, x: FROM, cls: tx.Type[TO], **kwargs) -> Key:
        return type(x), cls

    def __apply__(
        self, func: Converter, x: FROM, cls: tx.Type[TO], **kwargs
    ) -> TO:
        return func(x, **kwargs)

    def __error__(self, x: FROM, cls: tx.Type[TO], **kwargs) -> None:
        t1, t2 = self.key_from_args(x, cls)
        raise ConversionError(f"No converter found for: {t1} -> {t2}")

    def __post_check__(
        self, result: TO, x: FROM, cls: tx.Type[TO], **kwargs
    ) -> TO:
        t1, t2 = self.key_from_args(x, cls)
        if not isinstance(result, cls):
            raise ConversionError(
                f"No converter found for: {t1} -> {t2}. The nearest one "
                f"produced a {type(result).__name__}."
            )
        return result

    def __call__(self, x: FROM, cls: tx.Type[TO], **kwargs) -> TO:
        """Convert a transform to a different type.

        !!! note

            Raises [`ConversionError`][] when no converter applies, but
            also when the one that did apply produced something that is
            not of the requested type.

            The second check is not belt and braces: the dispatch below
            scores the *target* by type subclasing, which is finite
            whenever the converter produces a **supertype** of what was
            asked for.

            The catch-all same-type converter, declared
            `Transformation -> Transformation`, therefore matches every
            request, and without this check a conversion to a type
            nothing can produce would quietly hand back the original
            type instead of saying so.

        Parameters
        ----------
        x : Transformation
            The transform to convert.
        cls : type[Transformation]
            The type to convert to.

        Returns
        -------
        Transformation
            A transform of the requested type.

        Raises
        ------
        ConversionError
            When no converter applies, or when the one that did apply
            produced something that is not a `cls`.
        """
        return super().__call__(x, cls, **kwargs)


convert: Convert = Convert()
"""Public conversion function, that dipatches on its input types."""

converter = convert.register
"""Decorator to register a converter function."""

get_converter = convert.get
"""Get a converter function for a pair of transformation types."""
