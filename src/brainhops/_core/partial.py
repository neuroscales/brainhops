import inspect
from functools import update_wrapper

from typing_extensions import Any, Callable, Dict, get_type_hints

from .compat import partial
from .properties import lazyproperty


class AnnotatedPartial:
    def __init__(self, func: Callable, *args, **keywords) -> None:
        self.partial = partial(func, *args, **keywords)
        update_wrapper(
            self,
            func,
            assigned=("__module__", "__name__", "__qualname__", "__doc__"),
        )

    @property
    def func(self) -> Callable:
        return self.partial.func

    @property
    def args(self) -> tuple:
        return self.partial.args

    @property
    def keywords(self) -> dict:
        return self.partial.keywords

    @lazyproperty
    def __annotations__(self) -> Dict[str, Any]:
        resolved_types = get_type_hints(self.func)
        try:
            sig = inspect.signature(self.partial)
        except (ValueError, TypeError):
            return {}

        new_annotations = {}
        for param_name in sig.parameters:
            if param_name in resolved_types:
                new_annotations[param_name] = resolved_types[param_name]

        if "return" in resolved_types:
            new_annotations["return"] = resolved_types["return"]

        return new_annotations

    def __call__(self, *call_args: Any, **call_kwargs: Any) -> Any:
        return self.partial(*call_args, **call_kwargs)

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        return annotated_partial(
            self.func, instance, *self.args, **self.keywords
        )

    def __repr__(self) -> str:
        return (
            f"<annotated_partial {self.func.__name__} "
            f"args={self.args} keywords={self.keywords}>"
        )


AnnotatedPartialBase = AnnotatedPartial


def annotated_partial(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """
    Create a partial function with preserved type annotations.
    """

    # Define local class with its own __call__ method
    class AnnotatedPartial(AnnotatedPartialBase):
        def __call__(self, *args, **kwargs) -> Any:
            return super().__call__(*args, **kwargs)

    # Wrap function
    wrapped = AnnotatedPartial(func, *args, **kwargs)

    # Assign annotations to __call__ method
    def __annotations__(self: Callable) -> Dict[str, Any]:
        return wrapped.__annotations__

    AnnotatedPartial.__call__.__annotations__ = property(__annotations__)

    return wrapped
