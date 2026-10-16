"""Package attributes that are imported the first time they are used.

A package whose submodules are costly to import can still export their
public names. [`lazy_exports`][] builds the module-level `__getattr__`
and `__dir__` functions of the package (see PEP 562), which import the
submodule that defines a name the first time the name is accessed.
"""

__all__ = ["lazy_exports"]

import importlib
import sys

import typing_extensions as tx


def lazy_exports(
    package: str, exports: tx.Mapping[str, str]
) -> tx.Tuple[tx.Callable[[str], tx.Any], tx.Callable[[], tx.List[str]]]:
    """Build the `__getattr__` and `__dir__` functions of a package.

    The returned `__getattr__` imports the submodule that defines a name
    the first time the name is accessed, and stores the value in the
    package, so that later accesses do not call it again.

    !!! example
        ```python
        __getattr__, __dir__ = lazy_exports(
            __name__, {"FlirtTransform": "._xform"}
        )
        ```

    Parameters
    ----------
    package : str
        The name of the package, which is `__name__` in its `__init__`.
    exports : mapping of str to str
        The module that defines each exported name. A module that starts
        with a dot is relative to the package.

    Returns
    -------
    getattr : callable
        The `__getattr__` function of the package.
    dir : callable
        The `__dir__` function of the package.
    """
    exports = dict(exports)

    def __getattr__(name: str) -> tx.Any:  # noqa: N807
        if name not in exports:
            raise AttributeError(
                f"module {package!r} has no attribute {name!r}"
            )
        module = importlib.import_module(exports[name], package)
        value = getattr(module, name)
        setattr(sys.modules[package], name, value)
        return value

    def __dir__() -> tx.List[str]:  # noqa: N807
        return sorted(set(vars(sys.modules[package])) | set(exports))

    return __getattr__, __dir__
