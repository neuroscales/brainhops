"""
Lazy package namespaces ([PEP 562](https://peps.python.org/pep-0562/)).

A package `__init__` that imports all of its submodules makes importing
the package as slow as importing everything under it. `lazy_exports`
builds the module-level `__getattr__` and `__dir__` of a package from a
map of the names it exposes, so that each submodule is imported only
when one of its names is first looked up.

```python
__getattr__, __dir__ = lazy_exports(__name__, globals(), {
    "base": ".base",      # the submodule `base` itself
    "load": ".base",      # the name `load`, re-exported from `.base`
})
```
"""

__all__ = ["lazy_exports"]

# stdlib
import importlib

# dependencies
import typing_extensions as tx


def lazy_exports(
    package: str,
    namespace: tx.Dict[str, tx.Any],
    exports: tx.Mapping[str, str],
) -> tx.Tuple[tx.Callable[[str], tx.Any], tx.Callable[[], tx.List[str]]]:
    """
    Build the `__getattr__` and `__dir__` of a lazy package.

    Parameters
    ----------
    package : str
        The package's qualified name, generally `__name__`.
    namespace : dict
        The package's namespace, generally `globals()`. A re-exported
        name is stored there once resolved, so it is resolved only once.
    exports : dict[str, str]
        Each lazily exposed name, mapped to the module, relative to the
        package, that holds it. A name mapped to the module of the same
        name (`{"base": ".base"}`) is that submodule; any other name
        (`{"load": ".base"}`) is an attribute of its module.

    Returns
    -------
    __getattr__ : callable
        To be assigned to the package's `__getattr__`.
    __dir__ : callable
        To be assigned to the package's `__dir__`.
    """
    exports = dict(exports)

    def __getattr__(name: str) -> tx.Any:
        try:
            target = exports[name]
        except KeyError:
            raise AttributeError(
                f"module {package!r} has no attribute {name!r}"
            ) from None
        module = importlib.import_module(target, package)
        if target == "." + name:
            # Importing a submodule sets it on the package already.
            return module
        value = namespace[name] = getattr(module, name)
        return value

    def __dir__() -> tx.List[str]:
        return sorted(set(namespace) | set(exports))

    return __getattr__, __dir__
