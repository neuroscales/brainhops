# stdlib
import importlib
import importlib.metadata
import importlib.util
import sys

# dependencies
import typing_extensions as tx

# Each optional dependency: the short alias it is read under, its
# module, and its availability flag. Any of the three spellings looks it
# up (see `__getattr__`).
_DEPENDENCIES = (
    # ---- I/O ---------------------------------------------------------
    ("nb", "nibabel", "HAS_NIBABEL"),
    ("h5", "h5py", "HAS_H5PY"),
    ("abczarr", "abczarr", "HAS_ABCZARR"),
    ("pil", "PIL", "HAS_PILLOW"),
    ("tifffile", "tifffile", "HAS_TIFFFILE"),
    ("openslide", "openslide", "HAS_OPENSLIDE"),
    # ---- units -------------------------------------------------------
    ("pint", "pint", "HAS_PINT"),
    # ---- backends ----------------------------------------------------
    ("np", "numpy", "HAS_NUMPY"),
    ("cp", "cupy", "HAS_CUPY"),
    ("dk", "dask", "HAS_DASK"),
    ("da", "dask.array", "HAS_DASK_ARRAY"),
    ("sp", "scipy", "HAS_SCIPY"),
    ("npndi", "scipy.ndimage", "HAS_SCIPY_NDIMAGE"),
    ("cpndi", "cupyx.scipy.ndimage", "HAS_CUPY_NDIMAGE"),
    # The dask backend's ndimage functions are brainhops' own, and need
    # nothing but dask.
    ("dkndi", "brainhops._core.dask_ndimage", "HAS_DASK_NDIMAGE"),
)

_LAZY_NAMES = tuple(name for names in _DEPENDENCIES for name in names)

# The top-level modules that make a submodule importable. A flag is
# answered from the specs of top-level modules, without importing them,
# and finding the spec of `a.b` would import `a`.
_INSTALLED_WITH = {
    "dask.array": ("dask", "numpy"),  # dask's "array" extra
    "scipy.ndimage": ("scipy",),
    "cupyx.scipy.ndimage": ("cupy", "cupyx"),  # cupyx ships with cupy
    "brainhops._core.dask_ndimage": ("dask", "numpy"),
}


def __getattr__(name: str) -> tx.Any:
    # A flag says whether a dependency is installed, which is asked at
    # import time (to decide which formats exist): it is answered without
    # importing the dependency, which is what makes `import brainhops`
    # slow. An installed module may still fail to import (a binding whose
    # native library is missing, a broken install): the failure surfaces
    # where the module is first used, which reads it through its alias
    # and finds `None`.
    for alias, qualname, flag in _DEPENDENCIES:
        if name == flag:
            modules = _INSTALLED_WITH.get(qualname, (qualname,))
            available = globals()[flag] = all(map(_find_spec, modules))
            return available
        if name in (alias, qualname):
            return _lazy_import(globals(), name, qualname, alias)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The backends of abczarr's own drivers, by top-level module, with the
# distribution whose major version the driver needs at least.
_ABCZARR_BACKENDS = {
    "zarr": ("zarr", 2),
    "tensorstore": None,
    "zarrista": None,
}


def has_abczarr_driver() -> bool:
    """Whether abczarr is installed together with at least one backend driver.

    abczarr reads and writes Zarr through a driver, one of zarr-python,
    TensorStore, or zarrista. The core installs none of them, so abczarr
    being importable is not enough on its own. This returns `True` only when
    abczarr is present and at least one driver is available to it.

    Until abczarr is imported, this tells from whether the backend of one
    of its drivers is installed, without importing either: importing
    abczarr imports a driver, which takes about a second. Once it is
    imported, abczarr is asked, so that it counts the drivers registered
    by other packages and those whose backend fails to import.
    """
    if not __getattr__("HAS_ABCZARR"):
        return False
    if "abczarr" not in sys.modules:
        return any(
            _find_spec(module) and _has_version(requirement)
            for module, requirement in _ABCZARR_BACKENDS.items()
        )
    abczarr = __getattr__("abczarr")
    if abczarr is None:
        return False
    try:
        return bool(abczarr.available_drivers())
    except Exception:
        return False


def _has_version(requirement: tx.Optional[tx.Tuple[str, int]]) -> bool:
    """Whether a distribution's major version is at least the one given,
    or `True` when that cannot be told from its metadata."""
    if requirement is None:
        return True
    dist, major = requirement
    try:
        version = importlib.metadata.version(dist)
        return int(version.split(".")[0]) >= major
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return True


class _LazyType(type):
    """
    The metaclass of `lazy_type`: a class that stands for a class of an
    optional dependency, and imports it the first time it is used.
    """

    _module: str
    _attribute: str

    def resolve(cls) -> type:
        """The class it stands for, imported if it was not yet."""
        resolved = cls.__dict__.get("_resolved")
        if resolved is None:
            resolved = importlib.import_module(cls._module)
            for name in cls._attribute.split("."):
                resolved = getattr(resolved, name)
            type.__setattr__(cls, "_resolved", resolved)
        return resolved

    def __instancecheck__(cls, obj: tx.Any) -> bool:
        return isinstance(obj, cls.resolve())

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, cls.resolve())

    def __call__(cls, *args: tx.Any, **kwargs: tx.Any) -> tx.Any:
        return cls.resolve()(*args, **kwargs)

    def __repr__(cls) -> str:
        return f"{cls._module}.{cls._attribute}"


_LAZY_TYPES: tx.Dict[str, type] = {}


def lazy_type(name: str) -> type:
    """
    A class of an optional dependency, imported when it is first used.

    The fields of a format are annotated with the classes of the
    dependency that reads it (`h5py.File`, `nibabel.Nifti1Image`), from
    which their converters are built as the class is created. Naming the
    class itself imports the dependency with the format's module, which
    is imported to sniff any file. This stands for it instead: it is a
    class, so it makes a type hint, and it imports the dependency the
    first time a value is checked against it or built with it, that is
    the first time the format is used.

    Parameters
    ----------
    name : str
        The module and the class in it, as `"module:Class"`.
    """
    lazy = _LAZY_TYPES.get(name)
    if lazy is None:
        module, attribute = name.split(":")
        lazy = _LAZY_TYPES[name] = _LazyType(
            attribute.rsplit(".", 1)[-1],
            (),
            {
                "_module": module,
                "_attribute": attribute,
                "__module__": module,
                "__qualname__": attribute,
            },
        )
    return lazy


def __dir__() -> tx.List[str]:
    """
    List the lazily importable names alongside the usual ones.

    Without this, `dir()` and tab-completion show only what has already
    been imported, so a dependency that nobody has touched yet looks as
    though it does not exist.
    """
    return sorted(set(globals()) | set(_LAZY_NAMES))


def _find_spec(qualname: str) -> bool:
    """Whether a top-level module is installed, without importing it."""
    try:
        return importlib.util.find_spec(qualname) is not None
    except (ImportError, ValueError):
        # ValueError: a module already in `sys.modules` without a spec.
        return False


def _lazy_import(
    namespace: tx.Dict[str, tx.Any],
    query: str,
    qualname: str,
    shortname: str,
) -> tx.Any:
    """
    Lazily import a module and store it in the given namespace.

    This function is used in this module's `__getattr__` to lazily
    import optional dependencies.

    Parameters
    ----------
    namespace : dict
        The namespace in which to store the imported module.
        Generally, this is `globals()`.
    query : str
        The user query, which triggers the import: the module's
        qualified name, or its short alias.
    qualname : str
        The qualified name to use to import the module.
    shortname : str
        The short name to use to store the module in the namespace.

    Returns
    -------
    module : module
        The queried module.
        The queried module is set to `None` if its import fails,
        rather than raising an `ImportError`.
    """
    qualnames = qualname.split(".")
    rootname = qualnames[0]
    # Both must start as None: if the *root* import fails there is no
    # `root` to record, and referencing an unassigned local would raise
    # `UnboundLocalError` -- for exactly the uninstalled dependencies
    # this module exists to report on.
    root = leaf = None
    try:
        for i in range(len(qualnames)):
            name = ".".join(qualnames[: i + 1])
            leaf = importlib.import_module(name)
            if i == 0:
                root = leaf
    except (ImportError, OSError):
        # OSError: a binding whose native library is missing (OpenSlide).
        leaf = None

    # Only the root and the caller's chosen short name are recorded. The
    # leaf name alone is ambiguous: `scipy.ndimage` and
    # `cupyx.scipy.ndimage` would both claim `ndimage`, and whichever
    # imported last would win.
    namespace[rootname] = root
    namespace[shortname] = leaf

    # The query is one of the names just written, except for a fully
    # qualified submodule, which is the leaf itself.
    return namespace[query] if query in namespace else leaf
