# stdlib
import importlib

# dependencies
import typing_extensions as tx

# Every spelling that resolves to the same dependency: a short alias, the
# qualified module name, and the availability flag. Named once so that
# `__getattr__` and `__dir__` cannot drift apart.

# ---- I/O -------------------------------------------------------------
_NIBABEL = ("nb", "nibabel", "HAS_NIBABEL")
_H5PY = ("h5", "h5py", "HAS_H5PY")
_ABCZARR = ("abczarr", "abczarr", "HAS_ABCZARR")

# ---- backends --------------------------------------------------------
_NUMPY = ("np", "numpy", "HAS_NUMPY")
_CUPY = ("cp", "cupy", "HAS_CUPY")
_DASK = ("dk", "dask", "HAS_DASK")
_DASK_ARRAY = ("da", "dask.array", "HAS_DASK_ARRAY")
_SCIPY = ("sp", "scipy", "HAS_SCIPY")
_SCIPY_NDIMAGE = ("npndi", "scipy.ndimage", "HAS_SCIPY_NDIMAGE")
_CUPY_NDIMAGE = ("cpndi", "cupyx.scipy.ndimage", "HAS_CUPY_NDIMAGE")
_DASK_NDIMAGE = ("dkndi", "dask_image.ndinterp", "HAS_DASK_NDIMAGE")

_LAZY_NAMES = (
    _NIBABEL
    + _H5PY
    + _ABCZARR
    + _NUMPY
    + _CUPY
    + _DASK
    + _DASK_ARRAY
    + _SCIPY
    + _SCIPY_NDIMAGE
    + _CUPY_NDIMAGE
    + _DASK_NDIMAGE
)


def __getattr__(name: str) -> tx.Any:

    # ==================================================================
    #
    #                                  I/O
    #
    # ==================================================================

    if name in _NIBABEL:
        return _lazy_import(globals(), name, "nibabel", "nb")

    if name in _H5PY:
        return _lazy_import(globals(), name, "h5py", "h5")

    if name in _ABCZARR:
        return _lazy_import(globals(), name, "abczarr", "abczarr")

    # ==================================================================
    #
    #                              BACKENDS
    #
    # ==================================================================

    if name in _NUMPY:
        return _lazy_import(globals(), name, "numpy", "np")

    if name in _CUPY:
        return _lazy_import(globals(), name, "cupy", "cp")

    if name in _DASK:
        return _lazy_import(globals(), name, "dask", "dk")

    if name in _DASK_ARRAY:
        return _lazy_import(globals(), name, "dask.array", "da")

    if name in _SCIPY:
        return _lazy_import(globals(), name, "scipy", "sp")

    if name in _SCIPY_NDIMAGE:
        return _lazy_import(globals(), name, "scipy.ndimage", "npndi")

    if name in _CUPY_NDIMAGE:
        return _lazy_import(
            globals(), name, "cupyx.scipy.ndimage", "cpndi", "CUPY_NDIMAGE"
        )

    if name in _DASK_NDIMAGE:
        return _lazy_import(
            globals(), name, "dask_image.ndinterp", "dkndi", "DASK_NDIMAGE"
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def has_abczarr_driver() -> bool:
    """Whether abczarr is installed together with at least one backend driver.

    abczarr reads and writes Zarr through a driver, one of zarr-python,
    TensorStore, or zarrista. The core installs none of them, so abczarr
    being importable is not enough on its own. This returns `True` only when
    abczarr is present and at least one driver is available to it.
    """
    abczarr = __getattr__("abczarr")
    if abczarr is None:
        return False
    try:
        return bool(abczarr.available_drivers())
    except Exception:
        return False


def __dir__() -> tx.List[str]:
    """
    List the lazily importable names alongside the usual ones.

    Without this, `dir()` and tab-completion show only what has already
    been imported, so a dependency that nobody has touched yet looks as
    though it does not exist.
    """
    return sorted(set(globals()) | set(_LAZY_NAMES))


def _lazy_import(
    namespace: tx.Dict[str, tx.Any],
    query: str,
    qualname: tx.Optional[str] = None,
    shortname: tx.Optional[str] = None,
    uppername: tx.Optional[str] = None,
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
        The user query, which triggers the import.
        This may be the full qualified name of the module, or a short alias,
        or the name of a boolean constant that indicates whether the module
        is available.
    qualname : str, optional
        The qualified name to use to import the module.
    shortname : str, optional
        The short name to use to store the module in the namespace.
    uppername : str, optional
        The name of the boolean constant to store in the namespace,
        indicating whether the module is available.

    Returns
    -------
    module : module
        The queried module or constant.
        The queried module is set to `None` if its import fails,
        rather than raising an `ImportError`.
    """
    qualname = qualname or query
    qualnames = qualname.split(".")
    rootname = qualnames[0]
    leafname = qualnames[-1]
    shortname = shortname or leafname.lower()
    uppername = uppername or "_".join(map(str.upper, qualnames))
    uppername = "HAS_" + uppername
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
    except ImportError:
        leaf = None

    # Only the root and the caller's chosen short name are recorded. The
    # leaf name alone is ambiguous: `scipy.ndimage` and
    # `cupyx.scipy.ndimage` would both claim `ndimage`, and whichever
    # imported last would win.
    namespace[rootname] = root
    namespace[shortname] = leaf
    namespace[uppername] = leaf is not None

    # The query is one of the names just written, except for a fully
    # qualified submodule, which is the leaf itself.
    return namespace[query] if query in namespace else leaf
