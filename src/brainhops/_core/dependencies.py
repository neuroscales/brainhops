import importlib

import typing_extensions as tx

# The aliases of each dependency (short name, qualified module name and
# HAS_ flag) are defined once, so that __getattr__ and __dir__ agree.

_NIBABEL = ("nb", "nibabel", "HAS_NIBABEL")
_H5PY = ("h5", "h5py", "HAS_H5PY")
_ABCZARR = ("abczarr", "abczarr", "HAS_ABCZARR")
_PILLOW = ("pil", "PIL", "HAS_PILLOW")
_TIFFFILE = ("tifffile", "tifffile", "HAS_TIFFFILE")
_OPENSLIDE = ("openslide", "openslide", "HAS_OPENSLIDE")

_PINT = ("pint", "pint", "HAS_PINT")

_NUMPY = ("np", "numpy", "HAS_NUMPY")
_CUPY = ("cp", "cupy", "HAS_CUPY")
_DASK = ("dk", "dask", "HAS_DASK")
_DASK_ARRAY = ("da", "dask.array", "HAS_DASK_ARRAY")
_SCIPY = ("sp", "scipy", "HAS_SCIPY")
_SCIPY_NDIMAGE = ("npndi", "scipy.ndimage", "HAS_SCIPY_NDIMAGE")
_CUPY_NDIMAGE = ("cpndi", "cupyx.scipy.ndimage", "HAS_CUPY_NDIMAGE")
# The dask ndimage module belongs to brainhops and only needs dask.
_DASK_NDIMAGE = ("dkndi", "brainhops._core.dask_ndimage", "HAS_DASK_NDIMAGE")

_LAZY_NAMES = (
    _NIBABEL
    + _H5PY
    + _ABCZARR
    + _PILLOW
    + _TIFFFILE
    + _OPENSLIDE
    + _PINT
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

    if name in _PILLOW:
        return _lazy_import(globals(), name, "PIL", "pil", "PILLOW")

    if name in _TIFFFILE:
        return _lazy_import(globals(), name, "tifffile", "tifffile")

    if name in _OPENSLIDE:
        return _lazy_import(globals(), name, "openslide", "openslide")

    # ==================================================================
    #
    #                                UNITS
    #
    # ==================================================================

    if name in _PINT:
        return _lazy_import(globals(), name, "pint", "pint")

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
            globals(),
            name,
            "brainhops._core.dask_ndimage",
            "dkndi",
            "DASK_NDIMAGE",
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def has_abczarr_driver() -> bool:
    """Return whether abczarr can be imported and has a Zarr driver.

    The core installation of abczarr includes no driver, so the package
    being importable is not enough. At least one driver, such as
    zarr-python, TensorStore or zarrista, must also be available. Any error
    raised while listing the drivers is reported as `False`.
    """
    abczarr = __getattr__("abczarr")
    if abczarr is None:
        return False
    try:
        return bool(abczarr.available_drivers())
    except Exception:
        return False


def __dir__() -> tx.List[str]:
    """List the lazy names with the globals, for `dir` and tab completion."""
    return sorted(set(globals()) | set(_LAZY_NAMES))


def _lazy_import(
    namespace: tx.Dict[str, tx.Any],
    query: str,
    qualname: tx.Optional[str] = None,
    shortname: tx.Optional[str] = None,
    uppername: tx.Optional[str] = None,
) -> tx.Any:
    """Import a module lazily and record it in a namespace.

    The root package, the module under its short name, and a `HAS_<NAME>`
    flag are stored in the namespace. A module that cannot be imported is
    stored as `None` instead of raising an `ImportError`.

    Parameters
    ----------
    namespace : dict of str to Any
        Namespace in which the names are stored, usually `globals()`.
    query : str
        Name whose lookup triggered the import: the qualified module name,
        its short alias, or the name of its flag.
    qualname : str, optional
        Name of the module to import. The default is `query`.
    shortname : str, optional
        Name under which the module is stored. The default is the last
        component of `qualname`, in lower case.
    uppername : str, optional
        Suffix of the flag name. The default is the components of
        `qualname`, in upper case and joined by underscores.

    Returns
    -------
    module or bool or None
        The value stored under `query`, or the module itself when `query` is
        a qualified submodule name that is not stored.
    """
    qualname = qualname or query
    qualnames = qualname.split(".")
    rootname = qualnames[0]
    leafname = qualnames[-1]
    shortname = shortname or leafname.lower()
    uppername = uppername or "_".join(map(str.upper, qualnames))
    uppername = "HAS_" + uppername
    # Both names start as None so that a failed import of the root package,
    # the very case that this module reports, does not raise an
    # UnboundLocalError.
    root = leaf = None
    try:
        for i in range(len(qualnames)):
            name = ".".join(qualnames[: i + 1])
            leaf = importlib.import_module(name)
            if i == 0:
                root = leaf
    except (ImportError, OSError):
        # Bindings whose native library is missing, such as OpenSlide, raise an
        # OSError.
        leaf = None

    # The leaf name is not recorded because it is ambiguous: scipy.ndimage
    # and cupyx.scipy.ndimage would collide.
    namespace[rootname] = root
    namespace[shortname] = leaf
    namespace[uppername] = leaf is not None

    # A qualified submodule name, such as dask.array, is not stored.
    return namespace[query] if query in namespace else leaf
