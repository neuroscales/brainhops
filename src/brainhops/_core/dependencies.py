# stdlib
import importlib
import importlib.metadata
import importlib.util
import re

# dependencies
import typing_extensions as tx

# Every spelling that resolves to the same dependency: a short alias, the
# qualified module name, and the availability flag. Named once so that
# `__getattr__` and `__dir__` cannot drift apart.

# ---- I/O -------------------------------------------------------------
_NIBABEL = ("nb", "nibabel", "HAS_NIBABEL")
_H5PY = ("h5", "h5py", "HAS_H5PY")
_ABCZARR = ("abczarr", "abczarr", "HAS_ABCZARR")
_PILLOW = ("pil", "PIL", "HAS_PILLOW")
_TIFFFILE = ("tifffile", "tifffile", "HAS_TIFFFILE")
_OPENSLIDE = ("openslide", "openslide", "HAS_OPENSLIDE")

# ---- backends --------------------------------------------------------
_NUMPY = ("np", "numpy", "HAS_NUMPY")
_CUPY = ("cp", "cupy", "HAS_CUPY")
_DASK = ("dk", "dask", "HAS_DASK")
_DASK_ARRAY = ("da", "dask.array", "HAS_DASK_ARRAY")
_SCIPY = ("sp", "scipy", "HAS_SCIPY")
_SCIPY_NDIMAGE = ("npndi", "scipy.ndimage", "HAS_SCIPY_NDIMAGE")
_CUPY_NDIMAGE = ("cpndi", "cupyx.scipy.ndimage", "HAS_CUPY_NDIMAGE")
# The dask backend's ndimage functions are brainhops' own, and need
# nothing but dask.
_DASK_NDIMAGE = ("dkndi", "brainhops._core.dask_ndimage", "HAS_DASK_NDIMAGE")

_LAZY_NAMES = (
    _NIBABEL
    + _H5PY
    + _ABCZARR
    + _PILLOW
    + _TIFFFILE
    + _OPENSLIDE
    + _NUMPY
    + _CUPY
    + _DASK
    + _DASK_ARRAY
    + _SCIPY
    + _SCIPY_NDIMAGE
    + _CUPY_NDIMAGE
    + _DASK_NDIMAGE
)

# The availability flags of top-level modules, which are answered from
# the module's spec rather than by importing it: whether a dependency is
# installed is asked at import time (to decide which formats exist), and
# importing it just to find out is what makes `import brainhops` slow.
# A dotted module has no such shortcut, since finding the spec of
# `a.b` imports `a`; its flag is derived instead (see `_DERIVED_FLAGS`).
_SPEC_FLAGS = {
    flag: qualname
    for _, qualname, flag in (
        _NIBABEL,
        _H5PY,
        _ABCZARR,
        _PILLOW,
        _TIFFFILE,
        _OPENSLIDE,
        _NUMPY,
        _CUPY,
        _DASK,
        _SCIPY,
    )
}


# The availability flags of submodules, which `find_spec` cannot answer
# without importing their parent. Each is answered from what makes the
# submodule importable instead:
# - `dask.array` needs dask's "array" extra (numpy), which is read from
#   dask's metadata (see `_has_extra`);
# - `scipy.ndimage` is part of scipy;
# - `cupyx` is a top-level package that ships in the cupy wheels;
# - `brainhops._core.dask_ndimage` needs nothing beyond `dask.array`.
def _has_dask_array() -> bool:
    if not __getattr__("HAS_DASK"):
        return False
    has_extra = _has_extra("dask", "array")
    if has_extra is None:
        # No metadata to read: import it, as a last resort.
        return __getattr__("da") is not None
    return has_extra


_DERIVED_FLAGS = {
    "HAS_DASK_ARRAY": _has_dask_array,
    "HAS_SCIPY_NDIMAGE": lambda: __getattr__("HAS_SCIPY"),
    "HAS_CUPY_NDIMAGE": lambda: (
        __getattr__("HAS_CUPY") and _find_spec("cupyx")
    ),
    "HAS_DASK_NDIMAGE": lambda: __getattr__("HAS_DASK_ARRAY"),
}


def __getattr__(name: str) -> tx.Any:

    # An installed module may still fail to import (a binding whose native
    # library is missing, a broken install), so its flag being `True` only
    # says that it is installed. The failure surfaces where the module is
    # first used, which reads it through its alias and finds `None`.
    if name in _SPEC_FLAGS:
        available = globals()[name] = _find_spec(_SPEC_FLAGS[name])
        return available

    if name in _DERIVED_FLAGS:
        available = globals()[name] = bool(_DERIVED_FLAGS[name]())
        return available

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
    """Whether abczarr is installed together with at least one backend driver.

    abczarr reads and writes Zarr through a driver, one of zarr-python,
    TensorStore, or zarrista. The core installs none of them, so abczarr
    being importable is not enough on its own. This returns `True` only when
    abczarr is present and at least one driver is available to it.
    """
    if not __getattr__("HAS_ABCZARR"):
        return False
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


def _find_spec(qualname: str) -> bool:
    """Whether a top-level module is installed, without importing it."""
    try:
        return importlib.util.find_spec(qualname) is not None
    except (ImportError, ValueError):
        # ValueError: a module already in `sys.modules` without a spec.
        return False


# A requirement as `importlib.metadata.requires` lists it: a name, then
# optionally extras, a version specifier, and a marker after `;`.
_REQUIREMENT = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"\s*(?:\[(?P<extras>[^\]]*)\])?"
    r"[^;]*(?:;(?P<marker>.*))?$"
)
_EXTRA_MARKER = r"^\s*\(?\s*extra\s*==\s*(['\"]){}\1\s*\)?\s*$"


def _has_distribution(name: str) -> bool:
    """Whether a distribution is installed, without importing it."""
    # Older interpreters do not normalise the name they are asked for.
    for spelling in {name, name.replace("-", "_"), name.replace("_", "-")}:
        try:
            importlib.metadata.distribution(spelling)
            return True
        except importlib.metadata.PackageNotFoundError:
            pass
    return False


def _has_extra(
    dist: str, extra: str, _seen: tx.Optional[set] = None
) -> tx.Optional[bool]:
    """
    Whether a distribution is installed with the requirements of an extra.

    This reads the distribution's metadata and does not import anything.
    It checks that each distribution the extra requires is installed, but
    not its version.

    Returns
    -------
    bool or None
        `False` if the distribution declares extras but not this one.
        `None` if this cannot be told from the metadata: the distribution
        has none, or the extra requires something under a marker other
        than the extra itself (`python_version`, `sys_platform`, ...),
        which is not evaluated here.
    """
    _seen = set() if _seen is None else _seen
    _seen.add((dist, extra))
    try:
        provided = importlib.metadata.metadata(dist).get_all("Provides-Extra")
        requirements = importlib.metadata.requires(dist)
    except importlib.metadata.PackageNotFoundError:
        return None
    if provided is not None and extra not in provided:
        return False
    marker_of_extra = re.compile(_EXTRA_MARKER.format(re.escape(extra)))
    has_extra = True
    for requirement in requirements or []:
        match = _REQUIREMENT.match(requirement)
        if not match:
            return None
        marker = match["marker"] or ""
        if "extra" not in marker:
            continue  # a requirement of the distribution itself
        if not marker_of_extra.match(marker):
            if re.search(r"(['\"])" + re.escape(extra) + r"\1", marker):
                return None  # the extra, combined with another marker
            continue  # another extra
        name = match["name"]
        if not _has_distribution(name):
            return False
        # An extra may require another extra (`dask[array]`).
        for nested in filter(None, (match["extras"] or "").split(",")):
            nested = nested.strip()
            if (name, nested) in _seen:
                continue
            has_nested = _has_extra(name, nested, _seen)
            if has_nested is None:
                return None
            has_extra = has_extra and has_nested
    return has_extra


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
    except (ImportError, OSError):
        # OSError: a binding whose native library is missing (OpenSlide).
        leaf = None

    # Only the root and the caller's chosen short name are recorded. The
    # leaf name alone is ambiguous: `scipy.ndimage` and
    # `cupyx.scipy.ndimage` would both claim `ndimage`, and whichever
    # imported last would win.
    namespace[rootname] = root
    namespace[shortname] = leaf
    # A top-level module's flag says whether it is installed, and is
    # answered from its spec (see `_SPEC_FLAGS`); it is not overwritten
    # here, so that it does not change with the order of the queries.
    if uppername not in _SPEC_FLAGS and uppername not in _DERIVED_FLAGS:
        namespace[uppername] = leaf is not None

    # The query is one of the names just written, except for a fully
    # qualified submodule, which is the leaf itself.
    return namespace[query] if query in namespace else leaf
