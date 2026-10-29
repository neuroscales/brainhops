"""Tests that the modules of brainhops import names as their visibility allows.

Every name of `brainhops` is public, internal or private, as the design
memo `spec/design/internal-api.md` explains. These tests read the import
statements of every module under `src/` and `tests/` with `ast`, without
importing the modules, and check the rules of that memo:

- no module imports a private name (a name that starts with an
  underscore) from another module;
- a module imports an internal name of another package only from the
  `_internal_api.py` module of that package;
- no module imports the `_internal_api.py` module of its own package;
- an `_internal_api.py` module only re-exports internal names of its
  package that other packages import.

Tests may import internal names from the modules that define them, so
the second rule does not apply to them.

A package is a directory that holds an `__init__.py` file. A subpackage
whose name starts with an underscore, below the top-level packages of
`brainhops`, is part of the package that contains it: for example,
`brainhops.datamodel._transformations.compute` is part of
`brainhops.datamodel`.
"""

import ast
import functools
from pathlib import Path

import pytest
import typing_extensions as tx

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def _module_name(path: Path, base: Path, prefix: str = "") -> str:
    """Return the dotted name of the module stored in a file.

    Parameters
    ----------
    path : Path
        The file of the module.
    base : Path
        The directory that holds the top-level package of the module.
    prefix : str, default=""
        A prefix for the dotted name, such as `"tests."`.

    Returns
    -------
    str
        The dotted name of the module.
    """
    parts = list(path.relative_to(base).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return prefix + ".".join(parts)


SOURCES = {_module_name(p, SRC): p for p in sorted(SRC.rglob("*.py"))}
TEST_MODULES = {
    _module_name(p, TESTS, "tests."): p for p in sorted(TESTS.rglob("*.py"))
}
MODULES = {**SOURCES, **TEST_MODULES}
TREES = {name: ast.parse(p.read_text(), str(p)) for name, p in MODULES.items()}


@functools.lru_cache(maxsize=None)
def _is_package(module: str) -> bool:
    """Tell whether a module is the `__init__.py` file of a package."""
    return MODULES[module].name == "__init__.py"


def _imported_module(module: str, node: ast.ImportFrom) -> str:
    """Return the absolute name of the module that a statement imports from.

    Parameters
    ----------
    module : str
        The module that holds the statement.
    node : ast.ImportFrom
        The statement.

    Returns
    -------
    str
        The dotted name of the module after `from`.
    """
    if node.level == 0:
        return node.module or ""
    parts = module.split(".")
    if not _is_package(module):
        parts = parts[:-1]
    parts = parts[: len(parts) - (node.level - 1)]
    if node.module:
        parts += node.module.split(".")
    return ".".join(parts)


@functools.lru_cache(maxsize=None)
def _package(module: str) -> str:
    """Return the package that a module belongs to.

    A subpackage whose name starts with an underscore, below the
    top-level packages of `brainhops`, belongs to its parent package.

    Parameters
    ----------
    module : str
        The dotted name of the module.

    Returns
    -------
    str
        The dotted name of the package.
    """
    parts = module.split(".")
    if not _is_package(module):
        parts = parts[:-1]
    for i in range(2, len(parts)):
        if parts[i].startswith("_"):
            return ".".join(parts[:i])
    return ".".join(parts)


def _is_private_module(module: str) -> bool:
    """Tell whether a module, or a package that holds it, is private."""
    return any(part.startswith("_") for part in module.split(".")[1:])


def _top_level(body: tx.List[ast.stmt]) -> tx.Iterator[ast.stmt]:
    """Yield the statements of a module that run when it is imported.

    The statements in the branches of `if` and `try` blocks are included,
    and those in the bodies of functions and classes are not.

    Parameters
    ----------
    body : list[ast.stmt]
        The statements of the module.

    Yields
    ------
    ast.stmt
        A statement.
    """
    for statement in body:
        if isinstance(statement, ast.If):
            yield from _top_level(statement.body)
            yield from _top_level(statement.orelse)
        elif isinstance(statement, ast.Try):
            yield from _top_level(statement.body)
            for handler in statement.handlers:
                yield from _top_level(handler.body)
            yield from _top_level(statement.orelse)
            yield from _top_level(statement.finalbody)
        else:
            yield statement


@functools.lru_cache(maxsize=None)
def _imports(
    module: str,
) -> tx.Tuple[tx.Tuple[ast.ImportFrom, str, ast.alias], ...]:
    """Return the names that a module imports with `from ... import`.

    The statements inside functions and classes are included. Modules
    imported by name, such as `x` in `from . import x`, and star imports
    are left out.

    Parameters
    ----------
    module : str
        The dotted name of the module.

    Returns
    -------
    tuple[tuple[ast.ImportFrom, str, ast.alias], ...]
        For each imported name, the import statement, the dotted name of
        the module that the name is imported from, and the name.
    """
    found = []
    for node in ast.walk(TREES[module]):
        if not isinstance(node, ast.ImportFrom):
            continue
        source = _imported_module(module, node)
        for alias in node.names:
            if alias.name == "*" or f"{source}.{alias.name}" in MODULES:
                continue
            found.append((node, source, alias))
    return tuple(found)


@functools.lru_cache(maxsize=None)
def _bindings(module: str) -> tx.Dict[str, tx.Tuple[str, str]]:
    """Return the names that a module binds by importing them.

    Parameters
    ----------
    module : str
        The dotted name of the module.

    Returns
    -------
    dict[str, tuple[str, str]]
        For each local name, the module it is imported from and its name
        there.
    """
    bindings = {}
    for statement in _top_level(TREES[module].body):
        if isinstance(statement, ast.ImportFrom):
            source = _imported_module(module, statement)
            for alias in statement.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = (source, alias.name)
    return bindings


@functools.lru_cache(maxsize=None)
def _definitions(module: str) -> tx.Set[str]:
    """Return the names that a module defines at its top level."""
    names = set()
    for statement in _top_level(TREES[module].body):
        if isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            names.add(statement.name)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        names.add(node.id)
    return names


@functools.lru_cache(maxsize=None)
def _star_sources(module: str) -> tx.List[str]:
    """Return the modules whose names a module imports with `*`."""
    return [
        _imported_module(module, statement)
        for statement in _top_level(TREES[module].body)
        if isinstance(statement, ast.ImportFrom)
        and any(alias.name == "*" for alias in statement.names)
    ]


@functools.lru_cache(maxsize=None)
def _all(module: str) -> tx.Optional[tx.List[str]]:
    """Return the `__all__` list of a module, or `None` if it has none.

    A module may import its `__all__` from another module, as
    `brainhops.datamodel.transformations` does.

    Parameters
    ----------
    module : str
        The dotted name of the module.

    Returns
    -------
    list[str] or None
        The names listed in `__all__`.
    """
    for statement in _top_level(TREES[module].body):
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        ):
            return list(ast.literal_eval(statement.value))
    source = _bindings(module).get("__all__", (None,))[0]
    if source in SOURCES and source != module:
        return _all(source)
    return None


@functools.lru_cache(maxsize=None)
def _origin(module: str, name: str) -> tx.Tuple[str, str]:
    """Return the module that defines a name, following re-exports.

    The search stops at the last module of `brainhops` when the name
    comes from another package, such as `typing_extensions`.

    Parameters
    ----------
    module : str
        The dotted name of a module that binds the name.
    name : str
        The name in that module.

    Returns
    -------
    module : str
        The dotted name of the module that defines the name.
    name : str
        The name in that module.
    """
    seen = set()
    while (module, name) not in seen:
        seen.add((module, name))
        binding = _bindings(module).get(name)
        if binding is None and name not in _definitions(module):
            for source in _star_sources(module):
                if source in SOURCES and name in (_all(source) or ()):
                    binding = (source, name)
                    break
        if binding is None:
            return module, name
        source, original = binding
        if source not in SOURCES or f"{source}.{original}" in MODULES:
            return module, name
        module, name = source, original
    return module, name


def _public_names() -> tx.Set[tx.Tuple[str, str]]:
    """Return the public names, as pairs of a defining module and a name."""
    names = set()
    for module in SOURCES:
        if not _is_private_module(module):
            for name in _all(module) or ():
                names.add(_origin(module, name))
    return names


PUBLIC = _public_names()


def _where(module: str, node: ast.stmt) -> str:
    """Return the file and line of a statement, for an error message."""
    return f"{MODULES[module].relative_to(ROOT)}:{node.lineno}"


def test_no_private_names_are_imported() -> None:
    """No module of `src/` or `tests/` imports a private name of another."""
    errors = []
    for module in MODULES:
        for node, source, alias in _imports(module):
            name = alias.name
            if not name.startswith("_") or name.startswith("__"):
                continue
            errors.append(f"{_where(module, node)}: {source}.{name}")
    assert not errors, "Private names are imported:\n" + "\n".join(errors)


def _internal_imports() -> tx.Iterator[tx.Tuple[str, ast.stmt, str, str]]:
    """Yield the imports of internal names by the modules of `src/`.

    Yields
    ------
    module : str
        The importing module.
    node : ast.stmt
        The import statement.
    source : str
        The module that the name is imported from.
    name : str
        The imported name.
    """
    for module in SOURCES:
        for node, source, alias in _imports(module):
            name = alias.name
            if name.startswith("_") or source not in SOURCES:
                continue
            if not _is_private_module(source):
                continue
            if _origin(source, name) in PUBLIC:
                continue
            yield module, node, source, name


def test_internal_names_of_other_packages_come_from_internal_api() -> None:
    """A module imports the internal names of another package from its API.

    The API of a package is its `_internal_api.py` module. The modules of
    the package import the internal names directly from the modules that
    define them.
    """
    errors = []
    for module, node, source, name in _internal_imports():
        if _package(module) == _package(source):
            continue
        api = _package(_origin(source, name)[0]) + "._internal_api"
        if source != api:
            errors.append(f"{_where(module, node)}: {source}.{name} ({api})")
    assert not errors, (
        "Internal names of other packages are imported without their "
        "_internal_api module:\n" + "\n".join(errors)
    )


def test_no_package_imports_its_own_internal_api() -> None:
    """No module imports the `_internal_api.py` module of its own package."""
    errors = []
    for module in SOURCES:
        for node in ast.walk(TREES[module]):
            if isinstance(node, ast.ImportFrom):
                source = _imported_module(module, node)
                own = _package(module) + "._internal_api"
                names = [alias.name for alias in node.names]
                if source == own or (
                    source == _package(module) and "_internal_api" in names
                ):
                    errors.append(_where(module, node))
    assert not errors, "\n".join(errors)


API_MODULES = sorted(m for m in SOURCES if m.endswith("._internal_api"))


@pytest.mark.parametrize("api", API_MODULES)
def test_internal_api_only_reexports_used_internal_names(api: str) -> None:
    """An `_internal_api.py` module re-exports what other packages use.

    The module holds a docstring, its `__all__` list and imports from the
    modules of its own package. Every name that it imports is in
    `__all__`, is internal, and is imported from it by a module of
    another package.

    Parameters
    ----------
    api : str
        The dotted name of the `_internal_api.py` module.
    """
    package = api.rsplit(".", 1)[0]
    body = TREES[api].body
    assert ast.get_docstring(TREES[api])
    exported = set()
    for statement in body[1:]:
        if isinstance(statement, ast.ImportFrom):
            source = _imported_module(api, statement)
            assert _package(source) == package, _where(api, statement)
            for alias in statement.names:
                assert not alias.name.startswith("_"), _where(api, statement)
                origin = _origin(source, alias.name)
                assert origin not in PUBLIC, _where(api, statement)
                exported.add(alias.asname or alias.name)
        else:
            assert isinstance(statement, ast.Assign), _where(api, statement)
            assert [t.id for t in statement.targets] == ["__all__"]
    assert sorted(_all(api) or ()) == sorted(exported)
    used = {
        name
        for module in SOURCES
        if _package(module) != package
        for _, source, alias in _imports(module)
        if source == api
        for name in [alias.name]
    }
    assert exported == used, sorted(exported ^ used)
