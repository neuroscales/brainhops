"""Keeping the metadata of a parser in step with its raw record."""

__all__ = ["parent_post_init", "sync_metadata"]

# stdlib
from inspect import Parameter, signature

# dependencies
import typing_extensions as tx

# internals
from ._base import FileBasedMetadata


def sync_metadata(
    obj: tx.Any,
    cls: tx.Type[FileBasedMetadata],
    raw: tx.Any = None,
    *,
    same: tx.Optional[tx.Callable[[tx.Any], bool]] = None,
    force: bool = False,
    image: tx.Any = None,
) -> bool:
    """
    Give a parser the metadata of its raw record, unless the parser
    already holds it.

    A parser calls this function from its `__post_init__`. When the
    metadata that the parser holds already has the record as its `raw`,
    the metadata is kept as it is. Otherwise the record is decoded, and
    the fields that changed in the metadata the parser holds (because the
    metadata was given explicitly, or carried over by `replace()`) are
    set over the decoded values, as changes (see
    `FileBasedMetadata.update_from_raw`). Metadata with no record is
    decoded again every time, since nothing tells whether it was read
    already. The `metadata` field converts what it is given, so the
    metadata is already of class `cls` here.

    Parameters
    ----------
    obj : object
        The parser, which has a `metadata` field.
    cls : type
        The metadata class of the format.
    raw : object or callable
        The raw record, or a function without arguments that builds it.
        A function suits a record that is rebuilt on each read (Zarr) or
        costly to build (MGH): it is called only when needed.
    same : callable, optional
        Whether the record of the metadata held, passed as the argument,
        is the parser's record. By default, `held is raw`.
    force : bool, optional
        Decode the record afresh, and drop the changes.
    image : object, optional
        The object passed to the decoder (usually `obj`).

    Returns
    -------
    bool
        Whether the metadata was read again.
    """
    metadata = obj.metadata
    held = getattr(metadata, "raw", None)
    if not force and held is not None:
        if held is raw if same is None else same(held):
            return False
    if callable(raw):
        raw = raw()
    if force or metadata is None:
        obj.metadata = cls.from_raw(raw, image=image)
    else:
        obj.metadata = metadata.update_from_raw(raw, image=image)
    return True


_POSITIONAL = (
    Parameter.POSITIONAL_ONLY,
    Parameter.POSITIONAL_OR_KEYWORD,
    Parameter.VAR_POSITIONAL,
)


def parent_post_init(parent: tx.Any, arguments: tx.Any = None) -> None:
    """
    Run the `__post_init__` that a parser's own one overrides.

    A parser's hook takes the constructor's arguments (`bagof` hands them
    to a hook that declares a parameter), and passes them on to the hook
    it overrides when that one declares a parameter too -- as the hook of
    a concrete transformation does -- and nothing otherwise.

    Parameters
    ----------
    parent : super
        `super()`, from the parser's hook.
    arguments : object, optional
        The arguments the parser's hook was given.
    """
    hook = getattr(parent, "__post_init__", None)
    if hook is None:
        return
    try:
        parameters = signature(hook).parameters.values()
    except (TypeError, ValueError):
        hook(arguments)
        return
    if any(parameter.kind in _POSITIONAL for parameter in parameters):
        hook(arguments)
    else:
        hook()
