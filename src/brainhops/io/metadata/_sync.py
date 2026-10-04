"""Keeping the metadata of a parser in step with its raw record."""

__all__ = ["sync_metadata"]

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.metadata import FileBasedMetadata


def sync_metadata(
    obj: tx.Any,
    cls: tx.Type[FileBasedMetadata],
    raw: tx.Any = None,
    *,
    read: tx.Optional[tx.Callable[[], tx.Any]] = None,
    same: tx.Optional[tx.Callable[[tx.Any], bool]] = None,
    image: tx.Any = None,
) -> bool:
    """
    Give `obj` the metadata of its raw record, unless it holds it already
    (called from a parser's `__post_init__`).

    The raw record is `raw`, or, for a record rebuilt from a store on
    each read (Zarr), what `read()` returns, read only when needed. A
    metadata whose raw record is that record already (`same(held)`; by
    default, `held is raw`) is kept as it is. Otherwise the raw record
    is decoded, and the fields that changed in the metadata `obj` holds
    (given explicitly, or carried over by `replace()`) are set over the
    decoded ones, as changes (see `FileBasedMetadata.update_from_raw`).
    A metadata with no raw record is decoded again every time: nothing
    tells whether it was read already. The `metadata` field converts
    what it is given (`MetadataField`), so it is already of `cls` here.

    Returns whether the metadata was read again.
    """
    metadata = obj.metadata
    held = getattr(metadata, "raw", None)
    if held is not None and (held is raw if same is None else same(held)):
        return False
    if read is not None:
        raw = read()
    if metadata is None:
        metadata = cls()
    obj.metadata = metadata.update_from_raw(raw, image=image)
    return True
