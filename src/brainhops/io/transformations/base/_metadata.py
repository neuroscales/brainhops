"""Helpers shared by the transformation formats that carry metadata."""

__all__ = ["sync_metadata"]

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.metadata import FileBasedMetadata


def sync_metadata(
    obj: tx.Any,
    klass: tx.Type[FileBasedMetadata],
    raw: tx.Any,
    *,
    image: tx.Any = None,
    same: tx.Optional[tx.Callable[[tx.Any, tx.Any], bool]] = None,
) -> None:
    """
    Read the metadata of `obj` from the raw record `raw`, when it is not
    its raw record yet (called from a parser's `__post_init__`).

    A metadata whose raw record is `raw` already (by identity, or by
    `same`) is kept as it is. Otherwise the raw record is decoded, and
    the fields that changed in the metadata given (explicitly, or carried
    over by `replace()`) are set over the decoded ones, as changes (see
    `FileBasedMetadata.update_from_raw`). A raw record of `None` is
    decoded every time: there is nothing to tell whether it was read
    already. The field converts what it is given (`MetadataField`), so
    `metadata` is already of `klass` here.
    """
    metadata = obj.metadata
    if metadata is not None and raw is not None:
        held = metadata.raw
        if held is raw or (
            same is not None and held is not None and same(held, raw)
        ):
            return
    if metadata is None:
        metadata = klass()
    obj.metadata = metadata.update_from_raw(raw, image=image)
