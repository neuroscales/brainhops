"""Helpers shared by the transformation formats that carry metadata."""

__all__ = ["metadata_field", "sync_metadata"]

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.metadata import FormatMetadata, metadata_annotation


def metadata_field(klass: tx.Type[FormatMetadata], doc: str) -> tx.Any:
    """
    The annotation of the `metadata` field of a format class, narrowed
    to `klass`: keyword-only, out of `repr` and `==`, with a default.

    It must be declared on the class itself or on its first base (see
    [`brainhops.datamodel.metadata`][]).
    """
    return metadata_annotation(klass, doc, default=klass)


def sync_metadata(
    obj: tx.Any,
    klass: tx.Type[FormatMetadata],
    raw: tx.Any,
    *,
    image: tx.Any = None,
    same: tx.Optional[tx.Callable[[tx.Any, tx.Any], bool]] = None,
) -> None:
    """
    Read the metadata of `obj` from the record `raw`, when it is not its
    record yet (called from a parser's `__post_init__`).

    A metadata whose record is `raw` already (by identity, or by `same`)
    is kept as it is. Otherwise the record is decoded, and the fields
    that changed in the metadata given (explicitly, or carried over by
    `replace()`) are set over the decoded ones, as changes (see
    `FormatMetadata.with_record`). A record of `None` is decoded every
    time: there is nothing to tell whether it was read already. The
    field converts what it is given (`metadata_field`), so `metadata`
    is already of `klass` here.
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
    obj.metadata = metadata.with_record(raw, image=image)
