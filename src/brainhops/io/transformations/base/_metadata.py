"""Helpers shared by the transformation formats that carry metadata."""

__all__ = ["metadata_field", "sync_metadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import Factory, KwOnly, NoEq, NoRepr

# internals
from brainhops.datamodel.metadata import FormatMetadata


def metadata_field(klass: tx.Type[FormatMetadata], doc: str) -> tx.Any:
    """
    The annotation of the `metadata` field of a format class, narrowed
    to `klass`: keyword-only, out of `repr` and `==`, with a default.

    It must be declared on the class itself or on its first base (see
    [`brainhops.datamodel.metadata`][]).
    """
    return tx.Annotated[
        klass, tx.Doc(doc), Factory(klass), KwOnly(), NoRepr(), NoEq()
    ]


def sync_metadata(
    obj: tx.Any,
    klass: tx.Type[FormatMetadata],
    raw: tx.Any,
    *,
    image: tx.Any = None,
) -> None:
    """
    Read the metadata of `obj` from the record `raw`, when it has not
    been read yet (called from a parser's `__post_init__`).

    Fields given explicitly in a `metadata` without a record win over
    the decoded ones, and count as changes on write.
    """
    metadata = obj.metadata
    if metadata is not None and not isinstance(metadata, klass):
        # A class that does not convert its fields: convert (and report)
        # here, as the field converter would.
        metadata = klass.from_other(metadata)
    if metadata is not None and metadata.raw is not None:
        obj.metadata = metadata
        return
    decoded = klass.from_raw(raw, image=image)
    if metadata is None:
        obj.metadata = decoded
        return
    values = {}
    for name in klass.vocabulary_fields + ("extra",):
        value = getattr(metadata, name, None)
        if value is None or (name == "extra" and not value):
            value = getattr(decoded, name, None)
        values[name] = value
    obj.metadata = klass(raw=raw, decoded=decoded._decoded, **values)
