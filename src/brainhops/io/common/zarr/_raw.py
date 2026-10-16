"""The record of a Zarr array: its metadata documents, as stored."""

# stdlib
import json

# dependencies
import numpy as np
import typing_extensions as tx
from abczarr import ZarrArray
from abczarr.metadata.base import ArrayMetadata, NodeMetadata
from bagof.magic import NoRepr

# internals
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import ParserContentError, WriterError

# ----------------------------------------------------------------------
#   RECORD
# ----------------------------------------------------------------------


class ZarrRaw(Raw):
    """The metadata of a Zarr array, as its store keeps it.

    A Zarr array is described by a small JSON document: `zarr.json` in
    Zarr version 3, or `.zarray` with the attributes in `.zattrs` in
    version 2. The record holds that description as the typed model of
    abczarr, an `ArrayMetadata` of the version of the store, which gives
    the shape, the data type, the chunks, the codecs, the dimension names
    and the attributes of the array. It also keeps the bytes of each
    document as they were read, so that a document that is written again
    without changes keeps its bytes, whatever its layout.

    The record never reads the chunks of the array, and it holds no handle
    on the store, so it can be copied and pickled. It is held by
    [`ZarrMetadata`][brainhops.io.common.zarr.ZarrMetadata], which never
    changes it in place: an edit is made on a copy, as in
    `replace(record, document=record.document.update_attributes(...))`.

    The record is shaped so that the record of an OME-Zarr pyramid can
    hold one such record for each of its levels, next to the record of
    its group.
    """

    document: ArrayMetadata
    """The document of the array, as the typed model of abczarr."""

    stored: NoRepr[tx.Tuple[tx.Tuple[str, bytes], ...]] = ()
    """The bytes of each document as read, with the key of its file.

    The key is the name of the file in the store, such as `zarr.json`.
    The tuple is empty for a record built in memory. A writer writes
    these bytes again for each document that is the same as the one
    that the record describes, as [`documents`][] tells.
    """

    @property
    def zarr_version(self) -> int:
        """The version of the Zarr format of the store, 2 or 3."""
        return int(self.document.zarr_format)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the array, in the order of the store."""
        return tuple(int(n) for n in self.document.shape)

    @property
    def dtype(self) -> np.dtype:
        """The data type of the array, as a NumPy type."""
        if self.zarr_version >= 3:
            return np.dtype(self.document.data_type.numpy)
        return np.dtype(self.document.dtype.numpy)

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The shape of a chunk, or of a shard, or `None` if irregular.

        A sharded array of Zarr version 3 stores its chunks in shards, so
        the shape returned is the shape of a shard, which is the unit of
        the files of the store.
        """
        if self.zarr_version < 3:
            return tuple(int(n) for n in self.document.chunks)
        configuration = getattr(
            self.document.chunk_grid, "configuration", None
        )
        shape = getattr(configuration, "chunk_shape", None)
        if shape is None:
            return None
        return tuple(int(n) for n in shape)

    @property
    def dimension_names(self) -> tx.Optional[tx.Tuple[tx.Optional[str], ...]]:
        """The name of each dimension, or `None` when none is stored.

        Only Zarr version 3 stores dimension names.
        """
        names = getattr(self.document, "dimension_names", None)
        if names is None:
            return None
        return tuple(names)

    @property
    def attributes(self) -> tx.Dict[str, tx.Any]:
        """The attributes of the array, as plain JSON values."""
        return dict(self.document.to_json().get("attributes") or {})

    @classmethod
    def from_node(cls, node: tx.Any) -> "ZarrRaw":
        """Read the record of an opened Zarr array, without its chunks.

        The document is the model that the node holds, and the bytes of
        each document are read from the store of the node, when the store
        can be read through its path.

        Parameters
        ----------
        node : ZarrArray
            The opened array.

        Returns
        -------
        ZarrRaw
            The record of the array.

        Raises
        ------
        ParserContentError
            If the node is a group.
        """
        if not isinstance(node, ZarrArray):
            raise ParserContentError(
                "This Zarr store is a group, not a plain array, so it has "
                "no array document to read."
            )
        document = node.metadata
        stored = []
        for key in _document_keys(document.zarr_format):
            content = _read_key(node.store_path, key)
            if content is not None:
                stored.append((key, content))
        return cls(document=document, stored=tuple(stored))

    def documents(self) -> tx.Dict[str, bytes]:
        """Return the bytes of each document that a writer writes.

        A document is written with the bytes as read when the parsed bytes
        give the same document as the record. Both sides are read through
        the model of abczarr and then compared through a canonical
        serialization, which tells the whole number `1` from the number
        `1.0`, so that a document that was only read keeps its bytes, and
        a document whose model changed is serialized again. In version 2,
        the attributes are written to `.zattrs`, which is left out when it
        was not stored and the record has no attribute.

        Returns
        -------
        dict of str to bytes
            The bytes of each document, by the name of its file.

        Raises
        ------
        WriterError
            If the record is of a version other than 2 or 3.
        """
        encoded = _encoded_documents(self.document)
        stored = dict(self.stored)
        if (
            ".zattrs" in encoded
            and not encoded[".zattrs"]
            and stored
            and ".zattrs" not in stored
        ):
            del encoded[".zattrs"]
        kept = _parsed_documents(stored, self.document.zarr_format)
        result = {}
        for key, document in encoded.items():
            if key in stored and _same_json(kept.get(key), document):
                result[key] = stored[key]
            else:
                result[key] = _serialized(document)
        return result


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def same_storage(first: ArrayMetadata, second: ArrayMetadata) -> bool:
    """Tell whether two documents store the chunks of an array alike.

    Two documents store their chunks alike when they are the same, apart
    from the attributes and the dimension names, which describe the array
    without changing a chunk. The chunk files of a store can then be
    copied into a store that the other document describes.
    """
    return _same_json(_storage_part(first), _storage_part(second))


def _storage_part(document: ArrayMetadata) -> tx.Dict[str, tx.Any]:
    """Return a document without what leaves its chunks unchanged."""
    data = dict(document.to_json())
    data.pop("attributes", None)
    data.pop("dimension_names", None)
    return data


def _document_keys(version: int) -> tx.Tuple[str, ...]:
    """Return the names of the document files of an array of a version.

    Raises
    ------
    WriterError
        If the version is neither 2 nor 3.
    """
    if version == 3:
        return ("zarr.json",)
    if version == 2:
        return (".zarray", ".zattrs")
    raise WriterError(
        f"Zarr version {version} cannot be written; write the array in "
        "Zarr version 2 or 3 with the option `zarr_version`."
    )


def _encoded_documents(
    document: ArrayMetadata,
) -> tx.Dict[str, tx.Dict[str, tx.Any]]:
    """Return the JSON content of each document file of an array.

    Raises
    ------
    WriterError
        If the document is of a version other than 2 or 3.
    """
    data = dict(document.to_json())
    keys = _document_keys(document.zarr_format)
    if document.zarr_format == 3:
        return {keys[0]: data}
    # Version 2 tells an array from a group by the name of its file, so
    # `.zarray` holds no `node_type`.
    attributes = dict(data.pop("attributes", None) or {})
    data.pop("node_type", None)
    return {keys[0]: data, keys[1]: attributes}


def _parsed_documents(
    stored: tx.Mapping[str, bytes], version: int
) -> tx.Dict[str, tx.Any]:
    """Return the documents that stored bytes give, through abczarr.

    The bytes are parsed into the model of abczarr and serialized again,
    so that they are compared with a record in the same normal form. An
    empty dictionary is returned when the bytes cannot be parsed, so that
    every document is then serialized again.
    """
    try:
        keys = _document_keys(version)
        if version == 3:
            if keys[0] not in stored:
                return {}
            content = json.loads(stored[keys[0]])
        else:
            if keys[0] not in stored:
                return {}
            content = json.loads(stored[keys[0]])
            content.setdefault("zarr_format", version)
            content.setdefault("node_type", "array")
            attributes = stored.get(keys[1])
            content["attributes"] = (
                json.loads(attributes) if attributes is not None else {}
            )
        model = NodeMetadata.from_json(content)
        return _encoded_documents(model)
    except Exception:  # noqa: BLE001
        return {}


def _same_json(first: tx.Any, second: tx.Any) -> bool:
    """Tell whether two JSON values are the same, telling `1` from `1.0`.

    The values are compared through a canonical serialization, in which
    the keys are sorted, so a whole number and a float never compare
    equal, unlike `1 == 1.0` in Python.
    """
    options = {"sort_keys": True, "separators": (",", ":")}
    try:
        return json.dumps(first, **options) == json.dumps(second, **options)
    except (TypeError, ValueError):
        return False


def _serialized(document: tx.Any) -> bytes:
    """Return the bytes of a document that is written again."""
    return json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8")


def _read_key(location: tx.Any, key: str) -> tx.Optional[bytes]:
    """Read the bytes of a file of a store, or `None` if it cannot be read."""
    try:
        target = location / key
        if not target.exists():
            return None
        return target.read_bytes()
    except Exception:  # noqa: BLE001
        return None
