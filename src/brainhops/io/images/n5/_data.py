"""Where an N5 node lies, and the lazy array of one dataset."""

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoInit, NoRepr

# internals
from brainhops._core import path
from brainhops._core.dependencies import ts
from brainhops._core.proxy import ArrayProxy
from brainhops.io.base._utils_files import is_local
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserExistsError,
    ParserTypeError,
)

# this format
from ._raw import N5Raw, json_attributes, parse_attributes, with_levels

ATTRIBUTES = "attributes.json"
"""The name of the JSON file that holds the attributes of a node."""

READABLE_COMPRESSIONS = ("raw", "gzip", "bzip2", "xz", "blosc")
"""The compressions that the `n5` driver of TensorStore 0.1.45 decodes.

TensorStore refuses the others, such as `lz4` and `zstd`, and the
`compressionType` of version 0 of the specification, when it opens a
dataset.
"""


def node_location(filename: tx.Any) -> str:
    """Return the directory of the N5 node that a name designates.

    A node is designated by its directory or by its `attributes.json`
    file. Only local containers are read.

    Parameters
    ----------
    filename : path-like
        The directory or the `attributes.json` file of a node.

    Returns
    -------
    str
        The absolute path of the directory of the node.

    Raises
    ------
    ParserTypeError
        If the name is not a local path.
    ParserExistsError
        If the name is neither a directory nor an `attributes.json` file.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    if not is_local(filename):
        raise ParserTypeError(
            f"brainhops reads local N5 containers only, not {filename}."
        )
    local = os.path.abspath(os.fspath(filename))
    if os.path.isdir(local):
        return local
    if os.path.basename(local) == ATTRIBUTES and os.path.isfile(local):
        return os.path.dirname(local)
    raise ParserExistsError(f"No N5 node at {local}.")


def read_node(filename: tx.Any) -> tx.Tuple[str, N5Raw]:
    """Read the record of a dataset or of a pyramid, with its location.

    The attributes of the node are read, and a node that is not a
    dataset is read as a pyramid: its subdirectories `s0`, `s1`, ... are
    read as its levels, up to the first one that is missing. No block is
    read.

    Parameters
    ----------
    filename : path-like
        The directory or the `attributes.json` file of a node.

    Returns
    -------
    str
        The directory of the node.
    N5Raw
        The record of the node.

    Raises
    ------
    ParserExistsError
        If there is no node at that name.
    ParserContentError
        If the node is neither a dataset nor a pyramid, or its
        attributes break the specification.
    """
    location = node_location(filename)
    record = _read_attributes(location)
    if record.is_dataset:
        return location, record
    levels = []
    while True:
        level = os.path.join(location, f"s{len(levels)}")
        if not os.path.isfile(os.path.join(level, ATTRIBUTES)):
            break
        levels.append(_read_attributes(level))
    if not levels:
        raise ParserContentError(
            f"The N5 group at {location} is neither a dataset nor a pyramid "
            f"of datasets s0, s1, ..."
        )
    return location, with_levels(record, levels)


def dataset_location(location: str, record: N5Raw, level: int) -> str:
    """Return the directory of a dataset of a node.

    The dataset is the node itself, or the level `s<level>` of a pyramid.
    """
    if record.levels:
        return os.path.join(location, f"s{level}")
    return location


def stream_location(file: tx.Any) -> str:
    """Return the directory of the node whose attributes a stream reads.

    Raises
    ------
    ParserTypeError
        If the stream has no file name, which a node of several files
        needs.
    """
    name = getattr(file, "name", None)
    if not isinstance(name, (str, os.PathLike)):
        raise ParserTypeError(
            "An N5 dataset is a directory of files, which a stream without a "
            "file name cannot hold. Read it from the path of its directory "
            "or of its attributes.json file instead."
        )
    return node_location(name)


class _N5Proxy(ArrayProxy):
    """A lazy array of the samples of an N5 dataset.

    The proxy holds the directory of the dataset and its record, whose
    shape and type it answers without reading anything. The dataset is
    opened with the `n5` driver of TensorStore the first time that values
    are needed, and each read then decodes only the blocks that it needs.
    A missing block reads as zeros, since N5 datasets are sparse. The
    samples are indexed in the order of the `dimensions` of the dataset,
    fastest first. Indexing the proxy
    returns another lazy array (see [`ArrayProxy`][]), which shares the
    open dataset. The proxy keeps no open dataset when it is pickled.
    """

    location: tx.Annotated[tx.Any, tx.Doc("The directory of the dataset.")]
    record: NoRepr[
        N5Raw,
        tx.Doc("The record of the dataset, which gives its shape and type."),
    ]
    _store: NoInit[
        NoRepr[
            tx.Optional[tx.Any],
            tx.Doc("The dataset that TensorStore opened, or `None`."),
        ]
    ] = None

    @property
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the samples, read from the record."""
        return tuple(int(n) for n in self.record.dimensions)

    @property
    def dtype(self) -> np.dtype:
        """The type of the samples, in the native byte order."""
        return self.record.dtype

    @property
    def base_chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The shape of a block of the dataset, read from the record.

        Each block is a file of the dataset, so a Dask chunk of this
        shape reads a single file.
        """
        block = self.record.block_size
        return None if block is None else tuple(int(n) for n in block)

    def __getstate__(self) -> tx.Dict[str, tx.Any]:
        state = dict(self.__dict__)
        state["_store"] = None
        return state

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        """Read a box of the samples with TensorStore.

        TensorStore reads the integers and the slice bounds that the base
        class gives as absolute positions, which they are.

        Raises
        ------
        ParserContentError
            If the dataset cannot be opened or a block cannot be decoded.
        """
        store = self._open()
        try:
            return np.asarray(store[index].read().result())
        except Exception as e:  # noqa: BLE001
            raise ParserContentError(
                f"TensorStore cannot read the N5 dataset at {self.location}: "
                f"{e}"
            ) from e

    def _open(self) -> tx.Any:
        """Open the dataset with TensorStore, once.

        Raises
        ------
        ParserContentError
            If TensorStore is missing or cannot open the dataset, or the
            dataset no longer has the shape and the type of the record.
        """
        if self._store is not None:
            return self._store
        if ts is None:
            raise ParserContentError(
                "Reading the samples of an N5 dataset needs TensorStore "
                "(pip install brainhops[n5])."
            )
        spec = {
            "driver": "n5",
            "kvstore": {"driver": "file", "path": self.location + os.sep},
        }
        try:
            store = ts.open(spec, read=True, open=True).result()
        except Exception as e:  # noqa: BLE001
            raise ParserContentError(
                f"TensorStore cannot open the N5 dataset at {self.location}: "
                f"{e}"
            ) from e
        if tuple(
            store.shape
        ) != self.base_shape or store.dtype.numpy_dtype != (self.dtype):
            raise ParserContentError(
                f"The N5 dataset at {self.location} holds {store.dtype} "
                f"samples of shape {tuple(store.shape)}, but the attributes "
                f"that the image was read with say {self.dtype} and "
                f"{self.base_shape}."
            )
        self._store = store
        return store


def _read_attributes(location: str) -> N5Raw:
    """Read the record of the attributes of one directory, without levels.

    A directory without `attributes.json` is a group without attributes.

    Raises
    ------
    ParserContentError
        If the attributes break the specification.
    """
    filename = os.path.join(location, ATTRIBUTES)
    if not os.path.isfile(filename):
        return parse_attributes({})
    with open(filename, "rb") as file:
        return parse_attributes(json_attributes(file.read()))
