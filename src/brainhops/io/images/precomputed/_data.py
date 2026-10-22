"""Where a precomputed volume lies, and the lazy array of one scale."""

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
from ._raw import PrecomputedRaw, PrecomputedScale

_INFO = "info"
"""The name of the JSON file that describes a volume."""

READABLE_ENCODINGS = ("raw", "jpeg", "compressed_segmentation")
"""The encodings that the `neuroglancer_precomputed` driver of TensorStore
decodes. TensorStore 0.1.45 refuses the others (`png`, `compresso`,
`jxl` and `crackle`) when it opens a scale."""


def dataset_location(filename: tx.Any) -> str:
    """Return the location of the volume that a name designates.

    A volume is designated by its directory or by its `info` file. A
    local name gives the absolute path of the directory. A remote name,
    such as `gs://bucket/volume` or `https://host/volume`, gives the URL
    of the directory, and a URL whose last component is `info` names the
    `info` file. The trailing slash is removed.

    Parameters
    ----------
    filename : path-like
        The directory or the `info` file of a volume, local or remote.

    Returns
    -------
    str
        The local path or the URL of the directory of the volume.

    Raises
    ------
    ParserExistsError
        If a local name is neither a directory that holds an `info` file
        nor an `info` file.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    if is_local(filename):
        local = os.path.abspath(os.fspath(filename))
        if os.path.isdir(local):
            if not os.path.isfile(os.path.join(local, _INFO)):
                raise ParserExistsError(
                    f"No precomputed volume in {local}: it has no info file."
                )
            return local
        if os.path.basename(local) == _INFO and os.path.isfile(local):
            return os.path.dirname(local)
        raise ParserExistsError(f"No precomputed volume at {local}.")
    location = str(filename).rstrip("/")
    if location.rsplit("/", 1)[-1] == _INFO:
        location = location.rsplit("/", 1)[0]
    return location


def read_info(location: str) -> bytes:
    """Return the content of the `info` file of a volume.

    A local file is read directly, and a remote one through the key-value
    stores of TensorStore, which read `gs://`, `http://` and `https://`
    URLs.

    Parameters
    ----------
    location : str
        The location of the volume, as [`dataset_location`][] gives it.

    Returns
    -------
    bytes
        The content of the `info` file.

    Raises
    ------
    ParserExistsError
        If the volume has no `info` file.
    ParserTypeError
        If the volume is remote and TensorStore cannot read its URL.
    """
    if _is_local_location(location):
        with open(os.path.join(location, _INFO), "rb") as file:
            return file.read()
    store = _open_kvstore(location)
    result = store.read(_INFO).result()
    if result.state != "value":
        raise ParserExistsError(f"No precomputed volume at {location}.")
    return bytes(result.value)


def read_precomputed(filename: tx.Any) -> tx.Tuple[str, PrecomputedRaw]:
    """Read the record of a volume, and return it with the location.

    Parameters
    ----------
    filename : path-like
        The directory or the `info` file of a volume, local or remote.

    Returns
    -------
    str
        The location of the volume.
    PrecomputedRaw
        The record of its `info` file.

    Raises
    ------
    ParserExistsError
        If there is no volume at that name.
    ParserContentError
        If the `info` file breaks the specification.
    """
    location = dataset_location(filename)
    return location, PrecomputedRaw.from_bytes(read_info(location))


def stream_location(file: tx.Any) -> str:
    """Return the location of the volume whose `info` file a stream reads.

    Raises
    ------
    ParserTypeError
        If the stream has no file name, which a volume of several files
        needs.
    """
    name = getattr(file, "name", None)
    if not isinstance(name, (str, os.PathLike)):
        raise ParserTypeError(
            "A precomputed volume is a directory of files, which a stream "
            "without a file name cannot hold. Read it from the path of "
            "its directory or of its info file instead."
        )
    return dataset_location(name)


class _PrecomputedProxy(ArrayProxy):
    """A lazy array of the voxels of one scale of a precomputed volume.

    The proxy holds the location of the volume and the record of the
    scale, which give its shape and type without reading anything. The
    scale is opened with the `neuroglancer_precomputed` driver of
    TensorStore the first time that values are needed, and each read
    then decodes only the chunks that it needs. The voxels are presented
    as `[x, y, z, channel]`, with the first voxel at index 0 whatever the
    `voxel_offset` of the scale, and the channel axis is dropped when the
    volume has a single channel. Indexing the proxy returns another lazy
    array (see [`ArrayProxy`][]), which shares the open scale. The proxy
    keeps no open scale when it is pickled.
    """

    location: tx.Annotated[
        tx.Any,
        tx.Doc("The local path or the URL of the directory of the volume."),
    ]
    record: NoRepr[
        PrecomputedRaw,
        tx.Doc("The record of the volume, which gives its type and channels."),
    ]
    scale: NoRepr[
        PrecomputedScale,
        tx.Doc("The record of the scale that the proxy reads."),
    ]
    _store: NoInit[
        NoRepr[
            tx.Optional[tx.Any],
            tx.Doc("The scale that TensorStore opened, or `None`."),
        ]
    ] = None

    @property
    def num_channels(self) -> int:
        """The number of channels of the volume."""
        return int(self.record.num_channels)

    @property
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the voxels, read from the record."""
        shape = tuple(self.scale.size)
        if self.num_channels != 1:
            shape += (self.num_channels,)
        return shape

    @property
    def dtype(self) -> np.dtype:
        """The type of the voxels, read from the record."""
        return self.record.dtype

    @property
    def base_chunks(self) -> tx.Tuple[int, ...]:
        """The shape of a chunk of the scale, read from the record.

        The chunk is the first of the `chunk_sizes` of the scale, which
        is the one that a reader uses, and it holds every channel.
        """
        chunks = tuple(int(n) for n in self.scale.chunk_size)
        if self.num_channels != 1:
            chunks += (int(self.num_channels),)
        return chunks

    def __getstate__(self) -> tx.Dict[str, tx.Any]:
        state = dict(self.__dict__)
        state["_store"] = None
        return state

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        """Read a box of the voxels with TensorStore.

        TensorStore reads the integers and the slice bounds that the base
        class gives as absolute positions, which they are.

        Raises
        ------
        ParserContentError
            If the scale cannot be opened, or no longer has the shape of
            the record.
        """
        values = np.asarray(self._open()[index].read().result())
        return values

    def _open(self) -> tx.Any:
        """Open the scale with TensorStore, once.

        Raises
        ------
        ParserContentError
            If TensorStore cannot open the scale, or the scale no longer
            has the shape of the record.
        """
        if self._store is not None:
            return self._store
        if ts is None:
            raise ParserContentError(
                "Reading the voxels of a precomputed volume needs "
                "TensorStore (pip install brainhops[precomputed])."
            )
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": _kvstore_spec(self.location),
            "scale_metadata": {"key": self.scale.key},
        }
        try:
            store = ts.open(spec, read=True, open=True).result()
        except Exception as e:  # noqa: BLE001
            raise ParserContentError(
                f"TensorStore cannot open the scale {self.scale.key!r} of "
                f"the precomputed volume at {self.location}: {e}"
            ) from e
        # The domain starts at the voxel offset, and integer indices are
        # absolute in TensorStore, so the domain is moved to start at 0.
        store = store.translate_to[0]
        if self.num_channels == 1:
            store = store[..., 0]
        if tuple(store.shape) != self.base_shape:
            raise ParserContentError(
                f"The scale {self.scale.key!r} of the precomputed volume at "
                f"{self.location} has shape {tuple(store.shape)}, but the "
                f"info file that the image was read with says "
                f"{self.base_shape}."
            )
        self._store = store
        return store


def _is_local_location(location: str) -> bool:
    """Tell whether the location of a volume is a local directory."""
    return is_local(path.Path(location))


def _kvstore_spec(location: str) -> tx.Any:
    """Return the TensorStore key-value store of a volume's directory.

    A local directory gives a `file` store, and a URL is given to
    TensorStore as it is, with a trailing slash.
    """
    if _is_local_location(location):
        return {"driver": "file", "path": location + os.sep}
    return location + "/"


def _open_kvstore(location: str) -> tx.Any:
    """Open the TensorStore key-value store of a remote volume.

    Raises
    ------
    ParserTypeError
        If TensorStore is missing or cannot read the URL.
    """
    if ts is None:
        raise ParserTypeError(
            "Reading a remote precomputed volume needs TensorStore (pip "
            "install brainhops[precomputed])."
        )
    try:
        return ts.KvStore.open(_kvstore_spec(location)).result()
    except Exception as e:  # noqa: BLE001
        raise ParserTypeError(
            f"TensorStore cannot read the precomputed volume at {location}: "
            f"{e}"
        ) from e
