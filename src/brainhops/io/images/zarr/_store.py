"""Adapt the file-based reader and writer surface to a Zarr store.

A Zarr node is a directory-backed store rather than a single file, so the
file-handle machinery of the parser base does not apply to it. This mixin
opens the store through abczarr instead. It inherits the common parser
class so that dispatch, `load`, and `save` reach it the same way they reach
every other reader, and it routes their file operations to a store path.

A concrete image class supplies the store-specific behaviour through three
hooks. `_score_store` scores an opened node, `_read_node` builds the image
from one, and `_write_node` writes the image into one. The public entry
points pair an open-node door with a location door in each direction:
[from_node][brainhops.io.images.zarr._store.ZarrParser.from_node] and
[to_node][brainhops.io.images.zarr._store.ZarrParser.to_node] operate on an
opened node, and
[from_store][brainhops.io.images.zarr._store.ZarrParser.from_store] and
[to_store][brainhops.io.images.zarr._store.ZarrParser.to_store] operate on a
store location.
"""

# dependencies
import abczarr
import typing_extensions as tx
from abczarr.abc.sync import ZarrNode

# core
from brainhops._core import path

# internals
from brainhops.io.base.parsers import (
    Confidence,
    FileParserWriter,
    ParserExistsError,
    ParserTypeError,
    WriterError,
)

#: A location is a store path or an already-opened store or node object.
StoreLike = tx.Union[str, "path.PathLike", tx.Any]


def _wrap_native(source: tx.Any) -> tx.Optional[ZarrNode]:
    """Wrap a driver-native array or group in an abczarr node, or `None`.

    Each abczarr driver is tried only if its backend is installed, since
    the set of installed drivers is not known ahead of time. A source that
    is not a recognized native object returns `None`.
    """
    try:
        import zarr

        if isinstance(source, zarr.Group):
            from abczarr.drivers.zarr_python import ZarrPythonGroup

            return ZarrPythonGroup(source)
        if isinstance(source, zarr.Array):
            from abczarr.drivers.zarr_python import ZarrPythonArray

            return ZarrPythonArray(source)
    except ImportError:
        pass

    try:
        import tensorstore as ts

        if isinstance(source, ts.TensorStore):
            from abczarr.drivers.tensorstore import TensorStoreArray

            return TensorStoreArray(source)
    except ImportError:
        pass

    try:
        import zarrista

        if isinstance(source, zarrista.Array):
            from abczarr.drivers.zarrista import ZarristaArray

            location = getattr(source, "path", None) or getattr(
                source, "store_path", ""
            )
            return ZarristaArray(source, location)
    except ImportError:
        pass

    return None


def _as_node(source: tx.Any) -> tx.Optional[ZarrNode]:
    """Return `source` as an abczarr node, or `None` when it is a location.

    An abczarr node is returned unchanged. A driver-native array or group
    is wrapped. A string or path, which names a store rather than being an
    open node, returns `None`.
    """
    if isinstance(source, abczarr.ZarrNode):
        return source
    if isinstance(source, (str, path.PathLike)):
        return None
    return _wrap_native(source)


class ZarrParser(FileParserWriter):
    """Read and write a Zarr store through abczarr.

    A concrete image class mixes this in ahead of the file-based image
    bases. The class scores a store with `_score_store`, reads an image
    from an opened node with `_read_node`, and writes itself into an opened
    node with `_write_node`.
    """

    _READ_MODE = "r"

    # ---- public API --------------------------------------------------

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs) -> tx.Self:
        """Read the image from an opened Zarr array or group.

        `node` is an abczarr node, or a driver-native array or group from
        zarr-python, TensorStore, or zarrista, which is wrapped in an
        abczarr node before it is read. This is the reading counterpart of
        [to_node][brainhops.io.images.zarr._store.ZarrParser.to_node].
        """
        opened = _as_node(node)
        if opened is None:
            raise ParserTypeError(
                "from_node expects an opened Zarr array or group; pass a "
                "store path to from_store instead."
            )
        return cls._read_node(opened, **kwargs)

    @classmethod
    def from_store(cls, location: StoreLike, **kwargs) -> tx.Self:
        """Read the image from a store location or an opened store.

        `location` is a path, given as a string or an `os.PathLike`, or an
        already-opened abczarr or driver-native store.
        """
        node = cls._open(location, cls._READ_MODE)
        if node is None:
            raise ParserExistsError(f"No Zarr store at {location}")
        return cls._read_node(node, **kwargs)

    def to_node(self, node: tx.Any, **kwargs) -> ZarrNode:
        """Write the image into an opened Zarr node, and return it.

        `node` is an abczarr node, or a driver-native array or group that
        is wrapped before it is written.
        """
        wrapped = _as_node(node)
        if wrapped is None:
            raise WriterError(
                "to_node expects an opened Zarr array or group; pass a store "
                "path to to_store instead."
            )
        self._write_node(wrapped, **kwargs)
        return wrapped

    def to_store(self, location: StoreLike, **kwargs) -> None:
        """Write the image to a store location, or into an opened store.

        `location` is a path, given as a string or an `os.PathLike`, or an
        already-opened abczarr or driver-native store.
        """
        node = _as_node(location)
        if node is not None:
            self._write_node(node, **kwargs)
            return
        self._create_store(str(location).rstrip("/"), **kwargs)

    # ---- open --------------------------------------------------------

    @classmethod
    def _open(cls, source: tx.Any, mode: str) -> tx.Optional[ZarrNode]:
        """Open `source` as a node, or return `None` when it is absent."""
        node = _as_node(source)
        if node is not None:
            return node
        if not isinstance(source, (str, path.PathLike)):
            return None
        location = str(source).rstrip("/")
        if not path.Path(location).exists():
            return None
        try:
            return abczarr.open(location, mode=mode)
        except Exception:
            return None

    # ---- sniff -------------------------------------------------------

    @classmethod
    def sniff_file(
        cls,
        file: "path.FileLike",
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        node = cls._open(file, cls._READ_MODE)
        if node is None:
            return Confidence.NO
        try:
            return cls._score_store(node)
        except Exception:
            return Confidence.NO

    @classmethod
    def sniff_fileobj(cls, file: tx.IO, **kwargs) -> float:
        # A Zarr store is a directory, not a stream, so an open file object
        # is never one.
        return Confidence.NO

    @classmethod
    def sniff_bytes(cls, content: tx.Any, **kwargs) -> float:
        return Confidence.NO

    # ---- read --------------------------------------------------------

    @classmethod
    def from_file(cls, file: "path.FileLike", **kwargs) -> tx.Self:
        return cls.from_store(file, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        raise ParserTypeError(
            "A Zarr image is read from a store path, not from a file object."
        )

    @classmethod
    def from_bytes(cls, content: tx.Any, **kwargs) -> tx.Self:
        raise ParserTypeError(
            "A Zarr image is read from a store path, not from bytes."
        )

    # ---- write -------------------------------------------------------

    def to_file(self, file: "path.FileLike", **kwargs) -> None:
        self.to_store(file, **kwargs)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        raise WriterError(
            "A Zarr image is written to a store path, not to a file object."
        )

    # ---- hooks -------------------------------------------------------

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        """Score how well `node` matches this image class."""
        raise NotImplementedError

    @classmethod
    def _read_node(cls, node: ZarrNode, **kwargs) -> tx.Self:
        """Build the image from an opened Zarr `node`."""
        raise NotImplementedError

    def _write_node(self, node: ZarrNode, **kwargs) -> None:
        """Write this image into the opened Zarr `node`."""
        raise NotImplementedError

    def _create_store(self, location: str, **kwargs) -> None:
        """Create a store at `location` and write this image into it."""
        raise NotImplementedError
