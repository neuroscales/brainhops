"""
Adapt the file-based reader and writer surface to a Zarr store.

A Zarr node is a directory-backed store rather than a single file, so the
file-handle machinery of the parser base does not apply to it. This mixin
opens the store through abczarr instead. It inherits the common parser
class so that dispatch, `load`, and `save` reach it the same way they reach
every other reader, and it routes their file operations to a store path.

The mixin is object-agnostic. An image reader and a transformation reader
both mix it in and share its store handling. The public entry points
pair an open-node door with a location door in each direction:
[from_node][ZarrParser.from_node] and [to_node][ZarrParser.to_node]
operate on an opened node, and [from_store][ZarrParser.from_store] and
[to_store][ZarrParser.to_store] operate on a store location.
"""

__all__ = ["ZarrParser", "StoreLike"]

# dependencies
import typing_extensions as tx
from abczarr import ZarrNode
from abczarr import open as open_node

# core
from brainhops._core import path

# internals
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base.parsers import (
    Confidence,
    FileParser,
    FileParserWriter,
    ParserExistsError,
    ParserTypeError,
    WriterError,
)

#: A location is a store path or an already-opened store or node object.
StoreLike = tx.Union[str, path.PathLike, tx.Any]


class ZarrParser(DataModelBase, FileParser):
    """
    Read and write a Zarr store through abczarr.

    A concrete class mixes this in ahead of the file-based bases,
    whether it reads an image or a transformation.
    """

    HINTS = ("zarr",)

    # ---- attributes --------------------------------------------------

    node: tx.Optional[ZarrNode] = None

    # ---- sniff -------------------------------------------------------

    @classmethod
    def sniff_file(
        cls,
        file: path.FileLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        node = open_node(file, cls._READ_MODE)
        if node is None:
            score = Confidence.NO
        else:
            try:
                score = cls._score_store(node)
            except Exception:
                score = Confidence.NO

        if score == Confidence.NO and error is not False:
            if error is True:
                error = ParserExistsError
            raise error(f"No Zarr store at {file}")
        return score

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        # A store is a directory, so the inherited filename sniffer cannot
        # judge one: it opens the name and reads it as a stream, which raises
        # on a directory. The name is scored as a store instead. Without this
        # a store scores zero through `sniff`, and dispatch falls back to
        # breaking the tie on the declared extensions.
        return cls.sniff_file(filename, error=error, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        # A Zarr store is a directory, not a stream, so an open file object
        # is never one.
        if error is not False:
            if error is True:
                error = ParserTypeError
            raise error(f"Zarr stores are directories, not files: {file}")
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: tx.Any,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        if error is not False:
            if error is True:
                error = ParserTypeError
            raise error("Zarr stores are directories, not files")
        return Confidence.NO

    # ---- read --------------------------------------------------------

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs) -> tx.Self:
        """
        Read the object from an opened Zarr array or group.

        `node` is an abczarr node, or a driver-native array or group from
        zarr-python, TensorStore, or zarrista, which is wrapped in an
        abczarr node before it is read. This is the reading counterpart of
        [to_node][ZarrParser.to_node].
        """
        opened = _as_node(node)
        if opened is None:
            raise ParserTypeError(
                "from_node expects an opened Zarr array or group; pass a "
                "store path to from_store instead."
            )
        # The wrapped node is what is kept: a driver-native array is not a
        # `ZarrNode` and would be refused by the field's converter.
        return cls(node=opened, **kwargs)

    @classmethod
    def from_store(cls, location: StoreLike, **kwargs) -> tx.Self:
        """
        Read the object from a store location or an opened store.

        `location` is a path, given as a string or an [`os.PathLike`][],
        or an already-opened abczarr or driver-native store.
        """
        zarr_kwargs = {}
        zarr_kwargs["mode"] = kwargs.pop("mode", cls._READ_MODE)
        if "driver" in kwargs:
            zarr_kwargs["driver"] = kwargs.pop("driver")
        node = open_node(location, **zarr_kwargs)
        if node is None:
            raise ParserExistsError(f"No Zarr store at {location}")
        return cls.from_node(node, **kwargs)

    @classmethod
    def from_file(cls, file: "path.FileLike", **kwargs) -> tx.Self:
        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            return cls.from_store(file, **kwargs)

        if hasattr(file, "read"):
            return cls.from_fileobj(file, **kwargs)

        # Cannot parse this content -> return False or error
        raise ParserTypeError(f"Cannot parse file of type {type(file)}")

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        raise ParserTypeError(
            "A Zarr store is read from a store path, not from a file object."
        )

    @classmethod
    def from_bytes(cls, content: tx.Any, **kwargs) -> tx.Self:
        raise ParserTypeError(
            "A Zarr store is read from a store path, not from bytes."
        )

    # ---- hooks -------------------------------------------------------

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        """Score how well `node` matches this class."""
        raise NotImplementedError


class ZarrParserWriter(ZarrParser, FileParserWriter):
    # ---- write -------------------------------------------------------

    def to_node(self, node: tx.Any, **kwargs) -> ZarrNode:
        """
        Write the object into an opened Zarr node, and return it.

        `node` is an abczarr node, or a driver-native array or group that
        is wrapped before it is written.
        """
        wrapped = _as_node(node)
        if wrapped is None:
            raise WriterError(
                "to_node expects an opened Zarr array or group; "
                "pass a store path to to_store instead."
            )
        self.node.store_path.copy(wrapped.store_path)

    def to_store(self, location: StoreLike, **kwargs) -> None:
        """
        Write the object to a store location, or into an opened store.

        `location` is a path, given as a string or an [`os.PathLike`][],
        or an already-opened abczarr or driver-native store.
        """
        if self.node is not None:
            self.node.store_path.copy(location)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            return self.to_store(file, **kwargs)

        if hasattr(file, "write") or hasattr(file, "writelines"):
            return self.to_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot write file of type {type(file)}")

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        raise WriterError(
            "A Zarr store is written to a store path, not to a file object."
        )


def _as_node(source: tx.Any) -> tx.Optional[ZarrNode]:
    """Return `source` as an abczarr node, or `None` when it is a location.

    An abczarr node is returned unchanged. A driver-native array or group
    is wrapped. A string or path, which names a store rather than being an
    open node, returns `None`.
    """
    if isinstance(source, ZarrNode):
        return source
    if isinstance(source, (str, path.PathLike)):
        return None
    return _wrap_native(source)


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
