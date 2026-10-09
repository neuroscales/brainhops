"""The Zarr store adapters of the parser base."""

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
    FileWriter,
    ParserExistsError,
    ParserTypeError,
    WriterError,
)

# A path, or an object that already represents an opened store.
StoreLike = tx.Union[str, path.PathLike, tx.Any]


class ZarrParser(DataModelBase, FileParser):
    """Base class for parsers that read a Zarr store.

    A concrete parser lists this class before its file-based bases and
    implements `_score_store`.
    """

    HINTS = ("zarr",)

    node: tx.Optional[ZarrNode] = None

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
        # The inherited sniffer opens the name as a stream, which fails on a
        # directory.
        return cls.sniff_file(filename, error=error, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
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

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs) -> tx.Self:
        """Read an object from an opened Zarr array or group.

        A native zarr-python, TensorStore or zarrista object is first wrapped
        in an `abczarr` node. The keyword arguments are passed to the
        constructor.

        Raises
        ------
        ParserTypeError
            If `node` is a path or cannot be wrapped.
        """
        opened = _as_node(node)
        if opened is None:
            raise ParserTypeError(
                "from_node expects an opened Zarr array or group; pass a "
                "store path to from_store instead."
            )
        # The field converter rejects native objects.
        return cls(node=opened, **kwargs)

    @classmethod
    def from_store(cls, location: StoreLike, **kwargs) -> tx.Self:
        """Read an object from a Zarr store location.

        The keyword arguments `mode` and `driver` are passed to `abczarr.open`,
        and the remaining ones to [`from_node`][ZarrParser.from_node].

        Raises
        ------
        ParserExistsError
            If no Zarr store exists at `location`.
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

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        """Return the confidence that an opened node holds this format.

        Subclasses implement this hook; the base implementation raises
        `NotImplementedError`.
        """
        raise NotImplementedError


class ZarrParserWriter(ZarrParser, FileWriter):
    def to_node(self, node: tx.Any, **kwargs) -> ZarrNode:
        """Copy the stored object into an opened Zarr array or group.

        The parser must hold a node. A native object is wrapped first. Despite
        the annotation, the method returns `None`.

        Raises
        ------
        WriterError
            If `node` is a path or cannot be wrapped.
        """
        wrapped = _as_node(node)
        if wrapped is None:
            raise WriterError(
                "to_node expects an opened Zarr array or group; "
                "pass a store path to to_store instead."
            )
        self.node.store_path.copy(wrapped.store_path)

    def to_store(self, location: StoreLike, **kwargs) -> None:
        """Copy the stored object to a Zarr store location.

        Nothing is written when the parser holds no node.
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
    """Return `source` as an `abczarr` node, or `None` for a location.

    A native array or group is wrapped, and a string or path gives `None`.
    """
    if isinstance(source, ZarrNode):
        return source
    if isinstance(source, (str, path.PathLike)):
        return None
    return _wrap_native(source)


def _wrap_native(source: tx.Any) -> tx.Optional[ZarrNode]:
    """Wrap a native Zarr object in an `abczarr` node, or return `None`.

    Each backend is tried only if it can be imported. Zarr-python arrays and
    groups are recognised, but only arrays for TensorStore and zarrista.
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
