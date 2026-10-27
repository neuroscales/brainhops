"""The Zarr store adapters of the parser base."""

# stdlib
import os

# dependencies
import typing_extensions as tx
from abczarr import ZarrNode
from abczarr import open as open_node

# core
from brainhops._core import path
from brainhops._core.path import is_local

# internals
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    ParserTypeError,
    WriterError,
    disabled,
)

# A path, or an object that already represents an opened store.
StoreLike = tx.Union[str, path.PathLike, tx.Any]

OPEN_MODE = "r"
"""The mode in which abczarr opens a store that is only read."""


class ZarrReader(BinaryFileReader):
    """Base class for the formats that read a Zarr store.

    A Zarr store is a directory, or a prefix of keys in a remote store,
    and not a file that can be read as a stream. This adapter routes the
    file-based reader interface to store locations: a path is opened with
    abczarr and read by [`from_node`][ZarrReader.from_node], while bytes
    and streams are refused. The adapter holds no state, so a format
    class that inherits it gains methods and no field.

    A concrete format implements `_score_store`, which scores an opened
    node, and `from_node`, which reads it.
    """

    HINTS = ("zarr",)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        # The inherited sniffer opens the name as a stream, which fails on a
        # directory, so the store is opened as `from_store` opens it.
        # abczarr raises, rather than returning `None`, when a path holds no
        # store, and a sniffer only scores.
        try:
            location = absolute_location(path.Path(filename))
            node = open_node(location, OPEN_MODE)
            score = Confidence.NO if node is None else cls._score_store(node)
        except Exception:  # noqa: BLE001
            score = Confidence.NO

        if score == Confidence.NO and error is not False:
            if error is True:
                error = ParserExistsError
            raise error(f"No Zarr store at {filename}")
        return score

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

        Each format implements this method. A native zarr-python,
        TensorStore or zarrista object is first wrapped in an `abczarr`
        node.

        Parameters
        ----------
        node : ZarrNode or native Zarr object
            The opened array or group.
        **kwargs : Any
            Options of the reader of the format.

        Returns
        -------
        Self
            The object read from the node.

        Raises
        ------
        NotImplementedError
            In this base class.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_node."
        )

    @classmethod
    def from_store(cls, location: StoreLike, **kwargs) -> tx.Self:
        """Read an object from a Zarr store location.

        The keyword arguments `mode` and `driver` are passed to `abczarr.open`,
        and the remaining ones to [`from_node`][ZarrReader.from_node]. A
        local path is made absolute first, so that the opened node knows
        where its store lies whatever the working directory.

        Parameters
        ----------
        location : str or PathLike
            The location of the store.
        **kwargs : Any
            `mode` and `driver`, and the options of the reader.

        Returns
        -------
        Self
            The object read from the store.

        Raises
        ------
        ParserExistsError
            If no Zarr store exists at `location`.
        """
        zarr_kwargs = {}
        zarr_kwargs["mode"] = kwargs.pop("mode", OPEN_MODE)
        if "driver" in kwargs:
            zarr_kwargs["driver"] = kwargs.pop("driver")
        node = open_node(absolute_location(location), **zarr_kwargs)
        if node is None:
            raise ParserExistsError(f"No Zarr store at {location}")
        return cls.from_node(node, **kwargs)

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read an object from the Zarr store at a path.

        The inherited reader opens the path as a file, which fails because
        a Zarr store is a directory. This method reads the store with
        [`from_store`][ZarrReader.from_store] instead. The inherited
        `load` and `from_file` pass a path to this method, and a
        dispatcher calls it by name, so all of them read the store in the
        same way.

        Parameters
        ----------
        filename : str or PathLike
            The location of the store.
        **kwargs : Any
            `mode` and `driver`, and the options of the reader.

        Returns
        -------
        Self
            The object read from the store.

        Raises
        ------
        ParserExistsError
            If no Zarr store exists at `filename`.
        """
        return cls.from_store(path.Path(filename), **kwargs)

    from_fileobj = disabled(
        ParserTypeError,
        "A Zarr store is read from a store path, not from a file object.",
    )

    from_bytes = disabled(
        ParserTypeError,
        "A Zarr store is read from a store path, not from bytes.",
    )

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        """Return the confidence that an opened node holds this format.

        Subclasses implement this hook; the base implementation raises
        `NotImplementedError`.
        """
        raise NotImplementedError


class ZarrReaderWriter(ZarrReader, BinaryFileWriter):
    """Base class for the formats that read and write a Zarr store.

    The adapter routes the file-based writer interface to store
    locations: a path is written by `to_store`, and a stream is refused.
    It holds no state. A concrete format implements `to_store`, which
    writes to a location, and `to_node`, which writes into an opened
    node.
    """

    def to_node(self, node: tx.Any, **kwargs) -> ZarrNode:
        """Write the object into an opened Zarr array or group.

        Each format implements this method.

        Parameters
        ----------
        node : ZarrNode or native Zarr object
            The opened array or group to write into.
        **kwargs : Any
            Options of the writer of the format.

        Returns
        -------
        ZarrNode
            The node that was written into.

        Raises
        ------
        NotImplementedError
            In this base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_node."
        )

    def to_store(self, location: StoreLike, **kwargs) -> None:
        """Write the object to a Zarr store location.

        Each format implements this method.

        Parameters
        ----------
        location : str or PathLike
            The location of the store.
        **kwargs : Any
            Options of the writer of the format.

        Raises
        ------
        NotImplementedError
            In this base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_store."
        )

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            return self.to_store(file, **kwargs)

        if hasattr(file, "write") or hasattr(file, "writelines"):
            return self.to_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot write file of type {type(file)}")

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        # The inherited writer opens the name as a stream, which a store is
        # not.
        return self.to_store(filename, **kwargs)

    to_fileobj = disabled(
        WriterError,
        "A Zarr store is written to a store path, not to a file object.",
    )

    to_bytes = disabled(
        WriterError, "A Zarr store is written to a store path, not to bytes."
    )


class _ZarrNodeReaderWriter(DataModelBase, ZarrReaderWriter):
    """The stateful parser that the OME-Zarr formats still inherit.

    `OmeZarrImage` and `OmeZarrField` do not follow the design of format
    records yet, and they read their group through the field `node`. This
    class keeps that field, its position among the positional fields of
    those classes, and the stateful parser that `ZarrReaderWriter` was
    before step 4b-i of the fourth pass, until step 4b-ii moves the two
    classes to a record and a tuple of lazy arrays and removes it. No
    other class may inherit it.
    """

    node: tx.Optional[ZarrNode] = None


def opened_node(source: tx.Any) -> ZarrNode:
    """Return an opened Zarr array or group as an `abczarr` node.

    A native zarr-python, TensorStore or zarrista object is wrapped, and
    an `abczarr` node is returned as it is.

    Raises
    ------
    ParserTypeError
        If `source` is a path, or an object that cannot be wrapped.
    """
    opened = _as_node(source)
    if opened is None:
        raise ParserTypeError(
            "from_node expects an opened Zarr array or group; pass a "
            "store path to from_store instead."
        )
    return opened


def absolute_location(location: tx.Any) -> tx.Any:
    """Return a store location, with a local path made absolute.

    The zarr-python driver of abczarr 0.2.2 gives a node opened from a
    relative path the store path `.`, so a local path is made absolute
    before it is opened. A remote path is returned as it is.
    """
    if isinstance(location, str):
        location = path.Path(location)
    if isinstance(location, path.PathLike) and is_local(location):
        return path.Path(os.path.abspath(os.fspath(location)))
    return location


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
