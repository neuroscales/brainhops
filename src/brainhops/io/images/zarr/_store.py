"""Adapt the file-based reader and writer surface to a Zarr store.

A Zarr node is a directory-backed store rather than a single file, so the
file-handle machinery of the parser base does not apply to it. This mixin
opens the store through abczarr instead. It routes sniffing, reading, and
writing to a store path, and leaves the scoring, the node reading, and the
node writing to the concrete image class through the
`_score_store`, `_from_node`, and `_to_store` hooks.
"""

# dependencies
import abczarr
import typing_extensions as tx

# core
from brainhops._core import path

# internals
from brainhops.io.base.parsers import Confidence, ParserExistsError


class ZarrParser:
    """Read and write a Zarr store through abczarr.

    A concrete image class mixes this in ahead of the file-based image
    bases. The class scores a store with `_score_store`, reads an image
    from an opened node with `_from_node`, and writes itself to a store
    path with `_to_store`.
    """

    _READ_MODE = "r"

    @classmethod
    def _open(cls, file: path.FileLike, mode: str) -> tx.Any:
        """Open the Zarr node at `file`, or return `None` when it is absent."""
        if not isinstance(file, (str, path.PathLike)):
            return None
        location = str(file).rstrip("/")
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
        file: path.FileLike,
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
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        node = cls._open(file, cls._READ_MODE)
        if node is None:
            raise ParserExistsError(f"No Zarr store at {file}")
        return cls._from_node(node, **kwargs)

    # ---- write -------------------------------------------------------

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        location = str(file).rstrip("/")
        self._to_store(location, **kwargs)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        raise NotImplementedError(
            "A Zarr image is written to a store path, not to a file object."
        )

    # ---- hooks -------------------------------------------------------

    @classmethod
    def _score_store(cls, node: tx.Any) -> float:
        """Score how well `node` matches this image class."""
        raise NotImplementedError

    @classmethod
    def _from_node(cls, node: tx.Any, **kwargs) -> tx.Self:
        """Build the image from an opened Zarr `node`."""
        raise NotImplementedError

    def _to_store(self, location: str, **kwargs) -> None:
        """Write this image to the store at `location`."""
        raise NotImplementedError
