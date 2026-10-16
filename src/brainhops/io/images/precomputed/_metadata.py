"""The metadata of a Neuroglancer precomputed volume."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence
from brainhops.io.metadata import MetadataFormat, MetadataReader

# this format
from ._data import read_precomputed
from ._raw import PrecomputedRaw, _no_options, _score


@register_format
class PrecomputedMetadata(MetadataReader, MetadataFormat):
    """Metadata of a Neuroglancer precomputed volume, which is its `info`.

    The metadata of a precomputed volume is its `info` file, held as a
    [`PrecomputedRaw`][] record. It is read from the directory of the
    volume, from the `info` file itself, or from the content of that
    file, and reading it never reads a chunk. The metadata keeps no open
    file and no location. Precomputed volumes are not written, so the
    metadata cannot be written either.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("precomputed", "neuroglancer")
    RAW: tx.ClassVar[type] = PrecomputedRaw

    raw: NoRepr[tx.Optional[PrecomputedRaw]] = None
    """The record of the volume, or `None` for metadata without a record."""

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a name designates a volume.

        The name is the directory of the volume or its `info` file, local
        or remote, and the `info` file is read.

        Parameters
        ----------
        filename : path-like
            The directory or the `info` file of a volume.
        error : bool or type[Exception], default=False
            Whether to raise an error when the name designates no volume.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, `Confidence.CERTAIN` or `Confidence.NO`.
        """
        try:
            read_precomputed(filename)
        except Exception:  # noqa: BLE001
            return _score(Confidence.NO, error)
        return Confidence.CERTAIN

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "PrecomputedMetadata":
        """Read the metadata of a volume from its directory or `info` file.

        A remote volume, such as `gs://bucket/volume`, is read with
        TensorStore. The `precomputed://` prefix that Neuroglancer adds
        to its URLs is not part of the name.

        Parameters
        ----------
        filename : path-like
            The directory or the `info` file of a volume, local or remote.
        **kwargs : Any
            Refused, since the metadata has no option.

        Returns
        -------
        PrecomputedMetadata
            The metadata, which holds the record of the `info` file.

        Raises
        ------
        TypeError
            If an option is given.
        ParserExistsError
            If there is no volume at that name.
        ParserContentError
            If the `info` file breaks the specification.
        """
        _no_options(kwargs)
        _, record = read_precomputed(filename)
        return cls.from_raw(record)
