"""The record of an N5 node: the attributes of a dataset or of a pyramid."""

# stdlib
import copy
import json

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops._core._internal_api import preserve_position
from brainhops.io.base._internal_api import Raw
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    SnifferContentError,
)

# The conventions below follow the specification of N5 at
# saalfeldlab/n5@c489d1653e05b3bac38f433037befd79f62586c4 (tag n5-3.5.1):
# the "File-system specification" of README.md (version 4.0.0),
# src/main/java/org/janelia/saalfeldlab/n5/DatasetAttributes.java (the
# names of the mandatory attributes, and the `compressionType` of version
# 0) and src/main/java/org/janelia/saalfeldlab/n5/DataType.java (the data
# types).

DATA_TYPES = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "object",
)
"""The values of `dataType` that the specification lists."""

_DATASET_KEYS = ("dimensions", "blockSize", "dataType")
"""The mandatory attributes of a dataset, besides its compression."""

_SNIFF_SIZE = 16 * 1024 * 1024
"""The size above which a stream is not taken for an `attributes.json`."""


class N5Raw(Raw, BinaryFileReader):
    """The attributes of an N5 dataset, or of a group that holds a pyramid.

    An N5 container is a directory tree in which every directory is a
    group, and each group may hold its attributes in a JSON file named
    `attributes.json`. A dataset is a group whose attributes give its
    `dimensions`, its `blockSize`, its `dataType` and its `compression`,
    and whose blocks are files below it. The record holds the attributes
    of one directory as the file stores them, in `attributes`, together
    with the mandatory attributes of a dataset, read and checked.

    A group whose subdirectories `s0`, `s1`, ... are datasets holds a
    pyramid, finest first, as the n5-viewer, Paintera and BigDataViewer
    write it. The record of such a group has no mandatory attributes of
    its own, and lists the records of its datasets in `levels`.

    Reading a record never reads a block. A record is frozen, and two
    records are equal when their fields are. The record is held by
    [`N5Metadata`][brainhops.io.images.n5.N5Metadata].
    """

    attributes: tx.Dict[str, tx.Any]
    """The content of `attributes.json`, or an empty dictionary.

    Every attribute is kept, including those that the specification does
    not define, such as the spatial attributes that brainhops interprets
    (see [`brainhops.io.images.n5`][brainhops.io.images.n5]).
    """

    dimensions: tx.Optional[tx.Tuple[int, ...]] = None
    """The number of samples along each dimension, fastest first.

    The first dimension varies fastest in a block, so it is `x` in the
    usual order of the Java tools. It is `None` for a group.
    """

    block_size: tx.Optional[tx.Tuple[int, ...]] = None
    """The size of a block along each dimension, or `None` for a group."""

    data_type: tx.Optional[str] = None
    """The `dataType` of the samples, such as `"uint16"`, or `None`."""

    compression: tx.Optional[tx.Dict[str, tx.Any]] = None
    """The `compression` of the blocks, or `None` for a group.

    It is a dictionary whose `type` names the scheme, such as `"raw"` or
    `"gzip"`, beside the parameters of the scheme. A dataset of version 0
    of the specification gives only a `compressionType`, which is read as
    `{"type": compressionType}`.
    """

    levels: tx.Tuple["N5Raw", ...] = ()
    """The records of the datasets `s0`, `s1`, ... of a pyramid.

    They are listed finest first, and the tuple is empty for a dataset or
    a group that holds no pyramid.
    """

    @property
    def is_dataset(self) -> bool:
        """Whether the record describes a dataset rather than a group."""
        return self.dimensions is not None

    @property
    def dtype(self) -> np.dtype:
        """The NumPy type of the samples, in the native byte order.

        Raises
        ------
        ParserContentError
            If the record is a group, or its samples are serialized
            objects, which have no NumPy type.
        """
        if self.data_type is None or self.data_type == "object":
            raise ParserContentError(
                f"An N5 node of data type {self.data_type!r} has no NumPy "
                f"type."
            )
        return np.dtype(self.data_type)

    def copy(self) -> "N5Raw":
        """Return a copy of the record.

        The attributes and the compression are copied deeply, so that the
        copy shares no mutable value with the record.

        Returns
        -------
        N5Raw
            The copy.
        """
        return replace(
            self,
            attributes=copy.deepcopy(self.attributes),
            compression=copy.deepcopy(self.compression),
            levels=tuple(level.copy() for level in self.levels),
        )

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "N5Raw":
        """Read the attributes of an N5 node from a stream.

        The stream holds an `attributes.json` file. A record read from a
        stream has no levels, because the datasets of a pyramid are other
        files. The position of the stream is restored.

        Parameters
        ----------
        file : BinaryIO
            The stream of the `attributes.json` file, open in binary mode.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        N5Raw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream does not hold a JSON object, or holds the
            attributes of a dataset that break the specification.
        """
        no_options(kwargs)
        with preserve_position(file):
            content = file.read()
        return parse_attributes(json_attributes(content))

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds dataset attributes.

        The stream must hold a JSON object with the mandatory attributes
        of a dataset, which the specification requires. The attributes of
        a group are not recognized from their content alone, since any
        JSON object may be the attributes of a group. The position of the
        stream is restored.

        Parameters
        ----------
        file : IO
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not the
            attributes of a dataset. With `True`, the error is a
            `SnifferContentError`, and an exception class is raised
            instead when one is given.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            The confidence, `Confidence.CERTAIN` or `Confidence.NO`.

        Raises
        ------
        SnifferContentError
            If the content is not the attributes of a dataset and `error`
            is `True`.
        """
        return score(sniff_attributes(file), error)


# ----------------------------------------------------------------------
#   PARSING
# ----------------------------------------------------------------------


def sniff_attributes(file: tx.IO) -> float:
    """Score a stream as the `attributes.json` of an N5 dataset.

    The stream must start with a JSON object, after white space, and
    hold valid attributes of a dataset. A stream that cannot be read, or
    that is larger than 16 MiB, scores `Confidence.NO`. The position of
    the stream is restored.

    Parameters
    ----------
    file : IO
        The stream to test, open in binary mode.

    Returns
    -------
    float
        The confidence that the stream holds the attributes of a dataset.
    """
    try:
        with preserve_position(file):
            content = file.read(_SNIFF_SIZE + 1)
        if len(content) > _SNIFF_SIZE or not content.lstrip().startswith(b"{"):
            return Confidence.NO
        record = parse_attributes(json_attributes(content))
    except Exception:  # noqa: BLE001
        return Confidence.NO
    return Confidence.CERTAIN if record.is_dataset else Confidence.NO


def parse_attributes(attributes: tx.Any) -> N5Raw:
    """Build the record of the attributes of one node, checking them.

    A node that gives none of `dimensions`, `blockSize` and `dataType` is
    a group. A node that gives any of them must give all of them, and a
    compression, as a dataset.

    Parameters
    ----------
    attributes : Any
        The parsed content of an `attributes.json` file.

    Returns
    -------
    N5Raw
        The record, without levels.

    Raises
    ------
    ParserContentError
        If the content is not a JSON object, or gives the attributes of a
        dataset that break the specification.
    """
    if not isinstance(attributes, dict):
        raise ParserContentError(
            "The attributes of an N5 node must be a JSON object."
        )
    if not any(key in attributes for key in _DATASET_KEYS):
        return N5Raw(attributes=attributes)
    missing = [key for key in _DATASET_KEYS if key not in attributes]
    if missing:
        raise ParserContentError(
            f"The attributes of this N5 dataset lack {', '.join(missing)}."
        )
    dimensions = _vector(attributes, "dimensions", minimum=0)
    block_size = _vector(attributes, "blockSize", minimum=1)
    if len(block_size) != len(dimensions):
        raise ParserContentError(
            f"The blockSize of this N5 dataset has {len(block_size)} "
            f"values, but its dimensions have {len(dimensions)}."
        )
    data_type = attributes["dataType"]
    if not isinstance(data_type, str) or data_type.lower() not in DATA_TYPES:
        raise ParserContentError(
            f"The dataType of this N5 dataset is {data_type!r}, which is not "
            f"one of {', '.join(DATA_TYPES)}."
        )
    return N5Raw(
        attributes=attributes,
        dimensions=dimensions,
        block_size=block_size,
        data_type=data_type.lower(),
        compression=_compression(attributes),
    )


def with_levels(record: N5Raw, levels: tx.Sequence[N5Raw]) -> N5Raw:
    """Return the record of a group that holds the datasets of a pyramid.

    Raises
    ------
    ParserContentError
        If a level is not a dataset.
    """
    for index, level in enumerate(levels):
        if not level.is_dataset:
            raise ParserContentError(f"The level s{index} is not a dataset.")
    return replace(record, levels=tuple(levels))


def no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options of a reader that takes none.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(f"Unknown N5 reader option(s): {', '.join(kwargs)}")


def score(value: float, error: tx.Union[bool, tx.Type[Exception]]) -> float:
    """Return a score, or raise if it is zero and the caller asks to.

    Raises
    ------
    SnifferContentError
        If the score is zero and `error` is `True`, or the exception
        class that `error` gives.
    """
    if value or not error:
        return value
    if error is True:
        error = SnifferContentError
    raise error("Content is not an N5 dataset or pyramid.")


def json_attributes(content: bytes) -> tx.Any:
    """Decode the JSON of an `attributes.json` file.

    Raises
    ------
    ParserContentError
        If the content is not JSON.
    """
    try:
        return json.loads(bytes(content).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ParserContentError(
            f"The attributes of an N5 node are not valid JSON: {e}"
        ) from None


def _vector(
    attributes: tx.Mapping[str, tx.Any], name: str, minimum: int
) -> tx.Tuple[int, ...]:
    """Return an attribute that is a non-empty array of integers.

    Raises
    ------
    ParserContentError
        If the attribute is not such an array, or a value is less than
        `minimum`.
    """
    value = attributes.get(name)
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
        or any(v < minimum for v in value)
    ):
        raise ParserContentError(
            f"The {name} of this N5 dataset is {value!r}, which is not a "
            f"non-empty array of integers of at least {minimum}."
        )
    return tuple(value)


def _compression(attributes: tx.Mapping[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Return the compression of a dataset.

    Version 0 of the specification names the scheme in `compressionType`,
    which `DatasetAttributes.java` still reads, without parameters.

    Raises
    ------
    ParserContentError
        If the dataset gives no valid compression.
    """
    compression = attributes.get("compression")
    if compression is None and isinstance(
        attributes.get("compressionType"), str
    ):
        return {"type": attributes["compressionType"]}
    if not isinstance(compression, dict) or not isinstance(
        compression.get("type"), str
    ):
        raise ParserContentError(
            f"The compression of this N5 dataset is {compression!r}, which "
            f"is not an object with a type."
        )
    return dict(compression)
