"""The record of a Neuroglancer precomputed volume: its `info` file."""

# stdlib
import json
import math

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic, replace

# internals
from brainhops._core._internal_api import preserve_position
from brainhops.io.base._internal_api import Raw
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    SnifferContentError,
)

# The conventions below follow the specification of the format at
# google/neuroglancer@abbdfc5ea97b81c0c0bf04a414f5c59f4d84d27f:
# src/datasource/precomputed/volume.md (the `info` file and the chunks),
# src/datasource/precomputed/sharded.md (the sharding specification),
# src/util/data_type.ts (the data types) and
# src/datasource/precomputed/base.ts (the encodings).

_VOLUME_TYPE = "neuroglancer_multiscale_volume"
"""The value of `@type` in the `info` file of a volume."""

_SHARDING_TYPE = "neuroglancer_uint64_sharded_v1"
"""The value of `@type` in a sharding specification."""

_VOLUME_TYPES = ("image", "segmentation")
"""The values of `type`."""

_DATA_TYPES = (
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "float32",
)
"""The values of `data_type`, which are the members of `DataType` in
`data_type.ts`. The page `volume.md` lists only the unsigned ones and
`float32`, but it defers to `data_type.ts`."""

ENCODINGS = (
    "raw",
    "jpeg",
    "compressed_segmentation",
    "compresso",
    "png",
    "jxl",
    "crackle",
)
"""The encodings of the chunks, the members of `VolumeChunkEncoding`."""

_HASHES = ("identity", "murmurhash3_x86_128")
"""The hash functions of a sharding specification."""

_INDEX_ENCODINGS = ("raw", "gzip")
"""The encodings of the minishard indices and of the sharded data."""

_SNIFF_SIZE = 16 * 1024 * 1024
"""The size above which a stream is not taken for an `info` file."""


class PrecomputedSharding(Magic, frozen=True):
    """How the chunks of one scale are gathered into shard files.

    A sharded scale stores its chunks in `2**shard_bits` files named
    `<shard>.shard`, where `<shard>` is the shard number in lowercase
    hexadecimal, padded to `ceil(shard_bits / 4)` digits. A chunk is
    identified by the compressed Morton code of its position in the grid
    of chunks, and the hash of that code, shifted right by
    `preshift_bits`, gives its shard and its minishard. Each shard file
    starts with an index of its minishards, and each minishard index
    lists its chunks and where their data lie in the file. The record
    only describes the layout; the chunks are found and read by
    TensorStore.
    """

    preshift_bits: int
    """The number of low bits of a chunk code that are not hashed."""

    hash: tx.Literal["identity", "murmurhash3_x86_128"]
    """The hash function, `"identity"` or `"murmurhash3_x86_128"`."""

    minishard_bits: int
    """The number of bits of the hashed code that give the minishard."""

    shard_bits: int
    """The number of bits of the hashed code that give the shard."""

    minishard_index_encoding: tx.Literal["raw", "gzip"] = "raw"
    """The compression of the minishard indices, `"raw"` or `"gzip"`."""

    data_encoding: tx.Literal["raw", "gzip"] = "raw"
    """The compression of the chunk data, `"raw"` or `"gzip"`."""


class PrecomputedScale(Magic, frozen=True):
    """One resolution of a precomputed volume, as its `info` file lists it.

    A scale is a grid of `size` voxels whose chunks are stored under the
    path `key`, relative to the directory of the `info` file. The volume
    is cut into chunks of `chunk_sizes[0]` voxels, and the chunks at the
    upper edges are truncated to fit. Without sharding, each chunk is a
    file named `<xBegin>-<xEnd>_<yBegin>-<yEnd>_<zBegin>-<zEnd>`, where
    the bounds are voxel indices that include `voxel_offset`, the end
    being excluded. The data of a chunk is a `[x, y, z, channel]` array in
    Fortran order, so `x` varies fastest, encoded as `encoding` says.

    The voxel `i` of the scale covers the interval
    `[voxel_offset + i, voxel_offset + i + 1)` in units of `resolution`,
    which are nanometres, so its centre lies at
    `(voxel_offset + i + 0.5) * resolution` nanometres.
    """

    key: str
    """The path of the chunks, relative to the directory of `info`."""

    size: tx.Tuple[int, int, int]
    """The number of voxels along `x`, `y` and `z`."""

    resolution: tx.Tuple[float, float, float]
    """The size of a voxel along `x`, `y` and `z`, in nanometres."""

    voxel_offset: tx.Tuple[int, int, int] = (0, 0, 0)
    """The index of the first voxel along `x`, `y` and `z`, in voxels."""

    chunk_sizes: tx.Tuple[tx.Tuple[int, int, int], ...] = ()
    """The chunk sizes in voxels. A reader uses the first one."""

    encoding: tx.Literal[
        "raw",
        "jpeg",
        "compressed_segmentation",
        "compresso",
        "png",
        "jxl",
        "crackle",
    ] = "raw"
    """The encoding of each chunk, in lower case."""

    compressed_segmentation_block_size: tx.Optional[
        tx.Tuple[int, int, int]
    ] = None
    """The block size of the `compressed_segmentation` encoding."""

    sharding: tx.Optional[PrecomputedSharding] = None
    """The layout of the shard files, or `None` for one file per chunk."""

    hidden: bool = False
    """Whether Neuroglancer hides the scale. The scale is still read."""

    @property
    def chunk_size(self) -> tx.Tuple[int, int, int]:
        """The size of the chunks that a reader uses, the first listed."""
        return self.chunk_sizes[0]

    @property
    def grid_shape(self) -> tx.Tuple[int, int, int]:
        """The number of chunks along `x`, `y` and `z`."""
        return tuple(
            math.ceil(n / c) for n, c in zip(self.size, self.chunk_size)
        )


class PrecomputedRaw(Raw, BinaryFileReader):
    """The `info` file of a Neuroglancer precomputed volume.

    A precomputed volume is a directory, local or served over HTTP, that
    holds a JSON file named `info` and one subdirectory of chunks for
    each scale. The `info` file says what the voxels are (`type`,
    `data_type` and `num_channels`) and lists the scales, finest first.
    The record holds these members as the file stores them, and reading
    it never reads a chunk. Members that the specification does not
    define, such as the `jpeg_quality` that TensorStore writes, are not
    read. A record is frozen, and two records are equal when their
    fields are.

    The record is held by
    [`PrecomputedMetadata`][brainhops.io.images.precomputed.PrecomputedMetadata].
    """

    volume_type: str
    """The `type` member: `"image"` or `"segmentation"`."""

    data_type: str
    """The type of the voxels, such as `"uint8"`, in lower case."""

    num_channels: int
    """The number of channels, which is 1 for a segmentation."""

    scales: tx.Tuple[PrecomputedScale, ...] = ()
    """The scales of the volume, finest first."""

    mesh: tx.Optional[str] = None
    """The directory of the meshes of a segmentation, or `None`."""

    skeletons: tx.Optional[str] = None
    """The directory of the skeletons of a segmentation, or `None`."""

    segment_properties: tx.Optional[str] = None
    """The directory of the segment properties, or `None`."""

    @property
    def dtype(self) -> np.dtype:
        """The NumPy type of the voxels, in the native byte order."""
        return np.dtype(self.data_type)

    def copy(self) -> "PrecomputedRaw":
        """Return a copy of the record.

        The record is frozen and its fields are tuples of frozen records,
        so the copy shares them.

        Returns
        -------
        PrecomputedRaw
            The copy.
        """
        return replace(self)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "PrecomputedRaw":
        """Read the `info` file of a precomputed volume from a stream.

        The position of the stream is restored. The record has no
        options.

        Parameters
        ----------
        file : BinaryIO
            The stream of the `info` file, open in binary mode.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        PrecomputedRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream is not the `info` file of a precomputed volume,
            or breaks the specification.
        """
        no_options(kwargs)
        with preserve_position(file):
            content = file.read()
        return _parse_info(_json(content))

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds the `info` of a volume.

        The stream must hold a JSON object that is a valid `info` file of
        a volume. The position of the stream is restored.

        Parameters
        ----------
        file : IO
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not the `info`
            file of a volume. With `True`, the error is a
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
            If the content is not the `info` file of a volume and `error`
            is `True`.
        """
        return checked_score(sniff_info(file), error)


def sniff_info(file: tx.IO) -> float:
    """Score a stream as the `info` file of a precomputed volume.

    The stream must start with a JSON object, after white space, and hold
    a valid `info` file. A stream that cannot be read, or that is larger
    than 16 MiB, scores `Confidence.NO`. The position of the stream is
    restored.

    Parameters
    ----------
    file : IO
        The stream to test, open in binary mode.

    Returns
    -------
    float
        The confidence that the stream holds the `info` file of a volume.
    """
    try:
        with preserve_position(file):
            content = file.read(_SNIFF_SIZE + 1)
        if len(content) > _SNIFF_SIZE or not content.lstrip().startswith(b"{"):
            return Confidence.NO
        _parse_info(_json(content))
    except Exception:  # noqa: BLE001
        return Confidence.NO
    return Confidence.CERTAIN


def no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options of a reader that takes none.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            "Unknown Neuroglancer precomputed reader option(s): "
            f"{', '.join(kwargs)}"
        )


def checked_score(
    score: float, error: tx.Union[bool, tx.Type[Exception]]
) -> float:
    """Return a score, or raise if it is zero and the caller asks to."""
    if score or not error:
        return score
    if error is True:
        error = SnifferContentError
    raise error("Content is not the info file of a precomputed volume.")


def _json(content: bytes) -> tx.Any:
    """Decode the JSON of an `info` file.

    Raises
    ------
    ParserContentError
        If the content is not JSON.
    """
    try:
        return json.loads(bytes(content).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ParserContentError(
            f"The info file of a precomputed volume is not valid JSON: {e}"
        ) from None


def _parse_info(info: tx.Any) -> PrecomputedRaw:
    """Build the record of a parsed `info` file, checking it.

    Raises
    ------
    ParserContentError
        If the object breaks the specification of a volume.
    """
    if not isinstance(info, dict):
        raise ParserContentError(
            "The info file of a precomputed volume must hold a JSON object."
        )
    kind = info.get("@type", _VOLUME_TYPE)
    if kind != _VOLUME_TYPE:
        raise ParserContentError(
            f"This info file describes a {kind!r}, not a precomputed "
            f"volume ({_VOLUME_TYPE!r})."
        )
    volume_type = _choice(info, "type", _VOLUME_TYPES)
    data_type = _choice(info, "data_type", _DATA_TYPES)
    num_channels = _integer(info, "num_channels", minimum=1)
    if volume_type == "segmentation" and num_channels != 1:
        raise ParserContentError(
            "A precomputed segmentation must have one channel, not "
            f"{num_channels}."
        )
    scales = info.get("scales")
    if not isinstance(scales, list) or not scales:
        raise ParserContentError(
            "The info file of a precomputed volume must list its scales."
        )
    return PrecomputedRaw(
        volume_type=volume_type,
        data_type=data_type,
        num_channels=num_channels,
        scales=tuple(_parse_scale(scale, i) for i, scale in enumerate(scales)),
        mesh=_optional_string(info, "mesh"),
        skeletons=_optional_string(info, "skeletons"),
        segment_properties=_optional_string(info, "segment_properties"),
    )


def _parse_scale(scale: tx.Any, index: int) -> PrecomputedScale:
    """Build the record of one scale of an `info` file, checking it.

    Raises
    ------
    ParserContentError
        If the scale breaks the specification.
    """
    where = f"scale {index}"
    if not isinstance(scale, dict):
        raise ParserContentError(f"The {where} is not a JSON object.")
    key = scale.get("key")
    if not isinstance(key, str) or not key:
        raise ParserContentError(f"The {where} has no key.")
    where = f"scale {key!r}"
    size = _triple(scale, "size", where, int, minimum=1)
    resolution = _triple(scale, "resolution", where, float, minimum=0)
    offset = (0, 0, 0)
    if scale.get("voxel_offset") is not None:
        offset = _triple(scale, "voxel_offset", where, int)
    chunk_sizes = scale.get("chunk_sizes")
    if not isinstance(chunk_sizes, list) or not chunk_sizes:
        raise ParserContentError(f"The {where} lists no chunk size.")
    chunk_sizes = tuple(
        _triple({"chunk_sizes": c}, "chunk_sizes", where, int, minimum=1)
        for c in chunk_sizes
    )
    encoding = scale.get("encoding")
    if not isinstance(encoding, str) or encoding.lower() not in ENCODINGS:
        raise ParserContentError(
            f"The encoding of the {where} is {encoding!r}, which is not one "
            f"of {', '.join(ENCODINGS)}."
        )
    encoding = encoding.lower()
    block = None
    if encoding == "compressed_segmentation":
        block = _triple(
            scale, "compressed_segmentation_block_size", where, int, minimum=1
        )
    elif scale.get("compressed_segmentation_block_size") is not None:
        raise ParserContentError(
            f"The {where} gives a compressed_segmentation_block_size, but its "
            f"encoding is {encoding!r}."
        )
    sharding = None
    if scale.get("sharding") is not None:
        sharding = _parse_sharding(scale["sharding"], where)
        if len(chunk_sizes) != 1:
            raise ParserContentError(
                f"The sharded {where} must list a single chunk size."
            )
    hidden = scale.get("hidden", False)
    if not isinstance(hidden, bool):
        raise ParserContentError(
            f"The hidden flag of the {where} is not a boolean."
        )
    return PrecomputedScale(
        key=key,
        size=size,
        resolution=resolution,
        voxel_offset=offset,
        chunk_sizes=chunk_sizes,
        encoding=encoding,
        compressed_segmentation_block_size=block,
        sharding=sharding,
        hidden=hidden,
    )


def _parse_sharding(sharding: tx.Any, where: str) -> PrecomputedSharding:
    """Build the record of a sharding specification, checking it.

    Raises
    ------
    ParserContentError
        If the specification breaks `sharded.md`.
    """
    where = f"sharding of the {where}"
    if not isinstance(sharding, dict):
        raise ParserContentError(f"The {where} is not a JSON object.")
    if sharding.get("@type") != _SHARDING_TYPE:
        raise ParserContentError(
            f"The {where} has @type {sharding.get('@type')!r}, not "
            f"{_SHARDING_TYPE!r}."
        )
    return PrecomputedSharding(
        preshift_bits=_integer(sharding, "preshift_bits", minimum=0),
        hash=_choice(sharding, "hash", _HASHES),
        minishard_bits=_integer(sharding, "minishard_bits", minimum=0),
        shard_bits=_integer(sharding, "shard_bits", minimum=0),
        minishard_index_encoding=_choice(
            sharding, "minishard_index_encoding", _INDEX_ENCODINGS, "raw"
        ),
        data_encoding=_choice(
            sharding, "data_encoding", _INDEX_ENCODINGS, "raw"
        ),
    )


def _choice(
    obj: tx.Mapping[str, tx.Any],
    name: str,
    values: tx.Tuple[str, ...],
    default: tx.Optional[str] = None,
) -> str:
    """Return a string member that must be one of some values.

    The member is compared without regard to case, and returned in lower
    case.

    Raises
    ------
    ParserContentError
        If the member is missing without a default, or has another value.
    """
    value = obj.get(name, default)
    if isinstance(value, str) and value.lower() in values:
        return value.lower()
    raise ParserContentError(
        f"The member {name!r} is {value!r}, which is not one of "
        f"{', '.join(values)}."
    )


def _integer(obj: tx.Mapping[str, tx.Any], name: str, minimum: int = 0) -> int:
    """Return an integer member that must be at least `minimum`.

    Raises
    ------
    ParserContentError
        If the member is missing, is not an integer, or is too small.
    """
    value = obj.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParserContentError(f"The member {name!r} is not an integer.")
    if value < minimum:
        raise ParserContentError(
            f"The member {name!r} is {value}, which is less than {minimum}."
        )
    return value


def _triple(
    obj: tx.Mapping[str, tx.Any],
    name: str,
    where: str,
    kind: type,
    minimum: tx.Optional[float] = None,
) -> tx.Tuple[tx.Any, tx.Any, tx.Any]:
    """Return a member that is an `[x, y, z]` array of numbers.

    With `kind=int`, the numbers must be integers. With a `minimum`, an
    integer must be at least `minimum` and a float must be greater than
    `minimum` and finite.

    Raises
    ------
    ParserContentError
        If the member is missing or breaks these rules.
    """
    value = obj.get(name)
    numbers = (int,) if kind is int else (int, float)
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(v, bool) or not isinstance(v, numbers) for v in value
        )
    ):
        raise ParserContentError(
            f"The {name} of the {where} is {value!r}, not an [x, y, z] "
            f"array of {kind.__name__} values."
        )
    if minimum is not None:
        if kind is int:
            wrong = any(v < minimum for v in value)
        else:
            wrong = any(not (math.isfinite(v) and v > minimum) for v in value)
        if wrong:
            raise ParserContentError(
                f"The {name} of the {where} is {value!r}, which is out of "
                f"range."
            )
    return tuple(kind(v) for v in value)


def _optional_string(obj: tx.Mapping[str, tx.Any], name: str) -> tx.Any:
    """Return a string member, or `None` when it is missing.

    Raises
    ------
    ParserContentError
        If the member is present and is not a string.
    """
    value = obj.get(name)
    if value is not None and not isinstance(value, str):
        raise ParserContentError(f"The member {name!r} is not a string.")
    return value
