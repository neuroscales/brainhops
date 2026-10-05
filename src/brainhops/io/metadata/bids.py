"""
The BIDS JSON sidecar codec.

A sidecar is the vocabulary as a flat JSON object, in the form that
`brainhops.io.metadata._json` writes and reads (BIDS keys, or a field's
name in `CamelCase` where BIDS has none; every other key is `extra`).
The units already match (seconds, degrees, tesla). The fields of the
[`DiffusionVocabulary`][brainhops.datamodel.metadata._vocabulary.DiffusionVocabulary]
group are not sidecar keys (BIDS keeps them in `.bval`/`.bvec` files, in
voxel axes), and an encoding direction along no voxel axis has no BIDS
string, so `to_bids` reports them as lost.
"""

__all__ = ["BidsSidecar", "from_bids", "to_bids"]

# stdlib
import json
import os
from io import BytesIO

# externals
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.datamodel.metadata import (
    UNSUPPORTED,
    ConversionReport,
    EncodingDirection,
    Metadata,
)
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.datamodel.metadata._vocabulary import (
    GROUPS,
    VOCABULARY,
    DiffusionVocabulary,
)
from brainhops.io.base._base import register_format
from brainhops.io.base._metadata_parser import MetadataParser
from brainhops.io.base.parsers import Confidence, SnifferContentError

from ._json import decode_object, jsonable, sidecar_key, to_json


def from_bids(source: tx.Any) -> Metadata:
    """
    Read a BIDS JSON sidecar into
    [`Metadata`][brainhops.datamodel.metadata.Metadata].

    Parameters
    ----------
    source : mapping | str | PathLike | file
        The sidecar, as a decoded JSON object, a JSON string, a path, or
        an open file.

    Returns
    -------
    Metadata
        The vocabulary fields the sidecar names, and its other keys in
        `extra`.
    """
    values, extra = decode_object(_read(source), _SIDECAR_FIELDS)
    return Metadata(extra=extra, **values)


def to_bids(
    metadata: Metadata,
    *,
    on_loss: tx.Optional[OnLoss] = None,
) -> tx.Dict[str, tx.Any]:
    """
    Write metadata as a BIDS JSON sidecar.

    Parameters
    ----------
    metadata : Metadata
        The metadata to write; only its vocabulary and `extra` are used.
    on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
        What to do with the fields a sidecar cannot hold (the diffusion
        fields, an encoding direction along no voxel axis). Defaults to
        the policy in effect; a `ConversionReport` is filled instead.

    Returns
    -------
    dict
        A JSON-serialisable sidecar. A key of `extra` that collides with
        a vocabulary key is overridden by the vocabulary value.
    """
    report = ConversionReport(source=metadata.format, target="bids")
    sidecar: tx.Dict[str, tx.Any] = {}
    extra = metadata.extra
    if extra and extra is not UNSUPPORTED:
        sidecar.update(jsonable(dict(extra)))
    for name in VOCABULARY:
        value = getattr(metadata, name, None)
        if value is None or value is UNSUPPORTED:
            continue
        if name not in _SIDECAR_FIELDS or (
            isinstance(value, EncodingDirection) and value.to_bids() is None
        ):
            report.lost[name] = value
            continue
        sidecar[sidecar_key(name)] = to_json(name, value)
    apply_loss_policy(report, on_loss, stacklevel=3)
    return sidecar


@register_format
class BidsSidecar(MetadataParser):
    """
    The reader of BIDS JSON sidecars, for
    [`Metadata.load`][brainhops.datamodel.metadata.Metadata.load].

    A sidecar is a file of metadata only, with no format class of its
    own: it reads as generic `Metadata` (see [`from_bids`][]). Any JSON
    object is accepted, and its keys that are not BIDS keys of the
    vocabulary land in `extra`.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".json",)
    HINTS = ("bids", "json")

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the reader is that an open file holds a
        sidecar: a JSON object.

        Parameters
        ----------
        file : file object
            A binary stream. Its position is restored.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            `MAYBE` for a JSON object (many JSON files are not sidecars),
            else 0.
        """
        start = file.tell()
        try:
            is_object = isinstance(json.load(file), dict)
        except Exception:
            is_object = False
        finally:
            file.seek(start)
        if is_object:
            return Confidence.MAYBE
        if error:
            raise (SnifferContentError if error is True else error)(
                "Content is not a JSON object"
            )
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the reader is that bytes hold a sidecar.

        Parameters
        ----------
        content : bytes
            The content of a file.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        return cls.sniff_fileobj(BytesIO(content), error=error)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs: tx.Any) -> Metadata:
        """
        Read an open sidecar.

        Parameters
        ----------
        file : file object
            The sidecar, open for reading. Its position is restored.
        **kwargs
            Ignored.

        Returns
        -------
        Metadata
            Generic metadata (see [`from_bids`][]).
        """
        with preserve_position(file):
            return from_bids(_read(file))


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------

# Not the diffusion fields: BIDS stores them as `.bval`/`.bvec` files.
_SIDECAR_FIELDS = tuple(
    name for name in VOCABULARY if name not in GROUPS[DiffusionVocabulary]
)


def _read(source: tx.Any) -> tx.Dict[str, tx.Any]:
    if isinstance(source, tx.Mapping):
        return dict(source)
    if isinstance(source, str) and source.lstrip().startswith("{"):
        return json.loads(source)
    if isinstance(source, (str, os.PathLike, path.PathLike)):
        with path.Path(source).open("r") as f:
            return json.load(f)
    if hasattr(source, "read"):
        return json.load(source)
    raise TypeError(
        f"A BIDS sidecar is a mapping, a JSON string, a path or an open "
        f"file, not {type(source).__name__}."
    )
