"""
BIDS JSON sidecars.

A sidecar is the JSON file that BIDS writes next to an image to hold its
metadata. It is the vocabulary as a flat JSON object, in the form that
`brainhops.io.metadata._json` writes and reads: each field under its
BIDS key, or under its name in `CamelCase` where BIDS has none, and
every other key in `extra`. The units of the vocabulary are those of
BIDS (seconds, degrees, tesla).

The fields of the
[`DiffusionVocabulary`][brainhops.datamodel.metadata.DiffusionVocabulary]
group are not sidecar keys, because BIDS keeps them in `.bval` and
`.bvec` files, and an encoding direction along no voxel axis has no BIDS
string. A sidecar has no slot for them.
"""

__all__ = ["BidsSidecar", "from_bids", "to_bids"]

# stdlib
import json
import os

# externals
import typing_extensions as tx
from bagof.magic import Field, NoRepr

# internals
from brainhops._core import path
from brainhops.datamodel.metadata import (
    ConversionReport,
    DiffusionVocabulary,
    EncodingDirection,
    Metadata,
)
from brainhops.datamodel.metadata._report import (
    MetadataLossError,
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.datamodel.metadata._vocabulary import GROUPS, VOCABULARY
from brainhops.errors import UNSET, ErrorPolicy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
)

from ._base import MetadataFormat, consolidated
from ._json import decode_object, encode_changes, encode_extra

# The fields that a sidecar stores under their keys: the vocabulary but
# the diffusion fields, which BIDS keeps in `.bval` and `.bvec` files.
_KEYED_FIELDS = tuple(
    name for name in VOCABULARY if name not in GROUPS[DiffusionVocabulary]
)


@register_format
class BidsSidecar(TextFileReader, TextFileWriter, MetadataFormat):
    """
    Metadata of a BIDS JSON sidecar, which holds the decoded JSON object.

    A sidecar is a file of metadata only. Its record is the JSON object
    of the file, as a `dict`, and every vocabulary field except the
    diffusion fields has a key in it. The keys that name no field are
    `extra`. Any JSON object reads as a sidecar, with a low confidence,
    because many JSON files are not sidecars.

    | Field              | Key                                          |
    |--------------------|----------------------------------------------|
    | a vocabulary field | its BIDS key, or its name in `CamelCase`     |
    | `extra`            | every other key                              |

    The diffusion fields (`bvalues`, `bvectors`) are not supported, and
    an encoding direction along no voxel axis is reported as lost when
    the sidecar is consolidated.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".json",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("bids", "json")
    SUPPORTED_FIELDS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "extra",
        *_KEYED_FIELDS,
    )

    raw: tx.Annotated[
        tx.Optional[tx.Dict[str, tx.Any]], NoRepr(), Field(convert=False)
    ] = None
    """The JSON object of the sidecar, or `None` for metadata without a
    record."""

    @classmethod
    def from_raw(
        cls, raw: tx.Optional[tx.Dict[str, tx.Any]], **kwargs
    ) -> "BidsSidecar":
        """Build the metadata of a JSON object, decoding its keys.

        Parameters
        ----------
        raw : dict, optional
            The JSON object, held as it is given.
        **kwargs : Any
            Other fields, which take precedence over the decoded values.

        Returns
        -------
        BidsSidecar
            The metadata.
        """
        return cls(raw=raw, **{**_decode(raw), **kwargs})

    @takes_error_policy
    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_error: ErrorPolicy = UNSET,
        **fields: tx.Any,
    ) -> "BidsSidecar":
        """Encode the fields that differ from the JSON object into a copy.

        The contract is that of [`MetadataFormat.consolidate`][]: the
        changes are written into a copy of the object, or into a new
        object, and the result is decoded from it afresh.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this metadata rather than a copy.
        on_error : ErrorPolicy, optional
            What to do with the fields that a sidecar cannot hold, and
            with any other conversion error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        **fields : Any
            Fields to override before the encoding.

        Returns
        -------
        BidsSidecar
            The consolidated copy, or this metadata with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If a field is reported under the `"raise"` policy.
        """
        return consolidated(
            self,
            _encode,
            dict,
            inplace=inplace,
            on_error=on_error,
            fields=fields,
        )

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """Return the confidence that a text is a sidecar: a JSON object.

        Parameters
        ----------
        text : str
            The content of a file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not a JSON object.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            `Confidence.MAYBE` for a JSON object, because many JSON files
            are not sidecars, and `Confidence.NO` otherwise.

        Raises
        ------
        SnifferContentError
            If the text is not a JSON object and `error` is `True`.
        """
        try:
            is_object = isinstance(json.loads(text), dict)
        except ValueError:
            is_object = False
        if is_object:
            return Confidence.MAYBE
        if error:
            raise (SnifferContentError if error is True else error)(
                "Content is not a JSON object"
            )
        return Confidence.NO

    @classmethod
    def from_text(cls, text: str, **kwargs: tx.Any) -> "BidsSidecar":
        """Read a sidecar from its text.

        Parameters
        ----------
        text : str
            The JSON text of the sidecar.
        **kwargs : Any
            Ignored.

        Returns
        -------
        BidsSidecar
            The metadata, whose record is the JSON object.

        Raises
        ------
        ValueError
            If the text is not a JSON object.
        """
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("A BIDS sidecar is a JSON object.")
        return cls.from_raw(obj)

    @takes_error_policy
    def to_text(self, *, on_error: ErrorPolicy = UNSET, **fields) -> str:
        """Return the JSON text of the consolidated sidecar.

        The metadata is consolidated first, so a field that was changed
        is written. The metadata itself is not changed.

        Parameters
        ----------
        on_error : ErrorPolicy, optional
            What to do with the fields that a sidecar cannot hold, and
            with any other conversion error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        **fields : Any
            Fields to override, as in [`consolidate`][].

        Returns
        -------
        str
            The JSON text, indented by two spaces.
        """
        record = self.consolidate(on_error=on_error, **fields).raw
        return json.dumps({} if record is None else record, indent=2) + "\n"

    @takes_error_policy
    def to_lines(
        self, *, on_error: ErrorPolicy = UNSET, **fields
    ) -> tx.Iterator[str]:
        """Return the lines of the JSON text of the consolidated sidecar.

        Parameters
        ----------
        on_error : ErrorPolicy, optional
            What to do with the fields that a sidecar cannot hold, and
            with any other conversion error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        **fields : Any
            Fields to override, as in [`consolidate`][].

        Returns
        -------
        iterator of str
            The lines of [`to_text`][], without terminators.
        """
        return iter(self.to_text(on_error=on_error, **fields).splitlines())


def from_bids(source: tx.Any) -> Metadata:
    """
    Read a BIDS JSON sidecar into generic metadata.

    Parameters
    ----------
    source : mapping or str or PathLike or file object
        The sidecar, as a decoded JSON object, a JSON string, a path, or
        an open file.

    Returns
    -------
    Metadata
        The vocabulary fields that the sidecar names, and its other keys
        in `extra`.
    """
    return Metadata(**_decode(_read(source)))


@takes_error_policy
def to_bids(
    metadata: Metadata,
    *,
    on_error: ErrorPolicy = UNSET,
) -> tx.Dict[str, tx.Any]:
    """
    Write metadata as the JSON object of a BIDS sidecar.

    Parameters
    ----------
    metadata : Metadata
        The metadata to write. Only its vocabulary and `extra` are used.
    on_error : ErrorPolicy, optional
        What to do with the fields a sidecar cannot hold (the diffusion
        fields, an encoding direction along no voxel axis), and with any
        other error (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]).
        By default, an error is raised, and a loss follows the policy in
        effect. A report, `ConversionReport`, is filled instead of
        warning or raising.

    Returns
    -------
    dict
        A JSON object that can be serialised.

    Raises
    ------
    MetadataLossError
        If a field is reported under the `"raise"` policy.
    """
    report = ConversionReport()
    sidecar = BidsSidecar.from_instance(
        metadata, on_error={MetadataLossError: report}
    )
    record = sidecar.consolidate(on_error={MetadataLossError: report}).raw
    apply_loss_policy(report, on_error, stacklevel=2)
    return {} if record is None else record


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _decode(raw: tx.Optional[tx.Mapping[str, tx.Any]]) -> tx.Dict[str, tx.Any]:
    """Return the fields of the vocabulary that a JSON object holds."""
    if raw is None:
        return {}
    values, extra = decode_object(raw, _KEYED_FIELDS)
    values["extra"] = extra
    return values


def _encode(
    raw: tx.Dict[str, tx.Any],
    changed: tx.Dict[str, tx.Any],
    report: ConversionReport,
) -> tx.Dict[str, tx.Any]:
    """Write the changes into a copy of a JSON object.

    Each change that a sidecar can hold is written and taken from
    `changed`. An encoding direction along no voxel axis has no BIDS
    string, so it is left in `changed`, and the key is removed.
    """
    if "extra" in changed:
        encode_extra(raw, changed.pop("extra"), _KEYED_FIELDS, report=report)
    writable = {}
    for name in _KEYED_FIELDS:
        if name not in changed:
            continue
        value = changed[name]
        if isinstance(value, EncodingDirection) and value.to_bids() is None:
            encode_changes(raw, {name: None}, report=report)
            continue
        writable[name] = changed.pop(name)
    encode_changes(raw, writable, report=report)
    return raw


def _read(source: tx.Any) -> tx.Dict[str, tx.Any]:
    """Return the JSON object of a sidecar given in any form."""
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
