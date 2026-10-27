"""The metadata of an X5 file."""

__all__ = ["X5Metadata"]

# dependencies
import h5py
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import takes_error_policy
from brainhops.datamodel.metadata._vocabulary import VOCABULARY
from brainhops.errors import UNSET, ErrorAction, ErrorPolicy
from brainhops.io.base._base import register_format
from brainhops.io.common.hdf5 import Hdf5ReaderWriter
from brainhops.io.metadata import MetadataFormat
from brainhops.io.metadata._base import consolidated
from brainhops.io.metadata._json import (
    decode_object,
    encode_changes,
    encode_extra,
)

# this format
from ._raw import X5Raw


@register_format
class X5Metadata(Hdf5ReaderWriter, MetadataFormat):
    """Metadata of an X5 file, which holds the file as an [`X5Raw`][].

    An X5 file has no header that precedes its data. Its record holds the
    root attributes, the chains and every transform of the file, with the
    JSON metadata of each transform. The matrices are read at once, and the
    fields are held as lazy arrays that read the file again by its name
    when they are used, so reading the metadata never reads a field. An
    [`X5Transform`][brainhops.io.transformations.x5.X5Transform] decodes its
    chain of transformations from the record of its metadata.

    The file is read and written by the record: [`from_h5`][] reads an
    [`X5Raw`][] and wraps it, and [`to_h5`][] writes the consolidated
    record, with the fields copied from their file. The metadata keeps no
    open file.

    The vocabulary is stored in the JSON `Metadata` attribute of a
    transform, each field under its BIDS key, or under its name in
    `CamelCase` when BIDS has none (`History`, `Moving`, `InputSpace`),
    and the other keys of the JSON object are `extra`. A JSON object
    stores anything, so `SUPPORTED_FIELDS` lists every field. The
    transform that holds the vocabulary of the file is the single
    transform of its first chain, or the single transform of a file
    without chains. Composition does not merge metadata, so a file whose
    first chain has several transforms has no vocabulary of its own:
    its fields decode as `None`, and a field written to it is reported
    as lost. `Domain.Coordinates` is not decoded, because nitransforms
    writes the kind of coordinates there (`"cartesian"`) and not the
    label of a space.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".x5",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("x5", "bids")
    SUPPORTED_FIELDS: tx.ClassVar[tx.Tuple[str, ...]] = ("extra",) + VOCABULARY
    """Every field of the vocabulary, and `extra`."""

    raw: NoRepr[tx.Optional[X5Raw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def from_raw(
        cls, raw: tx.Optional[X5Raw], **kwargs: tx.Any
    ) -> "X5Metadata":
        """Build the metadata of a record, decoding its vocabulary.

        Parameters
        ----------
        raw : X5Raw, optional
            The record, held as it is given.
        **kwargs : Any
            Other fields, which take precedence over the decoded values.

        Returns
        -------
        X5Metadata
            The metadata.
        """
        return cls(raw=raw, **{**_decode(raw), **kwargs})

    @takes_error_policy
    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_error: ErrorPolicy = UNSET,
        on_loss: ErrorAction = UNSET,
        **fields: tx.Any,
    ) -> "X5Metadata":
        """Encode the fields that differ from the record into a copy of it.

        The contract is that of
        [`MetadataFormat.consolidate`][brainhops.io.metadata.MetadataFormat.consolidate]:
        the changes are written into the JSON metadata of the transform
        of a copy of the record, and the result is decoded from it
        afresh. A record without such a transform stores nothing, and a
        transform is never made up to hold the fields.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this metadata rather than a copy.
        on_error : ErrorPolicy, optional
            What to do with the fields that the record cannot hold, and
            with any other conversion error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        on_loss : ErrorAction, optional
            What to do with a loss of metadata, with the same values as a
            single action of `on_error`, over which it takes precedence for
            a loss. By default, a loss falls under `on_error`.
        **fields : Any
            Fields to override before the encoding.

        Returns
        -------
        X5Metadata
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
            None,
            inplace=inplace,
            on_error=on_error,
            fields=fields,
        )

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score an open HDF5 file by its root `Format` attribute.

        The file is scored by [`X5Raw.sniff_h5`][].

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an X5 file.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return X5Raw.sniff_h5(h5file, error=error)

    @classmethod
    def from_h5(cls, h5file: h5py.File, load: bool = False) -> "X5Metadata":
        """Read the metadata of an open X5 file.

        The record is read by [`X5Raw.from_h5`][].

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        load : bool, default=False
            Whether to read the fields into memory at once, rather than
            when they are used.

        Returns
        -------
        X5Metadata
            The metadata.
        """
        return cls.from_raw(X5Raw.from_h5(h5file, load=load))

    @takes_error_policy
    def to_h5(
        self,
        h5file: h5py.File,
        *,
        on_error: ErrorPolicy = UNSET,
        on_loss: ErrorAction = UNSET,
        **fields: tx.Any,
    ) -> None:
        """Write the consolidated record into an empty HDF5 file.

        The metadata is consolidated first, with the overrides given, so a
        field that was changed is written. Metadata without a record is
        written as a new, empty record, which is an X5 file without
        transforms.

        Parameters
        ----------
        h5file : h5py.File
            The empty file, open for writing.
        on_error : ErrorPolicy, optional
            What to do with the fields that the record cannot hold, and
            with any other conversion error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        on_loss : ErrorAction, optional
            What to do with a loss of metadata, with the same values as a
            single action of `on_error`, over which it takes precedence for
            a loss. By default, a loss falls under `on_error`.
        **fields : Any
            Fields to override, as in [`consolidate`][].
        """
        raw = self.consolidate(on_error=on_error, **fields).raw
        (X5Raw() if raw is None else raw).to_h5(h5file)


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _vocabulary_node(raw: tx.Optional[X5Raw]) -> tx.Optional[int]:
    """Return the index of the transform that holds the vocabulary.

    It is the single transform of the first chain, or the single
    transform of a file without chains, and `None` otherwise.
    """
    if raw is None or not raw.nodes:
        return None
    chains = raw.header.chains
    if chains:
        first = chains[0]
        return int(first[0]) if len(first) == 1 else None
    return 0 if len(raw.nodes) == 1 else None


def _decode(raw: tx.Optional[X5Raw]) -> tx.Dict[str, tx.Any]:
    """Return the fields that the JSON metadata of the transform holds."""
    index = _vocabulary_node(raw)
    if index is None:
        return {}
    json = raw.nodes[index].metadata
    if not isinstance(json, dict):
        return {}
    values, extra = decode_object(json, VOCABULARY)
    values["extra"] = extra
    return values


def _encode(
    raw: X5Raw, changed: tx.Dict[str, tx.Any], report: ConversionReport
) -> X5Raw:
    """Write the changed fields into the JSON metadata of the transform.

    The record is already a copy, whose nodes are copies. Without a
    transform that holds the vocabulary, every field is left in
    `changed`, so that it is reported as lost.
    """
    index = _vocabulary_node(raw)
    if index is None:
        return raw
    node = raw.nodes[index]
    json = node.metadata
    if isinstance(json, str) and json:
        report.approximated["extra"] = (
            "the Metadata of the transform was not JSON, and is replaced"
        )
    json = dict(json) if isinstance(json, dict) else {}
    if "extra" in changed:
        encode_extra(json, changed.pop("extra"), VOCABULARY, report=report)
    values = {
        name: changed.pop(name) for name in VOCABULARY if name in changed
    }
    encode_changes(json, values, report=report)
    node.metadata = json or None
    return raw
