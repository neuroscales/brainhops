"""The base of the metadata of file formats, and its consolidation."""

__all__ = [
    "MetadataFormat",
    "consolidated",
    "lost_fields",
    "unsupported_values",
]

import math

import typing_extensions as tx
from bagof.magic import NoRepr, replace

from brainhops._core.compare import differs
from brainhops.datamodel.metadata import ConversionReport, Metadata
from brainhops.datamodel.metadata._report import (
    OnLoss,
    apply_loss_policy,
    short,
)
from brainhops.datamodel.metadata._vocabulary import FIELDS
from brainhops.io.base._base import (
    Format,
    _FileBasedModelMixin,
    format_registry,
)
from brainhops.io.base.specs import register_parser

# An encoder of the vocabulary of a format: `encode(record, changed,
# report)` writes into `record`, which is already a copy, every field of
# `changed` that the format can store, pops each of them from
# `changed`, and returns the record.
_Encoder = tx.Callable[
    [tx.Any, tx.Dict[str, tx.Any], ConversionReport], tx.Any
]


@register_parser(Metadata)
@format_registry(isolated=True)
class MetadataFormat(_FileBasedModelMixin, Metadata, Format, eq=False):
    """Format dispatcher and common base for the metadata of files.

    The metadata of a file is what its header records beside the data. Each
    file format that has a header defines a subclass named after the format,
    such as `NiftiMetadata`, which holds the header in `raw` as the format
    stores it. `MetadataFormat.load` reads the header of a file in any
    registered format, and the same method of a subclass reads a file of
    its own format. In both cases, the data that follows the header is not
    read.

    The header held in `raw` is called the record of the file. The record
    describes the file as it was read, or as the object that holds it was
    last consolidated. A record is never changed in place: an object that
    encodes a new header, with its `consolidate` method or when it is
    written, encodes it into a copy and holds that copy in new metadata.
    Metadata keeps no open file and no reference to the data, so it is
    cheap to copy and it can be pickled.

    The metadata of a format also holds the common vocabulary of
    [`Metadata`][brainhops.datamodel.metadata.Metadata]. A format decodes
    every field that it can store from the record when the metadata is
    built with `from_raw`, and a field that the format has no slot for
    holds `None`. The fields that a format can store are listed in its
    constant `SUPPORTED_FIELDS`. A field can be assigned whatever the
    format supports: `consolidate` encodes into a copy of the record the
    fields that differ from a fresh decode of the record, and reports
    those that the format could not store.

    Images and transformations share their metadata with their copies,
    and metadata can be changed in place, so a change made through one
    copy is seen by the others. To change the metadata of one copy only,
    assign it new metadata, such as
    `image.metadata = replace(image.metadata, description="smoothed")`.
    Metadata compares by identity.

    The metadata registry is separate from the registry of
    [`Format`][], so that [`load`][brainhops.io.load] never returns
    metadata when it reads an image or a transformation. A concrete
    subclass is added to the metadata registry with `@register_format`.
    """

    SUPPORTED_FIELDS: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """The fields of the metadata that the format can store.

    The names are those of the vocabulary, and `"extra"` when the format
    has a free-form store. A field that is not listed is decoded as
    `None`, and its value is reported as lost when the metadata is
    consolidated. A conversion into the format reads this list to
    report a lower bound of what a write would lose. The base class
    lists no field.
    """

    raw: NoRepr[tx.Any] = None
    """The header of the file as the format stores it, or `None`.

    The type of the record depends on the format, and only the format
    reads it. The record is hidden from the representation of the
    metadata, because a header is usually long.
    """

    @classmethod
    def from_raw(cls, raw: tx.Any, **kwargs) -> tx.Self:
        """Build the metadata that holds a record.

        A reader calls this method once it has parsed the header of a file.
        The record is held as it is given, without a copy. The format
        decodes the fields of the vocabulary from the record here, and the
        base class, which knows no record, decodes none.

        Parameters
        ----------
        raw : object
            The header of a file, in the representation of the format.
        **kwargs : Any
            Other fields of the metadata, which take precedence over the
            decoded values.

        Returns
        -------
        MetadataFormat
            The metadata.
        """
        return cls(raw=raw, **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build metadata from a file, from other metadata, or from a value.

        Metadata of any class, generic or of another format, is copied
        field by field with [`from_instance`][]. A file is read as
        `_FileBasedModelMixin.from_any` reads it, and any other value is
        passed to the data model.

        Parameters
        ----------
        other : object
            A file, its content, metadata, or a mapping of fields.
        *args : Any
            Constructor arguments.
        **kwargs : Any
            Fields of the metadata, or the options of the reader for a
            file.

        Returns
        -------
        MetadataFormat
            The metadata.
        """
        if isinstance(other, Metadata):
            return cls.from_instance(other, *args, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(
        cls,
        other: tx.Any,
        *args: tx.Any,
        on_loss: tx.Optional[OnLoss] = None,
        **kwargs: tx.Any,
    ) -> tx.Self:
        """Copy metadata into this format, field by field.

        Every field of the vocabulary and `extra` are copied by name,
        including those that this format cannot store, which are dropped
        and reported only when the metadata is consolidated. The record
        of another format means nothing to this one, so it is copied only
        from metadata of this class. The fields that are set and that
        `SUPPORTED_FIELDS` does not list are reported here as lost,
        because a write will drop them, so the report is a lower bound of
        what a write loses.

        Parameters
        ----------
        other : object
            The metadata to copy.
        *args : Any
            Constructor arguments, which take precedence over the fields
            of `other`.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the fields that this format cannot store. By
            default, the policy in effect (see
            [`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy]).
            A report is filled instead of warning or raising.
        **kwargs : Any
            Fields to set on the result.

        Returns
        -------
        MetadataFormat
            The copy.

        Raises
        ------
        MetadataLossError
            If a field is reported under the `"raise"` policy.
        """
        obj = super().from_instance(other, *args, **kwargs)
        report = ConversionReport(
            source=type(other).__name__, target=cls.__name__
        )
        report.lost.update(unsupported_values(cls, obj))
        apply_loss_policy(report, on_loss, stacklevel=2)
        return obj

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the fields of the metadata into its record.

        The base class knows no record, so it encodes nothing: a field
        that differs from a fresh decode of the record, which gives
        `None` for every field, cannot be stored, and is reported as
        lost. A format that stores fields overrides this method, and
        encodes them with [`consolidated`][] and an encoder of its own.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this metadata rather than a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the fields that the format cannot store. By
            default, the policy in effect.
        **fields : Any
            Fields of the metadata to override before the encoding, by
            name.

        Returns
        -------
        Self
            The consolidated copy, or this metadata with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If a field is reported under the `"raise"` policy.
        """
        return consolidated(
            self, None, None, inplace=inplace, on_loss=on_loss, fields=fields
        )


# ----------------------------------------------------------------------
#   HELPERS OF THE FORMATS
# ----------------------------------------------------------------------


def consolidated(
    metadata: MetadataFormat,
    encode: tx.Optional[_Encoder],
    new_record: tx.Optional[tx.Callable[[], tx.Any]],
    *,
    inplace: bool,
    on_loss: tx.Optional[OnLoss],
    fields: tx.Mapping[str, tx.Any],
) -> tx.Any:
    """Return metadata whose record holds its fields.

    This function is the body of `consolidate` for every metadata
    format, which passes its own encoder. It follows the contract of
    consolidation of the format records.

    1. The fields given in `fields` replace those of the metadata.
    2. The fields that differ from a fresh decode of the record (the
       metadata that `from_raw` builds from it) are the changes. With
       none, the result shares the record, and metadata without a record
       and without a field set stays without a record.
    3. The changes are encoded into a copy of the record, or into a new,
       empty record when there is none. The encoder takes the name of
       each change that it writes, and the changes that are left are
       reported as lost. A change that the record holds only
       approximately is reported as approximated.
    4. The result is decoded afresh from the new record, so a second
       consolidation changes nothing.

    Parameters
    ----------
    metadata : MetadataFormat
        The metadata to consolidate.
    encode : callable, optional
        The encoder of the format, `encode(record, changed, report)`. It
        writes into `record`, which is a copy, the changes that the
        format can store, pops each of them from `changed`, reports what
        it stores approximately, and returns the record. `None` stores
        nothing.
    new_record : callable, optional
        A function that returns an empty record of the format, for
        metadata without a record. With `None`, such metadata stores
        nothing.
    inplace : bool
        Whether to change `metadata` rather than a copy.
    on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
        What to do with what was not stored. By default, the policy in
        effect.
    fields : mapping
        Fields to override before the encoding.

    Returns
    -------
    MetadataFormat
        The consolidated copy, or `metadata` itself with `inplace=True`.

    Raises
    ------
    TypeError
        If `fields` names no field of the metadata.
    MetadataLossError
        If something is reported under the `"raise"` policy.
    """
    unknown = sorted(set(fields) - set(FIELDS))
    if unknown:
        raise TypeError(
            f"{type(metadata).__name__}.consolidate() got unknown field(s): "
            f"{', '.join(unknown)}"
        )
    cls = type(metadata)
    source = replace(metadata, **fields) if fields else metadata
    fresh = cls.from_raw(source.raw)
    changed = {
        name: getattr(source, name)
        for name in FIELDS
        if _differs(getattr(source, name), getattr(fresh, name))
    }
    if not changed:
        if inplace:
            for name in fields:
                setattr(metadata, name, getattr(source, name))
            return metadata
        return source if fields else replace(metadata)
    report = ConversionReport(source=cls.__name__, target=cls.__name__)
    record = source.raw
    remaining = dict(changed)
    if encode is not None:
        if record is not None:
            record = record.copy()
        elif new_record is not None:
            record = new_record()
        if record is not None:
            record = encode(record, remaining, report)
    lost_fields(remaining, report)
    decoded = cls.from_raw(record)
    for name, value in changed.items():
        if name in remaining or name in report.approximated:
            continue
        held = getattr(decoded, name)
        if not _agrees(held, value):
            report.approximated[name] = f"the record holds {short(held)}"
    apply_loss_policy(report, on_loss, stacklevel=3)
    if inplace:
        metadata.raw = record
        for name in FIELDS:
            setattr(metadata, name, getattr(decoded, name))
        return metadata
    return replace(
        source,
        raw=record,
        **{name: getattr(decoded, name) for name in FIELDS},
    )


def lost_fields(
    remaining: tx.Mapping[str, tx.Any], report: ConversionReport
) -> ConversionReport:
    """Report as lost the changes that no slot of a record took.

    Parameters
    ----------
    remaining : mapping
        The changes that the encoder of a format left: field name to the
        value that it could not store.
    report : ConversionReport
        The report to fill.

    Returns
    -------
    ConversionReport
        `report` itself.
    """
    for name, value in remaining.items():
        report.lost[name] = value
    return report


def unsupported_values(
    cls: tx.Type[MetadataFormat], metadata: Metadata
) -> tx.Dict[str, tx.Any]:
    """Return the fields of metadata that a format declares no slot for.

    This function is the one reader of `SUPPORTED_FIELDS`. A field
    counts when its value is set, which means not `None`, and not empty
    for `extra`.

    Parameters
    ----------
    cls : type
        The metadata class of a format.
    metadata : Metadata
        The metadata to check.

    Returns
    -------
    dict
        Field name to value, for each field that is set and that
        `cls.SUPPORTED_FIELDS` does not list.
    """
    supported = frozenset(cls.SUPPORTED_FIELDS)
    values = {}
    for name in FIELDS:
        value = getattr(metadata, name, None)
        if name in supported or value is None:
            continue
        if name == "extra" and not value:
            continue
        values[name] = value
    return values


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _differs(value: tx.Any, decoded: tx.Any) -> bool:
    """Return whether a field differs from its decoded value.

    `None` is compared by identity, because NumPy reads `None` as the
    default data type, so that `np.dtype("float64") == None` holds.
    """
    if value is None or decoded is None:
        return value is not decoded
    return differs(value, decoded)


def _agrees(held: tx.Any, given: tx.Any) -> bool:
    """Return whether a record holds a value, up to single precision.

    Numbers, and sequences of numbers element by element, agree within
    the rounding of single precision, which is how most headers store
    them. Other values agree when they are equal.
    """
    if _is_number(held) and _is_number(given):
        return math.isclose(held, given, rel_tol=1e-6, abs_tol=1e-9)
    if (
        isinstance(held, (tuple, list))
        and isinstance(given, (tuple, list))
        and len(held) == len(given)
        and all(_is_number(v) for v in (*held, *given))
    ):
        return all(_agrees(h, g) for h, g in zip(held, given))
    return not _differs(held, given)


def _is_number(value: tx.Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
