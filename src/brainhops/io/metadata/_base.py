"""
`FileBasedMetadata`: the metadata of a file format, its hooks, and the
dispatcher among the formats whose files hold metadata.

It lives in `brainhops.io`, next to `FileBasedImage`, because it derives
from the dispatcher of the formats; the format-agnostic `Metadata` it
extends lives in `brainhops.datamodel.metadata`, which never imports it.
"""

__all__ = ["FileBasedMetadata", "OpaqueMetadata"]

# stdlib
import copy
import math

# externals
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr

# internals
from brainhops._core.compare import differs
from brainhops.datamodel.enums import AxisType
from brainhops.datamodel.metadata._base import (
    FIELDS,
    Metadata,
    _format_name,
    _History,
)
from brainhops.datamodel.metadata._report import (
    ConversionReport,
    OnLoss,
    apply_loss_policy,
    short,
)
from brainhops.datamodel.metadata._sentinel import UNSUPPORTED
from brainhops.io.base._base import FormatDispatcher, format_registry
from brainhops.io.base.parsers import ParserNotImplementedError

RawT = tx.TypeVar("RawT")
"""The type of the raw record of a format (see `FileBasedMetadata`)."""


@format_registry
class FileBasedMetadata(FormatDispatcher, Metadata, tx.Generic[RawT]):
    """
    The metadata of a file format: the common vocabulary, decoded from and
    encoded into the format's own raw record.

    This is the base of every `<Fmt>Metadata`, as `FileBasedImage` is the
    base of the image class of every format. A format declares what it
    can store with `supports=`, declares the type of its raw record as
    the type argument of its base (`FileBasedMetadata[nb.Nifti1Header]`;
    `FileBasedMetadata[None]` for a format without one), and decodes and
    encodes the record with the hooks described in the format author's
    guide (`docs/dev/metadata-formats.md`). The record itself and the
    read-time snapshot are fields of
    [`Metadata`][brainhops.datamodel.metadata.Metadata], so that generic
    metadata carries them through a conversion; reading and writing them
    is what this class adds.

    A new, empty record is the type called without arguments, so the
    type of a record must build one that way.

    The class is also the dispatcher of the formats whose files hold
    metadata: `FileBasedMetadata.load(path)` picks the registered format
    that best matches the file. The class of such a format lists its
    parser (a
    [`MetadataParser`][brainhops.io.base._metadata_parser.MetadataParser])
    first among its bases, and registers with
    [`register_format`][brainhops.io.base.register_format]. The registry
    is separate from that of
    [`FileBasedObject`][brainhops.io.base.FileBasedObject], which this
    class is not, so that `brainhops.io.load` never returns metadata
    where an image or a transformation was asked for.
    """

    raw: tx.Annotated[
        tx.Optional[RawT],
        tx.Doc(
            """
            The raw record of the file the metadata was read from, of the
            type the format declares, or `None` for metadata built in
            memory. Edit it only for what the vocabulary does not cover:
            on write, a field left untouched keeps the value of the
            record, and a field that was set wins over it.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    # --- reading ------------------------------------------------------

    @classmethod
    def load(cls, file: tx.Any, **kwargs: tx.Any) -> "Metadata":
        """
        Read the metadata of a file, without reading its data.

        On `FileBasedMetadata` itself, the dispatcher, the format is found
        among the registered formats whose files hold metadata (NIfTI,
        MGH, plain Zarr and OME-Zarr, x5, ITK `.h5`, and BIDS JSON
        sidecars), by the name of the file and by its content, as
        `brainhops.io.load` finds the format of an image; `hint=`
        restricts the candidates. On the class of a format, the file is
        read as a file of that format, by its parser (`MetadataParser`),
        which comes first among its bases; the class of a format whose
        files hold no metadata (FLIRT, ITK `.tfm` and `.mat`) refuses.
        Only the raw record is read: a NIfTI header, the footer and the
        tags of an MGH file, the attributes of a Zarr node, the JSON of
        an x5 node.

        Parameters
        ----------
        file : str, path-like, file object or bytes
            The file, or the Zarr store.
        **kwargs
            Options of the reader of the format, and, on the dispatcher,
            `hint=` (a format name such as `"nifti"`, or several).

        Returns
        -------
        Metadata
            The metadata of the file, with its raw record: the metadata
            of its format, or generic metadata for a BIDS sidecar.
            Convert it with `to(Metadata)` for generic metadata (which
            keeps the record).

        Raises
        ------
        ParserContentError
            On the dispatcher, if no registered format reads the file.
        ParserNotImplementedError
            On the class of a format whose files hold no metadata.
        ParserExistsError
            If the file does not exist.

        Examples
        --------
        ```python
        meta = FileBasedMetadata.load("sub-01_bold.nii.gz")
        meta.repetition_time  # read from the header alone
        FileBasedMetadata.load("sub-01_bold.json").extra["TaskName"]
        FileBasedMetadata.load("scan.mgz", hint="mgh")
        NiftiMetadata.load("sub-01_bold.nii.gz")  # as a NIfTI file
        ```
        """
        return super().load(file, **kwargs)

    @classmethod
    def from_raw(
        cls, raw: tx.Any, *, image: tx.Any = None, **values: tx.Any
    ) -> tx.Self:
        """
        Build the metadata of a raw record that was just read.

        The raw record is decoded into the common fields by the format's
        `_decode_raw` hook. The decoded values are kept as the read-time
        snapshot, so that a field the user leaves untouched keeps the
        value of the record when the record is written back.

        Parameters
        ----------
        raw : object
            The raw record, of the type the format declares for `raw`.
        image : object, optional
            The image or transformation the record was read with, for the
            fields that the format decodes from the data model.
        **values
            Fields to set over the decoded values. They count as changes
            on write.

        Returns
        -------
        FileBasedMetadata
            The metadata, with `raw` set to the record.

        Raises
        ------
        TypeError
            If `_decode_raw` returned a value for a field that this class does
            not support. The raw record belongs to the format, so this
            error never comes from the data: it reveals a format class
            whose `_decode_raw` disagrees with its `supports=` declaration,
            and which would otherwise drop the value without a report.
        """
        decoded = cls._checked_decode_raw(raw, image)
        obj = cls(raw=raw, **decoded)
        # Snapshot the *converted* values, so that a decoded list held as
        # a tuple does not count as a change.
        obj._snapshot = {
            key: copy.deepcopy(getattr(obj, key)) for key in decoded
        }
        for key, value in values.items():
            setattr(obj, key, value)
        return obj

    def update_from_raw(self, raw: tx.Any, *, image: tx.Any = None) -> tx.Self:
        """
        Build the metadata of another raw record, keeping the changes made
        to this metadata.

        The new record is decoded as `from_raw` decodes it. Every field
        that changed in this object since it was read is then set over the
        decoded values, and counts as a change. For an object built in
        memory, every field that is set counts as changed. The keys of
        `extra` are merged one by one. This is how an image that is given
        a new raw record (`replace(image, header=...)`) keeps the metadata
        edits it carried.

        Parameters
        ----------
        raw : object
            The new raw record.
        image : object, optional
            The image or transformation the record belongs to.

        Returns
        -------
        FileBasedMetadata
            A new metadata object of the same class.
        """
        changed = {
            key: value
            for key, value in self._changed_fields().items()
            if key in type(self).supported_fields
        }
        extra = changed.pop("extra", None)
        obj = type(self).from_raw(raw, image=image)
        for key, value in changed.items():
            setattr(obj, key, value)
        if extra:
            obj.extra = _apply_diff(obj.extra or {}, extra)
        return obj

    # --- the change-detecting write -----------------------------------

    def _changed_fields(self) -> tx.Dict[str, tx.Any]:
        """
        The common fields that differ from the read-time snapshot.

        An object read from a file and left untouched has none. An object
        built in memory has no snapshot, so every field that is not
        `None` is a change. A field set to `None` after a read is a change
        to `None` (the record's slot is cleared on write). `extra` is
        compared key by key: its entry is the per-key diff, where `None`
        removes a key.
        """
        snapshot = self._snapshot
        changed: tx.Dict[str, tx.Any] = {}
        for name in FIELDS:
            value = getattr(self, name, None)
            if value is UNSUPPORTED:
                continue
            before = snapshot.get(name)
            if name == "extra":
                diff = _extra_diff(before, value)
                if diff:
                    changed["extra"] = diff
            elif differs(value, before):
                changed[name] = value
        return changed

    def to_raw(
        self,
        *,
        image: tx.Any = None,
        on_loss: tx.Optional[OnLoss] = None,
    ) -> tx.Any:
        """
        Encode the common fields into a new raw record, and return it.

        This is the counterpart of `from_raw`: the record is a copy of
        `raw` (or a default record, for metadata built in memory), with
        the fields that changed since the read encoded over it. It is what
        a metadata writer writes (see `MetadataParser.to_file`). A writer
        that builds its own fresh record calls `update_raw` instead.

        Parameters
        ----------
        image : object, optional
            The image or transformation the metadata belongs to.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the losses. By default, the policy in effect.

        Returns
        -------
        object
            The encoded record. `raw` itself is left unchanged.

        Raises
        ------
        MetadataLossError
            If something is lost under the `"raise"` policy.
        """
        return self.update_raw(None, image=image, on_loss=on_loss)

    def update_raw(
        self,
        raw: tx.Any = None,
        *,
        image: tx.Any = None,
        on_loss: tx.Optional[OnLoss] = None,
        force: tx.Collection[str] = (),
    ) -> tx.Any:
        """
        Encode the common fields over a raw record, and return the record.

        Only the fields that changed since the read are encoded, so that a
        field the user did not touch keeps the value that the record holds.
        The fields named in `force` are encoded whether they changed or
        not: a writer uses `force` for a keyword argument that must win
        over the record, such as the `tr=` option of the MGH writer, and a
        forced `None` clears the slot.

        A field that the format does not support, but that was assigned
        after construction, is reported as lost, as are the
        value-dependent losses found by `_encode_raw`. What the writer
        then changes in the record (a slot that the data model owns, such
        as the time step of a NIfTI image) is found by `check_raw`, which
        the writer calls once the record is finished.

        Parameters
        ----------
        raw : object, optional
            The record to encode over. A writer passes its own fresh
            record, already filled with what it keeps from `self.raw`.
            By default, a copy of `self.raw` is used, or a default record
            when there is none.
        image : object, optional
            The image or transformation being written.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the losses. By default, the policy in effect
            (see
            [`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy]).
            A writer passes the report of its whole write, which it then
            acts on once.
        force : collection of str, optional
            The fields to encode even when they did not change.

        Returns
        -------
        object
            The encoded record.

        Raises
        ------
        MetadataLossError
            If something is lost under the `"raise"` policy.
        """
        sink = isinstance(on_loss, ConversionReport)
        if sink:
            report = on_loss
        else:
            report = ConversionReport(source=self.format, target=self.format)
        unsupported = type(self).unsupported_fields
        for name in unsupported:
            value = getattr(self, name, None)
            if value is not None and value is not UNSUPPORTED:
                report.lost[name] = value
        if raw is None:
            raw = self._raw_or_default()
        changed = {
            key: value
            for key, value in self._changed_fields().items()
            if key not in unsupported
        }
        for name in force:
            if name in FIELDS and name != "extra":
                if name not in unsupported:
                    changed[name] = getattr(self, name)
        raw = self._encode_raw(raw, changed, image=image, report=report)
        if not sink:
            apply_loss_policy(report, on_loss, stacklevel=2)
        return raw

    def check_raw(
        self,
        raw: tx.Any,
        *,
        image: tx.Any = None,
        on_loss: tx.Optional[OnLoss] = None,
    ) -> ConversionReport:
        """
        Check that a finished record holds the fields that changed.

        A writer encodes the fields with `update_raw`, then may set slots
        of the record from the data model, whatever the metadata says: the
        time step of a NIfTI image, the data type of the stored values.
        Once the record is finished, it calls this method, which decodes
        the record as a reader would (`from_raw`) and reports as
        approximated every changed field whose value the record does not
        hold. A field already reported (lost or approximated), an
        unsupported field and `extra` are not checked again.

        Parameters
        ----------
        raw : object
            The finished record.
        image : object, optional
            The image or transformation being written, as a reader would
            be given it.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the losses. By default, the policy in effect.
            A writer passes the report of its whole write.

        Returns
        -------
        ConversionReport
            The report, `on_loss` itself when it is one.

        Raises
        ------
        MetadataLossError
            If something is approximated under the `"raise"` policy.
        """
        sink = isinstance(on_loss, ConversionReport)
        if sink:
            report = on_loss
        else:
            report = ConversionReport(source=self.format, target=self.format)
        decoded = type(self).from_raw(raw, image=image)
        unsupported = type(self).unsupported_fields
        for name, value in self._changed_fields().items():
            if (
                name == "extra"
                or name in unsupported
                or name in report.lost
                or name in report.approximated
            ):
                continue
            held = getattr(decoded, name)
            if not _agrees(held, value):
                report.approximated[name] = f"the record holds {short(held)}"
        if not sink:
            apply_loss_policy(report, on_loss, stacklevel=2)
        return report

    @classmethod
    def writable(
        cls, metadata: Metadata
    ) -> tx.Tuple["FileBasedMetadata", ConversionReport]:
        """
        Prepare metadata for a writer of this format.

        Metadata that is already of this class is returned as it is.
        Metadata of another class is converted into this class, and the
        losses of the conversion seed the report. The loss policy is not
        applied here: the writer encodes the fields into the same report
        (`update_raw(..., on_loss=report)`) and applies the policy once, so
        that one write gives one report and at most one warning.

        Parameters
        ----------
        metadata : Metadata
            The metadata of the object being written.

        Returns
        -------
        metadata : FileBasedMetadata
            The metadata to write, of this class.
        report : ConversionReport
            The report of the write so far.
        """
        if isinstance(metadata, cls):
            target = _format_name(cls)
            return metadata, ConversionReport(
                source=metadata.format, target=target
            )
        return cls._convert_from(metadata)

    def check_writable(
        self, *, image: tx.Any = None, raw: tx.Any = None
    ) -> ConversionReport:
        """
        Report what a write of this metadata would lose, without writing
        anything.

        The class declarations only give a lower bound of what is lost.
        This method runs the encoder on a scratch record, so that the
        losses that depend on the values, such as an over-long description
        or an irregular slice timing, are included as well. A format whose
        writer starts from another record than a copy of `raw` overrides
        this method, builds that record, and calls `super()` with it.

        Parameters
        ----------
        image : object, optional
            The image or transformation that would be written, for the
            fields that depend on the data model.
        raw : object, optional
            The record to encode over, which may be edited. By default, a
            copy of `raw`, or a new, empty record when there is none.

        Returns
        -------
        ConversionReport
            What would be lost or approximated.
        """
        report = ConversionReport(source=self.format, target=self.format)
        raw = self.update_raw(raw, image=image, on_loss=report)
        self.check_raw(raw, image=image, on_loss=report)
        return report

    # --- the record of a conversion -----------------------------------

    @classmethod
    def _raw_type(cls) -> tx.Optional[type]:
        """
        The type of raw record that this class declares, as the type
        argument of its base (`FileBasedMetadata[T]`).

        Returns
        -------
        type or None
            The declared type, `type(None)` for a format without a record,
            or `None` when the class does not declare a type.
        """
        return cls._raw_class

    @classmethod
    def _accepts_raw(cls, raw: tx.Any) -> bool:
        """
        Whether a conversion into this class keeps a raw record: only
        when the record is of the type this class declares. Formats
        declare distinct types, so a record only ever goes back to its
        own format; a class that declares no type keeps none.

        Parameters
        ----------
        raw : object
            The raw record of the source metadata.

        Returns
        -------
        bool
            Whether the record is kept.
        """
        declared = cls._raw_type()
        return declared is not None and isinstance(raw, declared)

    # --- per-format hooks ---------------------------------------------

    @classmethod
    def _decode_raw(cls, raw: tx.Any, *, image: tx.Any = None) -> tx.Dict:
        """
        Decode a raw record into common fields.

        A format overrides this hook. The default decodes nothing.

        Parameters
        ----------
        raw : object
            The raw record.
        image : object, optional
            The image or transformation the record was read with.

        Returns
        -------
        dict
            Field name to decoded value. A value that is `None` or
            `UNSUPPORTED` is ignored.
        """
        return {}

    def _encode_raw(
        self,
        raw: tx.Any,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> tx.Any:
        """
        Encode changed common fields into a raw record.

        A format overrides this hook. The default leaves the record
        unchanged.

        Parameters
        ----------
        raw : object
            The record to encode into, which may be edited in place.
        changed : dict
            Field name to new value, for the fields to encode. A `None`
            value clears the slot of the field.
        image : object, optional
            The image or transformation being written.
        report : ConversionReport
            The report to fill with what cannot be stored exactly.

        Returns
        -------
        object
            The encoded record.
        """
        return raw

    # --- internals ----------------------------------------------------

    @classmethod
    def _checked_decode_raw(
        cls, raw: tx.Any, image: tx.Any
    ) -> tx.Dict[str, tx.Any]:
        """`_decode_raw`, without its absent values, and checked against the
        declarations of the class (see `from_raw`)."""
        decoded: tx.Dict[str, tx.Any] = {}
        for key, value in cls._decode_raw(raw, image=image).items():
            if value is None or value is UNSUPPORTED:
                continue
            if key not in cls.supported_fields:
                raise TypeError(
                    f"{cls.__name__}._decode_raw returned {key}={value!r}, "
                    f"but {cls.__name__} does not support {key!r}."
                )
            decoded[key] = value
        return decoded

    def _raw_or_default(self) -> tx.Any:
        """A copy of `raw`, or a new, empty record when there is none."""
        if self.raw is not None:
            return copy.deepcopy(self.raw)
        cls = type(self)._raw_class
        return None if cls is None else cls()

    def _derive_values(
        self,
        *,
        changed: tx.Mapping[AxisType, tx.Any],
        history: _History,
    ) -> tx.Dict[str, tx.Any]:
        # A copy of the raw record and the snapshot are kept, so that a
        # field `_reslice` or `_select` cleared is cleared in the record on
        # write. A format scrubs what else the change invalidates in its
        # own `_reslice` or `_select`, on the copy.
        values = super()._derive_values(changed=changed, history=history)
        values["snapshot"] = copy.copy(self._snapshot)
        values["raw"] = copy.deepcopy(self.raw)
        return values


class OpaqueMetadata(
    FileBasedMetadata[None], on={"format": "opaque"}, supports=()
):
    """
    The metadata of a format that stores none, not even in memory: every
    field is unsupported, and there is no raw record.

    Base class of the formats that store a bare matrix and are given
    nothing else (matrix text, ITK `.tfm`/`.mat`). A read-then-save
    loses nothing, because nothing is there. A format that keeps
    anything, if only in memory (FLIRT `moving`/`fixed`), is a
    `FileBasedMetadata` with a `supports=` list instead. Its `raw` is
    always `None`.
    """

    @classmethod
    def load(cls, file: tx.Any, **kwargs: tx.Any) -> "Metadata":
        """
        Refuse to read the metadata of a file: the files of the format
        hold none.

        Parameters
        ----------
        file : str, path-like or file object
            The file.
        **kwargs
            Ignored.

        Returns
        -------
        Metadata
            Never: the method always raises.

        Raises
        ------
        ParserNotImplementedError
            Always.
        """
        raise ParserNotImplementedError(
            f"{cls.__name__} stores no metadata that can be read on its own."
        )


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _extra_diff(
    before: tx.Optional[tx.Mapping], after: tx.Any
) -> tx.Dict[str, tx.Any]:
    before = dict(before or {})
    after = dict(after or {})
    diff = {
        key: value
        for key, value in after.items()
        if key not in before or differs(value, before[key])
    }
    for key in before:
        if key not in after:
            diff[key] = None
    return diff


def _apply_diff(
    mapping: tx.Mapping[str, tx.Any], diff: tx.Mapping[str, tx.Any]
) -> tx.Dict[str, tx.Any]:
    """A copy of `mapping` with an `extra` diff applied (`None` removes
    a key)."""
    out = dict(mapping)
    for key, value in diff.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def _agrees(value: tx.Any, given: tx.Any) -> bool:
    """Whether a value agrees with what a record holds: numbers, and
    sequences of numbers element by element, within single-precision
    rounding."""
    if _is_number(value) and _is_number(given):
        return math.isclose(value, given, rel_tol=1e-6, abs_tol=1e-9)
    if (
        isinstance(value, (tuple, list))
        and isinstance(given, (tuple, list))
        and len(value) == len(given)
        and all(_is_number(v) for v in (*value, *given))
    ):
        return all(_agrees(v, g) for v, g in zip(value, given))
    return not differs(value, given)


def _is_number(value: tx.Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
