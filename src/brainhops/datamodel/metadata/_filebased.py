"""`FileBasedMetadata`: the metadata of a file format, and its hooks."""

__all__ = ["FileBasedMetadata", "OpaqueMetadata"]

# stdlib
import copy
import math

# externals
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr, fields

# internals
from brainhops._core.compare import differs

from ..enums import AxisType
from ._base import FIELDS, Metadata, _format_name, _History
from ._report import ConversionReport, OnLoss, apply_loss_policy, short
from ._sentinel import UNSUPPORTED


class FileBasedMetadata(Metadata):
    """
    The metadata of a file format: the common vocabulary, decoded from and
    encoded into the format's own raw record.

    This is the base of every `<Fmt>Metadata`, as `FileBasedImage` is the
    base of the image class of every format. A format declares what it
    can store with `supports=`, declares the type of its raw record by
    annotating `raw`, and decodes and encodes the record with the hooks
    described in the format author's guide
    (`docs/dev/metadata-formats.md`). The record itself and the read-time
    snapshot are fields of [`Metadata`][], so that generic metadata
    carries them through a conversion; reading and writing them is what
    this class adds.
    """

    # --- reading ------------------------------------------------------

    @classmethod
    def load(cls, file: tx.Any, **kwargs: tx.Any) -> "Metadata":
        """
        Refuse to read the metadata of a file on its own.

        The metadata class of a format whose files hold metadata lists its
        parser (`MetadataParser`) before `FileBasedMetadata` among its
        bases, so that the parser's `load` wins; every other format, whose
        metadata is written with its data or not at all, inherits this
        refusal. [`Metadata.load`][brainhops.datamodel.metadata.Metadata.load]
        finds the format of a file.

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
        from brainhops.io.base.parsers import ParserNotImplementedError

        raise ParserNotImplementedError(
            f"{cls.__name__} stores no metadata that can be read on its own."
        )

    @classmethod
    def from_raw(
        cls, raw: tx.Any, *, image: tx.Any = None, **values: tx.Any
    ) -> tx.Self:
        """
        Build the metadata of a raw record that was just read.

        The raw record is decoded into the common fields by the format's
        `_decode` hook. The decoded values are kept as the read-time
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
            If `_decode` returned a value for a field that this class does
            not support. The raw record belongs to the format, so this
            error never comes from the data: it reveals a format class
            whose `_decode` disagrees with its `supports=` declaration,
            and which would otherwise drop the value without a report.
        """
        decoded = cls._checked_decode(raw, image)
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

        A field that the data model gives a value for (see `_geometry`) is
        not encoded, because the writer stores the value of the data
        model. When the field was changed to a value that disagrees with
        the data model, the field is reported as approximated. A field
        that the format does not support, but that was assigned after
        construction, is reported as lost, as are the value-dependent
        losses found by `_encode`.

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
            (see [`metadata_loss_policy`][]). A writer passes the report of
            its whole write, which it then acts on once.
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
        self._check_derived(changed, image, report)
        raw = self._encode(raw, changed, image=image, report=report)
        if not sink:
            apply_loss_policy(report, on_loss, stacklevel=2)
        return raw

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

    def check_writable(self, *, image: tx.Any = None) -> ConversionReport:
        """
        Report what a write of this metadata would lose, without writing
        anything.

        The class declarations only give a lower bound of what is lost.
        This method runs the encoder on the record that the writer of the
        format would start from (see `_check_raw`), so that the losses
        that depend on the values, such as an over-long description or an
        irregular slice timing, are included as well.

        Parameters
        ----------
        image : object, optional
            The image or transformation that would be written, for the
            fields that depend on the data model.

        Returns
        -------
        ConversionReport
            What would be lost or approximated.
        """
        report = ConversionReport(source=self.format, target=self.format)
        self.update_raw(self._check_raw(image), image=image, on_loss=report)
        return report

    # --- the record of a conversion -----------------------------------

    @classmethod
    def _raw_type(cls) -> tx.Optional[type]:
        """
        The type of raw record that this class declares, read from the
        annotation of its `raw` field (`Optional` removed).

        Returns
        -------
        type or None
            The declared type, `type(None)` for a format without a record,
            or `None` when the class does not declare a type (it inherits
            the `Any` of `Metadata`).
        """
        hint = next(f.type for f in fields(cls) if f.name == "raw")
        while tx.get_origin(hint) is tx.Annotated:
            hint = tx.get_args(hint)[0]
        if hint is None or hint is type(None):
            return type(None)
        if tx.get_origin(hint) is tx.Union:
            args = [a for a in tx.get_args(hint) if a is not type(None)]
            hint = args[0] if len(args) == 1 else None
        if hint is tx.Any or not isinstance(hint, type):
            return None
        return hint

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
    def _default_raw(cls) -> tx.Any:
        """
        Build a fresh, empty raw record of this format.

        Returns
        -------
        object
            The empty record, or `None` for a format without one.
        """
        return None

    @classmethod
    def _decode(cls, raw: tx.Any, *, image: tx.Any = None) -> tx.Dict:
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

    def _encode(
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

    def _geometry(self, image: tx.Any) -> tx.Dict[str, tx.Any]:
        """
        Give the values that the data model holds for some fields.

        Some formats store, in the same slot, a value that is both a
        common field and a piece of the geometry of the data model. NIfTI,
        for instance, stores the repetition time as the time step of the
        image, `pixdim[4]`. When such an image is written, the writer
        stores the time step of the data model, whatever the metadata
        says. For that reason, `update_raw` does not encode a field that
        this hook gives a value for, and reports the field as approximated
        when its changed value disagrees with the value of the data model.

        A field that this hook leaves out, or gives as `None`, is one the
        data model says nothing about, and `_encode` handles its changed
        value as usual. The default gives no field.

        Parameters
        ----------
        image : object
            The image or transformation being written.

        Returns
        -------
        dict
            Field name to the value that the data model gives.
        """
        return {}

    def _check_raw(self, image: tx.Any) -> tx.Any:
        """
        Build the raw record that `check_writable` encodes over.

        The record is the one that the writer of this format would pass
        to `update_raw` for `image`. The default is a copy of `raw`, or a
        default record when there is none.

        Parameters
        ----------
        image : object
            The image or transformation that would be written.

        Returns
        -------
        object
            A scratch record, which may be edited.
        """
        return self._raw_or_default()

    def _derive_raw(
        self,
        raw: tx.Any,
        *,
        changed: tx.Mapping[AxisType, tx.Any],
    ) -> tx.Any:
        """
        Build the raw record of derived metadata (see `derive`, and the
        hooks of the image operations, `_select` and `_reslice`).

        The record is a copy of `raw`. A format overrides this hook to
        remove from the copy what the changed axes invalidate but the
        vocabulary does not cover, such as the slice-timing slots of a
        NIfTI header when the spatial axes changed. The default makes a
        deep copy, so that the derived metadata never shares its record.

        Parameters
        ----------
        raw : object
            The record of this metadata.
        changed : mapping
            The changed axes: `AxisType` to what changed the axes of that
            type, the kept positions for `_select` and the linear voxel
            map (or `None`) for `_reslice`. It is empty for `derive`. A
            format only tests which types are in it
            (`AxisType.space in changed`).

        Returns
        -------
        object
            The record of the derived metadata.
        """
        return copy.deepcopy(raw)

    # --- internals ----------------------------------------------------

    @classmethod
    def _checked_decode(
        cls, raw: tx.Any, image: tx.Any
    ) -> tx.Dict[str, tx.Any]:
        """`_decode`, without its absent values, and checked against the
        declarations of the class (see `from_raw`)."""
        decoded: tx.Dict[str, tx.Any] = {}
        for key, value in cls._decode(raw, image=image).items():
            if value is None or value is UNSUPPORTED:
                continue
            if key not in cls.supported_fields:
                raise TypeError(
                    f"{cls.__name__}._decode returned {key}={value!r}, "
                    f"but {cls.__name__} does not support {key!r}."
                )
            decoded[key] = value
        return decoded

    def _check_derived(
        self,
        changed: tx.Dict[str, tx.Any],
        image: tx.Any,
        report: ConversionReport,
    ) -> None:
        """Take out of `changed` the fields the data model gives values
        for (`_geometry`), and report those that disagree with it."""
        if image is None or not changed:
            return
        for name, given in self._geometry(image).items():
            if given is None or name not in changed:
                continue
            value = changed.pop(name)
            if value is not None and not _agrees(value, given):
                report.approximated[name] = (
                    f"derived from the data model ({short(given)})"
                )

    def _raw_or_default(self) -> tx.Any:
        if self.raw is not None:
            return copy.deepcopy(self.raw)
        return self._default_raw()

    def _derive_values(
        self,
        *,
        changed: tx.Mapping[AxisType, tx.Any],
        history: _History,
    ) -> tx.Dict[str, tx.Any]:
        # The raw record and the snapshot are kept, so that a field
        # `_reslice` or `_select` cleared is cleared in the record on write;
        # the record is the format's scrubbed copy (`_derive_raw`).
        values = super()._derive_values(changed=changed, history=history)
        values["snapshot"] = dict(self._snapshot)
        values["raw"] = self._derive_raw(self.raw, changed=changed)
        return values


class OpaqueMetadata(FileBasedMetadata, on={"format": "opaque"}, supports=()):
    """
    The metadata of a format that stores none, not even in memory: every
    field is unsupported, and there is no raw record.

    Base class of the formats that store a bare matrix and are given
    nothing else (matrix text, ITK `.tfm`/`.mat`). A read-then-save
    loses nothing, because nothing is there. A format that keeps
    anything, if only in memory (FLIRT `moving`/`fixed`), is a
    `FileBasedMetadata` with a `supports=` list instead.
    """

    raw: tx.Annotated[
        None, tx.Doc("Always `None`: no raw record."), NoRepr(), NoEq()
    ] = None


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
    """Whether a value agrees with what the data model gives (numbers
    within single-precision rounding)."""
    if isinstance(value, (int, float)) and isinstance(given, (int, float)):
        return math.isclose(value, given, rel_tol=1e-6, abs_tol=1e-9)
    return not differs(value, given)
