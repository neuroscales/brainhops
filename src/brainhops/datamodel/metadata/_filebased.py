"""`FileBasedMetadata`: the metadata of a file format, and its hooks."""

__all__ = ["FileBasedMetadata", "OpaqueMetadata"]

# stdlib
import copy
import math

# externals
import typing_extensions as tx
from bagof.magic import Factory, NoEq, NoRepr

# internals
from brainhops._core.compare import differs
from brainhops._core.fields import Lazy

from ._base import FIELDS, Metadata, format_name
from ._report import ConversionReport, OnLoss, apply_loss_policy, short
from ._sentinel import UNSUPPORTED


class FileBasedMetadata(Metadata):
    """
    The metadata of a file format: the common vocabulary, plus the
    format's own raw record and the read-time snapshot.

    This is the base of every `<Fmt>Metadata` (as `FileBasedImage` is of
    every format's image class): a format declares what it can store
    with `supports=`, and decodes and encodes its raw record with the
    hooks described in the format author's guide
    (`docs/dev/metadata-formats.md`).
    """

    raw: tx.Annotated[
        tx.Any,
        tx.Doc(
            """
            The format's own raw record (a `nibabel` header, ...). Edit
            it only for what the vocabulary does not cover: an untouched
            common field keeps the record's value on write, a common
            field you set wins over it. Never copied across formats.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    _snapshot: tx.Annotated[
        tx.Dict[str, tx.Any],
        tx.Doc(
            """
            The read-time snapshot: field name -> the value the reader
            decoded from `raw` (empty for an object built in memory). A
            common field is written over the raw record only when it
            differs from it. Filled by the reader; never set by hand.
            """
        ),
        NoRepr(),
        NoEq(),
        Factory(),
    ]

    # --- reading ------------------------------------------------------

    @classmethod
    def from_raw(
        cls, raw: tx.Any, *, image: tx.Any = None, **values: tx.Any
    ) -> tx.Self:
        """
        Build the metadata of a raw record that was just read.

        The raw record is decoded into the common fields by the format's
        `_decode` hook. The decoded values are kept as the read-time
        snapshot, so that a field the user leaves untouched keeps the
        value of the record when the record is written back. A field that
        the format declares lazy (`lazy=`), and that `_decode` returns as a
        [`Lazy`][brainhops._core.fields.Lazy] value, is decoded the first
        time it is read instead.

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
            not support, or a `Lazy` value for a field that is not lazy.
            The raw record belongs to the format, so this error never
            comes from the data: it reveals a format class whose `_decode`
            disagrees with its `supports=` or `lazy=` declaration, and
            which would otherwise drop the value without a report.
        """
        decoded, pending = cls._checked_decode(raw, image)
        obj = cls(raw=raw, **decoded)
        # Snapshot the *converted* values, so that a decoded list held as
        # a tuple does not count as a change.
        obj._snapshot = {
            key: copy.deepcopy(getattr(obj, key)) for key in decoded
        }
        for key, value in pending.items():
            # Injected as is (no conversion): the first read loads it.
            obj.__dict__[key] = value
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
            target = format_name(cls)
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

    # --- copies -------------------------------------------------------

    def copy(self) -> tx.Self:
        """
        A copy of this metadata, sharing its raw record.

        The raw record (`raw`) is shared, as `replace()` shares it; the
        snapshot and `extra` are copied, so editing the copy never edits
        this object. A field still waiting to be decoded (see `lazy=`)
        stays so in both.
        """
        new = super().copy()
        new.__dict__["_snapshot"] = dict(self._snapshot)
        return new

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
        grid_changed: bool,
        volumes: tx.Optional[tx.Sequence[int]],
    ) -> tx.Any:
        """
        The raw record of a derived object (`derive`): a copy of `raw`,
        which a format scrubs of what is tied to the grid (when
        `grid_changed`) or to the volumes (`volumes`, the selected
        indices) but is outside the vocabulary. Default: a deep copy, so
        that the derived object never shares its record.
        """
        return copy.deepcopy(raw)

    # --- internals ----------------------------------------------------

    @classmethod
    def _checked_decode(
        cls, raw: tx.Any, image: tx.Any
    ) -> tx.Tuple[tx.Dict[str, tx.Any], tx.Dict[str, Lazy]]:
        """`_decode`, split into the values and the lazy ones, and
        checked against the declarations of the class (see `from_raw`)."""
        decoded: tx.Dict[str, tx.Any] = {}
        pending: tx.Dict[str, Lazy] = {}
        for key, value in cls._decode(raw, image=image).items():
            if value is None or value is UNSUPPORTED:
                continue
            if key not in cls.supported_fields:
                raise TypeError(
                    f"{cls.__name__}._decode returned {key}={value!r}, "
                    f"but {cls.__name__} does not support {key!r}."
                )
            if not isinstance(value, Lazy):
                decoded[key] = value
            elif key in cls.lazy_fields:
                pending[key] = value
            else:
                raise TypeError(
                    f"{cls.__name__}._decode returned a Lazy {key!r}, "
                    f"which is not one of its lazy= fields."
                )
        return decoded, pending

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
        grid_changed: bool,
        grid_map: tx.Any,
        volumes: tx.Optional[tx.Sequence[int]],
        volumes_changed: bool,
        step: tx.Optional[str],
    ) -> tx.Dict[str, tx.Any]:
        # The raw record and the snapshot are kept, so that a field
        # `derive` cleared is cleared in the record on write; the record
        # is the format's scrubbed copy (`_derive_raw`).
        values = super()._derive_values(
            grid_changed=grid_changed,
            grid_map=grid_map,
            volumes=volumes,
            volumes_changed=volumes_changed,
            step=step,
        )
        values.update(self._format_state())
        values["raw"] = self._derive_raw(
            self.raw, grid_changed=grid_changed, volumes=volumes
        )
        return values

    def _format_state(self) -> tx.Dict[str, tx.Any]:
        """What a same-format copy carries besides the fields: the raw
        record (shared) and the snapshot (copied)."""
        return {"raw": self.raw, "snapshot": dict(self._snapshot)}


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

    format: tx.Annotated[
        tx.Literal["opaque"], tx.Doc("Always `'opaque'`.")
    ] = "opaque"

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
