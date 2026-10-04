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
from brainhops._core.properties import Lazy

from ._base import FIELDS, Metadata
from ._report import ConversionReport, short
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
        Factory(dict),
    ]

    # --- reading ------------------------------------------------------

    @classmethod
    def from_raw(
        cls, raw: tx.Any, *, image: tx.Any = None, **values: tx.Any
    ) -> tx.Self:
        """
        Build the metadata of a raw record that was just read.

        The raw record is decoded into the common fields (`_decode`), and
        what was decoded is kept as the read-time snapshot, so that an
        untouched field keeps the record's value when it is written back.
        A lazy field (`lazy=`) decoded as
        [`Lazy`][brainhops._core.properties.Lazy] is decoded on first
        access instead. Keyword arguments set fields over the decoded
        values (they then count as changes).

        Raises
        ------
        TypeError
            If `_decode` returned a value for a field this class does not
            support, or a `Lazy` for a field not declared `lazy=`. The
            raw record is the format's own, so this is never about the
            data: it is a bug of the format class, whose `_decode` and
            `supports=` (or `lazy=`) disagree, and which would otherwise
            drop the value without a report.
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
        The metadata of another raw record, keeping the changes made here.

        `raw` is decoded (`from_raw`), and every field that changed since
        this object was read (`changed_fields()`: all the fields that
        are set, for an object built in memory) is set over the decoded
        values, as a change. `extra` is merged key by key. This is what
        an object given a new raw record holds (`replace(image,
        header=...)`), and what `metadata=` given along with a raw record
        becomes.
        """
        changed = {
            key: value
            for key, value in self.changed_fields().items()
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

    def changed_fields(self) -> tx.Dict[str, tx.Any]:
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
        report: tx.Optional[ConversionReport] = None,
        force: tx.Collection[str] = (),
    ) -> tx.Any:
        """
        Encode the common fields over a raw record, and return it.

        `raw` is the record to write over: a writer passes its own fresh
        record, already filled with what it keeps from `self.raw`. When
        it is `None`, a copy of `self.raw` (or a default record) is used.
        Only the fields that changed since the read are encoded (see
        [`changed_fields`][brainhops.datamodel.metadata.FileBasedMetadata.changed_fields]),
        and the fields named in `force`, whether they changed or not (a
        writer keyword that must win over the record, such as MGH `tr=`;
        a `None` there clears the slot).

        A field for which the data model gives a value (`_geometry`: a
        view of geometry, such as the NIfTI repetition time) is not
        encoded: the data model's value is what the writer stores, and a
        changed value that disagrees with it is reported as
        approximated. A field this format does not
        support but that was assigned after construction is recorded as
        lost in `report`, as are the value-dependent losses `_encode`
        finds.
        """
        if report is None:
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
            for key, value in self.changed_fields().items()
            if key not in unsupported
        }
        for name in force:
            if name in FIELDS and name != "extra":
                if name not in unsupported:
                    changed[name] = getattr(self, name)
        self._check_derived(changed, image, report)
        return self._encode(raw, changed, image=image, report=report)

    def check_writable(self, *, image: tx.Any = None) -> ConversionReport:
        """
        What a write of this object would lose, without writing it.

        Class-level declarations are the lower bound of loss; this runs
        the encoder on a scratch raw record (the one the format's writer
        would start from, see `_check_raw`), so value-dependent losses
        (an over-long description, an irregular slice timing) are
        included. Pass the image (or transformation) for the fields that
        need it.
        """
        report = ConversionReport(source=self.format, target=self.format)
        self.update_raw(self._check_raw(image), image=image, report=report)
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
        """A fresh, empty raw record for this format."""
        return None

    @classmethod
    def _decode(cls, raw: tx.Any, *, image: tx.Any = None) -> tx.Dict:
        """Raw record to common fields. Default: nothing to decode."""
        return {}

    def _encode(
        self,
        raw: tx.Any,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> tx.Any:
        """Common fields to raw record. Default: the record is unchanged."""
        return raw

    def _geometry(self, image: tx.Any) -> tx.Dict[str, tx.Any]:
        """
        The fields that are, for this format, a view of geometry the data
        model owns, and the values it gives for them (NIfTI: the time
        step of the image as `repetition_time`). This is what makes a
        field derived: on write, the data model's value wins, and a
        changed value that disagrees with it is reported. A field left
        out, or `None`, is one the data model says nothing about: its
        changed value is then left to `_encode`. Default: nothing.
        """
        return {}

    def _check_raw(self, image: tx.Any) -> tx.Any:
        """
        The raw record `check_writable` encodes over: what the writer of
        this format would pass to `update_raw` for `image`. Default: a
        copy of the record, or a default one.
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
