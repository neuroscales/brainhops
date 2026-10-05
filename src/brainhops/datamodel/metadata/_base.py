"""`Metadata`: the common vocabulary, the hub through which formats convert."""

__all__ = ["Metadata"]

# stdlib
import copy

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, NoEq, NoRepr, fields

# internals
from brainhops._core.typing import ArrayLike

from ..base import DataModelBase
from ..enums import AxisType
from ._meta import MetadataMeta
from ._report import (
    ConversionReport,
    OnLoss,
    apply_loss_policy,
)
from ._sentinel import UNSUPPORTED, Maybe
from ._terms import EncodingDirection, GeneratedBy
from ._vocabulary import (
    ALONG,
    FIELDS,
    SCOPES,
    VOCABULARY,
    DiffusionVocabulary,
    DisplayVocabulary,
    MicroscopyVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    Scope,
    Scoped,
    StorageVocabulary,
    TransformVocabulary,
)

# The `history=` argument of `derive`, `_select` and `_reslice`: one
# entry, several, or none.
_History = tx.Union[str, tx.Sequence[str], None]


class Metadata(
    DataModelBase,
    ProvenanceVocabulary,
    MRIVocabulary,
    DiffusionVocabulary,
    DisplayVocabulary,
    StorageVocabulary,
    MicroscopyVocabulary,
    TransformVocabulary,
    metaclass=MetadataMeta,
    polymorphic=True,
    kw_only=True,
    repr=False,
):
    """
    Metadata that does not depend on a file format: the common
    vocabulary, and the free-form store `extra`.

    In-memory images and transformations carry this class of metadata,
    and every conversion between two formats goes through it, so that a
    conversion from NIfTI to MGH loses exactly what the two conversions
    from NIfTI to `Metadata` and from `Metadata` to MGH lose. The BIDS
    sidecar codec reads and writes it too. `Metadata` supports every
    field of the vocabulary.

    `Metadata` is also the root of the metadata classes, and selects the
    subclass from the `format` field: once `brainhops.io` is imported,
    `Metadata(format="nifti", ...)` builds a `NiftiMetadata`, and an
    unknown format builds a plain `Metadata`.

    The vocabulary is declared by seven groups, which `Metadata` inherits:
    [`ProvenanceVocabulary`][], [`MRIVocabulary`][], [`DiffusionVocabulary`][],
    [`DisplayVocabulary`][], [`StorageVocabulary`][],
    [`MicroscopyVocabulary`][] and
    [`TransformVocabulary`][]. Each field holds a value, `None` when the
    value is unknown, or `UNSUPPORTED` when a format has no slot for the
    field. The hooks that a format implements are described in the format
    author's guide (`docs/dev/metadata-formats.md`).
    """

    # --- class attributes ---------------------------------------------

    supported_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc(
            "The vocabulary fields (and `'extra'`) this class can store, "
            "from its `supports=` declaration."
        ),
    ] = frozenset(("extra",) + VOCABULARY)

    unsupported_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc(
            "The vocabulary fields this class cannot store: the "
            "complement of `supported_fields`."
        ),
    ] = frozenset()

    lazy_fields: tx.Annotated[
        tx.ClassVar[tx.FrozenSet[str]],
        tx.Doc("The vocabulary fields decoded on first access."),
    ] = frozenset()

    # --- format and extras --------------------------------------------

    format: tx.Annotated[
        str,
        tx.Doc("The format this metadata belongs to; selects the subclass."),
    ] = "generic"

    extra: tx.Annotated[
        Maybe[tx.Dict[str, tx.Any]],
        tx.Doc(
            "Free-form keys the vocabulary does not cover, copied into "
            "any free-form store a format has."
        ),
        Scoped(Scope.FILE),
        # Not `Factory()`: inferred from `Maybe[...]`, a union with
        # `None`, the default would be `None`.
        Factory(dict),
    ]

    # --- the raw record -----------------------------------------------

    raw: tx.Annotated[
        tx.Any,
        tx.Doc(
            """
            The raw record of the file the metadata was read from (a
            `nibabel` header, the attributes of a Zarr array, ...), or
            `None`. Edit it only for what the vocabulary does not cover:
            on write, a field left untouched keeps the value of the
            record, and a field that was set wins over it.

            Generic metadata keeps the record of the metadata it was
            converted from, so that converting back to the format of the
            record keeps it: `NiftiMetadata -> Metadata -> NiftiMetadata`
            round-trips, header extensions included. A conversion into
            another format leaves the record behind, since every format
            declares its own type of record.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    _snapshot: tx.Annotated[
        tx.Dict[str, tx.Any],
        tx.Doc(
            """
            The read-time snapshot: field name to the value that the
            reader decoded from `raw`, and empty for metadata built in
            memory. On write, a field is encoded over the raw record only
            when it differs from its snapshot. The reader fills it; it is
            never set by hand.
            """
        ),
        NoRepr(),
        NoEq(),
        Factory(),
    ]

    # --- construction -------------------------------------------------

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        cls = type(self)
        for name in cls.unsupported_fields:
            value = getattr(self, name, None)
            if value is None:
                setattr(self, name, UNSUPPORTED)
            elif value is not UNSUPPORTED:
                raise ValueError(
                    f"{cls.__name__} cannot store {name!r} (it is "
                    f"UNSUPPORTED by this format), so {value!r} is "
                    f"refused."
                )

    def __repr__(self) -> str:
        # `UNSUPPORTED`, `None` and an empty `extra` are hidden, or a
        # format that stores three fields would print forty. `format` and
        # a format's own fields come first, then `extra`, then the
        # vocabulary in its declared order (not `bagof`'s reverse MRO).
        own = [
            field.name
            for field in fields(type(self))
            if field.name not in VOCABULARY
            and field.name != "extra"
            and field.repr
            and not field.var
        ]
        parts = []
        for name in own + list(FIELDS):
            value = getattr(self, name, None)
            if value is None or value is UNSUPPORTED:
                continue
            if name == "extra" and not value:
                continue
            parts.append(f"{name.lstrip('_')}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def copy(self) -> tx.Self:
        """
        Copy this metadata, so that the copy can be edited without editing
        the original.

        An image or a transformation holds a copy when it is given
        metadata that another object already holds (through `replace()`,
        `from_other` or `metadata=`).

        The raw record is shared, as `replace()` shares it, while the
        read-time snapshot and `extra` are copied, so that editing the
        copy never edits the original. A field that is still waiting to
        be decoded (see `lazy=`) stays so in both.

        Returns
        -------
        Metadata
            The copy, of the same class.
        """
        new = copy.copy(self)
        new.extra = copy.copy(new.extra)
        new._snapshot = dict(self._snapshot)
        return new

    # --- capabilities -------------------------------------------------

    @classmethod
    def supports(cls, name: str) -> bool:
        """
        Whether this class can store a vocabulary field, as its
        `supports=` declaration says.

        A format whose capabilities depend on the instance holds
        `UNSUPPORTED` in a field that a particular instance cannot store,
        so `meta.name is UNSUPPORTED` is the test for an instance.

        Parameters
        ----------
        name : str
            The name of a vocabulary field, or `"extra"`.

        Returns
        -------
        bool
            Whether the class can store the field.

        Raises
        ------
        KeyError
            If `name` is not a vocabulary field.
        """
        if name not in FIELDS:
            raise KeyError(f"{name!r} is not a vocabulary field.")
        return name in cls.supported_fields

    # --- conversion ---------------------------------------------------

    def to(
        self,
        cls: tx.Union[None, str, tx.Type["Metadata"]] = None,
        *,
        on_loss: tx.Optional[OnLoss] = None,
        **values: tx.Any,
    ) -> "Metadata":
        """
        Convert this metadata into another class, as images and
        transformations convert with `to()`.

        Each value of the vocabulary is copied, except where the target
        class cannot store the field. Such a value is lost, and reported.

        Parameters
        ----------
        cls : type or str, optional
            The `Metadata` subclass to convert to, or the name of its
            format (`"generic"`, `"nifti"`, ...). By default, the class of
            this object, which makes a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do when something is lost. By default, the policy in
            effect (see [`metadata_loss_policy`][]). A
            [`ConversionReport`][] is filled with what was lost or
            approximated, and nothing is warned or raised.
        **values
            Fields to set on the result.

        Returns
        -------
        Metadata
            The converted metadata.

        Raises
        ------
        MetadataLossError
            If something is lost under the `"raise"` policy.
        ValueError
            If `cls` names no known format.

        Examples
        --------
        ```python
        report = ConversionReport()
        nifti = meta.to(NiftiMetadata, on_loss=report)
        if report.lossy:
            ...
        ```
        """
        target = type(self) if cls is None else _metadata_class(cls)
        obj, found = target._convert_from(self, (), values)
        apply_loss_policy(found, on_loss, stacklevel=2)
        return obj

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Convert the metadata of another format into this class.

        This is the conversion that field converters trigger. It behaves
        as [`to`][brainhops.datamodel.metadata.Metadata.to] does, and hands
        the report to the loss policy in effect (see
        [`metadata_loss_policy`][]). An object that is not metadata is
        handed to the data model.

        Parameters
        ----------
        other : object
            The object to convert.
        *args, **kwargs
            Constructor arguments.

        Returns
        -------
        Metadata
            The converted metadata.

        Raises
        ------
        MetadataLossError
            If something is lost under the `"raise"` policy.
        """
        if not isinstance(other, Metadata):
            return super().from_instance(other, *args, **kwargs)
        obj, report = cls._convert_from(other, args, kwargs)
        apply_loss_policy(report, stacklevel=3)
        return obj

    # --- propagation --------------------------------------------------

    def derive(self, *, history: _History = None) -> tx.Self:
        """
        Build the metadata of an image computed from the image this
        metadata describes, on the same axes, such as a smoothed or a
        denoised version of it.

        Every field is kept, except three provenance fields, which record
        the derivation: `creation_time` is cleared, the entries of
        `history` are appended to `history`, and `generated_by` gains a
        brainhops entry, once.

        An image operation that changes axes (`image[index]`,
        `image.reslice(...)`) derives the metadata of its result itself,
        and records the derivation in the same way: an index along a
        time or a channel axis selects the entries of the fields that run
        along it, and a change of the spatial axes clears the fields tied
        to the spatial sampling, except an encoding direction, which
        follows the map of the voxel axes. See the user guide (`Derived
        images`).

        Parameters
        ----------
        history : str or sequence of str, optional
            A description of the derivation, appended to `history`: one
            entry, or several.

        Returns
        -------
        Metadata
            New metadata, of the same class. Generic metadata derived this
            way carries no raw record, since no format hook can scrub it.
            The metadata of a file format keeps a copy of its record, and
            of its read-time snapshot.

        Examples
        --------
        ```python
        meta = Metadata(description="T1w", history=("acquired",))
        meta.derive(history="smooth").history  # ('acquired', 'smooth')
        ```
        """
        return type(self)(**self._derive_values(changed={}, history=history))

    # --- files ------------------------------------------------------

    @classmethod
    def load(cls, file: tx.Any, **kwargs: tx.Any) -> "Metadata":
        """
        Read the metadata of a file, without reading its data.

        The format of the file is found as `brainhops.io` finds it, from
        the name of the file and from its content, among the formats whose
        metadata can be read on its own: NIfTI, MGH, plain Zarr and
        OME-Zarr, x5, ITK `.h5`, and BIDS JSON sidecars. `hint=` restricts
        the candidates, as for `brainhops.io.load`. Only the raw record is
        read: a NIfTI header, the footer and the tags of an MGH file, the
        attributes of a Zarr node, the JSON of an x5 node.

        The metadata class of a format overrides this method by its bases:
        one whose files hold metadata lists its parser before
        `FileBasedMetadata`, and reads the file as a file of that format;
        any other inherits `FileBasedMetadata.load`, which refuses.

        Parameters
        ----------
        file : str, path-like or file object
            The file, or the Zarr store.
        **kwargs
            Options of the reader of the format, and `hint=` (a format
            name such as `"nifti"`, or several).

        Returns
        -------
        Metadata
            The metadata of the format, with its raw record. Convert it
            with `to(Metadata)` for generic metadata (which keeps the
            record).

        Raises
        ------
        ParserContentError
            If no format reads the metadata of the file.
        ParserNotImplementedError
            If called on the class of a format whose files hold no
            metadata (FLIRT, ITK `.tfm` and `.mat`, ...).
        ParserExistsError
            If the file does not exist.

        Examples
        --------
        ```python
        meta = Metadata.load("sub-01_bold.nii.gz")
        meta.repetition_time        # read from the header alone
        Metadata.load("sub-01_bold.json").extra["TaskName"]  # a sidecar
        Metadata.load("scan.mgz", hint="mgh")
        NiftiMetadata.load("sub-01_bold.nii.gz")  # as a NIfTI file
        ```
        """
        import brainhops.io  # noqa: F401  (registers the formats)
        from brainhops.io.base._metadata_parser import MetadataParser

        return MetadataParser.load(file, **kwargs)

    # --- BIDS ---------------------------------------------------------

    @classmethod
    def from_bids(cls, sidecar: tx.Any) -> "Metadata":
        """
        Read a BIDS JSON sidecar.

        A key that names a vocabulary field, through its BIDS key, fills
        that field, and every other key lands in `extra`.

        Parameters
        ----------
        sidecar : mapping, str, path-like or file
            The sidecar, as a decoded JSON object, a JSON string, a path
            or an open file.

        Returns
        -------
        Metadata
            Generic metadata.
        """
        from brainhops.io.metadata.bids import from_bids

        return from_bids(sidecar)

    def to_bids(
        self, *, on_loss: tx.Optional[OnLoss] = None
    ) -> tx.Dict[str, tx.Any]:
        """
        Write this metadata as a BIDS JSON sidecar.

        The diffusion fields are not sidecar keys, and an encoding
        direction that is not along a voxel axis has no BIDS string, so
        both are reported as lost.

        Parameters
        ----------
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do when something is lost. By default, the policy in
            effect.

        Returns
        -------
        dict
            The sidecar, which can be serialised to JSON.
        """
        from brainhops.io.metadata.bids import to_bids

        return to_bids(self, on_loss=on_loss)

    # --- conversion hook (key/value formats) --------------------------

    @classmethod
    def _import(
        cls,
        other: "Metadata",
        values: tx.Dict[str, tx.Any],
        *,
        report: ConversionReport,
    ) -> None:
        """
        Recover losses of a conversion into this class: a hook for the
        key/value formats (MRtrix, NRRD), whose free-form store can hold
        what they have no slot for.

        Called by `to` and `from_instance` with the source object, the
        values about to be passed to the constructor (`values`, edited in
        place) and the report, whose `lost` already lists what this class
        cannot store. A format recovers a loss by moving the value into
        `values["extra"]`, removing it from `report.lost` and adding its
        name to `report.passed_through`. Default: nothing is recovered.

        Examples (none of the formats in the package needs it yet):

        - MRtrix (`.mif`) has dedicated keys for a few fields
          (`PhaseEncodingDirection`, `TotalReadoutTime`, `dw_scheme`)
          and none for `echo_time` or `flip_angle`, but its `keyval`
          holds any `key: value`. Converting an MGH image's metadata
          (`echo_time=0.0035`, `flip_angle=8.6`) to `MrtrixMetadata`
          would otherwise report both as lost; its `_import` moves them
          to `values["extra"]` under their BIDS keys (`EchoTime`,
          `FlipAngle`), as `mrconvert -json_import` does, and lists them
          in `report.passed_through`:

          ```python
          @classmethod
          def _import(cls, other, values, *, report):
              extra = dict(values.get("extra") or {})
              for name in [n for n in report.lost if n != "extra"]:
                  extra[BIDS_KEYS[name]] = report.lost.pop(name)
                  report.passed_through += (name,)
              values["extra"] = extra
          ```

        - NRRD has no field for slice timing or the phase encoding
          direction of a NIfTI image; its `_import` writes them as
          `keyvalue` pairs (`SliceTiming:=0 0.5 1 1.5`), so a NIfTI ->
          NRRD -> NIfTI round trip keeps them.
        - The other way round, a format with dedicated slots but no
          free-form store (MGH) loses the `extra` of an MRtrix source
          as a whole; its `_import` could take `extra["EchoTime"]` back
          into `values["echo_time"]` before the rest of `extra` is
          reported lost.
        """

    # --- internals ----------------------------------------------------

    @classmethod
    def _convert_from(
        cls,
        other: "Metadata",
        args: tx.Tuple[tx.Any, ...] = (),
        kwargs: tx.Optional[tx.Dict[str, tx.Any]] = None,
    ) -> tx.Tuple["Metadata", ConversionReport]:
        """`from_instance`, returning the report instead of acting on it."""
        kwargs = dict(kwargs or {})
        same = _is_already(other, cls)
        # A copy keeps the most specific class.
        target = type(other) if same else cls
        report = ConversionReport(
            source=_format_name(other), target=_format_name(target)
        )
        values: tx.Dict[str, tx.Any] = {}
        if same or (other.raw is not None and target._accepts_raw(other.raw)):
            # The record and its snapshot go along, shared and copied as
            # `copy()` does: to a copy, to the generic hub, or back to the
            # format whose class declares the type of the record.
            values["raw"] = other.raw
            values["snapshot"] = dict(other._snapshot)
        unsupported = target.unsupported_fields
        for name in FIELDS:
            value = getattr(other, name, None)
            if value is None or value is UNSUPPORTED:
                continue
            if name == "extra":
                if not value:
                    continue
                value = dict(value)
            if name in unsupported:
                report.lost[name] = value
                continue
            values[name] = value
        target._import(other, values, report=report)
        values.update(kwargs)
        return target(*args, **values), report

    def _select(
        self,
        axis: AxisType,
        positions: tx.Optional[ArrayLike],
        *,
        history: _History = None,
    ) -> tx.Self:
        """
        The metadata of an image indexed along its axes of one type, other
        than `space`: a hook of the image operations.

        A field in the `AXIS` scope that runs along `axis` keeps its
        entries at `positions`, the positions kept along the axis, which
        the image resolves from its index. `None` (the axis was dropped,
        or the selection is unknown), or a position beyond the field,
        clears the field. Every other field is kept, and the derivation
        is recorded as `derive` records it.

        Parameters
        ----------
        axis : AxisType
            The type of the indexed axis.
        positions : array-like of int, optional
            The kept positions, in order.
        history : str or sequence of str, optional
            A description of the derivation, appended to `history`.

        Returns
        -------
        Metadata
            New metadata, of the same class.

        Raises
        ------
        ValueError
            If `axis` is `space`: a change of the spatial axes goes
            through `_reslice`.
        """
        axis = AxisType(axis)
        if axis is AxisType.space:
            raise ValueError("The spatial axes change through _reslice().")
        values = self._derive_values(
            changed={axis: positions}, history=history
        )
        return type(self)(**values)

    def _reslice(
        self,
        linear: tx.Optional[ArrayLike],
        *,
        history: _History = None,
    ) -> tx.Self:
        """
        The metadata of an image whose spatial axes changed: a hook of the
        image operations.

        A field in the `SPATIAL` scope is cleared, except an encoding
        direction in voxel axes, which is mapped through `linear`, the
        linear part of the map from the old voxel coordinates to the new
        ones (new axes by old axes), which the image computes from its
        geometry. Without it (`None`: the map is unknown or not affine),
        the direction is cleared. A direction in a named world space is
        kept. Every other field is kept, and the derivation is recorded
        as `derive` records it.

        Parameters
        ----------
        linear : array-like, optional
            The linear map from the old voxel axes to the new ones.
        history : str or sequence of str, optional
            A description of the derivation, appended to `history`.

        Returns
        -------
        Metadata
            New metadata, of the same class.
        """
        values = self._derive_values(
            changed={AxisType.space: linear}, history=history
        )
        return type(self)(**values)

    def _derive_values(
        self,
        *,
        changed: tx.Mapping[AxisType, tx.Any],
        history: _History,
    ) -> tx.Dict[str, tx.Any]:
        """
        The constructor values of `derive`, `_select` and `_reslice`,
        field by field, from the scope of each field.

        `changed` maps the type of each changed axis to what changed it:
        the linear voxel map (or `None`) for `space`, the kept positions
        (or `None`) for any other type. It is empty for `derive`.
        """
        values: tx.Dict[str, tx.Any] = {}
        for name in FIELDS:
            value = getattr(self, name, None)
            if value is UNSUPPORTED or name in self.unsupported_fields:
                continue
            scope = SCOPES.get(name, Scope.FILE)
            if name == "extra":
                value = dict(value or {})
            elif scope is Scope.SPATIAL and AxisType.space in changed:
                value = _map_direction(value, changed[AxisType.space])
            elif scope is Scope.AXIS and ALONG[name] in changed:
                value = _index(value, changed[ALONG[name]])
            values[name] = value
        _derive_provenance(values, history)
        return values

    @classmethod
    def _accepts_raw(cls, raw: tx.Any) -> bool:
        """Whether a conversion into this class keeps a raw record:
        generic metadata keeps any record (see `raw`); a file format,
        only its own (see `FileBasedMetadata._accepts_raw`)."""
        return True


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _derive_provenance(
    values: tx.Dict[str, tx.Any], history: _History
) -> None:
    """
    Record a derivation in the provenance fields, in place.

    The derived object is a new object, so its `creation_time` is
    cleared. The entries of `history` (one, when it is a string) are
    appended to the `history` field, and brainhops is added to
    `generated_by` unless it is there already. A field that the format
    does not support is absent from `values`, and is left alone.
    """
    if isinstance(history, str):
        steps: tx.Tuple[str, ...] = (history,)
    else:
        steps = tuple(history or ())
    if "creation_time" in values:
        values["creation_time"] = None
    if "history" in values and steps:
        values["history"] = tuple(values["history"] or ()) + steps
    if "generated_by" in values:
        values["generated_by"] = _with_brainhops(values["generated_by"])


def _with_brainhops(
    generated_by: tx.Optional[tx.Tuple[GeneratedBy, ...]],
) -> tx.Tuple[GeneratedBy, ...]:
    entries = tuple(generated_by or ())
    if any(getattr(g, "name", None) == "brainhops" for g in entries):
        return entries
    try:
        from brainhops import __version__ as version
    except ImportError:  # pragma: no cover
        version = None
    return entries + (GeneratedBy(name="brainhops", version=version),)


def _map_direction(value: tx.Any, linear: tx.Any) -> tx.Any:
    """
    A `SPATIAL` field after a change of the spatial axes (see `_reslice`).

    An encoding direction in a world space is kept. One in voxel axes
    goes through `linear`, when it is given. The map may have more axes
    than the direction, which lies in the first ones (`ijk`), as for a
    4-D image: the direction is then mapped as a vector of the old axes
    that is zero beyond its own, and is kept only when it still lies in
    the first axes. Anything else is cleared.
    """
    if not isinstance(value, EncodingDirection):
        return None
    if value.space is not None:
        return value
    if linear is None:
        return None
    linear = np.asarray(linear, dtype=float)
    vector = np.asarray(value.vector, dtype=float)
    size = len(vector)
    if linear.ndim != 2 or min(linear.shape) < size:
        return None
    beyond = linear[size:, :size] @ vector
    if np.any(np.abs(beyond) > 1e-9 * np.abs(linear).max()):
        # The direction moved onto an axis it cannot be expressed on.
        return None
    try:
        return value.transform(linear[:size, :size])
    except ValueError:
        # Mapped to zero: its axis was dropped.
        return None


def _index(value: tx.Any, positions: tx.Any) -> tx.Any:
    """
    An `AXIS` field after an index of its axis (see `_select`): the
    entries at `positions`. No positions (the axis was dropped), or a
    position beyond the field, clears it.
    """
    if value is None or positions is None:
        return None
    positions = np.asarray(positions)
    if positions.ndim != 1 or positions.dtype.kind not in "iu":
        return None
    try:
        size = len(value)
    except TypeError:
        return None
    if positions.size and not (0 <= positions.min() <= positions.max() < size):
        return None
    return tuple(value[int(i)] for i in positions)


def _is_already(value: tx.Any, cls: type) -> bool:
    """
    Whether a metadata object can stand for `cls` as it is, without a
    conversion.

    An object of `cls` itself can, and so can an object of a subclass of
    a format class: a subclass of `NiftiMetadata` is still NIfTI
    metadata. Generic `Metadata` is the exception, because its subclasses
    are the formats: a `NiftiMetadata` given where generic metadata is
    expected is converted, so that the result supports every field.

    Parameters
    ----------
    value : object
        The metadata object.
    cls : type
        The metadata class that is expected.

    Returns
    -------
    bool
        Whether `value` is used as it is.
    """
    if type(value) is cls:
        return True
    return cls is not Metadata and isinstance(value, cls)


def _format_name(obj: tx.Any) -> str:
    """
    The name of the format of a metadata class or object.

    For a class, the name is the default of its `format` field, which the
    class pins with `on={"format": ...}` (`"nifti"` for `NiftiMetadata`,
    `"generic"` for `Metadata`). A class whose `format` has no string
    default is named after the class. For an object, the name is the
    value of its `format` field.

    Parameters
    ----------
    obj : type or object
        A metadata class or object.

    Returns
    -------
    str
        The name of the format, as conversion reports print it.
    """
    if isinstance(obj, type):
        for field in fields(obj):
            if field.name == "format":
                default = field.default
                if isinstance(default, str):
                    return default
        return obj.__name__
    return str(getattr(obj, "format", type(obj).__name__))


def _metadata_class(target: tx.Any) -> tx.Type[Metadata]:
    """
    The class that a `to()` target names.

    Parameters
    ----------
    target : type or str
        A `Metadata` subclass, or the name of a format. A name is resolved
        by the polymorphic constructor of `Metadata`, which selects the
        subclass from `format`.

    Returns
    -------
    type
        The metadata class.

    Raises
    ------
    ValueError
        If `target` names no known format.
    TypeError
        If `target` is neither a class nor a string.
    """
    if isinstance(target, type) and issubclass(target, Metadata):
        return target
    if isinstance(target, str):
        cls = type(Metadata(format=target))
        if cls is Metadata and target != _format_name(Metadata):
            raise ValueError(f"No metadata class for the format {target!r}.")
        return cls
    raise TypeError(
        f"Expected a Metadata subclass or a format name, got {target!r}."
    )
