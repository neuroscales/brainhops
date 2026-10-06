"""`Metadata`: the common vocabulary, the hub through which formats convert."""

__all__ = ["Metadata"]

# stdlib
import copy
import operator

# externals
import typing_extensions as tx
from bagof.magic import Factory, HideIf, NoEq, NoRepr, fields

# internals
from ..base import DataModelBase
from ._meta import MetadataMeta
from ._operations import Operation, propagate
from ._report import (
    ConversionReport,
    OnLoss,
    apply_loss_policy,
)
from ._sentinel import UNSUPPORTED, Maybe
from ._terms import GeneratedBy, _is_absent
from ._vocabulary import (
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

# The `history=` argument of `derive`: one entry, several, or none.
# Above `Metadata`, whose signatures evaluate it.
_History = tx.Union[str, tx.Sequence[str], None]


# The groups are listed in the reverse of their order because the fields
# are (`reverse=True`: the fields of a class before those it inherits),
# so that `repr` shows `format`, `extra`, then the vocabulary in its
# declared order. A format class declares the vocabulary again
# (`supports=`), so its `repr` shows its own fields, then `format`.
class Metadata(
    DataModelBase,
    TransformVocabulary,
    MicroscopyVocabulary,
    StorageVocabulary,
    DisplayVocabulary,
    DiffusionVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    metaclass=MetadataMeta,
    polymorphic=True,
    kw_only=True,
    reverse=True,
    # `UNSUPPORTED` and `None` are hidden from `repr`, or a format that
    # stores three fields would print forty (see also `extra`).
    repr=HideIf(_is_absent),
):
    """
    Metadata that does not depend on a file format: the common
    vocabulary, and the free-form store `extra`.

    In-memory images and transformations carry this class of metadata,
    and every conversion between two formats goes through it, so that a
    conversion from NIfTI to MGH loses exactly what the two conversions
    from NIfTI to `Metadata` and from `Metadata` to MGH lose. The BIDS
    sidecar codec (`brainhops.io.metadata.bids`) reads and writes it
    too. `Metadata` supports every field of the vocabulary, and does no
    input or output: the metadata of a file is read by
    [`FileBasedMetadata.load`][brainhops.io.metadata.FileBasedMetadata.load].

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

    # The type of raw record that the class declares, as the type argument
    # of a generic base (`FileBasedMetadata[nb.Nifti1Header]`), set by the
    # metaclass: `type(None)` for `[None]`. `None` on `Metadata`, which
    # keeps any record (see `raw`, and `_accepts_raw` below).
    _raw_class: tx.ClassVar[tx.Optional[type]] = None

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
        # Hidden from `repr` when empty, as well as when absent.
        HideIf(operator.not_),
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

    def copy(self) -> tx.Self:
        """
        Copy this metadata, so that the copy can be edited without editing
        the original.

        An image or a transformation holds a copy when it is given
        metadata that another object already holds (through `replace()`,
        `from_other` or `metadata=`).

        The raw record is shared, as `replace()` shares it, while the
        read-time snapshot and `extra` are copied, so that editing the
        copy never edits the original.

        Returns
        -------
        Metadata
            The copy, of the same class.
        """
        new = copy.copy(self)
        new.extra = copy.copy(new.extra)
        new._snapshot = copy.copy(self._snapshot)
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
        obj, found = _convert_from(target, self, (), values)
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
        obj, report = _convert_from(cls, other, args, kwargs)
        apply_loss_policy(report, stacklevel=3)
        return obj

    # --- propagation --------------------------------------------------

    def derive(
        self,
        operation: tx.Optional[Operation] = None,
        *,
        history: _History = None,
    ) -> tx.Self:
        """
        Build the metadata of an image computed from the image this
        metadata describes.

        Without an operation, the derived image lies on the same axes,
        as a smoothed or a denoised version of it does, and every field
        is kept. An image operation that changes axes describes what it
        did as an [`Operation`][brainhops.datamodel.metadata.Operation]:
        `image[index]` gives an
        [`Indexed`][brainhops.datamodel.metadata.Indexed], and
        `image.reslice(...)` a
        [`Resampled`][brainhops.datamodel.metadata.Resampled]. Each
        field then propagates through it, by the handler of its value
        or of its scope (see `brainhops.datamodel.metadata._operations`):
        an index along a time or a channel axis selects the entries of
        the fields that run along it, and a change of the spatial axes
        clears the fields tied to the spatial sampling, except an
        encoding direction, which follows the map of the voxel axes.

        Either way, three provenance fields record the derivation:
        `creation_time` is cleared, the entries of `history` are appended
        to `history`, and `generated_by` gains a brainhops entry, once.

        Parameters
        ----------
        operation : Operation, optional
            What the image operation did to the axes of the image.
        history : str or sequence of str, optional
            A description of the derivation, appended to `history`: one
            entry, or several.

        Returns
        -------
        Metadata
            New metadata, of the same class. Generic metadata derived this
            way carries no raw record, since no format hook can scrub it.
            The metadata of a file format keeps a copy of its record,
            propagated through the operation by the handler of the
            record's type (see `propagate_raw`), and a copy of its
            read-time snapshot.

        Examples
        --------
        ```python
        meta = Metadata(description="T1w", history=("acquired",))
        meta.derive(history="smooth").history  # ('acquired', 'smooth')
        ```
        """
        values = self._derive_values(operation=operation, history=history)
        return type(self)(**values)

    # --- internals ----------------------------------------------------

    def _derive_values(
        self,
        *,
        operation: tx.Optional[Operation],
        history: _History,
    ) -> tx.Dict[str, tx.Any]:
        """
        The constructor values of `derive`, field by field: each value
        propagated through `operation` (see `propagate`), and the
        derivation recorded in the provenance fields.
        """
        values: tx.Dict[str, tx.Any] = {}
        for name in FIELDS:
            value = getattr(self, name, None)
            if value is UNSUPPORTED or name in self.unsupported_fields:
                continue
            if name == "extra":
                value = dict(value or {})
            if operation is not None:
                value = propagate(
                    value,
                    operation,
                    name=name,
                    scope=SCOPES.get(name, Scope.FILE),
                    source=self,
                )
            values[name] = value
        _derive_provenance(values, history)
        return values


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _convert_from(
    cls: tx.Type[Metadata],
    other: Metadata,
    args: tx.Tuple[tx.Any, ...] = (),
    kwargs: tx.Optional[tx.Dict[str, tx.Any]] = None,
) -> tx.Tuple[Metadata, ConversionReport]:
    """
    Convert metadata into a class, as `Metadata.from_instance` does, and
    return the report instead of acting on it.

    It works across the metadata classes, which do not override it: each
    value of the vocabulary is copied, except where `cls` cannot store
    the field, and is then reported as lost. The raw record and its
    snapshot go along when `cls` keeps the record (see `_accepts_raw`).

    Parameters
    ----------
    cls : type
        The metadata class to convert into.
    other : Metadata
        The metadata to convert.
    args : tuple, optional
        Positional arguments of the constructor.
    kwargs : dict, optional
        Fields to set on the result.

    Returns
    -------
    metadata : Metadata
        The converted metadata: of `cls`, or of the class of `other` when
        `other` already stands for `cls` (a copy).
    report : ConversionReport
        What was lost.
    """
    kwargs = dict(kwargs or {})
    same = _is_already(other, cls)
    # A copy keeps the most specific class.
    target = type(other) if same else cls
    report = ConversionReport(
        source=_format_name(other), target=_format_name(target)
    )
    values: tx.Dict[str, tx.Any] = {}
    if same or (other.raw is not None and _accepts_raw(target, other.raw)):
        # The record and its snapshot go along, shared and copied as
        # `copy()` does: to a copy, to the generic hub, or back to the
        # format whose class declares the type of the record.
        values["raw"] = other.raw
        values["snapshot"] = copy.copy(other._snapshot)
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
    values.update(kwargs)
    return target(*args, **values), report


def _accepts_raw(cls: type, raw: tx.Any) -> bool:
    """
    Whether a conversion into a metadata class keeps a raw record.

    A class keeps a record of the type it declares (`_raw_class`), and a
    class that declares none (`None`: generic `Metadata`) keeps any
    record. Formats declare distinct types, so that a record only ever
    goes back to its own format; `FileBasedMetadata` declares
    `type(None)` until a format declares its own type, so that a format
    keeps no record of another one.

    Parameters
    ----------
    cls : type
        The metadata class converted into.
    raw : object
        The raw record of the source metadata, not `None`.

    Returns
    -------
    bool
        Whether the record is kept.
    """
    declared = cls._raw_class
    return declared is None or isinstance(raw, declared)


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
        # Not at the top: `brainhops` imports the data model before it
        # defines `__version__`.
        from brainhops import __version__ as version
    except ImportError:  # pragma: no cover
        version = None
    return entries + (GeneratedBy(name="brainhops", version=version),)


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
