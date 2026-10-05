"""`Metadata`: the common vocabulary, the hub through which formats convert."""

__all__ = ["Metadata"]

# stdlib
import copy

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, fields

# internals
from brainhops._core.typing import ArrayLike

from ..base import DataModelBase
from ._meta import MetadataMeta
from ._report import (
    ConversionReport,
    OnLoss,
    apply_loss_policy,
)
from ._sentinel import UNSUPPORTED, Maybe
from ._terms import EncodingDirection, GeneratedBy
from ._vocabulary import (
    FIELDS,
    FILE,
    GRID,
    SCOPES,
    VOCABULARY,
    VOLUME,
    DiffusionMetadata,
    DisplayMetadata,
    MicroscopyMetadata,
    MRIMetadata,
    ProvenanceMetadata,
    Scope,
    TransformMetadata,
)


class Metadata(
    DataModelBase,
    ProvenanceMetadata,
    MRIMetadata,
    DiffusionMetadata,
    DisplayMetadata,
    MicroscopyMetadata,
    TransformMetadata,
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

    The vocabulary is declared by six groups, which `Metadata` inherits:
    [`ProvenanceMetadata`][], [`MRIMetadata`][], [`DiffusionMetadata`][],
    [`DisplayMetadata`][], [`MicroscopyMetadata`][] and
    [`TransformMetadata`][]. Each field holds a value, `None` when the
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
        Scope(FILE),
        # Not `Factory()`: inferred from `Maybe[...]`, a union with
        # `None`, the default would be `None`.
        Factory(dict),
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

        Returns
        -------
        Metadata
            The copy, of the same class. `extra` is copied as well.
        """
        new = copy.copy(self)
        extra = self.__dict__.get("extra")
        if isinstance(extra, dict):
            new.__dict__["extra"] = dict(extra)
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

    def derive(
        self,
        *,
        grid_changed: bool = False,
        grid_map: tx.Optional[ArrayLike] = None,
        volumes: tx.Optional[tx.Sequence[int]] = None,
        volumes_changed: bool = False,
        step: tx.Optional[str] = None,
    ) -> tx.Self:
        """
        The metadata of an object derived from this one (resampled,
        cropped, a selection of volumes...). Always a new object.

        - `file` fields are kept, except `creation_time` (cleared) and
          `history`, to which `step` is appended; `generated_by` gains a
          brainhops entry once.
        - `acquisition` fields are kept.
        - `grid` fields are cleared when `grid_changed`, with one
          exception: given `grid_map`, the linear part of the map from
          the old voxel axes to the new ones, an encoding direction in
          voxel axes is mapped through it (`normalize(grid_map @ v)`)
          instead. A direction in a named world space is kept.
        - `volume` fields are indexed by `volumes` (the selected volume
          indices) when given, and cleared when `volumes_changed` and no
          selection is known; `display_range` and `data_unit`, one value
          for every volume, are kept.
        - `extra` is kept verbatim.

        The metadata of a file format also keeps its raw record (a copy,
        scrubbed by the format of what the grid or the volumes bound) and
        its snapshot, so that a field cleared here is cleared in the
        record on write.

        Examples
        --------
        Reslicing onto a grid whose voxel axes are the old ones swapped
        (`i <-> j`) or flipped: the phase encoding direction, given in
        voxel axes, follows the axes; slice timing, which belongs to the
        acquired slices, does not survive a resampling:

        ```python
        meta = Metadata(phase_encoding_direction="j-",
                        slice_timing=(0, 0.5, 1, 1.5), repetition_time=2)
        swap = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        new = meta.derive(grid_changed=True, grid_map=swap, step="reslice")
        new.phase_encoding_direction  # EncodingDirection('i-')
        new.slice_timing              # None
        new.repetition_time           # 2 (acquisition: kept)
        meta.derive(grid_changed=True, grid_map=np.diag([1, -1, 1]))
        # -> phase_encoding_direction EncodingDirection('j')
        meta.derive(grid_changed=True)  # no map: the direction is cleared
        ```

        Selecting volumes 0-2 of a DWI keeps the matching b-values and
        b-vectors (world frame: a reslicing leaves them alone):

        ```python
        dwi = Metadata(bvalues=(0, 1000, 1000, 2000),
                       bvectors=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))
        dwi.derive(volumes=[0, 1, 2]).bvalues      # (0, 1000, 1000)
        dwi.derive(volumes_changed=True).bvalues   # None (unknown selection)
        ```
        """
        values = self._derive_values(
            grid_changed=grid_changed,
            grid_map=grid_map,
            volumes=volumes,
            volumes_changed=volumes_changed,
            step=step,
        )
        return type(self)(**values)

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
        same = fits(other, cls)
        # A copy keeps the most specific class.
        target = type(other) if same else cls
        report = ConversionReport(
            source=format_name(other), target=format_name(target)
        )
        values = other._format_state() if same else {}
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

    def _derive_values(
        self,
        *,
        grid_changed: bool,
        grid_map: tx.Any,
        volumes: tx.Optional[tx.Sequence[int]],
        volumes_changed: bool,
        step: tx.Optional[str],
    ) -> tx.Dict[str, tx.Any]:
        """The constructor values of `derive`: a field rule
        (`_DERIVE_RULES`) where there is one, else its scope's."""
        values: tx.Dict[str, tx.Any] = {}
        for name in FIELDS:
            value = getattr(self, name, None)
            if value is UNSUPPORTED or name in self.unsupported_fields:
                continue
            scope = SCOPES.get(name)
            if name in _DERIVE_RULES:
                value = _DERIVE_RULES[name](value, step)
            elif scope == GRID and grid_changed:
                value = _map_direction(value, grid_map)
            elif scope == VOLUME and value is not None:
                value = _select_volumes(value, volumes, volumes_changed)
            values[name] = value
        return values

    def _format_state(self) -> tx.Dict[str, tx.Any]:
        """The constructor values a copy in the same format carries over
        besides the fields (`FileBasedMetadata`: the raw record and the
        snapshot). None here."""
        return {}


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


# The fields `derive` treats by name rather than by scope: field ->
# `rule(value, step)`, the derived value.
_DERIVE_RULES: tx.Dict[str, tx.Callable[[tx.Any, tx.Any], tx.Any]] = {
    "creation_time": lambda value, step: None,  # a new object
    "history": lambda value, step: _appended(value, step),
    "generated_by": lambda value, step: _with_brainhops(value),
    "extra": lambda value, step: dict(value or {}),
    # One value for every volume: a selection keeps it.
    "display_range": lambda value, step: value,
    "data_unit": lambda value, step: value,
}


def _appended(history: tx.Any, step: tx.Optional[str]) -> tx.Any:
    return history if step is None else tuple(history or ()) + (step,)


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


def _map_direction(value: tx.Any, grid_map: tx.Any) -> tx.Any:
    """A `grid` field after a grid change: an encoding direction in voxel
    axes goes through `grid_map` when it is given (and fits), one in a
    world space is kept; anything else is cleared."""
    if not isinstance(value, EncodingDirection):
        return None
    if value.space is not None:
        return value
    if grid_map is None:
        return None
    matrix = np.asarray(grid_map, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(value.vector):
        return None
    try:
        return value.transform(matrix)
    except ValueError:
        return None


def _select_volumes(
    value: tx.Any,
    volumes: tx.Optional[tx.Sequence[int]],
    volumes_changed: bool,
) -> tx.Any:
    """A `volume` field after a change of volumes: indexed by the
    selection when it is known, cleared when it is not."""
    if volumes is not None:
        try:
            return tuple(value[i] for i in volumes)
        except (IndexError, TypeError):
            return None
    return None if volumes_changed else value


def fits(value: tx.Any, cls: type) -> bool:
    """Whether a metadata object is one of `cls` already: of the class
    itself, or of a subclass of a format class (generic `Metadata` holds
    generic metadata only)."""
    if type(value) is cls:
        return True
    return cls is not Metadata and isinstance(value, cls)


def format_name(obj: tx.Any) -> str:
    if isinstance(obj, type):
        for field in fields(obj):
            if field.name == "format":
                default = field.default
                if isinstance(default, str):
                    return default
        return obj.__name__
    return str(getattr(obj, "format", type(obj).__name__))


def _metadata_class(target: tx.Any) -> tx.Type[Metadata]:
    """The class a `to()` target names: a class, or a format name."""
    if isinstance(target, type) and issubclass(target, Metadata):
        return target
    if isinstance(target, str):
        if target == "generic":
            return Metadata
        stack = list(Metadata.__subclasses__())
        while stack:
            klass = stack.pop()
            if format_name(klass) == target:
                return klass
            stack.extend(klass.__subclasses__())
        raise ValueError(f"No metadata class for the format {target!r}.")
    raise TypeError(
        f"Expected a Metadata subclass or a format name, got {target!r}."
    )
