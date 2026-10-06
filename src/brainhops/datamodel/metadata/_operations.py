"""
Operations: what an image operation did to the axes of an image, and how
each metadata value follows it.

An image operation that changes axes (`image[index]`,
`image.reslice(...)`) describes what it did as an [`Operation`][]:
[`Indexed`][] for an index, [`Resampled`][] for a resampling. It hands
the operation to
[`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive], which
gives every value of the metadata to [`propagate`][], and the raw record
of a format to [`propagate_raw`][]. Both look up a handler registered
with [`propagates`][], in this order:

1. a handler of the type of the value (or of the record), the most
   specific class of the value first, and, for each class, the most
   specific class of the operation first;
2. for a vocabulary value, a handler of the scope of its field (a
   [`Scope`][brainhops.datamodel.metadata.Scope] member), the most
   specific class of the operation first;
3. otherwise, a value is kept, and a record is deep-copied.

A handler lives next to the type it handles: that of an encoding
direction in `_terms`, the scope defaults in `_vocabulary`, that of the
NIfTI header next to `NiftiMetadata`. A format whose raw record holds
content tied to some axes, outside the vocabulary, registers a handler
for the class of its record, which is its own (never a widely used type
such as `dict`).

This module imports neither the transformations nor the geometry at
the top: the transformations import the metadata (their `metadata`
field). The two computations that need them import them when they run.
"""

__all__ = [
    "Handler",
    "Indexed",
    "Operation",
    "Resampled",
    "propagate",
    "propagate_raw",
    "propagates",
]

# stdlib
import copy

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import Field, NoEq, NoInit, NoRepr

# internals
from ..base import DataModelBase
from ..enums import AxisType
from ..systems import CoordinateSystem, _axes_or_unknown
from ._sentinel import UNSUPPORTED

if tx.TYPE_CHECKING:  # pragma: no cover
    from .._transformations.base import Transformation
    from ..geometry import Geometry
    from ._base import Metadata

Handler = tx.Callable[..., tx.Any]
"""
A propagation handler, `handler(value, operation, *, name, source)`: the
value after the operation. `name` is the name of the field (`"raw"` for
the raw record), and `source` is the metadata the value is read from.
"""


class Operation(DataModelBase, frozen=True, eq=False, repr=False):
    """
    What an image operation did to the axes of an image.

    An operation is handed to
    [`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive],
    which propagates each value of the metadata through it (see
    [`propagate`][]). A subclass that no handler names takes the
    handlers of its bases: the scope defaults read its `moves_space`,
    and the handler of an encoding direction its `voxel_map`. A
    subclass overrides those two properties; their defaults (the
    spatial axes moved, through an unknown map) clear whatever depends
    on the spatial axes.

    A warp would describe the local map of its voxels (its Jacobian),
    which an encoding direction could follow voxel by voxel; it is a
    possible extension, which no operation implements.
    """

    @property
    def moves_space(self) -> bool:
        """Whether the spatial axes changed (the default: they did)."""
        return True

    @property
    def voxel_map(self) -> tx.Optional[np.ndarray]:
        """
        The linear part of the map from the old voxel coordinates to the
        new ones (new axes by old axes), or `None` when it is unknown or
        not affine (the default).
        """
        return None


class Indexed(Operation):
    """
    An image indexed as its data array is, `image[index]`.

    The index is expanded once, on construction, into one component per
    axis of the data (`expanded`), and the type of each voxel axis is
    read from the coordinate system (`axes`).
    """

    index: tx.Annotated[
        tuple,
        tx.Doc("The full index, as given to `image[index]`."),
        Field(convert=False),
    ]
    shape: tx.Annotated[
        tx.Tuple[int, ...],
        tx.Doc("The shape of the indexed image."),
    ]
    system: tx.Annotated[
        tx.Optional[CoordinateSystem],
        tx.Doc(
            "The coordinate system of the voxels of the image (the input "
            "of its grid), which gives the type of each axis; `None` when "
            "it is not known."
        ),
        Field(convert=False),
    ] = None
    expanded: tx.Annotated[
        tuple,
        tx.Doc(
            "The index, with its `...` replaced by as many full slices as "
            "the axes it stands for, and the axes it leaves out filled at "
            "the end."
        ),
        NoInit(),
        NoRepr(),
        NoEq(),
    ] = ()
    axes: tx.Annotated[
        tx.Tuple[tx.Optional[AxisType], ...],
        tx.Doc(
            "The type of each axis of the image, `None` where it is not known."
        ),
        NoInit(),
        NoRepr(),
        NoEq(),
    ] = ()

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        index = self.index if isinstance(self.index, tuple) else (self.index,)
        ndim = len(self.shape)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "expanded", _expand_index(index, ndim))
        object.__setattr__(self, "axes", _axis_types(self.system, ndim))

    def positions(self, axis: AxisType) -> tx.Optional[np.ndarray]:
        """
        The positions that the index keeps along an axis of a type, in
        order.

        Parameters
        ----------
        axis : AxisType
            The type of the axis. When the image has several axes of this
            type, the first one is read.

        Returns
        -------
        np.ndarray or None
            The kept positions: the full range when the axis is
            untouched. `None` when the index drops the axis (an
            integer), when it keeps a selection that is not 1-D, when it
            is out of the range of the axis, or when the image has no
            axis of this type.
        """
        axis = AxisType(axis)
        old = 0
        for component in self.expanded:
            if component is None:
                continue
            if self.axes[old] is axis:
                try:
                    kept = np.arange(self.shape[old])[component]
                except IndexError:
                    return None
                return kept if np.ndim(kept) == 1 else None
            old += 1
        return None

    @property
    def moves_space(self) -> bool:
        """
        Whether the index changes a spatial axis, or an axis of unknown
        type, or moves one to another position (an integer or a `None`
        before it).
        """
        old = new = 0
        for component in self.expanded:
            if component is None:
                new += 1
                continue
            kind, size = self.axes[old], self.shape[old]
            changes = not (
                isinstance(component, slice)
                and range(*component.indices(size)) == range(size)
            )
            if kind in (None, AxisType.space) and (changes or old != new):
                return True
            old += 1
            if not isinstance(component, (int, np.integer)):
                new += 1
        return False

    @property
    def voxel_map(self) -> tx.Optional[np.ndarray]:
        """
        The linear part of the map from the old voxels to the new ones,
        with a zero column for each axis an integer dropped; `None` for
        an index other than integers, slices and `None`.
        """
        if not all(
            c is None or isinstance(c, (int, slice)) for c in self.expanded
        ):
            return None
        # Not at the top: see the docstring of the module.
        from ..geometry import _index2transform

        # The map of the index goes from the new voxels to the old ones;
        # its pseudo-inverse goes back.
        sub2full, _ = _index2transform(self.index, self.shape, self.system)
        matrix = np.asarray(sub2full.matrix, dtype=float)
        return np.linalg.pinv(matrix[:, :-1])


class Resampled(Operation):
    """
    An image resampled onto a new geometry, `image.reslice(geometry)`.
    """

    transformation: tx.Annotated[
        "Transformation",
        tx.Doc(
            "The map from the new voxel coordinates to the old ones, "
            "without the grid of the new geometry."
        ),
        Field(convert=False),
    ]
    geometry: tx.Annotated[
        tx.Optional["Geometry"],
        tx.Doc(
            "The geometry the image is resampled onto, or `None` when it "
            "is not known (a level of a pyramid, whose shape is not known "
            "until its data is read)."
        ),
        Field(convert=False),
    ] = None

    @property
    def moves_space(self) -> bool:
        """Always `True`: a resampling moves the spatial axes."""
        return True

    @property
    def voxel_map(self) -> tx.Optional[np.ndarray]:
        """
        The linear part of the inverse of `transformation`, from the old
        voxels to the new ones; `None` when it does not reduce to an
        affine.
        """
        # Not at the top: see the docstring of the module.
        from .._transformations.multiscale import _as_affine

        affine = _as_affine(self.transformation.inverse())
        if affine is None:
            return None
        return np.asarray(affine.matrix, dtype=float)[:, :-1]


def propagates(
    key: tx.Any, operation_type: tx.Type[Operation]
) -> tx.Callable[[Handler], Handler]:
    """
    Register a propagation handler: a decorator.

    Parameters
    ----------
    key : type or Scope
        The class of the values the handler propagates (a vocabulary
        value, or the class of a raw record), or a scope, whose handler
        propagates the values of the fields in that scope that no
        handler of their type takes.
    operation_type : type
        The class of the operations the handler takes, a subclass of
        [`Operation`][] (or `Operation` itself, for every operation).

    Returns
    -------
    callable
        The decorator, which registers the handler and returns it
        unchanged.

    Raises
    ------
    TypeError
        If `key` is `object`, which would take every value, or if
        `operation_type` is not an `Operation` class.
    """
    if key is object:
        raise TypeError("A handler of `object` would take every value.")
    if not (
        isinstance(operation_type, type)
        and issubclass(operation_type, Operation)
    ):
        raise TypeError(
            f"Expected an Operation class, got {operation_type!r}."
        )

    def register(handler: Handler) -> Handler:
        _HANDLERS[(key, operation_type)] = handler
        return handler

    return register


def propagate(
    value: tx.Any,
    operation: Operation,
    *,
    name: str,
    scope: tx.Any,
    source: "Metadata",
) -> tx.Any:
    """
    A value of the metadata after an operation, from the handler of its
    type, or else from that of its scope (see the lookup order in the
    docstring of the module). A value no handler takes is kept, and
    `UNSUPPORTED` always is.

    Parameters
    ----------
    value : object
        The value of the field.
    operation : Operation
        What the image operation did.
    name : str
        The name of the field.
    scope : Scope
        The scope of the field.
    source : Metadata
        The metadata the value is read from.

    Returns
    -------
    object
        The value of the field of the derived metadata.
    """
    if value is UNSUPPORTED:
        return value
    handler = _lookup(type(value).__mro__, operation)
    if handler is None:
        handler = _lookup((scope,), operation)
    if handler is None:
        return value
    return handler(value, operation, name=name, source=source)


def propagate_raw(
    raw: tx.Any,
    operation: tx.Optional[Operation],
    *,
    source: "Metadata",
) -> tx.Any:
    """
    The raw record of the metadata of a format after an operation, from
    the handler of its type, or else a deep copy of it.

    Parameters
    ----------
    raw : object
        The raw record, or `None`.
    operation : Operation, optional
        What the image operation did; `None` for a derivation on the same
        axes.
    source : Metadata
        The metadata the record belongs to.

    Returns
    -------
    object
        The record of the derived metadata: never the record itself, so
        that the derived metadata can edit it.
    """
    handler = None
    if operation is not None:
        handler = _lookup(type(raw).__mro__, operation)
    if handler is None:
        return copy.deepcopy(raw)
    return handler(raw, operation, name="raw", source=source)


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


# (class of value or scope, class of operation) -> handler.
_HANDLERS: tx.Dict[tx.Tuple[tx.Any, type], Handler] = {}


def _lookup(
    keys: tx.Iterable[tx.Any], operation: Operation
) -> tx.Optional[Handler]:
    """The handler of the first key that has one, the most specific class
    of the operation first, for each key."""
    for key in keys:
        for kind in type(operation).__mro__:
            handler = _HANDLERS.get((key, kind))
            if handler is not None:
                return handler
    return None


def _expand_index(
    index: tx.Tuple[tx.Any, ...], ndim: int
) -> tx.Tuple[tx.Any, ...]:
    """An index with its `...` replaced by as many full slices as the axes
    it stands for, and the axes it leaves out filled at the end, as
    `_index2transform` expands it."""
    # Compared by identity, so that an array in the index is not compared
    # with `...` element by element.
    at = next((i for i, c in enumerate(index) if c is ...), None)
    if at is None:
        index, at = (*index, ...), len(index)
    used = sum(1 for c in index if c is not None and c is not ...)
    fill = (slice(None),) * (ndim - used)
    return index[:at] + fill + index[at + 1 :]


def _axis_types(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> tx.Tuple[tx.Optional[AxisType], ...]:
    """The type of each voxel axis of an image, `None` where it is not
    known."""
    axes = _axes_or_unknown(system)
    if axes == [...]:
        return (None,) * ndim
    if axes.is_open:
        axes = axes.expand(ndim)
    kinds = []  # type: tx.List[tx.Optional[AxisType]]
    for axis in list(axes)[:ndim]:
        try:
            kinds.append(AxisType(getattr(axis, "type", None)))
        except ValueError:
            kinds.append(None)
    return tuple(kinds) + (None,) * (ndim - len(kinds))
