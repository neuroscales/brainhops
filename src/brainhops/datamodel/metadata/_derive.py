"""
What an image operation did to the axes of an image, and how each field
of the metadata follows it.

An image operation that changes axes or values describes what it did
with one of three records: `image[index]` builds an [`Indexed`][],
`image.reslice(...)` builds a [`Resampled`][], and `image.reframe(...)`
builds a [`Reframed`][]. The image hands the
record to
[`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive], which
applies the rule of the scope of each field, in one loop. The functions
of this module are the steps of that loop that need a computation: the
entries of a field along an axis, and an encoding direction mapped
through the voxels.

This module imports neither the transformations nor the geometry at the
top, because the transformations import the metadata. The two
computations that need them import them when they run.
"""

__all__ = ["Indexed", "Reframed", "Resampled"]

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import Field

# internals
from ..base import DataModelBase
from ..enums import AxisType
from ..systems import CoordinateSystem
from ._terms import EncodingDirection, GeneratedBy

if tx.TYPE_CHECKING:  # pragma: no cover
    from .._transformations.base import Transformation
    from ..geometry import Geometry


class Indexed(DataModelBase, frozen=True, eq=False, repr=False, kw_only=False):
    """
    An image indexed as its data array is, `image[index]`.

    The image builds this record and passes it to `Metadata.derive`. The
    index is expanded once, on construction, into one component per axis
    of the data (`expanded`), and the type of each voxel axis is read
    from the coordinate system (`axes`).
    """

    # The record is built by `SingleScaleImage.__getitem__`, never by a
    # user, from values that already have the right type. Converting the
    # index would also change its meaning: the tuple converter turns a
    # list such as `[0, 2]`, which selects two positions along one axis,
    # into the tuple `(0, 2)`, which selects one voxel.
    index: tx.Annotated[tuple, Field(convert=False)]
    """The full index, as given to `image[index]`."""

    shape: tx.Tuple[int, ...]
    """The shape of the indexed image."""

    system: tx.Annotated[
        tx.Optional[CoordinateSystem], Field(convert=False)
    ] = None
    """
    The coordinate system of the voxels of the image (the input of its
    grid), which gives the type of each axis; `None` when it is not known.
    """

    @property
    def expanded(self) -> tx.Tuple[tx.Any, ...]:
        """The index, with its `...` replaced by full slices and filled."""
        return _expand_index(self.index, len(self.shape))

    @property
    def axes(self) -> tx.Tuple[tx.Optional[AxisType], ...]:
        """The type of each axis of the image, `None` where it is not known."""
        return _axis_types(self.system, len(self.shape))

    def positions(self, axis: AxisType) -> tx.Optional[np.ndarray]:
        """
        Return the positions that the index keeps along an axis of a type.

        Parameters
        ----------
        axis : AxisType
            The type of the axis. When the image has several axes of this
            type, the first one is read.

        Returns
        -------
        np.ndarray or None
            The kept positions, in order: the full range when the axis is
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
        with a zero column for each axis an integer dropped, or `None`
        for an index other than integers, slices and `None`.
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


class Resampled(
    DataModelBase, frozen=True, eq=False, repr=False, kw_only=False
):
    """
    An image resampled onto a new geometry, `image.reslice(geometry)`.

    The image builds this record and passes it to `Metadata.derive`.
    """

    transformation: "Transformation" = Field(convert=False)
    """
    The map from the new voxel coordinates to the old ones, without the
    grid of the new geometry.
    """

    geometry: tx.Optional["Geometry"] = Field(default=None, convert=False)
    """
    The geometry the image is resampled onto, or `None` when it is not
    known (a level of a pyramid, whose shape is not known until its data
    is read).
    """

    @property
    def moves_space(self) -> bool:
        """Always `True`, because a resampling moves the spatial axes."""
        return True

    @property
    def voxel_map(self) -> tx.Optional[np.ndarray]:
        """
        The linear part of the inverse of `transformation`, from the old
        voxels to the new ones, or `None` when it does not reduce to an
        affine.
        """
        # Not at the top: see the docstring of the module.
        from .._transformations.compute.multiscale import _as_affine

        affine = _as_affine(self.transformation.inverse())
        if affine is None:
            return None
        return np.asarray(affine.matrix, dtype=float)[:, :-1]


class Reframed(
    DataModelBase, frozen=True, eq=False, repr=False, kw_only=False
):
    """
    The values of an image re-expressed in another frame,
    `image.reframe(...)`.

    The entries along a component axis (a `point`, `vector`,
    `displacement`, `coordinate` or `tensor` axis) are the components of
    one vector or tensor, and the image re-expressed them in another
    coordinate system. No voxel moves, but the numbers in the data
    change, so the fields that describe the values (the `VALUES` scope)
    no longer apply. Neither do the fields with one entry per index along
    a re-expressed axis, which described the components in the old frame
    and which may no longer match the length of the axis, because a
    tensor stored with only some of its components is stored with all of
    them afterwards. The image builds this record and passes it to
    `Metadata.derive`.
    """

    axes: tx.Tuple[AxisType, ...] = ()
    """The type of each axis whose values were re-expressed."""

    @property
    def moves_space(self) -> bool:
        """Always `False`, because no voxel moves."""
        return False


# ----------------------------------------------------------------------
#   HELPERS OF `Metadata.derive`
# ----------------------------------------------------------------------


def mapped_direction(
    value: EncodingDirection, operation: tx.Union[Indexed, Resampled]
) -> tx.Optional[EncodingDirection]:
    """
    Return an encoding direction after an operation that moved space.

    A direction expressed in a world space does not move with the
    voxels, so it is kept. A direction in voxel axes goes through the
    map of the voxels (`operation.voxel_map`). That map may have more
    axes than the direction, which lies in the first three axes, as for
    a 4-D image: the direction is then mapped as a vector of the old
    axes that is zero beyond its own, and is kept only when it still
    lies in the first axes.

    Parameters
    ----------
    value : EncodingDirection
        The direction.
    operation : Indexed or Resampled
        What the image operation did.

    Returns
    -------
    EncodingDirection or None
        The mapped direction, or `None` when the map is unknown, when it
        moves the direction off the first axes, or when it maps the
        direction to zero because its axis was dropped.
    """
    if value.space is not None:
        return value
    linear = operation.voxel_map
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


def entries_along(value: tx.Any, operation: Indexed, axis: AxisType) -> tx.Any:
    """
    Return the entries of a field that the index keeps along an axis.

    The field has one entry per position along the first axis of the
    given type. It is kept when the image has no such axis, or when the
    index leaves that axis untouched.

    Parameters
    ----------
    value : sequence
        The entries of the field.
    operation : Indexed
        The index.
    axis : AxisType
        The type of the axis the entries run along.

    Returns
    -------
    tuple or None
        The kept entries, or `None` when the index drops the axis, when
        the selection is not known, or when a kept position is beyond
        the entries of the field.
    """
    if axis not in operation.axes:
        return value
    positions = operation.positions(axis)
    size = operation.shape[operation.axes.index(axis)]
    if positions is not None and np.array_equal(positions, np.arange(size)):
        return value
    if positions is None or positions.dtype.kind not in "iu":
        return None
    try:
        count = len(value)
    except TypeError:
        return None
    if positions.size and not (
        0 <= positions.min() <= positions.max() < count
    ):
        return None
    return tuple(value[int(i)] for i in positions)


def with_history(
    history: tx.Optional[tx.Tuple[str, ...]],
    steps: tx.Union[str, tx.Sequence[str], None],
) -> tx.Optional[tx.Tuple[str, ...]]:
    """
    Return a history with the steps of a derivation appended.

    Parameters
    ----------
    history : tuple of str, optional
        The history of the source metadata.
    steps : str or sequence of str, optional
        One step, or several.

    Returns
    -------
    tuple of str or None
        The new history, or `history` itself when there is no step.
    """
    if isinstance(steps, str):
        steps = (steps,)
    steps = tuple(steps or ())
    if not steps:
        return history
    return tuple(history or ()) + steps


def with_brainhops(
    generated_by: tx.Optional[tx.Tuple[GeneratedBy, ...]],
) -> tx.Tuple[GeneratedBy, ...]:
    """
    Return the programs that produced the data, with brainhops added once.

    Parameters
    ----------
    generated_by : tuple of GeneratedBy, optional
        The programs of the source metadata.

    Returns
    -------
    tuple of GeneratedBy
        The programs, with a brainhops entry at the end unless one is
        there already.
    """
    entries = tuple(generated_by or ())
    if any(entry.name == "brainhops" for entry in entries):
        return entries
    try:
        # Not at the top: `brainhops` imports the data model before it
        # defines `__version__`.
        from brainhops import __version__ as version
    except ImportError:  # pragma: no cover
        version = None
    return entries + (GeneratedBy(name="brainhops", version=version),)


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


def _expand_index(
    index: tx.Tuple[tx.Any, ...], ndim: int
) -> tx.Tuple[tx.Any, ...]:
    """Return an index with its `...` replaced by as many full slices as
    the axes it stands for, and the axes it leaves out filled at the
    end, as `_index2transform` expands it."""
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
    """Return the type of each voxel axis of an image, `None` where it is
    not known."""
    if system is None:
        return (None,) * ndim
    axes = system.axes
    if axes.is_variadic:
        axes = axes.expand(ndim)
    kinds = []  # type: tx.List[tx.Optional[AxisType]]
    for axis in list(axes)[:ndim]:
        try:
            kinds.append(AxisType(getattr(axis, "type", None)))
        except ValueError:
            kinds.append(None)
    return tuple(kinds) + (None,) * (ndim - len(kinds))
