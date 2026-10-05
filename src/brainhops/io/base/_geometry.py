"""
Voxel-to-world geometry helpers shared by file formats.

Several formats (NIfTI, MGH, MRtrix, ...) store the geometry of an image
as a three-dimensional voxel-to-RAS affine. The helpers here reduce a
transformation to that matrix, and split off the axes that follow the
spatial ones (such as time), which those formats store apart. They also
place the axes of an image where such a format stores them, by the types
and names its spaces declare (see [`arrange_voxel_to_ras`][]). They
depend only on the datamodel and NumPy -- not on nibabel -- so that
formats that do not need nibabel can use them.
"""

__all__ = [
    "CHANNEL_TYPES",
    "RAS_FROM_ORIENTATION",
    "STORAGE_GROUPS",
    "Arrangement",
    "AxisLayout",
    "arrange_voxel_to_ras",
    "axis_group",
    "closed_world",
    "complete_basis",
    "declared_axes",
    "embed_affine",
    "plan_axes",
    "ras_conversion",
    "reduce_to_affine",
    "split_spatial",
]

# externals
import numpy as np
import typing_extensions as tx

# internals
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.orientation import Orientation
from brainhops.datamodel.systems import CoordinateSystem, _axes_or_unknown
from brainhops.datamodel.transformations import (
    Affine,
    ConversionError,
    Sequence,
    Transformation,
)
from brainhops.io.base.parsers import (
    UnrepresentableTransformationError,
    WriterError,
)

RAS_FROM_ORIENTATION = {
    "left-to-right": (0, 1.0),
    "right-to-left": (0, -1.0),
    "posterior-to-anterior": (1, 1.0),
    "anterior-to-posterior": (1, -1.0),
    "inferior-to-superior": (2, 1.0),
    "superior-to-inferior": (2, -1.0),
}
"""
The RAS axis and sign that an anatomical orientation points along.

Each key is the value of an anatomical orientation carried by an axis. The
first element of the pair is the index of the RAS axis the orientation runs
along, and the second is its sign. This drives the conversion of a
voxel-to-world affine into voxel-to-RAS from the axes themselves, rather
than from the world space's name.
"""


def ras_conversion(system: tx.Optional[CoordinateSystem]) -> np.ndarray:
    """
    The `(4, 4)` matrix that maps a world space's coordinates into RAS.

    The matrix is built from the anatomical orientation carried by each
    axis, not from the world space's name. An LPS space becomes a flip of
    the first two axes, an RSA space becomes a permutation, and a space
    already in RAS becomes the identity.

    The conversion is derived only when all three leading axes carry a
    recognized anatomical orientation. When any of them does not, the
    identity is returned, so a space with no orientation is stored as it
    is.
    """
    # An axis about which nothing is known, including the `...` of a
    # missing space, carries no orientation.
    axes = _axes_or_unknown(system)[:3]
    mapping = []
    for axis in axes:
        value = getattr(getattr(axis, "orientation", None), "value", None)
        if value not in RAS_FROM_ORIENTATION:
            return np.eye(4)
        mapping.append(RAS_FROM_ORIENTATION[value])
    if len(mapping) != 3:
        return np.eye(4)
    conversion = np.zeros((4, 4))
    conversion[3, 3] = 1.0
    for column, (row, sign) in enumerate(mapping):
        conversion[row, column] = sign
    return conversion


def reduce_to_affine(
    xform: Transformation, fmt: str, world: str = "world"
) -> Affine:
    """
    Reduce a voxel-to-world transformation to an `Affine`.

    An affine transformation is used directly. A transformation of any
    other kind that reduces to an affine, such as a `Scaling` or a
    `Sequence` of affines, is converted.

    Parameters
    ----------
    xform : Transformation
        The voxel-to-world transformation.
    fmt : str
        The name of the format, used in error messages.
    world : str
        The name the format gives its world space, used in error
        messages.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine representation, such as a
        displacement field.
    """
    reduced = xform.compute() if isinstance(xform, Sequence) else xform
    error = None
    affine = reduced
    if not isinstance(affine, Affine):
        try:
            affine = reduced.to(Affine)
        except ConversionError as exc:
            error = exc
    if not isinstance(affine, Affine):
        # A field returns itself from a conversion to `Affine`, and a
        # `Sequence` of a non-affine reduces to one, so the result has to
        # be checked rather than trusted.
        raise UnrepresentableTransformationError(
            f"A {type(xform).__name__} cannot be written as {fmt} "
            f"geometry: {fmt} stores an affine voxel-to-{world} matrix, "
            f"and this transformation has no affine representation."
        ) from error
    return affine


def embed_affine(
    matrix: np.ndarray, fmt: str, world: str = "world"
) -> np.ndarray:
    """
    Embed a homogeneous voxel-to-world matrix in a `(4, 4)` matrix.

    A two-dimensional map yields a `(3, 3)` homogeneous matrix, whose
    rotation and translation are placed in a `(4, 4)` matrix whose extra
    axis is the identity. A three-dimensional map is already `(4, 4)` and
    is returned unchanged.

    A map from fewer than three voxel axes into a three-dimensional world
    (a slice placed in space) already fixes the world origin of its
    missing voxel axes, but not their directions. Each is given a unit
    direction orthogonal to the voxel axes the map has (see
    [`complete_basis`][]): the normal of a slice, as DICOM and ITK give
    it, so the stored matrix is never singular.

    Parameters
    ----------
    matrix : ndarray
        The homogeneous voxel-to-world matrix.
    fmt : str
        The name of the format, used in error messages.
    world : str
        The name the format gives its world space, used in error
        messages.

    Raises
    ------
    WriterError
        If the map has more than three spatial dimensions.
    """
    out_dim = matrix.shape[0] - 1
    in_dim = matrix.shape[1] - 1
    if out_dim > 3 or in_dim > 3:
        raise WriterError(
            f"{fmt} stores a three-dimensional voxel-to-{world} affine, so "
            f"a {in_dim}D-to-{out_dim}D transformation cannot be written. "
            f"Reduce the transformation to three spatial dimensions before "
            f"writing it to {fmt}."
        )
    embedded = np.eye(4)
    embedded[:out_dim, :in_dim] = matrix[:out_dim, :in_dim]
    embedded[:out_dim, 3] = matrix[:out_dim, in_dim]
    if out_dim == 3 and in_dim < 3:
        embedded[:3, in_dim:3] = complete_basis(embedded[:3, :in_dim])
    return embedded


def complete_basis(columns: np.ndarray) -> np.ndarray:
    """
    Unit directions that complete the columns of a `(3, k)` block, `k < 3`.

    They are the directions of the voxel axes that a map from `k` voxel
    axes into a three-dimensional world does not state, with a spacing of
    one. Two columns that span a plane determine their complement up to
    its sense: it is their normalised cross product, the slice normal of
    DICOM and ITK, which makes the voxel axes right-handed. One column,
    or two that do not span a plane, do not determine it; the complement
    is then built from the world axes, `x`, `y` then `z`, each with the
    part along the columns (and the directions already chosen) removed,
    keeping the ones that remain longest. A column that is a world axis
    is thus completed by the other two, in order.
    """
    columns = np.asarray(columns, dtype=float).reshape(3, -1)
    missing = 3 - columns.shape[1]
    if missing <= 0:
        return np.zeros((3, 0))
    if columns.shape[1] == 2:
        normal = np.cross(columns[:, 0], columns[:, 1])
        scale = np.linalg.norm(columns[:, 0]) * np.linalg.norm(columns[:, 1])
        if np.linalg.norm(normal) > 1e-8 * max(scale, 1e-300):
            return (normal / np.linalg.norm(normal))[:, None]
    # >> An orthonormal basis of what the columns span.
    basis: tx.List[np.ndarray] = []
    for column in columns.T:
        residual = column - sum((b @ column) * b for b in basis)
        norm = np.linalg.norm(residual)
        if norm > 1e-8 * max(np.linalg.norm(column), 1e-300):
            basis.append(residual / norm)
    # >> The world axes, less what is already spanned, longest first.
    found = []
    while len(found) < missing:
        best = None
        for axis in np.eye(3):
            residual = axis - sum((b @ axis) * b for b in basis)
            norm = np.linalg.norm(residual)
            if best is None or norm > best[0] + 1e-12:
                best = (norm, residual)
        direction = best[1] / best[0]
        basis.append(direction)
        found.append(direction)
    # The directions are kept in the order of the world axes they come
    # from, so a column along `x` is completed by `y` then `z`.
    found.sort(key=lambda d: int(np.argmax(np.abs(d))))
    return np.stack(found, axis=1)


def split_spatial(
    matrix: np.ndarray, fmt: str, world: str = "world", nspace: int = 3
) -> tx.Tuple[np.ndarray, tx.List[tx.Tuple[float, float]]]:
    """
    Split a homogeneous voxel-to-world matrix into its spatial block and
    the scale and offset of each axis that follows the spatial ones.

    A format that stores a spatial affine stores the axes that follow the
    `nspace` spatial ones -- the time axis of a time series -- apart, each
    by a spacing and, for some, an origin. A map over more axes splits
    into those only when it does not couple the spatial axes with the
    others, and maps each other axis onto itself alone: its matrix is
    block-diagonal, with a diagonal second block.

    A map over at most `nspace` axes is returned unchanged, with no other
    axis.

    Parameters
    ----------
    matrix : ndarray
        The homogeneous voxel-to-world matrix, of shape
        `(n_out + 1, n_in + 1)`.
    fmt : str
        The name of the format, used in error messages.
    world : str
        The name the format gives its world space, used in error
        messages.
    nspace : int
        The number of leading axes that are spatial.

    Returns
    -------
    spatial : ndarray
        The homogeneous matrix of the spatial axes, of shape
        `(nspace + 1, nspace + 1)`, or `matrix` unchanged.
    others : list of (float, float)
        The scale and the offset of each axis that follows the spatial
        ones, in order.

    Raises
    ------
    UnrepresentableTransformationError
        If the map couples the spatial axes with the others, mixes two of
        the others, or maps different numbers of axes in and out.
    """
    n_out, n_in = matrix.shape[0] - 1, matrix.shape[1] - 1
    if n_out <= nspace and n_in <= nspace:
        return matrix, []
    if n_out != n_in:
        raise UnrepresentableTransformationError(
            f"{fmt} stores a {nspace}D voxel-to-{world} affine and a spacing "
            f"for each other axis, so a map from {n_in} to {n_out} axes "
            f"cannot be written."
        )
    linear = matrix[:n_out, :n_in]
    others = linear[nspace:, nspace:]
    coupled = (
        np.any(linear[:nspace, nspace:] != 0)
        or np.any(linear[nspace:, :nspace] != 0)
        or np.any(others != np.diag(np.diag(others)))
    )
    if coupled:
        raise UnrepresentableTransformationError(
            f"{fmt} stores a {nspace}D voxel-to-{world} affine over the "
            f"spatial axes and a spacing for each other axis (such as "
            f"time), so a map that mixes the spatial axes with the others, "
            f"or two of the others, cannot be written."
        )
    spatial = np.eye(nspace + 1)
    spatial[:nspace, :nspace] = matrix[:nspace, :nspace]
    spatial[:nspace, nspace] = matrix[:nspace, n_in]
    extra = [
        (float(matrix[d, d]), float(matrix[d, n_in]))
        for d in range(nspace, n_out)
    ]
    return spatial, extra


# ----------------------------------------------------------------------
#   PLACING AXES BY THEIR DECLARED TYPES
# ----------------------------------------------------------------------

STORAGE_GROUPS = ("space", "time", "channel", "other")
"""
The order in which a format that stores a spatial affine stores the axes
of an array, by group: the spatial axes first, then time, then the
components of a vector or the channels, then any other axis.
"""

CHANNEL_TYPES = ("channel", "displacement", "coordinate")
"""The axis types stored where the components of a vector are."""

_SPACE_NAMES = ("x", "y", "z")
"""The names that put spatial axes in their own order (see `plan_axes`)."""

_INDEX = "index"


def axis_group(axis: tx.Any) -> str:
    """The group of `STORAGE_GROUPS` that an axis is stored in."""
    type_ = getattr(axis, "type", None)
    if type_ in CHANNEL_TYPES:
        return "channel"
    if type_ in ("space", "time"):
        return str(type_)
    return "other"


def declared_axes(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> tx.Optional[tx.List[Axis]]:
    """
    The axes of a system, when they say where each of them is stored.

    They are the axes of `system` when it states exactly `ndim` of them
    and gives a type to at least one. Otherwise -- a missing or open
    system, one with another number of axes, or one whose axes carry no
    type at all, such as the `dim0, dim1, ...` of a plain array -- the
    system says nothing about where its axes go, and `None` is returned:
    the axes are then taken in the format's positional order.

    An axis with no type, in a system that types others, is neither
    spatial nor temporal: it is stored after them.
    """
    axes = getattr(system, "axes", None)
    if axes is None:
        return None
    axes = list(axes)
    if len(axes) != ndim or any(not isinstance(a, Axis) for a in axes):
        return None
    if all(getattr(a, "type", None) is None for a in axes):
        return None
    return axes


def _groups(axes: tx.Sequence[Axis]) -> tx.List[str]:
    """
    The group of each axis. When there is a time axis, the first axes of
    no type fill the spatial slots the spatial axes leave before it: an
    axis of no type may be spatial.
    """
    groups = [axis_group(axis) for axis in axes]
    if "time" in groups:
        untyped = [i for i, a in enumerate(axes) if a.type is None]
        free = max(0, 3 - groups.count("space"))
        for i in untyped[:free]:
            groups[i] = "space"
    return groups


class AxisLayout(tx.NamedTuple):
    """
    Where a format stores each axis of an array.

    `order[i]` is the axis of the array stored at position `i`, before
    the singleton axes are inserted; `inserted` lists the positions, in
    the stored array, of the singleton axes the format needs to place the
    others (a `z` of size one for a slice that has other axes, or a time
    axis before the channels), in increasing order. `axes` and `groups`
    are the axes stored at every position, inserted ones included, and
    their groups (`STORAGE_GROUPS`).
    """

    order: tx.List[int]
    inserted: tx.List[int]
    axes: tx.List[Axis]
    groups: tx.List[str]

    @property
    def trivial(self) -> bool:
        """Whether the array is stored as it is."""
        return not self.inserted and self.order == sorted(self.order)

    def apply(self, data: tx.Any) -> tx.Any:
        """
        Put an array in this layout: transpose it, then insert the
        singleton axes. Both are views, so a lazy (dask) array stays
        lazy.
        """
        if self.order != sorted(self.order):
            data = get_array_backend(data).transpose(data, self.order)
        for position in self.inserted:
            data = get_array_backend(data).expand_dims(data, axis=position)
        return data


def plan_axes(
    axes: tx.Sequence[Axis],
    side: str,
    fmt: str,
    *,
    fill_space: bool = True,
    fill_time: bool = False,
    time_slot: tx.Optional[int] = None,
    max_nonspatial: tx.Optional[int] = None,
) -> AxisLayout:
    """
    Place declared axes in the order a format stores them.

    The spatial axes come first, then the time axis, then the channel-like
    axes, then the others (`STORAGE_GROUPS`), each group in its declared
    order -- except that spatial axes that are all named among `x`, `y`
    and `z` are put in that order. A format stores at most three spatial
    axes and one time axis.

    Parameters
    ----------
    axes : sequence of Axis
        The declared axes.
    side : str
        `"voxel"` or `"world"`, used in error messages.
    fmt : str
        The name of the format, used in error messages.
    fill_space : bool
        When there are one or two spatial axes and any other axis, insert
        singleton spatial axes so the others follow three spatial ones,
        as the format reads them.
    fill_time : bool
        When there is no time axis and any channel or other axis, insert
        a singleton time axis before them, so they are not read as time.
    time_slot : int, optional
        The position the format stores time at. A time axis that cannot
        be put there raises `UnrepresentableTransformationError`.
    max_nonspatial : int, optional
        The number of axes, besides the spatial ones, the format stores.

    Raises
    ------
    UnrepresentableTransformationError
        If the axes do not fit the format.
    """
    axes = list(axes)
    groups = _groups(axes)
    nspace, ntime = groups.count("space"), groups.count("time")
    nother = len(axes) - nspace
    if nspace > 3:
        raise UnrepresentableTransformationError(
            f"{fmt} stores at most three spatial axes, but the {side} space "
            f"declares {nspace}."
        )
    if ntime > 1:
        raise UnrepresentableTransformationError(
            f"{fmt} stores one time axis, but the {side} space declares "
            f"{ntime}."
        )
    if max_nonspatial is not None and nother > max_nonspatial:
        raise UnrepresentableTransformationError(
            f"{fmt} stores {max_nonspatial} axis besides the spatial ones, "
            f"but the {side} space declares {nother}."
        )

    spatial = [i for i, g in enumerate(groups) if g == "space"]
    names = [getattr(axes[i], "name", None) for i in spatial]
    if all(n in _SPACE_NAMES for n in names) and len(set(names)) == len(names):
        spatial.sort(key=lambda i: _SPACE_NAMES.index(axes[i].name))
    rest = [i for i, g in enumerate(groups) if g != "space"]
    rest.sort(key=lambda i: (STORAGE_GROUPS.index(groups[i]), i))
    order = spatial + rest

    stored = [axes[i] for i in order]
    stored_groups = [groups[i] for i in order]
    inserted = []
    if fill_space and 0 < nspace < 3 and nother:
        free = [n for n in _SPACE_NAMES if n not in names]
        if not all(n in _SPACE_NAMES for n in names):
            free = [f"dim{k}" for k in range(nspace, 3)]
        for k in range(nspace, 3):
            stored.insert(k, Axis(free[k - nspace], "space", unit=_INDEX))
            stored_groups.insert(k, "space")
            inserted.append(k)
        nspace = 3
    if ntime and time_slot is not None and nspace != time_slot:
        raise UnrepresentableTransformationError(
            f"{fmt} stores the time axis after {time_slot} spatial axes, "
            f"but the {side} space declares {nspace} spatial axes."
        )
    if fill_time and not ntime and nother:
        stored.insert(nspace, Axis("t", "time", unit=_INDEX))
        stored_groups.insert(nspace, "time")
        inserted.append(nspace)
    return AxisLayout(order, inserted, stored, stored_groups)


def closed_world(
    system: tx.Optional[CoordinateSystem], ndim: int, fmt: str
) -> tx.Optional[CoordinateSystem]:
    """
    The world space, closed to the `ndim` axes the affine maps into.

    A format cannot store an open world space, one whose axes hold `...`,
    so it is closed from the shape of the voxel-to-world matrix. The axes
    that `...` stands for carry no orientation, as any axis the format
    knows nothing about. A world space that states more axes than the
    matrix has rows raises `WriterError`.
    """
    if system is None or system.ndim is not None:
        return system
    try:
        return system.expand(ndim)
    except ValueError as error:
        raise WriterError(
            f"The world space of this transformation states more axes than "
            f"the {ndim} its voxel-to-world matrix maps into, so it cannot "
            f"be written as {fmt} geometry."
        ) from error


class Arrangement(tx.NamedTuple):
    """
    The geometry of an image, with its axes placed where a format stores
    them (see [`arrange_voxel_to_ras`][]).

    `matrix` is the `(4, 4)` voxel-to-RAS matrix of the spatial axes, and
    `others` the scale and offset of each axis stored after them, in
    order. `layout` says where each axis of the data is stored, or is
    `None` when the data is stored as it is, by the format's positional
    convention. `voxel_groups` and `world_groups` are the groups of the
    stored voxel and world axes, when known, and `world` the world space
    with its axes in stored order. `filled_time` says whether the time
    axis is a singleton the layout inserted.
    """

    matrix: np.ndarray
    others: tx.List[tx.Tuple[float, float]]
    layout: tx.Optional[AxisLayout]
    voxel_groups: tx.Optional[tx.List[str]]
    world_groups: tx.Optional[tx.List[str]]
    world: tx.Optional[CoordinateSystem]
    filled_time: bool


_ANATOMICAL_POSITIVE = (
    "left-to-right",
    "posterior-to-anterior",
    "inferior-to-superior",
)


def _filled_world_axis(
    world_axes: tx.Sequence[Axis], groups: tx.Sequence[str], voxel_axis: Axis
) -> Axis:
    """
    The world axis of a spatial axis a layout inserted, in a world space
    whose spatial axes are fewer than three.

    It has the unit of the other spatial axes, and, when they carry
    anatomical orientations along two different RAS axes, the orientation
    along the third, in its positive sense, so the space still turns into
    RAS.
    """
    spatial = [a for a, g in zip(world_axes, groups) if g == "space"]
    unit = next(
        (a.unit for a in spatial if getattr(a, "unit", None) is not None),
        None,
    )
    names = [getattr(a, "name", None) for a in spatial]
    name = voxel_axis.name
    if name in names:
        name = next(
            (n for n in _SPACE_NAMES if n not in names), f"dim{len(names)}"
        )
    orientation = None
    used = set()
    for axis in spatial:
        value = getattr(getattr(axis, "orientation", None), "value", None)
        if value in RAS_FROM_ORIENTATION:
            used.add(RAS_FROM_ORIENTATION[value][0])
    if len(used) == len(spatial) == 2:
        (left,) = {0, 1, 2} - used
        orientation = Orientation(
            type="anatomical", value=_ANATOMICAL_POSITIVE[left]
        )
    return Axis(name, "space", unit=unit, orientation=orientation)


def arrange_voxel_to_ras(
    xform: tx.Any,
    voxel_axes: tx.Optional[tx.Sequence[Axis]],
    fmt: str,
    world_name: str = "world",
    **policy: tx.Any,
) -> Arrangement:
    """
    The voxel-to-RAS geometry of a transformation, with the axes placed
    where a format stores them.

    The transformation is reduced to an affine (see [`reduce_to_affine`][]).
    The axes are placed by what the spaces declare, not by position. The
    voxel axes are `voxel_axes`, the axes of the data (see
    [`declared_axes`][]), and the world axes are those of the affine's
    output space. Each side is put in the order the format stores (see
    [`plan_axes`][], which `policy` is passed to): its spatial axes, then
    its time axis, then the others. A side that declares nothing is taken
    in the format's positional order -- except a world space that declares
    nothing, of as many axes as the voxel space, which is taken to list
    its axes as the voxel space does.

    When the layout inserts singleton axes in the data, the matrix gains
    their columns, and the world the rows it lacks:

    * A spatial axis inserted next to the spatial axes of a world that has
      as few is mapped onto a world axis of its own, by the identity:
      scale one, offset zero.
    * A spatial axis inserted in a voxel space whose map already reaches a
      three-dimensional world (an injective map, such as a slice placed in
      space) takes its world origin from the map, which states it, and
      its direction from [`complete_basis`][], with a spacing of one.
    * An inserted time axis is mapped onto a world time axis by the
      identity, when the world has none, and counts frames
      (`filled_time`).

    The spatial block is then split from the other axes (see
    [`split_spatial`][]), embedded in a `(4, 4)` matrix (see
    [`embed_affine`][]), and turned into RAS from the anatomical
    orientation of the world axes (see [`ras_conversion`][]).

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine form, the axes do not fit the
        format, or the map mixes the spatial axes with the others.
    WriterError
        If the data, reordered, does not have as many axes as the map.
    """
    affine = reduce_to_affine(xform, fmt, world_name)
    matrix = affine.homogeneous_matrix
    matrix = np.eye(4) if matrix is None else np.asarray(matrix, float)
    n_out, n_in = matrix.shape[0] - 1, matrix.shape[1] - 1

    # >> The voxel side, in the order the data is stored in.
    layout = None
    if voxel_axes is not None:
        layout = plan_axes(voxel_axes, "voxel", fmt, **policy)
        if len(voxel_axes) != n_in:
            if layout.order != sorted(layout.order):
                raise WriterError(
                    f"The data has {len(voxel_axes)} axes, which are "
                    f"reordered to be written as {fmt}, but this "
                    f"voxel-to-{world_name} transformation maps {n_in}."
                )
            # The map does not say where the data's axes go: the data is
            # stored as it is.
            layout = None
        else:
            matrix = matrix[:, layout.order + [n_in]]
    vgroups = None
    if layout is not None:
        vgroups = [
            g for k, g in enumerate(layout.groups) if k not in layout.inserted
        ]

    # >> The world side, in the same order.
    world = closed_world(getattr(affine, "output", None), n_out, fmt)
    world_axes = declared_axes(world, n_out)
    stated = list(getattr(world, "axes", None) or [])
    wgroups = None
    rows = None
    if world_axes is not None:
        wgroups = _groups(world_axes)
        rows = plan_axes(world_axes, "world", fmt, fill_space=False).order
        world_axes = [world_axes[i] for i in rows]
        wgroups = [wgroups[i] for i in rows]
    elif layout is not None and n_out == n_in:
        rows = list(layout.order)
        wgroups = list(vgroups)
        if len(stated) == n_out and Ellipsis not in stated:
            world_axes = [stated[i] for i in rows]
    elif vgroups is not None:
        # The world lists its axes by position: as many non-spatial axes
        # as the voxel space, after its spatial ones.
        nonspatial = [g for g in vgroups if g != "space"]
        nspace = n_out - len(nonspatial)
        if 0 < nspace <= 3:
            wgroups = ["space"] * nspace + nonspatial
    if rows is not None:
        matrix = matrix[rows + [n_out], :]

    # >> The singleton axes the layout inserts.
    filled_time = False
    unplaced = []
    if layout is not None and layout.inserted:
        for position in layout.inserted:
            group = layout.groups[position]
            matrix = np.insert(matrix, position, 0.0, axis=1)
            if wgroups is None:
                continue
            wspace = wgroups.count("space")
            if group == "space":
                vspace = layout.groups[:position].count("space")
                if wspace == vspace and wspace < 3:
                    matrix = np.insert(matrix, wspace, 0.0, axis=0)
                    matrix[wspace, position] = 1.0
                    if world_axes is not None:
                        filled = _filled_world_axis(
                            world_axes, wgroups, layout.axes[position]
                        )
                        world_axes.insert(wspace, filled)
                    wgroups.insert(wspace, "space")
                else:
                    unplaced.append(position)
            elif group == "time" and "time" not in wgroups:
                matrix = np.insert(matrix, wspace, 0.0, axis=0)
                matrix[wspace, position] = 1.0
                wgroups.insert(wspace, "time")
                filled_time = True
                if world_axes is not None:
                    world_axes.insert(wspace, Axis("t", "time", unit=_INDEX))
        if unplaced and wgroups is not None and wgroups.count("space") == 3:
            # The map reaches a three-dimensional world from fewer voxel
            # axes: the missing ones point away from those it has.
            given = [k for k in range(3) if k not in unplaced]
            matrix[:3, unplaced] = complete_basis(matrix[:3, given])
        vgroups = list(layout.groups)

    if world_axes is not None:
        world = CoordinateSystem(axes=world_axes)
    spatial, others = split_spatial(matrix, fmt, world_name)
    embedded = embed_affine(spatial, fmt, world_name)
    return Arrangement(
        ras_conversion(world) @ embedded,
        others,
        layout,
        vgroups,
        wgroups,
        world,
        filled_time,
    )
