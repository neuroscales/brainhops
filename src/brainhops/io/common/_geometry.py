"""Voxel-to-RAS geometry shared by formats that store a 3D affine.

Formats such as NIfTI, MGH and MRtrix store a three-dimensional voxel-to-RAS
matrix, with trailing axes such as time stored apart. These helpers reduce a
transformation to that matrix and place image axes where the format stores
them; [`arrange_voxel_to_ras`][] combines them. They do not need nibabel.
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

import numpy as np
import typing_extensions as tx

from brainhops.backends import get_array_backend
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Sequence,
    Transformation,
)
from brainhops.errors import ConversionError
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
"""The RAS axis index and sign of each anatomical orientation."""


def ras_conversion(system: tx.Optional[CoordinateSystem]) -> np.ndarray:
    """Return the (4, 4) matrix that maps world coordinates to RAS.

    The matrix follows the anatomical orientations of the first three axes, not
    the name of the space: LPS flips two axes, RSA permutes them. If an axis
    has no recognized orientation, the identity is returned.
    """
    # An axis about which nothing is known, such as the ... of a missing space,
    # has no orientation.
    axes = get_axes(system)[:3]
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
    """Reduce a voxel-to-world transformation to an [`Affine`][].

    An affine is returned as it is, and a transformation that reduces to one,
    such as a `Scaling` or a `Sequence` of affines, is converted.

    Parameters
    ----------
    xform : Transformation
        The voxel-to-world transformation.
    fmt, world : str
        The names of the format and of the world space, for error messages.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine representation, as a displacement
        field has not.
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
        # A field returns itself from conversion to Affine, and a Sequence of a
        # non-affine reduces to one, so the result must be checked.
        raise UnrepresentableTransformationError(
            f"A {type(xform).__name__} cannot be written as {fmt} "
            f"geometry: {fmt} stores an affine voxel-to-{world} matrix, "
            f"and this transformation has no affine representation."
        ) from error
    return affine


def embed_affine(
    matrix: np.ndarray, fmt: str, world: str = "world"
) -> np.ndarray:
    """Embed a homogeneous voxel-to-world matrix in a (4, 4) matrix.

    Missing dimensions are completed by the identity. When fewer than three
    voxel axes map into a 3D world, as for a slice, the missing voxel axes get
    orthogonal unit directions from [`complete_basis`][], so that the matrix is
    never singular.

    Raises
    ------
    WriterError
        If either side has more than three dimensions.
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
    """Complete the k < 3 columns of a (3, k) block with unit directions.

    Two columns spanning a plane are completed by their normalized cross
    product, the slice normal of DICOM and ITK, which makes the axes
    right-handed. Otherwise, each new direction is the world axis with the
    longest part outside what is already spanned.
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
    # Orthonormal basis of what the columns span.
    basis: tx.List[np.ndarray] = []
    for column in columns.T:
        residual = column - sum((b @ column) * b for b in basis)
        norm = np.linalg.norm(residual)
        if norm > 1e-8 * max(np.linalg.norm(column), 1e-300):
            basis.append(residual / norm)
    # World axes minus what is already spanned, longest first.
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
    # Keep the order of the world axes the directions come from.
    found.sort(key=lambda d: int(np.argmax(np.abs(d))))
    return np.stack(found, axis=1)


def split_spatial(
    matrix: np.ndarray, fmt: str, world: str = "world", nspace: int = 3
) -> tx.Tuple[np.ndarray, tx.List[tx.Tuple[float, float]]]:
    """Split a voxel-to-world matrix into its spatial block and other axes.

    Formats store the axes after the `nspace` spatial ones, such as time, as a
    spacing and an offset. The map must keep spatial axes and other axes apart,
    and map each other axis onto itself. A map over at most `nspace` axes is
    returned unchanged.

    Returns
    -------
    spatial : ndarray
        The (nspace + 1, nspace + 1) spatial block, or the unchanged matrix.
    others : list of (float, float)
        The scale and offset of each trailing axis.

    Raises
    ------
    UnrepresentableTransformationError
        If the numbers of input and output axes differ, or the map couples axes
        that must stay apart.
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
"""The order in which formats store array axes, by group.

Spatial axes come first, then time, then channels or vector components, then
any other.
"""

CHANNEL_TYPES = ("channel", "displacement", "coordinate")
"""Axis types stored where channels are stored."""

_SPACE_NAMES = ("x", "y", "z")
"""Names that put spatial axes in their own order, used by [`plan_axes`][]."""

_INDEX = "index"


def axis_group(axis: tx.Any) -> str:
    """Return the storage group of an axis, one of [`STORAGE_GROUPS`][]."""
    type_ = getattr(axis, "type", None)
    if type_ in CHANNEL_TYPES:
        return "channel"
    if type_ in ("space", "time"):
        return str(type_)
    return "other"


def declared_axes(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> tx.Optional[tx.List[Axis]]:
    """Return the axes of a system if they say where each axis is stored.

    This requires exactly `ndim` axes, at least one of them typed. Otherwise
    the result is `None`, and the axes follow the positional order of the
    format. An untyped axis among typed ones is stored after the spatial and
    time axes.
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
    """Return the group of each axis.

    When a time axis is present, the first untyped axes fill the spatial slots
    left before it.
    """
    groups = [axis_group(axis) for axis in axes]
    if "time" in groups:
        untyped = [i for i, a in enumerate(axes) if a.type is None]
        free = max(0, 3 - groups.count("space"))
        for i in untyped[:free]:
            groups[i] = "space"
    return groups


class AxisLayout(tx.NamedTuple):
    """Where a format stores each axis of an array.

    `order[i]` is the array axis stored at position `i`, and `inserted` lists
    the increasing positions of singleton axes the format needs, such as the
    `z` of a slice. `axes` and `groups` describe every stored position.
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
        """Arrange an array as the layout stores it.

        Transposition and singleton insertion are views, so lazy arrays such as
        dask arrays stay lazy.
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
    """Place declared axes in the storage order of a format.

    Axes are stored by group, in the order of [`STORAGE_GROUPS`][], and in
    declared order within a group, except that spatial axes all named among
    `x`, `y` and `z` are sorted.

    Parameters
    ----------
    side, fmt : str
        The side, `"voxel"` or `"world"`, and the format, for error messages.
    fill_space : bool
        With one or two spatial axes and other axes, insert singleton spatial
        axes up to three.
    fill_time : bool
        With no time axis and other non-spatial axes, insert a singleton time
        axis before them.
    time_slot : int, optional
        The position at which the format stores time.
    max_nonspatial : int, optional
        The number of non-spatial axes the format can store.

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
    """Close an open world space, whose axes hold `...`, to `ndim` axes.

    Formats cannot store open spaces. The added axes carry no orientation, and
    a world that states more than `ndim` axes raises [`WriterError`][].
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
    """Geometry of an image with its axes placed where a format stores them.

    `matrix` is the (4, 4) voxel-to-RAS matrix of the spatial axes, and
    `others` the scale and offset of each later axis. `layout` is `None` when
    the data is stored as it is, and `filled_time` tells whether the time axis
    was inserted.
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
    """Create the world axis of an inserted spatial axis.

    It takes the unit of the other spatial axes and, if two of them are
    oriented along different RAS axes, the orientation along the third.
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
    """Compute the voxel-to-RAS geometry of a transformation for a format.

    The transformation is reduced by [`reduce_to_affine`][], and axes are
    placed by what the spaces declare: the voxel side by [`plan_axes`][] with
    `policy`, and the world side, the output space of the affine, without
    filling. A side that declares nothing keeps its positional order; an
    undeclared world with as many axes as the voxel space is assumed to list
    them in the same way.

    Singleton data axes inserted by the layout gain matrix columns, and the
    world gains rows as needed:

    - an inserted spatial axis next to the spatial axes of a world with as few
      is mapped to a new world axis by the identity;
    - an inserted spatial axis of a map already reaching a 3D world, such as a
      slice, takes its direction from [`complete_basis`][];
    - an inserted time axis is mapped to a new world time axis when the world
      has none, which sets `filled_time`.

    The matrix is then split by [`split_spatial`][], embedded by
    [`embed_affine`][] and converted by [`ras_conversion`][].

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine form or its axes do not fit the
        format.
    WriterError
        If the data must be reordered but has another number of axes than the
        map.
    """
    affine = reduce_to_affine(xform, fmt, world_name)
    matrix = affine.homogeneous_matrix
    matrix = np.eye(4) if matrix is None else np.asarray(matrix, float)
    n_out, n_in = matrix.shape[0] - 1, matrix.shape[1] - 1

    # Voxel side, in data storage order.
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
            # The map does not say where the data axes go, so the data is
            # stored as is.
            layout = None
        else:
            matrix = matrix[:, layout.order + [n_in]]
    vgroups = None
    if layout is not None:
        vgroups = [
            g for k, g in enumerate(layout.groups) if k not in layout.inserted
        ]

    # World side, in the same order.
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
        # A positional world lists as many non-spatial axes as the voxel space,
        # after its spatial ones.
        nonspatial = [g for g in vgroups if g != "space"]
        nspace = n_out - len(nonspatial)
        if 0 < nspace <= 3:
            wgroups = ["space"] * nspace + nonspatial
    if rows is not None:
        matrix = matrix[rows + [n_out], :]

    # Singleton axes inserted by the layout.
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
            # The map reaches a 3D world from fewer voxel axes: the missing
            # ones point
            # away from those it has.
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
