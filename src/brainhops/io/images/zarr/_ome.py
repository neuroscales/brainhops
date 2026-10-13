"""OME-Zarr multiscale metadata of images.

OME-Zarr stores an image pyramid as a Zarr group whose metadata lists
the array of each level and the transformations that place the levels in
space. The metadata is parsed and validated by abczarr, so metadata that
contradicts the schema is rejected when it is parsed. Whatever version
the store uses, the metadata is then normalized to OME-NGFF 0.6, which
describes each level by a single transformation. Each OME transformation
becomes a brainhops transformation of the same kind rather than being
flattened into an affine.
"""

import typing_extensions as tx
from abczarr.abc.sync import ZarrGroup
from abczarr.ome import v0_6 as _v06
from abczarr.ome.v0_6.images import Dataset, Multiscale
from abczarr.ome.v0_6.ome import OME
from abczarr.ome.v0_6.transformations import CoordinateTransformation
from bagof.magic import replace

from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    CoordinatesField,
    DisplacementField,
    Identity,
    Sequence,
    Transformation,
)
from brainhops.io.base.parsers import ParserContentError, WriterError
from brainhops.io.images.zarr import _axisorder
from brainhops.io.transformations.zarr import _map, _node
from brainhops.io.transformations.zarr._axes import _to_axis

# Spline degree of each OME interpolation name.
_INTERPOLATION_DEGREE = {"nearest": 0, "linear": 1, "bspline-cubic": 3}

# An `_Entry` is the JSON description of one OME transformation, and a
# `_Level` pairs the path of a level array with its entry.
_Entry = tx.Dict[str, tx.Any]
_Level = tx.Tuple[str, _Entry]

# Version to which the reader normalizes every group.
NORMALIZED_VERSION = "0.6"

# Default write version. Version 0.4 requires Zarr v2, which brainhops does not
# write, and 0.6 is the only released version that carries every placement. The
# version is pinned rather than tracking abczarr, because it decides which
# other tools can read the output.
DEFAULT_WRITE_VERSION = "0.6"


class OmeImageError(ParserContentError):
    """Error raised when an OME-Zarr image group cannot be read.

    The group either has no multiscale metadata, or its metadata does not
    parse into a valid pyramid.
    """


def looks_like_multiscale(node: ZarrGroup) -> bool:
    """Return whether a group carries multiscale image metadata.

    The metadata is found by abczarr under the `ome` attribute (0.5 and
    later) or among the top-level attributes (0.4). Other kinds of OME
    metadata, such as labels, do not count.

    !!! note
        Any error raised while reading the metadata makes the function
        return `False`. Reading the group with [`read_multiscale`][] reports
        the specific fault.
    """
    try:
        ome = node.ome
    except Exception:
        return False
    return bool(getattr(ome, "multiscales", None))


def read_multiscale(
    node: ZarrGroup,
) -> tx.Tuple[tx.Optional[Multiscale], tx.Optional[str]]:
    """Read the first multiscale of a group, normalized to OME-NGFF 0.6.

    Returns
    -------
    multiscale : Multiscale or None
        The first multiscale, or `None` if the group has no OME metadata or
        its metadata names no multiscale.
    source_version : str or None
        The OME version found in the group, or `None` when there is no
        multiscale.

    Raises
    ------
    OmeImageError
        If the metadata cannot be read or normalized.
    """
    try:
        ome = node.ome
    except Exception as error:
        raise OmeImageError(
            "This OME-Zarr group carries metadata that could not be read as "
            "a valid image pyramid. " + str(error)
        ) from error
    if ome is None:
        return None, None
    source_version = getattr(ome, "version", None)
    try:
        normalized = ome.to_version(NORMALIZED_VERSION)
    except Exception as error:
        raise OmeImageError(
            "This OME-Zarr image could not be converted to a form with one "
            "transformation per level. " + str(error)
        ) from error
    multiscales = getattr(normalized, "multiscales", None)
    if not multiscales:
        return None, None
    return multiscales[0], source_version


def _output_system(multiscale: Multiscale) -> tx.Any:
    # Return the system named by the output of the first dataset
    # transformation that names one, or the first declared system when no
    # transformation names a declared system.
    systems = list(multiscale.coordinateSystems)
    for dataset in multiscale.datasets:
        for transform in dataset.coordinateTransformations:
            output = getattr(transform, "output", None)
            name = getattr(output, "name", None)
            if isinstance(name, str):
                for system in systems:
                    if system.name == name:
                        return system
    return systems[0]


def multiscale_axes(multiscale: Multiscale) -> tx.List[Axis]:
    """Return the axes of a multiscale as brainhops axes, in stored order."""
    system = _output_system(multiscale)
    return [_to_axis(axis.to_json()) for axis in system.axes]


def intrinsic_name(multiscale: Multiscale) -> tx.Optional[str]:
    """Return the name of the intrinsic space onto which the levels map.

    Without common transformations, the intrinsic space is the world space,
    as described by
    [`MultiScaleImage`][brainhops.datamodel.images.MultiScaleImage].
    """
    return getattr(_output_system(multiscale), "name", None)


def system_axes(multiscale: Multiscale) -> tx.Dict[str, tx.List[Axis]]:
    """Return the axes of every named coordinate system, in stored order.

    There is one system for the intrinsic space and one for each world space.
    """
    axes = {}  # type: tx.Dict[str, tx.List[Axis]]
    for system in multiscale.coordinateSystems:
        name = getattr(system, "name", None)
        if isinstance(name, str):
            axes[name] = [_to_axis(axis.to_json()) for axis in system.axes]
    return axes


def _make_read_field(
    node: tx.Optional[ZarrGroup], store_axes: tx.Optional[tx.Sequence[Axis]]
) -> tx.Optional[tx.Callable]:
    # Build the callback with which `_map.from_ome` reads a displacement or
    # coordinate field. The callback reads the array that the transformation
    # names from `node`, and lays the field out in the brainhops order.
    if node is None:
        return None

    def read_field(transform: tx.Any, kind: str) -> Transformation:
        path = getattr(transform, "path", None)
        if not isinstance(path, str):
            raise OmeImageError(
                f"This OME-Zarr {kind} field names no array, so its field "
                "cannot be read."
            )
        field_node = _node.follow_path(node, path)
        raw = _node.read_array(field_node)
        typed = _node.typed_axes(field_node, field_node.ndim)
        if typed is not None:
            # Sorting the typed axes into the brainhops order puts the spatial
            # axes first and the component axis last.
            data = get_array_backend().transpose(
                raw, _axisorder.to_canonical(typed)
            )
        else:
            # A field array without OME metadata can only be laid out from
            # its dimension names.
            data = _field_from_names(raw, field_node, store_axes, kind, path)
        degree = _INTERPOLATION_DEGREE.get(
            getattr(transform, "interpolation", None), 1
        )
        field_cls = (
            DisplacementField if kind == "displacements" else CoordinatesField
        )
        return field_cls(field=data, degree=degree)

    return read_field


def _field_from_names(
    raw: tx.Any,
    field_node: tx.Any,
    store_axes: tx.Optional[tx.Sequence[Axis]],
    kind: str,
    path: str,
) -> tx.Any:
    # Every dimension named after an image axis is spatial, and the single
    # remaining dimension holds the vector components.
    names = _node.dimension_names(field_node)
    if names is None:
        raise OmeImageError(
            f"The {kind} field at {path!r} has neither OME metadata nor "
            "dimension names, so brainhops cannot tell which axis holds the "
            "vector components."
        )
    store_axes = list(store_axes or [])
    image_names = [axis.name for axis in store_axes]
    spatial_dims = [i for i, name in enumerate(names) if name in image_names]
    component_dims = [i for i in range(len(names)) if i not in spatial_dims]
    if len(component_dims) != 1 or len(spatial_dims) != len(store_axes):
        raise OmeImageError(
            f"The {kind} field at {path!r} has axes {tuple(names)}, which do "
            f"not match the image axes {tuple(image_names)} together with a "
            "single component axis."
        )
    canonical_axes = _axisorder.permute(
        store_axes, _axisorder.to_canonical(store_axes)
    )
    by_name = {names[i]: i for i in spatial_dims}
    target = [by_name[axis.name] for axis in canonical_axes] + component_dims
    return get_array_backend().transpose(raw, target)


def _map_transform(
    transform: CoordinateTransformation,
    perm: tx.Sequence[int],
    ndim: int,
    node: tx.Optional[ZarrGroup] = None,
    store_axes: tx.Optional[tx.Sequence[Axis]] = None,
) -> Transformation:
    read_field = _make_read_field(node, store_axes)
    try:
        return _map.from_ome(transform, perm, ndim, read_field)
    except _map.OmeMappingError as error:
        raise OmeImageError(str(error)) from error


def level_transformation(
    dataset: Dataset,
    perm: tx.Sequence[int],
    ndim: int,
    node: tx.Optional[ZarrGroup] = None,
    store_axes: tx.Optional[tx.Sequence[Axis]] = None,
    input: tx.Optional[CoordinateSystem] = None,
    output: tx.Optional[CoordinateSystem] = None,
) -> Transformation:
    """Return the voxel-to-intrinsic transformation of one level.

    A dataset is the metadata entry of one level. Only the transformations
    declared on the dataset are mapped here. The transformations declared on
    the whole multiscale belong to the pyramid and are read by
    [`common_transformations`][]. Several transformations form a
    [`Sequence`][] in application order, and a dataset without
    transformations yields an [`Identity`][].

    Parameters
    ----------
    dataset : Dataset
        OME dataset of the level.
    perm : sequence of int
        Permutation from the stored axis order to the brainhops order.
    ndim : int
        Dimensionality of the level.
    node : ZarrGroup, optional
        Group in which fields named by the transformations are looked up.
    store_axes : sequence of Axis, optional
        Image axes in stored order, used to lay out fields that have no OME
        metadata.
    input : CoordinateSystem, optional
        Input coordinate system attached to the result.
    output : CoordinateSystem, optional
        Output coordinate system attached to the result.

    Raises
    ------
    OmeImageError
        If a transformation cannot be mapped.
    """
    transforms = list(dataset.coordinateTransformations)
    if not transforms:
        return Identity(input=input, output=output)
    mapped = [
        _map_transform(one, perm, ndim, node, store_axes) for one in transforms
    ]
    if len(mapped) == 1:
        return replace(mapped[0], input=input, output=output)
    return Sequence(mapped, input=input, output=output)


def common_transformations(
    multiscale: Multiscale,
    perm: tx.Sequence[int],
    ndim: int,
    node: tx.Optional[ZarrGroup] = None,
    store_axes: tx.Optional[tx.Sequence[Axis]] = None,
    input: tx.Optional[CoordinateSystem] = None,
    systems: tx.Optional[tx.Mapping[str, CoordinateSystem]] = None,
) -> tx.List[Transformation]:
    """Return the intrinsic-to-world transformations shared by every level.

    The common transformations of a multiscale are declared once for the
    whole pyramid. They carry the intrinsic space, onto which every level
    maps, to one or more world spaces. Each transformation goes from one
    coordinate system to another, so together they form a graph, which is
    walked starting from the intrinsic space. A transformation that starts
    from a space reached by an earlier step extends the path to that space,
    and the path then becomes a [`Sequence`][].

    The result holds one transformation per world space reached, ordered by
    the last declaration that reaches each space, so that the last
    transformation is the preferred one, as
    [`MultiScaleImage`][brainhops.datamodel.images.MultiScaleImage]
    expects. The list is empty when the multiscale declares no common
    transformations, which is the usual case.

    Parameters
    ----------
    multiscale : Multiscale
        Normalized multiscale metadata.
    perm : sequence of int
        As in [`level_transformation`][].
    ndim : int
        As in [`level_transformation`][].
    node : ZarrGroup, optional
        As in [`level_transformation`][].
    store_axes : sequence of Axis, optional
        As in [`level_transformation`][].
    input : CoordinateSystem, optional
        Intrinsic coordinate system, from which the walk starts.
    systems : mapping of str to CoordinateSystem, optional
        Named coordinate systems, attached as the outputs of the results.

    Raises
    ------
    OmeImageError
        If a transformation starts from a space that cannot be reached from
        the intrinsic space, or if a transformation cannot be mapped.
    """
    common = getattr(multiscale, "coordinateTransformations", None)
    if not isinstance(common, list) or not common:
        return []
    systems = dict(systems or {})
    root = getattr(input, "name", None)

    edges = [
        (
            getattr(getattr(one, "input", None), "name", None),
            getattr(getattr(one, "output", None), "name", None),
            _map_transform(one, perm, ndim, node, store_axes),
        )
        for one in common
    ]

    # `paths` maps each reached space to the transformations that lead to it
    # from the intrinsic space. `order` lists the reached spaces by their last
    # declaration, so that the preferred space stays last.
    paths = {}  # type: tx.Dict[tx.Optional[str], tx.List[Transformation]]
    order = []  # type: tx.List[tx.Optional[str]]
    pending = list(edges)
    while pending:
        progress = False
        for edge in list(pending):
            source, target, transform = edge
            if source in paths:
                # The transformation continues a path that an earlier one
                # started. Versions 0.4 and 0.5 write such chains.
                prefix = paths[source]
            elif source is None or source == root:
                prefix = []
            else:
                continue
            pending.remove(edge)
            progress = True
            paths[target] = prefix + [transform]
            if target in order:
                order.remove(target)
            order.append(target)
        if not progress:
            unreachable = sorted(
                str(source) for source, _, _ in pending if source is not None
            )
            raise OmeImageError(
                "This OME multiscale places its levels through a coordinate "
                f"system that nothing reaches: {', '.join(unreachable)}. Its "
                "transformations do not start from the space the levels are "
                "mapped onto, so the pyramid cannot be placed."
            )

    transformations = []  # type: tx.List[Transformation]
    for target in order:
        parts = paths[target]
        output = systems.get(target) if isinstance(target, str) else None
        if len(parts) == 1:
            transformations.append(
                replace(parts[0], input=input, output=output)
            )
        else:
            transformations.append(Sequence(parts, input=input, output=output))
    return transformations


def axis_to_json(axis: Axis) -> tx.Dict[str, tx.Any]:
    """Return the OME-Zarr JSON description of one axis."""
    entry = {}  # type: tx.Dict[str, tx.Any]
    name = getattr(axis, "name", None)
    type_ = getattr(axis, "type", None)
    unit = getattr(axis, "unit", None)
    entry["name"] = name if isinstance(name, str) else ""
    if isinstance(type_, str):
        entry["type"] = type_
    if unit is not None:
        value = getattr(unit, "value", unit)
        if isinstance(value, str):
            entry["unit"] = value
    return entry


# Versions that only carry per-axis scale and translation.
_SCALE_ONLY_VERSIONS = frozenset({"0.1", "0.2", "0.3", "0.4", "0.5"})


def resolve_write_version(
    version: tx.Optional[str],
    source_version: tx.Optional[str],
    rich: bool = False,
) -> str:
    """Choose the OME-NGFF version to write.

    An explicit `version` is used as given. Otherwise, the version of the
    source store is used, or `DEFAULT_WRITE_VERSION` when there is no source
    version. A true `rich` states that the image is placed by a
    transformation that only later versions can carry, such as a rotation
    or an affine. In that case, a scale-only version that was chosen
    implicitly is raised to `NORMALIZED_VERSION`, whereas a scale-only
    version that was requested explicitly is refused.

    Raises
    ------
    WriterError
        If `rich` is true and the requested version is scale-only.
    """
    chosen = version or source_version or DEFAULT_WRITE_VERSION
    if rich and chosen in _SCALE_ONLY_VERSIONS:
        if version is not None:
            raise WriterError(
                f"This image is placed by a transformation that OME-NGFF "
                f"{version} cannot carry, such as a rotation or an affine. "
                f"Write it in version {NORMALIZED_VERSION} instead, which "
                "carries the full placement."
            )
        return NORMALIZED_VERSION
    return chosen


# Name of the world coordinate system that the writer emits.
WORLD_SYSTEM = "physical"

# Name of the intrinsic system. The writer emits the intrinsic system only
# when there are common transformations. Otherwise, the levels map directly
# onto `WORLD_SYSTEM`.
INTRINSIC_SYSTEM = "intrinsic"


def _level_transforms(
    entry: _Entry, path: str, output: str
) -> tx.List[_Entry]:
    refs = {"input": {"path": path}, "output": {"name": output}}
    return [dict(entry, **refs)]


def resolve_world_names(
    names: tx.Sequence[tx.Optional[str]],
) -> tx.List[str]:
    """Name the world coordinate system of each common transformation.

    Existing names are kept, so that they survive a round trip. An unnamed
    system is called `WORLD_SYSTEM`, followed by its position when it is not
    the first one. A name that is already taken, including the intrinsic
    name, gets a number appended, so that no two systems share a name.
    """
    used = {INTRINSIC_SYSTEM}
    resolved = []  # type: tx.List[str]
    for position, name in enumerate(names):
        candidate = name or (
            WORLD_SYSTEM if position == 0 else WORLD_SYSTEM + str(position)
        )
        base, suffix = candidate, 1
        while candidate in used:
            candidate = base + str(suffix)
            suffix += 1
        used.add(candidate)
        resolved.append(candidate)
    return resolved


def build_ome(
    axes: tx.Sequence[Axis],
    levels: tx.Sequence[_Level],
    commons: tx.Sequence[tx.Tuple[str, _Entry]],
    name: tx.Optional[str],
    version: str,
) -> OME:
    """Build the typed OME metadata of a pyramid.

    The metadata is built in OME-NGFF 0.6 and then converted to `version`.
    The intrinsic system is emitted only when `commons` is not empty, so
    that scale-only versions are not given a graph of coordinate systems
    that they cannot express.

    Parameters
    ----------
    axes : sequence of Axis
        Axes in stored order.
    levels : sequence of tuple
        Array path and OME transformation of each level.
    commons : sequence of tuple
        World system name and OME transformation from the intrinsic space,
        for each world space.
    name : str or None
        Optional name of the multiscale.
    version : str
        OME-NGFF version to produce.
    """
    json_axes = [axis_to_json(axis) for axis in axes]
    level_output = INTRINSIC_SYSTEM if commons else WORLD_SYSTEM
    if commons:
        systems = [{"name": INTRINSIC_SYSTEM, "axes": json_axes}]
        systems += [{"name": world, "axes": json_axes} for world, _ in commons]
    else:
        systems = [{"name": WORLD_SYSTEM, "axes": json_axes}]
    block = {
        "coordinateSystems": systems,
        "datasets": [
            {
                "path": path,
                "coordinateTransformations": _level_transforms(
                    entry, path, level_output
                ),
            }
            for path, entry in levels
        ],
    }  # type: tx.Dict[str, tx.Any]
    if commons:
        # Every common transformation starts from the intrinsic space, so
        # several world spaces are siblings rather than links of a chain.
        block["coordinateTransformations"] = [
            dict(
                entry,
                input={"name": INTRINSIC_SYSTEM},
                output={"name": world},
            )
            for world, entry in commons
        ]
    if name is not None:
        block["name"] = name
    ome = _v06.OME.from_json(
        {"version": NORMALIZED_VERSION, "multiscales": [block]}
    )
    if version != NORMALIZED_VERSION:
        ome = ome.to_version(version)
    return ome


def write_multiscale(
    node: ZarrGroup,
    axes: tx.Sequence[Axis],
    levels: tx.Sequence[_Level],
    commons: tx.Sequence[tx.Tuple[str, _Entry]],
    name: tx.Optional[str],
    version: str,
) -> None:
    """Write the OME metadata of a pyramid onto a group.

    The arguments are those of [`build_ome`][].
    """
    node.ome = build_ome(axes, levels, commons, name, version)
