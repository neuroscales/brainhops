"""Read and build OME-Zarr multiscale metadata through abczarr.

The OME-Zarr metadata of a group is read with abczarr, which parses it
into a typed object and validates it. The typed object is normalized to
OME-NGFF 0.6, in which each resolution level carries a single
coordinate transformation. The reader can therefore assume one
transformation specification per level, whatever version the group was
written in.

Each OME coordinate transformation is mapped to the brainhops
transformation of the same kind, rather than being collapsed into an
affine. A scale becomes a `Scaling`, a translation a `Translation`, a
rotation a `Rotation`, and a sequence a `Sequence` of the mapped children,
so the geometry a level carries is the one the metadata describes.

Malformed or missing metadata is rejected by abczarr while it parses, so a
level with no transformation, or one placed by a transformation that is
misspelled or of the wrong shape, raises rather than being read as an
identity placement.
"""

# dependencies
import typing_extensions as tx
from abczarr.abc.sync import ZarrGroup
from abczarr.ome import v0_6 as _v06
from abczarr.ome.v0_6.images import Dataset, Multiscale
from abczarr.ome.v0_6.ome import OME
from abczarr.ome.v0_6.transformations import CoordinateTransformation
from bagof.magic import replace

# internals
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

#: The brainhops spline order each OME interpolation name maps to.
_INTERPOLATION_ORDER = {"nearest": 0, "linear": 1, "bspline-cubic": 3}

#: The placement of one level: its array path with the OME coordinate
#: transformation that places it, ready for the metadata.
_Entry = tx.Dict[str, tx.Any]
_Level = tx.Tuple[str, _Entry]

#: The OME-NGFF version the reader normalizes every group to.
NORMALIZED_VERSION = "0.6"

#: The OME-NGFF version the writer emits when nothing else selects one.
#: Version 0.4 is stored as Zarr v2, and 0.5 and later as Zarr v3, so 0.4 is
#: not a candidate: brainhops writes Zarr v3. Of the rest the newest released
#: version is written, because it is the only one that can carry every
#: placement brainhops can hold -- a rotation, an affine, or a pyramid placed
#: in more than one world space -- so the writer does not have to raise the
#: version to express what it was given.
#:
#: This names a version rather than following abczarr's newest, for the same
#: reason `NORMALIZED_VERSION` does: which version brainhops writes by
#: default decides what other tools can read its output, so it is a
#: compatibility choice to make deliberately rather than inherit.
DEFAULT_WRITE_VERSION = "0.6"


class OmeImageError(ParserContentError):
    """Raised when an OME-Zarr image group cannot be read.

    A group with no multiscale metadata, or one whose metadata abczarr
    cannot parse into a valid pyramid, is refused with this error.
    """


def looks_like_multiscale(node: ZarrGroup) -> bool:
    """Whether a group carries image multiscale metadata.

    The metadata is read through abczarr, which finds it wherever the
    version the group was written in puts it: nested under ``ome`` from 0.5
    on, and at the top level of the attributes in 0.4. A group that carries
    OME metadata of another kind, such as labels, names no multiscale and is
    left to the reader of that kind.

    !!! note
        Metadata that abczarr cannot parse does *not* count as a multiscale.
        abczarr is liberal -- it converts what it can rather than validating
        strictly -- so it fails only when the metadata genuinely contradicts
        the schema, which is evidence against reading the group as a pyramid
        rather than a reason to claim it anyway.

        This scores a group; it does not report on one. A group that is
        *asked* to be read as a pyramid still gets the specific fault, since
        [read_multiscale][brainhops.io.images.zarr._ome.read_multiscale] does
        the parsing and raises.
    """
    try:
        ome = node.ome
    except Exception:
        return False
    return bool(getattr(ome, "multiscales", None))


def read_multiscale(
    node: ZarrGroup,
) -> tx.Tuple[tx.Optional[Multiscale], tx.Optional[str]]:
    """Return a group's first multiscale, normalized to 0.6.

    The result is ``(multiscale, source_version)``, where `multiscale` is
    the abczarr 0.6 multiscale object and `source_version` is the
    OME-NGFF version the group was written in. The multiscale is `None` when
    the group carries no OME metadata at all, or carries OME metadata that
    names no multiscale.

    Raises
    ------
    OmeImageError
        If the group's metadata contradicts the OME schema, or cannot be
        converted to a form with one transformation per level. This is the
        specific fault, so it is worth reaching: a group that only *looks*
        like a pyramid is filtered out before here, by
        [looks_like_multiscale][brainhops.io.images.zarr._ome.looks_like_multiscale].
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
    # The coordinate system a dataset maps its array onto, read from the
    # first dataset's transformation. The first coordinate system is used
    # when no transformation names one.
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
    """Return the name of the coordinate system the levels map onto.

    This is the intrinsic space that every level shares: it is what a
    level's own coordinate transformation outputs to, and what the
    multiscale's common transformations carry to world.

    A multiscale with no common transformations places its levels directly
    in world space, so the intrinsic system *is* the world system. This
    mirrors [MultiScaleImage][brainhops.datamodel.images.MultiScaleImage],
    whose level transformations end in the intrinsic space and whose own
    transformations carry that space to each world space.
    """
    return getattr(_output_system(multiscale), "name", None)


def system_axes(multiscale: Multiscale) -> tx.Dict[str, tx.List[Axis]]:
    """Return the axes of every named coordinate system, in stored order.

    A multiscale names one coordinate system per space it places its levels
    in: the intrinsic space the levels map onto, and every world space the
    common transformations reach.
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
    # Build the callback that reads a displacement or coordinate field from
    # the node a field transformation names. The callback lays the field out
    # in the brainhops order, so it fits the affine transformations around
    # it. The node reading itself is done by the shared OME field reader in
    # `io.transformations.zarr`.
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
            # The field node's own typed axes place the component axis and
            # order the spatial axes. Sorting into the brainhops order lays
            # the field out as (*spatial, component).
            data = get_array_backend().transpose(
                raw, _axisorder.to_canonical(typed)
            )
        else:
            # A field node that is only a bare array carries no typed axes,
            # just dimension names. Match those names against the image axes
            # to find the component axis.
            data = _field_from_names(raw, field_node, store_axes, kind, path)
        order = _INTERPOLATION_ORDER.get(
            getattr(transform, "interpolation", None), 1
        )
        field_cls = (
            DisplacementField if kind == "displacements" else CoordinatesField
        )
        return field_cls(field=data, order=order)

    return read_field


def _field_from_names(
    raw: tx.Any,
    field_node: tx.Any,
    store_axes: tx.Optional[tx.Sequence[Axis]],
    kind: str,
    path: str,
) -> tx.Any:
    # Lay a bare field array out as (*spatial, component) from its axis names
    # alone. The array shares the image's spatial axes by name; the remaining
    # axis is the component axis.
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
    # Map one 0.6 coordinate transformation to the brainhops
    # transformation of the same kind, through the shared OME mapping. A
    # field is read from the node it names by the image-side callback.
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

    Each OME coordinate transformation is mapped to the brainhops
    transformation of the same kind. Only the level's *own* transformations
    are mapped here: the multiscale transformations that apply to every
    level carry the intrinsic space to world, so they belong to the pyramid
    rather than to one of its levels, and
    [common_transformation][brainhops.io.images.zarr._ome.common_transformation]
    reads them. A level with more than one transformation becomes a
    `Sequence` in application order.

    !!! note
        This is the contract
        [MultiScaleImage][brainhops.datamodel.images.MultiScaleImage]
        states: a level's transformations end in the intrinsic space that
        every level shares, and the pyramid's own transformations carry
        that space to world.
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

    These are the multiscale's own `coordinateTransformations`, which apply
    to every level alike. A multiscale may place its levels in more than one
    world space, so one transformation is returned per world space its
    transformations reach, each carrying the intrinsic space to that space.
    They are ordered as the metadata declares them, so the last one is the
    preferred placement -- which is what
    [MultiScaleImage][brainhops.datamodel.images.MultiScaleImage] takes the
    last entry of `transformations` to be.

    Each transformation declares the system it maps from and the one it maps
    to, so they form a graph rather than a list. The graph is walked from the
    intrinsic space: an edge that leaves a space already reached is composed
    onto the path that reached it, and becomes a `Sequence` in application
    order. This is what makes both conventions read correctly -- the several
    entries of a 0.4 or 0.5 pyramid, which chain within one world space, and
    the several spaces a 0.6 pyramid can name.

    The result is empty when the multiscale declares no common
    transformations, which is the common case: the levels are then placed
    directly in world space and the pyramid needs no transformation of its
    own.

    Raises
    ------
    OmeImageError
        If a transformation maps from a space that nothing reaches from the
        intrinsic space, so it cannot place the levels.
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

    # The parts to apply, in order, to carry the intrinsic space to each
    # space that is reached. `order` keeps the declaration order, so the
    # preferred placement stays last.
    paths = {}  # type: tx.Dict[tx.Optional[str], tx.List[Transformation]]
    order = []  # type: tx.List[tx.Optional[str]]
    pending = list(edges)
    while pending:
        progress = False
        for edge in list(pending):
            source, target, transform = edge
            if source in paths:
                # Already-reached space: this edge continues that path. A
                # 0.4 or 0.5 pyramid whose common transformations all name
                # one space is read this way, as a chain.
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


#: The OME-NGFF versions that carry only a per-axis scale and translation.
#: A rotation, an affine, or a sequence of them needs a later version.
_SCALE_ONLY_VERSIONS = frozenset({"0.1", "0.2", "0.3", "0.4", "0.5"})


def resolve_write_version(
    version: tx.Optional[str],
    source_version: tx.Optional[str],
    rich: bool = False,
) -> str:
    """Choose the OME-NGFF version to write.

    An explicit `version` is used as given. Otherwise the version of the
    OME-Zarr the image was read from is used, so a pyramid is written back
    in the version it came from. When neither is available, the leanest
    version that carries a per-axis scale and translation is used.

    `rich` states that a level is placed by a transformation that only a
    later version carries, such as a rotation, an affine, or a sequence of
    them. When the chosen version cannot carry such a transformation, an
    explicit request for that version is refused with a
    [`WriterError`][brainhops.io.base.parsers.WriterError], and an implicit
    choice is raised to the version that can.
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


#: The name of the world coordinate system the writer emits.
WORLD_SYSTEM = "physical"

#: The name of the intrinsic coordinate system the writer emits: the space
#: every level maps onto, which the pyramid's common transformations then
#: carry to world. It is emitted only when there is such a transformation;
#: otherwise the levels map straight onto `WORLD_SYSTEM`.
INTRINSIC_SYSTEM = "intrinsic"


def _level_transforms(
    entry: _Entry, path: str, output: str
) -> tx.List[_Entry]:
    # Attach the input and output references to a level's coordinate
    # transformation, so it names the array it places and the system it maps
    # that array onto.
    refs = {"input": {"path": path}, "output": {"name": output}}
    return [dict(entry, **refs)]


def resolve_world_names(
    names: tx.Sequence[tx.Optional[str]],
) -> tx.List[str]:
    """Name the world coordinate system of each common transformation.

    A transformation that already carries the name of its output space keeps
    it, so a pyramid that was read from a store is written back naming the
    same spaces. One that names none is given `WORLD_SYSTEM`, and further
    ones are numbered after it. A name that would collide with another world
    space, or with the intrinsic space, is numbered too, so every system the
    metadata declares is named exactly once.
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
    """Build the typed OME metadata for an image pyramid.

    `axes` are the axes in the stored order. `levels` gives, for each
    resolution level, its array path and the OME coordinate transformation
    that places it in the intrinsic space the levels share. `commons` gives,
    for each world space the pyramid is placed in, that space's name and the
    transformation carrying the intrinsic space to it; these apply to every
    level alike. It is empty when the levels are placed directly in world
    space. The metadata is built in 0.6 and then converted to `version`,
    so any supported version can be written from one code path. The returned
    object is assigned to a group's ``ome``.

    !!! note
        The intrinsic system is emitted only when `commons` is non-empty. A
        pyramid whose levels already land in world space is written with the
        single coordinate system it needs, so the leaner versions are not
        handed a graph they cannot express.
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
        # Every common transformation leaves the intrinsic space, so the
        # several world spaces a pyramid names are siblings rather than a
        # chain. A chain that was read as one `Sequence` is written back as
        # one sequence transformation, so the graph keeps its shape.
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
    """Write an image pyramid's OME metadata onto a group through abczarr."""
    node.ome = build_ome(axes, levels, commons, name, version)
