"""OME-Zarr multiscale metadata of images.

OME-Zarr stores an image pyramid as a Zarr group whose metadata lists
the array of each level and the transformations that place the levels in
space. The metadata is parsed by abczarr, which refuses metadata that
does not fit its models, such as an unknown type of transformation, but
does not check every rule of the specification. Whatever version the
store uses, the metadata is then normalized to OME-NGFF 0.6, which
describes each level by a single transformation. Each OME transformation
becomes a brainhops transformation of the same kind rather than being
flattened into an affine. The conventions are described in
[`brainhops.io.images.zarr`][].
"""

# dependencies
import typing_extensions as tx
from abczarr.abc.sync import ZarrGroup
from abczarr.ome import v0_6 as _v06
from abczarr.ome.v0_6.images import Dataset, Multiscale
from abczarr.ome.v0_6.ome import OME
from abczarr.ome.v0_6.transformations import CoordinateTransformation
from bagof.magic import replace

# core
from brainhops._core.backends import array_backend

# datamodel
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    CoordinateField,
    DisplacementField,
    Identity,
    Projection,
    Sequence,
    SubspaceTransformation,
    Transformation,
)
from brainhops.datamodel.units import Unit

# io
from brainhops.errors import WriterError
from brainhops.io.base.parsers import ParserContentError
from brainhops.io.common.zarr._internal_api import check_ome
from brainhops.io.images.zarr import _axisorder
from brainhops.io.transformations.zarr import _map, _node
from brainhops.io.transformations.zarr._internal_api import to_axis

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


# The drafts 0.6.dev1 and 0.6.dev2 let a scale or a translation store its
# values in an array named by `path`. Versions 0.4 and 0.5, and every draft
# from 0.6.dev3 on, require the values inline in their schemas, as version
# 0.6 does (ngff-spec tag `0.6` at 89da1da,
# `schemas/coordinate_transformations.schema`, and its invalid conformance
# cases `bad_scale_path_not_allowed.json` and
# `bad_translate_path_not_allowed.json`).
_PARAMETER_PATH_DRAFTS = ("0.6.dev1", "0.6.dev2")


def check_parameter_paths(
    multiscale: Multiscale, version: tx.Optional[str]
) -> None:
    """Refuse a scale or a translation whose values are named by `path`.

    OME-NGFF 0.6 only lets an `affine` or a `rotation` store its parameters
    in an array named by `path`. A `scale` or a `translation` must give its
    values inline, whereas the drafts 0.6.dev1 and 0.6.dev2 allowed a
    `path` there too, and such a draft is still read.

    Parameters
    ----------
    multiscale : Multiscale
        The multiscale, normalized to version 0.6.
    version : str or None
        The OME version of the group, before normalization.

    Raises
    ------
    OmeImageError
        If a transformation of the multiscale or of one of its levels,
        including one nested in another transformation, is a `scale` or a
        `translation` whose values are named by `path`, and the version is
        not one of those drafts.
    """
    if version in _PARAMETER_PATH_DRAFTS:
        return
    transforms = list(
        getattr(multiscale, "coordinateTransformations", None) or []
    )
    for dataset in multiscale.datasets:
        transforms.extend(dataset.coordinateTransformations or [])
    for transform in transforms:
        found = _parameter_path(transform.to_json())
        if found is not None:
            kind, path = found
            raise OmeImageError(
                f"This OME-Zarr image has a {kind} transformation whose "
                f"values are stored in the array {path!r}. OME-NGFF "
                f"{version or NORMALIZED_VERSION} requires the values of a "
                f"{kind} inline, and only an affine or a rotation may name "
                "an array of parameters."
            )


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
    return [to_axis(axis.to_json()) for axis in system.axes]


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
            axes[name] = [to_axis(axis.to_json()) for axis in system.axes]
    return axes


# ---- names of coordinate systems -------------------------------------
# OME-NGFF 0.6 names a coordinate system by any non-empty string, unique
# among the systems of one `coordinateSystems` array (ngff-spec tag `0.6`
# at 89da1da, `index.md`, section "coordinateSystems metadata", and
# `schemas/coordinate_systems.schema`, where `name` is a string of
# `minLength` 1 without a pattern). A transformation refers to a system of
# another group by the `path` of that group, relative to its own group,
# and to a system of its own group by the name alone (same file, section
# "coordinateTransformations metadata", "Constraints"). Since no character
# is reserved in a name, the separator is chosen against the group path
# instead. A Zarr node name is never empty and never holds a `/` (Zarr v3
# core specification, section "Node names"), so `//` never occurs in a
# normalized group path, and it separates the path from the name.
GROUP_SEPARATOR = "//"


def system_name(name: str, group: str = "") -> str:
    """Return the brainhops name of an OME coordinate system.

    A system of the root group keeps its OME name, so that the common case
    is unchanged. A system of another group is named by the path of that
    group from the root, followed by `//` and its OME name, as in
    `"labels/cells//physical"`. A name of the root group that contains
    `//` is preceded by `//`, so that no two systems get the same name.
    The reader that opened the root of the hierarchy knows the path of
    each group, and passes it to the reader of that group.

    Parameters
    ----------
    name : str
        The OME name of the system.
    group : str, optional
        The normalized path of its group from the root of the hierarchy,
        as [`group_path`][] returns it. The empty string, the default, is
        the root group.

    Returns
    -------
    str
        The brainhops name of the system.
    """
    if group:
        return group + GROUP_SEPARATOR + name
    if GROUP_SEPARATOR in name:
        return GROUP_SEPARATOR + name
    return name


def split_system_name(name: str) -> tx.Tuple[str, str]:
    """Split a brainhops system name into its group path and OME name.

    This undoes [`system_name`][]: the group path is everything before the
    first `//`, which a group path never contains, and the empty string
    stands for the root group.

    Parameters
    ----------
    name : str
        The brainhops name of the system.

    Returns
    -------
    tuple of str
        The group path and the OME name.
    """
    group, separator, local = name.partition(GROUP_SEPARATOR)
    if not separator:
        return "", name
    return group, local


def group_path(group: str, path: tx.Optional[str] = None) -> str:
    """Return the normalized path from the root of a group that is referred to.

    The `path` of a reference is relative to the group that holds the
    metadata, as the labels example of ngff-spec 0.6 shows (section
    "multiscales metadata", "Example: Additional coordinate
    transformation"). A missing or empty path refers to that group itself.

    Parameters
    ----------
    group : str
        The path of the group that holds the metadata, from the root of the
        hierarchy. The empty string is the root group.
    path : str, optional
        The path of the reference.

    Returns
    -------
    str
        The path of the referred group from the root, without leading or
        trailing slashes and without `.` segments, or the empty string for
        the root group.

    Raises
    ------
    ValueError
        If either path is absolute, has an empty segment, or leaves the root
        of the hierarchy.
    """
    segments = []  # type: tx.List[str]
    for part in (group, path or ""):
        if part.startswith("/"):
            raise ValueError(
                f"The OME-Zarr group path {part!r} is absolute, while it must "
                "be relative to the group that holds the metadata."
            )
        part = part.rstrip("/")
        for segment in part.split("/") if part else []:
            if not segment:
                raise ValueError(
                    f"The OME-Zarr group path {part!r} has an empty segment."
                )
            if segment == "..":
                if not segments:
                    raise ValueError(
                        f"The OME-Zarr group path {part!r} leaves the root "
                        "of the hierarchy."
                    )
                segments.pop()
            elif segment != ".":
                segments.append(segment)
    return "/".join(segments)


def _make_read_field(
    node: tx.Optional[ZarrGroup], store_axes: tx.Optional[tx.Sequence[Axis]]
) -> tx.Optional[tx.Callable]:
    # Build the callback with which `_map.from_ome` reads a displacement or
    # coordinate field. The callback reads the node that the transformation
    # names from `node`, and returns a field that maps the brainhops order of
    # the input system to the brainhops order of the output system.
    if node is None:
        return None

    def read_field(
        transform: tx.Any,
        kind: str,
        source: tx.Sequence[int],
        target: tx.Sequence[int],
    ) -> Transformation:
        path = getattr(transform, "path", None)
        if not isinstance(path, str):
            raise OmeImageError(
                f"This OME-Zarr {kind} field names no array, so its field "
                "cannot be read."
            )
        try:
            field_node = _node.follow_path(node, path)
        except Exception as error:
            raise OmeImageError(
                f"The {kind} field at {path!r} could not be opened. {error}"
            ) from error
        degree = _interpolation_degree(transform, kind, path)
        if isinstance(field_node, ZarrGroup):
            return _field_group(field_node, kind, path, source, target, degree)
        return _field_array(field_node, store_axes, kind, path, target, degree)

    return read_field


def _field_group(
    field_node: ZarrGroup,
    kind: str,
    path: str,
    source: tx.Sequence[int],
    target: tx.Sequence[int],
    degree: int,
) -> Transformation:
    # In OME-NGFF 0.6, `path` names a multiscale group whose own
    # transformation places the vectors in the input system of the
    # transformation, and whose vector axis lists the components in the
    # stored order of the output system. The group is read as an
    # `OmeZarrField`, which works in the stored order, and the field is
    # surrounded by the permutations to and from that order.
    from brainhops.io.transformations.zarr import OmeZarrField

    try:
        field = OmeZarrField.from_node(field_node, degree=degree)
    except Exception as error:
        raise OmeImageError(
            f"The {kind} field at {path!r} could not be read as an "
            f"OME-Zarr field. {error}"
        ) from error
    types = [axis.type for axis in field.axes or []]
    if kind[:-1] not in types:
        raise OmeImageError(
            f"The field at {path!r} has no {kind[:-1]} axis, so it cannot "
            f"be read as {kind}."
        )
    count = len(types) - 1
    if count != len(source):
        raise OmeImageError(
            f"The {kind} field at {path!r} has {count} spatial axes, but "
            f"its input coordinate system has {len(source)} axes."
        )
    # The vector axis holds one component per output axis (ngff-spec 0.6,
    # "coordinates and displacements", "Vector dimension length").
    components = field.raw_levels[0].shape[types.index(kind[:-1])]
    if components != len(target):
        raise OmeImageError(
            f"The {kind} field at {path!r} has {components} components, "
            f"but its output coordinate system has {len(target)} axes."
        )
    return _map.in_stored_order(field, source, target)


def _field_array(
    field_node: tx.Any,
    store_axes: tx.Optional[tx.Sequence[Axis]],
    kind: str,
    path: str,
    target: tx.Sequence[int],
    degree: int,
) -> Transformation:
    # Drafts of OME-NGFF 0.6 let `path` name the array of the field itself,
    # with its axes named by its own typed metadata or by its dimension
    # names. Its spatial axes are sorted into the brainhops order, and the
    # component axis comes last.
    raw = _node.read_array(field_node)
    typed = _node.typed_axes(field_node, field_node.ndim)
    if typed is not None:
        data = array_backend(raw).transpose(
            raw, _axisorder.to_canonical(typed)
        )
    else:
        data = _field_from_names(raw, field_node, store_axes, kind, path)
    # Component `i` belongs to the stored output axis `i` (ngff-spec 0.6,
    # "coordinates and displacements", "Vector component mapping"), so the
    # components are reordered like the axes of the output system.
    if data.shape[-1] != len(target):
        raise OmeImageError(
            f"The {kind} field at {path!r} has {data.shape[-1]} components, "
            f"but its output coordinate system has {len(target)} axes."
        )
    if list(target) != list(range(len(target))):
        data = data[..., list(target)]
    field_cls = (
        DisplacementField if kind == "displacements" else CoordinateField
    )
    return field_cls(field=data, degree=degree)


def _interpolation_degree(transform: tx.Any, kind: str, path: str) -> int:
    # The interpolation of the vectors defaults to `linear` (ngff-spec 0.6,
    # "coordinates and displacements"). abczarr marks an absent field with
    # a sentinel that is false.
    name = getattr(transform, "interpolation", None)
    if not name:
        return _INTERPOLATION_DEGREE["linear"]
    if name not in _INTERPOLATION_DEGREE:
        known = ", ".join(repr(key) for key in _INTERPOLATION_DEGREE)
        raise OmeImageError(
            f"The {kind} field at {path!r} is interpolated with {name!r}, "
            f"which brainhops does not know. The known methods are {known}."
        )
    return _INTERPOLATION_DEGREE[name]


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
    return array_backend(raw).transpose(raw, target)


def _map_transform(
    transform: CoordinateTransformation,
    source: tx.Sequence[int],
    target: tx.Sequence[int],
    node: tx.Optional[ZarrGroup] = None,
    store_axes: tx.Optional[tx.Sequence[Axis]] = None,
    orders: tx.Optional[tx.Mapping[str, tx.Sequence[int]]] = None,
) -> Transformation:
    reading = _map.Reading(
        node=node,
        read_field=_make_read_field(node, store_axes),
        orders=dict(orders or {}),
    )
    try:
        return _map.from_ome(transform, source, target, reading)
    except _map.OmeMappingError as error:
        raise OmeImageError(str(error)) from error


def level_transformation(
    dataset: Dataset,
    order: tx.Sequence[int],
    node: tx.Optional[ZarrGroup] = None,
    store_axes: tx.Optional[tx.Sequence[Axis]] = None,
    input: tx.Optional[CoordinateSystem] = None,
    output: tx.Optional[CoordinateSystem] = None,
    orders: tx.Optional[tx.Mapping[str, tx.Sequence[int]]] = None,
) -> Transformation:
    """Return the voxel-to-intrinsic transformation of one level.

    A dataset is the metadata entry of one level. Only the transformations
    declared on the dataset are mapped here. The transformations declared on
    the whole multiscale belong to the pyramid and are read by
    [`common_transformations`][]. Several transformations form a
    [`Sequence`][] in application order, and a dataset without
    transformations yields an [`Identity`][].

    OME-NGFF 0.6 only allows a `scale`, an `identity`, or a `sequence` of a
    `scale` and a `translation` here, but the drafts that preceded it, and
    earlier versions of brainhops, wrote other types. Every type that the
    mapping knows is therefore read.

    Parameters
    ----------
    dataset : Dataset
        OME dataset of the level.
    order : sequence of int
        Order of the array axes, which is also the order of the intrinsic
        system: element `k` is the stored position of brainhops axis `k`.
    node : ZarrGroup, optional
        Group in which arrays named by the transformations are looked up.
    store_axes : sequence of Axis, optional
        Image axes in stored order, used to lay out fields that have no OME
        metadata.
    input : CoordinateSystem, optional
        Input coordinate system attached to the result.
    output : CoordinateSystem, optional
        Output coordinate system attached to the result.
    orders : mapping of str to sequence of int, optional
        Order of each named coordinate system, for the transformations
        inside a sequence that name their output system.

    Raises
    ------
    OmeImageError
        If a transformation cannot be mapped.
    """
    transforms = list(dataset.coordinateTransformations)
    if not transforms:
        return Identity(input=input, output=output)
    mapped = [
        _map_transform(one, order, order, node, store_axes, orders)
        for one in transforms
    ]
    if len(mapped) == 1:
        return replace(mapped[0], input=input, output=output)
    return Sequence(mapped, input=input, output=output)


def common_transformations(
    multiscale: Multiscale,
    orders: tx.Mapping[str, tx.Sequence[int]],
    root_order: tx.Sequence[int],
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

    OME-NGFF 0.6 lets a common transformation either start from the
    intrinsic space or end there: "One of `input` or `output` MUST
    reference the intrinsic coordinate system" (ngff-spec tag `0.6` at
    89da1da, `index.md`, section "multiscales metadata"). Its conformance
    case `tests/attributes/strict/valid/image/multiscales_example.json`
    declares a `scale` from `world` to `intrinsic`, and the example
    `examples/multiscales_strict/multiscales_reference_to_label.json` an
    `identity` from a labels group to the intrinsic space. A transformation
    that ends in the intrinsic space is read as the inverse of the declared
    transformation, so that every placement of the
    pyramid starts from the intrinsic space. The inverse is only computed
    in closed form for the types that have one, such as an affine. A
    `projectAxis` that drops axes has no inverse, so such a transformation
    is refused, while one that only creates axes is read as the projection
    that drops them again, which undoes it on the points that it reaches.
    A transformation that refers to a coordinate system of another
    group, such as a labels image, by a `path` does not place this pyramid
    and is skipped, while a `path` that leads back to the group of the
    multiscale refers to its own systems.

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
    orders : mapping of str to sequence of int
        Order of each named coordinate system: element `k` is the stored
        position of brainhops axis `k`.
    root_order : sequence of int
        Order of the intrinsic system, which is also used for a system that
        a transformation does not name.
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

    def order(name: tx.Optional[str]) -> tx.Sequence[int]:
        if isinstance(name, str) and name in orders:
            return orders[name]
        return root_order

    edges = []  # type: tx.List[tx.Tuple[tx.Any, tx.Any, Transformation]]
    for one in common:
        source = getattr(one, "input", None)
        target = getattr(one, "output", None)
        if _other_group(source) or _other_group(target):
            continue
        source = getattr(source, "name", None)
        target = getattr(target, "name", None)
        mapped = _map_transform(
            one, order(source), order(target), node, store_axes, orders
        )
        if target == root and source not in (None, root):
            # The transformation ends in the intrinsic space, so its inverse
            # starts there. A projection that drops axes has no inverse
            # (spec/design/compute-api-and-open-systems.md, section A.8).
            if _drops_axes(mapped):
                raise OmeImageError(
                    f"This OME multiscale reaches its intrinsic space from "
                    f"{source!r} by dropping axes, which cannot be inverted "
                    "to place the pyramid."
                )
            edges.append((target, source, mapped.inverse()))
        else:
            edges.append((source, target, mapped))

    # `paths` maps each reached space to the transformations that lead to it
    # from the intrinsic space. `order` lists the reached spaces by their last
    # declaration, so that the preferred space stays last.
    paths = {}  # type: tx.Dict[tx.Optional[str], tx.List[Transformation]]
    reached = []  # type: tx.List[tx.Optional[str]]
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
            if target in reached:
                reached.remove(target)
            reached.append(target)
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
    for target in reached:
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
    """Return the OME-Zarr JSON description of one axis.

    The description holds the name of the axis, and its type, its unit
    and whether it is discrete whenever the axis knows them. An unknown
    unit is left out rather than replaced by a default.
    """
    entry = {}  # type: tx.Dict[str, tx.Any]
    name = getattr(axis, "name", None)
    type_ = getattr(axis, "type", None)
    unit = getattr(axis, "unit", None)
    entry["name"] = name if isinstance(name, str) else ""
    if isinstance(type_, str):
        entry["type"] = str(getattr(type_, "value", type_))
    # A unit of the data model is written by its full name, such as
    # `millimeter`, which is how OME-NGFF spells its units.
    value = unit.name if isinstance(unit, Unit) else unit
    if isinstance(value, str):
        entry["unit"] = value
    discrete = getattr(axis, "discrete", None)
    if isinstance(discrete, bool):
        entry["discrete"] = discrete
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
                f"{version} cannot carry, such as a rotation, an affine or "
                "a flip, since it only carries positive scales and "
                "translations. "
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


def describe_axes(axes: tx.Sequence[Axis]) -> tx.List[Axis]:
    """Return the axes with every undescribed axis named and typed after
    the OME convention.

    OME-NGFF needs a distinct name for every axis. An undescribed axis,
    such as an axis that closes `...`, has neither a name nor a type. The
    writer of a pyramid calls this function before it sorts the axes
    into the stored order, because that sort groups the axes by type,
    so that each undescribed axis sorts into its conventional place.

    The conventional names are those of `default_canonical_axes`, in
    brainhops order: `x`, `y` and `z` in space, then `t` in time and `c`
    for the channels. The names that the other axes already use are
    removed, and the remaining names are given, with their types, to
    the undescribed axes in their brainhops order. When the five names
    are exhausted, an undescribed axis is named `dim<i>` after its
    position, and its type is left unset. An axis that has a name keeps
    it. An axis that has a type but no name keeps its type and takes the
    first free conventional name of that type, or `dim<i>` when there is
    none.

    Parameters
    ----------
    axes : sequence of Axis
        Axes in brainhops order.

    Returns
    -------
    list of Axis
        The same axes, in the same order, with every axis named.
    """
    # The pyramid module imports this one, so it is imported when used.
    from brainhops.io.images.zarr._multiscale import default_canonical_axes

    def name_of(axis: Axis) -> str:
        name = getattr(axis, "name", None)
        return name if isinstance(name, str) else ""

    taken = {name_of(axis) for axis in axes} - {""}
    free = [
        axis for axis in default_canonical_axes(5) if axis.name not in taken
    ]
    described = []
    for index, axis in enumerate(axes):
        if name_of(axis):
            described.append(axis)
            continue
        type_ = getattr(axis, "type", None)
        matches = [c for c in free if type_ is None or c.type == type_]
        if matches:
            convention = matches[0]
            free.remove(convention)
            name = convention.name
            described.append(replace(axis, name=name, type=convention.type))
        else:
            name = "dim" + str(index)
            while name in taken:
                name = "_" + name
            described.append(replace(axis, name=name))
        taken.add(name)
    return described


def write_multiscale(
    node: ZarrGroup,
    axes: tx.Sequence[Axis],
    levels: tx.Sequence[_Level],
    commons: tx.Sequence[tx.Tuple[str, _Entry]],
    name: tx.Optional[str],
    version: str,
) -> None:
    """Write the OME metadata of a pyramid onto a group.

    The arguments are those of [`build_ome`][]. The metadata is checked
    against the rules of OME-NGFF before it is written.

    Raises
    ------
    WriterError
        If the metadata breaks a rule of OME-NGFF.
    """
    ome = build_ome(axes, levels, commons, name, version)
    check_ome(ome)
    node.ome = ome


def _other_group(space: tx.Any) -> bool:
    # A space whose path leads away from its own group lies in another group
    # of the hierarchy. A path that cannot be resolved is treated the same
    # way, so that its transformation is skipped.
    path = getattr(space, "path", None)
    if not isinstance(path, str):
        return False
    try:
        return group_path("", path) != ""
    except ValueError:
        return True


def _drops_axes(xform: Transformation) -> bool:
    # Return whether a mapped transformation contains a projection that
    # drops axes, looking into sequences and subspace transformations.
    if isinstance(xform, Projection):
        return len(xform.dropped) > 0
    if isinstance(xform, SubspaceTransformation):
        return _drops_axes(xform.transformation)
    if isinstance(xform, Sequence):
        return any(_drops_axes(part) for part in xform.transformations)
    return False


def _parameter_path(entry: tx.Any) -> tx.Optional[tx.Tuple[str, str]]:
    # Return the type and the path of the first scale or translation of the
    # JSON of a transformation that names its values by `path`, looking into
    # every nested transformation.
    if isinstance(entry, list):
        for item in entry:
            found = _parameter_path(item)
            if found is not None:
                return found
        return None
    if not isinstance(entry, dict):
        return None
    kind = entry.get("type")
    path = entry.get("path")
    if kind in ("scale", "translation") and isinstance(path, str):
        return kind, path
    for value in entry.values():
        found = _parameter_path(value)
        if found is not None:
            return found
    return None
