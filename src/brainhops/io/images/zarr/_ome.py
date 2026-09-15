"""Read and build OME-Zarr multiscale metadata through abczarr.

The OME-Zarr metadata of a group is read with abczarr, which parses it
into a typed object and validates it. The typed object is normalized to
OME-NGFF 0.6rc0, in which each resolution level carries a single
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

# stdlib
from collections.abc import Mapping

# dependencies
import typing_extensions as tx
from abczarr.abc.sync import ZarrGroup
from abczarr.ome import v0_6rc0 as _v6
from abczarr.ome.v0_6rc0.images import Dataset, Multiscale
from abczarr.ome.v0_6rc0.ome import OME
from abczarr.ome.v0_6rc0.transformations import CoordinateTransformation
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
NORMALIZED_VERSION = "0.6rc0"

#: The OME-NGFF version the writer emits when nothing else selects one. A
#: per-axis scale and translation is expressible in every version, and this
#: is the leanest that carries it.
DEFAULT_WRITE_VERSION = "0.4"


class OmeImageError(ParserContentError):
    """Raised when an OME-Zarr image group cannot be read.

    A group with no multiscale metadata, or one whose metadata abczarr
    cannot parse into a valid pyramid, is refused with this error.
    """


def looks_like_multiscale(node: ZarrGroup) -> bool:
    """Whether a group's attributes carry image multiscale metadata.

    This inspects the raw attributes only, so a group that names a
    multiscale is recognized even when the metadata is malformed. The
    reader then reports the specific fault instead of the group being
    passed over.
    """
    attrs = dict(node.attrs)
    if "multiscales" in attrs:
        return True
    inner = attrs.get("ome")
    return isinstance(inner, Mapping) and "multiscales" in inner


def read_multiscale(
    node: ZarrGroup,
) -> tx.Optional[tx.Tuple[Multiscale, tx.Optional[str]]]:
    """Return a group's first multiscale, normalized to 0.6rc0.

    The result is ``(multiscale, source_version)``, where `multiscale` is
    the abczarr 0.6rc0 multiscale object and `source_version` is the
    OME-NGFF version the group was written in. The result is `None` when
    the group carries no OME metadata at all.
    """
    try:
        ome = node.ome
    except Exception as error:
        raise OmeImageError(
            "This OME-Zarr group carries metadata that could not be read as "
            "a valid image pyramid. " + str(error)
        ) from error
    if ome is None:
        return None
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
        return None
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
    # Map one 0.6rc0 coordinate transformation to the brainhops
    # transformation of the same kind, through the shared OME mapping. A
    # field is read from the node it names by the image-side callback.
    read_field = _make_read_field(node, store_axes)
    try:
        return _map.from_ome(transform, perm, ndim, read_field)
    except _map.OmeMappingError as error:
        raise OmeImageError(str(error)) from error


def level_transformation(
    multiscale: Multiscale,
    dataset: Dataset,
    perm: tx.Sequence[int],
    ndim: int,
    node: tx.Optional[ZarrGroup] = None,
    store_axes: tx.Optional[tx.Sequence[Axis]] = None,
    input: tx.Optional[CoordinateSystem] = None,
    output: tx.Optional[CoordinateSystem] = None,
) -> Transformation:
    """Return the voxel-to-world transformation of one level.

    Each OME coordinate transformation is mapped to the brainhops
    transformation of the same kind. The level's own transformation runs
    first, then the multiscale transformations that apply to every level. A
    level with more than one transformation becomes a `Sequence` in that
    application order.
    """
    mapped = []  # type: tx.List[Transformation]
    transforms = list(dataset.coordinateTransformations)
    if transforms:
        mapped.append(
            _map_transform(transforms[0], perm, ndim, node, store_axes)
        )
    common = getattr(multiscale, "coordinateTransformations", None)
    if isinstance(common, list):
        mapped.extend(
            _map_transform(one, perm, ndim, node, store_axes) for one in common
        )

    if not mapped:
        return Identity(input=input, output=output)
    if len(mapped) == 1:
        return replace(mapped[0], input=input, output=output)
    return Sequence(mapped, input=input, output=output)


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


def _level_transforms(entry: _Entry, path: str) -> tx.List[_Entry]:
    # Attach the input and output references to a level's coordinate
    # transformation, so it names the array it places and the world system.
    refs = {"input": {"path": path}, "output": {"name": "physical"}}
    return [dict(entry, **refs)]


def build_ome(
    axes: tx.Sequence[Axis],
    levels: tx.Sequence[_Level],
    name: tx.Optional[str],
    version: str,
) -> OME:
    """Build the typed OME metadata for an image pyramid.

    `axes` are the axes in the stored order. `levels` gives, for each
    resolution level, its array path and the OME coordinate transformation
    that places it in world space. The metadata is built in 0.6rc0 and then
    converted to `version`, so any supported version can be written from one
    code path. The returned object is assigned to a group's ``ome``.
    """
    block = {
        "coordinateSystems": [
            {
                "name": "physical",
                "axes": [axis_to_json(axis) for axis in axes],
            }
        ],
        "datasets": [
            {
                "path": path,
                "coordinateTransformations": _level_transforms(entry, path),
            }
            for path, entry in levels
        ],
    }  # type: tx.Dict[str, tx.Any]
    if name is not None:
        block["name"] = name
    ome = _v6.OME.from_json(
        {"version": NORMALIZED_VERSION, "multiscales": [block]}
    )
    if version != NORMALIZED_VERSION:
        ome = ome.to_version(version)
    return ome


def write_multiscale(
    node: ZarrGroup,
    axes: tx.Sequence[Axis],
    levels: tx.Sequence[_Level],
    name: tx.Optional[str],
    version: str,
) -> None:
    """Write an image pyramid's OME metadata onto a group through abczarr."""
    node.ome = build_ome(axes, levels, name, version)
