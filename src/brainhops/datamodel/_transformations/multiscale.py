import numpy as np
import typing_extensions as tx
from bagof.magic import Forget, replace

from brainhops._core.affines import axis_scales
from brainhops._core.affines import inv as _affine_inv
from brainhops._core.backends import to_backend
from brainhops._core.properties import smartproperty
from brainhops._core.pyramids import (
    downsample,
    level_geometry,
    rescale_vectors,
)
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel._multiscale import (
    PyramidLayout,
    adopt_scales,
    build_pyramid,
    build_scales,
    check_scales,
    downsampled_matrix,
    geometry_matrix,
    keeps_scale,
    level_matches,
    level_resolution,
    pop_pyramid_options,
    pyramid_layout,
)
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.axes import (
    AxisList,
    CoordinateAxis,
    DisplacementAxis,
    SpaceAxis,
)
from brainhops.datamodel.multiscale import Multiscale
from brainhops.errors import ConversionError

from .base import Transformation
from .compute.convert import converter
from .compute.simplify import SimplifyLike
from .concrete import Affine, CoordinateField, DisplacementField
from .modes import ModeLike
from .sequence import ImmutableSequence, Sequence


class FieldScale(tx.NamedTuple):
    """The parts of one scale of a multiscale field."""

    kind: str
    """`"displacement"` or `"coordinate"`."""

    world2voxel: np.ndarray
    """The compact matrix that maps the input space onto the grid."""

    field: Transformation
    """The field on the grid, in voxels of the grid for displacements."""

    voxel2world: tx.Optional[np.ndarray]
    """The compact matrix that maps the grid onto the output space.

    It is `None` for a field of coordinates, which ends in the output
    space.
    """


def field_scale(scale: Transformation) -> FieldScale:
    """Return the parts of one scale of a multiscale field.

    A scale of coordinates is a sequence of a world-to-voxel affine and
    a field of coordinates. A scale of displacements is a sequence of a
    world-to-voxel affine, a field of displacements in voxels of its
    grid and a voxel-to-world affine. The sequence is flattened first,
    and the elements before and after its field are each composed into
    one affine, so that a chain built with `@`, or a lazy inverse of a
    chain, is read too.

    Parameters
    ----------
    scale : Transformation
        The scale.

    Returns
    -------
    FieldScale
        Its parts, with the matrices of its affines.

    Raises
    ------
    ValueError
        If the scale has neither form, or if an affine of the scale is
        not an affine of the axes of the grid.
    """
    # A chain composed with `@` nests its sequences, which are
    # flattened.
    if isinstance(scale, Sequence) and not isinstance(scale, Multiscale):
        scale = scale.flatten()
    elements = list(getattr(scale, "transformations", None) or [])
    found = [
        index
        for index, element in enumerate(elements)
        if isinstance(element, (CoordinateField, DisplacementField))
    ]
    kind = None
    if len(found) == 1:
        field = elements[found[0]]
        before, after = elements[: found[0]], elements[found[0] + 1 :]
        lazy = getattr(type(field), "_operator", None) is not None
        if lazy or getattr(field, "data", None) is None:
            kind = None
        elif isinstance(field, CoordinateField) and not after:
            kind = "coordinate"
        elif (
            isinstance(field, DisplacementField)
            and after
            and not getattr(field, "log", False)
        ):
            kind = "displacement"
    if kind is None:
        raise ValueError(
            "A scale of a multiscale field is a sequence of a world-to-voxel "
            "affine and a field of coordinates, or of a world-to-voxel "
            "affine, a field of displacements and a voxel-to-world affine, "
            "whose field holds its values."
        )
    count = int(field.data.ndim) - 1
    world2voxel = _composed(before, count)
    voxel2world = _composed(after, count) if kind == "displacement" else None
    if world2voxel is None or (kind == "displacement" and voxel2world is None):
        raise ValueError(
            "The affines of a scale of a multiscale field map the axes of "
            f"its grid, which has {count} axes, onto as many axes."
        )
    return FieldScale(kind, world2voxel, field, voxel2world)


def grid_vectors(parts: FieldScale) -> tx.Optional[str]:
    """Return how the vectors of a scale depend on its grid.

    The displacements of a field of displacements are in voxels of its
    grid, which it maps onto itself. The coordinates of a field of
    coordinates are in its output space, which is the grid of the field
    itself when the field states that its output space is its input
    space, and another space, such as the world or the grid of another
    image, otherwise. A downsampled scale rescales the vectors that are
    in its own grid, as
    [`rescale_vectors`][brainhops._core.pyramids.rescale_vectors]
    explains, and only averages the others.

    Parameters
    ----------
    parts : FieldScale
        The parts of the scale, as [`field_scale`][] returns them.

    Returns
    -------
    str or None
        `"displacements"` or `"coordinates"` when the vectors are in the
        grid of the field, and `None` when they are not.
    """
    if parts.kind == "displacement":
        return "displacements"
    source = getattr(parts.field, "input", None)
    target = getattr(parts.field, "output", None)
    if source is not None and target is not None and source == target:
        return "coordinates"
    return None


def _composed(
    elements: tx.Sequence[Transformation], count: int
) -> tx.Optional[np.ndarray]:
    # The matrix of the elements applied in turn, or `None` when one of
    # them is not an affine of `count` axes. No element is the identity.
    if not elements:
        return np.eye(count, count + 1)
    if len(elements) == 1:
        return geometry_matrix(elements[0], count)
    return geometry_matrix(Sequence(list(elements)), count)


class MultiscaleField(Multiscale[Sequence], ImmutableSequence):
    """Field of coordinates or displacements at several resolutions.

    Each scale is a [`Sequence`][] that maps the input space to the output
    space on the grid of that scale. A scale of coordinates is a sequence of
    two elements, a world-to-voxel affine and a coordinate field. A scale of
    displacements has three: a world-to-voxel affine, a displacement field in
    voxel units and a voxel-to-world affine. The container treats a scale as a
    plain sequence, so a single class handles both kinds of field.

    The field behaves as its finest scale: it composes as the finest scale
    would, and `compute()` reduces it to that scale. Another scale is used only
    when it is selected with [`to_singlescale`][], which is also how a scale is
    edited: the returned sequence is edited in place, since the container
    supports no item assignment, insertion or deletion.
    """

    # --- attributes ---------------------------------------------------

    scales: ... = ()
    """Scales, from finest to coarsest.

    Each scale is a sequence that maps the input space to the output space,
    sampled on the grid of that scale.
    """

    # `transformations` is served from the finest scale rather than stored, so
    # it is not a constructor field. Forgetting the storage field of
    # `Sequence` keeps it out of `__init__`, `fields()` and `replace()`.
    _transformations: Forget

    @property
    def transformations(self) -> tx.Tuple[Transformation, ...]:
        """Transformations of the finest scale.

        They cannot be assigned: a scale is edited through `to_singlescale()`.
        """
        return getattr(self._finest, "transformations", None) or ()

    @transformations.setter
    def transformations(self, value: None) -> None:
        if value is not None:
            raise TypeError(
                "The elements of a multiscale field are determined by its "
                "scales. Select a scale with to_singlescale() and edit that "
                "sequence."
            )

    # The endpoints are stored as given, without a fallback. The fallback of
    # `Sequence` reads through `transformations`, hence through the scales, and
    # a reader that builds its scales lazily from these same endpoints would
    # recurse.
    input = smartproperty("input")
    output = smartproperty("output")

    # --- methods ------------------------------------------------------

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> Transformation:
        """Compute the field as a plain transformation.

        The finest scale is computed and returned. The result is an ordinary
        transformation, not a pyramid, and computes as the finest scale alone.
        """
        return self._finest.compute(mode, simplify=simplify, factor=factor)

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        scales = self.scales or []
        return replace(
            self,
            scales=[
                scale.inverse(compute=compute, **kwargs) for scale in scales
            ],
            input=self.output,
            output=self.input,
        )

    def flatten(self, endpoints: bool = True) -> Sequence:
        """Return the finest scale, flattened, as a plain sequence.

        A multiscale field behaves as its finest scale, so a field spliced
        into a surrounding chain contributes the elements of that scale.
        """
        # A multiscale field spliced into a surrounding sequence contributes
        # the elements of its finest scale.
        return self._finest.flatten(endpoints)

    # --- helpers ------------------------------------------------------

    def _as_sequence(self) -> Sequence:
        # Return the finest scale as a plain sequence that carries the
        # endpoints of the container. The simplifier works on this sequence,
        # because the derived `transformations` of the container cannot be
        # rebuilt by `replace()`.
        return Sequence(
            transformations=self.transformations,
            input=self.input,
            output=self.output,
        )

    # --- scales built from the finest scale ---------------------------


def _affine(matrix: np.ndarray, input: tx.Any, output: tx.Any) -> Affine:
    # An affine of `matrix` between two coordinate systems.
    return Affine(matrix=np.asarray(matrix), input=input, output=output)


def _scale_of(t: Transformation) -> Sequence:
    # Return the scale that a field on its grid is, as a sequence of
    # affines around the field. A field alone maps its grid onto itself,
    # so its affines are the identity.
    if isinstance(t, (DisplacementField, CoordinateField)):
        count = int(t.data.ndim) - 1 if t.data is not None else 0
        identity = np.eye(count, count + 1)
        chain = [Affine(matrix=identity), t]
        if isinstance(t, DisplacementField):
            chain.append(Affine(matrix=identity))
        scale = Sequence(chain, input=t.input, output=t.output)
    else:
        scale = t.flatten() if isinstance(t, Sequence) else t
    try:
        parts = field_scale(scale)
    except ValueError as error:
        raise ConversionError(
            f"A {type(t).__name__} becomes a multiscale field only when it "
            f"is a field on its grid. {error}"
        ) from None
    field = parts.field
    chain = [_affine(parts.world2voxel, t.input, field.input), field]
    if parts.voxel2world is not None:
        chain.append(_affine(parts.voxel2world, field.output, t.output))
    return Sequence(chain, input=t.input, output=t.output)


@converter(MultiscaleField, MultiscaleField)
def _multiscale_to_multiscale(
    t: MultiscaleField, cls: tx.Type[MultiscaleField], **kwargs
) -> MultiscaleField:
    # The options of `PYRAMID_OPTIONS` build the scales, and the other
    # options change the fields of the copy, which keeps its scales
    # unless `nscales=` is given.
    from .compute.converters import smart_replace

    options = pop_pyramid_options(kwargs)
    return build_pyramid(smart_replace(t, cls, **kwargs), options)


def _field_to_multiscale(
    t: Transformation, cls: tx.Type[MultiscaleField], **kwargs
) -> MultiscaleField:
    # A field on its grid becomes the finest scale of a multiscale
    # field, whose coarser scales `nscales=` builds, and a class that
    # builds its scales by default builds them without it.
    options = pop_pyramid_options(kwargs)
    if kwargs:
        names = ", ".join(f"{name}=" for name in sorted(kwargs))
        raise ConversionError(
            f"A field becomes a multiscale field with the options of its "
            f"scales, such as nscales=, and not with {names}."
        )
    result = cls(scales=[_scale_of(t)], input=t.input, output=t.output)
    return build_pyramid(result, options, default=cls.BUILDS_SCALES)


for _source in (Sequence, DisplacementField, CoordinateField):
    converter(_source, MultiscaleField)(_field_to_multiscale)


@pyramid_layout.register(MultiscaleField)
def _(obj: "MultiscaleField") -> PyramidLayout:
    scales = list(obj.scales or [])
    if not scales:
        raise ValueError(
            "This multiscale field has no scale, so no coarser scale can "
            "be built from it."
        )
    parts = field_scale(scales[0])
    shape = tuple(int(n) for n in parts.field.data.shape)
    count = len(shape) - 1
    grid = get_axes(parts.field.input)
    if len(grid) != count or Ellipsis in list(grid):
        grid = [SpaceAxis() for _ in range(count)]
    vector = (
        DisplacementAxis()
        if parts.kind == "displacement"
        else CoordinateAxis()
    )
    sizes = np.asarray(axis_scales(_affine_inv(parts.world2voxel)))
    return PyramidLayout(
        axes=AxisList(list(grid) + [vector]),
        shape=shape,
        sizes=tuple(float(size) for size in sizes) + (1.0,),
        spatial=tuple(range(count)),
        fixed=(count,),
        field=True,
    )


@build_scales.register(MultiscaleField)
def _(
    obj: "MultiscaleField",
    factors: tx.Sequence[tx.Tuple[int, ...]],
    mode: str,
    align: str,
    kept: int,
) -> "MultiscaleField":
    finest = obj.scales[0]
    parts = field_scale(finest)
    values = to_backend(parts.field.field)
    count = len(values.shape) - 1
    grid2world = np.asarray(_affine_inv(parts.world2voxel))
    kind = grid_vectors(parts)
    scales = list(obj.scales[: kept + 1])
    previous = None
    if kept:
        # The vectors of a kept scale are given back in the grid of
        # the finest scale, in which the modes average them.
        _, before, shifts = level_geometry(
            align, values.shape[:count], factors[kept - 1][:count]
        )
        kept_values = to_backend(field_scale(scales[kept]).field.field)
        kept_values = rescale_vectors(
            kept_values, before, shifts, kind, inverse=True
        )
        previous = (factors[kept - 1], kept_values)
    arrays = downsample(
        values,
        factors[kept:],
        mode,
        edge="linear",
        align=align,
        previous=previous,
    )
    for level, array in zip(factors[kept:], arrays):
        _, spatial, shift = level_geometry(
            align, values.shape[:count], level[:count]
        )
        world2voxel = _affine_inv(
            downsampled_matrix(grid2world, spatial, shift)
        )
        array = rescale_vectors(array, spatial, shift, kind)
        field = type(parts.field)(
            field=array,
            degree=parts.field.degree,
            bound=parts.field.bound,
            input=parts.field.input,
            output=parts.field.output,
        )
        chain = [
            _affine(world2voxel, finest.input, parts.field.input),
            field,
        ]
        if parts.kind == "displacement":
            voxel2world = downsampled_matrix(parts.voxel2world, spatial, shift)
            chain.append(
                _affine(voxel2world, parts.field.output, finest.output)
            )
        scales.append(
            Sequence(chain, input=finest.input, output=finest.output)
        )
    return replace(obj, scales=scales)


@keeps_scale.register(MultiscaleField)
def _(
    obj: "MultiscaleField", index: int, factors: tx.Tuple[int, ...], align: str
) -> bool:
    try:
        finest = field_scale(obj.scales[0])
        level = field_scale(obj.scales[index])
    except ValueError:
        return False
    if level.kind != finest.kind:
        return False
    shapes = [part.field.data.shape[:-1] for part in (finest, level)]
    return level_matches(
        np.asarray(_affine_inv(finest.world2voxel)),
        np.asarray(_affine_inv(level.world2voxel)),
        shapes[0],
        shapes[1],
        factors[: len(shapes[0])],
        align,
    )


@check_scales.register(MultiscaleField)
def _(obj: "MultiscaleField", scales: tx.Sequence[Sequence]) -> None:
    parts = []  # type: tx.List[FieldScale]
    for scale in scales:
        if not isinstance(scale, Sequence):
            raise TypeError("The scales of a multiscale field are sequences.")
        parts.append(field_scale(scale))
    if len({part.kind for part in parts}) > 1:
        raise ValueError(
            "The scales of a multiscale field hold the same kind of "
            "field, but these mix displacements and coordinates."
        )
    for side in ("input", "output"):
        systems = [getattr(scale, side, None) for scale in scales]
        systems.append(getattr(obj, side, None))
        known = [system for system in systems if system is not None]
        if any(system != known[0] for system in known):
            names = [getattr(system, "name", None) for system in systems]
            raise ValueError(
                f"The scales of a multiscale field share their {side} "
                f"space, but these have the {side} spaces {names[:-1]}."
            )
    sizes = [
        np.asarray(axis_scales(_affine_inv(part.world2voxel)))
        for part in parts
    ]
    for index in range(1, len(sizes)):
        if bool((sizes[index] < sizes[index - 1] * (1 - 1e-9)).any()):
            raise ValueError(
                "The scales are ordered from fine to coarse, but the "
                f"voxels of scale {index} ({sizes[index].tolist()}) are "
                f"smaller than those of scale {index - 1} "
                f"({sizes[index - 1].tolist()}) along some axis."
            )


@adopt_scales.register(MultiscaleField)
def _(
    obj: "MultiscaleField", scales: tx.Sequence[Sequence]
) -> "MultiscaleField":
    return replace(obj, scales=list(scales))


@level_resolution.register(MultiscaleField)
def _(obj: "MultiscaleField", index: int) -> tx.Optional[ArrayProtocol]:
    # The resolution of a scale is read from its leading world-to-voxel
    # affine: the norm of each column of its inverse is the voxel size
    # along that axis. If the leading element is not a defined affine, the
    # resolution is unknown.
    scales = obj.scales or []
    if not 0 <= index < len(scales):
        return None
    scale = scales[index]
    if not len(scale):
        return None
    affine = scale[0].to_affine(on_error=None)
    if affine is None:
        return None
    return axis_scales(_affine_inv(affine.matrix))
