# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.affines import axis_scales
from brainhops._core.affines import inv as _affine_inv
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel.base import DataModelBase

# internals
from .base import Transformation
from .concrete import Affine, CoordinatesField, DisplacementField, Identity
from .errors import ConversionError
from .inverse import Inverse
from .modes import SimplifyLike
from .sequence import ImmutableSequence, ModeLike, Sequence

# typing
SINGLE_SCALE = tx.TypeVar("SINGLE_SCALE", covariant=True)


class Multiscale(DataModelBase, tx.Generic[SINGLE_SCALE]):
    """A pyramid of resolution scales, ordered from finest to coarsest.

    This mixin gives a transformation a list of scales and the operations
    that select one of them. The scales are ordered from finest to
    coarsest, so the first scale is the highest resolution one.

    The mixin does not say what a scale is. A subclass supplies the scales
    and, through the `_level_resolution` hook, the physical grid size of
    each one. With those, `_nearest_level` picks the scale whose
    resolution is closest to a target grid.
    """

    # --- attributes ---------------------------------------------------

    scales: tx.Annotated[
        tx.List[tx.Any],
        tx.Doc("The resolution scales, ordered from finest to coarsest."),
    ] = ()

    @property
    def nscales(self) -> int:
        """The number of resolution scales."""
        return len(self.scales)

    @property
    def _finest(self) -> tx.Any:
        # The finest resolution scale, or `None` when there are no scales.
        scales = self.scales
        return scales[0] if scales else None

    # --- methods ------------------------------------------------------

    def to_singlescale(self, index: int = 0) -> tx.Any:
        """Return the resolution scale at a given index.

        Index `0` is the finest scale. The scale is returned as it is
        stored, so for a field it is the plain transformation of that
        scale rather than the multiscale field.
        """
        return self.scales[int(index)]

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # The physical grid size of a level, as a per-axis vector in the
        # input units of the multiscale. `None` means the resolution of
        # the level is unknown. A subclass overrides this hook.
        return None

    def _nearest_level(self, voxel2world: Transformation) -> int:
        # The index of the level whose resolution is closest to the grid
        # described by `voxel2world`. See `nearest_resolution`, which a
        # multiscale image shares, for how the comparison is made.
        scales = self.scales
        resolutions = [self._level_resolution(i) for i in range(len(scales))]
        return _nearest_resolution_index(
            resolutions, _as_affine_ignoring_fields(voxel2world)
        )


class MultiscaleField(Multiscale[Sequence], ImmutableSequence):
    """A field of coordinates or displacements at several resolutions.

    Each scale is a [`Sequence`][] that maps the multiscale's input space
    to its output space, sampled on that scale's grid. A coordinate scale
    is a two-element sequence of a world-to-voxel affine and a field of
    coordinates. A displacement scale is a three-element sequence of a
    world-to-voxel affine, a field of displacements in voxel units, and
    the voxel-to-world affine. The container treats a scale as a plain
    sequence, so the same class carries both kinds.

    A `MultiscaleField` behaves as its finest scale. It composes with
    other transformations exactly as the finest scale would, and reduces
    to the finest scale when it is computed. The finest scale is the one
    used unless a scale is selected with `to_singlescale`.

    The scales replace this class as the unit that is edited. A scale is
    selected with `to_singlescale`, and the returned sequence is edited in
    place. The container itself does not support item assignment,
    insertion, or deletion.
    """

    # --- attributes ---------------------------------------------------

    scales: tx.Annotated[
        tx.List[Sequence],
        tx.Doc(
            "The resolution scales, ordered from finest to coarsest. Each "
            "scale is a sequence that maps the input space to the output "
            "space, sampled on that scale's grid."
        ),
    ] = ()

    # `transformations` is served on demand from the finest scale rather
    # than stored, so it is not a constructor-taken field here. Declaring
    # it a `ClassVar` overrides the inherited init-field from `Sequence`
    # and keeps it out of `__init__`, `fields()` and `replace()`, while
    # the property keeps the container reading as the finest scale.
    transformations: tx.ClassVar[tx.Tuple[Transformation, ...]]

    @property
    def transformations(self) -> tx.Tuple[Transformation, ...]:
        """The transformations of the finest scale."""
        return getattr(self._finest, "transformations", None) or ()

    @transformations.setter
    def transformations(self, value: None) -> None:
        if value is not None:
            raise TypeError(
                "The elements of a multiscale field are determined by its "
                "scales. Select a scale with to_singlescale() and edit that "
                "sequence."
            )

    # A multiscale field declares the space its scales are built into, so
    # its endpoints are the ones it was given rather than the ones read
    # back off its finest scale. The inherited `Sequence` fallback would
    # read them through `transformations`, hence `_finest`, hence `scales`
    # -- and a reader that builds its scales lazily from those very
    # endpoints would recurse. Stored endpoints, no fallback.
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

        The finest scale is composed and returned. The result is an
        ordinary transformation, with no pyramid, so it computes exactly
        as the finest scale would on its own.
        """
        return self._finest.compute(mode, simplify=simplify, factor=factor)

    def inverse(self, compute: bool = False) -> tx.Self:
        scales = self.scales or []
        return replace(
            self,
            scales=[scale.inverse(compute) for scale in scales],
            input=self.output,
            output=self.input,
        )

    # --- helpers ------------------------------------------------------

    def _as_sequence(self) -> Sequence:
        # The finest scale as a plain sequence, carrying the container's
        # input and output. `compute` and `_flattened` go through this, so
        # they operate on a real init-field sequence rather than on the
        # container, whose `transformations` is derived and cannot be
        # rebuilt by `replace`.
        return Sequence(
            transformations=self.transformations,
            input=self.input,
            output=self.output,
        )

    def _flattened(self) -> Sequence:
        # Flatten through the finest scale. A multiscale field spliced
        # into a surrounding sequence contributes the elements of its
        # finest scale.
        return self._finest._flattened()

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # The physical grid size of a scale, read from the scale's leading
        # world-to-voxel affine. The inverse of that affine is the scale's
        # voxel-to-world transformation, and the norm of each of its
        # columns is the voxel size along one axis. A scale whose leading
        # element is not a defined affine has an unknown resolution.
        scales = self.scales or []
        if not 0 <= index < len(scales):
            return None
        scale = scales[index]
        if not len(scale):
            return None
        affine = _as_affine(scale[0])
        if affine is None:
            return None
        return axis_scales(_affine_inv(affine.matrix))


def _as_affine(xform: Transformation) -> tx.Optional[Affine]:
    # The transformation as an [`Affine`][], or `None`. A transformation
    # that does not reduce to an affine with a defined matrix has no affine
    # form, and is reported as `None` rather than refused. Callers that
    # must know whether a transformation really is an affine -- a writer
    # that can only emit one, say -- use this rather than
    # `_as_affine_ignoring_fields`, which answers a weaker question.
    try:
        affine = xform.compute().to(Affine)
    except ConversionError:
        return None
    if not isinstance(affine, Affine) or affine.matrix is None:
        return None
    return affine


def _as_affine_ignoring_fields(
    xform: Transformation,
) -> tx.Optional[Affine]:
    # The affine part of a transformation, with every field it carries
    # treated as the identity, or `None` when what is left still does not
    # reduce to an affine.
    #
    # This measures a chain that a field would otherwise make unmeasurable:
    # a warp has no affine form, but it is close enough to an isometry that
    # the *scale* of the chain around it is the scale of the whole. Only
    # use it for a question about scale. A caller asking whether a
    # transformation really is an affine wants `_as_affine`.
    return _as_affine(_fields_as_identity(xform))


def _fields_as_identity(xform: Transformation) -> Transformation:
    # Every field inside a transformation replaced by the identity, so that
    # what is left is the affine part of the chain. A field is recognized by
    # its family, which a typed inverse belongs to as well, so the inverse
    # of a field is replaced too.
    if isinstance(xform, (CoordinatesField, DisplacementField)):
        return Identity(input=xform.input, output=xform.output)
    if isinstance(xform, Inverse):
        forward = xform.forward
        if forward is None:
            return xform
        resolved = _fields_as_identity(forward)
        return xform if resolved is forward else resolved.inverse()
    if isinstance(xform, Sequence):
        parts = xform.transformations or []
        resolved = [_fields_as_identity(part) for part in parts]
        if any(new is not old for new, old in zip(resolved, parts)):
            return replace(xform, transformations=resolved)
        return xform
    return xform


def _nearest_resolution_index(
    resolutions: tx.Sequence[tx.Optional[ArrayProtocol]],
    voxel2world: tx.Optional[Affine],
) -> int:
    """The index of the resolution closest to a target grid.

    `resolutions` gives the physical grid size of each level, finest first,
    as a per-axis vector; `None` marks a level whose resolution is unknown.
    `voxel2world` is the affine of the grid being matched, already reduced
    by the caller, so that both sides of the comparison arrive in the same
    form; `None` marks a grid whose resolution is unknown.

    The comparison is made in logarithmic scale, so the level above and the
    level below the target are weighed evenly, and the finer level wins a
    tie. When any level's resolution, or the target's, is unknown, the
    finest level is chosen.

    This is shared by a multiscale field, whose levels are transformations,
    and a multiscale image, whose levels are images, so the two cannot
    disagree about which level matches a grid.
    """
    if len(resolutions) <= 1:
        return 0
    if any(resolution is None for resolution in resolutions):
        return 0
    if voxel2world is None or voxel2world.matrix is None:
        return 0
    target = axis_scales(voxel2world.matrix)
    if target is None:
        return 0
    ab = get_array_backend()
    log_target = float(ab.log(ab.abs(ab.asarray(target))).mean())
    best_index, best_distance = 0, None
    for index, resolution in enumerate(resolutions):
        log_resolution = float(ab.log(ab.abs(ab.asarray(resolution))).mean())
        distance = abs(log_resolution - log_target)
        if best_distance is None or distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def _at_resolution(
    transformation: Transformation, voxel2world: Transformation
) -> Transformation:
    # Select, inside a transformation, the scale of every multiscale field
    # whose resolution matches a target grid. The `voxel2world` is the
    # voxel-to-world transformation of the grid onto which an image is
    # resliced. Each multiscale field is replaced by its resolution-
    # matched scale, walking through any nesting of sequences. A
    # transformation that carries no multiscale field is returned
    # unchanged.
    if isinstance(transformation, Multiscale):
        return transformation.to_singlescale(
            transformation._nearest_level(voxel2world)
        )
    if isinstance(transformation, Sequence):
        parts = transformation.transformations or []
        resolved = [_at_resolution(part, voxel2world) for part in parts]
        if any(new is not old for new, old in zip(resolved, parts)):
            return replace(transformation, transformations=resolved)
        return transformation
    return transformation
