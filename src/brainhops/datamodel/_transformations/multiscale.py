import typing_extensions as tx
from bagof.magic import NotKwOnly, replace

from brainhops._core.affines import axis_scales
from brainhops._core.affines import inv as _affine_inv
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol, Derived
from brainhops.backends import get_array_backend
from brainhops.datamodel.base import DataModelBase
from brainhops.errors import ConversionError

from .base import Transformation
from .compute.simplify import SimplifyLike
from .concrete import Affine, CoordinatesField, DisplacementField, Identity
from .inverse import Inverse
from .modes import ModeLike
from .sequence import ImmutableSequence, Sequence

SINGLE_SCALE = tx.TypeVar("SINGLE_SCALE", covariant=True)


class Multiscale(DataModelBase, tx.Generic[SINGLE_SCALE]):
    """Mixin for a pyramid of scales, finest first.

    Index 0 is the scale with the highest resolution. The mixin provides the
    list of scales and the operations that select one. It does not define what
    a scale is: a subclass supplies the scales and, through the
    `_level_resolution` hook, the physical grid size of each one, from which
    `_nearest_level` picks the scale whose resolution is closest to a target
    grid.
    """

    # --- attributes ---------------------------------------------------

    scales: NotKwOnly[tx.List[SINGLE_SCALE]] = ()
    """Scales, from finest to coarsest."""

    @property
    def nscales(self) -> int:
        """Number of scales."""
        return len(self.scales)

    @property
    def _finest(self) -> SINGLE_SCALE:
        # Return the finest scale, or `None` if there are no scales.
        scales = self.scales
        return scales[0] if scales else None

    # --- methods ------------------------------------------------------

    def to_singlescale(self, index: int = 0) -> SINGLE_SCALE:
        """Return the scale at `index`, where 0 is the finest.

        The scale is returned as it is stored: for a field, it is the plain
        transformation of that scale, not a multiscale field.
        """
        return self.scales[int(index)]

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # Return the physical grid size of a level, as a vector with one entry
        # per axis in the input units, or `None` if the size is unknown. This
        # hook returns `None` here, and subclasses override it.
        return None

    def _nearest_level(self, voxel2world: Transformation) -> int:
        # Return the index of the scale whose resolution is closest to the
        # grid of `voxel2world`. The comparison is done by
        # `_nearest_resolution_index`, which the multiscale image also uses.
        scales = self.scales
        resolutions = [self._level_resolution(i) for i in range(len(scales))]
        return _nearest_resolution_index(
            resolutions, _as_affine_ignoring_fields(voxel2world)
        )


class MultiscaleField(Multiscale[Sequence], ImmutableSequence):
    """Field of coordinates or displacements at several resolutions.

    Each scale is a [`Sequence`][] that maps the input space to the output
    space on the grid of that scale. A scale of coordinates is a sequence of
    two elements, a world-to-voxel affine and a coordinates field. A scale of
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

    scales: NotKwOnly[tx.List[Sequence]] = ()
    """Scales, from finest to coarsest.

    Each scale is a sequence that maps the input space to the output space,
    sampled on the grid of that scale.
    """

    # `transformations` is served from the finest scale rather than stored, so
    # it is not a constructor field. Declaring the storage slot of `Sequence`
    # as derived keeps it out of `__init__`, `fields()` and `replace()`.
    _transformations: Derived[tx.Tuple[Transformation, ...]]

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

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # The resolution of a scale is read from its leading world-to-voxel
        # affine: the norm of each column of its inverse is the voxel size
        # along that axis. If the leading element is not a defined affine, the
        # resolution is unknown.
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
    # Return the transformation as an `Affine`. If the transformation does not
    # reduce to an affine with a defined matrix, `None` is returned rather than
    # an error. A caller that must know whether the transformation really is
    # affine uses this function, not `_as_affine_ignoring_fields`.
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
    # Return the affine part of the transformation, with every field treated
    # as the identity, or `None` if what remains is not affine. A warp has no
    # affine form, but it is close to an isometry, so the affine part still
    # measures the scale of a chain that contains a warp. The result answers
    # questions about scale only.
    return _as_affine(_fields_as_identity(xform))


def _fields_as_identity(xform: Transformation) -> Transformation:
    # Replace every field with the identity and keep the affine part. Fields
    # are recognised by their class, so the typed inverse of a field is also
    # replaced.
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
    """Return the index of the resolution closest to a target grid.

    `resolutions` holds the physical grid size of each level, finest first, as
    a vector with one entry per axis, or `None` where it is unknown.
    `voxel2world` is the affine of the grid being matched, already reduced by
    the caller, or `None`. Resolutions are compared in log scale, by the mean
    logarithm of the absolute axis scales, so that a level above the target and
    a level below it are weighed evenly, and the finer level wins a tie. When a
    level or the target is unknown, or when there is at most one level, the
    finest level (0) is returned. Multiscale fields and multiscale images share
    this function, so they cannot disagree.
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
    # Replace every multiscale field inside the transformation, walking nested
    # sequences, by its scale that best matches the grid of `voxel2world`, the
    # voxel-to-world transformation of the grid an image is resliced onto. A
    # transformation without a multiscale field is returned unchanged.
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
