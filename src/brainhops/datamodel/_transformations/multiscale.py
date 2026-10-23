import typing_extensions as tx
from bagof.magic import Forget, replace

from brainhops._core.affines import axis_scales
from brainhops._core.affines import inv as _affine_inv
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.multiscale import Multiscale

from .base import Transformation
from .compute.multiscale import _as_affine
from .compute.simplify import SimplifyLike
from .modes import ModeLike
from .sequence import ImmutableSequence, Sequence


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
