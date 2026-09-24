# stdlib
from numbers import Integral

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.properties import smartproperty
from brainhops._core.typing import npvector

# datamodel
from brainhops.datamodel import hierarchy
from brainhops.datamodel.axes import Axis

# internals
from .base import Transformation
from .modes import ModeLike
from .simplify import SimplifyLike
from .simplify import simplify as _simplify

# typing
if tx.TYPE_CHECKING:
    from brainhops.datamodel.systems import CoordinateSystem

TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)


class MetaTransformation(Transformation):
    """
    A transformation that is defined in terms of another transformation.

    This is a base class for transformations that are defined in terms of
    other transformations, such as `SubspaceTransformation` and
    `Bijection`. It is not meant to be instantiated directly.
    """

    def compute(
        self, mode: ModeLike = True, *, simplify: SimplifyLike = "analytic",
    ) -> tx.Self:
        # A meta transformation holds no parameter of its own to fuse, so
        # computing it is simplifying it: the registered simplifier for its
        # type recurses into what it wraps, gated by `simplify`. `mode`
        # gates which kinds *compose*, and there is nothing here to compose.
        return _simplify(self, policy=simplify)


class SubspaceTransformation(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """
    A transformation that is applied to a subset of the input and output axes.

    The transformation acts on the axes named by `input_axes` and
    `output_axes`, and leaves every other axis unchanged. The
    dimensionality of the space is preserved. An axis that is not named
    passes through as the identity. This lifts a transformation defined
    over a few axes, such as a spatial transformation over `(x, y, z)`,
    into a larger space, such as `(x, y, z, t)`, where it acts on the
    spatial axes and leaves time untouched.

    Generic in the wrapped transformation type:
    `SubspaceTransformation[TRANSFORMATION]` lifts a `TRANSFORMATION`. Its
    membership is decided by a checker registered in `checkers` that recurses
    into the wrapped transform.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = (
        "transformation", "input_axes", "output_axes",
    )

    # --- attributes ---------------------------------------------------

    transformation: tx.Annotated[
        tx.Optional[TRANSFORMATION], tx.Doc("The transformation to apply.")
    ] = None

    input_axes: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc("The axes of the input coordinate system to transform."),
    ] = None

    output_axes: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc("The axes of the output coordinate system to transform."),
    ] = None

    # --- properties ---------------------------------------------------

    @smartproperty
    def input(self) -> tx.Optional["CoordinateSystem"]:
        system = getattr(self.transformation, "input", None)
        return _subsystem(system, self.input_axes, full=self._input)

    @smartproperty
    def output(self) -> tx.Optional["CoordinateSystem"]:
        system = getattr(self.transformation, "output", None)
        return _subsystem(system, self.output_axes, full=self._output)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        if self.transformation is None:
            return self.to(
                input=self._output,
                output=self._input,
                input_axes=self.output_axes,
                output_axes=self.input_axes,
            )
        kwargs.setdefault("compute", compute)
        return self.to(
            transformation=self.transformation.inverse(**kwargs),
            input=self._output,
            output=self._input,
            input_axes=self.output_axes,
            output_axes=self.input_axes,
        )


class Projection(MetaTransformation):
    """
    A transformation that acts as a projection from a higher dimensional
    space to a lower-dimensional space by removing one or more axes.

    Or its inverse (i.e., an embedding) that adds one or more axes to a
    lower-dimensional space.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = "dropped", "created"

    # --- attributes ---------------------------------------------------

    dropped: npvector[Integral] = ()
    created: npvector[Integral] = ()

    # --- methods ------------------------------------------------------

    def inverse(self) -> tx.Self:
        return self.to(
            dropped=self.created,
            created=self.dropped,
            input=self.output,
            output=self.input,
        )


@hierarchy.BijectiveTransformation.register
class Bijection(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """
    A transformation whose inverse is explicitly defined.

    Generic in the forward transformation type: `Bijection[TRANSFORMATION]`
    wraps a `TRANSFORMATION`. Declared bijective; a checker registered in
    `checkers` refines its membership from the forward map.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = "forward", "backward"

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[TRANSFORMATION], tx.Doc("The forward transformation.")
    ] = None

    backward: tx.Annotated[
        tx.Optional[TRANSFORMATION], tx.Doc("The backward transformation.")
    ] = None

    # --- properties ---------------------------------------------------

    @smartproperty
    def input(self) -> tx.Optional["CoordinateSystem"]:
        if self.forward is not None and self.forward.input is not None:
            return self.forward.input
        if self.backward is not None and self.backward.output is not None:
            return self.backward.output
        return None

    @smartproperty
    def output(self) -> tx.Optional["CoordinateSystem"]:
        if self.forward is not None and self.forward.output is not None:
            return self.forward.output
        if self.backward is not None and self.backward.input is not None:
            return self.backward.input
        return None

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        obj = self.to(
            forward=self.backward,
            backward=self.forward,
            input=self._output,
            output=self._input,
        )
        if compute:
            obj = obj.compute(**kwargs)
        return obj


def _subsystem(
    system: tx.Optional["CoordinateSystem"] = None,
    index: tx.Optional[tx.Sequence[Integral]] = None,
    full: tx.Optional["CoordinateSystem"] = None,
) -> tx.Optional["CoordinateSystem"]:
    """Build the full-space system a subspace transform presents.

    `system` is the inner transform's own (subspace) coordinate system,
    which names one axis per acted-on dimension. `index` gives the
    positions those axes occupy in the full space.

    When a declared endpoint (`full`) is available, it is returned as is.
    Otherwise a full-space system is reconstructed: the inner system's
    axis `j` is placed at position `index[j]`, and any position no inner
    axis lands on is filled with a placeholder [`Axis`][]. This spans
    `max(index) + 1` axes, so an endpoint-less subspace whose axes exceed
    the inner system's length no longer indexes past its end.
    """
    if full is not None:
        return full
    if index is None:
        return system
    axes = getattr(system, "axes", None)
    if axes is None:
        return system
    index = list(index)
    n = max(index) + 1 if index else 0
    new_axes = [Axis() for _ in range(n)]
    for j, i in enumerate(index):
        if j < len(axes):
            new_axes[i] = axes[j]
    return replace(
        system,
        axes=new_axes,
        name=f"subspace({system.name})" if system.name else None,
    )
