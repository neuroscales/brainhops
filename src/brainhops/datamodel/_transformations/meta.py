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

# internals
from .base import Transformation

# typing
if tx.TYPE_CHECKING:
    from brainhops.datamodel.systems import CoordinateSystem


class MetaTransformation(Transformation):
    """
    A transformation that is defined in terms of another transformation.

    This is a base class for transformations that are defined in terms of
    other transformations, such as `SubspaceTransformation` and
    `Bijection`. It is not meant to be instantiated directly.
    """


class SubspaceTransformation(MetaTransformation):
    """
    A transformation that is applied to a subset of the input and output axes.

    The transformation acts on the axes named by `input_axes` and
    `output_axes`, and leaves every other axis unchanged. The
    dimensionality of the space is preserved. An axis that is not named
    passes through as the identity. This lifts a transformation defined
    over a few axes, such as a spatial transformation over `(x, y, z)`,
    into a larger space, such as `(x, y, z, t)`, where it acts on the
    spatial axes and leaves time untouched.
    """

    parameter_names: tx.ClassVar[str] = "transformation"

    # --- attributes ---------------------------------------------------

    transformation: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The transformation to apply.")
    ] = None

    input_axes: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc("The axes of the input coordinate system to transform."),
    ] = None

    output_axes: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc("The axes of the output coordinate system to transform."),
    ] = None

    @smartproperty
    def input(self) -> tx.Optional["CoordinateSystem"]:
        system = getattr(self.transformation, "input", None)
        return _subsystem(system, self.input_axes)

    @smartproperty
    def output(self) -> tx.Optional["CoordinateSystem"]:
        system = getattr(self.transformation, "output", None)
        return _subsystem(system, self.output_axes)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        if self.transformation is None:
            return replace(
                self,
                input=self._output,
                output=self._input,
                input_axes=self.output_axes,
                output_axes=self.input_axes,
            )
        kwargs.setdefault("compute", compute)
        return replace(
            self,
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

    # --- attributes ---------------------------------------------------

    dropped: npvector[Integral] = ()
    created: npvector[Integral] = ()

    # --- methods ------------------------------------------------------

    def inverse(self) -> tx.Self:
        return replace(
            self,
            dropped=self.created,
            created=self.dropped,
            input=self.output,
            output=self.input,
        )

    def compute(self, *args, **kwargs) -> tx.Self:
        return self


@hierarchy.BijectiveTransformation.register
class Bijection(Transformation):
    """
    A transformation whose inverse is explicitly defined.
    """

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The forward transformation.")
    ] = None

    backward: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The backward transformation.")
    ] = None

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
        obj = replace(
            self,
            forward=self.backward,
            backward=self.forward,
            input=self._output,
            output=self._input,
        )
        if compute:
            obj = obj.compute(**kwargs)
        return obj

    def compute(self, *args, **kwargs) -> tx.Self:
        forward, backward = self.forward, self.backward
        if forward is not None:
            forward = forward.compute(*args, **kwargs)
        if backward is not None:
            backward = backward.compute(*args, **kwargs)
        if forward is None and backward is None:
            return self
        return replace(self, forward=forward, backward=backward)


def _subsystem(
    system: tx.Optional["CoordinateSystem"] = None,
    index: tx.Optional[tx.Sequence[Integral]] = None,
) -> tx.Optional["CoordinateSystem"]:
    """Build a subsystem from a coordinate system."""
    if index is None:
        return system
    axes = getattr(system, "axes", None)
    if axes is None:
        return system
    subaxes = [axes[i] for i in index]
    return replace(
        system,
        axes=subaxes,
        name=f"subspace({system.name})" if system.name else None,
    )
