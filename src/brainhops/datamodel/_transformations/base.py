# dependencies
import typing_extensions as tx

# api
from brainhops._core.properties import smartproperty
from brainhops.datamodel import hierarchy
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem

# internals
from . import registries
from .convert import convert
from .errors import ConversionError, LossyConversionError

# typing
if tx.TYPE_CHECKING:
    from .concrete import CoordinatesField
    from .sequence import Sequence


@hierarchy.Transformation.register
class Transformation(DataModelBase, reverse=True):
    """
    A transformation between coordinate systems.

    It maps coordinates from an input coordinate system to an output
    coordinate system. Transformations can be applied and/or composed
    using different syntaxes:

    1. Functional: `t(x)` applies the transform `t` to coordinates `x`,
       and `t2(t1)` composes the transform `t1` with the transform `t2`
       (i.e., applies `t1` first, then `t2`).

    2. Matrix-like: `t @ x` applies the transform `t` to coordinates `x`,
       and `t2 @ t1` composes the transform `t1` with the transform `t2`.

       This abuses the matrix multiplication operator `@` because linear
       and affine transformations can be represented as matrices, and are
       typically applied to coordinates using matrix multiplication, with
       the input space "on the right" and the output space "on the left".

    !!! warning "Direction of transformation"
        This mapping direction is the opposite of the direction that is
        typically used to transform images. For example, a transformation
        that deforms an image from space A to space B, will actually
        map coordinates from space B to space A. In our model, this
        transformation would be represented as `Transform(input=B, output=A)`.
    """

    parameter_names: tx.Annotated[
        tx.ClassVar[tx.Union[str, tx.Tuple[str, ...]]],
        tx.Doc("The attributes that parameterize the transformation."),
    ] = ()

    # --- attributes ---------------------------------------------------

    # The endpoints are stored under private names, so the constructor
    # argument stays `input=`/`output=` while `replace()` carries over
    # what the transform was *given* rather than what its `input`/`output`
    # property reports. A subclass whose endpoints are derived -- a
    # `Sequence` reads them off its children -- would otherwise have every
    # `replace()` freeze the derived value into a declared one, and a
    # sequence that declares nothing would come back claiming the systems
    # its children happen to name.
    _input: tx.Annotated[
        tx.Optional[CoordinateSystem],
        tx.Doc(
            """
            The input coordinate system of the transformation.
            If not specified, it can be inferred from the context (e.g.,
            from the coordinate system of the image being transformed).
            """
        ),
    ] = None

    _output: tx.Annotated[
        tx.Optional[CoordinateSystem],
        tx.Doc(
            """
            The output coordinate system of the transformation.
            If not specified, it can be inferred from the context (e.g.,
            from the coordinate system of the image being transformed).
            """
        ),
    ] = None

    input = smartproperty("input")
    output = smartproperty("output")

    # --- methods ------------------------------------------------------

    def compute(self, simplify: bool = False) -> tx.Self:
        """
        Compute the transformation, if it is not already fully defined.
        """
        # `Sequence`, `Inverse`, `Bijection`, `Projection` and the
        # multiscale containers overload `compute()`, so here we can assume
        # that `self` is none of those. Every other transformation -- a
        # concrete one that holds a parameter, or a meta one that wraps
        # another transformation -- simplifies to the simplest compatible
        # kind, whose compatibility can be detected with (almost) no
        # overhead. For example, if the parameter of a transformation is
        # set to `None`, the transformation is treated as an identity
        # transformation.
        #
        # The checks live in `concrete`, which imports this module, so they
        # are imported lazily to avoid an import cycle.
        from .concrete import (
            Identity,
            Linear,
            Permutation,
            Rotation,
            Scaling,
            Translation,
            is_identity,
            is_linear,
            is_permutation,
            is_rotation,
            is_scale,
            is_translation,
        )

        CHECKS = [
            (is_identity, Identity),
            (is_translation, Translation),
            (is_scale, Scaling),
            (is_permutation, Permutation),
            (is_rotation, Rotation),
            (is_linear, Linear),
        ]
        for check, cls in CHECKS:
            if check(self, compute=simplify):
                return self.to(cls)
        return self

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        """
        Return the inverse of this transformation.

        Some classes of transformations return a lazy inverse by default,
        which is only evaluated when the `compute()` method is called.
        This allows for efficient composition of transformations.
        Or the inverse can be computed immediately by setting `compute=True`.

        The inverse can also be obtained using the `__invert__` operator:
        `T.inverse()` is equivalent to `~T`.
        """
        raise NotImplementedError("Transformation.inverse()")

    def to(
        self,
        cls: tx.Optional[tx.Type[tx.Self]] = None,
        *,
        lossy: bool = False,
        error: tx.Union[tx.Type[Exception], Exception, bool] = True,
        **kwargs,
    ) -> tx.Self:
        """
        Convert this transformation to a different type.

        Parameters
        ----------
        cls : type, optional
            The type to convert to. If `None`, keep the current type.
        lossy : bool, default=False
            Whether to allow lossy conversions.
        error : bool or Exception, default=True
            Whether to raise an error if the conversion fails:

            * If an `Exception`, raise it.
            * If `True`, raise the original error.
            * Otherwise, return the value of `error`.
        **kwargs : dict
            Attributes to override in the converted transform.
            This allows transformations to be modified within their type.
            For example, a [`DisplacementField`][] can be converted from
            a field of values to a field of spline coefficients by
            setting `coeff=True` in `kwargs`.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        cls = cls or type(self)
        try:
            return convert(self, cls, **kwargs)
        except LossyConversionError as e:
            if lossy:
                return e
        except ConversionError as e:
            if error is True:
                raise
            elif isinstance(error, Exception) or (
                isinstance(error, type) and issubclass(error, Exception)
            ):
                raise error from e
        return error

    # --- operators ----------------------------------------------------

    @tx.overload
    def __call__(
        self, x: "CoordinatesField", compute: tx.Literal[True]
    ) -> "CoordinatesField":
        """
        Transform coordinates from the input coordinate system to the
        output coordinate system.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "CoordinatesField", compute: tx.Literal[False]
    ) -> "Sequence":
        """
        Transform coordinates from the input coordinate system to the
        output coordinate system.

        This variant does not compute the transformed coordinates, but
        instead returns an object that holds the original coordinates
        and the transform, and can be computed later when needed.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "Transformation", compute: tx.Literal[True]
    ) -> "Transformation":
        """
        Compose this transform with another transform.

        The resulting transform will have the input coordinate system of
        the other transform and the output coordinate system of this
        transform.

        If the output coordinate system of the other transform does not
        match the input coordinate system of this transform, it will
        try to guess an intermediate adapter transform, and raise an
        error if it cannot find one.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "Transformation", compute: tx.Literal[False]
    ) -> "Sequence":
        """
        Compose this transform with another transform, without computing
        the resulting transform, but instead returning a sequence of the
        two transformations that can be computed later when needed.
        """
        ...

    def __call__(self, x, compute: bool = False) -> "Transformation":
        if isinstance(x, Transformation):
            x = registries.SEQUENCE([x, self])
        else:
            raise TypeError(f"Unsupported type for transformation: {type(x)}")
        if compute is not False:
            mode = None
            if compute is not True:
                mode = compute
            x = x.compute(mode=mode)
        return x

    def __matmul__(self, other: "Transformation") -> "Transformation":
        # A right operand that composes itself -- a `Geometry`, which folds
        # the composition into its transformation and stays a `Geometry` --
        # gets the first say. Python only offers the reflected operand that
        # precedence when its class is a proper subclass of this one, and
        # `Sequence` is a factory for `MutableSequence`, so a sibling class
        # would never be asked. Deferring here does not depend on the class
        # tree.
        rmatmul = getattr(type(other), "__rmatmul__", None)
        if rmatmul is not None and not isinstance(other, type(self)):
            result = rmatmul(other, self)
            if result is not NotImplemented:
                return result
        return self(other)

    def __or__(self, other: "Transformation") -> "Transformation":
        return other(self)

    def __invert__(self) -> "Transformation":
        return self.inverse()
