# dependencies
import typing_extensions as tx

# api
from brainhops._core.properties import smartproperty
from brainhops._core.typing import is_instance_or_subclass
from brainhops.datamodel import kinds
from brainhops.datamodel.base import DataModelBase, IdentityComparison
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import ConversionError, LossyConversionError

# internals
from . import nocycles
from .compute.check import is_kind
from .compute.convert import convert
from .compute.simplify import SimplifyLike
from .compute.utils import require_endomorphism
from .modes import ModeLike

# typing
if tx.TYPE_CHECKING:
    from .concrete import CoordinatesField
    from .sequence import Sequence


@kinds.Transformation.register  # virtual registration in hierarchy
@nocycles.register_transformation  # register in registry for cyclic imports
class Transformation(
    IdentityComparison, DataModelBase, reverse=True, eq=False, kw_only=True
):
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
        transformation would be represented as
        `Transformation(input=B, output=A)`.

    !!! note "Transformations compare by identity"
        `t1 == t2` is `t1 is t2`: two distinct transformations are never
        equal, even when they hold the same parameters in the same
        systems, and `==` never raises. This is to avoid triggering
        expensive array computations when comparing transformations.

        A transformation hashes by identity too, so it can be put in a
        set or used as a dictionary key.

        To test whether two transformations are equivalent, one solution
        is to check whether one composed with the inverse of the other
        is the identity:

        ```python
        is_identity((~t1 @ t2).compute(), compute=True)
        ```

        This still comes with caveats: this test requires one of the
        transformations to be invertible, and the numeric test performed
        by `is_identity` only returns `True` when the resulting
        transformation is exactly the identity. Numerical errors
        intrinsic to floating point arithmetic may cause two
        transformations that are equivalent to not be recognized as such.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """The attributes that parameterize the transformation."""

    metadata_fields: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """The meta-attributes that define the encoding."""

    derived_fields: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """The attributes that are derived from other attributes."""

    mutually_exclusive_fields: tx.ClassVar[
        tx.Optional[tx.Tuple[str, ...]]
    ] = None
    """
    Arguments that cannot be set together in `__init__`.

    If `None`, it is the one stored parameter and the views that spell it
    differently -- `data_fields + derived_fields` -- which is a group only
    for a class that stores a single parameter. A class whose data fields
    are independent of one another has no such group by default, and
    declares one here if it needs it.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        mutex = self.mutually_exclusive_fields
        if mutex is None:
            # `data`, `matrix`, `field`, `values`, `coefficients`... are
            # names for one parameter, so two of them is that parameter
            # twice. A class with several data fields -- a subspace's
            # `transformation` and its axes, a bijection's two directions
            # -- sets them together, so the default group does not apply.
            mutex = (
                self.data_fields + self.derived_fields
                if len(self.data_fields) <= 1 else ()
            )
        _mutually_exclusive(self, arguments, mutex)
        self._set_derived_fields(arguments)

    def _set_derived_fields(self, arguments: tx.Any) -> None:
        for name in self.derived_fields:
            if (value := arguments.get(name)) is not None:
                setattr(self, name, value)

    # --- attributes ---------------------------------------------------

    # The endpoints are stored under private names, so the constructor
    # argument stays `input=`/`output=` while `replace()` carries over
    # what the transform was *given* rather than what its `input`/`output`
    # property reports. A subclass whose endpoints are derived would
    # otherwise have every `replace()` freeze the derived value into a
    # declared one.
    #
    # The endpoints are keyword-only. A positional parameter's place in
    # `__init__` follows the order bagof collects fields in, which follows
    # the MRO, so it would move whenever a class reorders its bases and
    # differ between families. Only the fields that define a
    # transformation (`matrix`, `field`, `transformations`, `forward`...)
    # are positional. bagof does not carry `KwOnly` over to a field that
    # a subclass declares again: a subclass that gives an endpoint a new
    # default must write `KwOnly[...]` itself.

    _input: tx.Optional[CoordinateSystem] = None
    """
    The input coordinate system of the transformation.

    If not specified, it can be inferred from the context
    (e.g., from the coordinate system of the image being transformed).
    """

    _output: tx.Optional[CoordinateSystem] = None
    """
    The output coordinate system of the transformation.

    If not specified, it can be inferred from the context
    (e.g., from the coordinate system of the image being transformed).
    """

    input = smartproperty("input")
    output = smartproperty("output")

    # --- methods ------------------------------------------------------

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        """
        Compute the transformation, if it is not already fully defined.

        Parameters
        ----------
        mode : [list of] name or type, optional
            Which kinds of transformations to materialize.
            `True` (the default) admits every kind.
            On a leaf transformation, if a `mode` is given and this leaf
            is not admitted by it, the leaf is returned unchanged.
            Keys are transformation types, names or symbols
            (e.g., `"affine"`, `"Aff"`, `"rigid"`, `"SO(3)"`).
        simplify : SimplifyLike, default="analytic"
        What simplifications to apply to the transformation. See
            Whether to simplify the transformation prior if possible,
            and how hard to try to simplify them.
            * `"analytic"` (the default) looks at the type structure only;
            * `"numeric"` looks at the numeric values of the transformation;
            * `False`/`"none"`/`None` disables simplification.
        factor : bool, default=False
            Whether to rewrite the transformation into its axis-group
            normal form, splitting it into independent factors that each
            act on a group of axes that transform together. Off by default,
            so a plain `compute()` result is unchanged. Nothing is ever
            composed across axis groups; `mode` still decides whether the
            restricted pieces inside a group compose.
        """
        # `compute()` has no meaningful default: every family implements it
        # with the behaviour that fits its type -- `ConcreteTransformation`
        # runs the kind-checks, `MetaTransformation` returns itself once
        # mode-gated, and `Sequence`, `Inverse`, `Bijection`, `Projection`
        # and the multiscale containers compose or materialize their
        # contents. Raising here (rather than returning `self`) makes a
        # subclass that forgets to implement `compute()` fail loudly,
        # mirroring `inverse()`.
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute()"
        )

    def simplify(
        self,
        policy: SimplifyLike = "analytic",
        *,
        compute: tx.Union[ModeLike, bool, None] = False,
    ) -> tx.Self:
        """Simplify this transformation under a per-kind policy.

        Convenience sugar for
        [`compute`][]: `t.simplify(policy, compute=mode)` is
        `t.compute(mode, simplify=policy)`.

        By default `simplify()` does no computation at all: `compute=False`
        maps to `mode=False`, which composes nothing (no matrices multiplied,
        no fields sampled, no lazy inverse materialized). It only downcasts
        each leaf under `policy` (analytic by default). Pass an explicit
        `compute=<mode>` to also compose that kind.

        Parameters
        ----------
        policy : simplify policy, default="analytic"
            The simplify policy, in the grammar `compute` accepts.
        compute : [list of] name or type, default=False
            The compose mode. The default, `False`, composes nothing
            (`mode=False` in `compute`); `None` would compose every kind. A
            real mode passes straight through.
        """
        return self.compute(compute, simplify=policy)

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

    def square(self, compute: bool = False, **kwargs) -> "Transformation":
        """
        Return the square of this transformation, `self @ self`.

        The square is the sequence `[self, self]`, which composes when it
        is computed. It is defined for a transformation that maps a space
        to itself.

        Parameters
        ----------
        compute : bool, default=False
            Whether to compute the result now rather than return it lazily.
        **kwargs
            Passed to [`compute`][] when `compute` is true.

        Raises
        ------
        DomainError
            If the transformation does not map a space to itself.
        """
        require_endomorphism(self, "square")
        obj = nocycles.SEQUENCE([self, self])
        return obj.compute(**kwargs) if compute else obj

    def sqrt(self, compute: bool = False, **kwargs) -> "Transformation":
        """
        Return the principal square root of this transformation.

        The square root `S` of `T` is the transformation with
        `S @ S == T`, the half-transformation. The principal one, whose
        linear part has its eigenvalues in the open right half-plane, is
        unique, and it is of the same kind as `T`: the square root of a
        rotation is a rotation, of a translation a translation, of a
        scaling a scaling, of an affine an affine. The square root of a
        permutation is a [`Linear`][] transformation.

        The square root is lazy: an [`Sqrt`][] wrapper is returned, and
        computed when it is applied, computed or converted. A transformation
        that needs no wrapper (an identity) is returned as is. A
        [`Sequence`][] is reduced first (see [`Sequence.sqrt`][]).

        Parameters
        ----------
        compute : bool, default=False
            Whether to compute the result now rather than return it lazily.
        **kwargs
            Passed to [`compute`][] when `compute` is true.

        Raises
        ------
        DomainError
            If the transformation does not map a space to itself, or, when
            the result is computed, if its linear part has an eigenvalue on
            the closed negative real axis (a reflection, a rotation by a
            half turn, a singular matrix), so that it has no real principal
            square root.
        NotImplementedError
            If brainhops does not compute the square root of this kind of
            transformation. A displacement field has one only when it is a
            stationary velocity field (`log=True`), whose square root
            halves its velocity.
        """
        raise NotImplementedError(
            f"The square root of a {type(self).__name__} is not implemented."
        )

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

        Conversion can be

        * between types: `linear.to(Affine)`                       ; or
        * within a type: `displacement.to(store="coefficients")`   ; or
        * both: `coords.to(DisplacementField, store="values")`     .

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
            passing `store="coefficients"`: a change of encoding flag
            re-encodes the stored `data`, and keeps the map. A view's
            name (`field=`, `matrix=`, ...) sets the map, as values, and
            it is stored in the encoding of the result; `data=` is
            stored as given.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        # Conversion can be
        # * between types: `linear.to(Affine)`                       ; or
        # * within a type: `displacement.to(store="coefficients")`   ; or
        # * both:          `coords.to(DisplacementField, store=...)` .
        #
        # All of these things are handled by `convert()`. Within type
        # conversion calls the `T -> T` converter, whereas between types
        # conversion calls the `T1 -> T2` converter.
        cls = cls or self._target_class(kwargs)
        try:
            return convert(self, cls, **kwargs)
        except LossyConversionError as e:
            if lossy:
                # The conversion is possible but discards information,
                # and the caller asked for it anyway. The transform it
                # would have produced travels on the exception.
                return e.result
            failure = e
        except ConversionError as e:
            failure = e
        # The conversion failed and the caller did not opt into the loss.
        # `error` says what to do about it: re-raise, raise something
        # else, or stand in for the result.
        if error is True:
            raise failure
        if is_instance_or_subclass(error, Exception):
            raise error from failure
        return error

    def _target_class(
        self, kwargs: tx.Dict[str, tx.Any]
    ) -> tx.Type["Transformation"]:
        """The type [`to`][] converts to when the caller names none.

        This class, normally: `t.to(store="coefficients")` is still a `t`.
        A class that a flag *selects*, though, answers with the one the
        new value of that flag selects, since the flag and the class say
        the same thing: `log=False` on a tangent asks for the map, which
        the plain class of the family holds and the tangent does not.

        Naming the class here rather than inside the converter is what
        lets [`convert`][] check that what came back is of the type that
        was asked for.
        """
        return type(self)

    # --- kind checks --------------------------------------------------

    def is_kind(
        self, kind: tx.Type[kinds.Kind], compute: bool = False
    ) -> bool:
        """
        Return whether a transformation is of a given kind.

        A transformation is recognized as a kind when it is a (virtual)
        instance  of the specified kind, or when its parameters are
        compatible with that kind.

        Parameters
        ----------
        kind : type[kinds.Kind]
            The kind to check for.
        compute : bool, default=False
            Whether to look at the numeric values of the transformation's
            parameters, rather than just its type structure, to determine
            whether it is of the given kind.

        Returns
        -------
        bool
            True if the transformation is of the specified kind,
            False otherwise.
        """
        return is_kind(self, kind, compute)

    def is_identity(self, compute: bool = False) -> bool:
        """Return whether a transformation is the identity.

        A transformation is recognized as the identity when its parameters
        are unset, or when it is an instance of [`Identity`][]. When
        `compute` is true, the parameters of a transformation such as
        [`Translation`][], [`Scaling`][], [`Permutation`][], [`Linear`][],
        [`Affine`][], or [`DisplacementField`][] are also inspected, so that
        a transformation whose parameters happen to encode the identity is
        recognized as such even though it is not stored as one.

        A [`CartesianField`][] is a regular grid of coordinates, which is the
        identity map over its grid by construction. When `compute` is true, a
        [`CartesianField`][] is therefore recognized as the identity. When
        `compute` is false, a [`CartesianField`][] that carries a grid is not
        recognized as the identity, because its `shape` parameter is set. A
        [`CartesianField`][] with no grid has an unset `shape` parameter and
        is recognized as the identity by the parameter check under either
        value of `compute`. The `shape` is read, never the derived `field`, so
        the meshgrid is not built by a structural check.

        Recognizing a grid as the identity does not mean a grid may be dropped
        on sight. A grid also defines the sampling domain onto which data is
        resampled. The decision to factor a grid away is made by the sequence
        simplifier, which only does so for a grid that sits strictly between
        two other transformations.
        """
        return self.is_kind(kinds.Identity, compute)

    def is_translation(self, compute: bool = False) -> bool:
        """Return whether a transformation is a pure translation.

        A transformation is recognized as a translation when it is an
        instance of [`kinds.Translation`][], or when [`is_identity`][]
        recognizes it as the identity, which is itself a translation by zero.

        When `compute` is true, the matrix of an [`Affine`][] transformation
        is also inspected for a linear part equal to the identity.
        """
        return self.is_kind(kinds.Translation, compute)


    def is_scaling(self, compute: bool = False) -> bool:
        """Return whether a transformation is a pure scaling.

        A transformation is recognized as a scaling when it is an instance
        of [`kinds.Diagonal`][], or when [`is_identity`][]
        recognizes it as the identity, which is itself a scaling by one.

        When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
        transformation is also inspected for a diagonal structure.
        """
        return self.is_kind(kinds.Diagonal, compute)


    def is_permutation(self, compute: bool = False) -> bool:
        """Return whether a transformation is a pure permutation of axes.

        A transformation is recognized as a permutation when it is an
        instance of [`kinds.Permutation`][], or when [`is_identity`][]
        recognizes it as the identity, which is itself a trivial permutation.

        When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
        transformation is also inspected for a binary, one-per-row and
        one-per-column structure.
        """
        return self.is_kind(kinds.Permutation, compute)


    def is_rotation(self, compute: bool = False) -> bool:
        """Return whether a transformation is a pure rotation.

        A transformation is recognized as a rotation when it is an instance
        of [`kinds.SpecialOrthogonal`][], or when
        [`is_identity`][] recognizes it as the identity, which is itself a
        rotation by zero.

        When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
        transformation is also inspected for orthogonality and a positive
        determinant.
        """
        return self.is_kind(kinds.SpecialOrthogonal, compute)


    def is_linear(self, compute: bool = False) -> bool:
        """Return whether a transformation is linear, without a translation.

        A transformation is recognized as linear when it is an instance of
        [`kinds.Linear`][], or when [`is_identity`][]
        recognizes it as the identity, which is itself linear.

        When `compute` is true, the matrix of an [`Affine`][] transformation
        is also inspected for a zero translation component.
        """
        return self.is_kind(kinds.Linear, compute)


    def is_affine(self, compute: bool = False) -> bool:
        """Return whether a transformation is affine.

        A transformation is recognized as affine when it is an instance
        of [`kinds.Affine`][], or when [`is_identity`][]
        recognizes it as the identity, which is itself affine.
        """
        return self.is_kind(kinds.Affine, compute)

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

    @tx.overload
    def __call__(
        self, x: "CoordinatesField", compute: ModeLike
    ) -> "CoordinatesField":
        """
        Transform coordinates, computing the result under the given mode.

        Besides `True` and `False`, `compute` accepts a mode -- a
        transformation-type name, a type, or a list of either -- which is
        forwarded to [`compute`][] as its `mode` and selects which kinds of
        transformations are composed.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "Transformation", compute: ModeLike
    ) -> "Transformation":
        """
        Compose this transform with another transform, computing the
        result under the given mode.

        `compute` mirrors the `mode` argument of [`compute`][]:

        * `compute=True` computes with `mode=None` (every kind),
        * `compute=<mode>` computes with that mode, and
        * `compute=False` returns the uncomputed sequence.
        """
        ...

    def __call__(self, x, compute: bool = False) -> "Transformation":
        # `compute=True` computes with `mode=None` (every kind);
        # `compute=<mode>` computes with that mode; `compute=False` returns
        # the uncomputed sequence.
        if isinstance(x, Transformation):
            x = nocycles.SEQUENCE([x, self])
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


def _mutually_exclusive(
    xform: Transformation, arguments: tx.Any, names: tx.Tuple[str, ...]
) -> None:
    """
    Refuse a call that spells one parameter more than once.

    `names` is the stored parameter and the views that are other
    spellings of it: `data` and `matrix`, or `data`, `field`, `values`
    and `coefficients`. A view is the map and fills `data`, so no two of
    them can be given together. `replace()` reaches here too: it carries
    `data` over.
    """
    given = [f"{name}=" for name in names if arguments.get(name) is not None]
    if len(given) <= 1:
        return
    raise TypeError(
        f"{type(xform).__name__}() got {', '.join(given)}, which are "
        f"spellings of one parameter. To change the map of an existing "
        f"transformation, use t.to(...); to store an array as it is "
        f"encoded, pass data= alone."
    )
