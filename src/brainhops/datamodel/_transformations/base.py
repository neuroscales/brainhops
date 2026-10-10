import typing_extensions as tx

from brainhops._core.properties import smartproperty
from brainhops._core.typing import is_instance_or_subclass
from brainhops.datamodel import kinds
from brainhops.datamodel.base import DataModelBase, IdentityComparison
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import ConversionError, LossyConversionError

from . import nocycles
from .compute.check import is_kind
from .compute.convert import convert
from .compute.simplify import SimplifyLike
from .compute.utils import require_endomorphism
from .modes import ModeLike

if tx.TYPE_CHECKING:
    from .concrete import CoordinatesField
    from .sequence import Sequence


@kinds.Transformation.register
@nocycles.register_transformation
class Transformation(
    IdentityComparison, DataModelBase, reverse=True, eq=False, kw_only=True
):
    """A map of coordinates from an input system to an output system.

    In the functional syntax, `t(x)` applies `t` to `x`, and `t2(t1)` composes
    two transformations so that `t1` is applied first. The matrix syntax
    borrows `@` from linear and affine maps, which are matrices: `t @ x`
    applies `t`, and `t2 @ t1` is the same composition, with the input space on
    the right and the output space on the left. Applying a transformation to
    coordinates is also a composition, since a [`CoordinatesField`][] is a
    transformation. The pipe `t1 | t2` composes in the order of application,
    and `~t` is the inverse of `t`.

    !!! warning "Direction of transformation"
        The direction of the mapping is opposite to the usual convention for
        transforming images. A warp that deforms image A into image B maps the
        coordinates of B to those of A, so it is a
        `Transformation(input=B, output=A)`.

    !!! note "Transformations compare by identity"
        `t1 == t2` means `t1 is t2`, so the comparison never raises and never
        compares arrays. Transformations also hash by identity and can be used
        as dictionary keys. Two transformations can be tested for equivalence
        with

        ```python
        is_identity((~t1 @ t2).compute(), compute=True)
        ```

        This test requires `t1` to be invertible, and because it accepts only
        an exact identity, floating-point error can make it miss equivalent
        transformations.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """Names of the attributes that parameterize the transformation."""

    metadata_fields: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """Names of the attributes that define the encoding."""

    derived_fields: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """Names of the derived attributes, such as views and caches."""

    mutually_exclusive_fields: tx.ClassVar[tx.Optional[tx.Tuple[str, ...]]] = (
        None
    )
    """Constructor arguments that may not be given together.

    A view is another name under which the stored parameter can be given, such
    as `matrix` for the `data` of an affine transformation. The default,
    `None`, makes the stored parameter and all of its views mutually exclusive,
    so the group is `data_fields` followed by `derived_fields`. This default
    applies only to a class with at most one data field. A class with several
    independent data fields has no group unless it declares one.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        mutex = self.mutually_exclusive_fields
        if mutex is None:
            # `data` and its views, such as `matrix`, are names for a single
            # parameter, so giving two of them gives the parameter twice. A
            # class with several data fields, such as a subspace
            # transformation or a bijection, takes all of them together, so
            # the default group does not apply to it.
            mutex = (
                self.data_fields + self.derived_fields
                if len(self.data_fields) <= 1
                else ()
            )
        _mutually_exclusive(self, arguments, mutex)
        self._set_derived_fields(arguments)
        # A concrete file format lists the data model before its format
        # class, so the format class comes after this class in the method
        # resolution order. It finishes the construction here, when it
        # defines `__post_init__`.
        post_init = getattr(super(), "__post_init__", None)
        if post_init is not None:
            post_init(arguments)

    def _set_derived_fields(self, arguments: tx.Any) -> None:
        for name in self.derived_fields:
            if (value := arguments.get(name)) is not None:
                setattr(self, name, value)

    # --- attributes ---------------------------------------------------

    # The input and output systems are stored under private names, so that
    # `replace()` carries over the systems given to the constructor rather
    # than the values that the `input` and `output` properties report. A
    # subclass that derives its systems would otherwise have them frozen into
    # fixed values by every `replace()`. The systems are keyword-only because
    # the position of a positional field follows the MRO. bagof does not carry
    # `KwOnly` over to a field that a subclass declares again, so a subclass
    # that gives a system a new default must write `KwOnly[...]` itself.

    _input: tx.Optional[CoordinateSystem] = None
    """Input coordinate system, which may be inferred from context if unset."""

    _output: tx.Optional[CoordinateSystem] = None
    """Output coordinate system, which may be inferred from context if unset.
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
        """Compute the transformation, evaluating what is still lazy.

        Computing a transformation composes and evaluates the parts that
        `mode` admits, for example by multiplying matrices, sampling fields or
        evaluating a lazy inverse. It also simplifies the result according to
        `simplify`. The base class raises `NotImplementedError`.

        Parameters
        ----------
        mode : ModeLike, default=True
            Kinds of transformation that may be composed or materialized, given
            as types, names or symbols (`"affine"`, `"Aff"`, `"rigid"`,
            `"SO(3)"`). The default, `True`, admits every kind. A concrete
            transformation, which holds its own parameters, has nothing to
            compose and ignores the mode unless `factor` is true.
        simplify : SimplifyLike, default="analytic"
            How much effort to spend on simplification. `"analytic"` (the
            default) decides from the types of the transformations alone,
            `"numeric"` (or `True`) also reads their parameter values, and
            `False`, `"none"` or `None` disables simplification.
        factor : bool, default=False
            Whether to rewrite the result as independent factors, one for each
            group of axes that transform together. Nothing is composed across
            groups, and `mode` still governs composition within a group.

        Returns
        -------
        Transformation
            The computed transformation.
        """
        # Every family computes differently, so the base raises rather than
        # returning `self`: a subclass that forgets `compute()` fails loudly,
        # as with `inverse()`.
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute()"
        )

    def simplify(
        self,
        policy: SimplifyLike = "analytic",
        *,
        compute: tx.Union[ModeLike, bool, None] = False,
    ) -> tx.Self:
        """Simplify the transformation, without composing it unless asked.

        `t.simplify(policy, compute=mode)` is
        `t.compute(mode, simplify=policy)`. By default nothing is composed: no
        matrix is multiplied, no field is sampled and no lazy inverse is
        evaluated. Each concrete transformation is only converted to the
        cheapest type that still represents it, as far as `policy` can tell.

        Parameters
        ----------
        policy : SimplifyLike, default="analytic"
            How hard to simplify, with the values accepted by the `simplify`
            argument of [`compute`][].
        compute : ModeLike or bool or None, default=False
            Kinds of transformation that may be composed, with the values
            accepted by the `mode` argument of [`compute`][]. The default,
            `False`, composes nothing, whereas `None` composes every kind.

        Returns
        -------
        Transformation
            The simplified transformation.
        """
        return self.compute(compute, simplify=policy)

    def factor(
        self,
        *,
        compute: tx.Union[ModeLike, bool, None] = False,
        simplify: SimplifyLike = False,
    ) -> "Transformation":
        """Rewrite the transformation as one factor per group of axes.

        The axes of a transformation often fall into groups that never
        interact, as with a diagonal matrix, whose axes each scale on their
        own. Factoring finds those groups and gives each one a factor that
        acts on its axes alone, which lets a resampler work group by group
        rather than over the whole grid. The transformation is factored as
        a chain of one element, so a diagonal affine splits into one factor
        per axis. See [`Sequence.factor`][] for the normal form that the
        pass produces and for what is left unfactored.

        Parameters
        ----------
        compute : ModeLike or bool or None, default=False
            Kinds of transformation that may be composed inside a group,
            with the values accepted by the `mode` argument of
            [`compute`][]. It decides nothing about what is factored, and
            the default composes nothing.
        simplify : SimplifyLike, default=False
            How hard to simplify the pieces of each group, with the values
            accepted by the `simplify` argument of [`compute`][].

        Returns
        -------
        Transformation
            The factored transformation, or `self` when it does not factor.
        """
        # The sequence owns the pass, and it never asks its elements to
        # factor themselves, so this call does not recurse.
        factored = nocycles.SEQUENCE([self]).factor(
            compute=compute, simplify=simplify
        )
        leaves = getattr(factored, "transformations", None) or ()
        if len(leaves) == 1 and leaves[0] is self:
            # The chain of one did not factor, so the wrapper is dropped
            # rather than handed back as a chain of one element.
            return self
        return factored

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        """Return the inverse of the transformation.

        Some classes return a lazy inverse, which is evaluated only when it is
        computed. A transformation composed with its lazy inverse can then
        cancel out without any arithmetic. `~t` is `t.inverse()`. The base
        class raises `NotImplementedError`.

        Parameters
        ----------
        compute : bool, default=False
            Whether to compute the inverse now rather than lazily.
        **kwargs : Any
            Arguments passed to [`compute`][] when `compute` is true.

        Returns
        -------
        Transformation
            The inverse.
        """
        raise NotImplementedError("Transformation.inverse()")

    def square(self, compute: bool = False, **kwargs) -> "Transformation":
        """Return the transformation composed with itself.

        The square is the [`Sequence`][] `[self, self]`, which is composed when
        it is computed. It is defined only for a transformation that maps a
        space to itself.

        Parameters
        ----------
        compute : bool, default=False
            Whether to compute the square now rather than return the sequence.
        **kwargs : Any
            Arguments passed to [`compute`][] when `compute` is true.

        Returns
        -------
        Transformation
            The square.

        Raises
        ------
        DomainError
            If the transformation does not map a space to itself.
        """
        require_endomorphism(self, "square")
        obj = nocycles.SEQUENCE([self, self])
        return obj.compute(**kwargs) if compute else obj

    def sqrt(self, compute: bool = False, **kwargs) -> "Transformation":
        """Return the principal square root of the transformation.

        The square root `S` of `T` is the transformation such that `S @ S` is
        `T`. The principal root is the one whose linear part has its
        eigenvalues in the open right half-plane. It is unique and of the same
        kind as `T`: the root of a rotation is a rotation, and likewise for a
        translation, a scaling or an affine transformation. A permutation is
        refused, although its linear form `p.to(Linear)` may have a root.

        The root is a lazy [`Sqrt`][brainhops.datamodel.transformations.Sqrt]
        wrapper, resolved when it is applied, computed or converted. An
        identity is returned as it is, and a [`Sequence`][] is reduced first.
        The base class raises `NotImplementedError`.

        Parameters
        ----------
        compute : bool, default=False
            Whether to compute the root now rather than lazily.
        **kwargs : Any
            Arguments passed to [`compute`][] when `compute` is true.

        Returns
        -------
        Transformation
            The principal square root.

        Raises
        ------
        DomainError
            If the transformation does not map a space to itself or, when the
            root is resolved, if its linear part has an eigenvalue on the
            closed negative real axis, as for a reflection or a half turn.
        NotImplementedError
            If the kind of transformation has no square root here. A
            displacement field has one only as a stationary velocity field
            (`log=True`).
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
        """Convert the transformation to another type or encoding.

        The conversion takes one of three forms:

        * between types: `linear.to(Affine)`                       ; or
        * within a type: `displacement.to(store="coefficients")`   ; or
        * both: `coords.to(DisplacementField, store="values")`     .

        Parameters
        ----------
        cls : type of Transformation, optional
            Type to convert to. By default, the type is normally kept.
        lossy : bool, default=False
            Whether to allow a conversion that loses information.
        error : type of Exception or Exception or bool, default=True
            What to do when the conversion fails. `True` (the default) raises
            the original error again, an exception or exception type is raised
            from it, and any other value is returned in place of the result.
        **kwargs : Any
            Attributes of the converted transformation. `store="coefficients"`
            re-encodes the data of a displacement field and keeps its map. A
            view such as `field=` or `matrix=` sets the map, stored in the
            encoding of the result, whereas `data=` is stored as given.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        cls = cls or self._target_class(kwargs)
        try:
            return convert(self, cls, **kwargs)
        except LossyConversionError as e:
            if lossy:
                # The caller accepts a lossy result, which the exception
                # carries.
                return e.result
            failure = e
        except ConversionError as e:
            failure = e
        # The conversion failed and no loss was accepted, so `error` decides
        # what happens.
        if error is True:
            raise failure
        if is_instance_or_subclass(error, Exception):
            raise error from failure
        return error

    def _target_class(
        self, kwargs: tx.Dict[str, tx.Any]
    ) -> tx.Type["Transformation"]:
        """Return the type that `to()` converts to when the caller names none.

        The type is normally the class of the transformation. Some classes are
        chosen by a flag, as `log=True` chooses the tangent class of a family,
        and such a class returns the class that the new value of the flag
        selects. For example, `log=False` on a tangent class asks for the
        plain class. Because the type is chosen here rather than in the
        converter, `convert()` can check that it returns an instance of it.
        """
        return type(self)

    # --- kind checks --------------------------------------------------

    def is_kind(
        self, kind: tx.Type[kinds.Kind], compute: bool = False
    ) -> bool:
        """Return whether the transformation is of a given kind.

        A transformation is of a kind when it is an instance of the kind,
        possibly a virtual one, or when its parameters are compatible with it.

        Parameters
        ----------
        kind : type of Kind
            Kind to test, from [`kinds`][].
        compute : bool, default=False
            Whether to read the values of the parameters rather than only the
            structure of the types.

        Returns
        -------
        bool
            Whether the transformation is of the kind.
        """
        return is_kind(self, kind, compute)

    def is_identity(self, compute: bool = False) -> bool:
        """Return whether the transformation is the identity.

        A transformation is the identity when its parameters are unset or when
        it is an [`Identity`][brainhops.datamodel.transformations.Identity].
        With `compute=True`, the parameters of the matrix, vector and
        displacement types are inspected too.

        A grid of coordinates
        ([`CartesianField`][brainhops.datamodel.transformations.CartesianField])
        is the identity over itself. It is recognised as such with
        `compute=True`, but with `compute=False` only if its `shape` is unset,
        since a shape is a parameter. In neither case are the coordinates of
        the grid built. A grid that is the identity still defines where a
        sequence is sampled, so simplifying a sequence removes the grid only
        when it lies between two other transformations.
        """
        return self.is_kind(kinds.Identity, compute)

    def is_translation(self, compute: bool = False) -> bool:
        """Return whether the transformation is a translation.

        A transformation is a translation when it is an instance of
        [`kinds.Translation`][brainhops.datamodel.kinds.Translation] or the
        identity. With `compute=True`, an affine matrix is checked for an
        identity linear part and a displacement field for a constant value.
        """
        return self.is_kind(kinds.Translation, compute)

    def is_scaling(self, compute: bool = False) -> bool:
        """Return whether the transformation is a scaling.

        A transformation is a scaling when it is an instance of
        [`kinds.Diagonal`][brainhops.datamodel.kinds.Diagonal] or the identity.
        With `compute=True`, a linear or affine matrix is checked to be
        diagonal.
        """
        return self.is_kind(kinds.Diagonal, compute)

    def is_permutation(self, compute: bool = False) -> bool:
        """Return whether the transformation is a permutation.

        A transformation is a permutation when it is an instance of
        [`kinds.Permutation`][brainhops.datamodel.kinds.Permutation] or the
        identity. With `compute=True`, a linear or affine matrix is checked for
        a single one in each row and column.
        """
        return self.is_kind(kinds.Permutation, compute)

    def is_rotation(self, compute: bool = False) -> bool:
        """Return whether the transformation is a rotation.

        A transformation is a rotation when it is an instance of
        [`kinds.SpecialOrthogonal`][brainhops.datamodel.kinds.SpecialOrthogonal]
        or the identity. With `compute=True`, a linear or affine matrix is
        checked to be orthogonal with a positive determinant.
        """
        return self.is_kind(kinds.SpecialOrthogonal, compute)

    def is_linear(self, compute: bool = False) -> bool:
        """Return whether the transformation is linear.

        A transformation is linear when it is an instance of
        [`kinds.Linear`][brainhops.datamodel.kinds.Linear] or the identity.
        With `compute=True`, an affine matrix is checked for a zero
        translation.
        """
        return self.is_kind(kinds.Linear, compute)

    def is_affine(self, compute: bool = False) -> bool:
        """Return whether the transformation is affine.

        A transformation is affine when it is an instance of
        [`kinds.Affine`][brainhops.datamodel.kinds.Affine] or the identity.
        """
        return self.is_kind(kinds.Affine, compute)

    # --- operators ----------------------------------------------------

    @tx.overload
    def __call__(
        self, x: "CoordinatesField", compute: tx.Literal[True]
    ) -> "CoordinatesField":
        """Apply the transformation to coordinates and compute the result."""
        ...

    @tx.overload
    def __call__(
        self, x: "CoordinatesField", compute: tx.Literal[False]
    ) -> "Sequence":
        """Apply the transformation to coordinates lazily.

        The result is a sequence of the two, computed later.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "Transformation", compute: tx.Literal[True]
    ) -> "Transformation":
        """Compose `x` then `self`, and compute the result.

        The result maps the input of `x` to the output of `self`. If the output
        of `x` does not match the input of `self`, an adaptor is sought, and an
        error is raised if none is found.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "Transformation", compute: tx.Literal[False]
    ) -> "Sequence":
        """Compose `x` then `self` lazily, as a sequence of the two."""
        ...

    @tx.overload
    def __call__(
        self, x: "CoordinatesField", compute: ModeLike
    ) -> "CoordinatesField":
        """Apply the transformation to coordinates and compute the result.

        `compute` is a compose mode, which selects the kinds that are composed.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "Transformation", compute: ModeLike
    ) -> "Transformation":
        """Compose `x` then `self`, and compute the result under a compose
        mode.

        `True` composes every kind, a mode composes the kinds it admits, and
        `False` returns the sequence uncomputed.
        """
        ...

    def __call__(self, x, compute: bool = False) -> "Transformation":
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
        # Defer to a right operand that composes itself, such as a geometry
        # that stays a geometry. Python gives the reflected method precedence
        # only to a proper subclass, and `Sequence` is a factory for
        # `MutableSequence`, so a sibling class would never be asked.
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
    """Refuse a constructor call that gives one parameter under two names.

    `names` holds the stored parameter and the views under which it can also
    be given, such as `data` and `matrix`. Because a view fills `data`, no two
    of these names can be given together. The check also applies to
    `replace()`, which carries `data` over, so `replace(t, matrix=m)` is
    refused as well.

    Raises
    ------
    TypeError
        If more than one of `names` is given a value other than `None`.
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
