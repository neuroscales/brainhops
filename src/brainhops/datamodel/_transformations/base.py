import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops._core.deprecation import pop_renamed_keyword
from brainhops._core.properties import smartproperty
from brainhops.datamodel import kinds
from brainhops.datamodel._backends import _BackendsMixin
from brainhops.datamodel.base import DataModelBase, _ComparesByIdMixin
from brainhops.datamodel.metadata import Metadata
from brainhops.datamodel.metadata._report import call_under_policy
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import (
    UNSET,
    ConversionError,
    ErrorAction,
    ErrorPolicy,
)

from . import nocycles
from .compute.check import KindLike, is_kind, normalize_kind
from .compute.convert import convert
from .compute.simplify import SimplifyLike
from .compute.utils import require_endomorphism
from .modes import ModeLike

if tx.TYPE_CHECKING:
    from .compute.filter import MatchLike

_KIND_CLASSES: tx.Dict[str, tx.Tuple[tx.Type[kinds.Kind], str]] = {
    "translation": (kinds.Translation, "Translation"),
    "scaling": (kinds.Diagonal, "Scaling"),
    "permutation": (kinds.Permutation, "Permutation"),
    "rotation": (kinds.SpecialOrthogonal, "Rotation"),
    "linear": (kinds.Linear, "Linear"),
    "affine": (kinds.Affine, "Affine"),
}
"""Kinds that the `is_*` methods test and the `to_*` methods produce.

Each entry maps the word used in the method names to the kind that the
`is_*` method tests and to the name of the concrete class that the
`to_*` method returns. The class is named rather than imported because
the module that defines it imports this module.
"""

if tx.TYPE_CHECKING:
    from .concrete import CoordinateField
    from .sequence import Sequence


@kinds.Transformation.register
@nocycles.register_transformation
class Transformation(
    _BackendsMixin,
    _ComparesByIdMixin,
    DataModelBase,
    reverse=True,
    eq=False,
    kw_only=True,
):
    """A map of coordinates from an input system to an output system.

    In the functional syntax, `t(x)` applies `t` to `x`, and `t2(t1)` composes
    two transformations so that `t1` is applied first. The matrix syntax
    borrows `@` from linear and affine maps, which are matrices: `t @ x`
    applies `t`, and `t2 @ t1` is the same composition, with the input space on
    the right and the output space on the left. Applying a transformation to
    coordinates is also a composition, since a [`CoordinateField`][] is a
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

    encoding_fields: tx.ClassVar[tx.Tuple[str, ...]] = ()
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
    # the position of a positional field follows the MRO. A subclass that
    # declares a system again, to give it a new default, inherits the
    # `kw_only=True` option of this class, so the system stays keyword-only.
    # bagof takes the option from the first base in the MRO, however, even
    # when that base only keeps the default, so a subclass whose first base
    # is a file reader that is not keyword-only must write `KwOnly[...]`
    # itself.

    _input: tx.Optional[CoordinateSystem] = None
    """Input coordinate system, which may be inferred from context if unset."""

    _output: tx.Optional[CoordinateSystem] = None
    """Output coordinate system, which may be inferred from context if unset.
    """

    input = smartproperty("input")
    output = smartproperty("output")

    metadata: NoRepr[tx.Optional[Metadata]] = None
    """What describes the transformation beyond its parameters, or `None`.

    The metadata holds the common vocabulary of
    [`Metadata`][brainhops.datamodel.metadata.Metadata], such as a
    description or the images of a registration. A transformation built
    from parameters has none unless it is given some. A file format takes
    the field over to hold the metadata of its own class, which also
    holds the header of the file, and converts the metadata it is given
    into that class. The field is keyword-only and is left out of the
    representation of the transformation.
    """

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

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy an instance of this class, or convert another transformation.

        A transformation that is not an instance of this class is
        converted exactly as `other.to(cls, copy=True)` converts it, with
        the converters of its family, which check what this class
        promises, such as the orthogonal matrix of a rotation. An
        instance of this class is copied as
        [`DataModelBase.from_instance`][brainhops.datamodel.base.DataModelBase.from_instance]
        copies it. For a file format, an instance of its data model, such
        as the `Affine` of an LTA file, is copied as well, because the
        converters of the format build the format that way.

        Parameters
        ----------
        other : DataModelBase
            The transformation to copy or convert.
        *args : Any
            Constructor arguments of a copy. A conversion takes keyword
            arguments only.
        **kwargs : Any
            Fields of the result, which take precedence over those of
            `other`.

        Returns
        -------
        Self
            The new transformation.

        Raises
        ------
        TypeError
            If positional arguments accompany a conversion.
        ConversionError
            If no converter turns `other` into this class without loss.
        """
        if cls._converts(other):
            if args:
                raise TypeError(
                    f"{cls.__name__}.from_instance() converts a "
                    f"{type(other).__name__} with keyword arguments only, "
                    f"but was given {len(args)} positional argument(s)."
                )
            return other.to(cls, copy=True, **kwargs)
        return super().from_instance(other, *args, **kwargs)

    @classmethod
    def _converts(cls, other: tx.Any) -> bool:
        """Return whether `from_instance` converts `other` rather than copy it.

        A transformation is converted unless it is an instance of the data
        model of this class, which is the class itself for a data model and
        the data model that a file format stores.
        """
        return isinstance(other, Transformation) and not isinstance(
            other, cls._data_model()
        )

    def to(
        self,
        cls: tx.Optional[tx.Type[tx.Self]] = None,
        *,
        lossy: ErrorAction = None,
        on_error: ErrorPolicy = UNSET,
        copy: bool = False,
        **kwargs,
    ) -> tx.Self:
        """Convert the transformation to another type or encoding.

        The conversion takes one of three forms:

        * between types: `linear.to(Affine)`                       ; or
        * within a type: `displacement.to(store="coefficients")`   ; or
        * both: `coords.to(DisplacementField, store="values")`     .

        A conversion that would lose information raises a
        [`LossyConversionError`][brainhops.errors.LossyConversionError],
        which carries the lossy result. A conversion that cannot be
        made raises a [`ConversionError`][]. An error policy decides
        what happens to these errors, and to any other error of the
        conversion (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]).
        The policy that the call applies is
        `{LossyConversionError: lossy, **on_error}`, so that
        `lossy=True` returns the lossy result. A loss of metadata, in
        the conversions of metadata that the conversion triggers,
        follows the same policy, because a
        [`MetadataLossError`][brainhops.datamodel.metadata.MetadataLossError]
        is a lossy conversion error.

        !!! warning "Deprecated `error=`"
            The `error=` keyword is a deprecated alias of `on_error=`.
            The value `True` reads as `"raise"`, and any value `v`
            applies to a [`ConversionError`][] only, as
            `on_error={ConversionError: v}`.

        Parameters
        ----------
        cls : type of Transformation, optional
            Type to convert to. By default, the type is normally kept.
        lossy : ErrorAction, optional
            What to do with a conversion that loses information. `True`
            or `"ignore"` returns the lossy result, `"warn"` warns and
            returns it, and a report records the loss and returns it.
            By default, the loss is raised, unless `on_error` has an
            entry for it.
        on_error : ErrorPolicy, optional
            What to do when the conversion fails. One action applies to
            every error, and a mapping chooses an action by the class
            of the error. An entry for a loss in `on_error` takes
            precedence over `lossy`. By default, every error is raised.
        copy : bool, default=False
            Whether to return a new object when the conversion has nothing
            to change. By default, a transformation converted to its own
            class without any field is returned itself.
        **kwargs : Any
            Attributes of the converted transformation. `store="coefficients"`
            re-encodes the data of a displacement field and keeps its map. A
            view such as `field=` or `matrix=` sets the map, stored in the
            encoding of the result, whereas `data=` is stored as given.

        Returns
        -------
        Transformation
            The converted transformation, or what the policy returns in
            its place.

        Examples
        --------
        ```python
        # Accept a loss, and return None if no converter exists.
        t.to(Rotation, lossy=True, on_error={ConversionError: None})
        ```
        """
        on_error = _deprecated_error(kwargs, on_error, type(self))

        def convert_to() -> Transformation:
            target = cls or self._target_class(kwargs)
            return _copied(self, convert(self, target, **kwargs), copy)

        return call_under_policy(convert_to, on_error, lossy=lossy)

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
        return self.is_kind(_KIND_CLASSES["translation"][0], compute)

    def is_scaling(self, compute: bool = False) -> bool:
        """Return whether the transformation is a scaling.

        A transformation is a scaling when it is an instance of
        [`kinds.Diagonal`][brainhops.datamodel.kinds.Diagonal] or the identity.
        With `compute=True`, a linear or affine matrix is checked to be
        diagonal.
        """
        return self.is_kind(_KIND_CLASSES["scaling"][0], compute)

    def is_permutation(self, compute: bool = False) -> bool:
        """Return whether the transformation is a permutation.

        A transformation is a permutation when it is an instance of
        [`kinds.Permutation`][brainhops.datamodel.kinds.Permutation] or the
        identity. With `compute=True`, a linear or affine matrix is checked for
        a single one in each row and column.
        """
        return self.is_kind(_KIND_CLASSES["permutation"][0], compute)

    def is_rotation(self, compute: bool = False) -> bool:
        """Return whether the transformation is a rotation.

        A transformation is a rotation when it is an instance of
        [`kinds.SpecialOrthogonal`][brainhops.datamodel.kinds.SpecialOrthogonal]
        or the identity. With `compute=True`, a linear or affine matrix is
        checked to be orthogonal with a positive determinant.
        """
        return self.is_kind(_KIND_CLASSES["rotation"][0], compute)

    def is_linear(self, compute: bool = False) -> bool:
        """Return whether the transformation is linear.

        A transformation is linear when it is an instance of
        [`kinds.Linear`][brainhops.datamodel.kinds.Linear] or the identity.
        With `compute=True`, an affine matrix is checked for a zero
        translation.
        """
        return self.is_kind(_KIND_CLASSES["linear"][0], compute)

    def is_affine(self, compute: bool = False) -> bool:
        """Return whether the transformation is affine.

        A transformation is affine when it is an instance of
        [`kinds.Affine`][brainhops.datamodel.kinds.Affine] or the identity.
        """
        return self.is_kind(_KIND_CLASSES["affine"][0], compute)

    # --- filters ------------------------------------------------------

    def without(self, match: "MatchLike") -> "Transformation":
        """Return the transformation with some parts replaced by the identity.

        A caller uses this method to ignore a class of transformations,
        for example the displacement fields of a chain whose affine part
        it needs. Every part that the matcher selects becomes an
        [`Identity`][brainhops.datamodel.transformations.Identity] with
        the same input and output coordinate systems, so that the
        elements of a chain still connect and no bridge is guessed
        between them. The identities disappear when the result is
        computed.

        The transformation is visited from the outside in. A selected
        node is replaced as a whole, and the method looks inside a
        sequence or an inverse only when the node itself is not
        selected. The inverse of a replaced part is replaced as well.

        Parameters
        ----------
        match : MatchLike
            Parts to replace. A matcher is a kind or the name of a kind,
            such as `"affine"`, a concrete class of transformation, a
            predicate that takes a transformation and returns a boolean,
            or a sequence of these values, which selects a part when any
            of its items selects it.

        Returns
        -------
        Transformation
            The filtered transformation, or the transformation itself
            when no part is replaced.

        Raises
        ------
        ValueError
            If a part to replace changes the number of axes.
        """
        from .compute.filter import replace_parts

        return replace_parts(self, match, keep=False)

    def with_only(self, match: "MatchLike") -> "Transformation":
        """Return the transformation with only some parts kept.

        This method is the complement of
        [`without`][brainhops.datamodel.transformations.Transformation.without]:
        every part that the matcher does not select is replaced by an
        identity with the same coordinate systems. Because a kind can
        only describe the transformations that have a property,
        `with_only("affine")` is the way to remove everything that is not
        affine.

        A node that the matcher selects is kept as a whole, so a chain
        that is affine as a whole is not split. The method looks inside
        a sequence or an inverse only when the node itself is not
        selected, and replaces any other node that is not selected.

        Parameters
        ----------
        match : MatchLike
            Parts to keep, in the same forms as for `without`.

        Returns
        -------
        Transformation
            The filtered transformation, or the transformation itself
            when no part is replaced.

        Raises
        ------
        ValueError
            If a part to replace changes the number of axes.
        """
        from .compute.filter import replace_parts

        return replace_parts(self, match, keep=True)

    # --- conversions to a kind ----------------------------------------

    def to_kind(
        self,
        kind: KindLike,
        *,
        ignore: tx.Union["MatchLike", bool, None] = None,
        **kwargs,
    ) -> "Transformation":
        """Convert the transformation to the class that represents a kind.

        A caller uses this method to obtain, for example, the affine
        matrix of a chain. The transformation is first filtered with
        `ignore`, then computed, because a chain must be composed before
        it can be converted, and the result is converted with
        [`to`][brainhops.datamodel.transformations.Transformation.to].

        | Kind            | Kind tested by `is_*`     | Concrete class |
        | --------------- | ------------------------- | -------------- |
        | `"translation"` | `kinds.Translation`       | `Translation`  |
        | `"scaling"`     | `kinds.Diagonal`          | `Scaling`      |
        | `"permutation"` | `kinds.Permutation`       | `Permutation`  |
        | `"rotation"`    | `kinds.SpecialOrthogonal` | `Rotation`     |
        | `"linear"`      | `kinds.Linear`            | `Linear`       |
        | `"affine"`      | `kinds.Affine`            | `Affine`       |

        Parameters
        ----------
        kind : KindLike
            Target kind, as a word of the table above or as one of the
            kinds of the table.
        ignore : MatchLike or bool, optional
            Parts to replace by the identity before the conversion, as
            in `without`. The value `True` means every part that is not
            of the target kind, as in `with_only(kind)`.
        **kwargs : Any
            Keyword arguments of `to`, such as `lossy=` and `error=`.

        Returns
        -------
        Transformation
            The converted transformation, or the value that `to`
            returns when the conversion fails and is not raised.

        Raises
        ------
        ValueError
            If the kind has no concrete class.
        """
        from . import concrete

        kind_class, class_name = _kind_entry(kind)
        xform = self
        if ignore is True:
            xform = xform.with_only(kind_class)
        elif ignore is not None and ignore is not False:
            xform = xform.without(ignore)
        target = getattr(concrete, class_name)
        return xform.compute().to(target, **kwargs)

    def to_translation(
        self, *, ignore: tx.Union["MatchLike", bool, None] = None, **kwargs
    ) -> "Transformation":
        """Convert the transformation to a translation.

        See `to_kind` for how the transformation is filtered and
        converted.

        Parameters
        ----------
        ignore : MatchLike or bool, optional
            Parts to replace by the identity before the conversion.
        **kwargs : Any
            Keyword arguments of `to`.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        return self.to_kind("translation", ignore=ignore, **kwargs)

    def to_scaling(
        self, *, ignore: tx.Union["MatchLike", bool, None] = None, **kwargs
    ) -> "Transformation":
        """Convert the transformation to a scaling.

        See `to_kind` for how the transformation is filtered and
        converted.

        Parameters
        ----------
        ignore : MatchLike or bool, optional
            Parts to replace by the identity before the conversion.
        **kwargs : Any
            Keyword arguments of `to`.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        return self.to_kind("scaling", ignore=ignore, **kwargs)

    def to_permutation(
        self, *, ignore: tx.Union["MatchLike", bool, None] = None, **kwargs
    ) -> "Transformation":
        """Convert the transformation to a permutation.

        See `to_kind` for how the transformation is filtered and
        converted.

        Parameters
        ----------
        ignore : MatchLike or bool, optional
            Parts to replace by the identity before the conversion.
        **kwargs : Any
            Keyword arguments of `to`.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        return self.to_kind("permutation", ignore=ignore, **kwargs)

    def to_rotation(
        self, *, ignore: tx.Union["MatchLike", bool, None] = None, **kwargs
    ) -> "Transformation":
        """Convert the transformation to a rotation.

        See `to_kind` for how the transformation is filtered and
        converted.

        Parameters
        ----------
        ignore : MatchLike or bool, optional
            Parts to replace by the identity before the conversion.
        **kwargs : Any
            Keyword arguments of `to`.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        return self.to_kind("rotation", ignore=ignore, **kwargs)

    def to_linear(
        self, *, ignore: tx.Union["MatchLike", bool, None] = None, **kwargs
    ) -> "Transformation":
        """Convert the transformation to a linear map.

        See `to_kind` for how the transformation is filtered and
        converted.

        Parameters
        ----------
        ignore : MatchLike or bool, optional
            Parts to replace by the identity before the conversion.
        **kwargs : Any
            Keyword arguments of `to`.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        return self.to_kind("linear", ignore=ignore, **kwargs)

    def to_affine(
        self, *, ignore: tx.Union["MatchLike", bool, None] = None, **kwargs
    ) -> "Transformation":
        """Convert the transformation to an affine map.

        See `to_kind` for how the transformation is filtered and
        converted. `to_affine(error=None)` returns `None` when the
        transformation is not affine, and
        `to_affine(ignore=True, error=None)` returns the affine part of
        a chain whose other parts, such as fields, are ignored.

        Parameters
        ----------
        ignore : MatchLike or bool, optional
            Parts to replace by the identity before the conversion.
        **kwargs : Any
            Keyword arguments of `to`.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        return self.to_kind("affine", ignore=ignore, **kwargs)

    # --- operators ----------------------------------------------------

    @tx.overload
    def __call__(
        self, x: "CoordinateField", compute: tx.Literal[True]
    ) -> "CoordinateField":
        """Apply the transformation to coordinates and compute the result."""
        ...

    @tx.overload
    def __call__(
        self, x: "CoordinateField", compute: tx.Literal[False]
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
        self, x: "CoordinateField", compute: ModeLike
    ) -> "CoordinateField":
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


def _kind_entry(kind: KindLike) -> tx.Tuple[tx.Type[kinds.Kind], str]:
    """Return the kind and the class name that `to_kind` converts to.

    Raises
    ------
    ValueError
        If the kind is not one of the kinds in `_KIND_CLASSES`.
    """
    if isinstance(kind, str) and kind.lower() in _KIND_CLASSES:
        return _KIND_CLASSES[kind.lower()]
    resolved = normalize_kind(kind)
    for entry in _KIND_CLASSES.values():
        if entry[0] is resolved:
            return entry
    raise ValueError(
        f"No concrete class represents the kind {kind!r}. The kinds that "
        f"can be converted to are {', '.join(_KIND_CLASSES)}."
    )


def _copied(
    original: Transformation, result: Transformation, copy: bool
) -> Transformation:
    """Return `result`, or a copy of it when it is `original` and `copy`."""
    if copy and result is original:
        return type(original).from_instance(original)
    return result


def _deprecated_error(
    kwargs: tx.Dict[str, tx.Any], on_error: ErrorPolicy, cls: type
) -> ErrorPolicy:
    """Return the policy of `to()`, read from `error=` as well.

    `error=` was the keyword of `to()` before `on_error=`. It is removed
    from `kwargs` with a `DeprecationWarning`, and read with its old
    meaning: `True` raises, and any other value applies to a
    `ConversionError` only. The stack level of the warning points at
    the caller of `to()`.
    """
    if "error" in kwargs and on_error is not UNSET:
        raise TypeError(
            f"{cls.__name__}.to() got both error= and on_error=; pass "
            f"on_error= only."
        )
    error = pop_renamed_keyword(
        kwargs, "error", "on_error", f"{cls.__name__}.to", UNSET, stacklevel=4
    )
    if error is UNSET:
        return on_error
    return {ConversionError: "raise" if error is True else error}
