import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops._core.properties import smartproperty
from brainhops._core.typing import is_instance_or_subclass
from brainhops.datamodel import kinds
from brainhops.datamodel._backends import _BackendsMixin
from brainhops.datamodel.base import DataModelBase, _ComparesByIdMixin
from brainhops.datamodel.metadata import Metadata
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import ConversionError, LossyConversionError

from . import nocycles
from .compute.check import is_kind
from .compute.convert import convert
from .compute.simplify import SimplifyLike
from .compute.utils import require_endomorphism
from .modes import ModeLike

if tx.TYPE_CHECKING:
    from .concrete import CoordinateField
    from .restriction import AxisRefs
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
            `"numeric"` (or `True`) also reads the parameter values of
            matrix transformations, `{None: "numeric"}` reads the values
            of every transformation, including fields, and `False`,
            `"none"` or `None` disables simplification.
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
        policy: SimplifyLike = "analytic",
        *,
        compute: tx.Union[ModeLike, bool, None] = False,
    ) -> "Transformation":
        """Rewrite the transformation as one factor per group of axes.

        The axes of a transformation often fall into groups that never
        interact, as with a diagonal matrix, whose axes each scale on
        their own. Factoring finds those groups and gives each one a
        factor that acts on its axes alone, which lets a resampler work
        group by group rather than over the whole grid. The
        transformation is factored as a chain of one element, so that,
        under the numeric policy, a diagonal affine splits into one
        factor per axis. See [`Sequence.factor`][] for the normal form
        that the pass produces and for what is left unfactored.

        Parameters
        ----------
        policy : SimplifyLike, default="analytic"
            How closely the transformation may be read, with the values
            and the meaning of the policy of [`simplify`][]. Under the
            analytic policy, a matrix couples all its axes, so a
            diagonal affine splits only under the numeric policy.
        compute : ModeLike or bool or None, default=False
            Kinds of transformation that may be composed inside a group,
            with the values accepted by the `mode` argument of
            [`compute`][]. It decides nothing about what is factored, and
            the default composes nothing.

        Returns
        -------
        Transformation
            The factored transformation, or `self` when it does not factor.
        """
        # The sequence owns the pass, and it never asks its elements to
        # factor themselves, so this call does not recurse.
        factored = nocycles.SEQUENCE([self]).factor(policy, compute=compute)
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

    def restrict(
        self,
        input_axes: "AxisRefs",
        output_axes: "AxisRefs",
        at: tx.Optional[tx.Mapping[tx.Any, float]] = None,
    ) -> "Transformation":
        """Return the transformation restricted to some of its axes.

        A restriction keeps a part of a transformation, such as the
        spatial part of a map of space and time, or the map of one
        volume of a time series. It is lazy, like [`inverse`][]. The
        result is a [`Sequence`][] of three elements: a
        [`Projection`][brainhops.datamodel.transformations.Projection]
        that creates the inputs that `input_axes` leaves out, the
        transformation itself, and a projection that drops the outputs
        that `output_axes` leaves out. Nothing is computed until
        [`compute`][] is called, and [`simplify`][] then removes the
        axes that are not needed. When both sides keep every axis, the
        transformation itself is returned.

        A removed input takes the value that `at` gives it, which
        selects the section of the transformation at that value. A
        removed input that `at` does not name is free, which means that
        the kept outputs must not read it. [`compute`][] raises a
        `RestrictionError` when a kept output reads a free input,
        whereas `restrict` and [`simplify`][] never raise for that
        reason. With one affine per volume of a time series, for
        example, the spatial outputs read the time axis, so the
        restriction to the spatial axes needs a value of time, which
        selects one volume.

        Each side is a list of references, or a single reference. A
        reference is a position, the name of an axis, or any other query
        of [`AxisQuery`][brainhops.datamodel.axes.AxisQuery], such as an
        axis class. It is resolved against the input or the output
        system of the transformation, as
        [`AxisSequence.restrict`][brainhops.datamodel.axes.AxisSequence.restrict]
        resolves it, so a query never matches the unknown axes that the
        `...` of an open system stands for. A name must designate
        exactly one axis, and any other query must designate at least
        one. Each side is a set of axes, so the order of the references
        does not matter: the kept axes stay in the order that they have
        in the transformation, and an axis must not be designated twice.
        Positions also work on a transformation without systems,
        or with open systems, provided that its numbers of axes are
        known, for example from the shape of its matrix. The systems of
        the result are the systems of the transformation restricted to
        the kept axes.

        A subspace transformation, an instance of
        [`SubspaceTransformation`][brainhops.datamodel.transformations.SubspaceTransformation],
        often does not know how many axes it passes through, which a
        restriction needs. When it is restricted to the axes that it
        acts on, however, the result is the transformation that it
        wraps, whatever that number is, and this result is returned. A
        subspace that lists its axes in another order than the full
        space gives the wrapped transformation between two permutations,
        which put its axes in the order of the full space. A geometry,
        an instance of
        [`Geometry`][brainhops.datamodel.geometry.Geometry], restricts
        its grid as well when it can, as its own method describes.

        !!! example
            ```pycon
            >>> x, y = SpaceAxis(name="x"), SpaceAxis(name="y")
            >>> xyt = CoordinateSystem(axes=[x, y, TimeAxis(name="t")])
            >>> matrix = [[2, 0, 0.5, 1], [0, 3, 0, 2], [0, 0, 1, 0]]
            >>> t = Affine(matrix=matrix, input=xyt, output=xyt)
            >>> part = t.restrict(SpaceAxis, SpaceAxis, at={"t": 4})
            >>> part.compute().matrix
            array([[2., 0., 3.],
                   [0., 3., 2.]])
            ```

        Parameters
        ----------
        input_axes : AxisQuery or iterable of (int or AxisQuery)
            Input axes to keep, given by position, by name or by query.
        output_axes : AxisQuery or iterable of (int or AxisQuery)
            Output axes to keep, given by position, by name or by query.
        at : mapping of (int or AxisQuery) to float, optional
            Values of some of the removed inputs. Each key designates
            input axes as an item of `input_axes` does, and every axis
            that it designates takes its value. NaN makes an input free,
            as leaving it out of `at` does.

        Returns
        -------
        Transformation
            The lazy restriction, or the transformation itself when both
            sides keep every axis.

        Raises
        ------
        RestrictionError
            If the number of input or output axes of the transformation
            is unknown, because the axes to remove are then unknown.
        KeyError
            If no axis has a name that a reference gives.
        IndexError
            If a position lies outside the axes of the transformation.
        ValueError
            If a query matches no axis, a name is shared by several
            axes, an axis is designated twice, or `at` gives a value to
            an input that is kept.
        TypeError
            If a reference is neither a position nor a query, or if `at`
            is not a mapping of real numbers.
        """
        # The module imports the projection and the sequence classes,
        # which import this one.
        from .restriction import restrict

        return restrict(self, input_axes, output_axes, at)

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
        lossy: bool = False,
        error: tx.Union[tx.Type[Exception], Exception, bool] = True,
        copy: bool = False,
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
            The converted transformation.
        """
        cls = cls or self._target_class(kwargs)
        try:
            return _copied(self, convert(self, cls, **kwargs), copy)
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


def _copied(
    original: Transformation, result: Transformation, copy: bool
) -> Transformation:
    """Return `result`, or a copy of it when it is `original` and `copy`."""
    if copy and result is original:
        return type(original).from_instance(original)
    return result
