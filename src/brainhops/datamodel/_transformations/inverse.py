# stdlib
from numbers import Integral, Real

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops._core.typing import ArrayProtocol, npmatrix, npvector
from brainhops._ext.invfield import inverse as inverse_disp

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder

# internals
from .base import Transformation
from .concrete import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
)
from .meta import _simplify_inner
from .modes import (
    ModeLike,
    SimplifyLike,
    _lower_modes,
    _lower_simplify,
    _mode_admits,
)
from .registries import INVERSE_CACHE, INVERSE_WRAPPERS, register_inverse

# typing
TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)


class Inverse(Transformation, tx.Generic[TRANSFORMATION]):
    """The inverse of a transformation, resolved on demand.

    An `Inverse` holds a forward transformation and represents its
    inverse. The inverse is not computed when the wrapper is built. It is
    computed only when the wrapper is applied, computed, or converted to a
    concrete type. Placed next to its forward transformation in a
    [`Sequence`][], the two cancel to the identity, and no inverse is ever
    computed.

    Constructing `Inverse(forward=t)` represents the inverse of any
    transformation `t`. Each family of transformations also has its own
    typed inverse, such as [`InverseAffine`][] or
    [`InverseDisplacementField`][], which a transformation returns from its
    `inverse()` method. A typed inverse remains an instance of the family
    it inverts, so composition and the kind checks treat it exactly like a
    forward transformation of that family.
    """

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[TRANSFORMATION],
        tx.Doc("The forward transformation whose inverse this represents."),
    ] = None

    # The forward transformation type a typed inverse inverts. It is unset
    # on the generic `Inverse` front-door and set on each typed subclass,
    # which drives both the materialization below and the wrapper registry.
    _inverseof: tx.ClassVar[tx.Optional[tx.Type[Transformation]]] = None

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False) -> Transformation:
        """Return the forward transformation, with the endpoints restored.

        The inverse of an inverse is the original forward transformation.
        An endpoint edit made on the wrapper is carried onto it.
        """
        forward = self.forward
        if forward is None:
            return Identity(input=self.input, output=self.output)
        new_input = self.output if self.output is not None else forward.input
        new_output = self.input if self.input is not None else forward.output
        if new_input is not forward.input or new_output is not forward.output:
            forward = replace(forward, input=new_input, output=new_output)
        if compute:
            forward = forward.compute()
        return forward

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: SimplifyLike = "analytic",
    ) -> Transformation:
        # Materializing an inverse to a concrete instance is the *compose*
        # path, gated by `mode`: it only runs when the mode admits the
        # inverse (which membership resolves through to the forward, so a
        # restrictive mode such as `Inverse(<field>).compute(mode="affine")`
        # does not trigger the expensive field inversion).
        modes = _lower_modes(mode)
        if modes and _mode_admits(self, modes):
            return self._materialize().compute(mode, simplify=simplify)
        # Not admitted (including `mode=False`): stay lazy -- never
        # materialize, so an adjacent pair can still cancel in a sequence.
        # Simplification is decoupled from the compose mode (see issue #92),
        # so still run the leaf-only analytic downcast of the forward.
        forward = _simplify_inner(self.forward, _lower_simplify(simplify))
        if forward is self.forward:
            return self
        return replace(self, forward=forward)

    def to(
        self,
        cls: tx.Optional[tx.Type[Transformation]] = None,
        *,
        lossy: bool = False,
        **kwargs,
    ) -> Transformation:
        if cls is None or cls is type(self):
            # An endpoint or metadata edit keeps the inverse unresolved,
            # reusing the forward transform and its cached materialization.
            return replace(self, **kwargs) if kwargs else self
        # A conversion to another type, including the forward type,
        # materializes the concrete inverse first, then converts onward.
        return self._materialize().to(cls, lossy=lossy, **kwargs)

    # --- helpers ------------------------------------------------------

    def _materialize(self) -> Transformation:
        # The concrete inverse. A typed inverse builds a plain instance of
        # the forward type holding the inverted parameter, with the
        # wrapper's (swapped) endpoints. The generic front-door defers to
        # the forward transform's own inverse, which resolves to the typed
        # inverse of that family.
        forward = self.forward
        inverseof = type(self)._inverseof
        if inverseof is None:
            if forward is None:
                return Identity(input=self.input, output=self.output)
            resolved = forward.inverse()
            edits = {}
            if self.input is not None:
                edits["input"] = self.input
            if self.output is not None:
                edits["output"] = self.output
            return resolved.to(**edits) if edits else resolved
        if forward is None:
            return inverseof(input=self.input, output=self.output)
        return replace(
            forward,
            input=self.input,
            output=self.output,
            **{inverseof.parameter_names: self._cached_inverse_param()},
        )

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        # The inverted parameter, worked out from the forward transform.
        # Each typed inverse implements the actual inversion.
        raise NotImplementedError

    def _cached_inverse_param(self) -> tx.Optional[ArrayProtocol]:
        forward = self.forward
        param = type(self)._inverseof.parameter_names
        if forward is None or getattr(forward, param) is None:
            return None
        cached = forward.__dict__.get(INVERSE_CACHE)
        if cached is None:
            # A one-tuple is stored so that a genuine `None` result is
            # cached rather than recomputed.
            cached = (self._inverse_param(),)
            forward.__dict__[INVERSE_CACHE] = cached
        return cached[0]


register_inverse(Inverse)

# NOTE (issue #93): the typed inverses below parameterize the generic
# (`Inverse[Translation]`, ...) for introspectable hints only. Making
# `Inverse(forward=x)` construct the matching typed inverse polymorphically
# (via the bagof.magic constructor) is a separate follow-up, not done here;
# `Inverse(forward=x)` stays generic-until-materialize.


def _cancels(first: Transformation, second: Transformation) -> bool:
    """Whether ``[first, second]`` cancels to the identity for free.

    `first` is applied before `second`. The two cancel when `second` is the
    lazy inverse of `first`, or `first` is the lazy inverse of `second`. An
    `Inverse` names the transform it undoes as its `forward`, so the test is
    a plain identity check that materializes neither field: it is O(1) and
    decides from object identity alone. This covers both a typed inverse
    and a generic `Inverse(forward=X)`.
    """
    if isinstance(second, Inverse) and second.forward is first:
        return True
    if isinstance(first, Inverse) and first.forward is second:
        return True
    return False


class InverseTranslation(Inverse[Translation], Translation):
    """The inverse of a [`Translation`][], resolved on demand."""

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Translation

    forward: tx.Annotated[
        tx.Optional[Translation],
        tx.Doc("The translation whose inverse this represents."),
    ] = None

    # `translation` is computed on demand from `forward`, so it is not a
    # stored, constructor-taken field here. Declaring it a `ClassVar`
    # overrides the inherited init-field and keeps it out of `__init__`,
    # `fields()` and `replace()`, while the property below serves reads.
    translation: tx.ClassVar[tx.Optional[npvector[Real]]]

    @property
    def translation(self) -> tx.Optional[ArrayProtocol]:
        return self._cached_inverse_param()

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        return -self.forward.translation


class InverseScaling(Inverse[Scaling], Scaling):
    """The inverse of a [`Scaling`][], resolved on demand."""

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Scaling

    forward: tx.Annotated[
        tx.Optional[Scaling],
        tx.Doc("The scaling whose inverse this represents."),
    ] = None

    scale: tx.ClassVar[tx.Optional[npvector[Real]]]

    @property
    def scale(self) -> tx.Optional[ArrayProtocol]:
        return self._cached_inverse_param()

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        return 1.0 / self.forward.scale


class InverseRotation(Inverse[Rotation], Rotation):
    """The inverse of a [`Rotation`][], resolved on demand."""

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Rotation

    forward: tx.Annotated[
        tx.Optional[Rotation],
        tx.Doc("The rotation whose inverse this represents."),
    ] = None

    matrix: tx.ClassVar[tx.Optional[npmatrix[Real]]]

    @property
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return self._cached_inverse_param()

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        # A rotation is orthogonal, so its inverse is its transpose.
        return self.forward.matrix.T


class InversePermutation(Inverse[Permutation], Permutation):
    """The inverse of a [`Permutation`][], resolved on demand."""

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Permutation

    forward: tx.Annotated[
        tx.Optional[Permutation],
        tx.Doc("The permutation whose inverse this represents."),
    ] = None

    permutation: tx.ClassVar[tx.Optional[npvector[Integral]]]

    @property
    def permutation(self) -> tx.Optional[ArrayProtocol]:
        return self._cached_inverse_param()

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        forward = self.forward.permutation
        inverse_permutation = [0] * len(forward)
        for i, p in enumerate(forward):
            inverse_permutation[p] = i
        return inverse_permutation


class InverseLinear(Inverse[Linear], Linear):
    """The inverse of a [`Linear`][] transformation, resolved on demand."""

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Linear

    forward: tx.Annotated[
        tx.Optional[Linear],
        tx.Doc("The linear transformation whose inverse this represents."),
    ] = None

    matrix: tx.ClassVar[tx.Optional[npmatrix[Real]]]

    @property
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return self._cached_inverse_param()

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        ab = get_array_backend(self.forward.matrix)
        return ab.linalg.inv(self.forward.matrix)


class InverseAffine(Inverse[Affine], Affine):
    """The inverse of an [`Affine`][] transformation, resolved on demand."""

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Affine

    forward: tx.Annotated[
        tx.Optional[Affine],
        tx.Doc("The affine transformation whose inverse this represents."),
    ] = None

    matrix: tx.ClassVar[tx.Optional[npmatrix[Real]]]

    @property
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return self._cached_inverse_param()

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        ab = get_array_backend(self.forward.matrix)
        return ab.linalg.inv(self.forward.homogeneous_matrix)[:-1]


class InverseDisplacementField(Inverse[DisplacementField], DisplacementField):
    """The inverse of a [`DisplacementField`][], resolved on demand.

    The wrapper reports the `order`, `bound` and `coeff` of the forward
    field, and the inverse field it materializes preserves them. A field
    of spline coefficients is inverted by re-fitting, and its inverse is
    itself a field of coefficients.
    """

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = DisplacementField

    forward: tx.Annotated[
        tx.Optional[DisplacementField],
        tx.Doc("The displacement field whose inverse this represents."),
    ] = None

    # `field` is computed on demand from `forward`, and `order`, `bound`
    # and `coeff` are read from `forward`, so none of them is a stored,
    # constructor-taken field here. Declaring each a `ClassVar` overrides
    # the inherited init-field and keeps it out of `__init__`, `fields()`
    # and `replace()`, while the properties below serve reads.
    field: tx.ClassVar[tx.Optional[ArrayProtocol]]
    order: tx.ClassVar[InterpolationOrder]
    bound: tx.ClassVar[tx.Union[BoundaryCondition, float]]
    coeff: tx.ClassVar[bool]

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        return self._cached_inverse_param()

    @property
    def order(self) -> InterpolationOrder:
        return self.forward.order

    @property
    def bound(self) -> tx.Union[BoundaryCondition, float]:
        return self.forward.bound

    @property
    def coeff(self) -> bool:
        return self.forward.coeff

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        forward = self.forward
        if not forward.coeff:
            return inverse_disp(forward.field)
        # A coefficient field is inverted by re-fitting: the coefficients
        # are read out as values, the value field is inverted, and the
        # result is fitted back to coefficients.
        values = coeff2value_field(
            forward.field, order=forward.order, bound=forward.bound
        )
        inverse_values = inverse_disp(values)
        return value2coeff_field(
            inverse_values, order=forward.order, bound=forward.bound
        )


class InverseCoordinatesField(Inverse[CoordinatesField], CoordinatesField):
    """The inverse of a [`CoordinatesField`][], resolved on demand.

    A coordinate field has no cheap closed-form inverse, so it cannot be
    materialized directly. The wrapper reports the `order`, `bound` and
    `coeff` of the forward field, and it cancels against the original field
    in a sequence. Reading its `field`, or converting or computing it
    outside a cancellation, raises.
    """

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = CoordinatesField

    forward: tx.Annotated[
        tx.Optional[CoordinatesField],
        tx.Doc("The coordinate field whose inverse this represents."),
    ] = None

    field: tx.ClassVar[tx.Optional[ArrayProtocol]]
    order: tx.ClassVar[InterpolationOrder]
    bound: tx.ClassVar[tx.Union[BoundaryCondition, float]]
    coeff: tx.ClassVar[bool]

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        return self._cached_inverse_param()

    @property
    def order(self) -> InterpolationOrder:
        return self.forward.order

    @property
    def bound(self) -> tx.Union[BoundaryCondition, float]:
        return self.forward.bound

    @property
    def coeff(self) -> bool:
        return self.forward.coeff

    def _inverse_param(self) -> tx.Optional[ArrayProtocol]:
        raise NotImplementedError(
            "The inverse of a coordinate field cannot be materialized "
            "directly. Keep the inverse unresolved so that it cancels in a "
            "sequence, or compose it away, rather than reading, converting "
            "or computing its field on its own."
        )


# Pair each forward transformation type with the typed inverse that
# represents it, keyed by the type each inverse names in `_inverseof`.
INVERSE_WRAPPERS.update(
    {
        cls._inverseof: cls
        for cls in (
            InverseTranslation,
            InverseScaling,
            InverseRotation,
            InversePermutation,
            InverseLinear,
            InverseAffine,
            InverseDisplacementField,
            InverseCoordinatesField,
        )
    }
)
