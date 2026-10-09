# stdlib
import math
from numbers import Integral, Real

# dependencies
import typing_extensions as tx
from bagof.magic import InitVar, replace

# core
from brainhops._core.affines import expm as affine_expm
from brainhops._core.affines import logm as affine_logm
from brainhops._core.affines import to_compact, to_homogeneous
from brainhops._core.linalg import expm, logm, require_positive
from brainhops._core.properties import lazyproperty
from brainhops._core.typing import ArrayProtocol, npmatrix, npvector

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel import kinds
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
from brainhops.errors import DomainError

# internals
from .base import Transformation
from .compute.compose import compose
from .concrete import (
    Affine,
    DisplacementField,
    Linear,
    Rotation,
    Scaling,
    StoreEnum,
    _alias,
    _data2values,
    _invalidating_property,
    require_endomorphism,
)

_FieldNames = tx.ClassVar[tx.Tuple[str, ...]]


class TangentMixin:
    """
    What every class that stores the tangent of its map shares.

    `data` is the tangent about the identity, so the inverse, the square
    root and the square are exact in it: negate, halve, double. The plain
    class that holds the map instead is `_tangentof`, which `.to(log=False)`
    converts to.
    """

    _tangentof: tx.ClassVar[tx.Optional[tx.Type[Transformation]]] = None

    @lazyproperty
    def _sqrt(self) -> tx.Optional[ArrayProtocol]:
        if self.data is None:
            return None
        return self.data * 0.5

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        if self.data is None:
            return None
        return -self.data

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        # Half the tangent.
        require_endomorphism(self, "square root")
        obj = replace(self, data=self.data * 0.5)
        return obj.compute(**kwargs) if compute else obj

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        # Twice the tangent.
        require_endomorphism(self, "square")
        obj = replace(self, data=self.data * 2.0)
        return obj.compute(**kwargs) if compute else obj

    def _target_class(
        self, kwargs: tx.Dict[str, tx.Any]
    ) -> tx.Type[Transformation]:
        # `log=False` asks for the map itself, which the plain class of
        # the family holds and this one does not.
        if not kwargs.get("log", True):
            return type(self)._tangentof
        return super()._target_class(kwargs)


class StationaryVelocityField(
    TangentMixin, DisplacementField, on={"_log": True}
):
    """
    A displacement field stored as its stationary velocity (`log=True`).

    `data` holds the velocity `v`, the tangent of the map about the
    identity: its values, or their spline coefficients when `coeff` is
    true (as NiftyReg's `-vel -cpp` grids and torch-diffeo store it). The
    map is the flow of `v` at time one, `exp(v)`, and the `field` view is
    always its displacement, as values. It is integrated by scaling and
    squaring: `v` is divided by `2 ** steps`, which is its own flow to
    first order, and composed with itself `steps` times, with the field
    composition and the field's own `degree` and `bound`. A field of
    coefficients is refitted at each step, so its squaring stays in
    coefficients. The displacements are in the voxels of the field's own
    grid, as for any [`DisplacementField`][].

    !!! note "Squaring on the knot grid"
        A field of coefficients is squared on its own grid: the grid of
        its knots. NiftyReg evaluates a velocity grid (`-vel -cpp`) onto
        the dense reference grid first, and squares there, so the two do
        not match `reg_transform -def` exactly, although both converge to
        the same flow.

    `DisplacementField(data=v, log=True)`, `d.to(log=True)` (for an unset
    `d`) and `StationaryVelocityField(data=v)` build the same object.
    Unset or zero `data` is the identity.

    The tangent makes some operations exact: the inverse is `exp(-v)`, the
    square root `exp(v / 2)` and the square `exp(2 v)`. A displacement has
    no logarithm that brainhops computes, so `.to(log=True)` refuses a
    `DisplacementField` that holds one; `.to(log=False)` integrates a
    velocity into a plain `DisplacementField`.
    """

    # --- class attributes ---------------------------------------------

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = DisplacementField

    _base_metadata_fields: _FieldNames = DisplacementField.metadata_fields
    metadata_fields: _FieldNames = (*_base_metadata_fields, "steps")

    # --- attributes ---------------------------------------------------

    _steps: tx.Optional[int] = None
    """
    The number of squaring steps. If `None`, the smallest number for
    which the first step, `v / 2 ** steps`, moves no point by more than
    an eighth of a voxel.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        if arguments.get("field") is not None:
            raise NotImplementedError(
                f"{type(self).__name__}() got field=, which is the map, "
                f"as  displacement values: storing it as a velocity "
                f"needs the logarithm of a field, which is not implemented. "
                f"Pass the velocity as data=."
            )
        super().__post_init__(arguments)

    # --- attribute views ----------------------------------------------

    steps = _invalidating_property("steps")

    # --- views --------------------------------------------------------

    @property
    def _compute_steps(self) -> tx.Optional[int]:
        """
        The number of squarings the integration will actually use: the
        one `steps` declares, or the smallest that bounds the first step
        when it declares none. `None` for an unset velocity, which is
        the identity and is not integrated at all.
        """
        if self.steps is not None:
            return self.steps
        if self.data is None:
            return None
        return _squaring_steps(self.values)

    @lazyproperty
    def field(self) -> tx.Optional[ArrayProtocol]:
        """
        The displacement of the map, as values: the flow at time one of
        the velocity `data` holds, integrated once, then cached.
        """
        flags = self.store, self.degree, self.bound
        return _integrate_field(self.data, *flags, self._compute_steps)

    # --- methods ------------------------------------------------------

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        # Half the velocity, which integrates with one squaring fewer.
        # A declared number of steps follows the halving; an undeclared
        # one is left undeclared, and the rule finds the same number
        # again from the halved velocity.
        require_endomorphism(self, "square root")
        obj = self
        if self.data is not None:
            steps = self.steps
            if steps is not None:
                steps = max(steps - 1, 0)
            obj = replace(self, data=self.data / 2, steps=steps)
        return obj.compute(**kwargs) if compute else obj

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        # Twice the velocity, which integrates with one squaring more.
        require_endomorphism(self, "square")
        obj = self
        if self.data is not None:
            steps = self.steps
            if steps is not None:
                steps = steps + 1
            obj = replace(self, data=self.data * 2, steps=steps)
        return obj.compute(**kwargs) if compute else obj


@kinds.PositiveAffine
class AffineExponential(TangentMixin, Affine, on={"_log": True}):
    """
    An affine transformation stored as its tangent (`log=True`).

    `data` is the `(N, N + 1)` tangent `[L, l]` of the map about the
    identity, and the map is its exponential: `matrix` is the top `N`
    rows of `expm([[L, l], [0, ..., 0]])`, computed once, then cached.
    `L` may be singular -- a translation has the tangent `[0, t]`. Unset or
    zero `data` is the identity, and `AffineExponential(data=I)` is the
    scaling by `e`, not the identity: a tangent is never a matrix.

    The exponential of a real tangent has a positive determinant, so it is
    a `PositiveAffine`. Its inverse, square root and square are exact: the
    tangent negated, halved and doubled. `.to(log=False)` gives the plain
    [`Affine`][] of `matrix`, and `Affine.to(log=True)` takes the principal
    logarithm of a matrix, which is refused (`DomainError`) when its
    linear part has an eigenvalue on the closed negative real axis.
    """

    # --- class attributes ---------------------------------------------

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = Affine

    derived_fields: _FieldNames = (
        *Affine.derived_fields,
        "logmatrix",
        "homogeneous_logmatrix",
    )

    # --- attributes ---------------------------------------------------

    _logmatrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    "The log of the matrix, with shape (No, Ni+1). An alias for `data`."

    _homogeneous_logmatrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    "The homogeneous form of the log-matrix, with shape (No+1, Ni+1)."

    # --- attribute views ----------------------------------------------

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)

    # --- views --------------------------------------------------------

    logmatrix = _alias("logmatrix", "data")

    @lazyproperty
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        """
        The affine matrix, of shape `(N, N + 1)`: the exponential of the
        tangent `data`.
        """
        if self.logmatrix is None:
            return None
        return affine_expm(self.logmatrix)

    @matrix.setter
    def matrix(self, value: tx.Optional[ArrayProtocol]) -> None:
        if value is None:
            self.logmatrix = None
        else:
            what = f"The logarithm of this {self._tangentof.__name__}"
            self.logmatrix = affine_logm(value, what)

    @lazyproperty
    def homogeneous_logmatrix(self) -> tx.Optional[ArrayProtocol]:
        if self.logmatrix is None:
            return None
        return to_homogeneous(self.logmatrix, tangent=True)

    @homogeneous_logmatrix.setter
    def homogeneous_logmatrix(self, value: tx.Optional[ArrayProtocol]) -> None:
        self.logmatrix = to_compact(value) if value is not None else None


class MatrixTangentMixin(TangentMixin):
    """
    The matrix of a linear tangent: the exponential of what `data` holds.

    Shared by the linear and the rotation tangents. The rotation is *not*
    a subclass of the linear tangent: both are selected by `log=True`
    alone, so a rotation tangent registered under the linear one would be
    what `LinearExponential(data=L)` builds, whatever `L` is.
    """

    @lazyproperty
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        """The matrix, of shape `(N, N)`: the exponential of `data`."""
        if self.logmatrix is None:
            return None
        return expm(self.logmatrix)

    @matrix.setter
    def matrix(self, value: tx.Optional[ArrayProtocol]) -> None:
        if value is None:
            self.logmatrix = None
        else:
            what = f"The logarithm of this {self._tangentof.__name__}"
            self.logmatrix = logm(value, what)


@kinds.PositiveLinear
class LinearExponential(
    MatrixTangentMixin,
    Linear,
    on={"_log": True},
    # A `RotationExponential` is a `Linear` too, and `log=True` is all
    # either of them claims, so `Linear(log=True)` would otherwise build
    # the deeper of the two. It builds this one.
    priority=1,
):
    """
    A linear transformation stored as its tangent (`log=True`).

    `data` is the `(N, N)` tangent `L` of the map about the identity, and
    the map is its exponential: `matrix` is `expm(L)`, computed once, then
    cached. Unset or zero `data` is the identity, and
    `LinearExponential(data=I)` is the scaling by `e`.

    The exponential of a real tangent has a positive determinant, so it is
    a `PositiveLinear`. Its inverse, square root and square are exact: the
    tangent negated, halved and doubled. `.to(log=False)` gives the plain
    [`Linear`][] of `matrix`, and `Linear.to(log=True)` takes the principal
    logarithm of a matrix, which is refused (`DomainError`) when it has an
    eigenvalue on the closed negative real axis.
    """

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = Linear

    derived_fields: _FieldNames = (*Linear.derived_fields, "logmatrix")

    _logmatrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    "The log of the matrix, with shape (No, Ni). An alias for `data`."

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)

    logmatrix = _alias("logmatrix", "data")


class RotationExponential(MatrixTangentMixin, Rotation, on={"_log": True}):
    """
    A rotation stored as its tangent (`log=True`).

    `data` is the `(N, N)` antisymmetric tangent `L` of the rotation about
    the identity -- its axis and angle -- and the map is its exponential:
    `matrix` is `expm(L)`, computed once, then cached. Unset or zero `data`
    is the identity.

    Its inverse, square root and square are exact: the tangent negated,
    halved and doubled. `.to(log=False)` gives the plain [`Rotation`][] of
    `matrix`, and `Rotation.to(log=True)` takes the principal logarithm of
    a rotation, which is refused (`DomainError`) for a rotation by a half
    turn, whose logarithm is not unique.
    """

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = Rotation

    derived_fields: _FieldNames = (*Rotation.derived_fields, "logmatrix")

    _logmatrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    "The antisymmetric log of the matrix. An alias for `data`."

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)

    logmatrix = _alias("logmatrix", "data")


@kinds.PositiveDiagonal
class ScalingExponential(TangentMixin, Scaling, on={"_log": True}):
    """
    A scaling stored as the logarithm of its factors (`log=True`).

    `data` is the tangent `s` of the map about the identity, and `scale`
    is `exp(s)`, computed once, then cached: the factors are positive, so
    it is a `PositiveDiagonal`. Unset or zero `data` is the identity.

    Its inverse, square root and square are exact: the tangent negated,
    halved and doubled. `.to(log=False)` gives the plain [`Scaling`][] of
    `scale`, and `Scaling.to(log=True)` takes the logarithm of the factors,
    which is refused (`DomainError`) unless they are all positive.
    """

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = Scaling

    derived_fields: _FieldNames = (*Scaling.derived_fields, "logscale")

    _logscale: InitVar[tx.Optional[npvector[Real]]] = None
    "The log of the scaling factors, with shape (N,). An alias for `data`."

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)

    logscale = _alias("logscale", "data")

    @lazyproperty
    def scale(self) -> tx.Optional[ArrayProtocol]:
        """The scaling factors, of shape `(N,)`: `exp(data)`."""
        if self.logscale is None:
            return None
        return get_array_backend(self.logscale).exp(self.logscale)

    @scale.setter
    def scale(self, value: tx.Optional[ArrayProtocol]) -> None:
        if value is None:
            self.logscale = None
        else:
            # Only positive factors have a real logarithm, so a scaling
            # that flips an axis has no tangent.
            what = f"The logarithm of this {self._tangentof.__name__}"
            require_positive(value, what)
            self.logscale = get_array_backend(value).log(value)


# --- helpers ----------------------------------------------------------


_FIRST_STEP = 0.125
"""
The largest displacement, in voxels, of the first step of scaling and
squaring. Arsigny et al. (2006) bound it by half a voxel; a smaller step
keeps the first-order step `id + v / 2**steps` accurate for velocities
whose gradient is large, at the price of two or three more squarings.
"""

_MAX_STEPS = 64
"""A velocity that would need more squarings than this is refused."""


def _integrate_field(
    data: tx.Optional[ArrayProtocol],
    store: StoreEnum,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
    steps: tx.Optional[int],
) -> tx.Optional[ArrayProtocol]:
    # The displacement, as values, of the flow at time one of the
    # stationary velocity that `data` stores, by scaling and squaring.
    if data is None:
        return None
    if steps is None:
        steps = _squaring_steps(_data2values(data, store, degree, bound))
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError(
            f"The number of squaring steps must be a non-negative integer, "
            f"not {steps!r}."
        )
    # Scaling: a small enough velocity is its own flow, to first order.
    # Scaling is linear, so the coefficients of a velocity are scaled as
    # its values are.
    step = DisplacementField(
        data=data * (0.5**steps), degree=degree, bound=bound, store=store
    )
    # Squaring: the flow at time `2 s` is the flow at time `s`, twice. The
    # field composition keeps the encoding of its left operand, so a
    # velocity of coefficients is refitted at each step.
    for _ in range(steps):
        step = compose(step, step)
        # A lazy (dask) array is evaluated before the next step reads it:
        # each task of a lazy sample rebuilds the window of the field it
        # reads, so the cost of an unevaluated chain doubles with each
        # squaring.
        if hasattr(step.data, "persist"):
            step.data = step.data.persist()
    return step.field


def _squaring_steps(velocity: ArrayProtocol) -> int:
    # The smallest number of squarings for which the first step moves no
    # point by more than `_FIRST_STEP` voxels.
    backend = get_array_backend(velocity)
    norm = float(backend.max(backend.sqrt((velocity**2).sum(axis=-1))))
    if not math.isfinite(norm):
        raise DomainError(
            "The exponential of this velocity field is not defined: it is "
            "not finite."
        )
    if norm <= _FIRST_STEP:
        return 0
    steps = int(math.ceil(math.log2(norm / _FIRST_STEP)))
    if steps > _MAX_STEPS:
        raise DomainError(
            "The exponential of this velocity field is not computed: its "
            f"largest displacement, {norm} voxels, would need more than "
            f"{_MAX_STEPS} squaring steps."
        )
    return steps
