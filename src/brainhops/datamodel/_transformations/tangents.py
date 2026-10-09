import math
from numbers import Integral, Real

import typing_extensions as tx
from bagof.magic import InitVar, replace

from brainhops._core.affines import expm as affine_expm
from brainhops._core.affines import logm as affine_logm
from brainhops._core.affines import to_compact, to_homogeneous
from brainhops._core.linalg import expm, logm, require_positive
from brainhops._core.properties import lazyproperty
from brainhops._core.typing import ArrayProtocol, npmatrix, npvector
from brainhops.backends import get_array_backend
from brainhops.datamodel import kinds
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
from brainhops.errors import DomainError

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
    """Mixin shared by the classes that store the tangent of their map.

    The `data` of such a class is a tangent about the identity, so the inverse,
    the square root and the square are exact: they negate, halve and double the
    tangent. `_tangentof` names the plain class that holds the map itself,
    which is the target of `.to(log=False)`.
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
        require_endomorphism(self, "square root")
        obj = replace(self, data=self.data * 0.5)
        return obj.compute(**kwargs) if compute else obj

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        require_endomorphism(self, "square")
        obj = replace(self, data=self.data * 2.0)
        return obj.compute(**kwargs) if compute else obj

    def _target_class(
        self, kwargs: tx.Dict[str, tx.Any]
    ) -> tx.Type[Transformation]:
        # `log=False` asks for the map itself, which is held by the plain class
        # of the family rather than by this one.
        if not kwargs.get("log", True):
            return type(self)._tangentof
        return super()._target_class(kwargs)


class StationaryVelocityField(
    TangentMixin, DisplacementField, on={"_log": True}
):
    """A displacement field stored as a stationary velocity (`log=True`).

    The `data` is the velocity `v`, stored as values or, when `coeff` is true,
    as spline coefficients (as in NiftyReg `-vel -cpp` grids). The map is the
    flow of `v` at time one, `exp(v)`, and the `field` view holds its
    displacement as values, in voxels of the field's own grid. The flow is
    integrated by scaling and squaring: `v` is divided by `2**steps` and the
    result is composed with itself `steps` times, with the field's own `degree`
    and `bound`.

    Unset or zero data is the identity, and the inverse, square root and square
    are exact. Since the logarithm of a displacement is not implemented,
    `.to(log=True)` refuses a [`DisplacementField`][] that already holds a
    displacement, whereas `.to(log=False)` integrates the velocity into a plain
    [`DisplacementField`][].

    !!! note "Squaring on the knot grid"
        A coefficient field is squared on its own knot grid, whereas NiftyReg
        squares on the dense reference grid. The results therefore differ
        slightly from `reg_transform -def`, although both converge to the same
        flow.
    """

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = DisplacementField

    _base_metadata_fields: _FieldNames = DisplacementField.metadata_fields
    metadata_fields: _FieldNames = (*_base_metadata_fields, "steps")

    _steps: tx.Optional[int] = None
    """Number of squaring steps.

    `None` selects the smallest number for which the first step,
    `v / 2**steps`, moves no point by more than an eighth of a voxel.
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

    steps = _invalidating_property("steps")

    @property
    def _compute_steps(self) -> tx.Optional[int]:
        """Number of squarings that the integration actually uses.

        This is the declared `steps`, or else the smallest number that bounds
        the first step, or `None` for an unset velocity.
        """
        if self.steps is not None:
            return self.steps
        if self.data is None:
            return None
        return _squaring_steps(self.values)

    @lazyproperty
    def field(self) -> tx.Optional[ArrayProtocol]:
        """Displacement of the map as values.

        The flow at time one of the velocity in `data` is integrated once and
        then cached.
        """
        flags = self.store, self.degree, self.bound
        return _integrate_field(self.data, *flags, self._compute_steps)

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        # Half the velocity needs one squaring fewer; undeclared steps stay
        # undeclared, since the rule finds that count itself.
        require_endomorphism(self, "square root")
        obj = self
        if self.data is not None:
            steps = self.steps
            if steps is not None:
                steps = max(steps - 1, 0)
            obj = replace(self, data=self.data / 2, steps=steps)
        return obj.compute(**kwargs) if compute else obj

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        # Twice the velocity needs one squaring more.
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
    """An affine transformation stored as its tangent (`log=True`).

    The `data` is the `(N, N+1)` tangent `[L, l]` about the identity, and
    `matrix` holds the top `N` rows of `expm([[L, l], [0, ..., 0]])`. `L` may
    be singular: a translation has the tangent `[0, t]`. Zero data is the
    identity, whereas `AffineExponential(data=I)` is a scaling by `e`. The
    exponential of a real tangent has a positive determinant, hence the kind
    [`PositiveAffine`][brainhops.datamodel.kinds.PositiveAffine].
    `Affine.to(log=True)` takes the principal logarithm and raises a
    [`DomainError`][] if the linear part has an eigenvalue on the closed
    negative real axis.
    """

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = Affine

    derived_fields: _FieldNames = (
        *Affine.derived_fields,
        "logmatrix",
        "homogeneous_logmatrix",
    )

    _logmatrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    """Logarithm of the matrix, an init-only alias of `data`."""

    _homogeneous_logmatrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    """Homogeneous form of the logarithm, of shape `(No+1, Ni+1)`."""

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)

    logmatrix = _alias("logmatrix", "data")

    @lazyproperty
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        """Affine matrix, the exponential of the tangent `data`.

        Assigning a matrix stores its logarithm.
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
    """Mixin for linear tangents whose matrix is the exponential of `data`.

    The rotation tangent deliberately does not subclass the linear tangent.
    Both are selected by `log=True` alone, so a subclass would be what
    `LinearExponential(data=L)` builds for any `L`.
    """

    @lazyproperty
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        """Matrix of shape `(N, N)`, the exponential of `data`.

        Assigning a matrix stores its logarithm.
        """
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
    # `RotationExponential` is also a `Linear` that claims only `log=True`, so
    # without the priority `Linear(log=True)` would build it instead of
    # `LinearExponential`.
    priority=1,
):
    """A linear transformation stored as its tangent (`log=True`).

    The `data` is the `(N, N)` tangent `L` and `matrix` is `expm(L)`, so that
    `LinearExponential(data=I)` is a scaling by `e` and the determinant is
    positive ([`PositiveLinear`][brainhops.datamodel.kinds.PositiveLinear]).
    `Linear.to(log=True)` takes the principal logarithm and raises a
    [`DomainError`][] when the matrix has an eigenvalue on the closed negative
    real axis.
    """

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = Linear

    derived_fields: _FieldNames = (*Linear.derived_fields, "logmatrix")

    _logmatrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    """Logarithm of the matrix, an init-only alias of `data`."""

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)

    logmatrix = _alias("logmatrix", "data")


class RotationExponential(MatrixTangentMixin, Rotation, on={"_log": True}):
    """A rotation stored as its tangent (`log=True`).

    The `data` is the antisymmetric `(N, N)` tangent `L`, which encodes the
    axis and the angle, and `matrix` is `expm(L)`. `Rotation.to(log=True)`
    raises a [`DomainError`][] for a half turn, whose logarithm is not unique.
    """

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = Rotation

    derived_fields: _FieldNames = (*Rotation.derived_fields, "logmatrix")

    _logmatrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    """Antisymmetric logarithm of the matrix, an init-only alias of `data`."""

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)

    logmatrix = _alias("logmatrix", "data")


@kinds.PositiveDiagonal
class ScalingExponential(TangentMixin, Scaling, on={"_log": True}):
    """A scaling stored as the logarithm of its factors (`log=True`).

    The `data` is the tangent `s` and `scale` is `exp(s)`, so the factors are
    positive
    ([`PositiveDiagonal`][brainhops.datamodel.kinds.PositiveDiagonal]).
    `Scaling.to(log=True)` raises a [`DomainError`][] unless all factors are
    positive.
    """

    _tangentof: tx.ClassVar[tx.Type[Transformation]] = Scaling

    derived_fields: _FieldNames = (*Scaling.derived_fields, "logscale")

    _logscale: InitVar[tx.Optional[npvector[Real]]] = None
    """Logarithm of the scaling factors, an init-only alias of `data`."""

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)

    logscale = _alias("logscale", "data")

    @lazyproperty
    def scale(self) -> tx.Optional[ArrayProtocol]:
        """Scaling factors, the exponential of `data`.

        Assigning factors stores their logarithm.
        """
        if self.logscale is None:
            return None
        return get_array_backend(self.logscale).exp(self.logscale)

    @scale.setter
    def scale(self, value: tx.Optional[ArrayProtocol]) -> None:
        if value is None:
            self.logscale = None
        else:
            # Only positive factors have a real logarithm, so a scaling that
            # flips an axis has no tangent.
            what = f"The logarithm of this {self._tangentof.__name__}"
            require_positive(value, what)
            self.logscale = get_array_backend(value).log(value)


_FIRST_STEP = 0.125
"""Largest displacement, in voxels, of the first squaring step.

Arsigny et al. (2006) use half a voxel. A smaller bound keeps the first-order
step accurate for velocities with large gradients, at the cost of two or three
more squarings.
"""

_MAX_STEPS = 64
"""Largest number of squarings; a velocity that needs more is refused."""


def _integrate_field(
    data: tx.Optional[ArrayProtocol],
    store: StoreEnum,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
    steps: tx.Optional[int],
) -> tx.Optional[ArrayProtocol]:
    # Integrate the flow at time one of the stationary velocity by scaling and
    # squaring, and return its displacement as values.
    if data is None:
        return None
    if steps is None:
        steps = _squaring_steps(_data2values(data, store, degree, bound))
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError(
            f"The number of squaring steps must be a non-negative integer, "
            f"not {steps!r}."
        )
    # A small enough velocity is its own flow to first order. Scaling is
    # linear, so coefficients scale like values.
    step = DisplacementField(
        data=data * (0.5**steps), degree=degree, bound=bound, store=store
    )
    # Field composition keeps the encoding of its left operand, so a
    # coefficient velocity is refitted at each step.
    for _ in range(steps):
        step = compose(step, step)
        # Evaluate lazy arrays before the next step; otherwise the cost of the
        # unevaluated chain doubles with each squaring.
        if hasattr(step.data, "persist"):
            step.data = step.data.persist()
    return step.field


def _squaring_steps(velocity: ArrayProtocol) -> int:
    # Smallest number of squarings for which the first step moves no point by
    # more than `_FIRST_STEP` voxels.
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
