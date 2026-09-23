"""Kind-membership checkers, registered into `registries.CHECKERS`.

This is to `check` what `composers` is to `compose` and `converters` is to
`convert`: the machinery lives in `check`, and every concrete checker
implementation lives here and registers via `@checker(SourceType, KindNode)`
at import time. It also holds the lattice-fact tables the wrapper checkers
reason with (as module-level data keyed by node, not attributes on the
hierarchy classes) and the field/wrapper name aliases.

Each checker is *sound*: it returns True only when membership is genuinely
(or, at analytic, optimistically from shape) established, so OR-ing the
applicable ones (see `check.is_kind`) is correct.
"""

# stdlib
from functools import lru_cache

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import hierarchy

# internals
from .check import checker, is_kind, register_kind_alias
from .concrete import (
    Affine,
    ConcreteTransformation,
    CoordinatesField,
    DisplacementField,
    Linear,
    Scaling,
    _linear_part,
    _matrix_dims,
    _matrix_rank,
    is_affine,
    is_identity,
    is_linear,
    is_permutation,
    is_rotation,
    is_scale,
    is_translation,
)
from .inverse import Inverse
from .meta import (
    Bijection,
    MetaTransformation,
    Projection,
    SubspaceTransformation,
)

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation


# ======================================================================
#                        L A T T I C E   F A C T S
# ======================================================================
# Facts about the lattice, used to reason about the wrappers (a lift of, a
# permutation of, an inversion of an inner set). Kept as module-level tables
# keyed by node, so the hierarchy stays pure set/group theory.

# The strictly-invertible refinement of a (possibly non-invertible) set.
_INVERTIBLE_OF = {
    hierarchy.AffineTransformation: hierarchy.InvertibleAffineTransformation,
    hierarchy.LinearTransformation: hierarchy.InvertibleLinearTransformation,
    hierarchy.DiagonalTransformation: (
        hierarchy.InvertibleDiagonalTransformation
    ),
    hierarchy.MultiplicativeTransformation: (
        hierarchy.InvertibleMultiplicativeTransformation
    ),
    hierarchy.MatrixTransformation: hierarchy.InvertibleMatrixTransformation,
    hierarchy.Transformation: hierarchy.BijectiveTransformation,
}
_NONINVERTIBLE_OF = {v: k for k, v in _INVERTIBLE_OF.items()}

# Sets NOT closed under blockdiag(., I): the uniform-scale / conformal
# families (lifting a uniform scaling into a larger identity block breaks
# uniformity). Everything else is liftable, including Injective/Surjective.
_NOT_LIFTABLE = frozenset(
    {
        hierarchy.MultiplicativeTransformation,
        hierarchy.InvertibleMultiplicativeTransformation,
        hierarchy.PositiveMultiplicativeTransformation,
        hierarchy.Dilation,
        hierarchy.PositiveDilation,
        hierarchy.ConformalEuclideanTransformation,
        hierarchy.SpecialConformalEuclideanTransformation,
        hierarchy.ConformalOrthogonalTransformation,
        hierarchy.SpecialConformalOrthogonalTransformation,
    }
)

# Sets closed under composition with a coordinate permutation P (any parity).
_PERMUTATION_CLOSED = frozenset(
    {
        hierarchy.Transformation,
        hierarchy.InjectiveTransformation,
        hierarchy.SurjectiveTransformation,
        hierarchy.BijectiveTransformation,
        hierarchy.Isomorphism,
        hierarchy.Diffeomorphism,
        hierarchy.MatrixTransformation,
        hierarchy.InvertibleMatrixTransformation,
        hierarchy.AffineTransformation,
        hierarchy.InvertibleAffineTransformation,
        hierarchy.ConformalEuclideanTransformation,
        hierarchy.EuclideanTransformation,
        hierarchy.LinearTransformation,
        hierarchy.InvertibleLinearTransformation,
        hierarchy.ConformalOrthogonalTransformation,
        hierarchy.OrthogonalTransformation,
        hierarchy.GeneralizedPermutation,
        hierarchy.SignedPermutation,
        hierarchy.Permutation,
    }
)
# Additionally closed when P is EVEN (det P = +1).
_EVEN_PERMUTATION_CLOSED = _PERMUTATION_CLOSED | frozenset(
    {
        hierarchy.VolumePreservingDiffeomorphism,
        hierarchy.PositiveDefiniteMatrixTransformation,
        hierarchy.PositiveAffineTransformation,
        hierarchy.SpecialAffineTransformation,
        hierarchy.SpecialConformalEuclideanTransformation,
        hierarchy.SpecialEuclideanTransformation,
        hierarchy.PositiveLinearTransformation,
        hierarchy.SpecialLinearTransformation,
        hierarchy.SpecialConformalOrthogonalTransformation,
        hierarchy.SpecialOrthogonalTransformation,
    }
)


def _liftable(node: type) -> bool:
    return node not in _NOT_LIFTABLE


@lru_cache(maxsize=None)  # noqa: UP033
def _all_nodes() -> tx.FrozenSet[type]:
    return frozenset(hierarchy.NAMETOCLASS.values())


def _maximal(nodes: tx.Iterable[type]) -> tx.List[type]:
    # The maximal (largest) sets under inclusion: drop any node that is a
    # strict subset of another node in the set.
    nodes = list(nodes)
    return [
        n
        for n in nodes
        if not any(m is not n and issubclass(n, m) for m in nodes)
    ]


@lru_cache(maxsize=None)  # noqa: UP033
def _lift_targets(node: type) -> tx.Tuple[type, ...]:
    # `blockdiag(inner, I) in node` iff `inner in M` for some M in these
    # targets. If `node` is liftable it is its own target; else the maximal
    # liftable strict subnodes of `node`.
    if _liftable(node):
        return (node,)
    return tuple(
        _maximal(
            n
            for n in _all_nodes()
            if issubclass(n, node) and n is not node and _liftable(n)
        )
    )


@lru_cache(maxsize=None)  # noqa: UP033
def _permute_targets(node: type, even: bool) -> tx.Tuple[type, ...]:
    # `P @ blockdiag(...) in node` (P a coordinate permutation of the given
    # parity) iff the lifted map in M for some M in these targets.
    closed = _EVEN_PERMUTATION_CLOSED if even else _PERMUTATION_CLOSED
    if node in closed:
        return (node,)
    return tuple(
        _maximal(
            n
            for n in _all_nodes()
            if issubclass(n, node) and n is not node and n in closed
        )
    )


@lru_cache(maxsize=None)  # noqa: UP033
def _bijective_targets(node: type) -> tx.Tuple[type, ...]:
    # The inverse of `T` is in `node` iff `T in M` for some M in these
    # targets: `node` itself when it is bijective (closed under inversion),
    # else the maximal bijective subnodes of `node`.
    if issubclass(node, hierarchy.BijectiveTransformation):
        return (node,)
    return tuple(
        _maximal(
            n
            for n in _all_nodes()
            if issubclass(n, node)
            and issubclass(n, hierarchy.BijectiveTransformation)
        )
    )


def _reindex_is_even(input_axes: tx.Any, output_axes: tx.Any) -> bool:
    """Parity of the coordinate permutation a reindexing subspace applies:
    position `input_axes[j]` maps to `output_axes[j]`. Returns True for an
    even permutation (det +1). A length or set mismatch is treated as odd
    (det-sign nodes are then dropped by the caller's `even=False` path).
    """
    if input_axes is None or output_axes is None:
        return True
    a, b = list(input_axes), list(output_axes)
    if len(a) != len(b) or sorted(a) != sorted(b):
        return False
    pos = {axis: j for j, axis in enumerate(a)}
    perm = [pos[b[j]] for j in range(len(a))]
    seen = [False] * len(perm)
    swaps = 0
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        swaps += length - 1
    return swaps % 2 == 0


# ======================================================================
#                    C O N C R E T E   L E A F   C H E C K E R S
# ======================================================================
# `compute=False` is analytic (structure only, optimistic by shape);
# `compute=True` is numeric (values, rank via `_matrix_rank`).


def _matrix_invertible(t: "Transformation", compute: bool) -> bool:
    # A square matrix leaf: optimistically invertible from shape, confirmed
    # by full rank at `numeric`.
    dims = _matrix_dims(t)
    if dims is None:
        return False
    ni, no = dims
    if no != ni:
        return False
    if not compute:
        return True
    linear = _linear_part(t)
    if linear is None:
        return False
    return _matrix_rank(linear) == no


def _matrix_surjective(t: "Transformation", compute: bool) -> bool:
    # Wide or square: optimistically full row rank (onto), confirmed by rank.
    dims = _matrix_dims(t)
    if dims is None:
        return False
    ni, no = dims
    if not compute:
        return no <= ni
    linear = _linear_part(t)
    if linear is None:
        return False
    return _matrix_rank(linear) == no


def _matrix_injective(t: "Transformation", compute: bool) -> bool:
    # Tall or square: optimistically full column rank (one-to-one), confirmed
    # by rank.
    dims = _matrix_dims(t)
    if dims is None:
        return False
    ni, no = dims
    if not compute:
        return no >= ni
    linear = _linear_part(t)
    if linear is None:
        return False
    return _matrix_rank(linear) == ni


def _scaling_invertible(t: "Transformation", compute: bool) -> bool:
    # A scaling is square; assumed non-zero (hence invertible) at analytic,
    # confirmed by full rank at numeric. An all-`None` scaling is the
    # identity, established by the identity checker instead.
    if getattr(t, "scale", None) is None:
        return False
    if not compute:
        return True
    linear = _linear_part(t)
    if linear is None:
        return False
    return _matrix_rank(linear) == linear.shape[0]


def _diagonal_invertible(t: "Transformation", compute: bool) -> bool:
    # A matrix leaf whose values are diagonal AND full-rank square is an
    # invertible diagonal (the numeric refinement of the `Diagonal` fact for
    # a general `Linear`/`Affine`; analytic reads no values). `Scaling` is
    # covered by `_scaling_invertible`.
    if not compute or not is_scale(t, compute=True):
        return False
    linear = _linear_part(t)
    if linear is None:
        return False
    no, ni = linear.shape
    return no == ni and _matrix_rank(linear) == no


# The numeric kind-check ladder, shared by every concrete leaf: each check is
# sound at both levels (`is_X(t, False)` only recognizes a declared instance
# or the identity; `is_X(t, True)` reads the values).
for _kind, _check in (
    (hierarchy.IdentityTransformation, is_identity),
    (hierarchy.Translation, is_translation),
    (hierarchy.DiagonalTransformation, is_scale),
    (hierarchy.Permutation, is_permutation),
    (hierarchy.SpecialOrthogonalTransformation, is_rotation),
    (hierarchy.LinearTransformation, is_linear),
    (hierarchy.AffineTransformation, is_affine),
):
    checker(ConcreteTransformation, _kind)(_check)
del _kind, _check

# Optimistic-shape invertibility / injectivity / surjectivity for matrix
# leaves, refined by rank at numeric.
checker(Affine, hierarchy.InvertibleAffineTransformation)(_matrix_invertible)
checker(Affine, hierarchy.SurjectiveTransformation)(_matrix_surjective)
checker(Affine, hierarchy.InjectiveTransformation)(_matrix_injective)
checker(Linear, hierarchy.InvertibleLinearTransformation)(_matrix_invertible)
checker(Linear, hierarchy.SurjectiveTransformation)(_matrix_surjective)
checker(Linear, hierarchy.InjectiveTransformation)(_matrix_injective)
checker(Scaling, hierarchy.InvertibleDiagonalTransformation)(
    _scaling_invertible
)
checker(ConcreteTransformation, hierarchy.InvertibleDiagonalTransformation)(
    _diagonal_invertible
)


# ======================================================================
#                        W R A P P E R   C H E C K E R S
# ======================================================================
# Each wrapper's membership depends on its contents, so it is decided by a
# checker that recurses into the wrapped transform via `is_kind` (carrying
# `compute` through). A checker is registered per node `C`, deciding "is this
# wrapper a member of `C`"; `is_kind` OR-s the applicable ones.


def _inner_member(
    inner: tx.Optional["Transformation"], m: type, compute: bool
) -> bool:
    # Membership of a subspace's inner in node `m`. A `None` inner is the
    # identity (`blockdiag(I, I) = I`), a member of `m` exactly when `m`
    # contains the identity node.
    if inner is None:
        return issubclass(hierarchy.IdentityTransformation, m)
    return is_kind(inner, m, compute)


def _subspace_member(
    sub: SubspaceTransformation, compute: bool, node: type
) -> bool:
    # A subspace is `P @ blockdiag(inner, I)`: a lift of `inner` into the full
    # space, optionally composed with a coordinate permutation `P` when it
    # reindexes axes. Its membership in `node` follows from the inner's
    # membership in the lift (and permutation) targets of `node`.
    inner = sub.transformation
    same = (sub.input_axes is None and sub.output_axes is None) or (
        sub.input_axes is not None
        and sub.output_axes is not None
        and list(sub.input_axes) == list(sub.output_axes)
    )
    if same:
        return any(
            _inner_member(inner, m, compute) for m in _lift_targets(node)
        )
    even = _reindex_is_even(sub.input_axes, sub.output_axes)
    return any(
        _inner_member(inner, m, compute)
        for target in _permute_targets(node, even)
        for m in _lift_targets(target)
    )


def _projection_member(proj: Projection, compute: bool, node: type) -> bool:
    # Establish injectivity/surjectivity/identity from the axis lists. A
    # projection that only drops axes is surjective; one that only creates
    # axes is injective; one that does both establishes nothing beyond
    # `Transformation`.
    dropped = 0 if proj.dropped is None else len(proj.dropped)
    created = 0 if proj.created is None else len(proj.created)
    if not dropped and not created:
        return issubclass(hierarchy.IdentityTransformation, node)
    if dropped and not created:
        return issubclass(hierarchy.SurjectiveTransformation, node)
    if created and not dropped:
        return issubclass(hierarchy.InjectiveTransformation, node)
    return False


def _bijection_member(bij: Bijection, compute: bool, node: type) -> bool:
    # Delegate to the forward map (or the inverse of the backward when no
    # forward is given). A `Bijection` is declared bijective, so a node that
    # only its invertible variant would establish (e.g. `Affine` when the
    # forward is a bare affine) is relaxed to that non-invertible set.
    f = bij.forward
    if f is None and bij.backward is not None:
        f = Inverse(forward=bij.backward)
    if f is None:
        return False
    if is_kind(f, node, compute):
        return True
    relaxed = _NONINVERTIBLE_OF.get(node)
    return relaxed is not None and is_kind(f, relaxed, compute)


def _inverse_member(inv: Inverse, compute: bool, node: type) -> bool:
    # An inverse never reads its own parameter, at any level. Every set under
    # `Bijective` is closed under inversion, so `Inverse(T) in N` for such an
    # `N` exactly when `T in N`; for a general node it holds when `T` is in
    # the bijective sub-set of `N`. For `Injective`/`Surjective`,
    # `_bijective_targets` is `(Bijective,)`: the inverse of a non-bijection
    # is not a function.
    if inv.forward is None:
        return issubclass(hierarchy.IdentityTransformation, node)
    return any(
        is_kind(inv.forward, m, compute) for m in _bijective_targets(node)
    )


def _register_wrapper(source: type, member: tx.Callable) -> None:
    # Register `member(t, compute, node=C)` for every hierarchy node `C`.
    for node in _all_nodes():
        checker(source, node)(_bind_node(member, node))


def _bind_node(member: tx.Callable, node: type) -> tx.Callable:
    def _checker(t: "Transformation", compute: bool) -> bool:
        return member(t, compute, node)

    return _checker


_register_wrapper(SubspaceTransformation, _subspace_member)
_register_wrapper(Projection, _projection_member)
_register_wrapper(Bijection, _bijection_member)
_register_wrapper(Inverse, _inverse_member)


# ======================================================================
#                            K I N D   A L I A S E S
# ======================================================================
# Wrapper and field kinds have no hierarchy node (their membership depends on
# their contents), so a name that resolves to one is matched by `isinstance`.
# `"scaling"`/`"positivescaling"` resolve to the diagonal *sets* (nodes).
register_kind_alias("inverse", Inverse)
register_kind_alias("subspace", SubspaceTransformation)
register_kind_alias("projection", Projection)
register_kind_alias("meta", MetaTransformation)
register_kind_alias("field", (DisplacementField, CoordinatesField))
register_kind_alias("displacements", DisplacementField)
register_kind_alias("coordinates", CoordinatesField)
register_kind_alias("scaling", hierarchy.DiagonalTransformation)
register_kind_alias(
    "positivescaling", hierarchy.PositiveDiagonalTransformation
)
