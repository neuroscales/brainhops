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

# core
from brainhops._core.typing import ArrayProtocol

# datamodel
from brainhops.backends import get_array_backend
from brainhops.datamodel import hierarchy

# internals
from .base import Transformation
from .check import checker, is_kind, register_kind_alias
from .concrete import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Linear,
    Permutation,
    Scaling,
    Translation,
)
from .inverse import Inverse
from .meta import (
    Bijection,
    MetaTransformation,
    Projection,
    SubspaceTransformation,
)

# ======================================================================
#
#                           C H E C K E R S
#
# ======================================================================

# --- Identity ---------------------------------------------------------

IdentityTarget = hierarchy.IdentityTransformation


@checker(IdentityTarget)
def _(query: Translation, compute: bool) -> bool:
    if query.translation is None:
        return True
    if compute:
        return (query.translation == 0).all()
    return False


@checker(IdentityTarget)
def _(query: Scaling, compute: bool) -> bool:
    if query.scale is None:
        return True
    if compute:
        return (query.scale == 1).all()
    return False


@checker(IdentityTarget)
def _(query: Permutation, compute: bool) -> bool:
    if query.permutation is None:
        return True
    if compute:
        ndim = len(query.permutation)
        return (query.permutation == list(range(ndim))).all()
    return False


@checker(IdentityTarget)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if cols != rows:
            return False
        ab = get_array_backend(matrix)
        return (matrix == ab.eye(rows)).all()
    return False


@checker(IdentityTarget)
def _(query: Affine, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if cols != rows + 1:
            return False
        ab = get_array_backend(matrix)
        return (matrix == ab.eye(rows + 1)[:-1]).all()
    return False


@checker(IdentityTarget)
def _(query: DisplacementField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    if compute:
        return (field == 0).all()
    return False


@checker(IdentityTarget)
def _(query: CoordinatesField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False




# --- Grid -------------------------------------------------------------
# A [`CartesianField`][] is the identity map over its own grid: its `field`
# is the coordinates of the grid points themselves. So a grid establishes
# membership in the identity set -- and, through it, in every set that
# contains the identity.
#
# But *only* numerically. Structurally, a grid whose `shape` is set carries
# a parameter, exactly as a `Translation` whose vector is set does, and a
# parameter is not read at analytic. This matters: it is what keeps a grid
# from being downcast to `Identity` (and its sampling domain lost) by a
# structural pass. Only `shape` is ever read, never the derived `field`, so
# a membership question never builds the meshgrid.


def _grid_is_identity(query: CartesianField, compute: bool) -> bool:
    if query.shape is None:
        return True
    return bool(compute)


# --- Translation ------------------------------------------------------

TranslationTarget = hierarchy.Translation


@checker(TranslationTarget)
def _(query: Scaling, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(TranslationTarget)
def _(query: Permutation, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(TranslationTarget)
def _(query: Linear, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(TranslationTarget)
def _(query: Affine, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if cols != rows + 1:
            return False
        ab = get_array_backend(matrix)
        # The linear block must be the identity, not merely diagonal: a
        # constant map (a zero linear block) is not a translation.
        return bool((matrix[:, :-1] == ab.eye(rows)).all())
    return False


@checker(TranslationTarget)
def _(query: DisplacementField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    if compute:
        first = (0,) * (len(field.shape) - 1) + (slice(None),)
        return (field == field[first]).all()
    return False


@checker(TranslationTarget)
def _(query: CoordinatesField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False




# --- Scaling ----------------------------------------------------------

ScaleTarget = hierarchy.DiagonalTransformation


@checker(ScaleTarget)
def _(query: Translation, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(ScaleTarget)
def _(query: Permutation, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(ScaleTarget)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        return not (matrix * (1 - ab.eye(rows))).any()
    return False


@checker(ScaleTarget)
def _(query: Affine, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if cols != rows + 1:
            return False
        ab = get_array_backend(matrix)
        # `eye(rows + 1)[:-1]` has the same shape as the affine matrix and
        # marks its diagonal, so the mask keeps every off-diagonal entry
        # *and* the translation column: a diagonal affine has neither.
        return not (matrix * (1 - ab.eye(rows + 1)[:-1])).any()
    return False


@checker(ScaleTarget)
def _(query: DisplacementField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False


@checker(ScaleTarget)
def _(query: CoordinatesField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False




# <<< special cases >>>


# >>> refinements of the diagonal set
#
# Every fact below is a property of the diagonal *vector*, so it is written
# once, as a predicate on that vector, and registered for every leaf that
# can produce one: a `Scaling` reads it off its `scale`, and a `Linear` or
# `Affine` off the diagonal of a matrix already established as diagonal.
# Writing them as predicates rather than as one checker per (leaf, node)
# pair is what keeps a `Linear` that happens to be diagonal from refining
# any less far than the `Scaling` it is equivalent to.


def _isotropic(scale: ArrayProtocol) -> bool:
    # Every factor equal: the scaling is a multiple of the identity.
    return bool((scale == scale[:1]).all())


def _diag_invertible(scale: ArrayProtocol) -> bool:
    return bool((scale != 0).all())


def _diag_positive(scale: ArrayProtocol) -> bool:
    return bool((scale > 0).all())


def _diag_special(scale: ArrayProtocol) -> bool:
    ab = get_array_backend(scale)
    return bool((ab.abs(scale) == 1).all())


def _diag_multiplicative(scale: ArrayProtocol) -> bool:
    return _isotropic(scale)


def _diag_multiplicative_invertible(scale: ArrayProtocol) -> bool:
    return _isotropic(scale) and bool(scale[0] != 0)


def _diag_multiplicative_positive(scale: ArrayProtocol) -> bool:
    return _isotropic(scale) and bool(scale[0] > 0)


_DIAGONAL_FACTS: tx.Dict[type, tx.Callable[[ArrayProtocol], bool]] = {
    hierarchy.InvertibleDiagonalTransformation: _diag_invertible,
    hierarchy.PositiveDiagonalTransformation: _diag_positive,
    hierarchy.SpecialDiagonalTransformation: _diag_special,
    hierarchy.MultiplicativeTransformation: _diag_multiplicative,
    hierarchy.InvertibleMultiplicativeTransformation:
        _diag_multiplicative_invertible,
    hierarchy.PositiveMultiplicativeTransformation:
        _diag_multiplicative_positive,
}

# Invertibility is the one fact assumed optimistically from structure: a
# scaling is presumed non-degenerate until its values say otherwise, the
# same optimism the matrix leaves get from their shape (see
# `_matrix_invertible`). Every other refinement needs the values.
_OPTIMISTIC = (
    hierarchy.InvertibleDiagonalTransformation,
    hierarchy.InvertibleMultiplicativeTransformation,
)


def _diagonal_values(
    query: Transformation,
) -> tx.Optional[ArrayProtocol]:
    # The diagonal a leaf represents, read from its values, or `None` when
    # it does not represent a diagonal map at all. This reads the matrix
    # directly rather than asking `is_kind`, because the refinements below
    # are themselves subsets of the diagonal set: routing through `is_kind`
    # would ask them the question they are answering.
    scale = getattr(query, "scale", None)
    if scale is not None:
        return scale
    matrix = getattr(query, "matrix", None)
    if matrix is None:
        return None
    ab = get_array_backend(matrix)
    if isinstance(query, Affine):
        if bool(matrix[:, -1].any()):
            return None  # a translation: not linear, so not diagonal
        matrix = matrix[:, :-1]
    rows, cols = matrix.shape
    if rows != cols:
        return None
    if bool((matrix * (1 - ab.eye(rows))).any()):
        return None  # an off-diagonal entry
    return ab.diagonal(matrix)


def _unset_parameter(query: Transformation) -> bool:
    # Whether a leaf holds no parameter at all, in which case it is the
    # identity -- a member of every refinement below.
    for name in ("scale", "matrix", "permutation", "translation"):
        if hasattr(query, name):
            return getattr(query, name) is None
    return False


def _diagonal_checker(node: type) -> tx.Callable:
    fact = _DIAGONAL_FACTS[node]
    optimistic = node in _OPTIMISTIC

    def _checker(query: Transformation, compute: bool) -> bool:
        if _unset_parameter(query):
            return True
        if not compute:
            # Structure alone establishes diagonality only for a
            # `Scaling`, whose whole parameter is the diagonal. A matrix
            # has to be read.
            return optimistic and isinstance(query, Scaling)
        scale = _diagonal_values(query)
        return scale is not None and fact(scale)

    return _checker


for _node in _DIAGONAL_FACTS:
    for _src in (Scaling, Linear, Affine):
        checker(_src, _node)(_diagonal_checker(_node))
del _node, _src


# --- Permutation ------------------------------------------------------

PermTarget = hierarchy.Permutation


@checker(PermTarget)
def _(query: Translation, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(PermTarget)
def _(query: Scaling, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(PermTarget)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        is_binary = bool(ab.isin(matrix, [0, 1]).all())
        is_perm = bool(
            (matrix.sum(0) == 1).all() and (matrix.sum(1) == 1).all()
        )
        return is_binary and is_perm
    return False


@checker(PermTarget)
def _(query: Affine, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if cols != rows + 1:
            return False
        ab = get_array_backend(matrix)
        no_translation = bool((matrix[:, -1] == 0).all())
        if not no_translation:
            return False
        linpart = matrix[:, :-1]
        is_binary = bool(ab.isin(linpart, [0, 1]).all())
        is_perm = bool(
            (linpart.sum(0) == 1).all() and (linpart.sum(1) == 1).all()
        )
        return is_binary and is_perm
    return False


@checker(PermTarget)
def _(query: DisplacementField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False


@checker(PermTarget)
def _(query: CoordinatesField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False


# <<< special cases >>>


@checker(hierarchy.EvenPermutation)
def _(query: Permutation, compute: bool) -> bool:
    permutation = query.permutation
    if permutation is None:
        return True
    if compute:
        ndim = len(permutation)
        if ndim != len(set(permutation)):
            return False
        return _reindex_is_even(permutation, list(range(ndim)))
    return False


@checker(hierarchy.OddPermutation)
def _(query: Permutation, compute: bool) -> bool:
    permutation = query.permutation
    if permutation is None:
        return False
    return (
        is_kind(query, hierarchy.Permutation, compute) and not
        is_kind(query, hierarchy.EvenPermutation, compute)
    )


# --- Rotation ---------------------------------------------------------

RotTarget = hierarchy.SpecialOrthogonalTransformation


@checker(RotTarget)
def _(query: Translation, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(RotTarget)
def _(query: Scaling, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(RotTarget)
def _(query: Permutation, compute: bool) -> bool:
    permutation = query.permutation
    if permutation is None:
        return True
    if compute:
        ndim = len(permutation)
        if ndim != len(set(permutation)):
            return False
        return _reindex_is_even(permutation, list(range(ndim)))
    return False


@checker(RotTarget)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        is_orthogonal = bool((matrix @ matrix.T == ab.eye(rows)).all())
        is_posdef = bool(ab.linalg.det(matrix) > 0)
        return is_orthogonal and is_posdef
    return False


@checker(RotTarget)
def _(query: Affine, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if cols != rows + 1:
            return False
        ab = get_array_backend(matrix)
        no_translation = bool((matrix[:, -1] == 0).all())
        if not no_translation:
            return False
        linpart = matrix[:, :-1]
        is_orthogonal = bool((linpart @ linpart.T == ab.eye(rows)).all())
        is_posdef = bool(ab.linalg.det(linpart) > 0)
        return is_orthogonal and is_posdef
    return False


@checker(RotTarget)
def _(query: DisplacementField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False


@checker(RotTarget)
def _(query: CoordinatesField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False


# --- Linear ---------------------------------------------------------

LinTarget = hierarchy.LinearTransformation


@checker(LinTarget)
def _(query: Translation, compute: bool) -> bool:
    return is_kind(query, hierarchy.IdentityTransformation, compute)


@checker(LinTarget)
def _(query: Scaling, compute: bool) -> bool:
    return True


@checker(LinTarget)
def _(query: Permutation, compute: bool) -> bool:
    return True


@checker(LinTarget)
def _(query: Affine, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        # An affine is linear when its translation column vanishes. The
        # linear block may be rectangular, so no squareness is required.
        return bool((matrix[:, -1] == 0).all())
    return False


@checker(LinTarget)
def _(query: DisplacementField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False


@checker(LinTarget)
def _(query: CoordinatesField, compute: bool) -> bool:
    field = query.field
    if field is None:
        return True
    return False


# <<< special cases >>>


@checker(hierarchy.PositiveLinearTransformation)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        return bool(ab.linalg.det(matrix) > 0)
    return False


@checker(hierarchy.SpecialLinearTransformation)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        return bool(ab.linalg.det(matrix) == 1)
    return False


@checker(hierarchy.SpecialConformalOrthogonalTransformation)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        fro = ab.linalg.norm(matrix, ord="fro")
        if bool(fro == 0):
            return False
        matrix = matrix * (rows ** 0.5 / fro)
        is_orthogonal = bool((matrix @ matrix.T == ab.eye(rows)).all())
        is_posdef = bool(ab.linalg.det(matrix) > 0)
        return is_orthogonal and is_posdef
    return False


@checker(hierarchy.OrthogonalTransformation)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        return bool((matrix @ matrix.T == ab.eye(rows)).all())
    return False


def _is_monomial(matrix: ArrayProtocol) -> bool:
    # A monomial (generalized permutation) matrix: exactly one non-zero
    # entry in every row and in every column.
    rows, cols = matrix.shape
    if rows != cols:
        return False
    nonzero = matrix != 0
    return bool(
        (nonzero.sum(0) == 1).all() and (nonzero.sum(1) == 1).all()
    )


@checker(hierarchy.GeneralizedPermutation)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        return _is_monomial(matrix)
    return False


@checker(hierarchy.SignedPermutation)
def _(query: Linear, compute: bool) -> bool:
    matrix = query.matrix
    if matrix is None:
        return True
    if compute:
        # A signed permutation is a monomial matrix whose non-zero entries
        # are all +1 or -1.
        if not _is_monomial(matrix):
            return False
        ab = get_array_backend(matrix)
        return bool(ab.isin(matrix, [-1, 0, 1]).all())
    return False


# --- Grid registration ------------------------------------------------
# Registered against every target a grid could be established in. A grid
# inherits from `CoordinatesField`, whose checkers read `.field`; these
# shadow them per target, so a membership question on a grid never builds
# the meshgrid.

for _target in (
    IdentityTarget,
    TranslationTarget,
    ScaleTarget,
    PermTarget,
    RotTarget,
    LinTarget,
):
    checker(CartesianField, _target)(_grid_is_identity)
del _target


# ======================================================================
#
#                     M A T R I X   H E L P E R S
#
# ======================================================================


def _matrix_dims(t: Transformation) -> tx.Optional[tx.Tuple[int, int]]:
    """`(Ni, No)` of a matrix leaf from the stored `matrix.shape`, or `None`.

    `Affine`: the matrix is `(No, Ni + 1)`; `Linear`/`Rotation`: `(No, Ni)`.
    Reads only the shape, never a value; never called on an `Inverse`.
    Shared with `separable._element_dims` so the two readings cannot drift.
    """
    matrix = getattr(t, "matrix", None)
    if matrix is None:
        return None
    no, cols = matrix.shape
    ni = cols - 1 if isinstance(t, Affine) else cols
    return ni, no


def _linear_part(t: Transformation) -> tx.Optional[ArrayProtocol]:
    # The linear block whose rank decides injectivity/surjectivity at
    # `numeric`. `None` for a parameter that is invertible by declaration
    # (`Permutation`, `Translation`, `Identity`) or has no matrix.
    if isinstance(t, Affine):
        return None if t.matrix is None else t.matrix[:, :-1]
    if isinstance(t, Linear):  # includes Rotation
        return t.matrix
    if isinstance(t, Scaling):
        if t.scale is None:
            return None
        ab = get_array_backend(t.scale)
        return ab.diag(t.scale)
    return None


def _matrix_rank(linear: ArrayProtocol) -> int:
    # The rank of a linear block. `matrix_rank` is the single rank test,
    # with a `det`-based fallback for a square matrix when a backend lacks
    # it (a non-zero determinant means full rank).
    ab = get_array_backend(linear)
    matrix_rank = getattr(getattr(ab, "linalg", None), "matrix_rank", None)
    if matrix_rank is not None:
        return int(matrix_rank(linear))
    no, ni = linear.shape
    if no == ni:
        return ni if bool(ab.linalg.det(linear) != 0) else ni - 1
    raise NotImplementedError(
        "the array backend provides neither matrix_rank nor a square "
        "determinant fallback for rank computation"
    )


def _matrix_invertible(t: Transformation, compute: bool) -> bool:
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


def _matrix_surjective(t: Transformation, compute: bool) -> bool:
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


def _matrix_injective(t: Transformation, compute: bool) -> bool:
    # Tall or square: optimistically full column rank (one-to-one),
    # confirmed by rank.
    dims = _matrix_dims(t)
    if dims is None:
        return False
    ni, no = dims
    if not compute:
        return ni <= no
    linear = _linear_part(t)
    if linear is None:
        return False
    return _matrix_rank(linear) == ni


# --- Shape-established facts ------------------------------------------
# Invertibility, injectivity and surjectivity are the facts a matrix leaf
# establishes from the *shape* of its matrix, optimistically: a square
# matrix is presumed invertible, a wide one onto, a tall one one-to-one.
# Numeric confirms or retracts each with the rank of the linear block.
#
# Only the *invertible* node of each family is registered, never the
# bijective/injective/surjective roots directly: `is_kind` reaches a
# superset from any subset it establishes, so `InvertibleAffine` already
# answers `Bijective`, `Injective` and `Surjective` for a square affine.
# The two roots are registered on their own for the rectangular cases,
# which no invertible node covers.

checker(Affine, hierarchy.InvertibleAffineTransformation)(_matrix_invertible)
checker(Linear, hierarchy.InvertibleLinearTransformation)(_matrix_invertible)
checker(Affine, hierarchy.SurjectiveTransformation)(_matrix_surjective)
checker(Linear, hierarchy.SurjectiveTransformation)(_matrix_surjective)
checker(Affine, hierarchy.InjectiveTransformation)(_matrix_injective)
checker(Linear, hierarchy.InjectiveTransformation)(_matrix_injective)


# ======================================================================
#
#                        L A T T I C E   F A C T S
#
# ======================================================================
# Facts about the lattice, used to reason about the wrappers (a lift of, a
# permutation of, an inversion of an inner set). Kept as module-level tables
# keyed by node, so the hierarchy stays pure set/group theory.


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
    if hierarchy.is_liftable(node):
        return (node,)
    return tuple(
        _maximal(
            n for n in hierarchy.all_sets() if (
                issubclass(n, node) and n is not node and
                hierarchy.is_liftable(n)
            )
        )
    )


@lru_cache(maxsize=None)  # noqa: UP033
def _permute_targets(node: type, even: bool) -> tx.Tuple[type, ...]:
    # `P @ blockdiag(...) in node` (P a coordinate permutation of the given
    # parity) iff the lifted map in M for some M in these targets.
    perm = hierarchy.EvenPermutation if even else hierarchy.Permutation
    def is_closed(node: type) -> bool:
        return hierarchy.is_closedunder(node, perm)

    if is_closed(node):
        return (node,)
    return tuple(_maximal(
            n for n in hierarchy.all_sets()
            if issubclass(n, node) and n is not node and is_closed(n)
    ))


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
            for n in hierarchy.all_sets()
            if issubclass(n, node)
            and issubclass(n, hierarchy.BijectiveTransformation)
        )
    )


def _reindex_is_even(input_axes: tx.Any, output_axes: tx.Any) -> bool:
    """
    Parity of the coordinate permutation a reindexing subspace applies:
    position `input_axes[j]` maps to `output_axes[j]`.

    Returns True for an even permutation (det +1).

    A length or set mismatch is treated as odd (det-sign nodes are then
    dropped by the caller's `even=False` path).
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
#
#                        W R A P P E R   C H E C K E R S
#
# ======================================================================
# Each wrapper's membership depends on its contents, so it is decided by a
# checker that recurses into the wrapped transform via `is_kind` (carrying
# `compute` through). A checker is registered per node `C`, deciding "is this
# wrapper a member of `C`"; `is_kind` OR-s the applicable ones.


def _inner_member(
    inner: tx.Optional[Transformation], m: type, compute: bool
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
    relaxed = hierarchy.as_noninvertible(node)
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
    for node in hierarchy.all_sets():
        checker(source, node)(_bind_node(member, node))


def _bind_node(member: tx.Callable, node: type) -> tx.Callable:
    def _checker(t: Transformation, compute: bool) -> bool:
        return member(t, compute, node)

    return _checker


_register_wrapper(SubspaceTransformation, _subspace_member)
_register_wrapper(Projection, _projection_member)
_register_wrapper(Bijection, _bijection_member)
_register_wrapper(Inverse, _inverse_member)


# ======================================================================
#
#                        K I N D   A L I A S E S
#
# ======================================================================
# The alias table names a kind that a hierarchy NAME or SYMBOL does not
# already name. Two sorts of entry live here:
#
# * a **class kind** -- a wrapper, a field, a container. These have no
#   hierarchy node, because membership depends on their contents rather
#   than on a set they belong to, so a name that resolves to one is matched
#   by `isinstance`.
# * a **friendlier spelling of a set node** -- `"scaling"` for the diagonal
#   set. These keep set semantics; only the spelling is new.
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
