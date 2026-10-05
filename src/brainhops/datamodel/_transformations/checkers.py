"""Kind-membership checkers, registered into `check.is_kind`.

This is to `check` what `composers` is to `compose` and `converters` is to
`convert`: the machinery lives in `check`, and every concrete checker
implementation lives here and registers at import time. It also holds the
lattice-fact tables the wrapper checkers reason with (as module-level data
keyed by node, not attributes on the kind classes) and the field/wrapper
name aliases.

A checker is written the way it is called -- `(query, kind, compute)`, with
the pair it is keyed by read from the first two hints:

    @checker
    def _(query: Affine, kind: type[kinds.Translation], compute: bool) -> bool

Each checker is *sound*: it returns True only when membership is genuinely
(or, at analytic, optimistically from shape) established, so OR-ing the
applicable ones (see `check.is_kind`) is correct -- and each is registered
against the *smallest* set it decides, since `is_kind` reaches every
superset of a set it establishes.
"""

# stdlib
from functools import lru_cache, wraps

# dependencies
import typing_extensions as tx

# core
from brainhops._core.typing import ArrayProtocol

# datamodel
from brainhops.backends import get_array_backend
from brainhops.datamodel import kinds

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
    TransformationField,
    Translation,
)
from .inverse import Inverse
from .meta import (
    Bijection,
    MetaTransformation,
    Projection,
    SubspaceTransformation,
)


def identity_from(field: str) -> tx.Callable:
    """Decorator that reads an unset parameter as the identity.

    A leaf that holds no parameter *is* the identity map, so it is a member
    of every set that contains the identity -- which is every set a checker
    wrapped in this decorator is registered against. No value is read, so
    the answer holds at analytic as well as at numeric.
    """

    def decorator(func: tx.Callable) -> tx.Callable:

        @wraps(func)
        def wrapper(
            query: Transformation,
            kind: tx.Type[kinds.TransformationKind],
            compute: bool,
        ) -> bool:
            if getattr(query, field) is None:
                return True
            return func(query, kind, compute)

        return wrapper

    return decorator


# ======================================================================
#
#                           C H E C K E R S
#
# ======================================================================

# --- Identity ---------------------------------------------------------
# The identity is the bottom of the lattice, a member of nearly every set in
# it, so these checkers answer far more questions than their own node: they
# are what establishes a parameterless `Scaling` in the translations, a
# unit-diagonal `Linear` in the rotations, and a zero field in the affines.
# Every other checker in this module is written against the smallest set it
# decides, for the same reason.

IdentityType = tx.Type[kinds.Identity]


@checker
@identity_from("translation")
def _(query: Translation, kind: IdentityType, compute: bool) -> bool:
    if compute:
        return bool((query.translation == 0).all())
    return False


@checker
@identity_from("scale")
def _(query: Scaling, kind: IdentityType, compute: bool) -> bool:
    if compute:
        return bool((query.scale == 1).all())
    return False


@checker
@identity_from("permutation")
def _(query: Permutation, kind: IdentityType, compute: bool) -> bool:
    if compute:
        ndim = len(query.permutation)
        return bool((query.permutation == list(range(ndim))).all())
    return False


@checker
@identity_from("matrix")
def _(query: Linear, kind: IdentityType, compute: bool) -> bool:
    if compute:
        matrix = query.matrix
        rows, cols = matrix.shape
        if cols != rows:
            return False
        ab = get_array_backend(matrix)
        return bool((matrix == ab.eye(rows)).all())
    return False


@checker
@identity_from("matrix")
def _(query: Affine, kind: IdentityType, compute: bool) -> bool:
    if compute:
        matrix = query.matrix
        rows, cols = matrix.shape
        if cols != rows + 1:
            return False
        ab = get_array_backend(matrix)
        return bool((matrix == ab.eye(rows + 1)[:-1]).all())
    return False


@checker
@identity_from("data")
def _(query: DisplacementField, kind: IdentityType, compute: bool) -> bool:
    if compute:
        return bool((query.field == 0).all())
    return False


@checker
@identity_from("data")
def _(query: CoordinatesField, kind: IdentityType, compute: bool) -> bool:
    return False


# <<< special case >>>
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


@checker
@identity_from("shape")
def _(query: CartesianField, kind: IdentityType, compute: bool) -> bool:
    return bool(compute)


# --- Translation ------------------------------------------------------

TranslationType = tx.Type[kinds.Translation]


@checker
@identity_from("matrix")
def _(query: Affine, kind: TranslationType, compute: bool) -> bool:
    if compute:
        matrix = query.matrix
        rows, cols = matrix.shape
        if cols != rows + 1:
            return False
        ab = get_array_backend(matrix)
        return bool((matrix[:, :-1] == ab.eye(rows)).all())
    return False


@checker
@identity_from("data")
def _(query: DisplacementField, kind: TranslationType, compute: bool) -> bool:
    if compute:
        field = query.field
        first = (0,) * (len(field.shape) - 1) + (slice(None),)
        return bool((field == field[first]).all())
    return False


# --- Scaling ----------------------------------------------------------

ScaleType = tx.Type[kinds.Diagonal]


@checker
@identity_from("matrix")
def _(query: Linear, kind: ScaleType, compute: bool) -> bool:
    if compute:
        matrix = query.matrix
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        return not bool((matrix * (1 - ab.eye(rows))).any())
    return False


@checker
@identity_from("matrix")
def _(query: Affine, kind: ScaleType, compute: bool) -> bool:
    if compute:
        matrix = query.matrix
        rows, cols = matrix.shape
        if cols != rows + 1:
            return False
        ab = get_array_backend(matrix)
        # `eye(rows + 1)[:-1]` has the same shape as the affine matrix and
        # marks its diagonal, so the mask keeps every off-diagonal entry
        # *and* the translation column: a diagonal affine has neither.
        return not bool((matrix * (1 - ab.eye(rows + 1)[:-1])).any())
    return False


# <<< special cases >>>
# <<< refinements of the diagonal set >>>
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


def _diag_inv(scale: ArrayProtocol) -> bool:
    return bool((scale != 0).all())


def _diag_pos(scale: ArrayProtocol) -> bool:
    return bool((scale > 0).all())


def _diag_spec(scale: ArrayProtocol) -> bool:
    ab = get_array_backend(scale)
    return bool((ab.abs(scale) == 1).all())


def _diag_iso(scale: ArrayProtocol) -> bool:
    return _isotropic(scale)


def _diag_iso_inv(scale: ArrayProtocol) -> bool:
    return _isotropic(scale) and _diag_inv(scale)


def _diag_iso_pos(scale: ArrayProtocol) -> bool:
    return _isotropic(scale) and _diag_pos(scale)


_DIAGONAL_FACTS: tx.Dict[type, tx.Callable[[ArrayProtocol], bool]] = {
    kinds.InvertibleDiagonal: _diag_inv,
    kinds.PositiveDiagonal: _diag_pos,
    kinds.SpecialOrthogonalDiagonal: _diag_spec,
    kinds.Multiplicative: _diag_iso,
    kinds.InvertibleMultiplicative: _diag_iso_inv,
    kinds.PositiveMultiplicative: _diag_iso_pos,
}

# Invertibility is the one fact assumed optimistically from structure: a
# scaling is presumed non-degenerate until its values say otherwise, the
# same optimism the matrix leaves get from their shape (see
# `_matrix_invertible`). Every other refinement needs the values.
#
# So the only node a `Scaling` is established in structurally is the
# invertible *diagonal* set: diagonality is its whole parameter, and
# invertibility is the optimistic part. `InvertibleMultiplicative` is not
# here, even though it too is an invertible node -- it also asserts
# isotropy, which is a property of the values, and which `Multiplicative`
# itself (rightly) declines to establish at analytic.
_OPTIMISTIC = (kinds.InvertibleDiagonal,)

_DATAFIELD = {
    Scaling: "scale",
    Linear: "matrix",
    Affine: "matrix",
}


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


def _diagonal_checker(source: type, node: type) -> tx.Callable:
    # The checker deciding "is this `source` a member of `node`", for one of
    # the refinements above. Registered explicitly rather than by hints: the
    # pair it is keyed by is the loop variable, not something a generated
    # closure could annotate.
    fact = _DIAGONAL_FACTS[node]
    optimistic = node in _OPTIMISTIC

    @identity_from(_DATAFIELD[source])
    def _checker(query: Transformation, kind: type, compute: bool) -> bool:
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
        checker(_src, _node)(_diagonal_checker(_src, _node))
del _node, _src


# --- Permutation ------------------------------------------------------

PermType = tx.Type[kinds.Permutation]


@checker
@identity_from("matrix")
def _(query: Linear, kind: PermType, compute: bool) -> bool:
    if compute:
        matrix = query.matrix
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


@checker
@identity_from("matrix")
def _(query: Affine, kind: PermType, compute: bool) -> bool:
    if compute:
        matrix = query.matrix
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


# <<< special cases >>>


EvenPermType = tx.Type[kinds.EvenPermutation]
OddPermType = tx.Type[kinds.OddPermutation]


@checker
@identity_from("permutation")
def _(query: Permutation, kind: EvenPermType, compute: bool) -> bool:
    if compute:
        permutation = query.permutation
        ndim = len(permutation)
        if ndim != len(set(permutation)):
            return False
        return _reindex_is_even(permutation, list(range(ndim)))
    return False


@checker
def _(query: Permutation, kind: OddPermType, compute: bool) -> bool:
    # The identity is even, so -- unlike every other checker here -- an
    # unset parameter establishes nothing.
    if query.permutation is None:
        return False
    return is_kind(query, kinds.Permutation, compute) and not is_kind(
        query, kinds.EvenPermutation, compute
    )


# --- Rotation ---------------------------------------------------------

RotType = tx.Type[kinds.SpecialOrthogonal]


@checker
@identity_from("matrix")
def _(query: Linear, kind: RotType, compute: bool) -> bool:
    matrix = query.matrix
    if compute:
        rows, cols = matrix.shape
        if rows != cols:
            return False
        ab = get_array_backend(matrix)
        is_orthogonal = bool((matrix @ matrix.T == ab.eye(rows)).all())
        is_posdef = bool(ab.linalg.det(matrix) > 0)
        return is_orthogonal and is_posdef
    return False


@checker
@identity_from("matrix")
def _(query: Affine, kind: RotType, compute: bool) -> bool:
    matrix = query.matrix
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


# --- Linear ---------------------------------------------------------

LinType = tx.Type[kinds.Linear]


@checker
def _(query: Scaling, kind: LinType, compute: bool) -> bool:
    return True


@checker
def _(query: Permutation, kind: LinType, compute: bool) -> bool:
    return True


@checker
@identity_from("matrix")
def _(query: Affine, kind: LinType, compute: bool) -> bool:
    if not compute:
        return False
    # An affine is linear when its translation column vanishes. The
    # linear block may be rectangular, so no squareness is required.
    return bool((query.matrix[:, -1] == 0).all())


# <<< special cases >>>


PosLinType = tx.Type[kinds.PositiveLinear]
SLType = tx.Type[kinds.SpecialLinear]
COType = tx.Type[kinds.ConformalOrthogonal]
CSOType = tx.Type[kinds.SpecialConformalOrthogonal]
OType = tx.Type[kinds.Orthogonal]
MonomialType = tx.Type[kinds.GeneralizedPermutation]
SignedMonomialType = tx.Type[kinds.SignedPermutation]


@checker
@identity_from("matrix")
def _(query: Linear, kind: PosLinType, compute: bool) -> bool:
    if not compute:
        return False
    matrix = query.matrix
    rows, cols = matrix.shape
    if rows != cols:
        return False
    ab = get_array_backend(matrix)
    return bool(ab.linalg.det(matrix) > 0)


@checker
@identity_from("matrix")
def _(query: Linear, kind: SLType, compute: bool) -> bool:
    if not compute:
        return False
    matrix = query.matrix
    rows, cols = matrix.shape
    if rows != cols:
        return False
    ab = get_array_backend(matrix)
    return bool(ab.linalg.det(matrix) == 1)


def _is_conformal(matrix: ArrayProtocol) -> bool:
    # `A A' = c**2 I` for some `c > 0`: a positive multiple of an orthogonal
    # matrix, which is what "preserves angles" means. Such an `A` has
    # `||A||_F = c sqrt(n)`, so dividing it out turns the question into a
    # plain orthogonality test and the scale never has to be recovered.
    rows, cols = matrix.shape
    if rows != cols:
        return False
    ab = get_array_backend(matrix)
    fro = ab.linalg.norm(matrix, ord="fro")
    if bool(fro == 0):
        return False
    normalized = matrix * (rows**0.5 / fro)
    return bool((normalized @ normalized.T == ab.eye(rows)).all())


@checker
@identity_from("matrix")
def _(query: Linear, kind: COType, compute: bool) -> bool:
    if not compute:
        return False
    return _is_conformal(query.matrix)


@checker
@identity_from("matrix")
def _(query: Linear, kind: CSOType, compute: bool) -> bool:
    if not compute:
        return False
    matrix = query.matrix
    if not _is_conformal(matrix):
        return False
    # The *special* conformal group is the component with positive
    # determinant; normalising by a positive factor cannot change its sign.
    ab = get_array_backend(matrix)
    return bool(ab.linalg.det(matrix) > 0)


@checker
@identity_from("matrix")
def _(query: Linear, kind: OType, compute: bool) -> bool:
    if not compute:
        return False
    matrix = query.matrix
    rows, cols = matrix.shape
    if rows != cols:
        return False
    ab = get_array_backend(matrix)
    return bool((matrix @ matrix.T == ab.eye(rows)).all())


def _is_monomial(matrix: ArrayProtocol) -> bool:
    # A monomial (generalized permutation) matrix: exactly one non-zero
    # entry in every row and in every column.
    rows, cols = matrix.shape
    if rows != cols:
        return False
    nonzero = matrix != 0
    return bool((nonzero.sum(0) == 1).all() and (nonzero.sum(1) == 1).all())


@checker
@identity_from("matrix")
def _(query: Linear, kind: MonomialType, compute: bool) -> bool:
    if not compute:
        return False
    return _is_monomial(query.matrix)


@checker
@identity_from("matrix")
def _(query: Linear, kind: SignedMonomialType, compute: bool) -> bool:
    if not compute:
        return False
    # A signed permutation is a monomial matrix whose non-zero entries
    # are all +1 or -1.
    matrix = query.matrix
    if not _is_monomial(matrix):
        return False
    ab = get_array_backend(matrix)
    return bool(ab.isin(matrix, [-1, 0, 1]).all())


# ======================================================================
#
#                     M A T R I X   H E L P E R S
#
# ======================================================================


def _matrix_dims(t: Transformation) -> tx.Optional[tx.Tuple[int, int]]:
    """`(Ni, No)` of a matrix leaf from the stored `matrix.shape`, or `None`.

    `Affine`: the matrix is `(No, Ni + 1)`; `Linear`/`Rotation`: `(No, Ni)`.
    Reads only the shape, never a value; never called on an `Inverse`.
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


@identity_from("matrix")
def _matrix_invertible(t: Transformation, kind: type, compute: bool) -> bool:
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


@identity_from("matrix")
def _matrix_surjective(t: Transformation, kind: type, compute: bool) -> bool:
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


@identity_from("matrix")
def _matrix_injective(t: Transformation, kind: type, compute: bool) -> bool:
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


# --- The linear part of an affine -------------------------------------
# An `Affine` whose translation column vanishes *is* its linear part, so it
# is established in exactly the sets that part is established in. Written
# once, rather than as one checker per (Affine, node) pair: the fact is about
# the representation, not about any particular set. Without it the facts an
# `Affine` can establish would stop at the handful of nodes given a checker
# of their own, and a translation-free affine could not be shown orthogonal,
# special-linear, monomial, and so on.


def _affine_is_linear_part(
    query: Affine, kind: tx.Type[kinds.TransformationKind], compute: bool
) -> bool:
    matrix = query.matrix
    if matrix is None:
        # The identity -- a member of every set that *holds* the identity,
        # which is not every set on the linear side: `OddPermutation` is one
        # it is not in. So this is tested, not assumed (unlike
        # `identity_from`, whose checkers are each registered against a set
        # known to contain it).
        return issubclass(kinds.Identity, kind)
    if not compute:
        # Reading the translation column is reading a value.
        return False
    if bool(matrix[:, -1].any()):
        return False  # a genuine translation: not linear at all
    return is_kind(Linear(matrix=matrix[:, :-1]), kind, compute)


# Registered only where an `Affine` has no checker of its own: one key holds
# one checker, so registering this against a node the sections above already
# cover would *replace* the specialised one -- which decides the same
# question, but knows things this cannot (a `Scaling`-like diagonal is
# established from its own parameter, and an unset one is the identity).
for _node in kinds.all_sets():
    if issubclass(_node, kinds.Linear) and (Affine, _node) not in is_kind:
        checker(Affine, _node)(_affine_is_linear_part)
del _node


# --- Shape-established facts ------------------------------------------
# Invertibility, injectivity and surjectivity are the facts a matrix leaf
# establishes from the *shape* of its matrix, optimistically: a square
# matrix is presumed invertible, a wide one onto, a tall one one-to-one.
# Numeric confirms or retracts each with the rank of the linear block.
#
# Only the *invertible* node of each family is registered, never the
# bijective root directly: a query about a superset reaches the nearest
# subset that is registered, so `InvertibleAffine` already answers
# `Bijective` for a square affine. `Injection` and `Surjection` are
# registered on their own because they are *not* supersets of an invertible
# node only -- a rectangular matrix is one-to-one or onto without being
# invertible, and no invertible node covers that.

checker(Affine, kinds.InvertibleAffine)(_matrix_invertible)
checker(Linear, kinds.InvertibleLinear)(_matrix_invertible)
checker(Affine, kinds.Surjection)(_matrix_surjective)
checker(Linear, kinds.Surjection)(_matrix_surjective)
checker(Affine, kinds.Injection)(_matrix_injective)
checker(Linear, kinds.Injection)(_matrix_injective)


# ======================================================================
#
#                        L A T T I C E   F A C T S
#
# ======================================================================
# Facts about the lattice, used to reason about the wrappers (an embedding
# of, a permutation of, an inversion of an inner set). Kept as module tables
# keyed by node, so [`kinds`][] stays pure set/group theory.


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
def _embed_targets(node: type) -> tx.Tuple[type, ...]:
    # `blockdiag(inner, I) in node` iff `inner in M` for some M in these
    # targets. If `node` is embeddable it is its own target; else the maximal
    # embeddable strict subnodes of `node`.
    if kinds.is_embeddable(node):
        return (node,)
    return tuple(
        _maximal(
            n
            for n in kinds.all_sets()
            if (
                issubclass(n, node)
                and n is not node
                and kinds.is_embeddable(n)
            )
        )
    )


@lru_cache(maxsize=None)  # noqa: UP033
def _permute_targets(node: type, even: bool) -> tx.Tuple[type, ...]:
    # `P @ blockdiag(...) in node` (P a coordinate permutation of the given
    # parity) iff the embedded map in M for some M in these targets.
    perm = kinds.EvenPermutation if even else kinds.Permutation

    def is_closed(node: type) -> bool:
        return kinds.is_closedunder(node, perm)

    if is_closed(node):
        return (node,)
    return tuple(
        _maximal(
            n
            for n in kinds.all_sets()
            if issubclass(n, node) and n is not node and is_closed(n)
        )
    )


@lru_cache(maxsize=None)  # noqa: UP033
def _bijective_targets(node: type) -> tx.Tuple[type, ...]:
    # The inverse of `T` is in `node` iff `T in M` for some M in these
    # targets: `node` itself when it is bijective (closed under inversion),
    # else the maximal bijective subnodes of `node`.
    if issubclass(node, kinds.Bijection):
        return (node,)
    return tuple(
        _maximal(
            n
            for n in kinds.all_sets()
            if issubclass(n, node) and issubclass(n, kinds.Bijection)
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
# `compute` through). One function decides "is this wrapper a member of
# `node`" for every node, reasoning from the lattice facts above rather than
# from a table of registrations.


def _inner_member(
    inner: tx.Optional[Transformation], m: type, compute: bool
) -> bool:
    # Membership of a subspace's inner in node `m`. A `None` inner is the
    # identity (`blockdiag(I, I) = I`), a member of `m` exactly when `m`
    # contains the identity node.
    if inner is None:
        return issubclass(kinds.Identity, m)
    return is_kind(inner, m, compute)


def _subspace_member(
    sub: SubspaceTransformation, node: type, compute: bool
) -> bool:
    # A subspace is `P @ blockdiag(inner, I)`: an embedding of `inner` into
    # the full space, optionally composed with a coordinate permutation `P`
    # when it reindexes axes. Its membership in `node` follows from the
    # inner's membership in the embedding (and permutation) targets of
    # `node`.
    inner = sub.transformation
    same = (sub.input_axes is None and sub.output_axes is None) or (
        sub.input_axes is not None
        and sub.output_axes is not None
        and list(sub.input_axes) == list(sub.output_axes)
    )
    if same:
        return any(
            _inner_member(inner, m, compute) for m in _embed_targets(node)
        )
    even = _reindex_is_even(sub.input_axes, sub.output_axes)
    return any(
        _inner_member(inner, m, compute)
        for target in _permute_targets(node, even)
        for m in _embed_targets(target)
    )


def _projection_member(proj: Projection, node: type, compute: bool) -> bool:
    # Establish injectivity/surjectivity/identity from the axis lists. A
    # projection that only drops axes is surjective; one that only creates
    # axes is injective; one that does both establishes nothing beyond
    # `Transformation`.
    dropped = 0 if proj.dropped is None else len(proj.dropped)
    created = 0 if proj.created is None else len(proj.created)
    if not dropped and not created:
        return issubclass(kinds.Identity, node)
    if dropped and not created:
        return issubclass(kinds.Surjection, node)
    if created and not dropped:
        return issubclass(kinds.Injection, node)
    return False


def _bijection_member(bij: Bijection, node: type, compute: bool) -> bool:
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
    relaxed = kinds.as_unrestricted(node)
    return relaxed is not None and is_kind(f, relaxed, compute)


def _inverse_member(inv: Inverse, node: type, compute: bool) -> bool:
    # An inverse never reads its own parameter, at any level. Every set under
    # `Bijective` is closed under inversion, so `Inverse(T) in N` for such an
    # `N` exactly when `T in N`; for a general node it holds when `T` is in
    # the bijective sub-set of `N`. For `Injective`/`Surjective`,
    # `_bijective_targets` is `(Bijective,)`: the inverse of a non-bijection
    # is not a function.
    if inv.forward is None:
        return issubclass(kinds.Identity, node)
    return any(
        is_kind(inv.forward, m, compute) for m in _bijective_targets(node)
    )


def _register_wrapper(source: type, member: tx.Callable) -> None:
    # A wrapper reasons about whichever node it is asked about, so the same
    # function serves them all -- but it must be registered against every
    # one of them: a question about `C` is answered by the nearest checker
    # registered against a *subset* of `C`, and only an exact registration
    # is nearer than the leaf checkers the wrapper must shadow.
    for node in kinds.all_sets():
        checker(source, node)(member)


_register_wrapper(SubspaceTransformation, _subspace_member)
_register_wrapper(Projection, _projection_member)
_register_wrapper(Bijection, _bijection_member)
_register_wrapper(Inverse, _inverse_member)


# ======================================================================
#
#                        K I N D   A L I A S E S
#
# ======================================================================
# The alias table names the kinds a [`kinds`][] NAME or SYMBOL does not
# already name, and only those: a wrapper, a field, a container. They have no
# kind node, because membership depends on their contents rather than on a
# set they belong to, so a name that resolves to one is matched by
# `isinstance`. One name resolves to one class -- `"field"` names the base
# the two field types share, not both of them.
#
# A friendlier spelling of a *node* does not belong here: it belongs in that
# node's `NAME`, which is what `TransformationKind.parse` reads. `"scaling"`
# and `"positivescaling"` are already names of the diagonal and positive
# diagonal sets, so they need no entry.
register_kind_alias("inverse", Inverse)
register_kind_alias("subspace", SubspaceTransformation)
register_kind_alias("projection", Projection)
register_kind_alias("meta", MetaTransformation)
register_kind_alias("field", TransformationField)
register_kind_alias("displacements", DisplacementField)
register_kind_alias("coordinates", CoordinatesField)
