"""Checkers for kind membership.

This module defines the checkers that [`is_kind`][] uses to decide
whether a transformation belongs to a kind, for example whether an
`Affine` is in fact a translation. The checkers register themselves
when the module is imported. This module relates to `check` in the same
way that `composers` relates to `compose`: `check` holds the
dispatcher, and this module holds the implementations. A checker is
written in the form in which it is called, and the transformation class
and kind node that it serves are read from its first two type hints:

```python
@checker
def _(query: Affine, kind: type[kinds.Translation], compute: bool) -> bool
```

A checker returns True only when it has shown membership, or, at the
analytic level, when membership can be presumed from the shape of a
matrix. For this reason, the answers of all applicable checkers can be
combined safely, and a single success is enough. Each checker is
registered for the smallest set that it decides, because [`is_kind`][]
also asks it about every larger set. The module also records how the
sets of the hierarchy relate to each other, which the checkers for
wrappers rely on, and it registers names for the field and wrapper
kinds.
"""

from functools import lru_cache, wraps

import typing_extensions as tx

from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import kinds

from ..base import Transformation
from ..concrete import (
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
from ..inverse import Inverse
from ..meta import (
    Bijection,
    MetaTransformation,
    Projection,
    SubspaceTransformation,
)
from ..operators import Sqrt
from .check import checker, is_kind, register_kind_alias


def identity_from(field: str) -> tx.Callable:
    """Make a checker return True when the attribute `field` is unset.

    An unset parameter denotes the identity map, which belongs to every
    set that a wrapped checker is registered against. No value is read, so
    the shortcut is valid at both levels.
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
# The identity map belongs to every set of the hierarchy, so these checkers
# answer for many more sets than their own node. For example, a Scaling without
# parameters is a translation, and a displacement field of zeros is affine.

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


# A CartesianField is the identity over its own grid, but it is reported as
# such only at the numeric level. At the analytic level, a set `shape` counts
# as a parameter, because treating the field as the identity would let a
# structural simplification replace the grid with an Identity and lose its
# sampling domain. Only the shape is read, so no meshgrid is built.


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
        # The mask keeps the off-diagonal entries and the translation column,
        # both of which vanish for a diagonal affine.
        return not bool((matrix * (1 - ab.eye(rows + 1)[:-1])).any())
    return False


# The subsets of the diagonal set, such as the invertible or isotropic
# diagonals, are decided by tests on the vector of diagonal entries. These
# tests are registered for every concrete class that can produce such a
# vector, so that a diagonal Linear is classified as precisely as the
# equivalent Scaling.


def _isotropic(scale: ArrayProtocol) -> bool:
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

# Invertibility is the only fact that is presumed from structure alone, in the
# same way that matrix transformations presume it from their shape. A Scaling
# is therefore presumed to be an invertible diagonal at the analytic level.
# InvertibleMultiplicative is excluded, because it also requires isotropy,
# which depends on the values.
_OPTIMISTIC = (kinds.InvertibleDiagonal,)

_DATAFIELD = {
    Scaling: "scale",
    Linear: "matrix",
    Affine: "matrix",
}


def _diagonal_values(
    query: Transformation,
) -> tx.Optional[ArrayProtocol]:
    # The diagonal is read directly rather than through is_kind. The subsets
    # being decided lie inside the diagonal set, so is_kind would ask their
    # checkers the very question that they are answering.
    scale = getattr(query, "scale", None)
    if scale is not None:
        return scale
    matrix = getattr(query, "matrix", None)
    if matrix is None:
        return None
    ab = get_array_backend(matrix)
    if isinstance(query, Affine):
        if bool(matrix[:, -1].any()):
            return None  # a translation, which is not diagonal
        matrix = matrix[:, :-1]
    rows, cols = matrix.shape
    if rows != cols:
        return None
    if bool((matrix * (1 - ab.eye(rows))).any()):
        return None
    return ab.diagonal(matrix)


def _diagonal_checker(source: type, node: type) -> tx.Callable:
    # The checker is registered explicitly rather than through its hints,
    # because the source and node are loop variables, which a generated
    # closure cannot use as annotations.
    fact = _DIAGONAL_FACTS[node]
    optimistic = node in _OPTIMISTIC

    @identity_from(_DATAFIELD[source])
    def _checker(query: Transformation, kind: type, compute: bool) -> bool:
        if not compute:
            # Only for a Scaling is the diagonal the whole parameter, whereas
            # a matrix would have to be read.
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
    # The identity is an even permutation, so an unset parameter never shows
    # that the permutation is odd.
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
    # The linear block may be rectangular.
    return bool((query.matrix[:, -1] == 0).all())


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
    # A conformal matrix satisfies A A' = c**2 I with c > 0. Since ||A||_F = c
    # sqrt(n), normalizing by the Frobenius norm reduces the test to
    # orthogonality.
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
    # The conformal test normalizes the matrix by a positive factor, which
    # does not change the sign of the determinant, so the determinant of the
    # matrix itself decides the orientation.
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
    """Return the dimensions `(ni, no)` of a matrix transformation, or None.

    The dimensions are read from the shape of the stored matrix, which is
    `(no, ni + 1)` for an `Affine` and `(no, ni)` for a `Linear`. No values
    are read.
    """
    matrix = getattr(t, "matrix", None)
    if matrix is None:
        return None
    no, cols = matrix.shape
    ni = cols - 1 if isinstance(t, Affine) else cols
    return ni, no


def _linear_part(t: Transformation) -> tx.Optional[ArrayProtocol]:
    # Return the linear block, whose rank decides injectivity and surjectivity
    # at the numeric level. The result is None when the transformation has no
    # matrix, or when it is invertible by declaration, such as a Permutation
    # or a Translation.
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
    # When the backend has no matrix_rank, the determinant of a square matrix
    # can still tell full rank from rank deficiency, and that distinction is
    # all the callers need.
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
# An Affine whose translation column is zero equals its linear part, so it
# belongs to exactly the sets that its linear part belongs to. Without this
# rule, an Affine could be recognized only in the sets that have an Affine
# checker of their own.


def _affine_is_linear_part(
    query: Affine, kind: tx.Type[kinds.TransformationKind], compute: bool
) -> bool:
    matrix = query.matrix
    if matrix is None:
        # A missing matrix is the identity. Not every linear set contains the
        # identity (OddPermutation does not), so the test is explicit.
        return issubclass(kinds.Identity, kind)
    if not compute:
        # Testing the translation column requires reading values.
        return False
    if bool(matrix[:, -1].any()):
        return False  # a genuine translation, which is not linear
    return is_kind(Linear(matrix=matrix[:, :-1]), kind, compute)


# The rule is registered only for sets in which an Affine has no checker of its
# own, because each pair of source and node holds a single checker, and the
# specialized checker knows more.
for _node in kinds.all_sets():
    if issubclass(_node, kinds.Linear) and (Affine, _node) not in is_kind:
        checker(Affine, _node)(_affine_is_linear_part)
del _node


# --- Shape-established facts ------------------------------------------
# Invertibility, injectivity and surjectivity are presumed from the shape of
# the matrix at the analytic level, and the rank confirms or retracts the
# presumption at the numeric level. For each matrix class, only the invertible
# node is registered, since queries about larger sets reach it. Injection and
# Surjection are registered separately, because a rectangular matrix can be
# one-to-one or onto without being invertible.

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
# The functions below record how the sets of the hierarchy behave under
# embedding, permutation, inversion and square roots. The checkers for wrappers
# rely on them, and they are kept here so that the `kinds` module describes
# only the sets themselves.


def _maximal(nodes: tx.Iterable[type]) -> tx.List[type]:
    # Keep the sets that are not contained in another set of the list.
    nodes = list(nodes)
    return [
        n
        for n in nodes
        if not any(m is not n and issubclass(n, m) for m in nodes)
    ]


@lru_cache(maxsize=None)  # noqa: UP033
def _embed_targets(node: type) -> tx.Tuple[type, ...]:
    # The block-diagonal map blockdiag(inner, I) belongs to `node` if and only
    # if `inner` belongs to one of the returned sets. The result is `node`
    # itself when `node` is embeddable, and otherwise the largest embeddable
    # sets that are strictly contained in `node`.
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
    # For a permutation P of the coordinates with the given parity, the map
    # P @ blockdiag(...) belongs to `node` if and only if the embedded map
    # belongs to one of the returned sets. The result is `node` itself when
    # `node` is closed under such permutations, and otherwise the largest
    # closed sets that are strictly contained in `node`.
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
    # The inverse inv(T) belongs to `node` if and only if T belongs to one of
    # the returned sets. The result is `node` itself when `node` contains only
    # bijections, since such sets are closed under inversion, and otherwise the
    # largest subsets of `node` that contain only bijections.
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
    """Return whether a reindexing permutes the coordinates evenly.

    The reindexing maps `input_axes[j]` to `output_axes[j]`, and an even
    permutation has determinant +1. The result is True when either list is
    None, and False when the lists differ in length or content, which
    sends the caller down the path for odd permutations.
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
# Membership of a wrapper depends on its contents, so each wrapper has one
# checker that recurses into the wrapped transformation and reasons from the
# facts about the hierarchy that are recorded above.


def _inner_member(
    inner: tx.Optional[Transformation], m: type, compute: bool
) -> bool:
    # A missing inner transformation is the identity.
    if inner is None:
        return issubclass(kinds.Identity, m)
    return is_kind(inner, m, compute)


def _subspace_member(
    sub: SubspaceTransformation, node: type, compute: bool
) -> bool:
    # A subspace transformation acts as P @ blockdiag(inner, I), where the
    # permutation P is present only when the axes are reindexed.
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
    # A projection that only drops axes is surjective, one that only creates
    # axes is injective, and one that does neither is the identity.
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
    # The question is passed on to the forward map, or to the inverse of the
    # backward map when no forward map is given. A Bijection is bijective by
    # declaration, so when a query names an invertible set such as
    # InvertibleAffine, it is enough for the map to belong to the unrestricted
    # set, such as Affine.
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
    # The checker never inverts the wrapped transformation, at either level,
    # and reasons only about the wrapped transformation itself. For Injection
    # or Surjection, the only target is Bijection, because the inverse of a map
    # that is not a bijection is not a function.
    if inv.forward is None:
        return issubclass(kinds.Identity, node)
    return any(
        is_kind(inv.forward, m, compute) for m in _bijective_targets(node)
    )


@lru_cache(maxsize=None)  # noqa: UP033
def _sqrt_targets(node: type) -> tx.Tuple[type, ...]:
    # A set qualifies when it is closed under the principal square root. Every
    # subset of GeneralizedPermutation that is not diagonal is treated as not
    # closed.
    def is_closed(n: type) -> bool:
        return not issubclass(n, kinds.GeneralizedPermutation) or issubclass(
            n, kinds.Diagonal
        )

    if is_closed(node):
        return (node,)
    return tuple(
        _maximal(
            n for n in kinds.all_sets() if issubclass(n, node) and is_closed(n)
        )
    )


def _sqrt_member(op: Sqrt, node: type, compute: bool) -> bool:
    # The square root itself is never computed here, because computing it is
    # the job of `compute`, and the root may be undefined.
    return any(is_kind(op.forward, m, compute) for m in _sqrt_targets(node))


def _register_wrapper(source: type, member: tx.Callable) -> None:
    # One function serves every node, but it must be registered for each node,
    # because only a registration for the same node takes precedence over the
    # leaf checkers that a typed wrapper inherits from its wrapped class.
    for node in kinds.all_sets():
        checker(source, node)(member)


_register_wrapper(SubspaceTransformation, _subspace_member)
_register_wrapper(Projection, _projection_member)
_register_wrapper(Bijection, _bijection_member)
_register_wrapper(Inverse, _inverse_member)
_register_wrapper(Sqrt, _sqrt_member)


# ======================================================================
#
#                        K I N D   A L I A S E S
#
# ======================================================================
# Wrapper, field and container kinds have no node, so their names are matched
# by isinstance. A friendlier spelling of a node belongs in the NAME attribute
# of that node instead.
register_kind_alias("inverse", Inverse)
register_kind_alias("subspace", SubspaceTransformation)
register_kind_alias("projection", Projection)
register_kind_alias("meta", MetaTransformation)
register_kind_alias("field", TransformationField)
register_kind_alias("displacements", DisplacementField)
register_kind_alias("coordinates", CoordinatesField)
