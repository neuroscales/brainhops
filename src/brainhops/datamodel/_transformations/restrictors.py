"""The restriction and embedding rules of the built-in transformations.

Each rule restricts one type of transformation to a decoupled block of its
axes, or embeds one into a wider space. See the `restrict` module for the
contract they share. `rows` / `cols` are sorted output / input positions
of the block, among the `no` outputs and `ni` inputs of `t`, and a
restriction returns `None` for an identity piece.
"""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from .base import Transformation
from .concrete import (
    Affine,
    Identity,
    Permutation,
    Scaling,
    TransformationField,
    Translation,
)
from .inverse import Inverse
from .meta import SubspaceTransformation
from .restrict import embed, embedder, restrict, restrictor
from .sequence import Sequence, _interpolates
from .utils import UNREADABLE, affine_matrix, axis_list

# ----------------------------------------------------------------------
#   RESTRICTION
# ----------------------------------------------------------------------


@restrictor
def _(
    t: Transformation, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Transformation]:
    # The affine family (`Affine`, `Linear`, `Rotation`), and any other
    # transformation read through its affine: the matrix sub-block. A
    # matrix-less transform is the identity. A transform with no affine
    # reading is dropped as the identity too: the factor pass only
    # restricts what its dependency reader could read, which refuses such
    # a transform first.
    return _affine_block(t, rows, cols)


@restrictor
def _(
    t: Identity, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> None:
    return None


@restrictor
def _(
    t: Scaling, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Scaling]:
    if t.scale is None:
        return None
    return Scaling(scale=np.asarray(t.scale)[cols])


@restrictor
def _(
    t: Translation, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Translation]:
    if t.translation is None:
        return None
    return Translation(translation=np.asarray(t.translation)[rows])


@restrictor
def _(
    t: Permutation, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Transformation]:
    if t.permutation is None:
        return None
    permutation = np.asarray(t.permutation)
    col_pos = {c: j for j, c in enumerate(cols)}
    new_perm = []
    for r in rows:
        source = int(permutation[r])
        if source not in col_pos:
            # A row reads outside the block's columns: fall back to the
            # affine sub-block.
            return _affine_block(t, rows, cols)
        new_perm.append(col_pos[source])
    return Permutation(permutation=np.asarray(new_perm, dtype=int))


@restrictor
def _(
    t: TransformationField,
    rows: tx.List[int],
    cols: tx.List[int],
    ni: int,
    no: int,
) -> Transformation:
    # A raw field couples the whole space, so it is its own single group
    # and is kept whole.
    return t


@restrictor
def _(
    t: Inverse, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Transformation]:
    forward = t.forward
    if forward is None:
        return None
    if len(rows) == no and len(cols) == ni:
        # The block covers the whole inverse: return the object itself, so
        # it stays lazy and cancels by identity.
        return t
    # A block-diagonal inverse: restrict the forward over the swapped
    # block and invert that, which stays lazy.
    inner = restrict(forward, cols, rows, no, ni)
    if inner is None:
        return None
    return inner.inverse()


@restrictor
def _(
    t: SubspaceTransformation,
    rows: tx.List[int],
    cols: tx.List[int],
    ni: int,
    no: int,
) -> tx.Optional[Transformation]:
    # A subspace acts on its named axes and passes every other axis through
    # (in order). Its piece for a block is the same subspace re-expressed
    # over the block's *local* axes: the acted axes that fall in the block
    # keep their inner (restricted, when the block holds only part of a
    # separable inner), and any pass-through axis the block also holds --
    # one coupled to the acted axes by another stage -- stays pass-through.
    in_axes = axis_list(t.input_axes)
    out_axes = axis_list(t.output_axes)
    inner = t.transformation
    interpolates = _interpolates(inner)
    if not in_axes or not out_axes:
        if interpolates:
            # Unreachable from the factor pass, whose reader refuses such a
            # subspace.
            return t
        return _affine_block(t, rows, cols)
    local_in = [j for j, a in enumerate(in_axes) if a in cols]
    local_out = [j for j, a in enumerate(out_axes) if a in rows]
    if not local_in and not local_out:
        # The block only holds pass-through axes of this subspace, which it
        # maps in order: an identity piece.
        return None
    whole = len(local_in) == len(in_axes) and len(local_out) == len(out_axes)
    if whole or interpolates:
        # The block holds every acted axis (an interpolating inner couples
        # all of them, so it is never split across blocks): keep the inner
        # as the very same object, so `Sub(warp) . Sub(warp^-1)` still
        # cancels by identity, materializing no field.
        piece = inner
    elif inner is None:
        piece = None
    else:
        # Partial overlap of a non-interpolating (hence separable) inner:
        # restrict the inner over its own local axes, which keeps the
        # cheaper type (a diagonal `Sub(Scaling, ...)` restricts to a
        # `Scaling`, not a rank-deficient affine sub-block).
        piece = restrict(
            inner, local_out, local_in, len(in_axes), len(out_axes)
        )
    sub_in = [cols.index(in_axes[j]) for j in local_in]
    sub_out = [rows.index(out_axes[j]) for j in local_out]
    if sub_in == list(range(len(cols))) and sub_out == list(range(len(rows))):
        # The acted axes are the whole block, in order: the piece is the
        # (restricted) inner itself.
        return piece
    if piece is None and sub_in == sub_out:
        # An identity inner over axes the block maps in order.
        return None
    return embed(piece, sub_in, sub_out, len(cols), len(rows))


@restrictor
def _(
    t: Sequence, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Transformation]:
    # A nested sub-chain (a subspace inner that composed only partially).
    # One that holds a field couples the whole space and is kept whole.
    # Otherwise it is read through its affine, like any transformation.
    if _interpolates(t):
        return t
    return _affine_block(t, rows, cols)


# ----------------------------------------------------------------------
#   EMBEDDING
# ----------------------------------------------------------------------


@embedder
def _(
    t: Transformation,
    in_axes: tx.List[int],
    out_axes: tx.List[int],
    ni: int,
    no: int,
) -> Transformation:
    # An affine-ish transform embeds as a plain affine over the wider space.
    # One that interpolates, or that has no affine reading, stays wrapped.
    if not _interpolates(t):
        matrix = affine_matrix(t)
        if matrix is not UNREADABLE:
            return embed_matrix(matrix, in_axes, out_axes, ni, no)
    return _wrap(t, in_axes, out_axes)


@embedder
def _(
    t: None, in_axes: tx.List[int], out_axes: tx.List[int], ni: int, no: int
) -> Affine:
    # The identity: a reindex of `in_axes` onto `out_axes`.
    return embed_matrix(None, in_axes, out_axes, ni, no)


@embedder
def _(
    t: Inverse,
    in_axes: tx.List[int],
    out_axes: tx.List[int],
    ni: int,
    no: int,
) -> SubspaceTransformation:
    # A lazy inverse is left wrapped: reading its matrix would materialize
    # it.
    return _wrap(t, in_axes, out_axes)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _affine_block(
    t: Transformation, rows: tx.List[int], cols: tx.List[int]
) -> tx.Optional[Affine]:
    # The `(rows, cols)` sub-block of the affine matrix of `t`, with its
    # translation column. `None` for a matrix-less (identity) transform,
    # or for one with no affine reading (see the rule for `Transformation`).
    matrix = affine_matrix(t)
    if matrix is None or matrix is UNREADABLE:
        return None
    n_in = matrix.shape[1] - 1
    return Affine(matrix=matrix[np.ix_(rows, list(cols) + [n_in])])


def embed_matrix(
    matrix: tx.Optional[np.ndarray],
    in_axes: tx.List[int],
    out_axes: tx.List[int],
    ni: int,
    no: int,
) -> Affine:
    """The `(no, ni + 1)` affine that embeds `matrix` over some axes.

    It applies `matrix` (an identity when `None`) from the input axes
    `in_axes` to the output axes `out_axes`, and passes every other axis
    through, in order.
    """
    embedded = np.zeros((no, ni + 1))
    pass_in = [c for c in range(ni) if c not in in_axes]
    pass_out = [r for r in range(no) if r not in out_axes]
    for r, c in zip(pass_out, pass_in):
        embedded[r, c] = 1.0
    for a, r in enumerate(out_axes):
        if matrix is None:
            embedded[r, in_axes[a]] = 1.0
            continue
        for b, c in enumerate(in_axes):
            embedded[r, c] = matrix[a, b]
        embedded[r, ni] = matrix[a, -1]
    return Affine(matrix=embedded)


def _wrap(
    t: Transformation, in_axes: tx.List[int], out_axes: tx.List[int]
) -> SubspaceTransformation:
    return SubspaceTransformation(
        transformation=t,
        input_axes=np.asarray(in_axes, dtype=int),
        output_axes=np.asarray(out_axes, dtype=int),
    )
