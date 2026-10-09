"""Restriction and embedding rules for the built-in transformation types.

Each rule restricts one type to a decoupled block of axes, or embeds it into
a wider space, under the contract described in the `restrict` module. In a
restriction rule, `rows` and `cols` are the sorted positions of the block
among the `no` outputs and the `ni` inputs of `t`, and a return value of
None stands for an identity piece.
"""

import numpy as np
import typing_extensions as tx

from brainhops.errors import RestrictionError

from ..base import Transformation
from ..concrete import (
    Affine,
    Identity,
    Permutation,
    Scaling,
    TransformationField,
    Translation,
)
from ..inverse import Inverse
from ..meta import SubspaceTransformation
from ..sequence import Sequence, _interpolates
from .factor import _element_ndim, _read_pattern
from .restrict import embed, embedder, restrict, restrictor
from .utils import UNREADABLE, affine_matrix, axis_list

# ----------------------------------------------------------------------
#   RESTRICTION
# ----------------------------------------------------------------------


@restrictor
def _(
    t: Transformation, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Transformation]:
    # The generic rule extracts the block from the affine matrix of `t`. An
    # element whose affine matrix cannot be read raises a RestrictionError
    # instead of being dropped.
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
            # This row reads an axis outside the block, so the piece cannot
            # stay a Permutation and is read as an affine block instead.
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
    # A plain field couples every axis with every other axis, so it forms a
    # single group and is kept whole.
    return t


@restrictor
def _(
    t: Inverse, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Transformation]:
    forward = t.forward
    if forward is None:
        return None
    if len(rows) == no and len(cols) == ni:
        # Returning the same object keeps the inverse lazy, and the inverse
        # still cancels with its forward transformation by object identity.
        return t
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
    # The piece is the same subspace, expressed over the local axes of the
    # block. The block may contain axes that this subspace only passes
    # through, because another stage couples them to the acted axes, and
    # those axes still pass through in the piece.
    in_axes = axis_list(t.input_axes)
    out_axes = axis_list(t.output_axes)
    inner = t.transformation
    interpolates = _interpolates(inner)
    if not in_axes or not out_axes:
        if interpolates:
            # The pattern reader of the factor pass refuses such a subspace,
            # so the factor pass never reaches this case.
            return t
        return _affine_block(t, rows, cols)
    local_in = [j for j, a in enumerate(in_axes) if a in cols]
    local_out = [j for j, a in enumerate(out_axes) if a in rows]
    if not local_in and not local_out:
        return None
    whole = len(local_in) == len(in_axes) and len(local_out) == len(out_axes)
    if whole or interpolates:
        # An inner transformation that interpolates couples all of its axes,
        # so it is never split. Keeping it as the same object lets
        # Sub(warp) . Sub(warp^-1) cancel by object identity without
        # materializing a field.
        piece = inner
    elif inner is None:
        piece = None
    else:
        # An inner transformation that does not interpolate can be split.
        # Restricting it keeps the cheaper type, so a diagonal Sub(Scaling)
        # gives a Scaling rather than a rank-deficient affine block.
        piece = restrict(
            inner, local_out, local_in, len(in_axes), len(out_axes)
        )
    sub_in = [cols.index(in_axes[j]) for j in local_in]
    sub_out = [rows.index(out_axes[j]) for j in local_out]
    if sub_in == list(range(len(cols))) and sub_out == list(range(len(rows))):
        return piece
    if piece is None and sub_in == sub_out:
        return None
    return embed(piece, sub_in, sub_out, len(cols), len(rows))


@restrictor
def _(
    t: Sequence, rows: tx.List[int], cols: tx.List[int], ni: int, no: int
) -> tx.Optional[Transformation]:
    # A nested sub-chain is either the partially composed inner
    # transformation of a subspace or the forward transformation of a lazy
    # inverse. The block is followed through the chain using the dependency
    # pattern of each member. A member that couples the block to other axes
    # raises a RestrictionError instead of being cut incorrectly.
    if len(rows) == no and len(cols) == ni:
        return t
    pieces: tx.List[Transformation] = []
    block, ndim = list(cols), ni
    for member in t.transformations or []:
        dims = _element_ndim(member, ndim)
        if dims is None or dims[0] != ndim:
            raise RestrictionError("Cannot read a member of the sequence")
        m_ni, m_no = dims
        pattern = _read_pattern(member, m_ni, m_no)
        if pattern is None or pattern.shape != (m_no, m_ni):
            raise RestrictionError("Cannot read a member of the sequence")
        inside = np.zeros(m_ni, dtype=bool)
        inside[block] = True
        reads_block = pattern[:, inside].any(axis=1)
        if pattern[np.ix_(reads_block, ~inside)].any():
            raise RestrictionError(
                "The block is coupled to other axes inside the sequence"
            )
        out_block = np.nonzero(reads_block)[0].tolist()
        if block and out_block:
            piece = restrict(member, out_block, block, m_ni, m_no)
            if piece is not None:
                pieces.append(piece)
        block, ndim = out_block, m_no
    if ndim != no or block != list(rows):
        raise RestrictionError(
            "The block is coupled to other axes inside the sequence"
        )
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    return Sequence(transformations=pieces)


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
    if not _interpolates(t):
        matrix = affine_matrix(t)
        if matrix is not UNREADABLE:
            return embed_matrix(matrix, in_axes, out_axes, ni, no)
    return _wrap(t, in_axes, out_axes)


@embedder
def _(
    t: None, in_axes: tx.List[int], out_axes: tx.List[int], ni: int, no: int
) -> Affine:
    return embed_matrix(None, in_axes, out_axes, ni, no)


@embedder
def _(
    t: Inverse,
    in_axes: tx.List[int],
    out_axes: tx.List[int],
    ni: int,
    no: int,
) -> SubspaceTransformation:
    # Reading the matrix of a lazy inverse would materialize it.
    return _wrap(t, in_axes, out_axes)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _affine_block(
    t: Transformation, rows: tx.List[int], cols: tx.List[int]
) -> tx.Optional[Affine]:
    matrix = affine_matrix(t)
    if matrix is UNREADABLE:
        raise RestrictionError(
            f"Cannot restrict a {type(t).__name__}, which has no affine "
            "reading"
        )
    if matrix is None:
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
    """Embed an affine matrix over some axes into a wider space.

    The returned Affine has a `(no, ni + 1)` matrix that applies `matrix` from
    `in_axes` to `out_axes` and passes the other axes through in order. When
    `matrix` is None, it stands for the identity, and the result reindexes
    `in_axes` onto `out_axes`.
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
