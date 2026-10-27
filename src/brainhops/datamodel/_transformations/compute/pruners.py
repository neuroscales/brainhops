"""Pattern readers and block restrictors for the built-in types.

The pruning of the `prune` module removes the axes that a chain does
not need, and the factor pass groups the axes that transform together.
Both read which outputs of each element read which inputs, and both cut
elements down to some of their outputs and the inputs that these read.
This module registers, for each built-in type, the rule that reads that
dependency pattern and the rule that performs the cut.

A pattern reader is called as `f(t, policy)`. Under the analytic policy
it reads only the type and the structure of `t`, so that a [`Scaling`][]
reads as diagonal and an [`Affine`][] as fully coupled. Under the
numeric policy it also reads the exact zeros of a matrix. A block
restrictor is called as `f(t, inputs, outputs, policy)` with a closed
block, in which the outputs read only the inputs, and it keeps the type
of `t` when that type can represent the block. Both kinds of rule
receive a transformation that states its numbers of axes.

A field reads as fully coupled and its restrictor declines, because a
field cannot yet hold a number of components that differs from the
dimension of its grid. A field inside a [`SubspaceTransformation`][]
still passes the other axes through.

A lazy inverse is never computed under the analytic policy. Its pattern
is then bounded by the reflexive and transitive closure of the pattern
of its forward transformation, because the inverse of a matrix is a
polynomial in that matrix by the Cayley-Hamilton theorem, so that an
output of the inverse can read an input only along a path of the forward
pattern. A block of the inverse is cut as the lazy inverse of a block of
the forward transformation when the forward pattern shows that this is
exact, and otherwise the rule declines. Under the numeric policy, the
coefficients of the inverse of a matrix transformation are read for its
pattern, and the inverse stays lazy. It is computed only when it is cut
and no lazy block is exact.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.datamodel.systems import CoordinateSystem

from ..base import Transformation
from ..concrete import (
    Affine,
    Identity,
    Linear,
    Permutation,
    Scaling,
    Translation,
)
from ..derived import DerivedTransformation
from ..inverse import Inverse
from ..meta import (
    Bijection,
    Projection,
    SubspaceTransformation,
)
from ..multiscale import MultiscaleField
from ..sequence import Sequence
from .prune import (
    block_restrictor,
    close_counts,
    counts_of,
    pattern_reader,
    read_pattern,
    restrict_block,
    stage_counts,
    unnamed_system,
)
from .simplify import SimplifyTable
from .utils import UNREADABLE, axis_list

NUMERIC = SimplifyPolicy.numeric

# ======================================================================
#
#                       P A T T E R N   R E A D E R S
#
# ======================================================================


@pattern_reader
def _(t: Transformation, policy: SimplifyTable) -> tx.Any:
    # A type without a rule of its own counts as fully coupled.
    return UNREADABLE


@pattern_reader
def _(t: DerivedTransformation, policy: SimplifyTable) -> tx.Any:
    # A lazy operator such as a square root is never resolved to read
    # its pattern, so it counts as fully coupled.
    return UNREADABLE


@pattern_reader
def _(t: MultiscaleField, policy: SimplifyTable) -> tx.Any:
    return UNREADABLE


@pattern_reader
def _(t: Identity, policy: SimplifyTable) -> tx.Any:
    n, m = counts_of(t)
    n = n if n is not None else m
    return UNREADABLE if n is None else np.eye(n, dtype=bool)


@pattern_reader
def _(t: Translation, policy: SimplifyTable) -> tx.Any:
    n, _ = counts_of(t)
    return UNREADABLE if n is None else np.eye(n, dtype=bool)


@pattern_reader
def _(t: Scaling, policy: SimplifyTable) -> tx.Any:
    n, _ = counts_of(t)
    if n is None:
        return UNREADABLE
    if policy.resolve(t) is NUMERIC and t.scale is not None:
        return np.diag(np.asarray(t.scale) != 0)
    return np.eye(n, dtype=bool)


@pattern_reader
def _(t: Permutation, policy: SimplifyTable) -> tx.Any:
    if t.permutation is None:
        n, _ = counts_of(t)
        return UNREADABLE if n is None else np.eye(n, dtype=bool)
    return _permutation_pattern(axis_list(t.permutation))


@pattern_reader
def _(t: Linear, policy: SimplifyTable) -> tx.Any:
    return _matrix_pattern(t, policy, homogeneous=False)


@pattern_reader
def _(t: Affine, policy: SimplifyTable) -> tx.Any:
    return _matrix_pattern(t, policy, homogeneous=True)


@pattern_reader
def _(t: Projection, policy: SimplifyTable) -> tx.Any:
    sources = _projection_sources(t)
    if sources is None:
        return UNREADABLE
    n_in, _ = counts_of(t)
    pattern = np.zeros((len(sources), n_in), dtype=bool)
    for o, i in enumerate(sources):
        if i is not None:
            pattern[o, i] = True
    return pattern


@pattern_reader
def _(t: SubspaceTransformation, policy: SimplifyTable) -> tx.Any:
    n_in, n_out = counts_of(t)
    if n_in is None or n_out is None:
        return UNREADABLE
    if t.input_axes is None or t.output_axes is None:
        return UNREADABLE
    in_axes, out_axes = axis_list(t.input_axes), axis_list(t.output_axes)
    pass_in = [i for i in range(n_in) if i not in in_axes]
    pass_out = [o for o in range(n_out) if o not in out_axes]
    if len(pass_in) != len(pass_out):
        return UNREADABLE
    inner = t.transformation
    shape = (len(out_axes), len(in_axes))
    if inner is None:
        if shape[0] != shape[1]:
            return UNREADABLE
        block = np.eye(shape[0], dtype=bool)
    else:
        block = UNREADABLE
        closed = close_counts(inner, shape[1], shape[0])
        if closed is not None:
            block = read_pattern(closed, policy)
        if block is UNREADABLE or np.shape(block) != shape:
            # An inner transformation that cannot be read couples the
            # axes it acts on, and the other axes still pass through.
            block = np.ones(shape, dtype=bool)
    pattern = np.zeros((n_out, n_in), dtype=bool)
    pattern[np.ix_(out_axes, in_axes)] = block
    pattern[pass_out, pass_in] = True
    return pattern


@pattern_reader
def _(t: Sequence, policy: SimplifyTable) -> tx.Any:
    # The pattern of a nested chain is the product of the patterns of
    # its members, along the chain.
    members = list(t.transformations or [])
    counts = stage_counts(members)
    if counts is None:
        return UNREADABLE
    pattern = np.eye(counts[0], dtype=bool)
    for k, member in enumerate(members):
        shape = (counts[k + 1], counts[k])
        closed = close_counts(member, counts[k], counts[k + 1])
        part = UNREADABLE if closed is None else read_pattern(closed, policy)
        if part is UNREADABLE or np.shape(part) != shape:
            part = np.ones(shape, dtype=bool)
        pattern = (part.astype(int) @ pattern.astype(int)) > 0
    return pattern


@pattern_reader
def _(t: Bijection, policy: SimplifyTable) -> tx.Any:
    if t.forward is None:
        return UNREADABLE
    return read_pattern(t.forward, policy)


@pattern_reader
def _(t: Inverse, policy: SimplifyTable) -> tx.Any:
    forward = t.forward
    n_in, n_out = counts_of(t)
    if forward is None:
        if n_in is None or n_in != n_out:
            return UNREADABLE
        return np.eye(n_in, dtype=bool)
    if isinstance(t, Permutation):
        # The inverse of a permutation is the inverse permutation, whose
        # vector is part of its structure.
        if t.permutation is not None:
            return _permutation_pattern(axis_list(t.permutation))
    if isinstance(t, (Scaling, Translation)):
        return read_pattern(forward, policy)
    if policy.resolve(t) is NUMERIC and isinstance(t, (Affine, Linear)):
        # The coefficients of the inverse are computed and cached on the
        # forward transformation, and the inverse stays lazy.
        matrix = t.matrix
        if matrix is not None:
            matrix = np.asarray(matrix)
            linear = matrix[:, :-1] if isinstance(t, Affine) else matrix
            return linear != 0
    pattern = read_pattern(forward, policy)
    if pattern is UNREADABLE:
        return UNREADABLE
    pattern = np.asarray(pattern, dtype=bool)
    if pattern.shape[0] != pattern.shape[1]:
        return UNREADABLE
    return _closure(pattern)


# ======================================================================
#
#                     B L O C K   R E S T R I C T O R S
#
# ======================================================================

_Axes = tx.List[int]


@block_restrictor
def _(
    t: Transformation, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    # A type without a rule of its own, such as a field, declines.
    return None


@block_restrictor
def _(
    t: DerivedTransformation,
    inputs: _Axes,
    outputs: _Axes,
    policy: SimplifyTable,
) -> tx.Optional[Transformation]:
    return None


@block_restrictor
def _(
    t: MultiscaleField, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    return None


@block_restrictor
def _(
    t: Bijection, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    # The cut of the backward map of a closed block is not the inverse
    # of the cut of the forward map, so a bijection keeps all its axes.
    return None


@block_restrictor
def _(
    t: Identity, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if not outputs:
        return _empty(inputs)
    return Identity() if inputs == outputs else None


@block_restrictor
def _(
    t: Translation, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if not outputs:
        return _empty(inputs)
    if inputs != outputs:
        return None
    if t.translation is None:
        return Identity()
    return Translation(translation=np.asarray(t.translation)[outputs])


@block_restrictor
def _(
    t: Scaling, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if not outputs:
        return _empty(inputs)
    if t.scale is None:
        return Identity() if inputs == outputs else None
    scale = np.asarray(t.scale)
    if not inputs:
        return _creation([0.0] * len(outputs))
    if inputs != outputs:
        # A zero scale makes its output constant and leaves its input
        # unread, so a block of a scaling may have different axes on
        # its two sides. Such a block is cut as a matrix.
        return _matrix_block(np.diag(scale), inputs, outputs, Linear)
    return Scaling(scale=scale[outputs])


@block_restrictor
def _(
    t: Permutation, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if not outputs:
        return _empty(inputs)
    if t.permutation is None:
        return Identity() if inputs == outputs else None
    permutation = axis_list(t.permutation)
    sources = [permutation[o] for o in outputs]
    if sorted(sources) != inputs:
        return None
    local = [inputs.index(s) for s in sources]
    if local == list(range(len(local))):
        return Identity()
    return Permutation(permutation=np.asarray(local, dtype=int))


@block_restrictor
def _(
    t: Linear, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    # The block of a rotation is not a rotation in general, so every
    # linear map is cut into a Linear.
    if not outputs:
        return _empty(inputs)
    if t.matrix is None:
        return Identity() if inputs == outputs else None
    if not inputs:
        return _creation([0.0] * len(outputs))
    return _matrix_block(np.asarray(t.matrix), inputs, outputs, Linear)


@block_restrictor
def _(
    t: Affine, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if not outputs:
        return _empty(inputs)
    if t.matrix is None:
        return Identity() if inputs == outputs else None
    matrix = np.asarray(t.matrix)
    if not inputs:
        # Outputs that read no input are created at their translation.
        return _creation(list(matrix[outputs, -1]))
    columns = list(inputs) + [matrix.shape[1] - 1]
    return Affine(matrix=matrix[np.ix_(outputs, columns)])


@block_restrictor
def _(
    t: Projection, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if not outputs:
        return _empty(inputs)
    sources = _projection_sources(t)
    if sources is None:
        return None
    read = [sources[o] for o in outputs if sources[o] is not None]
    if any(i not in inputs for i in read):
        return None
    values = dict(zip(axis_list(t.created), [float(v) for v in t.at]))
    dropped = [p for p, i in enumerate(inputs) if i not in read]
    created = [q for q, o in enumerate(outputs) if sources[o] is None]
    if not dropped and not created:
        return Identity()
    return Projection(
        dropped=dropped,
        created=created,
        at=[values[outputs[q]] for q in created],
        input=unnamed_system(len(inputs)),
    )


@block_restrictor
def _(
    t: SubspaceTransformation,
    inputs: _Axes,
    outputs: _Axes,
    policy: SimplifyTable,
) -> tx.Optional[Transformation]:
    n_in, n_out = counts_of(t)
    if n_in is None or n_out is None:
        return None
    if t.input_axes is None or t.output_axes is None:
        return None
    in_axes, out_axes = axis_list(t.input_axes), axis_list(t.output_axes)
    pass_in = [i for i in range(n_in) if i not in in_axes]
    pass_out = [o for o in range(n_out) if o not in out_axes]
    if len(pass_in) != len(pass_out):
        return None
    # An axis that passes through is kept on both sides or on neither,
    # so that the axes that still pass through are paired in order.
    if any((i in inputs) != (o in outputs) for i, o in zip(pass_in, pass_out)):
        return None
    inner_in = [j for j, a in enumerate(in_axes) if a in inputs]
    inner_out = [j for j, a in enumerate(out_axes) if a in outputs]
    inner = t.transformation
    if not inner_out:
        return Identity() if not inner_in else None
    if len(inner_in) == len(in_axes) and len(inner_out) == len(out_axes):
        piece = inner
    elif inner is None:
        if inner_in != inner_out:
            return None
        piece = None
    else:
        closed = close_counts(inner, len(in_axes), len(out_axes))
        if closed is None:
            return None
        piece = restrict_block(closed, inner_in, inner_out, policy)
        if piece is None:
            return None
    new_in = [inputs.index(in_axes[j]) for j in inner_in]
    new_out = [outputs.index(out_axes[j]) for j in inner_out]
    if (
        piece is not None
        and new_in == list(range(len(inputs)))
        and new_out == list(range(len(outputs)))
    ):
        # Nothing passes through any more, so the piece is the inner
        # transformation itself.
        return piece
    if (piece is None or isinstance(piece, Identity)) and new_in == new_out:
        return Identity()
    return SubspaceTransformation(
        None if isinstance(piece, Identity) else piece,
        np.asarray(new_in, dtype=int),
        np.asarray(new_out, dtype=int),
        input=_restricted_system(t.input, inputs),
        output=_restricted_system(t.output, outputs),
    )


@block_restrictor
def _(
    t: Sequence, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    # A nested chain is cut member by member, from its end. Each member
    # is cut down to the outputs that the next member reads, with the
    # patterns that the policy lets the members show.
    if not outputs:
        return _empty(inputs)
    members = list(t.transformations or [])
    counts = stage_counts(members)
    if counts is None:
        return None
    pieces: tx.List[Transformation] = []
    need = list(outputs)
    for k in reversed(range(len(members))):
        closed = close_counts(members[k], counts[k], counts[k + 1])
        if closed is None or not need:
            return None
        shape = (counts[k + 1], counts[k])
        pattern = read_pattern(closed, policy)
        if pattern is UNREADABLE or np.shape(pattern) != shape:
            pattern = np.ones(shape, dtype=bool)
        read = np.nonzero(np.asarray(pattern)[need].any(axis=0))[0].tolist()
        if need == list(range(shape[0])) and read == list(range(shape[1])):
            piece = members[k]
        else:
            piece = restrict_block(closed, read, need, policy)
            if piece is None:
                return None
        pieces.insert(0, piece)
        need = read
    if need != list(inputs):
        return None
    pieces = [p for p in pieces if not isinstance(p, Identity)]
    if not pieces:
        return Identity()
    if len(pieces) == 1:
        return pieces[0]
    return Sequence(transformations=pieces)


@block_restrictor
def _(
    t: Inverse, inputs: _Axes, outputs: _Axes, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    forward = t.forward
    if not outputs:
        return _empty(inputs)
    if forward is None:
        return Identity() if inputs == outputs else None
    lazy = _lazy_block(forward, inputs, outputs, policy)
    if lazy is not None:
        return lazy
    if policy.resolve(t) is NUMERIC and isinstance(
        t, (Affine, Linear, Scaling, Translation, Permutation)
    ):
        # The inverse of a matrix transformation is computed and then
        # cut, because nothing smaller can be written lazily. The
        # analytic policy never computes it, so the rule declines.
        return restrict_block(t._materialize(), inputs, outputs, policy)
    return None


# ======================================================================
#
#                             H E L P E R S
#
# ======================================================================


def _empty(inputs: tx.List[int]) -> tx.Optional[Transformation]:
    # The empty block of a transformation that can be cut is the
    # identity on no axis, which tells the pruning that the
    # transformation can be removed when none of its outputs is needed.
    return None if inputs else Identity()


def _creation(values: tx.List[float]) -> Projection:
    # Return the projection from no axis that creates one axis at each
    # value, which is the block of outputs that read no input.
    return Projection(
        created=list(range(len(values))),
        at=[float(v) for v in values],
        input=unnamed_system(0),
    )


def _lazy_block(
    forward: Transformation,
    inputs: tx.List[int],
    outputs: tx.List[int],
    policy: SimplifyTable,
) -> tx.Optional[Transformation]:
    # Return the block of the inverse of `forward` as the lazy inverse
    # of a block of `forward`, or None when the pattern of `forward`
    # does not show that this is exact. The inputs of the inverse are
    # outputs of `forward`, and its outputs are inputs of `forward`.
    pattern = read_pattern(forward, policy)
    n_in, n_out = counts_of(forward)
    if pattern is UNREADABLE or np.shape(pattern) != (n_out, n_in):
        return None
    pattern = np.asarray(pattern, dtype=bool)
    other_rows = [i for i in range(n_out) if i not in inputs]
    other_cols = [j for j in range(n_in) if j not in outputs]
    block = None
    if (
        len(inputs) == len(outputs)
        and not pattern[np.ix_(inputs, other_cols)].any()
        and not pattern[np.ix_(other_rows, outputs)].any()
    ):
        # The forward map is block diagonal, so the block of its inverse
        # is the inverse of its block.
        block = restrict_block(forward, outputs, inputs, policy)
    elif inputs == outputs and not pattern[np.ix_(inputs, other_cols)].any():
        # The forward outputs of the block read only its inputs, so the
        # forward map is block triangular, and the block of its inverse
        # over these axes is again the inverse of its block.
        block = restrict_block(forward, inputs, outputs, policy)
    if block is None or isinstance(block, Identity):
        return block
    return block.inverse()


def _permutation_pattern(permutation: tx.List[int]) -> np.ndarray:
    # Output o of a permutation reads input permutation[o].
    n = len(permutation)
    pattern = np.zeros((n, n), dtype=bool)
    pattern[list(range(n)), permutation] = True
    return pattern


def _matrix_pattern(
    t: tx.Any, policy: SimplifyTable, homogeneous: bool
) -> tx.Any:
    # A matrix is fully coupled under the analytic policy. Under the
    # numeric policy, its exact zeros are read.
    n_in, n_out = counts_of(t)
    if n_in is None or n_out is None:
        return UNREADABLE
    if t.matrix is None:
        return np.eye(n_in, dtype=bool) if n_in == n_out else UNREADABLE
    if policy.resolve(t) is NUMERIC:
        matrix = np.asarray(t.matrix)
        linear = matrix[:, :-1] if homogeneous else matrix
        return linear != 0
    return np.ones((n_out, n_in), dtype=bool)


def _matrix_block(
    matrix: np.ndarray,
    inputs: tx.List[int],
    outputs: tx.List[int],
    cls: tx.Type[Transformation],
) -> Transformation:
    return cls(matrix=matrix[np.ix_(outputs, inputs)])


def _projection_sources(t: Projection) -> tx.Optional[tx.List[tx.Any]]:
    # Return, for each output of a projection, the input that it reads,
    # or None for an output that the projection creates. None is
    # returned when the numbers of axes are unknown or do not fit.
    n_in, n_out = counts_of(t)
    if n_in is None or n_out is None:
        return None
    dropped, created = axis_list(t.dropped), axis_list(t.created)
    kept = [i for i in range(n_in) if i not in dropped]
    rows = [o for o in range(n_out) if o not in created]
    if len(kept) != len(rows):
        return None
    sources: tx.List[tx.Any] = [None] * n_out
    for o, i in zip(rows, kept):
        sources[o] = i
    return sources


def _closure(pattern: np.ndarray) -> np.ndarray:
    # Return the reflexive and transitive closure of a square pattern,
    # read as a relation from an output to the input of the same index.
    closure = pattern | np.eye(pattern.shape[0], dtype=bool)
    while True:
        wider = closure | ((closure.astype(int) @ closure.astype(int)) > 0)
        if np.array_equal(wider, closure):
            return closure
        closure = wider


def _restricted_system(
    system: tx.Optional[CoordinateSystem], axes: tx.List[int]
) -> CoordinateSystem:
    # Restrict a closed system to some of its axes, or describe the axes
    # by an unnamed closed system when the system is not closed.
    if get_axes(system).ndim is not None:
        return system.restrict(axes)
    return unnamed_system(len(axes))
