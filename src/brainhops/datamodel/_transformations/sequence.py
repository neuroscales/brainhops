# stdlib
from collections.abc import MutableSequence as AbcMutableSequence
from collections.abc import Sequence as AbcSequence

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# api
from brainhops._core.properties import smartproperty
from brainhops.datamodel import hierarchy
from brainhops.datamodel.systems import CoordinateSystem

# internals
from . import registries
from .base import Transformation
from .compose import compose
from .concrete import (
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    is_identity,
)
from .errors import CompositionError
from .inverse import Inverse
from .meta import SubspaceTransformation
from .registries import register_sequence

# typing
_ModePair = tx.Tuple[tx.Type[hierarchy.Transformation], tx.Optional[int]]
_ModeCls = tx.Union[str, tx.Type[hierarchy.Transformation]]
_ModeLike = tx.Union[tx.Tuple[_ModeCls, tx.Optional[int]], _ModeCls, int]
ModeLike = tx.Union[_ModeLike, tx.Iterable[_ModeLike]]


class SequenceMixin(AbcSequence):
    # Implements the abc.Sequence API, assuming the existence of a
    # `transformations` attribute that is a sequence of `Transformation`
    # objects.

    def __len__(self) -> int:
        return len(self.transformations or [])

    def __getitem__(self, index: tx.Union[int, slice]) -> Transformation:
        return self.transformations[index]

    def __delitem__(self, index: tx.Union[int, slice]) -> None:
        del self.transformations[index]

    def __iter__(self) -> tx.Iterator[Transformation]:
        return iter(self.transformations or [])


class MutableSequenceMixin(SequenceMixin, AbcMutableSequence):
    # Implements the abc.MutableSequence API, assuming the existence of a
    # a `transformations` attribute that is a mutable sequence of
    # `Transformation` objects.

    def __setitem__(
        self, index: tx.Union[int, slice], value: Transformation
    ) -> None:
        self.transformations[index] = value

    def insert(self, index: int, value: Transformation) -> None:
        if self.transformations is None:
            self.transformations = []
        self.transformations.insert(index, value)

    def append(self, value: Transformation) -> None:
        if self.transformations is None:
            self.transformations = []
        self.transformations.append(value)

    def extend(self, values: tx.List[Transformation]) -> None:
        if self.transformations is None:
            self.transformations = []
        self.transformations.extend(values)

    def clear(self) -> None:
        if self.transformations:
            self.transformations.clear()

    def pop(self, index: int = -1) -> Transformation:
        if self.transformations is None:
            raise IndexError("pop from empty sequence")
        return self.transformations.pop(index)

    def remove(self, value: Transformation) -> None:
        if self.transformations is None:
            raise ValueError("remove from empty sequence")
        self.transformations.remove(value)


class Sequence(SequenceMixin, Transformation):
    """A sequence of transformations.

    This is a base class shared by mutable and immutable sequences.

    It can also be used as a factory itself, in which case it returns
    a `MutableSequence`.

    !!! note
        Transformations in a sequence are listed in the order in which
        they are applied. It reads as the opposite order to function
        composition (or matrix multiplication), which may be confusing.

        * `Sequence([t1, t2, t3])(x)` is equivalent to `t3(t2(t1(x)))`.
        * `Sequence([t1, t2, t3]) @ x` is equivalent to `t3 @ t2 @ t1 @ x`.
    """

    parameter_names: tx.ClassVar[str] = "transformations"

    # --- attributes ---------------------------------------------------

    transformations: tx.Annotated[
        tx.Optional[tx.Sequence[Transformation]],
        tx.Doc(
            """
            A list of transformations, in the order in which they are
            applied to an input coordinate system.
            """
        ),
    ] = None

    @smartproperty
    def input(self) -> tx.Optional[CoordinateSystem]:
        if self.transformations:
            return self.transformations[0].input
        return None

    @smartproperty
    def output(self) -> tx.Optional[CoordinateSystem]:
        if self.transformations:
            return self.transformations[-1].output
        return None

    # --- factory ------------------------------------------------------

    def __new__(cls, *args, **kwargs) -> tx.Type[tx.Self]:
        if cls is Sequence:
            return MutableSequence(*args, **kwargs)
        return super().__new__(cls)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        cls = type(self)
        if self.transformations is None:
            return cls(input=self.output, output=self.input)
        # Only the endpoints this sequence was *given* are carried over,
        # swapped. Reading `self.output`/`self.input` would hand the
        # derived values to the constructor, which stores them as
        # declared ones; the reversed children derive the same answer
        # anyway.
        return cls(
            transformations=[
                t.inverse(compute=compute, **kwargs)
                for t in reversed(self.transformations)
            ],
            input=getattr(self, "_output", None),
            output=getattr(self, "_input", None),
        )

    def compute(self, mode: tx.Optional[ModeLike] = None) -> Transformation:
        """
        Compute the resulting transform of the sequence of transformations.

        If all transformations in the sequence are affine-like transformations,
        `compute()` returns an affine-like transform.

        If the first (= rightmost) transform in the sequence is a
        coordinate field, `compute()` returns a coordinate field.

        If the first (= rightmost) transform in the sequence is an
        affine-like transform, and the sequence contains at least one
        non-affine-like transform, `compute()` returns a sequence of two
        transformations:
        1. the composition of all affine-like transformations that appear
           before the first non-affine-like transform in the sequence, and
        2. the composition of all transformations in the sequence, starting
           from the first non-affine-like transform in the sequence.

        A sequence may contain a stored field that no sampling domain
        precedes. Computing that sequence folds the rest of the sequence
        into the field and returns a field, rather than sampling the field
        on a grid. The returned field is exact within the field of view of
        the stored field. Outside that field of view the result is an
        approximation, because the boundary condition extrapolates the
        coordinates that fall beyond the grid. A sequence that instead
        begins with a sampling domain evaluates the field on that domain,
        and its result is exact everywhere. Reslicing an image and
        [`Points.compute`][] both begin with such a domain.

        Parameters
        ----------
        mode : [list of] str, optional
            Types of transformations to compute.
            * If `None` (default): compute all transformations in the sequence.
            * If the name of a transformation type: compute only consecutive
              sequences of transformations that match the specified type.
        """
        mode = _ensure_proper_modes(mode)
        if not mode:
            return self  # No-op
        return _compute_sequence(self, mode=mode)

    def _flattened(self) -> tx.Self:
        # Flatten nested sequences into a single sequence, and propagate
        # the sequence's own input and output onto its first and last
        # transformations. Subclasses that must keep a fixed shape (such
        # as `Geometry`) override this.
        if self.transformations is None:
            return self
        inp, out = self.input, self.output
        flattened = []
        for i, t in enumerate(self.transformations):
            if i == 0 and t.input is None and inp is not None:
                t = t.to(input=inp)
            elif i == len(self) - 1 and t.output is None and out is not None:
                t = t.to(output=out)
            if isinstance(t, Sequence):
                # A `Geometry` child contributes its grid followed by its
                # transformation, which may itself still be a sequence. That
                # leaves one level of nesting in the splice, which is fine:
                # `_compute_sequence` re-flattens on the next pass.
                flattened.extend(t._flattened().transformations or [])
            else:
                flattened.append(t)
        return replace(self, transformations=flattened)


register_sequence(Sequence)


class MutableSequence(MutableSequenceMixin, Sequence):
    """A sequence of transformations.

    !!! note
        Transformations in a sequence are listed in the order in which
        they are applied. It reads as the opposite order to function
        composition (or matrix multiplication), which may be confusing.

        * `Sequence([t1, t2, t3])(x)` is equivalent to `t3(t2(t1(x)))`.
        * `Sequence([t1, t2, t3]) @ x` is equivalent to `t3 @ t2 @ t1 @ x`.
    """

    # NOTE: this makes `transformations` a list instead of any sequence
    transformations: tx.Annotated[
        tx.Optional[tx.List[Transformation]],
        tx.Doc(
            """
            A list of transformations, in the order in which they are
            applied to an input coordinate system.
            """
        ),
    ] = None


class ImmutableSequence(Sequence):
    """A [`Sequence`][] whose contents cannot be edited in place.

    Every in-place edit -- item assignment, deletion, insertion -- raises
    `TypeError`.
    """

    def _refuse_in_place_edit(self, *args: tx.Any) -> tx.NoReturn:
        raise TypeError(f"{type(self).__name__} cannot be edited in place.")

    __setitem__ = _refuse_in_place_edit
    __delitem__ = _refuse_in_place_edit
    insert = _refuse_in_place_edit

    # NOTE: this makes `transformations` a tuple instead of any sequence
    transformations: tx.Annotated[
        tx.Optional[tx.Tuple[Transformation, ...]],
        tx.Doc(
            """
            A tuple of transformations, in the order in which they are
            applied to an input coordinate system.
            """
        ),
    ] = None


# ======================================================================
#
#                             H E L P E R S
#
# ======================================================================

# ----------------------------------------------------------------------
#    SEQUENCE COMPUTATION
# ----------------------------------------------------------------------


def _compute_sequence(
    seq: Sequence,
    mode: tx.List[_ModePair],
    memo: tx.Optional[tx.Set[_ModePair]] = None,
) -> Transformation:
    # We optimize by recursively finding the subclasses of all the modes
    # specified. This allows us to combine similar transformations first
    # before combining other transformations. For example say the user
    # lists Affine as the mode. This will then recursively call this
    # function with all subclasses then all subclasses of subclasses etc.
    # in a depth first search fashion. This means if there are translations
    # that are next to each other in the sequence it will combine the
    # translations before combining any of the affines.

    # --- If we are called from the public method, `mode` is a `list`.
    # > Simplify to a fixpoint, then recurse per mode with a memo.
    if memo is None:
        # Each simplification pass can expose a new adjacent
        # transform/inverse pair. Dropping a strictly interior grid
        # can make a pair adjacent, and so can composing a run. So the
        # cancellation runs *after* the grids are dropped, and re-runs
        # after each composition pass, until a pass no longer shrinks the
        # sequence. Counts are measured on the fully unnested element list,
        # so the loop terminates even when a composition folds into a
        # nested sequence.
        while True:
            # Flatten without rebuilding any endpoint, so a transform stays
            # the same object that its inverse names. Cancellation tests
            # that link by identity, and `_flattened` (used below to
            # propagate coordinate systems) would rebuild the first and
            # last elements and break it.

            # Reconcile any boundary where two adjacent transforms disagree
            # on the system they share, before those transforms are
            # flattened and composed. The composers assume compatible
            # systems, so the bridge that reorders, rescales, or flips the
            # mismatched axes is inserted here. Bridging runs on the direct
            # children, because a nested sequence carries its endpoint
            # systems on itself and flattening would drop them. A bridge is
            # built from exactly invertible pieces, so two opposite bridges
            # cancel and simplify away.
            bridged = _insert_bridges(seq.transformations or [])
            flat = _unnest(bridged)
            before = len(flat)
            seq = replace(seq, transformations=flat)
            if not flat:
                # Empty sequence -> return
                # TODO/FIXME: return an Identity instead?
                return seq

            # Factor away any strictly interior grid before composing. An
            # interior `CartesianField` is the identity map over its grid,
            # and its neighbours overwrite those coordinates, so it is
            # redundant. The first and last elements define the sampling
            # domain and are left in place.
            seq = _drop_interior_grids(seq)
            seq = _cancel_adjacent_inverses(seq)
            if not isinstance(seq, Sequence):
                return seq

            # Compose adjacent subspace transforms over the same axes, so a
            # subspace-wrapped field meets its own subspace-wrapped inverse
            # and cancels by identity rather than being inverted numerically.
            # A full cancellation collapses the sequence to the identity.
            seq = _merge_adjacent_subspaces(seq, mode)
            if not isinstance(seq, Sequence):
                return seq

            # Propagate the sequence's own endpoints onto its first and
            # last elements, but only when it carries any, so the identity
            # link is preserved in the common case of an endpoint-less
            # composition.
            if seq.input is not None or seq.output is not None:
                seq = seq._flattened()
            submemo: tx.Set[_ModePair] = set()
            for submode in mode:
                seq = _compute_sequence(seq, submode, memo=submemo)
                if not isinstance(seq, Sequence):
                    return seq
            if len(_unnest(seq.transformations)) >= before:
                # No pass shrank the sequence, so a further cancellation
                # cannot either. Nothing left to simplify.
                return seq

    # --- Flatten sequence
    if not _is_flat(seq):
        seq = seq._flattened()

    # --- Check if nothing to do
    if not seq.transformations:
        return seq

    # --- Otherwise, `mode` is a single mode
    # > if the mode has been seen, return the sequence as is
    if mode in memo:
        return seq

    # --- Else compute all children of the mode
    children = _mode_children(mode)
    for child in children:
        seq = _compute_sequence(seq, child, memo=memo)
        if not isinstance(seq, Sequence):
            return seq

    # Mark that we've been through this mode
    # !!! DO NOT RETURN HERE
    memo.add(mode)

    # --- Compose transformations that belong to this mode in order

    # Ensure a list of transformations (and make a copy so we can pop it)
    inputs = list(getattr(seq, "transformations", [seq]))
    outputs = []

    # Compose any consecutive sequence that matches the mode
    while inputs:
        item = inputs.pop(0)
        if _matches_mode(item, mode):
            while inputs and _matches_mode(inputs[0], mode):
                next_input = inputs.pop(0)
                try:
                    # NOTE: we compose to the left ! (see sequence definition)
                    item = compose(next_input, item)
                except CompositionError:
                    # NOTE(YB):
                    # When does this happen? When we don't know how to adapt?
                    outputs.append(item)
                    item = next_input
        outputs.append(item)

    # Return
    if len(outputs) == 1:
        return outputs[0]
    return Sequence(transformations=outputs)


# ----------------------------------------------------------------------
#    MODE
# ----------------------------------------------------------------------


def _ensure_proper_modes(mode: ModeLike) -> tx.List[_ModePair]:
    """
    Convert any (list of) mode-like input into a list of (type, ndim) pairs.

    !!! note "An empty list of modes yields a no-op"
    """

    if mode is None:
        # Default case
        mode = [hierarchy.Transformation]
    elif isinstance(mode, (str, int, type)):
        # We know that these are single modes -> wrap them already
        mode = [mode]
    elif _is_proper_mode(mode):
        # Already a proper mode -> wrap it
        mode = [mode]

    # Convert each element to a proper mode = a (type, ndim) pair
    return [
        hierarchy.parseType(m) if not _is_proper_mode(m) else m for m in mode
    ]


def _is_proper_mode(mode: ModeLike) -> bool:
    """
    A proper mode is a tuple (type, ndim),
    where type is a subclass of `hierarchy.Transformation`
    and ndim is an int or None.
    """
    if not isinstance(mode, tuple):
        return False
    if len(mode) != 2:
        return False
    if not isinstance(mode[0], type):
        return False
    if not isinstance(mode[1], (int, type(None))):
        return False
    return True


def _mode_admits(t: Transformation, mode: tx.List[_ModePair]) -> bool:
    # Whether the current mode would compose a transform of this type. The
    # mode is the list of `(type, ndim)` pairs the simplification runs
    # under, and a transform belongs to the mode when it matches any pair.
    return any(_matches_mode(t, m) for m in mode)


def _mode_children(mode: _ModePair) -> list:
    children = []
    seen = set()
    cls, ndim = mode
    for child in cls.__subclasses__():
        if (child, ndim) not in seen:
            seen.add((child, ndim))
            children.append((child, ndim))
    return children


def _matches_mode(t: Transformation, mode: _ModePair) -> bool:
    # FIXME
    #   In many transforms, the ndim can be guessed from the content
    #   of the xform, even if the input/output spaces are not set
    #   (eg. the shape of the matri or the field).
    #
    #   The current implementation is a stricter bound.
    cls, ndim = mode
    if not isinstance(t, cls):
        return False
    if ndim is None:
        return True
    if t.input is None or t.output is None:
        return False
    if t.input.axes is None or t.output.axes is None:
        return False
    return len(t.input.axes) == ndim and len(t.output.axes) == ndim


def _drop_interior_grids(seq: Sequence) -> Sequence:
    """Remove every strictly interior grid from a sequence.

    A [`CartesianField`][] is the identity map over its grid. A grid that
    sits strictly between two other transformations is therefore
    redundant. The transformation before it and the transformation after
    it overwrite its coordinates, so `A @ grid @ B` computes the same
    field as `A @ B`. Each such interior grid is removed, which lets the
    two neighbours compose directly.

    The first element, the last element, and a standalone element are left
    in place. A grid in one of those positions defines the sampling domain
    onto which data is resampled, and removing it would lose that domain.
    A sequence of one or two transformations has no interior, so it is
    returned unchanged.

    Only a [`CartesianField`][] is removed, because only a grid is the
    identity map by construction. A general field of coordinates or a
    field of displacements carries its own values, which the neighbours do
    not reproduce, so such a field is kept wherever it appears.
    """
    xforms = seq.transformations or []
    if len(xforms) <= 2:
        return seq
    kept = [xforms[0]]
    for elem in xforms[1:-1]:
        if isinstance(elem, CartesianField) and is_identity(
            elem, compute=True
        ):
            continue
        kept.append(elem)
    kept.append(xforms[-1])
    if len(kept) == len(xforms):
        return seq
    return replace(seq, transformations=kept)


def _subspaces_mergeable(
    prev: SubspaceTransformation,
    nxt: SubspaceTransformation,
    mode: tx.List[_ModePair],
) -> bool:
    # Whether two adjacent subspace transforms over the same axes may be
    # merged under the current mode. Two kinds of merge are distinguished.
    #
    # A merge that cancels by identity materializes no field, so it is
    # always allowed, whatever the mode. This is the case that lets a
    # subspace-wrapped field meet its own subspace-wrapped inverse. A pair of
    # inner transforms where one is the inverse of the other cancels this
    # way, as does a pair where either inner is the identity.
    #
    # Any other merge composes the two inner transforms numerically, which
    # for two fields resamples one through the other. That work is done only
    # when the current mode would compose those inner transforms anyway, so a
    # restrictive mode such as `compute(mode="Affine")` does not silently
    # compose two fields.
    inner_prev = prev.transformation
    inner_nxt = nxt.transformation
    if inner_prev is None or inner_nxt is None:
        return True
    if _cancels(inner_prev, inner_nxt):
        return True
    return _mode_admits(inner_prev, mode) and _mode_admits(inner_nxt, mode)


def _merge_adjacent_subspaces(
    seq: Sequence, mode: tx.List[_ModePair]
) -> Transformation:
    # Compose adjacent subspace transforms that act on the same axes. Two
    # subspace transforms that meet, where the axes the first writes are the
    # axes the second reads, compose into one subspace transform over those
    # axes. A pair that composes to the identity is dropped. This lets a
    # subspace-wrapped field meet its own subspace-wrapped inverse and
    # cancel by identity, rather than the field being inverted numerically.
    #
    # A merge that would compose two inner transforms numerically runs only
    # when the current mode admits those inner types, so a restrictive mode
    # does not compose transforms it was told to leave alone. A cancellation
    # by identity is always allowed, whatever the mode.
    xforms = seq.transformations or []
    if len(xforms) < 2:
        return seq
    merged: tx.List[Transformation] = [xforms[0]]
    changed = False
    last_identity: tx.Optional[Identity] = None
    for nxt in xforms[1:]:
        prev = merged[-1]
        if (
            isinstance(prev, SubspaceTransformation)
            and isinstance(nxt, SubspaceTransformation)
            and prev.output_axes is not None
            and nxt.input_axes is not None
            and list(prev.output_axes) == list(nxt.input_axes)
            and _subspaces_mergeable(prev, nxt, mode)
        ):
            composed = compose(nxt, prev)
            merged.pop()
            if isinstance(composed, Identity):
                last_identity = composed
            else:
                merged.append(composed)
            changed = True
        else:
            merged.append(nxt)
    if not changed:
        return seq
    if not merged and last_identity is not None:
        # Every element cancelled, so the sequence is the identity the last
        # cancelling pair produced, with its endpoints, rather than an empty
        # sequence.
        return last_identity
    return replace(seq, transformations=merged)


def _normalize_inverse(t: Transformation) -> Transformation:
    # Expand a generic `Inverse` front-door into the typed inverse of the
    # transform it holds, so the sequence engine computes it and
    # cancellation recognizes it like any other inverse. An endpoint
    # override on the `Inverse` is carried onto the result. A typed inverse
    # (its `_inverseof` is set) is already such a result and is left as is:
    # `Inverse(forward=X)` becomes `X.inverse()`, which for a field or an
    # affine is the typed inverse whose `forward` is `X`.
    if not isinstance(t, Inverse) or t._inverseof is not None:
        return t
    if t.forward is None:
        return Identity(input=t.input, output=t.output)
    inv = t.forward.inverse()
    kwargs = {}
    if t.input is not None:
        kwargs["input"] = t.input
    if t.output is not None:
        kwargs["output"] = t.output
    return inv.to(**kwargs) if kwargs else inv


def _cancels(first: Transformation, second: Transformation) -> bool:
    # `first` is applied before `second`. The two cancel when `second` is
    # the inverse of `first`, or `first` is the inverse of `second`. An
    # inverse names the transform it undoes as its `forward`, so the test
    # is a plain identity check that materializes neither field. This
    # covers both a typed inverse and a generic `Inverse(forward=X)`.
    if isinstance(second, Inverse) and second.forward is first:
        return True
    if isinstance(first, Inverse) and first.forward is second:
        return True
    return False


def _cancel_adjacent_inverses(seq: Sequence) -> Transformation:
    # Remove adjacent transform/inverse pairs before any numeric inversion
    # or composition. A transform placed next to its own lazy inverse
    # annihilates it, and each removal can expose a new adjacent pair, so
    # the scan keeps a stack and cancels the top of the stack against the
    # next transform.
    flat = _unnest(seq.transformations)
    stack = []
    cancelled = False
    for t in flat:
        if stack and _cancels(stack[-1], t):
            stack.pop()
            cancelled = True
        else:
            stack.append(t)
    if not cancelled:
        return seq
    if not stack:
        # A sequence that cancels entirely is the identity from the input
        # of its first element to the output of its last element. The
        # sequence's own endpoints are usually unset, so the element
        # endpoints are used, falling back to the sequence's endpoints
        # where an element leaves one unset.
        first, last = flat[0], flat[-1]
        return Identity(
            input=first.input if first.input is not None else seq.input,
            output=last.output if last.output is not None else seq.output,
        )
    return replace(seq, transformations=stack)


# ----------------------------------------------------------------------
#    UTILS
# ----------------------------------------------------------------------


def _is_flat(self: Sequence) -> bool:
    # Check if the sequence is flat (does not contain any nested sequences).
    if self.transformations is None:
        return True
    return all(not isinstance(t, Sequence) for t in self.transformations)


def _unnest(transformations: tx.Optional[tx.List[Transformation]]) -> list:
    # Flatten nested sequences into a single list, without touching the
    # endpoints of any transform (unlike `_flatten`, which may rebuild the
    # first and last transform to propagate coordinate systems, and in
    # doing so would read a lazy field). A generic `Inverse` front-door is
    # expanded to its typed inverse along the way.
    flattened = []
    for t in transformations or []:
        t = _normalize_inverse(t)
        if isinstance(t, Sequence):
            flattened.extend(_unnest(t.transformations))
        else:
            flattened.append(t)
    return flattened


def _interpolates(xform: Transformation) -> bool:
    # Whether applying a transform resamples data through a spline. A
    # transform interpolates when, looking past a sequence, a subspace
    # wrapper, and an inverse, it reaches a displacement field or a
    # coordinate field that is not a plain grid. A `CartesianField` is the
    # identity map over its grid and reads no value off it, so it does not
    # interpolate. An affine, a permutation, and the like never interpolate.
    if xform is None:
        return False
    if isinstance(xform, Inverse):
        return _interpolates(xform.forward)
    if isinstance(xform, Sequence):
        return any(_interpolates(t) for t in (xform.transformations or []))
    if isinstance(xform, SubspaceTransformation):
        return _interpolates(xform.transformation)
    if isinstance(xform, CartesianField):
        return False
    if isinstance(xform, (DisplacementField, CoordinatesField)):
        return True
    return False


# ----------------------------------------------------------------------
#   ADAPTORS
# ----------------------------------------------------------------------


def _splice(spliced: tx.List[Transformation], nxt: Transformation) -> None:
    # Add `nxt` to the running list `spliced`, reconciling the boundary it
    # shares with the transform already at the end of the list. The adaptor
    # returns the pair with whatever bridge or subspace lift the boundary
    # needs already placed between them. The two transforms it contains are
    # never rebuilt, so a leaf stays the same object its inverse names and
    # the adjacent-inverse cancellation still links the two by identity.
    if not spliced:
        spliced.append(nxt)
        return
    prev = spliced[-1]
    pieces = list(
        registries.ADAPT(prev, nxt, allow_type_grouped_positional=True)
    )
    if pieces[0] is not prev:
        # `prev` was lifted into the fuller space of `nxt`, so the piece
        # that replaces it now presents a different left boundary. The old
        # `prev` is dropped and the lifted piece is re-spliced against
        # `prev`'s own left neighbour, which may in turn need reconciling.
        spliced.pop()
        _splice(spliced, pieces[0])
        spliced.extend(pieces[1:])
    else:
        spliced.extend(pieces[1:])


def _insert_bridges(
    transformations: tx.List[Transformation],
) -> tx.List[Transformation]:
    # Reconcile every boundary where two adjacent transforms disagree on
    # the system they share. The output system of one and the input system
    # of the next are reconciled by the adaptor, whose pieces are spliced in
    # at that boundary so the result still contains both transforms. A
    # boundary whose systems already agree, or where either system is
    # unspecified, is left alone. Reconciling runs before the sequence is
    # flattened, because a nested sequence carries its endpoint systems on
    # the sequence and not on the leaves that flattening would expose.
    # A boundary can hide inside a nested sequence or behind a generic
    # inverse, both of which the flattening later removes. So each nested
    # sequence has its own children bridged first, keeping its endpoints,
    # and each generic inverse is expanded to the typed inverse the
    # flattening would produce, so the boundary is read from the system
    # that inverse actually presents. Neither step rebuilds a leaf's
    # endpoints, so a transform stays the same object its inverse names and
    # the adjacent-inverse cancellation still links the two by identity.
    prepared: tx.List[Transformation] = []
    for t in transformations:
        t = _normalize_inverse(t)
        # Only a plain sequence is rebuilt with its children bridged. A
        # specialized sequence, such as a multiscale field or a geometry,
        # keeps its own shape and derives its elements, so its boundaries
        # are read through it rather than rebuilt. `Sequence` itself is a
        # factory that builds a `MutableSequence`, so a plain sequence is
        # exactly one of the two concrete kinds.
        if type(t) in (MutableSequence, ImmutableSequence):
            t = replace(
                t, transformations=_insert_bridges(t.transformations or [])
            )
        prepared.append(t)
    if len(prepared) < 2:
        return prepared
    spliced: tx.List[Transformation] = []
    for nxt in prepared:
        _splice(spliced, nxt)
    return spliced
