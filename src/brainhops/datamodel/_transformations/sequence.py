# stdlib
from collections.abc import MutableSequence as AbcMutableSequence
from collections.abc import Sequence as AbcSequence

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# api
from brainhops._core.properties import smartproperty
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

# `_cancels` is the O(1), identity-only cancel test. It lives in `inverse`,
# next to the `Inverse` class it reads, and is used here by `_annihilates`.
from .inverse import Inverse, _cancels
from .meta import SubspaceTransformation

# The mode types and helpers live in their own module so that `base`,
# `inverse` and `meta` can import them at the top level without the
# `sequence` <-> `base` import cycle. They are re-exported here because
# other modules (and the package `__init__`) import `ModeLike` from
# `sequence`.
from .modes import (  # noqa: F401
    ModeLike,
    ModePair,
    SimplifyLike,
    SimplifyPolicy,
    SimplifyTable,
    _is_proper_mode,
    _lower_modes,
    _lower_simplify,
    _matches_mode,
    _mode_admits,
    _mode_children,
    _ModePair,
    _resolve_simplify,
    _table_is_noop,
    _table_needs_numeric,
)
from .registries import register_sequence


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

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> Transformation:
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
        mode : [list of] name or type, optional
            Kinds of transformations to compose.
            * If `None` (default): compose every kind in the sequence.
            * If a kind key (a set NAME, a hierarchy type, or a wrapper/field
              key): compose only consecutive runs of that kind.
        simplify : simplify policy, default="analytic"
            How hard each leaf may be looked at, per its kind (see the
            [`SimplifyPolicy`][brainhops.datamodel.enums.SimplifyPolicy]
            grammar). `"analytic"` (the default) downcasts each leaf from
            structure only; `False`/`"none"`/`None` disables it.
        factor : bool, default=False
            Whether to rewrite the sequence into its axis-group normal
            form: a leading grid (if any), an optional embedding of created
            axes, one axis-preserving factor per group of axes that
            transform together, and trailing reindex/drop projections. Off
            by default, so the result is byte-for-byte the plain
            `compute()` result. Nothing is ever composed across groups;
            `mode` still decides whether the restricted pieces inside a
            group compose.
        """
        modes = _lower_modes(mode)
        if not modes:
            return self  # No-op
        # The simplify table is resolved per leaf and applied inside the
        # fixpoint loop of `_compute_sequence`, every iteration, AFTER the
        # identity-cancel sweep (so a lazy transform/inverse pair cancels by
        # object identity rather than being materialized) and BEFORE per-mode
        # composition (so a numeric downcast such as field->affine feeds it).
        table = _lower_simplify(simplify)
        # The driver owns the per-leaf simplify cache (and its keepalive), so
        # the final pass below reuses the very same memo the fixpoint loop
        # built rather than re-scanning a leaf it already downcast (R1).
        simplify_cache: tx.Dict[int, Transformation] = {}
        simplify_keepalive: tx.List[Transformation] = []
        result = _compute_sequence(
            self,
            mode=modes,
            table=table,
            factor=factor,
            simplify_cache=simplify_cache,
            simplify_keepalive=simplify_keepalive,
        )
        # A final pass downcasts a composed result that escaped the loop as a
        # single transform. It is needed only when the table has a numeric
        # entry: the in-loop pass runs before composition each iteration and
        # the loop exits when nothing shrinks, so the only leaves left
        # unsimplified are the last composition's products, which have no
        # structural downcast left -- only value facts can appear, which is
        # `numeric`. It runs after all cancellation, so a lazy pair has
        # already cancelled by identity rather than being materialized. It
        # consults the driver-owned cache, so a leaf already downcast in the
        # loop is not scanned again.
        if _table_needs_numeric(table):
            result = _simplify_leaves(
                result, table, simplify_cache, simplify_keepalive
            )
        return result

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


def _simplify_leaves(
    result: Transformation,
    table: SimplifyTable,
    cache: tx.Dict[int, Transformation],
    keepalive: tx.List[Transformation],
) -> Transformation:
    # Apply the simplify table to a computed result, consulting the
    # driver-owned cache. A `Sequence` result has each of its leaves
    # simplified; any other result is simplified as a single leaf. This runs
    # after `_compute_sequence` has finished all composition and
    # cancellation, so it only downcasts what survives; a leaf already in the
    # cache (unchanged through the loop) is not scanned again (R1).
    if isinstance(result, Sequence):
        leaves = result.transformations or []
        simplified = [
            _cached_simplify_leaf(t, table, cache, keepalive) for t in leaves
        ]
        if any(a is not b for a, b in zip(simplified, leaves)):
            return replace(result, transformations=simplified)
        return result
    return _cached_simplify_leaf(result, table, cache, keepalive)


def _cached_simplify_leaf(
    t: Transformation,
    table: SimplifyTable,
    cache: tx.Dict[int, Transformation],
    keepalive: tx.List[Transformation],
) -> Transformation:
    # A leaf's simplify result, memoized by `id`. A leaf seen before (same
    # object) reuses its cached result rather than being re-scanned; the
    # keepalive list pins those leaves so an `id` is never reused while the
    # cache holds it.
    key = id(t)
    if key in cache:
        return cache[key]
    result = _simplify_leaf(t, table)
    cache[key] = result
    keepalive.append(t)
    return result


def _simplify_leaf(t: Transformation, table: SimplifyTable) -> Transformation:
    # Apply a leaf's resolved simplify policy. Called each fixpoint iteration
    # by `_simplify_pass` (after the identity-cancel sweep, before per-mode
    # composition) and once by `_simplify_leaves` on the final result. The
    # whole `table` is threaded to `compute`, never a scalar, so a wrapper's
    # inner resolves against the user's own keys.
    #
    # A `CartesianField` is never downcast. A grid is the identity map over
    # its coordinates, so the checks would collapse it to `Identity` and
    # drop the sampling domain it defines; grids only ever reach a leading,
    # trailing or standalone position, all of which are sampling domains.
    if isinstance(t, CartesianField):
        return t
    policy = _resolve_simplify(t, table)
    if policy is SimplifyPolicy.none:
        return t
    if policy is SimplifyPolicy.analytic and isinstance(t, Inverse):
        # Structure-only: a lazy inverse is left untouched (so an analytic
        # policy never materializes, e.g., an `InverseCoordinatesField`); the
        # always-on cancel sweep still collapses it in place.
        return t
    if isinstance(t, Inverse):
        # `numeric`: materializing may not be available (a coordinate-field
        # inverse cannot be materialized directly) -- leave it lazy so it can
        # still cancel, rather than raising.
        try:
            return t.compute(simplify=table)
        except NotImplementedError:
            return t
    return t.compute(simplify=table)


def _simplify_pass(
    seq: "Sequence",
    table: SimplifyTable,
    cache: tx.Dict[int, Transformation],
    keepalive: tx.List[Transformation],
) -> "Sequence":
    # One per-iteration simplify pass over a sequence's leaves. A leaf seen
    # before (same object across iterations) reuses its cached result rather
    # than being re-scanned; the keepalive list pins those leaves so an `id`
    # is never reused while the cache holds it.
    leaves = _unnest(seq.transformations)
    simplified = []
    changed = False
    for t in leaves:
        result = _cached_simplify_leaf(t, table, cache, keepalive)
        simplified.append(result)
        changed = changed or result is not t
    if changed:
        return replace(seq, transformations=simplified)
    return seq


def _compute_sequence(
    seq: Sequence,
    mode: tx.List[_ModePair],
    memo: tx.Optional[tx.Set[_ModePair]] = None,
    table: tx.Optional[SimplifyTable] = None,
    factor: bool = False,
    simplify_cache: tx.Optional[tx.Dict[int, Transformation]] = None,
    simplify_keepalive: tx.Optional[tx.List[Transformation]] = None,
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
        #
        # The per-type simplify policy is applied every iteration, AFTER the
        # identity-cancel sweep (so a lazy transform/inverse pair cancels by
        # object identity before a numeric method could materialize either
        # side) and BEFORE composition (so a downcast such as field->affine
        # still feeds separability/merging). Running it each iteration also
        # downcasts a *composed* result to its cheapest type, matching the
        # former final pass. An `id`-keyed cache keeps a surviving leaf from
        # being re-scanned on every iteration; the keepalive list pins those
        # leaves so an `id` is never reused while the cache holds it.
        skip_simplify = table is None or _table_is_noop(table)
        if simplify_cache is None:
            simplify_cache = {}
            simplify_keepalive = []
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

            # Identity-based cancellation, run as a single stack sweep before
            # any typed composition. A transform placed next to its own lazy
            # inverse annihilates it, and so does a subspace-wrapped
            # transform placed next to its own subspace-wrapped inverse; both
            # cancel by identity, materializing no field. Each removal can
            # expose a new adjacent pair, so the sweep keeps a stack and
            # cancels the top of the stack against the next transform.
            #
            # This runs *before* composition, and not after it, because
            # composition must not materialize a neighbour before an
            # adjacent pair has cancelled: otherwise `grid @ affine` folds
            # into a field before an adjacent `Sub(warp) @ Sub(warp^-1)` pair
            # can cancel, and the warp is resampled needlessly (or, for a
            # field that cannot be inverted, at all).
            flat = _unnest(seq.transformations)
            stack: tx.List[Transformation] = []
            cancelled = False
            for t in flat:
                if stack and _annihilates(stack[-1], t):
                    stack.pop()
                    cancelled = True
                else:
                    stack.append(t)
            if cancelled:
                if not stack:
                    # A sequence that cancels entirely is the identity from
                    # the input of its first element to the output of its
                    # last. The element endpoints are used, falling back to
                    # the sequence's own where an element leaves one unset.
                    # `or` relies on a `CoordinateSystem` never being falsy: it
                    # defines no `__bool__`/`__len__`, so it is always truthy
                    # and `or` selects the element endpoint when set, the
                    # sequence's otherwise. A future `__len__` on
                    # `CoordinateSystem` would make an empty system falsy and
                    # would need this revisited.
                    first, last = flat[0], flat[-1]
                    return Identity(
                        input=first.input or seq.input,
                        output=last.output or seq.output,
                    )
                seq = replace(seq, transformations=stack)

            # Per-type simplify, now that adjacent lazy-inverse pairs have
            # cancelled. Each leaf's resolved policy is applied before the
            # composition below, so a numeric downcast feeds it.
            if not skip_simplify:
                seq = _simplify_pass(
                    seq, table, simplify_cache, simplify_keepalive
                )

            # Propagate the sequence's own endpoints onto its first and
            # last elements, but only when it carries any, so the identity
            # link is preserved in the common case of an endpoint-less
            # composition.
            if seq.input is not None or seq.output is not None:
                seq = seq._flattened()
            submemo: tx.Set[_ModePair] = set()
            for submode in mode:
                # `table` is deliberately not threaded here: the memo'd
                # recursion only composes, and the simplify pass above (in
                # the memo-less driver) has already run this iteration.
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

    # Compose any consecutive run that matches the mode. `_matches_mode`
    # sees through wrappers via `is_member` (analytic), so a subspace that
    # lifts an affine matches `mode="affine"` and composes with its
    # neighbours -- replacing the old `_composes_under` special case.
    while inputs:
        item = inputs.pop(0)
        if _matches_mode(item, mode):
            while inputs and _matches_mode(inputs[0], mode):
                next_input = inputs.pop(0)
                try:
                    # NOTE: we compose to the left ! (see sequence definition)
                    # `compose` is mode-free: the gate that decides which
                    # adjacent transforms are handed to it is `_matches_mode`
                    # above, not the composer.
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


def _annihilates(first: Transformation, second: Transformation) -> bool:
    # Whether `[first, second]` (with `first` applied first) reduces to the
    # identity without materializing any field. This is the predicate the
    # single stack sweep in `_compute_sequence` cancels pairs by.
    #
    # A transform placed next to its own lazy inverse annihilates it, named
    # by identity through `forward` (see `_cancels`). And two subspace
    # transforms over the same axes annihilate when the axes chain, the net
    # map introduces no reindex (the axes the first reads are the axes the
    # second writes), and their inner transforms compose to the identity:
    # either because one inner is the lazy inverse of the other, or because
    # both inners are already the identity. This is exactly the case that
    # lets a subspace-wrapped field meet its own subspace-wrapped inverse and
    # cancel, rather than the field being resampled through a neighbour first.
    if _cancels(first, second):
        return True
    if not (
        isinstance(first, SubspaceTransformation)
        and isinstance(second, SubspaceTransformation)
    ):
        return False
    if (
        first.output_axes is None
        or second.input_axes is None
        or list(first.output_axes) != list(second.input_axes)
    ):
        return False
    # A *reindexing* inverse pair -- whose axes chain (checked above) but
    # whose net map still permutes axes (`first.input_axes` !=
    # `second.output_axes`) -- is deliberately NOT annihilated here. It does
    # not reduce to the bare identity (it is a pure axis reindex), so it is
    # left to the run loop to fold into a single reindexing subspace
    # transform; do not "fix" it into this sweep.
    same_axes = (first.input_axes is None) == (
        second.output_axes is None
    ) and (
        first.input_axes is None
        or list(first.input_axes) == list(second.output_axes)
    )
    if not same_axes:
        return False
    inner_first = first.transformation
    inner_second = second.transformation
    if _cancels(inner_first, inner_second):
        return True
    # `compute=False` only: this sweep is the always-on analytic pass, run for
    # every adjacent subspace pair on every fixpoint iteration. A numeric
    # `compute=True` check would scan a whole displacement field
    # (`(field == 0).all()`) each time; that belongs to the later per-type
    # policy, not here. Nothing depends on the numeric branch -- the
    # `transformation=None` case is `inner is None`, and the lazy-inverse case
    # is `_cancels` above.
    first_identity = inner_first is None or is_identity(
        inner_first, compute=False
    )
    second_identity = inner_second is None or is_identity(
        inner_second, compute=False
    )
    return first_identity and second_identity


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
