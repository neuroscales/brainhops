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
)
from .errors import CompositionError
from .inverse import Inverse
from .meta import SubspaceTransformation
from .modes import (
    Family,
    ModeLike,
    matches_mode,
    mode_children,
    normalize_modes,
)
from .registries import register_sequence
from .simplify import SimplifyLike, SimplifyTable
from .simplify import simplify as _simplify


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


@register_sequence
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

    data_fields: tx.ClassVar[tx.Tuple[str]] = "transformations",

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
        # NOTE
        #   Not all sequence subclasses can contain their inverse.
        #   We therefore return an exact `Sequence`, rather than using
        #   `type(self)`.
        if self.transformations is None:
            return Sequence(input=self.output, output=self.input)
        return Sequence(
            transformations=[
                t.inverse(compute=compute, **kwargs)
                for t in reversed(self.transformations)
            ],
            # NOTE
            #   we carry over the declared (private) endpoints, not the
            #   derived (public) ones, so that the latter keep being
            #   derived in the inverse object.
            input=self._output,
            output=self._input,
        )

    def compute(
        self, mode: ModeLike = True, *, simplify: SimplifyLike = "analytic",
    ) -> Transformation:
        """
        Compute the resulting transform of the sequence of transformations.

        Assuming that `mode=True`:

        * If all transformations in the sequence are affine-like
          transformations, `compute()` returns an affine-like transform.

       * If the first (= rightmost) transform in the sequence is a
          coordinate field, `compute()` returns a coordinate field.

        * If the first (= rightmost) transform in the sequence is an
          affine-like transform, and the sequence contains at least one
          non-affine-like transform, `compute()` returns a sequence of two
          transformations:

          1. the composition of all affine-like transformations that
             appear before the first non-affine-like transform in the
            sequence, and
          2. the composition of all transformations in the sequence,
             starting from the first non-affine-like transform in the
             sequence.

        Parameters
        ----------
        mode : [list of] name or type, optional
            Kinds of transformations to compose.
            * If `True` (default): compose every kind in the sequence.
            * If `False`: compose nothing (simplify-only).
            * If a (list of) transformation type(s): compose only pairs
              of transformations of these kinds.
        simplify : simplify policy, default="analytic"
            Whether to simplify sub-transformations prior to composition,
            and how hard to try to simplify them.
            * `"analytic"` (the default) looks at the type structure only;
            * `"numeric"` looks at the numeric values of the transformation;
            * `False`/`"none"`/`None` disables simplification.
        """
        modes = normalize_modes(mode)
        policy = SimplifyTable.from_like(simplify)
        if not modes:
            # `mode=False` (compose nothing): simplification alone. Each
            # leaf is downcast per its policy and each free pair collapses,
            # but nothing is composed, bridged or materialized -- this is
            # exactly what `simplify()` does.
            return _simplify(self, policy=policy)
        return _compute_sequence(self, modes, policy)

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
        return self.to(transformations=flattened)


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
    modes: tx.List[Family],
    policy: SimplifyTable,
) -> Transformation:
    """Compute a sequence: bridge, simplify, compose -- to a fixpoint.

    The three steps are ordered by what each of them is allowed to cost.

    1. **Bridge.** Reconcile every boundary where two adjacent transforms
       disagree on the system they share. The composers assume compatible
       systems, so the bridge that reorders, rescales or flips the
       mismatched axes is inserted here -- the one step that *adds* an
       element, which is why it belongs to computation and not to
       simplification. It runs first because a bridge reads its neighbours:
       a reversed array-index axis needs the extent that an adjacent grid
       carries, and that grid is exactly what step 2 may drop. Bridging
       reads no parameter and materializes nothing, so running it ahead of
       the cancellation below costs nothing. A bridge is built from exactly
       invertible pieces, so two opposite bridges cancel and simplify away
       on the next round.

    2. **Simplify.** Downcast every leaf and collapse every pair that
       collapses for free. This runs before every composition, because it
       is the only step that can make a lazy inverse disappear without
       paying for it: once a composer has folded a neighbour into a field,
       the pair that would have cancelled is gone. Simplification never
       lengthens the sequence.

    3. **Compose.** Fuse each run of adjacent transforms the mode admits,
       reading their parameters.

    A round that does not shorten the sequence cannot be improved on by
    another, so the loop stops there.
    """
    while True:
        # --- 1. bridge ---
        # Bridging runs on the direct children, because a nested sequence
        # carries its endpoint systems on itself and flattening would drop
        # them. Flattening then happens without rebuilding any endpoint, so
        # a transform stays the same object that its inverse names and the
        # cancellation below keeps linking the two by identity.
        flat = _unnest(_insert_bridges(seq.transformations or []))
        if not flat:
            return Identity(input=seq.input, output=seq.output)
        seq = seq.to(transformations=flat)
        before = len(flat)

        # Propagate the sequence's own endpoints onto its first and last
        # elements, but only when it carries any, so the identity link is
        # preserved in the common case of an endpoint-less composition.
        if seq.input is not None or seq.output is not None:
            seq = seq._flattened()

        # --- 2. simplify ---
        simplified = _simplify(seq, policy=policy)
        if not isinstance(simplified, Sequence):
            # It collapsed to a single transform: nothing left to compose.
            return simplified
        seq = simplified

        # --- 3. compose ---
        memo: tx.Set[Family] = set()
        for submode in modes:
            result = _compose_mode(seq, submode, memo)
            if not isinstance(result, Sequence):
                # A single transform: give it one last downcast, since the
                # products of a composition are exactly the leaves the
                # simplify pass above has not seen.
                return _simplify(result, policy=policy)
            seq = result

        if len(_unnest(seq.transformations)) >= before:
            # No pass shrank the sequence, so a further round cannot
            # either. One last simplify pass over the composition products.
            return _simplify(seq, policy=policy)


def _compose_mode(
    seq: Sequence,
    mode: Family,
    memo: tx.Set[Family],
) -> Transformation:
    # Compose the runs a single mode admits, depth first.
    #
    # We optimize by recursively descending into the subclasses of the
    # mode. This combines similar transformations first: given `mode
    # = affine`, two adjacent translations are folded into one translation
    # before anything is widened to an affine.

    # --- Flatten sequence
    if not _is_flat(seq):
        seq = seq._flattened()

    # --- Check if nothing to do
    if not seq.transformations:
        return seq

    # --- A mode already visited has nothing left to fold
    if mode in memo:
        return seq

    # --- First, fold every child mode (finer kinds first)
    for child in mode_children(mode):
        seq = _compose_mode(seq, child, memo)
        if not isinstance(seq, Sequence):
            return seq

    # Mark that we've been through this mode
    # !!! DO NOT RETURN HERE
    memo.add(mode)

    # --- Compose transformations that belong to this mode in order

    # Ensure a list of transformations (and make a copy so we can pop it)
    inputs = list(getattr(seq, "transformations", [seq]))
    outputs = []

    # Compose any consecutive run that matches the mode. `matches_mode`
    # sees through wrappers via `is_kind` (analytic), so a subspace that
    # lifts an affine matches `mode="affine"` and composes with its
    # neighbours.
    while inputs:
        item = inputs.pop(0)
        if matches_mode(item, mode):
            while inputs and matches_mode(inputs[0], mode):
                next_input = inputs.pop(0)
                try:
                    # NOTE: we compose to the left ! (see sequence definition)
                    # `compose` is mode-free: the gate that decides which
                    # adjacent transforms are handed to it is `matches_mode`
                    # above, not the composer.
                    item = compose(next_input, item)
                except CompositionError:
                    # The two are of admitted kinds but cannot be combined
                    # (e.g. two subspace transforms whose axes do not line
                    # up). Keep them side by side and carry on.
                    outputs.append(item)
                    item = next_input
        outputs.append(item)

    # Return
    if len(outputs) == 1:
        return outputs[0]
    return Sequence(transformations=outputs)


# NOTE
#   The interior-grid drop and the adjacent-inverse cancellation used to
#   live here, as `_drop_interior_grids` and `_annihilates`. Both are
#   cost-free rewrites of a *pair* of transforms, so both are now
#   simplifiers (see `simplifiers`), reached from step 1 of
#   `_compute_sequence` and from `compose`'s first tier.


# ----------------------------------------------------------------------
#    UTILS
# ----------------------------------------------------------------------


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
