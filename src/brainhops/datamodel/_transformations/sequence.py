from collections.abc import MutableSequence as AbcMutableSequence
from collections.abc import Sequence as AbcSequence

import typing_extensions as tx
from bagof.magic import NotKwOnly, replace

from brainhops._core.properties import smartproperty
from brainhops.datamodel import kinds
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import CompositionError, ConversionError

from . import nocycles
from .base import Transformation
from .compute.check import is_kind
from .compute.compose import compose
from .compute.factor import PatternCache, factor_sequence
from .compute.simplify import SimplifyLike, SimplifyTable
from .compute.simplify import simplify as _simplify
from .compute.utils import require_endomorphism
from .concrete import (
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
)
from .inverse import Inverse
from .meta import SubspaceTransformation
from .modes import (
    Family,
    ModeLike,
    is_family,
    mode_children,
    normalize_modes,
)
from .nocycles import register_sequence


class SequenceMixin(AbcSequence):
    # Sequence API over the `transformations` attribute.

    def __len__(self) -> int:
        return len(self.transformations or [])

    def __getitem__(self, index: tx.Union[int, slice]) -> Transformation:
        return self.transformations[index]

    def __iter__(self) -> tx.Iterator[Transformation]:
        return iter(self.transformations or [])


class MutableSequenceMixin(SequenceMixin, AbcMutableSequence):
    # Mutable sequence API over the `transformations` attribute.

    def __setitem__(
        self, index: tx.Union[int, slice], value: Transformation
    ) -> None:
        self.transformations[index] = value

    def __delitem__(self, index: tx.Union[int, slice]) -> None:
        del self.transformations[index]

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
    """An ordered chain of transformations.

    [`Sequence`][] is the base class of [`MutableSequence`][] and
    [`ImmutableSequence`][]. Calling `Sequence` directly acts as a factory and
    returns a [`MutableSequence`][].

    !!! note
        The transformations are listed in the order in which they are applied,
        which is the opposite of the order used when writing a composition of
        functions or a product of matrices. `Sequence([t1, t2, t3])(x)` is
        equivalent to `t3(t2(t1(x)))`, and `Sequence([t1, t2, t3]) @ x` is
        equivalent to `t3 @ t2 @ t1 @ x`.
    """

    data_fields = ("transformations",)

    # The chain is stored under a private name so that `replace()` carries over
    # what was given rather than what the property reports. In a subclass that
    # derives its chain, the copy would otherwise serve a stale chain that
    # shares its list with the original.
    _transformations: NotKwOnly[tx.Optional[tx.Sequence[Transformation]]] = (
        None
    )
    """Transformations, in the order in which they are applied."""

    transformations = smartproperty("transformations")

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

    def __new__(cls, *args, **kwargs) -> tx.Type[tx.Self]:
        if cls is Sequence:
            return MutableSequence(*args, **kwargs)
        return super().__new__(cls)

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        # Return a plain `Sequence` rather than `type(self)`, because not every
        # subclass can hold its own inverse.
        if self.transformations is None:
            return Sequence(input=self.output, output=self.input)
        return Sequence(
            transformations=[
                t.inverse(compute=compute, **kwargs)
                for t in reversed(self.transformations)
            ],
            # Carry over the declared endpoints only, so that derived endpoints
            # stay derived in the inverse.
            input=self._output,
            output=self._input,
        )

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the principal square root of the chain.

        The chain is first simplified. A chain `[P, *X, P^-1]`, where `P^-1` is
        the lazy inverse of `P` or the two ends are affine transformations
        whose product is exactly the identity, is a change of coordinates
        around `X`, and its square root is `[P, sqrt(X), P^-1]`. Any other
        chain is composed, and the square root of the result is returned.

        Raises
        ------
        DomainError
            If the chain does not map a space onto itself, or if the square
            root of its reduced form is undefined (see
            [`DomainError`][brainhops.errors.DomainError]).
        NotImplementedError
            If the chain is neither a change of coordinates nor composes to a
            single transformation.
        """
        return _chain_sqrt(self, compute, kwargs)

    def to(
        self, cls: tx.Optional[tx.Type[Transformation]] = None, **kwargs
    ) -> Transformation:
        """Convert the chain to another type or encoding.

        A chain has no tangent of its own, because the tangent of a composition
        is not the sum of the tangents. A `log=` keyword therefore re-encodes
        the transformation that the chain reduces to, which is possible in
        three cases:

        - A chain that simplifies to a single transformation is converted as
          that transformation.
        - A change of coordinates `[P, *X, P^-1]` (see
          [`Sequence.sqrt`][brainhops.datamodel.transformations.Sequence.sqrt])
          keeps its ends and re-encodes `X`, which is exact because the flow of
          a velocity commutes with a change of coordinates.
        - A chain of affine transformations is composed.

        Any other chain raises a [`ConversionError`][] before anything is
        computed. Without `log=`, the conversion follows
        [`Transformation.to`][brainhops.datamodel.transformations.Transformation.to].
        """
        if "log" in kwargs:
            return _chain_to(self, cls, kwargs)
        return super().to(cls, **kwargs)

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> Transformation:
        """Compute the transformation that results from the chain.

        With `mode=True`, a chain of affine-like transformations yields an
        affine-like transformation, and a chain whose first applied element is
        a coordinate field yields a coordinate field. When the first applied
        element is affine-like but the chain contains other elements, the
        result is a sequence of two: the composition of the affine-like
        elements before the first other element, then the composition of
        everything from that element onward.

        Parameters
        ----------
        mode : ModeLike, default=True
            Kinds to compose, as a name, a type or a list of them. `True`
            composes everything, and `False` only simplifies.
        simplify : SimplifyLike, default="analytic"
            Simplification policy applied before composition: `"analytic"` uses
            the type structure only, `"numeric"` also inspects values, and
            `False`, `"none"` or `None` disables simplification.
        factor : bool, default=False
            Rewrite the chain into its axis-group normal form
            `[grid?, F_1, ..., F_m, Pi_perm?]`, with one axis-preserving
            subspace factor per group of axes that transform together and an
            optional final permutation. Nothing is composed across groups, and
            a chain that creates or drops axes is left unfactored. The option
            is ignored with `mode=False`.
        """
        modes = normalize_modes(mode)
        policy = SimplifyTable.from_like(simplify)
        if not modes:
            # Simplification only, exactly as `simplify()` does: leaves are
            # downcast and free pairs collapse, but nothing is composed,
            # bridged or materialized.
            return _simplify(self, policy=policy)
        return _compute_sequence(self, modes, policy, factor=factor)

    def _flattened(self) -> tx.Self:
        # Flatten nested sequences and push the endpoints of the sequence onto
        # its first and last elements. Subclasses with a fixed shape, such as
        # geometries, override this method.
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
                # A nested child (such as a geometry, which contributes a grid
                # and then a transformation) may leave one level of nesting;
                # the next pass of `_compute_sequence` flattens it again.
                flattened.extend(t._flattened().transformations or [])
            else:
                flattened.append(t)
        return self.to(transformations=flattened)


class MutableSequence(MutableSequenceMixin, Sequence):
    """A mutable sequence of transformations, stored as a list.

    !!! note
        The transformations are listed in the order in which they are applied,
        which is the opposite of the order used when writing a composition of
        functions or a product of matrices. `Sequence([t1, t2, t3])(x)` is
        equivalent to `t3(t2(t1(x)))`, and `Sequence([t1, t2, t3]) @ x` is
        equivalent to `t3 @ t2 @ t1 @ x`.
    """

    # Narrow the chain to a list.
    _transformations: NotKwOnly[tx.Optional[tx.List[Transformation]]] = None


class ImmutableSequence(Sequence):
    """A [`Sequence`][] whose contents cannot be edited in place.

    The chain is stored as a tuple and the class has no mutating methods, so
    item assignment, deletion and insertion fail.
    """

    # Narrow the chain to a tuple.
    _transformations: NotKwOnly[tx.Optional[tx.Tuple[Transformation, ...]]] = (
        None
    )


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
    factor: bool = False,
) -> Transformation:
    """Compute a sequence by bridging, simplifying and composing to a fixpoint.

    The steps are ordered by the cost that each one is allowed to incur.

    1. Bridge every boundary at which adjacent transformations disagree on
       their shared coordinate system, since composers assume compatible
       systems. Bridging is the only step that adds elements, so it belongs to
       computation rather than simplification. It runs first because it reads
       the neighbours (a reversed array-index axis needs the extent of the
       adjacent grid), but it reads no parameter, and opposite bridges cancel
       in the next round.
    2. Simplify the leaves. This precedes every composition because it is the
       only step that can drop a lazy inverse for free, whereas a composer that
       folds a neighbour into a field destroys the cancelling pair.
    3. Compose the runs of adjacent transformations that the mode admits.

    A round that does not shorten the sequence cannot be improved by another,
    so the loop stops. With `factor=True`, the chain is also rewritten into its
    axis-group normal form (see [`factor_sequence`][]) between steps 2 and 3.
    Because factoring may lengthen the sequence, the fixpoint is instead a
    round that leaves every leaf as the same object, and an iteration cap
    raises an error rather than looping forever.
    """
    # The factoring pass memoizes the dependency pattern of each leaf across
    # rounds.
    cache = PatternCache()
    iteration = 0
    max_iter = 0
    while True:
        # Bridge the direct children, because a nested sequence carries its
        # endpoints on itself. Flattening then rebuilds no endpoint, which
        # keeps identity-based cancellation working.
        flat = _unnest(_insert_bridges(seq.transformations or []))
        if not flat:
            return Identity(input=seq.input, output=seq.output)
        seq = seq.to(transformations=flat)
        before = len(flat)
        if factor and not max_iter:
            max_iter = _factor_cap(flat)

        # Push the endpoints onto the ends only if the sequence has any, which
        # preserves the identity link in the common case without endpoints.
        if seq.input is not None or seq.output is not None:
            seq = seq._flattened()

        simplified = _simplify(seq, policy=policy)
        if not isinstance(simplified, Sequence):
            # The chain collapsed to a single transformation: nothing is left
            # to compose.
            return simplified
        seq = simplified

        if factor:
            factored = factor_sequence(
                seq, modes, simplify=policy, cache=cache
            )
            if not isinstance(factored, Sequence):
                return factored
            seq = factored

        memo: tx.Set[Family] = set()
        for submode in modes:
            result = _compose_mode(seq, submode, memo, factor)
            if not isinstance(result, Sequence):
                # One last downcast, since composition products are leaves that
                # the simplifier has not seen.
                return _simplify(result, policy=policy)
            seq = result

        if factor:
            # Fixpoint: the round left every leaf as the same object. The
            # simplifier has already seen these leaves, so no final pass is
            # needed.
            after = _unnest(seq.transformations)
            if len(after) == len(flat) and all(
                a is b for a, b in zip(after, flat)
            ):
                return seq
            iteration += 1
            if iteration > max_iter:
                raise RuntimeError(
                    "compute did not converge under factor=True after "
                    f"{iteration} iterations; a pass is not "
                    "identity-preserving on the normal form"
                )
        elif len(_unnest(seq.transformations)) >= before:
            # No pass shrank the sequence, so another round cannot either.
            # Simplify the composition products one last time.
            return _simplify(seq, policy=policy)


def _compose_mode(
    seq: Sequence,
    mode: Family,
    memo: tx.Set[Family],
    factor: bool = False,
) -> Transformation:
    # Compose the runs admitted by one mode, depth first, so that finer kinds
    # combine first: with `mode="affine"`, adjacent translations fold into one
    # translation before widening to an affine.

    if not _is_flat(seq):
        seq = seq._flattened()

    if not seq.transformations:
        return seq

    # A mode that was already visited has nothing left to fold.
    if mode in memo:
        return seq

    # Fold the finer child modes first.
    for child in mode_children(mode):
        seq = _compose_mode(seq, child, memo, factor)
        if not isinstance(seq, Sequence):
            return seq

    # Never visit this mode again.
    memo.add(mode)

    inputs = list(getattr(seq, "transformations", [seq]))
    outputs = []

    # Compose each consecutive run that matches the mode. `is_family` sees
    # through wrappers, so a subspace transformation embedding an affine
    # matches `mode="affine"` and composes with its neighbours.
    while inputs:
        item = inputs.pop(0)
        if is_family(item, mode):
            while inputs and is_family(inputs[0], mode):
                if factor and not _factor_pair_ok(item, inputs[0]):
                    # Composition must not undo the factoring pass.
                    break
                next_input = inputs.pop(0)
                try:
                    # Compose in sequence order. `compose` ignores modes;
                    # `is_family` is the gate that decides which pairs reach
                    # it.
                    item = compose(next_input, item)
                except CompositionError:
                    # Admitted kinds that cannot combine, such as subspace
                    # transformations over misaligned axes, are kept side by
                    # side.
                    outputs.append(item)
                    item = next_input
        outputs.append(item)

    if len(outputs) == 1:
        return outputs[0]
    return Sequence(transformations=outputs)


def _factor_pair_ok(a: Transformation, b: Transformation) -> bool:
    # Under `factor=True`, a `CartesianField` composes with nothing, and a
    # subspace factor composes only with another one over the same axes or with
    # an `Identity`. Next to anything else, the subspace composers would fold
    # the factor in and destroy the normal form.
    if isinstance(a, CartesianField) or isinstance(b, CartesianField):
        return False
    a_sub = isinstance(a, SubspaceTransformation)
    b_sub = isinstance(b, SubspaceTransformation)
    if not a_sub and not b_sub:
        return True
    if isinstance(a, Identity) or isinstance(b, Identity):
        return True
    if a_sub and b_sub:
        return _axes_tuple(a.input_axes) == _axes_tuple(
            b.input_axes
        ) and _axes_tuple(a.output_axes) == _axes_tuple(b.output_axes)
    return False


def _axes_tuple(axes: tx.Any) -> tx.Optional[tx.Tuple[int, ...]]:
    # Axis vectors may be arrays; compare them as tuples.
    return None if axes is None else tuple(int(x) for x in axes)


def _factor_cap(flat: tx.List[Transformation]) -> int:
    # Generous bound on productive rounds: `N` overestimates the working
    # dimension and `L` is the length of the chain. The cap is only a safety
    # net, since a pass that preserves the normal form converges quickly.
    n = 0
    for t in flat:
        shape = getattr(t, "shape", None)
        if shape is not None:
            n = max(n, len(shape))
        if not isinstance(t, Inverse):
            # Reading the matrix of an inverse would materialize it.
            matrix = getattr(t, "matrix", None)
            if matrix is not None:
                rows, cols = matrix.shape
                n = max(n, int(rows), int(cols))
        for name in ("input_axes", "output_axes"):
            axes = getattr(t, name, None)
            if axes is not None and len(axes):
                n = max(n, max(int(x) for x in axes) + 1)
    return max(2 * (n + len(flat)) + 4, 16)


# ----------------------------------------------------------------------
#    OPERATORS
# ----------------------------------------------------------------------


def _chain_sqrt(
    seq: Sequence, compute: bool, kwargs: tx.Dict[str, tx.Any]
) -> Transformation:
    # The square root does not distribute over a composition, but it commutes
    # with a change of coordinates: `sqrt(P^-1 X P) == P^-1 sqrt(X) P`.
    require_endomorphism(seq, "square root")
    chain = seq.simplify()
    if not isinstance(chain, Sequence):
        return chain.sqrt(compute, **kwargs)
    leaves = list(chain.transformations or [])
    if len(leaves) >= 3 and _undoes(leaves[0], leaves[-1]):
        middle = leaves[1:-1]
        if len(middle) == 1:
            inner = middle[0]
        else:
            inner = Sequence(transformations=middle)
        obj = Sequence(
            transformations=[leaves[0], inner.sqrt(), leaves[-1]],
            input=chain._input,
            output=chain._output,
        )
        return obj.compute(**kwargs) if compute else obj
    reduced = chain.compute()
    if isinstance(reduced, Sequence):
        names = ", ".join(type(t).__name__ for t in reduced.transformations)
        raise NotImplementedError(
            f"The square root of a chain is implemented only when it "
            f"composes to a single transformation, or when it is a change "
            f"of coordinates [P, ..., P^-1], but this one composes to "
            f"[{names}]."
        )
    return reduced.sqrt(compute, **kwargs)


def _chain_to(
    seq: Sequence,
    cls: tx.Optional[tx.Type[Transformation]],
    kwargs: tx.Dict[str, tx.Any],
) -> Transformation:
    # The endpoints belong to the chain; every other keyword re-encodes the
    # transformation that the chain reduces to.
    ends = {k: kwargs.pop(k) for k in ("input", "output") if k in kwargs}
    chain = seq.simplify()
    if not isinstance(chain, Sequence):
        return chain.to(cls, **kwargs, **ends)
    leaves = list(chain.transformations or [])
    if len(leaves) >= 3 and _undoes(leaves[0], leaves[-1]):
        middle = leaves[1:-1]
        if len(middle) == 1:
            inner = middle[0]
        else:
            inner = Sequence(transformations=middle)
        obj = Sequence(
            transformations=[leaves[0], inner.to(**kwargs), leaves[-1]],
            input=chain._input,
            output=chain._output,
        )
        return obj.to(**ends) if ends else obj
    if all(is_kind(t, kinds.Affine) for t in leaves):
        reduced = chain.compute()
        if not isinstance(reduced, Sequence):
            return reduced.to(cls, **kwargs, **ends)
    names = ", ".join(type(t).__name__ for t in leaves)
    raise ConversionError(
        f"A chain is stored as its tangent (log=) only when it reduces to "
        f"a single transformation, is a change of coordinates "
        f"[P, ..., P^-1], or is a chain of affines, but this one is "
        f"[{names}]. It is not composed to find out."
    )


def _undoes(first: Transformation, last: Transformation) -> bool:
    # Whether `last @ first` is exactly the identity: a lazy inverse undoes its
    # forward, and two affine ends undo each other when their product is the
    # identity. Matrices that are inverse only up to rounding do not qualify,
    # and a field end is never composed to decide.
    if isinstance(_simplify(first, last), Identity):
        return True
    if not (is_kind(first, kinds.Affine) and is_kind(last, kinds.Affine)):
        return False
    try:
        product = compose(last, first)
    except (CompositionError, ValueError):
        return False
    return product.is_identity(compute=True)


# ----------------------------------------------------------------------
#    UTILS
# ----------------------------------------------------------------------


def _normalize_inverse(t: Transformation) -> Transformation:
    # Expand the generic `Inverse` front door into the typed inverse of the
    # held transformation, so that the engine can compute it and cancellation
    # can recognize it. A typed inverse is left alone.
    if not isinstance(t, Inverse) or t._resultof is not None:
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
    if self.transformations is None:
        return True
    return all(not isinstance(t, Sequence) for t in self.transformations)


def _unnest(transformations: tx.Optional[tx.List[Transformation]]) -> list:
    # Flatten nested sequences without touching any endpoint (unlike
    # `_flattened`, which may rebuild the first or last element and so read a
    # lazy field). Generic inverses are expanded on the way.
    flattened = []
    for t in transformations or []:
        t = _normalize_inverse(t)
        if isinstance(t, Sequence):
            flattened.extend(_unnest(t.transformations))
        else:
            flattened.append(t)
    return flattened


def _interpolates(xform: Transformation) -> bool:
    # Whether applying a transformation resamples through a spline, that is,
    # whether it reaches a displacement or coordinates field. A
    # `CartesianField` is the identity over its grid, so it does not
    # interpolate.
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
    # Append `nxt`, letting the adaptor place a bridge or subspace embedding at
    # the boundary. Neither transformation is rebuilt, so each leaf stays the
    # same object that its inverse names.
    if not spliced:
        spliced.append(nxt)
        return
    prev = spliced[-1]
    pieces = list(
        nocycles.ADAPT(prev, nxt, allow_type_grouped_positional=True)
    )
    if pieces[0] is not prev:
        # `prev` was embedded in the fuller space of `nxt`, so its left
        # boundary changed; splice the embedded piece again against its left
        # neighbour.
        spliced.pop()
        _splice(spliced, pieces[0])
        spliced.extend(pieces[1:])
    else:
        spliced.extend(pieces[1:])


def _insert_bridges(
    transformations: tx.List[Transformation],
) -> tx.List[Transformation]:
    # Reconcile every boundary at which the output system of one transformation
    # disagrees with the input system of the next. Boundaries may hide inside
    # nested sequences or behind a generic inverse, so nested children are
    # bridged first and generic inverses are expanded. No leaf has its
    # endpoints rebuilt.
    prepared: tx.List[Transformation] = []
    for t in transformations:
        t = _normalize_inverse(t)
        # Only plain sequences are rebuilt; a specialized sequence (such as a
        # multiscale field or a geometry) keeps its own shape.
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
