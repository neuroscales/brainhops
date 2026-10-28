# stdlib
from collections.abc import MutableSequence as AbcMutableSequence
from collections.abc import Sequence as AbcSequence

# dependencies
import typing_extensions as tx
from bagof.magic import NotKwOnly, replace

# core
from brainhops._core.backends import BackendLike, DeviceLike
from brainhops._core.properties import smartproperty

# datamodel
from brainhops.datamodel import kinds
from brainhops.datamodel._backends import _BackendsMixin
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import CompositionError, ConversionError

# transformations
from . import nocycles
from .base import Transformation, _copied
from .compute.check import is_kind
from .compute.compose import compose
from .compute.factor import factor_sequence
from .compute.simplify import SimplifyLike, SimplifyTable
from .compute.simplify import simplify as _simplify
from .compute.utils import (
    compute_depth,
    refuse_free_axes,
    require_endomorphism,
)
from .concrete import (
    CartesianField,
    CoordinateField,
    DisplacementField,
    Identity,
    Permutation,
)
from .inverse import Inverse
from .meta import MetaTransformation, Projection, SubspaceTransformation
from .modes import (
    Family,
    ModeLike,
    is_family,
    mode_children,
    normalize_modes,
)
from .nocycles import register_sequence


class SequenceMixin(AbcSequence):
    # Implement the read-only sequence protocol on top of the
    # `transformations` attribute.

    def __len__(self) -> int:
        return len(self.transformations or [])

    def __getitem__(self, index: tx.Union[int, slice]) -> Transformation:
        return self.transformations[index]

    def __iter__(self) -> tx.Iterator[Transformation]:
        return iter(self.transformations or [])


class MutableSequenceMixin(SequenceMixin, AbcMutableSequence):
    # Implement the mutable sequence protocol on top of the `transformations`
    # attribute, creating the list on the first insertion.

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
class Sequence(SequenceMixin, MetaTransformation):
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

    meta_fields = ("transformations",)
    data_fields = ("transformations",)

    # --- attributes ---------------------------------------------------

    # The chain is stored under a private name so that `replace()` carries over
    # what was given rather than what the property reports. In a subclass that
    # derives its chain, the copy would otherwise serve a stale chain that
    # shares its list with the original.
    _transformations: tx.Annotated[
        tx.Optional[tx.Sequence[Transformation]],
        NotKwOnly(),
    ] = None
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

    # --- factory ------------------------------------------------------

    def __new__(cls, *args, **kwargs) -> tx.Type[tx.Self]:
        if cls is Sequence:
            return MutableSequence(*args, **kwargs)
        return super().__new__(cls)

    # --- methods ------------------------------------------------------

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

        The chain is first simplified. A chain of the form `[P, *X, P^-1]` is
        a change of coordinates around `X`, and its square root is
        `[P, sqrt(X), P^-1]`. The chain has this form when its last element is
        the lazy inverse of its first element, or when its two ends are affine
        transformations whose product is exactly the identity. Any other chain
        is composed, and the square root of the result is returned.

        Raises
        ------
        DomainError
            If the chain does not map a space onto itself, or if the square
            root of its reduced form is undefined (see
            [`DomainError`][brainhops.errors.DomainError]).
        NotImplementedError
            If the chain is not a change of coordinates and does not compose
            to a single transformation.
        """
        return _chain_sqrt(self, compute, kwargs)

    def to(
        self,
        cls: tx.Optional[tx.Type[Transformation]] = None,
        *,
        copy: bool = False,
        **kwargs,
    ) -> Transformation:
        """Convert the chain to another type or encoding.

        A chain has no tangent of its own, because the tangent of a
        composition is not the sum of the tangents. A `log=` keyword
        therefore re-encodes the transformation that the chain reduces
        to, which is possible in three cases:

        - A chain that simplifies to a single transformation is converted
          as that transformation.
        - A change of coordinates `[P, *X, P^-1]` (see [`Sequence.sqrt`][])
          keeps its ends and re-encodes `X`, which is exact because the
          flow of a velocity commutes with a change of coordinates.
        - A chain of affine transformations is composed.

        Any other chain raises a [`ConversionError`][] before anything is
        computed. Without `log=`, the conversion follows
        [`Transformation.to`][Transformation.to], and so does `copy=`.
        """
        if "log" in kwargs:
            return _copied(self, _chain_to(self, cls, kwargs), copy)
        return super().to(cls, copy=copy, **kwargs)

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
            the type structure only, `"numeric"` also inspects the values
            of matrix transformations, `{None: "numeric"}` inspects the
            values of every transformation, and `False`, `"none"` or
            `None` disables simplification.
        factor : bool, default=False
            Whether to rewrite the chain into a normal form organized by
            groups of axes,
            `[grid?, delta?, F_1, ..., F_m, Pi_perm?, kappa?]`. The normal
            form starts with an optional grid and an optional projection
            that drops the inputs that nothing reads. It continues with
            one subspace factor for each group of axes that transform
            together, where each factor preserves its axes, and it ends
            with an optional permutation and an optional projection that
            creates the outputs that read no input. Nothing is composed
            across groups. The option is ignored with `mode=False`.
        """
        modes = normalize_modes(mode)
        policy = SimplifyTable.from_like(simplify)
        with compute_depth() as depth:
            if not modes:
                # Without a mode, only simplify, exactly as `simplify()`
                # does. Leaves are downcast to more specific types and pairs
                # that cancel at no cost collapse, but nothing is composed,
                # bridged or materialized.
                result = _simplify(self, policy=policy)
            else:
                result = _compute_sequence(self, modes, policy, factor=factor)
        # Only a composition removes the free axes that nothing reads, so
        # the result is checked when something may be composed.
        if depth == 0 and modes:
            refuse_free_axes(result)
        return result

    def bridge(self) -> tx.Self:
        """Reconcile the coordinate systems at every boundary of the chain.

        Two neighbours in a chain share a coordinate system: the output
        system of one is the input system of the next. Where the two
        disagree, as when one works in millimetres and the next in voxels,
        or when the two name the same axes in a different order, a bridge is
        inserted between them. A bridge reads no parameter, so bridging
        costs nothing, and two opposite bridges cancel when the chain is
        simplified.

        Bridging is also the only pass that lengthens a chain, which is why
        [`compute`][] runs it before anything else: a bridge reads its
        neighbours, and a reversed array-index axis needs the extent of the
        adjacent grid, which composition may drop.

        A nested sequence is bridged in place and stays nested, so a
        boundary that it hides is reconciled inside it. Call [`flatten`][]
        afterwards for a flat chain. A sequence with a fixed shape, such as
        a geometry or a multiscale field, is left alone. A generic
        [`Inverse`][] is replaced by the typed inverse of its forward
        transformation, so that the engine can compute it and cancellation
        can recognize it. No leaf has its endpoints rebuilt.

        Returns
        -------
        Sequence
            The chain, with a bridge at each boundary that needed one.

        Raises
        ------
        AdaptationError
            If a boundary cannot be bridged, because an axis that must be
            matched has no counterpart on the other side.
        """
        if self.transformations is None:
            return self
        return self.to(
            transformations=_insert_bridges(list(self.transformations))
        )

    def factor(
        self,
        policy: SimplifyLike = "analytic",
        *,
        compute: tx.Union[ModeLike, bool, None] = False,
    ) -> Transformation:
        """Rewrite the chain as one factor per group of axes.

        The axes of a chain often fall into groups that never interact,
        as when each axis is scaled on its own and two axes are then
        swapped. Factoring finds those groups and rewrites the chain as
        `[grid?, delta?, F_1, ..., F_m, Pi_perm?, kappa?]`, with one
        axis-preserving factor `F_i = Subspace(inner_i, A_i, A_i)` per
        group, ordered by their first axis, and a permutation that
        carries each group to the axes that it occupies at the output
        (see [`factor_sequence`][]). The projection `delta` drops the
        inputs that no output reads, and the projection `kappa` creates
        the outputs that read no input, at their constant values.
        Nothing is composed across groups, so the coordinates of
        independent axes are never tiled into a grid over all of them,
        and each group can be resampled on its own.

        The chain is unnested first, and a leading [`CartesianField`][] is
        the sampling grid, which is kept as the same object. Every other
        element is cut into one piece per group, and an element that lies
        within a single group is kept as it is, so a lazy inverse stays
        lazy and still cancels with its forward transformation.

        !!! note
            The pass reads the chain as it stands, and it is idempotent.
            It declines a chain in which an element shifts or scales an
            output that reads no input, because it carries such a
            constant only through the elements that pass it through
            unchanged. [`compute`][] with `factor=True` composes such a
            chain, and then simplifies and factors the result.

        Parameters
        ----------
        policy : SimplifyLike, default="analytic"
            How closely the elements may be read, with the values and
            the meaning of the policy of [`simplify`][]. Under the
            analytic policy, a matrix couples all its axes, and under
            the numeric policy, its exact zeros are read, so that an
            affine with independent axes splits.
        compute : ModeLike or bool or None, default=False
            Kinds of transformation that may be composed inside a group,
            with the values accepted by the `mode` argument of
            [`compute`][]. The default, `False`, composes nothing, so each
            factor holds the sub-chain of its own group, whereas `True`
            leaves each factor holding a single transformation.

        Returns
        -------
        Transformation
            The normal form, or `self` when the chain is already in
            normal form or does not factor at all, as when it forms a
            single group and has neither end, its two ends differ in
            dimension in a way that the end projections do not explain,
            a grid sits inside it, or one of its elements cannot be read
            or restricted. An [`Identity`][] is returned when no factor
            is left.
        """
        return factor_sequence(self, policy, compute=compute)

    def flatten(self, endpoints: bool = True) -> tx.Self:
        """Splice the nested sequences of the chain into the chain itself.

        A chain whose elements are themselves chains applies exactly as the
        flat chain of their elements, so flattening changes nothing about
        what the transformation does. It is what lets the simplifier and
        the composers see two leaves as neighbours when a nesting boundary
        sits between them.

        A nested child is asked to flatten itself, so a child with a shape
        of its own contributes what it chooses: a geometry contributes its
        grid and its transformation, and a multiscale field contributes the
        elements of its finest scale.

        Parameters
        ----------
        endpoints : bool, default=True
            Whether to push the coordinate systems of the chain onto its
            first and last element, which is where the composers read them.
            A child with a shape of its own may contribute a sequence, so
            one level of nesting can survive this mode, and the next pass
            splices it. With `endpoints=False`, nesting is unwound all the
            way to the leaves in one pass, no leaf is rebuilt and each one
            stays the same object, so that a transformation still cancels
            with the lazy inverse that names it; a generic [`Inverse`][] is
            then also replaced by the typed inverse of its forward
            transformation.

        Returns
        -------
        Sequence
            The flattened chain.
        """
        # Subclasses with a fixed shape, such as geometries, override this
        # method.
        if self.transformations is None:
            return self
        if not endpoints:
            return self.to(transformations=_unnest(self.transformations))
        inp, out = self.input, self.output
        flattened = []
        for i, t in enumerate(self.transformations):
            if i == 0 and t.input is None and inp is not None:
                t = t.to(input=inp)
            elif i == len(self) - 1 and t.output is None and out is not None:
                t = t.to(output=out)
            if isinstance(t, Sequence):
                # A nested child, such as a geometry that contributes a
                # grid and then a transformation, may leave one level of
                # nesting, which a second call splices.
                flattened.extend(t.flatten().transformations or [])
            else:
                flattened.append(t)
        return self.to(transformations=flattened)

    # --- conversion to another backend --------------------------------

    def to_backend(
        self,
        backend: tx.Optional[BackendLike] = None,
        *,
        device: tx.Optional[DeviceLike] = None,
    ) -> tx.Any:
        """Refuse to return data, because a sequence holds none.

        A sequence holds transformations rather than data. Its members
        are converted with `as_backend`, or one by one.

        Parameters
        ----------
        backend : str or module, optional
            The setting. By default, the current setting is used.

        Raises
        ------
        TypeError
            Always.
        """
        raise TypeError(
            f"A {type(self).__name__} holds transformations rather than "
            "data. Convert it with as_backend, or convert its "
            "transformations."
        )


class MutableSequence(MutableSequenceMixin, Sequence):
    """A mutable sequence of transformations, stored as a list.

    !!! note
        The transformations are listed in the order in which they are applied,
        which is the opposite of the order used when writing a composition of
        functions or a product of matrices. `Sequence([t1, t2, t3])(x)` is
        equivalent to `t3(t2(t1(x)))`, and `Sequence([t1, t2, t3]) @ x` is
        equivalent to `t3 @ t2 @ t1 @ x`.
    """

    # Narrow the type of the chain to a list.
    _transformations: NotKwOnly[tx.Optional[tx.List[Transformation]]] = None


class ImmutableSequence(Sequence):
    """A [`Sequence`][] whose contents cannot be edited in place.

    The chain is stored as a tuple and the class has no mutating methods, so
    item assignment, deletion and insertion fail.
    """

    # Narrow the type of the chain to a tuple.
    _transformations: NotKwOnly[tx.Optional[tx.Tuple[Transformation, ...]]] = (
        None
    )


class DerivedSequence(ImmutableSequence):
    """An [`ImmutableSequence`][] whose members are derived from an array.

    A file format that stores a field, such as a displacement field read
    from a NIfTI file, is a sequence of the affine transformations and
    the field that the file describes, but what it stores is the array of
    the file. Its members are derived from that array and cached, and
    they are derived again when the array changes. Such a format derives
    from this class rather than from [`ImmutableSequence`][].

    The array is exposed as `data`, as in an image. The backend
    conversions apply to the array, so `to_backend` returns it
    and `as_backend` converts it, and the members are then derived from
    the converted array. A plain sequence instead refuses `to_backend`
    and converts its members one by one, which would leave the stored
    array as it is.

    A subclass declares the field `_data`, which holds an array that was
    assigned, derives its members from `data` in `transformations`, and
    usually gives `data` its own getter, which returns the lazy array of
    its file when no array was assigned. This class declares no field, so
    that the data model of such a format remains the sequence: another
    object is converted into it through its chain of transformations.
    """

    data = smartproperty("data", invalidates="transformations")
    """The stored array, as it is stored.

    Assigning it drops the members derived from the previous array.
    """

    # The conversions of an object that stores an array, rather than
    # those of a sequence, which convert the members.
    to_backend = _BackendsMixin.to_backend
    as_backend = _BackendsMixin.as_backend


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
    """Compute a sequence in a single pass of five stages.

    The stages are ordered by the cost that each one is allowed to
    incur.

    1. Insert a bridge at every boundary where two adjacent
       transformations disagree on the coordinate system that they
       share, because composition assumes compatible systems. Bridging
       is the only stage that adds elements, so it belongs to
       computation rather than simplification. It runs first because a
       bridge reads its neighbours: for example, a reversed array-index
       axis needs the extent of the adjacent grid, which simplification
       may drop. Bridging reads no parameter, so it costs nothing, and
       two opposite bridges cancel when the chain is simplified.
    2. Flatten the chain, so that the next stages see two leaves as
       neighbours even when a nesting boundary sits between them.
    3. Simplify the chain, which also removes the axes that it does not
       need. Simplification precedes every composition because it is the
       only stage that can remove a lazy inverse and its forward
       transformation at no cost. Once a neighbour has been composed
       into a field, the pair that would have cancelled no longer
       exists.
    4. With `factor=True`, rewrite the chain into a normal form
       organized by groups of axes (see [`factor_sequence`][]). The
       chain of each factor is composed inside its group by a
       computation of its own.
    5. Compose the runs of adjacent transformations that the mode
       admits. Each product is simplified as it is produced and is tried
       against its new neighbours, so the result needs no further
       simplification.

    The later stages only remove elements or replace neighbours by their
    product, which keeps the coordinate systems at each boundary, so
    every bridge is inserted by the first stage. What a product makes
    possible, such as the cancellation of the two neighbours of a
    product that is the identity, is done by the composition stage
    itself. The only work that this stage repeats concerns
    `factor=True`: a chain that the factor stage declined is simplified
    and factored once more after it is composed, because composition can
    fold constants that the factor stage cannot carry.
    """
    # --- 1. bridge ---
    # Bridge the direct children before flattening, because a nested
    # sequence carries its endpoints on itself. Flattening then has no
    # endpoint to rebuild, so every leaf stays the same object and
    # cancellation by object identity keeps working.
    bridged = _insert_bridges(seq.transformations or [])

    # --- 2. flatten ---
    flat = _unnest(bridged)
    if not flat:
        return Identity(input=seq.input, output=seq.output)
    seq = seq.to(transformations=flat)
    # Push the endpoints of the sequence onto its first and last
    # elements only if the sequence declares any. In the common case
    # without endpoints, every leaf therefore stays the same object.
    if seq.input is not None or seq.output is not None:
        seq = seq.flatten()

    # --- 3. simplify ---
    seq = _simplify(seq, policy=policy)
    if not isinstance(seq, Sequence):
        # The chain collapsed to a single transformation, so nothing is
        # left to factor or to compose.
        return seq

    # --- 4. factor ---
    if factor:
        seq = factor_sequence(seq, policy, compute=modes)
        if not isinstance(seq, Sequence):
            return seq

    # --- 5. compose ---
    return _compose_chain(seq, modes, policy, factor)


def _compose_chain(
    seq: Sequence,
    modes: tx.List[Family],
    policy: SimplifyTable,
    factor: bool = False,
) -> Transformation:
    # Compose the runs of transformations that the modes admit, which is
    # the last stage of `_compute_sequence`.
    memo: tx.Set[Family] = set()
    composed: Transformation = seq
    for mode in modes:
        composed = _compose_mode(composed, mode, memo, policy, factor)
        if not isinstance(composed, Sequence):
            return composed
    if factor and not _same_leaves(composed, seq):
        # The factor stage carries a constant output only through the
        # elements that pass it through unchanged, so it declines a
        # chain in which an element shifts or scales such an output.
        # Composition folds these constants into a matrix, whose exact
        # zeros the pruning and the factor pass can then read, so the
        # composed chain is simplified and factored. A chain in normal
        # form has nothing to compose outside its factors, so this
        # happens only when the factor stage declined the chain. The
        # factors of the result are composed inside, and nothing outside
        # them composes under `factor=True`.
        composed = _simplify(composed, policy=policy)
        if isinstance(composed, Sequence):
            composed = factor_sequence(composed, policy, compute=modes)
    return composed


def _compose_mode(
    seq: Sequence,
    mode: Family,
    memo: tx.Set[Family],
    policy: SimplifyTable,
    factor: bool = False,
) -> Transformation:
    # Compose the runs of transformations that one mode admits. The
    # child modes are handled first, depth first, so that finer kinds
    # combine before coarser ones. For example, with `mode="affine"`,
    # adjacent translations fold into a single translation before they
    # are widened to an affine.

    # The checks run from the cheapest to the most expensive: a set
    # lookup, a length test, then a walk over the whole chain.

    # --- A mode already visited has nothing left to fold
    if mode in memo:
        return seq

    # --- Check if nothing to do
    if not seq.transformations:
        return seq

    # --- Flatten sequence
    if not _is_flat(seq):
        seq = seq.flatten()

    # --- First, fold every child mode (finer kinds first)
    for child in mode_children(mode):
        seq = _compose_mode(seq, child, memo, policy, factor)
        if not isinstance(seq, Sequence):
            return seq

    # Never visit this mode again.
    memo.add(mode)

    # --- Compose transformations that belong to this mode in order

    # The sweep keeps a stack of the transformations that it has kept so
    # far, and merges each new transformation into the top of the stack
    # for as long as a merge applies. A product is therefore also tried
    # against the transformation on its left, which may have refused the
    # first factor of the product but accept the product itself.
    inputs = list(seq.transformations)
    # The pairs that are neighbours before the sweep have already been
    # offered to the pair simplifiers, by the simplification stage or by
    # the sweep that brought them together.
    neighbours = {(id(a), id(b)) for a, b in zip(inputs, inputs[1:])}
    stack: tx.List[Transformation] = []
    for nxt in inputs:
        while stack:
            merged = _merge(stack[-1], nxt, mode, policy, factor, neighbours)
            if merged is None:
                break
            stack.pop()
            nxt = merged
        stack.append(nxt)

    if len(stack) == 1:
        return stack[0]
    return Sequence(transformations=stack)


def _merge(
    first: Transformation,
    second: Transformation,
    mode: Family,
    policy: SimplifyTable,
    factor: bool,
    neighbours: tx.Set[tx.Tuple[int, int]],
) -> tx.Optional[Transformation]:
    # Return the transformation that replaces two neighbours, where
    # `first` is applied before `second`, or None when the two stay side
    # by side.
    merged = None
    # `is_family` sees through wrappers, so a subspace transformation
    # embedding an affine matches `mode="affine"` and composes with its
    # neighbours. `compose` ignores modes, so `is_family` is what
    # decides which pairs reach it.
    if (
        is_family(first, mode)
        and is_family(second, mode)
        and (not factor or _factor_pair_ok(first, second))
    ):
        try:
            # Compose in sequence order.
            merged = compose(second, first)
        except CompositionError:
            # Transformations that the mode admits but that cannot
            # combine, such as subspace transformations over misaligned
            # axes, are kept side by side.
            pass
    if merged is None and (id(first), id(second)) not in neighbours:
        # Two transformations that a merge has just brought together,
        # such as the two neighbours of a product that collapsed to the
        # identity, may cancel even when the mode does not admit them.
        merged = _simplify(first, second, policy=policy)
    if merged is None or merged is first or merged is second:
        return merged
    # A product is simplified as it is produced, so that, for example, a
    # product that is exactly the identity becomes an `Identity` and
    # merges into its neighbours.
    return _simplify(merged, policy=policy)


def _same_leaves(a: Sequence, b: Sequence) -> bool:
    # Return whether two sequences hold the same objects in the same
    # order.
    left = a.transformations or []
    right = b.transformations or []
    return len(left) == len(right) and all(x is y for x, y in zip(left, right))


def _factor_pair_ok(a: Transformation, b: Transformation) -> bool:
    # Under `factor=True`, a `CartesianField` composes with nothing, and
    # a subspace factor composes only with another one over the same
    # axes or with an `Identity`. Next to anything else, the subspace
    # composers would fold the factor in and destroy the normal form.
    # For the same reason, the projection at the end of the normal form
    # does not compose with the permutation before it. A subspace
    # transformation that moves its axes is never a factor, so it
    # composes as any other transformation does.
    if isinstance(a, CartesianField) or isinstance(b, CartesianField):
        return False
    if (isinstance(a, Projection) and isinstance(b, Permutation)) or (
        isinstance(a, Permutation) and isinstance(b, Projection)
    ):
        return False
    a_sub = _is_factor(a)
    b_sub = _is_factor(b)
    if not a_sub and not b_sub:
        return True
    if isinstance(a, Identity) or isinstance(b, Identity):
        return True
    if a_sub and b_sub:
        return _axes_tuple(a.input_axes) == _axes_tuple(
            b.input_axes
        ) and _axes_tuple(a.output_axes) == _axes_tuple(b.output_axes)
    return False


def _is_factor(t: Transformation) -> bool:
    # Return whether `t` has the shape of a factor of the normal form,
    # that is, a subspace transformation that keeps its axes in place.
    return isinstance(t, SubspaceTransformation) and _axes_tuple(
        t.input_axes
    ) == _axes_tuple(t.output_axes)


def _axes_tuple(axes: tx.Any) -> tx.Optional[tx.Tuple[int, ...]]:
    # Convert an axis vector, which may be an array, to a tuple so that
    # two vectors can be compared.
    return None if axes is None else tuple(int(x) for x in axes)


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
    # Return whether `last @ first` is exactly the identity. This is the case
    # when one end is the lazy inverse of the other, or when both ends are
    # affine and their product is the identity. Matrices that are inverses
    # only up to rounding do not qualify, and an end that is a field is never
    # composed to decide.
    if isinstance(_simplify(first, last), Identity):
        return True
    if not (first.is_affine() and last.is_affine()):
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
    # Replace a generic `Inverse`, which is not yet typed for the family of
    # its forward, with the typed inverse of that forward, so that the engine
    # can compute it and cancellation can recognize it. A typed inverse is
    # returned unchanged.
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
    # `flatten()`, which may rebuild the first or last element and so read
    # a lazy field). Generic inverses are expanded along the way.
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
    # whether it reaches a displacement or coordinate field. A
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
    if isinstance(xform, (DisplacementField, CoordinateField)):
        return True
    return False


# ----------------------------------------------------------------------
#   ADAPTORS
# ----------------------------------------------------------------------


def _splice(spliced: tx.List[Transformation], nxt: Transformation) -> None:
    # Append `nxt`, letting the adaptor place a bridge or a subspace embedding
    # at the boundary. Neither transformation is rebuilt, so each leaf stays
    # the same object that its inverse names.
    if not spliced:
        spliced.append(nxt)
        return
    prev = spliced[-1]
    pieces = list(
        nocycles.ADAPT(prev, nxt, allow_type_grouped_positional=True)
    )
    if pieces[0] is not prev:
        # `prev` was embedded in the larger space of `nxt`, so its left
        # boundary changed. Splice the embedded piece again against its left
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
        # Only plain sequences are rebuilt. A specialized sequence, such as a
        # multiscale field or a geometry, keeps its own shape.
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
