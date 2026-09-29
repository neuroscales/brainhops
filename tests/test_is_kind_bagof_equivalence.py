"""Regression guard: `is_kind`'s bagof-based dispatch matches the algorithm.

`is_kind` was moved off a bespoke `Dispatcher`/`type_distance` registry onto
a [`bagof.dispatchers`][] `Function` (source covariant, kind contravariant via
a `Super` lower bound, `compute` carried) with a per-node most-specific
reducer feeding an OR. The dispatch it must reproduce is written out here,
verbatim, as a frozen oracle: the old `type_distance` metric and the old
candidate selection (per-node nearest source by MRO, dedup by function
identity), reading the SAME registered checkers the live `is_kind` uses.

Two things are asserted over every (concrete-instance, kind-node) pair:

* the *set* of checkers the two select is identical -- an order-independent
  comparison that is the true statement of dispatch equivalence; and
* the end-to-end boolean is identical at both `compute` levels.

This is the migration's equivalence check kept as a regression test. If a
future change to the dispatch (or to `bagof.dispatchers`' selection) diverges
from the documented algorithm, one of these assertions fails.
"""

from functools import lru_cache

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel import kinds as H
from brainhops.datamodel._transformations.check import is_kind, normalize_kind
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    DisplacementField,
    Identity,
    Inverse,
    Linear,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)

# --- the pre-refactor dispatch, frozen as an oracle -------------------


def _type_distance(t1: type, t2: type, oriented: bool = True) -> float:
    if t1 is t2:
        return 0
    if issubclass(t1, t2):
        n = len(t1.__bases__)
        return 1 + min(
            _type_distance(base, t2, oriented=False) + (i / n)
            for i, base in enumerate(t1.__bases__)
        )
    if issubclass(t2, t1) and not oriented:
        return _type_distance(t2, t1, oriented=True)
    return float("inf")


_type_distance = lru_cache(maxsize=None)(_type_distance)  # noqa: UP033

Gated = tx.Union[bool, type]
Selection = tx.Dict[type, tx.Tuple[tx.Tuple[int, float], int, tx.Callable]]


def _gate(x: object, kind: type) -> Gated:
    """The shared entry gating: returns a bool answer, or the kind node."""
    kind = normalize_kind(kind)
    if isinstance(x, kind):
        return True
    if not H.is_transformation_set(kind):
        return False
    return kind


def _oracle_selection(x: object, kind: type) -> Selection:
    """The old per-node nearest-source selection over the live registry."""
    key_src = type(x)
    mro = key_src.__mro__
    nearest: Selection = {}  # node -> (distance, order, impl)
    for order, ((rsrc, rnode), shim) in enumerate(is_kind._registry.items()):
        if not issubclass(key_src, rsrc):
            continue
        source_rank = mro.index(rsrc) if rsrc in mro else len(mro)
        if _type_distance(rnode, kind) == float("inf"):
            continue
        dist = (source_rank, _type_distance(rnode, kind))
        current = nearest.get(rnode)
        if current is None or dist < current[0]:
            nearest[rnode] = (dist, order, shim._impl)
    return nearest


def oracle_is_kind(x: object, kind: type, compute: bool = False) -> bool:
    gated = _gate(x, kind)
    if gated is True or gated is False:
        return gated
    kind, compute = gated, bool(compute)
    nearest = _oracle_selection(x, kind)
    seen: tx.Set[int] = set()
    for _dist, _order, func in sorted(nearest.values(), key=lambda c: c[:2]):
        if id(func) in seen:
            continue
        seen.add(id(func))
        if func(x, kind, compute):
            return True
    return False


def oracle_selected_set(x: object, kind: type) -> tx.Union[bool, frozenset]:
    gated = _gate(x, kind)
    if gated is True or gated is False:
        return gated
    return frozenset(
        id(impl) for _d, _o, impl in _oracle_selection(x, gated).values()
    )


def live_selected_set(x: object, kind: type) -> tx.Union[bool, frozenset]:
    """The set of checker implementations the live `is_kind` would OR.

    Read through the same (memoized) selection `is_kind` itself uses, so
    the comparison covers the per-`(type, kind)` memo as well as the
    library's enumeration behind it.
    """
    gated = _gate(x, kind)
    if gated is True or gated is False:
        return gated
    return frozenset(id(impl) for impl in is_kind._selection(x, gated, False))


# --- valid, value-varied instances covering every checker branch ------

_I2 = np.eye(2)
_ROT = np.array([[0.0, -1.0], [1.0, 0.0]])
_PERM = np.array([[0.0, 1.0], [1.0, 0.0]])
_GEN = np.array([[2.0, 1.0], [0.0, 3.0]])
_WIDE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_TALL = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
_AFF_I = np.eye(3)[:2]
_AFF_ROT = np.concatenate([_ROT, np.zeros((2, 1))], axis=1)
_AFF_T = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]])
_AFF_GEN = np.concatenate([_GEN, np.array([[1.0], [2.0]])], axis=1)
_AFF_PERM = np.concatenate([_PERM, np.zeros((2, 1))], axis=1)
_AFF_DIAG = np.concatenate([np.diag([2.0, 3.0]), np.zeros((2, 1))], axis=1)

INSTANCES = [
    Identity(),
    Scaling(scale=[2.0, 3.0]),
    Scaling(scale=[2.0, 2.0]),
    Scaling(scale=[1.0, 1.0]),
    Scaling(scale=[-1.0, 1.0]),
    Scaling(),
    Translation(translation=[1.0, 2.0]),
    Translation(translation=[0.0, 0.0]),
    Translation(),
    Permutation(permutation=[1, 0]),
    Permutation(permutation=[0, 1]),
    Permutation(permutation=[1, 2, 0]),
    Permutation(),
    Linear(matrix=_I2),
    Linear(matrix=_ROT),
    Linear(matrix=_PERM),
    Linear(matrix=_GEN),
    Linear(matrix=np.diag([2.0, 3.0])),
    Linear(matrix=_WIDE),
    Linear(matrix=_TALL),
    Linear(),
    Rotation(matrix=_ROT),
    Rotation(),
    Affine(matrix=_AFF_I),
    Affine(matrix=_AFF_ROT),
    Affine(matrix=_AFF_T),
    Affine(matrix=_AFF_GEN),
    Affine(matrix=_AFF_PERM),
    Affine(matrix=_AFF_DIAG),
    Affine(),
    DisplacementField(field=np.zeros((2, 2, 2))),
    Affine(matrix=_AFF_ROT).inverse(),
    Scaling(scale=[2.0, 3.0]).inverse(),
    Translation(translation=[1.0, 2.0]).inverse(),
    Rotation(matrix=_ROT).inverse(),
    Permutation(permutation=[1, 0]).inverse(),
    Inverse(),
    Sequence([Scaling(scale=[2.0, 2.0]), Translation(translation=[1.0, 1.0])]),
    Sequence([Rotation(matrix=_ROT), Translation(translation=[1.0, 1.0])]),
    Sequence(),
    SubspaceTransformation(transformation=Rotation(matrix=_ROT)),
    SubspaceTransformation(
        transformation=Scaling(scale=[2.0]),
        input_axes=[0],
        output_axes=[0],
    ),
    SubspaceTransformation(),
    Projection(dropped=[1], created=[]),
    Projection(dropped=[], created=[1]),
    Projection(dropped=[0], created=[1]),
    Projection(),
    Bijection(forward=Affine(matrix=_AFF_GEN)),
    Bijection(),
]

NODES = sorted(H.all_sets(), key=lambda c: c.__name__)


@pytest.mark.parametrize("x", INSTANCES, ids=lambda x: type(x).__name__)
def test_selected_checker_set_matches_oracle(x: object) -> None:
    # Order-independent: the live dispatch selects exactly the checkers the
    # pre-refactor algorithm selects, per node, for every kind.
    for node in NODES:
        assert live_selected_set(x, node) == oracle_selected_set(x, node), (
            f"{type(x).__name__} / {node.__name__}"
        )


@pytest.mark.parametrize("x", INSTANCES, ids=lambda x: type(x).__name__)
def test_membership_boolean_matches_oracle(x: object) -> None:
    for node in NODES:
        for compute in (False, True):
            assert is_kind(x, node, compute=compute) == oracle_is_kind(
                x, node, compute=compute
            ), f"{type(x).__name__} / {node.__name__} / compute={compute}"
