"""Tests for the dispatch of simplifiers.

Pair simplifiers form a chain of responsibility over Function.candidates(),
tried from the most specific until one returns something other than None.
"""

import pytest
from bagof.dispatchers import Function

from brainhops.datamodel._transformations.compute import simplify as _S
from brainhops.datamodel._transformations.compute.simplify import (
    SimplifyTable,
    get_simplifiers,
)
from brainhops.datamodel._transformations.concrete import Affine, Identity
from brainhops.datamodel._transformations.inverse import Inverse


def _identity() -> Identity:
    return Identity()


# ----------------------------------------------------------------------
#   the collapsed, arity-dispatched accessor
# ----------------------------------------------------------------------


def test_get_simplifiers_leaf_is_optional_callable() -> None:
    leaf = get_simplifiers(Identity)
    assert callable(leaf)
    assert type(leaf).__name__ != "Method"
    # Every transformation finds a leaf rule through the catch-all rule for
    # Transformation, but a type that is not a transformation finds none.
    assert get_simplifiers(int) is None


def test_get_simplifiers_pair_is_ordered_candidate_tuple() -> None:
    pair = get_simplifiers(Identity, Inverse)
    assert isinstance(pair, tuple)
    assert all(callable(f) for f in pair)
    # Several rules tie in specificity.
    assert len(pair) >= 2
    # The accessor returns the chain in the order simplify walks it.
    expected = tuple(
        m.function
        for m in _S._simplify.resolve_candidates(
            Identity, Inverse, SimplifyTable
        )
    )
    assert pair == expected


def test_get_simplifiers_rejects_wrong_arity() -> None:
    with pytest.raises(TypeError):
        get_simplifiers()
    with pytest.raises(TypeError):
        get_simplifiers(Identity, Inverse, Affine)


def test_public_api_is_collapsed() -> None:
    assert "get_simplifiers" in _S.__all__
    assert "get_simplifier" not in _S.__all__
    assert "get_pair_simplifiers" not in _S.__all__
    assert not hasattr(_S, "get_simplifier")
    assert not hasattr(_S, "get_pair_simplifiers")


# ----------------------------------------------------------------------
#   chain-of-responsibility semantics
# ----------------------------------------------------------------------


def test_pairs_carry_no_priority() -> None:
    # Pair rules carry no priority, so ties are ordered by candidates().
    assert all(m.priority == 0 for m in _S._simplify.methods)


def test_chain_falls_through_when_the_most_specific_declines() -> None:
    # The test uses a fresh Function so that it does not depend on the real
    # rules.
    class A:
        pass

    class B(A):
        pass

    chain = Function("chain")

    @chain.register
    def specific(first: B, second: B, policy: object) -> object:
        return None

    @chain.register
    def general(first: A, second: A, policy: object) -> object:
        return "general"

    b = B()
    order = [m.function.__name__ for m in chain.candidates(b, b, object())]
    assert order == ["specific", "general"]

    def run() -> object:
        for cand in chain.candidates(b, b, object()):
            result = cand(b, b, object())
            if result is not None:
                return result
        return None

    assert run() == "general"


def test_simplify_pair_declines_return_none_not_raise() -> None:
    aff = Affine(matrix=[[2.0, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3]])
    assert _S.simplify(aff, aff, policy="analytic") is None


def test_simplify_identity_pair_collapses() -> None:
    aff = Affine(matrix=[[2.0, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3]])
    out = _S.simplify(_identity(), aff, policy="analytic")
    assert out is aff
