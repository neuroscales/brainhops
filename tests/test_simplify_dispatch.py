"""The bagof-dispatchers 0.3 pair rework: chain of responsibility + accessor.

Pairs are dispatched as a *chain of responsibility* over
`Function.candidates()` (bagof 0.3): every applicable pair simplifier is
tried most-specific-first (specificity ties in registration order) until one
returns non-`None`. Leaves stay single-winner. The two getters
(`get_simplifier` / `get_pair_simplifiers`) are collapsed into one
arity-dispatched `get_simplifiers`.
"""

import pytest
from bagof.dispatchers import Function

from brainhops.datamodel._transformations import simplify as _S
from brainhops.datamodel._transformations.concrete import Affine, Identity
from brainhops.datamodel._transformations.inverse import Inverse
from brainhops.datamodel._transformations.simplify import (
    SimplifyTable,
    get_simplifiers,
)


def _identity() -> Identity:
    return Identity()


# ----------------------------------------------------------------------
#   the collapsed, arity-dispatched accessor
# ----------------------------------------------------------------------


def test_get_simplifiers_leaf_is_optional_callable() -> None:
    leaf = get_simplifiers(Identity)
    assert callable(leaf)
    # A concrete callable, not a bagof `Method` wrapper.
    assert type(leaf).__name__ != "Method"
    # Every transform has a leaf (the `Transformation` catch-all); a
    # non-transform type has none.
    assert get_simplifiers(int) is None


def test_get_simplifiers_pair_is_ordered_candidate_tuple() -> None:
    pair = get_simplifiers(Identity, Inverse)
    assert isinstance(pair, tuple)
    assert all(callable(f) for f in pair)
    # `(Identity, Inverse)` is one of the ~255 genuine specificity ties, so
    # the chain has more than one applicable rule.
    assert len(pair) >= 2
    # The accessor's order is exactly the chain `simplify` walks: the
    # applicable methods most-specific-first, ties in registration order.
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
    # The rework dropped the registration-order `priority` tiebreak: every
    # registered simplifier (leaf and pair) sits at the default priority, and
    # ties are resolved by `candidates()` registration order instead.
    assert all(m.priority == 0 for m in _S._simplify.methods)


def test_chain_falls_through_when_the_most_specific_declines() -> None:
    # The property `simplify`'s pair loop relies on: a more specific rule may
    # decline (return `None`) and hand off to a less specific one. Modelled on
    # a fresh `Function` so the mechanism is pinned independently of the
    # concrete rule set.
    class A:
        pass

    class B(A):
        pass

    chain = Function("chain")

    @chain.register
    def specific(first: B, second: B, policy: object) -> object:
        return None  # declines

    @chain.register
    def general(first: A, second: A, policy: object) -> object:
        return "general"

    b = B()
    order = [m.function.__name__ for m in chain.candidates(b, b, object())]
    assert order == ["specific", "general"]  # most specific first

    def run() -> object:
        for cand in chain.candidates(b, b, object()):
            result = cand(b, b, object())
            if result is not None:
                return result
        return None

    assert run() == "general"  # fell through the declining specific rule


def test_simplify_pair_declines_return_none_not_raise() -> None:
    # Two transforms with no collapsing rule between them: the chain empties
    # (or every rule declines) and `simplify` answers `None`, never raising.
    aff = Affine(matrix=[[2.0, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3]])
    assert _S.simplify(aff, aff, policy="analytic") is None


def test_simplify_identity_pair_collapses() -> None:
    # An identity next to anything disappears: the pair collapses to the other
    # operand. Exercises the chain end to end.
    aff = Affine(matrix=[[2.0, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3]])
    out = _S.simplify(_identity(), aff, policy="analytic")
    assert out is aff
