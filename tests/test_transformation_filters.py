"""Tests for the filters and the conversions to a kind of #589.

The design is recorded in spec/design/transformation-filters.md.
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel import kinds
from brainhops.datamodel.axes import A, R, S, TimeAxis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    DisplacementField,
    Identity,
    Inverse,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Sequence,
    Transformation,
    Translation,
)
from brainhops.errors import ConversionError

MATRIX = np.array(
    [
        [2.0, 0.0, 0.0, 1.0],
        [0.0, 3.0, 0.0, 2.0],
        [0.0, 0.0, 4.0, 3.0],
    ]
)


def _system(name: str) -> CoordinateSystem:
    return CoordinateSystem(name=name, axes=[R(), A(), S()])


def _field() -> DisplacementField:
    return DisplacementField(
        np.zeros((4, 5, 6, 3)), input=_system("b"), output=_system("c")
    )


def _chain() -> Sequence:
    affine = Affine(MATRIX, input=_system("a"), output=_system("b"))
    return Sequence(transformations=[affine, _field()])


# --- matchers ---------------------------------------------------------


@pytest.mark.parametrize(
    "match",
    [
        "affine",
        kinds.Affine,
        Affine,
        lambda t: isinstance(t, Affine),
        [Rotation, "affine"],
        (Translation, Affine),
    ],
)
def test_every_matcher_form_selects_an_affine(match: tx.Any) -> None:
    filtered = _chain().without(match)
    first, second = filtered.transformations
    assert isinstance(first, Identity)
    assert isinstance(second, DisplacementField)


def test_a_class_matches_by_instance_and_not_by_kind() -> None:
    filtered = _chain().without(DisplacementField)
    first, second = filtered.transformations
    assert isinstance(first, Affine)
    assert isinstance(second, Identity)


def test_an_unknown_matcher_is_refused() -> None:
    with pytest.raises(TypeError):
        _chain().without(3)


# --- without and with_only --------------------------------------------


def test_a_replaced_part_keeps_its_systems() -> None:
    field = _field()
    identity = Sequence(transformations=[field]).without(DisplacementField)
    (part,) = identity.transformations
    assert isinstance(part, Identity)
    assert part.input is field.input
    assert part.output is field.output


def test_with_only_replaces_the_parts_that_do_not_match() -> None:
    first, second = _chain().with_only("affine").transformations
    assert isinstance(first, Affine)
    assert isinstance(second, Identity)


def test_a_filter_that_changes_nothing_returns_the_transformation() -> None:
    chain = _chain()
    assert chain.without(Rotation) is chain
    assert chain.with_only([Affine, DisplacementField]) is chain
    affine = Affine(MATRIX)
    assert affine.with_only("affine") is affine


def test_a_node_that_matches_as_a_whole_is_not_split() -> None:
    chain = _chain()

    def is_chain(xform: Transformation) -> bool:
        return xform is chain

    assert chain.with_only(is_chain) is chain
    assert isinstance(chain.without(is_chain), Identity)


def test_a_nested_sequence_is_filtered() -> None:
    inner = _chain()
    outer = Sequence(transformations=[inner, Affine(MATRIX)])
    filtered = outer.with_only("affine")
    nested, last = filtered.transformations
    assert last is outer.transformations[1]
    assert isinstance(nested.transformations[0], Affine)
    assert isinstance(nested.transformations[1], Identity)


def test_the_inverse_of_a_removed_part_is_removed() -> None:
    inverse = Inverse(_chain())
    filtered = inverse.with_only("affine")
    assert filtered is not inverse
    affine = filtered.to_affine()
    expected = Affine(MATRIX).inverse().to_affine()
    np.testing.assert_allclose(affine.matrix, expected.matrix)


def test_the_inverse_of_a_field_is_removed() -> None:
    inverse = Inverse(_field())
    assert isinstance(inverse.without(DisplacementField), Identity)


def test_a_part_that_changes_the_number_of_axes_is_refused() -> None:
    full = CoordinateSystem(name="f", axes=[R(), A(), S(), TimeAxis("t")])
    part = Affine(input=full, output=_system("s"))
    with pytest.raises(ValueError):
        part.without("affine")


# --- conversions to a kind --------------------------------------------


def test_to_affine_ignores_the_fields_of_a_chain() -> None:
    chain = _chain()
    assert chain.to_affine(on_error=None) is None
    affine = chain.to_affine(include="affine", on_error=None)
    np.testing.assert_allclose(affine.matrix, MATRIX)
    affine = chain.to_affine(exclude=DisplacementField)
    np.testing.assert_allclose(affine.matrix, MATRIX)


def test_to_affine_passes_its_options_to_to() -> None:
    assert _field().to_affine(on_error=None) is None
    with pytest.raises(ConversionError):
        _field().to_affine()


@pytest.mark.parametrize(
    "xform, method, cls",
    [
        (Translation([1.0, 2.0]), "to_translation", Translation),
        (Linear(np.diag([2.0, 3.0])), "to_scaling", Scaling),
        (
            Linear(np.array([[0.0, 1.0], [1.0, 0.0]])),
            "to_permutation",
            Permutation,
        ),
        (Linear(np.eye(2)), "to_rotation", Rotation),
        (Scaling([2.0, 3.0]), "to_linear", Linear),
        (Translation([1.0, 2.0]), "to_affine", Affine),
    ],
)
def test_each_to_method_returns_its_class(
    xform: Transformation, method: str, cls: tx.Type[Transformation]
) -> None:
    result = getattr(xform, method)()
    assert isinstance(result, cls)
    np.testing.assert_allclose(
        result.to_affine().matrix, xform.to_affine().matrix
    )


def test_to_kind_accepts_a_kind_and_refuses_one_without_a_class() -> None:
    assert isinstance(Scaling([2.0]).to_kind(kinds.Linear), Linear)
    with pytest.raises(ValueError):
        Scaling([2.0]).to_kind(kinds.Identity)


# --- include= and exclude= --------------------------------------------


def test_include_and_exclude_filter_before_the_conversion() -> None:
    chain = _chain()
    # Both may be given: the parts to keep, then the parts to drop.
    affine = chain.to_affine(
        include=["affine", DisplacementField], exclude=DisplacementField
    )
    np.testing.assert_allclose(affine.matrix, MATRIX)
    # Excluding the affine leaves the field, which is not affine.
    assert chain.to_affine(exclude="affine", on_error=None) is None


def test_ignore_is_no_longer_a_filter() -> None:
    # `ignore=` is not a keyword of the filters, so it reaches `to()`,
    # which refuses it.
    with pytest.raises(TypeError):
        _chain().to_affine(ignore=True)


# --- the collections of a matcher -------------------------------------


@pytest.mark.parametrize(
    "match",
    [["affine"], ("affine",), {"affine"}, frozenset({"affine"})],
)
def test_a_collection_selects_with_any_of_its_items(match: object) -> None:
    filtered = _chain().with_only(match)
    assert isinstance(filtered.transformations[1], Identity)


@pytest.mark.parametrize(
    "match",
    [
        Sequence(transformations=[Affine(MATRIX)]),
        (item for item in ["affine"]),
        iter(["affine"]),
        3,
    ],
    ids=["sequence", "generator", "iterator", "int"],
)
def test_another_iterable_is_not_a_matcher(match: object) -> None:
    with pytest.raises(TypeError):
        _chain().without(match)


# --- the parameters of the result -------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        "to_translation",
        "to_scaling",
        "to_permutation",
        "to_rotation",
        "to_linear",
        "to_affine",
    ],
)
@pytest.mark.parametrize(
    "xform",
    [Identity(), Inverse(forward=Affine())],
    ids=["identity", "inverse-of-an-unset-affine"],
)
def test_a_result_without_parameters_is_refused(
    method: str, xform: object
) -> None:
    # Without a dimension, the result would have no matrix or no data.
    with pytest.raises(ConversionError):
        getattr(xform, method)()
    assert getattr(xform, method)(on_error=None) is None


def test_to_makes_no_check_of_the_parameters() -> None:
    affine = Identity().to(Affine)
    assert type(affine) is Affine
    assert affine.matrix is None
