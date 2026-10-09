"""Tests for the set-theoretic hierarchy of transformation kinds."""

import pytest

from brainhops.datamodel import kinds
from brainhops.datamodel.transformations import Affine, Linear, Translation

# ----------------------------------------------------------------------
#   CASE-INSENSITIVE parse
# ----------------------------------------------------------------------


def test_parse_name_is_case_insensitive() -> None:
    for name in ("affine", "Affine", "AFFINE", "AfFiNe"):
        family = kinds.TransformationFamily.parse(name)
        assert family.kind is kinds.Affine
        assert family.ndim is None
    assert kinds.TransformationFamily.parse("translation").kind is (
        kinds.Translation
    )
    assert kinds.TransformationFamily.parse("rotation").kind is (
        kinds.SpecialOrthogonal
    )


def test_parse_symbols_stay_case_sensitive() -> None:
    # Mathematical symbols are matched case-sensitively.
    family = kinds.TransformationFamily.parse("SO(3)")
    assert family.kind is kinds.SpecialOrthogonal
    assert family.ndim == 3
    assert kinds.TransformationFamily.parse("SO").kind is (
        kinds.SpecialOrthogonal
    )


def test_parse_raises_on_unknown() -> None:
    with pytest.raises(ValueError):
        kinds.TransformationFamily.parse("DefinitelyNotAType")


# ----------------------------------------------------------------------
#   INJECTIVE / SURJECTIVE ADDED UNDER BIJECTIVE
# ----------------------------------------------------------------------


def test_injective_surjective_are_bases_of_bijective() -> None:
    assert issubclass(kinds.Bijection, kinds.Injection)
    assert issubclass(kinds.Bijection, kinds.Surjection)
    # Every invertible kind is both injective and surjective.
    assert issubclass(
        kinds.InvertibleAffine,
        kinds.Injection,
    )
    assert issubclass(
        kinds.SpecialOrthogonal,
        kinds.Surjection,
    )


def test_euclidean_lattice_edges() -> None:
    # Translations are special Euclidean, hence Euclidean, but not orthogonal.
    assert issubclass(
        kinds.SpecialEuclidean,
        kinds.Euclidean,
    )
    assert issubclass(kinds.Translation, kinds.SpecialEuclidean)
    assert issubclass(kinds.Translation, kinds.Euclidean)
    assert not issubclass(kinds.Translation, kinds.Orthogonal)
    # A concrete translation belongs to the Euclidean kind.
    from brainhops.datamodel.transformations import Translation, is_kind

    assert is_kind(Translation(translation=[1.0, 2.0]), "euclidean")


def test_injective_surjective_names_resolve() -> None:
    assert kinds.TransformationFamily.parse("injection").kind is (
        kinds.Injection
    )
    assert kinds.TransformationFamily.parse("surjective").kind is (
        kinds.Surjection
    )
    assert kinds.TransformationFamily.parse("bijection").kind is (
        kinds.Bijection
    )


# ----------------------------------------------------------------------
#   TRANSITIVE (isinstance) MEMBERSHIP VIA .register
# ----------------------------------------------------------------------


def test_concrete_membership_is_transitive() -> None:
    # Linear is registered under kinds.Linear, a subclass of kinds.Affine.
    assert isinstance(Linear(matrix=[[2.0, 0.0], [0.0, 3.0]]), kinds.Affine)
    # The converse fails: a general affine is not linear.
    assert not isinstance(
        Affine(matrix=[[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]),
        kinds.Linear,
    )
    assert isinstance(Translation(translation=[1.0, 2.0]), kinds.Affine)


# ----------------------------------------------------------------------
#   #88b MACHINERY / NODES ARE GONE
# ----------------------------------------------------------------------


def test_field_transformation_node_absent() -> None:
    assert not hasattr(kinds, "FieldTransformation")


def test_removed_88b_nodes_are_gone() -> None:
    for name in (
        "MetaTransformation",
        "SubspaceTransformation",
        "InverseTransformation",
    ):
        assert not hasattr(kinds, name), name


def test_removed_88b_helpers_are_gone() -> None:
    for name in (
        "member",
        "MEMBERS",
        "CLASS_TO_NODE",
        "NON_ADDRESSABLE",
        "node_of_class",
        "type_from_name",
        "is_member",
        "NAMETOCLASS_CI",
    ):
        assert not hasattr(kinds, name), name


def test_nametoclass_is_lowercase_keyed() -> None:
    assert "affine" in kinds.NAMETOCLASS
    assert "Affine" not in kinds.NAMETOCLASS
