"""Tests for the pure set/group-theory hierarchy (post-#88b revert).

C1 restores the hierarchy to plain `.register` membership with a
case-insensitive `NAMETOCLASS`, and adds `InjectiveTransformation` /
`SurjectiveTransformation` as bases of `BijectiveTransformation`. The
`@member` / `MEMBERS` / `node_of_class` / `type_from_name` / `NAMETOCLASS_CI`
machinery and the `Meta`/`Subspace`/`Inverse` nodes of #88b are gone;
wrapper and value-dependent keys are handled in the policy layer instead
(see `test_kind_membership.py`).
"""

import pytest

from brainhops.datamodel import hierarchy
from brainhops.datamodel.transformations import Affine, Linear, Translation

# ----------------------------------------------------------------------
#   CASE-INSENSITIVE parse
# ----------------------------------------------------------------------


def test_parse_name_is_case_insensitive() -> None:
    for name in ("affine", "Affine", "AFFINE", "AfFiNe"):
        cls, dim = hierarchy.TransformationFamily.parse(name)
        assert cls is hierarchy.AffineTransformation
        assert dim is None
    assert hierarchy.TransformationFamily.parse("translation")[0] is (
        hierarchy.Translation
    )
    assert hierarchy.TransformationFamily.parse("rotation")[0] is (
        hierarchy.SpecialOrthogonalTransformation
    )


def test_parse_symbols_stay_case_sensitive() -> None:
    # SYMBOL / FSYMBOL are mathematical symbols, matched case-sensitively.
    cls, dim = hierarchy.TransformationFamily.parse("SO(3)")
    assert cls is hierarchy.SpecialOrthogonalTransformation
    assert dim == 3
    assert hierarchy.TransformationFamily.parse("SO")[0] is (
        hierarchy.SpecialOrthogonalTransformation
    )


def test_parse_raises_on_unknown() -> None:
    with pytest.raises(ValueError):
        hierarchy.TransformationFamily.parse("DefinitelyNotAType")


# ----------------------------------------------------------------------
#   INJECTIVE / SURJECTIVE ADDED UNDER BIJECTIVE
# ----------------------------------------------------------------------


def test_injective_surjective_are_bases_of_bijective() -> None:
    assert issubclass(
        hierarchy.BijectiveTransformation, hierarchy.InjectiveTransformation
    )
    assert issubclass(
        hierarchy.BijectiveTransformation, hierarchy.SurjectiveTransformation
    )
    # Every invertible node is therefore injective and surjective.
    assert issubclass(
        hierarchy.InvertibleAffineTransformation,
        hierarchy.InjectiveTransformation,
    )
    assert issubclass(
        hierarchy.SpecialOrthogonalTransformation,
        hierarchy.SurjectiveTransformation,
    )


def test_euclidean_lattice_edges() -> None:
    # The approved lattice fix: a special-euclidean map is euclidean, and a
    # translation is special-euclidean (hence euclidean), while a translation
    # is still not orthogonal.
    assert issubclass(
        hierarchy.SpecialEuclideanTransformation,
        hierarchy.EuclideanTransformation,
    )
    assert issubclass(
        hierarchy.Translation, hierarchy.SpecialEuclideanTransformation
    )
    assert issubclass(hierarchy.Translation, hierarchy.EuclideanTransformation)
    assert not issubclass(
        hierarchy.Translation, hierarchy.OrthogonalTransformation
    )
    # And a concrete translation is a member of the euclidean set.
    from brainhops.datamodel.transformations import Translation, is_kind

    assert is_kind(Translation(translation=[1.0, 2.0]), "euclidean")


def test_injective_surjective_names_resolve() -> None:
    assert hierarchy.TransformationFamily.parse("injection")[0] is (
        hierarchy.InjectiveTransformation
    )
    assert hierarchy.TransformationFamily.parse("surjective")[0] is (
        hierarchy.SurjectiveTransformation
    )
    assert hierarchy.TransformationFamily.parse("bijection")[0] is (
        hierarchy.BijectiveTransformation
    )


# ----------------------------------------------------------------------
#   TRANSITIVE (isinstance) MEMBERSHIP VIA .register
# ----------------------------------------------------------------------


def test_concrete_membership_is_transitive() -> None:
    # `Linear` registers to `LinearTransformation`, a subclass of
    # `AffineTransformation`, so a linear instance is an affine.
    assert isinstance(
        Linear(matrix=[[2.0, 0.0], [0.0, 3.0]]), hierarchy.AffineTransformation
    )
    # The converse does not hold: a general affine is not linear.
    assert not isinstance(
        Affine(matrix=[[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]),
        hierarchy.LinearTransformation,
    )
    assert isinstance(
        Translation(translation=[1.0, 2.0]), hierarchy.AffineTransformation
    )


# ----------------------------------------------------------------------
#   #88b MACHINERY / NODES ARE GONE
# ----------------------------------------------------------------------


def test_field_transformation_node_absent() -> None:
    assert not hasattr(hierarchy, "FieldTransformation")


def test_removed_88b_nodes_are_gone() -> None:
    for name in (
        "MetaTransformation",
        "SubspaceTransformation",
        "InverseTransformation",
    ):
        assert not hasattr(hierarchy, name), name


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
        assert not hasattr(hierarchy, name), name


def test_nametoclass_is_lowercase_keyed() -> None:
    assert "affine" in hierarchy.NAMETOCLASS
    assert "Affine" not in hierarchy.NAMETOCLASS
