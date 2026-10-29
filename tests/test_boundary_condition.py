"""The names of boundary conditions and their aliases."""

import pytest
import typing_extensions as tx

from brainhops.datamodel.enums import BoundaryCondition, BoundaryName
from brainhops.datamodel.transformations import DisplacementField

ALIASES = {
    "nearest": ["edge", "border", "replicate"],
    "reflect": ["symmetric", "dct2", "grid-mirror"],
    "mirror": ["dct1"],
    "grid-wrap": ["wrap", "circular", "circulant", "dft"],
    "constant": ["zero", "zeros", "grid-constant"],
    "antimirror": ["dst1"],
    "antireflect": ["dst2"],
}


@pytest.mark.parametrize("value", sorted(ALIASES))
def test_every_alias_converts_to_its_member(value: str) -> None:
    member = BoundaryCondition(value)
    for alias in ALIASES[value]:
        assert BoundaryCondition(alias) is member
        assert BoundaryCondition(alias.upper()) is member
        assert getattr(BoundaryCondition, alias.replace("-", "")) is member


def test_the_literal_lists_every_name() -> None:
    names = {name for value in ALIASES for name in [value] + ALIASES[value]}
    assert set(tx.get_args(BoundaryName)) == names


def test_wrap_is_the_periodic_boundary() -> None:
    assert BoundaryCondition.wrap is BoundaryCondition.gridwrap
    assert BoundaryCondition.wrap == "grid-wrap"


def test_a_field_reads_an_alias_with_a_hyphen() -> None:
    np = pytest.importorskip("numpy")
    field = DisplacementField(data=np.zeros((2, 2, 2)), bound="grid-mirror")
    assert field.bound is BoundaryCondition.reflect


def test_an_unknown_name_is_refused() -> None:
    with pytest.raises(ValueError):
        BoundaryCondition("periodic")
