"""Tests for the ``from_*`` constructors on ``DataModelBase``."""

import typing_extensions as tx
from bagof.magic import Field

from brainhops.datamodel.base import DataModelBase


class _Plain(DataModelBase):
    x: tx.Optional[int] = None
    y: tx.Optional[int] = None


class _Aliased(DataModelBase):
    x: tx.Annotated[tx.Optional[int], Field(alias="ex")] = None


def test_from_dict_carries_plain_fields() -> None:
    # Regression (#61): the incoming values were keyed on `field.alias`,
    # which is unset for a plain field, so the value was silently dropped.
    obj = _Plain.from_dict({"x": 5, "y": 7})
    assert obj.x == 5
    assert obj.y == 7


def test_from_dict_honors_an_explicit_alias() -> None:
    obj = _Aliased.from_dict({"ex": 9})
    assert obj.x == 9


def test_from_dict_lets_keyword_arguments_take_precedence() -> None:
    obj = _Plain.from_dict({"x": 5, "y": 7}, y=99)
    assert obj.x == 5
    assert obj.y == 99


def test_from_instance_carries_plain_fields() -> None:
    # Regression (#61): the value was keyed on `field.alias`, so
    # reconstruction raised `TypeError: keywords must be strings`.
    obj = _Plain.from_instance(_Plain(x=5, y=7))
    assert obj.x == 5
    assert obj.y == 7


def test_from_instance_honors_an_explicit_alias() -> None:
    obj = _Aliased.from_instance(_Aliased(ex=9))
    assert obj.x == 9


def test_from_other_dispatches_to_dict_and_instance() -> None:
    from_dict = _Plain.from_other({"x": 5, "y": 7})
    assert (from_dict.x, from_dict.y) == (5, 7)
    from_instance = _Plain.from_other(_Plain(x=5, y=7))
    assert (from_instance.x, from_instance.y) == (5, 7)
