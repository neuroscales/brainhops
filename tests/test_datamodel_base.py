"""Tests for the ``from_*`` constructors on ``DataModelBase``, and for the
converter that uses them."""

import pytest
import typing_extensions as tx
from bagof.converters import ConversionError, get_converter
from bagof.magic import Field

from brainhops._core.typing import Const
from brainhops.datamodel.base import DataModelBase


class _Plain(DataModelBase):
    x: tx.Optional[int] = None
    y: tx.Optional[int] = None


class _Aliased(DataModelBase):
    x: tx.Annotated[tx.Optional[int], Field(alias="ex")] = None


class _Child(_Plain):
    z: tx.Optional[int] = None


class _Sibling(_Plain):
    w: tx.Optional[int] = None


class _Kinded(DataModelBase):
    kind: tx.Optional[str] = None
    x: tx.Optional[int] = None


class _FixedKind(_Kinded):
    kind: Const[str] = "fixed"


class _Holder(DataModelBase):
    child: tx.Optional[_Child] = None
    fixed: tx.Optional[_FixedKind] = None


class _StrictHolder(DataModelBase):
    # Not optional: a union would report its own failure instead.
    fixed: _FixedKind = _FixedKind()


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


def test_from_instance_accepts_a_fixed_field_that_agrees_or_is_unset() -> None:
    assert _FixedKind.from_instance(_Kinded(kind="fixed", x=1)).x == 1
    assert _FixedKind.from_instance(_Kinded(x=1)).x == 1


def test_from_instance_refuses_a_fixed_field_that_disagrees() -> None:
    # The parent may carry any `kind`, the child always has "fixed": the
    # parent's value cannot be dropped silently.
    with pytest.raises(ValueError, match="always 'fixed'"):
        _FixedKind.from_instance(_Kinded(kind="other"))


def test_from_dict_refuses_a_fixed_field_that_disagrees() -> None:
    assert _FixedKind.from_dict({"kind": "fixed", "x": 1}).x == 1
    with pytest.raises(ValueError, match="always 'fixed'"):
        _FixedKind.from_dict({"kind": "other"})


def test_from_other_does_not_read_a_plain_object_as_a_parent() -> None:
    # `object` is a parent of every class, but an instance of it has
    # nothing to read: it goes to the constructor, which refuses it,
    # rather than yielding a default instance.
    with pytest.raises(ConversionError):
        _Plain.from_other(object())


def test_converter_returns_an_instance_of_the_target_unchanged() -> None:
    child = _Child(x=1)
    assert get_converter(_Child)(child) is child
    assert get_converter(_Plain)(child) is child


def test_converter_reads_a_mapping_through_from_other() -> None:
    obj = get_converter(_Child)({"x": 5, "z": 7})
    assert type(obj) is _Child
    assert (obj.x, obj.y, obj.z) == (5, None, 7)


def test_converter_reads_a_parent_instance_through_from_other() -> None:
    # Regression: the converter called the class itself, which took the
    # parent instance as the value of the first field.
    obj = get_converter(_Child)(_Plain(x=5, y=7))
    assert type(obj) is _Child
    assert (obj.x, obj.y, obj.z) == (5, 7, None)


def test_converter_refuses_a_sibling_instance() -> None:
    with pytest.raises(ConversionError):
        get_converter(_Child)(_Sibling(x=5))


def test_a_field_converts_a_parent_instance_and_a_mapping() -> None:
    holder = _Holder(child=_Plain(x=5), fixed={"x": 3})
    assert holder.child == _Child(x=5)
    assert holder.fixed == _FixedKind(x=3)


def test_a_field_keeps_the_reason_a_value_was_refused() -> None:
    match = "_StrictHolder.fixed: .*always 'fixed'"
    with pytest.raises(ConversionError, match=match):
        _StrictHolder(fixed=_Kinded(kind="other"))
