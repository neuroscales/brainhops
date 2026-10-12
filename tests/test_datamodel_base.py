"""Tests of the `from_*` constructors of data models and of their converter."""

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


class _Refixed(_FixedKind):
    # A subclass may fix the field its own way and remain a `_FixedKind`.
    kind: Const[str] = "refixed"


class _Holder(DataModelBase):
    child: tx.Optional[_Child] = None
    fixed: tx.Optional[_FixedKind] = None


class _StrictHolder(DataModelBase):
    # Not optional, since a union would report its own failure instead.
    fixed: _FixedKind = _FixedKind()


def test_from_dict_carries_plain_fields() -> None:
    # Regression (#61): values were keyed on the unset alias and dropped.
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
    # Regression (#61): values were keyed on the unset alias, which raised.
    obj = _Plain.from_instance(_Plain(x=5, y=7))
    assert obj.x == 5
    assert obj.y == 7


def test_from_instance_honors_an_explicit_alias() -> None:
    obj = _Aliased.from_instance(_Aliased(ex=9))
    assert obj.x == 9


def test_from_other_dispatches_to_dict_and_instance() -> None:
    from_dict = _Plain.from_any({"x": 5, "y": 7})
    assert (from_dict.x, from_dict.y) == (5, 7)
    from_instance = _Plain.from_any(_Plain(x=5, y=7))
    assert (from_instance.x, from_instance.y) == (5, 7)


def test_from_instance_accepts_a_fixed_field_that_agrees_or_is_unset() -> None:
    assert _FixedKind.from_instance(_Kinded(kind="fixed", x=1)).x == 1
    assert _FixedKind.from_instance(_Kinded(x=1)).x == 1


def test_from_instance_refuses_a_fixed_field_that_disagrees() -> None:
    # The value of the parent cannot be dropped silently, so a disagreement is
    # refused.
    with pytest.raises(ValueError, match="always 'fixed'"):
        _FixedKind.from_instance(_Kinded(kind="other"))


def test_from_dict_refuses_a_fixed_field_that_disagrees() -> None:
    assert _FixedKind.from_dict({"kind": "fixed", "x": 1}).x == 1
    with pytest.raises(ValueError, match="always 'fixed'"):
        _FixedKind.from_dict({"kind": "other"})


def test_from_other_does_not_read_a_plain_object_as_a_parent() -> None:
    # Every class derives from `object`, but an `object` instance has nothing
    # to read, so the constructor refuses it.
    with pytest.raises(ConversionError):
        _Plain.from_any(object())


def test_converter_returns_an_instance_of_the_target_unchanged() -> None:
    child = _Child(x=1)
    assert get_converter(_Child)(child) is child
    assert get_converter(_Plain)(child) is child


def test_converter_reads_a_mapping_through_from_other() -> None:
    obj = get_converter(_Child)({"x": 5, "z": 7})
    assert type(obj) is _Child
    assert (obj.x, obj.y, obj.z) == (5, None, 7)


def test_converter_reads_a_parent_instance_through_from_other() -> None:
    # Regression: the parent instance was taken as the value of the first
    # field.
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


def test_from_instance_leaves_defaults_in_place_of_unset_attributes() -> None:
    class _Defaulted(_Plain):
        y: tx.Optional[int] = 3

    # `y` is unset on the parent, so the default of the child stays.
    assert _Defaulted.from_instance(_Plain(x=1)).y == 3


def test_from_dict_keeps_an_explicit_none() -> None:
    class _Defaulted(_Plain):
        y: tx.Optional[int] = 3

    assert _Defaulted.from_dict({"y": None}).y is None


def test_from_instance_skips_the_fixed_check_for_an_instance() -> None:
    obj = _FixedKind.from_instance(_Refixed(x=1))
    assert type(obj) is _FixedKind
    assert (obj.kind, obj.x) == ("fixed", 1)


def test_a_fixed_field_is_checked_before_the_instance_is_built() -> None:
    # `x` cannot be converted either, but the fixed field is checked first.
    with pytest.raises(ValueError, match="always 'fixed'"):
        _FixedKind.from_dict({"kind": "other", "x": "not a number"})


def test_a_fixed_value_that_cannot_be_converted_says_so() -> None:
    with pytest.raises(ValueError, match="could not be converted") as info:
        _FixedKind.from_dict({"kind": 5})
    assert isinstance(info.value.__cause__, ConversionError)


def test_from_dict_ignores_a_key_that_matches_no_field() -> None:
    assert _Plain.from_dict({"x": 1, "naem": 2}).x == 1


def test_from_other_refuses_a_key_that_matches_no_field() -> None:
    with pytest.raises(TypeError, match="no field named 'naem'"):
        _Plain.from_any({"x": 1, "naem": 2})


def test_converter_refuses_a_key_that_matches_no_field() -> None:
    with pytest.raises(ConversionError, match="no field named 'naem'"):
        get_converter(_Plain)({"x": 1, "naem": 2})
