"""Unit tests for the brainhops.struct module."""
from typing import Annotated

import pytest

from brainhops._ext.struct import (
    ConvertTo,
    Default,
    Factory,
    Field,
    Frozen,
    InitVar,
    Key,
    KwOnly,
    NoEq,
    NoHash,
    NoInit,
    NoOrder,
    NoRepr,
    NotFrozen,
    NotKey,
    Struct,
    Validate,
    struct,
)
from brainhops._ext.struct.constants import _FIELDS, _OPTIONS, MISSING
from brainhops._ext.struct.converters import ConversionError, HintConverter
from brainhops._ext.struct.options import Options
from brainhops._ext.struct.validators import HintValidator, ValidationError

# ======================================================================
# Constants
# ======================================================================


class TestMissing:

    def test_singleton(self) -> None:
        assert MISSING is MISSING
        from brainhops._ext.struct.constants import _MissingType
        assert _MissingType() is MISSING

    def test_bool_is_false(self) -> None:
        assert not MISSING

    def test_repr(self) -> None:
        assert repr(MISSING) == "<MISSING>"


# ======================================================================
# Options
# ======================================================================


class TestOptions:

    def test_make_default(self) -> None:
        opts = Options.make_default()
        assert opts.init is True
        assert opts.repr is True
        assert opts.eq is True
        assert opts.order is False
        assert opts.unsafe_hash is False
        assert opts.frozen is False
        assert opts.match_args is False
        assert opts.kw_only is False
        assert opts.slots is False
        assert opts.weakref_slot is False
        assert opts.factory is False
        assert opts.convert is False
        assert opts.validate is False

    def test_update(self) -> None:
        opts = Options.make_default()
        override = Options(frozen=True, kw_only=True)
        opts.update(override)
        assert opts.frozen is True
        assert opts.kw_only is True
        # unchanged
        assert opts.init is True

    def test_setdefault(self) -> None:
        opts = Options(frozen=MISSING, kw_only=True)
        defaults = Options.make_default()
        opts.setdefault(defaults)
        assert opts.frozen is False  # was MISSING -> filled
        assert opts.kw_only is True  # was set -> kept

    def test_repr(self) -> None:
        opts = Options(frozen=True)
        r = repr(opts)
        assert "frozen=True" in r


# ======================================================================
# Basic Struct (inheritance API)
# ======================================================================


class TestBasicStruct:

    def test_simple_class(self) -> None:
        class Point(Struct):
            x: int
            y: int

        p = Point(1, 2)
        assert p.x == 1
        assert p.y == 2

    def test_repr(self) -> None:
        class Point(Struct):
            x: int
            y: int

        p = Point(1, 2)
        assert repr(p) == "Point(x=1, y=2)"

    def test_eq(self) -> None:
        class Point(Struct):
            x: int
            y: int

        assert Point(1, 2) == Point(1, 2)
        assert Point(1, 2) != Point(1, 3)

    def test_eq_different_class(self) -> None:
        class A(Struct):
            x: int

        class B(Struct):
            x: int

        assert A(1) != B(1)
        assert A(1).__eq__(B(1)) is NotImplemented

    def test_keyword_args(self) -> None:
        class Point(Struct):
            x: int
            y: int

        p = Point(x=10, y=20)
        assert p.x == 10
        assert p.y == 20

    def test_mixed_args(self) -> None:
        class Point(Struct):
            x: int
            y: int

        p = Point(10, y=20)
        assert p.x == 10
        assert p.y == 20

    def test_missing_required_arg(self) -> None:
        class Point(Struct):
            x: int
            y: int

        with pytest.raises(
                TypeError,
                match="missing 1 required positional argument"):
            Point(1)

    def test_too_many_positional_args(self) -> None:
        class Point(Struct):
            x: int

        with pytest.raises(
                TypeError,
                match="takes 2 positional arguments but 3 were given"):
            Point(1, 2)

    def test_unexpected_kwarg(self) -> None:
        class Point(Struct):
            x: int

        with pytest.raises(
                TypeError, match="got an unexpected keyword argument"):
            Point(x=1, z=2)


# ======================================================================
# Struct via decorator
# ======================================================================


class TestStructDecorator:

    def test_decorator_no_args(self) -> None:
        @struct
        class Point:
            x: int
            y: int

        p = Point(1, 2)
        assert p.x == 1
        assert p.y == 2

    def test_decorator_with_options(self) -> None:
        @struct(frozen=True, eq=True)
        class Point:
            x: int
            y: int

        p = Point(3, 4)
        assert p.x == 3
        with pytest.raises(AttributeError):
            p.x = 10


# ======================================================================
# Default values
# ======================================================================


class TestDefaults:

    def test_default_via_class_attribute(self) -> None:
        class Point(Struct):
            x: int
            y: int = 0

        p = Point(1)
        assert p.x == 1
        assert p.y == 0

    def test_default_annotation(self) -> None:
        class Point(Struct):
            x: int
            y: Default[int, 0]

        p = Point(1)
        assert p.y == 0

    def test_default_factory_annotation(self) -> None:
        class Container(Struct):
            items: Factory[list]

        c = Container()
        assert c.items == []

    def test_default_factory_custom(self) -> None:
        class Container(Struct):
            items: Factory[list, lambda: [1, 2]]

        c = Container()
        assert c.items == [1, 2]

    def test_default_factory_independent_instances(self) -> None:
        class Container(Struct):
            items: Factory[list]

        a = Container()
        b = Container()
        a.items.append(1)
        assert b.items == []


# ======================================================================
# Frozen
# ======================================================================


class TestFrozen:

    def test_frozen_class(self) -> None:
        class Point(Struct, frozen=True):
            x: int
            y: int

        p = Point(1, 2)
        with pytest.raises(AttributeError, match="Cannot set frozen field"):
            p.x = 10

    def test_frozen_delete(self) -> None:
        class Point(Struct, frozen=True):
            x: int
            y: int

        p = Point(1, 2)
        with pytest.raises(AttributeError, match="Cannot delete frozen field"):
            del p.x

    def test_frozen_field_annotation(self) -> None:
        class Point(Struct):
            x: Frozen[int]
            y: int

        p = Point(1, 2)
        with pytest.raises(AttributeError, match="Cannot set frozen field"):
            p.x = 10
        # y is not frozen
        p.y = 20
        assert p.y == 20

    def test_not_frozen_field_annotation(self) -> None:
        class Point(Struct, frozen=True):
            x: NotFrozen[int]
            y: int

        p = Point(1, 2)
        # y is frozen (class-level)
        with pytest.raises(AttributeError):
            p.y = 10
        # x is explicitly not frozen
        p.x = 10
        assert p.x == 10


# ======================================================================
# KwOnly
# ======================================================================


class TestKwOnly:

    def test_kw_only_class(self) -> None:
        class Point(Struct, kw_only=True):
            x: int
            y: int

        p = Point(x=1, y=2)
        assert p.x == 1
        with pytest.raises(
                TypeError,
                match="takes 1 positional argument but 3 were given"):
            Point(1, 2)

    def test_kw_only_field_annotation(self) -> None:
        class Point(Struct):
            x: int
            y: KwOnly[int]

        p = Point(1, y=2)
        assert p.y == 2


# ======================================================================
# Init / NoInit
# ======================================================================


class TestInit:

    def test_no_init_field(self) -> None:
        class Point(Struct):
            x: int
            y: NoInit[int] = 0

        p = Point(1)
        assert p.x == 1
        assert p.y == 0

    def test_no_init_field_rejects_positional(self) -> None:
        class Point(Struct):
            x: int
            y: NoInit[int] = 0

        with pytest.raises(
                TypeError,
                match="takes 2 positional arguments but 3 were given"):
            Point(1, 2)

    def test_no_init_field_rejects_keyword(self) -> None:
        class Point(Struct):
            x: int
            y: NoInit[int] = 0

        with pytest.raises(
                TypeError, match="got an unexpected keyword argument 'y'"):
            Point(x=1, y=2)

    def test_no_init_class(self) -> None:
        class Point(Struct, init=False):
            x: int = 0
            y: int = 0

        p = Point()
        assert p.x == 0
        assert p.y == 0


# ======================================================================
# Repr / NoRepr
# ======================================================================


class TestRepr:

    def test_no_repr_field(self) -> None:
        class Point(Struct):
            x: int
            y: NoRepr[int]

        p = Point(1, 2)
        assert repr(p) == "Point(x=1)"

    def test_no_repr_class(self) -> None:
        class Point(Struct, repr=False):
            x: int
            y: int

        p = Point(1, 2)
        assert "Point" not in repr(p) or "x=" not in repr(p)


# ======================================================================
# Eq / Order
# ======================================================================


class TestEqOrder:

    def test_no_eq_field(self) -> None:
        class Point(Struct):
            x: int
            y: NoEq[int]

        # y is excluded from eq
        assert Point(1, 2) == Point(1, 99)

    def test_order(self) -> None:
        class Point(Struct, order=True):
            x: int
            y: int

        assert Point(1, 2) < Point(1, 3)
        assert not Point(1, 3) < Point(1, 2)

    def test_order_different_class(self) -> None:
        class A(Struct, order=True):
            x: int

        class B(Struct, order=True):
            x: int

        assert A(1).__lt__(B(2)) is NotImplemented

    def test_no_order_field(self) -> None:
        class Point(Struct, order=True):
            x: int
            y: NoOrder[int]

        # y excluded from ordering
        assert not Point(1, 2) < Point(1, 1)
        assert not Point(1, 1) < Point(1, 2)

    def test_order_requires_eq(self) -> None:
        with pytest.raises(ValueError, match="eq must be true"):
            class Bad(Struct):
                x: Annotated[int, Field(order=True, eq=False)]


# ======================================================================
# Hash
# ======================================================================


class TestHash:

    def test_frozen_eq_hashing(self) -> None:
        class Point(Struct, frozen=True, eq=True):
            x: int
            y: int

        p = Point(1, 2)
        assert hash(p) == hash(Point(1, 2))
        assert hash(p) != hash(Point(1, 3))

    def test_unsafe_hash(self) -> None:
        class Point(Struct, unsafe_hash=True):
            x: int
            y: int

        p = Point(1, 2)
        assert hash(p) == hash(Point(1, 2))

    def test_no_hash_field(self) -> None:
        class Point(Struct, frozen=True, eq=True):
            x: int
            y: NoHash[int]

        # NoHash removes field from hash but not from eq
        # hash should be same regardless of y
        # (NoHash sets hash=False; the hash_add function checks f.hash)
        assert hash(Point(1, 2)) == hash(Point(1, 99))

    def test_frozen_in_set(self) -> None:
        class Point(Struct, frozen=True, eq=True):
            x: int
            y: int

        s = {Point(1, 2), Point(1, 2), Point(3, 4)}
        assert len(s) == 2


# ======================================================================
# Slots
# ======================================================================


class TestSlots:

    def test_slots(self) -> None:
        class Point(Struct, slots=True):
            x: int
            y: int

        p = Point(1, 2)
        assert p.x == 1
        assert not hasattr(p, "__dict__")

    def test_slots_no_arbitrary_attrs(self) -> None:
        class Point(Struct, slots=True):
            x: int
            y: int

        p = Point(1, 2)
        with pytest.raises(AttributeError):
            p.z = 3

    def test_weakref_slot_requires_slots(self) -> None:
        with pytest.raises(
                TypeError, match="weakref_slot is True but slots is False"):
            class Bad(Struct, weakref_slot=True, slots=False):
                x: int

    def test_weakref_slot(self) -> None:
        import weakref

        class Point(Struct, slots=True, weakref_slot=True):
            x: int

        p = Point(1)
        ref = weakref.ref(p)
        assert ref() is p

    def test_slots_already_defined_error(self) -> None:
        with pytest.raises(TypeError, match="already specifies __slots__"):
            class Bad(Struct, slots=True):
                __slots__ = ('x',)
                x: int


# ======================================================================
# Inheritance
# ======================================================================


class TestInheritance:

    def test_inherit_fields(self) -> None:
        class Base(Struct):
            x: int

        class Derived(Base):
            y: int

        d = Derived(1, 2)
        assert d.x == 1
        assert d.y == 2

    def test_inherit_options(self) -> None:
        class Base(Struct, frozen=True):
            x: int

        class Derived(Base):
            y: int

        d = Derived(1, 2)
        with pytest.raises(AttributeError):
            d.x = 10
        with pytest.raises(AttributeError):
            d.y = 10

    def test_override_field(self) -> None:
        class Base(Struct):
            x: int
            y: int

        class Derived(Base):
            y: str

        d = Derived(1, "hello")
        assert d.y == "hello"

    def test_fields_stored(self) -> None:
        class Point(Struct):
            x: int
            y: int

        fields = getattr(Point, _FIELDS)
        assert "x" in fields
        assert "y" in fields

    def test_options_stored(self) -> None:
        class Point(Struct, frozen=True):
            x: int

        opts = getattr(Point, _OPTIONS)
        assert opts.frozen is True


# ======================================================================
# ConvertTo
# ======================================================================


class TestConvertTo:

    def test_convert_annotation(self) -> None:
        class Point(Struct):
            x: ConvertTo[int]

        p = Point("42")
        assert p.x == 42
        assert isinstance(p.x, int)

    def test_convert_custom_function(self) -> None:
        class Upper(Struct):
            name: ConvertTo[str, str.upper]

        u = Upper("hello")
        assert u.name == "HELLO"

    def test_convert_class_option(self) -> None:
        class Point(Struct, convert=True):
            x: int
            y: float

        p = Point("1", "2.5")
        assert p.x == 1
        assert p.y == 2.5


# ======================================================================
# Validate
# ======================================================================


class TestValidate:

    def test_validate_annotation(self) -> None:
        class Point(Struct):
            x: Validate[int]

        p = Point(42)
        assert p.x == 42

    def test_validate_annotation_fail(self) -> None:
        class Point(Struct):
            x: Validate[int]

        with pytest.raises(ValidationError):
            Point("not int")

    def test_validate_class_option(self) -> None:
        class Point(Struct, validate=True):
            x: int
            y: float

        Point(1, 2.5)
        with pytest.raises(ValidationError):
            Point("a", 2.5)


# ======================================================================
# Var / InitVar / ClassVar
# ======================================================================


class TestVarFields:

    def test_init_var(self) -> None:

        class WithInitVar(Struct):
            x: int
            scale: InitVar[int]

            def __post_init__(self, scale: float) -> None:
                self.x = self.x * scale

        w = WithInitVar(5, 10)
        assert w.x == 50
        assert not hasattr(w, "scale") or getattr(w, "scale", None) is None


# ======================================================================
# match_args
# ======================================================================


class TestMatchArgs:

    def test_match_args(self) -> None:
        class Point(Struct, match_args=True):
            x: int
            y: int

        assert Point.__match_args__ == ("x", "y")

    def test_match_args_excludes_kw_only(self) -> None:
        class Point(Struct, match_args=True):
            x: int
            y: KwOnly[int]

        assert Point.__match_args__ == ("x",)


# ======================================================================
# Field (direct)
# ======================================================================


class TestField:

    def test_field_from_hint_simple(self) -> None:
        f = Field.from_hint("x", int)
        assert f.name == "x"
        assert f.type is int

    def test_field_from_hint_with_default(self) -> None:
        f = Field.from_hint("x", int, 42)
        assert f.default == 42

    def test_field_from_hint_annotated(self) -> None:
        f = Field.from_hint("x", Annotated[int, Field(frozen=True)])
        assert f.frozen is True

    def test_field_repr(self) -> None:
        f = Field(init=True, repr=False)
        r = repr(f)
        assert "init=True" in r
        assert "repr=False" in r

    def test_field_compare_alias(self) -> None:
        f = Field(compare=True)
        assert f.eq is True
        assert f.order is True

    def test_field_no_annotation_error(self) -> None:
        with pytest.raises(
                TypeError, match="is a field but has no type annotation"):
            class Bad(Struct):
                x = Field()


# ======================================================================
# HintConverter
# ======================================================================


class TestHintConverter:

    def test_int_converter(self) -> None:
        c = HintConverter(int)
        assert c("42") == 42

    def test_float_converter(self) -> None:
        c = HintConverter(float)
        assert c("3.14") == pytest.approx(3.14)

    def test_str_converter(self) -> None:
        c = HintConverter(str)
        assert c(42) == "42"

    def test_bool_converter(self) -> None:
        c = HintConverter(bool)
        assert c(1) is True

    def test_none_converter(self) -> None:
        c = HintConverter(None)
        assert c(None) is None
        with pytest.raises(ConversionError):
            c(42)

    def test_list_converter(self) -> None:
        c = HintConverter(list)
        assert c((1, 2, 3)) == [1, 2, 3]

    def test_list_typed_converter(self) -> None:
        c = HintConverter(list[int])
        assert c(["1", "2"]) == [1, 2]

    def test_tuple_converter(self) -> None:
        c = HintConverter(tuple)
        assert c([1, 2, 3]) == (1, 2, 3)

    def test_tuple_typed_converter(self) -> None:
        c = HintConverter(tuple[int, str])
        assert c([1, 2]) == (1, "2")

    def test_set_converter(self) -> None:
        c = HintConverter(set)
        assert c([1, 2, 2]) == {1, 2}

    def test_set_typed_converter(self) -> None:
        c = HintConverter(set[int])
        assert c(["1", "2"]) == {1, 2}

    def test_frozenset_converter(self) -> None:
        c = HintConverter(frozenset)
        assert c([1, 2, 2]) == frozenset({1, 2})

    def test_dict_converter(self) -> None:
        c = HintConverter(dict)
        assert c([("a", 1)]) == {"a": 1}

    def test_dict_typed_converter(self) -> None:
        c = HintConverter(dict[str, int])
        result = c({"a": "1", "b": "2"})
        assert result == {"a": 1, "b": 2}

    def test_union_converter(self) -> None:
        c = HintConverter(int | str)
        assert c("42") == 42  # tries int first

    def test_optional_converter(self) -> None:
        from typing import Optional
        c = HintConverter(Optional[int])
        assert c(None) is None
        assert c("42") == 42

    def test_any_converter(self) -> None:
        from typing import Any
        c = HintConverter(Any)
        assert c("hello") == "hello"
        assert c(42) == 42

    def test_type_converter(self) -> None:
        c = HintConverter(type[int])
        assert c(int) is int
        assert c(bool) is bool  # bool is subclass of int
        with pytest.raises(ConversionError):
            c(str)

    def test_conversion_error(self) -> None:
        c = HintConverter(int)
        with pytest.raises(ConversionError):
            c("not a number")


# ======================================================================
# HintValidator
# ======================================================================


class TestHintValidator:

    def test_int_validator(self) -> None:
        v = HintValidator(int)
        assert v(42) == 42

    def test_int_validator_fail(self) -> None:
        v = HintValidator(int)
        with pytest.raises(ValidationError):
            v("42")

    def test_float_validator(self) -> None:
        v = HintValidator(float)
        assert v(3.14) == pytest.approx(3.14)
        with pytest.raises(ValidationError):
            v("3.14")

    def test_str_validator(self) -> None:
        v = HintValidator(str)
        assert v("hello") == "hello"
        with pytest.raises(ValidationError):
            v(42)

    def test_none_validator(self) -> None:
        v = HintValidator(None)
        assert v(None) is None
        with pytest.raises(ValidationError):
            v(0)

    def test_list_validator(self) -> None:
        v = HintValidator(list)
        assert v([1, 2]) == [1, 2]
        with pytest.raises(ValidationError):
            v((1, 2))

    def test_list_typed_validator(self) -> None:
        v = HintValidator(list[int])
        assert v([1, 2]) == [1, 2]
        with pytest.raises(ValidationError):
            v(["a", "b"])

    def test_tuple_validator(self) -> None:
        v = HintValidator(tuple)
        assert v((1, 2)) == (1, 2)
        with pytest.raises(ValidationError):
            v([1, 2])

    def test_tuple_typed_validator(self) -> None:
        v = HintValidator(tuple[int, str])
        assert v((1, "a")) == (1, "a")
        with pytest.raises(ValidationError):
            v((1, 2))  # second element not str

    def test_set_validator(self) -> None:
        v = HintValidator(set)
        assert v({1, 2}) == {1, 2}
        with pytest.raises(ValidationError):
            v([1, 2])

    def test_frozenset_validator(self) -> None:
        v = HintValidator(frozenset)
        assert v(frozenset({1, 2})) == frozenset({1, 2})
        with pytest.raises(ValidationError):
            v({1, 2})

    def test_dict_validator(self) -> None:
        v = HintValidator(dict)
        assert v({"a": 1}) == {"a": 1}
        with pytest.raises(ValidationError):
            v([("a", 1)])

    def test_dict_typed_validator(self) -> None:
        v = HintValidator(dict[str, int])
        assert v({"a": 1}) == {"a": 1}
        with pytest.raises(ValidationError):
            v({1: "a"})  # key not str

    def test_union_validator(self) -> None:
        v = HintValidator(int | str)
        assert v(42) == 42
        assert v("hello") == "hello"

    def test_optional_validator(self) -> None:
        from typing import Optional
        v = HintValidator(Optional[int])
        assert v(None) is None
        assert v(42) == 42
        with pytest.raises(ValidationError):
            v("hello")

    def test_any_validator(self) -> None:
        from typing import Any
        v = HintValidator(Any)
        assert v("anything") == "anything"
        assert v(42) == 42

    def test_type_validator(self) -> None:
        v = HintValidator(type[int])
        assert v(int) is int
        with pytest.raises(ValidationError):
            v(str)
        with pytest.raises(ValidationError):
            v(42)


# ======================================================================
# Mapping
# ======================================================================


class TestMapping:

    def test_mapping_getitem(self) -> None:
        class Point(Struct, mapping=True):
            x: int
            y: int

        p = Point(1, 2)
        assert p["x"] == 1
        assert p["y"] == 2

    def test_mapping_getitem_keyerror(self) -> None:
        class Point(Struct, mapping=True):
            x: int

        p = Point(1)
        with pytest.raises(KeyError):
            p["z"]

    def test_mapping_setitem(self) -> None:
        class Point(Struct, mapping=True):
            x: int
            y: int

        p = Point(1, 2)
        p["x"] = 10
        assert p.x == 10
        assert p["x"] == 10

    def test_mapping_setitem_keyerror(self) -> None:
        class Point(Struct, mapping=True):
            x: int

        p = Point(1)
        with pytest.raises(KeyError):
            p["z"] = 99

    def test_mapping_delitem(self) -> None:
        class Point(Struct, mapping=True):
            x: int
            y: int

        p = Point(1, 2)
        del p["x"]
        assert not hasattr(p, "x")

    def test_mapping_delitem_keyerror(self) -> None:
        class Point(Struct, mapping=True):
            x: int

        p = Point(1)
        with pytest.raises(KeyError):
            del p["z"]

    def test_mapping_iter(self) -> None:
        class Point(Struct, mapping=True):
            x: int
            y: int

        p = Point(1, 2)
        assert list(p) == ["x", "y"]

    def test_mapping_len(self) -> None:
        class Point(Struct, mapping=True):
            x: int
            y: int

        p = Point(1, 2)
        assert len(p) == 2

    def test_mapping_is_mutable_mapping(self) -> None:
        from collections.abc import MutableMapping

        class Point(Struct, mapping=True):
            x: int

        p = Point(1)
        assert isinstance(p, MutableMapping)

    def test_frozen_mapping_is_immutable_mapping(self) -> None:
        from collections.abc import Mapping, MutableMapping

        class Point(Struct, mapping=True, frozen=True):
            x: int
            y: int

        p = Point(1, 2)
        assert isinstance(p, Mapping)
        assert not isinstance(p, MutableMapping)

    def test_mapping_dict_conversion(self) -> None:
        class Point(Struct, mapping=True):
            x: int
            y: int

        p = Point(1, 2)
        assert dict(p) == {"x": 1, "y": 2}

    def test_mapping_not_key_field(self) -> None:
        class Point(Struct, mapping=True):
            x: int
            y: NotKey[int]

        p = Point(1, 2)
        assert list(p) == ["x"]
        assert p["x"] == 1
        with pytest.raises(KeyError):
            p["y"]

    def test_mapping_key_field_override(self) -> None:
        class Point(Struct):
            x: Key[int]
            y: int

        # mapping=False by default, but Key annotation sets field.key=True
        # The mapping interface is only generated if the class option is set,
        # so Key on its own doesn't add mapping methods.
        # Let's test with mapping=True and Key/NotKey mix.
        class Point2(Struct, mapping=True):
            x: Key[int]
            y: NotKey[int]

        p = Point2(1, 2)
        assert dict(p) == {"x": 1}

    def test_mapping_inherited(self) -> None:
        class Base(Struct, mapping=True):
            x: int

        class Derived(Base):
            y: int

        d = Derived(1, 2)
        assert dict(d) == {"x": 1, "y": 2}

    def test_mapping_default_off(self) -> None:
        class Point(Struct):
            x: int
            y: int

        p = Point(1, 2)
        # No mapping interface by default
        assert not hasattr(p, "__getitem__")


# ======================================================================
# Integration: combined features
# ======================================================================


class TestIntegration:

    def test_frozen_eq_hashable_as_dict_key(self) -> None:
        class Point(Struct, frozen=True, eq=True):
            x: int
            y: int

        d = {Point(1, 2): "a", Point(3, 4): "b"}
        assert d[Point(1, 2)] == "a"

    def test_convert_and_validate(self) -> None:
        class Config(Struct):
            x: Annotated[int, ConvertTo(), Validate()]

        c = Config("42")
        assert c.x == 42

    def test_inheritance_with_defaults(self) -> None:
        class Base(Struct):
            x: int

        class Derived(Base):
            x: int = 10
            y: int = 20

        d = Derived()
        assert d.x == 10
        assert d.y == 20

    def test_default_factory_class_option(self) -> None:
        class Container(Struct, factory=True):
            items: list

        c = Container()
        assert c.items == []

    def test_deeply_nested_struct(self) -> None:
        class A(Struct):
            a: int

        class B(A):
            b: int

        class C(B):
            c: int

        obj = C(1, 2, 3)
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == 3

    def test_eq_identity_shortcircuit(self) -> None:
        class Point(Struct):
            x: int

        p = Point(1)
        assert p == p  # same object -> True immediately
