"""Tests for `smartproperty` and `lazyproperty`: when the getter computes
the value (`unset`), what the setter stores, and the cache."""

import pytest
import typing_extensions as tx

from brainhops._core.properties import lazyproperty, smartproperty, smartsetter


def _box(unset: tx.Any = None, cache: bool = False) -> type:
    # A plain class with one property, `value`, stored under `_value`, whose
    # computed value counts how often it was computed.
    class Box:
        computed = 0

        def __init__(self, value: tx.Any = None) -> None:
            # A constructor that writes its argument, the default included,
            # through the setter -- as a data model does.
            self.value = value

        @smartproperty(unset=unset, cache=cache)
        def value(self) -> str:
            type(self).computed += 1
            return "computed"

    return Box


def _odd(value: tx.Any) -> bool:
    return isinstance(value, int) and value % 2 == 1


@pytest.mark.parametrize(
    "unset, stored, computed",
    [
        # `None` (the default): a stored `None` only.
        (None, None, True),
        (None, [], False),
        (None, 0, False),
        # "empty": an empty container only -- not `None`.
        ("empty", [], True),
        ("empty", (), True),
        ("empty", {}, True),
        ("empty", set(), True),
        ("empty", frozenset(), True),
        ("empty", [0], False),
        ("empty", "", False),
        ("empty", None, False),
        # A predicate.
        (_odd, 3, True),
        (_odd, 2, False),
        (_odd, None, False),
        # A tuple: any of them.
        ((None, "empty"), None, True),
        ((None, "empty"), [], True),
        ((None, "empty"), [1], False),
        ((None, _odd), 1, True),
        ((None, _odd), None, True),
        (("empty", _odd), 2, False),
    ],
)
def test_unset_says_when_the_getter_runs(
    unset: tx.Any, stored: tx.Any, computed: bool
) -> None:
    box = _box(unset)(stored)
    assert (box.value == "computed") is computed
    if not computed:
        assert box.value is stored


def test_the_default_is_a_stored_none() -> None:
    assert _box()(None).value == "computed"
    assert _box()([]).value == []


@pytest.mark.parametrize("unset", [None, "empty", (None, "empty"), _odd])
def test_the_setter_stores_the_value_as_given(unset: tx.Any) -> None:
    for value in (None, [], 3, [1]):
        box = _box(unset)(value)
        assert box._value is value


def test_an_unset_value_written_by_the_constructor_is_computed() -> None:
    # The case `unset=(None, "empty")` is for: a constructor that writes an
    # empty default through the setter does not shadow the getter.
    box = _box((None, "empty"))([])
    assert box.value == "computed"
    box.value = ["given"]
    assert box.value == ["given"]
    box.value = []
    assert box.value == "computed"


def test_the_cache_and_its_invalidation() -> None:
    cls = _box((None, "empty"), cache=True)
    box = cls([])
    assert box.value == "computed" and cls.computed == 1
    assert box.value == "computed" and cls.computed == 1  # cached
    # A set value is served, and setting clears the cache.
    box.value = ["given"]
    assert box.value == ["given"] and not hasattr(box, "_cache_value")
    box.value = []
    assert box.value == "computed" and cls.computed == 2
    # A predicate caches the same way.
    cls = _box(_odd, cache=True)
    box = cls(1)
    assert box.value == "computed" and box.value == "computed"
    assert cls.computed == 1


def test_lazyproperty_takes_unset() -> None:
    class Lazy:
        computed = 0

        @lazyproperty(unset=(None, "empty"))
        def value(self) -> str:
            type(self).computed += 1
            return "computed"

    lazy = Lazy()
    assert lazy.value == "computed" and lazy.value == "computed"
    assert Lazy.computed == 1


@pytest.mark.parametrize(
    "unset, error, match",
    [
        ("blank", ValueError, "'empty'"),
        ((None, "nothing"), ValueError, "'empty'"),
        ((), ValueError, "no form"),
        (0, TypeError, "not a int"),
        ([None, "empty"], TypeError, "not a list"),
        ((None, 1.0), TypeError, "not a float"),
    ],
)
def test_a_bad_unset_is_refused_when_declared(
    unset: tx.Any, error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        smartproperty(unset=unset)
    with pytest.raises(error, match=match):
        lazyproperty(unset=unset)


def test_the_former_options_are_gone() -> None:
    for option in ("empty_as_unset", "informative", "missing"):
        with pytest.raises(TypeError):
            smartproperty(**{option: True})


# --- smartsetter ------------------------------------------------------


def _setter_box() -> type:
    # Two properties: `value`, which reads `_value`, the name of its
    # setter, and `other`, which reads the `_stored` it is given. Each
    # setter stores the value; `value`'s counts its calls.
    class Box:
        calls = 0

        @smartsetter
        def value(self, value: tx.Any) -> None:
            type(self).calls += 1
            self._value = value

        @smartsetter("stored")
        def other(self, value: tx.Any) -> None:
            self._stored = value * 2

    return Box


def test_smartsetter_takes_its_name_from_the_function() -> None:
    box = _setter_box()()
    assert isinstance(type(box).value, property)
    assert box.value is None
    box.value = 3
    assert box.value == 3 and box._value == 3
    assert type(box).calls == 1


def test_smartsetter_takes_a_name() -> None:
    box = _setter_box()()
    assert isinstance(type(box).other, property)
    assert box.other is None
    box.other = 3
    assert box.other == 6 and box._stored == 6


def test_smartsetter_reads_the_private_attribute() -> None:
    box = _setter_box()()
    box._value = "stored"
    assert box.value == "stored"
    assert type(box).calls == 0


def test_smartsetter_on_a_magic_class_with_a_private_field() -> None:
    # As the transformations use it: the field is stored privately, the
    # constructor takes its public name, and an assignment runs the setter.
    from bagof.magic import Magic

    class Model(Magic):
        _data: tx.Optional[int] = None
        assigned: tx.ClassVar[int] = 0

        @smartsetter
        def data(self, value: tx.Optional[int]) -> None:
            self._data = value
            type(self).assigned += 1

    model = Model(data=4)
    assert model.data == 4 and model._data == 4
    before = Model.assigned
    model.data = 5
    assert model.data == 5 and Model.assigned == before + 1
    assert Model().data is None
