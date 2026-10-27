"""Tests of `smartproperty`, `lazyproperty` and `smartsetter`."""

import pytest
import typing_extensions as tx
from bagof.magic import Magic

from brainhops._core.properties import (
    InvalidatorInAttribute,
    alias,
    cached_value,
    drop_cache,
    lazyproperty,
    smartproperty,
    smartsetter,
)


def _box(cache: bool = False) -> type:
    # A class with one property `value`, stored as `_value`, that counts how
    # often its value is computed.
    class Box:
        computed = 0

        def __init__(self, value: tx.Any = None) -> None:
            # The constructor writes its argument, including the default,
            # through the setter, as a data model does.
            self.value = value

        @smartproperty(cache=cache)
        def value(self) -> str:
            type(self).computed += 1
            return "computed"

    return Box


def test_the_default_is_a_stored_none() -> None:
    assert _box()(None).value == "computed"
    assert _box()([]).value == []


def test_an_empty_value_is_used_as_it_is() -> None:
    # Only a stored `None` is computed, so an empty container is a value.
    box = _box()(None)
    assert box.value == "computed"
    box.value = []
    assert box.value == []
    box.value = None
    assert box.value == "computed"


def test_the_cache_and_its_invalidation() -> None:
    cls = _box(cache=True)
    box = cls(None)
    assert box.value == "computed" and cls.computed == 1
    assert box.value == "computed" and cls.computed == 1
    # A value that is set is served as is, and setting clears the cache.
    box.value = ["given"]
    assert box.value == ["given"] and not hasattr(box, "_cache_value")
    box.value = None
    assert box.value == "computed" and cls.computed == 2


def test_a_frozen_object_can_cache_a_read_only_property() -> None:
    class Frozen(Magic, frozen=True):
        @smartproperty(cache=True, fset=False)
        def value(self) -> object:
            return object()

    obj = Frozen()
    assert obj.value is obj.value
    with pytest.raises(AttributeError):
        obj.value = object()


def test_lazyproperty_computes_once_and_reads_a_stored_value() -> None:
    class Lazy:
        computed = 0

        @lazyproperty
        def value(self) -> str:
            type(self).computed += 1
            return "computed"

    lazy = Lazy()
    assert lazy.value == "computed" and lazy.value == "computed"
    assert Lazy.computed == 1
    with pytest.raises(AttributeError):
        lazy.value = "given"
    # A value that a constructor stores in the private attribute is read.
    lazy = Lazy()
    lazy._value = "stored"
    assert lazy.value == "stored" and Lazy.computed == 1


def test_the_former_options_are_gone() -> None:
    for option in ("empty_as_unset", "informative", "missing", "unset"):
        with pytest.raises(TypeError):
            smartproperty(**{option: True})
    with pytest.raises(TypeError):
        smartproperty(convert=str)
    from brainhops._core import properties

    assert not hasattr(properties, "always_unset")
    assert not hasattr(properties, "UnsetLike")


def _views_box() -> type:
    # `value` is computed from `_source`, and its setter is given with the
    # `setter` decorator. `twice` is a cached view that depends on `value`.
    class Box:
        def __init__(self, source: int) -> None:
            self._source = source

        @smartproperty(cache=True, invalidates="twice")
        def value(self) -> int:
            return self._source

        @value.setter
        def value(self, value: int) -> None:
            self._source = value

        @smartproperty(cache=True, fset=False)
        def twice(self) -> int:
            return 2 * self.value

    return Box


def test_a_setter_given_with_the_decorator_keeps_the_invalidation() -> None:
    box = _views_box()(1)
    assert box.value == 1 and box.twice == 2
    assert box._cache_value == 1 and box._cache_twice == 2
    box.value = 3
    assert "_cache_value" not in box.__dict__
    assert "_cache_twice" not in box.__dict__
    assert box.value == 3 and box.twice == 6


def test_the_getter_and_deleter_decorators_keep_the_invalidation() -> None:
    class Box:
        def __init__(self) -> None:
            self.deleted = False

        @smartproperty(cache=True, invalidates="other")
        def value(self) -> int:
            return 1

        @value.deleter
        def value(self) -> None:
            self.deleted = True

        @value.setter
        def value(self, value: int) -> None:
            self._value = value

    box = Box()
    assert box.value == 1 and box._cache_value == 1
    box._cache_other = "stale"
    box.value = 2
    assert "_cache_value" not in box.__dict__ and box.value == 2
    assert "_cache_other" not in box.__dict__
    del box.value
    assert box.deleted
    # A getter replaced after the declaration keeps the invalidation too.
    replaced = Box.value.getter(lambda self: 5)
    assert replaced.invalidates == ("other",) and replaced.cache


def _cached_box(how: str) -> type:
    # `value` is cached, and its setter, given with `fset=` or with the
    # `setter` decorator, stores the value in `_source`, which the getter
    # reads.
    def store(self: tx.Any, value: int) -> None:
        self._source = value

    class Box:
        def __init__(self, source: int) -> None:
            self._source = source

        if how == "fset":

            @smartproperty(cache=True, fset=store)
            def value(self) -> int:
                return self._source

        elif how == "setter":

            @smartproperty(cache=True)
            def value(self) -> int:
                return self._source

            value = value.setter(store)

        else:

            @lazyproperty
            def value(self) -> int:
                return self._source

            value = value.setter(store)

    return Box


@pytest.mark.parametrize("how", ["fset", "setter", "lazy"])
def test_every_setter_drops_the_cache_of_its_property(how: str) -> None:
    box = _cached_box(how)(1)
    assert box.value == 1 and box._cache_value == 1
    box.value = 3
    assert box.value == 3


def test_a_setter_may_fill_its_own_cache() -> None:
    # The cache is dropped before the setter runs, so a setter that caches
    # the value it was given keeps it, as the `values` setter of a field
    # does.
    class Box:
        def __init__(self) -> None:
            self._source = 1

        @lazyproperty(invalidates="other")
        def value(self) -> int:
            return self._source

        @value.setter
        def value(self, value: int) -> None:
            self._source = None
            self._cache_value = value

    box = Box()
    assert box.value == 1
    box._cache_other = "stale"
    box.value = 3
    assert box._cache_value == 3 and box.value == 3
    assert "_cache_other" not in box.__dict__


def test_invalidates_mixes_names_and_invalidators() -> None:
    class Box:
        views = ("twice",)

        def __init__(self) -> None:
            self._cache_twice = "stale"
            self._cache_other = "stale"
            self._cache_kept = "kept"

        value = smartproperty(
            "value", invalidates=("other", InvalidatorInAttribute("views"))
        )

    box = Box()
    box.value = 1
    assert box.value == 1
    assert "_cache_twice" not in box.__dict__
    assert "_cache_other" not in box.__dict__
    assert box._cache_kept == "kept"
    assert Box.value.invalidates[0] == "other"


def test_invalidates_refuses_other_items() -> None:
    with pytest.raises(TypeError, match="not a int"):
        smartproperty("value", invalidates=("other", 1))


def test_drop_cache_deletes_the_cached_values_it_names() -> None:
    box = _cached_box("setter")(1)
    assert box.value == 1
    box._source = 2
    assert box.value == 1
    drop_cache(box, "value", "never_cached")
    assert box.value == 2


def test_cached_value_reads_a_cache_without_computing_it() -> None:
    box = _cached_box("setter")(1)
    assert cached_value(box, "value") is None
    assert box.value == 1
    assert cached_value(box, "value") == 1
    drop_cache(box, "value")
    assert cached_value(box, "value") is None


def _sourced_box() -> type:
    # A class whose cached `view` is computed from `source`, which it names
    # in `invalidated_by`, and which counts how often it is computed.
    class Box:
        computed = 0

        def __init__(self, source: tx.Any) -> None:
            self.source = source
            self._view = None

        @smartproperty(cache=True, invalidated_by=("source",))
        def view(self) -> tx.Any:
            type(self).computed += 1
            return ("view of", self.source)

    return Box


def test_invalidated_by_computes_again_for_another_source() -> None:
    cls = _sourced_box()
    first, second = [1], [1]
    box = cls(first)
    view = box.view
    assert box.view is view and cls.computed == 1
    # The same object, even changed in place, keeps the cached value.
    first.append(2)
    assert box.view is view and cls.computed == 1
    # Another object, even an equal one, computes the value again.
    box.source = second
    assert box.view is not view and cls.computed == 2
    assert box.view[1] is second and cls.computed == 2


def test_cached_value_is_none_for_another_source() -> None:
    cls = _sourced_box()
    box = cls([1])
    view = box.view
    assert cached_value(box, "view") is view
    box.source = [1]
    assert cached_value(box, "view") is None
    drop_cache(box, "view")
    assert cached_value(box, "view") is None


def test_invalidated_by_does_not_keep_a_source_alive() -> None:
    # The cache refers to its sources weakly, so a source that is replaced
    # is collected, and a collected source followed by None is a change.
    import gc
    import weakref

    class Source:
        pass

    class Box:
        computed = 0

        def __init__(self) -> None:
            self.source = Source()
            self._view = None

        @smartproperty(cache=True, invalidated_by=("source",))
        def view(self) -> int:
            type(self).computed += 1
            return type(self).computed

    box = Box()
    assert box.view == 1
    old = weakref.ref(box.source)
    box.source = None
    gc.collect()
    assert old() is None
    assert box.view == 2
    assert box.view == 2


def test_invalidated_by_needs_a_cache() -> None:
    with pytest.raises(ValueError):
        smartproperty(lambda self: None, invalidated_by=("source",))


# smartsetter


# --- smartsetter ------------------------------------------------------


def _setter_box() -> type:
    # `value` reads `_value`, named after its setter, and counts its calls;
    # `other` reads the given `_stored`.
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
    # As in the transformations: the field is stored privately, the constructor
    # takes the public name, and assignment runs the setter.
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


def test_alias_reads_and_writes_its_source() -> None:
    # An alias reads its source and assigns it, unless it is read-only.
    class Box:
        value = 1
        other = alias("value")
        fixed = alias("value", readonly=True)

    box = Box()
    assert box.other == 1
    box.other = 2
    assert box.value == 2
    assert box.fixed == 2
    with pytest.raises(AttributeError):
        box.fixed = 3


def test_a_dotted_alias_is_read_only() -> None:
    # An alias of an attribute of another object reads it and never
    # writes into it.
    class Inner:
        value = 1

    class Outer:
        inner = Inner()
        value = alias("inner.value")

    outer = Outer()
    assert outer.value == 1
    with pytest.raises(AttributeError):
        outer.value = 2
