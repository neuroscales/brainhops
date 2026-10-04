"""Tests of `brainhops._core.fields`."""

import typing_extensions as tx

# --- LazyField --------------------------------------------------------


def _holder(**options):  # noqa: ANN003, ANN202
    from brainhops._core.fields import LazyField

    loads = []

    class Holder:
        value = LazyField(**options)

        def __setattr__(self, name: str, value: tx.Any) -> None:
            # Stands for a class that converts what it is given.
            if isinstance(value, list):
                value = tuple(value)
            super().__setattr__(name, value)

    return Holder, loads


def test_lazyfield_loads_on_first_read() -> None:
    from brainhops._core.fields import Lazy

    seen = []
    Holder, _ = _holder(on_load=lambda obj, name, value: seen.append(value))
    obj = Holder()
    calls = []
    obj.__dict__["value"] = Lazy(lambda: calls.append(1) or [1, 2])
    assert calls == []
    assert obj.value == (1, 2)  # assigned through `setattr`: converted
    assert obj.value == (1, 2)
    assert calls == [1]
    assert seen == [(1, 2)]


def test_lazyfield_loads_before_an_assignment() -> None:
    from brainhops._core.fields import Lazy

    seen = []
    Holder, _ = _holder(on_load=lambda obj, name, value: seen.append(value))
    obj = Holder()
    obj.__dict__["value"] = Lazy(lambda: "old")
    obj.value = "new"
    assert seen == ["old"]
    assert obj.value == "new"


def test_lazyfield_plain_values_and_defaults() -> None:
    from brainhops._core.fields import Lazy, LazyField

    Holder, _ = _holder(default="nothing", prepare=lambda v: v * 2)
    obj = Holder()
    assert obj.value == "nothing"
    obj.value = 3
    assert obj.value == 3
    assert not Holder.value.pending(obj)
    obj.__dict__["value"] = Lazy(lambda: 4)
    assert Holder.value.pending(obj)
    assert obj.value == 8
    assert isinstance(Holder.value, LazyField)
    assert Holder.value.name == "value"
    assert "Lazy(" in repr(Lazy(len))
