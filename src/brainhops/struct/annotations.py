__all__ = [
    "Default",
    "DefaultFactory",
    "ConvertTo",
    "Validate",
    "Init",
    "NoInit",
    "KwOnly",
    "NotKwOnly",
    "Frozen",
    "NotFrozen",
    "Var",
    "Field",
    "InitVar",
    "ClassVar",
    "Repr",
    "NotRepr",
    "Compare",
    "NotCompare",
    "Eq",
    "NotEq",
    "Order",
    "NotOrder",
    "Hash",
    "NotHash"
]

import typing_extensions as _tx

from .constants import MISSING, MaybeMissing
from .converters import HintConverter
from .validators import HintValidator

T = _tx.TypeVar("T")


class _InputDefaultsToTypeMixin:
    # Used by DefaultFactory and Converter

    def __getitem__(
        self, 
        t: type, 
        func: MaybeMissing[_tx.Callable[[], T]] = MISSING, 
        *other
    ) -> _tx.TypeAlias:
        """
        Act as a type annotation.
        
        ```python
        Annotation[T, F] -> Annotated[T, Annotation(F)]
        Annotation[T]    -> Annotated[T, Annotation(T)]
        ```
        """
        if func is MISSING:
            func = t
        cls = type(self)
        return _tx.Annotated[t, cls(func), *other]
    
    @classmethod
    def from_hint(cls, hint: _tx.Any) -> _tx.Optional[_tx.Self]:
        """
        Extract the annotation from a `Annotated` type.

        ```python
        Annotation.from_hint(Annotated[T, Annotation(F)]) -> Annotation(F)
        Annotation.from_hint(Annotated[T, Annotation])    -> Annotation(T)
        Annotation.from_hint(Annotated[T, ...])           -> MISSING
        ```
        """
        for m in getattr(hint, "__metadata__", ()):
            if isinstance(m, cls):
                return m
            if m is cls:
                return cls(*m.__args__)
        return MISSING


class _BooleanAnnotation(int):
    # Used by Init and KwOnly

    def __new__(cls, value: bool) -> _tx.Self:
        if not hasattr(cls, "_true"):
            cls._true = super().__new__(cls, True)
        if not hasattr(cls, "_false"):
            cls._false = super().__new__(cls, False)
        return cls._true if value else cls._false

    def __call__(self, inp: bool = True) -> _tx.Self:
        """
        Act as a factory.

        ```python
        Annotation(True) -> type(Annotation)(True)
        ```
        """
        return type(self)(inp)

    def __not__(self) -> _tx.Self:
        return type(self)(not bool(self))
    
    def __and__(self, value) -> _tx.Self:
        return type(self)(super().__and__(value))
    
    def __rand__(self, value) -> _tx.Self:
        return type(self)(super().__rand__(value))

    def __or__(self, value) -> _tx.Self:
        return type(self)(super().__or__(value))

    def __ror__(self, value) -> _tx.Self:
        return type(self)(super().__ror__(value))

    def __xor__(self, value) -> _tx.Self:
        return type(self)(super().__xor__(value))

    def __rxor__(self, value) -> _tx.Self:
        return type(self)(super().__rxor__(value))
    
    def __getitem__(self, t: type, *other) -> _tx.TypeAlias:
        """
        Act as a type annotation.

        ```python
        Annotation[T] -> Annotated[T, Annotation]
        Annotation[T, ...] -> Annotated[T, Annotation, ...]
        """
        return _tx.Annotated[t, self, *other]
    
    @classmethod
    def from_hint(cls, hint: _tx.Any) -> _tx.Optional[_tx.Self]:
        """
        Extract the annotation from a `Annotated` type.

        ```python
        Annotation.from_hint(Annotated[T, Annotation])        -> Annotation(True)
        Annotation.from_hint(Annotated[T, Annotation(True)])  -> Annotation(True)
        Annotation.from_hint(Annotated[T, Annotation(False)]) -> Annotation(False)
        Annotation.from_hint(Annotated[T, ...])               -> None
        ```
        """  # noqa: E501
        for m in getattr(hint, "__metadata__", ()):
            if isinstance(m, cls):
                return m
        return MISSING


class _Default(_tx.Generic[T]):

    def __init__(self, value: MaybeMissing[T] = MISSING) -> None:
        self.value = value

    @_tx.overload
    def __call__(self) -> T:
        """Return default value."""

    @_tx.overload
    def __call__(self, value: _tx.Callable[[], T]) -> _tx.Self:
        """Return a new default holder with the given value."""

    def __call__(self, value=MISSING):
        if (self.value is MISSING) == (value is MISSING):
            raise TypeError(
                "Wrong use of Default annotation. "
                "It should  either be used as a hint "
                "(`Default[T, value]`) "
                "or as an object in an annotation "
                "(`Annotated[T, Default(value)]`)."
            )
        if value is MISSING:
            return self.value
        else:
            return _Default[T](value)

    def __getitem__(
        self, 
        t: type, 
        value: MaybeMissing[T] = MISSING, 
        *other
    ) -> _tx.TypeAlias:
        """
        Act as a type annotation.
        
        ```python
        Default[T, X] -> Annotated[T, Default(X)]
        Default[T]    -> Annotated[T, Default(T())]
        ```
        """
        if value is MISSING:
            value = t()
        cls = type(self)
        return _tx.Annotated[t, cls(value), *other]
    
    @classmethod
    def from_hint(cls, hint: _tx.Any) -> _tx.Optional[_tx.Self]:
        """
        Extract the annotation from a `Annotated` type.

        ```python
        Default.from_hint(Annotated[T, Default(X)]) -> Default(X)
        Default.from_hint(Annotated[T, Default])    -> Default(T())
        Default.from_hint(Annotated[T, ...])        -> None
        ```
        """
        for m in getattr(hint, "__metadata__", ()):
            if isinstance(m, cls):
                return m
            if m is cls:
                return cls(m.__args__[0]())
        return MISSING
    

Default = _Default()


class _DefaultFactory(_InputDefaultsToTypeMixin, _tx.Generic[T]):

    def __init__(self, func: MaybeMissing[_tx.Callable[[], T]] = MISSING) -> None:
        if func is not MISSING and not callable(func):
            val = func
            func = lambda: val
        self.func = func

    @_tx.overload
    def __call__(self) -> T:
        """Generate value from factory."""

    @_tx.overload
    def __call__(self, func: _tx.Callable[[], T]) -> _tx.Self:
        """Return a new factory with the given function."""

    def __call__(self, func=MISSING):
        if (self.func is MISSING) == (func is MISSING):
            raise TypeError(
                "Wrong use of DefaultFactory annotation. "
                "It should  either be used as a hint "
                "(`DefaultFactory[T, factory]`) "
                "or as an object in an annotation "
                "(`Annotated[T, DefaultFactory(factory)]`)."
            )
        if func is MISSING:
            return self.func()
        else:
            return _DefaultFactory[T](func)


DefaultFactory = _DefaultFactory()


class _ConvertTo(_InputDefaultsToTypeMixin, _tx.Generic[T]):

    def __init__(
        self, 
        func: MaybeMissing[_tx.Callable[[_tx.Any], T]] = MISSING
    ) -> None:
        if func is not MISSING and not callable(func):
            raise ValueError("Converter function must be callable.")
        self.func = func

    def __call__(self, inp: _tx.Callable[[_tx.Any], T]) -> _tx.Union[T, _tx.Self]:
        if self.func is MISSING:
            return _ConvertTo[T](inp)
        else:
            return self.func(inp)

    def __getitem__(
        self, 
        t: type, 
        func: MaybeMissing[_tx.Callable[[], T]] = MISSING, 
        *other
    ) -> _tx.TypeAlias:
        """
        Act as a type annotation.
        
        ```python
        ConvertTo[T, V] -> Annotated[T, ConvertTo(V)]
        ConvertTo[T]    -> Annotated[T, ConvertTo(HintConverter(T))]
        ```
        """
        if func is MISSING:
            func = HintConverter(t)
        cls = type(self)
        return _tx.Annotated[t, cls(func), *other]
    
    @classmethod
    def from_hint(cls, hint: _tx.Any) -> _tx.Optional[_tx.Self]:
        """
        Extract the annotation from a `Annotated` type.

        ```python
        ConvertTo.from_hint(Annotated[T, ConvertTo(V)]) -> ConvertTo(V)
        ConvertTo.from_hint(Annotated[T, ConvertTo])    -> ConvertTo(HintConverter(T)))
        ConvertTo.from_hint(Annotated[T, ...])          -> None
        ```
        """
        for m in getattr(hint, "__metadata__", ()):
            if isinstance(m, cls):
                return m
            if m is cls:
                return cls(HintConverter(*m.__args__))
        return MISSING
    

ConvertTo = _ConvertTo()


class _Validate(_tx.Generic[T]):

    def __init__(
        self, 
        func: MaybeMissing[_tx.Callable[[_tx.Any], T]] = MISSING
    ) -> None:
        if func is not MISSING and not callable(func):
            raise ValueError("Validator function must be callable.")
        self.func = func

    def __call__(self, inp: _tx.Callable[[_tx.Any], T]) -> _tx.Union[T, _tx.Self]:
        if self.func is MISSING:
            return _Validate[T](inp)
        else:
            return self.func(inp)

    def __getitem__(
        self, 
        t: type, 
        func: MaybeMissing[_tx.Callable[[], T]] = MISSING, 
        *other
    ) -> _tx.TypeAlias:
        """
        Act as a type annotation.
        
        ```python
        Validate[T, V] -> Annotated[T, Validate(V)]
        Validate[T]    -> Annotated[T, Validate(HintValidator(T))]
        ```
        """
        if func is MISSING:
            func = HintValidator(t)
        cls = type(self)
        return _tx.Annotated[t, cls(func), *other]
    
    @classmethod
    def from_hint(cls, hint: _tx.Any) -> _tx.Optional[_tx.Self]:
        """
        Extract the annotation from a `Annotated` type.

        ```python
        Validate.from_hint(Annotated[T, Validate(V)]) -> Validate(V)
        Validate.from_hint(Annotated[T, Validate])    -> Validate(HintValidator(T)))
        Validate.from_hint(Annotated[T, ...])         -> None
        ```
        """
        for m in getattr(hint, "__metadata__", ()):
            if isinstance(m, cls):
                return m
            if m is cls:
                return cls(HintValidator(*m.__args__))
        return MISSING


Validate = _Validate()


class _Init(_BooleanAnnotation):

    def __repr__(self) -> str:
        return "Init" if self else "NoInit"


Init = _Init(True)
NoInit = _Init(False)


class _KwOnly(_BooleanAnnotation):

    def __repr__(self) -> str:
        return "KwOnly" if self else "NotKwOnly"


KwOnly = _KwOnly(True)
NotKwOnly = _KwOnly(False)


class _Frozen(_BooleanAnnotation): 

    def __repr__(self) -> str:
        return "Frozen" if self else "NotFrozen"


Frozen = _Frozen(True)
NotFrozen = _Frozen(False)


class _Var(_BooleanAnnotation): 

    def __repr__(self) -> str:
        return "Var" if self else "Field"


Var = _Var(True)
Field = _Var(False)
InitVar = _tx.Annotated[T, Init, Var]
ClassVar = _tx.Annotated[T, NoInit, Var]


class _Repr(_BooleanAnnotation): 

    def __repr__(self) -> str:
        return "Repr" if self else "NotRepr"


Repr = _Repr(True)
NotRepr = _Repr(False)


class _Eq(_BooleanAnnotation): 

    def __repr__(self) -> str:
        return "Eq" if self else "NotEq"


Eq = _Eq(True)
NotEq = _Eq(False)


class _Order(_BooleanAnnotation): 

    def __repr__(self) -> str:
        return "Order" if self else "NotOrder"


Order = _Order(True)
NotOrder = _Order(False)

Compare = _tx.Annotated[T, Eq, Order]


class _Hash(_BooleanAnnotation): 

    def __repr__(self) -> str:
        return "Hash" if self else "NotHash"


Hash = _Hash(True)
NotHash = _Hash(False)
