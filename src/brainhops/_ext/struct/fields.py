__all__ = [
    "Field",
    "Default",
    "Factory",
    "ConvertTo",
    "Validate",
    "Init",
    "NoInit",
    "Kw",
    "NotKw",
    "KwOnly",
    "NotKwOnly",
    "Positional",
    "NotPositional",
    "PositionalOnly",
    "NotPositionalOnly",
    "Frozen",
    "NotFrozen",
    "Var",
    "InitVar",
    "ClassVar",
    "Repr",
    "NoRepr",
    "Compare",
    "NoCompare",
    "Eq",
    "NoEq",
    "Order",
    "NoOrder",
    "Hash",
    "NoHash",
    "Key",
    "NotKey",
    "Doc",
]
import types as _t
import typing_extensions as _tx

from .constants import MISSING, REQUIRED, MaybeMissing
from .converters import HintConverter
from .options import Options
from .utils import SlotsBase, slots, _get_origin
from .validators import HintValidator

T = _tx.TypeVar("T")


@slots(
    'name',             # Field name
    'type',             # Field type (or type hint)
    'default',          # Default value for this field.
    'factory',  # A factory function that generates a default value for this field.
    'init',             # Include this field in the generated __init__ method.
    'repr',             # Include this field in the generated __repr__ method.
    'hash',             # Include this field in the generated __hash__ method.
    'eq',               # Include this field in the generated __eq__ method.
    'order',            # Include this field in the generated __lt__ methods.
    'metadata',         # User-defined metadata
    'kw',               # Make this field a keyword in the generated __init__ method.
    'positional',       # Make this field a positional argument in the generated __init__ method.
    'init_only',        # Make this field positional-only in the generated __init__ method.
    'frozen',           # Make this field immutable after initialization.
    'converter',        # A function that converts the input value for this field.
    'validator',        # A function that validates the input value for this field.
    'var',              # Whether this field is a pseudo-field (InitVar or ClassVar).
    'doc',              # Docstring for this field.
    'key',              # Include this field in the generated dict-like interface.
)
class Field(SlotsBase):

    def __init__(self, *arg, **kwargs) -> None:
        # The positional argument is a special case in which `Field`` 
        # acts as the opposite of `Var`.
        if arg and arg[0] is not MISSING:
            kwargs["var"] = not arg[0]
        # `compare` is a special alias for setting both `eq` and `order` 
        # at the same time.
        compare = kwargs.get("compare", MISSING)
        if compare is not MISSING:
            kwargs.setdefault("eq", compare)
            kwargs.setdefault("order", compare)
        # set slots from keywords
        super().__init__(**kwargs)
    
    def __class_getitem__(cls, t: _tx.Union[type, _tx.Tuple]) -> _tx.TypeAlias:
        if not isinstance(t, tuple):
            t = (t,)
        t, *args = t
        return _tx.Annotated[t, cls(True), *args]

    @classmethod
    def from_hint(
        cls, name: str, hint: _tx.Any, default: _tx.Any = MISSING
    ) -> "Field":
        type = hint
        origin = _get_origin(hint)

        if origin is _tx.ClassVar:
            # Replace python's ClassVar with our own.
            hint = ClassVar[_tx.get_args(hint)]
            return cls.from_hint(name, hint, default)
    
        field = Field()
        if origin is _tx.Annotated:
            type, *hints = _tx.get_args(hint)
            for hint in hints:
                if isinstance(hint, Field):
                    field.update(hint)
                elif isinstance(hint, _tx.Doc):
                    field.doc = hint.documentation
        field.update(Field(name=name, type=type, default=default))
        return field
    
    def setdefault(self, options: Options) -> None:
        # When field options are not explicitly set (MISSING), they are 
        # inherited from the class options.
        if options.kw_only and options.positional_only:
            raise ValueError(
                "Cannot set both kw_only and positional_only to True"
            )
        if self.doc is MISSING:
            self.doc = None
        if self.var is MISSING:
            self.var = False
        if self.init is MISSING:
            self.init = options.init
        if self.repr is MISSING:
            self.repr = options.repr if not self.var else False
        if self.hash is MISSING:
            self.hash = True
        if self.key is MISSING:
            self.key = options.mapping
        if self.eq is MISSING:
            self.eq = options.eq
        if self.order is MISSING:
            self.order = options.order
        if options.kw_only:
            if self.kw is MISSING:
                self.kw = True
            if self.positional is MISSING:
                self.positional = False
        elif options.positional_only:
            if self.initkw_only is MISSING:
                self.kw = False
            if self.positional is MISSING:
                self.positional = True
        else:
            if self.kw is MISSING:
                self.kw = True
            if self.positional is MISSING:
                self.positional = True
        if self.frozen is MISSING:
            self.frozen = options.frozen
        if self.converter is MISSING:
            self.converter = options.convert
        if self.converter is True:
            self.converter = HintConverter(self.type)
        if self.validator is MISSING:
            self.validator = options.validate
        if self.validator is True:
            self.validator = HintValidator(self.type)
        if self.factory is MISSING: 
            self.factory = options.factory
        if self.factory is True:
            factory = self.type
            origin = _get_origin(factory)
            if origin in (_t.UnionType, _tx.Union, _tx.Optional):
                factory = _tx.get_args(factory)[0]
            elif origin in (type, _tx.Type):
                factory = lambda: _tx.get_args(factory)[0]
            else:
                factory = origin
            self.factory = factory


# ----------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------


@slots
class AnnotatedField(Field):

    __set_value__ = MISSING
    __set_slots__ = {}

    @classmethod
    def _set_slots(cls) -> _tx.Dict[str, _tx.Any]:
        set_slots = {}
        for cls in reversed(cls.__mro__):
            cls_set_slots = getattr(cls, '__set_slots__', {})
            if isinstance(cls_set_slots, str):
                cls_set_slots = (cls_set_slots,)
            if isinstance(cls_set_slots, tuple):
                cls_set_slots = {
                    slot: cls.__set_value__ 
                    for slot in cls_set_slots
                }
            set_slots.update(cls_set_slots)
        return set_slots

    def __init__(self, *values, **kwvalues) -> None:
        cls = type(self)
        set_slots = cls._set_slots()

        for name, value in zip(set_slots, values):
            kwvalues[name] = value
        for name, value in set_slots.items():
            kwvalues.setdefault(name, value)
        if any(value is REQUIRED for value in kwvalues.values()):
            raise TypeError(f"Missing required argument for {cls.__name__!r}")
        super().__init__(**kwvalues)

    def __class_getitem__(
        cls, args: _tx.Union[type, _tx.Tuple]
    ) -> _tx.TypeAlias:
        set_slots = cls._set_slots()
        values = ()
        if not isinstance(args, tuple):
            args = (args,)
        t, *args = args
        if args:
            values, args = args[:len(set_slots)], args[len(set_slots):]
        if any(value is REQUIRED for value in values):
            raise TypeError(f"Missing required argument for {cls.__name__!r}[]")
        return _tx.Annotated[t, cls(*values), *args]


@slots
class BoolAnnotatedField(AnnotatedField):

    __set_value__ = True

    def __class_getitem__(
        cls, args: _tx.Union[type, _tx.Tuple]
    ) -> _tx.TypeAlias:
        if not isinstance(args, tuple):
            args = (args,)
        t, *args = args
        return _tx.Annotated[t, cls(True), *args]


@slots
class InversedBoolAnnotatedField(BoolAnnotatedField):

    def __init__(self, value: MaybeMissing[bool] = True, /) -> None:
        super().__init__(not value)


@slots
class Default(AnnotatedField):
    """
    Specifiy that a field has a default value.

    ```python
    Default(10)    ~> Field(default=10)
    Default[int, 10] ~> Annotated[T, Field(default=10)]
    ```
    """

    __set_slots__ = {'default': REQUIRED}


@slots
class Factory(AnnotatedField):
    """
    Specifiy that a field has a default factory.
    
    ```python
    Factory()             ~> Field(factory=True)
    Factory(list)         ~> Field(factory=list)
    Factory[list]         ~> Annotated[T, Field(factory=True)]
    Factory[list, mylist] ~> Annotated[T, Field(factory=mylist)]
    ```
    """

    __set_slots__ = {'factory': True}


@slots
class ConvertTo(AnnotatedField):
    """
    Specifiy that a field has a converter.
    
    ```python
    ConvertTo()             ~> Field(converter=True)
    ConvertTo(list)         ~> Field(converter=list)
    ConvertTo[list]         ~> Annotated[T, Field(converter=True)]
    ConvertTo[list, mylist] ~> Annotated[T, Field(converter=mylist)]
    ```
    """

    __set_slots__ = {'converter': True}


@slots
class Validate(AnnotatedField):
    """
    Specifiy that a field has a validator.
    
    ```python
    Validate()                  ~> Field(validator=True)
    Validate(myvalidator)       ~> Field(validator=myvalidator)
    Validate[list]              ~> Annotated[T, Field(validator=True)]
    Validate[list, myvalidator] ~> Annotated[T, Field(validator=myvalidator)]
    ```
    """

    __set_slots__ = {'validator': True}


@slots
class Init(BoolAnnotatedField):
    """
    Specifiy that a field should [not] be included in the generated 
    `__init__` method.

    ```python
    Init()      ~> Field(init=True)
    NoInit()    ~> Field(init=False)
    Init[int]   ~> Annotated[T, Field(init=True)]
    NoInit[int] ~> Annotated[T, Field(init=False)]
    """

    __set_slots__ = 'init'


@slots
class NoInit(Init, InversedBoolAnnotatedField): ...


@slots
class Kw(BoolAnnotatedField):
    """
    Specifiy that a field is [not] a keyword-only parameter.

    ```python
    Kw()        ~> Field(kw=True)
    NotKw()     ~> Field(kw=False)
    KwOnly()    ~> Field(kw=True, positional=False)
    Kw[int]     ~> Annotated[T, Field(kw=True)]
    NotKw[int]  ~> Annotated[T, Field(kw=False)]
    KwOnly[int] ~> Annotated[T, Field(kw=True, positional=False)]
    """

    __set_slots__ = 'kw'


@slots
class NotKw(Kw, InversedBoolAnnotatedField): ...


@slots
class Positional(BoolAnnotatedField):
    """
    Specifiy that a field is [not] a positional-only parameter.

    ```python
    Positional()        ~> Field(positional=True)
    NotPositional()     ~> Field(positional=False)
    PositionalOnly()    ~> Field(positional=True, kw=False)
    Positional[int]     ~> Annotated[T, Field(positional=True)]
    NotPositional[int]  ~> Annotated[T, Field(positional=False)]
    PositionalOnly[int] ~> Annotated[T, Field(positional=True, kw=False)]
    """

    __set_slots__ = 'positional'


@slots
class NotPositional(Positional, InversedBoolAnnotatedField): ...


@slots
class KwOnly(Kw, NotPositional): ...


@slots
class NotKwOnly(Kw, Positional): ...


@slots
class PositionalOnly(NotKw, Positional): ...


@slots
class NotPositionalOnly(Kw, Positional): ...


@slots
class Frozen(BoolAnnotatedField):
    """
    Specifiy that a field is [not] frozen.

    ```python
    Frozen()       ~> Field(frozen=True)
    NotFrozen()    ~> Field(frozen=False)
    Frozen[int]    ~> Annotated[T, Field(frozen=True)]
    NotFrozen[int] ~> Annotated[T, Field(frozen=False)]
    """

    __set_slots__ = 'frozen'


@slots
class NotFrozen(Frozen, InversedBoolAnnotatedField): ...


@slots
class Var(BoolAnnotatedField):
    """
    Specifiy that a field is a pseudo-field (InitVar or ClassVar).

    ```python
    Var()         ~> Field(var=True)
    InitVar()     ~> Field(var=True, init=True)
    ClassVar()    ~> Field(var=True, init=False)

    Var[int]      ~> Annotated[T, Field(var=True)]
    InitVar[int]  ~> Annotated[T, Field(var=True, init=True)]
    ClassVar[int] ~> Annotated[T, Field(var=True, init=False)]
    ```
    """

    __set_slots__ = 'var'


@slots
class InitVar(Var, Init): ...


@slots
class ClassVar(Var, NoInit): ...


@slots
class Repr(BoolAnnotatedField):
    """
    Specifiy that a field should [not] be included in the generated 
    `__repr__` method.

    ```python
    Repr()       ~> Field(repr=True)
    NoRepr()     ~> Field(repr=False)
    Repr[int]    ~> Annotated[T, Field(repr=True)]
    NoRepr[int]  ~> Annotated[T, Field(repr=False)]
    """

    __set_slots__ = ('repr',)


@slots
class NoRepr(Repr, InversedBoolAnnotatedField): ...


@slots
class Eq(BoolAnnotatedField): 
    """ Specifiy that a field should [not] be included in the generated `__eq__` method.

    ```python
    Eq()       ~> Field(eq=True)
    NoEq()     ~> Field(eq=False)
    Eq[int]    ~> Annotated[T, Field(eq=True)]
    NoEq[int]  ~> Annotated[T, Field(eq=False)]
    ```
    """
    
    __set_slots__ = ('eq',)


@slots
class NoEq(Eq, InversedBoolAnnotatedField): ...


@slots
class Order(BoolAnnotatedField): 
    """ Specifiy that a field should [not] be included in the generated `__lt__` method.

    ```python
    Order()       ~> Field(order=True)
    NoOrder()     ~> Field(order=False)
    Order[int]    ~> Annotated[T, Field(order=True)]
    NoOrder[int]  ~> Annotated[T, Field(order=False)]
    ```
    """

    __set_slots__ = ('order',)


@slots
class NoOrder(Order, InversedBoolAnnotatedField): ...


@slots
class Compare(Eq, Order):  ...


@slots
class NoCompare(Compare, InversedBoolAnnotatedField): ...


@slots
class Hash(BoolAnnotatedField): 
    """ Specifiy that a field should [not] be included in the generated 
    `__hash__` method.

    ```python
    Hash()      ~> Field(hash=True)
    NoHash()    ~> Field(hash=False)
    Hash[int]   ~> Annotated[T, Field(hash=True)]
    NoHash[int] ~> Annotated[T, Field(hash=False)]
    ```
    """

    __set_slots__ = ('hash',)


@slots
class NoHash(Hash, InversedBoolAnnotatedField): ...


@slots
class Key(BoolAnnotatedField): 
    """ Specifiy that a field should [not] be included in the generated 
    `__hash__` method.

    ```python
    Key()       ~> Field(key=True)
    NotKey()    ~> Field(key=False)
    Key[int]    ~> Annotated[T, Field(key=True)]
    NotKey[int] ~> Annotated[T, Field(key=False)]
    ```
    """

    __set_slots__ = ('key',)


@slots
class NotKey(Key, InversedBoolAnnotatedField): ...


@slots
class Doc(AnnotatedField, _tx.Doc):
    """
    Specifiy the docstring for a field.

    ```python
    Doc("This is a field")      ~> Field(doc="This is a field")
    Doc[int, "This is a field"] ~> Annotated[T, Field(doc="This is a field")]
    ```
    """

    __set_slots__ = ('doc',)

    def __init__(self, documentation: str, /) -> None:
        _tx.Doc.__init__(self, documentation)
        AnnotatedField.__init__(self, documentation)