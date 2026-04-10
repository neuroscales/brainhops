# TODO:
# - Add option `objvar` that adds a pseudo-field (InitVar) to the 
#   very beginning of the `__init__` method, allowing the object
#   to be initialized with an existing object of the same type.
# - Add option `dictvar` that adds a pseudo-field (InitVar) to the 
#   very beginning of the `__init__` method, allowing the object
#   to be initialized with a compatible dict-like object.
# - `objvar` and `dictvar` can have the same value (e.g. `"obj"`).
# - Add converters and validators for array-like types 
#   (np.ndarray, cp.ndarray, torch.tensor, dask.array, ...)
# - Refactor/Simplify converters and validators.
#   (many have identical semantics and could share a base class)
# - Implement from_dict, to_dict, etc.
# - dataclasses generate class-specific methods that compile to much
#   more efficient bytecode than my methods (which are general-purpose).
#   We could steal some of their magic, but this is for later.
"""
A `Struct` acts like a python `dataclass`, except that it operates
via inheritence, rather than via a decorator (although the @struct 
decorator can be used if preferred).

The options typically specified in the @dataclass decorator are instead 
specified as class keyword arguments, and are inherited (or overloaded) 
by subclasses.

```python
class Point(Struct, frozen=True):
    x: float
    y: float
```

Most options supported by dataclasses are supported, but there are 
some differences. Additional options are also implemented:

Parameters
----------
init: bool, default=True
    Generate `__init__` method
repr : bool, default=True             
    Generate `__repr__` method
eq : bool, default=True               
    Generate `__eq__` method
order : bool, default=False            
    Generate `__lt__` method
unsafe_hash : bool, default=False      
    Always generate `__hash__` method
frozen : bool, default=False           
    Disable `__setattr__` and `__delattr__`
match_args : bool, default=True       
    Generate `__match_args__` for pattern matching
kw_only : bool, default=False         
    Make all fields keyword-only by default
slots : bool, default=False           
    Generate `__slots__` and remove `__dict__`
weakref_slot : bool, default=False    
    Generate a weakref slot in `__slots__`
default_factory : bool, default=False 
    Use field type as factory if none is provided
convert : bool, default=False         
    Use field type as converter if none is provided
validate : bool, default=False        
    Use field type as validator if none is provided 

It also differs from a standard dataclass in that field-specific options 
are assigned via annotations, rather than via a `field` function:

```python
# - Default factories
#   instead of x: list = field(default_factory=list)
x: DefaultFactory[list, list_factory]
x: Annotated[list, DefaultFactory(list_factory)]

#   if no factory is provided, it will use the type as the default factory
x: DefaultFactory[list] -> x: Annotated[list, DefaultFactory(list)]

# - Include in the init method
#   instead of x: int = field(init=True)
x: Init[int]
x: Annotated[int, Init]
x: Annotated[int, Init(True)]
x: NoInit[int]
x: Annotated[int, NoInit]
x: Annotated[int, Init(False)]

# - Keyword-only arguments
#   instead of x: int = field(kw_only=True)
x: KwOnly[int]
x: Annotated[int, KwOnly]
x: Annotated[int, KwOnly(True)]
x: NotKwOnly[int]
x: Annotated[int, NotKwOnly]
x: Annotated[int, KwOnly(False)]
```

It supports additional features such as  automatic conversion of field 
values via annotations:

```python
x: ConvertTo[int, partial(int, base=16)]
x: Annotated[int, ConvertTo(partial(int, base=16))]

# if no converter is provided, it will use the type as the default converter
x: ConvertTo[int] -> x: Annotated[int, ConvertTo(int)]
```

Frozen or unfrozen fields:

```python
x: Frozen[int]
x: Annotated[int, Frozen]
x: Annotated[int, Frozen(True)]
x: NotFrozen[int]
x: Annotated[int, NotFrozen]
x: Annotated[int, Frozen(False)]
```
"""
__all__ = [
    "Struct", "struct",
    "Default", "DefaultFactory", "ConvertTo", "Validate", 
    "Frozen", "Init", "KwOnly", "Repr", 
    "Hash", "Eq", "Order", "Compare", "Var", "InitVar", "ClassVar",
]
from functools import partial
import types as _t
import typing_extensions as _tx

from .constants import _FIELDS, _OPTIONS, _DISCARD, _POST_INIT_NAME, MISSING
from .converters import HintConverter, _get_origin
from .validators import HintValidator
from .annotations import (
    Default, DefaultFactory, ConvertTo, Validate, Frozen, Init, KwOnly, 
    Repr, Hash, Eq, Order, Compare, Var, InitVar, ClassVar, 
)


# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------


class _SlotsBase:
    __slots__ = ()

    def __init__(self, **kwargs) -> None:
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot, MISSING))

    def __repr__(self):
        params = (f"{slot}={getattr(self, slot)!r}" for slot in self.__slots__)
        params = ", ".join(params)
        return f"{self.__class__.__name__}({params})"
    
    def update(self, options: _tx.Self) -> None:
        for slot in self.__slots__:
            if getattr(options, slot) is not MISSING:
                setattr(self, slot, getattr(options, slot))

    def setdefault(self, options: _tx.Self) -> None:
        for slot in self.__slots__:
            if getattr(self, slot) is MISSING:
                setattr(self, slot, getattr(options, slot))


class StructOptions(_SlotsBase):

    __slots__ = (
        'init',             # Generate __init__ method
        'repr',             # Generate __repr__ method
        'eq',               # Generate __eq__ method
        'order',            # Generate __lt__ method
        'unsafe_hash',      # Always generate __hash__ method
        'frozen',           # Disable __setattr__ and __delattr__
        'match_args',       # Generate __match_args__ for pattern matching
        'kw_only',          # Make all fields keyword-only by default
        'slots',            # Generate __slots__ and remove __dict__
        'weakref_slot',     # Generate a weakref slot in __slots__
        'default_factory',  # Use field type as factory if none is provided
        'convert',          # Use field type as converter if none is provided
        'validate',         # Use field type as validator if none is provided
    )

    _DEFAULTS: _tx.Dict[str, bool] = dict(
        init=True,
        repr=True,
        eq=True,
        order=False,
        unsafe_hash=False,
        frozen=False,
        match_args=False,
        kw_only=False,
        slots=False,
        weakref_slot=False,
        default_factory=False,
        convert=False,
        validate=False
    )
    
    @staticmethod
    def make_default() -> _tx.Self:
        return StructOptions(**StructOptions._DEFAULTS)


class StructField(_SlotsBase):

    __slots__ = (
        'name',             # Field name
        'type',             # Field type (or type hint)
        'default',          # Default value for this field.
        'default_factory',  # A factory function that generates a default value for this field.
        'init',             # Include this field in the generated __init__ method.
        'repr',             # Include this field in the generated __repr__ method.
        'hash',             # Include this field in the generated __hash__ method.
        'eq',               # Include this field in the generated __eq__ method.
        'order',            # Include this field in the generated __lt__ methods.
        'metadata',         # User-defined metadata
        'kw_only',          # Make this field keyword-only in the generated __init__ method.
        'frozen',           # Make this field immutable after initialization.
        'converter',        # A function that converts the input value for this field.
        'validator',        # A function that validates the input value for this field.
        'var',              # Whether this field is a pseudo-field (InitVar or ClassVar).
    )

    def __init__(self, **kwargs) -> None:
        compare = kwargs.get("compare", MISSING)
        if compare is not MISSING:
            kwargs.setdefault("eq", True)
            kwargs.setdefault("order", True)
        super().__init__(**kwargs)
    
    @classmethod
    def from_hint(
        cls, name: str, hint: _tx.Any, default: _tx.Any = MISSING
    ) -> "StructField":
        if default is MISSING:
            default = Default.from_hint(hint)
        type = hint
        if _get_origin(hint) is _tx.Annotated:
            type = _tx.get_args(hint)[0]
        return StructField(
            name=name,
            type=type,
            default=default,
            default_factory=DefaultFactory.from_hint(hint),
            init=Init.from_hint(hint),
            repr=Repr.from_hint(hint),
            hash=Hash.from_hint(hint),
            eq=Eq.from_hint(hint),
            order=Order.from_hint(hint),
            kw_only=KwOnly.from_hint(hint),
            frozen=Frozen.from_hint(hint),
            converter=ConvertTo.from_hint(hint),
            validator=Validate.from_hint(hint),
            var=(Var.from_hint(hint) == True),
        )
    
    def setdefault(self, options: StructOptions) -> None:
        # When field options are not explicitly set (MISSING), they are 
        # inherited from the class options.
        if self.init is MISSING:
            self.init = options.init
        if self.repr is MISSING:
            self.repr = options.repr
        if self.hash is MISSING:
            self.hash = True
        if self.eq is MISSING:
            self.eq = options.eq
        if self.order is MISSING:
            self.order = options.order
        if self.kw_only is MISSING:
            self.kw_only = options.kw_only
        if self.frozen is MISSING:
            self.frozen = options.frozen
        if self.converter is MISSING and options.convert:
            self.converter = HintConverter(self.type)
        if self.validator is MISSING and options.validate:
            self.validator = HintValidator(self.type)
        if self.default_factory is MISSING and options.default_factory:
            factory = self.type
            origin = _get_origin(factory)
            if origin in (_t.UnionType, _tx.Union, _tx.Optional):
                factory = _tx.get_args(factory)[0]
            elif origin in (type, _tx.Type):
                factory = lambda: _tx.get_args(factory)[0]
            else:
                factory = origin
            self.default_factory = factory

# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------
# Adapted from Python's standard library `dataclasses` module, which is 
# licensed under the Python Software Foundation License Version 2.

def __post_new__(cls: type) -> type:
    
    fields = getattr(cls, _FIELDS, {})
    real_fields = {name: f for name, f in fields.items() if not f.var}

    # These methods have to be assigned post-new, because they 
    # use super and therefore need to reference the class.
    __delattr__, __setattr__ = _make_frozen(cls, real_fields)
    if "__setattr___" not in cls.__dict__:
        cls.__setattr__ = __setattr__
    if "__delattr___" not in cls.__dict__:
        cls.__delattr__ = __delattr__

    return cls

def __pre_new__(
    metacls: "MetaStruct", 
    name: str, 
    namespace: dict, 
    bases: tuple[type, ...], 
    **kwargs
) -> dict:
    if name == _DISCARD:
        # This is a dummy class used to compute the MRO of our class 
        # without including the class itself.
        return namespace

    # Now that dicts retain insertion order, there's no reason to use
    # an ordered dict.  I am leveraging that ordering here, because
    # derived class fields overwrite base class fields, but the order
    # is defined by the base class, which is found first.
    fields = {}

    # Class options that are not explicitly set are inherited from 
    # base classes in MRO order. Derived classes only override base
    # classes if options are explicitly set (not MISSING).
    options = StructOptions.make_default()

    # Find our base classes in reverse MRO order, and exclude
    # ourselves. In reversed order so that more derived classes
    # override earlier field definitions in base classes.
    mro = type(_DISCARD, bases, {}).__mro__
    for b in reversed(mro):
        # Only process classes that have been processed by our
        # decorator.  That is, they have a _FIELDS attribute.
        base_fields = getattr(b, _FIELDS, None)
        if base_fields is not None:
            base_options = getattr(b, _OPTIONS)
            options.update(base_options)
            for f in base_fields.values():
                fields[f.name] = f

    # Save final options for this class.
    options.update(StructOptions(**kwargs))
    namespace[_OPTIONS] = options

    # Annotations that are defined in this class (not in base
    # classes).  If __annotations__ isn't present, then this class
    # adds no new   We use this to compute fields that are
    # added by this class.
    #
    # Fields are found from cls_annotations, which is guaranteed to be
    # ordered.  Default values are from class attributes, if a field
    # has a default.  If the default value is a Field(), then it
    # contains additional info beyond (and possibly including) the
    # actual default value.  Pseudo-fields ClassVars and InitVars are
    # included, despite the fact that they're not real fields.  That's
    # dealt with later.
    cls_annotations = namespace.get('__annotations__', {})

    # Now find fields in our class.  While doing so, validate some
    # things, and set the default values (as class attributes) where
    # we can.
    cls_fields = []
    for name, type_ in cls_annotations.items():
        default = namespace.get(name, MISSING)
        field = StructField.from_hint(name, type_, default)
        field.setdefault(options)
        cls_fields.append(field)

    for f in cls_fields:
        fields[f.name] = f

        # If the class attribute (which is the default value for this
        # field) exists and is of type 'StructField', replace it with 
        # the real default.  This is so that normal class introspection
        # sees a real default value, not a StructField.
        if isinstance(namespace.get(f.name), StructField):
            if f.default is MISSING:
                # If there's no default, delete the class attribute.
                # This happens if we specify field(repr=False), for
                # example (that is, we specified a field object, but
                # no default value).  Also if we're using a default
                # factory.  The class attribute should not be set at
                # all in the post-processed class.
                namespace.pop(f.name, None)
            else:
                namespace[f.name] = f.default

    # Do we have any Field members that don't also have annotations?
    for name, value in namespace.items():
        if isinstance(value, StructField) and not name in cls_annotations:
            raise TypeError(f'{name!r} is a field but has no type annotation')

    # Remember all of the fields on our class (including bases).
    namespace[_FIELDS] = fields

    # Was this class defined with an explicit __hash__?  Note that if
    # __eq__ is defined in this class, then python will automatically
    # set __hash__ to None.  This is a heuristic, as it's possible
    # that such a __hash__ == None was not auto-generated, but it's
    # close enough.
    class_hash = namespace.get('__hash__', MISSING)
    has_explicit_hash = not (class_hash is MISSING or
                             (class_hash is None and '__eq__' in namespace))

    # If we're generating ordering methods, we must be generating the
    # eq methods.
    for field in fields.values():
        if field.order and not field.eq:
            raise ValueError('eq must be true if order is true')

    if options.init:
        namespace.setdefault("__init__", _make_init(fields))

    # TODO
    # _set_new_attribute(cls, '__replace__', _replace)

    # Include only real fields.  This is used in all of the following methods.
    real_fields = {name: f for name, f in fields.items() if not f.var}

    if options.repr:
        namespace.setdefault("__repr__", _make_repr(real_fields))

    if options.eq:
        namespace.setdefault("__eq__", _make_eq(real_fields))

    if options.order:
        namespace.setdefault("__lt__", _make_lt(real_fields))

    # Decide if/how we're going to create a hash function.
    hash_action = _hash_action[bool(options.unsafe_hash),
                               bool(options.eq),
                               bool(options.frozen),
                               "__hash__" in namespace]
    if hash_action:
        namespace.setdefault("__hash__", hash_action(name, real_fields))

    if options.match_args:
        # I could probably compute this once.
        namespace.setdefault("__match_args__", tuple(
            f.name for f in fields.values() if f.init and not f.kw_only
        ))

    # It's an error to specify weakref_slot if slots is False.
    if options.weakref_slot and not options.slots:
        raise TypeError('weakref_slot is True but slots is False')
    if options.slots:
        if '__slots__' in namespace:
            raise TypeError(f'{name} already specifies __slots__')
        namespace['__slots__'] = tuple(real_fields.keys())
        if options.weakref_slot:
            namespace['__slots__'] += ('__weakref__',)

    return namespace


def _hash_set_none(name: str, fields: dict) -> None:
    return None


def _hash_exception(name: str, fields: dict) -> _tx.NoReturn:
    raise TypeError(
        f'Cannot overwrite attribute __hash__ in class {name}')


def _hash_add(name: str, fields: dict) -> int:
    fields = [
        f for f in fields.values() 
        if (f.compare if f.hash is None else f.hash)
    ]

    def __hash__(self) -> int:
        values = tuple(getattr(self, f.name) for f in fields)
        return hash(values)
    
    return __hash__


#
#                +-------------------------------------- unsafe_hash?
#                |      +------------------------------- eq?
#                |      |      +------------------------ frozen?
#                |      |      |      +----------------  has-explicit-hash?
#                |      |      |      |
#                |      |      |      |        +-------  action
#                |      |      |      |        |
#                v      v      v      v        v
_hash_action = {(False, False, False, False): None,
                (False, False, False, True ): None,
                (False, False, True,  False): None,
                (False, False, True,  True ): None,
                (False, True,  False, False): _hash_set_none,
                (False, True,  False, True ): None,
                (False, True,  True,  False): _hash_add,
                (False, True,  True,  True ): None,
                (True,  False, False, False): _hash_add,
                (True,  False, False, True ): _hash_exception,
                (True,  False, True,  False): _hash_add,
                (True,  False, True,  True ): _hash_exception,
                (True,  True,  False, False): _hash_add,
                (True,  True,  False, True ): _hash_exception,
                (True,  True,  True,  False): _hash_add,
                (True,  True,  True,  True ): _hash_exception,
                }


def _make_init(fields: dict[str, StructField]) -> _tx.Callable:

    self_name = "__struct_self__" if "self" in fields else "self"
    _std_init_fields = {
        name: field for name, field in fields.items()
        if field.init and not field.kw_only
    }
    _kw_only_init_fields = {
        name: field for name, field in fields.items()
        if field.init and field.kw_only
    }

    def __init__(*args, **kwargs) -> None:

        std_init_fields = dict(_std_init_fields)
        kw_only_init_fields = dict(_kw_only_init_fields)
        init_vars = {}

        args = list(args)
        if args:
            self, *args = args
        else:
            self = kwargs.pop(self_name)

        # First, unroll positional arguments.
        while args:
            if not std_init_fields:
                raise TypeError(f"Too many positional arguments: {len(args)}")
            arg = args.pop(0)
            field = std_init_fields.pop(next(iter(std_init_fields)))

            if field.converter is not MISSING:
                arg = field.converter(arg)

            if field.validator is not MISSING:
                arg = field.validator(arg)

            if field.var:
                init_vars[field.name] = arg
            
            else:
                setattr(self, field.name, arg)

        # Merge all remaining fields. They will either be in kwargs,
        # or will be assigned their default value.
        kw_only_init_fields.update(std_init_fields)

        # Unroll keyword arguments.
        for name, arg in kwargs.items():
            if name not in kw_only_init_fields:
                raise TypeError(f"Unexpected keyword argument: {name!r}")
            
            field = kw_only_init_fields.pop(name)

            if field.converter is not MISSING:
                arg = field.converter(arg)

            if field.validator is not MISSING:
                arg = field.validator(arg)

            if field.var:
                init_vars[field.name] = arg
            
            else:
                setattr(self, field.name, arg)

        # Assign default values for any remaining fields.
        for field in kw_only_init_fields.values():
            if field.default_factory is not MISSING:
                value = field.default_factory()
            elif field.default is not MISSING:
                value = field.default
            else:
                raise TypeError(f"Missing required argument: {field.name!r}")
            setattr(self, field.name, value)

        # Call post-init on InitVars
        if hasattr(self, _POST_INIT_NAME):
            postinit_args = []
            for field in fields.values():
                if not (field.init and field.var):
                    continue
                arg = getattr(init_vars, field.name, MISSING)
                if arg is MISSING:
                    if field.default_factory is not MISSING:
                        arg = field.default_factory()
                    elif field.default is not MISSING:
                        arg = field.default
                    else:
                        raise TypeError(
                            f"Missing required argument for post-init: "
                            f"{field.name!r}"
                        )
                postinit_args.append(arg)
            getattr(self, _POST_INIT_NAME)(*postinit_args)

    return __init__


def _make_repr(fields: dict[str, StructField]) -> _tx.Callable:

    def __repr__(self) -> str:
        params = [
            f"{field.name}={getattr(self, field.name)!r}" 
            for field in fields.values()
            if field.repr
        ]
        params = ", ".join(params)
        return f"{self.__class__.__name__}({params})"
    
    return __repr__


def _make_eq(fields: dict[str, StructField]) -> _tx.Callable:

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if other.__class__ is self.__class__:
            return all(
                getattr(self, field.name) == getattr(other, field.name)
                for field in fields.values()
                if field.eq
            )
        return NotImplemented
    
    return __eq__


def _make_lt(fields: dict[str, StructField]) -> _tx.Callable:

    def __lt__(self, other) -> bool:
        if other.__class__ is self.__class__:
            this_value = tuple(
                getattr(self, field.name) 
                for field in fields.values() 
                if field.order
            )
            other_value = tuple(
                getattr(other, field.name) 
                for field in fields.values() 
                if field.order
            )
            return this_value < other_value
        return NotImplemented
    
    return __lt__


def _make_frozen(cls: type, fields: dict[str, StructField]) -> type:

    __class__ = cls

    def __delattr__(self, name: str) -> None:
        field = fields.get(name)
        if getattr(field, "frozen", False):
            raise AttributeError(f"Cannot delete frozen field {name!r}")
        super(cls, self).__delattr__(name)

    def __setattr__(self, name: str, value: _tx.Any) -> None:
        field = fields.get(name)
        if getattr(field, "frozen", False):
            raise AttributeError(f"Cannot set frozen field {name!r}")
        super(cls, self).__setattr__(name, value)

    return __delattr__, __setattr__


# ----------------------------------------------------------------------
# Base
# ----------------------------------------------------------------------


class MetaStruct(type):
    """
    Examples
    --------
    ```python
    # Functional API
    MetaStruct(name, bases, namespace, **options) -> type: ...

    # Class-based API
    class Struct(*bases, metaclass=MetaStruct, **options):
        ...

    # Decorator API
    @struct(**options)
    class MyStruct:
        ...
    ```

    Parameters
    ----------
    name : str
        The name of the class being defined.
    bases : tuple[type, ...]
        The base classes of the class being defined.
    namespace : dict
        The namespace of the class being defined.
    
    Other Parameters
    ----------------
    init: bool, default=True
        Generate __init__ method
    repr : bool, default=True             
        Generate __repr__ method
    eq : bool, default=True               
        Generate __eq__ method
    order : bool, default=False            
        Generate __lt__ method
    unsafe_hash : bool, default=False      
        Always generate __hash__ method
    frozen : bool, default=False           
        Disable __setattr__ and __delattr__
    match_args : bool, default=True       
        Generate __match_args__ for pattern matching
    kw_only : bool, default=False         
        Make all fields keyword-only by default
    slots : bool, default=False           
        Generate __slots__ and remove __dict__
    weakref_slot : bool, default=False    
        Generate a weakref slot in __slots__
    default_factory : bool, default=False 
        Use field type as factory if none is provided
    convert : bool, default=False         
        Use field type as converter if none is provided
    validate : bool, default=False        
        Use field type as validator if none is provided

    Returns
    -------
    cls : type
        The class being defined.
    """

    def __new__(metacls, name, bases, namespace, **kwargs) -> type:
        namespace = __pre_new__(metacls, name, namespace, bases, **kwargs)
        cls = super().__new__(metacls, name, bases, namespace)
        cls = __post_new__(cls)
        return cls


class Struct(metaclass=MetaStruct, **StructOptions._DEFAULTS):
    """
    Base class for data structures.

    Examples
    --------
    ```python
    class Point(Struct, frozen=True):
        x: float
        y: float
    ```

    Parameters
    ----------
    init: bool, default=True
        Generate `__init__` method
    repr : bool, default=True             
        Generate `__repr__` method
    eq : bool, default=True               
        Generate `__eq__` method
    order : bool, default=False            
        Generate `__lt__` method
    unsafe_hash : bool, default=False      
        Always generate `__hash__` method
    frozen : bool, default=False           
        Disable `__setattr__` and `__delattr__`
    match_args : bool, default=True       
        Generate `__match_args__` for pattern matching
    kw_only : bool, default=False         
        Make all fields keyword-only by default
    slots : bool, default=False           
        Generate `__slots__` and remove `__dict__`
    weakref_slot : bool, default=False    
        Generate a weakref slot in `__slots__`
    default_factory : bool, default=False 
        Use field type as factory if none is provided
    convert : bool, default=False         
        Use field type as converter if none is provided
    validate : bool, default=False        
        Use field type as validator if none is provided
    """
    ...


# ----------------------------------------------------------------------
# Decorator
# ----------------------------------------------------------------------


def struct(cls: _tx.Optional[type] = None, **kwargs):
    """
    Decorator for defining a Struct class.
    See `MetaStruct` for parameters and examples.
    """
    if cls is None:
        return partial(struct, **kwargs)
    return MetaStruct(cls.__name__, cls.__bases__, cls.__dict__, **kwargs)

