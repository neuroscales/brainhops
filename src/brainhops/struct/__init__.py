# TODO:
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
x: Annotated[int, Init()]
x: Annotated[int, Init(True)]
x: NoInit[int]
x: Annotated[int, NoInit()]
x: Annotated[int, Init(False)]

# - Keyword-only arguments
#   instead of x: int = field(kw_only=True)
x: KwOnly[int]
x: Annotated[int, KwOnly()]
x: Annotated[int, KwOnly(True)]
x: NotKwOnly[int]
x: Annotated[int, NotKwOnly()]
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
x: Annotated[int, Frozen()]
x: Annotated[int, Frozen(True)]
x: NotFrozen[int]
x: Annotated[int, NotFrozen()]
x: Annotated[int, Frozen(False)]
```
"""
__all__ = ["Struct", "struct"]
from abc import ABCMeta
from collections import abc as _abc
from functools import partial
import types as _t
import typing_extensions as _tx

from .constants import _FIELDS, _OPTIONS, _DISCARD, _POST_INIT_NAME, _PRE_INIT_NAME, MISSING
from .utils import rebuild_cls
from .options import *
from .fields import *

from .options import __all__ as __all_options__
from .fields import __all__ as __all_fields__
__all__ += __all_fields__
__all__ += __all_options__


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------
# Adapted from Python's standard library `dataclasses` module, which is 
# licensed under the Python Software Foundation License Version 2.

def __post_new__(cls: type) -> type:
    # These methods have to be assigned post-new, because they 
    # use super and therefore need to reference the class.

    fields = getattr(cls, _FIELDS, {})
    fields = {name: field for name, field in fields.items() if not field.var}
    __delattr__, __setattr__ = _make_assign(cls)
    if "__setattr___" not in cls.__dict__:
        cls.__setattr__ = __setattr__
    if "__delattr___" not in cls.__dict__:
        cls.__delattr__ = __delattr__

    return cls


def __pre_new__(
    metacls: "MetaStruct", 
    clsname: str, 
    bases: tuple[type, ...], 
    namespace: dict, 
    **kwargs
) -> tuple[str, tuple[type, ...], dict]:
    if clsname == _DISCARD:
        # This is a dummy class used to compute the MRO of our class 
        # without including the class itself.
        return clsname, bases, namespace

    # Now that dicts retain insertion order, there's no reason to use
    # an ordered dict.  I am leveraging that ordering here, because
    # derived class fields overwrite base class fields, but the order
    # is defined by the base class, which is found first.
    fields = {}

    # Class options that are not explicitly set are inherited from 
    # base classes in MRO order. Derived classes only override base
    # classes if options are explicitly set (not MISSING).
    options = Options.make_default()

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
    options.update(Options(**kwargs))
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
    for field_name, type_ in cls_annotations.items():
        default = namespace.get(field_name, MISSING)
        field = Field.from_hint(field_name, type_, default)
        field.setdefault(options)
        cls_fields.append(field)

    for f in cls_fields:
        fields[f.name] = f

        # If the class attribute (which is the default value for this
        # field) exists and is of type 'StructField', replace it with 
        # the real default.  This is so that normal class introspection
        # sees a real default value, not a StructField.
        if isinstance(namespace.get(f.name), Field):
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
    for attr_name, value in namespace.items():
        if isinstance(value, Field) and not attr_name in cls_annotations:
            raise TypeError(
                f'{attr_name!r} is a field but has no type annotation'
            )

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
        repr_fields = {name: f for name, f in fields.items() if f.repr}
        namespace.setdefault("__repr__", _make_repr(repr_fields))

    if options.eq:
        namespace.setdefault("__eq__", _make_eq(real_fields))

    if options.order:
        namespace.setdefault("__lt__", _make_lt(real_fields))

    # Decide if/how we're going to create a hash function.
    _make_hash = _hash_action[bool(options.unsafe_hash),
                              bool(options.eq),
                              bool(options.frozen),
                              has_explicit_hash]
    if _make_hash:
        namespace.setdefault("__hash__", _make_hash(clsname, real_fields))

    if options.match_args:
        # I could probably compute this once.
        namespace.setdefault("__match_args__", tuple(
            f.name for f in fields.values() if f.init and f.positional
        ))

    if options.frozen:
        getstate, setstate = _make_state(real_fields)
        namespace.setdefault("__getstate__", getstate)
        namespace.setdefault("__setstate__", setstate)

    if options.mapping:
        dict_fields = {f.name: f for f in fields.values() if f.key}
        for name, func in _make_mapping(dict_fields).items():
            namespace.setdefault(name, func)
        Mapping = _abc.Mapping if options.frozen else _abc.MutableMapping
        if not any(isinstance(base, Mapping) for base in bases):
            bases += (Mapping,)

    # It's an error to specify weakref_slot if slots is False.
    if options.weakref_slot and not options.slots:
        raise TypeError('weakref_slot is True but slots is False')
    if options.slots:
        if '__slots__' in namespace:
            raise TypeError(f'{clsname} already specifies __slots__')
        weakref_slot = options.weakref_slot
        namespace["__slots__"] = _make_slots(bases, real_fields, weakref_slot)

    return clsname, bases, namespace


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


def _make_init(fields: dict[str, Field]) -> _tx.Callable:

    self_name = "__struct_self__" if "self" in fields else "self"
    _std_init_fields = {
        name: field for name, field in fields.items()
        if field.init and field.positional
    }
    _kw_only_init_fields = {
        name: field for name, field in fields.items()
        if field.init and field.kw and not field.positional
    }
    _positional_only_init_fields = {
        name: field for name, field in fields.items()
        if field.init and field.positional and not field.kw
    }

    def __init__(*args, **kwargs) -> None:

        # --------------------------------------------------------------
        # Get self
        if args:
            self, *args = args
        else:
            self = kwargs.pop(self_name)

        # --------------------------------------------------------------
        # Call pre-init.
        if hasattr(self, _PRE_INIT_NAME):
            args, kwargs = getattr(self, _PRE_INIT_NAME)(*args, **kwargs)

        # --------------------------------------------------------------
        # Make copies of the fields dicts, because we're going to be mutating
        std_init_fields = dict(_std_init_fields)
        kw_only_init_fields = dict(_kw_only_init_fields)
        positional_only_init_fields = dict(_positional_only_init_fields)
        init_vars = {}

        # --------------------------------------------------------------
        # First, unroll positional arguments.
        args = list(args)
        while args:
            if not std_init_fields:
                raise TypeError(f"Too many positional arguments: {len(args)}")
            arg = args.pop(0)
            field = std_init_fields.pop(next(iter(std_init_fields)))

            if field.var:
                init_vars[field.name] = arg
                continue

            if not field.frozen:
                # We let setattr() do the work
                setattr(self, field.name, arg)

            else:
                if field.converter:
                    arg = field.converter(arg)

                if field.validator:
                    arg = field.validator(arg)
            
                object.__setattr__(self, field.name, arg)

        # --------------------------------------------------------------
        # Merge all remaining fields. They will either be in kwargs,
        # or will be assigned their default value.
        kw_only_init_fields.update(std_init_fields)

        # --------------------------------------------------------------
        # Unroll keyword arguments.
        for name, arg in kwargs.items():

            if name in positional_only_init_fields:
                raise TypeError(
                    f"Got positional-only argument as keyword: {name!r}"
                )

            if name not in kw_only_init_fields:
                raise TypeError(f"Unexpected keyword argument: {name!r}")
            
            field = kw_only_init_fields.pop(name)

            if field.var:
                init_vars[field.name] = arg
                continue

            if not field.frozen:
                # We let setattr() do the work
                setattr(self, field.name, arg)

            else:
                if field.converter:
                    arg = field.converter(arg)

                if field.validator:
                    arg = field.validator(arg)
            
                object.__setattr__(self, field.name, arg)

        # --------------------------------------------------------------
        # Assign default values for any remaining fields.
        for field in kw_only_init_fields.values():

            if field.default is not MISSING:
                arg = field.default
            
            elif field.default_factory:
                arg = field.default_factory()

            else:
                raise TypeError(f"Missing required argument: {field.name!r}")
        
            if field.var:
                init_vars[field.name] = arg

            if not field.frozen:
                # We let setattr() do the work
                setattr(self, field.name, arg)

            else:
                if field.converter:
                    arg = field.converter(arg)

                if field.validator:
                    arg = field.validator(arg)

                object.__setattr__(self, field.name, arg)

        # --------------------------------------------------------------
        # Call post-init on InitVars
        if hasattr(self, _POST_INIT_NAME):
            postinit_args = []
            for field in fields.values():

                if not (field.init and field.var):
                    continue

                arg = init_vars.get(field.name, MISSING)
                if arg is MISSING:

                    if field.default is not MISSING:
                        arg = field.default

                    elif field.default_factory:
                        arg = field.default_factory()

                    else:
                        raise TypeError(
                            f"Missing required argument for post-init: "
                            f"{field.name!r}"
                        )
            
                if field.converter:
                    arg = field.converter(arg)

                if field.validator:
                    arg = field.validator(arg)

                postinit_args.append(arg)
            getattr(self, _POST_INIT_NAME)(*postinit_args)

    return __init__


def _make_repr(fields: dict[str, Field]) -> _tx.Callable:

    def __repr__(self) -> str:
        params = [
            f"{field.name}={getattr(self, field.name)!r}" 
            for field in fields.values()
            if field.repr is True or (
                field.repr == "hide_none" and 
                getattr(self, field.name) is not None
            )
        ]
        params = ", ".join(params)
        return f"{self.__class__.__name__}({params})"
    
    return __repr__


def _make_eq(fields: dict[str, Field]) -> _tx.Callable:

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


def _make_lt(fields: dict[str, Field]) -> _tx.Callable:

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


def _make_assign(cls: type) -> type:

    __class__ = cls
    fields = getattr(cls, _FIELDS, {})
    fields = {name: field for name, field in fields.items() if not field.var}

    # We are calling object methods instead of super(), beause
    # super() falls back to inherited struct methods, which we don't want.

    def __delattr__(self, name: str) -> None:
        field = fields.get(name)
        if field:
            if getattr(field, 'frozen', False):
                raise AttributeError(f"Cannot delete frozen field {name!r}")
        elif getattr(type(self), _OPTIONS).frozen:
            raise AttributeError(
                f"Cannot delete attribute {name!r} on frozen class"
            )
        object.__delattr__(self, name)

    def __setattr__(self, name: str, value: _tx.Any) -> None:
        field = fields.get(name)
        if field and not field.var:
            if field.frozen:
                raise AttributeError(f"Cannot set frozen field {name!r}")
            if field.converter:
                value = field.converter(value)
            if field.validator:
                value = field.validator(value)
        elif getattr(type(self), _OPTIONS).frozen:
            raise AttributeError(
                f"Cannot set attribute {name!r} on frozen class"
            )
        object.__setattr__(self, name, value)

    return __delattr__, __setattr__


def _make_state(fields: dict[str, Field]) -> _tx.Callable:

    def __getstate__(self) -> _tx.Tuple:
        fields = [f for f in fields.values() if not f.var]
        return tuple(getattr(self, f.name) for f in fields)

    def __setstate__(self, state: _tx.Tuple) -> None:
        fields = [f for f in fields.values() if not f.var]
        for field, value in zip(fields, state):
            # use setattr because dataclass may be frozen
            object.__setattr__(self, field.name, value)

    return __getstate__, __setstate__


def _get_slots(cls: type) -> _tx.Iterator[str]:
    slots = cls.__dict__.get('__slots__')
    if slots is None:
        # `__dictoffset__` and `__weakrefoffset__` can tell us whether
        # the base type has dict/weakref slots, in a way that works correctly
        # for both Python classes and C extension types. Extension types
        # don't use `__slots__` for slot creation
        if getattr(cls, '__weakrefoffset__', -1) != 0:
            yield '__weakref__'
        if getattr(cls, '__dictoffset__', -1) != 0:
            yield '__dict__'
    elif isinstance(slots, str):
        yield slots
    elif not hasattr(slots, '__next__'):
        # Slots may be any iterable, but we cannot handle an iterator
        # because it will already be (partially) consumed.
        yield from slots
    else:
        raise TypeError(f"Slots of '{cls.__name__}' cannot be determined")


def _make_slots(
    bases: tuple[type, ...], 
    fields: dict[str, Field], 
    weakref_slot: bool = False,
) -> _tx.Union[tuple[str, ...], dict[str, _tx.Optional[str]]]:
    mro = type(_DISCARD, bases, {}).__mro__[1:-1]
    inherited_slots = set(
        slot
        for base in mro
        for slot in _get_slots(base)
    )

    slots, has_doc = {}, False
    for field in fields.values():
        if field.name in inherited_slots:
            continue
        slots[field.name] = field.doc
        if field.doc:
            has_doc = True

    if weakref_slot and '__weakref__' not in inherited_slots:
        slots['__weakref__'] = None
    
    if not has_doc:
        slots = tuple(slots)

    return slots


def _make_mapping(fields: dict[str, Field]) -> _tx.Mapping[str, _tx.Callable]:

    def __getitem__(self, key: str) -> _tx.Any:
        field = fields.get(key)
        if field:
            value = getattr(self, field.name)
            if field.key == "hide_none" and value is None:
                raise KeyError(key)
            return value
        raise KeyError(key)

    def __setitem__(self, key: str, value: _tx.Any) -> None:
        field = fields.get(key)
        if field:
            setattr(self, field.name, value)
        else:
            raise KeyError(key)
        
    def __delitem__(self, key: str) -> None:
        field = fields.get(key)
        if field:
            delattr(self, field.name)
        else:
            raise KeyError(key)

    def __iter__(self) -> _tx.Iterator[str]:
        for field in fields.values():
            if field:
                if (field.key == "hide_none" and 
                    getattr(self, field.name) is None
                ):
                    continue
                yield field.name

    def __len__(self) -> int:
        return sum(
            field.key != "hide_none" or 
            getattr(self, field.name) is not None 
            for field in fields.values()
        )

    return {
        "__getitem__": __getitem__,
        "__setitem__": __setitem__,
        "__delitem__": __delitem__,
        "__iter__": __iter__,
        "__len__": __len__,
    }


# ----------------------------------------------------------------------
# Base
# ----------------------------------------------------------------------
# MetaStruct derives from ABCMeta so that derivatives of Struct can 
# derive from ABCs (e.g. Mapping).

class MetaStruct(ABCMeta):
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
    positional_only : bool, default=False
        Make all fields positional-only by default
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
    mapping : bool, default=False
        Implement the `Mapping` protocol.

    Returns
    -------
    cls : type
        The class being defined.
    """

    def __new__(metacls, name, bases, namespace, **kwargs) -> type:
        name, bases, namespace = __pre_new__(metacls, name, bases, namespace, **kwargs)
        cls = super().__new__(metacls, name, bases, namespace)
        cls = __post_new__(cls)
        return cls


class Struct(metaclass=MetaStruct, **Options._DEFAULTS):
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
    repr : bool | {"hide_none"}, default=True             
        Generate `__repr__` method.
        If "hide_none", then fields with value None are not included 
        in the repr.
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
    positional_only : bool, default=False
        Make all fields positional-only by default
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
    mapping : bool | {"hide_none"}, default=False
        Implement the `Mapping` protocol.
        If "hide_none", then fields with value None are not included 
        in the list of keys.

    """

    # Set __slots__ so that inheriting classes can have slot=True
    __slots__ = ()


# ----------------------------------------------------------------------
# Decorator
# ----------------------------------------------------------------------


@_tx.overload
def struct(**kwargs) -> _tx.Callable[[type], type]: ...


@_tx.overload
def struct(cls: type, **kwargs) -> type: ...


def struct(cls: _tx.Optional[type] = None, **kwargs):
    """
    Decorator for defining a Struct class.
    See `MetaStruct` for parameters and examples.
    """
    if cls is None:
        return partial(struct, **kwargs)
    return rebuild_cls(cls, partial(MetaStruct, **kwargs))

