# TODO:
# - Refactor/Simplify validators.
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
factory : bool, default=False
    Use field type as factory if none is provided
convert : bool, default=False
    Use field type as converter if none is provided
validate : bool, default=False
    Use field type as validator if none is provided

It also differs from a standard dataclass in that field-specific options
are assigned via annotations, rather than via a `field` function:

```python
# - Default factories
#   instead of x: list = field(factory=list)
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
__all__ = ["Struct", "struct", "converters", "validators"]
# stdlib
import sys
import types as _t
from abc import ABCMeta
from collections import abc as _abc
from functools import partial
from textwrap import dedent, indent

# externals
import typing_extensions as _tx

# internals
from .constants import (
    _FIELDS, _OPTIONS, _DISCARD, _POST_INIT_NAME, _PRE_INIT_NAME, _RETURN_TYPE,
    _SELF, _HasFactory, _DEFAULT, _TYPE, _CONVERTER, _VALIDATOR, MISSING
)
from .utils import _get_origin, rebuild_cls
from .options import *
from .fields import *

from . import converters, validators
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


def _add_fields(
    fields: dict[str, Field],
    new_fields: _tx.Iterable[Field],
    replace: bool = False,
    reverse: bool = False,
    inherit: _tx.List[str] = ("doc",),
) -> None:
    # Add fields to an existing dict of fields.
    #
    # This is used when constructing the dictionary of inherited fields.
    # * replace :
    #   If True, then new fields will replace existing fields.
    #   If False, then existing fields will be preserved.
    # * reverse :
    #   If True, then new fields will be added before existing fields.
    #   If False, then new fields will be added after existing fields.
    #   In both case, the order of `new_fields` is preserved.
    if replace and not reverse:
        if inherit:
            for new_field in new_fields:
                if new_field.name in fields:
                    old_field = fields[new_field.name]
                    for attr in inherit:
                        if getattr(new_field, attr, MISSING) is MISSING:
                            continue
                        setattr(new_field, attr, getattr(old_field, attr))
                fields[new_field.name] = new_field
        else:
            fields.update({f.name: f for f in new_fields})

    elif replace and reverse:
        prev_fields = fields.copy()
        fields.clear()
        fields.update({f.name: f for f in new_fields})
        for name, field in prev_fields.items():
            fields.setdefault(name, field)
            for attr in inherit:
                if getattr(fields[name], attr, MISSING) is MISSING:
                    setattr(fields[name], attr, getattr(field, attr))

    elif not replace and not reverse:
        for f in new_fields:
            fields.setdefault(f.name, f)
            for attr in inherit:
                if getattr(fields[f.name], attr, MISSING) is MISSING:
                    setattr(fields[f.name], attr, getattr(f, attr))

    elif not replace and reverse:
        prev_fields = fields.copy()
        fields.clear()
        fields.update(prev_fields)
        for f in new_fields:
            fields.setdefault(f.name, f)
            for attr in inherit:
                if getattr(fields[f.name], attr, MISSING) is MISSING:
                    setattr(fields[f.name], attr, getattr(f, attr))


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

    # Get globals of the module where this class is defined.
    if namespace["__module__"] in sys.modules:
        globals = sys.modules[namespace["__module__"]].__dict__
    else:
        # Theoretically this can happen if someone writes
        # a custom string to cls.__module__.  In which case
        # such dataclass won't be fully introspectable
        # (w.r.t. typing.get_type_hints) but will still function
        # correctly.
        globals = {}

    fnbuilder = _FuncBuilder(globals)

    # Save qualified name -- we will use it when generating methods.
    qualname = namespace["__qualname__"]

    # Now that dicts retain insertion order, there's no reason to use
    # an ordered dict.  I am leveraging that ordering here, because
    # derived class fields overwrite base class fields, but the order
    # is defined by the base class, which is found first.
    fields = {}

    # Class options that are not explicitly set are inherited from
    # base classes in MRO order. Derived classes only override base
    # classes if options are explicitly set (not MISSING).
    options = Options.make_default()

    # Find our base classes in reverse MRO order, so that order is
    # obtained from MRO, but value is obtained from most derived class.
    mro = type(_DISCARD, bases, {}).__mro__
    for b in reversed(mro):
        # Only process classes that have been processed by our
        # decorator.  That is, they have a _FIELDS attribute.
        base_fields = getattr(b, _FIELDS, None)
        if base_fields is not None:
            base_options = getattr(b, _OPTIONS)
            options.update(base_options)
            _add_fields(
                fields,
                base_fields.values(),
                replace=True,
                reverse=options.reverse
            )

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
    # things, and set the d
    cls_fields = []
    for field_name, type_ in cls_annotations.items():
        # Make Field from annotation
        field = Field.from_hint(field_name, type_)

        # If the class attribute (which is the default value for this
        # field) exists and is of type `Field`, replace it with the real
        # default. This is so that normal class introspection sees a
        # real default value, not a `Field`.
        if isinstance(namespace.get(field.name), Field):
            field.update(namespace[field.name])
            if field.default is MISSING:
                # If there's no default, delete the class attribute.
                # This happens if we specify field(repr=False), for
                # example (that is, we specified a field object, but
                # no default value).  Also if we're using a default
                # factory.  The class attribute should not be set at
                # all in the post-processed class.
                namespace.pop(field.name, None)
            else:
                namespace[field.name] = field.default

        # If the class attribute exists and is not a Field, then use it
        # as the default value for this field.
        elif field.name in namespace:
            field.default = namespace[field.name]

        cls_fields.append(field)

    # Insert fields from this class, in correct order.
    _add_fields(fields, cls_fields, replace=True, reverse=options.reverse)

    # Set unset field options from class options
    for field in fields.values():
        field.setdefault(options)

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

    # Check if pre and/or post init methods are defined in this class.
    prepost = []
    if _PRE_INIT_NAME in namespace:
        prepost += ["pre"]
    if _POST_INIT_NAME in namespace:
        prepost += ["post"]
    prepost = "+".join(prepost)

    # Build __init__
    if options.init:
        fnbuilder.add_fn(**_make_init_smart(fields, prepost))
        # namespace.setdefault("__init__", _make_init(qualname, fields))

    # TODO
    # _set_new_attribute(cls, '__replace__', _replace)

    # Include only real fields.  This is used in all of the following methods.
    real_fields = {name: f for name, f in fields.items() if not f.var}

    if options.repr:
        repr_fields = {name: f for name, f in fields.items() if f.repr}
        namespace.setdefault("__repr__", _make_repr(qualname, repr_fields))

    if options.eq:
        namespace.setdefault("__eq__", _make_eq(qualname, real_fields))

    if options.order:
        namespace.setdefault("__lt__", _make_lt(qualname, real_fields))

    # Decide if/how we're going to create a hash function.
    _make_hash = _hash_action[bool(options.unsafe_hash),
                              bool(options.eq),
                              bool(options.frozen),
                              has_explicit_hash]
    if _make_hash:
        namespace.setdefault("__hash__", _make_hash(qualname, real_fields))

    if options.match_args:
        # I could probably compute this once.
        namespace.setdefault("__match_args__", tuple(
            f.name for f in fields.values() if f.init and f.positional
        ))

    if options.frozen:
        getstate, setstate = _make_state(qualname, real_fields)
        namespace.setdefault("__getstate__", getstate)
        namespace.setdefault("__setstate__", setstate)

    if options.mapping:
        dict_fields = {f.name: f for f in fields.values() if f.key}
        for name, func in _make_mapping(qualname, dict_fields).items():
            namespace.setdefault(name, func)
        Mapping = _abc.Mapping if options.frozen else _abc.MutableMapping
        if not any(issubclass(base, Mapping) for base in bases):
            bases += (Mapping,)

    # It's an error to specify weakref_slot if slots is False.
    if options.weakref_slot and not options.slots:
        raise TypeError('weakref_slot is True but slots is False')
    if options.slots:
        if '__slots__' in namespace:
            raise TypeError(f'{clsname} already specifies __slots__')
        weakref_slot = options.weakref_slot
        namespace["__slots__"] = _make_slots(bases, real_fields, weakref_slot)

    fnbuilder.insert_fns(clsname, namespace)

    # Add attributes to class documentation
    if options.doc:
        doc = namespace.get('__doc__', '').rstrip("\n")
        doc = "\n\n".join([doc, _make_doc_class(fields)])
        namespace['__doc__'] = doc

    return clsname, bases, namespace


class _FuncBuilder:
    # Also adapted from dataclasses

    def __init__(self, globals: dict) -> None:
        self.methods = {}  # name -> function
        self.globals = globals
        self.locals = {}
        self.overwrite_errors = {}
        self.unconditional_adds = {}

    def add_fn(
        self, name: str, args: _tx.List[str], body: _tx.List[str], *,
        doc: _tx.Optional[_tx.List[str]] = None,
        locals: _tx.Optional[dict] = None,
        return_type: _tx.Any = MISSING,
        overwrite_error: _tx.Union[bool, str] = False,
        unconditional_add: bool = False,
        decorator: _tx.Optional[str] = None
    ) -> None:
        if locals is not None:
            self.locals.update(locals)

        if overwrite_error:
            self.overwrite_errors[name] = overwrite_error

        if unconditional_add:
            self.unconditional_adds[name] = True

        if return_type is not MISSING:
            self.locals[_RETURN_TYPE(name)] = return_type
            return_annotation = f'->{_RETURN_TYPE(name)}'
        else:
            return_annotation = ''

        args = ','.join(args or [])
        body = '\n'.join(body or ['pass'])
        doc = '\n'.join(['"""'] + (doc or []) + ['"""'])

        src = "\n".join([
            f"def {name}({args}){return_annotation}:",
            indent(doc, " " * 4),
            indent(body, " " * 4),
        ])
        if decorator:
            src = f'{decorator}\n{src}'
        self.methods[name] = src

    def insert_fns(self, clsname: str, namespace: dict) -> None:
        # The source to all of the functions we're generating.
        fns_src = '\n'.join(self.methods.values())

        # The locals they use.
        local_vars = ','.join(self.locals.keys())

        # The names of all of the functions, used for the return value of the
        # outer function.  Need to handle the 0-tuple specially.
        if len(self.methods) == 0:
            return_names = "()"
        else:
            return_names  =f'({",".join(self.methods.keys())},)'

        # txt is the entire function we're going to execute, including the
        # bodies of the functions we're defining.  Here's a greatly simplified
        # version:
        # def __create_fn__():
        #  def __init__(self, x, y):
        #   self.x = x
        #   self.y = y
        #  @recursive_repr
        #  def __repr__(self):
        #   return f"cls(x={self.x!r},y={self.y!r})"
        # return __init__,__repr__

        txt = "\n".join([
            f"def __create_fn__({local_vars}):",
            indent(f"{fns_src}", " " * 4),
            indent(f"return {return_names}", " " * 4)
        ])
        temporary_namespace = {}
        exec(txt, self.globals, temporary_namespace)
        fns = temporary_namespace['__create_fn__'](**self.locals)

        # Now that we've generated the functions, assign them into cls.
        qualname = namespace["__qualname__"]
        for name, fn in zip(self.methods, fns):
            fn.__qualname__ = f"{qualname}.{fn.__name__}"
            if self.unconditional_adds.get(name, False):
                namespace[name] = fn
            elif name not in namespace:
                namespace[name] = fn
            elif self.overwrite_errors.get(name, False):
                msg_extra = self.overwrite_errors[name]
                error_msg = (
                    f'Cannot overwrite attribute {name} in class {clsname}'
                )
                if msg_extra is not True:
                    error_msg = f'{error_msg} {msg_extra}'
                raise TypeError(error_msg)


def _hash_set_none(qualname: str, fields: dict) -> None:
    return None


def _hash_exception(qualname: str, fields: dict) -> _tx.NoReturn:
    raise TypeError(
        f'Cannot overwrite attribute __hash__ in class {qualname}')


def _hash_add(qualname: str, fields: dict) -> int:
    fields = [
        f for f in fields.values()
        if (f.compare if f.hash is None else f.hash)
    ]

    def __hash__(self) -> int:
        values = tuple(getattr(self, f.name) for f in fields)
        return hash(values)

    __hash__.__qualname__ = f"{qualname}.__hash__"
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


def _make_doc_class(fields: dict[str, Field]) -> str:
    attrdocs, classattrdocs = [], []
    for name, field in fields.items():
        if not field.var:
            attrdocs.append(_make_doc_elem(field, name))
        elif not field.init:
            classattrdocs.append(_make_doc_elem(field, name))
    attrdocs = "\n".join(attrdocs)
    classattrdocs = "\n".join(classattrdocs)
    if attrdocs:
        attrdocs = "Attributes\n----------\n" + attrdocs
    if classattrdocs:
        classattrdocs = "Class Attributes\n----------------\n" + classattrdocs
    return "\n\n".join([attrdocs, classattrdocs])


def _make_doc_elem(field: Field, name: _tx.Optional[str] = None) -> str:

    name = name or field.name

    default = field.default
    if field.factory:
        default = _HasFactory(field.factory)

    doctype = field.type
    if _get_origin(doctype) in (_tx.Optional, _tx.Annotated):
        doctype = _tx.get_args(doctype)[0]
    elif _get_origin(doctype) in (_tx.Union, _t.UnionType):
        # Simplify the representation of optional unions.
        if (
            len(_tx.get_args(doctype)) == 2 and (
                None in _tx.get_args(doctype) or
                type(None) in _tx.get_args(doctype)
            )
        ):
            doctype = next(iter(
                arg
                for arg in _tx.get_args(doctype)
                if arg not in (None, type(None))
            ))
        else:
            doctype = " | ".join([
                arg.__qualname__
                if isinstance(arg, type) else
                repr(arg)
                for arg in _tx.get_args(doctype)
            ])
    doctype = (
        doctype
        if isinstance(doctype, str) else
        doctype.__qualname__
        if isinstance(doctype, type) else
        repr(doctype)
    )
    doc = (
        f"{name} : {doctype}, optional"
        if default is None else
        f"{name} : {doctype}, default={default!r}"
        if default is not MISSING else
        f"{name} : {doctype}"
    )
    if field.doc:
        doc += "\n" + indent(dedent(field.doc).strip(), " " * 4)
    return doc


def _make_init_smart(
    fields: dict[str, Field], prepost: str=""
) -> dict:

    locals = {"object": object, "_HasFactory": _HasFactory}
    positional_onlys, args, kw_onlys = {}, {}, {}

    SELF = "self"
    for name, field in fields.items():
        if field.init and field.positional and not field.kw:
            positional_onlys[name] = field
        elif field.init and field.positional and field.kw:
            args[name] = field
        elif field.init and not field.positional and field.kw:
            kw_onlys[name] = field
        else:
            continue
        if name == "self":
            SELF = _SELF

    def _make_signature_elem(field: Field) -> _tx.Tuple[str, str]:
        name = field.name
        default = field.default
        if field.factory:
            default = _HasFactory(field.factory)
        locals[_TYPE(name)] = field.type
        if default is MISSING:
            signature = f"{name}: {_TYPE(name)}"
        else:
            locals[_DEFAULT(name)] = default
            signature = f"{name}: {_TYPE(name)}={_DEFAULT(name)}"
        doc = _make_doc_elem(field, name)
        return signature, doc

    def _check_signature(signature: _tx.List[str]) -> None:
        has_default = False
        for elem in signature:
            if elem == "*":
                break
            if elem == "/":
                continue
            if "=" in elem:
                has_default = True
            elif has_default:
                raise SyntaxError(f"parameter without a default follows "
                                  f"parameter with a default: {elem}")

    signature, doc = [], ["Parameters", "----------"]
    for name, field in positional_onlys.items():
        signature_elem, doc_elem = _make_signature_elem(field)
        signature.append(signature_elem)
        doc.append(doc_elem)
    if positional_onlys:
        signature.append("/")
    for name, field in args.items():
        signature_elem, doc_elem = _make_signature_elem(field)
        signature.append(signature_elem)
        doc.append(doc_elem)
    if kw_onlys:
        signature.append("*")
    for name, field in kw_onlys.items():
        signature_elem, doc_elem = _make_signature_elem(field)
        signature.append(signature_elem)
        doc.append(doc_elem)

    _check_signature(signature)

    def _make_prepost_call(func: str) -> str:
        prepost_args = []
        for name, field in positional_onlys.items():
            if field.var:
                prepost_args.append(f"{name}")
        for name, field in args.items():
            if field.var:
                prepost_args.append(f"{name}")
        for name, field in kw_onlys.items():
            if field.var:
                prepost_args.append(f"{name}={name}")
        prepost_args = ", ".join(prepost_args)
        return f"{SELF}.{func}({prepost_args})"

    def _make_body_elem(name: str, field: Field) -> str:
        body = ""
        if field.factory:
            body += dedent(f"""
            if isinstance({name}, _HasFactory):
                {name} = {name}()
            """)
        if field.converter:
            locals[_CONVERTER(name)] = field.converter
            body += dedent(f"""
            {name} = {_CONVERTER(name)}({name})
            """)
        if field.validator:
            locals[_VALIDATOR(name)] = field.validator
            body += dedent(f"""
            {name} = {_VALIDATOR(name)}({name})
            """)
        if not field.var:
            # NOTE: we by pass the object's __setattr__ to avoid running
            # through conversion and validation multiple times.
            body += dedent(f"""
            object.__setattr__({SELF}, {field.name!r}, {name})
            """)
        return body

    body = []
    if "pre" in prepost:
        body.append(_make_prepost_call(_PRE_INIT_NAME))
    for field in positional_onlys.values():
        body.append(_make_body_elem(name, field))
    for name, field in args.items():
        body.append(_make_body_elem(name, field))
    for name, field in kw_onlys.items():
        body.append(_make_body_elem(name, field))
    if "post" in prepost:
        body.append(_make_prepost_call(_POST_INIT_NAME))

    return {
        "name": "__init__",
        "args": [SELF] + signature,
        "body": body,
        "doc": doc,
        "locals": locals,
        "return_type": None,
    }


def _make_init(qualname: str, fields: dict[str, Field]) -> _tx.Callable:

    self_name = _SELF if "self" in fields else "self"

    _std_init_fields = {
        name: field
        for name, field in fields.items()
        if field.init and field.positional
    }
    _positional_only_init_fields = {
        name: field
        for name, field in fields.items()
        if field.init and field.positional and not field.kw
    }
    _kw_only_init_fields = {
        name: field
        for name, field in fields.items()
        if field.init and not field.positional and field.kw
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

            elif field.factory:
                arg = field.factory()

            else:
                raise TypeError(f"Missing required argument: {field.name!r}")

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

                    elif field.factory:
                        arg = field.factory()

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

    __init__.__qualname__ = f"{qualname}.__init__"
    return __init__


def _make_repr(qualname: str, fields: dict[str, Field]) -> _tx.Callable:

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

    __repr__.__qualname__ = f"{qualname}.__repr__"
    return __repr__


def _make_eq(qualname: str, fields: dict[str, Field]) -> _tx.Callable:

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

    __eq__.__qualname__ = f"{qualname}.__eq__"
    return __eq__


def _make_lt(qualname: str, fields: dict[str, Field]) -> _tx.Callable:

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

    __lt__.__qualname__ = f"{qualname}.__lt__"
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

    __delattr__.__qualname__ = f"{cls.__qualname__}.__delattr__"
    __setattr__.__qualname__ = f"{cls.__qualname__}.__setattr__"
    return __delattr__, __setattr__


def _make_state(qualname: str, fields: dict[str, Field]) -> _tx.Callable:

    def __getstate__(self) -> _tx.Tuple:
        fields = [f for f in fields.values() if not f.var]
        return tuple(getattr(self, f.name) for f in fields)

    def __setstate__(self, state: _tx.Tuple) -> None:
        fields = [f for f in fields.values() if not f.var]
        for field, value in zip(fields, state):
            # use setattr because dataclass may be frozen
            object.__setattr__(self, field.name, value)

    __getstate__.__qualname__ = f"{qualname}.__getstate__"
    __setstate__.__qualname__ = f"{qualname}.__setstate__"
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


def _make_mapping(qualname: str, fields: dict[str, Field]) -> _tx.Mapping[str, _tx.Callable]:

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

    __getitem__.__qualname__ = f"{qualname}.__getitem__"
    __setitem__.__qualname__ = f"{qualname}.__setitem__"
    __delitem__.__qualname__ = f"{qualname}.__delitem__"
    __iter__.__qualname__ = f"{qualname}.__iter__"
    __len__.__qualname__ = f"{qualname}.__len__"
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


_DOC_OPTIONS = """
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
factory : bool, default=False
    Use field type as factory if none is provided
convert : bool, default=False
    Use field type as converter if none is provided
validate : bool, default=False
    Use field type as validator if none is provided
mapping : bool, default=False
    Implement the `Mapping` protocol.
reverse : bool, default=False
    Use the reverse MRO order to determine field order.
    This only affects the relaive order of the fields of one class
    with respect to the fields of its base classes.
""".strip()


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
    {DOC_OPTIONS}

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


class Struct(metaclass=MetaStruct):
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
    {DOC_OPTIONS}
    """

    # Set __slots__ so that inheriting classes can have slot=True
    __slots__ = ()


MetaStruct.__doc__ = MetaStruct.__doc__.format(DOC_OPTIONS=_DOC_OPTIONS)
Struct.__doc__ = Struct.__doc__.format(DOC_OPTIONS=_DOC_OPTIONS)


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
