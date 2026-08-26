import typing_extensions as _tx

T = _tx.TypeVar("T")

# The name of an attribute on the class where we store the StructField
# objects.  Also used to check if a class is a @struct.
_FIELDS = '__struct_fields__'

# The name of an attribute on the class that stores the parameters to
# @struct.
_OPTIONS = '__struct_options__'

# The name of a method that is called before the __init__ method,
# if it exists.
# It returns (args, kwargs).
_PRE_INIT_NAME = "__pre_init__"

# The name of a method that is called after the __init__ method,
# if it exists.
_POST_INIT_NAME = "__post_init__"

# Name we give to classes that are only created temporarily to build the
# MRO and then discarded.
_DISCARD = "__struct_discard__"

# Name we give to the `self` variable, in cases where a field named `self`
# already exists.
_SELF = "__struct_self__"

# Name given to the local type variable when generating __init__


def _TYPE(x): return f"__struct_{x}_type__"

# Name given to the local default variable when generating __init__


def _DEFAULT(x): return f"__struct_{x}_default__"

# Name given to the local converter variable when generating __init__


def _CONVERTER(x): return f"__struct_{x}_converter__"

# Name given to the local validator variable when generating __init__


def _VALIDATOR(x): return f"__struct_{x}_validator__"

# Name given to a method's return type variable when generating it


def _RETURN_TYPE(x): return f"__struct_{x}_return_type__"


class _MissingType:

    def __new__(cls) -> _tx.Self:
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


MISSING = _MissingType()
MaybeMissing = _tx.Union[T, _MissingType]


class _RequiredType:

    def __new__(cls) -> _tx.Self:
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<REQUIRED>"

    def __bool__(self) -> bool:
        return True


REQUIRED = _RequiredType()


class _HasFactory:

    def __init__(self, factory: callable) -> None:
        self.factory = factory

    def __repr__(self) -> str:
        return '<factory>'

    def __call__(self):
        return self.factory()


class SHOW_ATTR:

    def __init__(
        self,
        key: _tx.Optional[str] = None,
        hide_if_none: bool = False
    ) -> None:
        self.key = key
        self.hide_if_none = hide_if_none

    def __call__(self, value: _tx.Any) -> bool:
        if self.key is False:
            return False
        if self.hide_if_none and value is None:
            return False
        return True

    def __bool__(self) -> bool:
        return self.key is not False

    def __str__(self) -> str:
        return str(self.key)

    def __repr__(self) -> str:
        if self.key is False:
            return "False"
        if self.key is True and self.hide_if_none:
            return "<if not None>"
        if self.hide_if_none:
            return f"{self.key!r} <if not None>"
        return f"{self.key!r}"


class HIDE_IF_NONE(SHOW_ATTR):

    def __init__(self, key: _tx.Optional[str] = None) -> None:
        super().__init__(key=key, hide_if_none=True)
