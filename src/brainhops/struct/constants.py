import typing_extensions as _tx

T = _tx.TypeVar("T")

# The name of an attribute on the class where we store the StructField
# objects.  Also used to check if a class is a @struct.
_FIELDS = '__struct_fields__'

# The name of an attribute on the class that stores the parameters to
# @struct.
_OPTIONS = '__struct_options__'

_POST_INIT_NAME = "__post_init__"

_DISCARD = "__struct_discard__"

class _MissingType:

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __repr__(self):
        return "<MISSING>"


MISSING = _MissingType()
MaybeMissing = _tx.Union[T, _MissingType]