import typing_extensions as _tx
from collections.abc import Mapping

from brainhops.struct import Struct


class SpecializedStruct(
    Struct, 
    convert=True, 
    mapping=True, 
    # default_factory=True,
):
    """
    We use this to set options that we want to propagate to all classes 
    in the hierarchy.
    """

    def __new__(cls, *args, **kwargs) -> _tx.Self:
        if (
            len(args) == 1 and
            len(kwargs) == 0 and
            isinstance(args[0], cls)
        ):
            # Do not make a copy
            return args[0]
        return super().__new__(cls, *args, **kwargs)

    def __pre_init__(
        self, *args, **kwargs
    ) -> _tx.Tuple[_tx.Tuple, _tx.Mapping]:
        if (
            args and
            isinstance(args[0], type(self))
        ):
            # Valid instance of the class
            obj, *args = args
            for field in self.__struct_fields__.values():
                if field.init and not field.var:
                    kwargs.setdefault(field.name, getattr(obj, field.name))
        elif (
            args and 
            isinstance(args[0], Mapping) and 
            all(key in self.keys() for key in args[0].keys())
        ):
            # Valid mapping-like object
            obj, *args = args
            for key, value in obj.items():
                kwargs.setdefault(key, value)
        return args, kwargs