import typing_extensions as _tx
from collections.abc import Mapping

from brainhops.struct import Struct


class SpecializedStruct(
    Struct, 
    convert=True,
    mapping="hide_none", 
    repr="hide_none",
):
    # We use this to set options that we want to propagate to all 
    # classes in the hierarchy.

    def __pre_init__(
        self, *args, **kwargs
    ) -> _tx.Tuple[_tx.Tuple, _tx.Mapping]:
        # Implements a copy constructor:
        #   If the first positional argument is an instance of the class,
        #   or a dictionary-like object with keys matching the fields of 
        #   the class, then we use it to populate the fields of the class.

        if (
            args and
            isinstance(args[0], type(self))
        ):
            # Valid instance of the class
            obj, *args = args
            for field in self.__struct_fields__.values():
                if field.init and not field.var:
                    kwargs.setdefault(field.name, getattr(obj, field.name))
            return args, kwargs
        
        valid_keys = (
            field.name
            for field in self.__struct_fields__.values()
            if field.init and field.kw
        )
        if (
            args and 
            isinstance(args[0], Mapping) and 
            all(key in valid_keys for key in args[0].keys())
        ):
            # Valid mapping-like object
            obj, *args = args
            for key, value in obj.items():
                kwargs.setdefault(key, value)
            return args, kwargs
        
        else:
            return args, kwargs