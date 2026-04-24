import typing_extensions as _tx
from collections.abc import Mapping

from brainhops.struct import Struct


class SpecializedStruct(
    Struct,
    convert=False,
    mapping=True,
    default_factory=True,
):
    """
    We use this to set options that we want to propagate to all classes 
    in the hierarchy.
    """

    def __pre_init__(
        self, *args, **kwargs
    ) -> _tx.Tuple[_tx.Tuple, _tx.Mapping]:
        if (
            args and
            isinstance(args[0], Mapping) and
            all(key in self.keys() for key in args[0].keys())
        ):
            # Valid mapping-like object
            obj, *args = args
            for key, value in obj.items():
                kwargs.setdefault(key, value)
        return args, kwargs
