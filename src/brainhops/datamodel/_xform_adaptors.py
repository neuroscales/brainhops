# TODO/WIP: not working at all yet

# internals
from .transformations import (
    Transformation,
    _adaptor,
)


@_adaptor
def _(inp: Transformation, out: Transformation) -> Transformation:

    raise NotImplementedError

    # TODO/WIP
    #
    #   This function should return a transform that "adapts" between
    #   two coordinate systems. It will be an educated guess that will
    #   make sense most of the time (hopefully). For example:
    #  - If the two coordinate systems have the same axes, but in different
    #    order, then a Permutation transform will be returned.
    #  - If the two coordinate systems have the same axes, but with different
    #    units, then a Scale transform will be returned.
    #  - If the axes have different names, but the same types and/or
    #    orientations, then they will be matched (and adapted if needed).
    #  - If only a subset of axes match, then the missing axes will be added
    #    and a "per-dimension" transform will be returned.
    #  - etc.
