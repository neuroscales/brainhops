# internals
from .systems import CoordinateSystem
from .transformations import Transformation, Identity, Permutation, Scaling, _adaptor


@_adaptor
def _(inp: CoordinateSystem, out: CoordinateSystem) -> Transformation:
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

    if inp == out:
        return Identity(input=inp, output=out)
    
    inames, iunits = zip(*[(axis.name, axis.unit) for axis in inp.axes])
    onames, ounits = zip(*[(axis.name, axis.unit) for axis in out.axes])

    if (len(inp.axes or []) == len(out.axes or [])):
        if (all(axis in out.axes for axis in inp.axes)):
            return Permutation(
                permutation=[out.axes.index(axis) for axis in inp.axes],
                input=inp,
                output=out
            )
        if all(iname in onames for iname in inames):
            perm = Permutation(
                permutation=[onames.index(iname) for iname in inames],
                input=inp,
                output=out
            )
            scale = Scaling()