# TODO/WIP: not working at all yet 

# internals
from .systems import CoordinateSystem
from .orientation import Orientation
from .transformations import Transformation, Identity, Permutation, Scaling, 
from .transformations import _adaptor, is_identity


def _get_names(axes):
    return [axis.name for axis in axes]

def _get_units(axes):
    return [axis.unit for axis in axes]

def _get_orientations(axes):
    return [axis.orientation for axis in axes]

def _get_by_name(axes, name):
    for axis in axes:
        if axis.name == name:
            yield axis
    return None

def _get_by_unit(axes, unit):
    for axis in axes:
        if axis.unit == unit:
            yield axis
    return None

def _get_by_unit_type(axes, unit_type):
    for axis in axes:
        if axis.unit and axis.unit.type == unit_type:
            yield axis
    return None

def _get_by_orientation(axes, orientation):
    for axis in axes:
        if axis.orientation == orientation:
            yield axis
    return None

def _get_by_orientation_type(axes, orientation_type):
    for axis in axes:
        if axis.orientation and axis.orientation.type == orientation_type:
            yield axis
    return None


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
    if inp is None or out is None:
        return Identity(input=inp, output=out)

    iaxes = {axis.name: axis for axis in inp.axes or []}
    oaxes = {axis.name: axis for axis in out.axes or []}
    inames, iunits = zip(*[(axis.name, axis.unit) for axis in iaxes.values()])
    onames, ounits = zip(*[(axis.name, axis.unit) for axis in oaxes.values()])

    if (len(inp.axes or []) == len(out.axes or [])):

        if (all(axis in out.axes for axis in inp.axes)):

            return Permutation(
                permutation=[out.axes.index(axis) for axis in inp.axes],
                input=inp,
                output=out
            )
        
        if all(iname in onames for iname in inames):

            permuted_inp = CoordinateSystem(
                axes=[iaxes[iname] for iname in onames],
                name=f"permuted({inp.name})"
            )
            perm = Permutation(
                permutation=[onames.index(iname) for iname in inames],
                input=inp,
                output=permuted_inp
            )
            scale = []
            for oaxis in oaxes.values():
                iaxis = iaxes.get(oaxis.name)
                if oaxis.unit is None or iaxis.unit is None:
                    scale.append(1.0)
                else:
                    scale.append(
                        getattr(oaxis.unit, "scale", 1.0) / 
                        getattr(iaxis.unit, "scale", 1.0)
                    )
            scale = Scaling(
                scale=scale,
                input=permuted_inp,
                output=out
            )
            if is_identity(perm, compute=True):
                if is_identity(scale, compute=True):
                    return Identity(input=inp, output=out)
                return Scaling(scale=scale.scale, input=inp, output=out)
            elif is_identity(scale, compute=True):
                return Permutation(permutation=perm.permutation, input=inp, output=out)
            else:
                return Sequence(
                    transformations=[perm, scale],
                    input=inp,
                    output=out
                )

    axis_matching = {}
    matchable_axes = list(iaxes.values())
    for oaxis in oaxes.values():
        if oaxis in matchable_axes:
            axis_matching[oaxis] = iaxes[oaxis.name]
        elif oaxis.name and oaxis.name in _get_names(matchable_axes):
            iaxis = next(_get_by_name(matchable_axes, oaxis.name))
            axis_matching[oaxis] = iaxis
            matchable_axes.remove(iaxis)
        elif oaxis.orientation and oaxis.orientation in _get_orientations(matchable_axes):
            iaxis = next(_get_by_orientation(matchable_axes, oaxis.orientation))
            axis_matching[oaxis] = iaxis
            matchable_axes.remove(iaxis)
