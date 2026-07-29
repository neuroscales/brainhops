# TODO/WIP: not working at all yet 

# internals
from brainhops.datamodel.axes import Axis

from .systems import CoordinateSystem
from .orientation import Orientation
from .transformations import Sequence, Transformation, Identity, Permutation, Scaling
from .transformations import _adaptor, is_identity
from brainhops.datamodel import systems


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


def _same_axis_type(a1: Axis, a2: Axis):
    if a1.type == a2.type:
        if a1.type != "spatial":
            return True
        return set(a1.name.split("-")) == set(a2.name.split("-"))
    return False


@_adaptor
def _(inp: Transformation, out: Transformation) -> Transformation:

    if inp.output == out.input:
        return Identity(input=inp.output, output=out.input)

    if len(inp.output.axes) != len(out.input.axes):
        raise NotImplementedError

    transformations = []

    perm = []
    for i in range(len(inp.output.axes)):
        for j in range(len(out.input.axes)):
            if _same_axis_type(inp.output.axes[i], out.input.axes[j]):
                perm.append(j)
        if len(perm) != i + 1:
            raise NotImplementedError

    permNeeded = False
    for i in range(len(perm)):
        if perm[i] != i:
            permNeeded = True

    intermediate_output = inp.output

    if permNeeded:
        intermediate_output = systems.SpatialCoordinateSystem(
            [inp.output.axes[i] for i in perm])
        transformations.append(Permutation(
            permutation=perm, input=inp.output, output=intermediate_output))

    scales = []
    for i in range(len(intermediate_output.axes)):
        inter = intermediate_output.axes[i]
        outer = out.input.axes[i]
        scale = 1.0
        if inter.unit is not None and outer.unit is not None:
            scale = inter.unit.scale / outer.unit.scale
        if inter.type != "spatial":
            scales.append(scale)
        else:
            scales.append((1 if inter.name == outer.name else -1) * scale)

    scalesNeeded = False
    for i in scales:
        if i != 1:
            scalesNeeded = True

    if scalesNeeded:
        transformations.append(
            Scaling(scales=scales, input=intermediate_output, output=out.input))

    if len(transformations) == 0:
        return Identity(input=inp.output, output=out.input)

    if len(transformations) == 1:
        return transformations[0]

    return Sequence(transformations=transformations, input=inp.output, output=out.input)

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
