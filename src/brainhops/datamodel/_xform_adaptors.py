# TODO/WIP: not working at all yet

# internals
from .systems import CoordinateSystem
from .transformations import _adaptor, _same_axis_type
from .transformations import Sequence, Transformation, Identity, Permutation, Scaling


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
def _(inp: Transformation, out: Transformation) -> Transformation:

    # if one of the two transformations has no axes than assume they are identity
    # if that is the case no adaptor is needed. return Identity
    if inp.ndims.output == 0 or out.ndims.input == 0:
        return Identity(input=inp.output, output=out.input)

    # The transformations that have been passed in should have the same dims assuming they are not identity
    if inp.ndims.output != out.ndims.input:
        raise NotImplementedError()

    # If the number of dims match but one or more transformation has unspecified axes assume that they already match
    if inp.output is None or out.input is None or inp.output == out.input:
        return Identity(input=inp.output, output=out.input)

    # Up to two transformations may be needed a permutation and a scaller
    transformations = []

    # Find what axes corrispond to output axes.
    # `perm[i] == j` means `inp.output.axes[i]` matches `out.input.axes[j]`.
    perm = []
    for i in range(len(inp.output.axes)):
        matches = []
        for j in range(len(out.input.axes)):
            if _same_axis_type(inp.output.axes[i], out.input.axes[j]):
                matches.append(j)
        if len(matches) == 0:
            raise ValueError(
                f"no matching axis found in `out.input` for axis "
                f"{inp.output.axes[i]!r} (index {i}) of `inp.output`."
            )
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous axis match: axis {inp.output.axes[i]!r} "
                f"(index {i}) of `inp.output` matches more than one axis "
                f"in `out.input` ({matches!r})."
            )
        perm.append(matches[0])

    # Check to see if a permutation transformation is needed
    permNeeded = False
    for i in range(len(perm)):
        if perm[i] != i:
            permNeeded = True

    # If we do need a permutation create one and make an intermediate output axis
    intermediate_output = inp.output
    if permNeeded:
        inverse_perm = [0] * len(perm)
        for i, j in enumerate(perm):
            inverse_perm[j] = i

        intermediate_output = CoordinateSystem(
            axes=[inp.output.axes[i] for i in inverse_perm])
        transformations.append(Permutation(
            permutation=perm, input=inp.output, output=intermediate_output))

    # Check to see if the axes use different units
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
            # because axes are already matched. If two axes don't share the same name assume they are facing opposite directions
            # TODO: this is hacky I am open to suggestions on how to better detect opposite directions
            scales.append((1 if inter.name == outer.name else -1) * scale)

    # Check to see if we need to make a scale transformation
    scalesNeeded = False
    for i in scales:
        if i != 1:
            scalesNeeded = True

    if scalesNeeded:
        transformations.append(
            Scaling(scale=scales, input=intermediate_output, output=out.input))

    # If permutation and scaling were not needed return Identity
    if len(transformations) == 0:
        return Identity(input=inp.output, output=out.input)

    # If only 1 transformation was needed just return that transformation,
    # but make sure its declared `output` is exactly `out.input` (rather
    # than the intermediate coordinate system), so callers can rely on
    # the adaptor's output always matching `out.input`.
    if len(transformations) == 1:
        return transformations[0].to(output=out.input)

    # If both permutation and scaling were needed return a sequence
    return Sequence(transformations=transformations, input=inp.output, output=out.input)
