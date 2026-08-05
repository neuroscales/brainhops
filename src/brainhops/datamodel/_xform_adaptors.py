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
def _(x1: Transformation, x2: Transformation) -> Transformation:

    # if one of the two transformations has no axes than assume they are identity
    # if that is the case no adaptor is needed. return Identity
    if x2.ndims.output == 0 or x1.ndims.input == 0:
        return Identity(input=x2.output, output=x1.input)

    # The transformations that have been passed in should have the same dims assuming they are not identity
    if x2.ndims.output != x1.ndims.input:
        raise NotImplementedError()

    # If the number of dims match but one or more transformation has unspecified axes assume that they already match
    if x2.output is None or x1.input is None or x2.output == x1.input:
        return Identity(input=x2.output, output=x1.input)

    # Up to two transformations may be needed a permutation and a scaller
    transformations = []

    # Find what axes corrispond to output axes.
    # `perm[i] == j` means `inp.output.axes[i]` matches `out.input.axes[j]`.
    perm = []
    for i in range(len(x2.output.axes)):
        matches = []
        for j in range(len(x1.input.axes)):
            if _same_axis_type(x2.output.axes[i], x1.input.axes[j]):
                matches.append(j)
        if len(matches) == 0:
            raise ValueError(
                f"no matching axis found in `out.input` for axis "
                f"{x2.output.axes[i]!r} (index {i}) of `inp.output`."
            )
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous axis match: axis {x2.output.axes[i]!r} "
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
    intermediate_output = x2.output
    if permNeeded:
        inverse_perm = [0] * len(perm)
        for i, j in enumerate(perm):
            inverse_perm[j] = i

        intermediate_output = CoordinateSystem(
            axes=[x2.output.axes[i] for i in inverse_perm])
        transformations.append(Permutation(
            permutation=perm, input=x2.output, output=intermediate_output))

    # Check to see if the axes use different units
    scales = []
    for i in range(len(intermediate_output.axes)):
        inter = intermediate_output.axes[i]
        outer = x1.input.axes[i]
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
            Scaling(scale=scales, input=intermediate_output, output=x1.input))

    # If permutation and scaling were not needed return Identity
    if len(transformations) == 0:
        return Identity(input=x2.output, output=x1.input)

    # If only 1 transformation was needed just return that transformation,
    if len(transformations) == 1:
        return transformations[0]

    # If both permutation and scaling were needed return a sequence
    return Sequence(transformations=transformations, input=x2.output, output=x1.input)
