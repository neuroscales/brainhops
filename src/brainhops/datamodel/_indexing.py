"""Affines that describe the sub-arrays that an index selects."""

import numbers

import numpy as np
import typing_extensions as tx

from brainhops._core._internal_api import CartesianIndex, Shape

from ._sugar import get_axes
from .axes import Axis, ChannelAxis, is_component_type
from .systems import (
    AxisSequence,
    CoordinateSystem,
)
from .transformations import (
    Affine,
)


def index2transform(
    index: CartesianIndex,
    shape: Shape,
    system: tx.Optional[CoordinateSystem] = None,
) -> tx.Tuple[Affine, Shape]:
    """Convert an array index into an affine from sub-array to array.

    Parameters
    ----------
    index : tuple of int or slice or None or ellipsis
        Tuple of integers (NumPy integers included), slices, `None` and
        at most one ellipsis, which is appended when missing. Booleans
        are rejected, because NumPy reads them as masks and not as
        positions.
    shape : tuple of int
        Shape of the original array.
    system : CoordinateSystem, optional
        Coordinate system of the original array.

    Returns
    -------
    affine : Affine
        Transformation from the indexed array to the original array.
    shape : tuple of int
        Shape of the indexed array.

    Raises
    ------
    ValueError
        If an index element is not an integer, a slice, `None` or an
        ellipsis, or if the index contains more than one ellipsis.
    """
    # Validate every element before anything else, so that an invalid
    # element raises an error whether or not the axes of `system` are
    # known. Integers are converted to built-in integers, which the code
    # below expects.
    if not isinstance(index, tuple):
        index = (index,)
    checked_index = []
    for idx in index:
        if isinstance(idx, numbers.Integral) and not isinstance(idx, bool):
            checked_index.append(int(idx))
        elif idx is None or idx is ... or isinstance(idx, slice):
            checked_index.append(idx)
        else:
            raise ValueError(f"Invalid index: {idx}")
    index = tuple(checked_index)
    if sum(1 for idx in index if idx is ...) > 1:
        raise ValueError(f"Invalid index: {index} has more than one ellipsis")

    if ... not in index:
        index = (*index, ...)
    index_ellipsis = index.index(...)
    nb_indexed_dims = sum(1 for idx in index if idx not in (None, ...))
    nb_implicit_dims = len(shape) - nb_indexed_dims
    fill = (slice(None),) * nb_implicit_dims
    index = index[:index_ellipsis] + fill + index[(index_ellipsis + 1) :]

    nb_output_dims = sum(1 for idx in index if not isinstance(idx, int))
    nb_input_dims = len(shape)

    input_axes: tx.Optional[AxisSequence] = get_axes(system)
    if input_axes == [...]:
        input_axes = None
    elif input_axes.is_variadic:
        input_axes = input_axes.expand(nb_input_dims)
    output_axes = None
    if input_axes:
        output_axes = []
        axis_walker = list(input_axes)
        dim = 0
        for idx in index:
            if isinstance(idx, int):
                axis_walker.pop(0)
                dim += 1
            elif isinstance(idx, slice):
                axis = axis_walker.pop(0)
                # The entries of a component axis are the components of one
                # object only when the axis keeps all of them, in order.
                # Otherwise they are independent values, as on a channel
                # axis.
                kept = range(*idx.indices(shape[dim]))
                if is_component_type(getattr(axis, "type", None)) and (
                    kept != range(shape[dim])
                ):
                    axis = ChannelAxis(
                        name=axis.name,
                        unit=axis.unit,
                        discrete=axis.discrete,
                    )
                output_axes.append(axis)
                dim += 1
            else:
                # The element is `None`, which inserts a new axis.
                output_axes.append(Axis())

    # Start from zeros and set every entry explicitly. An identity seed
    # would leave a spurious 1 in the row of a dropped (integer) axis and
    # in the column of an inserted (None) axis.
    # The matrix is a small parameter, so it is built with NumPy whatever
    # the backend setting.
    affine = np.zeros((nb_input_dims, nb_output_dims + 1))

    input_dim_walker = 0
    output_dim_walker = 0
    output_shape = []
    for idx in index:
        if idx is None:
            output_dim_walker += 1
            output_shape.append(1)

        elif isinstance(idx, int):
            affine[input_dim_walker, -1] = idx
            input_dim_walker += 1

        elif isinstance(idx, slice):
            shp = shape[input_dim_walker]
            step = idx.step or 1
            if idx.start is None:
                start = 0 if step > 0 else (shp - 1)
            else:
                start = (shp + idx.start) if idx.start < 0 else idx.start
            if idx.stop is None:
                stop = shp if step > 0 else -1
            else:
                stop = (shp + idx.stop) if idx.stop < 0 else idx.stop

            # Round the length up, with a bias that depends on the sign of
            # the step, to match `len(range(start, stop, step))`.
            if step > 0:
                stop = min(stop, shp)
                oshp = max(0, (stop - start + (step - 1)) // step)
            else:
                stop = max(stop, -1)
                oshp = max(0, (stop - start + (step + 1)) // step)
            output_shape.append(oshp)

            affine[input_dim_walker, output_dim_walker] = step
            affine[input_dim_walker, -1] = start

            input_dim_walker += 1
            output_dim_walker += 1

    # The input system is that of the indexed array and the output system
    # is that of the original array.
    return Affine(
        matrix=affine,
        input=CoordinateSystem(axes=output_axes) if output_axes else None,
        output=system,
    ), tuple(output_shape)
