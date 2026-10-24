"""Conversion between OME-Zarr axes and brainhops axes.

OME-NGFF 0.6 gives each axis a name, a type, a unit and whether it is
discrete, and it lets the type be a custom string (ngff-spec tag
`0.6`, `index.md`, section "axes metadata"). Its schema does not forbid
other fields in an axis (`schemas/axes.schema`). The types `point`,
`vector` and `tensor` of brainhops are such custom types, and the
custom fields `valence` and `layout` describe a vector or a tensor
further, as the section "Vectors and tensors" of
[`brainhops.io.images.zarr`][] lists them. Each value is a string,
because abczarr keeps only the scalar values of the custom fields of
an axis, so the valences of a tensor are joined by commas, as in
`"vector,covector"`. An explicit layout, which is a tuple of
components, therefore has no encoding, and [`axis_losses`][] reports
it.
"""

from collections.abc import Mapping

import typing_extensions as tx

from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    PointAxis,
    SpaceAxis,
    TensorAxis,
    TimeAxis,
    VectorAxis,
)
from brainhops.datamodel.enums import TensorLayout, Valence
from brainhops.datamodel.units import Unit, is_indexunit

_AXIS_TYPES = {
    "space": SpaceAxis,
    "time": TimeAxis,
    "channel": ChannelAxis,
    "displacement": DisplacementAxis,
    "coordinate": CoordinateAxis,
    "point": PointAxis,
    "vector": VectorAxis,
    "tensor": TensorAxis,
}

# The valences of a tensor are joined by this separator in its `valence`
# field.
_VALENCE_SEPARATOR = ","

# OME-NGFF 0.6 requires the axis of a displacement or a coordinate field
# to be discrete (section "coordinates and displacements", "Array
# Structure").
_DISCRETE_TYPES = ("displacement", "coordinate")


def _to_axis(ome_axis: tx.Any) -> Axis:
    # The OME axis is a mapping or an object with attributes. A known OME
    # type selects the matching Axis subclass, which fixes the axis type.
    # An unknown type is stored in the `type` field of a plain Axis. A
    # vector or a tensor axis also reads its custom `valence` and
    # `layout`.
    if not isinstance(ome_axis, Mapping):
        ome_axis = {
            key: getattr(ome_axis, key, None)
            for key in (
                "type",
                "name",
                "unit",
                "discrete",
                "valence",
                "layout",
            )
        }
    type_ = ome_axis.get("type")
    cls = _AXIS_TYPES.get(type_, Axis)
    kwargs = {}
    for key in ("name", "unit", "discrete"):
        value = ome_axis.get(key)
        if value is not None:
            kwargs[key] = value
    if cls is Axis and type_ is not None:
        kwargs["type"] = type_
    if cls is VectorAxis or cls is TensorAxis:
        kwargs.update(_component_fields(cls, ome_axis))
    return cls(**kwargs)


def _component_fields(
    cls: type, ome_axis: tx.Mapping[str, tx.Any]
) -> tx.Dict[str, tx.Any]:
    """Return the valence and the layout that an OME axis states.

    Raises
    ------
    ValueError
        If the valence or the layout is not a string that names valid
        values.
    """
    fields: tx.Dict[str, tx.Any] = {}
    valence = ome_axis.get("valence")
    if valence is not None:
        if not isinstance(valence, str):
            raise ValueError(
                f"The valence of an OME-Zarr {cls.__name__} must be a "
                f"string, not {valence!r}."
            )
        names = valence.split(_VALENCE_SEPARATOR)
        try:
            valences = tuple(Valence(name.strip()) for name in names)
        except ValueError:
            raise ValueError(
                f"The valence {valence!r} of an OME-Zarr axis names "
                f"another valence than 'vector' and 'covector'."
            ) from None
        if cls is VectorAxis:
            if len(valences) != 1:
                raise ValueError(
                    f"A vector axis has one valence, not {valence!r}."
                )
            fields["valence"] = valences[0]
        else:
            fields["valence"] = valences
    layout = ome_axis.get("layout")
    if layout is not None and cls is TensorAxis:
        try:
            fields["layout"] = TensorLayout(layout)
        except ValueError:
            raise ValueError(
                f"The layout {layout!r} of an OME-Zarr tensor axis is not "
                f"the name of a layout."
            ) from None
    return fields


def axis_to_json(axis: Axis) -> tx.Dict[str, tx.Any]:
    """Return the OME-Zarr JSON description of one axis.

    The name, the type and the unit are written, and whether the axis is
    discrete when the axis says so. The axis of a displacement or a
    coordinate field is discrete unless it says otherwise, as OME-NGFF
    0.6 requires. A vector or a tensor axis also gets the custom fields
    `valence` and `layout` when they differ from the defaults, so that
    it reads back as it was written. An explicit layout cannot be
    written, and [`axis_losses`][] reports it.

    Parameters
    ----------
    axis : Axis
        The axis to describe.

    Returns
    -------
    dict
        The JSON description of the axis.
    """
    entry: tx.Dict[str, tx.Any] = {}
    name = getattr(axis, "name", None)
    type_ = getattr(axis, "type", None)
    type_ = getattr(type_, "value", type_)
    unit = getattr(axis, "unit", None)
    entry["name"] = name if isinstance(name, str) else ""
    if isinstance(type_, str):
        entry["type"] = type_
    if unit is not None:
        # A brainhops unit is not a string, and its text is the long
        # name of the unit, as OME-NGFF spells it, such as "millimeter".
        # The index of an array has no unit in OME-NGFF.
        value = getattr(unit, "value", unit)
        if isinstance(unit, Unit):
            value = None if is_indexunit(unit) else str(unit)
        if isinstance(value, str):
            entry["unit"] = value
    discrete = getattr(axis, "discrete", None)
    if discrete is None and type_ in _DISCRETE_TYPES:
        discrete = True
    if discrete is not None:
        entry["discrete"] = bool(discrete)
    if isinstance(axis, VectorAxis) and axis.valence != Valence.vector:
        entry["valence"] = Valence(axis.valence).value
    if isinstance(axis, TensorAxis):
        valences = tuple(Valence(v) for v in axis.valence)
        if valences != (Valence.vector, Valence.vector):
            entry["valence"] = _VALENCE_SEPARATOR.join(
                v.value for v in valences
            )
        if isinstance(axis.layout, str):
            entry["layout"] = TensorLayout(axis.layout).value
    return entry


def axis_losses(axes: tx.Sequence[Axis]) -> tx.List[str]:
    """Return what the OME-Zarr description of some axes leaves out.

    The custom fields of [`axis_to_json`][] describe every vector and
    tensor axis, apart from a tensor whose layout is an explicit tuple
    of components, which a scalar field cannot hold. Such an axis is
    written without a layout, and reads back with the layout that the
    number of its entries implies, so the values lose their meaning.

    Parameters
    ----------
    axes : sequence of Axis
        The axes to write.

    Returns
    -------
    list of str
        One clause per axis that loses information, which completes
        "writing this image loses information about its values: ...".
    """
    losses = []
    for index, axis in enumerate(axes):
        if isinstance(axis, TensorAxis) and isinstance(axis.layout, tuple):
            name = repr(axis.name or index)
            losses.append(
                f"OME-Zarr cannot store the explicit layout of the tensor "
                f"along axis {name}, which would read back with the layout "
                f"that the number of its entries implies"
            )
    return losses
