"""The order in which a NRRD image presents the axes of its file.

The first NRRD axis varies fastest, and the axes of a file come in any
order. An image presents them grouped by their role: the spatial axes
first, then time, then channels, then the other axes, each group in file
order. The role of an axis is decided by the header alone, so the order
is a function of the record.
"""

# dependencies
import typing_extensions as tx

# this format
from ._raw import NrrdRaw

_CHANNEL_KINDS = frozenset(
    (
        "vector",
        "list",
        "point",
        "covariant-vector",
        "normal",
        "2-vector",
        "3-vector",
        "4-vector",
        "3-gradient",
        "3-normal",
        "rgb-color",
        "hsv-color",
        "xyz-color",
        "rgba-color",
        "complex",
        "quaternion",
        "2d-symmetric-matrix",
        "2d-masked-symmetric-matrix",
        "2d-matrix",
        "2d-masked-matrix",
        "3d-symmetric-matrix",
        "3d-masked-symmetric-matrix",
        "3d-matrix",
        "3d-masked-matrix",
    )
)
"""Kinds whose samples are the components of one value, read as channels."""

ROLES = ("space", "time", "channel", "other")
"""The roles of the axes, in the order in which an image presents them."""


def axis_roles(record: NrrdRaw) -> tx.List[str]:
    """Return the role of each NRRD axis, fastest first.

    An axis with a space direction is spatial. In a header without
    directions, an axis of kind `domain` or `space`, or one of the first
    three if there are no `kinds`, is spatial too. Other axes are time,
    channel or other according to their kind.
    """
    dirs = record.space_directions
    kinds = record.kinds
    has_kinds = "kinds" in record.fields
    roles = []
    for i, kind in enumerate(kinds):
        if dirs is not None and dirs[i] is not None:
            roles.append("space")
        elif kind == "time":
            roles.append("time")
        elif kind in _CHANNEL_KINDS:
            roles.append("channel")
        elif dirs is None and (
            kind in ("domain", "space") or (not has_kinds and i < 3)
        ):
            roles.append("space")
        else:
            roles.append("other")
    return roles


def image_order(record: NrrdRaw) -> tx.List[int]:
    """Return the file axis that each axis of the image presents.

    The roles are sorted in the order of [`ROLES`][], each in file order,
    so that element `j` of the result is the file axis that becomes axis
    `j` of the image.
    """
    roles = axis_roles(record)
    return [i for role in ROLES for i, r in enumerate(roles) if r == role]
