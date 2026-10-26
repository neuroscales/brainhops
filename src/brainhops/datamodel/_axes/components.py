"""Axes whose entries are the components of points, vectors or tensors.

Most axes of an image enumerate independent values, such as the colours
of an RGB image. Some axes hold instead the components of a geometric
object, such as a displacement, the gradient of an image or a diffusion
tensor. These axes are called component axes, and their type says what
the object is:

| Class                  | `type`           | Action of a linear map `J`   |
| ---------------------- | ---------------- | ---------------------------- |
| [`PointAxis`][]        | `"point"`        | none (see below)             |
| [`CoordinateAxis`][]   | `"coordinate"`   | none, as for `"point"`       |
| [`VectorAxis`][]       | `"vector"`       | `v ↦ J v`, or `g ↦ J⁻ᵀ g`    |
| [`DisplacementAxis`][] | `"displacement"` | `v ↦ J v`                    |
| [`TensorAxis`][]       | `"tensor"`       | `J` or `J⁻ᵀ` on each index   |

The variance of a vector or a tensor says, for each of its indices,
whether the index is multiplied by `J` (a `"contravariant"` index) or by
`J⁻ᵀ` (a `"covariant"` index).

When an image is moved or resampled, the components of vectors and
tensors must be re-expressed with the Jacobian of the transformation.
The components are expressed in a coordinate system, called the frame
of the values, which the image records in
[`SingleScaleImage.frames`][brainhops.datamodel.images.SingleScaleImage.frames].
The entries of a vector axis follow axes of the system of that frame, in
order, and
[`SingleScaleImage.reframe`][brainhops.datamodel.images.SingleScaleImage.reframe]
changes the frame. The coordinates of a point change with the map itself
rather than with its Jacobian, so they are not re-expressed.
"""

# The design is recorded in spec/design/vector-and-tensor-images.md.

__all__ = [
    "COMPONENT_TYPES",
    "is_component_type",
    "ComponentAxis",
    "PointAxis",
    "CoordinateAxis",
    "VectorAxis",
    "DisplacementAxis",
    "TensorAxis",
]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr

# brainhops
from brainhops._core.properties import smartproperty
from brainhops._core.typing import npmatrix

# internals
from ..enums import AxisType, IndexVariance, TensorLayout
from .base import Axis

Entries: tx.TypeAlias = tx.Tuple[tx.Tuple[int, ...], ...]
"""For each entry of a tensor axis, the frame axis of each index."""

COMPONENT_TYPES: tx.FrozenSet[str] = frozenset(
    {
        AxisType.point,
        AxisType.coordinate,
        AxisType.vector,
        AxisType.displacement,
        AxisType.tensor,
    }
)
"""The types of the axes whose entries are the components of an object."""


def is_component_type(value: tx.Any) -> bool:
    """Return whether an axis type describes a component axis.

    The readers and the writers of files group these axes with the
    channels, and only these axes can have a frame.

    Parameters
    ----------
    value : AxisType or str or None
        The type of an axis, such as `"vector"`. Any other value, such as
        `None`, gives `False`.

    Returns
    -------
    bool
        Whether the entries along an axis of this type are the components
        of a point, a vector or a tensor.
    """
    return isinstance(value, str) and value in COMPONENT_TYPES


def is_component_axis(axis: Axis) -> bool:
    """
    Return whether an axis holds the components of a point, vector or tensor.

    Parameters
    ----------
    axis : Axis
        The axis to test.

    Returns
    -------
    bool
        Whether the entries along the axis are the components of a point,
        a vector or a tensor.
    """
    return is_component_type(axis.type)


class ComponentAxis(
    Axis, on={"type": {str(t.value) for t in COMPONENT_TYPES}}
):
    """Axis whose entries are the components of a point, vector or tensor.

    This class is the common base of the classes of the component axes,
    and calling `Axis` with one of their types builds one of them. It
    says how many axes of the frame the entries follow, and how the
    entries change when the frame changes by a linear map.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"point", "coordinate", "vector" "displacement", "tensor"}, optional
        Type of the axis, such as `"vector"`.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    """

    orientation: NoRepr[None] = None

    @property
    def variances(self) -> tx.Optional[tx.Tuple[IndexVariance, ...]]:
        """The variance of each index of the object, or `None` for a point.

        * In tensors, `variance == variances`
        * In vectors, `variances = (variance,)`
        * In points, `variances = None`.
        """
        variance = getattr(self, "variance", None)
        if isinstance(variance, str):
            return (variance,)
        return variance

    @property
    def rank(self) -> int:
        """The number of indices of the tensor, the length of `variance`."""
        variances = self.variances
        if variances is None:
            # A point has rank 1 but no variance
            return 1
        return len(self.variances)


class PointAxis(ComponentAxis, on={"type": {"point", "coordinate"}}):
    """Axis whose entries are the coordinates of points.

    An image of points holds, at each voxel, the position of a point in
    some space, such as the position that a voxel maps to. Moving these
    points into another frame needs the map itself rather than its
    Jacobian, so
    `apply_linear_map`
    and
    [`SingleScaleImage.reframe`][brainhops.datamodel.images.SingleScaleImage.reframe]
    raise `NotImplementedError` on such an axis. The axis can still have
    a frame, which records the space of the points.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"point"}, default="point"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    """

    type: NoRepr[...] = "point"


class CoordinateAxis(PointAxis, on={"type": "coordinate"}):
    """Axis that enumerates the coordinates of the points of a field.

    A coordinate field of OME-NGFF has its spatial axes and one
    coordinate axis. The axis behaves as a [`PointAxis`][], and its type
    is kept so that the fields of OME-NGFF keep their type.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"coordinate"}, default="coordinate"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    """

    type: NoRepr[...]


class VectorAxis(ComponentAxis, on={"type": {"vector", "displacement"}}):
    """Axis whose entries are the components of vectors.

    A velocity or a fibre direction is a `"contravariant"` vector.

    When the frame changes by a linear map `J`, its components become
    `J v`. The gradient of an image and the normal of a surface are
    `"covariant"`, which are also called covectors. Their components
    become `J⁻ᵀ g`, so that the pairing of a covector with a vector
    does not change.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"vector"}, default="vector"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    variance : IndexVariance | str, default="contravariant"
        Whether the components are multiplied by `J` or by `J⁻ᵀ` when
        the frame changes by `J`.
    """

    type: NoRepr[...] = "vector"

    variance: IndexVariance = IndexVariance.vector
    """Whether the components change with `J` or with `J⁻ᵀ`."""

    @property
    def variances(self) -> tx.Tuple[IndexVariance, ...]:
        """The variance of the single index of the vector, its `variance`."""
        return (self.variance,)


class DisplacementAxis(
    VectorAxis,
    on={"type": "displacement", "variance": "contravariant"},
):
    """Axis that enumerates the components of displacement vectors.

    A displacement field has its spatial axes and one displacement axis.
    The axis behaves as a `"contravariant"` [`VectorAxis`][], which it
    has no field to change, and its type is kept so that the fields of
    OME-NGFF keep their type.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"displacement"}, default="displacement"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    variance : {"contravariant"} | str, default="contravariant"
        The pinned variance.
    """

    type: NoRepr[...]
    variance: NoRepr[...]


class TensorAxis(ComponentAxis, on={"type": "tensor"}, eq="_eq_fields"):
    """Axis whose entries are the components of tensors.

    A diffusion tensor, the Hessian of an image and the Jacobian of a
    map are tensors with two indices. Each index has a variance, either
    `"contravariant"` or `"covariant"`, and `variance` lists them, so
    that the number of indices is the rank `k` of the tensor. When a
    linear map `J` is applied to the underlying vector space, each
    contravariant index is multiplied by `J` and each covariant index
    by `J⁻ᵀ`:

    | `variance`             | Example          | New components |
    | ---------------------- | ---------------- | -------------- |
    | `("contra", "contra")` | diffusion tensor | `J T Jᵀ`       |
    | `("co", "co")`         | Hessian          | `J⁻ᵀ H J⁻¹`    |
    | `("contra", "co")`     | linear map       | `J A J⁻¹`      |

    A tensor whose two indices belong to different frames, such as the
    Jacobian of a warp between two spaces, is stored with two vector
    axes, one `"contravariant"` and one `"covariant"`, so that each
    axis has its own frame.

    `layout` says which components the axis stores, and in which order,
    for a frame of `n` axes. It is either a name, a member of
    [`TensorLayout`][] or its value, or an explicit tuple that gives,
    for each entry, the frame axis of each index. The names are listed
    below with their order for a tensor of rank 2 in a frame of three
    axes, where `ij` stands for the entry `(i, j)`:

    | `layout`                     | Order for `k = 2` and `n = 3` |
    | ---------------------------- | ----------------------------- |
    | `"C"`                        | `00 01 02 10 11 12 20 21 22`  |
    | `"F"`                        | `00 10 20 01 11 21 02 12 22`  |
    | `"symmetric"`                | `00 01 02 11 12 22`           |
    | `"symmetric-lower"`          | `00 10 11 20 21 22`           |
    | `"symmetric-diagonal-first"` | `00 11 22 01 02 12`           |
    | `"diagonal"`                 | `00 11 22`                    |

    `"C"` lists all the components with the last index varying fastest,
    and `"F"` with the first index varying fastest. `"symmetric"` lists
    the indices that do not decrease in lexicographic order, which is the
    upper triangle row by row for a matrix. `"symmetric-lower"` lists the
    indices that do not increase in lexicographic order, which is the
    lower triangle row by row. `"symmetric-diagonal-first"` lists the
    diagonal and then the upper triangle row by row, and it is only
    defined for tensors of rank 2. `"diagonal"` stores the components
    whose indices are all equal, and the other components are zero. The
    full layouts store `n**k` entries, the symmetric layouts store
    `C(n + k - 1, k)` entries, and the diagonal layout stores `n`.

    Software stores a symmetric matrix in these layouts:

    | Software               | `layout`                     |
    | ---------------------- | ---------------------------- |
    | FSL, ITK, NRRD, Camino | `"symmetric"`                |
    | MRtrix                 | `"symmetric-diagonal-first"` |
    | DIPY                   | `"symmetric-lower"`          |

    An explicit tuple lists each stored component once, and each listed
    component is stored independently, so that a full tensor can be
    listed in any order. A component that the tuple does not list takes
    the value of the listed component whose indices are a permutation of
    its own, if there is one, and it is zero otherwise.

    `None`, the default, reads the layout from the number of entries
    once the size `n` of the frame is known: `n**k` entries are `"C"`
    and `C(n + k - 1, k)` entries are `"symmetric"`. These two numbers
    differ when `n` and `k` are both at least 2, and any other number of
    entries is refused.

    A layout with components that are zero by construction, such as
    `"diagonal"` or an explicit tuple that leaves components out without
    a permutation to take their value from, cannot hold the tensor after
    a general change of frame. `apply_linear_map` then returns the tensor
    in the smallest layout that can, which is `"symmetric"` when the
    stored tensor is symmetric with indices of the same variance and `"C"`
    otherwise, and `_reframed_axis` gives the axis that describes it. The
    length of the axis changes.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"tensor"}, default="tensor"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    variance : tuple of (IndexVariance | str), default="contravariant"
        The variance of each index of the tensor, one per index.
    layout : TensorLayout | str | (tuple of tuple of int), optional
        Which components the axis stores and in which order, as a name
        of the table above or as an explicit tuple. By default, the
        layout is read from the number of entries.

    Raises
    ------
    ValueError
        If the tensor has fewer than two indices, if the layout is an
        unknown name or a name that is not defined for the rank of the
        tensor, or if an explicit layout lists a component twice or has
        entries of the wrong length.
    """

    type: NoRepr[...]

    _rank: tx.Optional[int] = None
    """
    The rank of the tensor.
    If not provided, guessed from `variance` or `layout`.
    """

    variance: tx.Tuple[IndexVariance, ...] = (IndexVariance.contravariant,)
    """The variance of each index of the tensor, one per index."""

    # An array cannot take part in the generated comparison, which needs a
    # single boolean per field, so `__eq__` compares the layout itself.
    layout: NoEq[tx.Optional[tx.Union[TensorLayout, npmatrix[int]]]] = None
    """Which components the axis stores, and in which order."""

    @smartproperty
    def rank(self) -> int:
        """The number of indices of the tensor.

        An explicit layout gives one frame axis per index. Otherwise, the
        rank is the length of `variance`, and a single variance, which
        applies to every index, gives a tensor of rank 2.
        """
        if self._rank is not None:
            return self._rank
        layout = self.layout
        if layout is not None and not isinstance(layout, str):
            return layout.shape[-1]
        if len(self.variance) == 1:
            return 2
        return len(self.variance)

    def __post_init__(self) -> None:
        # If variance was set as a scalar value, expand to rank
        rank = self.rank
        if len(self.variance) == 1 and rank > 1:
            self.variance = tuple(self.variance) * rank

        # Check minimum rank
        if rank < 2 or len(self.variance) < 2:
            raise ValueError(
                f"A tensor axis has at least two indices, not {rank}. A "
                f"tensor with one index is a vector."
            )

        layout = self.layout
        if isinstance(layout, str):
            # The converter of the field has already refused other names.
            if layout == "symmetric-diagonal-first" and rank != 2:
                raise ValueError(
                    f"The layout 'symmetric-diagonal-first' is defined for "
                    f"tensors of rank 2, not {rank}."
                )
            return
        if layout is None:
            return

        # explicit layout - must match rank
        if layout.shape[-1] != len(self.variance):
            raise ValueError(
                f"Each entry of the layout of a tensor of rank {rank} "
                f"gives {rank} frame axes, but the layout has shape "
                f"{layout.shape}."
            )

        # explicit layout - must be injective
        listed: tx.Set[tx.Tuple[int, ...]] = set()
        for entry in layout:
            entry = tuple(entry.tolist())
            if entry in listed:
                raise ValueError(
                    f"The layout lists the component {entry} twice."
                )
            listed.add(entry)

    def __eq__(self, other: tx.Any) -> bool:
        same = self._eq_fields(other)
        if same is not True:
            return same
        mine, theirs = self.layout, other.layout
        if isinstance(mine, str) or isinstance(theirs, str):
            return mine == theirs
        if mine is None or theirs is None:
            return mine is theirs
        return bool(np.array_equal(mine, theirs))
