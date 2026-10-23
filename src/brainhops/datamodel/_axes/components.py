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
| [`VectorAxis`][]       | `"vector"`       | `v ↦ J v`                    |
| [`DisplacementAxis`][] | `"displacement"` | `v ↦ J v`, as for `"vector"` |
| [`CovectorAxis`][]     | `"covector"`     | `g ↦ J⁻ᵀ g`                  |
| [`TensorAxis`][]       | `"tensor"`       | `J` or `J⁻ᵀ` on each index   |

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

The design is recorded in `spec/design/vector-and-tensor-images.md`.
"""

__all__ = [
    "COMPONENT_TYPES",
    "is_component_type",
    "ComponentAxis",
    "PointAxis",
    "CoordinateAxis",
    "VectorAxis",
    "DisplacementAxis",
    "CovectorAxis",
    "TensorAxis",
]

import itertools
import math

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops._core.typing import ArrayProtocol
from brainhops.backends import _is_dask, get_array_backend

from ..enums import AxisType
from .base import Axis

IndexKind: tx.TypeAlias = tx.Literal["vector", "covector"]
"""How one index of a tensor changes with the frame."""

Entries: tx.TypeAlias = tx.Tuple[tx.Tuple[int, ...], ...]
"""For each entry of a tensor axis, the frame axis of each index."""

COMPONENT_TYPES: tx.FrozenSet[str] = frozenset(
    {
        AxisType.point,
        AxisType.coordinate,
        AxisType.vector,
        AxisType.displacement,
        AxisType.covector,
        AxisType.tensor,
    }
)
"""The types of the axes whose entries are the components of an object."""

# The kind of each index of the object that an axis of this type holds.
# A point has no such kinds, because its coordinates are not changed by
# the Jacobian of a map. A tensor reads its kinds from its `indices`.
_INDEX_KINDS: tx.Dict[str, tx.Tuple[IndexKind, ...]] = {
    AxisType.vector: ("vector",),
    AxisType.displacement: ("vector",),
    AxisType.covector: ("covector",),
}


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


class ComponentAxis(Axis):
    """Axis whose entries are the components of a point, vector or tensor.

    This class is the common base of the classes of the component axes,
    and calling `Axis` with one of their types builds one of them. It
    says how many axes of the frame the entries follow, and how the
    entries change when the frame changes by a linear map.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : AxisType | str, optional
        Type of the axis, such as `"vector"`.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    @property
    def index_kinds(self) -> tx.Optional[tx.Tuple[IndexKind, ...]]:
        """The kind of each index of the object, or `None` for a point.

        A vector has one index of kind `"vector"`, a covector has one of
        kind `"covector"`, and a tensor has one per entry of its
        `indices`. A point has none, because its coordinates do not
        change with the Jacobian of a map.
        """
        return _INDEX_KINDS.get(self.type)

    def frame_ndim(self, size: int) -> int:
        """Return the number of axes of the frame that the entries follow.

        Each entry of a point, a vector or a covector follows one axis of
        the frame, so this number is the number of entries.

        Parameters
        ----------
        size : int
            The number of entries along the axis.

        Returns
        -------
        int
            The number of axes of the frame that the entries follow.
        """
        return size

    def apply(
        self, values: ArrayProtocol, matrix: ArrayProtocol, axis: int
    ) -> ArrayProtocol:
        """Apply a linear map of the frame to the entries along an axis.

        A vector index is multiplied by the map `J`, and a covector
        index by `J⁻ᵀ`, so that a vector becomes `J v` and a covector
        becomes `J⁻ᵀ g`.

        Parameters
        ----------
        values : ArrayProtocol
            The values of the image.
        matrix : ArrayProtocol
            The linear map `J`, of shape `(*batch, n, n)`, where `batch`
            broadcasts against the shape of `values` without `axis`.
        axis : int
            The axis of `values` whose entries are transformed.

        Returns
        -------
        ArrayProtocol
            The transformed values, with the shape of `values`.

        Raises
        ------
        NotImplementedError
            If the entries are the coordinates of points, which the
            Jacobian of a map does not re-express.
        """
        kinds = self.index_kinds
        if kinds is None:
            raise NotImplementedError(
                f"The entries of a {self.type!r} axis are the coordinates "
                f"of points, which change with the map itself rather than "
                f"with its Jacobian, so they cannot be re-expressed in "
                f"another frame."
            )
        if kinds == ("vector",):
            return _matvec(values, matrix, axis)
        return _matvec(values, _swap_last(_batched(_inv, matrix)), axis)


class PointAxis(ComponentAxis, on={"type": "point"}):
    """Axis whose entries are the coordinates of points.

    An image of points holds, at each voxel, the position of a point in
    some space, such as the position that a voxel maps to. Moving these
    points into another frame needs the map itself rather than its
    Jacobian, so [`apply`][brainhops.datamodel.axes.ComponentAxis.apply]
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
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class CoordinateAxis(ComponentAxis, on={"type": "coordinate"}):
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
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class VectorAxis(ComponentAxis, on={"type": "vector"}):
    """Axis whose entries are the components of vectors.

    A velocity or a fibre direction is a vector. When the frame changes
    by a linear map `J`, the components become `J v`.

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
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class DisplacementAxis(ComponentAxis, on={"type": "displacement"}):
    """Axis that enumerates the components of displacement vectors.

    A displacement field has its spatial axes and one displacement axis.
    The axis behaves as a [`VectorAxis`][], and its type is kept so that
    the fields of OME-NGFF keep their type.

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
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class CovectorAxis(ComponentAxis, on={"type": "covector"}):
    """Axis whose entries are the components of covectors.

    The gradient of an image and the normal of a surface are covectors.
    When the frame changes by a linear map `J`, the components become
    `J⁻ᵀ g`, so that the pairing of a covector with a vector does not
    change.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"covector"}, default="covector"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class TensorAxis(ComponentAxis, on={"type": "tensor"}):
    """Axis whose entries are the components of tensors.

    A diffusion tensor, the Hessian of an image and the Jacobian of a
    map are tensors with two indices. Each index is either a vector
    index or a covector index, and `indices` lists their kinds, so that
    the number of indices is the rank of the tensor. When the frame
    changes by a linear map `J`, each vector index is multiplied by `J`
    and each covector index by `J⁻ᵀ`:

    | `indices`                  | Example          | New components |
    | -------------------------- | ---------------- | -------------- |
    | `("vector", "vector")`     | diffusion tensor | `J T Jᵀ`       |
    | `("covector", "covector")` | Hessian          | `J⁻ᵀ H J⁻¹`    |
    | `("vector", "covector")`   | linear map       | `J A J⁻¹`      |

    A tensor whose two indices belong to different frames, such as the
    Jacobian of a warp between two spaces, is stored with two axes, a
    vector axis and a covector axis, so that each axis has its own frame.

    With the `"full"` layout, the axis stores the `n**k` components of a
    tensor of rank `k` in a frame of `n` axes. With the `"symmetric"`
    layout, it stores only the `C(n + k - 1, k)` components of a
    symmetric tensor whose indices do not decrease, and all the indices
    must then be of the same kind. `order` gives, for each entry, the
    frame axis of each index. By default, the full layout lists the
    components in row-major order, and the symmetric layout lists the
    non-decreasing indices in lexicographic order. FSL and MRtrix both
    store a diffusion tensor with the symmetric layout, but in different
    orders, and the order of FSL is the default:

    | Software | `order` for a frame of three axes                  |
    | -------- | -------------------------------------------------- |
    | FSL      | `((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))` |
    | MRtrix   | `((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))` |

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"tensor"}, default="tensor"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    indices : tuple of {"vector", "covector"}, default=("vector", "vector")
        The kind of each index of the tensor, one per index.
    layout : {"full", "symmetric"}, default="full"
        Whether the axis stores all the components of the tensor, or
        only those of a symmetric tensor whose indices do not decrease.
    order : tuple of tuple of int, optional
        For each entry along the axis, the frame axis of each index of
        the tensor. By default, the components are listed in row-major
        order for the full layout, and in lexicographic order for the
        symmetric layout.

    Raises
    ------
    ValueError
        If the tensor has no index, if a symmetric tensor has indices of
        different kinds, or if an entry of `order` does not have one
        frame axis per index.
    """

    type: NoRepr[...]

    indices: tx.Tuple[IndexKind, ...] = ("vector", "vector")
    """The kind of each index of the tensor, one per index."""

    layout: tx.Literal["full", "symmetric"] = "full"
    """Whether all the components are stored, or those of a symmetric
    tensor whose indices do not decrease."""

    order: tx.Optional[Entries] = None
    """For each entry, the frame axis of each index of the tensor."""

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("A tensor axis needs at least one index.")
        if self.layout == "symmetric" and len(set(self.indices)) > 1:
            raise ValueError(
                f"A symmetric tensor has indices of a single kind, but "
                f"these indices are {self.indices}. Store the tensor with "
                f"the full layout instead."
            )
        rank = len(self.indices)
        for entry in self.order or ():
            if len(entry) != rank:
                raise ValueError(
                    f"Each entry of the order of a tensor of rank {rank} "
                    f"gives {rank} frame axes, but {entry} gives "
                    f"{len(entry)}."
                )

    @property
    def index_kinds(self) -> tx.Tuple[IndexKind, ...]:
        """The kind of each index of the tensor, as in `indices`."""
        return tuple(self.indices)

    def frame_ndim(self, size: int) -> int:
        """Return the number of axes `n` of the frame of the tensor.

        Parameters
        ----------
        size : int
            The number of entries along the axis, which is `n**k` for the
            full layout and `C(n + k - 1, k)` for the symmetric layout,
            where `k` is the rank of the tensor.

        Returns
        -------
        int
            The number of axes of the frame that the indices follow.

        Raises
        ------
        ValueError
            If no frame gives `size` entries.
        """
        rank = len(self.indices)
        ndim = 1
        while self._size(ndim) < size:
            ndim += 1
        if self._size(ndim) != size:
            raise ValueError(
                f"{size} entries cannot hold the components of a "
                f"{self.layout} tensor of rank {rank}."
            )
        return ndim

    def entries(self, ndim: int) -> Entries:
        """Return the frame axis of each index, for each entry.

        Parameters
        ----------
        ndim : int
            The number of axes of the frame.

        Returns
        -------
        tuple of tuple of int
            For each entry, the frame axis of each index of the tensor.
            The indices of a symmetric tensor are listed in
            non-decreasing order.

        Raises
        ------
        ValueError
            If `order` does not list each component of the tensor once.
        """
        rank = len(self.indices)
        if self.layout == "symmetric":
            expected = tuple(
                itertools.combinations_with_replacement(range(ndim), rank)
            )
        else:
            expected = tuple(itertools.product(range(ndim), repeat=rank))
        if self.order is None:
            return expected
        entries = tuple(tuple(int(i) for i in entry) for entry in self.order)
        if self.layout == "symmetric":
            entries = tuple(tuple(sorted(entry)) for entry in entries)
        if len(entries) != len(expected) or set(entries) != set(expected):
            raise ValueError(
                f"The order {entries} does not list each component of a "
                f"{self.layout} tensor of rank {rank} in a frame of {ndim} "
                f"axes once."
            )
        return entries

    def apply(
        self, values: ArrayProtocol, matrix: ArrayProtocol, axis: int
    ) -> ArrayProtocol:
        """Return the components of the tensor in the new frame.

        The entries are unpacked into a full tensor, each of its indices
        is multiplied by `J` or by `J⁻ᵀ` according to its kind, and the
        result is packed again in the same order.

        Parameters
        ----------
        values : ArrayProtocol
            The values of the image.
        matrix : ArrayProtocol
            The linear map `J`, of shape `(*batch, n, n)`, where `batch`
            broadcasts against the shape of `values` without `axis`.
        axis : int
            The axis of `values` that holds the components.

        Returns
        -------
        ArrayProtocol
            The transformed values, with the shape of `values`.

        Raises
        ------
        ValueError
            If the number of entries or `order` does not describe a
            tensor of this rank and layout.
        """
        xp = get_array_backend(values)
        rank = len(self.indices)
        ndim = self.frame_ndim(values.shape[axis])
        entries = self.entries(ndim)
        symmetric = self.layout == "symmetric"
        position = {entry: k for k, entry in enumerate(entries)}
        unpack = np.array(
            [
                position[tuple(sorted(multi)) if symmetric else multi]
                for multi in itertools.product(range(ndim), repeat=rank)
            ]
        )
        pack = np.array(
            [np.ravel_multi_index(entry, (ndim,) * rank) for entry in entries]
        )
        packed = xp.moveaxis(values, axis, -1)
        full = packed[..., unpack]
        full = full.reshape(tuple(packed.shape[:-1]) + (ndim,) * rank)
        forward = _like(matrix, values)
        backward = None
        if "covector" in self.indices:
            backward = _swap_last(_batched(_inv, forward))
        for index, kind in enumerate(self.indices):
            factor = forward if kind == "vector" else backward
            full = _contract(full, factor, index, rank)
        flat = full.reshape(tuple(full.shape[:-rank]) + (ndim**rank,))
        return xp.moveaxis(flat[..., pack], -1, axis)

    def _size(self, ndim: int) -> int:
        """Return the number of entries of the tensor in `ndim` axes."""
        rank = len(self.indices)
        if self.layout == "symmetric":
            return math.comb(ndim + rank - 1, rank)
        return ndim**rank


# --- helpers ----------------------------------------------------------


def _like(matrix: ArrayProtocol, values: ArrayProtocol) -> ArrayProtocol:
    """Return `matrix` in the backend of `values`, in a floating type."""
    xp = get_array_backend(values)
    dtype = np.result_type(values.dtype, np.float32)
    if isinstance(matrix, np.ndarray) and not _is_dask(values):
        return xp.asarray(matrix.astype(dtype))
    return matrix.astype(dtype)


def _matvec(
    values: ArrayProtocol, matrix: ArrayProtocol, axis: int
) -> ArrayProtocol:
    """Multiply each vector along `axis` by `matrix`."""
    xp = get_array_backend(values)
    vectors = xp.moveaxis(values, axis, -1)
    matrix = _like(matrix, values)
    out = xp.matmul(matrix, vectors[..., None])[..., 0]
    return xp.moveaxis(out, -1, axis)


def _contract(
    tensor: ArrayProtocol, matrix: ArrayProtocol, index: int, rank: int
) -> ArrayProtocol:
    """Multiply one index of a stack of tensors by a stack of matrices.

    The last `rank` axes of `tensor` are its indices, and the batch of
    `matrix` broadcasts against the other axes.
    """
    xp = get_array_backend(tensor)
    position = tensor.ndim - rank + index
    moved = xp.moveaxis(tensor, position, -1)
    if matrix.ndim > 2:
        # The other indices of the tensor sit between the batch and the
        # index that is multiplied, so the matrix gets an axis of one entry
        # for each of them.
        shape = tuple(matrix.shape[:-2]) + (1,) * (rank - 1)
        matrix = matrix.reshape(shape + tuple(matrix.shape[-2:]))
    out = xp.matmul(matrix, moved[..., None])[..., 0]
    return xp.moveaxis(out, -1, position)


def _swap_last(matrix: ArrayProtocol) -> ArrayProtocol:
    """Transpose the last two axes of a stack of matrices."""
    xp = get_array_backend(matrix)
    return xp.swapaxes(matrix, -1, -2)


def _batched(
    function: tx.Callable[[ArrayProtocol], ArrayProtocol],
    matrix: ArrayProtocol,
) -> ArrayProtocol:
    """Apply a function of stacks of matrices to an array of any backend.

    The function receives a NumPy or CuPy array. Dask has no linear
    algebra on stacks of matrices, so the function is applied to each
    block, with the last two axes in a single block.
    """
    if _is_dask(matrix):
        whole = matrix.rechunk({matrix.ndim - 2: -1, matrix.ndim - 1: -1})
        return whole.map_blocks(function, dtype=matrix.dtype)
    return function(matrix)


def _inv(matrix: ArrayProtocol) -> ArrayProtocol:
    """Invert a stack of matrices."""
    return get_array_backend(matrix).linalg.inv(matrix)
