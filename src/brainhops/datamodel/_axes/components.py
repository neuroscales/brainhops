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

The valence of a vector or a tensor says, for each of its indices,
whether the index is multiplied by `J` (a `"vector"` index) or by `J⁻ᵀ`
(a `"covector"` index). The gradient of an image is a vector whose
valence is `"covector"`.

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

import itertools
import math

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr, replace

from brainhops._core.backends import array_backend, is_dask
from brainhops._core.typing import ArrayProtocol

from ..enums import AxisType, TensorLayout, Valence
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

# The valences of the object that an axis of this type holds, for the types
# whose valences are fixed. A point has no valence, because its coordinates
# are not changed by the Jacobian of a map. A vector and a tensor read their
# valences from their `valence` field.
_VALENCES: tx.Dict[str, tx.Tuple[Valence, ...]] = {
    AxisType.displacement: (Valence.vector,),
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
    def valences(self) -> tx.Optional[tx.Tuple[Valence, ...]]:
        """The valence of each index of the object, or `None` for a point.

        A vector has one index, whose valence is its `valence`, a
        displacement has one index of valence `"vector"`, and a tensor has
        one per entry of its `valence`. A point has no valence, because
        its coordinates do not change with the Jacobian of a map.
        """
        return _VALENCES.get(self.type)

    def frame_ndim(self, size: int, ndim: tx.Optional[int] = None) -> int:
        """Return the number of axes of the frame that the entries follow.

        Each entry of a point or a vector follows one axis of the frame,
        so this number is the number of entries.

        Parameters
        ----------
        size : int
            The number of entries along the axis.
        ndim : int, optional
            The number of axes of the frame, when it is known from
            elsewhere. It is checked against `size`.

        Returns
        -------
        int
            The number of axes of the frame that the entries follow.

        Raises
        ------
        ValueError
            If `ndim` is given and differs from `size`.
        """
        if ndim is not None and ndim != size:
            raise ValueError(
                f"{size} entries are not the components of a point or a "
                f"vector in a frame of {ndim} axes."
            )
        return size

    def apply(
        self, values: ArrayProtocol, matrix: ArrayProtocol, axis: int
    ) -> ArrayProtocol:
        """Apply a linear map of the frame to the entries along an axis.

        Each index of the object is multiplied by the map `J` when it is
        a vector index and by `J⁻ᵀ` when it is a covector index, so that
        a vector becomes `J v` and a vector of valence `"covector"`
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
        valences = self.valences
        if valences is None:
            raise NotImplementedError(
                f"The entries of a {self.type!r} axis are the coordinates "
                f"of points, which change with the map itself rather than "
                f"with its Jacobian, so they cannot be re-expressed in "
                f"another frame."
            )
        xp = array_backend(values)
        moved = xp.moveaxis(values, axis, -1)
        out = _multilinear(moved, matrix, valences, values)
        return xp.moveaxis(out, -1, axis)


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

    A velocity or a fibre direction is a vector of valence `"vector"`.
    When the frame changes by a linear map `J`, its components become
    `J v`. The gradient of an image and the normal of a surface are
    vectors of valence `"covector"`, which are also called covectors.
    Their components become `J⁻ᵀ g`, so that the pairing of a covector
    with a vector does not change.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"c"`.
    type : {"vector"}, default="vector"
        Pinned axis type.
    unit : Unit | str, optional
        Unit of the coordinates along the axis.
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    valence : Valence | str, default="vector"
        Whether the components are multiplied by `J` or by `J⁻ᵀ` when
        the frame changes by `J`.
    """

    type: NoRepr[...]

    valence: Valence = Valence.vector
    """Whether the components change with `J` or with `J⁻ᵀ`."""

    @property
    def valences(self) -> tx.Tuple[Valence, ...]:
        """The valence of the single index of the vector, its `valence`."""
        return (self.valence,)


class DisplacementAxis(ComponentAxis, on={"type": "displacement"}):
    """Axis that enumerates the components of displacement vectors.

    A displacement field has its spatial axes and one displacement axis.
    The axis behaves as a [`VectorAxis`][] of valence `"vector"`, which it
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
    orientation : Orientation | str, optional
        Meaningful direction of the axis, such as left to right.
    """

    type: NoRepr[...]


class TensorAxis(ComponentAxis, on={"type": "tensor"}):
    """Axis whose entries are the components of tensors.

    A diffusion tensor, the Hessian of an image and the Jacobian of a
    map are tensors with two indices. Each index has a valence, either
    `"vector"` or `"covector"`, and `valence` lists them, so that the
    number of indices is the rank `k` of the tensor. When the frame
    changes by a linear map `J`, each vector index is multiplied by `J`
    and each covector index by `J⁻ᵀ`:

    | `valence`                  | Example          | New components |
    | -------------------------- | ---------------- | -------------- |
    | `("vector", "vector")`     | diffusion tensor | `J T Jᵀ`       |
    | `("covector", "covector")` | Hessian          | `J⁻ᵀ H J⁻¹`    |
    | `("vector", "covector")`   | linear map       | `J A J⁻¹`      |

    A tensor whose two indices belong to different frames, such as the
    Jacobian of a warp between two spaces, is stored with two vector
    axes, one of valence `"vector"` and one of valence `"covector"`, so
    that each axis has its own frame.

    `layout` says which components the axis stores, and in which order,
    for a frame of `n` axes. It is either a name, a member of
    [`TensorLayout`][brainhops.datamodel.enums.TensorLayout] or its value,
    or an explicit tuple that gives, for each entry, the frame axis of
    each index. The names are listed below with their order for a tensor
    of rank 2 in a frame of three axes, where `ij` stands for the entry
    `(i, j)`:

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
    its own, if there is one, and it is zero otherwise. A tuple that
    leaves out a component with several listed permutations is refused,
    because their values may differ. Taking a value from a permutation
    makes the tensor symmetric, which needs all its indices to have the
    same valence, because the symmetry of a tensor whose indices have
    different valences depends on the frame. The symmetric layouts, and the
    tuples that take a value from a permutation, are therefore refused
    for a mixed valence.

    `None`, the default, reads the layout from the number of entries
    once the size `n` of the frame is known: `n**k` entries are `"C"`
    and `C(n + k - 1, k)` entries are `"symmetric"`. These two numbers
    differ when `n` and `k` are both at least 2, and any other number of
    entries is refused.

    A layout with components that are zero by construction, such as
    `"diagonal"` or an explicit tuple that leaves components out without
    a permutation to take their value from, cannot hold the tensor after
    a general change of frame. [`apply`][] then returns the tensor in the
    smallest layout that can, which is `"symmetric"` when the stored
    tensor is symmetric with indices of the same valence and `"C"`
    otherwise, and [`reframed`][] gives the axis that describes it. The
    length of the axis changes.

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
    valence : tuple of (Valence | str), default=("vector", "vector")
        The valence of each index of the tensor, one per index.
    layout : TensorLayout | str | tuple of tuple of int, optional
        Which components the axis stores and in which order, as a name
        of the table above or as an explicit tuple. By default, the
        layout is read from the number of entries.

    Raises
    ------
    ValueError
        If the tensor has fewer than two indices, if the layout is an
        unknown name or a name that is not defined for the rank of the
        tensor, if a symmetric layout has indices of different valences, or
        if an explicit layout lists a component twice, has entries of
        the wrong length, or fills a component from several listed
        permutations or from a permutation with a mixed valence.
    """

    type: NoRepr[...]

    valence: tx.Tuple[Valence, ...] = (Valence.vector, Valence.vector)
    """The valence of each index of the tensor, one per index."""

    layout: tx.Optional[tx.Union[TensorLayout, Entries]] = None
    """Which components the axis stores, and in which order."""

    def __post_init__(self) -> None:
        rank = len(self.valence)
        if rank < 2:
            raise ValueError(
                f"A tensor axis has at least two indices, not {rank}. A "
                f"tensor with one index is a vector."
            )
        mixed = len(set(self.valence)) > 1
        layout = self.layout
        if isinstance(layout, str):
            # The converter of the field has already refused other names.
            if layout == "symmetric-diagonal-first" and rank != 2:
                raise ValueError(
                    f"The layout 'symmetric-diagonal-first' is defined for "
                    f"tensors of rank 2, not {rank}."
                )
            if layout.startswith("symmetric") and mixed:
                raise ValueError(
                    f"A symmetric tensor has indices of the same valence, but "
                    f"its valence is {self.valence}. Use the layout 'C' "
                    f"instead."
                )
            return
        if layout is None:
            return
        listed: tx.Set[tx.Tuple[int, ...]] = set()
        for entry in layout:
            if len(entry) != rank:
                raise ValueError(
                    f"Each entry of the layout of a tensor of rank {rank} "
                    f"gives {rank} frame axes, but {entry} gives "
                    f"{len(entry)}."
                )
            if entry in listed:
                raise ValueError(
                    f"The layout lists the component {entry} twice."
                )
            listed.add(entry)
        # The components that the layout leaves out take the value of a
        # listed permutation of their indices, which must be unique and
        # needs indices of the same valence.
        for key, group in _permutation_groups(layout).items():
            missing = set(itertools.permutations(key)) - listed
            if not missing:
                continue
            if len(group) > 1:
                raise ValueError(
                    f"The layout leaves out {sorted(missing)[0]}, whose "
                    f"value could be taken from any of {group}."
                )
            if mixed:
                raise ValueError(
                    f"The layout takes the value of {sorted(missing)[0]} "
                    f"from {group[0]}, which needs indices of the same "
                    f"valence, but the valence is {self.valence}."
                )

    @property
    def valences(self) -> tx.Tuple[Valence, ...]:
        """The valence of each index of the tensor, as in `valence`."""
        return tuple(self.valence)

    def frame_ndim(self, size: int, ndim: tx.Optional[int] = None) -> int:
        """Return the number of axes `n` of the frame of the tensor.

        Parameters
        ----------
        size : int
            The number of entries along the axis.
        ndim : int, optional
            The number of axes of the frame, when it is known from
            elsewhere. It is checked against `size`. An explicit layout
            accepts any frame that has the axes that it names, and
            without `ndim` it gives the smallest such frame.

        Returns
        -------
        int
            The number of axes of the frame that the indices follow.

        Raises
        ------
        ValueError
            If no frame, or more than one, gives `size` entries.
        """
        rank = len(self.valence)
        layout = self.layout
        if isinstance(layout, tuple):
            # An explicit layout needs at least the axes that it names, and
            # it may leave the last axes of a larger frame out.
            least = 1 + max(max(entry) for entry in layout)
            found = set()
            if size == len(layout):
                found = {least}
                if ndim is not None and ndim >= least:
                    found.add(ndim)
        else:
            if layout is not None:
                names = [layout]
            elif len(set(self.valence)) > 1:
                names = ["C"]
            else:
                names = ["C", "symmetric"]
            found = set()
            for name in names:
                for n in range(1, size + 1):
                    if _layout_size(name, n, rank) == size:
                        found.add(n)
        if ndim is not None:
            if ndim not in found:
                raise ValueError(
                    f"{size} entries do not hold a tensor of rank {rank} "
                    f"in a frame of {ndim} axes with the layout "
                    f"{layout!r}."
                )
            return ndim
        if len(found) != 1:
            raise ValueError(
                f"{size} entries hold a tensor of rank {rank} with the "
                f"layout {layout!r} in frames of {sorted(found)} axes, "
                f"so the size of the frame is not known."
            )
        return found.pop()

    def resolved_layout(
        self, ndim: int, size: tx.Optional[int] = None
    ) -> tx.Union[str, Entries]:
        """Return the layout, read from the number of entries if needed.

        Parameters
        ----------
        ndim : int
            The number of axes of the frame.
        size : int, optional
            The number of entries along the axis, which reads a layout
            that is `None`.

        Returns
        -------
        str or tuple of tuple of int
            The layout, never `None`.

        Raises
        ------
        ValueError
            If the layout is `None` and `size` is neither `n**k` nor
            `C(n + k - 1, k)`.
        """
        if self.layout is not None:
            return self.layout
        rank = len(self.valence)
        for name in ("C", "symmetric"):
            if size is not None and _layout_size(name, ndim, rank) == size:
                if name == "symmetric" and len(set(self.valence)) > 1:
                    break
                return name
        raise ValueError(
            f"{size} entries are neither all the components nor the "
            f"symmetric components of a tensor of rank {rank} in a frame "
            f"of {ndim} axes, so the layout must be given."
        )

    def entries(self, ndim: int, size: tx.Optional[int] = None) -> Entries:
        """Return the frame axis of each index, for each entry.

        Parameters
        ----------
        ndim : int
            The number of axes of the frame.
        size : int, optional
            The number of entries along the axis, which reads a layout
            that is `None`.

        Returns
        -------
        tuple of tuple of int
            For each entry, the frame axis of each index of the tensor.

        Raises
        ------
        ValueError
            If an explicit layout names a frame axis that does not exist.
        """
        layout = self.resolved_layout(ndim, size)
        rank = len(self.valence)
        if isinstance(layout, str):
            return _layout_entries(layout, ndim, rank)
        entries = tuple(tuple(int(i) for i in entry) for entry in layout)
        if any(not 0 <= i < ndim for entry in entries for i in entry):
            raise ValueError(
                f"The layout {entries} names an axis that a frame of "
                f"{ndim} axes does not have."
            )
        return entries

    def closed(self, ndim: int, size: tx.Optional[int] = None) -> bool:
        """Return whether the layout holds the tensor in any frame.

        A layout is closed when every component of the tensor is either
        stored or taken from a stored component, so that no component is
        zero by construction.

        Parameters
        ----------
        ndim : int
            The number of axes of the frame.
        size : int, optional
            The number of entries along the axis, which reads a layout
            that is `None`.

        Returns
        -------
        bool
            Whether a change of frame keeps the tensor in this layout.
        """
        return -1 not in self._unpack(ndim, size)

    def reframed(
        self, ndim: int, size: tx.Optional[int] = None
    ) -> "TensorAxis":
        """Return the axis that describes the tensor after [`apply`][].

        Parameters
        ----------
        ndim : int
            The number of axes of the frame.
        size : int, optional
            The number of entries along the axis, which reads a layout
            that is `None`.

        Returns
        -------
        TensorAxis
            This axis when its layout is closed, and otherwise a copy in
            the layout `"symmetric"` when all the indices have the same
            valence, and `"C"` otherwise.
        """
        if self.closed(ndim, size):
            return self
        return replace(self, layout=self._closure())

    def apply(
        self, values: ArrayProtocol, matrix: ArrayProtocol, axis: int
    ) -> ArrayProtocol:
        """Return the components of the tensor in the new frame.

        The entries are unpacked into a full tensor, each of its indices
        is multiplied by `J` or by `J⁻ᵀ` according to its valence, and the
        result is packed again in the same layout, or in the layout of
        [`reframed`][] when this layout is not closed.

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
            The transformed values. The length of `axis` changes when the
            layout is not closed.

        Raises
        ------
        ValueError
            If the number of entries does not match the layout and the
            size of `matrix`.
        """
        xp = array_backend(values)
        rank = len(self.valence)
        size = values.shape[axis]
        ndim = self.frame_ndim(size, int(matrix.shape[-1]))
        unpack = self._unpack(ndim, size)
        packed = xp.moveaxis(values, axis, -1)
        if -1 in unpack:
            # The components that are zero by construction are read from an
            # extra entry of zeros.
            zeros = xp.zeros_like(packed[..., :1])
            packed = xp.concatenate([packed, zeros], axis=-1)
            unpack = np.where(unpack < 0, size, unpack)
        full = packed[..., unpack]
        full = full.reshape(tuple(packed.shape[:-1]) + (ndim,) * rank)
        full = _multilinear(full, matrix, tuple(self.valence), values)
        out = self.reframed(ndim, size)
        pack = np.array(
            [
                np.ravel_multi_index(entry, (ndim,) * rank)
                for entry in out.entries(ndim, size)
            ]
        )
        flat = full.reshape(tuple(full.shape[:-rank]) + (ndim**rank,))
        return xp.moveaxis(flat[..., pack], -1, axis)

    def _closure(self) -> str:
        """Return the smallest closed layout for the stored tensor.

        The tensor is symmetric when its indices are of the same valence and
        its layout does not store two components whose indices are
        permutations of each other, so that it stays symmetric in any
        frame.
        """
        if len(set(self.valence)) > 1:
            return "C"
        if isinstance(self.layout, tuple) and any(
            len(group) > 1
            for group in _permutation_groups(self.layout).values()
        ):
            return "C"
        return "symmetric"

    def _unpack(self, ndim: int, size: tx.Optional[int]) -> np.ndarray:
        """Return the entry of each component, in row-major order.

        A component that is zero by construction has the entry `-1`.
        """
        rank = len(self.valence)
        entries = self.entries(ndim, size)
        position = {entry: k for k, entry in enumerate(entries)}
        permuted = {tuple(sorted(entry)): k for k, entry in enumerate(entries)}
        symmetric = len(set(self.valence)) == 1
        unpack = []
        for multi in itertools.product(range(ndim), repeat=rank):
            if multi in position:
                unpack.append(position[multi])
            elif symmetric:
                unpack.append(permuted.get(tuple(sorted(multi)), -1))
            else:
                unpack.append(-1)
        return np.array(unpack, dtype=int)


def _permutation_groups(
    layout: Entries,
) -> tx.Dict[tx.Tuple[int, ...], tx.List[tx.Tuple[int, ...]]]:
    """Group the entries of an explicit layout by their sorted indices."""
    groups: tx.Dict[tx.Tuple[int, ...], tx.List[tx.Tuple[int, ...]]] = {}
    for entry in layout:
        groups.setdefault(tuple(sorted(entry)), []).append(tuple(entry))
    return groups


def _layout_size(name: str, ndim: int, rank: int) -> int:
    """Return the number of entries of a named layout."""
    if name in ("C", "F"):
        return ndim**rank
    if name == "diagonal":
        return ndim
    return math.comb(ndim + rank - 1, rank)


def _layout_entries(name: str, ndim: int, rank: int) -> Entries:
    """Return the entries of a named layout, in their order."""
    if name == "C":
        return tuple(itertools.product(range(ndim), repeat=rank))
    if name == "F":
        return tuple(
            entry[::-1]
            for entry in itertools.product(range(ndim), repeat=rank)
        )
    if name == "diagonal":
        return tuple((i,) * rank for i in range(ndim))
    upper = tuple(itertools.combinations_with_replacement(range(ndim), rank))
    if name == "symmetric":
        return upper
    if name == "symmetric-lower":
        return tuple(sorted(entry[::-1] for entry in upper))
    # "symmetric-diagonal-first", which is only defined for rank 2.
    diagonal = tuple(entry for entry in upper if entry[0] == entry[1])
    return diagonal + tuple(entry for entry in upper if entry[0] != entry[1])


# --- helpers ----------------------------------------------------------


def _like(matrix: ArrayProtocol, values: ArrayProtocol) -> ArrayProtocol:
    """Return `matrix` in the backend of `values`, in a floating type."""
    xp = array_backend(values)
    dtype = np.result_type(values.dtype, np.float32)
    if isinstance(matrix, np.ndarray) and not is_dask(values):
        return xp.asarray(matrix.astype(dtype))
    return matrix.astype(dtype)


def _multilinear(
    full: ArrayProtocol,
    matrix: ArrayProtocol,
    valence: tx.Tuple[Valence, ...],
    like: ArrayProtocol,
) -> ArrayProtocol:
    """Multiply each index of a stack of tensors by `J` or by `J⁻ᵀ`.

    The last `len(valence)` axes of `full` are the indices of the tensors.
    A vector index is multiplied by `matrix` and a covector index by the
    inverse of its transpose. A vector is a tensor with one index.
    """
    rank = len(valence)
    forward = _like(matrix, like)
    backward = None
    if "covector" in valence:
        backward = _swap_last(_batched(_inv, forward))
    for index, each in enumerate(valence):
        factor = forward if each == "vector" else backward
        full = _contract(full, factor, index, rank)
    return full


def _contract(
    tensor: ArrayProtocol, matrix: ArrayProtocol, index: int, rank: int
) -> ArrayProtocol:
    """Multiply one index of a stack of tensors by a stack of matrices.

    The last `rank` axes of `tensor` are its indices, and the batch of
    `matrix` broadcasts against the other axes.
    """
    xp = array_backend(tensor)
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
    xp = array_backend(matrix)
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
    if is_dask(matrix):
        whole = matrix.rechunk({matrix.ndim - 2: -1, matrix.ndim - 1: -1})
        return whole.map_blocks(function, dtype=matrix.dtype)
    return function(matrix)


def _inv(matrix: ArrayProtocol) -> ArrayProtocol:
    """Invert a stack of matrices."""
    return array_backend(matrix).linalg.inv(matrix)
