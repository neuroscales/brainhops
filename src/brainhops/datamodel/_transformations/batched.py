"""Batched transformations, which hold one transformation per element.

Some transformations are a family of transformations indexed by one or
more axes of the space. Motion correction, for example, estimates one
affine per volume of a time series. Such an axis is a batch axis: the
transformation carries it through unchanged, while its other outputs
depend on it.

A [`BatchedTransformation`][] holds the family in one of two forms. The
general form holds an array of transformations, which may be of any
type, such as a chain of affines for each time point. The stacked form,
whose subclasses are [`BatchedAffine`][], [`BatchedLinear`][],
[`BatchedScaling`][], [`BatchedTranslation`][] and
[`BatchedPermutation`][], holds the parameters of elements of one type
in one array, whose leading dimensions index the elements, and it
creates an element only when one is asked for.

The design is recorded in `spec/design/batched-transformations.md`.
"""

__all__ = [
    "BatchedTransformation",
    "BatchedAffine",
    "BatchedLinear",
    "BatchedScaling",
    "BatchedTranslation",
    "BatchedPermutation",
]

import functools
from numbers import Integral

import numpy as np
import typing_extensions as tx
from bagof.hints.numpy import ndarray
from bagof.magic import Forget, InitVar, NotKwOnly, fields

from brainhops._core.affines import expm as affine_expm
from brainhops._core.affines import inv as affine_inv
from brainhops._core.affines import logm as affine_logm
from brainhops._core.affines import to_compact, to_homogeneous
from brainhops._core.backends import array_backend
from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.linalg import expm, inv, logm
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol, npvector
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem

from . import nocycles
from .base import Transformation
from .compute.simplify import SimplifyLike
from .compute.simplify import simplify as _simplify
from .compute.utils import axis_counts, axis_list
from .concrete import (
    Affine,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
    is_tangent,
)
from .modes import ModeLike

TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)
_FieldNames = tx.ClassVar[tx.Tuple[str, ...]]
_Key = tx.Union[int, slice, tx.Tuple[tx.Union[int, slice], ...]]


def _subclass_of(family: type, element_type: tx.Any) -> bool:
    """Return whether `element_type` is a class of the family `family`."""
    return isinstance(element_type, type) and issubclass(element_type, family)


class BatchedTransformation(
    Transformation,
    tx.Generic[TRANSFORMATION],
    polymorphic=True,
):
    """A family of transformations, one for each element of a batch.

    A batched transformation stands for one transformation per element of
    its batch axes, such as one affine per volume of a time series. A batch
    axis is carried through unchanged, while the other outputs depend on
    it, so the transformation of a point is the element that sits at the
    coordinate of the point along the batch axes.

    This class is the general form of a batch. It holds an array of
    transformations, one per element, whose shape is the shape of the
    batch. The elements may be of any type, such as sequences, fields or
    lazy inverses, and they may differ from one element to the next, as
    long as they all have the same numbers of input and output axes. The
    stacked form holds elements of one matrix or vector type in one array
    of parameters, and it is built by the class method `batch` of the
    element type, such as `Affine.batch(matrices)`, which returns a
    subclass such as [`BatchedAffine`][].

    `input_axes` and `output_axes` give the positions of the axes of the
    elements in the input and output spaces, as for a
    [`SubspaceTransformation`][brainhops.datamodel.transformations.SubspaceTransformation].
    Every other axis is a batch axis, and the batch axes follow the order
    of the dimensions of the batch. When neither is given, the batch axes
    come first and the axes of the elements follow, so that the space is
    `(*batch_axes, *element_axes)`. `positions` gives the coordinate of
    each element along each batch axis, and its default is the index of
    the element.

    A batched transformation is not registered as any kind: a batch of
    affines is not an affine map of its inputs, because its other outputs
    depend on the batch axes.

    !!! example
        ```pycon
        >>> import numpy as np
        >>> from brainhops.datamodel.transformations import (
        ...     Affine,
        ...     BatchedTransformation,
        ...     Scaling,
        ...     Sequence,
        ... )
        >>> chains = [
        ...     Sequence([Scaling([1.0, 2.0, 3.0])]),
        ...     Sequence([Scaling([2.0, 2.0, 2.0]), Affine(np.eye(3, 4))]),
        ... ]
        >>> batch = BatchedTransformation(chains, input_axes=[0, 1, 2])
        >>> batch.batch_shape, batch.batch_input_axes
        ((2,), [3])
        >>> batch[1] is chains[1]
        True

        ```

    Parameters
    ----------
    elements : array_like of Transformation, optional
        Transformations of the elements, as a nested list or an array of
        objects whose shape is the shape of the batch.

    Other Parameters
    ----------------
    input_axes : array_like of int, optional
        Positions of the input axes of the elements in the input space.
    output_axes : array_like of int, optional
        Positions of the output axes of the elements in the output space.
        It defaults to `input_axes` when only `input_axes` is given.
    positions : sequence of array_like, optional
        Coordinates of the elements along each batch axis, one vector per
        batch axis.
    element_type : type, optional
        Class of the elements of a stacked batch, which selects the
        stacked subclass.
    data : ArrayLike, optional
        Data of the elements of a stacked batch, of shape `(*batch, ...)`.
    input : CoordinateSystem, optional
        Coordinate system of the whole input space.
    output : CoordinateSystem, optional
        Coordinate system of the whole output space.
    """

    # --- class attributes ---------------------------------------------

    data_fields: _FieldNames = (
        "elements",
        "input_axes",
        "output_axes",
        "positions",
    )

    # --- attributes ---------------------------------------------------

    _elements: NotKwOnly[tx.Optional[tx.Any]] = None
    """Transformations of the elements, as given."""

    _input_axes: tx.Optional[npvector[Integral]] = None
    """Positions of the input axes of the elements, as given."""

    _output_axes: tx.Optional[npvector[Integral]] = None
    """Positions of the output axes of the elements, as given."""

    _positions: tx.Optional[tx.Sequence[tx.Any]] = None
    """Coordinates of the elements along each batch axis, as given."""

    element_type: tx.Optional[tx.Type[Transformation]] = None
    """Class of the elements of a stacked batch, or None."""

    _data: tx.Optional[ArrayProtocol] = None
    """Data of the elements of a stacked batch, or None."""

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        self._check_storage()
        self._check_axes()
        self._check_positions()
        self._check_systems()

    # --- storage ------------------------------------------------------
    # The general form stores an array of transformations. A stacked
    # subclass overrides these methods to read its stacked data instead.

    def _check_storage(self) -> None:
        if self.element_type is not None or self._data is not None:
            raise TypeError(
                f"No stacked batch exists for elements of type "
                f"{getattr(self.element_type, '__name__', None)}. A batch "
                f"of other transformations is built from an array of "
                f"elements."
            )
        if self._elements is None:
            raise ValueError(
                "A batch needs the transformations of its elements."
            )
        self._elements = _object_array(self._elements)
        if self._elements.ndim == 0:
            raise ValueError("A batch needs at least one batch axis.")
        if self._elements.size == 0:
            raise ValueError("A batch needs at least one element.")
        counts = {axis_counts(t) for t in self._elements.flat}
        if len(counts) > 1:
            raise ValueError(
                f"The elements of a batch must have the same numbers of "
                f"input and output axes, but they have {sorted(counts)}."
            )
        if None in next(iter(counts), (None, None)):
            raise ValueError(
                "The numbers of axes of the elements of a batch must be "
                "known. Give the elements their input and output systems."
            )

    @property
    def batch_shape(self) -> tx.Tuple[int, ...]:
        """Number of elements along each batch axis."""
        return tuple(int(n) for n in self._elements.shape)

    @property
    def element_counts(self) -> tx.Tuple[int, int]:
        """Numbers of input and output axes of one element."""
        return axis_counts(self._elements.flat[0])

    @property
    def elements(self) -> np.ndarray:
        """Transformations of the elements, as an array of objects."""
        return self._elements

    def _element(self, key: tx.Tuple[int, ...]) -> Transformation:
        # The element at a full index of the batch.
        return self._elements[key]

    def _take(self, key: tx.Tuple[tx.Union[int, slice], ...]) -> tx.Any:
        # The storage of the batch that a key selects.
        return self._elements[key]

    def _rebuild(self, storage: tx.Any, **changes) -> "BatchedTransformation":
        # A batch of the same form, with other storage and other fields.
        return BatchedTransformation(storage, **changes)

    def _inverse_storage(self) -> tx.Any:
        # The storage of the batch of the inverses.
        inverses = np.empty(self.batch_shape, dtype=object)
        for index in np.ndindex(*self.batch_shape):
            inverses[index] = self._elements[index].inverse()
        return inverses

    # --- shape of the batch -------------------------------------------

    @property
    def batch_ndim(self) -> int:
        """Number of batch axes, which is the number of batch dimensions."""
        return len(self.batch_shape)

    @property
    def counts(self) -> tx.Tuple[int, int]:
        """Numbers of input and output axes of the whole space."""
        ni, no = self.element_counts
        return self.batch_ndim + ni, self.batch_ndim + no

    # --- axes ---------------------------------------------------------

    @property
    def input_axes(self) -> tx.List[int]:
        """Positions of the input axes of the elements."""
        if self._input_axes is not None:
            return axis_list(self._input_axes)
        ni = self.element_counts[0]
        return list(range(self.batch_ndim, self.batch_ndim + ni))

    @property
    def output_axes(self) -> tx.List[int]:
        """Positions of the output axes of the elements."""
        if self._output_axes is not None:
            return axis_list(self._output_axes)
        if self._input_axes is not None:
            return axis_list(self._input_axes)
        no = self.element_counts[1]
        return list(range(self.batch_ndim, self.batch_ndim + no))

    @property
    def batch_input_axes(self) -> tx.List[int]:
        """Positions of the batch axes in the input space."""
        taken = set(self.input_axes)
        return [i for i in range(self.counts[0]) if i not in taken]

    @property
    def batch_output_axes(self) -> tx.List[int]:
        """Positions of the batch axes in the output space."""
        taken = set(self.output_axes)
        return [i for i in range(self.counts[1]) if i not in taken]

    @property
    def positions(self) -> tx.Tuple[np.ndarray, ...]:
        """Coordinates of the elements along each batch axis.

        There is one vector per batch axis. When no positions are given,
        each element sits at its index.
        """
        if self._positions is None:
            return tuple(np.arange(n) for n in self.batch_shape)
        return tuple(np.asarray(p) for p in self._positions)

    # --- systems ------------------------------------------------------

    @property
    def element_input(self) -> tx.Optional[CoordinateSystem]:
        """Input system of one element, when the input system is closed."""
        return _subsystem(self.input, self.input_axes)

    @property
    def element_output(self) -> tx.Optional[CoordinateSystem]:
        """Output system of one element, when the output system is closed."""
        return _subsystem(self.output, self.output_axes)

    # --- elements -----------------------------------------------------

    def __getitem__(self, key: _Key) -> Transformation:
        """Return an element, or a smaller batch.

        The key indexes the batch dimensions in order, as an integer, a
        slice, or a tuple of them. An integer selects one element along its
        batch axis, which then disappears from the space. A slice keeps
        the batch axis and selects some of its elements. When every batch
        axis is selected by an integer, the result is the element itself.
        Otherwise, it is a batch of the same form.
        """
        key = _normalize_key(key, self.batch_ndim)
        removed = [i for i, k in enumerate(key) if not isinstance(k, slice)]
        if len(removed) == self.batch_ndim:
            return self._element(key)
        positions = tuple(
            np.asarray(p)[k]
            for p, k in zip(self.positions, key)
            if isinstance(k, slice)
        )
        gone_in = [self.batch_input_axes[i] for i in removed]
        gone_out = [self.batch_output_axes[i] for i in removed]
        return self._rebuild(
            self._take(key),
            input_axes=_shift(self.input_axes, gone_in),
            output_axes=_shift(self.output_axes, gone_out),
            positions=positions,
            input=_drop(self._input, gone_in, self.counts[0]),
            output=_drop(self._output, gone_out, self.counts[1]),
        )

    def __iter__(self) -> tx.Iterator[Transformation]:
        """Iterate over the elements along the first batch axis."""
        for index in range(self.batch_shape[0]):
            yield self[index]

    # --- methods ------------------------------------------------------

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> Transformation:
        """Simplify the batch.

        Nothing is composed across the elements of a batch, so the batch is
        only simplified, or computed as a sequence of one element when
        `factor` is true.

        Parameters
        ----------
        mode : ModeLike, default=True
            Kinds of transformation that may be composed, used only when
            `factor` is true.
        simplify : SimplifyLike, default="analytic"
            How much effort to spend on simplification.
        factor : bool, default=False
            Whether to split the transformation into independent factors.

        Returns
        -------
        Transformation
            The simplified transformation.
        """
        if factor:
            return nocycles.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        return _simplify(self, policy=simplify)

    def inverse(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the batch of the inverses of the elements.

        The batch axes and their positions are kept, and the input and
        output systems, and the axes of the elements, are swapped. The
        general form holds the inverse of each element, which may be a
        lazy inverse. A stacked batch computes the inverses on its whole
        stack.

        Parameters
        ----------
        compute : bool, default=False
            Whether to compute the result, which simplifies it.
        **kwargs : Any
            Arguments passed to [`compute`][] when `compute` is true.

        Returns
        -------
        BatchedTransformation
            The batch of the inverses.
        """
        obj = self._rebuild(
            self._inverse_storage(),
            input_axes=self.output_axes,
            output_axes=self.input_axes,
            positions=self._positions,
            input=self._output,
            output=self._input,
        )
        return obj.compute(**kwargs) if compute else obj

    def as_backend(
        self,
        backend: tx.Any = None,
        *,
        device: tx.Any = None,
        inplace: bool = False,
    ) -> tx.Self:
        """Return the batch with the data of its elements on a backend.

        Parameters
        ----------
        backend : str | module, optional
            The setting. By default, the current setting is used.
        device : str | int, optional
            The device for converted arrays.
        inplace : bool, default=False
            Whether the elements of this batch are replaced and this batch
            is returned, instead of a new batch.

        Returns
        -------
        BatchedTransformation
            A batch of the same class.
        """
        converted = np.empty(self.batch_shape, dtype=object)
        for index in np.ndindex(*self.batch_shape):
            element = self._elements[index]
            converted[index] = element.as_backend(
                backend, device=device, inplace=inplace
            )
        if inplace:
            self._elements = converted
            return self
        return self._rebuild(
            converted,
            input_axes=self._input_axes,
            output_axes=self._output_axes,
            positions=self._positions,
            input=self._input,
            output=self._output,
        )

    @classmethod
    def stack(
        cls,
        elements: tx.Iterable[Transformation],
        batch_axis: tx.Optional[Axis] = None,
    ) -> "BatchedTransformation":
        """Build a batch from transformations that already exist.

        The batch has one batch axis, which comes first in the space, and
        `batch_axis` describes it in the input and output systems. The
        elements may have different types, but they must have the same
        numbers of axes and, when they have any, the same systems. The
        result is a general batch, which holds the elements as they are.
        A batch built from an array that already holds the parameters of
        all its elements should use the class method `batch` of the
        element type instead.

        Parameters
        ----------
        elements : iterable of Transformation
            The transformations to stack.
        batch_axis : Axis, optional
            Description of the batch axis. An axis about which nothing is
            known is used when it is not given.

        Returns
        -------
        BatchedTransformation
            The batch of the elements.

        Raises
        ------
        ValueError
            If no element is given, or if the elements differ in their
            numbers of axes or in their systems.
        """
        elements = list(elements)
        if not elements:
            raise ValueError("A batch needs at least one element.")
        first = elements[0]
        for element in elements[1:]:
            if element.input != first.input or element.output != first.output:
                raise ValueError(
                    "The elements of a batch must have the same input and "
                    "output systems."
                )
        return BatchedTransformation(
            _object_array(elements),
            input=_prepend(first.input, batch_axis),
            output=_prepend(first.output, batch_axis),
        )

    # --- checks -------------------------------------------------------

    def _check_axes(self) -> None:
        # The axes of the elements must be distinct positions of the space,
        # and as many as the element has inputs or outputs.
        ni, no = self.counts
        for name, axes, count, total in (
            ("input_axes", self.input_axes, self.element_counts[0], ni),
            ("output_axes", self.output_axes, self.element_counts[1], no),
        ):
            if len(axes) != count:
                raise ValueError(
                    f"`{name}` must name {count} axes, one per axis of an "
                    f"element, but it names {len(axes)}."
                )
            if len(set(axes)) != len(axes):
                raise ValueError(f"`{name}` names an axis twice: {axes}.")
            if any(not 0 <= a < total for a in axes):
                raise ValueError(
                    f"`{name}` must hold positions of a space of {total} "
                    f"axes, but it is {axes}."
                )

    def _check_positions(self) -> None:
        # One vector of positions per batch axis, with one position per
        # element along that axis.
        if self._positions is None:
            return
        if len(self._positions) != self.batch_ndim:
            raise ValueError(
                f"`positions` must hold one vector per batch axis, that is "
                f"{self.batch_ndim}, but it holds {len(self._positions)}."
            )
        for i, (p, n) in enumerate(zip(self.positions, self.batch_shape)):
            if p.ndim != 1 or len(p) != n:
                raise ValueError(
                    f"The positions along batch axis {i} must be a vector "
                    f"of {n} values, one per element, but they have the "
                    f"shape {p.shape}."
                )

    def _check_systems(self) -> None:
        # A closed system must have the number of axes of its space.
        for name, system, count in (
            ("input", self._input, self.counts[0]),
            ("output", self._output, self.counts[1]),
        ):
            ndim = get_axes(system).ndim if system is not None else None
            if ndim is not None and ndim != count:
                raise ValueError(
                    f"The {name} system has {ndim} axes, but the batch "
                    f"maps a space of {count} axes."
                )


# ----------------------------------------------------------------------
#   STACKED BATCHES
# ----------------------------------------------------------------------


class _StackedBatch(BatchedTransformation[TRANSFORMATION]):
    """Base class of the batches that stack the data of their elements.

    The elements have one matrix or vector type, `element_type`, and their
    parameters are stacked in `data`, whose leading dimensions index the
    elements. An element is created only when it is asked for, as a view
    of the stack. The subclass that the constructor builds is chosen from
    the element type.
    """

    # --- class attributes ---------------------------------------------

    data_fields: _FieldNames = (
        "element_type",
        "data",
        "input_axes",
        "output_axes",
        "positions",
    )

    # --- attributes ---------------------------------------------------

    _elements: Forget

    data = smartproperty("data", fset=False)

    # --- storage ------------------------------------------------------

    def _check_storage(self) -> None:
        self.element_type = _element_type(self.element_type)
        if self._data is None:
            raise ValueError(
                f"A batch of {self.element_type.__name__} needs the data "
                f"of its elements."
            )
        ndim = _element_ndim(self.element_type)
        if np.ndim(self._data) <= ndim:
            raise ValueError(
                f"The data of a batch of {self.element_type.__name__} must "
                f"have more than {ndim} dimensions, because its leading "
                f"dimensions index the elements, but it has "
                f"{np.ndim(self._data)}."
            )

    @property
    def batch_shape(self) -> tx.Tuple[int, ...]:
        """Number of elements along each batch axis."""
        ndim = int(np.ndim(self.data)) - _element_ndim(self.element_type)
        return tuple(int(n) for n in np.shape(self.data)[:ndim])

    @property
    def element_counts(self) -> tx.Tuple[int, int]:
        """Numbers of input and output axes of one element."""
        return self._element_counts(np.shape(self.data))

    def _element_counts(self, shape: tx.Tuple[int, ...]) -> tx.Tuple[int, int]:
        # The numbers of input and output axes of an element, read from the
        # shape of the stack. A vector type maps a space to itself.
        n = int(shape[-1])
        return n, n

    @property
    def elements(self) -> np.ndarray:
        """Transformations of the elements, as an array of objects.

        Each element is created from a view of the stack.
        """
        elements = np.empty(self.batch_shape, dtype=object)
        for index in np.ndindex(*self.batch_shape):
            elements[index] = self._element(index)
        return elements

    def _element(self, key: tx.Tuple[int, ...]) -> Transformation:
        return self.element_type(
            self.data[key],
            input=self.element_input,
            output=self.element_output,
        )

    def _take(self, key: tx.Tuple[tx.Union[int, slice], ...]) -> tx.Any:
        return self.data[key]

    def _rebuild(self, storage: tx.Any, **changes) -> "BatchedTransformation":
        element_type = changes.pop("element_type", self.element_type)
        return BatchedTransformation(
            element_type=element_type, data=storage, **changes
        )

    def inverse(self, compute: bool = False, **kwargs) -> Transformation:
        # A type whose name states a direction, such as `VoxelToLPS`,
        # inverts to the type of its pair.
        reverse = getattr(self.element_type, "_reverseof", None)
        obj = self._rebuild(
            self._inverse_storage(),
            element_type=reverse or self.element_type,
            input_axes=self.output_axes,
            output_axes=self.input_axes,
            positions=self._positions,
            input=self._output,
            output=self._input,
        )
        return obj.compute(**kwargs) if compute else obj

    inverse.__doc__ = BatchedTransformation.inverse.__doc__

    def _inverse_storage(self) -> ArrayProtocol:
        # The data of the inverses of the elements. Each typed subclass
        # computes it on the stack.
        raise NotImplementedError(
            f"The inverse of a batch of {self.element_type.__name__} is "
            f"not implemented."
        )

    def as_backend(
        self,
        backend: tx.Any = None,
        *,
        device: tx.Any = None,
        inplace: bool = False,
    ) -> tx.Self:
        # The parameters of matrix and vector types are small NumPy arrays
        # whatever the backend, as for the types themselves.
        return self


class BatchedAffine(
    _StackedBatch[Affine],
    on={"element_type": partial(_subclass_of, Affine, PLACEHOLDER)},
):
    """A stacked batch of affine transformations.

    `data` has the shape `(*batch, No, Ni+1)`. Its `matrix` view gives the
    matrices of the elements, which are the exponentials of `data` when
    the element type stores tangents.
    """

    derived_fields: _FieldNames = ("matrix", "homogeneous_matrix")
    mutually_exclusive_fields: tx.ClassVar[tx.Tuple[str, ...]] = (
        "data",
        "matrix",
        "homogeneous_matrix",
    )

    _matrix: InitVar[tx.Optional[ArrayProtocol]] = None
    """Matrices of the elements, accepted in place of `data`."""

    _homogeneous_matrix: InitVar[tx.Optional[ArrayProtocol]] = None
    """Homogeneous matrices of the elements, accepted in place of `data`."""

    @property
    def matrix(self) -> ArrayProtocol:
        """Matrices of the elements, of shape `(*batch, No, Ni+1)`."""
        if is_tangent(self.element_type):
            return affine_expm(self.data)
        return self.data

    @matrix.setter
    def matrix(self, value: ArrayProtocol) -> None:
        tangent = is_tangent(self.element_type)
        self._data = affine_logm(value) if tangent else value

    @property
    def homogeneous_matrix(self) -> ArrayProtocol:
        """Homogeneous matrices of the elements.

        They have the shape `(*batch, No+1, Ni+1)`.
        """
        return to_homogeneous(self.matrix)

    @homogeneous_matrix.setter
    def homogeneous_matrix(self, value: ArrayProtocol) -> None:
        self.matrix = to_compact(value)

    def _element_counts(self, shape: tx.Tuple[int, ...]) -> tx.Tuple[int, int]:
        return int(shape[-1]) - 1, int(shape[-2])

    def _inverse_storage(self) -> ArrayProtocol:
        if is_tangent(self.element_type):
            return -self.data
        return affine_inv(self.data)


class BatchedLinear(
    _StackedBatch[Linear],
    on={"element_type": partial(_subclass_of, Linear, PLACEHOLDER)},
):
    """A stacked batch of linear transformations, rotations included.

    `data` has the shape `(*batch, No, Ni)`. Its `matrix` view gives the
    matrices of the elements, which are the exponentials of `data` when
    the element type stores tangents.
    """

    derived_fields: _FieldNames = ("matrix",)
    mutually_exclusive_fields: tx.ClassVar[tx.Tuple[str, ...]] = (
        "data",
        "matrix",
    )

    _matrix: InitVar[tx.Optional[ArrayProtocol]] = None
    """Matrices of the elements, accepted in place of `data`."""

    @property
    def matrix(self) -> ArrayProtocol:
        """Matrices of the elements, of shape `(*batch, No, Ni)`."""
        if is_tangent(self.element_type):
            return expm(self.data)
        return self.data

    @matrix.setter
    def matrix(self, value: ArrayProtocol) -> None:
        tangent = is_tangent(self.element_type)
        self._data = logm(value) if tangent else value

    def _element_counts(self, shape: tx.Tuple[int, ...]) -> tx.Tuple[int, int]:
        return int(shape[-1]), int(shape[-2])

    def _inverse_storage(self) -> ArrayProtocol:
        if is_tangent(self.element_type):
            return -self.data
        if issubclass(self.element_type, Rotation):
            return array_backend(self.data).swapaxes(self.data, -1, -2)
        return inv(self.data)


class BatchedScaling(
    _StackedBatch[Scaling],
    on={"element_type": partial(_subclass_of, Scaling, PLACEHOLDER)},
):
    """A stacked batch of scalings.

    `data` has the shape `(*batch, N)`. Its `scale` view gives the factors
    of the elements, which are the exponentials of `data` when the element
    type stores their logarithms.
    """

    derived_fields: _FieldNames = ("scale",)
    mutually_exclusive_fields: tx.ClassVar[tx.Tuple[str, ...]] = (
        "data",
        "scale",
    )

    _scale: InitVar[tx.Optional[ArrayProtocol]] = None
    """Factors of the elements, accepted in place of `data`."""

    @property
    def scale(self) -> ArrayProtocol:
        """Factors of the elements, of shape `(*batch, N)`."""
        if is_tangent(self.element_type):
            return array_backend(self.data).exp(self.data)
        return self.data

    @scale.setter
    def scale(self, value: ArrayProtocol) -> None:
        tangent = is_tangent(self.element_type)
        self._data = array_backend(value).log(value) if tangent else value

    def _inverse_storage(self) -> ArrayProtocol:
        if is_tangent(self.element_type):
            return -self.data
        return 1 / self.data


class BatchedTranslation(
    _StackedBatch[Translation],
    on={"element_type": partial(_subclass_of, Translation, PLACEHOLDER)},
):
    """A stacked batch of translations.

    `data` has the shape `(*batch, N)`, and its `translation` view gives the
    translations of the elements.
    """

    derived_fields: _FieldNames = ("translation",)
    mutually_exclusive_fields: tx.ClassVar[tx.Tuple[str, ...]] = (
        "data",
        "translation",
    )

    _translation: InitVar[tx.Optional[ArrayProtocol]] = None
    """Translations of the elements, accepted in place of `data`."""

    @property
    def translation(self) -> ArrayProtocol:
        """Translations of the elements, of shape `(*batch, N)`."""
        return self.data

    @translation.setter
    def translation(self, value: ArrayProtocol) -> None:
        self._data = value

    def _inverse_storage(self) -> ArrayProtocol:
        return -self.data


class BatchedPermutation(
    _StackedBatch[Permutation],
    on={"element_type": partial(_subclass_of, Permutation, PLACEHOLDER)},
):
    """A stacked batch of permutations of the axes.

    `data` has the shape `(*batch, N)`, and its `permutation` view gives the
    permutations of the elements.
    """

    derived_fields: _FieldNames = ("permutation",)
    mutually_exclusive_fields: tx.ClassVar[tx.Tuple[str, ...]] = (
        "data",
        "permutation",
    )

    _permutation: InitVar[tx.Optional[ArrayProtocol]] = None
    """Permutations of the elements, accepted in place of `data`."""

    @property
    def permutation(self) -> ArrayProtocol:
        """Permutations of the elements, of shape `(*batch, N)`."""
        return self.data

    @permutation.setter
    def permutation(self, value: ArrayProtocol) -> None:
        self._data = value

    def _inverse_storage(self) -> ArrayProtocol:
        return array_backend(self.data).argsort(self.data, axis=-1)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _object_array(elements: tx.Any) -> np.ndarray:
    """Return the transformations of a batch as an array of objects.

    A nested list or tuple is read as the grid of the batch, and each of
    its leaves must be a transformation. An array of objects is returned
    as it is. NumPy is not asked to infer the shape, because a sequence of
    transformations, or a batch, is itself iterable.
    """
    if isinstance(elements, np.ndarray) and elements.dtype == object:
        return elements
    shape = []
    level = elements
    while isinstance(level, (list, tuple)):
        shape.append(len(level))
        if not level:
            break
        level = level[0]
    array = np.empty(tuple(shape), dtype=object)
    for index in np.ndindex(*shape):
        value = elements
        for i in index:
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    "The elements of a batch must form a regular grid."
                )
            value = value[i]
        if not isinstance(value, Transformation):
            raise TypeError(
                f"The elements of a batch must be transformations, not "
                f"{type(value).__name__}."
            )
        array[index] = value
    return array


def _element_type(element_type: tx.Any) -> tx.Type[Transformation]:
    """Return the data-model class of the elements of a stacked batch.

    The elements of a stacked batch are matrix or vector types, whose
    data has a known number of dimensions. The class of a file format is
    replaced by its data model, because the metadata of a format
    describes a record of a file and not an element of a batch. A lazy
    wrapper, such as an inverse, is refused, since it holds no data of
    its own.
    """
    if not isinstance(element_type, type):
        raise TypeError(
            f"The element type of a batch must be a class, not "
            f"{element_type!r}."
        )
    model = element_type._data_model()
    wrapper = getattr(model, "_operator", None) is not None
    if (
        not issubclass(model, Transformation)
        or wrapper
        or _element_ndim(model) is None
    ):
        raise TypeError(
            f"A stacked batch holds matrix or vector transformations, such "
            f"as Affine or Scaling, not {element_type.__name__}."
        )
    return model


@functools.lru_cache(maxsize=None)
def _element_ndim(model: type) -> tx.Optional[int]:
    """Return the number of dimensions of the data of one element.

    The data of a matrix or vector type is annotated with the shape of
    the data of one transformation, such as `npmatrix` for `Affine` and
    `npvector` for `Scaling`. The number of dimensions is the length of
    that shape: 2 for a matrix type and 1 for a vector type. A batch
    reads it to tell its batch dimensions, which come first, from the
    dimensions of one element. The result is None for a type that has
    no data, such as `Identity`, and for a type whose data may have any
    number of dimensions, such as a field.
    """
    hint = next((f.type for f in fields(model) if f.name == "_data"), None)
    hints = tx.get_args(hint) if tx.get_origin(hint) is tx.Union else (hint,)
    for hint in hints:
        # On Python 3.8, the `ndarray` of bagof is a generic class of its
        # own rather than the class of NumPy, so it is the one compared.
        if tx.get_origin(hint) is not ndarray:
            continue
        shape = tx.get_args(tx.get_args(hint)[0])
        if Ellipsis not in shape:
            return len(shape)
    return None


def _normalize_key(
    key: _Key, ndim: int
) -> tx.Tuple[tx.Union[int, slice], ...]:
    """Return a key with one integer or slice per batch dimension."""
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) > ndim:
        raise IndexError(
            f"The batch has {ndim} batch axes, but the key indexes {len(key)}."
        )
    normalized = []
    for k in key:
        if isinstance(k, slice):
            normalized.append(k)
        elif isinstance(k, Integral):
            normalized.append(int(k))
        else:
            raise TypeError(
                f"A batch is indexed by integers and slices, not by "
                f"{type(k).__name__}."
            )
    normalized += [slice(None)] * (ndim - len(normalized))
    return tuple(normalized)


def _subsystem(
    system: tx.Optional[CoordinateSystem], axes: tx.List[int]
) -> tx.Optional[CoordinateSystem]:
    """Return the system of the axes at `axes`, or None if `system` is open."""
    if system is None or get_axes(system).ndim is None:
        return None
    return system.restrict(axes)


def _drop(
    system: tx.Optional[CoordinateSystem], gone: tx.List[int], ndim: int
) -> tx.Optional[CoordinateSystem]:
    """Return the system without the axes at `gone`, or None if it is open."""
    if system is None or get_axes(system).ndim is None:
        return None
    return system.restrict([i for i in range(ndim) if i not in gone])


def _shift(axes: tx.List[int], gone: tx.List[int]) -> tx.List[int]:
    """Return the positions of `axes` once the axes at `gone` are removed."""
    return [a - sum(g < a for g in gone) for a in axes]


def _prepend(
    system: tx.Optional[CoordinateSystem], axis: tx.Optional[Axis]
) -> tx.Optional[CoordinateSystem]:
    """Return the system with `axis`, or an unknown axis, placed first."""
    if system is None:
        return None
    axes = list(get_axes(system))
    return CoordinateSystem(axes=[axis if axis is not None else Axis()] + axes)
