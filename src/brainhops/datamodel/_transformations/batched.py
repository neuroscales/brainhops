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

The rules that `compute` and `simplify` apply to batches are registered
in the modules `pruners`, `simplifiers`, `converters` and `composers`
of the compute machinery. They call the kernels of the batches, such as
[`BatchedTransformation.compose_with`][], for the part that reads the
elements, so that a stacked subclass computes that part on its stack.

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
from brainhops._core.affines import matmul as affine_matmul
from brainhops._core.affines import to_compact, to_homogeneous
from brainhops._core.backends import array_backend
from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.linalg import expm, inv, logm
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol, npvector
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import CompositionError, ConversionError

from . import nocycles
from .base import Transformation
from .compute.check import KindLike, is_kind
from .compute.compose import compose
from .compute.prune import close_counts, read_pattern, restrict_block
from .compute.simplify import SimplifyLike, SimplifyTable
from .compute.simplify import simplify as _simplify
from .compute.utils import UNREADABLE, axis_counts, axis_list
from .concrete import (
    Affine,
    CoordinateField,
    DisplacementField,
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

    def select(
        self, values: tx.Mapping[int, float]
    ) -> tx.Optional[Transformation]:
        """Return the elements that sit at some coordinates of the batch.

        A point whose coordinates along the batch axes are known reaches
        its element through this method, which the compute rules use when
        a chain fixes a batch axis at a value. Each coordinate is matched
        to the position of an element up to the tolerance of
        [`match_positions`][]. When every batch axis is given, the result
        is the element itself. Otherwise, it is the smaller batch along
        the batch axes that are not given, as `b[k]` returns it.

        Parameters
        ----------
        values : Mapping[int, float]
            Coordinate along some batch axes, indexed by the position of
            the batch axis in the input space.

        Returns
        -------
        Transformation or None
            The element or the smaller batch, or None when a coordinate,
            such as NaN, matches no position.

        Raises
        ------
        ValueError
            If a key is not the position of a batch axis.
        """
        others = set(values) - set(self.batch_input_axes)
        if others:
            raise ValueError(
                f"The batch axes are at {self.batch_input_axes} in the "
                f"input space, so a batch cannot select elements by the "
                f"axes at {sorted(others)}."
            )
        key: tx.List[tx.Union[int, slice]] = []
        for axis, positions in zip(self.batch_input_axes, self.positions):
            if axis not in values:
                key.append(slice(None))
                continue
            index = int(match_positions(positions, [values[axis]])[0])
            if index < 0:
                return None
            key.append(index)
        return self[tuple(key)]

    # --- kernels of the compute rules ---------------------------------
    # The rules that the compute machinery registers for batches, in the
    # modules `pruners`, `simplifiers` and `composers`, handle the axes,
    # the positions and the systems. They call these methods for the part
    # that reads the elements. The general form reads the elements one at
    # a time, and these implementations therefore hold for every batch. A
    # stacked subclass overrides some of them to compute on its stack.

    def elements_are(self, kind: KindLike, compute: bool = False) -> bool:
        """Return whether every element is of a kind.

        A batch is not of any kind finer than `Transformation`, because
        its outputs depend on its batch axes, so `is_kind` is false for a
        batch of affines. This method asks the question of the elements
        instead.

        Parameters
        ----------
        kind : KindLike
            Kind to test, such as a node of
            [`kinds`][brainhops.datamodel.kinds] or a string that names
            one.
        compute : bool, default=False
            Whether the values of the elements may be read.

        Returns
        -------
        bool
            Whether every element can be shown to be of the kind.
        """
        return all(is_kind(e, kind, compute) for e in self.elements.flat)

    def resolve_policy(self, table: SimplifyTable) -> SimplifyPolicy:
        """Return the simplification policy that applies to the batch.

        A batch belongs to no family of transformations. A table that
        names families, such as the bare numeric policy, which reads the
        values of matrix transformations only, is therefore resolved
        against the elements of the batch, and the safest of their
        policies applies. [`SimplifyTable`][] calls this method whenever
        it resolves a batch.

        Parameters
        ----------
        table : SimplifyTable
            Table of the policies of each family.

        Returns
        -------
        SimplifyPolicy
            The policy of the elements.
        """
        return table.resolve(*self._representatives())

    def _representatives(self) -> tx.List[Transformation]:
        # The elements whose families decide the policy of the batch. The
        # elements of a stacked batch share one type, so one is enough.
        return list(self._elements.flat)

    def varies_along(self, dim: int) -> bool:
        """Return whether the elements differ along a batch dimension.

        Two elements of the general form are the same when they are the
        same object, so this test reads no value.

        Parameters
        ----------
        dim : int
            Index of the batch dimension.

        Returns
        -------
        bool
            False when every element equals the first element along the
            dimension, and True when it may differ.
        """
        first = np.take(self.elements, [0], axis=dim)
        first = np.broadcast_to(first, self.batch_shape)
        return any(a is not b for a, b in zip(self.elements.flat, first.flat))

    def element_pattern(
        self, policy: SimplifyTable
    ) -> tx.Tuple[np.ndarray, np.ndarray]:
        """Return which inputs and which batch axes each element output reads.

        The pattern reader of a batch places this pattern in the pattern
        of the whole space. The pattern of the elements is the union of
        the patterns of each element, and an element whose pattern cannot
        be read counts as fully coupled. Under the analytic policy, every
        output of the elements reads every batch axis. Under the numeric
        policy, an output reads a batch axis only when it may differ along
        that axis.

        Parameters
        ----------
        policy : SimplifyTable
            Policy that says how closely the elements may be read.

        Returns
        -------
        block : numpy.ndarray
            Boolean array of shape `(no, ni)`, where `ni` and `no` are the
            numbers of inputs and outputs of one element, whose entry
            `(o, i)` is true when output `o` may read input `i` in some
            element.
        reads_batch : numpy.ndarray
            Boolean array of shape `(no, batch_ndim)` whose entry `(o, j)`
            is true when output `o` may read the batch axis `j`.
        """
        ni, no = self.element_counts
        block = np.zeros((no, ni), dtype=bool)
        for element in self.elements.flat:
            pattern = read_pattern(element, policy)
            if pattern is UNREADABLE or np.shape(pattern) != (no, ni):
                block[...] = True
                break
            block |= np.asarray(pattern, dtype=bool)
        numeric = self.resolve_policy(policy) is SimplifyPolicy.numeric
        varies = [
            not numeric or self.varies_along(j) for j in range(self.batch_ndim)
        ]
        reads_batch = np.tile(np.asarray(varies, dtype=bool), (no, 1))
        return block, reads_batch

    def restrict_elements(
        self,
        inputs: tx.List[int],
        outputs: tx.List[int],
        policy: SimplifyTable,
        **fields,
    ) -> tx.Optional["BatchedTransformation"]:
        """Return the batch of the elements cut down to a closed block.

        The block restrictor of a batch calls this method once it has
        checked that the block keeps every batch axis. Each element is cut
        down by its own restrictor, and the method declines when one of
        them declines.

        Parameters
        ----------
        inputs : list of int
            Sorted positions of the inputs of the block among the inputs
            of one element.
        outputs : list of int
            Sorted positions of the outputs of the block among the outputs
            of one element.
        policy : SimplifyTable
            Policy that says how closely the elements may be read.
        **fields : Any
            Axes, positions and systems of the result, as the constructor
            takes them.

        Returns
        -------
        BatchedTransformation or None
            The batch of the cut elements, or None to decline.
        """
        elements = self.elements
        pieces = np.empty(self.batch_shape, dtype=object)
        for index in np.ndindex(*self.batch_shape):
            piece = restrict_block(elements[index], inputs, outputs, policy)
            if piece is not None:
                # An identity piece states no number of axes, which every
                # element of a batch must state.
                piece = close_counts(piece, len(inputs), len(outputs))
            if piece is None:
                return None
            pieces[index] = piece
        return BatchedTransformation(pieces, **fields)

    def compose_with(
        self, inner: "BatchedTransformation", **fields
    ) -> "BatchedTransformation":
        """Return the batch of the compositions of the elements of two batches.

        Each element of this batch is applied after the element of `inner`
        at the same index, and a batch dimension of length 1 is broadcast
        against the other batch, as in NumPy. The composers of batches call
        this method once they have checked that the two batches meet on
        the same axes. The general form composes the elements one at a
        time with
        [`compose`][brainhops.datamodel._transformations.compute.compose.compose].

        Parameters
        ----------
        inner : BatchedTransformation
            Batch applied first, with as many batch dimensions as this one.
        **fields : Any
            Axes, positions and systems of the result, as the constructor
            takes them.

        Returns
        -------
        BatchedTransformation
            The batch of the compositions.

        Raises
        ------
        CompositionError
            If two elements cannot be composed.
        """
        outer, first = np.broadcast_arrays(self.elements, inner.elements)
        products = np.empty(outer.shape, dtype=object)
        for index in np.ndindex(*outer.shape):
            products[index] = compose(outer[index], first[index])
        return BatchedTransformation(products, **fields)

    def apply_to_points(
        self, points: np.ndarray, index: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """Apply to each point the element that the point selects.

        The composer of a batch with a coordinate field reads the batch
        coordinates of each point, matches them to the positions of the
        elements, and calls this method with the coordinates of the point
        along the input axes of the elements. The general form groups the
        points by element and applies each element to its group.

        Parameters
        ----------
        points : numpy.ndarray
            Coordinates of `N` points along the input axes of the elements,
            of shape `(N, ni)`.
        index : numpy.ndarray
            Index of the element of each point along each batch dimension,
            of shape `(N, batch_ndim)`.
        valid : numpy.ndarray
            Boolean vector of shape `(N,)`, false for a point whose batch
            coordinates match no element.

        Returns
        -------
        numpy.ndarray
            Coordinates of the points along the output axes of the
            elements, of shape `(N, no)`, and NaN for a point that is not
            valid.
        """
        values = np.full((len(points), self.element_counts[1]), np.nan)
        rows = np.nonzero(valid)[0]
        if not len(rows):
            return values
        flat = np.ravel_multi_index(tuple(index[rows].T), self.batch_shape)
        elements = self.elements.reshape(-1)
        for k in np.unique(flat):
            group = rows[flat == k]
            values[group] = _apply_element(elements[k], points[group])
        return values

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

    def _target_class(
        self, kwargs: tx.Dict[str, tx.Any]
    ) -> tx.Type[Transformation]:
        # A new element type selects its own stacked subclass, which the
        # constructor of the general form chooses.
        if "element_type" in kwargs:
            return BatchedTransformation
        return type(self)

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
            # A coordinate selects the element whose position it matches,
            # so two elements cannot share a position.
            if len(np.unique(p)) != len(p):
                raise ValueError(
                    f"The positions along batch axis {i} must be distinct, "
                    f"but they are {p.tolist()}."
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

    # --- kernels of the compute rules ---------------------------------
    # Each typed subclass supplies `_stack_as`, which gives its stack in
    # the encoding of a wider family, and the small kernels `_product`,
    # `_apply_gathered` and `restrict_elements`. Every other kernel reads
    # the stack in the encoding of an affine, so that it is written once.

    def _representatives(self) -> tx.List[Transformation]:
        return [self._element((0,) * self.batch_ndim)]

    def varies_along(self, dim: int) -> bool:
        data = np.asarray(self.data)
        return bool(np.any(data != np.take(data, [0], axis=dim)))

    def element_pattern(
        self, policy: SimplifyTable
    ) -> tx.Tuple[np.ndarray, np.ndarray]:
        ni, no = self.element_counts
        if self.resolve_policy(policy) is not SimplifyPolicy.numeric:
            # The analytic policy reads the structure of the type, which
            # one element gives, and never the values of the stack.
            reads_batch = np.ones((no, self.batch_ndim), dtype=bool)
            return self._structure(policy), reads_batch
        # Under the numeric policy, the exact zeros of the whole stack are
        # read, and a row reads a batch axis when it changes along it.
        affine = np.asarray(self._stack_as(Affine))
        batch_dims = tuple(range(self.batch_ndim))
        block = np.any(affine[..., :-1] != 0, axis=batch_dims)
        reads_batch = np.stack(
            [
                np.any(
                    affine != np.take(affine, [0], axis=j),
                    axis=batch_dims + (-1,),
                )
                for j in batch_dims
            ],
            axis=-1,
        )
        return block, reads_batch

    def _structure(self, policy: SimplifyTable) -> np.ndarray:
        # The pattern of the type, read from one element.
        element = self._element((0,) * self.batch_ndim)
        return np.asarray(read_pattern(element, policy), dtype=bool)

    def compose_with(
        self, inner: BatchedTransformation, **fields
    ) -> BatchedTransformation:
        if not isinstance(inner, _StackedBatch):
            return super().compose_with(inner, **fields)
        # The two stacks are read in the family of the composition of two
        # of their elements, so that one product computes every element.
        family = result_type(self.element_type, inner.element_type)
        outer, inner = widen(self, family), widen(inner, family)
        data = outer._product(inner)
        return outer._rebuild(data, element_type=family, **fields)

    compose_with.__doc__ = BatchedTransformation.compose_with.__doc__

    def apply_to_points(
        self, points: np.ndarray, index: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        # Each point gathers the parameters of its element from the stack,
        # in the encoding that the elements apply, so that a tangent is
        # exponentiated once for each element and not for each point.
        stack = np.asarray(self._stack_as(_family(self.element_type)))
        key = tuple(np.where(valid[:, None], index, 0).T)
        values = self._apply_gathered(stack[key], points).astype(float)
        values[~valid] = np.nan
        return values

    apply_to_points.__doc__ = BatchedTransformation.apply_to_points.__doc__

    def _stack_as(self, family: tx.Type[Transformation]) -> ArrayProtocol:
        # The stack in the encoding of the plain class `family`, such as
        # the matrices of a batch of scalings for `Linear`. A family that
        # cannot hold the elements without a loss is refused. The affine
        # encoding of a linear map is its matrix with a zero translation.
        if family is Affine:
            linear = self._stack_as(Linear)
            zeros = np.zeros(np.shape(linear)[:-1] + (1,))
            return np.concatenate([linear, zeros], axis=-1)
        raise ConversionError(
            f"A batch of {self.element_type.__name__} cannot be held as a "
            f"batch of {family.__name__} without a loss."
        )

    def _product(self, inner: "_StackedBatch") -> ArrayProtocol:
        # The data of the compositions of the elements of two stacks of
        # the same plain class, this one being applied second.
        raise NotImplementedError

    def _apply_gathered(
        self, stack: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        # Apply one element to each point, where `stack` holds, for each
        # point, the parameters of its element in the encoding of the
        # family of the elements.
        raise NotImplementedError


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

    def _stack_as(self, family: tx.Type[Transformation]) -> ArrayProtocol:
        if family is Affine:
            return self.matrix
        return super()._stack_as(family)

    def restrict_elements(
        self,
        inputs: tx.List[int],
        outputs: tx.List[int],
        policy: SimplifyTable,
        **fields,
    ) -> tx.Optional[BatchedTransformation]:
        if not inputs:
            # Outputs that read no input are created at a value, which
            # only the projections that the elements give can hold.
            return super().restrict_elements(inputs, outputs, policy, **fields)
        matrix = self.matrix
        columns = list(inputs) + [np.shape(matrix)[-1] - 1]
        data = matrix[..., outputs, :][..., columns]
        return self._rebuild(data, element_type=Affine, **fields)

    def _product(self, inner: _StackedBatch) -> ArrayProtocol:
        # A NaN in the translation of `inner` is a free axis, which only
        # the outputs that read it inherit, as for two affines.
        return affine_matmul(self.data, inner.data, zero_cancels_nan=True)

    def _apply_gathered(
        self, stack: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        linear = np.einsum("nij,nj->ni", stack[..., :-1], points)
        return linear + stack[..., -1]


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

    def _stack_as(self, family: tx.Type[Transformation]) -> ArrayProtocol:
        if family is Linear or (
            family is Rotation and issubclass(self.element_type, Rotation)
        ):
            return self.matrix
        return super()._stack_as(family)

    def restrict_elements(
        self,
        inputs: tx.List[int],
        outputs: tx.List[int],
        policy: SimplifyTable,
        **fields,
    ) -> tx.Optional[BatchedTransformation]:
        # The block of a rotation is not a rotation in general, so every
        # stack of linear maps is cut into a stack of `Linear`.
        if not inputs:
            return super().restrict_elements(inputs, outputs, policy, **fields)
        data = self.matrix[..., outputs, :][..., inputs]
        return self._rebuild(data, element_type=Linear, **fields)

    def _product(self, inner: _StackedBatch) -> ArrayProtocol:
        return self.data @ inner.data

    def _apply_gathered(
        self, stack: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        return np.einsum("nij,nj->ni", stack, points)


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

    def _stack_as(self, family: tx.Type[Transformation]) -> ArrayProtocol:
        if family is Scaling:
            return self.scale
        if family is Linear:
            scale = np.asarray(self.scale)
            return scale[..., :, None] * np.eye(scale.shape[-1])
        return super()._stack_as(family)

    def restrict_elements(
        self,
        inputs: tx.List[int],
        outputs: tx.List[int],
        policy: SimplifyTable,
        **fields,
    ) -> tx.Optional[BatchedTransformation]:
        if not inputs:
            return super().restrict_elements(inputs, outputs, policy, **fields)
        if inputs == outputs:
            data = self.scale[..., outputs]
            return self._rebuild(data, element_type=Scaling, **fields)
        # A zero scale leaves its input unread, so a block may have
        # different axes on its two sides, and it is cut as a matrix.
        data = self._stack_as(Linear)[..., outputs, :][..., inputs]
        return self._rebuild(data, element_type=Linear, **fields)

    def _product(self, inner: _StackedBatch) -> ArrayProtocol:
        return self.data * inner.data

    def _apply_gathered(
        self, stack: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        return points * stack


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

    def _stack_as(self, family: tx.Type[Transformation]) -> ArrayProtocol:
        if family is Translation:
            return self.translation
        if family is Affine:
            translation = np.asarray(self.translation)
            n = translation.shape[-1]
            eye = np.broadcast_to(np.eye(n), translation.shape[:-1] + (n, n))
            return np.concatenate([eye, translation[..., None]], axis=-1)
        return super()._stack_as(family)

    def restrict_elements(
        self,
        inputs: tx.List[int],
        outputs: tx.List[int],
        policy: SimplifyTable,
        **fields,
    ) -> tx.Optional[BatchedTransformation]:
        if not inputs or inputs != outputs:
            return super().restrict_elements(inputs, outputs, policy, **fields)
        data = self.translation[..., outputs]
        return self._rebuild(data, element_type=Translation, **fields)

    def _product(self, inner: _StackedBatch) -> ArrayProtocol:
        return self.data + inner.data

    def _apply_gathered(
        self, stack: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        return points + stack


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

    def _stack_as(self, family: tx.Type[Transformation]) -> ArrayProtocol:
        if family is Permutation:
            return self.permutation
        if family is Linear:
            permutation = np.asarray(self.permutation)
            n = permutation.shape[-1]
            return (permutation[..., :, None] == np.arange(n)).astype(float)
        return super()._stack_as(family)

    def _structure(self, policy: SimplifyTable) -> np.ndarray:
        # The vector of a permutation is its structure, and it differs
        # from one element to the next, so the patterns of all the
        # elements are read.
        batch_dims = tuple(range(self.batch_ndim))
        return np.any(self._stack_as(Linear) != 0, axis=batch_dims)

    def restrict_elements(
        self,
        inputs: tx.List[int],
        outputs: tx.List[int],
        policy: SimplifyTable,
        **fields,
    ) -> tx.Optional[BatchedTransformation]:
        if not inputs or len(inputs) != len(outputs):
            return super().restrict_elements(inputs, outputs, policy, **fields)
        permutation = np.asarray(self.permutation)
        sources = permutation[..., outputs]
        if np.any(np.sort(sources, axis=-1) != np.asarray(inputs)):
            return None
        # The inputs of the block are numbered from zero in the cut.
        local = np.full(permutation.shape[-1], -1)
        local[inputs] = np.arange(len(inputs))
        data = local[sources]
        return self._rebuild(data, element_type=Permutation, **fields)

    def _product(self, inner: _StackedBatch) -> ArrayProtocol:
        # Output i of the composition reads input inner[outer[i]].
        outer, first = np.broadcast_arrays(self.data, inner.data)
        return np.take_along_axis(first, outer, axis=-1)

    def _apply_gathered(
        self, stack: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        return np.take_along_axis(points, stack.astype(int), axis=-1)


# ----------------------------------------------------------------------
#   POSITIONS AND FAMILIES
# ----------------------------------------------------------------------


def match_positions(
    positions: tx.Sequence[float], values: tx.Sequence[float]
) -> np.ndarray:
    """Return the index of the position that each value matches.

    A coordinate along a batch axis selects the element whose position
    it matches. The match tolerates the rounding of a coordinate computed
    in floating point, such as `t0 + k * TR`, so that it still selects
    volume `k`. The tolerance is a millionth of the smallest spacing
    between two positions, or a millionth of the largest of 1 and the
    magnitude of the position when there is only one. A value that lies
    between two positions, or outside their range, matches none.

    Parameters
    ----------
    positions : sequence of float
        Distinct positions of the elements along one batch axis.
    values : sequence of float
        Coordinates to match.

    Returns
    -------
    numpy.ndarray
        Vector of integers with one entry per value, which is the index of
        the matched position, or -1 when the value, such as NaN, matches
        none.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(positions)
    ordered = positions[order]
    if len(ordered) > 1:
        tolerance = 1e-6 * np.min(np.diff(ordered))
    else:
        tolerance = 1e-6 * max(1.0, abs(float(ordered[0])))
    # The nearest position is one of the two that surround the value.
    right = np.clip(np.searchsorted(ordered, values), 0, len(ordered) - 1)
    left = np.clip(right - 1, 0, len(ordered) - 1)
    nearest = np.where(
        np.abs(ordered[left] - values) < np.abs(ordered[right] - values),
        left,
        right,
    )
    # A NaN compares false, so it matches no position.
    hit = np.abs(ordered[nearest] - values) <= tolerance
    return np.where(hit, order[nearest], -1)


# The plain classes of the stacked batches, from the most specific to the
# most general. The family of an element type is the first of them that
# it derives from, so that a tangent class or a class with a direction,
# such as `VoxelToLPS`, belongs to the family of its plain class.
_FAMILIES = (Rotation, Scaling, Translation, Permutation, Linear, Affine)
_LINEAR_FAMILIES = (Rotation, Scaling, Permutation, Linear)


def _family(element_type: tx.Type[Transformation]) -> tx.Type[Transformation]:
    """Return the plain class of the family of an element type."""
    return next(f for f in _FAMILIES if issubclass(element_type, f))


def result_type(
    outer: tx.Type[Transformation], inner: tx.Type[Transformation]
) -> tx.Type[Transformation]:
    """Return the class of the composition of two stacked element types.

    The classes follow the composers of the elements. Two elements of
    the same family compose into that family, so that two rotations give
    a rotation, and two linear maps of other families give a `Linear`.
    Any other pair gives an `Affine`. A tangent class composes through
    the map that it encodes, so the result is always a plain class.

    Parameters
    ----------
    outer : type
        Class of the element applied second.
    inner : type
        Class of the element applied first.

    Returns
    -------
    type
        Plain class of the composition.
    """
    outer, inner = _family(outer), _family(inner)
    if outer is inner:
        return outer
    if outer in _LINEAR_FAMILIES and inner in _LINEAR_FAMILIES:
        return Linear
    return Affine


def widen(
    batch: "_StackedBatch", family: tx.Type[Transformation]
) -> "_StackedBatch":
    """Return a stacked batch as a batch of a wider plain class.

    A batch of scalings becomes a batch of `Linear` or of `Affine`, and
    a batch of translations becomes a batch of `Affine`. A tangent class
    is read through the map that it encodes. The batch is returned as it
    is when its element type is already `family`.

    Parameters
    ----------
    batch : BatchedTransformation
        A stacked batch.
    family : type
        Plain class of the elements of the result.

    Returns
    -------
    BatchedTransformation
        The batch with elements of class `family`.

    Raises
    ------
    ConversionError
        If `family` cannot hold the elements without a loss.
    """
    if batch.element_type is family:
        return batch
    return batch._rebuild(
        batch._stack_as(family),
        element_type=family,
        input_axes=batch._input_axes,
        output_axes=batch._output_axes,
        positions=batch._positions,
        input=batch._input,
        output=batch._output,
    )


def restack(
    batch: BatchedTransformation, element_type: tx.Type[Transformation]
) -> BatchedTransformation:
    """Return a batch as a stacked batch of an element type.

    A general batch copies the data of its elements, converted to
    `element_type`, into one stack, which is the cheaper form to compute
    with. A stacked batch is widened, as by [`widen`][], and is returned
    as it is when it already has this element type. `t.to(element_type=...)`
    calls this function.

    Parameters
    ----------
    batch : BatchedTransformation
        The batch to convert.
    element_type : type
        Matrix or vector type of the elements of the result.

    Returns
    -------
    BatchedTransformation
        The stacked batch.

    Raises
    ------
    ConversionError
        If `element_type` cannot be stacked, if an element does not
        convert to it, or if the elements disagree on their systems, or
        state systems that the batch does not, because the elements of a
        stacked batch take their systems from the batch.
    """
    try:
        model = _element_type(element_type)
    except TypeError as error:
        raise ConversionError(str(error)) from None
    if batch.element_type is not None:
        return batch if model is batch.element_type else widen(batch, model)
    elements = list(batch.elements.flat)
    first = elements[0]
    if any(
        e.input != first.input or e.output != first.output for e in elements
    ):
        raise ConversionError(
            "The elements of a stacked batch share the systems of the "
            "batch, but these elements have different systems."
        )
    if (first.input is not None and batch.element_input is None) or (
        first.output is not None and batch.element_output is None
    ):
        raise ConversionError(
            "The elements of this batch state systems that the batch does "
            "not, and a stacked batch would lose them."
        )
    data = np.stack([np.asarray(e.to(model).data) for e in elements])
    return BatchedTransformation(
        element_type=model,
        data=data.reshape(batch.batch_shape + data.shape[1:]),
        input_axes=batch._input_axes,
        output_axes=batch._output_axes,
        positions=batch._positions,
        input=batch._input,
        output=batch._output,
    )


def broadcast_element(
    t: Transformation, batch_ndim: int
) -> BatchedTransformation:
    """Return a batch of one element that broadcasts along every batch axis.

    A composer of a batch with a transformation that acts on the axes of
    its elements only composes the batch with this batch, whose batch
    dimensions all have length 1. A matrix or vector transformation
    gives a stacked batch, so that a stacked neighbour composes with it
    on its whole stack, and any other transformation gives a general
    batch.

    Parameters
    ----------
    t : Transformation
        Transformation, which states its numbers of axes.
    batch_ndim : int
        Number of batch dimensions.

    Returns
    -------
    BatchedTransformation
        The batch of shape `(1,) * batch_ndim` that holds `t`.
    """
    family = next((f for f in _FAMILIES if isinstance(t, f)), None)
    if family is not None:
        # The view of a lazy inverse or of a tangent gives its map.
        data = np.asarray(getattr(t, _VIEWS[family]))
        return family.batch(data[(None,) * batch_ndim])
    elements = np.empty((1,) * batch_ndim, dtype=object)
    elements[(0,) * batch_ndim] = t
    return BatchedTransformation(elements)


# The view of each plain class that holds the map of an element.
_VIEWS = {
    Rotation: "matrix",
    Linear: "matrix",
    Affine: "matrix",
    Scaling: "scale",
    Translation: "translation",
    Permutation: "permutation",
}


def _apply_element(element: Transformation, points: np.ndarray) -> np.ndarray:
    """Return the coordinates of some points through one element."""
    domain = CoordinateField(field=points, store="values")
    result = nocycles.SEQUENCE([domain, element]).compute()
    if isinstance(result, DisplacementField):
        result = result.to(CoordinateField)
    if not isinstance(result, CoordinateField):
        raise CompositionError(
            f"An element of a batch, of type {type(element).__name__}, did "
            f"not reduce to a field of coordinates when it was applied to "
            f"points."
        )
    return np.asarray(result.field)


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
