# dependencies
import numpy as np
import typing_extensions as tx
from bagof.dispatchers import Function
from bagof.magic import replace

# core
from brainhops._core.anybackend import moveaxis
from brainhops._core.backends import array_backend, like
from brainhops._core.linalg import batchT, inv, matvec
from brainhops._core.typing import ArrayProtocol, npmatrix

# datamodel
from ._frames import _tensor_sub2ind, get_frame_ndim
from .axes import ComponentAxis, TensorAxis
from .enums import IndexVariance

# dispatch
apply_linear_map = Function("apply_linear_map")


@apply_linear_map.register
def _apply_to_vector(
    vector: ArrayProtocol,
    matrix: ArrayProtocol,
    axis_type: ComponentAxis,
    axis: int = -1,
) -> ArrayProtocol:
    """Return the components of the vector in the new frame.

    The entries are unpacked into a full tensor, each of its indices
    is multiplied by `J` or by `J⁻ᵀ` according to its variance, and the
    result is packed again in the same layout, or in the layout of
    [`reframed`][] when this layout is not closed.

    Parameters
    ----------
    vector : ArrayProtocol
        A vector field, with shape `(*pre, n, *post)`.
    matrix : ArrayProtocol
        The linear map `J`, of shape `(*batch, n, n)`, where `batch`
        broadcasts against the shape of `values` without `axis`.
    axis : int
        The axis of `values` that holds the vector.

    Returns
    -------
    ArrayProtocol
        The transformed tensor field, with shape `(*pre, n, *post)`.

    Raises
    ------
    ValueError
        If the number of entries does not match the size of `matrix`.
    NotImplementedError
        If the entries are the coordinates of points, which the
        Jacobian of a map does not re-express.
    """
    if axis_type.variances is None:
        raise NotImplementedError(
            f"The entries of a {axis_type.type!r} axis are the coordinates "
            f"of points, which change with the map itself rather than "
            f"with its Jacobian, so they cannot be re-expressed in "
            f"another frame."
        )
    vector = moveaxis(vector, axis, -1)
    vector = _apply_to_full_tensor(vector, matrix, axis_type.variances)
    return moveaxis(vector, -1, axis)


# --- tensors ----------------------------------------------------------


@apply_linear_map.register
def _apply_to_compact_tensor(
    tensor: ArrayProtocol,
    matrix: ArrayProtocol,
    axis_type: TensorAxis,
    axis: int = -1,
) -> ArrayProtocol:
    """Return the components of the tensor in the new frame.

    The entries are unpacked into a full tensor, each of its indices
    is multiplied by `J` or by `J⁻ᵀ` according to its variance, and the
    result is packed again in the same layout, or in the layout of
    [`reframed`][] when this layout is not closed.

    Parameters
    ----------
    tensor : ArrayProtocol
        A tensor field, with shape `(*pre, axis_length, *post)`, where
        `axis_length` is the length of the flattened compact tensor.
    matrix : ArrayProtocol
        The linear map `J`, of shape `(*batch, n, n)`, where `batch`
        broadcasts against the shape of `values` without `axis`.
    axis : int
        The axis of `values` that holds the compact tensor.

    Returns
    -------
    ArrayProtocol
        The transformed tensor field, with shape
        `(*pre, *output_shape, *post)`.
        The length of `axis` changes when the
        layout is not closed.

    Raises
    ------
    ValueError
        If the number of entries does not match the layout and the
        size of `matrix`.
    """
    rank = axis_type.rank
    variance = axis_type.variances
    xp = array_backend(tensor)
    size = tensor.shape[axis]
    ndim = get_frame_ndim(axis_type, size, int(matrix.shape[-1]))
    sub2ind = _tensor_sub2ind(axis_type, ndim, size)
    packed = xp.moveaxis(tensor, axis, -1)
    if -1 in sub2ind:
        # The components that are zero by construction are read from an
        # extra entry of zeros.
        zeros = xp.zeros_like(packed[..., :1])
        packed = xp.concatenate([packed, zeros], axis=-1)
        sub2ind = np.where(sub2ind < 0, size, sub2ind)
    full = packed[..., sub2ind]
    full = full.reshape(tuple(packed.shape[:-1]) + (ndim,) * rank)
    full = _apply_to_full_tensor(full, matrix, variance)
    out = _reframed_axis(axis_type, ndim, size)
    pack = np.array(
        [
            np.ravel_multi_index(entry, (ndim,) * rank)
            for entry in out.entries(ndim, size)
        ]
    )
    flat = full.reshape(tuple(full.shape[:-rank]) + (ndim**rank,))
    return xp.moveaxis(flat[..., pack], -1, axis)


def _apply_to_full_tensor(
    tensor: ArrayProtocol,
    matrix: ArrayProtocol,
    variance: tx.Tuple[IndexVariance, ...],
) -> ArrayProtocol:
    """Multiply each index of a batch of tensors by `J` or by `J⁻ᵀ`.

    The rank of the tensor is `len(variance)`.

    The last `rank` axes of `full` are the indices of the tensors.
    A vector index is multiplied by `matrix` and a covector index by the
    inverse of its transpose. A vector is a tensor with one index.

    Parameters
    ----------
    tensor : ArrayProtocol
        A stack of tensors, with shape `(*batch, n, ..., n)`, where
        the last `rank` axes have length `n` and correspond to the
        indices of the tensors.
    matrix : ArrayProtocol
        The linear map `J`, of shape `(*batch, n, n)`, where
    variance : tuple[IndexVariance, ...]
        The variance of each index of the tensor, which is either
        `IndexVariance.vector` or `IndexVariance.covector`.
    """
    rank = len(variance)
    forward = like(matrix, tensor, atleast=np.float32)
    xp = array_backend(tensor)
    backward = None
    if IndexVariance.covariant in variance:
        backward = batchT(inv(forward))
    if rank > 1:
        # The other indices of the tensor sit between the batch and the
        # index that is multiplied, so the matrices get an axis of one
        # entry for each of them.
        shape = tuple(matrix.shape)
        broadcast_shape = shape[:-2] + (1,) * (rank - 1) + shape[-2:]
        forward = forward.reshape(broadcast_shape)
        if backward is not None:
            backward = backward.reshape(broadcast_shape)
    for each in reversed(variance):
        factor = forward if each == IndexVariance.contravariant else backward
        tensor = matvec(factor, tensor)
        tensor = xp.moveaxis(tensor, -1, -rank)
    return tensor


def _reframed_axis(
    axis: ComponentAxis, ndim: int, size: tx.Optional[int] = None
) -> ComponentAxis:
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
        variance, and `"C"` otherwise.
    """
    if not isinstance(axis, TensorAxis):
        return axis
    sub2ind = _tensor_sub2ind(axis, ndim, size)
    if -1 not in sub2ind:
        return axis
    return replace(axis, layout=_reframed_layout(axis))


def _reframed_layout(axis: TensorAxis) -> str:
    """Return the smallest closed layout for the stored tensor.

    The tensor is symmetric when its indices are of the same variance and
    its layout does not store two components whose indices are
    permutations of each other, so that it stays symmetric in any
    frame.
    """
    maybe_symmetric = len(set(axis.variance)) == 1
    if maybe_symmetric and isinstance(axis.layout, tuple):
        groups = _permutation_groups(axis.layout)
        is_symmetric = all(len(group) == 1 for group in groups.values())
    else:
        is_symmetric = maybe_symmetric
    return "symmetric" if is_symmetric else "C"


def _permutation_groups(
    layout: npmatrix[int],
) -> tx.Dict[tx.Tuple[int, ...], tx.List[tx.Tuple[int, ...]]]:
    """Group the entries of an explicit layout by their sorted indices."""
    groups: tx.Dict[tx.Tuple[int, ...], tx.List[tx.Tuple[int, ...]]] = {}
    for entry in layout:
        entry = tuple(int(i) for i in entry)
        groups.setdefault(tuple(sorted(entry)), []).append(entry)
    return groups
