"""
Private helpers that turn the blocks of an ITK file into a chain and back.

The three ITK formats that store parameters, the text `.tfm` files, the
MATLAB `.mat` files and the HDF5 `.h5` files, hold the same content: a list
of blocks in the order of the file, where a `CompositeTransform` header may
come first. Each format keeps that list in its record, and the helpers of
this module decode the chain of a transformation from the list, encode a
chain back into a list, and tell whether a chain is still the one that a
record decodes to. Each format calls them explicitly.
"""

from warnings import warn

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.io.base.parsers import (
    ParserContentError,
    UnrepresentableTransformationError,
)

from ._common import ItkPrecision, ItkStruct, ItkTransformClass
from ._systems import _make_system

# ----------------------------------------------------------------------
#   DECODING
# ----------------------------------------------------------------------


def _check_composite(blocks: tx.Sequence[ItkStruct]) -> None:
    """Refuse a list of blocks with a composite header that is not first.

    ITK writes a `CompositeTransform` header only as the first block of a
    file, and composites cannot be nested.

    Raises
    ------
    ParserContentError
        If a composite header is not the first block.
    """
    if any(_is_composite(block) for block in blocks[1:]):
        raise ParserContentError(
            "ITK only writes a CompositeTransform as the first block "
            "of a file, and it cannot be nested."
        )


def _top_level(
    blocks: tx.Sequence[ItkStruct],
) -> tx.List[tx.List[ItkStruct]]:
    """Return the top-level transforms of a file, in application order.

    ITK writes a `CompositeTransform` as a header followed by its queue of
    transforms, from front to back, but it applies the queue from back to
    front, so that a queue `[T0, T1]` maps `x` to `T0(T1(x))`. A
    [`Sequence`][brainhops.datamodel.transformations.Sequence] lists its
    transformations in the order in which they apply, so the blocks of a
    composite are reversed, and the file holds one top-level transform. In
    a file without a composite, every block is a separate top-level
    transform.
    """
    if blocks and _is_composite(blocks[0]):
        return [list(reversed(blocks[1:]))]
    return [[block] for block in blocks]


def _check_position(
    blocks: tx.Sequence[ItkStruct], position: tx.Optional[int]
) -> None:
    """Refuse a position that a list of blocks has no transform at.

    Raises
    ------
    ParserContentError
        If there is no top-level transform at `position`.
    """
    if position is None:
        return
    transforms = _top_level(blocks)
    if not transforms and position == 0:
        return
    if not 0 <= position < len(transforms):
        raise ParserContentError(
            f"This ITK file has {len(transforms)} transform(s), so it has "
            f"no transform {position}."
        )


def _decode_chain(
    blocks: tx.Sequence[ItkStruct],
    position: tx.Optional[int] = None,
    quiet: bool = False,
) -> tx.List[ItkStruct]:
    """Return the chain of one top-level transform of a list of blocks.

    The blocks of the chain are the blocks of the list, so that a chain
    decoded twice holds the same objects. The first top-level transform is
    returned by default, with a warning when the file holds several,
    unless `quiet` is true.

    Parameters
    ----------
    blocks : sequence of ItkStruct
        The blocks of the file, in the order of the file.
    position : int, optional
        The index of the top-level transform to return.
    quiet : bool, default=False
        Whether to leave out the warning about other transforms.

    Returns
    -------
    list of ItkStruct
        The blocks of the transform, in application order.

    Raises
    ------
    ParserContentError
        If there is no top-level transform at `position`.
    """
    _check_position(blocks, position)
    transforms = _top_level(blocks)
    if position is None:
        if len(transforms) > 1 and not quiet:
            warn(
                f"This ITK file holds {len(transforms)} transforms and no "
                f"composite, so only the first one is read. Pass "
                f"`position=` to read another one.",
                stacklevel=3,
            )
        position = 0
    if not transforms:
        return []
    return list(transforms[position])


def _same_chain(
    chain: tx.Sequence[_xforms.Transformation],
    decoded: tx.Sequence[_xforms.Transformation],
) -> bool:
    """Tell whether a chain holds the blocks of a decoded chain, in order.

    The blocks are compared by identity, since a decoded chain holds the
    blocks of the record.
    """
    return len(chain) == len(decoded) and all(
        first is second for first, second in zip(chain, decoded)
    )


# ----------------------------------------------------------------------
#   ENCODING
# ----------------------------------------------------------------------


def _encode_chain(
    chain: tx.Sequence[_xforms.Transformation],
) -> tx.Tuple[ItkStruct, ...]:
    """Return the blocks of a file that holds a chain, in the file order.

    Each transformation becomes a block, as [`_block`][] encodes it. A
    chain of several blocks is stored as a composite, whose header comes
    first and whose queue lists the blocks from the last applied to the
    first applied, as ITK writes it. The header takes the precision and
    the dimensions of the first block of its queue.

    Raises
    ------
    UnrepresentableTransformationError
        If a transformation is neither an ITK block nor an affine between
        ITK spaces.
    """
    blocks = [_block(xform) for xform in chain]
    if len(blocks) < 2:
        return tuple(blocks)
    queue = tuple(reversed(blocks))
    header = ItkStruct(
        type=ItkTransformClass.CompositeTransform,
        precision=queue[0].precision,
        ndim_input=queue[0].ndim_input,
        ndim_output=queue[0].ndim_output,
        parameters=_read_only(np.array([])),
        fixed_parameters=_read_only(np.array([])),
    )
    return (header,) + queue


def _block(xform: _xforms.Transformation) -> ItkStruct:
    """Return the block that stores a transformation.

    An ITK block is stored as it is, center included. Any other
    transformation that converts to an affine is stored as an
    `AffineTransform` centered on the origin, which ITK reads back as the
    same matrix.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation is a composite header, or if it is neither
        an ITK block nor an affine between ITK spaces.
    """
    if isinstance(xform, ItkStruct):
        if _is_composite(xform):
            raise UnrepresentableTransformationError(
                "A CompositeTransform header is written by the file that "
                "holds its blocks, and is not a transformation of its own."
            )
        return xform
    return _affine_block(xform)


def _affine_block(xform: _xforms.Transformation) -> ItkStruct:
    """Encode an affine between ITK spaces as an origin-centered block.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation is not an affine between ITK spaces of the
        same dimension.
    """
    affine = xform
    if not isinstance(affine, _xforms.Affine):
        affine = xform.to(_xforms.Affine, error=None)
    matrix = None if affine is None else affine.matrix
    if matrix is None:
        raise UnrepresentableTransformationError(
            f"An ITK file holds ITK transforms or affines, and a "
            f"{type(xform).__name__} cannot be written as either."
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    ndim = matrix.shape[0]
    if matrix.shape != (ndim, ndim + 1):
        raise UnrepresentableTransformationError(
            f"An ITK affine maps a space to a space of the same "
            f"dimension, of shape (D, D + 1), but the matrix has shape "
            f"{matrix.shape}."
        )
    space = _make_system(ndim)
    for end in ("input", "output"):
        system = getattr(affine, end)
        if system is not None and system != space:
            raise UnrepresentableTransformationError(
                f"An ITK affine maps LPS millimetres to LPS millimetres, "
                f"but the {end} of this one is {type(system).__name__}. "
                f"Convert it to ITK's space first."
            )
    return ItkStruct(
        type=ItkTransformClass.AffineTransform,
        precision=ItkPrecision.Double,
        ndim_input=ndim,
        ndim_output=ndim,
        # ITK stores the matrix in row-major order, followed by the
        # translation. With the center at the origin, the ITK translation
        # is the translation of the affine.
        parameters=_read_only(
            np.concatenate([matrix[:, :-1].ravel(), matrix[:, -1]])
        ),
        fixed_parameters=_read_only(np.zeros(ndim)),
    )


def _check_systems(
    xform: _xforms.Sequence, chain: tx.Sequence[_xforms.Transformation]
) -> None:
    """Refuse assigned systems that the file of a chain cannot express.

    An ITK file stores no coordinate system: every block maps the LPS
    millimetres of its dimension. An `input` or an `output` that was
    assigned to `xform` is therefore accepted only when it is the system
    that its chain, `chain`, already maps from or to.

    Raises
    ------
    UnrepresentableTransformationError
        If an assigned system differs from the system of the chain.
    """
    implied = {
        "input": chain[0].input if chain else None,
        "output": chain[-1].output if chain else None,
    }
    for end, expected in implied.items():
        system = getattr(xform, "_" + end)
        if system is not None and system != expected:
            raise UnrepresentableTransformationError(
                f"An ITK file stores no coordinate system, and its chain "
                f"maps {expected!r}, so the {end} {system!r} cannot be "
                f"written."
            )


def _read_only(values: tx.Any) -> tx.Any:
    """Return parameters read from a file as a read-only array.

    A record is never changed in place, and the blocks that it holds are
    also the elements of the chain that a transformation decodes, so their
    arrays are made read-only when they are read. A lazy array is returned
    as it is, since it reads the file each time.
    """
    if not isinstance(values, np.ndarray):
        return values
    values = values.view()
    values.flags.writeable = False
    return values


def _is_composite(block: ItkStruct) -> bool:
    return block.type == ItkTransformClass.CompositeTransform


def _type_name(block: ItkStruct) -> str:
    """Return the name of the type of a block, as an ITK file writes it.

    The name joins the class, the precision and the dimensions of the
    block, as in `AffineTransform_double_3_3`. A B-spline whose degree
    is not 3 appends its degree.
    """
    name = (
        f"{ItkTransformClass(block.type).value}_"
        f"{ItkPrecision(block.precision).value}_"
        f"{block.ndim_input}_{block.ndim_output}"
    )
    if block.type == ItkTransformClass.BSplineTransform and block.degree != 3:
        name += f"_{block.degree}"
    return name
