"""The frames in which NIfTI files store the components of vectors.

Some NIfTI intents mark a displacement field, whose fifth axis holds the
components of a vector at each voxel, or whose fourth axis holds them in
the 4-D files that FSL writes. Each of these intents implies the
coordinate system in which the components are expressed, which is
called the frame of the values. The table names each intent by its
value in the `intent` field of the metadata:

| Code | Intent                  | Frame of the vectors                    |
| ---- | ----------------------- | --------------------------------------- |
| 1006 | displacement vector     | the world of the best affine, in RAS    |
| 2006 | fnirt disp field        | FSL scaled millimetres of the reference |
| 2007 | fnirt cubic spline coef | the same, moved by the initial affine   |
| 2009 | fnirt quad spline coef  | the same, moved by the initial affine   |
| 2008 | fnirt dct coef          | the same, as a foreign frame            |

The functions of this module build these frames from the header, so
that [`NiftiImage`][brainhops.io.images.nifti.NiftiImage] records them
when it reads a file, and they restrict the frame that a writer targets
to the components of the values.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel._frames import (
    FrameSlot,
    _free_rows,
    _structure,
    array_axes,
    component_axes,
    component_count,
    explicit_frame,
)
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Projection,
    Transformation,
)

from ._constants import _NIFTI_FSL_INTENTS, _NIFTI_INTENT_DISPVECT
from ._header import _nifti_intent

FRAMED_INTENTS = frozenset({_NIFTI_INTENT_DISPVECT, *_NIFTI_FSL_INTENTS})
"""The intent codes whose files state the frame of their vectors."""


def header_frames(
    metadata: tx.Any,
    preferred: tx.Optional[Transformation],
    system: tx.Optional[CoordinateSystem],
    shape: tx.Tuple[int, ...],
) -> tx.Dict[int, FrameSlot]:
    """Return the frame of each component axis, as a header states it.

    A `DISPVECT` file stores RAS vectors in millimetres, in the world of
    its best affine, as ITK 5.4 and later read it and as the NIfTI
    displacement fields of brainhops do. The frame is then the best
    affine, which is `preferred`, the last transformation that the
    header decodes to.

    An FSL file stores its vectors in FSL scaled millimetres, which are
    the voxel indices multiplied by the voxel sizes, with the first axis
    flipped when the best affine has a positive determinant. FSL adds a
    vector to the position of a voxel of the reference in these
    coordinates, after the initial affine of a coefficient field has
    moved that position, and reads the sum as a position in the scaled
    millimetres of the moving image. A vector is therefore a difference
    between positions in two coordinate systems. brainhops assumes, as
    FSL does when it adds them, that the two systems share their axes,
    and takes the frame to be the scaled millimetres of the reference.
    The frame is the map from the voxels of the file to these
    coordinates, as
    [`FnirtWarpField`][brainhops.io.transformations.fsl.FnirtWarpField]
    computes it without a reference image, which assumes that a
    deformation field lies on the grid of its reference, as FNIRT and
    `convertwarp` write it. The vectors re-expressed in RAS through this
    frame are the displacements of the map in RAS only when the moving
    image has the same voxel-to-RAS map as the reference, up to the voxel
    size: the same direction cosines, the same flip of the first axis
    and the same position in RAS of the origin of the scaled
    millimetres. Otherwise they differ by a linear term and by a
    constant offset, and only `FnirtWarpField`, which takes the moving
    image, computes the displacements of the map in RAS.

    The values of an FSL deformation field may be absolute positions
    rather than displacements. The header does not tell the two apart,
    and `FnirtWarpField` guesses from the values. This reader does not
    read the values to decide the type of an axis, so it reads them as
    displacements in the same frame. Absolute positions re-expressed in
    another frame are then wrong, because positions also change with the
    translation of the frame.

    A field of DCT coefficients is not a grid of positions, so no map
    from its array reaches the scaled millimetres. Its frame is foreign:
    an
    [`FslCoordinateSystem`][brainhops.io.transformations.fsl.FslCoordinateSystem]
    that no transformation of the image reaches.

    Parameters
    ----------
    metadata : NiftiMetadata
        The metadata of the file, which holds its header.
    preferred : Transformation, optional
        The preferred transformation that the header decodes to.
    system : CoordinateSystem, optional
        The array system that the header decodes to.
    shape : tuple of int
        The shape of the data.

    Returns
    -------
    dict of int to slot
        The frame of each component axis, or an empty mapping when the
        intent does not state a frame or the image has no component axis.
    """
    header = metadata.raw.header
    intent = _nifti_intent(header)
    axes = array_axes(system, len(shape))
    components = component_axes(axes)
    if intent not in FRAMED_INTENTS or not components:
        return {}
    if intent == _NIFTI_INTENT_DISPVECT:
        if preferred is None:
            return {}
        frame = preferred
    else:
        frame = _fsl_frame(metadata, system, len(shape))
    if isinstance(frame, Transformation):
        return {
            i: restricted_frame(frame, axes, i, shape[i]) for i in components
        }
    return {i: frame for i in components}


def restricted_frame(
    frame: Transformation,
    axes: tx.Sequence[Axis],
    index: int,
    size: int,
) -> FrameSlot:
    """Restrict a frame of NIfTI to the components of an axis.

    The world of a NIfTI file has three spatial axes, which the
    components of a vector follow, and may have a time axis, which they
    do not. The frame is restricted to the spatial axes as
    `SingleScaleImage.frames` restricts it, by composing a projection
    that removes the other axes. A field of fewer components than spatial
    axes, such as a two-dimensional field, holds the components along the
    first spatial axes, as ITK reads it. A frame that has fewer axes than
    there are components cannot hold them, and its output system is
    returned instead, as a foreign frame.

    Parameters
    ----------
    frame : Transformation
        The map from the array system to the frame of the file.
    axes : sequence of Axis
        The axes of the array system.
    index : int
        The index of the component axis.
    size : int
        The number of entries along the component axis.

    Returns
    -------
    Transformation or CoordinateSystem
        The restricted frame, or the foreign frame.
    """
    try:
        count = component_count(axes, index, size, [frame])
        return explicit_frame(frame, axes, count)
    except ValueError:
        pass
    structure = _structure(frame, len(axes))
    outputs = None
    if structure is not None:
        outputs = array_axes(frame.output, structure.outputs.shape[0])
    if outputs is None:
        return frame.output
    free = _free_rows(structure.outputs, component_axes(axes))
    try:
        count = axes[index].frame_ndim(size)
    except ValueError:
        return frame.output
    if len(free) < count:
        return frame.output
    dropped = free[count:]
    system = CoordinateSystem(
        [axis for r, axis in enumerate(outputs) if r not in dropped],
        name=getattr(frame.output, "name", None),
    )
    projection = Projection(dropped=dropped, input=frame.output, output=system)
    return projection @ frame


def _fsl_frame(
    metadata: tx.Any, system: tx.Optional[CoordinateSystem], ndim: int
) -> FrameSlot:
    """Return the frame of the vectors of an FSL file.

    The frame is an affine map from the array system to an
    [`FslCoordinateSystem`][brainhops.io.transformations.fsl.FslCoordinateSystem],
    which ignores the axes of the array beyond the first three, or that
    coordinate system alone, as a foreign frame, when the header does not
    give the map.
    """
    # The FSL formats read NIfTI files through this package, so they are
    # imported when a frame is needed rather than when it is loaded.
    from brainhops.io.transformations.fsl._systems import (
        FslCoordinateSystem,
    )
    from brainhops.io.transformations.fsl.fnirt._base import _vector_frame

    output = FslCoordinateSystem()
    matrix = _vector_frame(metadata)
    if matrix is None or system is None or ndim < 3:
        return output
    # The vectors follow the three spatial axes of the file, so the other
    # axes of the array, time and the components, have zero columns.
    full = np.zeros((3, ndim + 1))
    full[:, :3] = matrix[:3, :3]
    full[:, -1] = matrix[:3, 3]
    return Affine(matrix=full, input=system, output=output)
