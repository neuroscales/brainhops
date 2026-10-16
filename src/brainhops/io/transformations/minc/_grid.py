"""Grid transforms, whose displacements are stored in a MINC volume."""

__all__ = ["XfmGridTransform"]

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import (
    BoundaryCondition,
    InterpolationOrder,
    StoreEnum,
)
from brainhops.io.base._utils_files import sibling
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserError,
    ParserExistsError,
    ParserNotImplementedError,
)
from brainhops.io.transformations.base.fields import ras_displacement_chain

_SPATIAL = ("xspace", "yspace", "zspace")
"""The names of the spatial dimensions of a MINC volume."""

_COMPONENTS = 3
"""The number of components of a displacement, one per world axis."""


class XfmGridTransform(_xforms.ImmutableSequence):
    """A grid transform of an MNI transform file, as a chain.

    A grid transform maps a world point `x`, in RAS millimetres, to
    `x + d(x)`, where `d` interpolates the displacements of a MINC volume
    at `x` (`grid_transform_point` in libminc's
    `volume_io/MNI_formats/grid_transforms.c`). The volume has three
    spatial dimensions and a vector dimension of three components, which
    are the displacements along the world axes x, y and z, in
    millimetres. The transform is read as a chain of three
    transformations, as a NIfTI displacement field is (see
    [`ras_displacement_chain`][brainhops.io.transformations.base.fields.ras_displacement_chain]).

    | Slot           | Transformation                                |
    | -------------- | --------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the voxels of the grid |
    | `displacement` | the displacement field, in voxel units        |
    | `voxel2ras`    | the voxels of the grid back to RAS world      |

    libminc interpolates the displacements with cubic interpolating
    splines, and lowers the degree to linear and then to nearest within
    two voxels of the edge of the grid. It applies no displacement at more
    than half a voxel outside the grid. The field of the chain is
    interpolated with cubic B-splines that pass through the stored values,
    and it is zero outside the grid. Both pass through the displacements
    at the nodes of the grid, and they differ slightly between the nodes
    and near the edges.
    """

    # The interpolation differs from that of libminc (item 144 of the
    # design memo).
    degree: tx.ClassVar[InterpolationOrder] = InterpolationOrder.cubic
    """The degree of the spline that interpolates the field."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.zero
    """The boundary condition outside the grid."""

    store: tx.ClassVar[StoreEnum] = StoreEnum.values
    """Whether the field holds spline coefficients or values."""

    @classmethod
    def from_ras(cls, vectors: ArrayProtocol, vox2ras: np.ndarray) -> tx.Self:
        """Build the chain from RAS displacements and the grid's affine.

        Parameters
        ----------
        vectors : ArrayProtocol
            The displacements, in RAS millimetres, of shape `(X, Y, Z, 3)`.
        vox2ras : np.ndarray
            The `(4, 4)` voxel-to-RAS matrix of the grid.

        Returns
        -------
        Self
            The chain of the RAS-to-voxel affine, the field of
            displacements in voxel units, and the voxel-to-RAS affine.
        """
        return cls(
            transformations=ras_displacement_chain(
                vectors,
                vox2ras,
                degree=cls.degree,
                bound=cls.bound,
                store=cls.store,
            )
        )

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The map from RAS world coordinates to the voxels of the grid."""
        return self.transformations[0]

    @property
    def displacement(self) -> _xforms.Transformation:
        """The displacement field, in voxel units."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> _xforms.Transformation:
        """The map from the voxels of the grid to RAS world coordinates."""
        return self.transformations[2]


# ----------------------------------------------------------------------
#   READING A DISPLACEMENT VOLUME
# ----------------------------------------------------------------------


def read_volume(
    name: str, location: tx.Optional[path.FilenameLike]
) -> tx.Optional[tx.Any]:
    """Read the header of the displacement volume of a grid transform.

    The volume is read as a
    [`MincImage`][brainhops.io.images.minc.MincImage], whose voxels are
    read only when they are needed. A relative name is relative to the
    directory of the transform file, as in libminc's
    `input_one_transform`. Without the location of the transform file, a
    relative name cannot be resolved, and `None` is returned.

    Parameters
    ----------
    name : str
        The name of the volume, as the transform file stores it.
    location : path-like, optional
        The path of the transform file.

    Returns
    -------
    MincImage or None
        The volume, or `None` if it cannot be located.

    Raises
    ------
    ParserExistsError
        If the volume does not exist.
    ParserContentError
        If the volume is not a MINC file that brainhops reads.
    ParserNotImplementedError
        If nibabel, which reads MINC files, is not installed.
    """
    if location is None:
        if not os.path.isabs(name):
            return None
        filename = path.Path(name)
    else:
        filename = sibling(location, name)
    try:
        from brainhops.io.images.minc import MincImage
    except ImportError as error:
        raise ParserNotImplementedError(
            f"Reading the displacement volume {name!r} of a grid transform "
            f"needs nibabel, which reads MINC files: pip install "
            f"brainhops[minc]."
        ) from error
    # A MINC1 volume written by libminc has no variable for its vector
    # dimension, and the MINC reader refuses it (item 142 of the design
    # memo).
    try:
        return MincImage.from_filename(filename)
    except ParserExistsError as error:
        raise ParserExistsError(
            f"The displacement volume {name!r} of a grid transform does not "
            f"exist: {filename}."
        ) from error
    except ParserError as error:
        raise ParserContentError(
            f"Cannot read the displacement volume {name!r} of a grid "
            f"transform ({filename}): {error}"
        ) from error


def grid_from_volume(image: tx.Any, name: str) -> XfmGridTransform:
    """Decode the grid transform of a displacement volume.

    The volume must have the spatial dimensions `xspace`, `yspace` and
    `zspace` and one other dimension of three components, in any order,
    as libminc's `create_grid_transform` requires. The grid is placed by
    the voxel-to-world matrix of the volume, and component `c` of the
    other dimension is the displacement along world axis `c`.

    Parameters
    ----------
    image : MincImage
        The displacement volume.
    name : str
        The name of the volume, for error messages.

    Returns
    -------
    XfmGridTransform
        The chain of the grid transform.

    Raises
    ------
    ParserContentError
        If the volume is not a displacement volume.
    """
    record = image.metadata.raw
    names = [dimension.name for dimension in record.dimensions]
    others = [i for i, n in enumerate(names) if n not in _SPATIAL]
    spatial = sorted(n for n in names if n in _SPATIAL)
    vector = others[0] if len(others) == 1 else None
    if (
        len(names) != 4
        or spatial != sorted(_SPATIAL)
        or vector is None
        or record.dimensions[vector].length != _COMPONENTS
    ):
        raise ParserContentError(
            f"The displacement volume {name!r} of a grid transform must "
            f"have the dimensions xspace, yspace and zspace and one of "
            f"{_COMPONENTS} components, not {', '.join(names)} of lengths "
            f"{', '.join(str(d.length) for d in record.dimensions)}."
        )
    # The data of the image reverses the dimensions of the file, and the
    # voxel-to-world matrix orders the spatial dimensions in the same
    # way, so moving the vector axis last aligns the grid with it.
    axis = len(names) - 1 - vector
    order = [i for i in range(len(names)) if i != axis] + [axis]
    vectors = image.data.transpose(order)
    return XfmGridTransform.from_ras(vectors, record.vox2world)
