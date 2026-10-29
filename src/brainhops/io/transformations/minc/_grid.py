"""Grid transforms, whose displacements are stored in a MINC volume."""

__all__ = [
    "PendingVolume",
    "XfmGridTransform",
    "grid_chain",
    "write_volume",
]

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core import path

# io
from brainhops._core._internal_api import ArrayProtocol, to_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import (
    BoundaryCondition,
    InterpolationOrder,
    StoreEnum,
)
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserError,
    ParserExistsError,
    ParserNotImplementedError,
)
from brainhops.io.transformations.base._internal_api import (
    affine_between,
    displacement_parts,
    split_field_chain,
    undoes,
    world_pieces,
)
from brainhops.io.transformations.base.affines import (
    RASToVoxel,
    VoxelToRAS,
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
    [`ras_displacement_chain`][]).

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
        filename = path.Path(location).parent / name
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
    if isinstance(image, PendingVolume):
        return XfmGridTransform.from_ras(image.vectors, image.vox2ras)
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


# ----------------------------------------------------------------------
#   WRITING A DISPLACEMENT VOLUME
# ----------------------------------------------------------------------
# A grid transform that was not read from a file has no volume to copy,
# so its displacements are held by a pending volume until the transform
# file is written, and the volume is then written beside it as a MINC2
# file.

_VECTOR = "vector_dimension"
"""The name that libminc gives the dimension of the components."""

_PENDING_NAME = "grid_{}.mnc"
"""The name of a pending volume until the transform file is named."""


class PendingVolume:
    """
    The displacements of a grid transform that is written to a new volume.

    A grid transform that was not read from a displacement volume, such
    as one that a field was converted to, holds its displacements here
    until the transform file is written, and [`write_volume`][] then
    writes them beside it. Nothing is read or computed before.

    Parameters
    ----------
    vectors : ArrayProtocol
        The displacements, in RAS millimetres, of shape `(X, Y, Z, 3)`.
    vox2ras : np.ndarray
        The `(4, 4)` voxel-to-RAS matrix of the grid.
    """

    def __init__(self, vectors: ArrayProtocol, vox2ras: np.ndarray) -> None:
        self.vectors = vectors
        self.vox2ras = np.asarray(vox2ras, dtype=np.float64)


def pending_name(number: int) -> str:
    """Return the name that a pending volume has before it is placed.

    Parameters
    ----------
    number : int
        The number of the grid transform in the file.

    Returns
    -------
    str
        The name, which the writer replaces with one after the file.
    """
    return _PENDING_NAME.format(number)


def write_volume(filename: path.FilenameLike, volume: PendingVolume) -> None:
    """
    Write the displacements of a grid transform as a MINC2 volume.

    The volume has the dimensions `zspace`, `yspace`, `xspace` and
    `vector_dimension`, slowest first, as libminc writes a displacement
    volume. Each voxel axis of the grid is named after the world axis
    that it runs closest to, and its direction cosines, step and start
    place it exactly, so that the voxel-to-world matrix of the volume is
    that of the grid. The displacements are written as `float32`.

    Parameters
    ----------
    filename : path-like
        The path of the volume.
    volume : PendingVolume
        The displacements and the grid.
    """
    import h5py

    vox2ras = volume.vox2ras
    linear = vox2ras[:3, :3]
    steps = np.linalg.norm(linear, axis=0)
    cosines = linear / steps
    names = _axis_names(cosines)
    starts = np.linalg.solve(cosines, vox2ras[:3, 3])
    # Slowest first: z, y, x, then the components.
    order = [names.index(name) for name in reversed(_SPATIAL)]
    vectors = np.asarray(to_backend(volume.vectors, "numpy"), np.float32)
    data = np.ascontiguousarray(vectors.transpose(*order, 3))
    with h5py.File(str(filename), "w") as file:
        root = file.create_group("minc-2.0")
        root.attrs["minc_version"] = np.bytes_("2.0")
        dimensions = root.create_group("dimensions")
        for axis, name in enumerate(names):
            dimension = dimensions.create_dataset(name, data=0, dtype="i4")
            dimension.attrs["varid"] = np.bytes_("MINC standard variable")
            dimension.attrs["vartype"] = np.bytes_("dimension____")
            dimension.attrs["version"] = np.bytes_("MINC Version    1.0")
            dimension.attrs["spacing"] = np.bytes_("regular__")
            dimension.attrs["alignment"] = np.bytes_("centre")
            dimension.attrs["units"] = np.bytes_("mm")
            dimension.attrs["length"] = np.int32(data.shape[order.index(axis)])
            dimension.attrs["step"] = np.float64(steps[axis])
            dimension.attrs["start"] = np.float64(starts[axis])
            dimension.attrs["direction_cosines"] = cosines[:, axis]
        vector = dimensions.create_dataset(_VECTOR, data=0, dtype="i4")
        vector.attrs["length"] = np.int32(_COMPONENTS)
        image = root.create_group("image").create_group("0")
        dataset = image.create_dataset("image", data=data)
        dataset.attrs["dimorder"] = np.bytes_(
            ",".join([*reversed(_SPATIAL), _VECTOR])
        )
        dataset.attrs["complete"] = np.bytes_("true_")
        finite = data[np.isfinite(data)]
        low = float(finite.min()) if finite.size else 0.0
        high = float(finite.max()) if finite.size else 0.0
        image.create_dataset("image-min", data=np.float64(low))
        image.create_dataset("image-max", data=np.float64(high))


def _axis_names(cosines: np.ndarray) -> tx.List[str]:
    """Name each voxel axis after the world axis that it runs closest to."""
    free = list(range(3))
    names = [""] * 3
    weights = np.abs(cosines)
    for axis in np.argsort(-weights.max(axis=0)):
        best = max(free, key=lambda world: weights[world, axis])
        names[axis] = _SPATIAL[best]
        free.remove(best)
    return names


# ----------------------------------------------------------------------
#   CONVERTING A CHAIN OF FIELDS INTO GRID TRANSFORMS
# ----------------------------------------------------------------------


def grid_chain(
    t: _xforms.Transformation, cls: type
) -> tx.Tuple[tx.List[_xforms.Transformation], tx.Optional[str]]:
    """
    Return `t` as the transforms of an MNI transform file, and its loss.

    `t` is read from RAS and written in RAS, and an endpoint that is not
    known is taken to be RAS. It is cut into pieces that each map RAS to
    RAS, as
    [`world_pieces`][brainhops.io.transformations.base._conversions.world_pieces]
    cuts it. Each piece of affines becomes an `Affine`, which the file
    writes as a linear transform, and each field becomes an
    [`XfmGridTransform`][], whose displacements the file writes as a new
    MINC volume. A field of coordinates becomes the displacement of each
    node, and a velocity is integrated. A field keeps the grid on which
    it is placed, and the affines around it become linear transforms of
    their own; where the chain does not say how the field is placed, the
    affines around it are carried into it when they do not undo each
    other.

    libminc interpolates the displacements of a grid with cubic splines,
    which brainhops reads as cubic B-splines of the values. A field that
    is interpolated so is held exactly, and any other field is stored as
    its values at its nodes, which is the same map at the nodes but not
    between them, so it is reported as a loss.

    Parameters
    ----------
    t : Transformation
        The transformation to convert.
    cls : type
        The format, which names the error when `t` cannot be held.

    Returns
    -------
    transforms : list of Transformation
        The transforms of the file, in the order in which they apply.
    loss : str or None
        What the transforms lose, or `None` when they are the map `t` is.
    """
    ras = _systems.RASmm()
    voxel = _systems.VoxelCoordinateSystem()
    transforms: tx.List[_xforms.Transformation] = []
    losses: tx.List[str] = []
    for piece in world_pieces(t, ras, cls):
        if isinstance(piece, _xforms.Affine):
            # A piece may keep endpoints of its own, such as a block of an
            # ITK chain between LPS spaces, so it is bridged to RAS.
            matrix = affine_between(piece, ras, ras, cls)
            transforms.append(
                _xforms.Affine(matrix=matrix[:-1], input=ras, output=ras)
            )
            continue
        before, field, after = split_field_chain(piece, ras, ras, cls)
        cubic = (
            isinstance(field, _xforms.DisplacementField)
            and not field.log
            and int(field.degree) == int(XfmGridTransform.degree)
            and field.data is not None
            and undoes(before, after)
        )
        if cubic:
            values = field.values
            world2grid, grid2world = before, after
        else:
            parts = displacement_parts(piece, ras, cls)
            if parts.field.data is None:
                continue
            values = parts.field.values
            world2grid, grid2world = parts.world2grid, parts.grid2world
            losses.append(
                "libminc interpolates the displacements of a grid with "
                "cubic splines, and its field is stored as its values at "
                "its nodes, which are the same map at the nodes only."
            )
        displacement = _xforms.DisplacementField(
            data=values,
            input=voxel,
            output=voxel,
            degree=XfmGridTransform.degree,
            bound=XfmGridTransform.bound,
            store=XfmGridTransform.store,
        )
        transforms.append(
            XfmGridTransform(
                transformations=(
                    RASToVoxel(matrix=world2grid[:-1]),
                    displacement,
                    VoxelToRAS(matrix=grid2world[:-1]),
                )
            )
        )
    loss = " ".join(dict.fromkeys(losses)) if losses else None
    return transforms, loss
