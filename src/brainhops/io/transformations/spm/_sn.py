"""The chain of transformations stored in an SPM normalisation file."""

__all__ = ["SpmSnTransformation", "dct_basis", "sn_positions"]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr, replace

# internals
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader
from brainhops.io.transformations.base import TransformationFormat
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import voxel_grid_coordinates

# this format
from ._sn_metadata import SpmSnMetadata
from ._sn_raw import SPM_TO_NIFTI, SpmSnRaw

# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------
# A normalisation file is parsed in one pass, so the coefficients of its
# warp live in the record of its metadata. The chain is decoded from the
# record when it is first used: the warp is then evaluated on the grid of
# the template, with the array backend, so that it stays lazy with Dask.


def _record(xform: "SpmSnTransformation") -> tx.Optional[SpmSnRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _decode(record: SpmSnRaw) -> tx.Tuple[_xforms.Transformation, ...]:
    """Return the chain of transformations that a record describes.

    The chain is described in [`SpmSnTransformation`][].
    """
    ras2grid = RASToVoxel(matrix=np.linalg.inv(record.template.vox2world)[:3])
    # `spm_sn2def` maps template voxel x, numbered from 1, to
    # VF.mat * Affine * (x + warp(x)). A voxel numbered from 0 is x - 1.
    grid2source = record.source.mat @ record.affine @ SPM_TO_NIFTI
    voxel2ras = VoxelToRAS(matrix=grid2source[:3])
    if record.affine_only:
        return (ras2grid, voxel2ras)
    voxel = _systems.VoxelCoordinateSystem()
    field = _xforms.CoordinatesField(
        data=sn_positions(record),
        input=voxel,
        output=voxel,
        degree=InterpolationOrder.linear,
        bound=BoundaryCondition.nearest,
    )
    return (ras2grid, field, voxel2ras)


# ----------------------------------------------------------------------
#   FORMAT
# ----------------------------------------------------------------------


@register_format
class SpmSnTransformation(
    BinaryFileReader,
    _xforms.ImmutableSequence,
    TransformationFormat,
):
    """Spatial normalisation stored in an SPM `*_sn.mat` file.

    SPM2 to SPM8 normalise an image to a template with an affine and a
    smooth nonlinear warp, and save both in a file whose name ends in
    `_sn.mat`. The unified segmentation of SPM5 and SPM8 saves the same
    file as `*_seg_sn.mat`, and its inverse as `*_seg_inv_sn.mat`, and
    the `OldNorm` toolbox of SPM12 still writes it. The file can only be
    read: brainhops writes the same map as an SPM deformation field
    instead, with `t.to(SpmCoordinatesField)`.

    The transformation maps world coordinates of the template, which is
    the target, to world coordinates of the source image. This is the
    direction in which `spm_write_sn` pulls the source image onto the
    grid of the template. It is the
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of

    1. a [`RASToVoxel`][], from template millimetres to the voxels of the
       template, numbered from 0;
    2. a
       [`CoordinatesField`][brainhops.datamodel.transformations.CoordinatesField]
       on the grid of the template, which holds the position of each
       voxel moved by the warp, still in template voxels;
    3. a [`VoxelToRAS`][], which carries these positions through the affine
       to the voxels of the source image, and then to its millimetres.

    When the file holds no warp, the normalisation is affine only, and
    the chain has no field. The values of the field at the voxels of the
    template are those that SPM computes. Between voxels, the field is
    interpolated linearly, and outside the grid it takes the nearest
    value, as the field of an SPM deformation file (`y_*.nii`) is read.
    The warp itself is a sum of cosines that SPM can evaluate anywhere,
    so these values differ from it between voxels and outside the grid.

    A normalisation file is read in one pass, so the whole file is held
    as an [`SpmSnRaw`][] record by the [`SpmSnMetadata`][] of the
    transformation. The chain is decoded from the record when it is
    first used, and it is then cached. For example, the grid of the
    template is `xform.metadata.raw.template`.

    A chain that is assigned replaces the decoded one, and it stays
    assigned, since the file cannot be written.
    """

    HINTS = ("spm", "sn")
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = ("_sn.mat",)

    _metadata: NoRepr[tx.Optional[SpmSnMetadata]] = None

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds the file as a record.

    A transformation built from a chain has no metadata. Assigning other
    metadata drops the chain decoded from the old record, and a chain that
    was assigned is kept.
    """

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain, in the order of application.

        The chain is decoded from the record when it is first used, and
        it is then cached. Decoding evaluates the warp on the grid of the
        template, with the array backend. A transformation without a
        record and without an assigned chain has an empty chain.
        Assigning a chain replaces the decoded one.
        """
        record = _record(self)
        if record is None:
            return ()
        return _decode(record)

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.Sequence[_xforms.Transformation]]
    ) -> None:
        # The assigned chain is stored as a tuple, the type of the chain
        # that is decoded, and the setter drops the decoded chain.
        self._transformations = None if value is None else tuple(value)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open binary file.

        The file is scored by [`SpmSnMetadata.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The open binary file, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an SPM
            normalisation file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return SpmSnMetadata.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of a file.

        The bytes are scored by [`SpmSnMetadata.sniff_bytes`][].

        Parameters
        ----------
        content : bytes
            The content of a file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not an SPM
            normalisation file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return SpmSnMetadata.sniff_bytes(content, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read a normalisation from the rest of an open binary file.

        The record is read by [`SpmSnMetadata.from_fileobj`][], and the
        chain is decoded from it when it is first used. Keyword arguments
        go to the constructor.

        Parameters
        ----------
        file : IO
            The open binary file.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The normalisation.

        Raises
        ------
        TypeError
            If a keyword argument is not a field.
        """
        return cls(metadata=SpmSnMetadata.from_fileobj(file), **kwargs)

    @classmethod
    def from_raw(cls, raw: SpmSnRaw, **kwargs) -> tx.Self:
        """Build a normalisation from an already read [`SpmSnRaw`][].

        The record is held by new metadata, without a copy. Keyword
        arguments go to the constructor.

        Parameters
        ----------
        raw : SpmSnRaw
            The record of the file.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The normalisation.
        """
        return cls(metadata=SpmSnMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a normalisation from a record, a file or a data model.

        An [`SpmSnRaw`][] is read with [`from_raw`][], and any other value
        as the bases read it.

        Parameters
        ----------
        other : object
            A record, a file, its content, a transformation, a mapping, or
            an instance of a similar class.
        *args : Any
            Constructor arguments. A record and a file take keyword
            arguments only.
        **kwargs : Any
            Options of the reader, and field values otherwise.

        Returns
        -------
        Self
            The new normalisation.
        """
        if isinstance(other, SpmSnRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- consolidation ------------------------------------------------

    def consolidate(self, *, inplace: bool = False) -> tx.Self:
        """Return the normalisation with nothing left to encode.

        A normalisation file cannot be written, so no record can be
        encoded, and nothing that is assigned can be moved into one. The
        copy shares the metadata, and it keeps an assigned chain and
        assigned systems as they are, in the same way as an
        [`Minc1Image`][brainhops.io.images.minc.Minc1Image] keeps assigned
        transformations.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to return this normalisation rather than a copy.

        Returns
        -------
        Self
            The copy, or this normalisation with `inplace=True`.
        """
        return self if inplace else replace(self)


# ----------------------------------------------------------------------
#   EVALUATING THE WARP
# ----------------------------------------------------------------------


def dct_basis(size: int, count: int) -> np.ndarray:
    """Return the first discrete cosines on a grid, as SPM samples them.

    This is `spm_dctmtx(size, count)` of SPM, the orthonormal basis of
    the type-II discrete cosine transform: column `k` holds
    `sqrt(2 / size) * cos(pi * (2 * n + 1) * k / (2 * size))` at the
    points `n = 0, ..., size - 1`, and the first column is the constant
    `1 / sqrt(size)`. SPM evaluates it at `x - 1` for the voxels `x`
    numbered from 1, which are the voxels numbered from 0.

    Parameters
    ----------
    size : int
        The number of voxels along the axis.
    count : int
        The number of basis functions.

    Returns
    -------
    np.ndarray
        The `(size, count)` basis.
    """
    points = np.arange(size, dtype=np.float64)[:, None]
    orders = np.arange(count, dtype=np.float64)[None, :]
    basis = np.sqrt(2.0 / size) * np.cos(
        np.pi * (2.0 * points + 1.0) * orders / (2.0 * size)
    )
    basis[:, 0] = 1.0 / np.sqrt(size)
    return basis


def sn_positions(
    record: SpmSnRaw, backend: tx.Any = None
) -> tx.Optional[ArrayProtocol]:
    """Evaluate the warp of a record on the grid of its template.

    Each voxel `i` of the template, numbered from 0, is moved by the warp
    to `i + warp(i)`, in template voxels. Component `c` of the warp is
    `sum(Tr[a, b, d, c] * Bx[i, a] * By[j, b] * Bz[k, d])`, with the bases
    of [`dct_basis`][] along the three axes of the template, as
    `spm_sn2def` computes it plane by plane. The sums are taken one axis
    at a time.

    Parameters
    ----------
    record : SpmSnRaw
        The record of a normalisation file.
    backend : ArrayBackend, optional
        The backend of the result, by default the current one.

    Returns
    -------
    ArrayProtocol or None
        The `(X, Y, Z, 3)` positions, in float64, or `None` for an affine
        normalisation. A NumPy array is read-only.
    """
    if record.affine_only:
        return None
    if backend is None:
        backend = get_array_backend()
    shape = record.template.dim
    counts = record.tr.shape[:3]
    bx, by, bz = (
        backend.asarray(dct_basis(size, count))
        for size, count in zip(shape, counts)
    )
    warp = backend.asarray(record.tr)
    warp = backend.tensordot(warp, bz, axes=([2], [1]))  # (Kx, Ky, 3, Z)
    warp = backend.tensordot(warp, by, axes=([1], [1]))  # (Kx, 3, Z, Y)
    warp = backend.tensordot(warp, bx, axes=([0], [1]))  # (3, Z, Y, X)
    warp = backend.transpose(warp, (3, 2, 1, 0))
    positions = voxel_grid_coordinates(shape, np.eye(4), backend) + warp
    if isinstance(positions, np.ndarray):
        positions.flags.writeable = False
    return positions
