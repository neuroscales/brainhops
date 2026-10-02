"""AFNI nonlinear warps (`3dQwarp`'s `_WARP` datasets), stored as AFNI
datasets (`.HEAD` + `.BRIK`)."""

__all__ = ["AfniBrikWarp", "AfniWarp"]

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.io.base._base import register_format
from brainhops.io.base._geometry import ras_conversion
from brainhops.io.base.afni import (
    DICOM_TO_RAS,
    AfniHeader,
    AfniParser,
    afni_world,
    scale_bricks,
)
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.images.afni import AfniImage
from brainhops.io.transformations.base import WritableFileBasedTransformation
from brainhops.io.transformations.base.fields import (
    ras_displacement_chain,
    split_ras_displacement_chain,
)

from ._formats import AfniWarpFormat

WARP_LABELS: tx.Tuple[str, str, str] = ("x_delta", "y_delta", "z_delta")
"""The labels AFNI gives the three sub-bricks of a warp
(`IW3D_to_dataset`, `mri_nwarp.c`)."""

_DICOM_RAS_VECTOR = np.array([-1.0, -1.0, 1.0])
"""Turns a DICOM (LPS) vector into RAS, and back."""

_WHAT = "An AFNI warp"


def is_warp_labels(labels: tx.Optional[tx.Sequence[str]]) -> bool:
    """Whether sub-brick labels are those of an AFNI warp."""
    if not labels or len(labels) < 3:
        return False
    return tuple(label.strip() for label in labels[:3]) == WARP_LABELS


def _cardinal(matrix: np.ndarray, tol: float = 1e-6) -> bool:
    """Whether each voxel axis of a voxel-to-world matrix runs along one
    world axis."""
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    norms = np.linalg.norm(linear, axis=0)
    if not np.all(norms > 0):
        return False
    unit = np.abs(linear / norms)
    return bool(np.all(np.abs(unit.max(axis=0) - 1) < tol))


class AfniWarp(AfniWarpFormat, _xforms.ImmutableSequence):
    """
    A nonlinear warp stored in an AFNI format: three sub-bricks of
    displacements, in DICOM (LPS) millimetres.

    The value at a point `x` of the base's grid is the displacement that
    takes it to the source: `x_source = x + u(x)`, both in DICOM
    coordinates. The grid is the dataset's *cardinal* grid, the one AFNI
    computes on (`IW3D_from_dataset`, `mri_nwarp.c`).

    The warp is read as the chain shared with the other RAS displacement
    formats
    ([`ras_displacement_chain`][brainhops.io.transformations.base.fields.ras_displacement_chain]),
    the DICOM displacements and grid being flipped into RAS -- an exact
    change of frame:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the warp's voxels  |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the warp's voxels back to RAS world         |

    The displacements are interpolated linearly. Outside its grid, AFNI
    extrapolates a warp linearly from each face of its box; the data
    model has no such boundary condition, so the nearest value is used.

    Abstract: each storage has its own registered class.
    """

    degree: tx.ClassVar[int] = 1
    """The spline degree used to interpolate the field."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition used outside of the field of view."""

    # --- decoding (implemented by each storage) -----------------------

    def _vox2dicom(self) -> np.ndarray:
        """The `(4, 4)` cardinal voxel-to-DICOM matrix of the grid."""
        raise NotImplementedError

    def _dicom_vectors(self) -> ArrayProtocol:
        """The `(X, Y, Z, 3)` displacements, in DICOM millimetres."""
        raise NotImplementedError

    # --- endpoints ----------------------------------------------------
    #
    # Declared rather than read off the chain, which would decode the
    # displacements.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps from: the base's RAS world."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps to: the source's RAS world."""
        return _systems.RASmm()

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations the warp encodes, built from the
        file on first access and cached. Assigning to it overrides the
        derived chain, which is how a warp that was not read from a file
        is built.
        """
        vox2ras = DICOM_TO_RAS @ self._vox2dicom()
        vectors = self._dicom_vectors()
        backend = get_array_backend(vectors)
        flip = backend.asarray(_DICOM_RAS_VECTOR, dtype=vectors.dtype)
        return ras_displacement_chain(
            vectors * flip, vox2ras, degree=self.degree, bound=self.bound
        )

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The affine from RAS world coordinates to the warp's voxels."""
        return self.transformations[0]

    @property
    def displacement(self) -> _xforms.Transformation:
        """The displacement field, in the voxel units of its grid."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> _xforms.Transformation:
        """The affine from the warp's voxels back to RAS world."""
        return self.transformations[2]

    def to_image(self) -> AfniImage:
        """
        The warp as an AFNI image: what AFNI stores for it.

        The image has three `float32` sub-bricks, the `x`, `y` and `z`
        displacements in DICOM (LPS) millimetres, on the warp's cardinal
        grid, labelled `x_delta`, `y_delta`, `z_delta` as AFNI labels a
        warp's. Saved as an AFNI dataset, it is read back as a warp.

        The attributes of the AFNI header the warp was read from, if any,
        are kept (its view among them).

        Raises
        ------
        WriterError
            If the chain is not a displacement field between two affines,
            or its grid is oblique: AFNI places a warp on a cardinal grid.
        """
        vox2dicom, vectors = self._dicom_field()
        header = getattr(self, "header", None)
        source = header if isinstance(header, AfniHeader) else None
        world = afni_world(source.view) if source is not None else None
        affine = _xforms.Affine(
            input=_systems.VoxelCoordinateSystem(),
            output=world or _systems.LPSmm(),
            matrix=vox2dicom[:3],
        )
        data = np.asarray(vectors, dtype=np.float32)
        image = AfniImage(data=data, transformations=[affine], header=source)
        labels = {"BRICK_LABS": "\0".join(WARP_LABELS)}
        header = image._afni_header(datatype="float", attributes=labels)
        return AfniImage(data=data, transformations=[affine], header=header)

    # --- conversion ---------------------------------------------------

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Build the warp from another sequence, such as a warp of another
        format: its chain is copied.

        A field read from a file builds its chain on access, so its
        `transformations` -- not the chain it was constructed with, which
        is empty -- is what is copied.
        """
        if isinstance(other, _xforms.Sequence):
            kwargs.setdefault("transformations", tuple(other.transformations))
        return super().from_instance(other, *args, **kwargs)

    # --- writing ------------------------------------------------------

    def _dicom_field(self) -> tx.Tuple[np.ndarray, ArrayProtocol]:
        """
        The `(4, 4)` voxel-to-DICOM matrix of the grid, and the
        `(X, Y, Z, 3)` displacements in DICOM millimetres, from the chain.

        The world of the chain's last slot is converted from its
        anatomical orientation (a world without one is taken to be RAS).

        Raises
        ------
        WriterError
            If the chain is not a displacement field between two affines,
            or its grid is oblique: AFNI places a warp on a cardinal grid.
        """
        chain = tuple(self.transformations or ())
        vox2world, vectors = split_ras_displacement_chain(chain, _WHAT, 3)
        to_dicom = DICOM_TO_RAS @ ras_conversion(
            getattr(chain[2], "output", None)
        )
        vox2dicom = to_dicom @ vox2world
        if not _cardinal(vox2dicom):
            raise WriterError(
                "AFNI computes a warp on a cardinal grid, whose axes run "
                "along the DICOM axes, and this warp's grid is oblique. "
                "Resample it on a cardinal grid first."
            )
        backend = get_array_backend(vectors)
        rotate = backend.asarray(to_dicom[:3, :3], dtype=vectors.dtype)
        vectors = backend.matmul(rotate, vectors[..., None])[..., 0]
        return vox2dicom, vectors


@register_format
class AfniBrikWarp(AfniWarp, AfniParser, WritableFileBasedTransformation):
    """
    An AFNI nonlinear warp stored as an AFNI dataset (`.HEAD` +
    `.BRIK`), such as `3dQwarp`'s `<prefix>_WARP+tlrc.HEAD`.

    The dataset has three sub-bricks, the `x`, `y` and `z` displacements
    in DICOM millimetres, labelled `x_delta`, `y_delta`, `z_delta`
    (`3dQwarp -saveall`-style warps may have three more: `hexvol`,
    `BulkEn`, `ShearEn`, which are not read). See [`AfniWarp`][] for what
    they mean.

    AFNI puts nothing in the header that says "warp" but those labels, so
    they are what this reader claims a dataset on, with certainty. A
    three-sub-brick dataset without them is read as an image; read it as
    a warp with `hint="afni.warp"`, or with this class directly.

    **Writing** goes through the image of
    [`to_image`][brainhops.io.transformations.afni.AfniWarp.to_image]:
    the warp is written as three `float32` sub-bricks labelled as AFNI
    labels them, on its cardinal grid. The attributes of the header the
    warp was read from are written back, and the writer options of
    [`AfniImage`][brainhops.io.images.afni.AfniImage] (`view`,
    `attributes`) are accepted.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = AfniImage.EXTENSIONS
    HINTS = ("brik",)

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(cls, header: AfniHeader) -> float:
        """`CERTAIN` for sub-bricks labelled as AFNI labels a warp's,
        `NO` otherwise."""
        if header.nvals >= 3 and is_warp_labels(header.labels):
            return Confidence.CERTAIN
        return Confidence.NO

    def _vox2dicom(self) -> np.ndarray:
        if self.header is None:
            raise ParserContentError(
                "This warp has no AFNI header to read its grid from."
            )
        return self.header.cardinal_matrix

    def _dicom_vectors(self) -> ArrayProtocol:
        header = self.header
        if header is None or self.dataobj is None:
            raise ParserContentError("This warp has no data to read.")
        if header.nvals < 3:
            raise ParserContentError(
                f"An AFNI warp has three sub-bricks (x, y and z "
                f"displacements), and this dataset has {header.nvals}."
            )
        data = scale_bricks(header, self.dataobj)[..., :3]
        return get_array_backend(data).asarray(data)

    # --- writing ------------------------------------------------------

    def to_filename(self, filename: tx.Any, **kwargs) -> None:
        """
        Write the warp as an AFNI dataset: its `.HEAD` and its `.BRIK`.

        The options of [`AfniImage`][brainhops.io.images.afni.AfniImage]'s
        writer are accepted (`view`, `attributes`); the stored type is
        always `float`, as AFNI writes it.
        """
        kwargs.pop("datatype", None)
        self.to_image().to_filename(filename, datatype="float", **kwargs)
