"""DARTEL flow fields, stored in NIfTI files prefixed with `u_`."""

__all__ = ["DARTEL_DESCRIP", "DARTEL_STEPS", "DartelFlowField"]

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import KwOnly, NoRepr, replace

# internals
from brainhops._core.properties import smartproperty
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import (
    BoundaryCondition,
    InterpolationOrder,
    StoreEnum,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _header,
    _nifti_shape,
    _NiftiObject,
    _record_image,
    _stored_parts,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import homogeneous_matrix
from brainhops.io.transformations.nifti import (
    NiftiBasedTransformation,
    NiftiRASCoordinatesField,
)
from brainhops.io.transformations.nifti._base import _consolidated_metadata
from brainhops.io.transformations.nifti._views import (
    _field_to_disk,
    _field_to_model,
)

DARTEL_STEPS = 6
"""The number of squaring steps that SPM12 uses by default (`K`).

Every `K` option of `config/spm_cfg_deformations.m` and
`toolbox/DARTEL/tbx_cfg_dartel.m` defaults to 6, which integrates the
velocity with 64 time steps.
"""

DARTEL_DESCRIP = b"Flow Field"
"""The description that `spm_dartel_warp` gives a flow field."""

DARTEL_BOUND = BoundaryCondition.gridwrap
"""The boundary condition under which DARTEL composes its fields.

`shoot_boundary.c` uses `circulant_boundary` unless `set_bound` is
called, which `dartel3('Exp', ...)` does not do, so a field wraps around
its grid, with the period of the grid.
"""

_XFORM_CODE = 2
"""The NIfTI code of the qform and the sform that DARTEL writes.

The code is `Aligned`, the intent that `spm_preproc_write8` gives the
`mat` and the `mat0` of the imported images, which `spm_dartel_warp`
copies into the flow field.
"""

_NDIM = 3
"""The number of spatial dimensions of a flow field."""

_WHAT = "A DARTEL flow field"


@register_format
class DartelFlowField(_xforms.ImmutableSequence, NiftiBasedTransformation):
    """
    DARTEL flow field written by SPM, in a file whose name starts with `u_`.

    DARTEL registers the images that SPM imported (`rc1*.nii`) to a
    template that it builds from them, and saves, for each image, the
    stationary velocity whose flow is the registration. The file is a
    NIfTI vector image of shape (X, Y, Z, 1, 3), on the grid of the
    imported image, which holds the velocity in voxels of that grid.

    The field maps RAS to RAS as SPM12's Deformations utility does for a
    forward flow (`times = [0 1]` in `spm_deformations.m`). The field is
    an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of three named slots:

    | Slot        | Transformation                                 |
    | ----------- | ---------------------------------------------- |
    | `ras2voxel` | the inverse of the qform (`mat0`)              |
    | `velocity`  | the velocity, in voxels of the grid            |
    | `voxel2ras` | the sform (`mat`), from the voxels to RAS      |

    The velocity is a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.StationaryVelocityField],
    interpolated linearly and wrapped around its grid, which is how
    DARTEL composes fields, and integrated with `steps` squaring steps,
    which is the `K` of SPM. The file does not store `K`, so `steps` is
    an option of the reader, whose default is SPM12's, 6. The backward
    flow, which SPM's Deformations utility computes by default and which
    maps the template to the subject, is the inverse of the field,
    `field.inverse()`.

    A field whose velocity is assigned is written with the qform and the
    sform of its chain. Another transformation is converted to this format
    exactly, or not at all (see
    [`brainhops.io.transformations.spm._converters`][]).
    """

    HINTS = ("dartel",)

    PREFIXES: tx.ClassVar[tx.Tuple[str, ...]] = ("u_",)
    """File name prefixes of DARTEL flow fields."""

    steps: tx.Annotated[
        int,
        tx.Doc(
            """
            The number of squaring steps that integrate the velocity, which
            is the `K` of SPM and which the file does not store.
            """
        ),
        KwOnly(),
    ] = DARTEL_STEPS

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The velocity given to the field, or `None` to decode it from `raw`.

    A velocity stored here is pending until [`consolidate`][] or a write
    encodes it into `raw`.
    """

    @smartproperty(cache=True, invalidates="transformations")
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The velocity as an (X, Y, Z, 3) array, in voxels of its grid.

        A velocity that is assigned is returned as it is, and stays pending
        until [`consolidate`][] or a write encodes it. Otherwise it is
        decoded from `raw`: the file stores the field with shape
        (X, Y, Z, 1, 3), and the singleton axis is dropped. Assigning the
        data drops the chain decoded from the previous data.
        """
        if self.raw is None:
            return None
        return _field_to_model(self.raw)

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the chain decoded from the previous
    header.
    """

    # --- reading ------------------------------------------------------
    # The readers hand their keyword arguments to nibabel, so `steps` is
    # popped first and set on the field read.

    @classmethod
    def from_filename(cls, filename: tx.Any, **kwargs) -> tx.Self:
        """Read a flow field from the path of a NIfTI file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The number of squaring steps `steps`, and the options of the
            nibabel readers.

        Returns
        -------
        DartelFlowField
            The field, whose array is read lazily.
        """
        steps = kwargs.pop("steps", None)
        return _with_steps(super().from_filename(filename, **kwargs), steps)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read a flow field from an open NIfTI stream.

        Parameters
        ----------
        fileobj : BinaryIO
            The stream, which must stay open while the data may be read.
        **kwargs : Any
            The number of squaring steps `steps`, and the options of the
            nibabel readers.

        Returns
        -------
        DartelFlowField
            The field, whose array is read lazily.
        """
        steps = kwargs.pop("steps", None)
        return _with_steps(super().from_fileobj(fileobj, **kwargs), steps)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a header as a DARTEL flow field.

        A field of shape (X, Y, Z, 1, 3) described as `"Flow Field"`, as
        `spm_dartel_warp` writes it, scores `Confidence.CERTAIN`. Any other
        header scores as a field of coordinates, since the content of the
        file cannot tell the two apart, and the `u_` prefix decides the
        tie, through
        [`PREFIXES`][brainhops.io.transformations.spm.DartelFlowField.PREFIXES].
        """
        shape = _nifti_shape(header)
        if (
            shape
            and len(shape) == 5
            and shape[3:] == (1, _NDIM)
            and _descrip(header) == DARTEL_DESCRIP
        ):
            return Confidence.CERTAIN
        return NiftiRASCoordinatesField._score_nibabel(header)

    # --- endpoints ----------------------------------------------------
    # The endpoints are declared rather than read off the chain, which
    # would decode the data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The world space of the qform, in RAS millimetres."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The world space of the sform, in RAS millimetres."""
        return _systems.RASmm()

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations that the field encodes.

        The chain is built from the header and the data when it is first
        used, and then cached. Assigning a chain overrides the decoded
        one, which is how fields that do not come from a file are built.

        Raises
        ------
        ParserContentError
            If the field has no velocity, if the velocity is not a 3-D
            grid of 3-vectors, or if the header has no qform.
        """
        qform, sform = _grid_affines(_header(self))
        voxel = _systems.VoxelCoordinateSystem()
        velocity = _xforms.DisplacementField(
            data=_velocity(self.data),
            input=voxel,
            output=voxel,
            log=True,
            steps=self.steps,
            degree=InterpolationOrder.linear,
            bound=DARTEL_BOUND,
        )
        return (
            RASToVoxel(matrix=np.linalg.inv(qform)[:-1]),
            velocity,
            VoxelToRAS(matrix=sform[:-1]),
        )

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.Sequence[_xforms.Transformation]]
    ) -> None:
        # None means that the chain is decoded from the NIfTI file.
        self._transformations = None if value is None else tuple(value)

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The affine from the world of the qform to the voxels."""
        return self.transformations[0]

    @property
    def velocity(self) -> _xforms.Transformation:
        """The stationary velocity, in voxels of the grid."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> _xforms.Transformation:
        """The affine from the voxels to the world of the sform."""
        return self.transformations[2]

    # --- writing ------------------------------------------------------

    def consolidate(
        self, *, inplace: bool = False, like: tx.Any = None, **overrides
    ) -> tx.Self:
        """
        Encode the pending velocity or chain into `raw` and a header.

        A field whose chain is decoded from its stored velocity, assigned
        or read, keeps the velocity under a copy of the header of its
        metadata, whose geometry is kept. A field whose chain was assigned
        is encoded from the chain: the velocity becomes the array, the
        inverse of the first affine becomes the qform and the last affine
        becomes the sform, both with the code `Aligned` (2) that DARTEL
        writes. The file does not store the number of squaring steps, so
        the copy takes the steps of the velocity as its `steps` option, and
        the file must be read with them. A field without metadata is
        described as `"Flow Field"`, as `spm_dartel_warp` describes it.
        Non-encoding header fields are copied from `like`, and `overrides`
        are applied last. A field that has a header and nothing pending,
        consolidated without options, is left as it is.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this field rather than a copy.
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        **overrides : Any
            Header fields that take precedence over the derived values and
            `like`.

        Returns
        -------
        Self
            The consolidated copy, or this field with `inplace=True`.

        Raises
        ------
        UnrepresentableTransformationError
            If the assigned chain is not one that DARTEL writes, or if the
            inverse of its first affine is not a qform.
        WriterError
            If the field has no velocity.
        """
        if (
            self._data is None
            and self._transformations is None
            and like is None
            and not overrides
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            return self if inplace else replace(self)
        image, steps = _dartel_encoded(self, like, overrides)
        raw, record = _stored_parts(image)
        metadata = _consolidated_metadata(self, record)
        if not inplace:
            return replace(
                self,
                raw=raw,
                metadata=metadata,
                data=None,
                transformations=None,
                steps=steps,
            )
        self.raw = raw
        self.metadata = metadata
        self.data = None
        self.transformations = None
        self.steps = steps
        return self


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _with_steps(field: DartelFlowField, steps: tx.Any) -> DartelFlowField:
    """Return a field read from a file, with the steps of the reader."""
    if steps is None:
        return field
    return replace(field, steps=_check_steps(steps))


def _check_steps(steps: tx.Any) -> int:
    """Return a number of squaring steps, checked.

    A specification such as `u_rc1.nii|dartel|steps:6` gives the steps as
    text, which is read as a decimal integer.

    Raises
    ------
    TypeError
        If `steps` is not a non-negative integer.
    """
    if isinstance(steps, str) and steps.strip().isdigit():
        steps = int(steps)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise TypeError(
            f"steps is a non-negative integer, the K of SPM, not {steps!r}."
        )
    if steps < 0:
        raise TypeError(f"steps is a non-negative integer, not {steps}.")
    return int(steps)


def _descrip(header: _NiftiObject) -> tx.Optional[bytes]:
    """Return the description of a header, or `None` if unreadable."""
    try:
        if isinstance(header, nb.Nifti1Image):
            header = header.header
        return bytes(np.asarray(header["descrip"]).item()).rstrip(b"\0")
    except Exception:  # noqa: BLE001  an unreadable header has none
        return None


def _grid_affines(
    header: tx.Optional[nb.Nifti1Header],
) -> tx.Tuple[np.ndarray, np.ndarray]:
    """Return the qform and the sform that place the grid of a flow field.

    SPM reads the qform as `mat0`, and the sform as `mat`, or the qform
    when the sform code is 0 (`@nifti/subsref.m`). A field without a
    header has the identity for both.

    Raises
    ------
    ParserContentError
        If the qform code is 0. SPM then guesses an orientation from its
        defaults (`decode_qform0.m`), which brainhops does not reproduce.
    """
    if header is None:
        return np.eye(4), np.eye(4)
    if int(header["qform_code"]) <= 0:
        raise ParserContentError(
            "SPM places the grid of a DARTEL flow field with its qform "
            "(mat0), and this header has no qform."
        )
    qform = np.asarray(header.get_qform(), dtype=np.float64)
    if int(header["sform_code"]) <= 0:
        return qform, qform
    return qform, np.asarray(header.get_sform(), dtype=np.float64)


def _velocity(data: tx.Optional[ArrayProtocol]) -> ArrayProtocol:
    """Return the stored velocity, checked.

    Raises
    ------
    ParserContentError
        If there is no velocity, or if it is not a 3-D grid of 3-vectors.
    """
    if data is None:
        raise ParserContentError("This field has no velocity to read.")
    if len(data.shape) != _NDIM + 1 or int(data.shape[-1]) != _NDIM:
        shape = tuple(int(d) for d in data.shape)
        raise ParserContentError(
            f"A DARTEL flow field is stored as a (X, Y, Z, 1, 3) array, and "
            f"its velocity has the shape {shape}."
        )
    return data


def velocity_refusal(velocity: _xforms.Transformation) -> tx.Optional[str]:
    """Return why DARTEL would integrate a velocity into another map.

    DARTEL stores the values of a stationary velocity, interpolates them
    linearly and wraps them around their grid. The reason ends a sentence
    that names the object, and it is `None` when the velocity can be
    stored.
    """
    if not isinstance(velocity, _xforms.StationaryVelocityField):
        return (
            f"it holds a {type(velocity).__name__} where a DARTEL flow field "
            f"holds a stationary velocity, and brainhops computes no "
            f"logarithm of a field."
        )
    degree = int(velocity.degree)
    bound = BoundaryCondition(velocity.bound)
    if degree != 1 or bound is not DARTEL_BOUND:
        return (
            f"its velocity has degree {degree} and bound {bound!s}, and "
            f"DARTEL interpolates a velocity linearly and wraps it around "
            f"its grid (degree 1, bound grid-wrap)."
        )
    return None


def qform_refusal(qform: np.ndarray) -> tx.Optional[str]:
    """Return why a qform cannot hold the affine that places a grid.

    A qform holds a rotation, possibly with a flip of the third axis,
    zooms and a translation, stored in single precision. An affine with
    shears changes when it is stored, by much more than the rounding to
    single precision. The reason ends a sentence that names the object,
    and it is `None` when a qform holds the affine.
    """
    header = nb.Nifti1Header()
    header.set_qform(qform, code=_XFORM_CODE)
    stored = header.get_qform()
    scale = max(float(np.abs(qform).max()), 1.0)
    if np.allclose(stored, qform, rtol=0, atol=1e-5 * scale):
        return None
    return (
        "the affine that places its grid is not a rotation with zooms and "
        "a translation, which is all that a qform holds, and DARTEL places "
        "the grid with its qform (mat0)."
    )


def _dartel_encoded(
    field: DartelFlowField,
    like: tx.Any,
    overrides: tx.Mapping[str, tx.Any],
) -> tx.Tuple[_NiftiObject, int]:
    """Encode a flow field over a copy of its record.

    The stored velocity, assigned or read, is kept when the chain was not
    assigned, and the chain is encoded otherwise. The steps that the copy
    reads the file with are returned with the image.

    Raises
    ------
    UnrepresentableTransformationError
        If the chain is not one that DARTEL writes.
    WriterError
        If the field has no velocity.
    """
    if field._data is not None:
        stored = _field_to_disk(field._data)
    else:
        stored = field.raw
    if field._transformations is None:
        if stored is None:
            raise WriterError(
                "This field has no velocity, so there is nothing to write."
            )
        image = _record_image(stored, field._record(), overrides=overrides)
        steps = field.steps
    else:
        image, steps = _chain_encoded(field, overrides)
    if field.metadata is None:
        # A new header is given the geometry that DARTEL writes: a qform and
        # an sform, both coded `Aligned`, and the description of a flow
        # field.
        header = image.header
        if field._transformations is None:
            header.set_qform(image.affine, code=_XFORM_CODE)
            header.set_sform(image.affine, code=_XFORM_CODE)
        header["descrip"] = DARTEL_DESCRIP
    _apply_like(image, like)
    _apply_overrides(image, overrides)
    return image, steps


def _chain_encoded(
    field: DartelFlowField, overrides: tx.Mapping[str, tx.Any]
) -> tx.Tuple[_NiftiObject, int]:
    """Encode the assigned chain of a flow field over a copy of its record.

    Raises
    ------
    UnrepresentableTransformationError
        If the chain does not have the three slots of a flow field, or if
        its velocity or its first affine cannot be stored.
    WriterError
        If the velocity has no data.
    """
    chain = tuple(field.transformations)
    if len(chain) != 3:
        raise UnrepresentableTransformationError(
            f"{_WHAT} is written from a chain of three transformations: RAS "
            f"to voxel, a stationary velocity, and voxel to RAS."
        )
    velocity = chain[1]
    reason = velocity_refusal(velocity)
    if reason is not None:
        raise UnrepresentableTransformationError(
            f"{_WHAT} cannot hold this chain: {reason}"
        )
    values = velocity.to(store=StoreEnum.values).data
    if values is None:
        raise WriterError(
            "This field has no velocity, so there is nothing to write."
        )
    qform = np.linalg.inv(homogeneous_matrix(chain[0], _WHAT, _NDIM))
    sform = homogeneous_matrix(chain[2], _WHAT, _NDIM)
    reason = qform_refusal(qform)
    if reason is not None:
        raise UnrepresentableTransformationError(
            f"{_WHAT} cannot hold this chain: {reason}"
        )
    backend = get_array_backend(values)
    stored = _field_to_disk(backend.asarray(values))
    image = _record_image(stored, field._record(), sform, overrides)
    image.header.set_qform(qform, code=_XFORM_CODE)
    image.header.set_sform(sform, code=_XFORM_CODE)
    return image, int(velocity._compute_steps)
