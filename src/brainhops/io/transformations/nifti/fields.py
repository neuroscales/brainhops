"""
Fields of RAS coordinates and of RAS displacements, stored in NIfTI files.

Which intent code each one reads and writes, and why, is described in
the package docstring, [`brainhops.io.transformations.nifti`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import KwOnly, replace

# core
from brainhops._core.properties import smartproperty
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import (
    _NIFTI_INTENT_DISPVECT,
    _NIFTI_INTENT_NAME_MAPPING,
    _NIFTI_INTENT_NAME_NIFTYREG,
    _NIFTI_INTENT_VECTOR,
    NiftiParser,
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _nifti_intent,
    _nifti_intent_name,
    _nifti_shape,
    _nifti_vector_field,
    _NiftiObject,
)
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.transformations.base.conversions import (
    convert_instance,
    converts_to,
)
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    ras_displacement_chain,
    split_ras_displacement_chain,
)
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation

_NDIM = 3
"""The number of spatial dimensions the displacement reader supports."""


@register_format
class NiftiRASCoordinatesField(RASCoordinatesField, NiftiBasedTransformation):
    """
    Field of RAS coordinates, stored in a NIfTI file.
    """

    HINTS = ("coordinates",)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a NIfTI header as a field of RAS coordinates.

        `VECTOR` (1007) is the intent code this reader writes, and the
        one SPM writes for its `y_` coordinate maps, so it is claimed
        with certainty. `DISPVECT` (1006) is not claimed at all: the
        standard reserves it for displacements, which
        [`NiftiRASDisplacementField`][] reads. FSL intent codes are left
        to the FSL readers, which decode them.

        A `VECTOR` file in ITK's layout is also claimed with certainty by
        the ITK displacement reader, which reads its vectors as LPS,
        unless its intent name is `"Mapping"` -- which SPM and this
        reader write and ITK never does. A `VECTOR` file without that
        name says nothing of the frame of its vectors, so the two tie
        and a hint decides. See
        [`brainhops.io.transformations.itk.nifti`][].

        A file named `"NREG_TRANS"` is NiftyReg's, and is left to the
        NiftyReg readers ([`brainhops.io.transformations.niftyreg`][]).
        """
        intent = _nifti_intent(header)
        if _nifti_intent_name(header) == _NIFTI_INTENT_NAME_NIFTYREG:
            # NiftyReg's own fields: `intent_p1` says whether they hold
            # positions, displacements, spline coefficients or
            # velocities, which only the NiftyReg readers decode.
            return Confidence.NO
        if intent == _NIFTI_INTENT_VECTOR:
            return Confidence.CERTAIN
        if intent == _NIFTI_INTENT_DISPVECT:
            return Confidence.NO
        # The intent code is often left unset, so fall back to the
        # shape: a field of RAS coordinates carries a trailing axis of
        # length 3. A plain image has no such axis, and scores zero
        # rather than competing with the affine readers.
        shape = _nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.MAYBE
        return Confidence.NO

    # --- conversions --------------------------------------------------
    # Another transformation is converted, as `t.to(cls)` converts it;
    # anything else is read or copied as the bases do.

    @classmethod
    def from_other(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_other(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_instance(other, *args, **kwargs)

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The field of RAS coordinates, as an `(X, Y, Z, 3)` array.

        It is the image data that the NIfTI parser reads from the file,
        and the array this field stores: its `field` view reads it.
        NIfTI stores a vector field as `(X, Y, Z, 1, 3)`, with the
        components in the fifth axis, and SPM writes its `y_` fields that
        way. The singleton axis before the components is dropped, or the
        field would be sampled as a 4-D grid of 3-vectors.
        """
        # `CoordinatesField` exposes its stored `_data` as `data`, which
        # shadows the parser's lazy `data` property. The parser keeps the
        # image it reads in `_data` too, so the two are one value here,
        # read through the parser.
        data = NiftiParser.data.fget(self)
        if data is None:
            return None
        return _nifti_vector_field(data)

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        NiftiParser.data.fset(self, value)
        self._forget_views()

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that encodes this field of RAS coordinates.

        The field array becomes the NIfTI data array, and the header
        carries the `VECTOR` (1007) intent code, with SPM's intent name
        `"Mapping"`, that marks the file as a field of coordinates rather
        than a plain image. `DISPVECT` (1006) is not used: the standard
        reserves it for displacements, and ITK and brainhops read it as
        such. The voxel-to-RAS affine of the
        grid is taken from the source header when the field was read from
        one, and is the identity otherwise.

        The field array keeps its own array backend. A `cupy` or `dask`
        array is passed through rather than coerced into `numpy`. When
        `like` is given, non-encoding header fields are copied from it.
        Keyword arguments override header fields last, so an explicit value
        wins.
        """
        # NIfTI stores sampled coordinates.
        field = self.to(coeff=False).data
        if self.header is not None:
            affine = self.header.get_best_affine()
        else:
            affine = np.eye(4)
        image = ras_coordinates_nifti(field, affine)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


@register_format
class NiftiRASDisplacementField(
    _xforms.ImmutableSequence, NiftiBasedTransformation
):
    """
    Field of RAS displacements, stored in a NIfTI file.

    This is the `DISPVECT` (1006) field of the NIfTI-1 standard: each
    voxel holds the displacement, in RAS millimetres, of the point at
    its centre, and the field maps RAS to RAS as `x -> x + u(x)`. It is
    how ITK 5.4 and later reads and writes a `DISPVECT` image.

    A `DisplacementField` adds its values in the units of its own grid,
    so the field is the
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of three named slots:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    The displacements are interpolated linearly and extended with the
    nearest value outside the grid, as ITK's `DisplacementFieldTransform`
    does by default.

    The file may hold a stationary velocity instead, whose flow at time
    one is the map: the standard has no code for one, so it is said with
    the `log` option (`warp.nii.gz|displacements|log:true`, or its alias
    `warp.nii.gz|svf`; in Python, `load(path, log=True)`). The
    `displacement` slot is then a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.\
StationaryVelocityField], integrated with `steps` squaring steps
    (`|svf|steps:6`). The field is written in the encoding the options
    say: a velocity when `log` is set, and otherwise the displacement,
    which a velocity is integrated into.

    Another transformation is converted to this format by its converters
    ([`brainhops.io.transformations.nifti.converters`][]), exactly or not
    at all. Its chain is carried over, rather than re-read from a NIfTI
    header that comes with it and says something else (a NiftyReg file
    holds positions, say). Its encoding is not: `log` and `steps` are this
    format's options, so a velocity converted here is written as its
    displacement unless `log=True` is given -- to the conversion, or to
    `save`.
    """

    HINTS = ("displacements",)

    degree: tx.ClassVar[int] = 1
    """The spline degree used to interpolate the field."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition used outside of the field of view."""

    log: tx.Annotated[
        bool,
        tx.Doc(
            """
            Whether the file holds the stationary velocity whose flow is
            the map, rather than its displacement.
            """
        ),
        KwOnly(),
    ] = False

    steps: tx.Annotated[
        tx.Optional[int],
        tx.Doc(
            """
            The number of squaring steps that integrate a velocity (`log`
            only). If `None`, the default of
            [`StationaryVelocityField`][brainhops.datamodel.\
transformations.StationaryVelocityField].
            """
        ),
        KwOnly(),
    ] = None

    # --- reading ------------------------------------------------------
    # The NIfTI parser hands its keywords to `nibabel`, not to the
    # constructor, so the encoding options are taken out first and set
    # on the field read.

    @classmethod
    def from_file(cls, file: tx.Any, **kwargs) -> tx.Self:
        encoding = _pop_encoding(kwargs)
        return _with_encoding(super().from_file(file, **kwargs), encoding)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        encoding = _pop_encoding(kwargs)
        obj = super().from_fileobj(fileobj, **kwargs)
        return _with_encoding(obj, encoding)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a NIfTI header as a field of RAS displacements.

        `DISPVECT` (1006) is the code the standard reserves for
        displacements, so a file that carries it is claimed with
        certainty. Nothing else is: a field without it may as well hold
        coordinates, and is left to [`NiftiRASCoordinatesField`][].
        """
        if _nifti_intent(header) == _NIFTI_INTENT_DISPVECT:
            return Confidence.CERTAIN
        return Confidence.NO

    # --- conversions --------------------------------------------------
    # Another transformation is converted, as `t.to(cls)` converts it;
    # anything else is read or copied as the bases do.

    @classmethod
    def from_other(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_other(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_instance(other, *args, **kwargs)

    # --- endpoints ----------------------------------------------------
    #
    # Declared rather than read off the chain: reading them off the
    # chain would build it, and building it decodes the field data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the field maps from."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space the field maps to."""
        return _systems.RASmm()

    # --- decoding -----------------------------------------------------

    def _vox2ras(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-RAS affine of the field's grid."""
        header = self.header
        if header is None:
            raise ParserContentError(
                "This field has no NIfTI header to read its grid from."
            )
        return np.asarray(header.get_best_affine(), dtype=np.float64)

    def _ras_vectors(self) -> ArrayProtocol:
        """
        The stored displacements, as an `(X, Y, Z, 3)` array in RAS mm.

        The standard layout is five-dimensional, `(X, Y, Z, 1, 3)`, with
        the components in the fifth axis; its singleton axis is dropped.
        """
        data = self.data
        if data is None:
            raise ParserContentError("This field has no data to read.")
        backend = get_array_backend(data)
        data = backend.asarray(data)
        shape = tuple(int(d) for d in data.shape)
        data = _nifti_vector_field(data)
        if data.ndim != 4:
            raise ParserContentError(
                f"A NIfTI displacement field is stored as a (X, Y, Z, 1, 3) "
                f"array, not as an array of shape {shape}."
            )
        if data.shape[-1] != _NDIM:
            raise ParserContentError(
                f"Only three-dimensional displacement fields are "
                f"supported, and this one has {data.shape[-1]} components."
            )
        return data

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        It is built from the NIfTI header and data on first access, and
        cached. Assigning to it overrides the derived chain, which is how
        a field that was not read from a file is built. Either way it is a
        tuple, and the field refuses in-place edits: its slots name fixed
        positions in the chain, so a copy with other slots is made with
        `replace`.
        """
        return ras_displacement_chain(
            self._ras_vectors(),
            self._vox2ras(),
            degree=self.degree,
            bound=self.bound,
            log=self.log,
            steps=self.steps,
        )

    @property
    def ras2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def displacement(self) -> tx.Optional[_xforms.Transformation]:
        """The displacement field, in the voxel units of its grid (a
        velocity, with `log`)."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from the field's voxels back to RAS world."""
        return self.transformations[2]

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self,
        like: tx.Any = None,
        log: tx.Optional[bool] = None,
        **overrides,
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that encodes this field of displacements.

        The displacements are rotated from voxel units into RAS
        millimetres and written as a `DISPVECT` (1006) image of shape
        `(X, Y, Z, 1, 3)`, whose voxel-to-RAS affine is the grid's. With
        `log` (this field's own, unless one is given here, as in
        `save(path, log=True)`), the velocity is written instead;
        otherwise a velocity is integrated into its displacement.

        When `like` is given, non-encoding header fields are copied from
        it. Keyword arguments override header fields last.
        """
        what = "A NIfTI displacement field"
        log = self.log if log is None else log
        vox2ras, vectors = split_ras_displacement_chain(
            self.transformations, what, ndim=_NDIM, log=log
        )
        backend = get_array_backend(vectors)
        # NIfTI stores a vector field as a five-dimensional array, with
        # the components in the fifth axis.
        vectors = backend.expand_dims(vectors, axis=3)
        image = _new_nifti(vectors, vox2ras)
        image.header.set_intent(_NIFTI_INTENT_DISPVECT)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


def ras_coordinates_nifti(
    field: tx.Optional[ArrayProtocol], vox2ras: np.ndarray
) -> nb.Nifti1Image:
    """
    The `nibabel` image of a field of RAS coordinates.

    The field array becomes the NIfTI data array, and the header carries
    the `VECTOR` (1007) intent code, with SPM's intent name `"Mapping"`.
    Both [`NiftiRASCoordinatesField`][] and SPM's `y_` fields are written
    this way.

    Parameters
    ----------
    field : array, shape `(X, Y, Z, 3)` or `(X, Y, Z, 1, 3)`
        The coordinates, as values. The array keeps its backend.
    vox2ras : array, shape `(4, 4)`
        The voxel-to-RAS affine of the grid.
    """
    if field is None:
        raise WriterError(
            "This field has no coordinates, so there is nothing to write."
        )
    backend = get_array_backend(field)
    field = backend.asarray(field)
    if field.ndim == 4:
        # NIfTI stores a vector field as a five-dimensional array, with the
        # components in the fifth axis and a singleton axis before them. A
        # four-dimensional array would put the components in the time
        # axis, which the reader misreads.
        field = backend.expand_dims(field, axis=3)
    image = _new_nifti(field, vox2ras)
    image.header.set_intent(
        _NIFTI_INTENT_VECTOR, name=_NIFTI_INTENT_NAME_MAPPING
    )
    return image


def _pop_encoding(kwargs: tx.Dict[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    # The encoding options of a displacement field, taken out of `kwargs`.
    return {
        name: kwargs.pop(name) for name in ("log", "steps") if name in kwargs
    }


def _with_encoding(
    obj: NiftiRASDisplacementField, encoding: tx.Dict[str, tx.Any]
) -> NiftiRASDisplacementField:
    # The field read, with its encoding options. `steps` is the number of
    # squaring steps of a velocity, so it needs `log`.
    if encoding.get("steps") is not None and not encoding.get("log"):
        raise TypeError(
            "steps is the number of squaring steps of a velocity, so it is "
            "given with log=true (or the hint svf)."
        )
    return replace(obj, **encoding) if encoding else obj
