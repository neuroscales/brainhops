"""
RAS coordinates fields and RAS displacement fields stored in NIfTI files.

The intent codes and the reasons for them are described in
[`brainhops.io.transformations.nifti`][].
"""

import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import KwOnly, replace

from brainhops._core.properties import (
    InvalidatorInAttribute,
    smartproperty,
)
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.common.nifti import NiftiParser
from brainhops.io.common.nifti._constants import (
    _NIFTI_INTENT_DISPVECT,
    _NIFTI_INTENT_NAME_MAPPING,
    _NIFTI_INTENT_NAME_NIFTYREG,
    _NIFTI_INTENT_VECTOR,
)
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _nifti_intent,
    _nifti_intent_name,
    _nifti_shape,
    _nifti_vector_field,
    _NiftiObject,
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
"""Spatial dimension supported by the displacement reader."""


def _always(value: tx.Any) -> bool:
    """Treat every stored value as unset, so that the getter always runs."""
    return True


def _store_through_the_parser(
    self: tx.Any, value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the data where the parser keeps it, which the getter reads."""
    NiftiParser.data.fset(self, value)


@register_format
class NiftiRASCoordinatesField(RASCoordinatesField, NiftiBasedTransformation):
    """Field of RAS coordinates stored in a NIfTI file."""

    HINTS = ("coordinates",)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a header as a field of RAS coordinates.

        A `VECTOR` image (1007), the intent that this reader and SPM write,
        scores `Confidence.CERTAIN`. A `DISPVECT` image (1006) is left to
        [`NiftiRASDisplacementField`][], and a file named `"NREG_TRANS"` to
        [`brainhops.io.transformations.niftyreg`][]. The ITK displacement
        reader ties on a `VECTOR` image in the ITK layout unless its intent
        name is `"Mapping"`, so a hint decides (see
        [`brainhops.io.transformations.itk.nifti`][]). Any other file whose
        trailing axis has length 3 scores `Confidence.MAYBE`.
        """
        intent = _nifti_intent(header)
        if _nifti_intent_name(header) == _NIFTI_INTENT_NAME_NIFTYREG:
            # Only the NiftyReg readers decode what intent_p1 says the file
            # holds.
            return Confidence.NO
        if intent == _NIFTI_INTENT_VECTOR:
            return Confidence.CERTAIN
        if intent == _NIFTI_INTENT_DISPVECT:
            return Confidence.NO
        # The intent is often unset, so fall back on the shape. A plain image
        # scores zero rather than compete with the affine readers.
        shape = _nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.MAYBE
        return Confidence.NO

    # --- conversions --------------------------------------------------
    # Another transformation is converted, as `t.to(cls)` converts it;
    # anything else is read or copied as the bases do.

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_instance(other, *args, **kwargs)

    @smartproperty(
        # The getter reshapes what the parser stores, so it runs on every read.
        unset=_always,
        fset=_store_through_the_parser,
        # Assigning the data clears the views keyed on it.
        invalidates=InvalidatorInAttribute("derived_fields"),
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The field as an (X, Y, Z, 3) array, which its `field` view reads.

        NIfTI stores a vector field with shape (X, Y, Z, 1, 3). The singleton
        axis is dropped, since the array would otherwise be sampled as a 4-D
        grid of vectors.
        """
        # The parser keeps its image in _data, which is also the value of this
        # field, so the data is read through the parser.
        data = NiftiParser.data.fget(self)
        if data is None:
            return None
        return _nifti_vector_field(data)

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build a `nibabel` image that encodes the field.

        The header carries the `VECTOR` intent (1007) with the SPM intent name
        `"Mapping"`, which marks a coordinates field, rather than `DISPVECT`
        (1006), which ITK and brainhops read as displacements. The voxel-to-RAS
        affine comes from the source header, or is the identity. The array
        backend is kept, so a `cupy` or `dask` array is not converted to
        `numpy`. Non-encoding header fields are copied from `like`, and
        `overrides` are applied last.
        """
        # NIfTI stores sampled coordinates.
        field = self.to(store="values").data
        if field is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        backend = get_array_backend(field)
        field = backend.asarray(field)
        if field.ndim == 4:
            # A NIfTI vector field is 5-D, with a singleton axis before the
            # components. A 4-D array would put the components in the time
            # axis.
            field = backend.expand_dims(field, axis=3)
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
    Field of RAS displacements stored in a NIfTI file.

    The file holds a `DISPVECT` field (1006), in which each voxel holds the
    displacement, in RAS millimetres, of the point at its centre. The field
    maps RAS to RAS as `x -> x + u(x)`, which is how ITK 5.4 and later read and
    write `DISPVECT`. A displacement field adds its values in the units of its
    own grid, so this field is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of three named slots:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    The displacements are interpolated linearly and extrapolated with the
    nearest value outside the grid, as by the default
    `DisplacementFieldTransform` of ITK.

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
    """Spline degree of the interpolation."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """Boundary condition outside the field of view."""

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

    # The NIfTI parser hands its keyword arguments to nibabel, so the
    # encoding options are popped first and set on the field read.

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
        Score a header as a field of RAS displacements.

        Only a `DISPVECT` image (1006) is claimed, with `Confidence.CERTAIN`.
        Other images may hold coordinates and are left to
        [`NiftiRASCoordinatesField`][].
        """
        if _nifti_intent(header) == _NIFTI_INTENT_DISPVECT:
            return Confidence.CERTAIN
        return Confidence.NO

    # --- conversions --------------------------------------------------
    # Another transformation is converted, as `t.to(cls)` converts it;
    # anything else is read or copied as the bases do.

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create a field from an instance of a similar class.

        The chain of the other transformation is carried over rather than
        re-read from its NIfTI header, which may mean something else. The
        encoding options `log` and `steps` are not carried over, so a copied
        velocity is written as a displacement unless `log=True` is given.
        """
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        if not isinstance(other, NiftiRASDisplacementField):
            kwargs.setdefault("log", False)
            kwargs.setdefault("steps", None)
            if isinstance(other, _xforms.Sequence):
                kwargs.setdefault("transformations", tuple(other))
        return super().from_instance(other, *args, **kwargs)

    # The endpoints are declared rather than read off the chain, which would
    # decode the data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """
        The anatomical space that the field maps from, in RAS millimetres.
        """
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space that the field maps to, in RAS millimetres."""
        return _systems.RASmm()

    def _vox2ras(self) -> np.ndarray:
        """Return the (4, 4) voxel-to-RAS affine of the grid."""
        header = self.header
        if header is None:
            raise ParserContentError(
                "This field has no NIfTI header to read its grid from."
            )
        return np.asarray(header.get_best_affine(), dtype=np.float64)

    def _ras_vectors(self) -> ArrayProtocol:
        """
        Return the stored displacements, in RAS millimetres.

        The standard layout is (X, Y, Z, 1, 3), and its singleton axis is
        dropped. A field whose vectors do not have three components is refused.
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

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        The chain is built lazily from the header and the data, and then
        cached. Assigning a chain overrides the derived one, which is how
        fields that do not come from a file are built. Since the slots name
        fixed positions, a copy with other slots is made with `replace` rather
        than by editing the chain in place.
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
        """The affine from RAS world coordinates to the voxels of the field."""
        return self.transformations[0]

    @property
    def displacement(self) -> tx.Optional[_xforms.Transformation]:
        """
        The displacement field in voxel units, or the velocity with `log`.
        """
        return self.transformations[1]

    @property
    def voxel2ras(self) -> tx.Optional[_xforms.Transformation]:
        """
        The affine from the voxels of the field back to RAS world coordinates.
        """
        return self.transformations[2]

    def to_nibabel(
        self,
        like: tx.Any = None,
        log: tx.Optional[bool] = None,
        **overrides,
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image of the displacement field.

        The displacements are rotated to RAS millimetres and written under the
        `DISPVECT` intent (1006), with shape (X, Y, Z, 1, 3). With `log`, which
        defaults to the option of the field, the velocity is written instead,
        and otherwise a velocity is integrated first. Non-encoding header
        fields are copied from `like`, and `overrides` are applied last.
        """
        what = "A NIfTI displacement field"
        log = self.log if log is None else log
        vox2ras, vectors = split_ras_displacement_chain(
            self.transformations, what, ndim=_NDIM, log=log
        )
        backend = get_array_backend(vectors)
        # A NIfTI vector field is 5-D, with the components in the fifth axis.
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
    return {
        name: kwargs.pop(name) for name in ("log", "steps") if name in kwargs
    }


def _with_encoding(
    obj: NiftiRASDisplacementField, encoding: tx.Dict[str, tx.Any]
) -> NiftiRASDisplacementField:
    # steps only makes sense for a velocity, so it requires log.
    if encoding.get("steps") is not None and not encoding.get("log"):
        raise TypeError(
            "steps is the number of squaring steps of a velocity, so it is "
            "given with log=true (or the hint svf)."
        )
    return replace(obj, **encoding) if encoding else obj
