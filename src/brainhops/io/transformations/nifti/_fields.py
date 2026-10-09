"""
RAS coordinates fields and RAS displacement fields stored in NIfTI files.

The intent codes and the reasons for them are described in
[`brainhops.io.transformations.nifti`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import KwOnly, replace

# internals
from brainhops._core.properties import (
    InvalidatorInAttribute,
    smartproperty,
)
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.common.nifti._constants import (
    _NIFTI_INTENT_DISPVECT,
    _NIFTI_INTENT_NAME_MAPPING,
    _NIFTI_INTENT_NAME_NIFTYREG,
    _NIFTI_INTENT_VECTOR,
)
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _header,
    _nifti_intent,
    _nifti_intent_name,
    _nifti_shape,
    _NiftiObject,
)
from brainhops.io.common.nifti._views import _field_to_disk, _field_to_model
from brainhops.io.transformations.base._conversions import (
    convert_instance,
    converts_to,
)
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    ras_displacement_chain,
    split_ras_displacement_chain,
)

# this format
from ._base import NiftiBasedTransformation, _always, _record_image

_NDIM = 3
"""Spatial dimension supported by the displacement reader."""


_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


def _set_field_data(
    self: NiftiBasedTransformation, value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the data of a NIfTI vector field in `raw`, as NIfTI stores it.

    The cached data is dropped, and the views derived from it are dropped
    by the invalidation of the property.
    """
    self.raw = None if value is None else _field_to_disk(value)
    self.__dict__.pop("_cache_data", None)


def _set_coordinates_data(
    self: "NiftiRASCoordinatesField", value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the data of a field of coordinates in `raw`.

    The private field in which the data model stores its data is emptied,
    so that a copy made with `replace` takes the data from `raw`.
    """
    _set_field_data(self, value)
    self._data = None


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
        if isinstance(other, cls):
            # Within the format, `raw` and the metadata carry the data, so
            # the data is not read, and a proxy stays lazy in the copy.
            kwargs.setdefault("data", None)
        return super().from_instance(other, *args, **kwargs)

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        # The constructor stores `data=` in the private field of the data
        # model, which the view does not read, and the default of `raw`
        # comes after it. The array is therefore stored again, in `raw`.
        # When both are given, `data` takes precedence over `raw`.
        if arguments.get("data") is not None:
            self.data = arguments["data"]

    @smartproperty(
        cache=True,
        unset=_always,
        fset=_set_coordinates_data,
        invalidates=_FORGET_VIEWS,
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The field as an (X, Y, Z, 3) array, which its `field` view reads.

        NIfTI stores a vector field with shape (X, Y, Z, 1, 3). The data is
        decoded from `raw` without the singleton axis, since the array would
        otherwise be sampled as a 4-D grid of vectors. Setting the data
        stores it in `raw` with the singleton axis again.
        """
        if self.raw is None:
            return None
        return _field_to_model(self.raw)

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build a `nibabel` image that encodes the field.

        The header carries the `VECTOR` intent (1007) with the SPM intent name
        `"Mapping"`, which marks a coordinates field, rather than `DISPVECT`
        (1006), which ITK and brainhops read as displacements. The field
        holds no grid of its own, so a copy of the header of the metadata,
        with its voxel-to-RAS affine, is written with the coordinates, and a
        field without metadata gets the identity. The array is written as
        `raw` holds it, so a proxy that was read is copied as the file
        stored it, and a `cupy` or `dask` array is not converted to `numpy`.
        Non-encoding header fields are copied from `like`, and `overrides`
        are applied last.
        """
        # NIfTI stores sampled coordinates.
        if StoreEnum(self.store) is StoreEnum.values:
            raw = self.raw
        else:
            values = self.to(store="values").data
            raw = None if values is None else _field_to_disk(values)
        if raw is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        image = _record_image(raw, self.metadata, overrides=overrides)
        _set_coordinates_intent(image.header)
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
    ([`brainhops.io.transformations.nifti._converters`][]), exactly or not
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

    @smartproperty(
        cache=True,
        unset=_always,
        fset=_set_field_data,
        invalidates=("transformations",),
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The stored vectors as an (X, Y, Z, 3) array, decoded from `raw`.

        NIfTI stores the field with shape (X, Y, Z, 1, 3), and the singleton
        axis is dropped. The vectors are the RAS displacements of the file,
        or its velocity with `log`. Setting the data stores it in `raw` and
        drops the chain decoded from the previous data.
        """
        if self.raw is None:
            return None
        return _field_to_model(self.raw)

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the chain decoded from the previous
    header.
    """

    # --- reading ------------------------------------------------------
    # The readers hand their keyword arguments to nibabel, so the encoding
    # options are popped first and set on the field read.

    @classmethod
    def from_filename(cls, filename: tx.Any, **kwargs) -> tx.Self:
        encoding = _pop_encoding(kwargs)
        obj = super().from_filename(filename, **kwargs)
        return _with_encoding(obj, encoding)

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

    # --- copies -------------------------------------------------------

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

    # --- endpoints ----------------------------------------------------
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

    # --- decoding -----------------------------------------------------

    def _vox2ras(self) -> np.ndarray:
        """Return the (4, 4) voxel-to-RAS affine of the grid.

        The affine is the best affine of the header, or the identity for a
        field without metadata.
        """
        header = _header(self)
        if header is None:
            return np.eye(4)
        return np.asarray(header.get_best_affine(), dtype=np.float64)

    def _ras_vectors(self) -> ArrayProtocol:
        """
        Return the stored displacements, in RAS millimetres.

        A field whose array is not a grid of vectors with three components
        is refused.
        """
        data = self.data
        if data is None:
            raise ParserContentError("This field has no data to read.")
        if len(data.shape) != 4:
            shape = tuple(int(d) for d in self.raw.shape)
            raise ParserContentError(
                f"A NIfTI displacement field is stored as a (X, Y, Z, 1, 3) "
                f"array, not as an array of shape {shape}."
            )
        if int(data.shape[-1]) != _NDIM:
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

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self,
        like: tx.Any = None,
        log: tx.Optional[bool] = None,
        **overrides,
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image of the displacement field.

        The displacements are written in RAS millimetres under the `DISPVECT`
        intent (1006), with shape (X, Y, Z, 1, 3). With `log`, which defaults
        to the option of the field, the velocity is written instead, and
        otherwise a velocity is integrated first.

        A field whose chain is decoded from its file, in the encoding that it
        is written in, is written back as it is stored, under a copy of the
        header of its metadata, whose geometry is kept. A field whose chain
        was assigned is encoded from the chain over that header, whose
        geometry is replaced.
        Non-encoding header fields are copied from `like`, and `overrides`
        are applied last.
        """
        what = "A NIfTI displacement field"
        log = self.log if log is None else log
        stored = (
            self.raw is not None
            and getattr(self, "_transformations", None) is None
            and log == self.log
        )
        if stored:
            image = _record_image(self.raw, self.metadata, overrides=overrides)
            if _nifti_intent(image.header) != _NIFTI_INTENT_DISPVECT:
                image.header.set_intent(_NIFTI_INTENT_DISPVECT)
        else:
            vox2ras, vectors = split_ras_displacement_chain(
                self.transformations, what, ndim=_NDIM, log=log
            )
            vectors = _field_to_disk(vectors)
            image = _record_image(vectors, self.metadata, vox2ras, overrides)
            image.header.set_intent(_NIFTI_INTENT_DISPVECT)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image


def _set_coordinates_intent(header: nb.Nifti1Header) -> None:
    """Give a header the intent of a field of coordinates.

    The intent is `VECTOR` (1007) with the intent name `"Mapping"`. A
    header that already has this intent is left as it is, so that its
    intent parameters are kept.
    """
    intent = (_nifti_intent(header), _nifti_intent_name(header))
    if intent != (_NIFTI_INTENT_VECTOR, _NIFTI_INTENT_NAME_MAPPING):
        header.set_intent(
            _NIFTI_INTENT_VECTOR, name=_NIFTI_INTENT_NAME_MAPPING
        )


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
