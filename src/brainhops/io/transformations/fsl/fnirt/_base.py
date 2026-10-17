import numpy as np
import typing_extensions as tx
from bagof.magic import Alias, Inherit, KwOnly, NoRepr, replace

from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._backends import _stored_data, dataproperty
from brainhops.datamodel.enums import StoreEnum
from brainhops.datamodel.images import Image
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common.nifti import NiftiMetadata
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _header,
    _keep_given_fields,
    _nifti_intent,
    _NiftiObject,
    _record_image,
    _split_options,
    _stored_parts,
)
from brainhops.io.images.nifti._views import (
    _image_to_disk,
    _image_to_model,
)
from brainhops.io.metadata._base import (
    metadata_to_write,
    overruled_fields,
    requested_fields,
)
from brainhops.io.transformations.base.fields import (
    StoredFieldMixin,
    voxel_grid_coordinates,
)
from brainhops.io.transformations.nifti import NiftiBasedTransformation
from brainhops.io.transformations.nifti._base import _consolidated_metadata
from brainhops.io.transformations.nifti._views import _field_to_model

from .._affines import _ImageGeometry
from .._fields import RASToWarpField, WarpFieldToRAS
from .._formats import FslTransformationFormat
from .._keywords import pop_aliased
from .._repr import stored_repr

# Intent codes of FSL, which its own copy of `nifti1.h` defines, as
# listed in `fsl/data/constants.py` of fslpy 3.29.1.
FSL_FNIRT_DISPLACEMENT_FIELD = 2006
FSL_CUBIC_SPLINE_COEFFICIENTS = 2007
FSL_DCT_COEFFICIENTS = 2008
FSL_QUADRATIC_SPLINE_COEFFICIENTS = 2009

_FNIRT_INTENTS = frozenset(
    {
        FSL_FNIRT_DISPLACEMENT_FIELD,
        FSL_CUBIC_SPLINE_COEFFICIENTS,
        FSL_DCT_COEFFICIENTS,
        FSL_QUADRATIC_SPLINE_COEFFICIENTS,
    }
)

_COEFFICIENT_INTENTS = frozenset(
    {
        FSL_CUBIC_SPLINE_COEFFICIENTS,
        FSL_QUADRATIC_SPLINE_COEFFICIENTS,
        FSL_DCT_COEFFICIENTS,
    }
)

# Deformation fields are linear B-splines on the grid of their header;
# coefficient fields are quadratic or cubic on a coarse knot grid.
_SPLINE_DEGREE = {
    0: 1,  # Native convertwarp/invwarp dense fields omit the FSL intent.
    FSL_FNIRT_DISPLACEMENT_FIELD: 1,
    FSL_QUADRATIC_SPLINE_COEFFICIENTS: 2,
    FSL_CUBIC_SPLINE_COEFFICIENTS: 3,
}

# A nibabel image or header, or a brainhops image.
_ImageLike = tx.Union[_NiftiObject, Image]


# Developer notes on the coefficient files (see also items 220 to 223 of
# spec/design/format-records.md):
# - The header conventions are those of `src/warpfns/fnirt_file_writer.cpp`,
#   `src/warpfns/fnirt_file_reader.cpp` and `src/basisfield/fsl_splines.h`
#   of FSL 5.0.9, as mirrored at https://github.com/neurodebian/fsl (commit
#   513f6a0), and they agree with `fsl/transform/fnirt.py` and
#   `fsl/transform/nonlinear.py` of fslpy 3.29.1
#   (https://github.com/pauldmccarthy/fslpy, commit 0eaff98).
# - A coefficient field is a `DisplacementField` that stores coefficients
#   because FSL evaluates the same centred B-splines.
# - FSL places the knots in the radiological order of the reference, so for
#   a reference whose voxel-to-world matrix has a positive determinant this
#   reader follows FSL, while fslpy places them from the stored voxels.
# - A reference of another resolution is placed by its scaled-mm positions,
#   as fslpy does, while FSL's reader evaluates the field on the grid stored
#   in the file (item 223).
# - FSL cannot write a DCT coefficient field (intent 2008) either.
@register_format
class FnirtWarpField(
    FslTransformationFormat,
    NiftiBasedTransformation,
    StoredFieldMixin,
    _xforms.ImmutableSequence,
):
    """Non-linear transformation stored in an FSL FNIRT NIfTI file.

    FNIRT saves the warp that it computes in two kinds of files, and the
    intent code of the file tells them apart. A deformation field, which
    FNIRT writes with `--fout`, stores for every reference voxel the
    matching moving location in scaled millimetres, or the displacement to
    it. A coefficient field, which FNIRT writes with `--cout`, stores the
    coefficients of a B-spline on a coarse grid of knots over the
    reference. Both are B-spline fields on a regular grid, so one reader
    handles them.

    | Intent | Kind | Degree | Grid |
    | ------ | ---- | ------ | ---- |
    | 0 | deformation (explicit `fnirt` hint) | 1 | header voxels |
    | 2006 | deformation | 1 | header voxels |
    | 2007 | cubic coefficients | 3 | knot grid |
    | 2009 | quadratic coefficients | 2 | knot grid |

    The transformation is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of three transformations from reference RAS to moving RAS: an affine to
    warp-grid coordinates, the field, and an affine to moving RAS. The field
    stays on its own grid and the spline is evaluated only when the sequence
    is computed. A coefficient field becomes a
    [`DisplacementField`][brainhops.datamodel.transformations.DisplacementField]
    that stores coefficients, and the two affines place its knots where FSL
    places them.

    A coefficient file uses its header in a way of its own. The pixdims
    hold the knot spacing in voxels of the reference, the intent parameters
    hold the voxel sizes of the reference, the offsets of the qform hold the
    shape of the reference, and the sform holds the initial affine from
    the source to the reference in scaled millimetres. The values are
    relative displacements in scaled millimetres, which FSL adds to the
    reference position moved by the inverse of the initial affine.
    Coefficients beyond the grid count as zero.

    A deformation field carries its grid geometry in its header. Its
    storage axes and resolution can differ from the reference image.
    Vector components stay in FSL scaled millimetres regardless of that
    storage layout. Without a separate reference image, the header also
    defines the reference geometry, so only the moving image is required.
    A coefficient field carries neither image's geometry, so both the
    reference and the moving image are required. A
    reference of another resolution than the one used at registration is
    placed by its scaled-mm positions. A discrete-cosine-transform
    coefficient field (intent 2008) is recognized but not supported.

    !!! note "Images remain external"
        A FNIRT file does not hold everything its map depends on: the moving
        image, whose geometry places the warped points in the world, is given
        separately, and so is whether a deformation field holds absolute or
        relative positions, which is otherwise guessed from the values. Save
        writes this FNIRT file, but those inputs are not embedded; retain them
        to read the same map back. A coefficient file keeps its knot spacing,
        the voxel sizes and the shape of its reference and its initial affine
        in its header, so a field read from such a file is written back as
        a coefficient file. A field built from an array alone is written as
        a deformation field.
    """

    HINTS = ("fnirt",)

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata")
    """The metadata of the file, which holds its header, or `None`.

    A field built from data has no metadata. The field is decoded from
    `raw` and from the header, and assigning other metadata keeps what
    was decoded.
    """

    moving: tx.Annotated[
        tx.Optional[_ImageLike], Alias(("moving", "mov", "src"))
    ] = None
    """The moving (source) image."""

    reference: tx.Annotated[
        tx.Optional[_ImageLike], Alias(("reference", "ref"))
    ] = None
    """The reference image, by default the geometry of a deformation field."""

    deformation_type: tx.Optional[str] = None
    """How a deformation field stores its values.

    `"absolute"` for positions, `"relative"` for displacements, or `None` to
    infer it from the data. Coefficient fields ignore it.
    """

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The array given to the field, or `None` to decode it from `rawdata`.

    An array stored here is pending until [`consolidate`][] or a write
    encodes it into `rawdata`.
    """

    data = dataproperty()
    """The array of the file, as FSL stores it.

    A deformation field has shape (X, Y, Z, 3), and a coefficient field
    holds the coefficients on its knot grid. An array that is assigned
    is returned as it is, and stays pending until [`consolidate`][] or
    a write encodes it. Otherwise the array is decoded from `rawdata`. The
    chain is built again from a new array.

    The first read converts the lazy array of the file with the
    backend setting in effect and caches the result. Each later read
    returns the cached array converted to the setting in effect then,
    such as `with backend("dask"):`, and the cached array does not
    change.
    """

    @property
    def dataobj(self) -> tx.Optional[ArrayProtocol]:
        """The array of the field in the layout of `data`, without reading it.

        An array assigned as `data` is returned as it is. Otherwise, the
        lazy array of the file is returned through the same view that
        `data` reads, and nothing is read until it is converted.
        """
        if self._data is not None:
            return self._data
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        if _nifti_intent(header) in _FNIRT_INTENTS:
            return Confidence.CERTAIN
        return Confidence.NO

    # --- image keyword handling ---------------------------------------
    # The readers forward their keywords to nibabel, which ignores
    # `moving=` and `reference=`, so these are set after parsing.

    @classmethod
    def _pop_images(
        cls, kwargs: tx.Dict[str, tx.Any]
    ) -> tx.Tuple[tx.Optional[_ImageLike], tx.Optional[_ImageLike]]:
        moving = pop_aliased(cls, kwargs, "moving")
        reference = pop_aliased(cls, kwargs, "reference")
        return moving, reference

    @classmethod
    def _with_images(
        cls,
        obj: tx.Self,
        moving: tx.Optional[_ImageLike],
        reference: tx.Optional[_ImageLike],
    ) -> tx.Self:
        if moving is not None:
            obj.moving = moving
        if reference is not None:
            obj.reference = reference
        return obj

    @classmethod
    def from_filename(cls, filename: tx.Any, **kwargs) -> tx.Self:
        """Read a FNIRT field from the path of a NIfTI file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The `moving` and `reference` images under any of their aliases,
            `deformation_type`, and the options of the nibabel readers.

        Returns
        -------
        FnirtWarpField
            The field, whose array is read lazily.
        """
        deformation_type = kwargs.pop("deformation_type", None)
        moving, reference = cls._pop_images(kwargs)
        obj = super().from_filename(filename, **kwargs)
        if deformation_type is not None:
            obj.deformation_type = deformation_type
        return cls._with_images(obj, moving, reference)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read a FNIRT field from an open NIfTI stream.

        Parameters
        ----------
        fileobj : BinaryIO
            The stream, which must stay open while the data may be read.
        **kwargs : Any
            The `moving` and `reference` images under any of their aliases,
            `deformation_type`, and the options of the nibabel readers.

        Returns
        -------
        FnirtWarpField
            The field, whose array is read lazily.
        """
        deformation_type = kwargs.pop("deformation_type", None)
        moving, reference = cls._pop_images(kwargs)
        obj = super().from_fileobj(fileobj, **kwargs)
        if deformation_type is not None:
            obj.deformation_type = deformation_type
        return cls._with_images(obj, moving, reference)

    def consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        on_loss: tx.Optional[OnLoss] = None,
        **options: tx.Any,
    ) -> tx.Self:
        """Encode the pending array into `rawdata` and a new header.

        FNIRT writes the field and the parameters of its spline in the
        header, which no other transformation can be converted to, so the
        array is kept under a copy of the header of the metadata, with its
        intent. A field built from an array alone is encoded as a
        deformation field (intent 2006) whose voxel-to-world affine is the
        identity. FNIRT cannot encode a chain, so consolidating a field
        whose chain was assigned raises an error. Non-encoding header fields
        are copied from `like`, and `overrides` are applied last. A field
        that has a header and nothing pending, consolidated without
        options, is left as it is.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the options that name its fields, and its record is
        the base of the new header. The fields given by keyword take
        precedence over `like`, and a `dtype` that is given, or changed
        in the metadata, is the type of the stored array. What the write
        loses or approximates is acted on once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this field rather than a copy.
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the header cannot hold. By
            default, the policy in effect.
        **options : Any
            Fields of the metadata to override, such as `description` or
            `dtype`, and header fields, which take precedence over the
            derived values and `like`. A header field that holds a field
            of the metadata, such as `descrip`, is refused.

        Returns
        -------
        Self
            The consolidated copy, or this field with `inplace=True`.

        Raises
        ------
        TypeError
            If an option names a header field that holds a field of the
            metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If the field has no array.
        UnrepresentableTransformationError
            If a chain was assigned to the field.
        """
        fields, overrides = _split_options(options)
        name = type(self).__name__
        report = ConversionReport(source=name, target=name)
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, NiftiMetadata, fields, report
        )
        if (
            self._data is None
            and self._transformations is None
            and like is None
            and not overrides
            and not requested
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            apply_loss_policy(report, on_loss, stacklevel=2)
            return self if inplace else replace(self)
        if "dtype" in requested:
            overrides["dtype"] = requested["dtype"]
        base = None if metadata is None else metadata.raw
        raw, record = _stored_parts(
            _fnirt_encoded(self, like, overrides, base, fields)
        )
        metadata = _consolidated_metadata(self, record)
        overruled_fields(requested, metadata, report)
        apply_loss_policy(report, on_loss, stacklevel=2)
        if not inplace:
            return replace(
                self,
                rawdata=raw,
                metadata=metadata,
                data=None,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.data = None
        return self

    # --- exposed parameters -------------------------------------------

    @property
    def degree(self) -> tx.Optional[int]:
        """The B-spline degree, or `None` for DCT fields."""
        header = _header(self)
        if header is None:
            return None
        return _SPLINE_DEGREE.get(_nifti_intent(header))

    @property
    def store(self) -> tx.Optional[StoreEnum]:
        """Whether the field stores spline coefficients or values."""
        header = _header(self)
        if header is None:
            return None
        return StoreEnum.from_coefficients(
            _nifti_intent(header) in _COEFFICIENT_INTENTS
        )

    # --- intent-driven behaviour --------------------------------------

    def _intent(self) -> tx.Optional[int]:
        header = _header(self)
        if header is None:
            return None
        return _nifti_intent(header)

    def _is_coeff(self) -> bool:
        return self._intent() in _COEFFICIENT_INTENTS

    def _degree(self) -> int:
        intent = self._intent()
        if intent == FSL_DCT_COEFFICIENTS:
            raise NotImplementedError(
                "FNIRT discrete-cosine-transform coefficient fields are not "
                "supported. Convert the field to a deformation field with "
                "FSL (`fnirtfileutils`) and read that instead."
            )
        degree = _SPLINE_DEGREE.get(intent)
        if degree is None:
            raise NotImplementedError(
                f"Unsupported FNIRT intent code {intent!r}."
            )
        return degree

    def _bound(self) -> tx.Union[str, float]:
        # FSL adds up only the splines whose coefficients exist, so the
        # coefficients beyond the grid count as zero (`Peek` and
        # `RangeOfSplines` in FSL, see `_ref_to_field`). A dense field is
        # never sampled outside, so nearest is harmless.
        return "constant" if self._is_coeff() else "nearest"

    def _stored_knot_spacing(self) -> np.ndarray:
        """Knot spacing in reference voxels, as stored in the pixdims.

        The pixdims are rounded to whole numbers of voxels, as FSL reads
        them.
        """
        # `FnirtFileReader::read_coef_file` in
        # `src/warpfns/fnirt_file_reader.cpp` of FSL 5.0.9.
        zooms = np.abs(np.asarray(_header(self).get_zooms()[:3], float))
        return np.floor(zooms + 0.5)

    def _reference_pixdim(self) -> np.ndarray:
        """Reference voxel sizes, stored in the intent parameters."""
        header = _header(self)
        return np.array(
            [
                float(header["intent_p1"]),
                float(header["intent_p2"]),
                float(header["intent_p3"]),
            ],
            dtype=np.float64,
        )

    def _has_spline_parameters(self) -> bool:
        """Whether the header gives usable knot and voxel sizes.

        A coefficient field cannot be placed without a positive knot
        spacing and positive voxel sizes of the reference. A dense field
        does not need them.
        """
        if not self._is_coeff():
            return True
        sizes = np.concatenate(
            [self._stored_knot_spacing(), self._reference_pixdim()]
        )
        return bool(np.all(np.isfinite(sizes) & (sizes > 0)))

    def _ref_to_field(self, ref: _ImageGeometry) -> np.ndarray:
        """Affine from reference voxels to the coordinates of the warp grid.

        A dense field uses the voxel-to-world geometry in its own header.
        Its storage axes and resolution can differ from the reference.
        The grid of a coefficient field is placed from the scaled-mm
        positions of the reference, divided by the knot spacing in
        millimetres and shifted by the offset of the knots of FSL.
        """
        # FSL places coefficient `c` of an axis at the voxel
        # `(c - degree // 2) * ksp` of the reference that it registered to,
        # or at voxel `c` when the knot spacing `ksp` is one
        # (`Spline1D::OffsetIntoField` in `src/basisfield/fsl_splines.h`).
        # These voxels are those of the image as FSL holds it in memory,
        # which FSL flips to the radiological order on read when the
        # voxel-to-world matrix has a positive determinant (`read_volumeROI`
        # in `src/newimage/generalio.cc`, and `read_volume` in
        # `src/fnirt/fnirt.cpp`), while it writes the coefficients without
        # any flip (`common_coef_construction` in
        # `src/warpfns/fnirt_file_writer.cpp`). A voxel in memory is
        # therefore the FSL scaled-mm position divided by the voxel size of
        # the reference. These files are those of FSL 5.0.9 in the
        # NeuroDebian mirror, https://github.com/neurodebian/fsl at commit
        # 513f6a0.
        #
        # A reference of another resolution than the one used at
        # registration is placed by its scaled-mm positions too, which
        # fslpy also assumes (`_readFnirtCoefficientField` in
        # `fsl/transform/fnirt.py` of fslpy 3.29.1). fslpy places the grid
        # from the stored voxels instead, which differs from FSL when the
        # reference has a positive determinant (item 220 of the memo).
        if not self._is_coeff():
            grid = _ImageGeometry(_header(self))
            return grid.ras2vox @ ref.vox2ras
        knot_mm = self._reference_pixdim() * self._stored_knot_spacing()
        grid = np.diag(np.concatenate([1.0 / knot_mm, [1.0]])) @ ref.vox2fsl
        grid[:3, 3] += _anchor_offsets(
            self._degree(), self._stored_knot_spacing()
        )
        return grid

    def _ref_to_source(self) -> np.ndarray:
        """The initial affine, from reference to source scaled millimetres.

        A deformation field already includes it, so it is the identity. A
        coefficient field stores its inverse in the sform, and
        `(inv(sform) - I) @ x` is added to the displacement at the
        reference position `x` in scaled millimetres.
        """
        # `add_affine_part` in `src/warpfns/fnirt_file_reader.cpp` of FSL
        # 5.0.9, and `coefficientFieldToDeformationField` in
        # `fsl/transform/nonlinear.py` of fslpy 3.29.1.
        if not self._is_coeff():
            return np.eye(4, dtype=np.float64)
        sform = np.asarray(_header(self).get_sform(), dtype=np.float64)
        if not np.all(np.isfinite(sform)):
            return np.eye(4, dtype=np.float64)
        return np.linalg.inv(sform)

    def _reference_geometry(self) -> _ImageGeometry:
        reference = self.reference
        if reference is None:
            if self._is_coeff():
                raise ValueError(
                    "A FNIRT coefficient field is stored on a coarse knot "
                    "grid, not on the reference grid, so the reference image "
                    "is needed to place the warp in world coordinates. Pass "
                    "reference=... when loading, or set it on the "
                    "transformation."
                )
            reference = _header(self)
        return _ImageGeometry(reference)

    def _raw_field(self) -> tx.Any:
        # A 5-D vector field in the layout of the NIfTI standard is
        # collapsed to `(*grid, 3)`.
        return _field_to_model(_stored_data(self))

    def _field_array(self, ref: _ImageGeometry) -> tx.Any:
        """The field on its own grid, as scaled millimetre displacements.

        Coefficients are returned unchanged and absolute deformations are made
        relative. The array backend is preserved.
        """
        field = self._raw_field()
        backend = get_array_backend(field)
        if self._is_coeff():
            return field

        shape = tuple(int(s) for s in field.shape[:3])
        grid_to_scaled = ref.vox2fsl @ np.linalg.inv(self._ref_to_field(ref))
        ref_scaled = _voxel_grid_in_scaled_mm(shape, grid_to_scaled, backend)

        deformation_type = self.deformation_type
        if deformation_type is None:
            deformation_type = _detect_deformation_type(field, ref_scaled)
        if deformation_type == "absolute":
            return field - ref_scaled
        if deformation_type == "relative":
            return field
        raise ValueError(
            'deformation_type must be "absolute", "relative" or None, '
            f"not {deformation_type!r}."
        )

    # --- the chain ----------------------------------------------------

    def _resolvable(self) -> bool:
        if _header(self) is None or self.moving is None:
            return False
        if self._intent() == FSL_DCT_COEFFICIENTS:
            return False
        if self._is_coeff() and self.reference is None:
            return False
        if not self._has_spline_parameters():
            return False
        # A coefficient field ignores `deformation_type`, so an invalid
        # value only matters for a deformation field.
        if not self._is_coeff() and self.deformation_type not in (
            None,
            "absolute",
            "relative",
        ):
            return False
        return True

    def _build_chain(self) -> tx.Tuple[_xforms.Transformation, ...]:
        if self.moving is None:
            raise ValueError(
                "A FNIRT warp maps to the moving image, whose geometry the "
                "warp file does not contain, so the moving image is needed "
                "to place the warp in world coordinates. Pass moving=... "
                "when loading, or set it on the transformation."
            )
        ref = self._reference_geometry()
        mov = _ImageGeometry(self.moving)
        if not self._has_spline_parameters():
            raise ValueError(
                "A FNIRT coefficient field stores its knot spacing in the "
                "pixdims and the voxel sizes of its reference in "
                "intent_p1 to intent_p3, and this header has a value that "
                "is not positive."
            )
        return _warp_chain(
            field=self._field_array(ref),
            degree=self._degree(),
            store=StoreEnum.from_coefficients(self._is_coeff()),
            bound=self._bound(),
            ref_to_field=self._ref_to_field(ref),
            ref=ref,
            mov=mov,
            ref_to_source=self._ref_to_source(),
        )

    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The transformations from reference RAS to moving RAS.

        The chain is built from the warp and both geometries, and raises when a
        required image is missing. It is cached until the images, the
        metadata, the array or `deformation_type` change, and it is a tuple
        so that the cache cannot be edited in place.
        """
        explicit = self._transformations
        if explicit is not None:
            return explicit
        if _header(self) is None:
            return ()
        key = self._chain_key()
        cached = getattr(self, "_chain_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        chain = self._build_chain()
        self._chain_cache = (key, chain)
        return chain

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.Sequence[_xforms.Transformation]]
    ) -> None:
        self._transformations = None if value is None else tuple(value)

    def _chain_key(self) -> tuple:
        return (
            id(self.moving),
            id(self.reference),
            id(self.metadata),
            id(self.rawdata),
            id(self._data),
            self.deformation_type,
        )

    def _cached_transformations(
        self,
    ) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain used by `len`, iteration and indexing.

        An assigned chain is returned first, otherwise the resolved chain, or
        an empty tuple when it cannot be built, so that inspection never
        raises.
        """
        explicit = self._transformations
        if explicit is not None:
            return explicit
        if not self._resolvable():
            return ()
        return self.transformations

    def __repr__(self) -> str:
        return stored_repr(
            self,
            ("degree", "store", "deformation_type", "moving", "reference"),
        )

    def __len__(self) -> int:
        return len(self._cached_transformations())

    def __iter__(self) -> tx.Iterator[_xforms.Transformation]:
        return iter(self._cached_transformations())

    def __getitem__(
        self, index: tx.Union[int, slice]
    ) -> _xforms.Transformation:
        return self._cached_transformations()[index]


def _fnirt_encoded(
    field: FnirtWarpField,
    like: tx.Any,
    overrides: tx.Mapping[str, tx.Any],
    base: tx.Any,
    fields: tx.Mapping[str, tx.Any],
) -> tx.Any:
    """Encode the array of a FNIRT field over a copy of a record.

    The record, `base`, is that of the consolidated metadata.

    Raises
    ------
    WriterError
        If the field has no array.
    UnrepresentableTransformationError
        If a chain was assigned to the field.
    """
    if field._transformations is not None:
        raise UnrepresentableTransformationError(
            "A FNIRT file stores a warp field and the parameters of its "
            "spline, so a chain assigned to the field cannot be written. "
            "Assign the field data instead."
        )
    if field._data is not None:
        stored = _image_to_disk(field._data)
    else:
        stored = field.rawdata
    if stored is None:
        raise WriterError("A FNIRT file needs its field data to be written.")
    record = None if base is None else base.copy()
    image = _record_image(stored, record, overrides=overrides)
    if _header(field) is None:
        image.header.set_intent(FSL_FNIRT_DISPLACEMENT_FIELD)
    _apply_like(image, like)
    _keep_given_fields(image, fields)
    _apply_overrides(image, overrides)
    return image


# ----------------------------------------------------------------------
#   CHAIN CONSTRUCTION
# ----------------------------------------------------------------------


def _warp_chain(
    field: np.ndarray,
    degree: int,
    store: tx.Any,
    bound: tx.Union[str, float],
    ref_to_field: np.ndarray,
    ref: _ImageGeometry,
    mov: _ImageGeometry,
    ref_to_source: np.ndarray,
) -> tx.Tuple[_xforms.Transformation, ...]:
    """Build the lazy reference-RAS to moving-RAS chain of a warp field.

    `ref_to_field` maps the voxels of the reference to the coordinates of
    the warp grid, at which the spline of the data model is evaluated, so
    it holds both the knot spacing and the offset of the knots of FSL. The
    field holds displacements in scaled millimetres, and FSL adds them to
    the reference position after the initial affine `ref_to_source` has
    moved that position (see `FnirtWarpField._ref_to_source`). The
    displacements are therefore expressed in units of the warp grid in the
    frame of the source, so that the affine at the end of the chain maps
    both the position and the displacement.
    """
    degree = int(degree)

    # Warp-grid coordinates to reference scaled millimetres.
    grid_to_fsl = ref.vox2fsl @ np.linalg.inv(ref_to_field)
    grid_to_fsl_lin = grid_to_fsl[:3, :3]
    grid_to_fsl_off = grid_to_fsl[:3, 3]
    src_lin = ref_to_source[:3, :3]
    src_off = ref_to_source[:3, 3]

    # Displacements in grid units of the source-aligned frame, so that the
    # initial affine applies to the base position.
    combined = src_lin @ grid_to_fsl_lin
    prescale = np.linalg.inv(combined)
    backend = get_array_backend(field)
    prescaled = backend.asarray(field) @ backend.asarray(prescale.T)

    ras_to_grid = ref_to_field @ ref.ras2vox

    fsl_to_ras = mov.fsl2ras
    grid_to_ras = np.eye(4, dtype=np.float64)
    grid_to_ras[:3, :3] = fsl_to_ras[:3, :3] @ combined
    grid_to_ras[:3, 3] = (
        fsl_to_ras[:3, :3] @ (src_lin @ grid_to_fsl_off + src_off)
        + fsl_to_ras[:3, 3]
    )

    return (
        RASToWarpField(matrix=ras_to_grid[:-1]),
        _xforms.DisplacementField(
            data=prescaled, degree=degree, bound=bound, store=store
        ),
        WarpFieldToRAS(matrix=grid_to_ras[:-1]),
    )


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _anchor_offsets(degree: int, knot_spacing: np.ndarray) -> np.ndarray:
    """Return the shift of the knots of FSL on each axis, in knots.

    FSL centres coefficient `c` at `(c - degree // 2) * ksp`, and at `c`
    when the knot spacing `ksp` is one, while the data model centres it at
    `c`. The shift is therefore `degree // 2` grid units, or zero where the
    stored knot spacing is one. A dense field has a knot spacing of one, so
    it has no shift.
    """
    # `Spline1D::OffsetIntoField` in `src/basisfield/fsl_splines.h` of FSL
    # 5.0.9.
    spacing = np.asarray(knot_spacing, dtype=np.float64)
    step = float(int(degree) // 2)
    return np.where(spacing == 1.0, 0.0, step).astype(np.float64)


def _voxel_grid_in_scaled_mm(
    shape: tx.Tuple[int, ...], vox2fsl: np.ndarray, backend: tx.Any
) -> tx.Any:
    """Scaled millimetre coordinates of every voxel of a grid."""
    return voxel_grid_coordinates(shape, vox2fsl, backend)


def _detect_deformation_type(field: tx.Any, ref_scaled: tx.Any) -> str:
    """Infer whether a deformation field stores positions or displacements.

    Positions grow with the voxel position and spread widely. The field is
    taken as absolute when it spreads more than its difference from the voxel
    coordinates.
    """
    backend = get_array_backend(field)
    axes = (0, 1, 2)
    std_absolute = float(backend.sum(backend.std(field, axis=axes)))
    std_relative = float(
        backend.sum(backend.std(field - ref_scaled, axis=axes))
    )
    return "absolute" if std_absolute > std_relative else "relative"
