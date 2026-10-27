"""AFNI nonlinear warps stored as NIfTI files (`3dQwarp -prefix x.nii`)."""

# stdlib
import re

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, KwOnly, NoRepr, replace

# internals
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.errors import UNSET, ErrorPolicy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.common.afni._constants import DICOM_TO_RAS
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _header,
    _keep_given_fields,
    _nifti_shape,
    _NiftiObject,
    _record_image,
    _split_options,
    _stored_parts,
)
from brainhops.io.common.nifti._raw import _sniffed_header
from brainhops.io.images.afni import AfniImage
from brainhops.io.metadata._base import (
    metadata_to_write,
    overruled_fields,
    requested_fields,
)
from brainhops.io.transformations.base._conversions import (
    FIELD_SOURCES,
    format_options,
    held,
    register_converter,
)
from brainhops.io.transformations.nifti import NiftiBasedTransformation
from brainhops.io.transformations.nifti._base import _consolidated_metadata
from brainhops.io.transformations.nifti._views import (
    _field_to_disk,
    _field_to_model,
)

# this format
from ._converters import afni_chain, afni_displacement
from ._formats import AfniWarpFormat
from ._geometry import _nifti_matrix, nifti_cardinal_matrix
from ._warp import (
    WARP_LABELS,
    _check_ras,
    _dicom_chain,
    _dicom_field,
    _dicom_vectors,
    is_warp_labels,
)

_NIFTI_ECODE_AFNI = 4
"""The code of the NIfTI header extension of AFNI (`NIFTI_ECODE_AFNI`)."""

_WHAT = "An AFNI NIfTI warp"
"""The name of the format in the messages of its errors."""

_XFORM_SCANNER_ANAT = 1
"""The code of the sform and the qform that are written when none is
known."""

_BRICK_LABS = re.compile(
    r"atr_name\s*=\s*[\"']BRICK_LABS[\"'][^>]*>\s*[\"']([^\"']*)[\"']"
)
_NIFTI_NUMS = re.compile(r"NIfTI_nums\s*=\s*[\"']([^\"']*)[\"']")


@register_format
class AfniNiftiWarp(
    AfniWarpFormat,
    NiftiBasedTransformation,
    _xforms.DerivedSequence,
):
    """
    An AFNI nonlinear warp stored as a NIfTI file, such as the
    `<prefix>_WARP.nii.gz` of `3dQwarp`.

    AFNI writes a dataset of three sub-bricks as a `(X, Y, Z, 1, 3)` NIfTI
    array without an intent code, with its usual voxel-to-RAS sform and
    qform, and with the displacements unchanged: they are in DICOM (LPS)
    millimetres, `x`, `y` and `z`, and not in RAS, unlike a NIfTI
    `DISPVECT` field. See [`AfniWarpFormat`][] for what they mean. The grid is
    the cardinal grid that AFNI derives from the qform, or else the sform,
    as `3dNwarpApply` does when it reads the file. It is the stored grid
    unless that grid is oblique.

    The warp holds the header of its file as
    [`NiftiMetadata`][brainhops.io.common.nifti.NiftiMetadata] and the
    array of the file in `rawdata`, as every NIfTI transformation does.
    The data is the array without its singleton axis, and the chain is
    decoded from it and from the header on first access. Values and chains
    that are assigned are pending until [`consolidate`][] encodes them
    into a copy, or until the warp is written.

    The header has nothing that says "AFNI warp" but the AFNI extension
    that AFNI writes into every NIfTI file, unless `AFNI_NIFTI_NOEXT` is
    set. When the labels of the sub-bricks in that extension are
    `x_delta`, `y_delta` and `z_delta`, the file is claimed with
    certainty. Without them, the same array is also what other software
    writes, such as a field of RAS coordinates or displacements or an ITK
    field, so the file is not claimed, and it is read with
    `hint="afni.warp"` or `hint="afni.warp.nifti"`, or with this class
    directly.

    A warp is written with its displacements in the layout of AFNI. The
    AFNI extension of the header is kept when it still describes the
    file, since AFNI warns when its `NIfTI_nums` do not match the file,
    and a minimal extension with the labels of a warp replaces it
    otherwise.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nifti",)

    PRIORITY: tx.ClassVar[int] = -1
    """
    Priority below the NIfTI field of RAS displacements.

    A `.nii.gz` name cannot tell an AFNI warp from a NIfTI `DISPVECT`
    field, and both hold a field of RAS displacements exactly. `io.save`
    writes such a field as a
    [`NiftiRASDisplacementField`][brainhops.io.transformations.nifti.NiftiRASDisplacementField],
    unless it already is an AFNI warp, which is built with
    `AfniNiftiWarp.from_any(t)`. A file is read as an AFNI warp from its
    extension, whose score decides before the priority.
    """

    degree: tx.ClassVar[int] = 1
    """The spline degree of the interpolation."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition outside the field of view."""

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The sub-bricks given to the warp, or `None` to decode them.

    Sub-bricks stored here are pending until [`consolidate`][] or a write
    encodes them into `rawdata`.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The sub-bricks as an (X, Y, Z, n) array, in DICOM millimetres.

        Sub-bricks that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them. Otherwise
        they are decoded from `rawdata`: AFNI stores the sub-bricks with shape
        (X, Y, Z, 1, n), and the singleton axis is dropped. Assigning the
        data drops the chain decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return _field_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the chain decoded from the previous
    header.
    """

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an AFNI warp.

        A stream that starts with a NIfTI header in the layout of AFNI is
        read again with its extensions, which the labels of the
        sub-bricks are in, and scored with `_score_nibabel`. Any other
        NIfTI header scores `Confidence.NO`, and a stream that holds no
        NIfTI header is declined by
        [`NiftiRaw.sniff_fileobj`][brainhops.io.common.nifti.NiftiRaw.sniff_fileobj].
        The position of the stream is restored.

        Parameters
        ----------
        file : IO
            The stream, which is read from its current position.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no NIfTI
            header.
        **kwargs : Any
            Options of the reader. The NIfTI version is read from
            `version`.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        header = _sniffed_header(file, kwargs.get("version"))
        if header is None:
            return NiftiRaw.sniff_fileobj(file, error=error, **kwargs)
        bricks = _bricks(_nifti_shape(header))
        if bricks is None or bricks < 3:
            return Confidence.NO
        try:
            with preserve_position(file):
                record = NiftiRaw.from_fileobj(file, check=False)
        except Exception:  # noqa: BLE001
            return Confidence.NO
        return cls._score_nibabel(record.header)

    @classmethod
    def from_filename(cls, filename: tx.Any, **kwargs) -> "AfniNiftiWarp":
        """Read a warp from the path of a NIfTI file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the nibabel readers, such as `mmap`.

        Returns
        -------
        AfniNiftiWarp
            The warp, whose array is read lazily.

        Raises
        ------
        ParserContentError
            If the file does not hold at least three sub-bricks.
        """
        return _checked(super().from_filename(filename, **kwargs))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "AfniNiftiWarp":
        """Read a warp from an open NIfTI stream.

        Parameters
        ----------
        file : BinaryIO
            The stream, which must stay open while the data may be read.
        **kwargs : Any
            Options of the nibabel readers.

        Returns
        -------
        AfniNiftiWarp
            The warp, whose array is read lazily.

        Raises
        ------
        ParserContentError
            If the file does not hold at least three sub-bricks.
        """
        return _checked(super().from_fileobj(file, **kwargs))

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> "AfniNiftiWarp":
        """Build a warp from a nibabel image or header.

        Parameters
        ----------
        nifti : nibabel.Nifti1Image or nibabel.Nifti1Header
            The nibabel image or header.
        **kwargs : Any
            Other fields of the warp.

        Returns
        -------
        AfniNiftiWarp
            The warp.

        Raises
        ------
        ParserContentError
            If the image does not hold at least three sub-bricks.
        """
        return _checked(super().from_nibabel(nifti, **kwargs))

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a header as an AFNI warp.

        The layout of AFNI, with an AFNI extension that labels the
        sub-bricks as those of a warp, scores `Confidence.CERTAIN`, and
        any other header `Confidence.NO`.
        """
        bricks = _bricks(_nifti_shape(header))
        if bricks is None or bricks < 3:
            return Confidence.NO
        if is_warp_labels(_extension_labels(header)):
            return Confidence.CERTAIN
        return Confidence.NO

    # --- endpoints ----------------------------------------------------

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space that the warp maps from: the RAS world of the base."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space that the warp maps to: the RAS world of the source."""
        return _systems.RASmm()

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations that the warp encodes.

        The chain is decoded from the cardinal grid that AFNI derives from
        the header and from the first three sub-bricks on first access,
        and cached. Its displacements in voxel units are read-only.
        Assigning a chain overrides the decoded one, and the chain is
        pending until [`consolidate`][] or a write encodes it.
        """
        return _dicom_chain(_dicom_vectors(self.data), _vox2dicom(self))

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
        """Return the warp as an AFNI image, which is what AFNI stores.

        The image has three `float32` sub-bricks, the `x`, `y` and `z`
        displacements in DICOM (LPS) millimetres, on the cardinal grid of
        the warp, labelled `x_delta`, `y_delta` and `z_delta`. Saved as an
        AFNI dataset, it is read back as a warp.

        Returns
        -------
        AfniImage
            The consolidated image.

        Raises
        ------
        WriterError
            If the chain is not a displacement field between two affines.
        UnrepresentableTransformationError
            If the grid of the chain is oblique, since AFNI places a warp
            on a cardinal grid.
        """
        vox2dicom, vectors = _dicom_field(self.transformations, _WHAT)
        affine = _xforms.Affine(
            matrix=vox2dicom[:3],
            input=_systems.VoxelCoordinateSystem(),
            output=_systems.LPSmm(name="orig"),
        )
        image = AfniImage(
            data=np.asarray(vectors, dtype=np.float32),
            transformations=[affine],
        )
        labels = {"BRICK_LABS": "\0".join(WARP_LABELS)}
        return image.consolidate(datatype="float", attributes=labels)

    # --- writing ------------------------------------------------------

    @takes_error_policy
    def consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        on_error: ErrorPolicy = UNSET,
        **options: tx.Any,
    ) -> "AfniNiftiWarp":
        """Encode the pending sub-bricks or chain into `rawdata` and a header.

        A warp whose chain was not assigned keeps its sub-bricks, assigned
        or read, under a copy of the header of its metadata, whose
        geometry is kept. A warp whose chain was assigned is encoded from
        the chain: its displacements are written in DICOM millimetres as a
        `float32` `(X, Y, Z, 1, 3)` array, with the cardinal grid of the
        chain as sform and qform, whose codes are kept from the header,
        or else are scanner-anatomical. Non-encoding header fields are
        copied from `like`, and `overrides` are applied last. The AFNI
        extension of the header is then kept if it still describes the
        file, and replaced by a minimal one that labels the sub-bricks
        otherwise. A warp that has a header and nothing pending,
        consolidated without options, is left as it is.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the options that name its fields, and its record is
        the base of the new header. The fields given by keyword take
        precedence over `like`, and a `dtype` that is given, or changed
        in the metadata, is the type of the stored array. What the write
        loses or approximates is acted on once, by `on_error`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this warp rather than a copy.
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        on_error : ErrorPolicy, optional
            What to do with the metadata that the header cannot hold,
            and with any other conversion error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        **options : Any
            Fields of the metadata to override, such as `description` or
            `dtype`, and header fields, which take precedence over the
            derived values and `like`. A header field that holds a field
            of the metadata, such as `descrip`, is refused.

        Returns
        -------
        AfniNiftiWarp
            The consolidated copy, or this warp with `inplace=True`.

        Raises
        ------
        TypeError
            If an option names a header field that holds a field of the
            metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If the warp has no values, or its chain is not a displacement
            field between two affines.
        UnrepresentableTransformationError
            If the chain, or an assigned `input` or `output`, does not map
            RAS to RAS, or the grid of the chain is oblique.
        """
        fields, overrides = _split_options(options)
        report = ConversionReport(
            source="AfniNiftiWarp", target="AfniNiftiWarp"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, NiftiMetadata, fields, report
        )
        declared = self._input is not None or self._output is not None
        if declared:
            _check_ras(self.input, self.output, _WHAT)
        if (
            self._data is None
            and self._transformations is None
            and not declared
            and like is None
            and not overrides
            and not requested
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            apply_loss_policy(report, on_error, stacklevel=2)
            return self if inplace else replace(self)
        if "dtype" in requested:
            overrides["dtype"] = requested["dtype"]
        base = None if metadata is None else metadata.raw
        raw, record = _stored_parts(
            _encoded(self, like, overrides, base, fields)
        )
        metadata = _consolidated_metadata(self, record)
        overruled_fields(requested, metadata, report)
        apply_loss_policy(report, on_error, stacklevel=2)
        if not inplace:
            return replace(
                self,
                rawdata=raw,
                metadata=metadata,
                data=None,
                transformations=None,
                input=None,
                output=None,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.data = None
        self.transformations = None
        self.input = None
        self.output = None
        return self


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _checked(xform: AfniNiftiWarp) -> AfniNiftiWarp:
    """Return a warp read from a file that holds at least three sub-bricks.

    Raises
    ------
    ParserContentError
        If the file does not hold at least three sub-bricks in the layout
        of AFNI.
    """
    shape = _nifti_shape(_header(xform))
    bricks = _bricks(shape)
    if bricks is None or bricks < 3:
        raise ParserContentError(
            f"An AFNI warp is stored as a (X, Y, Z, 1, 3) array, not as an "
            f"array of shape {shape}."
        )
    return xform


def _vox2dicom(xform: AfniNiftiWarp) -> np.ndarray:
    """Return the cardinal voxel-to-DICOM matrix that AFNI reads.

    The matrix is derived from the qform, or else the sform, of the
    header, and is the identity for a warp without a header.
    """
    header = _header(xform)
    if header is None:
        return np.eye(4)
    real = DICOM_TO_RAS @ _nifti_matrix(header)
    return nifti_cardinal_matrix(real)


def _encoded(
    xform: AfniNiftiWarp,
    like: tx.Any,
    overrides: tx.Mapping[str, tx.Any],
    base: tx.Optional[NiftiRaw],
    fields: tx.Mapping[str, tx.Any],
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Encode a warp over a copy of a record.

    The record, `base`, is that of the consolidated metadata.

    The stored sub-bricks, assigned or read, are kept when the chain was
    not assigned, and the chain is encoded otherwise.

    Raises
    ------
    WriterError
        If the warp has no values.
    """
    if xform._transformations is None:
        if xform._data is not None:
            stored = _field_to_disk(xform._data)
        else:
            stored = xform.rawdata
        if stored is None:
            raise WriterError(
                "This warp has no values, so there is nothing to write."
            )
        record = None if base is None else base.copy()
        image = _record_image(stored, record, overrides=overrides)
    else:
        vox2dicom, vectors = _dicom_field(xform.transformations, _WHAT)
        spatial = tuple(int(d) for d in vectors.shape[:3])
        data = np.asarray(vectors, dtype=np.float32).reshape(spatial + (1, 3))
        vox2ras = DICOM_TO_RAS @ vox2dicom
        record = None if base is None else base.copy()
        scode, qcode = _xform_codes(None if record is None else record.header)
        image = _record_image(data, record, vox2ras, overrides)
        image.header.set_sform(vox2ras, code=scode)
        image.header.set_qform(vox2ras, code=qcode)
        image.header.set_xyzt_units("mm")
    _apply_like(image, like)
    _keep_given_fields(image, fields)
    _apply_overrides(image, overrides)
    _set_afni_extension(image.header)
    return image


def _xform_codes(header: tx.Any) -> tx.Tuple[int, int]:
    """Return the codes of the sform and the qform to store.

    The codes of the header are kept, and a header without one gets the
    scanner-anatomical code for both.
    """
    if header is not None and hasattr(header, "get_sform"):
        _, scode = header.get_sform(coded=True)
        _, qcode = header.get_qform(coded=True)
        if scode or qcode:
            return int(scode or qcode), int(qcode or scode)
    return _XFORM_SCANNER_ANAT, _XFORM_SCANNER_ANAT


def _afni_extension(header: _NiftiObject) -> tx.Optional[str]:
    """Return the text of the AFNI extension of a header, or `None`."""
    for extension in getattr(header, "extensions", None) or ():
        try:
            if int(extension.get_code()) != _NIFTI_ECODE_AFNI:
                continue
            content = extension.get_content()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(content, bytes):
            content = content.decode("latin-1")
        return str(content)
    return None


def _extension_labels(header: _NiftiObject) -> tx.Optional[tx.List[str]]:
    """Return the sub-brick labels that the AFNI extension stores, or `None`.

    The labels are the `BRICK_LABS` attribute, in which the NUL
    characters are written as `~`.
    """
    text = _afni_extension(header)
    match = _BRICK_LABS.search(text) if text else None
    if match is None:
        return None
    return match.group(1).split("~")


def _bricks(shape: tx.Optional[tx.Sequence[int]]) -> tx.Optional[int]:
    """Return the number of sub-bricks of a NIfTI file of AFNI, or `None`.

    AFNI writes a dataset of several sub-bricks that is not a time series
    with `dim[0] = 5` and its sub-bricks in `dim[5]`, which gives a shape
    `(X, Y, Z, 1, n)` (`thd_niftiwrite.c`). A file whose sub-bricks are on
    the time axis, `(X, Y, Z, n)`, is read in the same way by AFNI, and
    here.
    """
    if shape is None:
        return None
    shape = tuple(int(d) for d in shape)
    if len(shape) == 5 and shape[3] == 1:
        return shape[4]
    if len(shape) == 4:
        return shape[3]
    return None


def _nifti_nums(header: nb.Nifti1Header) -> str:
    """Return the `NIfTI_nums` of AFNI that describe a header.

    They are `nx,ny,nz,nt,nu,datatype`, which AFNI compares with the file
    that it reads.
    """
    shape = tuple(int(d) for d in header.get_data_shape())
    shape = (shape + (1,) * 5)[:5]
    return ",".join(str(d) for d in shape + (int(header["datatype"]),))


def _set_afni_extension(header: nb.Nifti1Header) -> None:
    """Keep the AFNI extension of a header if it describes it, or replace it.

    An extension whose `NIfTI_nums` match the header and that labels the
    sub-bricks of a warp is kept, and any other AFNI extension is
    replaced by a minimal one that holds those labels.
    """
    text = _afni_extension(header)
    nums = _nifti_nums(header)
    if text:
        match = _NIFTI_NUMS.search(text)
        if (
            match
            and match.group(1).replace(" ", "") == nums
            and is_warp_labels(_extension_labels(header))
        ):
            return
    kept = [
        extension
        for extension in header.extensions
        if int(extension.get_code()) != _NIFTI_ECODE_AFNI
    ]
    header.extensions[:] = kept
    header.extensions.append(
        nb.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, _niml_extension(nums))
    )


def _niml_extension(nums: str) -> bytes:
    """Return a minimal AFNI extension, which labels the sub-bricks of a
    warp.

    AFNI warns when the `NIfTI_nums` of the extension do not match the
    file (`thd_niftiread.c`), so they are written. The geometry is never
    taken from the extension, only from the NIfTI header.
    """
    labels = "~".join(WARP_LABELS)
    text = (
        "<?xml version='1.0' ?>\n"
        "<AFNI_attributes\n"
        f'  NIfTI_nums="{nums}"\n'
        '  ni_form="ni_group" >\n'
        "<AFNI_atr\n"
        '  ni_type="String"\n'
        '  ni_dimen="1"\n'
        '  atr_name="BRICK_LABS" >\n'
        f' "{labels}"\n'
        "</AFNI_atr>\n"
        "</AFNI_attributes>\n"
    )
    return text.encode("latin-1") + b"\0"


# ----------------------------------------------------------------------
#   CONVERTERS
# ----------------------------------------------------------------------
# The converters into this format live with it, because the format needs
# nibabel and the other AFNI formats do not. They are those of
# `AfniBrikWarp`, described in `brainhops.io.transformations.afni.
# _converters`.


@register_converter(FIELD_SOURCES, AfniNiftiWarp)
def _(
    t: _xforms.Transformation, cls: tx.Type[AfniNiftiWarp], **kwargs
) -> AfniNiftiWarp:
    # One field of displacements, between a world-to-grid affine and its
    # inverse, on a cardinal grid, sampled as AFNI stores it. Anything
    # else is turned into one, and what loses is reported.
    cls = AfniNiftiWarp
    options = format_options(t, cls, kwargs, ("metadata",))
    # The metadata of `t` rides along, converted by the field, as for the
    # NIfTI transformations. The header of another NIfTI format is left
    # out, since its intent would claim the file for that format.
    metadata = t.metadata
    if isinstance(metadata, NiftiMetadata) and metadata.raw is not None:
        metadata = replace(metadata, raw=None)
    options.setdefault("metadata", metadata)
    parts = afni_displacement(t, cls)
    return held(cls(transformations=afni_chain(parts), **options), parts)


@converter
def _(
    t: AfniNiftiWarp, cls: tx.Type[AfniNiftiWarp], **kwargs
) -> AfniNiftiWarp:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
