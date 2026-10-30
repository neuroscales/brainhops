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
from brainhops.datamodel._internal_api import converter, smart_replace
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, ParserContentError
from brainhops.io.common.afni._internal_api import DICOM_TO_RAS
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw
from brainhops.io.common.nifti._internal_api import (
    NiftiObject,
    header_of,
    nifti_shape,
    sniffed_header,
)
from brainhops.io.images.nifti import NiftiImage
from brainhops.io.transformations.base._internal_api import (
    FIELD_SOURCES,
    format_options,
    held,
    register_converter,
)
from brainhops.io.transformations.nifti._internal_api import (
    NiftiImageBackedTransformation,
    consolidate_through,
    data_image,
    field_to_disk,
    field_to_model,
    image_with,
)

# this format
from ._converters import afni_chain, afni_displacement
from ._formats import AfniWarpFormat
from ._geometry import nifti_cardinal_matrix, nifti_matrix
from ._warp import (
    WARP_LABELS,
    check_ras,
    dicom_chain,
    dicom_field,
    dicom_vectors,
    is_warp_labels,
)

_NIFTI_ECODE_AFNI = 4
"""The code of the NIfTI header extension of AFNI (`NIFTI_ECODE_AFNI`)."""

_NIFTI_UNITS_MM = 2
"""The code of millimetres in `xyzt_units` (`NIFTI_UNITS_MM`)."""

_WHAT = "An AFNI NIfTI warp"
"""The name of the format in the messages of its errors."""

_BRICK_LABS = re.compile(
    r"atr_name\s*=\s*[\"']BRICK_LABS[\"'][^>]*>\s*[\"']([^\"']*)[\"']"
)
_NIFTI_NUMS = re.compile(r"NIfTI_nums\s*=\s*[\"']([^\"']*)[\"']")


@register_format
class AfniNiftiWarp(
    AfniWarpFormat,
    NiftiImageBackedTransformation,
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

    The warp keeps its file in a
    [`NiftiImage`][brainhops.io.images.nifti.NiftiImage], as it was read
    or as the warp was last consolidated, and `rawdata` and `metadata`
    read that image. The data is the array without its singleton axis,
    and the chain is decoded from it and from the header on first
    access. Values and chains that are assigned are pending until
    [`consolidate`][] encodes them into a copy, or until the warp is
    written. [`from_image`][] builds a warp from such an image, and
    [`to_image`][] returns the NIfTI image that a file of the warp
    holds. The AFNI dataset that AFNI itself writes is reached by
    converting the warp to an
    [`AfniBrikWarp`][brainhops.io.transformations.afni.AfniBrikWarp],
    as in `warp.to(AfniBrikWarp).to_image()`, which returns an
    [`AfniImage`][brainhops.io.images.afni.AfniImage].

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
        (X, Y, Z, 1, n), and the singleton axis is dropped. A file whose
        fourth axis is longer than one holds a field per volume, which
        is not supported yet, so reading its data raises
        `ParserContentError`. Assigning the data drops the chain decoded
        from the previous data.
        """
        if self.rawdata is None:
            return None
        return field_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None, var=True
    )

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
        sub-bricks are in, and scored with `score_header`. Any other
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
        header = sniffed_header(file, kwargs.get("version"))
        if header is None:
            return NiftiRaw.sniff_fileobj(file, error=error, **kwargs)
        bricks = _bricks(nifti_shape(header))
        if bricks is None or bricks < 3:
            return Confidence.NO
        try:
            with preserve_position(file):
                record = NiftiRaw.from_fileobj(file, check=False)
        except Exception:  # noqa: BLE001
            return Confidence.NO
        return cls.score_header(record.header)

    @classmethod
    def from_image(cls, image: NiftiImage, **options: tx.Any) -> tx.Self:
        """Build a warp from the NIfTI image that stores it.

        The readers call this method, so a file that does not hold at
        least three sub-bricks in the layout of AFNI is refused here.

        Parameters
        ----------
        image : NiftiImage
            The image, which is consolidated first if it has pending
            data or transformations.
        **options : Any
            The other fields of the warp.

        Returns
        -------
        AfniNiftiWarp
            The warp.

        Raises
        ------
        TypeError
            If the image is not a `NiftiImage`.
        ParserContentError
            If the image does not hold at least three sub-bricks.
        """
        return _checked(super().from_image(image, **options))

    @classmethod
    def from_nibabel(cls, nifti: NiftiObject, **kwargs) -> "AfniNiftiWarp":
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
    def score_header(cls, header: NiftiObject) -> float:
        """Score a header as an AFNI warp.

        The layout of AFNI, with an AFNI extension that labels the
        sub-bricks as those of a warp, scores `Confidence.CERTAIN`, and
        any other header `Confidence.NO`.
        """
        bricks = _bricks(nifti_shape(header))
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

    @smartproperty(cache=True, invalidated_by="_image")
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations that the warp encodes.

        The chain is decoded from the cardinal grid that AFNI derives from
        the header and from the first three sub-bricks on first access,
        and cached. Its displacements in voxel units are read-only.
        Assigning a chain overrides the decoded one, and the chain is
        pending until [`consolidate`][] or a write encodes it.
        """
        return dicom_chain(dicom_vectors(self.data), _vox2dicom(self))

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

    # --- writing ------------------------------------------------------

    def _consolidate(self, **options: tx.Any) -> tx.Self:
        """Encode the pending sub-bricks or chain through the stored image.

        A warp whose chain was not assigned keeps its sub-bricks,
        assigned or read, and the geometry of its header. A warp whose
        chain was assigned is encoded from the chain: its displacements
        are written in DICOM millimetres as a `float32` (X, Y, Z, 1, 3)
        array, and the cardinal grid of the chain replaces the geometry
        of the header, in millimetres. An `input` or an `output` that
        was assigned must be RAS, and it is dropped from the copy. AFNI
        warns when the `NIfTI_nums` of its extension do not match the
        file, so the AFNI extension of a header that is encoded again is
        replaced when it no longer describes the file.

        Raises
        ------
        WriterError
            If the chain is not a displacement field between two
            affines.
        UnrepresentableTransformationError
            If the chain, or an assigned `input` or `output`, does not
            map RAS to RAS, or the grid of the chain is oblique.
        """
        if self._input is not None or self._output is not None:
            check_ras(self.input, self.output, _WHAT)
        changes = {
            "data": None,
            "transformations": None,
            "input": None,
            "output": None,
        }
        intent = {}
        if self._transformations is None:
            image = data_image(self, field_to_disk)
        else:
            vox2dicom, vectors = dicom_field(self.transformations, _WHAT)
            raw = field_to_disk(np.asarray(vectors, dtype=np.float32))
            image = image_with(self, raw, DICOM_TO_RAS @ vox2dicom)
            # AFNI writes its grids in millimetres.
            intent = {"xyzt_units": _NIFTI_UNITS_MM}
        stored = self.metadata
        xform = consolidate_through(self, image, intent, changes, **options)
        if xform.metadata is not stored:
            xform.metadata = _with_afni_extension(xform.metadata)
        return xform


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
    shape = nifti_shape(header_of(xform))
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
    header = header_of(xform)
    if header is None:
        return np.eye(4)
    real = DICOM_TO_RAS @ nifti_matrix(header)
    return nifti_cardinal_matrix(real)


def _afni_extension(header: NiftiObject) -> tx.Optional[str]:
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


def _extension_labels(header: NiftiObject) -> tx.Optional[tx.List[str]]:
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


def _with_afni_extension(metadata: NiftiMetadata) -> NiftiMetadata:
    """Return metadata whose AFNI extension describes its header.

    An extension whose `NIfTI_nums` match the header and that labels the
    sub-bricks of a warp is kept, and the metadata is returned as it is.
    Any other AFNI extension is replaced, in a copy of the record, by a
    minimal one that holds those labels.
    """
    header = metadata.raw.header
    text = _afni_extension(header)
    nums = _nifti_nums(header)
    if text:
        match = _NIFTI_NUMS.search(text)
        if (
            match
            and match.group(1).replace(" ", "") == nums
            and is_warp_labels(_extension_labels(header))
        ):
            return metadata
    record = metadata.raw.copy()
    extensions = record.header.extensions
    extensions[:] = [
        extension
        for extension in extensions
        if int(extension.get_code()) != _NIFTI_ECODE_AFNI
    ]
    extensions.append(
        nb.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, _niml_extension(nums))
    )
    return type(metadata).from_raw(record)


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
