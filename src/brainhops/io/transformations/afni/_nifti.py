"""AFNI nonlinear warps stored as NIfTI files (`3dQwarp -prefix x.nii`)."""

__all__ = ["AfniNiftiWarp"]

# stdlib
import re

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.io.base._base import register_format
from brainhops.io.base.afni import DICOM_TO_RAS
from brainhops.io.base.nifti import (
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _nifti_shape,
    _NiftiObject,
)
from brainhops.io.base.parsers import Confidence, ParserContentError
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation

from ._utils import _nifti_matrix, nifti_cardinal_matrix
from ._warp import WARP_LABELS, AfniWarp, is_warp_labels

_NIFTI_ECODE_AFNI = 4
"""The code of AFNI's NIfTI header extension (`NIFTI_ECODE_AFNI`)."""

_NIFTI_FLOAT32 = 16
"""The NIfTI datatype code of `float32`, as written in `NIfTI_nums`."""

_XFORM_SCANNER_ANAT = 1
"""The sform and qform code written when none is known."""

_BRICK_LABS = re.compile(
    r"atr_name\s*=\s*[\"']BRICK_LABS[\"'][^>]*>\s*[\"']([^\"']*)[\"']"
)
_NIFTI_NUMS = re.compile(r"NIfTI_nums\s*=\s*[\"']([^\"']*)[\"']")


def _afni_extension(header: _NiftiObject) -> tx.Optional[str]:
    """The text of AFNI's extension (NIML attributes), or `None`."""
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
    """The sub-brick labels stored in AFNI's extension (`BRICK_LABS`,
    NUL characters written as `~`), or `None`."""
    text = _afni_extension(header)
    match = _BRICK_LABS.search(text) if text else None
    if match is None:
        return None
    return match.group(1).split("~")


def _bricks(shape: tx.Optional[tx.Sequence[int]]) -> tx.Optional[int]:
    """
    The number of sub-bricks of a NIfTI file in AFNI's layout, or `None`.

    AFNI writes a dataset of several sub-bricks that is not a time series
    with `dim[0] = 5` and its sub-bricks in `dim[5]`: `(X, Y, Z, 1, n)`
    (`thd_niftiwrite.c`). A file whose sub-bricks are on the time axis,
    `(X, Y, Z, n)`, is read the same way by AFNI, and here.
    """
    if shape is None:
        return None
    shape = tuple(int(d) for d in shape)
    if len(shape) == 5 and shape[3] == 1:
        return shape[4]
    if len(shape) == 4:
        return shape[3]
    return None


def _niml_extension(shape: tx.Sequence[int]) -> bytes:
    """
    A minimal AFNI extension, with the labels of a warp's sub-bricks.

    AFNI warps when the extension's `NIfTI_nums`
    (`nx,ny,nz,nt,nu,datatype`) do not match the file
    (`thd_niftiread.c`), so they are written. The geometry is never
    taken from the extension, only from the NIfTI header.
    """
    nums = ",".join(str(int(d)) for d in (*shape[:3], 1, 3, _NIFTI_FLOAT32))
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


@register_format
class AfniNiftiWarp(AfniWarp, NiftiBasedTransformation):
    """
    An AFNI nonlinear warp stored as a NIfTI file, such as `3dQwarp`'s
    `<prefix>_WARP.nii.gz`.

    AFNI writes a three-sub-brick dataset as a `(X, Y, Z, 1, 3)` NIfTI
    array with no intent code, its usual voxel-to-RAS sform and qform,
    and the displacements unchanged: in DICOM (LPS) millimetres, `x`,
    `y`, `z` -- not RAS, unlike a NIfTI `DISPVECT` field. See
    [`AfniWarp`][] for what they mean. The grid is the cardinal grid
    AFNI derives from the qform (else the sform), as `3dNwarpApply` does
    when it reads the file; it is the stored grid unless that grid is
    oblique.

    **Telling it from other NIfTI fields.** The header has nothing that
    says "AFNI warp" but the AFNI extension AFNI writes into every NIfTI
    file (unless `AFNI_NIFTI_NOEXT` is set): when its sub-brick labels
    are `x_delta`, `y_delta`, `z_delta`, the file is claimed with
    certainty. Without them, the same array is also what other software
    writes (a field of RAS coordinates or displacements, an ITK field),
    so the file is not claimed: read it with `hint="afni.warp"` (or
    `"afni.warp.nifti"`), or with this class directly.

    **Writing** stores the displacements as `float32` in AFNI's
    `(X, Y, Z, 1, 3)` layout, with the cardinal grid as sform and qform
    (codes kept from the header read, else scanner-anatomical), and the
    AFNI extension of the header read when it still describes the file,
    else a minimal one holding the sub-brick labels.
    """

    HINTS = ("nifti",)

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """`CERTAIN` for AFNI's layout with an AFNI extension that
        labels the sub-bricks as a warp's; `NO` otherwise."""
        bricks = _bricks(_nifti_shape(header))
        if bricks is None or bricks < 3:
            return Confidence.NO
        if is_warp_labels(_extension_labels(header)):
            return Confidence.CERTAIN
        return Confidence.NO

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """Build the warp from a `nibabel` header or image, refusing one
        that does not hold at least three sub-bricks."""
        header = nifti.header if isinstance(nifti, nb.Nifti1Image) else nifti
        shape = _nifti_shape(header)
        bricks = _bricks(shape)
        if bricks is None or bricks < 3:
            raise ParserContentError(
                f"An AFNI warp is stored as a (X, Y, Z, 1, 3) array, not "
                f"as an array of shape {shape}."
            )
        return super().from_nibabel(nifti, **kwargs)

    def _vox2dicom(self) -> np.ndarray:
        if self.header is None:
            raise ParserContentError(
                "This warp has no NIfTI header to read its grid from."
            )
        real = DICOM_TO_RAS @ _nifti_matrix(self.header)
        return nifti_cardinal_matrix(real)

    def _dicom_vectors(self) -> ArrayProtocol:
        data = self.data
        if data is None:
            raise ParserContentError("This warp has no data to read.")
        backend = get_array_backend(data)
        data = backend.asarray(data)
        shape = tuple(int(d) for d in data.shape)
        bricks = shape[-1]
        data = backend.reshape(data, (*shape[:3], bricks))
        return data[..., :3]

    # --- writing ------------------------------------------------------

    def _xform_codes(self) -> tx.Tuple[int, int]:
        """The sform and qform codes to store."""
        header = self.header
        if header is not None and hasattr(header, "get_sform"):
            _, scode = header.get_sform(coded=True)
            _, qcode = header.get_qform(coded=True)
            if scode or qcode:
                return int(scode or qcode), int(qcode or scode)
        return _XFORM_SCANNER_ANAT, _XFORM_SCANNER_ANAT

    def _extension(self, shape: tx.Sequence[int]) -> bytes:
        """The AFNI extension to write: the one read, if it labels a warp
        and its `NIfTI_nums` still describe the file (AFNI warps when
        they do not), else a minimal one."""
        header = self.header
        text = None if header is None else _afni_extension(header)
        if text:
            match = _NIFTI_NUMS.search(text)
            nums = (*shape[:3], 1, 3, _NIFTI_FLOAT32)
            expected = ",".join(str(int(d)) for d in nums)
            if match and match.group(1).replace(" ", "") == expected:
                if is_warp_labels(_extension_labels(header)):
                    content = text.encode("latin-1")
                    return (
                        content if content.endswith(b"\0") else content + b"\0"
                    )
        return _niml_extension(shape)

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image AFNI would write for this warp.

        When `like` is given, non-encoding header fields are copied from
        it. Keyword arguments override header fields last.
        """
        vox2dicom, vectors = self._dicom_field()
        spatial = tuple(int(d) for d in vectors.shape[:3])
        data = np.asarray(vectors, dtype=np.float32).reshape((*spatial, 1, 3))
        vox2ras = DICOM_TO_RAS @ vox2dicom
        image = _new_nifti(data, vox2ras)
        scode, qcode = self._xform_codes()
        image.header.set_sform(vox2ras, code=scode)
        image.header.set_qform(vox2ras, code=qcode)
        image.header.set_xyzt_units("mm")
        image.header.extensions.append(
            nb.nifti1.Nifti1Extension(
                _NIFTI_ECODE_AFNI, self._extension(spatial)
            )
        )
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image
