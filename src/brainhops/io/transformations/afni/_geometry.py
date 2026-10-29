"""
The geometry that the AFNI transformation formats share: the true and
the cardinal voxel-to-DICOM matrices of an image, and the map between
its cardinal and its true world coordinates.

AFNI programs such as `3dAllineate`, `3dvolreg`, `3dQwarp` and
`3dNwarpApply` compute on the *cardinal* grid of each dataset, which is
the `ijk_to_dicom` matrix built from `ORIENT_SPECIFIC`, `ORIGIN` and
`DELTA`. They ignore the obliquity of `IJK_TO_DICOM_REAL`
(`DSET_CMAT(dset, use_realaxes)`, with `use_realaxes = 0` unless asked,
in `3dAllineate.c`, and `IW3D_from_dataset` in `mri_nwarp.c`). The
matrices and warps that they write therefore map cardinal coordinates.
See [`brainhops.io.transformations.afni`][].
"""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.common.afni import AfniRaw
from brainhops.io.common.afni._internal_api import (
    DICOM_TO_RAS,
    afni_cardinal_matrix,
    afni_geometry_from_matrix,
    afni_voxel_to_dicom,
)


def nifti_cardinal_matrix(real: np.ndarray) -> np.ndarray:
    """
    Return the cardinal voxel-to-DICOM matrix that AFNI gives a grid that
    it reads from a NIfTI file, or from any format other than its own.

    `thd_niftiread.c` keeps the closest orientation of each voxel axis
    and the length of its column as the voxel size, as
    [`afni_geometry_from_matrix`][brainhops.io.common.afni._geometry.afni_geometry_from_matrix]
    does. It takes as origin the translation itself, along the DICOM axis
    that each voxel axis runs along, rather than its projection on the
    column, so the cardinal and the true matrices share their
    translation. The two are the same matrix unless the grid is oblique.

    Parameters
    ----------
    real : array, shape `(4, 4)`
        The true voxel-to-DICOM matrix.

    Returns
    -------
    array, shape `(4, 4)`
        The cardinal voxel-to-DICOM matrix.
    """
    real = np.asarray(real, dtype=np.float64)
    orient, _, delta = afni_geometry_from_matrix(real)
    origin = [float(real[code // 2, 3]) for code in orient]
    return afni_cardinal_matrix(orient, origin, delta)


def image_grids(image: tx.Any) -> tx.Tuple[np.ndarray, np.ndarray]:
    """
    Return the true and the cardinal (4, 4) voxel-to-DICOM matrices of an
    image, as AFNI sees them.

    Parameters
    ----------
    image : AfniRaw or AfniImage or nibabel image or header or Image
        An AFNI record, or an object whose metadata holds one, such as an
        [`AfniImage`][brainhops.io.images.afni.AfniImage], gives
        `IJK_TO_DICOM_REAL` and its own cardinal grid. A NIfTI header or
        image, from nibabel or a brainhops image whose metadata holds a
        NIfTI header, gives the form that AFNI reads, and the cardinal
        grid that AFNI derives from it. Any other brainhops image gives
        its preferred voxel-to-world transformation, and the cardinal
        grid that AFNI would derive from it on import.

    Returns
    -------
    real : array, shape `(4, 4)`
        The true voxel-to-DICOM matrix.
    cardinal : array, shape `(4, 4)`
        The cardinal voxel-to-DICOM matrix.

    Raises
    ------
    TypeError
        If the geometry of `image` cannot be found.
    """
    record = _record(image)
    if isinstance(record, AfniRaw):
        return record.voxel_to_dicom, record.cardinal_matrix

    nifti = _nifti_matrix(image)
    if nifti is None:
        nifti = _nifti_matrix(getattr(record, "header", None))
    if nifti is not None:
        real = DICOM_TO_RAS @ nifti
        return real, nifti_cardinal_matrix(real)

    affine = getattr(image, "affine", None)
    if affine is not None and not isinstance(affine, _xforms.Transformation):
        real = DICOM_TO_RAS @ np.asarray(affine, dtype=np.float64)
        return real, nifti_cardinal_matrix(real)

    xform = getattr(image, "transformation", None)
    if isinstance(xform, _xforms.Transformation):
        real = afni_voxel_to_dicom(xform)
        return real, nifti_cardinal_matrix(real)

    raise TypeError(
        f"Cannot find the voxel-to-world geometry of a "
        f"{type(image).__name__}: pass an AFNI image or record, a NIfTI "
        f"(nibabel) image or header, or a brainhops image."
    )


def cardinal_to_real_matrix(image: tx.Any) -> np.ndarray:
    """Return the (4, 4) map from the cardinal DICOM coordinates of an
    image to its true ones, which is `real @ inv(cardinal)`."""
    real, cardinal = image_grids(image)
    return real @ np.linalg.inv(cardinal)


def cardinal_to_real(image: tx.Any) -> _xforms.Affine:
    """
    Return the affine from the cardinal DICOM coordinates of an image to
    its true coordinates, in LPS millimetres.

    AFNI programs ignore the obliquity of a dataset: they compute on its
    *cardinal* grid, which is the closest grid whose axes run along the
    DICOM axes. A point of the image has cardinal coordinates, in which
    the matrices and the warps of AFNI are expressed, and true
    coordinates, which say where it lies in the scanner or template
    space. This affine maps the former to the latter, and it is the
    identity for an image that is not oblique.

    The affine is composed with an AFNI transformation to place it in
    true coordinates, as
    [`AfniAffine`][brainhops.io.transformations.afni.AfniAffine] does when
    it is given its `base` and `source` images. For a warp `W` from the
    cardinal space of the base to that of the source, the map between
    true coordinates is the following.

    ```python
    from brainhops.datamodel.transformations import Sequence

    true = Sequence([
        cardinal_to_real(base).inverse(),  # base: true -> cardinal
        W,                                 # base -> source, cardinal
        cardinal_to_real(source),          # source: cardinal -> true
    ])
    ```

    Parameters
    ----------
    image : AfniRaw or AfniImage or nibabel image or header or Image
        An AFNI image or record, a NIfTI (nibabel) image or header, or a
        brainhops image.

    Returns
    -------
    Affine
        The affine from the cardinal to the true DICOM coordinates of the
        image, from `LPSmm` to `LPSmm`.

    Raises
    ------
    TypeError
        If the geometry of `image` cannot be found.
    """
    return _xforms.Affine(
        matrix=cardinal_to_real_matrix(image)[:3],
        input=_systems.LPSmm(),
        output=_systems.LPSmm(),
    )


def _record(image: tx.Any) -> tx.Any:
    """Return the record that an image, its metadata or a record holds.

    An object without metadata is returned as it is.
    """
    metadata = getattr(image, "metadata", None)
    raw = getattr(metadata, "raw", None)
    if raw is not None:
        return raw
    raw = getattr(image, "raw", None)
    if isinstance(raw, AfniRaw) or hasattr(raw, "header"):
        return raw
    return image


def _nifti_matrix(header: tx.Any) -> tx.Optional[np.ndarray]:
    """
    Return the voxel-to-RAS matrix that AFNI reads from a NIfTI header.

    AFNI prefers the qform when its code is set, then the sform
    (`thd_niftiread.c`, unless `NIFTI_FORM_PRIORITY` says otherwise).
    A header with neither code falls back to the best affine of nibabel.
    A nibabel image, or a NIfTI record, is read through its header, and
    `None` is returned for anything else.
    """
    if not hasattr(header, "get_best_affine"):
        header = getattr(header, "header", None)
    if not (hasattr(header, "get_qform") and hasattr(header, "get_sform")):
        return None
    qform, qcode = header.get_qform(coded=True)
    if qcode and qform is not None:
        return np.asarray(qform, dtype=np.float64)
    sform, scode = header.get_sform(coded=True)
    if scode and sform is not None:
        return np.asarray(sform, dtype=np.float64)
    return np.asarray(header.get_best_affine(), dtype=np.float64)
