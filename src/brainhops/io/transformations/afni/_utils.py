"""
The geometry the AFNI transformation formats share: the true and the
cardinal voxel-to-DICOM matrices of an image, and the map between its
cardinal and its true world coordinates.

AFNI programs (`3dAllineate`, `3dvolreg`, `3dQwarp`, `3dNwarpApply`, ...)
compute on each dataset's *cardinal* grid -- the `ijk_to_dicom` matrix,
built from `ORIENT_SPECIFIC`, `ORIGIN` and `DELTA` -- and ignore the
obliquity of `IJK_TO_DICOM_REAL` (`DSET_CMAT(dset, use_realaxes)`, with
`use_realaxes = 0` unless asked, in `3dAllineate.c`; `IW3D_from_dataset`
in `mri_nwarp.c`). The matrices and warps they write therefore map
cardinal coordinates. See [`brainhops.io.transformations.afni`][].
"""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base.afni import (
    DICOM_TO_RAS,
    AfniHeader,
    afni_cardinal_matrix,
    afni_geometry_from_matrix,
    afni_voxel_to_dicom,
)


def _nifti_matrix(header: tx.Any) -> tx.Optional[np.ndarray]:
    """
    The voxel-to-RAS matrix AFNI reads from a NIfTI header, or `None`.

    AFNI prefers the qform when its code is set, then the sform
    (`thd_niftiread.c`, unless `NIFTI_FORM_PRIORITY` says otherwise).
    A header with neither code falls back to `nibabel`'s best affine.
    """
    if not (hasattr(header, "get_qform") and hasattr(header, "get_sform")):
        return None
    qform, qcode = header.get_qform(coded=True)
    if qcode and qform is not None:
        return np.asarray(qform, dtype=np.float64)
    sform, scode = header.get_sform(coded=True)
    if scode and sform is not None:
        return np.asarray(sform, dtype=np.float64)
    return np.asarray(header.get_best_affine(), dtype=np.float64)


def nifti_cardinal_matrix(real: np.ndarray) -> np.ndarray:
    """
    The cardinal voxel-to-DICOM matrix AFNI gives a grid it reads from a
    NIfTI file (or any format other than its own), from its true
    voxel-to-DICOM matrix.

    `thd_niftiread.c` keeps the closest orientation of each voxel axis
    and the length of its column as voxel size -- as
    [`afni_geometry_from_matrix`][brainhops.io.base.afni.afni_geometry_from_matrix]
    does -- but takes as origin the translation itself, along the DICOM
    axis each voxel axis runs along, rather than its projection on the
    column: the cardinal and the true matrices share their translation.
    The two are the same matrix unless the grid is oblique.
    """
    real = np.asarray(real, dtype=np.float64)
    orient, _, delta = afni_geometry_from_matrix(real)
    origin = [float(real[code // 2, 3]) for code in orient]
    return afni_cardinal_matrix(orient, origin, delta)


def image_grids(image: tx.Any) -> tx.Tuple[np.ndarray, np.ndarray]:
    """
    The true and the cardinal `(4, 4)` voxel-to-DICOM matrices of an
    image, as AFNI sees them.

    Parameters
    ----------
    image : AfniHeader | AfniImage | nibabel image or header | Image
        An AFNI header, or an object whose `header` is one (an
        [`AfniImage`][brainhops.io.images.afni.AfniImage]), gives
        `IJK_TO_DICOM_REAL` and its own cardinal grid. A NIfTI header or
        image (`nibabel`'s, or a brainhops image that keeps one in
        `header`) gives the form AFNI reads, and the cardinal grid AFNI
        derives from it. Any other brainhops image gives its preferred
        voxel-to-world transformation, and the cardinal grid AFNI would
        derive from it on import.

    Returns
    -------
    real, cardinal : (4, 4) arrays
    """
    header = image if isinstance(image, AfniHeader) else None
    if header is None and isinstance(
        getattr(image, "header", None), AfniHeader
    ):
        header = image.header
    if header is not None:
        return header.voxel_to_dicom, header.cardinal_matrix

    nifti = _nifti_matrix(image)
    if nifti is None:
        nifti = _nifti_matrix(getattr(image, "header", None))
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
        f"{type(image).__name__}: pass an AFNI image or header, a NIfTI "
        f"(nibabel) image or header, or a brainhops image."
    )


def cardinal_to_real_matrix(image: tx.Any) -> np.ndarray:
    """The `(4, 4)` map from an image's cardinal DICOM coordinates to its
    true ones: `real @ inv(cardinal)`."""
    real, cardinal = image_grids(image)
    return real @ np.linalg.inv(cardinal)


def cardinal_to_real(image: tx.Any) -> _xforms.Affine:
    """
    The affine from an image's cardinal DICOM coordinates to its true
    ones, in LPS millimetres.

    AFNI programs ignore the obliquity of a dataset: they compute on its
    *cardinal* grid, the closest grid whose axes run along the DICOM
    axes. A point of the image has cardinal coordinates (what AFNI's
    matrices and warps are expressed in) and true coordinates (where it
    is in the scanner, or template, space). This affine maps the former
    to the latter: it is the identity for an image that is not oblique.

    Compose it with an AFNI transformation to place it in true
    coordinates, as
    [`AfniAffine`][brainhops.io.transformations.afni.AfniAffine] does
    when given its `base` and `source` images. For a warp `W` from
    the cardinal space of the base to that of the source:

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
    image : AfniHeader | AfniImage | nibabel image or header | Image
        An AFNI image or header, a NIfTI (`nibabel`) image or header, or
        a brainhops image (see `image_grids`).
    """
    return _xforms.Affine(
        matrix=cardinal_to_real_matrix(image)[:3],
        input=_systems.LPSmm(),
        output=_systems.LPSmm(),
    )
