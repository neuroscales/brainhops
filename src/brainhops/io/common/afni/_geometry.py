"""AFNI geometry: the cardinal and real voxel-to-DICOM matrices,
and views."""

# stdlib
import itertools
import os
import re

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel.systems import LPSmm
from brainhops.datamodel.transformations import Transformation
from brainhops.io.common._geometry import (
    embed_affine,
    ras_conversion,
    reduce_to_affine,
)

# this format
from ._constants import (
    _ORIENT_CODE,
    _ORIENT_SIGN,
    _VIEW_NAMES,
    DICOM_TO_RAS,
)

# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def afni_cardinal_matrix(
    orient: tx.Sequence[int],
    origin: tx.Sequence[float],
    delta: tx.Sequence[float],
) -> np.ndarray:
    """
    The `(4, 4)` cardinal voxel-to-DICOM matrix (`THD_daxes_to_mat44`).

    Voxel axis `i` runs along DICOM axis `orient[i] // 2`: that row of
    column `i` holds `delta[i]`, and that row of the last column holds
    `origin[i]`.

    Raises
    ------
    ValueError
        If the orientation codes are not valid or do not name three
        different DICOM axes.
    """
    orient = [int(o) for o in orient[:3]]
    if len(orient) != 3 or any(not 0 <= o <= 5 for o in orient):
        raise ValueError(f"Invalid AFNI orientation codes: {orient}")
    rows = [o // 2 for o in orient]
    if len(set(rows)) != 3:
        raise ValueError(
            f"The AFNI orientation codes {orient} do not name three "
            f"different axes."
        )
    matrix = np.eye(4)
    matrix[:3, :3] = 0.0
    for column, row in enumerate(rows):
        matrix[row, column] = float(delta[column])
        matrix[row, 3] = float(origin[column])
    return matrix


def afni_geometry_from_matrix(
    matrix: np.ndarray,
) -> tx.Tuple[
    tx.Tuple[int, int, int],
    tx.Tuple[float, float, float],
    tx.Tuple[float, float, float],
]:
    """
    Decompose a voxel-to-DICOM matrix into AFNI's orientation codes,
    origin and voxel sizes (`THD_daxes_from_mat44`).

    Each voxel axis is given the DICOM axis and direction its column is
    closest to, among the assignments that give the three axes different
    DICOM axes. Its voxel size is the length of its column, and its
    origin is the projection of the translation on the unit column, both
    negated for an orientation that runs towards `R`, `A` or `I`. A
    cardinal matrix gives back exactly its `ORIGIN` and `DELTA`.

    Returns
    -------
    orient : (int, int, int)
    origin : (float, float, float)
    delta : (float, float, float)
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    norms = np.linalg.norm(linear, axis=0)
    safe = np.where(norms > 0, norms, 1.0)
    unit = linear / safe

    # The closest orientation: the permutation of DICOM axes whose
    # entries are the largest, which is what NIfTI's
    # `nifti_mat44_to_orientation` settles on for any matrix that is not
    # wildly sheared.
    best = max(
        itertools.permutations(range(3)),
        key=lambda rows: sum(abs(unit[r, c]) for c, r in enumerate(rows)),
    )
    orient = []
    origin = []
    delta = []
    for column, row in enumerate(best):
        code = _ORIENT_CODE[row, bool(unit[row, column] >= 0)]
        sign = _ORIENT_SIGN[code]
        orient.append(code)
        delta.append(float(sign * norms[column]))
        origin.append(float(sign * (unit[:, column] @ translation)))
    return tuple(orient), tuple(origin), tuple(delta)


def afni_voxel_to_dicom(xform: Transformation) -> np.ndarray:
    """
    The `(4, 4)` voxel-to-DICOM (LPS) matrix of a voxel-to-world
    transformation.

    The transformation is reduced to an affine (a `Scaling`, a
    `Sequence` of affines, ...), embedded in three dimensions, turned into
    RAS from the anatomical orientation of its world axes (a world with
    no orientation is taken to be RAS already, as every format does), and
    then into DICOM.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine representation.
    WriterError
        If it maps more than three spatial dimensions.
    """
    affine = reduce_to_affine(xform, "AFNI", "DICOM")
    matrix = affine.homogeneous_matrix
    matrix = np.eye(4) if matrix is None else np.asarray(matrix, float)
    matrix = _spatial_block(matrix)
    embedded = embed_affine(matrix, "AFNI", "DICOM")
    output = getattr(affine, "output", None)
    try:
        if output is not None and output.ndim is None:
            output = output.expand(matrix.shape[0] - 1)
    except Exception:
        output = None
    return DICOM_TO_RAS @ ras_conversion(output) @ embedded


def _spatial_block(matrix: np.ndarray) -> np.ndarray:
    """
    The homogeneous matrix of the three leading axes of a wider one.

    A map of more than three axes -- such as the `Scaling` of a time
    series, which also scales its time axis -- keeps its three leading
    axes when they do not mix with the others. Any other matrix is
    returned unchanged (and refused by `embed_affine` if too wide).
    """
    out_dim, in_dim = matrix.shape[0] - 1, matrix.shape[1] - 1
    if out_dim <= 3 and in_dim <= 3:
        return matrix
    if out_dim < 3 or in_dim < 3:
        return matrix
    if np.any(matrix[:3, 3:in_dim]) or np.any(matrix[3:out_dim, :3]):
        return matrix
    block = np.eye(4)
    block[:3, :3] = matrix[:3, :3]
    block[:3, 3] = matrix[:3, in_dim]
    return block


def afni_world(name: str) -> LPSmm:
    """The DICOM (LPS millimetre) world space of an AFNI dataset, named
    after its view (`"orig"`, `"tlrc"`, ...)."""
    return LPSmm(name=name)


def afni_view(name: tx.Any) -> tx.Optional[str]:
    """
    The AFNI view a name stands for, or `None`.

    `name` may be a view (`"tlrc"`), the name of a world space
    (`"talairach"` and `"mni"` mean `tlrc`, `"orig-cardinal"` means
    `orig`), or a dataset file name (`"anat+tlrc.HEAD"`).
    """
    if not isinstance(name, (str, os.PathLike)):
        return None
    text = os.fspath(name)
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    base = text.replace("\\", "/").rsplit("/", 1)[-1]
    match = re.search(r"\+(orig|acpc|tlrc)(\.|$)", base)
    if match:
        return match.group(1)
    key = base.lower()
    if key.endswith(_CARDINAL):
        key = key[: -len(_CARDINAL)]
    return _VIEW_NAMES.get(key)


_CARDINAL = "-cardinal"
"""The suffix of the name of the cardinal world space of a view."""
