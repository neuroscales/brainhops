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
    Build the cardinal voxel-to-DICOM matrix of an AFNI grid.

    As in `THD_daxes_to_mat44`, voxel axis `i` runs on DICOM axis
    `orient[i] // 2`; in that row, column `i` holds `delta[i]` and the last
    column holds `origin[i]`.

    Raises
    ------
    ValueError
        If an orientation code is invalid, or if the codes do not name three
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
    Decompose a voxel-to-DICOM matrix into `(orient, origin, delta)`.

    As in `THD_daxes_from_mat44`, each voxel axis takes its closest
    orientation, under the constraint that the three axes run on different
    DICOM axes. The voxel size is the length of the column and the origin is
    the translation projected on the unit column, both negated for the R, A
    and I orientations. A cardinal matrix round-trips exactly.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    norms = np.linalg.norm(linear, axis=0)
    safe = np.where(norms > 0, norms, 1.0)
    unit = linear / safe

    # The closest orientation is the permutation with the largest entries,
    # as in nifti_mat44_to_orientation.
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
    Compute the voxel-to-DICOM matrix of a voxel-to-world transformation.

    The world coordinates are converted to RAS according to the orientation
    of the world axes (axes without one are taken to be RAS already), and
    then to DICOM.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine representation.
    WriterError
        If the transformation has more than three spatial dimensions.
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
    Keep the three leading axes of a matrix when they do not mix with the
    others.

    A `Scaling` of a time series that also scales time is an example. Any
    other matrix is returned unchanged.
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
    """The DICOM (LPS mm) world space of an AFNI view."""
    return LPSmm(name=name)


def afni_view(name: tx.Any) -> tx.Optional[str]:
    """
    Return the AFNI view that a name stands for, or `None`.

    The name may be a view (`"tlrc"`), a world-space name (`"mni"` stands
    for `"tlrc"`, `"orig-cardinal"` for `"orig"`), or a dataset file name
    (`"anat+tlrc.HEAD"`).
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
"""The suffix of the cardinal world-space name of a view."""
