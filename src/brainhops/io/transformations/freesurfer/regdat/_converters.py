"""
Private converters into the `register.dat` format, and between it and LTA.

A `register.dat` file stores an affine between the tkregister RAS spaces
of two volumes, and it holds no geometry of either volume. An LTA file
stores an affine together with the geometries of its two volumes. When
the geometries of both volumes are known, the two formats describe the
same registration, and FreeSurfer converts one into the other exactly,
by algebra on the voxel-to-RAS matrices of the volumes
(`MRItkReg2Native` and `MRItkRegMtx` in `utils/mri.cpp`, and
`LTAchangeType` in `utils/transform.cpp`, FreeSurfer commit 4f8728e).
This module registers these two exact converters:

- a `RegisterDatTransformation` that knows both geometries converts to
  an `LtaTransformation` of type `LINEAR_RAS_TO_RAS`, the type that
  `lta_convert --inreg` writes, whose source volume is the reference and
  whose destination volume is the movable volume;
- an `LtaTransformation` whose record knows both geometries converts to
  a `RegisterDatTransformation` whose reference is the source volume and
  whose movable volume is the destination volume. A record of type
  `REGISTER_DAT` already holds the matrix between tkregister spaces, so
  it converts without its geometries.

Both conversions keep the map, from the input to the output, as every
conversion does. FreeSurfer instead keeps the roles of the volumes: an
LTA file written by `lta_convert --inreg` maps the movable volume to the
reference, so it is the inverse of the LTA file built here, which is the
file that `lta_convert --inreg ... --invert` writes. Either conversion
refuses, with a reason, when a geometry that it needs is missing.

Any other family with an affine form is relabelled as a
`RegisterDatTransformation` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the
transformation only when its coordinate systems are not known or are
compatible with those of the format. Whether the format can write those
systems is decided when it is written. A `RegisterDatTransformation`
that is converted to its own class is changed by the rules of any
affine. Another variant of the format that reads its matrix from a
record reads the same record in its own coordinate systems, which is
exact, and one that holds a matrix of its own is relabelled, as any
other transformation is.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    smart_replace,
)
from brainhops.io.base.parsers import UnrepresentableTransformationError
from brainhops.io.common.freesurfer import FreesurferTkrSystem
from brainhops.io.transformations.base._conversions import unrepresentable
from brainhops.io.transformations.base._converters import (
    relabel,
    relabelling,
)
from brainhops.io.transformations.base._records import matrix_to_disk

from ..lta import LtaRaw, LtaTransformation, LtaType, LtaValidity
from ..lta._matrix_utils import (
    _get_tkr2tkr,
    _volume_geometry,
    _volume_info,
)
from ..lta._xforms import _lta_written
from ._raw import RegisterDatRaw
from ._xforms import (
    RegisterDatTransformation,
    _tkr_to_scanner,
)

# A `register.dat` file stores an affine between tkregister spaces, and
# no exact conversion into it is written yet for the families of the
# data model, so each family with an affine form is relabelled.
converter(_xforms.Identity, RegisterDatTransformation)(relabelling)
converter(_xforms.Translation, RegisterDatTransformation)(relabelling)
converter(_xforms.Scaling, RegisterDatTransformation)(relabelling)
converter(_xforms.Permutation, RegisterDatTransformation)(relabelling)
converter(_xforms.Linear, RegisterDatTransformation)(relabelling)
converter(_xforms.Affine, RegisterDatTransformation)(relabelling)
converter(_xforms.AffineExponential, RegisterDatTransformation)(relabelling)
converter(_xforms.SubspaceTransformation, RegisterDatTransformation)(
    relabelling
)
converter(_xforms.Sequence, RegisterDatTransformation)(relabelling)


@converter
def _(
    t: RegisterDatTransformation,
    cls: tx.Type[RegisterDatTransformation],
    **kwargs,
) -> RegisterDatTransformation:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type. Another variant of the format reads
    # the record of `t`, when it has one, in its own systems, with the
    # same geometries. A matrix of its own is relabelled only when its
    # systems are compatible.
    if isinstance(t, cls):
        return _convert_withlog(t, cls, "matrix", **kwargs)
    if t._data is None:
        return _view(t, cls, **kwargs)
    return relabel(t, cls, **kwargs)


@converter
def _(
    t: RegisterDatTransformation,
    cls: tx.Type[LtaTransformation],
    **kwargs,
) -> LtaTransformation:
    # The scanner RAS affine from the reference to the movable volume,
    # written as `lta_convert --inreg` writes its LINEAR_RAS_TO_RAS file,
    # but in the direction of `t`, so that the map is kept.
    if t.reference is None or t.moving is None:
        raise unrepresentable(
            t,
            cls,
            "an LTA file places its affine with the geometries of its "
            "two volumes, and a register.dat file holds none. Give the "
            "geometries of both volumes, as reference= and moving=, to "
            "convert it.",
        )
    scanner = _scanner_matrix(t, cls)
    record = LtaRaw(
        type=LtaType.LINEAR_RAS_TO_RAS,
        affine=LtaRaw.Affine(matrix=matrix_to_disk(scanner[:-1])),
        src=_volume_info(t.reference, LtaRaw.SrcVolumeInfo),
        dst=_volume_info(t.moving, LtaRaw.DstVolumeInfo),
    )
    return cls.from_raw(record, **kwargs)


@converter
def _(
    t: LtaTransformation,
    cls: tx.Type[RegisterDatTransformation],
    **kwargs,
) -> RegisterDatTransformation:
    # The affine between the tkregister spaces of the source and the
    # destination volume of the record that `t` writes, whose geometries
    # become those of the reference and the movable volume.
    for name in ("reference", "moving"):
        if name in kwargs:
            raise TypeError(
                f"The {name} geometry of a register.dat converted from an "
                f"LTA file is that of the LTA file, so {name}= cannot be "
                f"given."
            )
    try:
        record = _lta_written(t)
    except UnrepresentableTransformationError as error:
        raise unrepresentable(t, cls, str(error)) from error
    src, dst = _valid(record.src), _valid(record.dst)
    if record.type != LtaType.REGISTER_DAT and (src is None or dst is None):
        raise unrepresentable(
            t,
            cls,
            "a register.dat file maps tkregister RAS spaces, which the "
            "geometries of both volumes give, and the LTA file does not "
            "hold a valid geometry for both of them.",
        )
    matrix = _get_tkr2tkr(record)
    raw = _new_raw(matrix, dst)
    return cls.from_raw(
        raw,
        reference=None if src is None else _volume_geometry(src),
        moving=None if dst is None else _volume_geometry(dst),
        **kwargs,
    )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _view(
    t: RegisterDatTransformation,
    cls: tx.Type[RegisterDatTransformation],
    **kwargs,
) -> RegisterDatTransformation:
    # Another variant of the format that shares the record of `t`, whose
    # matrix it decodes rather than copies.
    kwargs.setdefault("data", None)
    return smart_replace(t, cls, **kwargs)


def _scanner_matrix(t: RegisterDatTransformation, cls: type) -> np.ndarray:
    # The (4, 4) scanner RAS affine of `t`, from its matrix in its own
    # systems: the tkregister spaces of the format, or scanner RAS for
    # the view, assigned or decoded from the record.
    data = t.matrix
    if data is not None and np.shape(data) != (3, 4):
        raise unrepresentable(
            t,
            cls,
            f"an LTA file holds a 3D affine, of shape (3, 4), and the "
            f"matrix has shape {np.shape(data)}.",
        )
    homogeneous = np.asarray(matrix_to_disk(data), dtype=np.float64)
    src_system, dst_system = t.input, t.output

    def both(system_cls: type) -> bool:
        return isinstance(src_system, system_cls) and isinstance(
            dst_system, system_cls
        )

    if both(FreesurferTkrSystem):
        return _tkr_to_scanner(homogeneous, t.reference, t.moving)
    if both(_systems.RASmm):
        return homogeneous
    raise unrepresentable(
        t,
        cls,
        "its input and output are neither both tkregister spaces nor "
        "both scanner RAS, so the affine that it maps is not known.",
    )


def _valid(info: tx.Optional[LtaRaw.VolumeInfo]) -> tx.Any:
    # The geometry block of a volume, or `None` when the record has none
    # or marks it invalid, which is how FreeSurfer writes a geometry that
    # is not known.
    if info is None or info.valid != LtaValidity.VOLUME_INFO_VALID:
        return None
    return info


def _new_raw(
    matrix: np.ndarray, moving: tx.Optional[LtaRaw.VolumeInfo]
) -> RegisterDatRaw:
    # A new record, whose resolution lines hold the first and third voxel
    # sizes of the movable volume, as `tkregister2` and `LTAwriteEx` in
    # `utils/transform.cpp` write them, or 1 when it is not known.
    homogeneous = tuple(tuple(float(v) for v in row) for row in matrix)
    if moving is None:
        return RegisterDatRaw(matrix=homogeneous)
    return RegisterDatRaw(
        inplane_resolution=float(moving.voxelsize[0]),
        slice_thickness=float(moving.voxelsize[2]),
        matrix=homogeneous,
    )
