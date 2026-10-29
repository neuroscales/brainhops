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

Every other family with an affine form, and a `Sequence` of affines,
converts exactly to a `RegisterDatTransformation` when the geometries of
both volumes are known. An affine between two tkregister spaces is kept
as it is, with the geometries that its systems hold. Any other affine is
read between the scanner RAS spaces of the two volumes, with its
endpoints bridged to RAS, which flips an affine between LPS spaces, and
an endpoint that is not known is taken to be RAS. Its matrix between
the tkregister spaces is then computed from the geometries, which the
conversion takes as the options `reference=`, the volume of the input,
and `moving=`, the volume of the output, as a FLIRT matrix does. The
conversion is refused, with the names of the options, when either is
missing. An LTA file without valid geometries also takes them as
options. A geometry is a
[`Geometry`][brainhops.datamodel.geometry.Geometry], an image, whose
geometry is used, or the geometry block of an LTA file.

A `RegisterDatTransformation` that is converted to its own class is
changed by the rules of any affine. Another variant of the format that
reads its matrix from a record reads the same record in its own
coordinate systems, which is exact, and one that holds a matrix of its
own is converted as any other affine is.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import (
    convert_withlog,
    converter,
    smart_replace,
)
from brainhops.io.base.parsers import UnrepresentableTransformationError
from brainhops.io.common.freesurfer import FreesurferTkrSystem
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    affine_between,
    format_options,
    register_converter,
    unrepresentable,
)
from brainhops.io.transformations.base._records import matrix_to_disk

from ..lta import LtaRaw, LtaTransformation, LtaType, LtaValidity
from ..lta._converters import _check_tkr
from ..lta._matrix_utils import (
    _get_tkr2tkr,
    _volume_geometry,
    _volume_info,
)
from ..lta._xforms import _lta_written
from ._raw import RegisterDatRaw
from ._xforms import (
    RegisterDatTransformation,
    _as_moving_geometry,
    _as_reference_geometry,
    _scanner_to_tkr,
    _tkr_to_scanner,
)

OPTIONS = ("reference", "moving")
"""What a conversion to a `register.dat` file takes: the geometries of
the volume of its input and of the volume of its output."""

RAS = _systems.RASmm()
"""The scanner RAS world, in millimetres, of a volume."""


@register_converter(AFFINE_SOURCES, RegisterDatTransformation)
def _(
    t: _xforms.Transformation,
    cls: tx.Type[RegisterDatTransformation],
    **kwargs,
) -> RegisterDatTransformation:
    # Exactly, or raise: the affine between the tkregister spaces of the
    # two volumes, kept as it is or computed from the scanner RAS affine
    # and the geometries of both volumes.
    options = format_options(t, cls, kwargs, OPTIONS)
    reference, moving = _geometries(t, cls, options)
    src, dst = t.input, t.output
    if isinstance(src, FreesurferTkrSystem) and isinstance(
        dst, FreesurferTkrSystem
    ):
        matrix = affine_between(t, src, dst, cls)
        reference = src.geometry if reference is None else reference
        moving = dst.geometry if moving is None else moving
    else:
        _check_tkr(t, cls, src, dst)
        missing = [
            name
            for name, value in zip(OPTIONS, (reference, moving))
            if value is None
        ]
        if missing:
            names = " and ".join(f"{name}=" for name in missing)
            raise unrepresentable(
                t,
                cls,
                f"a register.dat file maps the tkregister spaces of two "
                f"volumes, which the geometries of both volumes place in "
                f"scanner RAS, and the conversion is not given {names}. "
                f"Pass the geometry of the volume of its input as "
                f"reference= and that of its output as moving=.",
            )
        scanner = affine_between(t, RAS, RAS, cls)
        matrix = _scanner_to_tkr(scanner, reference, moving)
    if matrix.shape != (4, 4):
        raise unrepresentable(
            t,
            cls,
            f"a register.dat file holds a 3-D affine, and this one has "
            f"{len(matrix) - 1} dimensions.",
        )
    result = RegisterDatTransformation(
        matrix=matrix[:-1], reference=reference, moving=moving
    )
    if cls is RegisterDatTransformation:
        return result
    # A view reads the record in its own systems, so the matrix between
    # the tkregister spaces is encoded first.
    return _view(result.consolidate(), cls)


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
    # same geometries. A matrix of its own is converted as the matrix of
    # any affine.
    if type(t) is cls:
        return convert_withlog(t, cls, "matrix", **kwargs)
    if t._data is None:
        return _view(t, cls, **kwargs)
    return _plain(t).to(cls, **kwargs)


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
    # become those of the reference and the movable volume. A record
    # without valid geometries takes them as options, as any affine does.
    try:
        record, _ = _lta_written(t, {})
    except UnrepresentableTransformationError as error:
        raise unrepresentable(t, cls, str(error)) from error
    src, dst = _valid(record.src), _valid(record.dst)
    given = [name for name in OPTIONS if kwargs.get(name) is not None]
    if given and record.type != LtaType.REGISTER_DAT:
        if src is not None or dst is not None:
            names = " and ".join(f"{name}=" for name in given)
            raise unrepresentable(
                t,
                cls,
                f"the geometries of a register.dat converted from an LTA "
                f"file are those of the LTA file, so {names} cannot be "
                f"given.",
            )
        return _plain(t).to(cls, **kwargs)
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


def _plain(t: _xforms.Affine) -> _xforms.Affine:
    # The affine that a format holds, between its own systems, as a plain
    # affine of the data model.
    return _xforms.Affine(matrix=t.matrix, input=t.input, output=t.output)


def _geometries(
    t: _xforms.Transformation, cls: type, options: tx.Dict[str, tx.Any]
) -> tx.Tuple[tx.Any, tx.Any]:
    # The geometries that the options give, as a register.dat file checks
    # them, or `None` for a geometry that is not given.
    try:
        return (
            _as_reference_geometry(options.get("reference")),
            _as_moving_geometry(options.get("moving")),
        )
    except (TypeError, ValueError) as error:
        raise unrepresentable(t, cls, str(error)) from error


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
