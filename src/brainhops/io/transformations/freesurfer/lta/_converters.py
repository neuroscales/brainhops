"""
Private converters into the LTA format.

`t.to(LtaTransformation)`, `LtaTransformation.from_any(t)` and
`io.save(t, "t.lta")` all run these converters. Each one returns an
`LtaTransformation` that is the map `t` is, or raises a
`ConversionError` that says what the format cannot hold.

An LTA file stores an affine together with a type, which names the
coordinate systems that the affine maps, and the geometries of its
source and destination volumes. The type of the file is chosen from the
endpoints of `t`:

| Endpoints of `t`                            | LTA type                    |
| ------------------------------------------- | --------------------------- |
| both `FreesurferTkrSystem`                  | `REGISTER_DAT`              |
| both `LtaVoxelSystem`                       | `LINEAR_VOX_TO_VOX`         |
| both `LtaPhysicalSystem`                    | `LINEAR_PHYSVOX_TO_PHYSVOX` |
| both `RSAmm`                                | `LINEAR_RSA_TO_RSA`         |
| both voxel systems, with both geometries    | `LINEAR_VOX_TO_VOX`         |
| any other world, or not known               | `LINEAR_RAS_TO_RAS`         |

The systems of an LTA file are kept as they are. A transformation
between two other voxel systems is a map between the voxels of two
volumes, which the file stores only with the geometries of both
volumes, so the conversion takes them as the options `reference=`, the
volume of the input, and `moving=`, the volume of the output. These are
the names that a `register.dat` file and a FLIRT matrix give to the
volumes of their input and output. Any other transformation is read as
one affine between RAS worlds: its endpoints are bridged to RAS, which
flips an affine between LPS spaces, and an endpoint that is not known
is taken to be RAS. Such a file needs no geometry, but FreeSurfer tools
use the geometries when they are there, so the options `reference=` and
`moving=` fill them, and they are otherwise marked invalid, as
FreeSurfer writes a geometry that is not known. A geometry is a
[`Geometry`][brainhops.datamodel.geometry.Geometry], an image, whose
geometry is used, or the geometry block of an LTA file.

An `LtaTransformation` that is converted to its own class is changed by
the rules of any affine. Another variant of the format that reads its
matrix from a record reads the same record in its own coordinate
systems, which is exact. One that holds a matrix of its own is
converted as any other affine is.

A `register.dat` transformation converts exactly to an LTA file when
the geometries of its two volumes are known. That converter, and the one
in the other direction, are registered by the private module
`brainhops.io.transformations.freesurfer.regdat._converters`, which
knows both formats.
"""

import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import (
    convert_withlog,
    converter,
    smart_replace,
)
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.images import Image
from brainhops.io.base.parsers import UnrepresentableTransformationError
from brainhops.io.common.freesurfer import FreesurferTkrSystem
from brainhops.io.transformations.base._internal_api import (
    AFFINE_SOURCES,
    affine_between,
    compatible,
    format_options,
    register_converter,
    unrepresentable,
)

from ._matrix_utils import volume_info
from ._metadata import LtaMetadata
from ._raw import LtaRaw
from ._systems import LtaPhysicalSystem, LtaVoxelSystem
from ._xforms import LtaTransformation, as_block, build_raw

OPTIONS = ("reference", "moving")
"""What a conversion to an LTA file takes: the geometries of the volume
of its input and of the volume of its output."""

RAS = _systems.RASmm()
"""The RAS world, in millimetres, of the `LINEAR_RAS_TO_RAS` type."""

_KEPT = (
    FreesurferTkrSystem,
    LtaVoxelSystem,
    LtaPhysicalSystem,
    _systems.RSAmm,
)
"""The systems that an LTA file stores as they are, each with its type."""


@register_converter(AFFINE_SOURCES, LtaTransformation)
def _(
    t: _xforms.Transformation, cls: tx.Type[LtaTransformation], **kwargs
) -> LtaTransformation:
    # Exactly, or raise: the type follows from the endpoints of `t`, and
    # the record is encoded as the format encodes an affine that it holds.
    options = format_options(t, cls, kwargs, OPTIONS)
    src, dst = t.input, t.output
    for kind in _KEPT:
        if isinstance(src, kind) and isinstance(dst, kind):
            matrix = affine_between(t, src, dst, cls)
            return _held(t, cls, matrix, src, dst, None)
    check_tkr(t, cls, src, dst)
    if _is_voxels(src) and _is_voxels(dst):
        return _voxels(t, cls, options)
    matrix = affine_between(t, RAS, RAS, cls)
    blocks = None
    if any(options.get(name) is not None for name in OPTIONS):
        blocks = LtaRaw(
            src=_block(options.get("reference"), LtaRaw.SrcVolumeInfo, t, cls),
            dst=_block(options.get("moving"), LtaRaw.DstVolumeInfo, t, cls),
        )
    return _held(t, cls, matrix, RAS, RAS, blocks)


@converter
def _(
    t: LtaTransformation, cls: tx.Type[LtaTransformation], **kwargs
) -> LtaTransformation:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type. Another variant of the format reads
    # the record of `t`, when it has one, in its own systems. A matrix of
    # its own is converted as the matrix of any affine.
    if type(t) is cls:
        return convert_withlog(t, cls, "matrix", **kwargs)
    if t._data is None:
        return _view(t, cls, **kwargs)
    plain = _xforms.Affine(matrix=t.matrix, input=t.input, output=t.output)
    return plain.to(cls, **kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _view(
    t: LtaTransformation, cls: tx.Type[LtaTransformation], **kwargs
) -> LtaTransformation:
    # Another variant of the format that shares the record of `t`, whose
    # matrix it decodes rather than copies.
    kwargs.setdefault("data", None)
    return smart_replace(t, cls, **kwargs)


def _is_voxels(system: tx.Optional[_systems.CoordinateSystem]) -> bool:
    # A voxel system, whose axes count voxels, as the voxels of a NIfTI
    # file or of an MGH volume do.
    return isinstance(system, _systems.VoxelCoordinateSystem)


def _voxels(
    t: _xforms.Transformation,
    cls: tx.Type[LtaTransformation],
    options: tx.Dict[str, tx.Any],
) -> LtaTransformation:
    # A map between the voxels of two volumes, stored with the geometries
    # of both, which the options give.
    missing = [name for name in OPTIONS if options.get(name) is None]
    if missing:
        names = " and ".join(f"{name}=" for name in missing)
        raise unrepresentable(
            t,
            cls,
            f"it maps the voxels of two volumes, which an LTA file places "
            f"with the geometries of both volumes, and the conversion is "
            f"not given {names}. Pass the geometry of the volume of its "
            f"input as reference= and that of its output as moving=.",
        )
    src_info = _block(options["reference"], LtaRaw.SrcVolumeInfo, t, cls)
    dst_info = _block(options["moving"], LtaRaw.DstVolumeInfo, t, cls)
    src = LtaVoxelSystem.from_raw(src_info)
    dst = LtaVoxelSystem.from_raw(dst_info)
    matrix = affine_between(t, src, dst, cls)
    return _held(t, cls, matrix, src, dst, None)


def _block(
    value: tx.Any, kind: tx.Type[LtaRaw.VolumeInfo], t: tx.Any, cls: type
) -> LtaRaw.VolumeInfo:
    # The geometry block of a volume, from a geometry, an image or a block,
    # or an invalid block when the geometry is not given.
    if value is None or isinstance(value, LtaRaw.VolumeInfo):
        return as_block(value, kind)
    if isinstance(value, Image):
        value = value.geometry
    if not isinstance(value, Geometry):
        raise unrepresentable(
            t,
            cls,
            f"the geometry of a volume is a Geometry, an image or an "
            f"LtaRaw.VolumeInfo, not a {type(value).__name__}.",
        )
    try:
        return volume_info(value, kind)
    except (TypeError, ValueError) as error:
        raise unrepresentable(t, cls, str(error)) from error


def _held(
    t: _xforms.Transformation,
    cls: tx.Type[LtaTransformation],
    matrix: tx.Any,
    src: _systems.CoordinateSystem,
    dst: _systems.CoordinateSystem,
    blocks: tx.Optional[LtaRaw],
) -> LtaTransformation:
    # The format that holds `matrix` between `src` and `dst`, with the
    # record that the format encodes for them. The geometries of `blocks`
    # are those of the RAS type, which takes them from the current record.
    if matrix.shape != (4, 4):
        raise unrepresentable(
            t,
            cls,
            f"an LTA file holds a 3-D affine, and this one has "
            f"{len(matrix) - 1} dimensions.",
        )
    metadata = None if blocks is None else LtaMetadata.from_raw(blocks)
    pending = LtaTransformation(
        matrix=matrix[:-1], input=src, output=dst, metadata=metadata
    )
    try:
        record = build_raw(pending)
    except UnrepresentableTransformationError as error:
        raise unrepresentable(t, cls, str(error)) from error
    _check_view(t, cls, src, dst)
    if blocks is None:
        # The matrix and the systems stay pending, as for any affine that
        # a format holds without a record, and are encoded when written.
        return cls(matrix=matrix[:-1], input=src, output=dst)
    # The geometries of the RAS type live in a record.
    return cls.from_raw(record)


def _check_view(
    t: _xforms.Transformation,
    cls: tx.Type[LtaTransformation],
    src: _systems.CoordinateSystem,
    dst: _systems.CoordinateSystem,
) -> None:
    # A view of the format, such as the voxel-to-voxel view, reads an LTA
    # file in its own systems, so it holds only an affine between them.
    if cls is LtaTransformation:
        return
    view = cls()
    for side, system in (("input", src), ("output", dst)):
        own = getattr(view, side)
        if own is not None and not compatible(system, own):
            raise unrepresentable(
                t,
                cls,
                f"the view reads an LTA file in its own systems, and the "
                f"{side} of the LTA file is {type(system).__name__}, not "
                f"{type(own).__name__}. Convert it to LtaTransformation.",
            )


def check_tkr(
    t: _xforms.Transformation,
    cls: type,
    src: tx.Optional[_systems.CoordinateSystem],
    dst: tx.Optional[_systems.CoordinateSystem],
) -> None:
    # A tkregister space has the axes of RAS but another origin, which
    # only the geometry of its volume gives, so it is never read as the
    # scanner RAS of the other side.
    tkr = [isinstance(system, FreesurferTkrSystem) for system in (src, dst)]
    if any(tkr) and not all(tkr):
        raise unrepresentable(
            t,
            cls,
            "one of its endpoints is a tkregister space and the other is "
            "not. A tkregister space is placed in scanner RAS only by the "
            "geometry of its volume, so the affine is not one that the "
            "format stores. Compose it with the tkregister-to-scanner "
            "affine of the volume first.",
        )
