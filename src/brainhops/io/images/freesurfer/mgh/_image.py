# stdlib
import math

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace
from nibabel.freesurfer import mghformat as _mgh

# internals
from brainhops._core import path
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import (
    ConversionReport,
    apply_loss_policy,
    preferred_dtype,
)
from brainhops.datamodel.orientation import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base._mgh_metadata import MghRaw
from brainhops.io.base.mgh import _MRI_PARAMS, MghMetadata, MghParser
from brainhops.io.base.nifti import (
    _scale_spatial,
    _unit_scale,
    _voxel_to_ras,
)
from brainhops.io.base.parsers import WriterError
from brainhops.io.images.base import WritableFileBasedImage

_SCANNER = "scanner"
"""Name of the scanner RAS space, the preferred world space."""

_TKR = "tkr"
"""Name of the tkr (surface) RAS space."""

_PHYSICAL = "physical"
"""Name of the scaled voxel space."""

_RAS_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}

# The voxel types MGH stores, by numpy dtype.
_MGH_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float32),
}


@register_format
class MghImage(MghParser, WritableFileBasedImage, SingleScaleImage):
    """
    An image that is encoded by a FreeSurfer MGH or MGZ file.

    The voxels are stored x fastest (F order), so `data` has shape
    `(x, y, z)` or, for a multi-frame volume, `(x, y, z, frames)`, with
    the frames read as a time axis.

    The transformations are, in order:

    1. a [`Scaling`][brainhops.datamodel.transformations.Scaling] from
       voxels to the scaled voxel space `"physical"`: the voxel size in
       mm, and, for a multi-frame volume, the repetition time in ms (or 1,
       with no unit, when the footer records none);
    2. the voxel-to-tkr (surface) RAS affine, whose output is named
       `"tkr"` (`header.get_vox2ras_tkr()`);
    3. the voxel-to-scanner RAS affine, whose output is named
       `"scanner"` (`header.get_vox2ras()`). It is the last one, so it is
       the preferred transformation.

    FreeSurfer-specific header content -- the MRI acquisition parameters
    of the footer, the raw `goodRASFlag` and the trailing tags -- is kept
    on the object ([`mri_params`][brainhops.io.base.mgh.MghParser.
    mri_params], the private `_good_ras` and
    [`tags`][brainhops.io.base.mgh.MghParser.tags]) and written back.

    !!! note "`goodRASFlag`"
        When the header's `goodRASFlag` is not positive, FreeSurfer
        ignores the stored geometry and uses 1 mm voxels, coronal (LIA)
        direction cosines and a zero centre. So does this reader. A file
        written back always records its geometry, with `goodRASFlag = 1`.

    !!! note "Why the bases are in this order"
        As for [`NiftiImage`][brainhops.io.images.nifti.NiftiImage]:
        `SingleScaleImage.data` has no default, so it comes last, and
        `MghParser` leads so that its lazy `data`/`system` properties win.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".mgh",
        ".mgz",
        ".mgh.gz",
    )

    @property
    def transformations(self) -> tx.List[Transformation]:
        """The voxel-to-world transformations recorded by the header,
        decoded on first access unless set explicitly.

        An image built from data alone has no header, so it records no
        transformation and the list is empty.
        """
        if getattr(self, "_transformations", None):
            return self._transformations
        if self.header is None:
            return list(getattr(self, "_transformations", None) or [])
        return _mgh_to_transformations(self)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value

    def to_nibabel(self, like: tx.Any = None, **overrides) -> _mgh.MGHImage:
        """
        Build the `nibabel` image that encodes this image.

        The image data becomes the voxels. The voxel-to-scanner RAS
        matrix is taken from the transformation whose output is named
        `"scanner"`, or else from the preferred transformation unless it
        maps to tkr RAS (which MGH cannot store, as it follows from the
        shape and the voxel size). It is converted to millimetres and
        decomposed into voxel sizes, direction cosines and a centre. An
        image with no transformation is written with 1 mm voxels and RAS
        axes. A transformation with no affine representation raises
        `UnrepresentableTransformationError`.

        The MRI parameters of the footer (`tr`, `flip_angle`, `te`, `ti`,
        `fov`) are copied from the raw record of the file this image was read
        from (`metadata.raw`), then from `like` when it is given (a path
        to an MGH/MGZ file, a `nibabel` MGH image or header, or another
        object read from MGH). The fields of `metadata` changed since the
        read are written over them (all of them for metadata built in
        memory or converted from another format), and what MGH cannot
        hold is reported according to `on_loss` (`"ignore"`, `"warn"` or
        `"raise"`; the policy in effect by default).

        The keyword arguments `tr`, `te`, `ti` (ms) and `flip_angle`
        (radians) set the matching metadata fields
        (`repetition_time`, ... in seconds and degrees) and win over
        everything else; any other keyword, such as `fov`, sets that
        header field last. `dtype` sets the stored voxel type. Without
        it, `metadata.data_type` (the type of the file that was read)
        does, when the array's values are of its kind (an integer type
        for integer values); else the array's type is kept when MGH can
        store it (uint8, int16, int32, float32), and otherwise converted
        to the nearest one MGH can: booleans to uint8, other floats to
        float32, other integers to int16 or int32. Integers that int32
        cannot hold raise `WriterError`.
        """
        return self._to_nibabel_and_tags(like, **overrides)[0]

    def _to_nibabel_and_tags(
        self, like: tx.Any = None, **overrides
    ) -> tx.Tuple[_mgh.MGHImage, bytes]:
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        ndim = len(getattr(data, "shape", ()))
        if not 1 <= ndim <= 4:
            raise WriterError(
                f"MGH stores volumes of up to four dimensions "
                f"(x, y, z, frames), not {ndim}."
            )

        overrides = dict(overrides)
        on_loss = overrides.pop("on_loss", None)
        dtype = overrides.pop("dtype", None)
        metadata, report, force = _writable_metadata(self.metadata, overrides)
        dtype = _stored_dtype(data, dtype, metadata, report)

        record = metadata.raw
        base = record.header if record is not None else self.header
        header = _mgh.MGHHeader()
        for source in (base, _like_header(like)):
            if source is None:
                continue
            for name in _MRI_PARAMS:
                header[name] = source[name]
        tags = record.tags if record is not None else (self.tags or b"")
        target = metadata.update_raw(
            MghRaw(header, tags), image=self, report=report, force=force
        )
        apply_loss_policy(report, on_loss, stacklevel=4)
        header = target.header
        for name, value in overrides.items():
            header[name] = value
        header.set_data_dtype(dtype)

        vox2ras = _scanner_matrix(self.transformations)
        return _mgh.MGHImage(data, vox2ras, header=header), target.tags


# The footer keywords of the writer, routed through the metadata: keyword
# -> (vocabulary field, factor from the keyword's unit to the field's).
_LEGACY_KEYWORDS = {
    "tr": ("repetition_time", 1e-3),
    "te": ("echo_time", 1e-3),
    "ti": ("inversion_time", 1e-3),
    "flip_angle": ("flip_angle", None),  # radians -> degrees
}


def _writable_metadata(
    metadata: tx.Any, overrides: tx.Dict[str, tx.Any]
) -> tx.Tuple[MghMetadata, ConversionReport, tx.Tuple[str, ...]]:
    """
    The `MghMetadata` to write, the report the write starts from, and
    the fields to write whatever the snapshot says: the image's own
    metadata (converted from another format if need be, see
    `FileBasedMetadata.writable`), with the footer keywords popped from
    `overrides` set over it. A keyword always wins: it is written even when it
    equals the value that was read, and a zero clears the slot.
    """
    if metadata is None:
        metadata = MghMetadata()
    metadata, report = MghMetadata.writable(metadata)
    values = {}
    for keyword, (name, factor) in _LEGACY_KEYWORDS.items():
        if keyword not in overrides:
            continue
        value = overrides.pop(keyword)
        if value is None or not float(value):
            values[name] = None
        elif factor is None:
            values[name] = math.degrees(float(value))
        else:
            values[name] = float(value) * factor
    if not values:
        return metadata, report, ()
    return replace(metadata, **values), report, tuple(values)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _mgh_to_transformations(image: MghParser) -> tx.List[Transformation]:
    """
    Convert the header of an MGH image to its list of transformations:
    voxel-to-physical scaling, voxel-to-tkr RAS and voxel-to-scanner RAS.
    """
    voxel_space = image.system
    axes = list(voxel_space.axes)
    tr = image.mri_params["tr"]

    # >> Physical space: the same axes, in mm (and ms for the frames,
    # when the repetition time is known -- it is never invented).
    units = {"space": "mm", "time": "ms" if tr > 0 else None}
    phys_axes = [replace(axis, unit=units.get(axis.type)) for axis in axes]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)

    # >> RAS space: the spatial axes run along R, A, S.
    ras_axes = [
        replace(
            axis,
            orientation=Orientation(
                type="anatomical", value=_RAS_ORIENTATION[axis.name]
            ),
        )
        if axis.name in _RAS_ORIENTATION
        else axis
        for axis in phys_axes
    ]
    ras_space = CoordinateSystem(name="RAS", axes=ras_axes)

    zooms = list(image.voxel_size)
    if len(axes) > 3:
        zooms.append(tr if tr > 0 else 1.0)
    vox2phys = Scaling(
        input=voxel_space, output=phys_space, scale=zooms[: len(axes)]
    )

    def _affine(matrix: np.ndarray, name: str) -> Affine:
        return Affine(
            input=voxel_space,
            output=replace(ras_space, name=name),
            matrix=np.asarray(matrix, dtype=np.float64)[:3],
        )

    return [
        vox2phys,
        _affine(image.vox2tkr, _TKR),
        _affine(image.vox2ras, _SCANNER),
    ]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _scanner_matrix(
    transformations: tx.Sequence[Transformation],
) -> np.ndarray:
    """
    The `(4, 4)` voxel-to-scanner RAS matrix, in mm, to store.

    The transformation named `"scanner"` wins; failing that, the
    preferred one, unless it maps to tkr RAS, in which case the last
    transformation that does not. No transformation gives the identity.
    """
    transformations = list(transformations or [])
    chosen = None
    for xform in transformations:
        if getattr(getattr(xform, "output", None), "name", None) == _SCANNER:
            chosen = xform
    if chosen is None:
        for xform in reversed(transformations):
            name = getattr(getattr(xform, "output", None), "name", None)
            if name != _TKR:
                chosen = xform
                break
    if chosen is None:
        return np.eye(4)
    matrix = _voxel_to_ras(chosen)
    output = getattr(chosen, "output", None)
    return _scale_spatial(matrix, _unit_scale(output, "mm"))


def _mgh_dtype(data: tx.Any, dtype: tx.Any = None) -> np.dtype:
    """The voxel type to store an array as (see `MghImage.to_nibabel`)."""
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype not in _MGH_DTYPES:
            raise WriterError(
                f"MGH cannot store voxels of type {dtype}; it stores uint8, "
                f"int16, int32 and float32."
            )
        return dtype
    dtype = np.dtype(getattr(data, "dtype", np.float32))
    if dtype in _MGH_DTYPES:
        return dtype
    if dtype.kind == "b":
        return np.dtype(np.uint8)
    if dtype.kind == "f":
        return np.dtype(np.float32)
    if dtype.kind in "iu":
        if np.can_cast(dtype, np.int16):
            return np.dtype(np.int16)
        if np.can_cast(dtype, np.int32):
            return np.dtype(np.int32)
        info = np.iinfo(np.int32)
        values = np.asarray(data)
        if values.size == 0 or (
            values.min() >= info.min and values.max() <= info.max
        ):
            return np.dtype(np.int32)
        raise WriterError(
            f"MGH stores integers in at most 32 bits, and this {dtype} "
            f"array holds values outside the int32 range."
        )
    raise WriterError(
        f"MGH cannot store voxels of type {dtype}; it stores uint8, int16, "
        f"int32 and float32."
    )


def _stored_dtype(
    data: tx.Any,
    dtype: tx.Any,
    metadata: MghMetadata,
    report: ConversionReport,
) -> np.dtype:
    """
    The voxel type to store: `dtype` when given, else the metadata's
    `data_type` when the values are of its kind (see `preferred_dtype`),
    as the nearest type MGH stores (approximated when it is not the
    same), else the array's own (see `_mgh_dtype`).
    """
    if dtype is not None:
        return _mgh_dtype(data, dtype)
    array_dtype = np.dtype(getattr(data, "dtype", np.float32))
    wanted = preferred_dtype(metadata, array_dtype, report=report)
    if wanted == array_dtype:
        return _mgh_dtype(data)
    nearest = _mgh_dtype(np.empty(0, dtype=wanted))
    if nearest != wanted:
        report.approximated["data_type"] = (
            f"stored as {nearest.name} (MGH stores uint8, int16, int32 "
            f"and float32)"
        )
    return _mgh_dtype(data, nearest)


def _like_header(like: tx.Any) -> tx.Optional[_mgh.MGHHeader]:
    """Resolve a `like` template to the MGH header to copy fields from."""
    if like is None:
        return None
    if isinstance(like, _mgh.MGHHeader):
        return like
    if isinstance(like, _mgh.MGHImage):
        return like.header
    header = getattr(like, "header", None)
    if isinstance(header, _mgh.MGHHeader):
        return header
    if isinstance(like, (str, path.PathLike)):
        return MghImage.from_file(like).header
    return None
