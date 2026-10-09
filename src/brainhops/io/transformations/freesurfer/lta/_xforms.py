__all__ = [
    "LtaFormat",
    "LtaTransformation",
    "LtaTransformationVoxToVox",
    "LtaTransformationPhysToPhys",
    "LtaTransformationRASToRAS",
]

from functools import partial
from warnings import catch_warnings, simplefilter, warn

import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, fields

from brainhops._core import path
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    TextFileParserWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    WritableFileBasedTransformation,
)

from .._formats import FreesurferTransformationFormat
from ._enums import LtaType, LtaValidity
from ._matrix_utils import _get_phys2phys, _get_ras2ras, _get_vox2vox
from ._struct import LtaStruct
from ._systems import LtaCoordinateSystem, LtaPhysicalSystem, LtaVoxelSystem


def _system(
    cls: tx.Type[LtaCoordinateSystem],
    info: tx.Optional[LtaStruct.VolumeInfo],
) -> tx.Optional[LtaCoordinateSystem]:
    """Return the system of a volume.

    `None` is returned, and the system is unknown, when the struct records no
    geometry for the volume.
    """
    return None if info is None else cls.from_struct(info)


class LtaFormat(FreesurferTransformationFormat, AffineTransformationFormat):
    """Marker of affine transformations stored in an LTA file."""

    HINTS = ("lta",)


# `TextFileParserWriter` bridges bytes and text, which the writer of
# `WritableFileBasedTransformation` cannot do, so it must come first.
@register_format
class LtaTransformation(
    LtaFormat,
    TextFileParserWriter,
    _xforms.Affine,
    WritableFileBasedTransformation,
    reverse=False,  # `struct` must be the last field
):
    """Transformation that can be encoded as a Linear Transform Array.

    LTA, the default linear-transform format of FreeSurfer, represents
    several kinds of affine. This class is the registered format for `.lta`
    files, read by `io.load` and written by `io.save`. The views
    [`LtaTransformationVoxToVox`][], [`LtaTransformationPhysToPhys`][] and
    [`LtaTransformationRASToRAS`][] read the same files but are not
    registered, since each would claim every `.lta` file.

    !!! note "What is written"
        An untouched transformation is written back as its struct. One whose
        `matrix`, `input` or `output` was set, including one converted from
        another affine, is written with the LTA type of its systems:

        | `input` and `output`  | LTA type                    |
        | --------------------- | --------------------------- |
        | both `RASmm`          | `LINEAR_RAS_TO_RAS`         |
        | both `RSAmm`          | `LINEAR_RSA_TO_RSA`         |
        | both `LtaVoxelSystem` | `LINEAR_VOX_TO_VOX`         |
        | both `LtaPhysicalSystem` | `LINEAR_PHYSVOX_TO_PHYSVOX` |

        Writing any other pair raises [`UnrepresentableTransformationError`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".lta",)

    struct: LtaStruct = Factory(LtaStruct, repr=False)

    @property
    def input(self) -> LtaCoordinateSystem:
        """The input coordinate system.

        Unless set, it is `RASmm` or `RSAmm` for the RAS and RSA types, and is
        built from the source geometry for the voxel and physical types (`None`
        without a geometry).
        """
        if getattr(self, "_input", None) is not None:
            return self._input
        if self.struct.type == LtaType.LINEAR_RAS_TO_RAS:
            return _systems.RASmm()
        elif self.struct.type == LtaType.LINEAR_RSA_TO_RSA:
            return _systems.RSAmm()
        elif self.struct.type == LtaType.LINEAR_VOX_TO_VOX:
            return _system(LtaVoxelSystem, self.struct.src)
        elif self.struct.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
            return _system(LtaPhysicalSystem, self.struct.src)
        raise AssertionError(f"unsupported LTA type: {self.struct.type}")

    @property
    def output(self) -> LtaCoordinateSystem:
        """The output coordinate system.

        Unless set, it follows from the struct type and the destination
        geometry.
        """
        if getattr(self, "_output", None) is not None:
            return self._output
        if self.struct.type == LtaType.LINEAR_RAS_TO_RAS:
            return _systems.RASmm()
        elif self.struct.type == LtaType.LINEAR_RSA_TO_RSA:
            return _systems.RSAmm()
        elif self.struct.type == LtaType.LINEAR_VOX_TO_VOX:
            return _system(LtaVoxelSystem, self.struct.dst)
        elif self.struct.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
            return _system(LtaPhysicalSystem, self.struct.dst)
        raise AssertionError(f"unsupported LTA type: {self.struct.type}")

    @property
    def data(self) -> np.ndarray:
        """The `(3, 4)` affine matrix.

        Unless set, it is the affine block of the struct, without its last row.
        """
        if getattr(self, "_data", None) is not None:
            return self._data
        return np.asarray(self.struct.affine.matrix, dtype=np.float64)[:-1]

    @input.setter
    def input(self, value: LtaCoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LtaCoordinateSystem) -> None:
        self._output = value

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = value

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a line as the first line of an LTA file."""
        return LtaStruct.sniff_line(line, error=error, **kwargs)

    @classmethod
    def from_(cls, other: tx.Any) -> tx.Self:
        """Build a transformation from a struct, a file or its content.

        !!! warning "Deprecated"
            Use `from_struct` for a struct, `load` for a file, a file object or
            bytes, and `from_text` or `from_lines` for content held in memory.
        """
        if isinstance(other, LtaStruct):
            hint = "use from_struct()"
        else:
            hint = (
                "use load() for a file, a file object or bytes, and "
                "from_text() or from_lines() for content held in memory"
            )
        warn(
            f"{cls.__name__}.from_() is deprecated: {hint}.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(other, LtaStruct):
            return cls.from_struct(other)
        with catch_warnings():
            # The deprecation was reported above, in terms of this class.
            simplefilter("ignore", DeprecationWarning)
            return cls.from_struct(LtaStruct.from_(other))

    @classmethod
    def from_struct(cls, struct: LtaStruct, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed [`LtaStruct`][].

        Keyword arguments such as `input`, `output` or `matrix` go to the
        constructor and override the struct.
        """
        return cls(struct=struct, **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from LTA lines.

        Keyword arguments such as `input`, `output` or `matrix` go to the
        constructor and override the file.
        """
        return cls.from_struct(LtaStruct.from_lines(lines), **kwargs)

    def to_struct(self) -> LtaStruct:
        """Return the [`LtaStruct`][] that encodes the transformation.

        If the matrix, input and output all come from the current struct, that
        struct is returned. Otherwise a new struct takes its type from the
        systems (see [`LtaTransformation`][]), its matrix from `matrix`, its
        volume geometries from the systems (voxel and physical types) or from
        the current struct (RAS and RSA types), and its other header fields
        from the current struct.

        Raises
        ------
        UnrepresentableTransformationError
            If the LTA format cannot encode the systems or the matrix.
        """
        if all(
            getattr(self, name, None) is None
            for name in ("_data", "_input", "_output")
        ):
            return self.struct
        return _build_struct(self)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the transformation to a path or file object in the LTA format.

        The struct is built first, so a transformation that cannot be encoded
        leaves the file untouched.
        """
        return self.to_struct().to_file(file, **kwargs)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the LTA file that encodes the transformation."""
        return self.to_struct().to_lines(**kwargs)

    def to_text(self, **kwargs) -> str:
        """Return the LTA file content that encodes the transformation."""
        return self.to_struct().to_text(**kwargs)


class LtaTransformationVoxToVox(LtaTransformation):
    """LTA file interpreted as a voxel-to-voxel affine."""

    struct: LtaStruct = Factory(
        partial(
            LtaStruct,
            type=LtaType.LINEAR_VOX_TO_VOX,
            src=LtaStruct.SrcVolumeInfo(),
            dst=LtaStruct.DstVolumeInfo(),
        ),
        repr=False,
    )

    @property
    def input(self) -> LtaCoordinateSystem:
        """The voxel system of the source volume, unless set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _system(LtaVoxelSystem, self.struct.src)

    @property
    def output(self) -> LtaCoordinateSystem:
        """The voxel system of the destination volume, unless set."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _system(LtaVoxelSystem, self.struct.dst)

    @property
    def data(self) -> np.ndarray:
        """The voxel-to-voxel matrix derived from the struct, unless set."""
        if getattr(self, "_data", None) is not None:
            return self._data
        return _get_vox2vox(self.struct)[:-1]

    @input.setter
    def input(self, value: LtaCoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LtaCoordinateSystem) -> None:
        self._output = value

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = value


class LtaTransformationPhysToPhys(LtaTransformation):
    """LTA file interpreted as a physical-to-physical affine."""

    struct: LtaStruct = Factory(
        partial(
            LtaStruct,
            type=LtaType.LINEAR_PHYSVOX_TO_PHYSVOX,
            src=LtaStruct.SrcVolumeInfo(),
            dst=LtaStruct.DstVolumeInfo(),
        ),
        repr=False,
    )

    @property
    def input(self) -> LtaCoordinateSystem:
        """The physical system of the source volume, unless set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _system(LtaPhysicalSystem, self.struct.src)

    @property
    def output(self) -> LtaCoordinateSystem:
        """The physical system of the destination volume, unless set."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _system(LtaPhysicalSystem, self.struct.dst)

    @property
    def data(self) -> np.ndarray:
        """Physical-to-physical matrix derived from the struct, unless set."""
        if getattr(self, "_data", None) is not None:
            return self._data
        return _get_phys2phys(self.struct)[:-1]

    @input.setter
    def input(self, value: LtaCoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LtaCoordinateSystem) -> None:
        self._output = value

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = value


class LtaTransformationRASToRAS(LtaTransformation):
    """LTA file interpreted as a RAS-to-RAS affine."""

    struct: LtaStruct = Factory(
        partial(
            LtaStruct,
            type=LtaType.LINEAR_RAS_TO_RAS,
        ),
        repr=False,
    )

    @property
    def input(self) -> LtaCoordinateSystem:
        """The RAS system, unless set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _systems.RASmm()

    @property
    def output(self) -> LtaCoordinateSystem:
        """The RAS system, unless set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _systems.RASmm()

    @property
    def data(self) -> np.ndarray:
        """The RAS-to-RAS matrix derived from the struct, unless set."""
        if getattr(self, "_data", None) is not None:
            return self._data
        return _get_ras2ras(self.struct)[:-1]

    @input.setter
    def input(self, value: LtaCoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LtaCoordinateSystem) -> None:
        self._output = value

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = value


# ----------------------------------------------------------------------
#   Encoding an affine as an LTA struct
# ----------------------------------------------------------------------


def _as_block(
    info: tx.Optional[LtaStruct.VolumeInfo], cls: type
) -> LtaStruct.VolumeInfo:
    """Return a volume geometry as a block of type `cls`.

    Without a geometry the block is invalid, as FreeSurfer writes it.
    """
    if info is None:
        return cls(valid=LtaValidity.VOLUME_INFO_INVALID)
    if type(info) is cls:
        return info
    return cls(**{f.name: getattr(info, f.name) for f in fields(cls)})


def _build_struct(xform: LtaTransformation) -> LtaStruct:
    """Encode the matrix and coordinate systems of `xform` as a struct.

    See [`LtaTransformation.to_struct`][].
    """
    src_system, dst_system = xform.input, xform.output
    old = xform.struct
    Src, Dst = LtaStruct.SrcVolumeInfo, LtaStruct.DstVolumeInfo

    def both(cls: type) -> bool:
        return isinstance(src_system, cls) and isinstance(dst_system, cls)

    if both(_systems.RASmm) or both(_systems.RSAmm):
        # A RAS-to-RAS affine does not depend on the geometry, but FreeSurfer
        # tools use it, so it is kept.
        lta_type = (
            LtaType.LINEAR_RAS_TO_RAS
            if both(_systems.RASmm)
            else LtaType.LINEAR_RSA_TO_RSA
        )
        src, dst = _as_block(old.src, Src), _as_block(old.dst, Dst)
    elif both(LtaVoxelSystem) or both(LtaPhysicalSystem):
        lta_type = (
            LtaType.LINEAR_VOX_TO_VOX
            if both(LtaVoxelSystem)
            else LtaType.LINEAR_PHYSVOX_TO_PHYSVOX
        )
        src = _as_block(getattr(src_system, "struct", None), Src)
        dst = _as_block(getattr(dst_system, "struct", None), Dst)
    else:

        def name(system: tx.Any) -> str:
            if system is None:
                return "an unspecified system"
            return type(system).__name__

        raise UnrepresentableTransformationError(
            f"An LTA file cannot encode an affine from {name(src_system)} "
            f"to {name(dst_system)}: it encodes RASmm to RASmm, RSAmm to "
            f"RSAmm, LtaVoxelSystem to LtaVoxelSystem, and "
            f"LtaPhysicalSystem to LtaPhysicalSystem. Set the input and "
            f"output of the affine to say which it is."
        )

    matrix = xform.matrix
    if matrix is None:
        homogeneous = np.eye(4)
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 4):
            raise UnrepresentableTransformationError(
                f"An LTA file encodes a 3D affine, of shape (3, 4), but "
                f"the matrix has shape {matrix.shape}."
            )
        homogeneous = np.concatenate([matrix, [[0.0, 0.0, 0.0, 1.0]]])

    return LtaStruct(
        type=lta_type,
        nxforms=old.nxforms,
        mean=old.mean,
        sigma=old.sigma,
        affine=LtaStruct.Affine(
            matrix=tuple(tuple(float(v) for v in row) for row in homogeneous)
        ),
        label=old.label,
        src=src,
        dst=dst,
    )
