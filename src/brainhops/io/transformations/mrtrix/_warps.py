"""
MRtrix warps: deformation, displacement and `warpfull` fields stored in
MRtrix images (`.mif`, `.mif.gz`, `.mih`).

What each one holds, and the MRtrix3 sources it was checked against, is
described in the package docstring,
[`brainhops.io.transformations.mrtrix`][].
"""

__all__ = [
    "MrtrixWarp",
    "MrtrixDeformationField",
    "MrtrixDisplacementField",
    "MrtrixWarpFull",
]

# stdlib
import math
import os

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# core
from brainhops._core import affines as _affines
from brainhops._core.properties import smartproperty
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.mrtrix import (
    MrtrixHeader,
    MrtrixParser,
    _merge_keyval,
    _writer_layout,
    dtype_to_mrtrix,
    split_voxel_to_scanner,
)
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.transformations.base import WritableFileBasedTransformation
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    homogeneous_matrix,
    ras_displacement_chain,
    split_ras_displacement_chain,
)

from ._formats import MrtrixTransformationFormat

_NDIM = 3
"""The number of spatial dimensions of an MRtrix warp."""

_NWARPS = 4
"""The number of warps in a `warpfull` file (`check_warp_full`)."""

_LINEAR = ("linear1", "linear2")
"""The header keys of the linear transforms of a `warpfull` file."""

_HISTORY = "command_history"

# The commands whose 4-D output is known to hold one kind of warp. The
# last `command_history` line names the command that wrote the file.
_DEFORMATION_COMMANDS = ("warpinit", "warpcorrect", "mrregister")
_DEFORMATION, _DISPLACEMENT = "deformation", "displacement"


def _is_warp(header: MrtrixHeader) -> bool:
    """Whether a header has the shape of a 4-D warp (`check_warp`)."""
    return header.ndim == 4 and header.dim[3] == _NDIM


def _is_warpfull(header: MrtrixHeader) -> bool:
    """Whether a header has the shape of a `warpfull` (`check_warp_full`)."""
    return (
        header.ndim == 5
        and header.dim[3] == _NDIM
        and header.dim[4] == _NWARPS
    )


def _history_kind(header: MrtrixHeader) -> tx.Optional[str]:
    """
    The kind of warp the command that wrote the file outputs, if known.

    MRtrix appends `argv` to `command_history` in every image it writes
    (`core/app.cpp`), so the last line names the command that wrote this
    file. A few commands output one kind of 4-D warp only:

    - `warpconvert <in> <type> <out>`: `*2displacement` or
      `*2deformation` says which;
    - `warpinvert`: a displacement with `-displacement`, else a
      deformation;
    - `warpinit`, `warpcorrect`, `mrregister -nl_warp`: a deformation.

    Anything else (a file processed further, or written by another
    program) says nothing, and `None` is returned.
    """
    history = header.keyval.get(_HISTORY)
    if not history:
        return None
    last = str(history).strip().split("\n")[-1]
    # Drop the trailing "  (version=...)" that MRtrix appends.
    last = last.split("  (version=", 1)[0]
    tokens = [t.strip("'\"") for t in last.split()]
    if not tokens:
        return None
    command = os.path.basename(tokens[0])
    args = tokens[1:]
    if command == "warpconvert":
        for arg in args:
            if arg.endswith("2" + _DISPLACEMENT):
                return _DISPLACEMENT
            if arg.endswith("2" + _DEFORMATION):
                return _DEFORMATION
        return None
    if command == "warpinvert":
        if any(arg.lstrip("-") == _DISPLACEMENT for arg in args):
            return _DISPLACEMENT
        return _DEFORMATION
    if command in _DEFORMATION_COMMANDS:
        return _DEFORMATION
    return None


def _check_ras(chain: tx.Sequence[_xforms.Transformation], what: str) -> None:
    """Refuse a chain that does not map RAS to RAS world coordinates."""
    src, dst = chain[0].input, chain[-1].output
    if isinstance(src, _systems.RASmm) and isinstance(dst, _systems.RASmm):
        return
    names = [
        "an unspecified system" if s is None else type(s).__name__
        for s in (src, dst)
    ]
    raise UnrepresentableTransformationError(
        f"{what} maps scanner RAS to scanner RAS world coordinates, and "
        f"this chain maps {names[0]} to {names[1]}."
    )


def _field_header(
    shape: tx.Tuple[int, ...],
    vox2ras: np.ndarray,
    dtype: tx.Any,
    source: tx.Optional[MrtrixHeader],
    layout: tx.Optional[tx.Union[str, tx.Sequence[int]]] = None,
    datatype: tx.Optional[tx.Any] = None,
    keyval: tx.Optional[tx.Mapping[str, tx.Optional[str]]] = None,
    **kwargs,
) -> MrtrixHeader:
    """
    The header of a warp of `shape` sampled on the grid `vox2ras`.

    The grid is split into a unit-direction `transform` and voxel sizes,
    as MRtrix stores them. The axes beyond the third have no physical
    size: they keep the one of the `source` header, if it has as many
    axes, and are `nan` otherwise, as MRtrix leaves a new axis. The data
    type defaults to the values' own floating type, else `Float32`, the
    type MRtrix writes its warps in.
    """
    if kwargs:
        raise TypeError(
            f"Unknown MRtrix writer option(s): {', '.join(kwargs)}"
        )
    ndim = len(shape)
    transform, vox = split_voxel_to_scanner(vox2ras)
    vox = list(vox)
    if source is not None and source.ndim == ndim and len(source.vox) == ndim:
        vox += list(source.vox[_NDIM:])
    else:
        vox += [math.nan] * (ndim - _NDIM)
    if datatype is None:
        dtype = np.dtype(dtype)
        datatype = dtype if dtype.kind == "f" else np.float32
    return MrtrixHeader(
        dim=shape,
        vox=vox,
        layout=_writer_layout(layout, source, ndim),
        datatype=dtype_to_mrtrix(datatype),
        transform=transform,
        keyval=_merge_keyval(source, keyval),
    )


def _coordinates_chain(
    coordinates: ArrayProtocol, vox2ras: np.ndarray
) -> tx.Tuple[RASToVoxel, RASCoordinatesField]:
    """The chain that maps RAS to RAS through a field of RAS coordinates
    sampled on the grid `vox2ras`."""
    compact = np.asarray(vox2ras, dtype=np.float64)[:_NDIM]
    return (
        RASToVoxel(matrix=_affines.inv(compact)),
        RASCoordinatesField(field=coordinates),
    )


class MrtrixWarp(
    MrtrixTransformationFormat,
    WritableFileBasedTransformation,
    MrtrixParser,
):
    """
    A warp stored in an MRtrix image.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. [`MrtrixDeformationField`][],
    [`MrtrixDisplacementField`][] and [`MrtrixWarpFull`][] derive from
    it, and answer to `hint="mrtrix.warp"`.

    The header and the values are read by the shared
    [`MrtrixParser`][brainhops.io.base.mrtrix.MrtrixParser], as for an
    MRtrix image: an uncompressed local file stays memory-mapped until
    the field is used. The header is kept, so that its free-form keys
    (`command_history`, ...) are written back.
    """

    # A concrete warp also lists `MrtrixTransformationFormat` among its
    # bases, after this class: hints compose along the declared bases, so
    # that is what qualifies its own hint by `"mrtrix"` alone
    # (`"mrtrix.deformation"`) as well as by `"mrtrix.warp"`
    # (`"mrtrix.warp.deformation"`).
    HINTS = ("warp",)
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mif", ".mif.gz", ".mih")

    order: tx.ClassVar[int] = 1
    """The spline order used to interpolate the field (MRtrix composes
    warps with `Interp::Linear`)."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition used outside of the field of view."""

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        return Confidence.NO

    # --- endpoints ----------------------------------------------------
    #
    # Declared rather than read off the chain: reading them off the
    # chain would build it, and building it decodes the field data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps from: scanner RAS millimetres."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps to: scanner RAS millimetres."""
        return _systems.RASmm()

    # --- decoding -----------------------------------------------------

    def _vox2ras(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-scanner affine of the field's grid."""
        header = self.header
        if header is None:
            raise ParserContentError(
                "This warp has no MRtrix header to read its grid from."
            )
        return header.voxel_to_scanner()

    def _values(self) -> ArrayProtocol:
        """The stored vectors (scaled), `[x, y, z, component, ...]`."""
        data = self._scaled_data()
        if data is None:
            raise ParserContentError("This warp has no data to read.")
        return data

    def _vectors(self) -> ArrayProtocol:
        """The vectors of a 4-D warp, as an `(X, Y, Z, 3)` array."""
        data = self._values()
        shape = tuple(int(d) for d in data.shape)
        if len(shape) != 4 or shape[3] != _NDIM:
            raise ParserContentError(
                f"An MRtrix warp is a 4-D image with 3 volumes (x, y, z), "
                f"not an image of shape {shape}."
            )
        return data


@register_format
class MrtrixDeformationField(
    _xforms.ImmutableSequence, MrtrixWarp, MrtrixTransformationFormat
):
    """
    Field of scanner RAS coordinates (an MRtrix *deformation*), stored
    in an MRtrix image.

    Each voxel holds the scanner-space position, in RAS millimetres, of
    the point that its own position maps to: the "pull-back" warp that
    `mrtransform -warp` takes. Mapping a template point gives the point
    of the moving image to sample.

    | Slot          | Transformation                              |
    | ------------- | ------------------------------------------- |
    | `ras2voxel`   | RAS world coordinates to the field's voxels |
    | `coordinates` | the field of RAS coordinates                |

    Nothing in an MRtrix header tells a deformation from a displacement,
    so a 4-D image with three volumes is read as a deformation -- the
    kind every MRtrix command takes -- unless the command that wrote it
    is known to write displacements. See the package docstring.
    """

    HINTS = ("deformation", "coordinates")

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        """
        Score an MRtrix header as a deformation.

        Any 4-D image with three volumes may be one; it is `MAYBE`, which
        is more than an MRtrix image of that shape scores (`WEAK`). It is
        `LIKELY` if the last command in its history writes deformations,
        and `WEAK` if it writes displacements.
        """
        if not _is_warp(header):
            return Confidence.NO
        kind = _history_kind(header)
        if kind == _DEFORMATION:
            return Confidence.LIKELY
        if kind == _DISPLACEMENT:
            return Confidence.WEAK
        return Confidence.MAYBE

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy another field of RAS coordinates: a chain of an affine
        and a `CoordinatesField`."""
        chain = tuple(getattr(other, "transformations", None) or ())
        if len(chain) != 2 or not isinstance(
            chain[1], _xforms.CoordinatesField
        ):
            raise TypeError(
                f"An MRtrix deformation is a chain of an affine and a "
                f"field of coordinates, not a {type(other).__name__}."
            )
        kwargs.setdefault("transformations", chain)
        return super().from_instance(other, *args, **kwargs)

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        It is built from the MRtrix header and data on first access, and
        cached. Assigning to it overrides the derived chain, which is how
        a field that was not read from a file is built.
        """
        return _coordinates_chain(self._vectors(), self._vox2ras())

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The affine from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def coordinates(self) -> _xforms.Transformation:
        """The field of RAS coordinates, defined on the field's voxels."""
        return self.transformations[1]

    # --- writing ------------------------------------------------------

    def _chain(self) -> tx.Tuple[np.ndarray, ArrayProtocol]:
        """The grid and the `(X, Y, Z, 3)` coordinates to write."""
        what = "An MRtrix deformation"
        chain = tuple(self.transformations or ())
        if len(chain) != 2 or not isinstance(
            chain[1], _xforms.CoordinatesField
        ):
            raise WriterError(
                f"{what} is written from a chain of two transformations: "
                f"RAS to voxel, and a field of RAS coordinates."
            )
        _check_ras(chain, what)
        field = chain[1]
        if field.field is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        if field.coeff:
            raise WriterError(
                f"{what} stores sampled coordinates, and this field holds "
                f"spline coefficients. Convert it to values first."
            )
        ras2vox = homogeneous_matrix(chain[0], what, ndim=_NDIM)
        coordinates = field.field
        shape = tuple(int(s) for s in coordinates.shape)
        if len(shape) == _NDIM + 2 and shape[_NDIM] == 1:
            # The NIfTI layout, (X, Y, Z, 1, 3), keeps a singleton axis.
            coordinates = coordinates[:, :, :, 0]
            shape = shape[:_NDIM] + shape[_NDIM + 1 :]
        if len(shape) != _NDIM + 1 or shape[-1] != _NDIM:
            raise WriterError(
                f"{what} holds one 3-vector per voxel of a 3-D grid, not "
                f"an array of shape {shape}."
            )
        return np.linalg.inv(ras2vox), coordinates

    def _mrtrix_header(self, **kwargs) -> MrtrixHeader:
        """
        Build the header of the field.

        Parameters
        ----------
        layout : str | sequence of int, optional
            The layout to store the values in (default: the layout the
            field was read with, else `+0,+1,+2,+3`).
        datatype : str | dtype, optional
            The MRtrix data type (default: the values' own floating type,
            else `Float32`).
        keyval : mapping, optional
            Extra header keys; a value of `None` removes a key.
        """
        vox2ras, coordinates = self._chain()
        return _field_header(
            tuple(int(s) for s in coordinates.shape),
            vox2ras,
            getattr(coordinates, "dtype", np.float32),
            self.header,
            **kwargs,
        )

    def _mrtrix_data(self) -> tx.Any:
        return np.asarray(self._chain()[1])


@register_format
class MrtrixDisplacementField(
    _xforms.ImmutableSequence, MrtrixWarp, MrtrixTransformationFormat
):
    """
    Field of scanner RAS displacements (an MRtrix *displacement*),
    stored in an MRtrix image.

    Each voxel holds the displacement, in RAS millimetres, from its own
    scanner-space position to the point it maps to: the field maps RAS
    to RAS as `x -> x + u(x)`. This is what `warpconvert ...
    deformation2displacement` writes.

    A `DisplacementField` adds its values in the units of its own grid,
    so the field is the chain of three named slots, as a NIfTI
    `DISPVECT` field is:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    Nothing in an MRtrix header tells a displacement from a deformation,
    so a file is read as a displacement only when the command that wrote
    it is known to write displacements, or when asked for:
    `hint="mrtrix.displacement"`.
    """

    HINTS = ("displacement", "displacements")

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        """
        Score an MRtrix header as a displacement.

        Any 4-D image with three volumes may be one, but MRtrix takes
        deformations, so it is only `WEAK`, below a deformation. It is
        `LIKELY` if the last command in its history writes
        displacements.
        """
        if not _is_warp(header):
            return Confidence.NO
        if _history_kind(header) == _DISPLACEMENT:
            return Confidence.LIKELY
        return Confidence.WEAK

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy another field of RAS displacements: a chain of an affine,
        a `DisplacementField` and an affine."""
        chain = tuple(getattr(other, "transformations", None) or ())
        if len(chain) != 3 or not isinstance(
            chain[1], _xforms.DisplacementField
        ):
            raise TypeError(
                f"An MRtrix displacement is a chain of an affine, a "
                f"displacement field and an affine, not a "
                f"{type(other).__name__}."
            )
        kwargs.setdefault("transformations", chain)
        return super().from_instance(other, *args, **kwargs)

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        It is built from the MRtrix header and data on first access, and
        cached. Assigning to it overrides the derived chain.
        """
        return ras_displacement_chain(
            self._vectors(),
            self._vox2ras(),
            order=self.order,
            bound=self.bound,
        )

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The affine from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def displacement(self) -> _xforms.Transformation:
        """The displacement field, in the voxel units of its grid."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> _xforms.Transformation:
        """The affine from the field's voxels back to RAS world."""
        return self.transformations[2]

    # --- writing ------------------------------------------------------

    def _chain(self) -> tx.Tuple[np.ndarray, ArrayProtocol]:
        """The grid and the `(X, Y, Z, 3)` RAS displacements to write."""
        what = "An MRtrix displacement"
        chain = tuple(self.transformations or ())
        if chain:
            _check_ras(chain, what)
        return split_ras_displacement_chain(chain, what, ndim=_NDIM)

    def _mrtrix_header(self, **kwargs) -> MrtrixHeader:
        """
        Build the header of the field. Takes the options of
        [`MrtrixDeformationField`][]: `layout`, `datatype`, `keyval`.
        """
        vox2ras, vectors = self._chain()
        return _field_header(
            tuple(int(s) for s in vectors.shape),
            vox2ras,
            getattr(vectors, "dtype", np.float32),
            self.header,
            **kwargs,
        )

    def _mrtrix_data(self) -> tx.Any:
        return np.asarray(self._chain()[1])


@register_format
class MrtrixWarpFull(
    _xforms.ImmutableSequence, MrtrixWarp, MrtrixTransformationFormat
):
    """
    The four midway-space warps and two linear transforms of a
    registration, as `mrregister -nl_warp_full` writes them (a 5-D
    MRtrix image, the *warpfull* format).

    The image is sampled on the grid of the *midway* space. Its fourth
    axis holds the `x, y, z` of scanner RAS positions, and its fifth one
    indexes four deformations (`nonlinear.cpp`, `get_output_warps`):

    | Index | Name         | Maps (as a point map)                        |
    | ----- | ------------ | -------------------------------------------- |
    | 0     | `im1_to_mid` | midway -> image 1 (before `linear1`)         |
    | 1     | `mid_to_im1` | image 1 (after `linear1⁻¹`) -> midway        |
    | 2     | `im2_to_mid` | midway -> image 2 (before `linear2`)         |
    | 3     | `mid_to_im2` | image 2 (after `linear2⁻¹`) -> midway        |

    Each is named after the image it moves, in MRtrix's "reverse"
    convention: `im1_to_mid` brings image 1 into the midway space, so it
    maps midway points to image-1 points. The header keys `linear1` and
    `linear2` hold the halves of the linear registration, which map
    midway points to scanner points of image 1 and image 2.

    The transformation this object *is* is the one `mrtransform
    -warp_full` applies (`compose.h`):

    - by default, the whole map from image 2 to image 1, which warps
      image 1 onto image 2 (`-from 1`): `linear2⁻¹`, `mid_to_im2`,
      `im1_to_mid`, `linear1`;
    - with `from_image=2`, the map from image 1 to image 2 (`-from 2`):
      `linear1⁻¹`, `mid_to_im1`, `im2_to_mid`, `linear2`;
    - with `midway=True`, the map from the midway space to image 1 (or 2
      with `from_image=2`), which warps that image to the midway space
      (`-midway_space`): `im1_to_mid` then `linear1`.

    The pieces are available on their own: `linear1`, `linear2`,
    `im1_to_mid`, `mid_to_im1`, `im2_to_mid`, `mid_to_im2`, and
    `chain(from_image, midway)` builds any of the maps above.

    Writing writes back the stored warps and linear transforms (and
    every other header key), whatever `from_image` and `midway` are; a
    new file is built with [`from_warps`][].
    """

    HINTS = ("warpfull", "warp_full")

    from_image: int = 1
    """Which image the warp moves: `1` (the default) maps image-2 points
    (or midway points, with `midway`) to image-1 points; `2` the other
    way. MRtrix's `-from`."""

    midway: bool = False
    """Whether the warp stops at the midway space (MRtrix's
    `-midway_space`), rather than mapping between the two images."""

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        """
        Score an MRtrix header as a warpfull file: a 5-D image of shape
        `(X, Y, Z, 3, 4)`, `CERTAIN` when it also has the `linear1` and
        `linear2` keys MRtrix writes in it.
        """
        if not _is_warpfull(header):
            return Confidence.NO
        if all(key in header.keyval for key in _LINEAR):
            return Confidence.CERTAIN
        return Confidence.LIKELY

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy another warpfull file only: no other transformation
        holds its four warps."""
        if not isinstance(other, MrtrixWarpFull):
            raise TypeError(
                f"An MRtrix warpfull file holds four midway-space warps "
                f"and two linear transforms, and cannot be built from a "
                f"{type(other).__name__}. Build it with `from_warps`."
            )
        return super().from_instance(other, *args, **kwargs)

    @classmethod
    def from_warps(
        cls,
        warps: ArrayProtocol,
        vox2ras: np.ndarray,
        linear1: tx.Optional[tx.Any] = None,
        linear2: tx.Optional[tx.Any] = None,
        keyval: tx.Optional[tx.Mapping[str, str]] = None,
        **kwargs,
    ) -> tx.Self:
        """
        Build a warpfull from its parts.

        Parameters
        ----------
        warps : array, shape `(X, Y, Z, 3, 4)`
            The four deformations, in the order of the table above, as
            scanner RAS positions in millimetres.
        vox2ras : array, shape `(4, 4)`
            The voxel-to-scanner affine of the midway grid.
        linear1, linear2 : array or Affine, optional
            The `(3, 4)` or `(4, 4)` linear transforms, from midway
            points to the points of image 1 and image 2. Default:
            identity.
        keyval : mapping, optional
            Other header keys.
        **kwargs
            `from_image` and `midway`.
        """
        warps = np.asarray(warps)
        shape = tuple(int(s) for s in warps.shape)
        if len(shape) != 5 or shape[3:] != (_NDIM, _NWARPS):
            raise ValueError(
                f"A warpfull holds an array of shape (X, Y, Z, 3, 4), not "
                f"{shape}."
            )
        values = {}
        for key, linear in zip(_LINEAR, (linear1, linear2)):
            values[key] = _format_linear(_linear_matrix(linear))
        values.update(keyval or {})
        header = _field_header(
            shape, np.asarray(vox2ras, float), warps.dtype, None, keyval=values
        )
        return cls(header=header, dataobj=warps, **kwargs)

    # --- parts --------------------------------------------------------

    def _warp(self, index: int) -> _xforms.ImmutableSequence:
        data = self._values()
        shape = tuple(int(d) for d in data.shape)
        if len(shape) != 5 or shape[3:] != (_NDIM, _NWARPS):
            raise ParserContentError(
                f"An MRtrix warpfull is a 5-D image of shape "
                f"(X, Y, Z, 3, 4), not an image of shape {shape}."
            )
        chain = _coordinates_chain(data[..., index], self._vox2ras())
        return _xforms.ImmutableSequence(transformations=chain)

    def _linear(self, key: str) -> _xforms.Affine:
        header = self.header
        value = None if header is None else header.keyval.get(key)
        if value is None:
            raise ParserContentError(
                f"This warpfull has no {key!r} linear transform in its "
                f"header. MRtrix keeps it there only in a .mif/.mih file "
                f"(or a NIfTI with its JSON sidecar)."
            )
        rows = [r.split() for r in str(value).split("\n") if r.strip()]
        try:
            matrix = np.asarray(rows, dtype=np.float64)
        except ValueError as e:
            raise ParserContentError(
                f"The {key!r} linear transform is not numeric: {value!r}"
            ) from e
        if matrix.shape != (_NDIM, _NDIM + 1):
            raise ParserContentError(
                f"The {key!r} linear transform must have 3 rows of 4 "
                f"values, not shape {matrix.shape}."
            )
        return _xforms.Affine(
            matrix=matrix, input=_systems.RASmm(), output=_systems.RASmm()
        )

    @property
    def linear1(self) -> _xforms.Affine:
        """The linear transform from midway points to image-1 points."""
        return self._linear("linear1")

    @property
    def linear2(self) -> _xforms.Affine:
        """The linear transform from midway points to image-2 points."""
        return self._linear("linear2")

    @property
    def im1_to_mid(self) -> _xforms.ImmutableSequence:
        """Warp 0: midway points to image-1 points (before `linear1`)."""
        return self._warp(0)

    @property
    def mid_to_im1(self) -> _xforms.ImmutableSequence:
        """Warp 1: image-1 points (after `linear1⁻¹`) to midway points."""
        return self._warp(1)

    @property
    def im2_to_mid(self) -> _xforms.ImmutableSequence:
        """Warp 2: midway points to image-2 points (before `linear2`)."""
        return self._warp(2)

    @property
    def mid_to_im2(self) -> _xforms.ImmutableSequence:
        """Warp 3: image-2 points (after `linear2⁻¹`) to midway points."""
        return self._warp(3)

    def chain(
        self, from_image: int = 1, midway: bool = False
    ) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of one of the maps the file encodes (see the class
        docstring), as `mrtransform -warp_full [-from N]
        [-midway_space]` composes it.
        """
        if from_image not in (1, 2):
            raise ValueError(f"from_image must be 1 or 2, not {from_image!r}")
        if from_image == 1:
            to_mid, from_mid = self.im1_to_mid, self.mid_to_im2
            linear, other = self.linear1, self.linear2
        else:
            to_mid, from_mid = self.im2_to_mid, self.mid_to_im1
            linear, other = self.linear2, self.linear1
        if midway:
            return (*to_mid, linear)
        return (other.inverse(compute=True), *from_mid, *to_mid, linear)

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations of the map selected by `from_image`
        and `midway`, built on first access and cached.
        """
        return self.chain(self.from_image, self.midway)

    # --- writing ------------------------------------------------------

    def _mrtrix_header(self, **kwargs) -> MrtrixHeader:
        """
        The header the warps were read with (or built with), with the
        writer options of [`MrtrixDeformationField`][] applied:
        `layout`, `datatype`, `keyval`.
        """
        header = self.header
        if header is None or self.dataobj is None:
            raise WriterError(
                "This warpfull has no stored warps to write. Build it with "
                "`MrtrixWarpFull.from_warps`."
            )
        data = self._mrtrix_data()
        return _field_header(
            tuple(int(s) for s in data.shape),
            header.voxel_to_scanner(),
            data.dtype,
            header,
            **kwargs,
        )

    def _mrtrix_data(self) -> tx.Any:
        data = self._values()
        backend = get_array_backend(data)
        return np.asarray(backend.asarray(data))


def _linear_matrix(linear: tx.Any) -> np.ndarray:
    """A `(3, 4)` matrix from an `Affine`, a `(3, 4)` / `(4, 4)` array,
    or `None` (the identity)."""
    if linear is None:
        return np.eye(_NDIM, _NDIM + 1)
    if isinstance(linear, _xforms.Transformation):
        linear = linear.to(_xforms.Affine).homogeneous_matrix
    matrix = np.asarray(linear, dtype=np.float64)
    if matrix.shape == (_NDIM + 1, _NDIM + 1):
        matrix = matrix[:_NDIM]
    if matrix.shape != (_NDIM, _NDIM + 1):
        raise ValueError(
            f"A linear transform is a (3, 4) or (4, 4) matrix, not "
            f"{matrix.shape}."
        )
    return matrix


def _format_linear(matrix: np.ndarray) -> str:
    """A `(3, 4)` matrix as MRtrix writes it in a header key: one row per
    line, values separated by spaces (`str(Eigen::Matrix)`)."""
    return "\n".join(" ".join(repr(float(v)) for v in row) for row in matrix)
