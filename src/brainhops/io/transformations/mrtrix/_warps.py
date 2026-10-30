"""
MRtrix warps: deformation, displacement and `warpfull` fields stored in
MRtrix images (`.mif`, `.mif.gz`, `.mih`).

What each one holds, and the MRtrix3 sources it was checked against, is
described in the package docstring,
[`brainhops.io.transformations.mrtrix`][].
"""

# stdlib
import math
import os
import warnings
from collections import OrderedDict

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import Inherit, KwOnly, NoRepr

# internals
from brainhops._core import affines as _affines
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._base import register_format
from brainhops.io.base._internal_api import declined, sniffed_raw
from brainhops.io.base.parsers import (
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common.mrtrix import MrtrixMetadata, MrtrixRaw
from brainhops.io.common.mrtrix._internal_api import (
    DECLINE_MESSAGE,
    mrtrix_bytes,
    read_mrtrix,
    reads_from,
    write_mrtrix,
    writer_options,
    written_files,
)
from brainhops.io.images.mrtrix import MrtrixImage
from brainhops.io.metadata._internal_api import metadata_options
from brainhops.io.transformations.base._internal_api import (
    ImageBackedTransformation,
    consolidate_through,
    data_image,
    image_with,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import (
    COORDINATE_HINTS,
    DISPLACEMENT_HINTS,
    RASCoordinateField,
    homogeneous_matrix,
    ras_displacement_chain,
    split_ras_displacement_chain,
    voxel_grid_coordinates,
)

# this format
from ._formats import MrtrixTransformationFormat
from ._views import warp_to_disk, warp_to_model

_NDIM = 3
"""The number of spatial dimensions of an MRtrix warp."""

_NWARPS = 4
"""The number of warps in a `warpfull` file (`check_warp_full`)."""

_LINEAR = ("linear1", "linear2")
"""The header keys of the linear transforms of a `warpfull` file."""

_HISTORY = "command_history"
"""The header key in which MRtrix records the commands that wrote a file."""

# The commands whose 4-D output is known to hold one kind of warp. The
# last `command_history` line names the command that wrote the file.
_DEFORMATION_COMMANDS = ("warpinit", "warpcorrect", "mrregister")
_DEFORMATION, _DISPLACEMENT = "deformation", "displacement"

_FIELD_TYPE = "brainhops_field_type"
"""The header key in which brainhops records the kind of a 4-D warp.

MRtrix keeps a header key that it does not know as free-form text
(`core/formats/mrtrix_utils.h`), and copies it into the images that its
commands derive from the file (`Header::create`, in `core/header.cpp`).
"""

# The values of `brainhops_field_type` that brainhops writes, and those
# that it reads, mapped to the kind of warp.
_FIELD_TYPE_VALUES = {
    _DEFORMATION: "coordinates",
    _DISPLACEMENT: "displacements",
}
_FIELD_TYPE_KINDS = {
    "coordinates": _DEFORMATION,
    "coord": _DEFORMATION,
    "deformation": _DEFORMATION,
    "displacement": _DISPLACEMENT,
    "displacements": _DISPLACEMENT,
    "disp": _DISPLACEMENT,
}

_BRAINHOPS = "brainhops"
"""The command name that brainhops writes in `command_history`."""

_SNIFF_VOXELS = 64**3
"""The number of voxels, at most, that the content of a warp is sniffed
on. A larger warp is subsampled with a regular step."""


class MrtrixWarp(
    BinaryFileWriter, ImageBackedTransformation, MrtrixTransformationFormat
):
    """A warp stored in an MRtrix image.

    A warp keeps its file in a private
    [`MrtrixImage`][brainhops.io.images.mrtrix.MrtrixImage], as it was
    read or as the warp was last consolidated, and `rawdata` and
    `metadata` read that image, as
    [`ImageBackedTransformation`][brainhops.io.transformations.base._image_backed.ImageBackedTransformation]
    describes. The metadata is
    [`MrtrixMetadata`][brainhops.io.common.mrtrix.MrtrixMetadata], whose
    record is an [`MrtrixRaw`][brainhops.io.common.mrtrix.MrtrixRaw].
    When the warp is read from a file, `rawdata` is a lazy array that
    reads the values only when they are needed, as for an MRtrix image.
    Each concrete format decodes its `data`, and the chain of the data
    model, from `rawdata` and from the record, and caches what it
    decodes.

    Values and chains that are assigned are pending until
    [`consolidate`][] encodes them through the image into a copy, or
    until the warp is written, which writes such a copy and leaves the
    warp as it was. The record of the metadata is the base of the header
    that is written, so its free-form keys, such as `command_history`,
    are kept, and a warp that is read and written again without changes
    is copied byte for byte.

    This class is abstract and is not registered as a format, so it never
    takes part in dispatch. [`MrtrixCoordinateField`][],
    [`MrtrixDisplacementField`][] and [`MrtrixWarpFull`][] derive from it,
    and answer to `hint="mrtrix.field"`.
    """

    # A concrete warp also lists `MrtrixTransformationFormat` among its
    # bases, after this class: hints compose along the declared bases, so
    # that is what qualifies its own hint by `"mrtrix"` alone
    # (`"mrtrix.coord"`) as well as by `"mrtrix.field"`
    # (`"mrtrix.field.coord"`).
    HINTS = ("field",)
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mif", ".mih")
    IMAGE = MrtrixImage

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(
        cls, header: MrtrixRaw, file: tx.Optional[tx.IO] = None
    ) -> float:
        """Return how well a valid MRtrix header matches this class.

        Each concrete format overrides this method, and the base returns
        `Confidence.NO`. The stream of the file, when it is given, lets a
        format look at the values when the header does not decide.
        """
        return Confidence.NO

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds this kind of warp.

        The header is read with
        [`MrtrixRaw.from_fileobj`][brainhops.io.common.mrtrix.MrtrixRaw.from_fileobj]
        and scored with `_score_header`. The stream may be gzipped, and its
        position is restored.

        Parameters
        ----------
        file : IO
            The stream, which may be gzipped.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no MRtrix
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        raw, failure = sniffed_raw(MrtrixRaw.from_fileobj, file)
        if raw is None:
            return declined(error, DECLINE_MESSAGE, failure)
        return cls._score_header(raw, file)

    # --- writing ------------------------------------------------------

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of a single-file, uncompressed `.mif` file.

        The warp is consolidated into a copy with the keyword arguments, as
        by [`consolidate`][], and the header of the copy, padded up to the
        first value, is written with its `rawdata`. A lazy array that reaches
        the writer unchanged is written as the file stores it. The warp
        itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            The options of [`consolidate`][]. The fields of the metadata
            are given to [`consolidate`][].

        Returns
        -------
        bytes
            The content of the `.mif` file.

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        fields, kwargs = metadata_options(kwargs)
        xform = self.consolidate(**writer_options(kwargs), **fields)
        return mrtrix_bytes(xform.metadata.raw, xform.rawdata)

    @takes_error_policy
    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write to a path, in the variant that its extension names.

        A `.mif` file holds the header and the values, gzipped for
        `.mif.gz`. A `.mih` header names its values, written next to it in
        a `.dat` file. The warp is consolidated into a copy with the
        keyword arguments, as by [`consolidate`][], which is written, and
        the warp itself is not changed. Each local file is replaced at
        once, so a warp can be written over the files that it was read
        from. When the lazy array of the warp reads a file that is
        written, the warp is consolidated in place instead, so that it
        holds what the files hold (#426), and its lazy array reads the new
        files.

        Parameters
        ----------
        filename : path-like
            The path of the `.mif`, `.mif.gz` or `.mih` file.
        **kwargs : Any
            The options of [`consolidate`][]. The fields of the metadata
            are given to [`consolidate`][].

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        fields, kwargs = metadata_options(kwargs)
        options = {**writer_options(kwargs), **fields}
        reading = reads_from(self.rawdata, written_files(filename))
        if reading:
            mmap = self.rawdata.root.mmap
            xform = self.consolidate(inplace=True, **options)
        else:
            xform = self.consolidate(**options)
        write_mrtrix(filename, xform.metadata.raw, xform.rawdata)
        if reading:
            # The warp holds what the files hold, and its lazy array reads
            # the new files, since the lazy array of the old files describes
            # how they stored the values.
            self.rawdata = read_mrtrix(filename, mmap=mmap)[1]
            self.data = None


@register_format
class MrtrixCoordinateField(
    MrtrixWarp,
    MrtrixTransformationFormat,
    _xforms.DerivedSequence,
):
    """
    Field of scanner RAS coordinates (an MRtrix *deformation*), stored
    in an MRtrix image.

    Each voxel holds the scanner-space position, in RAS millimetres, of
    the point that its own position maps to. This is the "pull-back" warp
    that `mrtransform -warp` takes, so mapping a template point gives the
    point of the moving image to sample.

    The field is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of two named slots:

    | Slot          | Transformation                              |
    | ------------- | ------------------------------------------- |
    | `ras2voxel`   | RAS world coordinates to the field's voxels |
    | `coordinates` | the field of RAS coordinates                |

    An MRtrix header does not say whether a warp holds coordinates or
    displacements. A 4-D image with three volumes is read as a field of
    coordinates, the kind that every MRtrix command takes, unless the
    header key `brainhops_field_type`, the command that wrote the file
    or the values say otherwise, as the package docstring describes.
    The field records `brainhops_field_type: coordinates` when it is
    written.
    """

    HINTS = COORDINATE_HINTS

    degree: tx.ClassVar[int] = 1
    """The spline degree of the interpolation (MRtrix composes warps with
    `Interp::Linear`)."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition outside the field of view."""

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The coordinates given to the field, or `None` to decode them.

    Coordinates stored here are pending until [`consolidate`][] or a write
    encodes them into `rawdata`.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The RAS coordinates as an (X, Y, Z, 3) array.

        Coordinates that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them.
        Otherwise they are the lazy array of the file, which reads its
        values when a computation needs them. Assigning the data drops
        the chain decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return warp_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[MrtrixMetadata]] = Inherit(
        "metadata", default=None, var=True
    )

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(
        cls, header: MrtrixRaw, file: tx.Optional[tx.IO] = None
    ) -> float:
        """Score an MRtrix file as a field of coordinates.

        Any 4-D image with three volumes may be one, so it scores
        `Confidence.MAYBE`, which is more than an MRtrix image of that
        shape scores (`Confidence.WEAK`). The header, and then the values,
        may say more, as the package docstring describes: a field that
        they show to hold coordinates scores the confidence that they
        give, and one that they show to hold displacements scores
        `Confidence.WEAK`.
        """
        if not _is_warp(header):
            return Confidence.NO
        kind, confidence = _field_kind(header, file)
        if kind == _DEFORMATION:
            return confidence
        if kind == _DISPLACEMENT:
            return Confidence.WEAK
        return Confidence.MAYBE

    # --- endpoints ----------------------------------------------------
    # The endpoints are declared rather than read off the chain, which would
    # decode the data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps from: scanner RAS millimetres."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps to: scanner RAS millimetres."""
        return _systems.RASmm()

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True, invalidated_by="_image")
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations that the field encodes.

        The chain is decoded from the record and the data on first access,
        and cached. Assigning a chain overrides the decoded one, which is
        how a field that does not come from a file is built, and the chain
        is pending until [`consolidate`][] or a write encodes it.
        """
        return _coordinates_chain(_vectors(self), _vox2ras(self))

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The affine from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def coordinates(self) -> _xforms.Transformation:
        """The field of RAS coordinates, defined on the field's voxels."""
        return self.transformations[1]

    # --- writing ------------------------------------------------------

    def _consolidate(
        self,
        *,
        keyval: tx.Optional[tx.Mapping[str, tx.Optional[str]]] = None,
        **options: tx.Any,
    ) -> tx.Self:
        """Encode the pending coordinates or chain through the image.

        A field whose chain was not assigned keeps its coordinates,
        assigned or read, under the geometry of its record. A field
        whose chain was assigned is encoded from the chain, whose grid
        replaces the geometry of the record. The image stores the
        values as an MRtrix image stores its data, and the kind of warp
        is recorded in the header key `brainhops_field_type`, with a
        `command_history` line that names brainhops unless it is
        already the last one, so that the file is read back as a field
        of coordinates. An `input` or `output` that was assigned must
        be scanner RAS, which is what the file stores.

        Parameters
        ----------
        keyval : mapping, optional
            Header keys merged into those of the base, after the keys
            that record the kind of warp. A value of `None` removes a
            key. The keys that the other options, the data and the
            geometry decide are refused.
        **options : Any
            The other options of
            [`MrtrixImage.consolidate`][brainhops.io.images.mrtrix.MrtrixImage.consolidate],
            such as `layout`, `datatype` and `scaling`, and the fields
            of the metadata.

        Raises
        ------
        WriterError
            If the field has no coordinates, if its chain is not a grid
            and a field of coordinates, or if `keyval` sets a key that
            the data or the geometry decide.
        UnrepresentableTransformationError
            If the chain, or an assigned `input` or `output`, does not map
            scanner RAS to scanner RAS.
        """
        what = "An MRtrix deformation"
        _check_endpoints(self, what)
        if self._transformations is None:
            image = data_image(self, warp_to_disk)
        else:
            vox2ras, values = _coordinate_parts(self.transformations, what)
            image = image_with(self, warp_to_disk(values), vox2ras)
        return _consolidate_warp(self, image, _DEFORMATION, keyval, options)


@register_format
class MrtrixDisplacementField(
    MrtrixWarp,
    MrtrixTransformationFormat,
    _xforms.DerivedSequence,
):
    """
    Field of scanner RAS displacements (an MRtrix *displacement*),
    stored in an MRtrix image.

    Each voxel holds the displacement, in RAS millimetres, from its own
    scanner-space position to the point it maps to, so the field maps RAS
    to RAS as `x -> x + u(x)`. This is what `warpconvert ...
    deformation2displacement` writes.

    A `DisplacementField` adds its values in the units of its own grid,
    so the field is the chain of three named slots, as a NIfTI `DISPVECT`
    field is:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    An MRtrix header does not say whether a warp holds coordinates or
    displacements, so a file is read as a displacement when the header
    key `brainhops_field_type`, the command that wrote the file or the
    values say so, as the package docstring describes, or when it is
    asked for with `hint="mrtrix.disp"`. The field records
    `brainhops_field_type: displacement` when it is written, so that it
    is read back as a displacement.
    """

    HINTS = DISPLACEMENT_HINTS

    PRIORITY: tx.ClassVar[int] = -1
    """
    Priority below the MRtrix field of coordinates.

    A `.mif` name cannot tell a field of displacements from a field of
    coordinates, and both hold any field exactly. `io.save` writes each
    kind of field in the format of that kind, and a field of another
    kind, such as a velocity, as a field of coordinates, which is what
    every MRtrix command takes.
    """

    degree: tx.ClassVar[int] = 1
    """The spline degree of the interpolation."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition outside the field of view."""

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The displacements given to the field, or `None` to decode them.

    Displacements stored here are pending until [`consolidate`][] or a
    write encodes them into `rawdata`.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The RAS displacements as an (X, Y, Z, 3) array.

        Displacements that are assigned are returned as they are, and
        stay pending until [`consolidate`][] or a write encodes them.
        Otherwise they are the lazy array of the file, which reads its
        values when a computation needs them. Assigning the data drops
        the chain decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return warp_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[MrtrixMetadata]] = Inherit(
        "metadata", default=None, var=True
    )

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(
        cls, header: MrtrixRaw, file: tx.Optional[tx.IO] = None
    ) -> float:
        """Score an MRtrix file as a field of displacements.

        Any 4-D image with three volumes may be one, but MRtrix takes
        fields of coordinates, so it scores only `Confidence.WEAK`, below
        a field of coordinates, unless the header or the values say that
        it holds displacements, as the package docstring describes. It
        then scores the confidence that they give.
        """
        if not _is_warp(header):
            return Confidence.NO
        kind, confidence = _field_kind(header, file)
        if kind == _DISPLACEMENT:
            return confidence
        return Confidence.WEAK

    # --- endpoints ----------------------------------------------------

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps from: scanner RAS millimetres."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps to: scanner RAS millimetres."""
        return _systems.RASmm()

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True, invalidated_by="_image")
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations that the field encodes.

        The chain is decoded from the record and the data on first access,
        and cached. Its displacements in voxel units are read-only.
        Assigning a chain overrides the decoded one, and the chain is
        pending until [`consolidate`][] or a write encodes it.
        """
        chain = ras_displacement_chain(
            _vectors(self),
            _vox2ras(self),
            degree=self.degree,
            bound=self.bound,
        )
        _read_only(chain[1].data)
        return chain

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

    def _consolidate(
        self,
        *,
        keyval: tx.Optional[tx.Mapping[str, tx.Optional[str]]] = None,
        **options: tx.Any,
    ) -> tx.Self:
        """Encode pending displacements or a chain through the image.

        The encoding is that of `MrtrixCoordinateField._consolidate`,
        with RAS displacements in place of RAS coordinates, and with
        `brainhops_field_type: displacement` in the header. A chain that
        was assigned is split into its grid and its displacements,
        rotated back into RAS millimetres.

        Parameters
        ----------
        keyval : mapping, optional
            Header keys merged into those of the base, after the keys
            that record the kind of warp. A value of `None` removes a
            key. The keys that the other options, the data and the
            geometry decide are refused.
        **options : Any
            The other options of
            [`MrtrixImage.consolidate`][brainhops.io.images.mrtrix.MrtrixImage.consolidate],
            such as `layout`, `datatype` and `scaling`, and the fields
            of the metadata.

        Raises
        ------
        WriterError
            If the field has no displacements, if its chain is not a grid
            around a displacement field, or if `keyval` sets a key that
            the data or the geometry decide.
        UnrepresentableTransformationError
            If the chain, or an assigned `input` or `output`, does not map
            scanner RAS to scanner RAS.
        """
        what = "An MRtrix displacement"
        _check_endpoints(self, what)
        if self._transformations is None:
            image = data_image(self, warp_to_disk)
        else:
            chain = tuple(self.transformations or ())
            if chain:
                _check_ras(chain[0].input, chain[-1].output, what)
            vox2ras, values = split_ras_displacement_chain(
                chain, what, ndim=_NDIM
            )
            image = image_with(self, warp_to_disk(values), vox2ras)
        return _consolidate_warp(self, image, _DISPLACEMENT, keyval, options)


@register_format
class MrtrixWarpFull(
    MrtrixWarp,
    MrtrixTransformationFormat,
    _xforms.DerivedSequence,
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

    Each warp is named after the image it moves, in the "reverse"
    convention of MRtrix: `im1_to_mid` brings image 1 into the midway
    space, so it maps midway points to image-1 points. The header keys
    `linear1` and `linear2` hold the halves of the linear registration,
    which map midway points to scanner points of image 1 and image 2.

    The transformation that this object is, its chain, is the one that
    `mrtransform -warp_full` applies (`compose.h`):

    - by default, the whole map from image 2 to image 1, which warps
      image 1 onto image 2 (`-from 1`): `linear2⁻¹`, `mid_to_im2`,
      `im1_to_mid`, `linear1`;
    - with `from_index=2`, the map from image 1 to image 2 (`-from 2`):
      `linear1⁻¹`, `mid_to_im1`, `im2_to_mid`, `linear2`;
    - with `midway=True`, the map from the midway space to image 1 (or 2
      with `from_index=2`), which warps that image to the midway space
      (`-midway_space`): `im1_to_mid` then `linear1`.

    The pieces are available on their own: `linear1`, `linear2`,
    `im1_to_mid`, `mid_to_im1`, `im2_to_mid` and `mid_to_im2`.

    The file holds the pieces, which a chain cannot be split back into, so
    the warps are written from `data`, and the linear transforms and every
    other header key from the record, whatever `from_index` and `midway`
    are. An assigned chain is refused when the file is written. A new file
    is built with [`from_warps`][].
    """

    HINTS = ("warpfull", "warp_full")

    from_index: int = 1
    """
    The index, 1 or 2, of the image that the warp moves: `1`, the
    default, maps image-2 points (or midway points, with `midway`) to
    image-1 points, and `2` maps the other way. This is the `-from`
    option of MRtrix.
    """

    midway: bool = False
    """
    Whether the warp stops at the midway space, which is the
    `-midway_space` option of MRtrix, rather than mapping between the
    two images.
    """

    _data: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The warps given to the object, or `None` to decode them.

    Warps stored here are pending until [`consolidate`][] or a write
    encodes them into `rawdata`.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The four warps as an (X, Y, Z, 3, 4) array of RAS positions.

        Warps that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them.
        Otherwise they are the lazy array of the file, which reads its
        values when a computation needs them. Assigning the data drops
        the chain decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return warp_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[MrtrixMetadata]] = Inherit(
        "metadata", default=None, var=True
    )

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(
        cls, header: MrtrixRaw, file: tx.Optional[tx.IO] = None
    ) -> float:
        """Score an MRtrix header as a warpfull file.

        A 5-D image of shape `(X, Y, Z, 3, 4)` scores `Confidence.LIKELY`,
        and `Confidence.CERTAIN` when it also has the `linear1` and
        `linear2` keys that MRtrix writes in it.
        """
        if not _is_warpfull(header):
            return Confidence.NO
        if all(key in header.keyval for key in _LINEAR):
            return Confidence.CERTAIN
        return Confidence.LIKELY

    @classmethod
    def from_warps(
        cls,
        warps: ArrayProtocol,
        vox2ras: np.ndarray,
        linear1: tx.Optional[tx.Any] = None,
        linear2: tx.Optional[tx.Any] = None,
        *,
        keyval: tx.Optional[tx.Mapping[str, str]] = None,
        from_index: int = 1,
        midway: bool = False,
    ) -> "MrtrixWarpFull":
        """Build a warpfull file from its parts.

        The parts are encoded as a file stores them, so the object holds
        the record and the values that a file written from it holds, and
        nothing is pending.

        Parameters
        ----------
        warps : array, shape `(X, Y, Z, 3, 4)`
            The four deformations, in the order of the table in the class
            docstring, as scanner RAS positions in millimetres.
        vox2ras : array, shape `(4, 4)`
            The voxel-to-scanner affine of the midway grid.
        linear1 : array or Affine, optional
            The `(3, 4)` or `(4, 4)` linear transform from midway points
            to the points of image 1. It defaults to the identity.
        linear2 : array or Affine, optional
            The linear transform from midway points to the points of
            image 2. It defaults to the identity.
        keyval : mapping, optional
            Other header keys.
        from_index : int, default=1
            The index, 1 or 2, of the image that the warp moves.
        midway : bool, default=False
            Whether the warp stops at the midway space.

        Returns
        -------
        MrtrixWarpFull
            The warps.

        Raises
        ------
        ValueError
            If the warps or a linear transform has the wrong shape.
        """
        warps = np.asarray(warps)
        shape = tuple(int(s) for s in warps.shape)
        if len(shape) != 5 or shape[3:] != (_NDIM, _NWARPS):
            raise ValueError(
                f"A warpfull holds an array of shape (X, Y, Z, 3, 4), not "
                f"{shape}."
            )
        values = OrderedDict()
        for key, linear in zip(_LINEAR, (linear1, linear2)):
            values[key] = _format_linear(_linear_matrix(linear))
        values.update(keyval or {})
        grid = [VoxelToRAS(matrix=np.asarray(vox2ras, dtype=np.float64)[:3])]
        image = MrtrixImage(data=warps, transformations=grid)
        return cls(
            image=image.consolidate(keyval=values),
            from_index=from_index,
            midway=midway,
        )

    # --- endpoints ----------------------------------------------------

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps from: scanner RAS millimetres."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space the warp maps to: scanner RAS millimetres."""
        return _systems.RASmm()

    # --- parts --------------------------------------------------------

    @property
    def linear1(self) -> _xforms.Affine:
        """The linear transform from midway points to image-1 points."""
        return _linear(self, "linear1")

    @property
    def linear2(self) -> _xforms.Affine:
        """The linear transform from midway points to image-2 points."""
        return _linear(self, "linear2")

    @property
    def im1_to_mid(self) -> _xforms.ImmutableSequence:
        """Warp 0: midway points to image-1 points (before `linear1`)."""
        return _midway_warp(self, 0)

    @property
    def mid_to_im1(self) -> _xforms.ImmutableSequence:
        """Warp 1: image-1 points (after `linear1⁻¹`) to midway points."""
        return _midway_warp(self, 1)

    @property
    def im2_to_mid(self) -> _xforms.ImmutableSequence:
        """Warp 2: midway points to image-2 points (before `linear2`)."""
        return _midway_warp(self, 2)

    @property
    def mid_to_im2(self) -> _xforms.ImmutableSequence:
        """Warp 3: image-2 points (after `linear2⁻¹`) to midway points."""
        return _midway_warp(self, 3)

    @smartproperty(cache=True, invalidated_by="_image")
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of the map that `from_index` and `midway` select.

        The chain is decoded on first access and cached, as `mrtransform
        -warp_full [-from N] [-midway_space]` composes it (see the class
        docstring). A chain can be assigned, but a file cannot hold it, so
        it is refused when the warps are consolidated or written.
        """
        if self.from_index not in (1, 2):
            raise ValueError(
                f"from_index must be 1 or 2, not {self.from_index!r}"
            )
        if self.from_index == 1:
            to_mid, from_mid = self.im1_to_mid, self.mid_to_im2
            linear, other = self.linear1, self.linear2
        else:
            to_mid, from_mid = self.im2_to_mid, self.mid_to_im1
            linear, other = self.linear2, self.linear1
        if self.midway:
            return (*to_mid, linear)
        return (other.inverse(compute=True), *from_mid, *to_mid, linear)

    # --- writing ------------------------------------------------------

    def _consolidate(self, **options: tx.Any) -> "MrtrixWarpFull":
        """Encode the pending warps through the image.

        The warps, assigned or read, are stored under the geometry of
        the record, whose linear transforms and other keys are kept.
        `from_index` and `midway` select what is decoded, and are not
        written. A chain, an `input` or an `output` that was assigned is
        refused, because the file holds the pieces of the chain and not
        the chain.

        Raises
        ------
        WriterError
            If there are no warps, or if they do not have the shape
            `(X, Y, Z, 3, 4)`.
        UnrepresentableTransformationError
            If a chain, an `input` or an `output` was assigned.
        """
        if (
            self._transformations is not None
            or self._input is not None
            or self._output is not None
        ):
            raise UnrepresentableTransformationError(
                "An MRtrix warpfull file holds four midway-space warps and "
                "two linear transforms, and an assigned chain or endpoint "
                "cannot be split into them. Assign the warps as data, or "
                "build the file with MrtrixWarpFull.from_warps."
            )
        image = data_image(self, warp_to_disk)
        changes = {"data": None}
        return consolidate_through(
            self, image, {}, changes, check=_check_warpfull, **options
        )


# ----------------------------------------------------------------------
#   READING THE HEADER
# ----------------------------------------------------------------------


def _is_warp(header: MrtrixRaw) -> bool:
    """Tell whether a header has the shape of a 4-D warp (`check_warp`)."""
    return header.ndim == 4 and header.dim[3] == _NDIM


def _is_warpfull(header: MrtrixRaw) -> bool:
    """Tell whether a header has the shape of a `warpfull` file
    (`check_warp_full`)."""
    return (
        header.ndim == 5
        and header.dim[3] == _NDIM
        and header.dim[4] == _NWARPS
    )


def _history_kind(header: MrtrixRaw) -> tx.Optional[str]:
    """Return the kind of warp that the command that wrote a file outputs.

    MRtrix appends `argv` to `command_history` in every image it writes
    (`core/app.cpp`), so the last line names the command that wrote this
    file. A few commands output one kind of 4-D warp only.

    - `warpconvert <in> <type> <out>` writes a displacement or a
      deformation, which `*2displacement` or `*2deformation` says.
    - `warpinvert` writes a displacement with `-displacement`, and a
      deformation otherwise.
    - `warpinit`, `warpcorrect` and `mrregister -nl_warp` write a
      deformation.

    Anything else, such as a file processed further or written by another
    program, says nothing, and `None` is returned.
    """
    history = header.keyval.get(_HISTORY)
    if not history:
        return None
    last = str(history).strip().split("\n")[-1]
    # The trailing "  (version=...)" that MRtrix appends is dropped.
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


def _history_lines(header: tx.Optional[MrtrixRaw]) -> tx.List[str]:
    """Return the lines of the `command_history` of a header."""
    if header is None:
        return []
    history = header.keyval.get(_HISTORY)
    if not history:
        return []
    return [line for line in str(history).split("\n") if line.strip()]


def _brainhops_history() -> str:
    """Return the `command_history` line that brainhops writes.

    MRtrix appends the command line and its version to the
    `command_history` of every file that it writes (`core/app.cpp`).
    """
    from brainhops import __version__

    return f"{_BRAINHOPS}  (version={__version__})"


def _written_by_brainhops(header: MrtrixRaw) -> bool:
    """Tell whether the last `command_history` line names brainhops."""
    lines = _history_lines(header)
    if not lines:
        return False
    tokens = lines[-1].split()
    return bool(tokens) and tokens[0] == _BRAINHOPS


def _stored_kind(header: MrtrixRaw) -> tx.Optional[str]:
    """Return the kind of warp that `brainhops_field_type` records."""
    value = header.keyval.get(_FIELD_TYPE)
    if value is None:
        return None
    return _FIELD_TYPE_KINDS.get(str(value).strip().lower())


def _field_kind(
    header: MrtrixRaw, file: tx.Optional[tx.IO] = None
) -> tx.Tuple[tx.Optional[str], float]:
    """Return what a 4-D warp holds, and how sure that is.

    An MRtrix header does not say whether a warp holds coordinates (a
    *deformation*) or displacements, so the clues are weighed in this
    order:

    1. The `brainhops_field_type` key, when brainhops wrote the file
       last, which the last `command_history` line says. The kind that
       the key records is then `Confidence.CERTAIN`.
    2. The last `command_history` line, when it names an MRtrix command
       that writes one kind of warp (`_history_kind`), which gives
       `Confidence.LIKELY`. MRtrix copies the key of brainhops into the
       files that it derives, so a command such as `warpconvert` makes
       the key stale, and the command decides.
    3. The `brainhops_field_type` key under a later command that does
       not change the kind of warp, such as `mrconvert`, which gives
       `Confidence.LIKELY`.
    4. The values, when the stream of the file is given
       (`_content_kind`), which give `Confidence.MAYBE`.

    The kind is `"deformation"` or `"displacement"`, or `None` with
    `Confidence.MAYBE` when nothing decides.
    """
    stored = _stored_kind(header)
    if stored is not None and _written_by_brainhops(header):
        return stored, Confidence.CERTAIN
    kind = _history_kind(header)
    if kind is not None:
        return kind, Confidence.LIKELY
    if stored is not None:
        return stored, Confidence.LIKELY
    if file is not None:
        kind = _content_kind(header, file)
        if kind is not None:
            return kind, Confidence.MAYBE
    return None, Confidence.MAYBE


def _content_kind(header: MrtrixRaw, file: tx.IO) -> tx.Optional[str]:
    """Guess from its values whether a 4-D warp holds coordinates.

    Coordinates follow the scanner position of their voxel, so they
    spread across the field of view, and displacements stay close to
    zero. The warp is taken to hold coordinates when its values spread
    more than their difference from the scanner position of their
    voxels, as the FNIRT reader decides between absolute and relative
    fields. Values of `nan`, which mark points outside the field of
    view, are ignored, and a warp larger than `_SNIFF_VOXELS` voxels is
    subsampled.

    The stream is left where it was. `None` is returned when the values
    cannot be read, for example when the data file of a `.mih` header
    cannot be found from the stream, or when they do not decide, as on a
    single voxel.
    """
    try:
        with preserve_position(file):
            _, proxy = read_mrtrix(file, mmap=False)
            values = np.asarray(proxy, dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None
    spatial = values.shape[:_NDIM]
    count = int(np.prod(spatial))
    step = max(1, int(math.ceil((count / _SNIFF_VOXELS) ** (1 / _NDIM))))
    values = values[::step, ::step, ::step]
    vox2ras = header.voxel_to_scanner() @ np.diag([step] * _NDIM + [1])
    grid = voxel_grid_coordinates(values.shape[:_NDIM], vox2ras)
    axes = tuple(range(_NDIM))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # A component that is `nan` everywhere warns, and is then ignored.
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nansum(np.nanstd(values, axis=axes))
        offset = np.nansum(np.nanstd(values - grid, axis=axes))
    if not (np.isfinite(spread) and np.isfinite(offset)):
        return None
    if spread > offset:
        return _DEFORMATION
    if offset > spread:
        return _DISPLACEMENT
    return None


# ----------------------------------------------------------------------
#   DECODING
# ----------------------------------------------------------------------


def _record(xform: MrtrixWarp) -> tx.Optional[MrtrixRaw]:
    """Return the record of the metadata of a warp, or `None`."""
    if xform.metadata is None:
        return None
    return xform.metadata.raw


def _vox2ras(xform: MrtrixWarp) -> np.ndarray:
    """Return the (4, 4) voxel-to-scanner affine of the grid of a warp.

    The affine is that of the record, or the identity for a warp without
    a record.
    """
    record = _record(xform)
    if record is None:
        return np.eye(4)
    return record.voxel_to_scanner()


def _vectors(xform: MrtrixWarp) -> ArrayProtocol:
    """Return the vectors of a 4-D warp, as an (X, Y, Z, 3) array.

    Raises
    ------
    ParserContentError
        If the warp has no data, or data of another shape.
    """
    data = xform.data
    if data is None:
        raise ParserContentError("This warp has no data to read.")
    shape = tuple(int(d) for d in data.shape)
    if len(shape) != 4 or shape[3] != _NDIM:
        raise ParserContentError(
            f"An MRtrix warp is a 4-D image with 3 volumes (x, y, z), not "
            f"an image of shape {shape}."
        )
    return data


def _coordinates_chain(
    coordinates: ArrayProtocol, vox2ras: np.ndarray
) -> tx.Tuple[RASToVoxel, RASCoordinateField]:
    """Return the RAS-to-RAS chain through a field of RAS coordinates.

    The field is sampled on the grid that `vox2ras` places.
    """
    compact = np.asarray(vox2ras, dtype=np.float64)[:_NDIM]
    return (
        RASToVoxel(matrix=_affines.inv(compact)),
        RASCoordinateField(field=coordinates),
    )


def _midway_warp(
    xform: MrtrixWarpFull, index: int
) -> _xforms.ImmutableSequence:
    """Return one of the four warps of a warpfull file, as a chain.

    Raises
    ------
    ParserContentError
        If the object has no warps, or warps of another shape.
    """
    data = xform.data
    if data is None:
        raise ParserContentError("This warpfull has no warps to read.")
    shape = tuple(int(d) for d in data.shape)
    if len(shape) != 5 or shape[3:] != (_NDIM, _NWARPS):
        raise ParserContentError(
            f"An MRtrix warpfull is a 5-D image of shape (X, Y, Z, 3, 4), "
            f"not an image of shape {shape}."
        )
    chain = _coordinates_chain(data[..., index], _vox2ras(xform))
    return _xforms.ImmutableSequence(transformations=chain)


def _linear(xform: MrtrixWarpFull, key: str) -> _xforms.Affine:
    """Return a linear transform of a warpfull file, read from its record.

    The matrix is read-only, because the writer writes the record and not
    this matrix.

    Raises
    ------
    ParserContentError
        If the record has no such key, or a value that is not a (3, 4)
        matrix.
    """
    record = _record(xform)
    value = None if record is None else record.keyval.get(key)
    if value is None:
        raise ParserContentError(
            f"This warpfull has no {key!r} linear transform in its header. "
            f"MRtrix keeps it there only in a .mif or .mih file, or in the "
            f"JSON sidecar of a NIfTI file."
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
            f"The {key!r} linear transform must have 3 rows of 4 values, "
            f"not shape {matrix.shape}."
        )
    return _xforms.Affine(
        matrix=_read_only(matrix),
        input=_systems.RASmm(),
        output=_systems.RASmm(),
    )


def _read_only(array: tx.Any) -> tx.Any:
    """Make a NumPy array that a view decodes read-only, and return it.

    The writer writes `rawdata` and the record, so an edit in place of a
    decoded array would be lost. Any other array is returned as it is.
    """
    if isinstance(array, np.ndarray):
        array.flags.writeable = False
    return array


# ----------------------------------------------------------------------
#   ENCODING
# ----------------------------------------------------------------------


def _check_ras(
    src: tx.Optional[_systems.CoordinateSystem],
    dst: tx.Optional[_systems.CoordinateSystem],
    what: str,
) -> None:
    """Refuse endpoints that are not scanner RAS world coordinates.

    Raises
    ------
    UnrepresentableTransformationError
        If either endpoint is not `RASmm`.
    """
    if isinstance(src, _systems.RASmm) and isinstance(dst, _systems.RASmm):
        return
    names = [
        "an unspecified system" if s is None else type(s).__name__
        for s in (src, dst)
    ]
    raise UnrepresentableTransformationError(
        f"{what} maps scanner RAS to scanner RAS world coordinates, and "
        f"this one maps {names[0]} to {names[1]}."
    )


def _coordinate_parts(
    chain: tx.Sequence[_xforms.Transformation], what: str
) -> tx.Tuple[np.ndarray, ArrayProtocol]:
    """Return the grid and the (X, Y, Z, 3) coordinates of a chain.

    Raises
    ------
    WriterError
        If the chain is not an affine and a field of coordinates, or if
        the field has no coordinates.
    UnrepresentableTransformationError
        If the chain does not map scanner RAS to scanner RAS.
    """
    chain = tuple(chain or ())
    if len(chain) != 2 or not isinstance(chain[1], _xforms.CoordinateField):
        raise WriterError(
            f"{what} is written from a chain of two transformations: RAS to "
            f"voxel, and a field of RAS coordinates."
        )
    _check_ras(chain[0].input, chain[-1].output, what)
    # MRtrix stores sampled coordinates, so a field of spline coefficients
    # is written as its values.
    coordinates = chain[1].values
    if coordinates is None:
        raise WriterError(
            "This field has no coordinates, so there is nothing to write."
        )
    ras2vox = homogeneous_matrix(chain[0], what, ndim=_NDIM)
    shape = tuple(int(s) for s in coordinates.shape)
    if len(shape) == _NDIM + 2 and shape[_NDIM] == 1:
        # The NIfTI layout, (X, Y, Z, 1, 3), keeps a singleton axis.
        coordinates = coordinates[:, :, :, 0]
        shape = shape[:_NDIM] + shape[_NDIM + 1 :]
    if len(shape) != _NDIM + 1 or shape[-1] != _NDIM:
        raise WriterError(
            f"{what} holds one 3-vector per voxel of a 3-D grid, not an "
            f"array of shape {shape}."
        )
    return np.linalg.inv(ras2vox), coordinates


def _typed_keyval(
    held: tx.Optional[MrtrixRaw],
    keyval: tx.Optional[tx.Mapping[str, tx.Optional[str]]],
    kind: str,
) -> tx.Dict[str, tx.Optional[str]]:
    """Return the header keys that record what a 4-D warp holds.

    MRtrix cannot tell a field of coordinates from a field of
    displacements, so brainhops records the kind in
    `brainhops_field_type`, which MRtrix keeps without reading it, and
    appends its own line to `command_history`, as an MRtrix command
    does, so that a sniffer knows that the key is current
    (`_field_kind`). The line is not repeated when it is already the
    last one. The keys given by the caller are merged last, so that
    `keyval={"brainhops_field_type": None}` leaves the key out.
    """
    line = _brainhops_history()
    lines = _history_lines(held)
    if not lines or lines[-1] != line:
        lines.append(line)
    typed: tx.Dict[str, tx.Optional[str]] = {
        _FIELD_TYPE: _FIELD_TYPE_VALUES[kind],
        _HISTORY: "\n".join(lines),
    }
    typed.update(keyval or {})
    return typed


def _check_warpfull(image: MrtrixImage) -> None:
    """Refuse an image that does not hold the four warps of a warpfull.

    Raises
    ------
    WriterError
        If the image does not have the shape `(X, Y, Z, 3, 4)`.
    """
    shape = tuple(int(d) for d in image.shape)
    if len(shape) != 5 or shape[3:] != (_NDIM, _NWARPS):
        raise WriterError(
            f"An MRtrix warpfull holds an array of shape (X, Y, Z, 3, 4), "
            f"not {shape}."
        )


def _check_endpoints(xform: MrtrixWarp, what: str) -> None:
    """Refuse an assigned `input` or `output` that is not scanner RAS.

    Raises
    ------
    UnrepresentableTransformationError
        If an endpoint was assigned and either endpoint is not `RASmm`.
    """
    if xform._input is not None or xform._output is not None:
        _check_ras(xform.input, xform.output, what)


def _consolidate_warp(
    xform: MrtrixWarp,
    image: MrtrixImage,
    kind: str,
    keyval: tx.Optional[tx.Mapping[str, tx.Optional[str]]],
    options: tx.Dict[str, tx.Any],
) -> MrtrixWarp:
    """Consolidate a 4-D warp through the image of its pending state.

    The header keys that record the kind of warp are given to the image
    when it is encoded again, merged with `keyval`. Keys that the caller
    gives are an option of the call, so the image is then encoded again.
    """
    typed = _typed_keyval(_record(xform), keyval, kind)
    if keyval is not None:
        options["keyval"] = typed
    changes = dict(data=None, transformations=None, input=None, output=None)
    return consolidate_through(
        xform, image, {"keyval": typed}, changes, **options
    )


def _linear_matrix(linear: tx.Any) -> np.ndarray:
    """Return a (3, 4) matrix from an `Affine`, a (3, 4) or (4, 4) array,
    or `None`, which is the identity.

    Raises
    ------
    ValueError
        If the matrix has another shape.
    """
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
    """Return a (3, 4) matrix as MRtrix writes it in a header key.

    Each row is a line, with its values separated by spaces, as
    `str(Eigen::Matrix)` writes them.
    """
    return "\n".join(" ".join(repr(float(v)) for v in row) for row in matrix)
