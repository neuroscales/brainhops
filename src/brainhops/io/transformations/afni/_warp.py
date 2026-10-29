"""AFNI nonlinear warps (the `_WARP` datasets of `3dQwarp`), stored as
AFNI datasets (`.HEAD` and `.BRIK`)."""

# stdlib

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, KwOnly, NoRepr, replace

# core
from brainhops._core import path
from brainhops._core._internal_api import (
    ArrayProtocol,
    array_backend,
    atomic_write_bytes,
    smartproperty,
)

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.datamodel.metadata._internal_api import (
    FIELDS,
    takes_error_policy,
)

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    UnrepresentableTransformationError,
    WriterError,
    disabled,
)
from brainhops.io.common._geometry import ras_conversion
from brainhops.io.common.afni import AfniMetadata, AfniRaw
from brainhops.io.common.afni._constants import DICOM_TO_RAS
from brainhops.io.common.afni._data import (
    brik_proxy,
    read_afni,
    reads_brik,
    write_brik,
)
from brainhops.io.common.afni._files import afni_written_files
from brainhops.io.common.afni._geometry import afni_view, afni_world
from brainhops.io.common.afni._raw import (
    _declined,
    _no_options,
    _sniffed_raw,
    _sniffed_raw_at,
)
from brainhops.io.images.afni import AfniImage
from brainhops.io.metadata._internal_api import requested_fields
from brainhops.io.transformations.base.fields import (
    ras_displacement_chain,
    split_ras_displacement_chain,
)

# this format
from ._formats import AfniWarpFormat
from ._views import _warp_to_disk, _warp_to_model

WARP_LABELS: tx.Tuple[str, str, str] = ("x_delta", "y_delta", "z_delta")
"""The labels that AFNI gives the three sub-bricks of a warp
(`IW3D_to_dataset`, in `mri_nwarp.c`)."""

_DICOM_RAS_VECTOR = np.array([-1.0, -1.0, 1.0])
"""The flip that turns a DICOM (LPS) vector into RAS, and back."""

_NDIM = 3
"""The number of spatial dimensions of an AFNI warp."""


@register_format
class AfniBrikWarp(
    BinaryFileReader,
    BinaryFileWriter,
    AfniWarpFormat,
    _xforms.DerivedSequence,
):
    """
    An AFNI nonlinear warp stored as an AFNI dataset (`.HEAD` and
    `.BRIK`), such as the `<prefix>_WARP+tlrc.HEAD` of `3dQwarp`.

    The dataset has three sub-bricks, the `x`, `y` and `z` displacements
    in DICOM millimetres, labelled `x_delta`, `y_delta` and `z_delta`.
    A warp saved with `3dQwarp -saveall` may have three more, `hexvol`,
    `BulkEn` and `ShearEn`, which are kept but not read. See
    [`AfniWarpFormat`][] for what the displacements mean.

    The warp holds the header of its dataset as [`AfniMetadata`][], whose
    record is an [`AfniRaw`][], and the sub-bricks as the BRIK stores
    them in `rawdata`, as an AFNI image does. When the warp is read from
    a file, `rawdata` is a lazy array that reads the BRIK only when the
    values are needed, and the chain is decoded from `rawdata` and from
    the cardinal grid of the record on first access. Sub-bricks and
    chains that are assigned are pending until [`consolidate`][] encodes
    them into a copy, or until the warp is written, which writes such a
    copy and leaves the warp as it was. The record of the metadata is
    the base of the header that is written, as for an
    [`AfniImage`][brainhops.io.images.afni.AfniImage], so its attributes,
    such as `HISTORY_NOTE`, are kept.

    AFNI puts nothing in the header that says "warp" but the labels of
    the sub-bricks, so the labels are what this reader claims a dataset
    on, with certainty. A dataset of three sub-bricks without them is
    read as an image, and it is read as a warp with `hint="afni.warp"`,
    or with this class directly. A warp that is encoded is given the
    labels when its record does not have them.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = AfniImage.EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("brik",)

    degree: tx.ClassVar[int] = 1
    """The spline degree of the interpolation."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition outside the field of view."""

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The sub-bricks as the BRIK stores them, or `None` without data.

    After a read, `rawdata` is a lazy array of the BRIK, which applies
    the scale factors of the header when it is read. `rawdata` describes
    the dataset as it was read, or as the warp was last consolidated, so
    assigning `data` leaves it as it is.
    """

    _metadata: NoRepr[tx.Optional[AfniMetadata]] = Inherit(
        "metadata", default=None
    )

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The sub-bricks given to the warp, or `None` to decode them.

    Sub-bricks stored here are pending until [`consolidate`][] or a write
    encodes them into `rawdata`.
    """

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the dataset, which holds its record, or `None`.

    Assigning other metadata drops the chain decoded from the previous
    record.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The sub-bricks as an (X, Y, Z, n) array, in DICOM millimetres.

        Sub-bricks that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them.
        Otherwise they are the lazy array of the file, scaled, which
        reads its values when a computation needs them. The first three
        are the `x`, `y` and `z` displacements. Assigning the data drops
        the chain decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return _warp_to_model(self.rawdata)

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(cls, header: AfniRaw) -> float:
        """Score a valid header as a warp.

        A dataset whose first three sub-bricks are labelled as AFNI labels
        the sub-bricks of a warp scores `Confidence.CERTAIN`, and any other
        dataset `Confidence.NO`.
        """
        if header.nvals >= _NDIM and is_warp_labels(header.labels):
            return Confidence.CERTAIN
        return Confidence.NO

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds the header of a warp.

        The position of the stream is restored.

        Parameters
        ----------
        file : IO
            The stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no AFNI
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        raw, failure = _sniffed_raw(file)
        if raw is None:
            return _declined(error, failure)
        return cls._score_header(raw)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a path names the dataset of a warp.

        The `.HEAD` file of the dataset is tested, whichever file of the
        dataset the path names.

        Parameters
        ----------
        filename : path-like
            Any file of the dataset, or the bare dataset name.
        error : bool or type[Exception], default=False
            Whether to raise an error when the path names no AFNI dataset.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        raw, failure = _sniffed_raw_at(filename)
        if raw is None:
            return _declined(error, failure)
        return cls._score_header(raw)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> "AfniBrikWarp":
        """Read a warp from the path of a `.HEAD`, a `.BRIK` or a dataset.

        The record is read from the `.HEAD` file, and the BRIK is checked
        but not read: the lazy array reads it when the values are needed.

        Parameters
        ----------
        filename : path-like
            Any file of the dataset, or the bare dataset name.
        mmap : bool, default=True
            Whether the lazy array memory-maps a local BRIK that is not
            compressed and whose sub-bricks share a type.
        **kwargs : Any
            Refused, since the warp has no other option.

        Returns
        -------
        AfniBrikWarp
            The warp, whose values are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If the `.HEAD` or the `.BRIK` file does not exist.
        ParserContentError
            If the header is invalid, or the BRIK cannot be read.
        """
        _no_options(kwargs)
        raw, proxy = read_afni(filename, mmap=mmap)
        return cls(rawdata=proxy, metadata=AfniMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, mmap: bool = False, **kwargs
    ) -> "AfniBrikWarp":
        """Read a warp from an open `.HEAD` or `.BRIK` file.

        The stream is read as
        [`AfniImage.from_fileobj`][brainhops.io.images.afni.AfniImage.from_fileobj]
        reads it: the other file of the dataset is found from the name of
        the stream, which must have one.

        Parameters
        ----------
        file : BinaryIO
            The stream of the `.HEAD` or the `.BRIK` file.
        mmap : bool, default=False
            Accepted for the readers of paths, and without effect.
        **kwargs : Any
            Refused, since the warp has no other option.

        Returns
        -------
        AfniBrikWarp
            The warp, whose values are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If the stream has no file name.
        """
        image = AfniImage.from_fileobj(file, mmap=mmap, **kwargs)
        return cls(rawdata=image.rawdata, metadata=image.metadata)

    from_bytes = disabled(
        ParserContentError,
        "An AFNI dataset is two files (.HEAD and .BRIK), which bytes "
        "alone cannot hold. Read it from its path instead.",
    )

    # --- endpoints ----------------------------------------------------
    # The endpoints are declared rather than read off the chain, which would
    # decode the data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space that the warp maps from: the RAS world of the base."""
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The space that the warp maps to: the RAS world of the source."""
        return _systems.RASmm()

    # --- chain --------------------------------------------------------

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations that the warp encodes.

        The chain is decoded from the cardinal grid of the record and from
        the first three sub-bricks on first access, and cached. Its
        displacements in voxel units are read-only. Assigning a chain
        overrides the decoded one, and the chain is pending until
        [`consolidate`][] or a write encodes it.
        """
        record = _record(self)
        vox2dicom = np.eye(4) if record is None else record.cardinal_matrix
        return _dicom_chain(_dicom_vectors(self.data), vox2dicom)

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The affine from RAS world coordinates to the warp's voxels."""
        return self.transformations[0]

    @property
    def displacement(self) -> _xforms.Transformation:
        """The displacement field, in the voxel units of its grid."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> _xforms.Transformation:
        """The affine from the warp's voxels back to RAS world."""
        return self.transformations[2]

    def to_image(self) -> AfniImage:
        """Return the warp as an AFNI image, which is what AFNI stores.

        The image holds what a dataset written from the warp holds: the
        sub-bricks of displacements in DICOM (LPS) millimetres, on the
        cardinal grid of the warp, labelled `x_delta`, `y_delta` and
        `z_delta`, and the other attributes of the record of the warp.
        Saved as an AFNI dataset, it is read back as a warp.

        Returns
        -------
        AfniImage
            The image, which shares the consolidated record and values of
            the warp.

        Raises
        ------
        WriterError
            If the chain is not a displacement field between two affines,
            or its grid is oblique, since AFNI places a warp on a
            cardinal grid.
        """
        xform = self.consolidate()
        return AfniImage(rawdata=xform.rawdata, metadata=xform.metadata)

    # --- writing ------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        view: tx.Optional[str] = None,
        datatype: tx.Optional[tx.Any] = None,
        attributes: tx.Optional[tx.Mapping[str, tx.Any]] = None,
        **fields: tx.Any,
    ) -> "AfniBrikWarp":
        """Encode the pending sub-bricks or chain into `rawdata` and a record.

        The warp is encoded as an
        [`AfniImage`][brainhops.io.images.afni.AfniImage] is, by
        `AfniImage._consolidate`,
        over the record of its metadata. A warp whose chain was not
        assigned keeps its sub-bricks, assigned or read, under the geometry
        of its record. A warp whose chain was assigned is encoded from the
        chain: its displacements are written in DICOM millimetres as
        `float32` sub-bricks, on the cardinal grid of the chain, which
        replaces the geometry of the record. The sub-bricks are given the
        labels of a warp when the record does not have them, or has
        another number of sub-bricks. When this encoding gives the record
        of the metadata, the metadata is kept. A warp that has a record
        and nothing pending, consolidated without options, is left as it
        is: the copy shares `rawdata` and the metadata.

        The metadata is consolidated first, as by the AFNI image, with
        the fields given by keyword. The record of an AFNI dataset holds
        no field of the vocabulary yet, so a field that is set is
        reported as lost, once, under the error policy of the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this warp rather than a copy.
        view : {"orig", "acpc", "tlrc"}, optional
            The view of the dataset.
        datatype : str or dtype, optional
            AFNI type name or NumPy type of the stored values.
        attributes : mapping, optional
            Attributes set last, as for an AFNI image.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        AfniBrikWarp
            The consolidated copy, or this warp with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If the warp has no values, if its chain is not a displacement
            field between two affines, or if the options are refused as
            for an AFNI image.
        UnrepresentableTransformationError
            If the chain, or an assigned `input` or `output`, does not map
            RAS to RAS, or the grid of the chain is oblique.
        """
        what = "An AFNI warp"
        options = dict(view=view, datatype=datatype, attributes=attributes)
        declared = self._input is not None or self._output is not None
        if declared:
            _check_ras(self.input, self.output, what)
        if (
            self._data is None
            and self._transformations is None
            and not declared
            and all(value is None for value in options.values())
            and not requested_fields(self.metadata, fields)
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            return self if inplace else replace(self)
        image = _warp_image(self, what).consolidate(
            view=view,
            datatype=datatype,
            attributes=_labelled(self, attributes),
            **fields,
        )
        if self.metadata is not None and image.metadata is self.metadata:
            metadata = self.metadata
        else:
            metadata = image.metadata
        if not inplace:
            return replace(
                self,
                rawdata=image.rawdata,
                metadata=metadata,
                data=None,
                transformations=None,
                input=None,
                output=None,
            )
        self.rawdata = image.rawdata
        self.metadata = metadata
        self.data = None
        self.transformations = None
        self.input = None
        self.output = None
        return self

    @takes_error_policy
    def to_filename(
        self,
        filename: path.FilenameLike,
        view: tx.Optional[str] = None,
        datatype: tx.Optional[tx.Any] = None,
        attributes: tx.Optional[tx.Mapping[str, tx.Any]] = None,
        **fields: tx.Any,
    ) -> None:
        """Write the `.HEAD` and `.BRIK` files of the dataset of the warp.

        The path may name the `.HEAD` file, the `.BRIK` file, whose `.gz`
        or `.bz2` suffix compresses it, or the bare dataset. The warp is
        consolidated into a copy with the options, as by
        [`consolidate`][], whose view defaults to the view in the file
        name, and the header and `rawdata` of the copy are written. The
        warp itself is not changed. As AFNI does, a sibling BRIK with
        another suffix is removed. Each local file is replaced at once.
        When the lazy array of the warp reads a BRIK that is written or
        removed, the warp is consolidated in place instead, so that it
        holds what the dataset holds (#426), and its lazy array reads
        the new BRIK.

        Parameters
        ----------
        filename : path-like
            The `.HEAD` file, the `.BRIK` file or the bare dataset name.
        view : {"orig", "acpc", "tlrc"}, optional
            The view of the dataset, as for [`consolidate`][].
        datatype : str or dtype, optional
            The type of the stored values, as for [`consolidate`][].
        attributes : mapping, optional
            Attributes set last, as for [`consolidate`][].
        **fields : Any
            Fields of the metadata to override, as for [`consolidate`][].
            Any other option is refused.

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        unknown = [name for name in fields if name not in FIELDS]
        if unknown:
            raise TypeError(
                f"Unknown AFNI writer option(s): {', '.join(unknown)}"
            )
        if isinstance(filename, str):
            filename = path.Path(filename)
        head, brik, stale = afni_written_files(filename)
        options = dict(view=view or afni_view(filename), datatype=datatype)
        options["attributes"] = attributes
        options.update(fields)
        reading = reads_brik(self.rawdata, brik) or any(
            reads_brik(self.rawdata, sibling) for sibling in stale
        )
        if reading:
            mmap = self.rawdata.root.mmap
            xform = self.consolidate(inplace=True, **options)
        else:
            xform = self.consolidate(**options)
        if xform.rawdata is None:
            raise WriterError(
                "This warp has no values, so there is nothing to write."
            )
        record = xform.metadata.raw
        write_brik(record, xform.rawdata, brik)
        atomic_write_bytes(head, record.to_bytes())
        for sibling in stale:
            sibling.unlink()
        if reading:
            # The warp holds what the dataset holds, and its lazy array reads
            # the new BRIK, since the lazy array of the old BRIK describes how
            # that BRIK stored the values.
            self.rawdata = brik_proxy(record, brik, mmap)
            self.data = None

    to_fileobj = disabled(
        WriterError,
        "An AFNI dataset is two files (.HEAD and .BRIK), which a "
        "single stream cannot hold. Write it to a path instead.",
    )

    to_bytes = disabled(
        WriterError,
        "An AFNI dataset is two files (.HEAD and .BRIK), which bytes "
        "alone cannot hold. Write it to a path instead.",
    )


# ----------------------------------------------------------------------
#   HELPERS SHARED BY THE WARPS
# ----------------------------------------------------------------------


def is_warp_labels(labels: tx.Optional[tx.Sequence[str]]) -> bool:
    """Tell whether sub-brick labels are those of an AFNI warp.

    Parameters
    ----------
    labels : sequence of str, optional
        The labels of the sub-bricks.

    Returns
    -------
    bool
        Whether the first three labels are `x_delta`, `y_delta` and
        `z_delta`.
    """
    if not labels or len(labels) < _NDIM:
        return False
    return tuple(label.strip() for label in labels[:_NDIM]) == WARP_LABELS


def _dicom_vectors(data: tx.Optional[ArrayProtocol]) -> ArrayProtocol:
    """Return the (X, Y, Z, 3) DICOM displacements of the sub-bricks.

    Raises
    ------
    ParserContentError
        If there are no sub-bricks, or fewer than three.
    """
    if data is None:
        raise ParserContentError("This warp has no data to read.")
    xp = array_backend(data)
    shape = tuple(int(d) for d in data.shape)
    if len(shape) == _NDIM + 2 and shape[_NDIM] == 1:
        # The NIfTI layout, (X, Y, Z, 1, n), keeps a singleton axis.
        data = xp.reshape(data, shape[:_NDIM] + shape[_NDIM + 1 :])
        shape = shape[:_NDIM] + shape[_NDIM + 1 :]
    if len(shape) != _NDIM + 1 or shape[-1] < _NDIM:
        raise ParserContentError(
            f"An AFNI warp has three sub-bricks (x, y and z "
            f"displacements), and these data have the shape {shape}."
        )
    return data[..., :_NDIM]


def _dicom_chain(
    vectors: ArrayProtocol, vox2dicom: np.ndarray
) -> tx.Tuple[_xforms.Transformation, ...]:
    """Return the RAS chain of DICOM displacements on a voxel-to-DICOM grid.

    The displacements and the grid are flipped into RAS, which is an exact
    change of frame, and the displacements in voxel units of the chain are
    read-only.
    """
    vox2ras = DICOM_TO_RAS @ np.asarray(vox2dicom, dtype=np.float64)
    xp = array_backend(vectors)
    flip = xp.asarray(_DICOM_RAS_VECTOR.astype(vectors.dtype))
    chain = ras_displacement_chain(
        vectors * flip,
        vox2ras,
        degree=AfniBrikWarp.degree,
        bound=AfniBrikWarp.bound,
    )
    displacements = chain[1].data
    if isinstance(displacements, np.ndarray):
        displacements.flags.writeable = False
    return chain


def _dicom_field(
    chain: tx.Sequence[_xforms.Transformation], what: str
) -> tx.Tuple[np.ndarray, ArrayProtocol]:
    """Return the cardinal grid and the DICOM displacements of a chain.

    The grid is the (4, 4) voxel-to-DICOM matrix, and the displacements
    are an (X, Y, Z, 3) array in DICOM millimetres. The world of the last
    link of the chain is converted from its anatomical orientation, and a
    world without one is taken to be RAS.

    Raises
    ------
    WriterError
        If the chain is not a displacement field between two affines.
    UnrepresentableTransformationError
        If the chain does not map RAS to RAS, or its grid is oblique,
        since AFNI places a warp on a cardinal grid.
    """
    chain = tuple(chain or ())
    if chain:
        _check_ras(chain[0].input, chain[-1].output, what)
    vox2world, vectors = split_ras_displacement_chain(chain, what, _NDIM)
    to_dicom = DICOM_TO_RAS @ ras_conversion(getattr(chain[2], "output", None))
    vox2dicom = to_dicom @ vox2world
    if not _cardinal(vox2dicom):
        raise UnrepresentableTransformationError(
            "AFNI computes a warp on a cardinal grid, whose axes run along "
            "the DICOM axes, and the grid of this warp is oblique. Resample "
            "it on a cardinal grid first."
        )
    xp = array_backend(vectors)
    rotate = xp.asarray(to_dicom[:3, :3].astype(vectors.dtype))
    vectors = xp.matmul(rotate, vectors[..., None])[..., 0]
    return vox2dicom, vectors


def _check_ras(
    src: tx.Optional[_systems.CoordinateSystem],
    dst: tx.Optional[_systems.CoordinateSystem],
    what: str,
) -> None:
    """Refuse endpoints that are not RAS world coordinates.

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
        f"{what} maps RAS to RAS world coordinates, and this one maps "
        f"{names[0]} to {names[1]}."
    )


def _cardinal(matrix: np.ndarray, tol: float = 1e-6) -> bool:
    """Tell whether each voxel axis of a matrix runs along one world axis."""
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    norms = np.linalg.norm(linear, axis=0)
    if not np.all(norms > 0):
        return False
    unit = np.abs(linear / norms)
    return bool(np.all(np.abs(unit.max(axis=0) - 1) < tol))


# ----------------------------------------------------------------------
#   HELPERS OF THE DATASET WARP
# ----------------------------------------------------------------------


def _record(xform: AfniBrikWarp) -> tx.Optional[AfniRaw]:
    """Return the record of the metadata of a warp, or `None`."""
    if xform.metadata is None:
        return None
    return xform.metadata.raw


def _warp_image(xform: AfniBrikWarp, what: str) -> AfniImage:
    """Return the AFNI image whose consolidation encodes a warp.

    A warp whose chain was not assigned is the image of its sub-bricks,
    assigned or read, under its record. A warp whose chain was assigned
    is the image of the displacements of the chain, as `float32`
    sub-bricks, on the cardinal grid of the chain, in the view of the
    record.

    Raises
    ------
    WriterError
        If the warp has no values, or its chain is not a displacement
        field between two affines.
    UnrepresentableTransformationError
        If the chain does not map RAS to RAS, or its grid is oblique.
    """
    if xform._transformations is None:
        if xform._data is None and xform.rawdata is None:
            raise WriterError(
                "This warp has no values, so there is nothing to write."
            )
        data = None if xform._data is None else _warp_to_disk(xform._data)
        return AfniImage(
            data=data, rawdata=xform.rawdata, metadata=xform.metadata
        )
    vox2dicom, vectors = _dicom_field(xform.transformations, what)
    record = _record(xform)
    world = afni_world(record.view if record is not None else "orig")
    affine = _xforms.Affine(
        matrix=vox2dicom[:3],
        input=_systems.VoxelCoordinateSystem(),
        output=world,
    )
    data = np.asarray(vectors, dtype=np.float32)
    return AfniImage(
        data=data, transformations=[affine], metadata=xform.metadata
    )


def _labelled(
    xform: AfniBrikWarp, attributes: tx.Optional[tx.Mapping[str, tx.Any]]
) -> tx.Optional[tx.Dict[str, tx.Any]]:
    """Return the attributes that encoding a warp sets.

    The labels of a warp are set when the record does not have them, or
    has another number of sub-bricks than what is encoded, unless the
    attributes set the labels themselves.
    """
    attributes = None if attributes is None else dict(attributes)
    if attributes is not None and "BRICK_LABS" in attributes:
        return attributes
    record = _record(xform)
    if xform._transformations is not None:
        nvals = _NDIM
    elif xform._data is not None:
        shape = tuple(int(d) for d in xform._data.shape)
        nvals = shape[_NDIM] if len(shape) > _NDIM else 1
    else:
        nvals = None if record is None else record.nvals
    if (
        record is not None
        and record.nvals == nvals
        and is_warp_labels(record.labels)
    ):
        return attributes
    attributes = {} if attributes is None else attributes
    labels = list(WARP_LABELS)
    if nvals is not None and nvals > _NDIM:
        old = [] if record is None else list(record.labels or [])
        labels += old[_NDIM:nvals] + [""] * (nvals - max(len(old), _NDIM))
    attributes["BRICK_LABS"] = "\0".join(labels)
    return attributes
