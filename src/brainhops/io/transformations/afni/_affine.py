"""AFNI affine matrices, stored as text (`.aff12.1D`, `.1D`)."""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Alias, Inherit, NoRepr, replace

# internals
from brainhops._core._internal_api import InvalidatorInAttribute, smartproperty
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.errors import ConversionError
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    ParserContentError,
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base._internal_api import affine_between

# this format
from ._formats import AfniAffineFormat
from ._geometry import cardinal_to_real_matrix
from ._metadata import AfniAffineMetadata
from ._raw import COMMENT, AfniAffineRaw

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""

_LPS = _systems.LPSmm()
"""The DICOM world of AFNI, which is LPS in millimetres."""

_BASE = ("base", "reference", "ref")
"""The names of the option that gives the base image."""

_SOURCE = ("source", "moving", "mov", "src")
"""The names of the option that gives the source image."""


@register_format
class AfniAffine(
    TextFileReader,
    TextFileWriter,
    AfniAffineFormat,
    _xforms.Affine,
):
    """
    An affine transformation stored in an AFNI matrix file
    (`.aff12.1D`, `.1D`).

    The file holds the 12 numbers of the `(3, 4)` matrix `M` that maps
    the DICOM (LPS) coordinates of the base dataset to those of the
    source dataset, `x_source = M @ x_base`. `3dAllineate
    -1Dmatrix_save`, `3dvolreg -1Dmatrix_save`, `align_epi_anat.py` and
    `cat_matvec -ONELINE` write one matrix per row, and `cat_matvec`
    writes a single matrix as 3 lines of 4 numbers. This is the direction
    in which the data model resamples, from the base to the source, so
    the matrix of the affine is `M` itself, from `LPSmm` to `LPSmm`.

    A matrix file is short and is read in one pass, so the whole file is
    held as an [`AfniAffineRaw`][] record by the [`AfniAffineMetadata`][]
    of the affine, with every matrix of the file and the comment above
    them. The matrix of the affine is a read-only view of that record. A
    matrix that is assigned, and an `input` or `output` that is set, are
    pending until [`consolidate`][] encodes them into a new record. An
    affine without a record, such as one built from data alone, holds its
    matrix as any affine does and has no metadata. An affine that is read
    and written again without changes keeps the bytes of its file.

    A file may hold one matrix per volume of the source, as `3dvolreg`
    writes, or `3dAllineate` on a time series. The data model has no
    affine that changes along the time axis, so the affine is the matrix
    of one volume, `volume`, which is the first one by default
    (`load(path, volume=12)`), and `matrices` gives all of them. A matrix
    that is assigned replaces the matrix of that volume, and the file is
    written back whole.

    AFNI computes on the cardinal grid of each dataset and ignores its
    obliquity, so `M` maps cardinal coordinates. When the `base` and the
    `source` images are given, the matrix of the affine maps their true
    coordinates instead, `c2r(source) @ M @ inv(c2r(base))`, where `c2r`
    is the affine that
    [`cardinal_to_real`][brainhops.io.transformations.afni.cardinal_to_real]
    gives.
    Without them, the two are taken to be the same, as they are for
    datasets that are not oblique, such as templates and most bases of
    `3dQwarp`. `volume`, `base` and `source` say how the record is read,
    and are not written. The matrix is decoded from the record when it is
    first read, so after one of them changes, assigning `data = None`
    decodes the matrix again.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".aff12.1D", ".1D")
    HINTS = ("aff12",)

    _metadata: NoRepr[tx.Optional[AfniAffineMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    An affine built from data alone has no metadata, and setting its
    data does not create any. Assigning other metadata drops the matrix
    decoded from the old record. A matrix that was assigned is pending
    and is kept, and it wins over the matrix of the new record until the
    affine is consolidated.
    """

    volume: int = 0
    """
    The matrix of the file that this affine is, counted from zero.
    Negative values count from the end.
    """

    base: tx.Annotated[tx.Optional[tx.Any], Alias(_BASE)] = None
    """
    The base, or reference, image: an AFNI image or record, a NIfTI
    image or header, or a brainhops image. It is used for its obliquity
    only.
    """

    source: tx.Annotated[tx.Optional[tx.Any], Alias(_SOURCE)] = None
    """
    The source, or moving, image, which is used for its obliquity only.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The DICOM world of the base, `LPSmm`, unless set explicitly."""
        return _systems.LPSmm()

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The DICOM world of the source, `LPSmm`, unless set explicitly."""
        return _systems.LPSmm()

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(3, 4)` base-to-source matrix, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record. Otherwise the
        matrix is that of volume `volume` of the record, in the true
        coordinates of `base` and `source` when they are given, and is
        read-only, or `None` without a record.

        Raises
        ------
        ParserContentError
            If `volume` is out of the range of the matrices of the file.
        """
        record = _record(self)
        if record is None:
            return None
        return _read_only(_world(self, _row(record, self.volume)))

    @property
    def nvolumes(self) -> int:
        """The number of matrices in the file, or 0 without a file."""
        record = _record(self)
        return 0 if record is None else len(record.matrices)

    @property
    def matrices(self) -> tx.Optional[np.ndarray]:
        """Every matrix of the file, as a read-only `(n, 3, 4)` array.

        The matrices are in the coordinates of `matrix`, which are the
        true coordinates of `base` and `source` when they are given.
        """
        record = _record(self)
        if record is None:
            return None
        return _read_only(
            np.stack([_world(self, raw) for raw in record.matrices])
        )

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a text stream holds AFNI matrices.

        The stream is scored by [`AfniAffineRaw.sniff_fileobj`][], which
        takes the name of the stream into account.

        Parameters
        ----------
        file : IO
            The stream, open in text mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no AFNI
            matrices.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return AfniAffineRaw.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that lines hold AFNI matrices.

        The lines are scored by [`AfniAffineRaw.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines hold no AFNI
            matrices.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return AfniAffineRaw.sniff_lines(lines, error=error)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "AfniAffine":
        """Read an affine from the text of an AFNI matrix file.

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            The options `volume`, `base` (or `reference`, `ref`) and
            `source` (or `moving`, `mov`, `src`). Any other option is
            refused.

        Returns
        -------
        AfniAffine
            The affine.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If the text does not hold AFNI matrices, or `volume` is out of
            their range.
        """
        options = _reader_options(kwargs)
        metadata = AfniAffineMetadata.from_text(text, **kwargs)
        _row(metadata.raw, options.get("volume", 0))
        return cls(metadata=metadata, **options)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "AfniAffine":
        """Read an affine from the lines of an AFNI matrix file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without their line breaks.
        **kwargs : Any
            The options of [`from_text`][].

        Returns
        -------
        AfniAffine
            The affine.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ParserContentError
            If the lines do not hold AFNI matrices, or `volume` is out of
            their range.
        """
        return cls.from_text("".join(line + "\n" for line in lines), **kwargs)

    # --- writing ------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        oneline: tx.Optional[bool] = None,
        **fields: tx.Any,
    ) -> "AfniAffine":
        """Encode the pending matrix and systems into a new record.

        An affine that holds a record, whose matrix, `input` and `output`
        were not set, and whose layout `oneline` does not change, is left
        as it is: the copy shares the metadata. Otherwise a new record is
        built. The pending matrix is read from `input` and written in the
        DICOM world of AFNI, where an endpoint that is not known is taken
        to be DICOM, and it is carried to the cardinal coordinates of
        `base` and `source` when they are given. It replaces the matrix of
        volume `volume` of the record, whose other matrices and comment
        are kept. An affine without a record gets a record of its single
        matrix, after the comment that AFNI writes. The new record is held
        by new metadata, and the record held before is never changed.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an AFNI
        affine file holds no field of the vocabulary yet, so a field
        that is set is reported as lost, once, under the error policy of
        the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this affine rather than a copy.
        oneline : bool, optional
            Whether to write each matrix as one row of 12 numbers
            (`-1Dmatrix_save`), or a single matrix as 3 lines of 4
            numbers (`cat_matvec`). It defaults to the layout of the
            record, or to one row per matrix without a record.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        AfniAffine
            The consolidated copy, or this affine with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If several matrices would be written as 3 lines of 4 numbers,
            or the matrix is not that of a 3-D affine.
        UnrepresentableTransformationError
            If the endpoints cannot be bridged to DICOM.
        """
        report = ConversionReport(source="AfniAffine", target="AfniAffine")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, AfniAffineMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        record = _record(self)
        declared = self._input is not None or self._output is not None
        changed = self._data is not None or declared
        if (
            record is not None
            and not changed
            and (oneline is None or bool(oneline) == record.oneline)
        ):
            if not inplace:
                return replace(self, metadata=metadata)
            if requested:
                self.metadata = metadata
            return self
        if record is not None and not changed:
            matrices = record.matrices
        else:
            matrices = _encoded_matrices(self, record)
        if oneline is None:
            oneline = True if record is None else record.oneline
        if not oneline and len(matrices) != 1:
            raise WriterError(
                f"Only a single matrix can be written as 3 lines of 4 "
                f"numbers, and there are {len(matrices)}: use oneline=True."
            )
        if record is not None:
            comment = record.comment
        else:
            comment = COMMENT if oneline else None
        record = AfniAffineRaw(
            matrices=matrices, comment=comment, oneline=oneline
        )
        if metadata is None:
            metadata = AfniAffineMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self, metadata=metadata, data=None, input=None, output=None
            )
        self.metadata = metadata
        self.data = None
        self.input = None
        self.output = None
        return self

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the content of the AFNI matrix file of the affine.

        The affine is consolidated into a copy with the options, as by
        [`consolidate`][], whose record is written: the text that it was
        read from, when nothing changed, and otherwise the comment and the
        matrices with every digit of a double. The affine itself is not
        changed.

        Parameters
        ----------
        **kwargs : Any
            The options of [`consolidate`][].

        Returns
        -------
        str
            The content of the file, ending with a newline.

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        return self.consolidate(**kwargs).metadata.to_text()

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.List[str]:
        """Return the lines of the AFNI matrix file of the affine.

        Parameters
        ----------
        **kwargs : Any
            The options of [`consolidate`][].

        Returns
        -------
        list of str
            The lines of the file, without their line breaks.
        """
        return self.to_text(**kwargs).splitlines()


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _record(xform: AfniAffine) -> tx.Optional[AfniAffineRaw]:
    """Return the record of the metadata of an affine, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _row(record: AfniAffineRaw, volume: int) -> np.ndarray:
    """Return the stored matrix of a volume of a record.

    Raises
    ------
    ParserContentError
        If the volume is out of the range of the matrices of the record.
    """
    count = len(record.matrices)
    if not -count <= int(volume) < count:
        raise ParserContentError(
            f"Volume {volume} is out of range: the file holds {count} "
            f"matrices."
        )
    return record.matrices[int(volume)]


def _world(xform: AfniAffine, raw: np.ndarray) -> np.ndarray:
    """Return a stored `(3, 4)` matrix in the true coordinates of the images.

    The matrix is `c2r(source) @ raw @ inv(c2r(base))` when the images
    are given.
    """
    matrix = _homogeneous(raw)
    if xform.base is not None:
        matrix = matrix @ np.linalg.inv(cardinal_to_real_matrix(xform.base))
    if xform.source is not None:
        matrix = cardinal_to_real_matrix(xform.source) @ matrix
    return matrix[:3]


def _encoded_matrices(
    xform: AfniAffine, record: tx.Optional[AfniAffineRaw]
) -> np.ndarray:
    """Return the matrices that a record stores for a changed affine.

    The matrix of the affine is bridged from its endpoints to DICOM, and
    carried to the cardinal coordinates of its images. It replaces the
    matrix of its volume in the record, or is the only matrix without a
    record.

    Raises
    ------
    WriterError
        If the matrix is not that of a 3-D affine.
    UnrepresentableTransformationError
        If the endpoints cannot be bridged to DICOM.
    """
    matrix = xform.matrix
    matrix = np.eye(3, 4) if matrix is None else np.asarray(matrix, float)
    if matrix.shape != (3, 4):
        raise WriterError(
            f"An AFNI matrix maps 3-D coordinates, and this one has shape "
            f"{matrix.shape}."
        )
    plain = _xforms.Affine(
        matrix=matrix, input=xform._input, output=xform._output
    )
    try:
        lps = affine_between(plain, _LPS, _LPS, AfniAffine)
    except ConversionError as error:
        raise UnrepresentableTransformationError(str(error)) from error
    if xform.base is not None:
        lps = lps @ cardinal_to_real_matrix(xform.base)
    if xform.source is not None:
        lps = np.linalg.inv(cardinal_to_real_matrix(xform.source)) @ lps
    if record is None:
        return lps[None, :3]
    matrices = np.array(record.matrices)
    matrices[int(xform.volume)] = lps[:3]
    return matrices


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Return a `(3, 4)` matrix as a `(4, 4)` homogeneous matrix."""
    out = np.eye(4)
    out[:3] = np.asarray(matrix, dtype=np.float64)[:3]
    return out


def _read_only(matrix: np.ndarray) -> np.ndarray:
    """Return a read-only copy of a matrix decoded from a record.

    The writer reads the record and not the decoded matrix, so an edit in
    place would be lost. To change the matrix, a new matrix is assigned as
    data instead, which stays pending until the affine is consolidated.
    """
    matrix = np.array(matrix, dtype=np.float64)
    matrix.flags.writeable = False
    return matrix


def _reader_options(kwargs: tx.Dict[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Pop the reader options of an affine, under any of their names.

    Raises
    ------
    TypeError
        If an option is given under several of its names.
    """
    options = {}
    if "volume" in kwargs:
        options["volume"] = kwargs.pop("volume")
    for name, aliases in (("base", _BASE), ("source", _SOURCE)):
        given = [alias for alias in aliases if alias in kwargs]
        if len(given) > 1:
            raise TypeError(
                f"The {name} image is given more than once: "
                f"{', '.join(given)}."
            )
        if given:
            options[name] = kwargs.pop(given[0])
    return options
