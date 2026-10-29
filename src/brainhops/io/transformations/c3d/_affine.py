"""Affine transformations stored in Convert3D matrix files."""

__all__ = ["C3dAffine"]

import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

from brainhops._core import path
from brainhops._core._internal_api import (
    ArrayProtocol,
    InvalidatorInAttribute,
    smartproperty,
)
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.errors import (
    IncompatibleSystemError,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.metadata._internal_api import (
    metadata_options,
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)
from brainhops.io.transformations.base.affines import (
    _RAS,
    RASToRAS,
    _check_endpoint,
)

from ._metadata import C3dAffineMetadata
from ._raw import C3dAffineRaw

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


# Text adapters supply byte decoding and encoding before the generic bases.
@register_format
class C3dAffine(
    TextFileReader,
    TextFileWriter,
    AffineTransformationFormat,
    RASToRAS,
    TransformationFormat,
):
    """Affine saved by Greedy, `c3d_affine_tool` or ITK-SNAP.

    These three programs of the ITK-SNAP family save an affine as a text
    file of sixteen numbers, the rows of a homogeneous (4, 4) matrix. The
    format is called the RAS or Convert3D format by these programs, and
    such files are usually named `*.mat` or `*.txt`. This class reads and
    writes them.

    ## Conventions

    The matrix maps the RAS world coordinates, in millimetres, of a point
    of the fixed image to those of the matching point of the moving
    image. This is the direction in which a moving image is resampled
    onto the fixed image, which is the direction of the data model, so
    the matrix is read without inversion, from `RASmm` to `RASmm`.

    The matrix is the ITK affine of the same registration, which maps
    fixed to moving points in LPS, with the signs of the first two world
    axes flipped on both sides. Each program converts between the two in
    the same way: `itk_read` and `itk_write` of `c3d_affine_tool` (the
    options `-itk` and `-oitk`), `MapITKTransformToRASMatrix` and
    `MapRASMatrixToITKTransform` of Greedy, and `GetRASMatrix` and
    `ReadAsRASMatrix` of ITK-SNAP. Greedy applies the matrix to the
    points of the fixed image in `MapRASAffineToPhysicalWarp`, and
    `fsl_to_ras` of `c3d_affine_tool` (the option `-fsl2ras`) builds it
    from the inverse of a FLIRT matrix, which maps the reference to the
    moving image. The file carries neither the images nor an inversion:
    `-inv` of `c3d_affine_tool` and the exponent `,-1` of Greedy invert
    the matrix in memory, and the file that they write holds the
    inverted matrix in the same convention.

    ## Dispatch

    Nothing in the content of the file tells it from a FLIRT matrix, a
    NiftyReg affine or any other plain matrix, so the format is never
    chosen from the content. It is selected with `hint="c3d"`,
    `hint="greedy"` or `hint="itksnap"`, or by calling
    `C3dAffine.load(path)`. Without a hint, a `.mat` file of sixteen
    numbers is read as a FLIRT matrix, and a `.txt` file as a plain
    matrix. Writing a RAS affine to a `.mat` file uses this format.

    ## Record

    The file is read in one pass, and its content is held as a
    [`C3dAffineRaw`][] record by the [`C3dAffineMetadata`][] of the
    transformation. The matrix lives in that record. A matrix that is
    assigned is pending: it wins over the record until [`consolidate`][]
    encodes it into a new record. A transformation that was read and
    not changed is written as its file stored it, byte for byte.

    !!! note "Not supported"
        Greedy and `c3d_affine_tool` also write (3, 3) matrices for 2-D
        registrations, which this class refuses. Greedy writes its
        deformable warps as ITK displacement fields, which
        [`ItkNiftiDisplacementField`][brainhops.io.transformations.itk.nifti.ItkNiftiDisplacementField]
        reads with `hint="itk"`.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("c3d", "greedy", "itksnap")

    _metadata: NoRepr[tx.Optional[C3dAffineMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata, and setting
    its data does not create any. Assigning other metadata drops the
    matrix decoded from the old record. A matrix that was assigned is
    pending and is kept, and it wins over the matrix of the new record
    until the transformation is consolidated.
    """

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(3, 4)` affine matrix, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record. Otherwise the
        matrix is the matrix of the record, without its last row, and is
        read-only, or `None` without a record.
        """
        record = _record(self)
        if record is None:
            return None
        return _read_only(_matrix_to_model(record.matrix))

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Decline to recognize a Convert3D matrix from its lines.

        The lines are scored by [`C3dAffineRaw.sniff_lines`][], which
        always declines, so that the format is selected by a hint.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning a confidence.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            `Confidence.NO`.
        """
        return C3dAffineRaw.sniff_lines(lines, error=error, **kwargs)

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Decline to recognize a Convert3D matrix from its content.

        The text is scored by [`C3dAffineRaw.sniff_text`][], which always
        declines, so that the format is selected by a hint.

        Parameters
        ----------
        text : str
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning a confidence.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            `Confidence.NO`.
        """
        return C3dAffineRaw.sniff_text(text, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: C3dAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed [`C3dAffineRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : C3dAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, such as `matrix`, which go to the constructor
            and override the record.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=C3dAffineMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build a transformation from the text of a Convert3D matrix file.

        The text is read by [`C3dAffineMetadata.from_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            Field values, such as `matrix`, which go to the constructor
            and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=C3dAffineMetadata.from_text(text), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from the lines of a Convert3D matrix file.

        The lines are read by [`C3dAffineMetadata.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Field values, such as `matrix`, which go to the constructor
            and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=C3dAffineMetadata.from_lines(lines), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`C3dAffineRaw`][] is read with [`from_raw`][], and any other
        value as the bases read it.

        Parameters
        ----------
        other : object
            A record, a file, its content, a transformation, a mapping, or
            an instance of a similar class.
        *args : Any
            Constructor arguments. A record and a file take keyword
            arguments only.
        **kwargs : Any
            Options of the reader, and field values otherwise.

        Returns
        -------
        Self
            The new transformation.
        """
        if isinstance(other, C3dAffineRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix into a new record.

        A transformation that holds a record, and whose matrix was not
        assigned, is left as it is, since its record also holds its
        matrix. Otherwise, a new record is built from `matrix`, which
        keeps the layout of the rows of the current record, and is held by
        new metadata. In every case, the input and the output must still
        be RAS, since the file can hold nothing else.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a
        Convert3D matrix holds no field of the vocabulary yet, so a
        field that is set is reported as lost, once, under the error
        policy of the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        Self
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If the input or the output is not RAS, or if the matrix is not
            the matrix of a 3-D affine.
        """
        report = ConversionReport(source="C3dAffine", target="C3dAffine")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, C3dAffineMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        _check_systems(self)
        record = _record(self)
        if record is not None and self._data is None:
            if not inplace:
                return replace(self, metadata=metadata)
            if requested:
                self.metadata = metadata
            return self
        record = _build_raw(self, record)
        if metadata is None:
            metadata = C3dAffineMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(self, metadata=metadata, data=None)
        self.metadata = metadata
        self.data = None
        return self

    @takes_error_policy
    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the transformation to a path or a file object.

        The transformation is consolidated into a copy, whose record is
        written, so a transformation that cannot be encoded leaves the file
        untouched, and the transformation itself is not changed.

        Parameters
        ----------
        file : path or file object
            The path or the open text file.
        **kwargs : Any
            Options of the writer of the record, which takes none. The
            fields of the metadata are given to [`consolidate`][].

        Raises
        ------
        UnrepresentableTransformationError
            If the transformation cannot be encoded.
        """
        record, options = _written(self, kwargs)
        return record.to_file(file, **options)

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the content of the file that encodes the transformation.

        The transformation is consolidated into a copy, whose record is
        written.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which takes none. The
            fields of the metadata are given to [`consolidate`][].

        Returns
        -------
        str
            The content of the file.

        Raises
        ------
        UnrepresentableTransformationError
            If the transformation cannot be encoded.
        """
        record, options = _written(self, kwargs)
        return record.to_text(**options)

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the file that encodes the transformation.

        The transformation is consolidated into a copy, whose record is
        written.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which takes none. The
            fields of the metadata are given to [`consolidate`][].

        Yields
        ------
        str
            The lines of the file, without line breaks.

        Raises
        ------
        UnrepresentableTransformationError
            If the transformation cannot be encoded.
        """
        record, options = _written(self, kwargs)
        return record.to_lines(**options)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _record(xform: C3dAffine) -> tx.Optional[C3dAffineRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _read_only(matrix: np.ndarray) -> np.ndarray:
    """Return a matrix decoded from a record, made read-only.

    The writer reads the record and not the cached matrix, so an edit in
    place would be lost. A new matrix is assigned as data instead.
    """
    matrix = np.array(matrix, dtype=np.float64)
    matrix.flags.writeable = False
    return matrix


def _matrix_to_model(matrix: tx.Any) -> np.ndarray:
    """Return the data of a transformation from the matrix of a record.

    The record stores the homogeneous matrix, whose last row is dropped.
    """
    return np.asarray(matrix, dtype=np.float64)[:-1]


def _matrix_to_disk(data: tx.Optional[ArrayProtocol]) -> tx.Tuple:
    """Return the matrix that a record stores for the data of a model.

    The record stores the homogeneous matrix as a tuple of rows of floats.
    Data of `None`, which the data model reads as the identity, is stored
    as the identity.
    """
    if data is None:
        homogeneous = np.eye(4)
    else:
        data = np.asarray(data, dtype=np.float64)
        last = np.zeros((1, data.shape[-1]))
        last[0, -1] = 1.0
        homogeneous = np.concatenate([data, last])
    return tuple(tuple(float(v) for v in row) for row in homogeneous)


def _check_systems(xform: C3dAffine) -> None:
    """Refuse a transformation whose input or output is not RAS.

    `RASToRAS` checks the systems when the transformation is built, and
    this check catches a system that was assigned afterwards.

    Raises
    ------
    UnrepresentableTransformationError
        If the input or the output is not compatible with `RASmm`.
    """
    for name in ("input", "output"):
        try:
            _check_endpoint(xform, name, _RAS)
        except IncompatibleSystemError as e:
            raise UnrepresentableTransformationError(
                f"A Convert3D matrix maps RAS to RAS, but the {name} of the "
                f"transformation is not RAS: {e}"
            ) from e


def _build_raw(
    xform: C3dAffine, old: tx.Optional[C3dAffineRaw]
) -> C3dAffineRaw:
    """Encode the matrix of `xform` as a record.

    The layout of the rows is that of the old record, or that of
    `c3d_affine_tool` without one.

    Raises
    ------
    UnrepresentableTransformationError
        If the matrix is not of shape (3, 4).
    """
    data = xform.data
    shape = None if data is None else tuple(np.shape(data))
    if shape is not None and shape != (3, 4):
        raise UnrepresentableTransformationError(
            f"A Convert3D matrix encodes a 3-D affine, of shape (3, 4), "
            f"but the matrix has shape {shape}."
        )
    return C3dAffineRaw(
        matrix=_matrix_to_disk(data),
        row_end="" if old is None else old.row_end,
    )


def _written(
    xform: C3dAffine, kwargs: tx.Mapping[str, tx.Any]
) -> tx.Tuple[C3dAffineRaw, tx.Dict[str, tx.Any]]:
    """Return the record that a file written from a transformation holds.

    The record is that of the copy consolidated with the fields of the
    metadata among `kwargs`, and the other options, for the writer of
    the record, are returned with it.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation cannot be encoded.
    """
    metadata, options = metadata_options(kwargs)
    return _record(xform.consolidate(**metadata)), options
