"""The metadata of plain matrix files, one class for each container."""

__all__ = [
    "MatrixMetadata",
    "TxtMatrixMetadata",
    "CsvMatrixMetadata",
    "TsvMatrixMetadata",
    "NpyMatrixMetadata",
    "NpzMatrixMetadata",
    "MatLegacyMatrixMetadata",
    "Mat73MatrixMetadata",
]

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops._core.streams import preserve_position
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, ParserContentError
from brainhops.io.common._arrays import (
    ArrayContainerError,
    ArrayReader,
    CsvArrayReader,
    Mat73ArrayReader,
    MatLegacyArrayReader,
    NpyArrayReader,
    NpzArrayReader,
    TsvArrayReader,
    TxtArrayReader,
    _read_mat_legacy,
    is_numeric_array,
    read_mat73,
    read_npy,
    read_npz,
    read_text_array,
    select_array,
)
from brainhops.io.metadata import MetadataFormat

from . import _conventions
from ._raw import MatrixAffineRaw


class MatrixMetadata(MetadataFormat, ArrayReader):
    """Metadata of a plain matrix file, which holds the file as a record.

    A plain matrix file holds a small matrix and nothing else, so it is
    read in one pass, and its record, a [`MatrixAffineRaw`][], holds the
    whole file together with the conventions that the caller stated when
    the file was read. The data of a plain matrix format, such as a
    [`TxtMatrixAffine`][brainhops.io.transformations.matrix.TxtMatrixAffine],
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The metadata keeps no open file.

    This class is the abstract base of the metadata of each container,
    which is registered, and it is not registered itself. The readers
    take the conventions of the matrix as options, and the readers of
    `.npz` and `.mat` files also take `variable` (or its alias `key`).
    Any other option raises `TypeError`. The metadata cannot be written,
    because the plain matrix formats have no writer.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("matrix",)
    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    # Plain matrices are tiny, so a large file is not claimed.
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 1 << 20

    raw: NoRepr[tx.Optional[MatrixAffineRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        return is_numeric_array(array) and array.ndim == 2

    @classmethod
    def _array_confidence(cls, array: np.ndarray, **kwargs) -> float:
        """Score `Confidence.WEAK` for an array that reads as an affine."""
        vector = kwargs.get("vector", "column")
        ndim = kwargs.get("ndim", None)
        if not _conventions.is_affine_matrix(array, vector, ndim):
            return Confidence.NO
        return cls.SNIFF_CONFIDENCE


# ----------------------------------------------------------------------
#   TEXT
# ----------------------------------------------------------------------


@register_format
class TxtMatrixMetadata(TxtArrayReader, MatrixMetadata):
    """Metadata of a whitespace-separated text matrix file.

    This covers `.txt`, `.dat` and AFNI `.1D` files. The record keeps the
    text of the file. See [`MatrixMetadata`][].
    """

    HINTS = TxtArrayReader.HINTS

    @classmethod
    def from_text(cls, text: str, **options) -> "TxtMatrixMetadata":
        """Read the metadata from the text of the file.

        Parameters
        ----------
        text : str
            The content of the file.
        **options : Any
            The conventions of the matrix, as
            [`MatrixAffineRaw.from_array`][brainhops.io.transformations.matrix.MatrixAffineRaw.from_array]
            takes them.

        Returns
        -------
        TxtMatrixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(
            _text_record(cls, text, TxtArrayReader.SEPARATORS, options)
        )

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **options
    ) -> "TxtMatrixMetadata":
        """Read the metadata from the lines of the file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **options : Any
            The conventions of the matrix.

        Returns
        -------
        TxtMatrixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_text(_joined(lines), **options)


@register_format
class CsvMatrixMetadata(CsvArrayReader, MatrixMetadata):
    """Metadata of a comma-separated `.csv` matrix file.

    The record keeps the text of the file. See [`MatrixMetadata`][].
    """

    HINTS = CsvArrayReader.HINTS

    @classmethod
    def from_text(cls, text: str, **options) -> "CsvMatrixMetadata":
        """Read the metadata from the text of the file.

        Parameters
        ----------
        text : str
            The content of the file.
        **options : Any
            The conventions of the matrix.

        Returns
        -------
        CsvMatrixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(
            _text_record(cls, text, CsvArrayReader.SEPARATORS, options)
        )

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **options
    ) -> "CsvMatrixMetadata":
        """Read the metadata from the lines of the file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **options : Any
            The conventions of the matrix.

        Returns
        -------
        CsvMatrixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_text(_joined(lines), **options)


@register_format
class TsvMatrixMetadata(TsvArrayReader, MatrixMetadata):
    """Metadata of a tab-separated `.tsv` matrix file.

    The record keeps the text of the file. See [`MatrixMetadata`][].
    """

    HINTS = TsvArrayReader.HINTS

    @classmethod
    def from_text(cls, text: str, **options) -> "TsvMatrixMetadata":
        """Read the metadata from the text of the file.

        Parameters
        ----------
        text : str
            The content of the file.
        **options : Any
            The conventions of the matrix.

        Returns
        -------
        TsvMatrixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(
            _text_record(cls, text, TsvArrayReader.SEPARATORS, options)
        )

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **options
    ) -> "TsvMatrixMetadata":
        """Read the metadata from the lines of the file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **options : Any
            The conventions of the matrix.

        Returns
        -------
        TsvMatrixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_text(_joined(lines), **options)


# ----------------------------------------------------------------------
#   NUMPY
# ----------------------------------------------------------------------


@register_format
class NpyMatrixMetadata(NpyArrayReader, MatrixMetadata):
    """Metadata of a NumPy `.npy` matrix file, read without unpickling.

    See [`MatrixMetadata`][].
    """

    HINTS = NpyArrayReader.HINTS

    @classmethod
    def from_fileobj(cls, file: tx.IO, **options) -> "NpyMatrixMetadata":
        """Read the metadata from a binary file object.

        The whole stream is read, because the file holds nothing but the
        matrix.

        Parameters
        ----------
        file : file object
            The file, open for binary reading.
        **options : Any
            The conventions of the matrix.

        Returns
        -------
        NpyMatrixMetadata
            The metadata, which holds the record of the file.
        """
        with preserve_position(file):
            content = file.read()
        return cls.from_bytes(content, **options)

    @classmethod
    def from_bytes(cls, content: bytes, **options) -> "NpyMatrixMetadata":
        """Read the metadata from the content of the file.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **options : Any
            The conventions of the matrix.

        Returns
        -------
        NpyMatrixMetadata
            The metadata, which holds the record of the file.

        Raises
        ------
        TypeError
            If an option is not a convention, such as `variable`, which a
            file of a single array does not take.
        ParserContentError
            If the content is not a `.npy` file of an affine matrix.
        """
        _no_variable(cls, options)
        try:
            array = read_npy(bytes(content))
        except ArrayContainerError as e:
            raise ParserContentError(str(e)) from e
        _check_accepted(cls, None, array)
        return cls.from_raw(MatrixAffineRaw.from_array(array, **options))


@register_format
class NpzMatrixMetadata(NpzArrayReader, MatrixMetadata):
    """Metadata of a NumPy `.npz` matrix file, read without unpickling.

    The array is selected with `variable` or `key`, and defaults to the
    only 2-D numeric array. The record keeps its name. See
    [`MatrixMetadata`][].
    """

    HINTS = NpzArrayReader.HINTS

    @classmethod
    def from_fileobj(cls, file: tx.IO, **options) -> "NpzMatrixMetadata":
        """Read the metadata from a binary file object.

        Parameters
        ----------
        file : file object
            The file, open for binary reading.
        **options : Any
            The conventions of the matrix, and `variable` or `key`.

        Returns
        -------
        NpzMatrixMetadata
            The metadata, which holds the record of the file.
        """
        with preserve_position(file):
            content = file.read()
        return cls.from_bytes(content, **options)

    @classmethod
    def from_bytes(cls, content: bytes, **options) -> "NpzMatrixMetadata":
        """Read the metadata from the content of the file.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **options : Any
            The conventions of the matrix, and `variable` or `key`.

        Returns
        -------
        NpzMatrixMetadata
            The metadata, which holds the record of the file.

        Raises
        ------
        TypeError
            If an option is not a convention, `variable` or `key`.
        ParserContentError
            If the content is not a `.npz` file, or holds no single array
            that reads as an affine matrix.
        """
        variable = _variable(options)
        try:
            arrays = read_npz(bytes(content))
        except ArrayContainerError as e:
            raise ParserContentError(str(e)) from e
        name, array = _selected(cls, arrays, variable)
        return cls.from_raw(
            MatrixAffineRaw.from_array(array, variable=name, **options)
        )


# ----------------------------------------------------------------------
#   MATLAB
# ----------------------------------------------------------------------


@register_format
class MatLegacyMatrixMetadata(MatLegacyArrayReader, MatrixMetadata):
    """Metadata of a MATLAB v4 or v5 to v7 `.mat` matrix file.

    The file is read with `scipy.io`. The variable is selected with
    `variable` or `key`, and defaults to the only 2-D numeric array. The
    record keeps its name. See [`MatrixMetadata`][].
    """

    @classmethod
    def from_fileobj(
        cls, file: tx.IO, **options
    ) -> "MatLegacyMatrixMetadata":
        """Read the metadata from a binary file object.

        Parameters
        ----------
        file : file object
            The file, open for binary reading.
        **options : Any
            The conventions of the matrix, and `variable` or `key`.

        Returns
        -------
        MatLegacyMatrixMetadata
            The metadata, which holds the record of the file.
        """
        with preserve_position(file):
            content = file.read()
        return cls.from_bytes(content, **options)

    @classmethod
    def from_bytes(
        cls, content: bytes, **options
    ) -> "MatLegacyMatrixMetadata":
        """Read the metadata from the content of the file.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **options : Any
            The conventions of the matrix, and `variable` or `key`.

        Returns
        -------
        MatLegacyMatrixMetadata
            The metadata, which holds the record of the file.

        Raises
        ------
        TypeError
            If an option is not a convention, `variable` or `key`.
        ParserContentError
            If the content is not a MATLAB v4 to v7 file, or holds no
            single variable that reads as an affine matrix.
        """
        variable = _variable(options)
        try:
            arrays = _read_mat_legacy(bytes(content))
        except ArrayContainerError as e:
            raise ParserContentError(str(e)) from e
        name, array = _selected(cls, arrays, variable)
        return cls.from_raw(
            MatrixAffineRaw.from_array(array, variable=name, **options)
        )


@register_format
class Mat73MatrixMetadata(Mat73ArrayReader, MatrixMetadata):
    """Metadata of a MATLAB v7.3 (HDF5) `.mat` matrix file.

    The file is read with `h5py`, and MATLAB stores arrays transposed,
    which the reader undoes. The variable is selected with `variable` or
    `key`, and defaults to the only 2-D numeric array. The record keeps
    its name. See [`MatrixMetadata`][].
    """

    HINTS = Mat73ArrayReader.HINTS

    @classmethod
    def from_fileobj(cls, file: tx.IO, **options) -> "Mat73MatrixMetadata":
        """Read the metadata from a binary file object.

        Parameters
        ----------
        file : file object
            The file, open for binary reading.
        **options : Any
            The conventions of the matrix, and `variable` or `key`.

        Returns
        -------
        Mat73MatrixMetadata
            The metadata, which holds the record of the file.
        """
        with preserve_position(file):
            content = file.read()
        return cls.from_bytes(content, **options)

    @classmethod
    def from_bytes(cls, content: bytes, **options) -> "Mat73MatrixMetadata":
        """Read the metadata from the content of the file.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **options : Any
            The conventions of the matrix, and `variable` or `key`.

        Returns
        -------
        Mat73MatrixMetadata
            The metadata, which holds the record of the file.

        Raises
        ------
        TypeError
            If an option is not a convention, `variable` or `key`.
        ParserContentError
            If the content is not a MATLAB v7.3 file, or holds no single
            variable that reads as an affine matrix.
        ImportError
            If h5py is not installed.
        """
        variable = _variable(options)
        try:
            arrays = read_mat73(bytes(content))
        except ArrayContainerError as e:
            raise ParserContentError(str(e)) from e
        name, array = _selected(cls, arrays, variable)
        return cls.from_raw(
            MatrixAffineRaw.from_array(array, variable=name, **options)
        )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _joined(lines: tx.Iterable[str]) -> str:
    """Return the text of lines, each ending with a line break."""
    return "".join(
        line if line.endswith("\n") else line + "\n"
        for line in lines
        if isinstance(line, str)
    )


def _text_record(
    cls: type,
    text: str,
    separators: tx.Optional[str],
    options: tx.Dict[str, tx.Any],
) -> MatrixAffineRaw:
    """Parse the text of a matrix file into a record that keeps the text.

    Raises
    ------
    TypeError
        If an option is not a convention.
    ParserContentError
        If the text is not a matrix with these separators, or not the
        matrix of an affine.
    """
    _no_variable(cls, options)
    try:
        array = read_text_array(text.splitlines(), separators=separators)
    except ArrayContainerError as e:
        raise ParserContentError(str(e)) from e
    return MatrixAffineRaw.from_array(array, text=text, **options)


def _no_variable(cls: type, options: tx.Dict[str, tx.Any]) -> None:
    """Refuse `variable` and `key`, which a single matrix does not take."""
    given = sorted(name for name in ("variable", "key") if name in options)
    if given:
        raise TypeError(
            f"{cls.__name__} reads a file that holds a single matrix, so it "
            f"takes no {' or '.join(given)}."
        )


def _variable(options: tx.Dict[str, tx.Any]) -> tx.Optional[str]:
    """Pop the name of the array to read, given as `variable` or `key`.

    Raises
    ------
    TypeError
        If both are given with different values.
    """
    variable = options.pop("variable", None)
    key = options.pop("key", None)
    if variable is not None and key is not None and variable != key:
        raise TypeError(
            f"variable={variable!r} and key={key!r} name different arrays; "
            f"key is an alias of variable, so give only one."
        )
    return variable if variable is not None else key


def _selected(
    cls: type,
    arrays: tx.Mapping[str, np.ndarray],
    variable: tx.Optional[str],
) -> tx.Tuple[str, np.ndarray]:
    """Pick the matrix of a container of several arrays.

    Raises
    ------
    ParserContentError
        If no array has the given name, if no single array can be chosen,
        or if the chosen array is not a two-dimensional numeric array.
    """
    try:
        name, array = select_array(
            arrays, variable, cls._accepts_array, f"{cls.CONTAINER} array"
        )
    except ArrayContainerError as e:
        raise ParserContentError(str(e)) from e
    _check_accepted(cls, name, array)
    return name, array


def _check_accepted(
    cls: type, name: tx.Optional[str], array: np.ndarray
) -> None:
    """Refuse an array that is not a two-dimensional numeric array.

    Raises
    ------
    ParserContentError
        If the array is not accepted.
    """
    if not cls._accepts_array(array):
        what = "The array" if name is None else repr(name)
        raise ParserContentError(
            f"{what} cannot be read by {cls.__name__} "
            f"({array.dtype} {array.shape})."
        )
