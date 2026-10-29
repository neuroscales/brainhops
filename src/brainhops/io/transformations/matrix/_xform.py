"""Affine transformations stored as plain matrices."""

__all__ = [
    "MatrixAffine",
    "TxtMatrixAffine",
    "CsvMatrixAffine",
    "TsvMatrixAffine",
    "NpyMatrixAffine",
    "NpzMatrixAffine",
    "MatMatrixAffine",
    "MatLegacyMatrixAffine",
    "Mat73MatrixAffine",
]

import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Inherit, NoRepr, replace

from brainhops._core.properties import InvalidatorInAttribute, smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import apply_loss_policy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserNotImplementedError,
    UnrepresentableTransformationError,
)
from brainhops.io.common._internal_api import (
    ArrayReader,
    CsvArrayReader,
    Mat73ArrayReader,
    MatArrayReader,
    MatLegacyArrayReader,
    NpyArrayReader,
    NpzArrayReader,
    TsvArrayReader,
    TxtArrayReader,
    is_numeric_array,
)
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)
from brainhops.io.transformations.base._internal_api import read_only

from . import _conventions
from ._metadata import (
    CsvMatrixMetadata,
    Mat73MatrixMetadata,
    MatLegacyMatrixMetadata,
    MatrixMetadata,
    NpyMatrixMetadata,
    NpzMatrixMetadata,
    TsvMatrixMetadata,
    TxtMatrixMetadata,
)
from ._raw import MatrixAffineRaw

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""

_READER_OPTIONS = _conventions.CONVENTIONS + ("variable", "key")
"""The options that the readers pass to the metadata, not to the object."""


class MatrixAffine(
    ArrayReader,
    AffineTransformationFormat,
    _xforms.Affine,
    TransformationFormat,
    repr=HIDE_IF_NONE,
):
    """
    Affine transformation stored as a bare matrix in a generic array container.

    The file holds a (3, 3), (3, 4) or (4, 4) matrix, or a (2, 3) or (3, 3)
    matrix in 2-D, and nothing else, so the caller gives its conventions as
    keyword arguments to `load` or `from_file`. This class is abstract, and
    each container has a registered subclass that mixes in its
    [`ArrayReader`][]:

    - [`TxtMatrixAffine`][]: whitespace-separated text (`.txt`, `.dat`, `.1D`),
      hint `"matrix.txt"`;
    - [`CsvMatrixAffine`][]: `.csv`, hint `"matrix.csv"`;
    - [`TsvMatrixAffine`][]: `.tsv`, hint `"matrix.tsv"`;
    - [`NpyMatrixAffine`][]: `.npy`, hint `"matrix.npy"`;
    - [`NpzMatrixAffine`][]: `.npz`, hint `"matrix.npz"`;
    - [`MatLegacyMatrixAffine`][]: MATLAB v4 to v7 `.mat`, read with
      `scipy.io`, hint `"matrix.mat"`;
    - [`Mat73MatrixAffine`][]: MATLAB v7.3 `.mat`, read with `h5py`, hints
      `"matrix.mat"`, `"mat.73"` and `"matrix.mat.73"`.

    [`MatMatrixAffine`][] reads a MATLAB file of any version by dispatching to
    the last two. Every reader answers to `hint="matrix"`, in which case the
    container is recognized from the content. In a container that holds several
    arrays, the matrix is selected with `variable=` (or `key=`), and defaults
    to the only 2-D numeric array. A container that holds a single array
    refuses `variable=` and `key=` with `TypeError`, and so does every
    reader for an option that it does not know.

    ## Conventions

    The conventions below are applied in order: transposition, inversion, index
    shift and image placement.

    - `vector="column"`: with `"row"`, the matrix is applied as `y = x @ A` and
      is transposed.
    - `ndim=None`: only used for a (3, 3) matrix, which is 3-D linear unless
      `ndim=2` makes it a 2-D homogeneous affine.
    - `direction="forward"`: with `"inverse"`, the file stores the map from
      output to input, and the matrix is inverted.
    - `input=None`, `output=None`: the spaces of the forward matrix, as
      `"voxel"` (or `"pixel"`, `"index"`), `"ras"`, `"lps"` or a coordinate
      system. `None` stands for a generic unnamed space.
    - `index_base=0`: 1 when voxel indices are 1-based, as in MATLAB and SPM,
      or an `(input, output)` pair. A 1-based voxel endpoint is shifted to
      0-based indices, in which an integer is the centre of a voxel. A single
      value applies to every voxel endpoint and requires at least one.
    - `source=None`, `target=None`: a `nibabel` image or header, or a brainhops
      image. The voxel input or output is mapped to the world space of that
      image through its voxel-to-world affine, and passing an image makes that
      endpoint default to `"voxel"`.

    ## Record

    A plain matrix file is read in one pass. Its matrix, and the
    conventions that it was read in, are held as a [`MatrixAffineRaw`][]
    record by the metadata of the transformation, such as a
    [`TxtMatrixMetadata`][], and the matrix and the coordinate systems of
    the transformation is decoded from that record. The record of a text
    file also keeps the text of the file. The coordinate systems of the
    record are given to the transformation when it is read, as its own
    `input` and `output`, so that a copy into another class keeps them. A
    matrix that is assigned, and an `input` or `output` that is not the
    system of the record, are pending: they win over the record until
    `consolidate()` encodes them into a new record, in the conventions
    and the layout of the record that it replaces. A transformation
    without a record, such as one built from data alone, holds its matrix
    as any affine does and has no metadata.

    These formats read files but do not write them, so a plain matrix is
    saved through another format, such as NIfTI or LTA.

    ## Dispatch

    Any small numeric table reads as a matrix, so this reader scores
    `Confidence.WEAK` and loses to every positively recognized format, such as
    FLIRT or ITK. A text matrix whose file extension matches its separator
    scores higher, so that it is not read as a FLIRT matrix. Files larger than
    1 MiB are not sniffed. `hint="matrix"`, or a container hint, selects this
    reader explicitly.
    """

    HINTS = ("matrix",)
    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    # Plain matrices are tiny, so a large file is not claimed, and a huge
    # image is never read whole only to be rejected.
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 1 << 20

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(N, N + 1)` affine matrix, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until the transformation is consolidated. Otherwise the matrix is
        decoded from the record in its conventions, without its last row,
        and is read-only, or `None` without a record.
        """
        record = _record(self)
        if record is None:
            return None
        return read_only(record.affine()[:-1])

    # --- ArrayReader hooks ----------------------------------------------

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

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: MatrixAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        Each container reads its own records, so the abstract classes
        refuse.

        Parameters
        ----------
        raw : MatrixAffineRaw
            The record of a file.
        **kwargs : Any
            Field values.

        Returns
        -------
        Self
            The transformation.

        Raises
        ------
        ParserNotImplementedError
            In an abstract class.
        """
        raise ParserNotImplementedError(
            f"{cls.__name__} is abstract and holds no record. Use the class "
            f"of a container, such as TxtMatrixAffine."
        )


@register_format
class TxtMatrixAffine(TxtArrayReader, MatrixAffine):
    """
    Affine stored in a whitespace-separated text file.

    This covers `.txt`, `.dat` and AFNI `.1D` files, with one matrix row per
    line. Comments start with `#`, and blank lines are skipped. The record
    keeps the text of the file. See [`MatrixAffine`][].
    """

    HINTS = TxtArrayReader.HINTS

    _metadata: NoRepr[tx.Optional[TxtMatrixMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata. Assigning
    other metadata drops the matrix decoded from the old record, and a
    matrix that was assigned stays pending. The systems of the
    transformation are kept, and they are pending unless they are those
    of the new record.
    """

    @classmethod
    def from_raw(cls, raw: MatrixAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : MatrixAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, which go to the constructor and override the
            record.

        Returns
        -------
        Self
            The transformation.
        """
        return _with_record(cls, TxtMatrixMetadata.from_raw(raw), kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build a transformation from the text of a file.

        The text is read by [`TxtMatrixMetadata.from_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            The conventions of the matrix, and field values, which go to
            the constructor and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = TxtMatrixMetadata.from_text(text, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from the lines of a file.

        The lines are read by [`TxtMatrixMetadata.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            The conventions of the matrix, and field values, which go to
            the constructor and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = TxtMatrixMetadata.from_lines(lines, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`MatrixAffineRaw`][] is read with `from_raw`, and any other
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
        if isinstance(other, MatrixAffineRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise a new record is built, which stores
        `matrix` in the conventions and the layout of the current record,
        or as a homogeneous matrix of column vectors with 0-based indices
        without one, and which holds `input` and `output`. The new record
        is held by new metadata, and the matrix and the systems are then
        decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a matrix
        file holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

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
            If the conventions of the record cannot store the matrix or
            the systems.
        """
        return _consolidated(self, TxtMatrixMetadata, inplace, fields)


@register_format
class CsvMatrixAffine(CsvArrayReader, MatrixAffine):
    """
    Affine stored in a comma-separated `.csv` file.

    The record keeps the text of the file. See [`MatrixAffine`][].
    """

    HINTS = CsvArrayReader.HINTS

    _metadata: NoRepr[tx.Optional[CsvMatrixMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata. Assigning
    other metadata drops the matrix decoded from the old record, and a
    matrix that was assigned stays pending. The systems of the
    transformation are kept, and they are pending unless they are those
    of the new record.
    """

    @classmethod
    def from_raw(cls, raw: MatrixAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : MatrixAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, which go to the constructor and override the
            record.

        Returns
        -------
        Self
            The transformation.
        """
        return _with_record(cls, CsvMatrixMetadata.from_raw(raw), kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build a transformation from the text of a file.

        The text is read by [`CsvMatrixMetadata.from_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            The conventions of the matrix, and field values, which go to
            the constructor and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = CsvMatrixMetadata.from_text(text, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from the lines of a file.

        The lines are read by [`CsvMatrixMetadata.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            The conventions of the matrix, and field values, which go to
            the constructor and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = CsvMatrixMetadata.from_lines(lines, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`MatrixAffineRaw`][] is read with `from_raw`, and any other
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
        if isinstance(other, MatrixAffineRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise a new record is built, which stores
        `matrix` in the conventions and the layout of the current record,
        or as a homogeneous matrix of column vectors with 0-based indices
        without one, and which holds `input` and `output`. The new record
        is held by new metadata, and the matrix and the systems are then
        decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a matrix
        file holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

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
            If the conventions of the record cannot store the matrix or
            the systems.
        """
        return _consolidated(self, CsvMatrixMetadata, inplace, fields)


@register_format
class TsvMatrixAffine(TsvArrayReader, MatrixAffine):
    """
    Affine stored in a tab-separated `.tsv` file.

    The record keeps the text of the file. See [`MatrixAffine`][].
    """

    HINTS = TsvArrayReader.HINTS

    _metadata: NoRepr[tx.Optional[TsvMatrixMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata. Assigning
    other metadata drops the matrix decoded from the old record, and a
    matrix that was assigned stays pending. The systems of the
    transformation are kept, and they are pending unless they are those
    of the new record.
    """

    @classmethod
    def from_raw(cls, raw: MatrixAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : MatrixAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, which go to the constructor and override the
            record.

        Returns
        -------
        Self
            The transformation.
        """
        return _with_record(cls, TsvMatrixMetadata.from_raw(raw), kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build a transformation from the text of a file.

        The text is read by [`TsvMatrixMetadata.from_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            The conventions of the matrix, and field values, which go to
            the constructor and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = TsvMatrixMetadata.from_text(text, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from the lines of a file.

        The lines are read by [`TsvMatrixMetadata.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            The conventions of the matrix, and field values, which go to
            the constructor and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = TsvMatrixMetadata.from_lines(lines, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`MatrixAffineRaw`][] is read with `from_raw`, and any other
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
        if isinstance(other, MatrixAffineRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise a new record is built, which stores
        `matrix` in the conventions and the layout of the current record,
        or as a homogeneous matrix of column vectors with 0-based indices
        without one, and which holds `input` and `output`. The new record
        is held by new metadata, and the matrix and the systems are then
        decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a matrix
        file holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

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
            If the conventions of the record cannot store the matrix or
            the systems.
        """
        return _consolidated(self, TsvMatrixMetadata, inplace, fields)


@register_format
class NpyMatrixAffine(NpyArrayReader, MatrixAffine):
    """
    Affine stored in a NumPy `.npy` file, read without unpickling.

    The file holds a single array, so the reader refuses `variable=` and
    `key=`. See [`MatrixAffine`][].
    """

    HINTS = NpyArrayReader.HINTS

    _metadata: NoRepr[tx.Optional[NpyMatrixMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata. Assigning
    other metadata drops the matrix decoded from the old record, and a
    matrix that was assigned stays pending. The systems of the
    transformation are kept, and they are pending unless they are those
    of the new record.
    """

    @classmethod
    def from_raw(cls, raw: MatrixAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : MatrixAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, which go to the constructor and override the
            record.

        Returns
        -------
        Self
            The transformation.
        """
        return _with_record(cls, NpyMatrixMetadata.from_raw(raw), kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Build a transformation from a binary file object.

        The file is read by [`NpyMatrixMetadata.from_fileobj`][].

        Parameters
        ----------
        file : file object
            The file, open for binary reading.
        **kwargs : Any
            The conventions of the matrix, the options of the container,
            and field values, which go to the constructor and override
            the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = NpyMatrixMetadata.from_fileobj(file, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build a transformation from the content of a file.

        The content is read by [`NpyMatrixMetadata.from_bytes`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            The conventions of the matrix, the options of the container,
            and field values, which go to the constructor and override
            the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = NpyMatrixMetadata.from_bytes(content, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`MatrixAffineRaw`][] is read with `from_raw`, and any other
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
        if isinstance(other, MatrixAffineRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise a new record is built, which stores
        `matrix` in the conventions and the layout of the current record,
        or as a homogeneous matrix of column vectors with 0-based indices
        without one, and which holds `input` and `output`. The new record
        is held by new metadata, and the matrix and the systems are then
        decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a matrix
        file holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

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
            If the conventions of the record cannot store the matrix or
            the systems.
        """
        return _consolidated(self, NpyMatrixMetadata, inplace, fields)


@register_format
class NpzMatrixAffine(NpzArrayReader, MatrixAffine):
    """
    Affine stored in a NumPy `.npz` file, read without unpickling.

    The array is selected with `variable=` or `key=`, and defaults to the only
    2-D numeric array. The record keeps its name. See [`MatrixAffine`][].
    """

    HINTS = NpzArrayReader.HINTS

    _metadata: NoRepr[tx.Optional[NpzMatrixMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata. Assigning
    other metadata drops the matrix decoded from the old record, and a
    matrix that was assigned stays pending. The systems of the
    transformation are kept, and they are pending unless they are those
    of the new record.
    """

    @classmethod
    def from_raw(cls, raw: MatrixAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : MatrixAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, which go to the constructor and override the
            record.

        Returns
        -------
        Self
            The transformation.
        """
        return _with_record(cls, NpzMatrixMetadata.from_raw(raw), kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Build a transformation from a binary file object.

        The file is read by [`NpzMatrixMetadata.from_fileobj`][].

        Parameters
        ----------
        file : file object
            The file, open for binary reading.
        **kwargs : Any
            The conventions of the matrix, the options of the container,
            and field values, which go to the constructor and override
            the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = NpzMatrixMetadata.from_fileobj(file, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build a transformation from the content of a file.

        The content is read by [`NpzMatrixMetadata.from_bytes`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            The conventions of the matrix, the options of the container,
            and field values, which go to the constructor and override
            the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = NpzMatrixMetadata.from_bytes(content, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`MatrixAffineRaw`][] is read with `from_raw`, and any other
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
        if isinstance(other, MatrixAffineRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise a new record is built, which stores
        `matrix` in the conventions and the layout of the current record,
        or as a homogeneous matrix of column vectors with 0-based indices
        without one, and which holds `input` and `output`. The new record
        is held by new metadata, and the matrix and the systems are then
        decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a matrix
        file holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

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
            If the conventions of the record cannot store the matrix or
            the systems.
        """
        return _consolidated(self, NpzMatrixMetadata, inplace, fields)


class MatMatrixAffine(MatArrayReader, MatrixAffine):
    """
    Affine stored in a MATLAB `.mat` file of any version.

    The variable is selected with `variable=`, and defaults to the only 2-D
    numeric array. The file is dispatched by content to
    [`MatLegacyMatrixAffine`][] or [`Mat73MatrixAffine`][], which returns an
    object of that class. Only these variants are registered, so that this
    class does not compete with them, and only they hold metadata. See
    [`MatrixAffine`][].
    """

    HINTS = MatArrayReader.HINTS


@register_format
class MatLegacyMatrixAffine(MatLegacyArrayReader, MatMatrixAffine):
    """
    Affine stored in a MATLAB v4 or v5 to v7 `.mat` file, read with `scipy.io`.

    The record keeps the name of the variable. See [`MatMatrixAffine`][].
    """

    _metadata: NoRepr[tx.Optional[MatLegacyMatrixMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata. Assigning
    other metadata drops the matrix decoded from the old record, and a
    matrix that was assigned stays pending. The systems of the
    transformation are kept, and they are pending unless they are those
    of the new record.
    """

    @classmethod
    def from_raw(cls, raw: MatrixAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : MatrixAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, which go to the constructor and override the
            record.

        Returns
        -------
        Self
            The transformation.
        """
        return _with_record(cls, MatLegacyMatrixMetadata.from_raw(raw), kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Build a transformation from a binary file object.

        The file is read by [`MatLegacyMatrixMetadata.from_fileobj`][].

        Parameters
        ----------
        file : file object
            The file, open for binary reading.
        **kwargs : Any
            The conventions of the matrix, the options of the container,
            and field values, which go to the constructor and override
            the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = MatLegacyMatrixMetadata.from_fileobj(file, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build a transformation from the content of a file.

        The content is read by [`MatLegacyMatrixMetadata.from_bytes`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            The conventions of the matrix, the options of the container,
            and field values, which go to the constructor and override
            the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = MatLegacyMatrixMetadata.from_bytes(content, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`MatrixAffineRaw`][] is read with `from_raw`, and any other
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
        if isinstance(other, MatrixAffineRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise a new record is built, which stores
        `matrix` in the conventions and the layout of the current record,
        or as a homogeneous matrix of column vectors with 0-based indices
        without one, and which holds `input` and `output`. The new record
        is held by new metadata, and the matrix and the systems are then
        decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a matrix
        file holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

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
            If the conventions of the record cannot store the matrix or
            the systems.
        """
        return _consolidated(self, MatLegacyMatrixMetadata, inplace, fields)


@register_format
class Mat73MatrixAffine(Mat73ArrayReader, MatMatrixAffine):
    """
    Affine stored in a MATLAB v7.3 (HDF5) `.mat` file, read with `h5py`.

    MATLAB v7.3 stores arrays transposed, which the reader undoes. The
    record keeps the name of the variable. See [`MatMatrixAffine`][].
    """

    HINTS = Mat73ArrayReader.HINTS

    _metadata: NoRepr[tx.Optional[Mat73MatrixMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata. Assigning
    other metadata drops the matrix decoded from the old record, and a
    matrix that was assigned stays pending. The systems of the
    transformation are kept, and they are pending unless they are those
    of the new record.
    """

    @classmethod
    def from_raw(cls, raw: MatrixAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : MatrixAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, which go to the constructor and override the
            record.

        Returns
        -------
        Self
            The transformation.
        """
        return _with_record(cls, Mat73MatrixMetadata.from_raw(raw), kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Build a transformation from a binary file object.

        The file is read by [`Mat73MatrixMetadata.from_fileobj`][].

        Parameters
        ----------
        file : file object
            The file, open for binary reading.
        **kwargs : Any
            The conventions of the matrix, the options of the container,
            and field values, which go to the constructor and override
            the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = Mat73MatrixMetadata.from_fileobj(file, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build a transformation from the content of a file.

        The content is read by [`Mat73MatrixMetadata.from_bytes`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            The conventions of the matrix, the options of the container,
            and field values, which go to the constructor and override
            the file.

        Returns
        -------
        Self
            The transformation.
        """
        options = _reader_options(kwargs)
        metadata = Mat73MatrixMetadata.from_bytes(content, **options)
        return _with_record(cls, metadata, kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`MatrixAffineRaw`][] is read with `from_raw`, and any other
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
        if isinstance(other, MatrixAffineRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise a new record is built, which stores
        `matrix` in the conventions and the layout of the current record,
        or as a homogeneous matrix of column vectors with 0-based indices
        without one, and which holds `input` and `output`. The new record
        is held by new metadata, and the matrix and the systems are then
        decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a matrix
        file holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

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
            If the conventions of the record cannot store the matrix or
            the systems.
        """
        return _consolidated(self, Mat73MatrixMetadata, inplace, fields)


MatMatrixAffine.VARIANTS = (MatLegacyMatrixAffine, Mat73MatrixAffine)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _record(xform: MatrixAffine) -> tx.Optional[MatrixAffineRaw]:
    """Return the record of the metadata of a transformation, or `None`.

    The abstract classes hold no metadata, so they have no record.
    """
    metadata = getattr(xform, "metadata", None)
    return None if metadata is None else metadata.raw


def _with_record(
    cls: type, metadata: MatrixMetadata, kwargs: tx.Dict[str, tx.Any]
) -> MatrixAffine:
    """Build a transformation from metadata and the systems of its record.

    The coordinate systems of the record are given to the transformation
    as its own, unless the keyword arguments set them, so that a copy into
    another class keeps them. They count as assigned only when they are
    not the very objects that the record holds.
    """
    kwargs.setdefault("input", metadata.raw.input)
    kwargs.setdefault("output", metadata.raw.output)
    return cls(metadata=metadata, **kwargs)


def _reader_options(kwargs: tx.Dict[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Pop the options that the metadata reads from the keyword arguments.

    They are the conventions of the matrix and the name of the array. The
    keyword arguments left are field values, and an unknown one raises
    `TypeError` in the constructor.
    """
    return {
        name: kwargs.pop(name) for name in _READER_OPTIONS if name in kwargs
    }


def _consolidated(
    xform: MatrixAffine,
    metadata_class: tx.Type[MatrixMetadata],
    inplace: bool,
    fields: tx.Mapping[str, tx.Any],
) -> MatrixAffine:
    """Encode the pending matrix and systems of a transformation.

    A transformation that holds a record, and whose matrix, `input` and
    `output` were not set, is left as it is, since its record also holds
    its matrix. Otherwise a new record is encoded by [`_encoded`][], and it
    is held by a copy of the metadata, or by new metadata of
    `metadata_class` when the transformation has none. The matrix and the
    systems are then decoded from the new record, so nothing is pending.
    The transformation itself is changed only with `inplace=True`.

    Raises
    ------
    UnrepresentableTransformationError
        If the conventions of the record cannot store the matrix or the
        systems.
    """
    report = ConversionReport(
        source=type(xform).__name__, target=type(xform).__name__
    )
    requested = requested_fields(xform.metadata, fields)
    metadata = metadata_to_write(
        xform.metadata, metadata_class, fields, report
    )
    apply_loss_policy(report, stacklevel=3)
    if not requested:
        metadata = xform.metadata
    record = _record(xform)
    if record is not None and not _overridden(xform, record):
        if not inplace:
            return replace(xform, metadata=metadata)
        if requested:
            xform.metadata = metadata
        return xform
    record = _encoded(xform, record)
    if metadata is None:
        metadata = metadata_class.from_raw(record)
    else:
        metadata = replace(metadata, raw=record)
    systems = {"input": record.input, "output": record.output}
    if not inplace:
        return replace(xform, metadata=metadata, data=None, **systems)
    xform.metadata = metadata
    xform.data = None
    xform.input = record.input
    xform.output = record.output
    return xform


def _overridden(xform: MatrixAffine, record: MatrixAffineRaw) -> bool:
    """Tell whether a transformation holds a matrix or systems of its own.

    A matrix is assigned when the private data is set. A system is
    assigned when it is not the very object that the record holds, which
    the readers give to the transformation.
    """
    return (
        xform._data is not None
        or xform._input is not record.input
        or xform._output is not record.output
    )


def _encoded(
    xform: MatrixAffine, old: tx.Optional[MatrixAffineRaw]
) -> MatrixAffineRaw:
    """Encode the matrix and the systems of a transformation as a record.

    The matrix is stored in the conventions and in the layout of the old
    record, so that a matrix read as 1-based, inverted or transposed is
    stored in the same way. Without an old record, the matrix is stored
    as a homogeneous matrix of column vectors, with 0-based indices and no
    image. The systems of the transformation become those of the record,
    and the record keeps no text, since it no longer describes a file
    that was read.

    Raises
    ------
    UnrepresentableTransformationError
        If the matrix does not have the dimensions of the old record, if
        the old record stores a linear map and the matrix has a
        translation, or if a 1-based endpoint is no longer a voxel space.
    """
    data = xform.data
    ndim = 3 if data is None else int(np.shape(data)[0])
    homogeneous = np.eye(ndim + 1)
    if data is not None:
        homogeneous[:ndim] = np.asarray(data, dtype=np.float64)
    if old is None:
        # The identity only gives the layout of the record that is encoded.
        identity = tuple(tuple(row) for row in np.eye(ndim + 1).tolist())
        old = MatrixAffineRaw(matrix=identity, ndim=ndim)
    systems = {"input": xform.input, "output": xform.output}
    placed = {"input": old.source_to_world, "output": old.target_to_world}
    for end, base in zip(("input", "output"), old.index_base):
        system = systems[end]
        if base and placed[end] is None and system is not None:
            if not _conventions.is_index_system(system):
                raise UnrepresentableTransformationError(
                    f"The matrix was read with 1-based indices for its "
                    f"{end}, which is no longer a voxel space but "
                    f"{type(system).__name__}."
                )
    try:
        matrix = _conventions.encode(homogeneous, old)
    except ValueError as e:
        raise UnrepresentableTransformationError(str(e)) from e
    return replace(old, matrix=matrix, text=None, **systems)
