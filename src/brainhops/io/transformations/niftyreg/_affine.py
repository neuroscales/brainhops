"""NiftyReg affine matrices, as written by `reg_aladin`."""

import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

from brainhops._core import path
from brainhops._core._internal_api import InvalidatorInAttribute, smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import convert_withlog, converter
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
    Confidence,
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.common._internal_api import TxtArrayReader, is_numeric_array
from brainhops.io.metadata._internal_api import (
    metadata_options,
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import TransformationFormat
from brainhops.io.transformations.base._conversions import (
    affine_between,
    format_options,
)
from brainhops.io.transformations.base._records import (
    matrix_to_disk,
    matrix_to_model,
    read_only,
    record_of,
)
from brainhops.io.transformations.base.affines import (
    _RAS,
    RASToRAS,
    _check_endpoint,
)

from ._formats import NiftyRegAffineFormat
from ._metadata import NiftyRegAffineMetadata
from ._raw import NiftyRegAffineRaw, is_homogeneous

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


@register_format
class NiftyRegAffine(
    TxtArrayReader,
    TextFileReader,
    TextFileWriter,
    NiftyRegAffineFormat,
    RASToRAS,
    TransformationFormat,
):
    """
    Affine written by `reg_aladin -aff`.

    The file is plain text holding the four rows of a (4, 4) homogeneous
    matrix, with four whitespace-separated numbers per line
    (`reg_tool_WriteAffineFile`). NiftyReg documents the matrix as `Affine *
    Reference = Floating`: it maps the reference world to the floating world,
    which is the resampling direction of the data model, so it is read without
    inversion. The world is NIfTI RAS in millimetres, with no sign flip, and no
    image is needed to read the file.

    Nothing marks the file as NiftyReg's, so this reader scores
    `Confidence.WEAK`, below the generic matrix reader
    ([`TxtMatrixAffine`][brainhops.io.transformations.matrix.TxtMatrixAffine])
    and FLIRT. It is selected with `hint="niftyreg"`, `"niftyreg.aladin"` or
    `"aladin"`.

    ## Record

    The file is read in one pass, and its content is held as a
    [`NiftyRegAffineRaw`][] record by the [`NiftyRegAffineMetadata`][] of
    the transformation. The matrix lives in that record. A matrix that is
    assigned is pending: it wins over the record until [`consolidate`][]
    encodes it into a new record. A transformation that was read and not
    changed is written as its file stored it, byte for byte. A matrix
    that is encoded is written in the same layout as NiftyReg's, with
    enough digits for each number to be read back exactly, whereas
    NiftyReg itself prints `%.7g`.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)
    HINTS = ("aladin",)
    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    NAMED_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    # The affine file is tiny, so a large file is not read whole only to be
    # rejected.
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 1 << 16

    _metadata: NoRepr[tx.Optional[NiftyRegAffineMetadata]] = Inherit(
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
        record = record_of(self)
        if record is None:
            return None
        return read_only(matrix_to_model(record.matrix))

    # --- ArrayReader hooks --------------------------------------------

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        return is_numeric_array(array) and array.ndim == 2

    @classmethod
    def _array_confidence(cls, array: np.ndarray, **kwargs) -> float:
        """Score `Confidence.WEAK` for a homogeneous (4, 4) matrix."""
        return cls.SNIFF_CONFIDENCE if is_homogeneous(array) else 0.0

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: NiftyRegAffineRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed record.

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : NiftyRegAffineRaw
            The record of the file.
        **kwargs : Any
            Field values, such as `matrix`, which go to the constructor
            and override the record.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=NiftyRegAffineMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build a transformation from the text of a NiftyReg affine file.

        The text is read by [`NiftyRegAffineMetadata.from_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            Field values, such as `matrix`, which go to the constructor
            and override the file. Any other option raises `TypeError`.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=NiftyRegAffineMetadata.from_text(text), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from the lines of a NiftyReg affine file.

        The lines are read by [`NiftyRegAffineMetadata.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Field values, such as `matrix`, which go to the constructor
            and override the file. Any other option raises `TypeError`.

        Returns
        -------
        Self
            The transformation.
        """
        metadata = NiftyRegAffineMetadata.from_lines(lines)
        return cls(metadata=metadata, **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`NiftyRegAffineRaw`][] is read with [`from_raw`][], and any
        other value as the bases read it.

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
        if isinstance(other, NiftyRegAffineRaw) and not args:
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
        matrix. Otherwise, a new record is built from `matrix` and is held
        by new metadata. In every case, the input and the output must
        still be RAS, since the file can hold nothing else.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a
        NiftyReg affine holds no field of the vocabulary yet, so a field
        that is set is reported as lost, once, under the error policy of
        the call.

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
        report = ConversionReport(
            source="NiftyRegAffine", target="NiftyRegAffine"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, NiftyRegAffineMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        _check_systems(self)
        record = record_of(self)
        if record is not None and self._data is None:
            if not inplace:
                return replace(self, metadata=metadata)
            if requested:
                self.metadata = metadata
            return self
        record = _build_raw(self)
        if metadata is None:
            metadata = NiftyRegAffineMetadata.from_raw(record)
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


@converter(_xforms.Identity, NiftyRegAffine)
@converter(_xforms.Translation, NiftyRegAffine)
@converter(_xforms.Scaling, NiftyRegAffine)
@converter(_xforms.Permutation, NiftyRegAffine)
@converter(_xforms.Linear, NiftyRegAffine)
@converter(_xforms.AffineExponential, NiftyRegAffine)
@converter(_xforms.SubspaceTransformation, NiftyRegAffine)
@converter(_xforms.Sequence, NiftyRegAffine)
@converter
def _(
    t: _xforms.Affine, cls: tx.Type[NiftyRegAffine], **kwargs
) -> NiftyRegAffine:
    # `affine_between` either converts `t` exactly or raises. It bridges
    # the endpoints of `t` to RAS, which flips an affine between LPS
    # spaces, and it reduces a chain to one affine. An affine whose
    # systems are not known is taken to map RAS to RAS, because the
    # endpoints of a NiftyReg affine accept unknown systems.
    cls = NiftyRegAffine
    format_options(t, cls, kwargs)
    matrix = affine_between(t, _RAS, _RAS, cls)
    return cls(matrix=matrix[:-1])


@converter
def _(
    t: NiftyRegAffine, cls: tx.Type[NiftyRegAffine], **kwargs
) -> NiftyRegAffine:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type.
    return convert_withlog(t, cls, "matrix", **kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _check_systems(xform: NiftyRegAffine) -> None:
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
                f"A NiftyReg affine maps RAS to RAS, but the {name} of the "
                f"transformation is not RAS: {e}"
            ) from e


def _build_raw(xform: NiftyRegAffine) -> NiftyRegAffineRaw:
    """Encode the matrix of `xform` as a record.

    Raises
    ------
    UnrepresentableTransformationError
        If the matrix is not of shape (3, 4).
    """
    data = xform.data
    shape = None if data is None else tuple(np.shape(data))
    if shape is not None and shape != (3, 4):
        raise UnrepresentableTransformationError(
            f"A NiftyReg affine maps 3-D to 3-D, and this one has a "
            f"matrix of shape {shape}."
        )
    return NiftyRegAffineRaw(matrix=matrix_to_disk(data))


def _written(
    xform: NiftyRegAffine, kwargs: tx.Mapping[str, tx.Any]
) -> tx.Tuple[NiftyRegAffineRaw, tx.Dict[str, tx.Any]]:
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
    return record_of(xform.consolidate(**metadata)), options
