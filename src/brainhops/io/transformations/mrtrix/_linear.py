"""
MRtrix linear transforms: the text files of `mrtransform -linear` and
`mrregister -rigid` or `-affine`.

What they hold, and the MRtrix3 sources it was checked against, is
described in the package docstring,
[`brainhops.io.transformations.mrtrix`][].
"""

# stdlib
from collections import OrderedDict

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core._internal_api import InvalidatorInAttribute, smartproperty
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import (
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
from brainhops.io.metadata._base import (
    metadata_options,
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import AffineTransformationFormat
from brainhops.io.transformations.base.affines import (
    _RAS,
    RASToRAS,
    _check_endpoint,
)

# this format
from ._formats import MrtrixTransformationFormat
from ._metadata import MrtrixLinearMetadata
from ._raw import MrtrixLinearRaw

_NDIM = 3
"""The number of spatial dimensions of an MRtrix linear transform."""

_HISTORY = "command_history"
"""The key in which MRtrix records the commands that wrote a file."""

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


@register_format
class MrtrixLinearTransform(
    TextFileReader,
    TextFileWriter,
    MrtrixTransformationFormat,
    AffineTransformationFormat,
    RASToRAS,
):
    """
    An affine stored in an MRtrix linear transform file, a `(3, 4)` or
    `(4, 4)` text matrix.

    The matrix maps scanner RAS points of the *template* image to scanner
    RAS points of the *moving* image, in millimetres: the "reverse"
    convention that `mrtransform -linear` and `mrregister` use. It is read
    as it is, as an affine from `RASmm` to `RASmm`, since that is also the
    direction of the data model, from template points to the points to
    sample.

    A linear transform file is short and is read in one pass, so the whole
    file is held as an [`MrtrixLinearRaw`][] record by the
    [`MrtrixLinearMetadata`][] of the transformation, including its
    `# key: value` comments, such as `command_history` and `centre`. The
    matrix lives in that record, and the data of the transformation is a
    read-only view of it. A matrix that is assigned is pending until
    [`consolidate`][] encodes it into a new record, which keeps the
    comments of the old one. A transformation without a record, such as
    one built from data alone, holds its matrix as any affine does and has
    no metadata. A transformation that is read and written again without
    changes keeps the bytes of its file.

    Plain text matrices look all alike, so a file is claimed only when its
    comments say that MRtrix wrote it, with a `command_history` or a
    `centre` entry. Otherwise, the file is read with
    `hint="mrtrix.linear"`.
    """

    HINTS = ("linear", "affine")
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)

    PRIORITY: tx.ClassVar[int] = -1
    """
    Priority below the NiftyReg affine.

    A `.txt` name cannot tell an MRtrix transform from a NiftyReg affine,
    and both hold an affine between RAS spaces exactly. `io.save` writes
    such an affine as a NiftyReg affine (#343), unless it already is an
    MRtrix transform, which is built with `t.to(MrtrixLinearTransform)`.
    A file is read as an MRtrix transform from its comments, whose score
    decides before the priority.
    """

    _metadata: NoRepr[tx.Optional[MrtrixLinearMetadata]] = Inherit(
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
        matrix is a read-only copy of the matrix of the record, or `None`
        without a record.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        return _read_only(self.metadata.raw.matrix)

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a text stream holds a transform.

        The stream is scored by [`MrtrixLinearRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The stream, open in text mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no transform
            written by MRtrix.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MrtrixLinearRaw.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that lines hold a transform of MRtrix.

        The lines are scored by [`MrtrixLinearRaw.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines hold no transform
            written by MRtrix.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MrtrixLinearRaw.sniff_lines(lines, error=error)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "MrtrixLinearTransform":
        """Read a transformation from the text of a linear transform file.

        The text is read by [`MrtrixLinearMetadata.from_text`][], and only
        the first three rows of the matrix are read, as MRtrix reads them.

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            Refused, since the reader has no option.

        Returns
        -------
        MrtrixLinearTransform
            The transformation.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the text does not hold a `(3, 4)` or `(4, 4)` matrix.
        """
        return cls(metadata=MrtrixLinearMetadata.from_text(text, **kwargs))

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **kwargs
    ) -> "MrtrixLinearTransform":
        """Read a transformation from the lines of a linear transform file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without their line breaks.
        **kwargs : Any
            Refused, since the reader has no option.

        Returns
        -------
        MrtrixLinearTransform
            The transformation.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the lines do not hold a `(3, 4)` or `(4, 4)` matrix.
        """
        return cls(metadata=MrtrixLinearMetadata.from_lines(lines, **kwargs))

    # --- writing ------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> "MrtrixLinearTransform":
        """Encode the pending matrix into a new record.

        A transformation that holds a record, and whose matrix was not
        assigned, is left as it is, since its record also holds its
        matrix: the copy shares the metadata. Otherwise a new record holds
        the matrix, and the comments of the current record, and new
        metadata holds the record. A transformation without a record gets
        a record whose `command_history` names brainhops and its version,
        as every MRtrix command records itself, so that the file it writes
        is recognized as an MRtrix transform when it is read again. The
        record held before is never changed. An `input` or `output` that
        was assigned must be compatible with scanner RAS, which is what
        the file stores.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an
        MRtrix linear transform holds no field of the vocabulary yet, so
        a field that is set is reported as lost, once, under the error
        policy of the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        MrtrixLinearTransform
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If the matrix is not that of a 3-D affine, or if the input or
            the output is not compatible with scanner RAS.
        """
        report = ConversionReport(
            source="MrtrixLinearTransform", target="MrtrixLinearTransform"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, MrtrixLinearMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        _check_ras(self)
        record = None if self.metadata is None else self.metadata.raw
        if record is not None and self._data is None:
            if not inplace:
                return replace(self, metadata=metadata)
            if requested:
                self.metadata = metadata
            return self
        matrix = np.asarray(self.homogeneous_matrix, dtype=np.float64)
        if matrix.shape != (_NDIM + 1, _NDIM + 1):
            raise UnrepresentableTransformationError(
                f"An MRtrix linear transform is 3-D, and this affine has a "
                f"matrix of shape {matrix.shape[0] - 1}x{matrix.shape[1] - 1}."
            )
        keyval = _new_keyval() if record is None else record.keyval
        record = MrtrixLinearRaw(matrix=matrix, keyval=keyval)
        if metadata is None:
            metadata = MrtrixLinearMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(self, metadata=metadata, data=None)
        self.metadata = metadata
        self.data = None
        return self

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the content of the file that encodes the transformation.

        The transformation is consolidated into a copy, whose record is
        written: the text that it was read from, when the matrix was not
        changed, and otherwise the lines that MRtrix writes
        (`save_transform`), which are the comments, three rows of four
        values and `0 0 0 1`. The transformation itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            Refused, since the writer has no option. The fields of the
            metadata are given to [`consolidate`][].

        Returns
        -------
        str
            The content of the file, ending with a newline.

        Raises
        ------
        TypeError
            If an option is given.
        UnrepresentableTransformationError
            If the affine does not map scanner RAS to scanner RAS, or is
            not 3-D.
        """
        fields, kwargs = metadata_options(kwargs)
        return self.consolidate(**fields).metadata.to_text(**kwargs)

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.List[str]:
        """Return the lines of the file that encodes the transformation.

        Parameters
        ----------
        **kwargs : Any
            Refused, since the writer has no option.

        Returns
        -------
        list of str
            The lines of the file, without their line breaks.
        """
        return self.to_text(**kwargs).splitlines()


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _read_only(matrix: np.ndarray) -> np.ndarray:
    """Return a read-only copy of a matrix decoded from a record.

    The writer reads the record and not the decoded matrix, so an edit in
    place would be lost. To change the matrix, a new matrix is assigned as
    data instead, which stays pending until the transformation is
    consolidated.
    """
    matrix = np.array(matrix, dtype=np.float64)
    matrix.flags.writeable = False
    return matrix


def _new_keyval() -> "OrderedDict[str, str]":
    """Return the comments of a new record, which name the writer.

    MRtrix appends the command line and its version to the
    `command_history` of every file that it writes (`core/app.cpp`).
    """
    from brainhops import __version__

    return OrderedDict([(_HISTORY, f"brainhops  (version={__version__})")])


def _check_ras(xform: MrtrixLinearTransform) -> None:
    """Refuse endpoints that are not compatible with scanner RAS.

    The endpoints are checked when the transformation is built, and again
    here, since they may have been assigned since.

    Raises
    ------
    UnrepresentableTransformationError
        If the input or the output is not compatible with `RASmm`.
    """
    for side in ("input", "output"):
        try:
            _check_endpoint(xform, side, _RAS)
        except IncompatibleSystemError as error:
            raise UnrepresentableTransformationError(
                f"An MRtrix linear transform maps scanner RAS to scanner "
                f"RAS. {error}"
            ) from error
