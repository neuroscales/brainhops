"""The chain of transformations stored in an ITK binary MATLAB file."""

__all__ = ["MatTransform"]

import sys

import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, KwOnly, NoRepr, replace

from brainhops._core._internal_api import smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common._internal_api import held_transformations
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.itk._internal_api import (
    block_of,
    check_position,
    check_systems,
    decode_chain,
    read_only,
    same_chain,
)

from .._common import ItkPrecision, ItkStruct
from .._xform import ItkTransform
from ._metadata import MatMetadata
from ._raw import MatRaw


def _record(xform: "MatTransform") -> tx.Optional[MatRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


@register_format
class MatTransform(BinaryFileReader, BinaryFileWriter, ItkTransform):
    """Transformation stored in an ITK binary MATLAB (`.mat`) file.

    ANTs writes every linear transform in this format. FSL FLIRT also writes
    `.mat` files, but as plain text, so the two formats are told apart by their
    content.

    The transformation is the chain of blocks of one top-level transform of
    the file, in the order in which they apply. The blocks and the bytes of
    the file are held as a [`MatRaw`][] record by the [`MatMetadata`][] of
    the transformation, and the chain is decoded from the record when it is
    first used. The decoded chain is a list that is cached by the
    transformation, so a change made to it in place, such as a replaced
    block, is kept.

    The record describes the file as it was read, or as the transformation
    was last consolidated. A chain that is assigned or changed, and an
    `input` or `output` that is set, are pending until [`consolidate`][]
    encodes them into a new record, or until the transformation is
    written.

    !!! note "What is written"
        A transformation read from a file, whose chain still holds the
        blocks of the record, is written back as read, with every block
        of the file, byte for byte. The option `position` only selects
        what is read, so it never changes what is written. The writer
        options `byteorder` and `precision` encode the blocks again when
        they differ from those of the record.

        Otherwise, the transformation is consolidated, and its chain must
        hold a single transformation, as ANTs writes one transform per
        file. An ITK block is written as it is, center included, and an
        affine between ITK spaces as an `AffineTransform` centered on the
        origin.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("mat",)

    PRIORITY: tx.ClassVar[int] = -1
    """
    Priority below the Convert3D matrix.

    A `.mat` name is claimed by an ITK MATLAB file, a Convert3D matrix
    and a FLIRT matrix. `io.save` writes an affine in the format that
    keeps its endpoints: an affine between LPS spaces as an ITK file,
    and one between RAS spaces as a Convert3D matrix. An affine whose
    systems are not known is written as a Convert3D matrix, which this
    priority decides, and a FLIRT matrix is written only when the
    images that it needs are given as conversion options. A file is
    read by the confidence of each format, which decides before the
    priority.
    """

    _metadata: NoRepr[tx.Optional[MatMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds the blocks as a record.

    A transformation built from a chain has no metadata. Assigning other
    metadata drops the chain decoded from the old record, so that it is
    decoded from the new one, and keeps a chain that was assigned.
    """

    _position: KwOnly[tx.Optional[int]] = None

    position = smartproperty("position", invalidates="transformations")
    """The index of the top-level transform of the file that is read.

    A file whose first block is a `CompositeTransform` header holds a single
    composite transform, and any other file holds one transform per block. By
    default, the first transform is read, with a warning if the file holds
    several. Assigning another index drops the chain decoded for the old index
    and keeps a chain that was assigned. The index only selects what is read,
    so it does not change the record.
    """

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[_xforms.Transformation]:
        """The blocks of the transform, in the order in which they apply.

        The chain is decoded from the record on first access and cached, apart
        from an assigned chain, so a change made to the list in place is kept.
        Its blocks are the blocks of the record, whose parameters are
        read-only. Assigning other metadata or another `position` drops the
        decoded chain, and a change made to it in place, and keeps an assigned
        chain. Assigning a list replaces the stored one. In both cases, the
        list is compared at [`consolidate`][], and by a write, with the chain
        that the record decodes to, block by block. A transformation without a
        record has no chain.

        Raises
        ------
        ParserContentError
            If the record has no transform at `position`.
        """
        record = _record(self)
        if record is None:
            return []
        return _decode(record, self.position)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a binary stream as an ITK `.mat` file.

        The stream is scored by [`MatMetadata.sniff_fileobj`][].

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not an ITK
            `.mat` file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MatMetadata.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes as those of an ITK `.mat` file.

        The bytes are scored by [`MatMetadata.sniff_bytes`][].

        Parameters
        ----------
        content : bytes
            The content, or its beginning.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not an ITK
            `.mat` file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MatMetadata.sniff_bytes(content, error=error, **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Reject text, which is never an ITK `.mat` file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the content.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning 0.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            Always `Confidence.NO`.
        """
        return MatMetadata.sniff_lines(lines, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(
        cls,
        file: tx.IO,
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """Read a transformation from a binary stream.

        The stream is read by [`MatMetadata.from_fileobj`][], and the chain
        is decoded from the record when it is first used.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        position : int, optional
            The index of the top-level transform to read. A file with a
            composite holds the composite as its only top-level transform,
            and a file without one holds one top-level transform per block.
            By default the first one is read, with a warning when there are
            several.
        **kwargs : Any
            Other fields of the transformation. Any other option is
            refused.

        Returns
        -------
        MatTransform
            The transformation.

        Raises
        ------
        ParserContentError
            If the content is not an ITK `.mat` file, or if it has no
            transform at `position`.
        """
        metadata = MatMetadata.from_fileobj(file)
        check_position(metadata.raw.blocks, position)
        return cls(metadata=metadata, position=position, **kwargs)

    @classmethod
    def from_raw(cls, raw: MatRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already read [`MatRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : MatRaw
            The record of the file.
        **kwargs : Any
            Field values, such as `position`, which go to the constructor.

        Returns
        -------
        MatTransform
            The transformation.

        Raises
        ------
        ParserContentError
            If the record has no transform at `position`.
        """
        check_position(raw.blocks, kwargs.get("position"))
        return cls(metadata=MatMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`MatRaw`][] is read with [`from_raw`][], and any other value as
        the bases read it.

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
        MatTransform
            The new transformation.
        """
        if isinstance(other, MatRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        byteorder: tx.Optional[str] = None,
        precision: tx.Optional[tx.Union[ItkPrecision, str]] = None,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending chain, systems and options into a new record.

        A transformation that holds a record is left as it is when its
        chain was never read, or when it still holds the blocks that the
        record decodes to, in the same order, and when the options agree
        with the record. An `input` or `output` that was set must be the
        system that the chain maps from or to, since the file stores no
        system, and it is then dropped. The option `position` only selects
        what is read, so it does not change the record.

        When only the options differ, the new record holds every block of
        the current record in the new byte order and precision. Otherwise,
        the chain must hold a single transformation, as ANTs writes one
        transform per file, and the new record holds its block: an ITK
        block as it is, and an affine between ITK spaces as an
        `AffineTransform` centered on the origin. The new record keeps the
        byte order of the current record, unless `byteorder` is given, and
        it is little-endian without a record. The new record is held by new
        metadata, and the option `position`, which indexed the old record,
        is reset.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an ITK
        MATLAB file holds no field of the vocabulary yet, so a field
        that is set is reported as lost, once, under the error policy of
        the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        byteorder : str, optional
            The byte order, `"<"`, `">"` or `"="` for the order of this
            machine. ITK writes the order of its machine, which is
            little-endian on common machines.
        precision : ItkPrecision or str, optional
            The precision of the parameters, `"double"` or `"float"`,
            which also appears in the name of each block. By default, each
            block keeps its precision. The fixed parameters of a block
            that is encoded again are written in double precision.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        MatTransform
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If the chain does not hold a single ITK block or affine
            between ITK spaces, or if a system that was set differs from
            the system of the chain.
        WriterError
            If `byteorder` is invalid.
        ValueError
            If `precision` is invalid.
        """
        report = ConversionReport(source="MatTransform", target="MatTransform")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, MatMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        byteorder = _byteorder(byteorder)
        precision = None if precision is None else ItkPrecision(precision)
        record = _record(self)
        decoded = (
            [] if record is None else _decode(record, self.position, True)
        )
        chain = held_transformations(self)
        check_systems(self, decoded if chain is None else chain)
        untouched = record is not None and (
            chain is None or same_chain(chain, decoded)
        )
        if untouched:
            blocks = record.blocks
            if byteorder in (None, record.byteorder) and all(
                precision in (None, block.precision) for block in blocks
            ):
                if not inplace:
                    return replace(
                        self,
                        transformations=None,
                        input=None,
                        output=None,
                        metadata=metadata,
                    )
                self.transformations = None
                self.input = None
                self.output = None
                self.metadata = metadata
                return self
        else:
            blocks = (_single_block(chain or []),)
        if precision is not None:
            blocks = tuple(_with_precision(b, precision) for b in blocks)
        if byteorder is None:
            byteorder = "<" if record is None else record.byteorder
        record = MatRaw(blocks=blocks, byteorder=byteorder)
        if metadata is None:
            metadata = MatMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self,
                metadata=metadata,
                transformations=None,
                input=None,
                output=None,
                position=None,
            )
        self.metadata = metadata
        self.input = None
        self.output = None
        self.position = None
        self.transformations = None
        return self

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the content of the ITK `.mat` file of the transformation.

        The transformation is consolidated into a copy with the writer
        options, and the record of the copy is written, so the
        transformation itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            The writer options `byteorder` and `precision`, as
            [`consolidate`][] takes them.

        Returns
        -------
        bytes
            The file content.

        Raises
        ------
        UnrepresentableTransformationError
            If the chain does not hold a single ITK block or affine
            between ITK spaces.
        WriterError
            If `byteorder` is invalid.
        """
        return _record(self.consolidate(**kwargs)).to_bytes()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _decode(
    record: MatRaw, position: tx.Optional[int], quiet: bool = False
) -> tx.List[_xforms.Transformation]:
    """Decode the chain of a record, as [`decode_chain`][] does."""
    return decode_chain(record.blocks, position, quiet=quiet)


def _byteorder(byteorder: tx.Optional[str]) -> tx.Optional[str]:
    """Return a byte order as `"<"` or `">"`, or `None` when not given.

    Raises
    ------
    WriterError
        If the byte order is not `"<"`, `">"` or `"="`.
    """
    if byteorder is None:
        return None
    if byteorder == "=":
        return "<" if sys.byteorder == "little" else ">"
    if byteorder not in ("<", ">"):
        raise WriterError(
            f"byteorder must be '<', '>' or '=', not {byteorder!r}."
        )
    return byteorder


def _single_block(chain: tx.Sequence[_xforms.Transformation]) -> ItkStruct:
    """Return the single block that stores a chain.

    ANTs writes a single linear transform per `.mat` file, so a chain of
    several transformations is refused.

    Raises
    ------
    UnrepresentableTransformationError
        If the chain does not hold a single ITK block or affine between
        ITK spaces.
    """
    if len(chain) != 1:
        raise UnrepresentableTransformationError(
            f"An ITK MATLAB file is written with a single transform, as "
            f"ANTs writes it, but this chain holds {len(chain)}. "
            f"Compose them first, for example with `.compute()`."
        )
    return block_of(chain[0])


def _with_precision(block: ItkStruct, precision: ItkPrecision) -> ItkStruct:
    """Return a block whose parameters are stored in a precision.

    The parameters hold the values that a file in that precision stores.
    """
    if block.precision == precision:
        return block
    dtype = np.float32 if precision == ItkPrecision.Float else np.float64
    parameters = np.asarray(block.parameters, dtype=np.float64).astype(dtype)
    return replace(
        block, precision=precision, parameters=read_only(parameters)
    )
