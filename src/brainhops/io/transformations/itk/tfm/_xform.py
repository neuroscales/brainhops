"""The chain of transformations stored in an ITK text `.tfm` file."""

__all__ = ["TfmTransform"]

import typing_extensions as tx
from bagof.magic import Inherit, KwOnly, NoRepr, replace

from brainhops._core.properties import smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.errors import UNSET, ErrorPolicy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import TextFileReader, TextFileWriter
from brainhops.io.common._geometry import held_transformations
from brainhops.io.metadata._base import (
    metadata_options,
    metadata_to_write,
    requested_fields,
)

from .._chains import (
    _check_position,
    _check_systems,
    _decode_chain,
    _encode_chain,
    _same_chain,
)
from .._xform import ItkTransform
from ._metadata import TfmMetadata
from ._raw import TfmRaw


def _record(xform: "TfmTransform") -> tx.Optional[TfmRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


@register_format
class TfmTransform(TextFileReader, TextFileWriter, ItkTransform):
    """Transformation stored in an ITK text `.tfm` file.

    The transformation is the chain of blocks of one top-level transform of
    the file, in the order in which they apply (see
    [`brainhops.io.transformations.itk.tfm`][]). The blocks of the file are
    held as a [`TfmRaw`][] record by the [`TfmMetadata`][] of the
    transformation, and the chain is decoded from the record when it is
    first used. The decoded chain is a list that is cached by the
    transformation, so a change made to it in place, such as an appended
    block, is kept.

    The record describes the file as it was read, or as the transformation
    was last consolidated. A chain that is assigned or changed, and an
    `input` or `output` that is set, are pending until [`consolidate`][]
    encodes them into a new record, or until the transformation is
    written.

    !!! note "What is written"
        The transformation is consolidated, and the blocks of its record
        are written as ITK writes a text file (see
        [`TfmRaw.to_lines`][brainhops.io.transformations.itk.tfm.TfmRaw.to_lines]).
        A transformation read from a file whose chain still holds the
        blocks of the record is written with every block of the file,
        including the transforms that `position` did not select. The
        comments of the file and the digits of its numbers are not kept,
        but every number reads back as the same value. An ITK block is
        written as it is, center included, and an affine between ITK
        spaces as an `AffineTransform` centered on the origin. A chain
        of several blocks is written as a composite.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tfm",)
    HINTS = ("tfm",)

    _metadata: NoRepr[tx.Optional[TfmMetadata]] = Inherit(
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

    A file that starts with a `CompositeTransform` header holds a single
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
        list is compared at [`consolidate`][] with the chain that the record
        decodes to, block by block. A transformation without a record has no
        chain.

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
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a line as the first block of a `.tfm` file.

        The line is scored by [`TfmMetadata.sniff_line`][].

        Parameters
        ----------
        line : str
            The first line of the file that is not a comment.
        error : bool or type[Exception], default=False
            Whether to raise an error when the line does not start a
            block.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return TfmMetadata.sniff_line(line, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_lines(
        cls,
        lines: tx.Iterable[str],
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """Read a transformation from the lines of a `.tfm` file.

        The lines are read by [`TfmMetadata.from_lines`][], and the chain
        is decoded from the record when it is first used.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        position : int, optional
            The index of the top-level transform to read. By default, the
            first transform is read, with a warning if the file holds
            several.
        **kwargs : Any
            Other fields of the transformation. Any other option is
            refused.

        Returns
        -------
        TfmTransform
            The transformation.

        Raises
        ------
        ParserContentError
            If the file has no transform at `position`, or if a composite
            header is not its first block.
        """
        metadata = TfmMetadata.from_lines(lines)
        _check_position(metadata.raw.blocks, position)
        return cls(metadata=metadata, position=position, **kwargs)

    @classmethod
    def from_raw(cls, raw: TfmRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already read [`TfmRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : TfmRaw
            The record of the file.
        **kwargs : Any
            Field values, such as `position`, which go to the constructor.

        Returns
        -------
        TfmTransform
            The transformation.

        Raises
        ------
        ParserContentError
            If the record has no transform at `position`.
        """
        _check_position(raw.blocks, kwargs.get("position"))
        return cls(metadata=TfmMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`TfmRaw`][] is read with [`from_raw`][], and any other value as
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
        TfmTransform
            The new transformation.
        """
        if isinstance(other, TfmRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    @takes_error_policy
    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_error: ErrorPolicy = UNSET,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending chain and systems into a new record.

        A transformation that holds a record is left as it is when its
        chain was never read, or when it still holds the blocks that the
        record decodes to, in the same order. An `input` or `output` that
        was set must be the system that the chain maps from or to, since
        the file stores no system, and it is then dropped. The option
        `position` only selects what is read, so it does not change the
        record.

        Otherwise, a new record holds the blocks of the chain: an ITK
        block as it is, and an affine between ITK spaces as an
        `AffineTransform` centered on the origin. A chain of several
        blocks is stored as a composite, as ITK writes it. A
        transformation without a record and without a chain gives an
        empty record. The new record is held by new metadata, and the
        option `position`, which indexed the old record, is reset.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an ITK
        text file holds no field of the vocabulary yet, so a field that
        is set is reported as lost, once, by `on_error`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        on_error : ErrorPolicy, optional
            What to do with the metadata that the record cannot hold,
            and with any other conversion error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        TfmTransform
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If a transformation of the chain is neither an ITK block nor
            an affine between ITK spaces, or if a system that was set
            differs from the system of the chain.
        """
        report = ConversionReport(source="TfmTransform", target="TfmTransform")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, TfmMetadata, fields, report
        )
        apply_loss_policy(report, on_error, stacklevel=2)
        if not requested:
            metadata = self.metadata
        record = _record(self)
        decoded = (
            [] if record is None else _decode(record, self.position, True)
        )
        chain = held_transformations(self)
        _check_systems(self, decoded if chain is None else chain)
        if record is not None and (
            chain is None or _same_chain(chain, decoded)
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
        record = TfmRaw(blocks=_encode_chain(chain or []))
        if metadata is None:
            metadata = TfmMetadata.from_raw(record)
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

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the `.tfm` file of the transformation.

        The transformation is consolidated into a copy, whose record is
        written, so the transformation itself is not changed, and a
        transformation that cannot be encoded raises before any line is
        written.

        Parameters
        ----------
        **kwargs : Any
            `on_error` and the fields of the metadata, which are given
            to [`consolidate`][]. The writer of the record takes no
            other option.

        Returns
        -------
        iterator of str
            The lines of the file, without line breaks.

        Raises
        ------
        UnrepresentableTransformationError
            If a transformation of the chain is neither an ITK block nor
            an affine between ITK spaces.
        """
        fields, options = metadata_options(kwargs)
        return _record(self.consolidate(**fields)).to_lines(**options)

    def to_text(self, **kwargs) -> str:
        """Return the content of the `.tfm` file of the transformation.

        The content is made of the lines of [`to_lines`][].

        Parameters
        ----------
        **kwargs : Any
            `on_error` and the fields of the metadata, which are given
            to [`consolidate`][].

        Returns
        -------
        str
            The content of the file, ending with a newline.

        Raises
        ------
        UnrepresentableTransformationError
            If a transformation of the chain is neither an ITK block nor
            an affine between ITK spaces.
        """
        return "".join(line + "\n" for line in self.to_lines(**kwargs))


def _decode(
    record: TfmRaw, position: tx.Optional[int], quiet: bool = False
) -> tx.List[_xforms.Transformation]:
    """Decode the chain of a record, as [`_decode_chain`][] does."""
    return _decode_chain(record.blocks, position, quiet=quiet)
