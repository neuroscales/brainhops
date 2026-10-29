"""The chain of transforms stored in an MNI transform file."""

__all__ = ["XfmTransform"]

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core._internal_api import (
    atomic_write_bytes,
    is_same_file,
    smartproperty,
)
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    ParserNotImplementedError,
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.metadata._base import (
    metadata_options,
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import TransformationFormat
from brainhops.io.transformations.base.fields import (
    split_ras_displacement_chain,
)

# this format
from ._formats import XfmFormat
from ._grid import (
    PendingVolume,
    XfmGridTransform,
    grid_from_volume,
    pending_name,
    read_volume,
    write_volume,
)
from ._metadata import XfmMetadata
from ._raw import XfmRaw, _decode, _read_bytes, _score, sniff_transforms

_Volumes = tx.Optional[tx.Tuple[tx.Any, ...]]
"""The displacement volumes of the grid transforms of a file, or `None`."""


# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------
# A transform file is parsed in one pass, so the matrices of its linear
# transforms live in the record of its metadata. The displacements of a
# grid transform live in a MINC volume, which the format reads into
# `rawdata`, one lazy image per grid transform. The chain is decoded
# from both when it is first used. A chain that is assigned is held by
# the data model, and it is encoded into a new record when the transform
# is consolidated or written. The design memo
# `spec/design/format-records.md` records this layout as item 140.


def _record(xform: "XfmTransform") -> tx.Optional[XfmRaw]:
    """Return the record of the metadata of a transform, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _grid_number(record: XfmRaw, index: int) -> int:
    """Return the number of grid transforms before transform `index`."""
    return sum(
        isinstance(block, XfmRaw.Grid) for block in record.transforms[:index]
    )


def _read_volumes(
    record: XfmRaw, location: tx.Optional[path.FilenameLike]
) -> _Volumes:
    """Read the displacement volumes of the grid transforms of a record.

    A volume whose relative name cannot be resolved, because the
    location of the transform file is not known, is `None`. A record
    without a grid transform has no volume, and `None` is returned.
    """
    grids = [b for b in record.transforms if isinstance(b, XfmRaw.Grid)]
    if not grids:
        return None
    return tuple(read_volume(block.volume, location) for block in grids)


def _decode_block(
    record: XfmRaw, index: int, volumes: _Volumes
) -> _xforms.Transformation:
    """Decode transform `index` of a record into a transformation.

    Raises
    ------
    ParserContentError
        If the displacement volume of a grid transform was not read.
    ParserNotImplementedError
        If the transform is a thin-plate spline.
    """
    block = record.transforms[index]
    if isinstance(block, XfmRaw.Linear):
        matrix = np.array(block.homogeneous[:-1])
        matrix.flags.writeable = False
        xform = _xforms.Affine(
            matrix=matrix, input=_systems.RASmm(), output=_systems.RASmm()
        )
    elif isinstance(block, XfmRaw.Grid):
        number = _grid_number(record, index)
        volume = None
        if volumes is not None and number < len(volumes):
            volume = volumes[number]
        if volume is None:
            raise ParserContentError(
                f"The displacement volume {block.volume!r} of this grid "
                f"transform was not read, because its name is relative to "
                f"the transform file, which was read without its path. "
                f"Read the transform file from its path."
            )
        xform = grid_from_volume(volume, block.volume)
    else:
        raise ParserNotImplementedError(
            "This MNI transform file holds a thin-plate spline, which "
            "brainhops cannot represent. The transform is kept in the "
            "record and written back unchanged."
        )
    return xform.inverse() if block.inverted else xform


# ----------------------------------------------------------------------
#   FORMAT
# ----------------------------------------------------------------------


@register_format
class XfmTransform(
    TextFileReader,
    TextFileWriter,
    XfmFormat,
    _xforms.ImmutableSequence,
    TransformationFormat,
):
    """A chain of transforms stored in an MNI transform file (`.xfm`).

    An MNI transform file concatenates one or more transforms, which are
    applied in the order of the file, so the file is read as an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of one transformation per transform. Each maps RAS millimetre world
    coordinates to RAS millimetre world coordinates, from the source to
    the target.

    | Transform of the file               | Transformation                |
    | ----------------------------------- | ----------------------------- |
    | `Linear`                            | `Affine` (`RASmm` to `RASmm`) |
    | `Grid_Transform`                    | [`XfmGridTransform`][]        |
    | `Thin_Plate_Spline_Transform`       | not supported                 |

    A transform with `Invert_Flag = True;` is read as the inverse of the
    transformation, as libminc reads it. The inverse of an affine is
    exact, and the inverse of a grid transform is computed by the data
    model when it is applied (see
    [`InverseDisplacementField`][brainhops.datamodel.transformations.InverseDisplacementField]).
    A file of one linear transform is also read by
    [`XfmAffine`][brainhops.io.transformations.minc.XfmAffine], which
    `load` prefers for such a file.

    The file is read in one pass, so the whole file is held as an
    [`XfmRaw`][] record by the [`XfmMetadata`][] of the transform. The
    displacement volumes of its grid transforms are read into `rawdata`,
    as MINC images whose voxels are read only when the chain is decoded.
    The chain is decoded when it is first used, and it is then cached.

    The record describes the file as it was read, or as the transform was
    last consolidated. A chain that is assigned, and an `input` or
    `output` that is set, are pending until [`consolidate`][] encodes them
    into a new record, or until the transform is written.

    !!! note "What is written"
        A transform whose chain was not assigned is written as its
        record, byte for byte, with its comments. A transform whose chain
        was assigned is consolidated: each element that this transform
        decoded from its own record is written as that transform of the
        file. Any other [`XfmGridTransform`][] is written as a grid
        transform whose displacements are a new MINC2 volume, and any
        other element must convert to an affine from `RASmm` to `RASmm`,
        which is written as a linear transform. An element taken from
        another object, or from a copy made with `replace`, is not
        recognized and is encoded again.

        A transform written to a path also places the displacement volume
        of each grid transform where the file names it, relative to the
        new file: a volume that was read is copied, and a new volume is
        written as `<name>_grid_<n>.mnc`, which libminc names it. Written
        to a stream, only the transform file is written, and the names of
        the new volumes are `grid_<n>.mnc`.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xfm",)

    PRIORITY: tx.ClassVar[int] = -1
    """
    Priority below the MNI affine.

    An `.xfm` name cannot tell a chain of transforms from a single
    affine, and both formats hold an affine exactly. `io.save` writes an
    affine as an
    [`XfmAffine`][brainhops.io.transformations.minc.XfmAffine], unless it
    already is an `XfmTransform`, which is built with
    `t.to(XfmTransform)`. A file is read by the confidence of each
    format, which decides before the priority.
    """

    _metadata: NoRepr[tx.Optional[XfmMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty(
        "metadata", invalidates=("transformations", "decoded_blocks")
    )
    """The metadata of the file, which holds the file as a record.

    A transform built from a chain has no metadata. Assigning other
    metadata drops the chain decoded from the old record, and a chain that
    was assigned is kept.
    """

    _rawdata: NoRepr[tx.Optional[tx.Tuple[tx.Any, ...]]] = None

    rawdata = smartproperty(
        "rawdata", invalidates=("transformations", "decoded_blocks")
    )
    """The displacement volumes of the grid transforms, or `None`.

    The volumes are [`MincImage`][brainhops.io.images.minc.MincImage]
    objects, one per grid transform in the order of the file, which read
    their voxels only when they are needed. A volume is `None` when its
    name is relative and the file was read without its path. A file
    without a grid transform, and a transform built from a chain, have no
    volume.
    """

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain, in the order of application.

        The chain is decoded from the record when it is first used, and
        it is then cached. A transform without a record and without an
        assigned chain has an empty chain. Assigning a chain replaces the
        decoded one, and the assigned chain is pending until
        [`consolidate`][] or the writer encodes it.

        Raises
        ------
        ParserContentError
            If the displacement volume of a grid transform was not read.
        ParserNotImplementedError
            If the file holds a thin-plate spline.
        """
        record = _record(self)
        if record is None:
            return ()
        return tuple(
            self.block_transformation(i) for i in range(len(record.transforms))
        )

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.Sequence[_xforms.Transformation]]
    ) -> None:
        # The assigned chain is stored as a tuple, the type of the chain
        # that is decoded, and the setter drops the decoded chain.
        self._transformations = None if value is None else tuple(value)

    def block_transformation(self, index: int) -> _xforms.Transformation:
        """Decode transform `index` of the record into a transformation.

        The transform is decoded once, and the same object is returned
        afterwards, so that the writer can recognize it. Assigning other
        metadata or other volumes forgets the decoded transforms.

        Parameters
        ----------
        index : int
            The index of the transform in the file.

        Returns
        -------
        Transformation
            The transformation that the transform encodes.

        Raises
        ------
        ParserContentError
            If the transformation has no record, or if the displacement
            volume of a grid transform was not read.
        ParserNotImplementedError
            If the transform is a thin-plate spline.
        """
        record = _record(self)
        if record is None:
            raise ParserContentError("This transformation has no record.")
        decoded = self.__dict__.setdefault("_cache_decoded_blocks", {})
        if index not in decoded:
            decoded[index] = _decode_block(record, index, self.rawdata)
        return decoded[index]

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the text of a file as an MNI transform file.

        A file that libminc reads is certainly of this format, unless it
        holds a single linear transform, which is only likely, so that
        [`XfmAffine`][brainhops.io.transformations.minc.XfmAffine] is
        preferred for it. A text that starts with the header line but that
        libminc would not read scores `Confidence.MAYBE`, so that reading
        it reports what is wrong.

        Parameters
        ----------
        text : str
            The text of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not an MNI transform
            file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        transforms = sniff_transforms(text)
        if transforms is None:
            return _score(False, text, error)
        if len(transforms) == 1 and isinstance(transforms[0], XfmRaw.Linear):
            return Confidence.LIKELY
        return Confidence.CERTAIN

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file as an MNI transform file.

        The lines are joined and scored by [`sniff_text`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines are not an MNI
            transform file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_text("\n".join(lines), error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of a file as an MNI transform file.

        The bytes are decoded as [`XfmRaw.from_bytes`][] decodes them, and
        scored by [`sniff_text`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not an MNI
            transform file.
        **kwargs : Any
            Options of the reader. `encoding` is used, and the others are
            ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        text = _decode(content, kwargs.get("encoding", "utf-8"))
        return cls.sniff_text(text, error=error)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a file as an MNI transform file.

        The file is read as bytes and scored by [`sniff_bytes`][].

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an MNI transform
            file.
        **kwargs : Any
            Options of the reader, passed to [`sniff_bytes`][].

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        content = _read_bytes(filename)
        if content is None:
            return super().sniff_filename(filename, error=error, **kwargs)
        return cls.sniff_bytes(content, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(
        cls,
        raw: XfmRaw,
        location: tx.Optional[path.FilenameLike] = None,
        **kwargs,
    ) -> tx.Self:
        """Build a transform from an already parsed [`XfmRaw`][].

        The record is held by new metadata, without a copy, and the
        displacement volumes of its grid transforms are read.

        Parameters
        ----------
        raw : XfmRaw
            The record of the file.
        location : path-like, optional
            The path of the file, against which the relative names of the
            displacement volumes are resolved. Without it, only the
            volumes with an absolute name are read.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The transform.

        Raises
        ------
        ParserExistsError
            If a displacement volume does not exist.
        ParserContentError
            If a displacement volume is not a MINC file that brainhops
            reads.
        """
        volumes = _read_volumes(raw, location)
        return cls(
            metadata=XfmMetadata.from_raw(raw), rawdata=volumes, **kwargs
        )

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build a transform from the text of an MNI transform file.

        The text has no location, so only the displacement volumes with an
        absolute name are read.

        Parameters
        ----------
        text : str
            The text of the file.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The transform.

        Raises
        ------
        ParserContentError
            If libminc would not read the text.
        """
        return cls.from_raw(XfmRaw.from_text(text), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transform from the lines of an MNI transform file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The transform.
        """
        return cls.from_raw(XfmRaw.from_lines(lines), **kwargs)

    @classmethod
    def from_bytes(
        cls, content: bytes, encoding: str = "utf-8", **kwargs
    ) -> tx.Self:
        """Build a transform from the bytes of an MNI transform file.

        The bytes have no location, so only the displacement volumes with
        an absolute name are read.

        Parameters
        ----------
        content : bytes
            The content of the file.
        encoding : str, default="utf-8"
            The encoding of the text.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The transform.
        """
        return cls.from_raw(XfmRaw.from_bytes(content, encoding), **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read a transform from an open file, in text or binary mode.

        The relative names of the displacement volumes are resolved
        against the name of the file, when the file has one.

        Parameters
        ----------
        file : IO
            The open file.
        **kwargs : Any
            Options of [`from_bytes`][], and field values.

        Returns
        -------
        Self
            The transform.
        """
        content = file.read()
        if isinstance(content, str):
            raw = XfmRaw.from_text(content)
        else:
            raw = XfmRaw.from_bytes(content, kwargs.pop("encoding", "utf-8"))
        name = getattr(file, "name", None)
        location = name if isinstance(name, (str, os.PathLike)) else None
        return cls.from_raw(raw, location=location, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, encoding: str = "utf-8", **kwargs
    ) -> tx.Self:
        """Read a transform from an MNI transform file.

        The file is read as bytes by [`XfmRaw.from_filename`][], and the
        displacement volumes of its grid transforms are read lazily, from
        the names that the file stores, relative to its directory.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        encoding : str, default="utf-8"
            The encoding of the text.
        **kwargs : Any
            Field values, which go to the constructor.

        Returns
        -------
        Self
            The transform.

        Raises
        ------
        ParserExistsError
            If the file or a displacement volume does not exist.
        ParserContentError
            If libminc would not read the file, or if a displacement
            volume is not a MINC file that brainhops reads.
        """
        raw = XfmRaw.from_filename(filename, encoding=encoding)
        return cls.from_raw(raw, location=filename, **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transform from a record, a file or a data model.

        An [`XfmRaw`][] is read with [`from_raw`][], and any other value
        as the bases read it.

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
            The new transform.
        """
        if isinstance(other, XfmRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending chain and systems into a new record.

        Systems that were assigned to `input` or `output` must be `RASmm`,
        since every transform of the file maps RAS to RAS. When no chain
        has been assigned, the record and the volumes are kept as they
        are, so that the file is written as it was read. An assigned chain
        is encoded into a new record, held by new metadata, with the
        comments of the current record. Each element that this transform
        decoded from its own record is encoded as that transform, and keeps
        its displacement volume. Any other [`XfmGridTransform`][] is
        encoded as a grid transform with a new displacement volume, which
        is written with the file. Any other element must convert to an
        affine from `RASmm` to `RASmm`, and a sequence that does not is
        encoded element by element. Afterwards the chain is decoded from
        the new record, with the precision of the file.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a MINC
        transform file holds no field of the vocabulary yet, so a field
        that is set is reported as lost, once, under the error policy of
        the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transform rather than a copy.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        Self
            The consolidated copy, or this transform with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If an element cannot be encoded, or if a system that was
            assigned to `input` or `output` is not `RASmm`.
        WriterError
            If the transform has neither a record nor a chain.
        """
        report = ConversionReport(source="XfmTransform", target="XfmTransform")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, XfmMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        record = _record(self)
        assigned = self._transformations is not None
        declared = self._input is not None or self._output is not None
        if declared:
            _check_ras(self, "chain")
        if not assigned:
            if record is None:
                raise WriterError("This transform has no content to write.")
            if not declared:
                if not inplace:
                    return replace(self, metadata=metadata)
                if requested:
                    self.metadata = metadata
                return self
            volumes = self.rawdata
        else:
            decoded = self.__dict__.get("_cache_decoded_blocks", {})
            record, volumes = _encode(
                self.transformations, record, self.rawdata, decoded
            )
            if metadata is None:
                metadata = XfmMetadata.from_raw(record)
            else:
                metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self,
                metadata=metadata,
                rawdata=volumes,
                transformations=None,
                input=None,
                output=None,
            )
        self.metadata = metadata
        self.rawdata = volumes
        self.transformations = None
        self.input = None
        self.output = None
        return self

    @takes_error_policy
    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the transform to a file, with its displacement volumes.

        The transform is consolidated into a copy before anything is
        written, so a transform that cannot be encoded leaves the files
        untouched, and the transform itself is not changed. The
        displacement volume of each grid transform is then copied where
        the file names it, relative to the new file, unless it is already
        there, and the file is written last. A local file is replaced at
        once.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][]. The fields of the metadata
            are given to [`consolidate`][].

        Raises
        ------
        UnrepresentableTransformationError
            If the transform cannot be encoded.
        WriterError
            If a displacement volume cannot be placed beside the file,
            because it was not read from a file or because its voxels were
            changed.
        """
        fields, kwargs = metadata_options(kwargs)
        if isinstance(filename, str):
            filename = path.Path(filename)
        copy = self.consolidate(**fields)
        record = _named(_record(copy), copy.rawdata, filename)
        content = record.to_bytes(**kwargs)
        _place_volumes(record, copy.rawdata, filename)
        atomic_write_bytes(filename, content)

    @takes_error_policy
    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the transform file to an open file, without its volumes.

        Parameters
        ----------
        file : IO
            The open file, in text or binary mode.
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][]. The fields of the metadata
            are given to [`consolidate`][].

        Raises
        ------
        UnrepresentableTransformationError
            If the transform cannot be encoded.
        """
        fields, kwargs = metadata_options(kwargs)
        _record(self.consolidate(**fields)).to_fileobj(file, **kwargs)

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the text of the MNI transform file that encodes the chain.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which refuses them. The
            fields of the metadata are given to [`consolidate`][].

        Returns
        -------
        str
            The text of the file.

        Raises
        ------
        UnrepresentableTransformationError
            If the transform cannot be encoded.
        """
        fields, kwargs = metadata_options(kwargs)
        return _record(self.consolidate(**fields)).to_text(**kwargs)

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the MNI transform file that encodes the chain.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which refuses them. The
            fields of the metadata are given to [`consolidate`][].

        Yields
        ------
        str
            The lines of the file, without line endings.

        Raises
        ------
        UnrepresentableTransformationError
            If the transform cannot be encoded.
        """
        fields, kwargs = metadata_options(kwargs)
        return _record(self.consolidate(**fields)).to_lines(**kwargs)

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the MNI transform file that encodes the chain.

        Parameters
        ----------
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][]. The fields of the metadata
            are given to [`consolidate`][].

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        UnrepresentableTransformationError
            If the transform cannot be encoded.
        """
        fields, kwargs = metadata_options(kwargs)
        return _record(self.consolidate(**fields)).to_bytes(**kwargs)


# ----------------------------------------------------------------------
#   ENCODING A CHAIN AS A RECORD
# ----------------------------------------------------------------------


def _system_name(system: tx.Optional[_systems.CoordinateSystem]) -> str:
    """Return the name of a coordinate system for an error message."""
    name = getattr(system, "name", None)
    if name:
        return name
    if system is not None:
        return type(system).__name__
    return "an unspecified system"


def _check_ras(xform: _xforms.Transformation, what: str) -> None:
    """Refuse a transformation that does not map `RASmm` to `RASmm`.

    Raises
    ------
    UnrepresentableTransformationError
        If the input or the output is not `RASmm`.
    """
    src, dst = xform.input, xform.output
    if isinstance(src, _systems.RASmm) and isinstance(dst, _systems.RASmm):
        return
    raise UnrepresentableTransformationError(
        f"An MNI transform file maps RAS to RAS world coordinates, and this "
        f"{what} maps {_system_name(src)} to {_system_name(dst)}. Set its "
        f"input and output to RASmm to say that it is one."
    )


def _encode(
    chain: tx.Sequence[_xforms.Transformation],
    old: tx.Optional[XfmRaw],
    volumes: _Volumes,
    decoded: tx.Dict[int, _xforms.Transformation],
) -> tx.Tuple[XfmRaw, _Volumes]:
    """Encode a chain of transformations as a new record and its volumes.

    The rules are described in `XfmTransform._consolidate`. An element
    that was decoded from transform `i` of `old`, which `decoded` lists,
    is recognized by its identity.

    Raises
    ------
    UnrepresentableTransformationError
        If an element cannot be encoded.
    WriterError
        If the chain is empty.
    """
    # A grid transform is recognized by its identity, as an X5 node is
    # (items 36 and 141 of the design memo).
    reused = {id(xform): index for index, xform in decoded.items()}
    blocks: tx.List[tx.Any] = []
    grids: tx.List[tx.Any] = []
    for xform in chain:
        if id(xform) in reused:
            index = reused[id(xform)]
            block = old.transforms[index]
            blocks.append(block)
            if isinstance(block, XfmRaw.Grid):
                number = _grid_number(old, index)
                grids.append(None if volumes is None else volumes[number])
        elif isinstance(xform, XfmGridTransform):
            blocks.append(XfmRaw.Grid(volume=pending_name(len(grids))))
            grids.append(_pending(xform))
        else:
            blocks.extend(_encode_linear(xform))
    if not blocks:
        raise WriterError(
            "An MNI transform file holds at least one transform, and this "
            "chain is empty."
        )
    comments = () if old is None else old.comments
    record = XfmRaw(comments=comments, transforms=tuple(blocks))
    return record, tuple(grids) if grids else None


def _pending(xform: XfmGridTransform) -> PendingVolume:
    """Return the volume of displacements that a grid transform writes.

    The displacements are rotated back to RAS millimetres, in the backend
    of the field, and nothing is computed until the volume is written.

    Raises
    ------
    UnrepresentableTransformationError
        If the grid transform does not map RAS to RAS.
    """
    _check_ras(xform, "grid transform")
    vox2ras, vectors = split_ras_displacement_chain(
        xform.transformations, "A grid transform", ndim=3
    )
    return PendingVolume(vectors, vox2ras)


def _named(
    record: XfmRaw, volumes: _Volumes, filename: path.FilenameLike
) -> XfmRaw:
    """Name the new displacement volumes after the transform file.

    libminc names the volume of grid transform `n` of `<name>.xfm` as
    `<name>_grid_<n>.mnc`, beside the file, and so does this writer for
    a volume that it writes.
    """
    stem = os.path.basename(str(filename))
    if stem.lower().endswith(".xfm"):
        stem = stem[: -len(".xfm")]
    blocks = list(record.transforms)
    number = 0
    for index, block in enumerate(blocks):
        if not isinstance(block, XfmRaw.Grid):
            continue
        volume = None if volumes is None else volumes[number]
        if isinstance(volume, PendingVolume):
            blocks[index] = replace(block, volume=f"{stem}_grid_{number}.mnc")
        number += 1
    return replace(record, transforms=tuple(blocks))


def _encode_linear(xform: _xforms.Transformation) -> tx.List[XfmRaw.Linear]:
    """Encode a transformation as linear transforms.

    A transformation that converts to an affine is one linear transform,
    and a sequence that does not is encoded element by element.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation, or an element of it, does not convert to an
        affine from `RASmm` to `RASmm`.
    """
    try:
        affine = xform.to(_xforms.Affine)
    except Exception as error:  # noqa: BLE001
        if isinstance(xform, _xforms.Sequence):
            try:
                return [
                    block for item in xform for block in _encode_linear(item)
                ]
            except UnrepresentableTransformationError:
                pass
        raise UnrepresentableTransformationError(
            f"An MNI transform file cannot encode a {type(xform).__name__}: "
            f"it stores affines and grid transforms. A field is converted "
            f"to grid transforms by `t.to(XfmTransform)`."
        ) from error
    _check_ras(xform, "affine")
    matrix = affine.homogeneous_matrix
    matrix = np.eye(4) if matrix is None else np.asarray(matrix, np.float64)
    if matrix.shape != (4, 4):
        raise UnrepresentableTransformationError(
            f"An MNI transform file stores 3-D affines, and this one has "
            f"shape {matrix.shape[0] - 1}x{matrix.shape[1] - 1}."
        )
    rows = tuple(tuple(float(v) for v in row) for row in matrix[:-1])
    return [XfmRaw.Linear(matrix=rows)]


# ----------------------------------------------------------------------
#   WRITING THE DISPLACEMENT VOLUMES
# ----------------------------------------------------------------------


def _volume_source(volume: tx.Any) -> tx.Optional[tx.Any]:
    """Return the path of the file that a displacement volume was read from.

    `None` is returned for a volume that was not read from a path, and for
    a volume whose voxels were assigned, since brainhops cannot write
    MINC volumes.
    """
    from brainhops.io.common.minc._data import _MincProxy

    if volume is None or volume._data is not None:
        return None
    proxy = volume.rawdata
    if isinstance(proxy, _MincProxy) and isinstance(
        proxy.source, (str, os.PathLike)
    ):
        return proxy.source
    return None


# A volume that was read is copied as it was read (item 141 of the design
# memo), and a new volume is written from its displacements.


def _place_volumes(
    record: XfmRaw, volumes: _Volumes, filename: path.FilenameLike
) -> None:
    """Copy the displacement volumes where a new transform file names them.

    A volume with an absolute name stays where it is. A volume with a
    relative name is copied beside the new file, under that name, unless
    the copy would be the file that the volume reads, or a file with the
    same content.

    Raises
    ------
    WriterError
        If a volume with a relative name was not read from a file, or if
        its voxels were changed.
    """
    grids = [b for b in record.transforms if isinstance(b, XfmRaw.Grid)]
    for number, block in enumerate(grids):
        if os.path.isabs(block.volume):
            continue
        target = path.Path(filename).parent / block.volume
        volume = None if volumes is None else volumes[number]
        if isinstance(volume, PendingVolume):
            write_volume(target, volume)
            continue
        source = _volume_source(volume)
        if source is None:
            raise WriterError(
                f"The displacement volume {block.volume!r} of a grid "
                f"transform must be placed beside {filename}, and it was "
                f"not read from a file, or its voxels were changed. Convert "
                f"the grid transform to an XfmGridTransform of its "
                f"displacements to write it as a new volume."
            )
        if is_same_file(source, target):
            continue
        with path.Path(source).open("rb") as file:
            content = file.read()
        if path.exists(target):
            with path.Path(target).open("rb") as file:
                if file.read() == content:
                    continue
        atomic_write_bytes(target, content)
