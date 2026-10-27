"""The chain of transformations stored in elastix transform parameter files."""

__all__ = [
    "ElastixTransform",
    "ElastixParameterTransform",
    "ElastixTomlTransform",
]

import os
from warnings import warn

import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

from brainhops._core import path
from brainhops._core.path import is_same_file, write_file
from brainhops._core.properties import lazyproperty, smartproperty
from brainhops._core.streams import preserve_position
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    ParserNotImplementedError,
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.metadata._base import metadata_to_write, requested_fields
from brainhops.io.transformations.base import TransformationFormat
from brainhops.io.transformations.base.affines import _compatible
from brainhops.io.transformations.itk._common import ItkStruct
from brainhops.io.transformations.itk._systems import _make_system

from ._blocks import (
    fixed_geometry,
    map_to_block,
    ndim_of,
    transformation_to_map,
)
from ._metadata import ElastixMetadata
from ._parser import get_bool, get_string
from ._raw import ElastixRaw, is_absolute

_Initial = tx.Union[bool, None, path.FilenameLike, _xforms.Transformation]
"""What a reader does with the initial transform that a file names."""


# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------
# A parameter file is parsed in one pass, and the files of its chain of
# initial transforms are read with it, so the whole chain lives in the
# record of the metadata. The block of each file is decoded from its
# record when it is first used, and its parameters are a read-only view
# of the array of the record. A chain that is assigned is held by the data
# model, and it is encoded into a new record when the transformation is
# consolidated or written.


def _record(xform: "ElastixTransform") -> tx.Optional[ElastixRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


# ----------------------------------------------------------------------
#   FORMATS
# ----------------------------------------------------------------------


# Text adapters supply byte decoding and encoding before the generic bases.
class ElastixTransform(
    TextFileReader,
    TextFileWriter,
    TransformationFormat,
    _xforms.Sequence,
):
    """Transformation stored in an elastix transform parameter file.

    The transformation is a
    [`Sequence`][brainhops.datamodel.transformations.Sequence] of ITK
    blocks, which are described in
    [`brainhops.io.transformations.elastix`][]. It maps LPS coordinates of
    the fixed image to LPS coordinates of the moving image, which is the
    pull direction that transformix uses. The blocks of the chained
    initial transforms come first, followed by the block of the file's own
    transform.

    A parameter file is read in one pass, so the whole file is held as an
    [`ElastixRaw`][] record by the [`ElastixMetadata`][] of the
    transformation, together with the records of the files of its chain.
    For example, the parameter map of the file is
    `xform.metadata.raw.parameters`. The chain is decoded from the record
    when it is first used, and it is then cached. A transformation built
    from a chain has no metadata.

    The record describes the files as they were read, or as the
    transformation was last consolidated. A chain that is assigned, and an
    `input` or `output` that is set, are pending until [`consolidate`][]
    encodes them into a new record, or until the transformation is
    written.

    This class is abstract. Its subclasses [`ElastixParameterTransform`][]
    (`.txt`) and [`ElastixTomlTransform`][] (`.toml`) are the registered
    formats.

    !!! note "What is written"
        A transformation whose chain was not assigned is written as its
        record, whose text is written back as it was read. When it is
        written to a path, each file of its chain whose name is relative
        is written where the new file names it, as described in
        [`to_filename`][]. A transformation whose chain was assigned is
        consolidated, and written as a single transform without an initial
        transform, as described in [`consolidate`][].
    """

    HINTS = ("elastix", "transformix")

    _metadata: NoRepr[tx.Optional[ElastixMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty(
        "metadata", invalidates=("transformations", "block", "initial")
    )
    """The metadata of the file, which holds the file as a record.

    A transformation built from a chain has no metadata. Assigning other
    metadata drops the chain, the block and the initial transform decoded
    from the old record. A chain that was assigned is kept.
    """

    _initial: NoRepr[tx.Optional[_xforms.Transformation]] = None

    @smartproperty(cache=True, invalidates="transformations")
    def initial(self) -> tx.Optional[_xforms.Transformation]:
        """The initial transform, normally an `ElastixTransform`.

        The initial transform is decoded from the record of the initial
        file that the record holds, as an [`ElastixParameterTransform`][]
        or an [`ElastixTomlTransform`][] that holds that record. It is
        `None` when the file names no initial transform, or when the file
        was read with `initial=False`.

        A transformation given as `initial` when the file is read, or
        assigned, is returned instead. Like the options of other readers,
        it selects what is decoded, and does not change the record: the
        file is still written with the initial file name that it stores.
        """
        record = _record(self)
        if record is None or record.initial is None:
            return None
        return _view(record.initial)

    @lazyproperty
    def block(self) -> tx.Optional[ItkStruct]:
        """The ITK block of the file's own transform, or `None`.

        The block is decoded from the record, and its parameters are a
        read-only view of the parameters of the record. A transformation
        without a record has no block.
        """
        record = _record(self)
        if record is None:
            return None
        return map_to_block(record.parameters)

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain, in the order of application.

        The flattened initial transforms come first and the block of the
        file's own transform comes last, since elastix computes
        `T1(T0(x))`. The chain is decoded when it is first used, and it is
        then cached. A chain that is assigned replaces the decoded chain,
        and it is pending until [`consolidate`][] or the writer encodes it.

        Raises
        ------
        ParserNotImplementedError
            If the file adds its transform to its initial transform
            (`HowToCombineTransforms "Add"`), which a chain cannot express.
        """
        _check_composed(self)
        chain: tx.List[_xforms.Transformation] = []
        initial = self.initial
        if isinstance(initial, _xforms.Sequence):
            chain.extend(initial.transformations or ())
        elif initial is not None:
            chain.append(initial)
        if self.block is not None:
            chain.append(self.block)
        return tuple(chain)

    @property
    def initial_filename(self) -> tx.Optional[str]:
        """The file name of the initial transform, as the file stores it.

        The name is `None` when the file names no initial transform, or
        when the transformation has no record.
        """
        record = _record(self)
        return None if record is None else record.initial_filename

    @property
    def fixed_geometry(self) -> tx.Optional[Geometry]:
        """The geometry of the fixed image, or `None` if the file omits it.

        The geometry is the grid onto which transformix resamples the moving
        image. It is built from `Size`, `Index`, `Spacing`, `Origin` and
        `Direction`, and its transformation maps voxels to LPS millimetres.
        """
        record = _record(self)
        return None if record is None else fixed_geometry(record.parameters)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: ElastixRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already read [`ElastixRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : ElastixRaw
            The record of the file.
        **kwargs : Any
            Field values, such as `initial`, which go to the constructor.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=ElastixMetadata.from_raw(raw), **kwargs)

    # --- to -----------------------------------------------------------

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the transformation to a file, with the files of its chain.

        The transformation is consolidated into a copy before anything is
        written, so a transformation that cannot be encoded leaves the
        files untouched, and the transformation itself is not changed. Each
        file of the chain whose name is relative is then written where the
        new file names it, which is beside it, unless that file already
        holds the same bytes. A name that is absolute is left as it is,
        and nothing is written for it, nor for a chain that was not read,
        as with `initial=False`. The file itself is written last, and a
        local file is replaced at once.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            `on_loss` and the fields of the metadata, given to
            [`consolidate`][].

        Raises
        ------
        UnrepresentableTransformationError
            If the chain cannot be written as a single elastix transform.
        WriterError
            If a file of the chain would be written over the new file, or
            over another file that holds different bytes.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        record = _record(self.consolidate(**kwargs))
        content = record.to_bytes()
        _place_initial(record, filename, filename)
        write_file(filename, content)

    def to_text(self, **kwargs) -> str:
        """Return the text of the parameter file that encodes the chain.

        The transformation is consolidated into a copy, whose record is
        written, without the files of its chain.

        Parameters
        ----------
        **kwargs : Any
            `on_loss` and the fields of the metadata, given to
            [`consolidate`][].

        Returns
        -------
        str
            The text of the file, ending with a newline.

        Raises
        ------
        UnrepresentableTransformationError
            If the chain cannot be written as a single elastix transform.
        """
        return _record(self.consolidate(**kwargs)).to_text()

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        return _record(self.consolidate(**kwargs)).to_lines()


@register_format
class ElastixParameterTransform(ElastixTransform):
    """Classic `(Name value ...)` elastix parameter file (`.txt`).

    See [`ElastixTransform`][] for the content and for what is written.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)
    HINTS = ("params",)
    # A `.txt` file holds a bare matrix more often than an elastix map,
    # so an affine whose systems are not known, which every `.txt`
    # format converts exactly, is written as a matrix rather than as an
    # elastix transform (see `io.save`).
    PRIORITY: tx.ClassVar[int] = -1

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        An [`ElastixRaw`][] is read with `from_raw`, and any other value as
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
        Self
            The new transformation.
        """
        if isinstance(other, ElastixRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return ElastixRaw.sniff_text(text, error=error, syntax="text")

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        *,
        initial: _Initial = True,
        encoding: str = "utf-8",
    ) -> tx.Self:
        return _from_filename(cls, "text", filename, initial, encoding)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        return _from_fileobj(cls, file, kwargs)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        initial: _Initial = True,
        origin: tx.Optional[path.FilenameLike] = None,
    ) -> tx.Self:
        """Read a transformation from the text of a file.

        Parameters
        ----------
        text : str
            The text of the file.
        initial : bool, path-like or Transformation, default=True
            The initial transform named in
            `InitialTransformParameterFileName`. With `True`, the initial
            transform is read, together with the initial transforms that
            it names in turn, as transformix does (see
            [`resolve_initial`][brainhops.io.transformations.elastix._raw.resolve_initial]).
            With `False`, only the file's own transform is read. A file
            name replaces the file that the map names, and a
            transformation replaces the initial transform.
        origin : path-like, optional
            The file that the text comes from, against which a relative
            initial file is resolved.

        Returns
        -------
        Self
            The transformation.

        Raises
        ------
        ParserNotImplementedError
            If a transform is not supported, or if it is added to its
            initial transform (`HowToCombineTransforms "Add"`), which a
            chain cannot express.
        """
        return _from_text(cls, "text", text, None, initial, origin)

    @classmethod
    def from_lines(
        cls,
        lines: tx.Iterable[str],
        *,
        initial: _Initial = True,
        origin: tx.Optional[path.FilenameLike] = None,
    ) -> tx.Self:
        return _from_text(cls, "text", None, lines, initial, origin)

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending chain and systems into a new record.

        The rules are those of [`ElastixTomlTransform.consolidate`][],
        with the classic text syntax.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
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
            If the transformation is empty, if the chain cannot be written
            as a single elastix transform, or if a system that was
            assigned is not LPS.
        """
        return _consolidated(self, "text", inplace, on_loss, fields)


@register_format
class ElastixTomlTransform(ElastixTransform):
    """TOML `Name = value` elastix parameter file (`.toml`).

    See [`ElastixTransform`][] for the content and for what is written.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".toml",)
    HINTS = ("toml",)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        An [`ElastixRaw`][] is read with `from_raw`, and any other value as
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
        Self
            The new transformation.
        """
        if isinstance(other, ElastixRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return ElastixRaw.sniff_text(text, error=error, syntax="toml")

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        *,
        initial: _Initial = True,
        encoding: str = "utf-8",
    ) -> tx.Self:
        return _from_filename(cls, "toml", filename, initial, encoding)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        return _from_fileobj(cls, file, kwargs)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        initial: _Initial = True,
        origin: tx.Optional[path.FilenameLike] = None,
    ) -> tx.Self:
        """Same as [`ElastixParameterTransform.from_text`][], in TOML."""
        return _from_text(cls, "toml", text, None, initial, origin)

    @classmethod
    def from_lines(
        cls,
        lines: tx.Iterable[str],
        *,
        initial: _Initial = True,
        origin: tx.Optional[path.FilenameLike] = None,
    ) -> tx.Self:
        return _from_text(cls, "toml", None, lines, initial, origin)

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending chain and systems into a new record.

        Systems that were assigned to `input` or `output` must be the LPS
        system of the transform, since elastix maps LPS to LPS. When no
        chain has been assigned, the record that the metadata holds is
        kept as it is, with the records of its chain, so that it is
        written as it was read. A record in the other syntax, as after a
        conversion from an [`ElastixParameterTransform`][], is copied
        into a record of this syntax with the same parameter map. An
        assigned chain is encoded into a new record, held by new metadata,
        and the chain is then decoded from it.

        The new record holds a single transform, without an initial
        transform. A chain of one ITK block whose class elastix shares
        (translation, Euler, similarity, affine or B-spline) keeps its
        class, its center and its grid. Any other chain that reduces to an
        affine is written as an `AffineTransform` centered on the origin.
        The parameters of the current record that do not describe the
        transform, such as the fixed geometry and the options of the
        resampler, are kept. The option `initial`, which replaced the
        initial transform of the old record, is reset.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an elastix
        file holds no field of the vocabulary yet, so a field that is set
        is reported as lost, once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
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
            If the transformation is empty, if the chain cannot be written
            as a single elastix transform, or if a system that was
            assigned is not LPS.
        """
        return _consolidated(self, "toml", inplace, on_loss, fields)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _from_filename(
    cls: type,
    syntax: str,
    filename: path.FilenameLike,
    initial: _Initial,
    encoding: str,
) -> ElastixTransform:
    """Read a transformation from a file, in the syntax of its class.

    The bytes of the file are read, so that the record keeps the line
    endings of the file.
    """
    if isinstance(initial, _xforms.Transformation):
        given, initial = initial, False
    else:
        given = None
    raw = ElastixRaw.from_filename(
        filename, syntax=syntax, initial=initial, encoding=encoding
    )
    return _checked(cls.from_raw(raw, initial=given))


def _from_fileobj(
    cls: type, file: tx.IO, kwargs: tx.Dict[str, tx.Any]
) -> ElastixTransform:
    """Read a transformation from an open file, in text or binary mode.

    The name of the file, when it has one, is the `origin` of `from_text`.
    """
    kwargs.setdefault("origin", getattr(file, "name", None))
    with preserve_position(file):
        content = file.read()
    if isinstance(content, (bytes, bytearray)):
        return cls.from_bytes(content, **kwargs)
    return cls.from_text(content, **kwargs)


def _from_text(
    cls: type,
    syntax: str,
    text: tx.Optional[str],
    lines: tx.Optional[tx.Iterable[str]],
    initial: _Initial,
    origin: tx.Optional[path.FilenameLike],
) -> ElastixTransform:
    """Read a transformation from text, or from lines when `text` is `None`."""
    if isinstance(initial, _xforms.Transformation):
        given, initial = initial, False
    else:
        given = None
    options = dict(syntax=syntax, initial=initial, origin=origin)
    if text is None:
        raw = ElastixRaw.from_lines(lines, **options)
    else:
        raw = ElastixRaw.from_text(text, **options)
    return _checked(cls.from_raw(raw, initial=given))


def _checked(xform: ElastixTransform) -> ElastixTransform:
    """Decode the blocks of a transformation that was read, and return it.

    The block of each file of the chain is decoded now, so that an
    unsupported transform fails when the file is read, but the chain is
    not, so that reading stays cheap. A file whose transform ignored the
    direction cosines is read with a warning.

    Raises
    ------
    ParserNotImplementedError
        If a transform is not supported, or if it is added to its initial
        transform.
    """
    current = xform
    while isinstance(current, ElastixTransform):
        record = _record(current)
        if not get_bool(record.parameters, "UseDirectionCosines", True):
            warn(
                "This elastix transform was computed with "
                'UseDirectionCosines "false": elastix ignored the '
                "direction cosines of both images, so the transform maps "
                "their direction-less physical spaces, not LPS. It is read "
                "as if it mapped LPS.",
                stacklevel=4,
            )
        current.block  # noqa: B018
        _check_composed(current)
        current = current.initial
    return xform


def _check_composed(xform: ElastixTransform) -> None:
    """Refuse a file whose transform is added to its initial transform.

    Raises
    ------
    ParserNotImplementedError
        If the file has an initial transform, and combines with it by
        another rule than composition.
    """
    record = _record(xform)
    if record is None or xform.initial is None:
        return
    how = get_string(record.parameters, "HowToCombineTransforms", "Compose")
    if how != "Compose":
        raise ParserNotImplementedError(
            f"This elastix transform combines with its initial transform "
            f"by {how!r} (HowToCombineTransforms): T(x) = T1(x) + T0(x) - x. "
            f"A sequence of transformations can only compose them, so it "
            f"cannot be read. Pass initial=False to read the file's own "
            f"transform alone."
        )


def _view(record: ElastixRaw) -> ElastixTransform:
    """Return the transformation that the record of an initial file holds.

    The class is the format of the syntax of the record.
    """
    if record.syntax == "toml":
        return ElastixTomlTransform.from_raw(record)
    return ElastixParameterTransform.from_raw(record)


# ----------------------------------------------------------------------
#   ENCODING
# ----------------------------------------------------------------------


def _consolidated(
    xform: ElastixTransform,
    syntax: str,
    inplace: bool,
    on_loss: tx.Optional[OnLoss],
    fields: tx.Mapping[str, tx.Any],
) -> ElastixTransform:
    """Encode the pending state of a transformation into a new record.

    The rules are described in [`ElastixTomlTransform.consolidate`][].

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation cannot be encoded.
    """
    report = ConversionReport(
        source=type(xform).__name__, target=type(xform).__name__
    )
    requested = requested_fields(xform.metadata, fields)
    metadata = metadata_to_write(
        xform.metadata, ElastixMetadata, fields, report
    )
    apply_loss_policy(report, on_loss, stacklevel=3)
    if not requested:
        metadata = xform.metadata
    record = _record(xform)
    assigned = xform._transformations is not None
    declared = xform._input is not None or xform._output is not None
    if not assigned:
        if record is None:
            raise UnrepresentableTransformationError(
                "This elastix transformation is empty."
            )
        if declared:
            _check_lps(xform, ndim_of(record.parameters))
        if record.syntax == syntax:
            if not declared:
                if not inplace:
                    return replace(xform, metadata=metadata)
                if requested:
                    xform.metadata = metadata
                return xform
        else:
            raw = ElastixRaw(
                parameters=record.parameters,
                syntax=syntax,
                initial=record.initial,
            )
            metadata = replace(metadata, raw=raw)
        options = {}
    else:
        base = None if record is None else record.parameters
        pmap = _encode(xform._transformations, base)
        if declared:
            _check_lps(xform, ndim_of(pmap))
        raw = ElastixRaw(parameters=pmap, syntax=syntax)
        if metadata is None:
            metadata = ElastixMetadata.from_raw(raw)
        else:
            metadata = replace(metadata, raw=raw)
        # The initial transform that was given replaced the initial
        # transform of the old record, which the new record does not have.
        options = {"initial": None}
    if not inplace:
        return replace(
            xform,
            metadata=metadata,
            transformations=None,
            input=None,
            output=None,
            **options,
        )
    xform.metadata = metadata
    xform.transformations = None
    xform.input = None
    xform.output = None
    if options:
        xform.initial = None
    return xform


def _encode(
    chain: tx.Sequence[_xforms.Transformation],
    base: tx.Optional[tx.Dict[str, tx.Any]],
) -> tx.Dict[str, tx.Any]:
    """Return the parameter map that encodes a chain as a single transform.

    Raises
    ------
    UnrepresentableTransformationError
        If the chain is empty, or does not reduce to a single transform.
    """
    chain = list(chain)
    if not chain:
        raise UnrepresentableTransformationError(
            "This elastix transformation is empty."
        )
    if len(chain) == 1:
        return transformation_to_map(chain[0], base)
    sequence = _xforms.Sequence(transformations=chain)
    try:
        return transformation_to_map(sequence, base)
    except UnrepresentableTransformationError as e:
        raise UnrepresentableTransformationError(
            f"An elastix parameter file holds a single transform, and "
            f"this chain of {len(chain)} does not reduce to an affine. "
            f"Write each one to its own file, and chain them with "
            f"InitialTransformParameterFileName."
        ) from e


def _check_lps(xform: ElastixTransform, ndim: int) -> None:
    """Refuse a transformation whose assigned systems are not LPS.

    Raises
    ------
    UnrepresentableTransformationError
        If a system that was assigned to `input` or `output` is not
        compatible with the LPS system of `ndim` dimensions.
    """
    lps = _make_system(ndim)
    for side in ("input", "output"):
        system = getattr(xform, "_" + side)
        if system is None or _compatible(system, lps):
            continue
        raise UnrepresentableTransformationError(
            f"An elastix transform maps LPS millimetres to LPS "
            f"millimetres, but the {side} of this one is "
            f"{type(system).__name__}. Convert it to LPS first."
        )


# ----------------------------------------------------------------------
#   WRITING THE FILES OF A CHAIN
# ----------------------------------------------------------------------


def _place_initial(
    record: ElastixRaw,
    filename: path.FilenameLike,
    written: path.FilenameLike,
) -> None:
    """Write the files of the chain of a record where its file names them.

    `filename` is the file of `record`, and `written` is the file that
    the writer writes last. A file of the chain with a relative name is
    written beside the file that names it, under that name, unless it
    already holds the same bytes. A file with an absolute name is left as
    it is, and so is a chain that the record does not hold.

    Raises
    ------
    WriterError
        If a file of the chain would be written over the file that the
        writer writes, or over a file that holds different bytes.
    """
    name = record.initial_filename
    initial = record.initial
    if initial is None or name is None or is_absolute(name):
        return
    target = path.Path(filename).parent / name
    if is_same_file(target, written) or (
        os.fspath(target) == os.fspath(written)
    ):
        raise WriterError(
            f"The elastix initial transform {name!r} would be written over "
            f"{written}, which names it."
        )
    content = initial.to_bytes()
    if path.exists(target):
        with path.Path(target).open("rb") as file:
            if file.read() != content:
                raise WriterError(
                    f"The elastix initial transform {name!r} must be "
                    f"written to {target}, which holds another file. "
                    f"Write the transformation to another directory, or "
                    f"remove that file."
                )
    else:
        write_file(target, content)
    _place_initial(initial, target, written)
