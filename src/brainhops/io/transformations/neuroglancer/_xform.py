"""Affine transformations stored as Neuroglancer coordinate transforms."""

__all__ = ["NeuroglancerTransform"]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core.properties import InvalidatorInAttribute, smartproperty
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata._report import OnLoss
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    ParserContentError,
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.metadata._base import requested_fields
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)

# this format
from ._dimensions import (
    decode_matrix,
    decode_system,
    encode_dimensions,
    encode_matrix,
)
from ._metadata import NeuroglancerTransformMetadata
from ._raw import NeuroglancerTransformRaw

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


@register_format
class NeuroglancerTransform(
    TextFileReader,
    TextFileWriter,
    AffineTransformationFormat,
    _xforms.Affine,
    TransformationFormat,
):
    """Affine transformation stored as a Neuroglancer coordinate transform.

    Neuroglancer places each data source of a layer with an affine
    coordinate transform, written in the state of the viewer as the
    `transform` of the layer `source`. This class reads and writes that
    JSON object on its own, so that an affine computed by brainhops can
    be pasted into the state of a viewer, or read back from it.
    Neuroglancer has no file for a transform, so the class claims no
    generic `.json` file: it reads a `.json` file whose keys are those of
    a transform (see [`NeuroglancerTransformRaw.sniff_text`][]), any file
    with the hint `"neuroglancer.affine"`, and the files whose names end
    in `.ngtransform.json`, a convention of brainhops. `io.save` writes
    this format to a name with that extension, or to any name with the
    hint, as in `io.save(affine, "transform.json",
    hint="neuroglancer.affine")`. The hint `"neuroglancer"` names the
    Neuroglancer precomputed volumes.

    The JSON is short and is read in one pass, so the whole object is
    held as a [`NeuroglancerTransformRaw`][] record by the
    [`NeuroglancerTransformMetadata`][] of the transformation, and the
    matrix lives in that record, as for an LTA file. The data and the
    coordinate systems of the transformation are decoded from it. The
    record describes the file as it was read, or as the transformation
    was last consolidated. A matrix that is assigned, and an `input` or
    `output` that is set, are pending until [`consolidate`][] encodes them
    into a new record.

    !!! note "Coordinate systems and units"
        The matrix of the file maps physical input coordinates to
        physical output coordinates, and its translation is a number of
        output scales. The input and output of the transformation have
        one axis for each dimension, named after it, whose unit is the
        unit of the dimension with the prefix that Neuroglancer displays
        for its scale, so `[4e-09, "m"]` is an axis in nanometers. The
        decoded matrix maps coordinates in those units. A transform
        without `inputDimensions` has an input whose axes have no name
        and the units of its output. The package documentation gives the
        conventions and their sources in Neuroglancer.

    !!! note "What is written"
        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is written as the text of that record.
        Otherwise the matrix is encoded with the dimensions of the record,
        and a system that is set, or the system of a transformation
        without a record, is encoded as dimensions: each axis must have a
        name that Neuroglancer accepts and a unit of length, of time, of
        frequency or of angular velocity whose size is a power of ten,
        such as `millimeter`, which is written as `[0.001, "m"]`, or no
        unit. A voxel axis is refused, because Neuroglancer places
        voxels as its data source does, not as the transform says.
        Anything else raises [`UnrepresentableTransformationError`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ngtransform.json",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("neuroglancer.affine",)

    _metadata: NoRepr[tx.Optional[NeuroglancerTransformMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the JSON object as a record.

    A transformation built from data alone has no metadata, and setting
    its data does not create any. Assigning other metadata drops the
    matrix decoded from the old record. A matrix that was assigned is
    pending and is kept, and it wins over the matrix of the new record
    until the transformation is consolidated.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The input coordinate system.

        Unless the system was set explicitly, it is decoded from the
        `inputDimensions` of the record, or from its `outputDimensions`
        without names when the record has no input dimensions. Without a
        record, it is `None`.
        """
        record = _record(self)
        if record is None:
            return None
        return decode_system(record.content, "inputDimensions")

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The output coordinate system.

        Unless the system was set explicitly, it is decoded from the
        `outputDimensions` of the record. Without a record, it is `None`.
        """
        record = _record(self)
        if record is None:
            return None
        return decode_system(record.content, "outputDimensions")

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(rank, rank + 1)` affine matrix, which `matrix` reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record. Otherwise the
        matrix is decoded from the record, in the units of the input and
        output axes, and is read-only, or `None` without a record.
        """
        record = _record(self)
        if record is None:
            return None
        matrix = decode_matrix(record.content)
        matrix.flags.writeable = False
        return matrix

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score text as a Neuroglancer coordinate transform.

        The text is scored by [`NeuroglancerTransformRaw.sniff_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not a transform.
        **kwargs : Any
            Options of the reader, which do not change the score.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return NeuroglancerTransformRaw.sniff_text(text, error=error)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: NeuroglancerTransformRaw, **kwargs) -> tx.Self:
        """Build a transformation from a [`NeuroglancerTransformRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : NeuroglancerTransformRaw
            The record of the transform.
        **kwargs : Any
            Field values, such as `input`, `output` or `matrix`, which go
            to the constructor and override the record.

        Returns
        -------
        Self
            The transformation.
        """
        metadata = NeuroglancerTransformMetadata.from_raw(raw)
        return cls(metadata=metadata, **kwargs)

    @classmethod
    def from_text(cls, text: str) -> tx.Self:
        """Build a transformation from the JSON text of a transform.

        The text is read by [`NeuroglancerTransformMetadata.from_text`][].

        Parameters
        ----------
        text : str
            The JSON text of the transform.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=NeuroglancerTransformMetadata.from_text(text))

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str]) -> tx.Self:
        """Build a transformation from the lines of the JSON text.

        Parameters
        ----------
        lines : iterable of str
            The lines, without line endings.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=NeuroglancerTransformMetadata.from_lines(lines))

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`NeuroglancerTransformRaw`][] is read with [`from_raw`][], and
        any other value as the bases read it.

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
        if isinstance(other, NeuroglancerTransformRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise a new record is built from a copy of
        the content of the current record, which keeps the keys that
        Neuroglancer does not read, and keeps `sourceRank` unless the rank
        changes. A system that is set,
        or both systems of a transformation without a record, are encoded
        as dimensions, and the matrix is encoded with the dimensions of
        the new record. The new record is held by new metadata, and the
        matrix and the systems are then decoded from it.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with metadata that the format cannot store. By
            default, the policy in effect.
        **fields : Any
            Metadata fields to override before encoding the transform.

        Returns
        -------
        Self
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        UnrepresentableTransformationError
            If a transform cannot encode the systems or the matrix.
        """
        metadata = self.metadata
        requested = requested_fields(metadata, fields)
        if metadata is not None:
            metadata = metadata.consolidate(on_loss=on_loss, **fields)
            if not requested:
                metadata = self.metadata
        elif fields:
            metadata = NeuroglancerTransformMetadata().consolidate(
                on_loss=on_loss, **fields
            )
        record = _record(self)
        overridden = (
            self._data is not None
            or self._input is not None
            or self._output is not None
        )
        if record is not None and not overridden:
            if inplace:
                self.metadata = metadata
                return self
            return replace(self, metadata=metadata)
        record = _encoded(self)
        if metadata is None:
            metadata = NeuroglancerTransformMetadata.from_raw(record)
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

    def to_text(self, **kwargs) -> str:
        """Return the JSON text of the transform.

        The transformation is consolidated into a copy, whose record is
        written, and the transformation itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            Options of [`consolidate`][].

        Returns
        -------
        str
            The JSON text.

        Raises
        ------
        UnrepresentableTransformationError
            If a transform cannot encode the systems or the matrix.
        """
        return _record(self.consolidate(**kwargs)).to_text()

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the JSON text of the transform.

        Parameters
        ----------
        **kwargs : Any
            Options of [`consolidate`][].

        Yields
        ------
        str
            The lines, without line endings.

        Raises
        ------
        UnrepresentableTransformationError
            If a transform cannot encode the systems or the matrix.
        """
        return iter(self.to_text(**kwargs).splitlines())


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _record(
    xform: NeuroglancerTransform,
) -> tx.Optional[NeuroglancerTransformRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _encoded(xform: NeuroglancerTransform) -> NeuroglancerTransformRaw:
    """Encode the matrix and the systems of a transformation as a record.

    See [`NeuroglancerTransform.consolidate`][].

    Raises
    ------
    UnrepresentableTransformationError
        If a transform cannot encode the systems or the matrix.
    """
    matrix = np.asarray(xform.matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != matrix.shape[0] + 1:
        raise UnrepresentableTransformationError(
            f"A Neuroglancer coordinate transform has as many input as "
            f"output dimensions, so its matrix has shape (rank, rank + 1), "
            f"but the matrix has shape {matrix.shape}."
        )
    rank = matrix.shape[0]
    old = _record(xform)
    if old is None:
        content = {"matrix": None}
        set_output = set_input = True
    else:
        content = dict(old.content)
        set_output = xform._output is not None
        set_input = xform._input is not None
        if old.rank != rank and not (set_output and set_input):
            raise UnrepresentableTransformationError(
                f"The matrix has rank {rank}, but the transform has "
                f"{old.rank} dimensions. Set the input and output of the "
                f"transformation to give the new dimensions."
            )
        if (
            set_output
            and not set_input
            and "inputDimensions" not in old.content
        ):
            # The input took the units of the old output, which it keeps.
            content["inputDimensions"] = old.content["outputDimensions"]
    if set_output:
        content["outputDimensions"] = encode_dimensions(
            xform.output, "outputDimensions", rank
        )
    if set_input:
        content["inputDimensions"] = encode_dimensions(
            xform.input, "inputDimensions", rank
        )
    if old is not None and old.rank != rank:
        # The rank of the source described the old dimensions.
        content.pop("sourceRank", None)
    content["matrix"] = encode_matrix(matrix, content)
    try:
        return NeuroglancerTransformRaw(content=content)
    except ParserContentError as error:
        raise UnrepresentableTransformationError(str(error)) from error
