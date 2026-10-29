"""The affine of one patch of a TrakEM2 project."""

__all__ = ["TrakEM2PatchAffine"]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core._internal_api import (
    ArrayProtocol,
    InvalidatorInAttribute,
    smartproperty,
)
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    ParserContentError,
    UnrepresentableTransformationError,
)
from brainhops.io.metadata._internal_api import (
    metadata_options,
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import TransformationFormat

# this format
from ._metadata import TrakEM2Metadata
from ._raw import TrakEM2Raw

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""

PATCH = "patch"
"""The name of the pixel system of a patch."""

LAYER = "layer"
"""The name of the plane of a layer, in which the patches are placed."""

_AFFINE = "mpicbg.trakem2.transform.AffineModel2D"
"""The Java class of the affine model of a coordinate transform."""

_SIMILARITY = "mpicbg.trakem2.transform.SimilarityModel2D"
"""The Java class of the similarity model of a coordinate transform."""


# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------
# The XML file is parsed in one pass, so the six values of the affine of
# each patch live in the record, as the matrix of an LTA file does. The
# data of the transformation is decoded from the patch that the option
# `patch` selects, and an assigned matrix is pending until it is encoded
# into a new record, in which only the `transform` attribute of that
# patch is written again.


def _record(xform: "TrakEM2PatchAffine") -> tx.Optional[TrakEM2Raw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def matrix_to_model(values: tx.Sequence[float]) -> np.ndarray:
    """Return the `(2, 3)` matrix of the six values of a `transform`.

    The values are those of `java.awt.geom.AffineTransform.getMatrix`, in
    the order `m00 m10 m01 m11 m02 m12`, so they are the columns of the
    matrix one after the other. The matrix is a new array, made
    read-only, because the writer reads the record and not the matrix,
    so an edit in place would be lost.
    """
    matrix = np.array(values, dtype=np.float64).reshape(3, 2).T.copy()
    matrix.flags.writeable = False
    return matrix


def matrix_to_disk(data: ArrayProtocol) -> tx.Tuple[float, ...]:
    """Return the six values that a `transform` stores for a matrix.

    The matrix is a `(2, 3)` matrix, or a `(3, 3)` homogeneous one.

    Raises
    ------
    UnrepresentableTransformationError
        If the matrix is not the matrix of a 2D affine.
    """
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.shape == (3, 3) and np.array_equal(matrix[2], [0, 0, 1]):
        matrix = matrix[:2]
    if matrix.shape != (2, 3):
        raise UnrepresentableTransformationError(
            f"A TrakEM2 patch is placed by a 2D affine, of shape (2, 3), "
            f"and this matrix has shape {matrix.shape}."
        )
    return tuple(float(v) for v in matrix.T.reshape(-1))


# ----------------------------------------------------------------------
#   FORMAT
# ----------------------------------------------------------------------


@register_format
class TrakEM2PatchAffine(
    BinaryFileReader,
    BinaryFileWriter,
    TransformationFormat,
    _xforms.Affine,
):
    """The affine that places one patch of a TrakEM2 project in its layer.

    TrakEM2 projects are made of layers, one for each section of a
    series, and each layer holds patches, which are images placed in the
    plane of the layer by a 2D affine (see
    [`brainhops.io.transformations.trakem2`][]). This class reads the
    affine of one patch, which the reader option `patch` selects by its
    `oid` or its title, by default the first patch of the file.

    The transformation is a 2D
    [`Affine`][brainhops.datamodel.transformations.Affine] of shape
    `(2, 3)`. Its input is the pixel system of the patch, with axes x
    and y and integers at the centers of the pixels, and its output is
    the plane of the layer, in pixels of the layer set, which has no
    unit. When the patch also has a coordinate transform, the affine is
    applied after it, so its input is not the pixel system of the image,
    and `input` is `None` (see [`coordinate_transform`][]). The z of the
    layer and the calibration are not part of the transformation; they
    are kept by the record, in
    [`TrakEM2Raw.Patch`][brainhops.io.transformations.trakem2.TrakEM2Raw.Patch].

    The whole XML file is held as a [`TrakEM2Raw`][] record by the
    [`TrakEM2Metadata`][] of the transformation. A matrix that is
    assigned, and an `input` or `output` that is set, are pending until
    [`consolidate`][] encodes them into a new record, or until the
    transformation is written.

    !!! note "What is written"
        A transformation whose matrix and systems were not assigned is
        written back as the record holds the file, byte for byte. A
        transformation whose matrix was assigned is written as the same
        file, in which only the value of the `transform` attribute of its
        patch is written again. A transformation without a record has no
        project to be written into, and is refused, since a project
        cannot be made from an affine alone.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xml",)
    HINTS = ("trakem2",)

    _metadata: NoRepr[tx.Optional[TrakEM2Metadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the project, which holds its XML file as a record.

    A transformation built from data alone has no metadata. Assigning
    other metadata drops the matrix decoded from the old record, and a
    matrix that was assigned is kept.
    """

    _patch: tx.Optional[tx.Union[int, str]] = None

    patch = smartproperty("patch", invalidates=("data", _FORGET_VIEWS))
    """The patch whose affine is read, by its `oid` or its title.

    It is the reader option `patch`, and a reader stores the `oid` of the
    patch that it read. `None` stands for the first patch of the file.
    Assigning another patch drops the matrix decoded from the record. The
    option only selects what is read: the file is written with every
    patch.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The pixel system of the patch, unless set explicitly.

        It is `None` when the patch has a coordinate transform, since the
        affine then maps the image after that transform.
        """
        record = _record(self)
        if record is not None and _has_coordinate_transform(
            record.patch(self.patch)
        ):
            return None
        return patch_system()

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The plane of the layer, unless set explicitly."""
        return layer_system()

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(2, 3)` affine matrix, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record. Otherwise the
        matrix is read from the `transform` attribute of the patch, and is
        read-only, or `None` without a record.
        """
        record = _record(self)
        if record is None:
            return None
        return matrix_to_model(record.patch(self.patch).transform)

    # --- coordinate transform -----------------------------------------

    def coordinate_transform(self) -> tx.Optional[_xforms.Affine]:
        """Decode the coordinate transform of the patch, if it has one.

        A patch may carry a coordinate transform, which TrakEM2 applies
        to the pixels of the image before the affine of the patch. Only
        an affine model and a similarity model written inside the patch
        are decoded. The affine of the patch is not composed with it,
        because TrakEM2 shifts the image between the two by the corner of
        the bounding box of a mesh (`Patch.getFullCoordinateTransform` in
        `ini/trakem2/display/Patch.java`, TrakEM2 commit cdd8b8f), which
        brainhops does not compute.

        Returns
        -------
        Affine or None
            A 2D affine whose input is the pixel system of the patch, or
            `None` when the patch has no coordinate transform.

        Raises
        ------
        ParserContentError
            If the transformation has no record, or if the coordinate
            transform is stored in a separate file, is a list, or is
            another model.
        """
        record = _record(self)
        if record is None:
            raise ParserContentError(
                "This TrakEM2 transformation has no record."
            )
        patch = record.patch(self.patch)
        item = patch.coordinate_transform
        if item is None:
            if patch.ct_id is None:
                return None
            raise ParserContentError(
                f"The coordinate transform of patch {patch.oid} is stored in "
                f"a separate file (ct_id {patch.ct_id}), which brainhops does "
                f"not read."
            )
        return _decode_coordinate_transform(item, patch.oid)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a stream from the document type of its XML.

        Parameters
        ----------
        file : IO
            The binary stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a TrakEM2
            project.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return TrakEM2Metadata.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes from the document type of their XML.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not a TrakEM2
            project.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return TrakEM2Metadata.sniff_bytes(content, error=error)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(
        cls,
        file: tx.BinaryIO,
        patch: tx.Optional[tx.Union[int, str]] = None,
        **kwargs,
    ) -> tx.Self:
        """Read the affine of a patch from a binary stream.

        The record is read by [`TrakEM2Metadata.from_fileobj`][], and the
        matrix is decoded from it when it is first used.

        Parameters
        ----------
        file : BinaryIO
            The stream of the XML file.
        patch : int or str, optional
            The `oid` of the patch, or its title. By default, the first
            patch of the file.
        **kwargs : Any
            Other options, which are refused.

        Returns
        -------
        Self
            The transformation.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ParserContentError
            If the content is not a TrakEM2 project, or if it has no such
            patch.
        """
        metadata = TrakEM2Metadata.from_fileobj(file, **kwargs)
        return cls.from_raw(metadata.raw, patch=patch)

    @classmethod
    def from_raw(cls, raw: TrakEM2Raw, **kwargs) -> tx.Self:
        """Build a transformation from an already read [`TrakEM2Raw`][].

        The record is held by new metadata, without a copy, and the patch
        is resolved to its `oid`.

        Parameters
        ----------
        raw : TrakEM2Raw
            The record of the XML file.
        **kwargs : Any
            Field values, such as `patch`, which go to the constructor.

        Returns
        -------
        Self
            The transformation.

        Raises
        ------
        ParserContentError
            If the record has no such patch.
        """
        oid = raw.patch(kwargs.pop("patch", None)).oid
        return cls(metadata=TrakEM2Metadata.from_raw(raw), patch=oid, **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`TrakEM2Raw`][] is read with [`from_raw`][], and any other
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
        if isinstance(other, TrakEM2Raw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        If neither the matrix nor the `input` and `output` systems were
        assigned, the transformation is left as it is, so that the file
        is written back as it was read. The option `patch` only selects
        what is decoded, so it does not change the record. Otherwise, the
        new record is the current one, in which the value of the
        `transform` attribute of the patch is written again from the
        matrix, as described in [`TrakEM2PatchAffine`][]. Systems that
        were assigned must be compatible with the pixel system of a patch
        and with the plane of a layer, since the file stores no system,
        and a patch that has a coordinate transform takes no input
        system. The new record is held by new metadata, and the matrix is
        then decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a
        TrakEM2 project holds no field of the vocabulary yet, so a field
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
            If the transformation has no record, if the matrix is not a
            2D affine, or if an assigned system is not compatible with the
            systems of the file.
        WriterError
            If the record cannot be changed in place.
        """
        report = ConversionReport(
            source="TrakEM2PatchAffine", target="TrakEM2PatchAffine"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, TrakEM2Metadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        record = _record(self)
        overridden = (
            self._data is not None
            or self._input is not None
            or self._output is not None
        )
        if record is not None and not overridden:
            if not inplace:
                return replace(self, metadata=metadata)
            if requested:
                self.metadata = metadata
            return self
        if record is None:
            raise UnrepresentableTransformationError(
                "A TrakEM2 affine is written into the project that it was "
                "read from, and this one has no project. Load the project, "
                "assign the matrix to the affine of one of its patches, and "
                "save it."
            )
        patch = record.patch(self.patch)
        _check_systems(self, patch)
        record = record.with_transform(patch.oid, matrix_to_disk(self.matrix))
        metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self,
                metadata=metadata,
                data=None,
                input=None,
                output=None,
                patch=patch.oid,
            )
        self.metadata = metadata
        self.data = None
        self.input = None
        self.output = None
        self.patch = patch.oid
        return self

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the XML file that encodes the transformation.

        The transformation is consolidated into a copy, whose record is
        written, and the transformation itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer, which are refused. The fields of the
            metadata are given to [`consolidate`][].

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        TypeError
            If an option is given.
        UnrepresentableTransformationError
            If the matrix or the systems cannot be encoded, or if the
            transformation has no record.
        """
        fields, kwargs = metadata_options(kwargs)
        return _record(self.consolidate(**fields)).to_bytes(**kwargs)


# ----------------------------------------------------------------------
#   SYSTEMS
# ----------------------------------------------------------------------


def patch_system() -> _systems.CoordinateSystem:
    """Return the pixel system of a patch, with axes x and y.

    Integers are at the centers of the pixels, as TrakEM2 and mpicbg
    count them (`Patch.paint` in `ini/trakem2/display/Patch.java`,
    TrakEM2 commit cdd8b8f, which shifts the image by half a pixel for
    AWT, whose integers are at the corners).
    """
    axes = [Axis(name, "space", unit="index") for name in ("x", "y")]
    return _systems.ArrayCoordinateSystem(name=PATCH, axes=axes, order="F")


def layer_system() -> _systems.CoordinateSystem:
    """Return the plane of a layer, whose axes have no unit."""
    axes = [Axis(name, "space") for name in ("x", "y")]
    return _systems.CoordinateSystem(name=LAYER, axes=axes)


def _check_systems(xform: TrakEM2PatchAffine, patch: TrakEM2Raw.Patch) -> None:
    """Refuse assigned systems that the file cannot express.

    Raises
    ------
    UnrepresentableTransformationError
        If the input is not compatible with the pixel system of a patch,
        or the output with the plane of a layer, or if an input is
        assigned to a patch that has a coordinate transform.
    """
    if xform._input is not None and _has_coordinate_transform(patch):
        raise UnrepresentableTransformationError(
            f"Patch {patch.oid} has a coordinate transform, so its affine "
            f"maps the image after that transform, and the file cannot say "
            f"that its input is {xform._input!r}."
        )
    for name, system, expected in (
        ("input", xform._input, patch_system()),
        ("output", xform._output, layer_system()),
    ):
        if system is not None and not expected.compatible_with(system):
            raise UnrepresentableTransformationError(
                f"A TrakEM2 affine maps the pixels of a patch, with axes x "
                f"and y, to the plane of its layer, and the {name} of this "
                f"affine is {system!r}."
            )


def _has_coordinate_transform(patch: TrakEM2Raw.Patch) -> bool:
    """Tell whether a patch has a coordinate transform.

    TrakEM2 gives a patch a coordinate transform when its `ct_id` is not
    0 (`Patch.hasCoordinateTransform`), and reads one written inside the
    patch first (`TMLHandler.endElement`).
    """
    return patch.coordinate_transform is not None or bool(patch.ct_id)


def _decode_coordinate_transform(
    item: TrakEM2Raw.CoordinateTransform, oid: int
) -> _xforms.Affine:
    """Decode an affine or a similarity model into a 2D affine.

    The models read their `data` attribute in `init`
    (`mpicbg/trakem2/transform/AffineModel2D.java` and
    `SimilarityModel2D.java` in trakem2-transform, commit 5a92a82). The
    affine model reads six doubles `m00 m10 m01 m11 m02 m12`. The
    similarity model reads four floats `scos ssin tx ty`, in single
    precision, and maps `(x, y)` to `(scos x - ssin y + tx, ssin x +
    scos y + ty)` (`SimilarityModel2D.applyInPlace` in mpicbg, commit
    fc319fc).

    Raises
    ------
    ParserContentError
        If the transform is a list or another model, or if its values
        cannot be read.
    """
    if item.element.endswith("_list") or item.cls not in (
        _AFFINE,
        _SIMILARITY,
    ):
        kind = item.element if item.cls is None else item.cls
        raise ParserContentError(
            f"The coordinate transform of patch {oid} is a {kind}, and "
            f"brainhops decodes only {_AFFINE} and {_SIMILARITY}."
        )
    fields = (item.data or "").split()
    try:
        if item.cls == _AFFINE and len(fields) == 6:
            matrix = matrix_to_model([float(v) for v in fields])
        elif item.cls == _SIMILARITY and len(fields) == 4:
            scos, ssin, tx_, ty_ = (float(np.float32(v)) for v in fields)
            matrix = np.array([[scos, -ssin, tx_], [ssin, scos, ty_]])
            matrix.flags.writeable = False
        else:
            raise ValueError(f"{len(fields)} values")
    except ValueError as error:
        raise ParserContentError(
            f"The coordinate transform of patch {oid} cannot be read: "
            f"{item.data!r} ({error})."
        ) from error
    return _xforms.Affine(matrix=matrix, input=patch_system(), output=None)
