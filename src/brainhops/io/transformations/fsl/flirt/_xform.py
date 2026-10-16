"""The affine stored in an FSL FLIRT `.mat` file."""

__all__ = ["FlirtTransform"]

import numpy as np
import typing_extensions as tx
from bagof.magic import (
    Alias,
    ConvertTo,
    Forget,
    Inherit,
    KwOnly,
    NoRepr,
    NotKwOnly,
    replace,
)

from brainhops._core.properties import InvalidatorInAttribute, smartproperty
from brainhops._core.typing import ArrayLike
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.images import Image
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.errors import IncompatibleSystemError
from brainhops.io.base.parsers import (
    TextFileReader,
    UnrepresentableTransformationError,
)
from brainhops.io.common.nifti._header import _NiftiObject
from brainhops.io.metadata._base import metadata_to_write, requested_fields
from brainhops.io.transformations.base import TransformationFormat
from brainhops.io.transformations.base.affines import _RAS, _check_endpoint

from .._affines import _ImageGeometry
from . import FlirtTransformStub
from ._metadata import FlirtMetadata
from ._raw import FlirtRaw

# A nibabel image or header, or a brainhops image.
_ImageLike = tx.Union[_NiftiObject, Image]

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


def _record(xform: "FlirtTransform") -> tx.Optional[FlirtRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


class FlirtTransform(
    TextFileReader,
    FlirtTransformStub,
    _xforms.Affine,
    TransformationFormat,
):
    """Linear transformation stored in an FSL FLIRT `.mat` file.

    The FLIRT matrix maps the scaled millimetre coordinates of the moving image
    to those of the reference image. This class exposes it as an affine whose
    `matrix` maps reference world (RAS) coordinates to moving world (RAS)
    coordinates, which is the direction the data model uses to resample a
    moving image onto a reference.

    Because the file carries no image geometry, the reference and moving
    images must be supplied. They can be passed as the `reference=` and
    `moving=` keyword arguments when the file is loaded, or set on the object
    before its `matrix` is read. They are options of the transformation and
    are never part of its record.

    The file is held as a [`FlirtRaw`][] record by the [`FlirtMetadata`][]
    of the transformation, and the FLIRT matrix is read from it. The FLIRT
    matrix is the data of the transformation: a FLIRT matrix that is
    assigned, or given to the constructor, is pending until
    [`consolidate`][] encodes it into a new record. The affine in world
    coordinates is computed from the FLIRT matrix and the two images, so
    it cannot be assigned. brainhops does not write FLIRT files, so the
    record is only held in memory.
    """

    # The data field is the FLIRT matrix. The `matrix` view is derived
    # from the FLIRT matrix and the two images, so computing it is costly
    # and may raise. Naming the FLIRT matrix here keeps the identity check
    # of `inverse()` from computing the affine.
    data_fields: tx.ClassVar[tx.Tuple[str, ...]] = ("flirt_matrix",)

    _input: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()

    # The affine is computed from the FLIRT matrix, so it is not a
    # constructor field. Forgetting the fields `_data` and `_matrix`
    # inherited from `Affine` keeps `data` and `matrix` out of `__init__`
    # and `fields()`. The FLIRT matrix takes the place of `data`.
    _data: Forget
    _matrix: Forget

    _flirt_matrix: NotKwOnly[tx.Optional[ArrayLike]] = None

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def flirt_matrix(self) -> tx.Optional[np.ndarray]:
        """The FLIRT matrix, from moving to reference scaled millimetres.

        A FLIRT matrix that is assigned is returned as it is, and stays
        pending until [`consolidate`][] encodes it in a new record.
        Otherwise the matrix is the `(4, 4)` matrix of the record, which
        is read-only, or `None` without a record.
        """
        record = _record(self)
        return None if record is None else _read_only(record.matrix)

    _metadata: NoRepr[tx.Optional[FlirtMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty(
        "metadata", invalidates=("flirt_matrix", _FORGET_VIEWS)
    )
    """The metadata of the file, which holds the FLIRT matrix as a record.

    A transformation built from a FLIRT matrix has no metadata. Assigning
    other metadata drops the FLIRT matrix decoded from the old record. A
    FLIRT matrix that was assigned is pending and is kept, and it wins over
    the matrix of the new record until the transformation is consolidated.
    """

    # The images are kept as they are given, without a conversion to their
    # declared type, so that any object that exposes a geometry is
    # accepted, as it was by the reader before the record design.
    _moving: KwOnly[
        tx.Annotated[
            tx.Optional[_ImageLike],
            Alias(("moving", "mov", "src")),
            ConvertTo(False),
        ]
    ] = None

    moving = smartproperty("moving", invalidates=_FORGET_VIEWS)
    """The moving (source) image, or `None`.

    It is an option of the reader, since the file does not store it, and
    it is not part of the record. It is given as `moving=`, `mov=` or
    `src=`.
    """

    _reference: KwOnly[
        tx.Annotated[
            tx.Optional[_ImageLike],
            Alias(("reference", "ref")),
            ConvertTo(False),
        ]
    ] = None

    reference = smartproperty("reference", invalidates=_FORGET_VIEWS)
    """The reference image, or `None`.

    It is an option of the reader, since the file does not store it, and
    it is not part of the record. It is given as `reference=` or `ref=`.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        _check_endpoint(self, "input", _RAS)
        _check_endpoint(self, "output", _RAS)

    @property
    def data(self) -> tx.Optional[np.ndarray]:
        """The affine from reference RAS to moving RAS, as a (3, 4) array.

        The affine is computed from the FLIRT matrix and both image
        geometries, and is `None` when there is no FLIRT matrix. It cannot
        be assigned.

        Raises
        ------
        ValueError
            If there is a FLIRT matrix but an image is missing.
        """
        flirt = self.flirt_matrix
        if flirt is None:
            return None
        if self.reference is None or self.moving is None:
            raise ValueError(
                "A FLIRT matrix carries no image geometry, so both the "
                "reference and the moving image are needed to place it in "
                "world coordinates. Pass reference=... and moving=... when "
                "loading the matrix, or set them on the transformation."
            )
        ref = _ImageGeometry(self.reference)
        mov = _ImageGeometry(self.moving)
        flirt = np.asarray(flirt, dtype=np.float64)
        full = mov.fsl2ras @ np.linalg.inv(flirt) @ ref.ras2fsl
        return full[:-1]

    @data.setter
    def data(self, value: None) -> None:
        if value is not None:
            raise ValueError(
                "The FLIRT affine is computed from the FLIRT matrix and the "
                "images, so it cannot be set. Set flirt_matrix instead."
            )

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file as those of a FLIRT file.

        The lines are scored by [`FlirtMetadata.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines do not hold a FLIRT
            matrix.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return FlirtMetadata.sniff_lines(lines, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Read a transformation from the lines of a FLIRT file.

        The lines are read by [`FlirtMetadata.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Field values, such as the images `moving` and `reference`,
            under any of their names, which go to the constructor. Any
            other option is refused.

        Returns
        -------
        FlirtTransform
            The transformation.
        """
        return cls(metadata=FlirtMetadata.from_lines(lines), **kwargs)

    @classmethod
    def from_raw(cls, raw: FlirtRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed [`FlirtRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : FlirtRaw
            The record of the file.
        **kwargs : Any
            Field values, such as the images `moving` and `reference`,
            which go to the constructor.

        Returns
        -------
        FlirtTransform
            The transformation.
        """
        return cls(metadata=FlirtMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`FlirtRaw`][] is read with [`from_raw`][], and any other value
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
        FlirtTransform
            The new transformation.
        """
        if isinstance(other, FlirtRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy a transformation, sharing the record of a FLIRT transform.

        Within the format, the stored fields are copied as they are: the
        metadata, which holds the FLIRT matrix of the file, a pending FLIRT
        matrix, the images and the systems. Nothing is decoded, and the
        affine in world coordinates, which needs both images, is never
        computed. Any other transformation is converted as `other.to(cls)`
        converts it, which a FLIRT transform refuses, since a conversion is
        not given the images.

        Parameters
        ----------
        other : object
            The instance to copy or convert.
        *args : Any
            Constructor arguments, which take precedence over the fields
            of `other`. A transformation that is converted takes keyword
            options only.
        **kwargs : Any
            Constructor arguments, which take precedence over the fields
            of `other`, or options of the conversion.

        Returns
        -------
        FlirtTransform
            The copy.

        Raises
        ------
        ConversionError
            If the format cannot hold a transformation that is converted.
        """
        if isinstance(other, FlirtTransform) and not args:
            for name in (
                "flirt_matrix",
                "metadata",
                "moving",
                "reference",
                "input",
                "output",
            ):
                kwargs.setdefault(name, getattr(other, "_" + name))
            return cls(**kwargs)
        return super().from_instance(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending FLIRT matrix into a new record.

        A transformation whose FLIRT matrix was not assigned is left as it
        is, since its record holds its FLIRT matrix. The images `moving`
        and `reference` are not stored in the file, so setting them
        changes no record. The `input` and `output` of a FLIRT transform
        are always RAS, which the file implies and does not store, so
        another system is refused. Otherwise, a new record holds the
        assigned FLIRT matrix, which must be a `(4, 4)` matrix. The new
        record is held by new metadata, and the FLIRT matrix is then
        decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a FLIRT
        matrix holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, by `on_loss`.

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
        FlirtTransform
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If the FLIRT matrix is not a `(4, 4)` matrix, or if the
            `input` or `output` is not RAS.
        """
        report = ConversionReport(
            source="FlirtTransform", target="FlirtTransform"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, FlirtMetadata, fields, report
        )
        apply_loss_policy(report, on_loss, stacklevel=2)
        if not requested:
            metadata = self.metadata
        for end in ("input", "output"):
            try:
                _check_endpoint(self, end, _RAS)
            except IncompatibleSystemError as e:
                raise UnrepresentableTransformationError(str(e)) from None
        if self._flirt_matrix is None:
            if not inplace:
                return replace(self, metadata=metadata)
            if requested:
                self.metadata = metadata
            return self
        record = FlirtRaw(matrix=_matrix_to_disk(self._flirt_matrix))
        if metadata is None:
            metadata = FlirtMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(self, metadata=metadata, flirt_matrix=None)
        self.metadata = metadata
        self.flirt_matrix = None
        return self

    def inverse(self, compute: bool = False, **kwargs) -> _xforms.Affine:
        # The inverse is a plain affine, because the FLIRT matrix and the
        # two images of a FLIRT transform have no meaning after inversion.
        return _xforms.Affine(
            matrix=self.matrix, input=self.input, output=self.output
        ).inverse(compute=compute, **kwargs)

    def __repr__(self) -> str:
        # The repr reads the stored fields, so that it decodes nothing
        # into the cache and never computes the affine, which may raise.
        parts = []
        flirt = self._flirt_matrix
        if flirt is None and _record(self) is not None:
            flirt = np.asarray(_record(self).matrix)
        for name, value in (
            ("flirt_matrix", flirt),
            ("moving", self.moving),
            ("reference", self.reference),
        ):
            if value is not None:
                parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _read_only(matrix: tx.Any) -> np.ndarray:
    """Return a matrix decoded from a record, made read-only.

    The matrix is cached, but a record is never changed in place, so an
    edit in place would be lost. To change the matrix, a new matrix is
    assigned instead, which stays pending until the transformation is
    consolidated.
    """
    matrix = np.array(matrix, dtype=np.float64)
    matrix.flags.writeable = False
    return matrix


def _matrix_to_disk(matrix: ArrayLike) -> tx.Tuple[tx.Tuple[float, ...], ...]:
    """Return the matrix that a record stores for a FLIRT matrix.

    Raises
    ------
    UnrepresentableTransformationError
        If the matrix is not a `(4, 4)` matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise UnrepresentableTransformationError(
            f"A FLIRT file holds a (4, 4) matrix, but this one has shape "
            f"{matrix.shape}."
        )
    return tuple(tuple(float(v) for v in row) for row in matrix)


# Importing the private converters registers them with the data model,
# so that a conversion into this format, such as `t.to(FlirtTransform)`,
# fails with a reason instead of building a wrong object.
from . import _converters  # noqa: E402, F401  isort: skip
