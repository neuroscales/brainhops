"""The registration of one view of a BigDataViewer dataset."""

__all__ = ["BdvViewRegistration"]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.errors import UNSET, ErrorPolicy
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import write_file
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    UnrepresentableTransformationError,
)
from brainhops.io.common.bdv import BdvMetadata, BdvRaw
from brainhops.io.common.bdv._geometry import global_system, voxel_system
from brainhops.io.metadata._base import (
    metadata_options,
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import TransformationFormat

# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------
# The XML file is short and parsed in one pass, so the matrices of the
# transforms live in the record, as the matrix of an LTA file does. The
# chain is decoded from the record of the view that `setup` and
# `timepoint` select, and each element is decoded once, so that the
# writer recognizes the elements that come from the record by their
# identity and writes their elements back as they are, with their names.


def _record(xform: "BdvViewRegistration") -> tx.Optional[BdvRaw]:
    """Return the record of the metadata of a registration, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _matrix_to_model(values: tx.Sequence[float]) -> np.ndarray:
    """Return the `(3, 4)` matrix of the twelve values of a transform.

    The values are the first three rows of the matrix, row after row
    (`AffineTransform3D.set(double...)` in imglib2-realtransform, commit
    ca9039d). The matrix is a new array, made read-only, because the
    writer reads the record and not the matrix, so an edit in place
    would be lost.
    """
    matrix = np.array(values, dtype=np.float64).reshape(3, 4)
    matrix.flags.writeable = False
    return matrix


def _matrix_to_disk(data: ArrayProtocol) -> tx.Tuple[float, ...]:
    """Return the twelve values that a transform stores for a matrix.

    The matrix is a `(3, 4)` matrix, or a `(4, 4)` homogeneous one.

    Raises
    ------
    UnrepresentableTransformationError
        If the matrix is not the matrix of a 3D affine.
    """
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3]
    if matrix.shape != (3, 4):
        raise UnrepresentableTransformationError(
            f"BigDataViewer stores 3D affines, of shape (3, 4), and this "
            f"matrix has shape {matrix.shape}."
        )
    return tuple(float(v) for v in matrix.reshape(-1))


def _decode(registration: BdvRaw.Registration, index: int) -> _xforms.Affine:
    """Decode the transform of a registration at an index of the file.

    The transform applied first, which is the last one of the file, maps
    the voxel system of the view, and the transform applied last, which
    is the first one of the file, maps to the global system. The systems
    between two transforms are not known.

    Raises
    ------
    ParserContentError
        If the transform is generic.
    """
    item = registration.transforms[index]
    if item.affine is None:
        raise ParserContentError(
            f"Transform {index} of the view of time point "
            f"{registration.timepoint} and setup {registration.setup} is a "
            f"generic transform, which brainhops cannot read."
        )
    last = len(registration.transforms) - 1
    return _xforms.Affine(
        matrix=_matrix_to_model(item.affine),
        input=voxel_system() if index == last else None,
        output=global_system() if index == 0 else None,
    )


# ----------------------------------------------------------------------
#   FORMAT
# ----------------------------------------------------------------------


@register_format
class BdvViewRegistration(
    BinaryFileReader,
    BinaryFileWriter,
    TransformationFormat,
    _xforms.ImmutableSequence,
):
    """The registration of one view of a BigDataViewer dataset.

    BigDataViewer and BigStitcher keep, in the XML file of a dataset, a
    list of affine transforms for each view, which together map the
    voxels of the view to the global space of the dataset (see
    [`brainhops.io.common.bdv`][]). This class reads the list of one
    view, which the reader options `setup` and `timepoint` select, by
    default the first setup at the first time point, in the order of
    their ids. Any image loader is accepted, so the registrations of a
    dataset stored in N5 or in TIFF files are read as well.

    The transformation is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of [`Affine`][brainhops.datamodel.transformations.Affine] elements,
    one for each transform, in the order in which they are applied, which
    is the reverse of the order of the file. Its input is the voxel
    system of the view, with axes x, y and z and integers at the centers
    of the voxels, and its output is the `"global"` system, which has no
    unit. The composed affine is `xform.to(Affine)`. The names of the
    transforms, such as `calibration`, are kept by the record.

    The whole XML file is held as a [`BdvRaw`][] record by the
    [`BdvMetadata`][] of the transformation. A chain that is assigned,
    and an `input` or `output` that is set, are pending until
    [`consolidate`][] encodes them into a new record, or until the
    transformation is written.

    !!! note "What is written"
        A registration whose chain and systems were not assigned is
        written back as the record holds the file, byte for byte. A
        registration whose chain was assigned is written as the same
        file, in which only the `<ViewRegistration>` element of its view
        is written again. An element that this transformation decoded
        from its own record is written as it was, with its name and the
        text of its values, and any other element is written as a new
        affine transform without a name. A registration without a record
        is written as a new XML file that holds a single view, without
        images.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xml",)
    HINTS = ("bdv", "bigdataviewer", "bigstitcher")

    _metadata: NoRepr[tx.Optional[BdvMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty(
        "metadata", invalidates=("transformations", "decoded")
    )
    """The metadata of the dataset, which holds its XML file as a record.

    A registration built from a chain has no metadata. Assigning other
    metadata drops the chain decoded from the old record, and a chain
    that was assigned is kept.
    """

    _setup: tx.Optional[int] = None

    setup = smartproperty("setup", invalidates=("transformations", "decoded"))
    """The id of the setup of the view.

    It is the reader option `setup`, and a reader stores the id of the
    setup that it read. `None` stands for the first setup, or setup 0
    for a registration without a record. Assigning another id drops the
    chain decoded from the record. The id only selects what is read: the
    file is written with every view.
    """

    _timepoint: tx.Optional[int] = None

    timepoint = smartproperty(
        "timepoint", invalidates=("transformations", "decoded")
    )
    """The id of the time point of the view.

    It is the reader option `timepoint`, and it behaves as `setup` does.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The voxel system of the view, unless set explicitly."""
        return voxel_system()

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The global system of the dataset, unless set explicitly."""
        return global_system()

    # --- chain --------------------------------------------------------

    def view(self) -> tx.Tuple[int, int]:
        """Return the ids of the setup and of the time point of the view.

        Returns
        -------
        setup : int
            The id of the setup.
        timepoint : int
            The id of the time point.

        Raises
        ------
        ParserContentError
            If the record has no such setup or time point.
        """
        record = _record(self)
        if record is None:
            return int(self.setup or 0), int(self.timepoint or 0)
        return record.view(self.setup, self.timepoint)

    def transform(self, index: int) -> _xforms.Affine:
        """Decode the transform at an index of the file.

        The transform is decoded once and the same object is returned
        afterwards, so that the writer can recognize it by its identity.
        Assigning other metadata, or another view, forgets the decoded
        transforms.

        Parameters
        ----------
        index : int
            The index of the `<ViewTransform>` element in the
            `<ViewRegistration>` element of the view, in the order of the
            file.

        Returns
        -------
        Affine
            The affine that the transform encodes.

        Raises
        ------
        ParserContentError
            If the registration has no record or no such view, or if the
            transform is generic.
        """
        record = _record(self)
        if record is None:
            raise ParserContentError(
                "This BigDataViewer registration has no record."
            )
        setup, timepoint = self.view()
        decoded = self.__dict__.setdefault("_cache_decoded", {})
        if index not in decoded:
            registration = record.registration(timepoint, setup)
            decoded[index] = _decode(registration, index)
        return decoded[index]

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of affines, in the order in which they are applied.

        The chain is decoded from the record when it is first used, and
        it is then cached. A registration without a record and without an
        assigned chain has an empty chain, which is the identity.
        Assigning a chain replaces the decoded one, and the assigned chain
        is pending until [`consolidate`][] or the writer encodes it.
        """
        record = _record(self)
        if record is None:
            return ()
        setup, timepoint = self.view()
        count = len(record.registration(timepoint, setup).transforms)
        return tuple(self.transform(i) for i in reversed(range(count)))

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.Sequence[_xforms.Transformation]]
    ) -> None:
        # The assigned chain is stored as a tuple, the type of the chain
        # that is decoded, and the setter drops the decoded chain.
        self._transformations = None if value is None else tuple(value)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a stream from the root element of its XML.

        The score is `MAYBE` for a BigDataViewer XML file, below the
        `CERTAIN` of
        [`BdvMultiScaleImage`][brainhops.io.images.bdv.BdvMultiScaleImage],
        so that `io.load` reads a dataset stored in HDF5 as an image,
        whose transformation is the registration, as it reads a NIfTI
        file as an image rather than as an affine.
        [`transformations.load`][brainhops.io.transformations.load] reads
        it as this class.

        Parameters
        ----------
        file : IO
            The binary stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a
            BigDataViewer XML file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if BdvMetadata.sniff_fileobj(file, error=error):
            return Confidence.MAYBE
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes from the root element of their XML.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not a
            BigDataViewer XML file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, as for [`sniff_fileobj`][].
        """
        if BdvMetadata.sniff_bytes(content, error=error):
            return Confidence.MAYBE
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(
        cls,
        file: tx.BinaryIO,
        setup: tx.Optional[int] = None,
        timepoint: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """Read the registration of a view from a binary stream.

        The record is read by [`BdvMetadata.from_fileobj`][], and the
        chain is decoded from it when it is first used.

        Parameters
        ----------
        file : BinaryIO
            The stream of the XML file.
        setup : int, optional
            The id of the setup of the view, by default the first one.
        timepoint : int, optional
            The id of the time point of the view, by default the first
            one.
        **kwargs : Any
            Other options, which are refused.

        Returns
        -------
        Self
            The registration.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ParserContentError
            If the content is not a BigDataViewer XML file, or if it has
            no registration for the view.
        """
        metadata = BdvMetadata.from_fileobj(file, **kwargs)
        return cls.from_raw(metadata.raw, setup=setup, timepoint=timepoint)

    @classmethod
    def from_raw(cls, raw: BdvRaw, **kwargs) -> tx.Self:
        """Build a registration from an already read [`BdvRaw`][].

        The record is held by new metadata, without a copy, and the ids
        of the view are resolved and checked.

        Parameters
        ----------
        raw : BdvRaw
            The record of the XML file.
        **kwargs : Any
            Field values, such as `setup` and `timepoint`, which go to the
            constructor.

        Returns
        -------
        Self
            The registration.

        Raises
        ------
        ParserContentError
            If the record has no registration for the view.
        """
        setup, timepoint = raw.view(
            kwargs.pop("setup", None), kwargs.pop("timepoint", None)
        )
        raw.registration(timepoint, setup)
        return cls(
            metadata=BdvMetadata.from_raw(raw),
            setup=setup,
            timepoint=timepoint,
            **kwargs,
        )

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a registration from a record, a file or a data model.

        A [`BdvRaw`][] is read with [`from_raw`][], and any other value
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
            The new registration.
        """
        if isinstance(other, BdvRaw) and not args:
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

        If neither the chain nor the `input` and `output` systems were
        assigned, and the metadata holds a record, the registration is
        left as it is, so that the file is written back as it was read.
        The options `setup` and `timepoint` only select what is decoded,
        so they do not change the record. Otherwise, the new record is the
        current one, in which the `<ViewRegistration>` element of the view
        is written again from the chain, as described in
        [`BdvViewRegistration`][]. Systems that were assigned must be
        compatible with the voxel system of a view and with the global
        system, since the file stores no system. A registration without a
        record is encoded as a new file with a single view. The new record
        is held by new metadata, and the chain is then decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a
        BigDataViewer dataset holds no field of the vocabulary yet, so a
        field that is set is reported as lost, once, by `on_error`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this registration rather than a copy.
        on_error : ErrorPolicy, optional
            What to do with the metadata that the record cannot hold,
            and with any other error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        Self
            The consolidated copy, or this registration with
            `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If an element is not a 3D affine, or if an assigned system is
            not compatible with the systems of the file.
        WriterError
            If the record cannot be changed in place.
        """
        report = ConversionReport(
            source="BdvViewRegistration", target="BdvViewRegistration"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, BdvMetadata, fields, report
        )
        apply_loss_policy(report, on_error, stacklevel=2)
        if not requested:
            metadata = self.metadata
        record = _record(self)
        assigned = self._transformations is not None
        declared = self._input is not None or self._output is not None
        if not assigned and not declared and record is not None:
            if not inplace:
                return replace(self, metadata=metadata)
            if requested:
                self.metadata = metadata
            return self
        if declared:
            _check_systems(self)
        setup, timepoint = self.view()
        record = _encode(self, record, setup, timepoint)
        if metadata is None:
            metadata = BdvMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self,
                metadata=metadata,
                transformations=None,
                input=None,
                output=None,
                setup=setup,
                timepoint=timepoint,
            )
        self.metadata = metadata
        self.transformations = None
        self.input = None
        self.output = None
        self.setup = setup
        self.timepoint = timepoint
        return self

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the XML file that encodes the registration.

        The registration is consolidated into a copy, whose record is
        written, and the registration itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer, which are refused. `on_error` and the
            fields of the metadata are given to [`consolidate`][].

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        TypeError
            If an option is given.
        UnrepresentableTransformationError
            If the chain or the systems cannot be encoded.
        """
        fields, kwargs = metadata_options(kwargs)
        return _record(self.consolidate(**fields)).to_bytes(**kwargs)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the registration to the file at a path, replacing it.

        The registration is consolidated into a copy before the file is
        written, so a registration that cannot be encoded leaves the file
        untouched. A local file is replaced at once.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the writer, which are refused.

        Raises
        ------
        TypeError
            If an option is given.
        UnrepresentableTransformationError
            If the chain or the systems cannot be encoded.
        """
        write_file(filename, self.to_bytes(**kwargs))


# ----------------------------------------------------------------------
#   ENCODING A CHAIN AS A RECORD
# ----------------------------------------------------------------------


def _check_systems(xform: BdvViewRegistration) -> None:
    """Refuse assigned systems that the file cannot express.

    Raises
    ------
    UnrepresentableTransformationError
        If the input is not compatible with the voxel system of a view,
        or the output with the global system.
    """
    for name, system, expected in (
        ("input", xform.input, voxel_system()),
        ("output", xform.output, global_system()),
    ):
        if not expected.compatible_with(system):
            raise UnrepresentableTransformationError(
                f"A BigDataViewer registration maps the voxels of a view, "
                f"with axes x, y and z, to the global space, and the "
                f"{name} of this registration is {system!r}."
            )


def _encode(
    xform: BdvViewRegistration,
    record: tx.Optional[BdvRaw],
    setup: int,
    timepoint: int,
) -> BdvRaw:
    """Encode the chain of a registration into a new record.

    An element that this registration decoded from its record is
    recognized by its identity and written as its element of the file.
    Any other element is converted to an affine and written without a
    name.

    Raises
    ------
    UnrepresentableTransformationError
        If an element is not a 3D affine.
    """
    decoded = xform.__dict__.get("_cache_decoded", {})
    reused = {id(item): index for index, item in decoded.items()}
    old = None if record is None else record.registration(timepoint, setup)
    listed = []
    for item in reversed(tuple(xform.transformations)):
        if old is not None and id(item) in reused:
            listed.append(old.transforms[reused[id(item)]])
            continue
        try:
            matrix = item.to(_xforms.Affine).matrix
        except Exception as error:  # noqa: BLE001
            raise UnrepresentableTransformationError(
                f"BigDataViewer stores affine transforms, and a "
                f"{type(item).__name__} is not one."
            ) from error
        listed.append(BdvRaw.Transform(affine=_matrix_to_disk(matrix)))
    if record is None:
        record = BdvRaw()
    return record.with_registration(timepoint, setup, listed)
