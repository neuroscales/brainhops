"""The NIfTI image format."""

# stdlib
from io import BytesIO

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    WriterError,
)
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw
from brainhops.io.common.nifti._constants import (
    _NIFTI_FIELD_INTENTS,
    _NIFTI_INTENT_NONE,
    _NIFTI_XCODES,
)
from brainhops.io.common.nifti._geometry import _image_with_geometry
from brainhops.io.common.nifti._header import (
    _nifti_intent,
    _nifti_shape,
    _nifti_to_axes,
    _NiftiObject,
)
from brainhops.io.common.nifti._raw import (
    _sniffed_header,
    read_nifti,
    write_nifti,
)
from brainhops.io.common.nifti._units import nifti_to_unit
from brainhops.io.common.nifti._views import _image_to_disk, _image_to_model
from brainhops.io.images.base import ImageFormat


@register_format
class NiftiImage(
    ImageFormat, SingleScaleImage, BinaryFileReader, BinaryFileWriter
):
    """An image stored in a NIfTI file.

    A NIfTI image holds the header of its file as [`NiftiMetadata`][] and
    the voxels as stored in `raw`. When the image is read from a file,
    `raw` is a nibabel proxy, which reads the voxels only when they are
    needed, and `data` is decoded from it on first access. The
    transformations are decoded from the header in the same way, unless
    other transformations are assigned.

    When the image is written, the header of its file is the base of the
    new header. The fields that describe the layout of the voxels and the
    geometry are encoded again from the image, and the other fields, such
    as the description, the intent and the extensions, are kept. The
    voxels of an image that is read and written again without changes are
    copied as the file stores them. The header is then the same as well,
    unless the file stores the geometry in a way that the writer does not
    reproduce, for example with a qform or an sform whose code is zero.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nifti",)

    raw: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The voxels as the file stores them, or `None` without data.

    After a read, `raw` is a nibabel proxy, which applies the scaling of
    the header when it is read. Setting `data` stores the new array here.
    """

    _metadata: KwOnly[NoRepr[tx.Optional[NiftiMetadata]]] = None

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the file, which holds its header, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous header.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        # The constructor assigns the fields in order, so the default of
        # `raw`, assigned after `data`, erases the array stored by `data=`.
        # The array is therefore stored again. When both are given, `data`
        # takes precedence over `raw`.
        if arguments.get("data") is not None:
            self.data = arguments["data"]

    @smartproperty(cache=True, invalidates=("data",))
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, decoded from `raw` on first access and cached.

        Data decoded from a file is read-only, because changing it in
        place would not change `raw`, which is what the image writes. To
        change the voxels, assign `data`: setting it stores the new array
        in `raw` and drops the cached value.
        """
        if self.raw is None:
            return None
        return _image_to_model(self.raw)

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self.raw = None if value is None else _image_to_disk(value)

    @smartproperty(cache=True, unset=(None, "empty"))
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the header.

        The decoded list is cached, so `img.transformation is
        img.transformation`, and a transformation appended to the list in
        place is kept. Assigning a list replaces the decoded one, and
        assigning other metadata drops it. A header modified in place is
        not noticed, so a modified copy should be assigned as new metadata
        instead.
        """
        if self.metadata is None or self.metadata.raw is None:
            return []
        return _nifti_to_transformations(self.metadata.raw.header)

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system described by the header.

        The axes are those of the stored array, in Fortran order, with the
        names and types that the intent code gives them. An image without
        metadata has no system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        axes = [
            axis
            for axis in _nifti_to_axes(self.metadata.raw.header)
            if axis.name is not None
        ]
        # In F order, the first axis changes fastest.
        return CoordinateSystem(axes=axes, name="voxel", order="F")

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a NIfTI image.

        A stream that starts with a NIfTI header is scored with
        `_score_nibabel`. Any other stream is declined by
        [`NiftiRaw.sniff_fileobj`][], which raises the requested error.
        """
        header = _sniffed_header(file, kwargs.get("version"))
        if header is None:
            return NiftiRaw.sniff_fileobj(file, error=error, **kwargs)
        return cls._score_nibabel(header)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a NIfTI image."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a NIfTI header as a plain image.

        Any NIfTI file can be read as an image, but a displacement-field
        intent suggests that the file is meant as a transformation, so it
        is only a weak match.
        """
        intent = _nifti_intent(header)
        if intent is None:
            return Confidence.MAYBE
        if intent in _NIFTI_FIELD_INTENTS:
            return Confidence.WEAK
        # The intent is often unset, so the shape is checked too: a trailing
        # axis of length 3 on a 4D or larger image may be a deformation.
        shape = _nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.WEAK
        if intent == _NIFTI_INTENT_NONE:
            return Confidence.LIKELY
        return Confidence.MAYBE

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "NiftiImage":
        """Read an image from the path of a NIfTI file.

        A local file is handed to nibabel by name, so that nibabel opens
        it whenever the voxels are read and can memory-map them. A remote
        file is read into memory instead. The keyword arguments are passed
        to nibabel.

        Raises
        ------
        ParserExistsError
            If the path does not exist.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        raw, proxy = read_nifti(filename, **kwargs)
        return cls(raw=proxy, metadata=NiftiMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "NiftiImage":
        """Read an image from an open NIfTI stream.

        The proxy reads the voxels from the stream when they are needed,
        so the stream must stay open while the data may be read. The
        keyword arguments are passed to nibabel.
        """
        raw, proxy = read_nifti(file, **kwargs)
        return cls(raw=proxy, metadata=NiftiMetadata.from_raw(raw))

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> "NiftiImage":
        """Build an image from a nibabel image or header.

        The voxels of an image become `raw` as they are, and a copy of its
        header becomes the record of the metadata. A header alone gives an
        image without data.

        Parameters
        ----------
        nifti : nibabel.Nifti1Image or nibabel.Nifti1Header
            The nibabel image or header. NIfTI-2 images and headers are
            accepted too.
        **kwargs : Any
            Other fields of the image.

        Returns
        -------
        NiftiImage
            The image.

        Raises
        ------
        TypeError
            If `nifti` is neither a NIfTI image nor a NIfTI header.
        """
        if isinstance(nifti, nb.Nifti1Image):
            raw, header = nifti.dataobj, nifti.header
        elif isinstance(nifti, nb.Nifti1Header):
            raw, header = None, nifti
        else:
            raise TypeError(
                f"Expected a NIfTI image or header, got {type(nifti)}"
            )
        metadata = NiftiMetadata.from_raw(NiftiRaw(header=header.copy()))
        return cls(raw=raw, metadata=metadata, **kwargs)

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """Build the nibabel image that encodes this image.

        A copy of the header of the metadata is the base of the new header,
        and the fields that describe the layout of the voxels and the
        geometry are encoded again over it. The data become the NIfTI
        array. The preferred transformation becomes the sform, and the
        qform is a rigid transformation among the others or the rigid part
        of the sform. Large arrays are written as NIfTI-2, and so is an
        image whose header is NIfTI-2.

        Parameters
        ----------
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose description and intent are copied. The geometry
            always comes from this image.
        **overrides : Any
            Values of `dtype`, `intent` and `descrip`, or of any other
            header field, which take precedence over the derived values and
            `like`.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.

        Raises
        ------
        WriterError
            If the image has no data.
        UnrepresentableTransformationError
            If the preferred transformation is not affine, as for a
            displacement field.
        """
        if self.raw is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        base = None if self.metadata is None else self.metadata.to_raw()
        return _image_with_geometry(
            self.raw,
            self.transformation,
            self.transformations,
            base=base,
            like=like,
            overrides=overrides,
        )

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a path, compressed if its name ends with `.gz`.

        The keyword arguments are those of [`to_nibabel`][].
        """
        write_nifti(self.to_nibabel(**kwargs), filename)

    def to_fileobj(self, file: tx.BinaryIO, **kwargs) -> None:
        """Write the image to a stream as an uncompressed NIfTI file.

        The keyword arguments are those of [`to_nibabel`][].
        """
        write_nifti(self.to_nibabel(**kwargs), file)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the uncompressed NIfTI encoding of the image.

        The keyword arguments are those of [`to_nibabel`][].
        """
        return self.to_nibabel(**kwargs).to_bytes()


def _nifti_to_transformations(
    header: nb.Nifti1Header,
) -> tx.List[Transformation]:
    """Convert a NIfTI header to transformations.

    The transformations are, in order:

    1. A transformation from voxel to scaled voxel space, named "physical";
    2. The qform voxel-to-RAS rigid transformation, named "qform";
    3. The sform voxel-to-RAS affine transformation, named "sform";
    4. The qform voxel-to-RAS rigid transformation, named after its code.
    5. The sform voxel-to-RAS affine transformation, named after its code.

    The code names are `"unknown"`, `"scanner"`, `"aligned"`,
    `"talairach"`, `"mni"` and `"template"`. A form whose code is zero is
    not set and is omitted, along with its entry named after the code.
    When the qform and the sform have the same code, the fourth entry is
    omitted. As a result, the last transformation is the affine that
    nibabel's `get_best_affine()` returns, named after its code.

    The NIfTI affine applies to the spatial axes. When there are other
    axes, each voxel-to-RAS transformation is a `Sequence` made of a
    `SubspaceTransformation` of the spatial axes and, for a time axis, one
    that scales by `pixdim[4]` and translates by `toffset`. Other axes pass
    through, and a time axis without a valid spacing stays a frame index
    in every space. An image with only spatial axes gets a plain `Affine`.
    """

    xforms = []

    # --- preliminaries ------------------------------------------------

    axes = _nifti_to_axes(header)
    # Units go through the single NIfTI-to-brainhops converter, which reads
    # an unknown spatial unit as millimeters and leaves an unknown time
    # unit unspecified (see `brainhops.io.common.nifti._units`).
    space, time = header.get_xyzt_units()
    units = {
        "space": nifti_to_unit(space, "space"),
        "time": nifti_to_unit(time, "time"),
    }
    orientation = {
        "x": "left-to-right",
        "y": "posterior-to-anterior",
        "z": "inferior-to-superior",
    }
    keep_dims = [
        i
        for i, axis in enumerate(axes)
        if axis.name is not None and axis.type == "space"
    ]

    # --- coordinate systems -------------------------------------------

    named_axes = [axis for axis in axes if axis.name is not None]

    # A time axis whose spacing is zero or non-finite (pixdim[4] = 0 when
    # the step is missing) has no TR. It is not mapped to time and stays a
    # frame index in every space.
    zooms = header.get_zooms()
    zooms = [zooms[i] for i, axis in enumerate(axes) if axis.name is not None]
    untimed = {
        j
        for j, axis in enumerate(named_axes)
        if axis.type == "time" and not (np.isfinite(zooms[j]) and zooms[j])
    }
    zooms = [1.0 if j in untimed else z for j, z in enumerate(zooms)]

    # The axes count samples, in Fortran order as nibabel reads them.
    voxel_space = CoordinateSystem(name="voxel", axes=named_axes, order="F")

    # Axes that are neither space nor time, such as channels, have no unit
    # rather than inheriting the index unit.
    phys_axes = [
        axis if j in untimed else replace(axis, unit=units.get(axis.type))
        for j, axis in enumerate(named_axes)
    ]
    phys_space = CoordinateSystem(name="physical", axes=phys_axes)

    ras_axes = [
        replace(
            axis,
            orientation=Orientation(
                type="anatomical", value=orientation[axis.name]
            ),
        )
        if axis.name in orientation
        else axis
        for axis in phys_axes
    ]
    ras_space = CoordinateSystem(name="RAS", axes=ras_axes)

    # --- voxel-to-physical --------------------------------------------
    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=zooms)
    xforms.append(vox2phys)

    # The voxel space holds only the named axes, so these positions are not
    # always the positions in the header.
    named = [i for i, axis in enumerate(axes) if axis.name is not None]
    space_pos = [named.index(i) for i in keep_dims]
    time_pos = [
        j
        for j, axis in enumerate(named_axes)
        if axis.type == "time" and j not in untimed
    ]
    product = bool(space_pos) and len(space_pos) < len(named_axes)

    def _coded_affine(matrix: np.ndarray, label: str) -> Transformation:
        """Build the voxel-to-RAS transformation of a qform or sform."""
        matrix = matrix[keep_dims + [-1], :][:, keep_dims + [-1]]
        if product:
            return _coded_product(matrix, label)
        return Affine(
            input=voxel_space,
            output=replace(ras_space, name=label),
            matrix=matrix[:-1],
        )

    def _coded_product(matrix: np.ndarray, label: str) -> Sequence:
        """Build the voxel-to-RAS transformation of an image with more axes.

        The affine maps the spatial axes and the time axis is mapped by its
        own spacing and origin, as a sequence of subspace transformations.
        """
        world = replace(ras_space, name=label)
        # Between the two steps, the spatial axes are in world coordinates
        # while the others are still voxel indices. That space is neither
        # the voxel nor the world space, so it has no name.
        middle_axes = list(voxel_space.axes)
        for j in space_pos:
            middle_axes[j] = world.axes[j]
        middle = CoordinateSystem(axes=middle_axes) if time_pos else world
        spatial = SubspaceTransformation(
            transformation=Affine(
                matrix=matrix[:-1],
                input=voxel_space.restrict(space_pos),
                output=world.restrict(space_pos),
            ),
            input_axes=space_pos,
            output_axes=space_pos,
            input=voxel_space,
            output=middle,
        )
        steps = [spatial]
        if time_pos:
            # t_world = toffset + dt * t_index: the spacing gives the time
            # since the first frame, and `toffset` shifts it to world time.
            (it,) = time_pos
            dt = zooms[it]
            toffset = float(header["toffset"])
            temporal = Sequence(
                transformations=[
                    Scaling(
                        scale=[dt],
                        input=voxel_space.restrict([it]),
                        output=phys_space.restrict([it]),
                    ),
                    Translation(
                        translation=[toffset],
                        input=phys_space.restrict([it]),
                        output=world.restrict([it]),
                    ),
                ],
                input=voxel_space.restrict([it]),
                output=world.restrict([it]),
            )
            steps.append(
                SubspaceTransformation(
                    transformation=temporal,
                    input_axes=[it],
                    output_axes=[it],
                    input=middle,
                    output=world,
                )
            )
        return Sequence(transformations=steps, input=voxel_space, output=world)

    # --- qform --------------------------------------------------------
    # get_qform and get_sform return None for a form whose code is 0, and
    # only one of the two forms is required.
    qmatrix, qcode = header.get_qform(coded=True)
    qform = None
    if qmatrix is not None:
        qname = _NIFTI_XCODES.get(qcode, "unknown")
        qform = _coded_affine(qmatrix, "qform")
        xforms.append(qform)

    # --- sform --------------------------------------------------------
    smatrix, scode = header.get_sform(coded=True)
    sform = None
    if smatrix is not None:
        sname = _NIFTI_XCODES.get(scode, "unknown")
        sform = _coded_affine(smatrix, "sform")
        xforms.append(sform)

    # --- named & best affines -----------------------------------------
    # The last transformation must be nibabel's best affine, named after
    # its code. It is rebuilt under that name rather than renamed, so that
    # the world space is named consistently in every step of a sequence.
    if sform is not None and qform is not None:
        if scode == qcode:
            xforms.append(_coded_affine(smatrix, sname))
        else:
            xforms.append(_coded_affine(qmatrix, qname))
            xforms.append(_coded_affine(smatrix, sname))
    elif sform is not None:
        xforms.append(_coded_affine(smatrix, sname))
    elif qform is not None:
        xforms.append(_coded_affine(qmatrix, qname))

    return xforms
