__all__ = ["NiftiParser"]

# stdlib
from io import BytesIO

# dependencies
import nibabel as nb
import typing_extensions as tx

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.io.base.parsers import BinaryFileParser, SnifferContentError

# typing
NiftiObject = tx.Union[nb.Nifti1Header, nb.Nifti1Image]


_NIFTI_AXES = [
    Axis("x", "space"),
    Axis("y", "space"),
    Axis("z", "space"),
    Axis("t", "time"),
    Axis("c", "channel"),
    Axis("dim5"),
    Axis("dim6"),
]
_FLAT_AXES = {
    0: Axis("n"),  # number of points / vertices / triangles / ...
    1: Axis("x"),
    2: Axis("y"),
    3: Axis("z"),
}
_FLAT_AXES_CHANNEL = {**_FLAT_AXES, 4: Axis("c", "channel")}
_FLAT_AXES_TIME = {**_FLAT_AXES, 4: Axis("t", "time")}
_AXES_DISP = {4: Axis("c", "displacement")}
_NIFTI_SPECIFIC_AXES = {
    1004: {5: Axis("k", "channel")},  # GENMATRIX
    1006: _AXES_DISP,  # DISPVECT
    1008: _FLAT_AXES_CHANNEL,  # POINTSET
    1009: _FLAT_AXES_CHANNEL,  # TRIANGLE
    # --- GIFTI ---
    2001: _FLAT_AXES_TIME,  # TIME_SERIES
    2002: _FLAT_AXES_CHANNEL,  # NODE_INDEX
    2003: _FLAT_AXES_CHANNEL,  # RGB_VECTOR
    2004: _FLAT_AXES_CHANNEL,  # RGBA_VECTOR
    2005: _FLAT_AXES_CHANNEL,  # SHAPE
    # --- FSL ---
    2006: _AXES_DISP,  # FSL_FNIRT_DISPLACEMENT_FIELD
    2007: _AXES_DISP,  # FSL_CUBIC_SPLINE_COEFFICIENTS
    2008: _AXES_DISP,  # FSL_DCT_COEFFICIENTS
    2009: _AXES_DISP,  # FSL_QUADRATIC_SPLINE_COEFFICIENTS
}


class NiftiParser(DataModelBase, BinaryFileParser):
    """
    Base class for objects that are encoded by a NIfTI file.

    This class is a base for `NiftiBasedImage` and `NiftiBasedTransformation`.
    """

    # --- NIfTI API ----------------------------------------------------

    image: tx.Annotated[
        tx.Optional[nb.Nifti1Image],
        tx.Doc(
            """
            The NIfTI image associated with this object.

            If no image is provided, but a header is, the `image` will
            be `None`, and the object will be constructed from the header
            only.
            """
        ),
    ] = None

    _header: tx.Annotated[
        tx.Optional[nb.Nifti1Header],
        tx.Doc(
            """
            The NIfTI header associated with this object.

            If the object was created from a NIfTI image, this will be
            pointing to the header of that image, unless a different
            header was explicitly provided to the constructor.

            If the object was created from a NIfTI header, this will be
            pointing to that header.
            """
        ),
    ] = None

    @property
    def header(self) -> tx.Optional[nb.Nifti1Header]:
        """
        The NIfTI header associated with this object.

        If a header was explicitly set by the user (at construction or
        later), this will be pointing to that header.

        Otherwise, if the object was created from a NIfTI header, this
        will be pointing to that header.

        Otherwise, if the object was created from a NIfTI image, this
        will be pointing to the header of that image.

        !!! example
        ```python
            import nibabel as nb
            image1 = nb.load("image1.nii")
            image2 = nb.load("image2.nii")
            NiftiParser(image1).header                        # `image1.header`
            NiftiParser(header=image2.header).header          # `image2.header`
            NiftiParser(image1, header=image2.header).header  # `image2.header`
            obj = NiftiParser(image1)
            obj.header = image2.header
            obj.header                                        # `image2.header`
        ```
        """
        if getattr(self, "_header", None) is not None:
            return self._header
        if self.image is not None:
            return self.image.header
        return None

    @header.setter
    def header(self, value: tx.Optional[nb.Nifti1Header]) -> None:
        self._header = value

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        if getattr(self, "_data", None) is not None:
            return self._data

        if self.image is not None:
            # Get the nibabel array
            data = get_array_backend().asarray(self.image.dataobj)

            # Drop irrelevant axes as specified by the intent code.
            slicer = [
                slice(None) if axis.name is not None else 0
                for axis in _nifti_to_axes(self.header)
            ]
            self._data = data[tuple(slicer)]
            return self._data

        return None

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._data = value

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        if getattr(self, "_system", None) is not None:
            return self._system

        if self.header is None:
            return None

        # Drop irrelevant axes, as specified by the intent code.
        axes = [
            axis
            for axis in _nifti_to_axes(self.header)
            if axis.name is not None
        ]

        return CoordinateSystem(axes=axes, name="voxel")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    # --- BinaryFileParser API -----------------------------------------

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        try:
            obj = nb.Nifti1Image.from_file_map(
                {"header": fileobj, "image": fileobj}, **kwargs
            )
        except Exception:
            obj = nb.Nifti1Header.from_fileobj(fileobj, **kwargs)
        return cls.from_nibabel(obj)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> tx.Self:
        return cls.from_fileobj(BytesIO(data), **kwargs)

    @classmethod
    def from_nibabel(cls, nifti: NiftiObject, **kwargs) -> tx.Self:
        if isinstance(nifti, nb.Nifti1Header):
            return cls(header=nifti, **kwargs)
        if isinstance(nifti, nb.Nifti1Image):
            return cls(image=nifti, header=nifti.header, **kwargs)
        raise TypeError(f"Expected a NIfTI image or header, got {type(nifti)}")

    # --- BinaryFileSniffer API ----------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        *,
        version: tx.Optional[int] = None,
        **kwargs,
    ) -> bool:

        # --- If nifti version not provided, try both NIfTI-1 and NIfTI-2
        if version is None:
            for version in (1, 2):
                if cls.sniff_fileobj(
                    file, error=False, version=version, **kwargs
                ):
                    return True
            if error:
                if error is True:
                    error = SnifferContentError
                raise error("Content is not a valid NIfTI-1 or NIfTI-2 file")
            return False

        # --- Version hint is provided, use appropriate nibabel class
        NiftiHeader = {1: nb.Nifti1Header, 2: nb.Nifti2Header}[version]
        kwargs["error"] = error
        try:
            pos = file.tell() if hasattr(file, "tell") else 0
        except Exception:
            pos = 0
        base_error = None
        try:
            with nb.openers.ImageOpener(file, cls._READ_MODE) as f:
                nbkwargs = {}
                if "endianness" in kwargs:
                    nbkwargs["endianness"] = kwargs.pop("endianness")
                if "check" in kwargs:
                    nbkwargs["check"] = kwargs.pop("check")
                else:
                    nbkwargs["check"] = False
                obj = NiftiHeader.from_fileobj(f, **nbkwargs)
                result = cls.sniff_nibabel(obj, **kwargs)
        except Exception as e:
            base_error = e
            result = False
        finally:
            if hasattr(file, "seek"):
                try:
                    file.seek(pos)
                except Exception:
                    pass

        if result:
            return True

        # Cannot parse this content -> return False or error
        if error:
            if error is True:
                error = SnifferContentError
            error = error(f"Content is not a valid NIfTI-{version} file")
            if base_error is not None:
                raise error from base_error
            else:
                raise error
        return False

    @classmethod
    def sniff_nibabel(
        cls,
        nifti: NiftiObject,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> bool:
        if isinstance(nifti, nb.Nifti1Image):
            return cls.sniff_nibabel(nifti.header, error=error, **kwargs)
        if isinstance(nifti, nb.Nifti2Header):
            result = nifti["sizeof_hdr"] == 540
        elif isinstance(nifti, nb.Nifti1Header):
            result = nifti["sizeof_hdr"] == 348
        if result:
            return True
        if error:
            if error is True:
                error = SnifferContentError
            raise error(
                f"Magic number does not match NIfTI header: "
                f"{nifti['sizeof_hdr']}"
            )
        return False

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> bool:
        kwargs["error"] = error
        return cls.sniff_fileobj(BytesIO(data), **kwargs)


def _nifti_to_axes(header: nb.Nifti1Header) -> tx.List[Axis]:
    """
    Compute the axes of a NIfTI file, based on its header.

    This function takes into account the intent code of the NIfTI file
    to change the names and types of the axes, if necessary.

    Axes that are deemed irrelevant by the intent code are given a name
    of `None`.
    """

    ndim = len(header.get_data_shape())

    # Get names and types of existing axes
    axes = _NIFTI_AXES[:ndim]

    # Apply specific knowledge from intent code
    if header["intent_code"] in _NIFTI_SPECIFIC_AXES:
        axes_map = _NIFTI_SPECIFIC_AXES[header["intent_code"]]
        axes = [axes_map.get(i, axis) for i, axis in enumerate(axes)]

    return axes
