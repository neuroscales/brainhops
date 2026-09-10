# stdlib
from os import PathLike

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel import transformations as _xforms

# optionals
if tx.TYPE_CHECKING:
    import nibabel as nb
else:
    try:
        import nibabel as nb
    except ImportError:

        class nb:
            Nifti1Header = None
            Nifti1Image = None


# typing
_NiftiLike = tx.Union[
    nb.Nifti1Header, nb.Nifti1Image, str, PathLike, tx.BinaryIO
]


class NiftiBasedTransformation(_xforms.Transformation):
    header: tx.Optional[nb.Nifti1Header] = None
    image: tx.Optional[nb.Nifti1Image] = None

    @property
    def header(self) -> tx.Optional[nb.Nifti1Header]:  # noqa: F811
        """The NIfTI header from which the transformation was derived."""
        if getattr(self, "_header", None) is not None:
            return self._header
        if getattr(self, "_image", None) is not None:
            return self._image.header
        return None

    @header.setter
    def header(self, value: nb.Nifti1Header) -> None:
        self._header = value

    @property
    def image(self) -> tx.Optional[nb.Nifti1Image]:  # noqa: F811
        """The NIfTI image from which the transformation was derived."""
        if getattr(self, "_image", None) is not None:
            return self._image
        return None

    @image.setter
    def image(self, value: nb.Nifti1Image) -> None:
        self._image = value

    @classmethod
    def from_(cls, other: _NiftiLike) -> tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI image."""
        return cls.from_nifti(other)

    @classmethod
    def from_file(cls, fileobj: tx.Union[str, PathLike]) -> tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI file."""
        return cls.from_nifti(nb.load(fileobj))

    @classmethod
    def from_bytes(cls, data: bytes) -> tx.Self:
        """
        Create a NiftiVoxelToRAS transformation from a NIfTI file in bytes.
        """
        return cls.from_nifti(nb.Nifti1Image.from_bytes(data))

    @classmethod
    def from_nifti(cls, nifti: _NiftiLike) -> tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI image."""
        if isinstance(nifti, nb.Nifti1Header):
            return cls(header=nifti)
        if isinstance(nifti, nb.Nifti1Image):
            return cls(image=nifti, header=nifti.header)
        return cls.from_nifti(nb.load(nifti))
