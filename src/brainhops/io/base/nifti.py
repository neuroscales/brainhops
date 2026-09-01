# stdlib
from os import PathLike

# dependencies
import typing_extensions as _tx

# internals
from brainhops.io.base.parsers import FileParser

# optionals
if _tx.TYPE_CHECKING:
    import nibabel as nb

    # typing
    _NiftiLike = _tx.Union[
        nb.Nifti1Header, nb.Nifti1Image, str, PathLike, _tx.BinaryIO
    ]
else:
    try:
        import nibabel as nb

        _NiftiLike = _tx.Union[
            nb.Nifti1Header, nb.Nifti1Image, str, PathLike, _tx.BinaryIO
        ]
    except ImportError:
        nb = None
        _NiftiLike = _tx.Any


class NiftiBasedParser(FileParser):

    header: _tx.Optional[nb.Nifti1Header] = None
    image: _tx.Optional[nb.Nifti1Image] = None

    @property
    def header(self) -> _tx.Optional[nb.Nifti1Header]:
        """The NIfTI header from which the transformation was derived."""
        if getattr(self, "_header", None) is not None:
            return self._header
        if getattr(self, "_image", None) is not None:
            return self._image.header
        return None

    @property
    def image(self) -> _tx.Optional[nb.Nifti1Image]:
        """The NIfTI image from which the transformation was derived."""
        if getattr(self, "_image", None) is not None:
            return self._image
        return None

    @classmethod
    def from_(cls, other: _NiftiLike) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI image."""
        return cls.from_nifti(other)

    @classmethod
    def from_file(cls, fileobj: _tx.Union[str, PathLike]) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI file."""
        return cls.from_nifti(nb.load(fileobj))

    @classmethod
    def from_bytes(cls, data: bytes) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI file in bytes."""
        return cls.from_nifti(nb.Nifti1Image.from_bytes(data))

    @classmethod
    def from_nifti(cls, nifti: _NiftiLike) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI image."""
        if isinstance(nifti, nb.Nifti1Header):
            return cls(header=nifti)
        if isinstance(nifti, nb.Nifti1Image):
            return cls(image=nifti)
        return cls.from_nifti(nb.load(nifti))
