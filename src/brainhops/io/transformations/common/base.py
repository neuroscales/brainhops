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

    @property
    def image(self) -> tx.Optional[nb.Nifti1Image]:  # noqa: F811
        """The NIfTI image from which the transformation was derived."""
        if getattr(self, "_image", None) is not None:
            return self._image
        return None

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
            return cls(image=nifti)
        return cls.from_nifti(nb.load(nifti))

    @classmethod
    def sniff(cls, other: _NiftiLike) -> bool:
        """
        Check whether `other` looks like a valid NIfTI image, header,
        path, or byte stream, without fully loading/validating it.
        """
        if nb is None:
            return False
        if isinstance(
            other,
            (nb.Nifti1Header, nb.Nifti2Header, nb.Nifti1Image, nb.Nifti2Image),
        ):
            return True
        if isinstance(other, bytes):
            return cls.sniff_bytes(other)
        if isinstance(other, (str, PathLike)):
            return cls.sniff_file(other)
        if hasattr(other, "read"):
            pos = other.tell() if hasattr(other, "tell") else None
            try:
                return cls.sniff_bytes(other.read(352))
            finally:
                if pos is not None:
                    other.seek(pos)
        return False

    @classmethod
    def sniff_file(cls, fileobj: tx.Union[str, PathLike]) -> bool:
        """
        Check whether the file at `fileobj` looks like a valid NIfTI
        file (NIfTI-1 or NIfTI-2), without fully loading it.

        Transparently handles gzip-compressed files (`.nii.gz`) via
        nibabel's `ImageOpener`, same as `nb.load` would.
        """
        if nb is None:
            return False
        try:
            with nb.openers.ImageOpener(str(fileobj), "rb") as f:
                # NIfTI-1/2 headers are 348/540 bytes; a few extra
                # bytes of slack doesn't hurt.
                binaryblock = f.read(552)
        except OSError:
            return False
        return cls.sniff_bytes(binaryblock)

    @classmethod
    def sniff_bytes(cls, data: bytes) -> bool:
        """
        Check whether `data` looks like the start of a valid NIfTI-1
        or NIfTI-2 header (magic bytes + expected size), without
        fully parsing it.
        """
        if nb is None:
            return False
        return nb.Nifti1Header.may_contain_header(
            data
        ) or nb.Nifti2Header.may_contain_header(data)
