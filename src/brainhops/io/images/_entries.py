"""
The image formats, declared ahead of import.

Dispatch finds a format through its entry here, and imports the format's
module only when it needs the format (see `brainhops.io.base._registry`).
An entry must say what its class says -- its hints, extensions, prefixes,
priority and dispatchers -- which `tests/test_io_registry_entries.py`
checks, and prints the line to write when they disagree.

It imports nothing but the registry, so that importing
`brainhops.io.images` imports neither the data model nor any format.
"""

# stdlib
import itertools

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.dependencies import has_abczarr_driver
from brainhops.io.base._registry import (
    IMAGE,
    OBJECT,
    WRITABLE,
    WRITABLE_IMAGE,
    FormatEntry,
    declare,
)


def _image(
    module: str,
    qualname: str,
    hints: str,
    extensions: tx.Tuple[str, ...],
    writable: bool = True,
    priority: int = 10,
    **kwargs,
) -> FormatEntry:
    """An image format, of a module of `brainhops.io.images`."""
    if writable:
        dispatchers = (WRITABLE_IMAGE, IMAGE, WRITABLE, OBJECT)
    else:
        dispatchers = (IMAGE, OBJECT)
    return FormatEntry(
        f"brainhops.io.images.{module}",
        qualname,
        hints,
        extensions,
        priority=priority,
        dispatchers=dispatchers,
        **kwargs,
    )


# The vendors OpenSlide reads, each with a single-scale and a multiscale
# format: name, hints, extensions.
_OPENSLIDE = (
    ("Aperio", "aperio svs", (".svs",)),
    ("Hamamatsu", "hamamatsu ndpi vms vmu", (".ndpi", ".vms", ".vmu")),
    ("Mirax", "mirax mrxs 3dhistech", (".mrxs",)),
    ("Leica", "leica scn", (".scn",)),
    ("Philips", "philips", (".tiff",)),
    ("Ventana", "ventana bif", (".bif", ".tif")),
    ("Sakura", "sakura svslide", (".svslide",)),
    ("Trestle", "trestle", (".tif",)),
    ("Zeiss", "zeiss czi", (".czi",)),
    ("DicomWsi", "dicom-wsi", (".dcm",)),
    ("GenericTiff", "generic-tiff", (".tif", ".tiff")),
)

_PILLOW = (
    ".png", ".apng", ".jpg", ".jpeg", ".jpe", ".jfif", ".mpo", ".bmp",
    ".dib", ".gif", ".webp", ".ppm", ".pgm", ".pbm", ".pnm", ".pfm",
    ".jp2", ".j2k", ".jpx", ".jpf", ".j2c", ".tga", ".pcx", ".qoi",
)  # fmt: skip
_TIFF = (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".btf", ".tf8", ".tf2")

_NIBABEL = {"requires": ("nibabel",)}
_MINC = {"requires": ("nibabel",), "extra": "minc"}
# The Zarr formats need abczarr and at least one of its backend drivers:
# abczarr alone cannot open a store. Which drivers are present is only
# known by importing abczarr, which is left to the first dispatch.
_ZARR = {
    "requires": ("abczarr",),
    "extra": "zarr",
    "check": has_abczarr_driver,
}


def _openslide() -> tx.List[FormatEntry]:
    """The single-scale and multiscale formats of each OpenSlide vendor,
    which also answer to their hints qualified by `openslide`."""
    entries = []
    for (name, words, extensions), multiscale in itertools.product(
        _OPENSLIDE, (False, True)
    ):
        words = words.split()
        hints = ["openslide", *words, *(f"openslide.{w}" for w in words)]
        entries.append(
            _image(
                "openslide._formats",
                f"{name}MultiScaleImage" if multiscale else f"{name}Image",
                " ".join(hints),
                extensions,
                writable=False,
                priority=11 if multiscale else 10,
                requires=("openslide-python",),
                extra="openslide",
            )
        )
    return entries


declare(
    [
        _image(
            "afni._image",
            "AfniImage",
            "afni afni.brik brik",
            (".HEAD", ".BRIK", ".BRIK.gz", ".BRIK.bz2")
            + (".head", ".brik", ".brik.gz", ".brik.bz2"),
        ),
        _image(
            "mrtrix._image",
            "MrtrixImage",
            "mrtrix",
            (".mif", ".mif.gz", ".mih"),
        ),
        _image(
            "nrrd._image",
            "AttachedNrrdImage",
            "attached nrrd nrrd.attached",
            (".nrrd",),
        ),
        _image(
            "nrrd._image",
            "DetachedNrrdImage",
            "detached nhdr nrrd nrrd.detached nrrd.nhdr",
            (".nhdr",),
        ),
        _image(
            "freesurfer.mgh._image",
            "MghImage",
            "freesurfer freesurfer.mgh freesurfer.mgz mgh mgz",
            (".mgh", ".mgz", ".mgh.gz"),
            extra="mgh",
            **_NIBABEL,
        ),
        _image(
            "minc._image",
            "Minc1Image",
            "minc minc1 1 minc.minc1 minc.1",
            (".mnc", ".mnc.gz"),
            writable=False,
            **_MINC,
        ),
        # MINC2 is an HDF5 file, read only with h5py.
        _image(
            "minc._image",
            "Minc2Image",
            "minc minc2 2 minc.minc2 minc.2",
            (".mnc",),
            writable=False,
            requires=("nibabel", "h5py"),
            extra="minc",
        ),
        _image(
            "nifti._image",
            "NiftiImage",
            "nifti",
            (".nii", ".nii.gz"),
            extra="nifti",
            **_NIBABEL,
        ),
        # Raster images (PNG, JPEG, ...) are read and written with Pillow.
        _image(
            "pillow._image",
            "PillowImage",
            "pillow",
            _PILLOW,
            requires=("Pillow",),
            extra="pillow",
        ),
        # TIFF images are read and written with tifffile. Without it,
        # Pillow (whose TIFF sniff is weaker) reads them as raster images.
        *(
            _image(
                module,
                qualname,
                "tiff tifffile",
                _TIFF,
                requires=("tifffile",),
                extra="tiff",
            )
            for module, qualname in (
                ("tiff._image", "TiffImage"),
                ("tiff._multiscale", "TiffMultiScaleImage"),
            )
        ),
        # Whole-slide images are read with OpenSlide (openslide-python and
        # the OpenSlide library). Without it, the TIFF-based slides are
        # read by the TIFF reader.
        *_openslide(),
        _image("zarr._image", "ZarrImage", "zarr", (".zarr",), **_ZARR),
        _image(
            "zarr._multiscale",
            "OmeZarrImage",
            "zarr",
            (".zarr", ".ome.zarr"),
            **_ZARR,
        ),
    ]
)
