"""State of a multiscale image (a pyramid) that a format compares on write.

A pyramid format holds one record for its whole container, such as the
group document of an OME-Zarr store, which describes the pyramid and each
of its levels. The pyramid and every one of its levels hold the same
metadata object, by identity. Each level is an image of the single-scale
format of the same package, which holds the lazy array of its level in
`rawdata` and the index `level` of the level in the record. The pyramid holds
the tuple of the level arrays in `rawdata`, and it decodes its levels
from `rawdata` and the record on first access. The memo
`spec/design/format-records.md` describes this design in its section on
pyramids.

When a pyramid is consolidated or written, its format must tell which
levels are still the levels of its file, so that it keeps their arrays
and their descriptions as they are stored, and encodes only the others.
The helpers of this module answer that question. They hold no state and
take what is specific to a format, the class of its levels and the
function that decodes the transformations of a level, as arguments, so
that each format calls them explicitly from its own `consolidate`.
"""

__all__ = ["held_images", "own_level", "own_levels"]

import typing_extensions as tx

from brainhops._core._internal_api import cached_value
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Transformation
from brainhops.io.common._geometry import (
    held_transformations,
    same_transformations,
)

LevelDecoder = tx.Callable[[tx.Any, int], tx.List[Transformation]]
"""Function that decodes the transformations of a level from metadata.

It receives the metadata of the pyramid and the index of the level in its
record, and returns the list that a level of that index decodes to.
"""


def held_images(pyramid: tx.Any) -> tx.Optional[tx.List[SingleScaleImage]]:
    """Return the levels that a pyramid holds, or `None`.

    A pyramid format takes the `images` field of its model over as
    `_images`, which holds an assigned list only, and its `images`
    property decodes the levels from `rawdata` and the record on first access
    and caches them apart. The levels that the pyramid holds are the
    assigned list, or else the cached list, whose levels may have been
    changed in place, for example by assigning their `data`. `None`
    means that the levels were neither assigned nor read, so every level
    is kept as it is stored.

    Parameters
    ----------
    pyramid : MultiScaleImage
        A pyramid whose `images` is a cached `smartproperty`.

    Returns
    -------
    list of SingleScaleImage or None
        The assigned list, or else the decoded list, or `None`.
    """
    if pyramid._images is not None:
        return pyramid._images
    return cached_value(pyramid, "images")


def own_level(
    pyramid: tx.Any,
    image: tx.Any,
    index: int,
    level_type: type,
    decode: LevelDecoder,
) -> bool:
    """Tell whether an image is still a level of the file of a pyramid.

    An image is level `index` of the file of a pyramid when it is an
    object of the level class of the format, it holds the metadata of the
    pyramid, by identity, at the same index, its `rawdata` is the array
    of that level in the `rawdata` of the pyramid, by identity, and
    nothing is pending in it: no data was assigned, and its
    transformations were neither read nor assigned, or are the same as a
    fresh decode of the record. Its array and its description can then
    be kept as they are stored, byte for byte. The test reads no data.

    Parameters
    ----------
    pyramid : MultiScaleImage
        The pyramid, which holds `rawdata` as a tuple of level arrays and its
        metadata.
    image : SingleScaleImage
        The image held by the pyramid at `index`.
    index : int
        The position of the image in the levels of the pyramid.
    level_type : type
        The level class of the format. An object of a subclass is not a
        level of the file, because its fields may mean something else.
    decode : callable
        The function that decodes the transformations of a level from
        the metadata of the pyramid and the index of the level.

    Returns
    -------
    bool
        Whether `image` is level `index` of the file of `pyramid`.
    """
    raw = pyramid.rawdata
    metadata = pyramid.metadata
    if type(image) is not level_type or raw is None or metadata is None:
        return False
    if not 0 <= index < len(raw):
        return False
    if (
        image.metadata is not metadata
        or image.level != index
        or image.rawdata is not raw[index]
        or image._data is not None
    ):
        return False
    held = held_transformations(image)
    return held is None or same_transformations(held, decode(metadata, index))


def own_levels(
    pyramid: tx.Any, level_type: type, decode: LevelDecoder
) -> bool:
    """Tell whether every level that a pyramid holds is a level of its file.

    The levels are kept when they were neither assigned nor read, or when
    the pyramid holds as many levels as its `rawdata` holds arrays and each of
    them is the level of its file at its position, as [`own_level`][]
    tells. A pyramid for which this is true, whose placements are kept,
    and which is consolidated without options, keeps its `rawdata` and
    its metadata.

    Parameters
    ----------
    pyramid : MultiScaleImage
        The pyramid, which holds `rawdata` as a tuple of level arrays and its
        metadata.
    level_type : type
        The level class of the format.
    decode : callable
        The function that decodes the transformations of a level from
        the metadata of the pyramid and the index of the level.

    Returns
    -------
    bool
        Whether the levels of `pyramid` are those of its file.
    """
    images = held_images(pyramid)
    if images is None:
        return True
    if pyramid.rawdata is None or len(images) != len(pyramid.rawdata):
        return False
    return all(
        own_level(pyramid, image, index, level_type, decode)
        for index, image in enumerate(images)
    )
