"""The record of an Imaris file: its groups and attributes, without voxels.

The layout is the one of the Imaris file format description,
"Imaris5.5 File Format Description (IMS)", which Oxford Instruments
publishes as `doc/Imaris5FileFormat.pdf` in
https://github.com/imaris/ImarisWriter (commit `b128e6e`). The details
that the description leaves out are taken from the writer of the same
repository, `writer/bpWriterHDF5.cxx`, and are cited where they are used.
"""

# stdlib
import re

# dependencies
import h5py
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, Magic, replace

# internals
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
)
from brainhops.io.common.hdf5 import Hdf5Reader

_DATASET = "DataSet"
"""The root group that holds the voxels of every level."""

_DATASET_INFO = "DataSetInfo"
"""The root group whose subgroups hold the parameters of the image."""

_LEVEL = "ResolutionLevel "
_TIME = "TimePoint "
_CHANNEL = "Channel "

_SIZES = ("ImageSizeX", "ImageSizeY", "ImageSizeZ")
"""The attributes of a `Channel` group that give the size of the image."""

_EXTENT_MIN = ("ExtMin0", "ExtMin1", "ExtMin2")
_EXTENT_MAX = ("ExtMax0", "ExtMax1", "ExtMax2")


class ImarisLevel(Magic, frozen=True):
    """One resolution level of an Imaris file, as its groups describe it.

    Each time point and each channel of a level is stored as its own
    three-dimensional `Data` dataset, whose shape is a whole number of
    chunks. The size of the image, which the padding hides, is recorded
    in the attributes of the `Channel` group. A level is frozen, and two
    levels are equal when their fields are.
    """

    size: tx.Tuple[int, int, int]
    """The size of the image along x, y and z, from `ImageSizeX`,
    `ImageSizeY` and `ImageSizeZ`."""

    stored: tx.Tuple[int, int, int]
    """The shape of each `Data` dataset, which is padded to whole chunks
    and ordered z, y, x."""

    datatype: str
    """The NumPy type string of the voxels, such as `"<u2"`."""

    @property
    def dtype(self) -> np.dtype:
        """The NumPy type of the voxels."""
        return np.dtype(self.datatype)


class ImarisRaw(Magic, Hdf5Reader, frozen=True):
    """The groups and attributes of an Imaris file, as brainhops reads them.

    An Imaris file (`.ims`) is an HDF5 file whose group
    `/DataSet/ResolutionLevel R/TimePoint T/Channel C` holds the voxels of
    one channel at one time point and one resolution level, and whose
    group `/DataSetInfo` holds the parameters of the image as text. The
    record holds the version of the file, the size and the stored shape
    of each level, the number of time points and channels, and every
    parameter of the subgroups of `/DataSetInfo`, such as the extent of
    the image, which places it in space. The histograms and the thumbnail
    are not read.

    The record never reads the voxels, so it is the same for a file whose
    `Data` datasets cannot be read. It is held by
    [`ImarisMetadata`][brainhops.io.common.imaris.ImarisMetadata]. A
    record is frozen, and two records are equal when their fields are.
    """

    version: tx.Optional[str] = None
    """The `ImarisVersion` attribute of the root, such as `"5.5.0"`."""

    levels: tx.Tuple[ImarisLevel, ...] = ()
    """The resolution levels, finest first."""

    timepoints: int = 1
    """The number of time points, which every level holds."""

    channels: int = 1
    """The number of channels, which every time point holds."""

    info: tx.Dict[str, tx.Dict[str, str]] = Factory(dict)
    """The parameters of the image, by subgroup of `/DataSetInfo`.

    Each subgroup, such as `"Image"`, `"Channel 0"` or `"TimeInfo"`,
    gives a dictionary from the name of each parameter to its text.
    Imaris stores a parameter as an attribute that holds an array of
    single characters, and ImarisWriter stores a parameter longer than
    1024 characters as a dataset instead, under a name in which `%`
    becomes `%p` and `/` becomes `%s` (`EncodeName` in
    `writer/bpWriterHDF5.cxx`). Both are read as text, under their
    original name.
    """

    @property
    def extent_min(self) -> tx.Tuple[float, float, float]:
        """The corner of the image with the smallest coordinates, along x,
        y and z.

        It is read from the `ExtMin0`, `ExtMin1` and `ExtMin2` parameters
        of the `"Image"` subgroup. The description of the format says that
        the extent of the image lies on the outer border of the voxels on
        its edge, not at their centers.
        """
        return _extent(self.info, _EXTENT_MIN)

    @property
    def extent_max(self) -> tx.Tuple[float, float, float]:
        """The corner of the image with the largest coordinates, along x,
        y and z.

        It is read from the `ExtMax0`, `ExtMax1` and `ExtMax2` parameters
        of the `"Image"` subgroup.
        """
        return _extent(self.info, _EXTENT_MAX)

    @property
    def unit(self) -> tx.Optional[str]:
        """The `Unit` parameter of the `"Image"` subgroup, or `None`.

        The description of the format lists `"m"`, `"mm"`, `"um"` and
        `"nm"`.
        """
        return self.info.get("Image", {}).get("Unit")

    def copy(self) -> "ImarisRaw":
        """Return a copy that can be changed without changing this record.

        The dictionaries of parameters are copied. The levels are frozen,
        so the copy shares them.

        Returns
        -------
        ImarisRaw
            The copy.
        """
        info = {name: dict(group) for name, group in self.info.items()}
        return replace(self, info=info)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Return the confidence that an open HDF5 file is an Imaris file.

        The file is an Imaris file when it has the groups `/DataSetInfo`
        and `/DataSet/ResolutionLevel 0`, which the description of the
        format requires.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an Imaris file.
            With `True`, the error is a `SnifferContentError`, and an
            exception class is raised instead when one is given.

        Returns
        -------
        float
            The confidence, between 0 and 1.

        Raises
        ------
        SnifferContentError
            If the file is not an Imaris file and `error` is `True`.
        """
        if is_imaris(h5file):
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error(
                "HDF5 file is not an Imaris file: it has no /DataSetInfo "
                "or no /DataSet/ResolutionLevel 0 group."
            )
        return Confidence.NO

    # --- read ---------------------------------------------------------

    @classmethod
    def from_h5(cls, h5file: h5py.File, **kwargs) -> "ImarisRaw":
        """Read the record of an open Imaris file, without its voxels.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        ImarisRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the file is not an Imaris file that brainhops can read.
        """
        no_options(kwargs)
        return read_imaris(h5file)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def is_imaris(h5file: h5py.File) -> bool:
    """Return whether an open HDF5 file has the groups of an Imaris file.

    Parameters
    ----------
    h5file : h5py.File
        The open file.

    Returns
    -------
    bool
        Whether the file has the groups `/DataSetInfo` and
        `/DataSet/ResolutionLevel 0`.
    """
    dataset = h5file.get(_DATASET)
    return (
        isinstance(h5file.get(_DATASET_INFO), h5py.Group)
        and isinstance(dataset, h5py.Group)
        and isinstance(dataset.get(_LEVEL + "0"), h5py.Group)
    )


def read_imaris(h5file: h5py.File) -> ImarisRaw:
    """Read the record of an open Imaris file, without its voxels.

    The levels, time points and channels are the groups numbered from 0
    without a gap. Every level must hold the same number of time points
    and channels. The size and the type of a level are read from its
    first time point and channel, since Imaris stores one image whose
    channels and time points share them.

    Parameters
    ----------
    h5file : h5py.File
        The open file.

    Returns
    -------
    ImarisRaw
        The record.

    Raises
    ------
    ParserContentError
        If the file is not an Imaris file, or a group or an attribute that
        the description of the format requires is missing or invalid.
    """
    if not is_imaris(h5file):
        raise ParserContentError(
            "Not an Imaris file: no /DataSetInfo or no "
            "/DataSet/ResolutionLevel 0 group."
        )
    groups = _numbered(h5file[_DATASET], _LEVEL)
    times = _numbered(groups[0], _TIME)
    if not times:
        raise ParserContentError(
            "The Imaris file has no group /DataSet/ResolutionLevel 0/"
            "TimePoint 0."
        )
    channels = _numbered(times[0], _CHANNEL)
    if not channels:
        raise ParserContentError(
            "The Imaris file has no group /DataSet/ResolutionLevel 0/"
            "TimePoint 0/Channel 0."
        )
    levels = tuple(
        _read_level(group, len(times), len(channels)) for group in groups
    )
    info = _read_info(h5file[_DATASET_INFO])
    try:
        low, high = _extent(info, _EXTENT_MIN), _extent(info, _EXTENT_MAX)
    except ValueError as e:
        raise ParserContentError(str(e)) from None
    # The description says that the extent is a bounding box whose
    # maximum is above its minimum, even for a single slice, and
    # ImarisWriter writes it so (`AdaptExtents`).
    if any(b <= a for a, b in zip(low, high)):
        raise ParserContentError(
            f"The Imaris extent goes from {low} to {high}, which is not a "
            f"box of positive size."
        )
    return ImarisRaw(
        version=attribute_text(h5file.attrs.get("ImarisVersion")),
        levels=levels,
        timepoints=len(times),
        channels=len(channels),
        info=info,
    )


def attribute_text(value: tx.Any) -> tx.Optional[str]:
    """Decode the text of an Imaris attribute, whatever its storage.

    Imaris stores text as an array of single characters (`H5T_C_S1` with
    one element per character, `WriteAttribute` in
    `writer/bpWriterHDF5.cxx`), so the characters are joined, and the
    text stops at the first null character, which ends a C string. Text
    stored as one string, as h5py writes it, is read as well, and numbers
    are written out separated by spaces. The description of the format
    does not give an encoding, so the bytes are decoded as UTF-8, or as
    Latin-1 when they are not valid UTF-8.

    Parameters
    ----------
    value : Any
        The value of the attribute, or a dataset that holds it.

    Returns
    -------
    str or None
        The text, or `None` when the value is `None`.
    """
    if value is None:
        return None
    if isinstance(value, h5py.Dataset):
        value = value[()]
    array = np.asarray(value)
    if array.dtype.kind == "S":
        return _decode(array.tobytes())
    if array.dtype.kind in "OU":
        parts = [
            _decode(bytes(item)) if isinstance(item, bytes) else str(item)
            for item in array.ravel().tolist()
        ]
        return "".join(parts).split("\x00", 1)[0]
    return " ".join(str(item) for item in array.ravel().tolist())


def no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options of a reader that takes none.

    Parameters
    ----------
    kwargs : Mapping[str, Any]
        The options given to the reader.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            f"Unknown Imaris reader option(s): {', '.join(sorted(kwargs))}"
        )


def _decode(content: bytes) -> str:
    """Decode the bytes of a C string, up to its first null character."""
    content = content.split(b"\x00", 1)[0]
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _numbered(group: h5py.Group, prefix: str) -> tx.List[h5py.Group]:
    """Return the subgroups `<prefix>0`, `<prefix>1` and so on, up to the
    first number that is missing."""
    out = []
    while True:
        child = group.get(f"{prefix}{len(out)}")
        if not isinstance(child, h5py.Group):
            return out
        out.append(child)


def _read_level(group: h5py.Group, times: int, channels: int) -> ImarisLevel:
    """Read one resolution level.

    Raises
    ------
    ParserContentError
        If the level does not hold every time point and channel, or its
        sizes or its `Data` dataset are invalid.
    """
    name = group.name
    if len(_numbered(group, _TIME)) != times:
        raise ParserContentError(
            f"The Imaris level {name} does not hold {times} time points, "
            f"as the first level does."
        )
    for index in range(times):
        found = len(_numbered(group[f"{_TIME}{index}"], _CHANNEL))
        if found != channels:
            raise ParserContentError(
                f"The time point {index} of the Imaris level {name} holds "
                f"{found} channels instead of {channels}."
            )
    channel = group[f"{_TIME}0/{_CHANNEL}0"]
    try:
        size = tuple(int(attribute_text(channel.attrs[n])) for n in _SIZES)
    except (KeyError, TypeError, ValueError):
        raise ParserContentError(
            f"The Imaris group {channel.name} has no valid ImageSizeX, "
            f"ImageSizeY and ImageSizeZ attributes."
        ) from None
    data = channel.get("Data")
    if not isinstance(data, h5py.Dataset) or data.ndim != 3:
        raise ParserContentError(
            f"The Imaris group {channel.name} has no three-dimensional Data "
            f"dataset."
        )
    if data.dtype.kind not in "uif":
        raise ParserContentError(
            f"The Imaris dataset {data.name} holds {data.dtype}, which is "
            f"not a number type."
        )
    stored = tuple(int(n) for n in data.shape)
    if any(s < 1 for s in size) or any(
        n > s for n, s in zip(reversed(size), stored)
    ):
        raise ParserContentError(
            f"The Imaris image size {size} (x, y, z) of {channel.name} does "
            f"not fit in its Data dataset of shape {stored} (z, y, x)."
        )
    return ImarisLevel(size=size, stored=stored, datatype=data.dtype.str)


def _read_info(group: h5py.Group) -> tx.Dict[str, tx.Dict[str, str]]:
    """Read the parameters of every subgroup of `/DataSetInfo` as text.

    A parameter that ImarisWriter stored as a one-dimensional dataset of
    characters, because it is long, is read under its decoded name.
    """
    info = {}
    for name, child in group.items():
        if not isinstance(child, h5py.Group):
            continue
        params = {
            key: attribute_text(value) for key, value in child.attrs.items()
        }
        for key, item in child.items():
            if (
                isinstance(item, h5py.Dataset)
                and item.ndim == 1
                and item.dtype.kind == "S"
            ):
                params[_decode_name(key)] = attribute_text(item)
        info[name] = params
    return info


def _decode_name(name: str) -> str:
    """Undo `EncodeName` of `writer/bpWriterHDF5.cxx`, which writes `%` as
    `%p` and `/` as `%s`."""
    return re.sub(
        r"%([ps])", lambda m: "%" if m.group(1) == "p" else "/", name
    )


def _extent(
    info: tx.Mapping[str, tx.Mapping[str, str]], names: tx.Sequence[str]
) -> tx.Tuple[float, float, float]:
    """Read three coordinates of the extent from the `"Image"` parameters.

    Raises
    ------
    ValueError
        If a coordinate is missing or is not a finite number.
    """
    image = info.get("Image", {})
    try:
        values = tuple(float(image[name]) for name in names)
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"The Imaris file has no valid {', '.join(names)} attributes "
            f"in /DataSetInfo/Image."
        ) from None
    if not all(np.isfinite(values)):
        raise ValueError(f"The Imaris extent {values} is not finite.")
    return values
