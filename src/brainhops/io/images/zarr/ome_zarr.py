import io
from os import PathLike

import dask.array as da
import numpy as np
import typing_extensions as _tx

from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import Image, MultiImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Identity,
    Scaling,
    Sequence,
    Transformation,
    Translation,
)

# optionals
if _tx.TYPE_CHECKING:
    import zarr
else:
    try:
        import zarr
    except ImportError:
        zarr = None

# typing
_OmeZarrLike = _tx.Union[
    "zarr.Group",
    str,
    PathLike,
]


class OmeZarrImage(MultiImage):
    group: _tx.Optional[zarr.Group] = None

    @property
    def _multiscale(self) -> _tx.Optional[dict]:
        if self.group is None:
            return None
        metadata = dict(self.group.attrs)
        # NGFF 0.5+ (zarr v3) nests everything under "ome"; earlier
        # versions (zarr v2, .zattrs) put `multiscales` at the top level.
        ome = metadata.get("ome", metadata)
        multiscales = ome.get("multiscales")
        if not multiscales:
            return None
        return multiscales[0]

    @property
    def _axes(self) -> _tx.Optional[_tx.List[Axis]]:
        """The Axis objects declared in this multiscale's metadata."""
        multiscale = self._multiscale
        if multiscale is None:
            return None
        return [self._axis_from_ngff(a) for a in multiscale["axes"]]

    @classmethod
    def from_(cls, other: _OmeZarrLike) -> _tx.Self:
        """Create an OmeZarrImage from a zarr group or a path to one."""
        return cls.from_zarr(other)

    @classmethod
    def from_file(
        cls, path: _tx.Union[str, PathLike]
    ) -> _tx.Self:
        """Create an OmeZarrImage from a path to an OME-Zarr store."""
        cls._require_zarr()
        return cls.from_zarr(zarr.open_group(str(path), mode="r"))

    @classmethod
    def from_bytes(cls, data: bytes) -> _tx.Self:
        """
        Create an OmeZarrImage from an OME-Zarr store made from byes
        """
        cls._require_zarr()
        store = zarr.storage.ZipStore(io.BytesIO(data), mode="r")
        return cls.from_zarr(zarr.open_group(store=store, mode="r"))

    @classmethod
    def from_zarr(cls, source: _OmeZarrLike) -> _tx.Self:
        """Create an OmeZarrImage from a zarr group, or a path to one."""
        cls._require_zarr()
        if isinstance(source, (str, PathLike)):
            return cls.from_zarr(
                zarr.open_group(str(source), mode="r")
            )
        if isinstance(source, zarr.Group):
            return cls(group=source)
        raise TypeError(
            f"Unsupported source type for OME-Zarr image: {type(source)}"
        )

    @classmethod
    def sniff(cls, other: _OmeZarrLike) -> bool:
        """
        Check whether `other` looks like a valid OME-Zarr store --
        i.e. a zarr group whose attributes declare NGFF `multiscales`
        metadata (in either the flat or NGFF-0.5 `"ome"`-nested form)
        -- without fully loading it as an `OmeZarrImage`.

        A plain zarr store with no OME metadata at the given node
        will correctly sniff as `False`, even though it opens fine as
        a zarr group.
        """
        if zarr is None:
            return False
        if isinstance(other, zarr.Group):
            return cls._looks_like_ome_zarr(other)
        if isinstance(other, bytes):
            return cls.sniff_bytes(other)
        if isinstance(other, (str, PathLike)):
            return cls.sniff_file(other)
        return False

    @classmethod
    def sniff_file(cls, path: _tx.Union[str, PathLike]) -> bool:
        """
        Check whether the store at `path` looks like a valid OME-Zarr
        store, without fully loading it.
        """
        if zarr is None:
            return False
        try:
            group = zarr.open_group(str(path), mode="r")
        except Exception:
            # not a zarr store at all (missing, not a group, etc.)
            return False
        return cls._looks_like_ome_zarr(group)

    @classmethod
    def sniff_bytes(cls, data: bytes) -> bool:
        """
        Check whether `data` (an OME-Zarr store packaged as a zip
        archive, e.g. a `.ome.zarr.zip` file) looks like a valid
        OME-Zarr store, without fully loading it.
        """
        if zarr is None:
            return False
        try:
            store = zarr.storage.ZipStore(io.BytesIO(data), mode="r")
            group = zarr.open_group(store=store, mode="r")
        except Exception:
            # not a valid zip, or not a zarr group once unzipped
            return False
        return cls._looks_like_ome_zarr(group)

    def _looks_like_ome_zarr(group: "zarr.Group") -> bool:
        """Does `group`'s own attrs declare NGFF `multiscales` metadata?"""
        try:
            metadata = dict(group.attrs)
            ome = metadata.get("ome", metadata)
            return bool(ome.get("multiscales"))
        except Exception:
            return False

    @property
    def images(self) -> _tx.List[Image]:
        if getattr(self, "_images", None) is None:
            if self.group is None or self._multiscale is None:
                self._images = None
            else:
                self._images = [
                    Image(data=da.from_array(self.group[ds["path"]]),
                          transformations=self._transform_from_multiscale(
                              self._multiscale, i))
                    for i, ds in enumerate(self._multiscale["datasets"])
                ]
        return self._images

    @images.setter
    def images(self, value: _tx.Optional[_tx.List[Image]]) -> None:
        self._images = value

    # --- helpers -----------------------------------------------------

    @staticmethod
    def _require_zarr() -> None:
        if zarr is None:
            raise ImportError(
                "The `zarr` package is required to read OME-Zarr images. "
                "Install it with `pip install zarr`."
            )

    @staticmethod
    def _axis_from_ngff(axis_meta: dict) -> Axis:
        """Build an Axis from one entry of axis metadata."""
        return Axis(
            name=axis_meta["name"],
            type=axis_meta.get("type"),
            unit=axis_meta.get("unit"),
        )

    @classmethod
    def _transform_from_multiscale(
        cls, multiscale: dict, level: int
    ) -> Transformation:
        """
        Build the Transformation for one resolution level of multiscales entry.
        """
        axes = [cls._axis_from_ngff(a) for a in multiscale["axes"]]
        voxel_space = CoordinateSystem(axes=axes)
        world_space = CoordinateSystem(axes=axes)

        dataset = multiscale["datasets"][level]
        # a multiscale-wide transform (applied to every level) may also
        # be declared alongside each per-dataset one; apply it first.
        raw_transforms = (
            list(multiscale.get("coordinateTransformations", []))
            + list(dataset.get("coordinateTransformations", []))
        )

        pieces: _tx.List[Transformation] = []
        for t in raw_transforms:
            kind = t.get("type")
            if kind == "scale":
                pieces.append(Scaling(
                    scale=np.asarray(t["scale"], dtype=float),
                    input=voxel_space,
                    output=world_space,
                ))
            elif kind == "translation":
                pieces.append(Translation(
                    translation=np.asarray(t["translation"], dtype=float),
                    input=voxel_space,
                    output=world_space,
                ))
            elif kind == "identity":
                continue
            else:
                raise NotImplementedError(
                    "Unsupported NGFF coordinateTransformation type: "
                    f"{kind!r}. Only 'scale' and 'translation' are "
                    "currently handled."
                )

        if not pieces:
            return Identity(input=voxel_space, output=world_space)
        if len(pieces) == 1:
            return pieces[0]
        return Sequence(
            transformations=pieces, input=voxel_space, output=world_space
        )
