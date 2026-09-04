import typing_extensions as _tx
from numpy.typing import ArrayLike

# datamodel
from brainhops._core.backends import get_ndimage_backend
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base.omezarr import OmeZarrParser


class OmeZarrTransformation(_xforms.LayeredTransformation, OmeZarrParser):
    """
    Base class for TIRL transformations.

    Concrete classes implement the `transform_group` attribute.
    """

    @property
    def _ome_version(self) -> _tx.Optional[str]:
        """The OME-NGFF spec version declared in this store's metadata."""
        if self.group is None:
            return None
        metadata = dict(self.group.attrs)
        ome = metadata.get("ome")
        if not isinstance(ome, dict):
            return None
        return ome.get("version")

    @staticmethod
    def _supports_displacement_fields(version: _tx.Optional[str]) -> bool:
        """
        NGFF 0.6+ (RFC-5, https://ngff.openmicroscopy.org/rfc/5/)
        introduced the `"displacement"` axis type. Earlier versions
        don't define it at all, so treat any pre-0.6 (or unparsable)
        version as unsupported.
        """
        if version is None:
            return False
        try:
            major, minor = (int(p) for p in version.split(".")[:2])
        except ValueError:
            return False
        return (major, minor) >= (0, 6)

    @property
    def _displacement_axis_mask(self) -> _tx.Optional[_tx.List[bool]]:
        """
        A boolean mask, one entry per `self._axes`, marking which
        axes are `"displacement"`-typed (RFC-5 / NGFF 0.6+). `None`
        if this store doesn't support the `"displacement"` axis type,
        or axes are unavailable.
        """
        if not self._supports_displacement_fields(self._ome_version):
            return None
        axes = self._axes
        if axes is None:
            return None
        return [getattr(axis, "type", None) == "displacement" for axis in axes]

    @property
    def _is_displacement_field(self) -> bool:
        """
        Whether every declared axis is `"displacement"`-typed -- in
        which case `self.fields[i]` should be wrapped as a
        `DisplacementField`, unchanged, rather than a
        `CoordinatesField`.
        """
        mask = self._displacement_axis_mask
        return bool(mask) and all(mask)

    @property
    def _has_mixed_displacement_axes(self) -> bool:
        """
        Whether some (but not all) declared axes are
        `"displacement"`-typed -- in which case `self.fields[i]`
        should be converted to absolute coordinates (by adding each
        such axis' own grid position to it) and wrapped as a
        `CoordinatesField`.
        """
        mask = self._displacement_axis_mask
        return bool(mask) and any(mask) and not all(mask)

    @staticmethod
    def _displacement_to_coordinate_field(
        field: ArrayLike, displacement_mask: list[bool]
    ) -> ArrayLike:
        """
        Convert the `"displacement"`-typed channels of `field`
        (shape `(*grid, N)`, one channel per axis) into absolute
        coordinates, by adding each such channel's own grid position
        to it -- i.e. `coordinate = grid_index + displacement`, the
        same relationship `DisplacementField`/`CoordinatesField` use
        elsewhere in this codebase. Channels that are not
        displacement-typed are left unchanged.

        Parameters
        ----------
        field: ArrayLike
            The zarr field that contains some displacement field.
        displacement_mask: list[bool]
            Value's coorisponding axis is displacement if True.

        Returns
        -------
        ArrayLike
            fully coordinate field
        """
        grid_shape = field.shape[:-1]
        for k, is_displacement in enumerate(displacement_mask):
            if is_displacement:
                index_shape = [1] * len(grid_shape)
                index_shape[k] = grid_shape[k]
                index_k = (
                    get_ndimage_backend()
                    .arange(grid_shape[k], dtype=field[..., k].dtype)
                    .reshape(index_shape)
                )
                field[..., k] = field[..., k] + index_k
        return field

    def _field_for_layer(self, field: ArrayLike) -> _xforms.Transformation:
        """
        Build the appropriate first-`Sequence`-element `Transformation`
        for one layer's raw `field` array:

        * every axis `"displacement"`-typed -> `DisplacementField`,
          unchanged.
        * no axes `"displacement"`-typed -> `CoordinatesField`,
          unchanged.
        * some (but not all) axes `"displacement"`-typed ->
          `CoordinatesField`, after converting each displacement axis
          to an absolute coordinate (see
          `_displacement_to_coordinate_field`).
        """
        if self._is_displacement_field:
            return _xforms.DisplacementField(
                field=field, input=self._axes, output=self._axes
            )
        if self._has_mixed_displacement_axes:
            field = self._displacement_to_coordinate_field(
                field, self._displacement_axis_mask
            )
        return _xforms.CoordinatesField(
            field=field, input=self._axes, output=self._axes
        )

    @property
    def layers(self) -> _tx.List[_xforms.Transformation]:
        if getattr(self, "_images", None) is None:
            if self.group is None or self._multiscale is None:
                self._layers = None
            else:
                self._layers = [
                    _xforms.Sequence(
                        transformations=[
                            self._field_for_layer(self.fields[i]),
                            *self._transform_from_multiscale(
                                self._multiscale, i
                            ),
                        ],
                        input=self._axes,
                        output=self._axes,
                    )
                    for i, ds in enumerate(self._multiscale["datasets"])
                ]
        return self._layers

    @layers.setter
    def layers(
        self, value: _tx.Optional[_tx.List[_xforms.Transformation]]
    ) -> None:
        self._layers = value
