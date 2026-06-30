import typing as _tx

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import map_coordinates

from brainhops._core.backends import da
from brainhops._core.typing import ArrayProtocol
from brainhops.io.transformations.base.fields import RASDisplacementField
from brainhops.io.transformations.common.base import NiftiBasedTransformation

FSL_CUBIC_SPLINE_COEFFICIENTS = 2007
FSL_DCT_COEFFICIENTS = 2008
FSL_QUADRATIC_SPLINE_COEFFICIENTS = 2009
FSL_FNIRT_DISPLACEMENT_FIELD = 2006


class FSLRASDisplacementField(RASDisplacementField, NiftiBasedTransformation):
    """
    Field of RAS displacements, stored in a NIfTI file.
    """

    _is_spline_cache: _tx.Optional[bool] = None
    _bspline_order: int = 3  # cubic, FNIRT's default
    _kernel: _tx.Optional[BSpline] = None
    _kernel_half_width: _tx.Optional[int] = None

    @property
    def is_spline_coefficients(self) -> bool:
        if self._is_spline_cache is not None:
            return self._is_spline_cache

        header = self.header
        if header is None:
            raise ValueError(
                "No header/image available to determine transform format."
            )

        intent_code = int(header.get("intent_code", 0))
        if intent_code == FSL_CUBIC_SPLINE_COEFFICIENTS:
            self._is_spline_cache = True
            self._bspline_order = 3
        elif intent_code == FSL_QUADRATIC_SPLINE_COEFFICIENTS:
            self._is_spline_cache = True
            self._bspline_order = 2
        elif intent_code == FSL_DCT_COEFFICIENTS:
            raise NotImplementedError(
                "DCT-basis FNIRT coefficient fields are not supported."
            )
        elif intent_code == FSL_FNIRT_DISPLACEMENT_FIELD:
            self._is_spline_cache = False
        else:
            raise ValueError(f"Unrecognized intent code: {intent_code}")

        return self._is_spline_cache

    # ------------------------------------------------------------------
    # Reconstruction (order-agnostic, scipy-backed)
    # ------------------------------------------------------------------

    def _reconstruct_field_slice(
        self, idx: _tx.Union[int, slice, tuple]
    ) -> np.ndarray:
        """
        Reconstruct only the requested slice of the dense displacement
        field from spline coefficients (cubic or quadratic) using
        `scipy.ndimage.map_coordinates`.

        TODO: Test against ground truth with FSL file
        """
        order = self._bspline_order
        if order not in (2, 3):
            raise NotImplementedError(f"Unsupported B-spline order: {order}")

        coef_data = self.image.get_fdata()  # shape: (nx, ny, nz, 3)

        target_shape = getattr(self, "_target_shape", None)
        if target_shape is None:
            raise ValueError(
                "Reconstructing from spline coefficients requires a "
                "target/reference grid shape. Set `_target_shape` "
                "before slicing."
            )

        grid_indices = np.indices(target_shape)  # (3, X, Y, Z)
        sliced_indices = grid_indices[(
            slice(None),) + self._normalize_idx(idx, target_shape)]

        out_shape = sliced_indices.shape[1:]
        flat_coords = sliced_indices.reshape(3, -1).T  # (N, 3)

        knot_coords = self._voxel_to_knot_coords(flat_coords)

        displacement = np.stack([
            map_coordinates(
                coef_data[..., c],
                knot_coords.T,
                order=order,
                prefilter=False,
                mode="grid-constant",
                cval=0.0,
            )
            for c in range(coef_data.shape[-1])
        ], axis=-1)

        return displacement.reshape(*out_shape, coef_data.shape[-1])

    @property
    def field(self) -> _tx.Optional[ArrayProtocol]:
        """The field of RAS displacements."""
        if da is None:
            return self.displacement_field  # fallback: eager numpy array

        return da.from_array(
            self,
            chunks="auto",
            name=f"displacement-field-{id(self)}",
        )

    @property
    def shape(self) -> tuple:
        if not self.is_spline_coefficients:
            return self.image.shape
        target_shape = getattr(self, "_target_shape", None)
        if target_shape is None:
            raise ValueError(
                "Cannot determine field shape from spline coefficients "
                "without `_target_shape` set."
            )
        return (*target_shape, 3)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    # ------------------------------------------------------------------
    # Lazy conversion: spline coefficients -> displacement field
    # ------------------------------------------------------------------

    @property
    def displacement_field(self) -> np.ndarray:
        """
        Full dense displacement field, materialized eagerly.
        Prefer slicing (`obj[idx]`) instead of this when you only need
        part of the volume — this property reconstructs the *entire*
        field if the source is spline coefficients, which can be costly.
        """
        if not self.is_spline_coefficients:
            return self.image.get_fdata()
        return self._reconstruct_full_field()

    def __getitem__(
        self, idx: _tx.Union[int, slice, tuple]
    ) -> np.ndarray:
        """
        Slice the displacement field. If the underlying data is stored as
        spline coefficients, only the requested region is reconstructed —
        full-volume reconstruction is avoided.
        """
        if not self.is_spline_coefficients:
            data = self.image.get_fdata()
            return data[idx]

        return self._reconstruct_field_slice(idx)

    def _voxel_to_knot_coords(
        self, voxel_coords: np.ndarray
    ) -> np.ndarray:
        """
        Map target-image voxel coordinates into coefficient-grid
        (knot) coordinate space, using the relative voxel sizes of the
        coefficient image vs. the implied dense-field resolution.

        NOTE: This assumes the coefficient image's affine/zooms encode
        the knot spacing directly, which matches FNIRT's convention but
        should be verified against `fnirtfileutils` output if exactness
        matters for your use case.
        """
        coef_zooms = np.array(self.header.get_zooms()[:3])
        # Caller is responsible for passing voxel_coords already in the
        # *target* (dense-field) voxel grid; here we rescale into knot
        # spacing units.
        return voxel_coords / coef_zooms

    def _reconstruct_full_field(self) -> np.ndarray:
        """Reconstruct the entire dense field (no slicing shortcut)."""
        target_shape = getattr(self, "_target_shape", None)
        if target_shape is None:
            raise ValueError(
                "Reconstructing from spline coefficients requires a "
                "target/reference grid shape set via `_target_shape`."
            )
        full_idx = tuple(slice(None) for _ in target_shape)
        return self._reconstruct_field_slice(full_idx)

    @staticmethod
    def _normalize_idx(idx: _tx.Union[int, slice, tuple],
                       shape: _tx.Union[int, slice, tuple]) -> tuple:
        """Normalize int/slice/tuple indexing into a full tuple of slices."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = list(idx)
        while len(idx) < len(shape):
            idx.append(slice(None))
        normalized = []
        for i, dim in zip(idx, shape):
            if isinstance(i, int):
                normalized.append(slice(i, i + 1))
            else:
                normalized.append(i)
        return tuple(normalized)
