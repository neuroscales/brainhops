import itertools
import typing as _tx

import numpy as np
from scipy.ndimage import map_coordinates

from brainhops._core.backends import da
from brainhops._core.typing import ArrayProtocol
from brainhops.io.transformations.base.fields import RASDisplacementField
from brainhops.io.transformations.common.base import NiftiBasedTransformation

# constants retrieved from fslpy on Jun 29th 2026
FSL_CUBIC_SPLINE_COEFFICIENTS = 2007
FSL_DCT_COEFFICIENTS = 2008
FSL_QUADRATIC_SPLINE_COEFFICIENTS = 2009
FSL_FNIRT_DISPLACEMENT_FIELD = 2006


class FSLRASDisplacementField(RASDisplacementField, NiftiBasedTransformation):
    """
    Field of RAS displacements, stored in a NIfTI file.
    """

    is_spline_cache: _tx.Optional[bool] = None
    bspline_order: int = 3  # cubic, FNIRT's default

    @property
    def is_spline_coefficients(self) -> bool:
        if self.is_spline_cache is not None:
            return self.is_spline_cache

        header = self.header
        if header is None:
            raise ValueError(
                "No header/image available to determine transform format."
            )

        intent_code = int(header.get("intent_code", 0))
        if intent_code == FSL_CUBIC_SPLINE_COEFFICIENTS:
            self.is_spline_cache = True
            self.bspline_order = 3
        elif intent_code == FSL_QUADRATIC_SPLINE_COEFFICIENTS:
            self.is_spline_cache = True
            self.bspline_order = 2
        elif intent_code == FSL_DCT_COEFFICIENTS:
            raise NotImplementedError(
                "DCT-basis FNIRT coefficient fields are not supported."
            )
        elif intent_code == FSL_FNIRT_DISPLACEMENT_FIELD:
            self.is_spline_cache = False
        else:
            raise ValueError(f"Unrecognized intent code: {intent_code}")

        return self.is_spline_cache

    # ------------------------------------------------------------------
    # Reconstruction (order-agnostic, scipy-backed)
    # ------------------------------------------------------------------

    def _reconstruct_field_slice(
        self, idx: _tx.Union[int, slice, tuple]
    ) -> np.ndarray:
        """
        Code taken from fslpy's displacement function

        Reconstruct only the requested slice of the dense displacement
        field from spline coefficients using the cubic B-spline basis
        functions from Rueckert et al. (1999).

        Reference:
        Rueckert et al., "Nonrigid Registration Using Free-Form
        Deformations", IEEE TMI 1999.
        https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf
        """

        order = self.bspline_order
        if order != 3:
            raise NotImplementedError(
                "Only cubic (order=3) B-spline reconstruction is "
                "currently supported via the Rueckert basis functions. "
                "Quadratic (order=2) requires its own basis functions "
                "fslpy does not support this yet"
            )

        coef_data = self.image.get_fdata()  # (nx, ny, nz, 3)
        nx, ny, nz = coef_data.shape[:3]

        target_shape = getattr(self, "target_shape", None)
        if target_shape is None:
            raise ValueError(
                "Reconstructing from spline coefficients requires a "
                "target/reference grid shape. Set `target_shape` "
                "before slicing."
            )

        # Cubic B-spline basis functions (Rueckert et al.)
        # u is the fractional position within the current spline segment,
        # always in [0, 1). l indexes which of the 4 neighboring knots
        # is being weighted — asymmetric by design.
        def b0(u): return ((1 - u) ** 3) / 6
        def b1(u): return (3 * (u ** 3) - 6 * (u ** 2) + 4) / 6
        def b2(u): return (-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6
        def b3(u): return (u ** 3) / 6
        b = [b0, b1, b2, b3]

        grid_indices = np.indices(target_shape)  # (3, X, Y, Z)
        sliced_indices = grid_indices[(
            slice(None),) + self._normalize_idx(idx, target_shape)]

        out_shape = list(sliced_indices.shape[1:])
        flatten_idx = idx
        if not isinstance(flatten_idx, _tx.Tuple):
            flatten_idx = tuple(flatten_idx, )
        for i in range(len(flatten_idx)-1, -1, -1):
            if isinstance(flatten_idx[i], int) or isinstance(flatten_idx[i], np.floating):
                out_shape.pop(i)

        flat_coords = sliced_indices.reshape(3, -1).T  # (N, 3)

        knot_coords = self._voxel_to_knot_coords(flat_coords)  # (N, 3)

        # Fractional and integer parts of knot coordinates
        u = np.remainder(knot_coords[:, 0], 1)
        v = np.remainder(knot_coords[:, 1], 1)
        w = np.remainder(knot_coords[:, 2], 1)
        i = np.floor(knot_coords[:, 0]).astype(np.int32)
        j = np.floor(knot_coords[:, 1]).astype(np.int32)
        k = np.floor(knot_coords[:, 2]).astype(np.int32)

        disps = np.zeros((flat_coords.shape[0], 3), dtype=float)

        for l, m, n in itertools.product(range(4), range(4), range(4)):
            il = i + l
            jm = j + m
            kn = k + n

            mask = (
                (il >= 0) & (il < nx) &
                (jm >= 0) & (jm < ny) &
                (kn >= 0) & (kn < nz)
            )
            if not np.any(mask):
                continue

            c = b[l](u[mask]) * b[m](v[mask]) * b[n](w[mask])
            disps[mask] += coef_data[il[mask],
                                     jm[mask], kn[mask], :] * c[:, None]

        return disps.reshape(*out_shape, 3)

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

    @field.setter
    def field(self, value: _tx.Optional[ArrayProtocol]):
        NotImplementedError(
            "cannot set field, make another displacement field")

    @property
    def shape(self) -> tuple:
        if not self.is_spline_coefficients:
            return self.image.shape
        target_shape = getattr(self, "target_shape", None)
        if target_shape is None:
            raise ValueError(
                "Cannot determine field shape from spline coefficients "
                "without `target_shape` set."
            )
        return (*target_shape, 3)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    @property
    def ndim(self) -> np.int32:
        return 3

    # ------------------------------------------------------------------
    # Lazy conversion: spline coefficients -> displacement field
    # ------------------------------------------------------------------

    @property
    def displacement_field(self) -> np.ndarray:
        """
        Full dense displacement field, materialized eagerly.
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
        """
        coef_zooms = np.array(self.header.get_zooms()[:3])
        return voxel_coords / coef_zooms

    def _reconstruct_full_field(self) -> np.ndarray:
        """Reconstruct the entire dense field (no slicing shortcut)."""
        target_shape = getattr(self, "target_shape", None)
        if target_shape is None:
            raise ValueError(
                "Reconstructing from spline coefficients requires a "
                "target/reference grid shape set via `target_shape`."
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
            idx.append(slice(0, shape[len(idx)]))
        normalized = []
        for i, dim in zip(idx, shape):
            if isinstance(i, int) or isinstance(i, np.floating):
                normalized.append(slice(int(i), int(i) + 1))
            else:
                normalized.append(i)
        return tuple(normalized)
