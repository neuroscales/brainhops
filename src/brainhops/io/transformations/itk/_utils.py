# stdlib
import math
from dataclasses import dataclass, field
import numpy as np
import typing_extensions as _tx
from brainhops._ext.struct import Struct
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class TransformBlock(Struct):
    _HAS_KEYS = True
    TransformType: str = ""
    TransformParameters: list[float] = field(default_factory=list)
    TransformFixedParameters: list[float] = field(default_factory=list)

    def __init__(self):
        super().__init__()
        self._affine: _tx.Optional[np.ndarray] = None
        self._displacement: _tx.Optional[np.ndarray] = None

    @property
    def ndim(self) -> _tx.Optional[int]:
        """
        Spatial dimension inferred from ITK transform name.

        Examples
        --------
        Euler2DTransform_double_2_2
        -> 2

        AffineTransform_double_3_3
        -> 3
        """
        parts = self.TransformType.split("_")
        if len(parts) < 2:
            return None
        try:
            return int(parts[-1])
        except ValueError:
            return None

    def itk_meta_to_affine(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        origin = arr[self.ndim:self.ndim*2]
        spacing = arr[self.ndim*2:self.ndim*3]
        direction = arr[self.ndim*3:(self.ndim*3 +
                                     self.ndim**2)].reshape(self.ndim, self.ndim)

        M = direction @ np.diag(spacing)

        affine = np.eye(4)
        affine[:3, :3] = M
        affine[:3, 3] = origin
        return affine

    @property
    def is_displacement(self) -> bool:
        return self.TransformType.startswith("DisplacementFieldTransform")

    def to_transformation(self):
        if self.is_displacement:
            return self.to_displacement()
        return self.to_affine()

    def to_displacement(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        (voxel_to_world_affine, displacement_field)
        """
        if not self.is_displacement:
            raise TypeError(
                f"{self.TransformType} is not a displacement transform")

        if self._displacement is None:
            # ---- displacement field ----
            if self.TransformParameters:
                shape = tuple(int(p)
                              for p in self.TransformFixedParameters[:self.ndim]) + (self.ndim,)
                self._displacement = np.asarray(
                    self.TransformParameters).reshape(shape)
            else:
                raise ValueError("No displacement field data found")

        if self._affine is None:
            # ---- voxel -> world affine ----
            # ITK usually stores this in FixedParameters for displacement fields
            if len(self.TransformFixedParameters) >= self.ndim * (self.ndim + self.ndim):
                self._affine = self.itk_meta_to_affine(
                    self.TransformFixedParameters)
            else:
                raise ValueError(
                    f"Insufficient FixedParameters for displacement affine (need ≥{self.ndim * (self.ndim + self.ndim)})"
                )

        return self._affine, self._displacement

    # TODO allow to output to other transform types
    def to_affine(self) -> np.ndarray:
        """Return a homogeneous affine matrix (ndim+1 x ndim+1).

        ITK's centre-of-rotation convention is applied:
            T(x) = A (x - c) + t + c
        where ``c`` comes from ``FixedParameters`` and ``t`` from the
        trailing entries of ``Parameters``.
        """
        if self._affine is not None:
            return self._affine

        name = self.TransformType
        if name.startswith("TranslationTransform"):
            self._affine = self._translation_to_affine()
        elif name.startswith("AffineTransform"):
            self._affine = self._matrix_offset_to_affine()
        elif name.startswith("MatrixOffsetTransformBase"):
            self._affine = self._matrix_offset_to_affine()
        elif name.startswith("Rigid2DTransform") or name.startswith("Euler2DTransform"):
            self._affine = self._euler2d_to_affine()
        elif name.startswith("Euler3DTransform"):
            self._affine = self._euler3d_to_affine()
        elif name.startswith("VersorRigid3DTransform") or name.startswith("QuaternionRigidTransform"):
            self._affine = self._versor3d_to_affine()
        elif name.startswith("ScaleTransform"):
            self._affine = self._scale_to_affine()
        elif name.startswith("IdentityTransform"):
            d = self.ndim or 3
            self._affine = np.eye(d + 1)
        else:
            raise NotImplementedError(
                f"No affine converter for '{name}'.  "
                f"Add a handler or call to_transformation() and check the type."
            )
        return self._affine

    # ---- helpers that all respect the ITK centre-of-rotation convention ----

    def _center(self) -> np.ndarray:
        """Centre of rotation from FixedParameters (zeros if absent)."""
        d = self.ndim or 3
        if len(self.TransformFixedParameters) >= d:
            return np.array(self.TransformFixedParameters[:d], dtype=np.float64)
        return np.zeros(d, dtype=np.float64)

    def _apply_center(self, A: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Build (d+1)x(d+1) matrix for  T(x) = A(x-c) + t + c."""
        d = len(c)
        M = np.eye(d + 1)
        M[:d, :d] = A
        M[:d, d] = -A @ c + t + c
        return M

    def _translation_to_affine(self) -> np.ndarray:
        d = self.ndim or len(self.TransformParameters)
        t = np.array(self.TransformParameters[:d], dtype=np.float64)
        M = np.eye(d + 1)
        M[:d, d] = t
        return M

    def _matrix_offset_to_affine(self) -> np.ndarray:
        """AffineTransform / MatrixOffsetTransformBase.
        Parameters = [A (row-major, d²)] + [t (d)]
        FixedParameters = [c (d)]
        """
        d = self.ndim or 3
        p = self.TransformParameters
        A = np.array(p[:d * d], dtype=np.float64).reshape(d, d)
        t = np.array(p[d * d:d * d + d], dtype=np.float64)
        c = self._center()
        return self._apply_center(A, t, c)

    def _euler2d_to_affine(self) -> np.ndarray:
        """Rigid2DTransform / Euler2DTransform.
        Parameters = [angle, tx, ty]
        FixedParameters = [cx, cy]
        """
        angle, tx, ty = self.TransformParameters[:3]
        t = np.array([tx, ty], dtype=np.float64)
        c = self._center()
        co, si = math.cos(angle), math.sin(angle)
        A = np.array([[co, -si], [si, co]], dtype=np.float64)
        return self._apply_center(A, t, c)

    def _euler3d_to_affine(self) -> np.ndarray:
        """Euler3DTransform.
        Parameters = [rx, ry, rz, tx, ty, tz]
        FixedParameters = [cx, cy, cz]
        ITK convention: R = Rz · Ry · Rx  (extrinsic x→y→z)
        """
        rx, ry, rz = self.TransformParameters[:3]
        t = np.array(self.TransformParameters[3:6], dtype=np.float64)
        c = self._center()

        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        Rx = np.array([[1, 0, 0], [0, cx, -sx],
                      [0, sx, cx]], dtype=np.float64)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
                      dtype=np.float64)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0],
                      [0, 0, 1]], dtype=np.float64)
        A = Rz @ Ry @ Rx
        return self._apply_center(A, t, c)

    def _versor3d_to_affine(self) -> np.ndarray:
        """VersorRigid3DTransform / QuaternionRigidTransform.
        Parameters = [qx, qy, qz, tx, ty, tz]
        (ITK stores the vector part; w = sqrt(1 - |v|²))
        FixedParameters = [cx, cy, cz]
        """
        qx, qy, qz = self.TransformParameters[:3]
        t = np.array(self.TransformParameters[3:6], dtype=np.float64)
        c = self._center()
        norm_sq = qx ** 2 + qy ** 2 + qz ** 2
        if norm_sq > 1.0:
            raise ValueError(
                "Versor quaternion vector part has magnitude > 1")
        qw = math.sqrt(max(0.0, 1.0 - norm_sq))
        # Convert unit quaternion to rotation matrix
        A = np.array([
            [1 - 2*(qy**2 + qz**2),     2 *
             (qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)
        return self._apply_center(A, t, c)

    def _scale_to_affine(self) -> np.ndarray:
        """ScaleTransform.
        Parameters = [sx, sy(, sz)]
        FixedParameters = [cx, cy(, cz)]
        """
        d = self.ndim or len(self.TransformParameters)
        s = np.array(self.TransformParameters[:d], dtype=np.float64)
        c = self._center()
        A = np.diag(s)
        t = np.zeros(d, dtype=np.float64)
        return self._apply_center(A, t, c)


class VoxelToLPS(_xforms.Sequence):
    """Affine transformation from voxel space to LPS space."""

    input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.LPSCoordinateSystem()


class LPSToVoxel(_xforms.Sequence):
    """Affine transformation from LPS space to voxel space."""

    input: _systems.CoordinateSystem = _systems.LPSCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
