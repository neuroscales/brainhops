import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly

from brainhops._core.typing import Deactivated
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.transformations.base import TransformationFormat

from .._affines import _ImageGeometry
from .._formats import FslAffineFormat
from .._repr import stored_repr
from ._parser import FlirtMatrixReader


@register_format
class FlirtTransform(
    FslAffineFormat,
    FlirtMatrixReader,
    _xforms.Affine,
    TransformationFormat,
):
    """Linear transformation stored in an FSL FLIRT `.mat` file.

    The FLIRT matrix maps the scaled millimetre coordinates of the moving image
    to those of the reference image. This class exposes it as an affine whose
    `matrix` maps reference world (RAS) coordinates to moving world (RAS)
    coordinates, which is the direction the data model uses to resample a
    moving image onto a reference.

    Because the file carries no image geometry, the reference and moving
    images must be supplied. They can be passed as the `reference=` and
    `moving=` keyword arguments when the file is loaded, or set on the object
    before its `matrix` is read.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("flirt",)
    # The data field is the raw FLIRT matrix. The `matrix` view is derived
    # from the raw matrix and the two images, so computing it is costly and
    # may raise. Naming the raw matrix here keeps the identity check of
    # `inverse()` from computing the affine.
    data_fields: tx.ClassVar[tx.Tuple[str, ...]] = ("flirt_matrix",)

    _input: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()

    # The affine is computed on demand, so it is not a constructor field.
    # Declaring `_data` as a ClassVar overrides the init field inherited from
    # `Affine` and keeps `data` out of `__init__` and `fields()`.
    _data: tx.ClassVar[tx.Optional[tx.Any]]
    _matrix: Deactivated[None]

    @property
    def data(self) -> tx.Optional[np.ndarray]:
        """The affine from reference RAS to moving RAS, as a (3, 4) array.

        The affine is computed from the raw matrix and both image geometries,
        and is `None` when no raw matrix is set. It cannot be assigned.

        Raises
        ------
        ValueError
            If a raw matrix is set but an image is missing.
        """
        raw = self.flirt_matrix
        if raw is None:
            return None
        if self.reference is None or self.moving is None:
            raise ValueError(
                "A FLIRT matrix carries no image geometry, so both the "
                "reference and the moving image are needed to place it in "
                "world coordinates. Pass reference=... and moving=... when "
                "loading the matrix, or set them on the transformation."
            )
        ref = _ImageGeometry(self.reference)
        mov = _ImageGeometry(self.moving)
        flirt = np.asarray(raw, dtype=np.float64)
        full = mov.fsl2ras @ np.linalg.inv(flirt) @ ref.ras2fsl
        return full[:-1]

    @data.setter
    def data(self, value: None) -> None:
        if value is not None:
            raise ValueError(
                "The FLIRT affine is computed from the raw matrix and the "
                "images, so it cannot be set. Set flirt_matrix instead."
            )

    def inverse(self, compute: bool = False, **kwargs) -> _xforms.Affine:
        # The inverse is a plain affine, because the raw matrix and the two
        # images of a FLIRT transform have no meaning after inversion.
        return _xforms.Affine(
            matrix=self.matrix, input=self.input, output=self.output
        ).inverse(compute=compute, **kwargs)

    def __repr__(self) -> str:
        return stored_repr(self, ("flirt_matrix", "moving", "reference"))
