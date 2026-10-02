# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly

# externals
# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.base._base import register_format
from brainhops.io.transformations.base import FileBasedTransformation

from .._affines import _ImageGeometry
from .._formats import FslAffineFormat
from .._repr import stored_repr
from ._parser import FlirtMatrixParser


@register_format
class FlirtTransform(
    FslAffineFormat,
    FlirtMatrixParser,
    _xforms.Affine,
    FileBasedTransformation,
):
    """A linear transformation stored in a FLIRT `.mat` file.

    A FLIRT matrix maps moving-image scaled-mm coordinates to
    reference-image scaled-mm coordinates. This reader exposes it as an
    affine whose `matrix` maps reference-image world (RAS) coordinates to
    moving-image world (RAS) coordinates, which is the direction the data
    model uses to resample a moving image onto a reference.

    The reference and moving images must be supplied, because the `.mat`
    file carries no image geometry. They may be passed as keyword
    arguments to `load` or `from_file` (`reference=`, `moving=`), or set
    on the object before its `matrix` is read.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("flirt",)
    # The stored parameter is the raw FLIRT matrix: `matrix` is derived from
    # it and the two image geometries, so reading it is not free and may
    # raise. Naming the raw field here keeps `_is_unparameterized()` (and so
    # `inverse()`) off that path.
    data_fields: tx.ClassVar[tx.Tuple[str, ...]] = ("flirt_matrix",)

    _input: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()
    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()

    # `matrix` is computed on demand from the raw FLIRT matrix and the two
    # image geometries, so it is not a stored, constructor-taken field
    # here. Declaring it a `ClassVar` overrides the inherited init-field
    # from `Affine` and keeps `matrix` out of `__init__` and `fields()`,
    # while the property below serves reads.
    matrix: tx.ClassVar[tx.Optional[tx.Any]]

    @property
    def matrix(self) -> tx.Optional[np.ndarray]:
        """The reference-RAS to moving-RAS affine, as a `(3, 4)` matrix.

        Reading this resolves the affine from the raw FLIRT matrix and the
        two image geometries. It raises when the raw matrix is present but
        either image is missing, because the affine cannot be placed in
        world coordinates without both.
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

    @matrix.setter
    def matrix(self, value: None) -> None:
        if value is not None:
            raise ValueError(
                "The FLIRT affine is computed from the raw matrix and the "
                "images, so it cannot be set. Set flirt_matrix instead."
            )

    def inverse(self, compute: bool = False, **kwargs) -> _xforms.Affine:
        # The inverse of a resolved FLIRT affine is a plain affine, because
        # the raw-matrix and image structure of a FLIRT transform does not
        # survive inversion.
        return _xforms.Affine(
            matrix=self.matrix, input=self.input, output=self.output
        ).inverse(compute=compute, **kwargs)

    def __repr__(self) -> str:
        return stored_repr(self, ("flirt_matrix", "moving", "reference"))
