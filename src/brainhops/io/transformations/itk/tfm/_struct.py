__all__ = ["TxtTransformStruct"]

# externals
import math

import typing_extensions as _tx

# _ext
from brainhops._ext.struct import Factory
from brainhops.io.transformations.itk._utils import TransformBlock

# internals
from ._parser import TxtTransformParser


# ---------------------------------------------------------------------
# Type hints
# ---------------------------------------------------------------------

_Floats = _tx.Tuple[float, ...]


# ---------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------


class TxtTransformStruct(TxtTransformParser):
    """
    In-memory representation of an ITK TxtTransformIO file.

    Parsing and writing are implemented in `TxtTransformParser`.

    Example
    -------
    ```
    #Insight Transform File V1.0

    #Transform 0
    Transform: Euler2DTransform_double_2_2
    Parameters: 0.2 10 -5
    FixedParameters: 128 128
    ```
    """

    # -----------------------------------------------------------------
    # File-level fields
    # -----------------------------------------------------------------

    transform_blocks: _tx.List[TransformBlock] = []

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------

    @property
    def n_transforms(self) -> int:
        return len(self.transform_blocks)

    @property
    def is_composite(self) -> bool:
        return self.n_transforms > 1

    @property
    def first_transform(self) -> _tx.Optional[TransformBlock]:

        if not self.transform_blocks:
            return None

        return self.transform_blocks[0]

    @property
    def last_transform(self) -> _tx.Optional[TransformBlock]:

        if not self.transform_blocks:
            return None

        return self.transform_blocks[-1]

    def append_transform(
        self,
        transform: TransformBlock,
    ):

        self.transform_blocks = (
            *self.transform_blocks,
            transform,
        )

    def new_transform(
        self,
        transform: str,
        parameters=(),
        fixed_parameters=(),
    ):

        obj = TransformBlock()
        obj.TransformType = transform
        obj.TransformParameters = parameters
        obj.TransformFixedParameters = fixed_parameters

        self.append_transform(
            obj
        )
