__all__ = ["H5TransformStruct"]

# externals

import typing_extensions as _tx

# _ext
from brainhops.io.transformations.itk._utils import TransformBlock

# internals
from ._parser import H5TransformParser


# ---------------------------------------------------------------------
# Type hints
# ---------------------------------------------------------------------

_Floats = _tx.Tuple[float, ...]


# ---------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------


class H5TransformStruct(H5TransformParser):
    """
    In-memory representation of an ITK H5TransformIO file.

    Parsing and writing are implemented in `H5TransformParser`.
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

        self.append_transform(
            TransformBlock(
                transform=transform,
                parameters=tuple(parameters),
                fixed_parameters=tuple(
                    fixed_parameters
                )
            )
        )
