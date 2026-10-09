__all__ = [
    "ElastixTransform",
    "ElastixParameterTransform",
    "ElastixTomlTransform",
]

import os
from warnings import warn

import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Factory, NoEq

from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.geometry import Geometry
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    ParserExistsError,
    ParserNotImplementedError,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base import WritableFileBasedTransformation
from brainhops.io.transformations.itk._common import ItkStruct

from ._blocks import fixed_geometry, map_to_block, transformation_to_map
from ._parser import (
    ParameterMap,
    format_text_map,
    format_toml_map,
    get_bool,
    get_string,
    is_transform_map,
    read_text_map,
    read_toml_map,
)

#: The value elastix writes when there is no initial transform.
_NO_INITIAL = "NoInitialTransform"


# Text adapters supply byte decoding and encoding before the generic bases.
class ElastixTransform(
    TextFileReader,
    TextFileWriter,
    _xforms.Sequence,
    WritableFileBasedTransformation,
    repr=HIDE_IF_NONE,
    reverse=False,  # keep `transformations` the first positional field
):
    """Transformation stored in an elastix transform parameter file.

    The transformation is a
    [`Sequence`][brainhops.datamodel.transformations.Sequence] of ITK blocks
    (see [`brainhops.io.transformations.elastix`][]) that maps fixed-image
    LPS to moving-image LPS, the pull direction of transformix. The blocks of
    the chained initial transforms come first, followed by the file's own.
    The raw map is kept in `parameter_map` and the initial transform in
    `initial`.

    This class is abstract; its subclasses [`ElastixParameterTransform`][]
    (`.txt`) and [`ElastixTomlTransform`][] (`.toml`) are the registered
    formats.
    """

    HINTS = ("elastix", "transformix")

    # Not compared, because the transform parameters are an array.
    parameter_map: tx.Annotated[ParameterMap, NoEq()] = Factory(
        dict, repr=False
    )
    """The parameters of the file, in the order they were read.

    Each name maps to a tuple of strings and numbers, except the transform
    parameters, which are a float64 array.
    """

    initial: tx.Optional[tx.Any] = None
    """The initial transform, as an `ElastixTransform`.

    It is `None` when the file names none, or with `initial=False`.
    """

    @classmethod
    def _read_map(cls, lines: tx.Iterable[str]) -> ParameterMap:
        raise NotImplementedError

    @classmethod
    def _format_map(cls, pmap: ParameterMap) -> tx.Iterator[str]:
        raise NotImplementedError

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that lines form an elastix transform file.

        The map must name a `Transform` and carry its parameters, which
        excludes registration parameter files.
        """
        try:
            pmap = cls._read_map(lines)
        except (ParserContentError, ValueError) as e:
            return _reject(error, f"Not an elastix parameter file: {e}")
        if is_transform_map(pmap):
            return Confidence.CERTAIN
        return _reject(
            error,
            "Not an elastix transform parameter file: it names no "
            "Transform, or no TransformParameters.",
        )

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the text of a file, as in [`sniff_lines`][]."""
        return cls.sniff_lines(text.splitlines(), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Build the transformation from an open file.

        A relative initial transform is looked for next to the file's `name`.
        """
        kwargs.setdefault("origin", getattr(file, "name", None))
        with preserve_position(file):
            content = file.read()
        if isinstance(content, (bytes, bytearray)):
            return cls.from_bytes(content, **kwargs)
        return cls.from_text(content, **kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build the transformation from the text of a file."""
        return cls.from_lines(text.splitlines(), **kwargs)

    @classmethod
    def from_lines(
        cls,
        lines: tx.Iterable[str],
        initial: tx.Union[
            bool, path.FilenameLike, _xforms.Transformation
        ] = True,
        origin: tx.Optional[path.FilenameLike] = None,
        **kwargs,
    ) -> tx.Self:
        """Build the transformation from the lines of a file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        initial : bool, file name or Transformation, default=True
            The initial transform named in `InitialTransformParameterFileName`.
            `True` reads it, and those it names in turn, as transformix does
            (see [`resolve_initial`][]). `False` reads the file's own transform
            alone. A file name or a transformation replaces the named initial
            transform.
        origin : file name, optional
            The file that the lines come from, against which a relative initial
            transform is resolved. `from_filename` sets it.

        Raises
        ------
        ParserNotImplementedError
            If the transform is not supported, or if it is added to its initial
            transform (`HowToCombineTransforms "Add"`), which a chain cannot
            express.
        """
        pmap = cls._read_map(lines)
        if not is_transform_map(pmap):
            raise ParserContentError(
                "Not an elastix transform parameter file: it names no "
                "Transform, or no TransformParameters."
            )
        if not get_bool(pmap, "UseDirectionCosines", True):
            warn(
                "This elastix transform was computed with "
                'UseDirectionCosines "false": elastix ignored the '
                "direction cosines of both images, so the transform maps "
                "their direction-less physical spaces, not LPS. It is read "
                "as if it mapped LPS.",
                stacklevel=2,
            )
        visited = kwargs.pop("_visited", ())
        name = initial_filename(pmap)
        initial_xform = None
        if initial is True and name is not None:
            filename = resolve_initial(name, origin)
            key = os.path.realpath(str(filename))
            if key in visited or (
                origin is not None and key == os.path.realpath(str(origin))
            ):
                raise ParserContentError(
                    f"The elastix initial transforms loop back to {filename}."
                )
            visited = (*visited, key)
            if origin is not None:
                visited = (*visited, os.path.realpath(str(origin)))
            initial_xform = _load_initial(filename, _visited=visited)
        elif initial not in (True, False, None):
            if isinstance(initial, _xforms.Transformation):
                initial_xform = initial
            else:
                initial_xform = _load_initial(initial)
        if initial_xform is not None:
            how = get_string(pmap, "HowToCombineTransforms", "Compose")
            if how != "Compose":
                raise ParserNotImplementedError(
                    f"This elastix transform combines with its initial "
                    f"transform by {how!r} (HowToCombineTransforms): "
                    f"T(x) = T1(x) + T0(x) - x. A sequence of "
                    f"transformations can only compose them, so it "
                    f"cannot be read. Pass initial=False to read the "
                    f"file's own transform alone."
                )
        obj = cls(parameter_map=pmap, initial=initial_xform)
        # Decode now, so that an unsupported transform fails here.
        obj.block  # noqa: B018
        return obj

    @property
    def initial_filename(self) -> tx.Optional[str]:
        """The file name of the initial transform, if the map names one."""
        return initial_filename(self.parameter_map)

    @smartproperty(cache=True)
    def block(self) -> tx.Optional[ItkStruct]:
        """The ITK block of the file's own transform."""
        if not self.parameter_map:
            return None
        return map_to_block(self.parameter_map)

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain, in the order of application.

        The flattened initial transforms come first and the file's own block
        last, since elastix computes `T1(T0(x))`. An assigned chain overrides
        the decoded one and is what the writer encodes.
        """
        chain: tx.List[_xforms.Transformation] = []
        if self.initial is not None:
            if isinstance(self.initial, _xforms.Sequence):
                chain.extend(self.initial.transformations or ())
            else:
                chain.append(self.initial)
        if self.block is not None:
            chain.append(self.block)
        return tuple(chain)

    @property
    def fixed_geometry(self) -> tx.Optional[Geometry]:
        """The geometry of the fixed image, or `None` if the file omits it.

        It is the grid onto which transformix resamples the moving image, built
        from `Size`, `Index`, `Spacing`, `Origin` and `Direction`, and its
        transformation maps voxels to LPS millimetres.
        """
        return fixed_geometry(self.parameter_map)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the transformation to a file.

        The content is built first, so an unrepresentable transformation leaves
        the file untouched.
        """
        content = self.to_text(**kwargs) + "\n"
        with path.Path(filename).open("w") as f:
            f.write(content)

    def to_map(self) -> ParameterMap:
        """Return the parameter map that encodes the transformation.

        An unmodified transformation is written as it was read, including the
        file name of its initial transform, which lives in its own file.
        Otherwise the chain must be a single elastix or ITK block (translation,
        Euler, similarity, affine or B-spline), or anything that reduces to an
        affine, which becomes an `AffineTransform` centered on the origin. The
        fixed geometry and other parameters of `parameter_map` are kept, and
        the result has no initial transform.

        Raises
        ------
        UnrepresentableTransformationError
            If the chain cannot be written as a single elastix transform.
        """
        if getattr(self, "_transformations", None) is None:
            if not self.parameter_map:
                raise UnrepresentableTransformationError(
                    "This elastix transformation is empty."
                )
            return dict(self.parameter_map)
        chain = list(self._transformations)
        if len(chain) == 1:
            return transformation_to_map(chain[0], self.parameter_map)
        if not chain:
            raise UnrepresentableTransformationError(
                "This elastix transformation is empty."
            )
        sequence = _xforms.Sequence(transformations=chain)
        try:
            return transformation_to_map(sequence, self.parameter_map)
        except UnrepresentableTransformationError as e:
            raise UnrepresentableTransformationError(
                f"An elastix parameter file holds a single transform, and "
                f"this chain of {len(chain)} does not reduce to an affine. "
                f"Write each one to its own file, and chain them with "
                f"InitialTransformParameterFileName."
            ) from e

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the parameter file (see [`to_map`][])."""
        return self._format_map(self.to_map())


# ----------------------------------------------------------------------
#   SYNTAXES
# ----------------------------------------------------------------------


@register_format
class ElastixParameterTransform(ElastixTransform):
    """Classic `(Name value ...)` elastix parameter file (`.txt`)."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)
    HINTS = ("params",)

    @classmethod
    def _read_map(cls, lines: tx.Iterable[str]) -> ParameterMap:
        return read_text_map(lines)

    @classmethod
    def _format_map(cls, pmap: ParameterMap) -> tx.Iterator[str]:
        return format_text_map(pmap)


@register_format
class ElastixTomlTransform(ElastixTransform):
    """TOML `Name = value` elastix parameter file (`.toml`)."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".toml",)
    HINTS = ("toml",)

    @classmethod
    def _read_map(cls, lines: tx.Iterable[str]) -> ParameterMap:
        return read_toml_map(lines)

    @classmethod
    def _format_map(cls, pmap: ParameterMap) -> tx.Iterator[str]:
        return format_toml_map(pmap)


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def initial_filename(pmap: ParameterMap) -> tx.Optional[str]:
    """Return the initial transform that a parameter map names, if any.

    elastix reads `InitialTransformParameterFileName`, or else the deprecated
    `InitialTransformParametersFileName`; `"NoInitialTransform"` means none.
    """
    for key in (
        "InitialTransformParameterFileName",
        "InitialTransformParametersFileName",
    ):
        name = get_string(pmap, key)
        if name:
            return None if name == _NO_INITIAL else name
    return None


def _is_absolute(name: str) -> bool:
    # elastix's own test: a leading slash or backslash, or a drive letter.
    return name.startswith(("/", "\\")) or (
        len(name) > 1 and name[0].isalpha() and name[1] == ":"
    )


def resolve_initial(
    name: str, origin: tx.Optional[path.FilenameLike] = None
) -> path.Path:
    """Find the initial transform file that a parameter file names.

    As in elastix (`TransformBase::ReadFromFile`), the name is used as it is
    when it is absolute or exists relative to the working directory, and is
    otherwise resolved against the directory of the naming file. Since the
    absolute paths that elastix writes break when the output folder moves, a
    name found nowhere is also looked up by its base name next to the file.

    Raises
    ------
    ParserExistsError
        If the file cannot be found. It is a `FileNotFoundError`.
    """
    candidates = []
    directory = None
    if origin is not None:
        directory = os.path.dirname(os.fspath(origin))
    if _is_absolute(name) or os.path.exists(name) or not directory:
        candidates.append(name)
    if directory:
        if not _is_absolute(name):
            candidates.append(os.path.join(directory, name))
        base = name.replace("\\", "/").rsplit("/", 1)[-1]
        candidates.append(os.path.join(directory, base))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return path.Path(candidate)
    where = f" (named by {origin})" if origin is not None else ""
    raise ParserExistsError(
        f"Cannot find the elastix initial transform {name!r}{where}. "
        f"Pass initial=<file name> to say where it is, or initial=False "
        f"to read this transform alone."
    )


def _load_initial(filename: path.FilenameLike, **kwargs) -> ElastixTransform:
    """Read an initial transform, in whichever syntax it is written."""
    name = os.fspath(filename)
    cls = (
        ElastixTomlTransform
        if name.lower().endswith(".toml")
        else ElastixParameterTransform
    )
    return cls.from_filename(filename, **kwargs)


def _reject(error: tx.Union[bool, tx.Type[Exception]], message: str) -> float:
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message)
    return Confidence.NO
