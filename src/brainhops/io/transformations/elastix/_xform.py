__all__ = [
    "ElastixTransform",
    "ElastixParameterTransform",
    "ElastixTomlTransform",
]

# stdlib
import os
from warnings import warn

# dependencies
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, NoEq

# core
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.geometry import Geometry

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    ParserExistsError,
    ParserNotImplementedError,
    SnifferContentError,
    TextFileParserWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base import WritableFileBasedTransformation
from brainhops.io.transformations.itk._common import ItkStruct

# locals
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

#: The value elastix writes when a transform has no initial transform.
_NO_INITIAL = "NoInitialTransform"


# `TextFileParserWriter` bridges bytes and text for a text format, and
# must come before `WritableFileBasedTransformation`, whose writer knows
# no encoding (see `LtaTransformation`).
class ElastixTransform(
    TextFileParserWriter,
    _xforms.Sequence,
    WritableFileBasedTransformation,
    repr=HIDE_IF_NONE,
    reverse=False,  # `transformations` stays the first positional field.
):
    """
    A transformation stored in an elastix transform parameter file.

    It is the
    [`Sequence`][brainhops.datamodel.transformations.Sequence] that maps
    fixed-image LPS coordinates to moving-image LPS coordinates (the
    direction in which transformix pulls the moving image onto the fixed
    grid): the blocks of the initial transforms that the file chains to,
    first, then the block of the file's own transform. Every block is an
    ITK block (see [`brainhops.io.transformations.elastix`][]).

    The raw parameter map is kept in `parameter_map`, and the initial
    transform -- itself an `ElastixTransform` -- in `initial`.

    Abstract: it is not decorated with `@register_format`. Its two
    syntaxes, [`ElastixParameterTransform`][] (`.txt`) and
    [`ElastixTomlTransform`][] (`.toml`), register themselves.
    """

    HINTS = ("elastix", "transformix")

    # Not compared: the transform parameters are an array.
    parameter_map: tx.Annotated[ParameterMap, NoEq()] = Factory(
        dict, repr=False
    )
    """
    The parameters of the file, in the order in which they were read.

    Each name maps to a tuple of values -- a `str` for a quoted value, an
    `int` or a `float` for a number -- except the transform parameters,
    which are a `float64` array.
    """

    initial: tx.Optional[tx.Any] = None
    """
    The initial transform that the file chains to, as an
    `ElastixTransform`, or `None` when it has none or when it was not
    followed (`initial=False`).
    """

    # --- syntax -------------------------------------------------------

    @classmethod
    def _read_map(cls, lines: tx.Iterable[str]) -> ParameterMap:
        raise NotImplementedError

    @classmethod
    def _format_map(cls, pmap: ParameterMap) -> tx.Iterator[str]:
        raise NotImplementedError

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the parser is that the lines are an
        elastix transform parameter file.

        The whole map is parsed: it must name a `Transform` and carry its
        parameters. elastix's *registration* parameter files share the
        syntax, and name a `Transform` too, but carry no parameters, and
        are not claimed.
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
        """Score the text of a file (see `sniff_lines`)."""
        return cls.sniff_lines(text.splitlines(), error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Build the object from an open file. Its `name`, when it has
        one, is where a relative initial transform is looked for."""
        kwargs.setdefault("origin", getattr(file, "name", None))
        with preserve_position(file):
            content = file.read()
        if isinstance(content, (bytes, bytearray)):
            return cls.from_bytes(content, **kwargs)
        return cls.from_text(content, **kwargs)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Build the object from the text of a file."""
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
        """
        Build the object from the lines of an elastix parameter file.

        Parameters
        ----------
        lines : Iterable[str]
            The lines of the file.
        initial : bool | FilenameLike | Transformation, default=True
            What to do with the initial transform that the file names in
            `InitialTransformParameterFileName`:

            - `True`: read it, and the ones it names in turn, as
              transformix does (see [`resolve_initial`][..resolve_initial]);
            - `False`: do not read it -- the chain then holds the file's
              own transform alone;
            - a file name: read that file instead;
            - a transformation: use it as the initial transform.
        origin : FilenameLike, optional
            The file the lines were read from, against which a relative
            initial transform is resolved. It is set by `from_filename`.

        Raises
        ------
        ParserNotImplementedError
            If the file uses a transform that is not supported, or
            combines with its initial transform by addition
            (`HowToCombineTransforms "Add"`), which a chain of
            transformations cannot express.
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
        # Decode the file's own transform now, so that an unsupported
        # transform fails here rather than on first use.
        obj.block  # noqa: B018
        return obj

    # --- chain --------------------------------------------------------

    @property
    def initial_filename(self) -> tx.Optional[str]:
        """The initial transform that the file names, if any."""
        return initial_filename(self.parameter_map)

    @smartproperty(cache=True)
    def block(self) -> tx.Optional[ItkStruct]:
        """The ITK block of the file's own transform."""
        if not self.parameter_map:
            return None
        return map_to_block(self.parameter_map)

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations, in the order they are applied: the
        initial transform's (flattened), then the file's own block.

        elastix composes a transform with its initial transform as
        `T(x) = T1(T0(x))` (`AdvancedCombinationTransform::
        TransformPointUseComposition`): the initial transform applies
        first. Assigning to it overrides the decoded chain, and is what
        the writer then encodes.
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
        """
        The geometry of the fixed image (`Size`, `Index`, `Spacing`,
        `Origin`, `Direction`), which is the grid transformix resamples
        the moving image onto. Its transformation maps voxels of that
        grid to LPS millimetres. `None` when the file does not say.
        """
        return fixed_geometry(self.parameter_map)

    # --- to -----------------------------------------------------------

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the transformation to a file.

        The content is built before the file is opened, so a
        transformation that the format cannot hold is refused without
        creating or truncating the file.
        """
        content = self.to_text(**kwargs) + "\n"
        with path.Path(filename).open("w") as f:
            f.write(content)

    def to_map(self) -> ParameterMap:
        """
        The parameter map that encodes this transformation.

        - A transformation read from a file, and not modified, is written
          as it was read, initial transform file name included. The
          initial transform itself is not written: it is its own file.
        - Otherwise the chain must be a single transformation: an elastix
          or ITK block (a translation, an Euler, a similarity, an affine
          or a B-spline), or anything that reduces to an affine, which is
          written as an `AffineTransform` centered on the origin. The
          fixed-image geometry and the other non-transform parameters of
          `parameter_map` are kept. It has no initial transform.

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
        """The lines of the parameter file (see `to_map`)."""
        return self._format_map(self.to_map())


# ----------------------------------------------------------------------
#   SYNTAXES
# ----------------------------------------------------------------------


@register_format
class ElastixParameterTransform(ElastixTransform):
    """
    A transformation stored in a classic elastix transform parameter file
    (`TransformParameters.0.txt`), one `(Name value ...)` per line.
    """

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
    """
    A transformation stored in an elastix TOML transform parameter file
    (`TransformParameters.0.toml`), one `Name = value` per line.
    """

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
    """The initial transform a map names, if any.

    elastix reads `InitialTransformParameterFileName` and, failing it,
    the deprecated `InitialTransformParametersFileName` (with an "s");
    `"NoInitialTransform"` means none.
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
    """
    Find the initial transform file that a parameter file names.

    elastix (`TransformBase::ReadFromFile`) uses the name as it is when
    it is absolute, or when it exists relative to the working directory;
    otherwise it looks for it relative to the directory of the parameter
    file that names it. The same is done here, with one more step: a name
    that is found nowhere is looked for, by its base name, next to the
    parameter file -- elastix writes absolute paths, which break when a
    registration's output folder is moved.

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
