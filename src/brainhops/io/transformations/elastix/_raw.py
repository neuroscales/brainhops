"""The record of an elastix transform parameter file.

A transform parameter file is short and is read in one pass, so its
record holds the whole file: the parameter map, the syntax in which the
file is written, and the text that it was read from. A file may name an
initial transform, which lives in another file, and the record of that
file is held in the record that names it, so that the record of the last
file of a chain holds the whole chain.
"""

__all__ = ["ElastixRaw", "initial_filename", "resolve_initial"]

import os

import typing_extensions as tx
from bagof.magic import Factory, NoRepr, replace

from brainhops._core import path
from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._internal_api import Raw
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
    _passthrough_from_fileobj,
)

from ._parser import (
    ParameterMap,
    format_text_map,
    format_toml_map,
    get_string,
    is_transform_map,
    read_text_map,
    read_toml_map,
)

#: The value elastix writes when there is no initial transform.
_NO_INITIAL = "NoInitialTransform"

#: The two syntaxes of a parameter file.
_SYNTAXES = ("text", "toml")

_InitialOption = tx.Union[bool, None, path.FilenameLike]
"""What a reader does with the initial transform that a file names."""


class ElastixRaw(
    Raw,
    TextFileReader,
    TextFileWriter,
    eq=False,
):
    """Record of an elastix transform parameter file, read in one pass.

    A transform parameter file, such as `TransformParameters.0.txt`, is a
    short text file, so the record holds all of it: the parameter map in
    file order, the syntax of the file, which is the classic text syntax
    or TOML, and the text that the file was read from. The metadata of a
    file,
    [`ElastixMetadata`][brainhops.io.transformations.elastix.ElastixMetadata],
    holds a record, and an
    [`ElastixTransform`][brainhops.io.transformations.elastix.ElastixTransform]
    decodes its chain of transformations from the record of its metadata.

    A file may name an initial transform in
    `InitialTransformParameterFileName`, which lives in another file. The
    record of that file is read when the record is read, and it is held in
    `initial`, so a record holds the whole chain of files that it names.
    The name in the parameter map is kept as the file stores it.

    The text is written back as it is while it still parses to the
    parameter map, so that a file that is read and written again keeps
    its bytes, whichever program wrote it. Otherwise, and for a record
    built from a parameter map, the map is formatted as elastix writes it.

    A record is never changed in place, so that several objects can share
    it. The class is frozen, and the transform parameters, which are an
    array, are read-only after a read. A changed record is built with
    [`replace`][bagof.magic.replace], and [`copy`][] returns a record
    whose parameter map can be changed freely. Records compare by
    identity, because the transform parameters are an array.
    """

    parameters: NoRepr[ParameterMap] = Factory(dict)
    """The parameter map, in the order of the file.

    Each name maps to a tuple of strings and numbers, except the transform
    parameters, which are a float64 array.
    """

    syntax: str = "text"
    """The syntax of the file, `"text"` for `(Name value ...)` lines or
    `"toml"` for `Name = value` lines."""

    text: NoRepr[tx.Optional[str]] = None
    """The text that the parameter map was read from, or `None`.

    The text is `None` for a record built from a parameter map, or read
    from lines, whose line endings are not known.
    """

    initial: NoRepr[tx.Optional["ElastixRaw"]] = None
    """The record of the file of the initial transform, or `None`.

    It is `None` when the file names no initial transform, and when the
    record was read without its initial transform (`initial=False`).
    """

    def __post_init__(self) -> None:
        if self.syntax not in _SYNTAXES:
            raise ValueError(
                f"The syntax of an elastix parameter file is 'text' or "
                f"'toml', not {self.syntax!r}."
            )

    # --- views --------------------------------------------------------

    @property
    def initial_filename(self) -> tx.Optional[str]:
        """The file name of the initial transform, as the map stores it.

        The name is `None` when the map names no initial transform.
        """
        return initial_filename(self.parameters)

    def copy(self) -> "ElastixRaw":
        """Return a copy of the record whose parameter map can be changed.

        The parameter map is copied into a new dictionary, whose values are
        shared, and the text and the record of the initial transform are
        kept, since [`to_text`][] writes the text only while it still
        parses to the map.

        Returns
        -------
        ElastixRaw
            The copy.
        """
        return replace(self, parameters=dict(self.parameters))

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score text as the text of a transform parameter file.

        The parameters must name a `Transform` and include its parameters,
        or at least their number. This requirement excludes registration
        parameter files, which share the syntax of transform files.

        Parameters
        ----------
        text : str
            The text of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not a transform
            parameter file.
        **kwargs : Any
            Options of the reader. The option `syntax` selects the syntax,
            which is otherwise recognised from the text, and any other
            option is ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        try:
            pmap, _ = _parse(text.splitlines(), kwargs.get("syntax"))
        except (ParserContentError, ValueError) as e:
            return _reject(error, f"Not an elastix parameter file: {e}")
        if is_transform_map(pmap):
            return Confidence.CERTAIN
        return _reject(
            error,
            "Not an elastix transform parameter file: it names no "
            "Transform, or no TransformParameters.",
        )

    # --- from ---------------------------------------------------------

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        *,
        syntax: tx.Optional[str] = None,
        initial: _InitialOption = True,
        encoding: str = "utf-8",
    ) -> "ElastixRaw":
        """Read the record of a file, and of the files of its chain.

        The bytes of the file are read and decoded, so that the text keeps
        its line endings. As in elastix, the syntax is TOML when the name
        ends in `.toml`, and the classic text syntax otherwise.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        syntax : {"text", "toml"}, optional
            The syntax of the file, which by default follows its extension.
        initial : bool or path-like, default=True
            The initial transform named in `InitialTransformParameterFileName`.
            With `True`, the file is found as elastix finds it (see
            [`resolve_initial`][]), and its record is read with the records
            of the files that it names in turn. With `False`, no initial
            file is read. A file name gives the file to read instead of
            the one that the map names.
        encoding : str, default="utf-8"
            The encoding of the file.

        Returns
        -------
        ElastixRaw
            The record.

        Raises
        ------
        ParserExistsError
            If the file or a file of its chain does not exist.
        ParserContentError
            If a file is not a transform parameter file, or if the chain
            loops back to one of its files.
        """
        return _read_file(filename, syntax, initial, encoding, ())

    @classmethod
    @_passthrough_from_fileobj
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "ElastixRaw":
        """Read the record from an open file, in text or binary mode.

        The name of the file, when it has one, is the `origin` of
        [`from_text`][], so it selects the syntax and locates a relative
        initial file.
        """
        kwargs.setdefault("origin", getattr(file, "name", None))
        return super().from_fileobj(file, **kwargs)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        syntax: tx.Optional[str] = None,
        initial: _InitialOption = True,
        origin: tx.Optional[path.FilenameLike] = None,
    ) -> "ElastixRaw":
        """Read the record from the text of a file, which it keeps.

        Parameters
        ----------
        text : str
            The text of the file.
        syntax : {"text", "toml"}, optional
            The syntax of the file. By default, it follows the extension of
            `origin`, and it is recognised from the text when there is no
            `origin`.
        initial : bool or path-like, default=True
            The initial transform, as for [`from_filename`][].
        origin : path-like, optional
            The file that the text comes from, against which a relative
            initial file is resolved. Without it, a relative name is looked
            up in the working directory only.

        Returns
        -------
        ElastixRaw
            The record.
        """
        return _read(text.splitlines(), text, syntax, initial, origin, ())

    @classmethod
    def from_lines(
        cls,
        lines: tx.Iterable[str],
        *,
        syntax: tx.Optional[str] = None,
        initial: _InitialOption = True,
        origin: tx.Optional[path.FilenameLike] = None,
    ) -> "ElastixRaw":
        """Same as [`from_text`][], for lines.

        The line endings are not known, so the record keeps no text, and
        it is formatted from its map when it is written.
        """
        return _read(list(lines), None, syntax, initial, origin, ())

    # --- to -----------------------------------------------------------

    @takes_error_policy
    def to_text(self) -> str:
        """Return the text of the file, ending with a newline.

        The text that the record was read from is returned while it still
        parses to the parameter map, in the syntax of the record.
        Otherwise, the map is formatted as elastix writes it, one parameter
        per line. The files of the chain are not written.

        Returns
        -------
        str
            The text of the file.

        Raises
        ------
        ValueError
            If a string of the map cannot be written in the syntax of the
            record, such as a string with a quote in the text syntax.
        """
        formatted = _format(self.parameters, self.syntax)
        if self.text is not None:
            try:
                pmap = _parse_map(self.text.splitlines(), self.syntax)
            except ParserContentError:
                pmap = None
            if pmap is not None and _format(pmap, self.syntax) == formatted:
                return self.text
        return formatted

    @takes_error_policy
    def to_lines(self) -> tx.Iterator[str]:
        return iter(self.to_text().splitlines())


# ----------------------------------------------------------------------
#   CHAINS
# ----------------------------------------------------------------------


def initial_filename(pmap: ParameterMap) -> tx.Optional[str]:
    """Return the file name of the initial transform in a parameter map.

    elastix reads `InitialTransformParameterFileName`, or else the
    deprecated `InitialTransformParametersFileName` (`TransformBase::
    ReadFromFile` in `Core/ComponentBaseClasses/elxTransformBase.hxx`). The
    value `"NoInitialTransform"` means that there is no initial transform,
    so the function then returns `None`, as it does when neither key is
    set.

    Parameters
    ----------
    pmap : dict
        The parameter map.

    Returns
    -------
    str or None
        The name, as the map stores it.
    """
    for key in (
        "InitialTransformParameterFileName",
        "InitialTransformParametersFileName",
    ):
        name = get_string(pmap, key)
        if name:
            return None if name == _NO_INITIAL else name
    return None


def resolve_initial(
    name: str, origin: tx.Optional[path.FilenameLike] = None
) -> path.Path:
    """Find the initial transform file that a parameter file names.

    As in elastix (`TransformBase::ReadFromFile`), the name is used as it
    is when it is absolute or exists relative to the working directory,
    and it is otherwise resolved against the directory of the naming file.
    The absolute paths that elastix writes break when the output folder
    moves, so a name that is found nowhere else is also looked up by its
    base name next to the naming file.

    Parameters
    ----------
    name : str
        The name that the parameter file stores.
    origin : path-like, optional
        The naming file.

    Returns
    -------
    path.Path
        The path of the initial file.

    Raises
    ------
    ParserExistsError
        If the file cannot be found. This error is a subclass of
        `FileNotFoundError`.
    """
    candidates = []
    directory = None
    if origin is not None:
        directory = os.path.dirname(os.fspath(origin))
    if is_absolute(name) or os.path.exists(name) or not directory:
        candidates.append(name)
    if directory:
        if not is_absolute(name):
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


def is_absolute(name: str) -> bool:
    """Tell whether elastix takes a file name as absolute.

    This is the test of `TransformBase::ReadFromFile`: a name is absolute
    if it starts with a slash, a backslash or a drive letter.

    Parameters
    ----------
    name : str
        The file name.

    Returns
    -------
    bool
        Whether the name is absolute.
    """
    return name.startswith(("/", "\\")) or (
        len(name) > 1 and name[0].isalpha() and name[1] == ":"
    )


def syntax_of(filename: tx.Optional[path.FilenameLike]) -> tx.Optional[str]:
    """Return the syntax of a file from its name, or `None` without a name.

    elastix reads a file whose last extension is `.toml` as TOML, and any
    other file in the classic text syntax (`ParameterFileParser::
    ReadParameterFile` in
    `Common/ParameterFileParser/itkParameterFileParser.cxx`, elastix
    3d044f8).

    Parameters
    ----------
    filename : path-like, optional
        The name of the file.

    Returns
    -------
    str or None
        `"toml"`, `"text"`, or `None` when there is no name.
    """
    if filename is None:
        return None
    return "toml" if os.fspath(filename).lower().endswith(".toml") else "text"


def _read_file(
    filename: path.FilenameLike,
    syntax: tx.Optional[str],
    initial: _InitialOption,
    encoding: str,
    visited: tx.Tuple[str, ...],
) -> ElastixRaw:
    """Read the record of a file by its name, with its chain.

    Raises
    ------
    ParserExistsError
        If the file does not exist.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    if not path.exists(filename):
        raise ParserExistsError(f"No such file: {filename}")
    with path.Path(filename).open("rb") as file:
        text = file.read().decode(encoding)
    if syntax is None:
        syntax = syntax_of(filename)
    return _read(text.splitlines(), text, syntax, initial, filename, visited)


def _read(
    lines: tx.List[str],
    text: tx.Optional[str],
    syntax: tx.Optional[str],
    initial: _InitialOption,
    origin: tx.Optional[path.FilenameLike],
    visited: tx.Tuple[str, ...],
) -> ElastixRaw:
    """Parse a record and read the records of the chain that it names.

    `visited` lists the real paths of the files that name this one, so
    that a chain that loops back is refused.

    Raises
    ------
    ParserContentError
        If the lines are not a transform parameter file, or if the chain
        loops back to one of its files.
    """
    if syntax is None:
        syntax = syntax_of(origin)
    pmap, syntax = _parse(lines, syntax)
    if not is_transform_map(pmap):
        raise ParserContentError(
            "Not an elastix transform parameter file: it names no "
            "Transform, or no TransformParameters."
        )
    name = initial_filename(pmap)
    record = None
    if initial is True and name is not None:
        filename = resolve_initial(name, origin)
        key = os.path.realpath(os.fspath(filename))
        here = None if origin is None else os.path.realpath(os.fspath(origin))
        if key in visited or key == here:
            raise ParserContentError(
                f"The elastix initial transforms loop back to {filename}."
            )
        visited = (*visited, key) if here is None else (*visited, key, here)
        record = _read_file(filename, None, True, "utf-8", visited)
    elif initial not in (True, False, None):
        record = _read_file(initial, None, True, "utf-8", ())
    return ElastixRaw(
        parameters=pmap, syntax=syntax, text=text, initial=record
    )


# ----------------------------------------------------------------------
#   SYNTAXES
# ----------------------------------------------------------------------


def _parse_map(lines: tx.Iterable[str], syntax: str) -> ParameterMap:
    """Parse the lines of a file in a known syntax."""
    if syntax == "toml":
        return read_toml_map(lines)
    if syntax == "text":
        return read_text_map(lines)
    raise ValueError(
        f"The syntax of an elastix parameter file is 'text' or 'toml', not "
        f"{syntax!r}."
    )


def _parse(
    lines: tx.List[str], syntax: tx.Optional[str]
) -> tx.Tuple[ParameterMap, str]:
    """Parse the lines of a file, and return the map and its syntax.

    Without a syntax, the classic text syntax is tried first, and TOML
    next. The two syntaxes share no line that holds a parameter: a text
    line is between brackets, and a TOML line has an equal sign outside
    of any brackets.
    """
    if syntax is not None:
        return _parse_map(lines, syntax), syntax
    try:
        return read_text_map(lines), "text"
    except ParserContentError:
        return read_toml_map(lines), "toml"


def _format(pmap: ParameterMap, syntax: str) -> str:
    """Format a parameter map in a syntax, one parameter per line."""
    if syntax == "toml":
        lines = format_toml_map(pmap)
    else:
        lines = format_text_map(pmap)
    return "".join(line + "\n" for line in lines)


def _reject(error: tx.Union[bool, tx.Type[Exception]], message: str) -> float:
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message)
    return Confidence.NO
