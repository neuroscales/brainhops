"""Choice of the registered parser that reads an input.

The functions in this module choose, among the parser classes collected in a
registry, the one that reads a given input. The choice depends only on the
registry and on the input, not on the class hierarchy. When no single parser
can be chosen, the error message either reports the error raised by each
parser that was tried, or lists the formats that could not be told apart and
explains how to select one of them.
"""

__all__ = ["Source", "parse", "sniff"]

import inspect
from collections.abc import Iterable
from io import BytesIO, StringIO
from os import DirEntry, PathLike, sep
from os.path import basename
from urllib.parse import urlsplit

import typing_extensions as tx
from bagof.magic import fields

from brainhops._core import path
from brainhops.io.base.parsers import (
    AmbiguousFormatError,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    SnifferExistsError,
)
from brainhops.io.base.specs import SourceSpec, format_hints, parser_for

_T = tx.TypeVar("_T")

# Formats that are not registered because an optional dependency is missing.
# Each entry maps a format hint to the package that the format needs and to the
# brainhops extra that installs it.
_MISSING_FORMATS: tx.Dict[str, tx.Tuple[str, str]] = {}


def register_missing_format(
    hints: tx.Iterable[str], package: str, extra: str
) -> None:
    """Record that the formats answering to `hints` need `package`.

    When one of these formats is later requested by hint, the error message
    names the extra to install.
    """
    for hint in hints:
        _MISSING_FORMATS[str(hint).lower()] = (package, extra)


def _missing_formats(hints: tx.Iterable[str]) -> str:
    needs = sorted(
        {_MISSING_FORMATS[h] for h in hints if h in _MISSING_FORMATS}
    )
    return "".join(
        f" This format needs {package}, which is not installed: "
        f"pip install brainhops[{extra}]"
        for package, extra in needs
    )


class Source:
    """An input to parse, which can be read again from the start.

    Dispatch sniffs and parses an input with several parsers in turn, and each
    attempt may consume the input. To make repeated reads possible, paths and
    in-memory content are kept as they are, seekable streams are rewound
    before each read, non-seekable streams are read once into memory, and
    one-shot iterables of lines are collected into a list.

    A `str` is a path. Text held in memory must be wrapped with [`content`][],
    because the type alone cannot tell the two apart.
    """

    def __init__(self, other: tx.Any) -> None:
        self.other = other
        self.pos = None
        self.buffer = None
        self.factory = None
        self.path = other if isinstance(other, (str, PathLike)) else None

        if hasattr(other, "read"):
            if self._seekable(other):
                try:
                    self.pos = other.tell()
                except Exception:
                    self.pos = None
            else:
                self.buffer = other.read()
                self.factory = (
                    BytesIO
                    if isinstance(self.buffer, (bytes, bytearray))
                    else StringIO
                )
        elif isinstance(other, (str, bytes, bytearray, PathLike)):
            pass  # str and bytes are Iterable, so they are tested first.
        elif isinstance(other, Iterable) and not isinstance(
            other, (list, tuple)
        ):
            self.other = list(other)

    @staticmethod
    def _seekable(other: tx.Any) -> bool:
        if not hasattr(other, "seek"):
            return False
        try:
            return bool(other.seekable())
        except AttributeError:
            return True  # Old-style file objects lack seekable().
        except Exception:
            return False

    def get(self) -> tx.Any:
        """Return a fresh view of the input, positioned at its start."""
        if self.buffer is not None:
            return self.factory(self.buffer)
        if self.pos is not None:
            try:
                self.other.seek(self.pos)
            except Exception:
                pass
        if isinstance(self.other, list):
            return iter(self.other)
        return self.other

    @classmethod
    def content(cls, other: tx.Any) -> "Source":
        """Wrap in-memory content, where a `str` is text and not a path."""
        source = cls(other)
        source.path = None
        return source

    @property
    def missing(self) -> tx.Optional[path.Path]:
        """The path that the input names, if that file is missing.

        The value is `None` if the file exists or its existence cannot be
        checked.
        """
        if self.path is None:
            return None
        filename = path.Path(self.path)
        try:
            exists = filename.exists()
        except Exception:
            return None  # A remote store may not support existence checks.
        return None if exists else filename

    @property
    def name(self) -> tx.Optional[str]:
        """The base name of the file that the input names, or `None`."""
        if self.path is not None:
            return _to_filename(self.path)
        if isinstance(self.other, (str, bytes, bytearray, PathLike)):
            # Content wrapped by Source.content is never a file name.
            return None
        return _to_filename(self.other)  # An open file is named by its .name.

    def __repr__(self) -> str:
        name = self.name
        return f"file {name!r}" if name else "input content"


def _to_filename(other: tx.Any) -> tx.Optional[str]:
    """Return the base name of the file that `other` names, or `None`.

    The name is taken from the text of the path without touching storage,
    because `os.fspath` raises on remote paths and downloads cloud paths.
    For a URL, the name is the last segment of its path, without query or
    fragment, so `https://host/x.nii.gz?token=...` names `x.nii.gz`. For an
    fsspec chain such as `simplecache::s3://...`, the name comes from the
    last link.
    """
    # TODO: move to brainhops._core.path, and rename, since it returns a
    # base name rather than a file name.

    if isinstance(other, DirEntry):
        # str() of a DirEntry gives its repr.
        text = other.path
    elif isinstance(other, (str, path.PathLike)):
        text = str(other)
    else:
        text = getattr(other, "name", None)
        if not isinstance(text, str):
            return None
    return _base_name(text)


def _base_name(text: str) -> str:
    """Return the last component of a local path or of the path of a URL."""
    # TODO: use bagof.paths.Path instead.
    if "::" in text and "://" in text:
        # In an fsspec chain, the last link is the file, and the other links
        # are layers such as caches or archives.
        text = text.rsplit("::", 1)[-1]
    if _has_scheme(text):
        # Only the path of the URL is kept, because the query and the
        # fragment may contain slashes and dots.
        text = urlsplit(text).path
        # Trailing slashes are stripped so that a directory store, such as a
        # .zarr folder, still gets a name.
        return text.rstrip("/").rsplit("/", 1)[-1]
    return basename(text.rstrip("/" + sep))


def _has_scheme(text: str) -> bool:
    """Return whether `text` starts with a URL scheme such as `s3:`.

    A scheme has at least two characters, so a drive such as `C:` is not one.
    """
    # TODO: use bagof.paths.Path instead.
    scheme, colon, _ = text.partition(":")
    return (
        bool(colon)
        and len(scheme) > 1
        and scheme.isascii()
        and scheme[0].isalpha()
        and all(c.isalnum() or c in "+-." for c in scheme)
    )


def _match_name(name: str, cls: type) -> tx.Optional[tx.Tuple[int, int]]:
    """Measure how much of a file name a format accounts for.

    The result holds the lengths of the longest matching declared extension and
    prefix, so that `.nii.gz` beats `.gz` and `iy_` beats `y_`. The result is
    `None` when no extension matches, or when the format declares prefixes
    and none of them matches, because a prefix is a requirement rather than a
    hint.
    """
    extension = None
    for ext in cls.EXTENSIONS:
        if name.endswith(ext) and (extension is None or len(ext) > extension):
            extension = len(ext)
    if extension is None:
        return None

    prefix = 0
    if cls.PREFIXES:
        for pre in cls.PREFIXES:
            if name.startswith(pre) and len(pre) > prefix:
                prefix = len(pre)
        if not prefix:
            return None

    return extension, prefix


def _registry_depth(cls: type) -> int:
    """Count the dispatcher levels above a format.

    A format registered under `ImageFormat` is classified more finely than
    one registered only under the root, and therefore wins a tie. Only the
    ancestors that own a registry are counted, rather than all ancestors,
    because the depth of the class hierarchy says nothing about how well a
    format matches.
    """
    return sum(1 for base in cls.__mro__ if "_REGISTRY" in base.__dict__)


def _drop_base_classes(
    candidates: tx.List[tx.Tuple[type, tx.Any]],
) -> tx.List[tx.Tuple[type, tx.Any]]:
    """Drop candidates that are base classes of another candidate.

    A subclass is more specific than its base class, so the base class is
    dropped, as in `functools.singledispatch`.
    """
    classes = [cls for cls, _ in candidates]
    return [
        (cls, info)
        for cls, info in candidates
        if not any(
            other is not cls and issubclass(other, cls) for other in classes
        )
    ]


def _specificity(
    cls: type,
    match: tx.Optional[tx.Tuple[int, int]],
    score: tx.Optional[tx.Dict[type, float]],
) -> tx.Tuple:
    """Return the sort key of a candidate, most specific first.

    The criteria, in order of precedence, concern what the file or the parser
    declares and never the import order:

    1. the confidence of the sniffer;
    2. the length of the matching extension;
    3. the length of the matching required prefix;
    4. the registry depth, that is, the number of dispatchers above the
       format;
    5. the narrowness of the declared extensions: a parser declaring `.lta`
       alone beats one declaring five, and both beat one declaring none;
    6. the explicit `PRIORITY`.

    Candidates with equal keys cannot be told apart, and dispatch does not
    guess between them.
    """
    extension, prefix = match or (0, 0)
    return (
        -(score[cls] if score else 0.0),
        -extension,
        -prefix,
        -_registry_depth(cls),
        0 if cls.EXTENSIONS else 1,
        len(cls.EXTENSIONS),
        -cls.PRIORITY,
    )


def _tiers(
    candidates: tx.List[tx.Tuple[type, tx.Any]],
    score: tx.Optional[tx.Dict[type, float]] = None,
) -> tx.List[tx.List[type]]:
    """Group candidates into tiers of equal specificity, best first.

    A tier with several members holds parsers that no criterion can separate.
    [`parse`][] tries every member of such a tier and reports an ambiguity if
    more than one of them succeeds.
    """
    groups = {}
    for cls, match in _drop_base_classes(candidates):
        groups.setdefault(_specificity(cls, match, score), []).append(cls)
    return [groups[key] for key in sorted(groups)]


def _candidates(
    source: "Source",
    registry: tx.Set[type],
    fn_sniff: str,
    errors: tx.Optional[tx.List[tx.Tuple[type, str, Exception]]] = None,
    allowed: tx.Optional[tx.Set[type]] = None,
    **kwargs,
) -> tx.List[tx.List[type]]:
    """Rank the formats that could read `source`, best tier first.

    [`parse`][] and [`sniff`][] share this ranking, so they always agree. The
    sniffer score and the file name match are combined into a single ranking.
    Ranking by extension alone would put every `.nii` reader into one
    ambiguous tier and would discard the scores that separate them.
    """
    name = source.name
    scores: tx.Dict[type, float] = {}
    candidates: tx.List[tx.Tuple[type, tx.Optional[tx.Tuple[int, int]]]] = []
    for subclass in registry:
        if allowed is not None and subclass not in allowed:
            continue
        match = _match_name(name, subclass) if name else None
        try:
            score = float(
                getattr(subclass, fn_sniff)(
                    source.get(), error=False, **kwargs
                )
            )
        except Exception as e:
            if errors is not None:
                errors.append((subclass, fn_sniff, e))
            score = 0.0
        # Sniffers are optional and the file name is evidence too, so a
        # parser whose extension matches stays in the running, ranked below
        # every parser that scored.
        if score > 0 or match is not None:
            scores[subclass] = score
            candidates.append((subclass, match))

    return _tiers(candidates, scores)


def parse(
    source: "Source",
    registry: tx.Set[tx.Type[_T]],
    fn_parse: str,
    fn_sniff: str,
    brute: bool = False,
    hints: tx.Iterable[str] = (),
    hint: tx.Optional[tx.Union[str, tx.Iterable[str]]] = None,
    options: tx.Optional[tx.Mapping[str, tx.Union[str, SourceSpec]]] = None,
    **kwargs,
) -> _T:
    """Read `source` with the registered parser that matches it best.

    Formats are ranked by sniffer score and file name match, and they are
    tried from the most specific. When several formats tie, every tied format
    is tried: if exactly one succeeds, its result is returned, and if several
    succeed, the input is ambiguous. If `brute` is set and no ranked format
    succeeds, every remaining allowed format is tried in turn.

    !!! note "Why brute force is opt-in"
        A reader applied to arbitrary bytes may succeed on a partial parse and
        quietly return an object of the wrong format.

    Parameters
    ----------
    source : Source
        The input.
    registry : set of type
        The formats to choose between.
    fn_parse : str
        The name of the parsing method, such as `"from_file"`.
    fn_sniff : str
        The name of the sniffing method, such as `"sniff_file"`.
    brute : bool, default=False
        Whether to try every format when none recognizes the input.
    hints : str or iterable of str, default=()
        Only formats that answer to one of these hints are allowed, and these
        formats are tried even if none of them recognizes the input.
    hint : str or iterable of str, optional
        Further hints, merged with `hints` and applied in the same way.
    options : mapping, optional
        Source options. Only formats that accept every option are allowed.

    Raises
    ------
    AmbiguousFormatError
        If several equally specific parsers can all read the input.
    ParserExistsError
        If the input names a file that does not exist.
    ParserContentError
        If an option is given twice, if no parser accepts the hints and
        options, or if no parser could read the input.
    """

    errors: tx.List[tx.Tuple[type, str, Exception]] = []
    tried: tx.List[type] = []
    if not registry:
        raise _failure(source, [], errors)
    requested_hints = _normalize_hints(hints, hint)
    source_options = dict(options or {})
    duplicate = set(kwargs).intersection(source_options)
    if duplicate:
        names = ", ".join(sorted(duplicate))
        raise ParserContentError(
            f"Source options were supplied more than once: {names}."
        )

    allowed = {
        subclass
        for subclass in registry
        if (not requested_hints or format_hints(subclass) & requested_hints)
        and _accepts_options(subclass, source_options)
    }
    if not allowed:
        details = []
        if requested_hints:
            details.append(
                "hints " + ", ".join(sorted(repr(h) for h in requested_hints))
            )
        if source_options:
            details.append(
                "options "
                + ", ".join(sorted(repr(name) for name in source_options))
            )
        suffix = " for " + " and ".join(details) if details else ""
        raise ParserContentError(
            f"No registered parser accepts {source}{suffix}."
            + _missing_formats(requested_hints)
        )

    def attempt(subclass: type) -> tx.Tuple[bool, tx.Any]:
        tried.append(subclass)
        try:
            bound = _bind_options(subclass, source_options)
            bound.update(kwargs)
            return True, getattr(subclass, fn_parse)(source.get(), **bound)
        except Exception as e:
            errors.append((subclass, fn_parse, e))
            return False, None

    def walk(tiers: tx.List[tx.List[type]]) -> tx.Tuple[bool, tx.Any]:
        for tier in tiers:
            if len(tier) == 1:
                ok, result = attempt(tier[0])
                if ok:
                    return True, result
                continue
            # If only one member of the tier can read the content, the tie
            # was only apparent.
            winners = []
            for subclass in tier:
                ok, result = attempt(subclass)
                if ok:
                    winners.append((subclass, result))
            if len(winners) == 1:
                return True, winners[0][1]
            if len(winners) > 1:
                raise AmbiguousFormatError(
                    _ambiguity_message(
                        f"{source}", [cls for cls, _ in winners]
                    )
                )
        return False, None

    tiers = _candidates(
        source, registry, fn_sniff, errors, allowed=allowed, **kwargs
    )
    # Hints are an explicit allowlist: if nothing in it sniffs positively, for
    # example because of an unknown extension, try the allowed formats rather
    # than widening to the full registry.
    if requested_hints and not tiers:
        tiers = _tiers([(subclass, None) for subclass in allowed])
    ok, result = walk(tiers)
    if ok:
        return result

    # --- Brute force ---------------------------------------------------
    if brute:
        for subclass in sorted(allowed, key=lambda c: c.__qualname__):
            if subclass in tried:
                continue
            ok, result = attempt(subclass)
            if ok:
                return result

    # --- Failure) Raise -----------------------------------------------
    raise _failure(source, list(allowed), errors)


def _field_annotations(cls: type) -> tx.Dict[str, tx.Any]:
    """Return the resolved field annotations of `cls`.

    `Annotated` metadata is kept. If resolution fails, the raw annotations are
    merged along the MRO.
    """
    try:
        annotations = tx.get_type_hints(cls, include_extras=True)
    except Exception:
        annotations = {}
        for base in reversed(cls.__mro__):
            annotations.update(base.__dict__.get("__annotations__", {}))
    return annotations


def _option_fields(cls: type) -> tx.Dict[str, tx.Tuple[str, tx.Any]]:
    """Map each source option of `cls` to the field that it sets.

    Option names are the public aliases of keyword fields, and each name maps
    to the public name and the annotation of its field.
    """
    try:
        magic_fields = fields(cls)
    except (TypeError, AttributeError):
        magic_fields = ()
    annotations = _field_annotations(cls)
    result = {}
    for field in magic_fields:
        if not field.init or not field.kw:
            continue
        aliases = getattr(field, "aliases", (field.public_name,))
        for alias in aliases:
            if not alias.startswith("_"):
                result[alias] = (
                    field.public_name,
                    annotations.get(field.name, field.type),
                )
    return result


def _accepts_options(
    cls: type,
    options: tx.Mapping[str, tx.Union[str, SourceSpec]],
) -> bool:
    accepted = _option_fields(cls)
    return all(name in accepted for name in options)


def _bind_options(
    cls: type,
    options: tx.Mapping[str, tx.Union[str, SourceSpec]],
) -> tx.Dict[str, tx.Any]:
    accepted = _option_fields(cls)
    bound = {}
    for option, value in options.items():
        field_name, annotation = accepted[option]
        bound[field_name] = _parse_field_value(
            cls, field_name, annotation, value
        )
    return bound


def _parse_field_value(
    owner: type,
    field_name: str,
    annotation: tx.Any,
    value: tx.Union[str, SourceSpec],
) -> tx.Any:
    parser = parser_for(annotation)
    spec = value if isinstance(value, SourceSpec) else SourceSpec(path=value)
    if parser is None:
        if isinstance(value, SourceSpec):
            raise TypeError(
                f"Nested source for {owner.__name__}.{field_name} has no "
                "registered field parser."
            )
        return value
    if isinstance(parser, type) and hasattr(parser, "from_spec"):
        return parser.from_spec(spec)
    if hasattr(parser, "from_spec"):
        return parser.from_spec(spec)
    if callable(parser):
        return parser(spec)
    raise TypeError(
        f"Parser for {owner.__name__}.{field_name} is not callable: "
        f"{parser!r}."
    )


def _failure(
    source: Source,
    registry: tx.List[type],
    errors: tx.List[tx.Tuple[type, str, Exception]],
) -> Exception:
    """Build the error that reports why parsing failed.

    The message lists the error raised by each parser, so that a bug in the
    correct reader is not hidden among the other failures, and the last error
    is chained as the cause. A missing file is reported as a
    [`ParserExistsError`][] rather than as every parser failing to open it.
    """
    if not registry:
        return ParserContentError(
            f"Cannot parse {source}: no parser is registered. Formats "
            f"register themselves on import -- is the format module "
            f"imported?"
        )
    missing = source.missing
    if missing is not None:
        failure = ParserExistsError(f"No such file: {missing}")
        if errors:
            failure.__cause__ = errors[-1][2]
        return failure
    if not errors:
        return ParserContentError(
            f"Cannot parse {source}: none of the "
            f"{len(registry)} registered parsers recognized it "
            f"({', '.join(sorted(cls.__name__ for cls in registry))})."
        )
    detail = "\n".join(
        f"  - {cls.__name__}.{fn}: {type(e).__name__}: {e}"
        for cls, fn, e in errors
    )
    failure = ParserContentError(f"Cannot parse {source}. Tried:\n{detail}")
    # The exception is returned rather than raised, so the cause is set by
    # hand.
    failure.__cause__ = errors[-1][2]
    return failure


def sniff(
    source: "Source",
    registry: tx.Set[type],
    fn_sniff: str,
    error: tx.Union[bool, tx.Type[Exception]] = False,
    what: str = "input content",
    hints: tx.Iterable[str] = (),
    hint: tx.Optional[tx.Union[str, tx.Iterable[str]]] = None,
    **kwargs,
) -> tx.Optional[type]:
    """Identify the registered format that would read `source`.

    The function returns a format class rather than a confidence score, and it
    ranks formats in the same way as [`parse`][].

    !!! note "`None` is not the same as a failed load"
        [`parse`][] can resolve a tie by trying each candidate. This function
        does not parse, so it reports a tie as `None`.

    Parameters
    ----------
    error : bool or type of Exception, default=False
        Whether to raise instead of returning `None`. An exception class
        replaces the default error type.
    what : str, default="input content"
        A description of the input for error messages.

    Returns
    -------
    type or None
        The best-matching format, or `None` if no single format stands out.

    Raises
    ------
    SnifferExistsError
        If `error` is set, no single format stands out, and the input names a
        missing file.
    AmbiguousFormatError
        If `error` is set and the top tier is tied.
    SnifferContentError
        If `error` is set and no format recognizes the input.
    """
    requested_hints = _normalize_hints(hints, hint)
    allowed = {
        subclass
        for subclass in registry
        if not requested_hints or format_hints(subclass) & requested_hints
    }
    tiers = _candidates(source, registry, fn_sniff, allowed=allowed, **kwargs)

    if tiers and len(tiers[0]) == 1:
        return tiers[0][0]

    if error:
        missing = source.missing
        if missing is not None:
            if error is True:
                error = SnifferExistsError
            raise error(f"No such file: {missing}")
        if tiers:
            if error is True:
                error = AmbiguousFormatError
            raise error(_ambiguity_message(what, tiers[0]))
        if error is True:
            error = SnifferContentError
        raise error(f"Nothing to sniff in {what}")

    return None


def _describe(cls: type) -> str:
    """Describe a format by the first paragraph of its own docstring.

    The paragraph is joined into one line without its final period. Inherited
    docstrings describe a base class and are ignored, and so is the docstring
    that bagof generates for a data model, which opens with an underlined
    heading.
    """
    doc = cls.__dict__.get("__doc__")
    if not isinstance(doc, str) or not doc.strip():
        return ""
    paragraph = inspect.cleandoc(doc).split("\n\n", 1)[0]
    lines = paragraph.splitlines()
    if len(lines) > 1 and set(lines[1].strip()) == {"-"}:
        return ""
    return " ".join(paragraph.split()).rstrip(".")


def _selecting_hint(
    cls: type, candidates: tx.Sequence[type]
) -> tx.Optional[str]:
    """Return the hint that selects `cls` among tied formats, or `None`.

    A selecting hint is one that no other tied format answers to. When several
    hints qualify, the hint with the fewest dotted parts is preferred, and then
    the shortest, so `itk` beats `itk.displacements`.
    """
    taken = set()
    for other in candidates:
        if other is not cls:
            taken |= format_hints(other)
    found = sorted(
        format_hints(cls) - taken,
        key=lambda hint: (hint.count("."), len(hint), hint),
    )
    return found[0] if found else None


def _ambiguity_message(subject: str, candidates: tx.Iterable[type]) -> str:
    """Explain which formats an input could be in.

    Each candidate is listed with the `hint=` value or the `load` call that
    selects it.
    """
    candidates = sorted(candidates, key=lambda cls: cls.__name__)
    lines = [
        f"Cannot tell which format {subject} is in: it can be read "
        f"equally well as any of these {len(candidates)} formats, which "
        f"would give different results."
    ]
    example = None
    for cls in candidates:
        line = f"  - {cls.__name__}"
        about = _describe(cls)
        if about:
            line += f" ({about})"
        hint = _selecting_hint(cls, candidates)
        if hint is None:
            line += f": `{cls.__name__}.load(path)`"
        else:
            line += f': hint="{hint}"'
            if example is None:
                example = hint
        lines.append(line)
    own_load = f"`{candidates[0].__name__}.load(path)`"
    if example is not None:
        lines.append(
            f"Choose one by passing its hint to `load`, as in `load(path, "
            f'hint="{example}")`. Each format can also read the file '
            f"itself, as in {own_load}."
        )
    else:
        lines.append(
            f"Choose one by reading the file with that format's own "
            f"`load`, as in {own_load}."
        )
    return "\n".join(lines)


def _normalize_hints(
    hints: tx.Iterable[str],
    hint: tx.Optional[tx.Union[str, tx.Iterable[str]]],
) -> tx.FrozenSet[str]:
    """Merge `hint` and `hints` into one lowercase set.

    Each argument may be a string or an iterable of strings.
    """
    result = [hints] if isinstance(hints, str) else list(hints)
    if hint is not None:
        result.extend([hint] if isinstance(hint, str) else hint)
    return frozenset(str(item).lower() for item in result)
