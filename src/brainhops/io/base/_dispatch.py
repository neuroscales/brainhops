"""
Format dispatch: choosing which registered parser should read an input.

Kept apart from `_base` because none of it depends on the class
hierarchy -- it works on a registry (any set of parser classes) and an
input, and is the part worth reading on its own when reasoning about why
a given file was read by a given parser.
"""

__all__ = ["Source", "parse", "sniff"]

# stdlib
from collections.abc import Iterable
from io import BytesIO, StringIO
from os import PathLike, fspath

# dependencies
import typing_extensions as tx
from bagof.magic import fields

# internals
from brainhops._core import path
from brainhops.io.base.parsers import (
    AmbiguousFormatError,
    ParserContentError,
    SnifferContentError,
)
from brainhops.io.base.specs import SourceSpec, format_hints, parser_for

_T = tx.TypeVar("_T")


class Source:
    """
    Hands out a fresh, rewound view of a parse input.

    Dispatch is speculative: we may sniff a stream and then parse it, or
    try several parsers in turn. Each of those consumes the input, so
    every attempt must start from the same place.

    - Paths and byte/text content are immutable and handed back as-is.
    - Seekable streams are rewound to their initial position.
    - Non-seekable streams (stdin, sockets, pipes) are read once into
      memory and re-wrapped in a fresh buffer for every attempt.
    - One-shot iterables of lines are materialized once into a list.
    """

    def __init__(self, other: tx.Any) -> None:
        self.other = other
        self.pos = None
        self.buffer = None
        self.factory = None

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
            pass  # immutable, and `str`/`bytes` are Iterable: check first
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
            return True  # old-style file objects have no `seekable()`
        except Exception:
            return False

    def get(self) -> tx.Any:
        """A fresh view of the input, positioned at the start."""
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

    @property
    def name(self) -> tx.Optional[str]:
        """The file name, if the input is a named file."""
        other = self.other
        if isinstance(other, PathLike):
            other = fspath(other)
        if not isinstance(other, str):
            other = getattr(other, "name", None)
            if not isinstance(other, str):
                return None
        # Trailing slashes matter for directory-based formats (.zarr)
        return path.Path(other.rstrip("/")).name

    def __repr__(self) -> str:
        """Describe the source by its file name, or as plain content when
        it has none."""
        name = self.name
        return f"file {name!r}" if name else "input content"


def _match_name(name: str, cls: type) -> tx.Optional[tx.Tuple[int, int]]:
    """
    How much of `name` this format accounts for, or `None` if it cannot.

    Returns `(extension, prefix)`, the lengths of the longest declared
    extension and prefix that matched. Longer is more specific: a parser
    declaring `".nii.gz"` beats one declaring `".gz"`, and one requiring
    `"iy_"` beats one requiring `"y_"`.

    A format that declares prefixes and matches none of them is out of
    the running entirely -- the constraint is a requirement, not a hint.
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
    """
    How many dispatcher levels a format sits under.

    A format registered under both `FileBasedImage` and the root
    `FileBasedObject` has been classified more finely than one sitting
    under the root alone, and is preferred when nothing else separates
    them.

    This is the one defensible reading of "MRO depth": it counts
    *registry* levels, which carry meaning, rather than classes, which
    do not -- a deep hierarchy inside one format family says nothing
    about how well that format matches the file at hand.
    """
    return sum(1 for base in cls.__mro__ if "_REGISTRY" in base.__dict__)


def _drop_base_classes(
    candidates: tx.List[tx.Tuple[type, tx.Any]],
) -> tx.List[tx.Tuple[type, tx.Any]]:
    """
    Drop candidates that are proper base classes of another candidate.

    A subclass is strictly more specific than its base, so when both
    claim the same content the subclass wins. This is the rule
    `functools.singledispatch` uses.
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
    """
    Sort key ordering candidates from most to least specific.

    Every component is a statement about the *file*, or about what the
    parser claims to handle -- never about accidents of import order.
    In decreasing precedence:

    1. Sniffer confidence.
    2. How much of the filename the extension accounted for.
    3. How much of it the required prefix accounted for.
    4. How finely the format is classified (`_registry_depth`).
    5. How narrow a surface it declares: a parser claiming `.lta` alone
       is more specific than one claiming five extensions, and both are
       more specific than one claiming none.
    6. Explicit `PRIORITY`.

    Candidates that tie on all of these are genuinely indistinguishable,
    and dispatch refuses to guess between them.
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
    """
    Group candidates into tiers of equal specificity, best tier first.

    A tier with more than one member holds parsers that nothing can
    separate; `_parse` treats that as an ambiguity rather than picking
    one at random.
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
    """
    Rank the formats that could read `source`, best tier first.

    Shared by `parse` and `sniff`, so that "which format would read this"
    and "which format did read this" can never disagree.

    Extension and confidence are weighed together rather than in separate
    passes. An extension pass that short-circuited would put every reader
    of a shared container -- every `.nii` reader, say -- into one
    undifferentiated tier and call it an ambiguity, throwing away the
    very scores that can tell them apart.
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
        # A matching extension keeps a parser in the running even when
        # its sniffer declines: sniffers are optional, and a name is
        # evidence too. It still sorts below anything that scored.
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
    """
    Find the parser in `registry` that best matches `source`, and use it.

    Every registered format is scored by its sniffer and by how well
    the filename matches, and the most specific one wins. If none of
    them recognizes the content, and only if asked, every format is
    tried in turn.

    !!! note "Why brute force is opt-in"
        Running arbitrary readers over arbitrary bytes can succeed on a
        partial parse and quietly return a wrong-format object, so it
        must be asked for rather than happening behind the caller's back.

    Parameters
    ----------
    source : Source
        The input, wrapped so that every attempt sees it from the same
        position.
    registry : set[type]
        The formats to choose between.
    fn_parse : str
        Name of the parsing method to call, e.g. `"from_file"`.
    fn_sniff : str
        Name of the matching sniffing method, e.g. `"sniff_file"`.
    brute : bool
        Try every parser in turn if none recognized the input.
    **kwargs
        Parser-specific options.

    Returns
    -------
    obj
        The parsed object.

    Raises
    ------
    AmbiguousFormatError
        If several equally specific parsers can all read the content.
    ParserContentError
        If no parser could read the content.
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
            # Nothing separates these parsers. Try them all: if only one
            # can actually read the content, the ambiguity was merely
            # apparent and there is nothing to complain about.
            winners = []
            for subclass in tier:
                ok, result = attempt(subclass)
                if ok:
                    winners.append((subclass, result))
            if len(winners) == 1:
                return True, winners[0][1]
            if len(winners) > 1:
                names = sorted(cls.__name__ for cls, _ in winners)
                raise AmbiguousFormatError(
                    f"Cannot choose a parser for {source}: "
                    f"{', '.join(names)} all read it, with equal "
                    f"confidence and equal specificity, but build "
                    f"different objects. Give one of them a sniffer that "
                    f"tells them apart, or an explicit PRIORITY."
                )
        return False, None

    tiers = _candidates(
        source, registry, fn_sniff, errors, allowed=allowed, **kwargs
    )
    # Hints are an explicit allowlist. If nothing in that allowlist sniffs
    # the source (an unknown extension is a common reason), try the allowed
    # formats instead of silently widening back to the full registry.
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
    """Resolved field annotations, including custom ``Annotated`` data."""
    try:
        annotations = tx.get_type_hints(cls, include_extras=True)
    except Exception:
        annotations = {}
        for base in reversed(cls.__mro__):
            annotations.update(base.__dict__.get("__annotations__", {}))
    return annotations


def _option_fields(cls: type) -> tx.Dict[str, tx.Tuple[str, tx.Any]]:
    """Source option names mapped to concrete Magic fields and types."""
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
    """
    Build an informative error out of everything that went wrong.

    Swallowing every exception and reporting a bare "cannot parse" hides
    real bugs inside the *correct* reader, so we report what each parser
    actually complained about, and chain the last one.
    """
    if not registry:
        return ParserContentError(
            f"Cannot parse {source}: no parser is registered. Formats "
            f"register themselves on import -- is the format module "
            f"imported?"
        )
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
    # `raise ... from ...` is not available to a function that *returns*
    # the exception, so chain it by hand.
    failure.__cause__ = errors[-1][2]
    return failure


def sniff(
    content: tx.Any,
    registry: tx.Set[type],
    fn_sniff: str,
    error: tx.Union[bool, tx.Type[Exception]] = False,
    what: str = "input content",
    hints: tx.Iterable[str] = (),
    hint: tx.Optional[tx.Union[str, tx.Iterable[str]]] = None,
    **kwargs,
) -> tx.Optional[type]:
    """
    Identify which registered format would read this content.

    A dispatcher answers *which* format, not *how confident*: the score
    belongs to a format judging itself, and once several formats have
    been compared the interesting answer is the winner. Ask the returned
    class to score itself if the number is what you wanted.

    The ranking is the one `parse` uses, so the answer is what `load`
    would actually reach for.

    !!! note "`None` is not the same as a failed load"
        `parse` can resolve an ambiguous top tier by *trying* each
        candidate -- if only one parses, there was no real ambiguity.
        `sniff` does not parse, so it cannot do that, and reports no
        single answer instead. A `None` here does not guarantee that
        `load` will fail.

    Parameters
    ----------
    content : Any
        The input to identify.
    registry : set[type]
        The formats to choose between.
    fn_sniff : str
        Name of the sniffing method to call, e.g. `"sniff_file"`.
    error : bool | type[Exception]
        If not False, raise instead of returning `None`.
    what : str
        How to describe the input in an error message.
    **kwargs
        Parser-specific options.

    Returns
    -------
    format : type | None
        The best-matching format, or `None` if no single one stands out.
    """
    requested_hints = _normalize_hints(hints, hint)
    allowed = {
        subclass
        for subclass in registry
        if not requested_hints or format_hints(subclass) & requested_hints
    }
    tiers = _candidates(
        Source(content), registry, fn_sniff, allowed=allowed, **kwargs
    )

    if tiers and len(tiers[0]) == 1:
        return tiers[0][0]

    if error:
        if tiers:
            names = ", ".join(sorted(cls.__name__ for cls in tiers[0]))
            if error is True:
                error = AmbiguousFormatError
            raise error(
                f"Cannot identify {what}: {names} match it equally well."
            )
        if error is True:
            error = SnifferContentError
        raise error(f"Nothing to sniff in {what}")

    return None


def _normalize_hints(
    hints: tx.Iterable[str],
    hint: tx.Optional[tx.Union[str, tx.Iterable[str]]],
) -> tx.FrozenSet[str]:
    """Normalize the singular convenience spelling and union spelling."""
    result = [hints] if isinstance(hints, str) else list(hints)
    if hint is not None:
        result.extend([hint] if isinstance(hint, str) else hint)
    return frozenset(str(item).lower() for item in result)
