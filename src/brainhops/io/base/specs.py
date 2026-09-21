"""Structured file-source specifications and field parser registration."""

__all__ = [
    "Parser",
    "SourceSpec",
    "format_hints",
    "parser_for",
    "register_parser",
]

from dataclasses import dataclass
from urllib.parse import unquote

import typing_extensions as tx


@dataclass(frozen=True)
class Parser:
    """``Annotated`` metadata selecting the parser for a field.

    The value may be a registered target type or a parser callable/class.
    An explicit ``Parser`` always takes precedence over registry lookup.
    """

    value: tx.Any


@dataclass(frozen=True)
class SourceSpec:
    """A source plus format hints, named options, and operations.

    Options remain ordered pairs until they are bound to a concrete format.
    This makes duplicate detection deterministic and avoids prematurely
    interpreting a value whose field type is not known yet.
    """

    value: str
    hints: tx.Tuple[str, ...] = ()
    options: tx.Tuple[tx.Tuple[str, tx.Union[str, "SourceSpec"]], ...] = ()
    operations: tx.Tuple[str, ...] = ()

    @classmethod
    def parse(
        cls,
        text: str,
        operations: tx.Iterable[str] = (),
    ) -> "SourceSpec":
        """Parse ``path|hint|key:value|op`` syntax.

        Square brackets delimit a nested source specification. Colons are
        meaningful only in modifier segments and are split once, so URI
        schemes in source values remain untouched. A literal pipe is written
        as ``%7C`` (case-insensitive).
        """
        operation_names = {str(op).lower() for op in operations}
        segments = _split_top_level(text)
        if not segments or not segments[0]:
            raise ValueError("A source specification must start with a value.")

        value = unquote(segments[0])
        hints: tx.List[str] = []
        options_list: tx.List[tx.Tuple[str, tx.Union[str, SourceSpec]]] = []
        parsed_operations: tx.List[str] = []
        seen_options: tx.Set[str] = set()

        for segment in segments[1:]:
            if not segment:
                raise ValueError("Empty pipe element in source specification.")
            lowered = segment.lower()
            if lowered in operation_names:
                parsed_operations.append(lowered)
                continue
            if ":" not in segment:
                hints.extend(_parse_hints(segment))
                continue

            key, raw_value = segment.split(":", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Missing option name in {segment!r}.")
            if key == "hint":
                hints.extend(_parse_hints(raw_value))
                continue
            if key in seen_options:
                raise ValueError(f"Duplicate source option {key!r}.")
            seen_options.add(key)
            if raw_value.startswith("[") and raw_value.endswith("]"):
                option_value: tx.Union[str, SourceSpec] = cls.parse(
                    raw_value[1:-1], operations=operations
                )
            else:
                option_value = unquote(raw_value)
            options_list.append((key, option_value))

        # A hint is an unordered allowlist. Repetition adds no information.
        normalized_hints = tuple(dict.fromkeys(h.lower() for h in hints))
        return cls(
            value=value,
            hints=normalized_hints,
            options=tuple(options_list),
            operations=tuple(parsed_operations),
        )


def _parse_hints(value: str) -> tx.List[str]:
    hints = [hint.strip().lower() for hint in value.split(",")]
    if not all(hints):
        raise ValueError(f"Invalid empty format hint in {value!r}.")
    return hints


def _split_top_level(text: str) -> tx.List[str]:
    """Split on pipes outside square brackets, validating nesting."""
    parts: tx.List[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("Unmatched closing bracket in source spec.")
        elif char == "|" and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    if depth:
        raise ValueError("Unclosed bracket in source specification.")
    parts.append(text[start:])
    return parts


_PARSERS: tx.Dict[type, tx.Any] = {}


def register_parser(target: type, parser: tx.Any) -> tx.Any:
    """Register the default parser for fields whose type is ``target``."""
    _PARSERS[target] = parser
    return parser


def parser_for(annotation: tx.Any) -> tx.Optional[tx.Any]:
    """Resolve an explicit or registered parser for a field annotation.

    ``Optional`` is transparent. A union resolves automatically only when
    all parseable members agree on one parser; otherwise an explicit
    ``Annotated[..., Parser(...)]`` is required.
    """
    annotation, explicit = _unwrap_annotated(annotation)
    if explicit is not None:
        value = explicit.value
        return _PARSERS.get(value, value) if isinstance(value, type) else value

    origin = tx.get_origin(annotation)
    if origin is tx.Union or str(origin) == "<class 'types.UnionType'>":
        members = [
            member
            for member in tx.get_args(annotation)
            if member not in (None, type(None))
        ]
        resolved = [parser_for(member) for member in members]
        parsers = [parser for parser in resolved if parser is not None]
        if not parsers:
            return None
        first = parsers[0]
        if len(parsers) == len(members) and all(
            parser is first for parser in parsers
        ):
            return first
        return None

    if not isinstance(annotation, type):
        return None
    candidates = [
        registered
        for registered in _PARSERS
        if issubclass(annotation, registered)
    ]
    if not candidates:
        return None
    nearest = min(candidates, key=annotation.__mro__.index)
    return _PARSERS[nearest]


def _unwrap_annotated(
    annotation: tx.Any,
) -> tx.Tuple[tx.Any, tx.Optional[Parser]]:
    explicit = None
    while tx.get_origin(annotation) is tx.Annotated:
        annotation, *metadata = tx.get_args(annotation)
        selected = [item for item in metadata if isinstance(item, Parser)]
        if len(selected) > 1:
            raise TypeError("A field annotation may specify only one Parser.")
        if selected:
            explicit = selected[0]
    return annotation, explicit


def format_hints(cls: type) -> tx.FrozenSet[str]:
    """Collect a format's hints additively through its class hierarchy."""
    hints: tx.Set[str] = set()
    for base in reversed(cls.__mro__):
        declared = base.__dict__.get("FORMAT_HINTS", ())
        hints.update(str(hint).lower() for hint in declared)
    return frozenset(hints)
