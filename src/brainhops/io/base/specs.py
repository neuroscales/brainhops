"""Structured file-source specifications and field parser registration."""

__all__ = [
    "OperationSpec",
    "Parser",
    "SourceSpec",
    "TransformationSpec",
    "format_hints",
    "parser_for",
    "register_parser",
]

import re

import typing_extensions as tx
from bagof.core.magic import get_from_registry
from bagof.magic import ConvertTo, Factory, Magic

from brainhops._core.path import Path


class Parser(Magic, frozen=True):
    """``Annotated`` metadata selecting the parser for a field.

    The value may be a registered target type or a parser callable/class.
    An explicit ``Parser`` always takes precedence over registry lookup.
    """

    value: tx.Any


class SourceSpec(Magic, frozen=True):
    """A source path plus format hints and typed named options."""

    path: ConvertTo[Path]
    hints: tx.Tuple[str, ...] = ()
    options: Factory[tx.Dict[str, tx.Union[str, "SourceSpec"]]]

    @classmethod
    def from_arg(cls, text: str) -> tx.Self:
        """Parse ``path|hint|key:value`` CLI syntax.

        Brackets delimit a nested source specification. Colons are syntax
        only in modifier segments, so URI schemes in source values remain
        intact. A literal pipe must be percent-encoded as ``%7C`` because
        ``|`` is always structural.
        """
        segments = _split_top_level(text)
        if not segments or not segments[0]:
            raise ValueError("A source specification must start with a path.")

        hints: tx.List[str] = []
        options: tx.Dict[str, tx.Union[str, SourceSpec]] = {}
        operation_specs: tx.List[OperationSpec] = []

        for segment in segments[1:]:
            if not segment:
                raise ValueError("Empty pipe element in source specification.")
            operation = cls._parse_operation(segment)
            if operation is not None:
                operation_specs.append(operation)
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
            if key in options:
                raise ValueError(f"Duplicate source option {key!r}.")
            if raw_value.startswith("[") and raw_value.endswith("]"):
                options[key] = SourceSpec.from_arg(raw_value[1:-1])
            else:
                options[key] = _decode_pipe(raw_value)

        kwargs: tx.Dict[str, tx.Any] = {
            "path": _decode_pipe(segments[0]),
            "hints": tuple(dict.fromkeys(h.lower() for h in hints)),
            "options": options,
        }
        if operation_specs:
            kwargs["operations"] = tuple(operation_specs)
        return cls(**kwargs)

    @classmethod
    def _parse_operation(cls, segment: str) -> tx.Optional["OperationSpec"]:
        """Parse a class-owned operation, if this spec type declares it."""
        return None


class OperationSpec(Magic, frozen=True):
    """A validated operation attached to a structured source."""

    name: str

    @classmethod
    def from_arg(cls, text: str) -> tx.Self:
        """Parse this operation's segment from a source argument."""
        if ":" in text:
            name = text.split(":", 1)[0]
            raise ValueError(f"Operation {name!r} takes no value.")
        return cls(name=text.lower())

    def apply(self, value: tx.Any) -> tx.Any:
        """Apply this operation to a loaded source value."""
        raise NotImplementedError


class TransformationSpec(SourceSpec, frozen=True):
    """A transformation source with validated transformation operations."""

    operations: tx.Tuple[OperationSpec, ...] = ()
    _OPERATIONS: tx.ClassVar[tx.Dict[str, tx.Type[OperationSpec]]] = {}

    @classmethod
    def register_operation(
        cls, name: str
    ) -> tx.Callable[[tx.Type[OperationSpec]], tx.Type[OperationSpec]]:
        """Register an operation understood by this source-spec class."""

        def decorator(
            operation: tx.Type[OperationSpec],
        ) -> tx.Type[OperationSpec]:
            cls._OPERATIONS[name.lower()] = operation
            return operation

        return decorator

    @classmethod
    def _parse_operation(cls, segment: str) -> tx.Optional[OperationSpec]:
        name = segment.split(":", 1)[0].lower()
        operation = cls._OPERATIONS.get(name)
        return operation.from_arg(segment) if operation else None

    def apply_operations(self, value: tx.Any) -> tx.Any:
        """Apply registered operations in their written order."""
        for operation in self.operations:
            value = operation.apply(value)
        return value


@TransformationSpec.register_operation("inv")
class InvertOperation(OperationSpec, frozen=True):
    """Invert a loaded transformation."""

    def apply(self, value: tx.Any) -> tx.Any:
        return value.inverse()


def _parse_hints(value: str) -> tx.List[str]:
    hints = [hint.strip().lower() for hint in value.split(",")]
    if not all(hints):
        raise ValueError(f"Invalid empty format hint in {value!r}.")
    return hints


def _decode_pipe(value: str) -> str:
    """Decode the one percent escape reserved by the source grammar."""
    return re.sub("%7c", "|", value, flags=re.IGNORECASE)


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


_PARSERS: tx.Dict[tx.Hashable, tx.Any] = {}


def register_parser(
    target: tx.Hashable,
) -> tx.Callable[[tx.Any], tx.Any]:
    """Register a default parser for ``target`` as a class decorator."""

    def decorator(parser: tx.Any) -> tx.Any:
        _PARSERS[target] = parser
        return parser

    return decorator


def parser_for(annotation: tx.Any) -> tx.Optional[tx.Any]:
    """Resolve explicit parser metadata or the best registered parser."""
    annotation, explicit = _unwrap_annotated(annotation)
    if explicit is not None:
        value = explicit.value
        return get_from_registry(value, _PARSERS) or value

    parser = get_from_registry(annotation, _PARSERS)
    if parser is not None:
        return parser

    origin = tx.get_origin(annotation)
    if origin is tx.Union or str(origin) == "<class 'types.UnionType'>":
        members = [
            member
            for member in tx.get_args(annotation)
            if member not in (None, type(None))
        ]
        parsers = [parser_for(member) for member in members]
        if parsers and all(parser is not None for parser in parsers):
            first = parsers[0]
            if all(parser is first for parser in parsers):
                return first
    return None


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


def _hint_paths(cls: type) -> tx.Set[tx.Tuple[str, ...]]:
    """Build namespace paths along individual inheritance branches."""
    declared = tuple(
        str(hint).lower() for hint in cls.__dict__.get("HINTS", ())
    )
    parent_paths: tx.Set[tx.Tuple[str, ...]] = set()
    for base in cls.__bases__:
        parent_paths.update(_hint_paths(base))
    if not declared:
        return parent_paths

    paths = {(hint,) for hint in declared}
    for hint in declared:
        for parent in parent_paths:
            paths.add(parent if parent[-1] == hint else (*parent, hint))
    return paths


def format_hints(cls: type) -> tx.FrozenSet[str]:
    """Collect leaf and qualified hints along semantic inheritance branches."""
    paths: tx.Set[tx.Tuple[str, ...]] = set()
    for base in cls.__mro__:
        paths.update(_hint_paths(base))
    return frozenset(".".join(path) for path in paths)
