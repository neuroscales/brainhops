"""Structured file sources, written `path|hint|key:value`.

The module also registers the parsers that convert option values.
"""

__all__ = [
    "ImageSpec",
    "OperationSpec",
    "Parser",
    "SourceSpec",
    "TransformationSpec",
    "format_hints",
    "parse_bool",
    "parser_for",
    "register_parser",
]

import re

import typing_extensions as tx
from bagof.core.magic import get_from_registry
from bagof.magic import ConvertTo, Factory, Magic, replace

from brainhops._core.path import Path


class Parser(Magic, frozen=True):
    """Annotation metadata choosing the parser of a field.

    A field annotated as `Annotated[T, Parser(value)]` converts its option
    string with `value` instead of the parser registered for `T`. The value is
    either a parser or a target registered with [`register_parser`][], in
    which case the parser registered for that target is used.
    """

    value: tx.Any


class SourceSpec(Magic, frozen=True):
    """An immutable source path with format hints and named options.

    A source specification names a file to read. Its format hints restrict
    the formats that may read the file, and its options set fields of the
    format that reads it. Option values are strings or nested specifications.
    """

    path: ConvertTo[Path]
    hints: tx.Tuple[str, ...] = ()
    options: Factory[tx.Dict[str, tx.Union[str, "SourceSpec"]]]

    @classmethod
    def from_arg(cls, text: str) -> tx.Self:
        """Parse a source argument written as `path|hint|key:value`.

        The segments after the path are format hints, options, or operations
        that the class declares. Hints may also be given as `hint:a,b`. An
        option value written in brackets, as in `key:[path|hint]`, is a nested
        specification. Colons have a special meaning only in the segments after
        the path, and only the first colon of a segment separates the option
        name from its value, so a URI scheme in the path or in a value is kept
        intact. A literal pipe is written `%7C`.

        Raises
        ------
        ValueError
            If the path is missing, a segment is empty, or an option is unnamed
            or repeated.
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
        """Parse a segment as a declared operation, or return `None`."""
        return None


class ImageSpec(SourceSpec, frozen=True):
    """Source specification of an image."""


class OperationSpec(Magic, frozen=True):
    """A named operation applied to a loaded source."""

    name: str

    @classmethod
    def from_arg(cls, text: str) -> tx.Self:
        """Parse the segment of this operation.

        Raises
        ------
        ValueError
            If the segment gives the operation a value.
        """
        if ":" in text:
            name = text.split(":", 1)[0]
            raise ValueError(f"Operation {name!r} takes no value.")
        return cls(name=text.lower())

    def apply(self, value: tx.Any) -> tx.Any:
        """Apply the operation to a loaded value."""
        raise NotImplementedError


class TransformationSpec(SourceSpec, frozen=True):
    """Source specification of a transformation, with optional operations.

    The hint `svf` is shorthand for `displacements|log:true`, which reads a
    displacement field as the stationary velocity field of the transformation,
    as in `warp.nii.gz|svf|steps:6`. Operations such as `inv` are applied to
    the loaded transformation in the order in which they are written.
    """

    operations: tx.Tuple[OperationSpec, ...] = ()
    _OPERATIONS: tx.ClassVar[tx.Dict[str, tx.Type[OperationSpec]]] = {}

    @classmethod
    def from_arg(cls, text: str) -> tx.Self:
        """Parse a source argument, expanding the `svf` alias.

        The syntax is that of [`SourceSpec.from_arg`][].
        """
        return _expand_svf(super().from_arg(text))

    @classmethod
    def register_operation(
        cls, name: str
    ) -> tx.Callable[[tx.Type[OperationSpec]], tx.Type[OperationSpec]]:
        """Register an operation under `name` for this class and subclasses."""

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
        """Apply the operations in the order in which they were written."""
        for operation in self.operations:
            value = operation.apply(value)
        return value


@TransformationSpec.register_operation("inv")
class InvertOperation(OperationSpec, frozen=True):
    """Invert a loaded transformation, written `inv` in a specification."""

    def apply(self, value: tx.Any) -> tx.Any:
        return value.inverse()


def _expand_svf(spec: TransformationSpec) -> TransformationSpec:
    # The svf hint already implies log:true, so it cannot be combined with an
    # explicit log option.
    if "svf" not in spec.hints:
        return spec
    if "log" in spec.options:
        raise ValueError(
            "The hint svf already means log:true, so it cannot be combined "
            "with a log: option."
        )
    hints = ("displacements" if hint == "svf" else hint for hint in spec.hints)
    return replace(
        spec,
        hints=tuple(dict.fromkeys(hints)),
        options={**spec.options, "log": "true"},
    )


def _parse_hints(value: str) -> tx.List[str]:
    hints = [hint.strip().lower() for hint in value.split(",")]
    if not all(hints):
        raise ValueError(f"Invalid empty format hint in {value!r}.")
    return hints


def _decode_pipe(value: str) -> str:
    """Decode `%7C`, the only escape of the grammar, into a pipe."""
    return re.sub("%7c", "|", value, flags=re.IGNORECASE)


def _split_top_level(text: str) -> tx.List[str]:
    """Split on the pipes outside nested specifications.

    A bracket opens a nested specification only when it directly follows an
    option name `key:`, other than `hint:`, in a segment after the path. The
    matching bracket must then end the option value. Every other bracket is
    part of a path.
    """
    parts: tx.List[str] = []
    start = 0
    depth = 0
    segment_starts = [0]
    segment_numbers = [0]
    literal_brackets = [0]
    for index, char in enumerate(text):
        if char == "[":
            if _starts_nested_source(
                text,
                segment_starts[depth],
                index,
                segment_numbers[depth],
            ):
                depth += 1
                segment_starts.append(index + 1)
                segment_numbers.append(0)
                literal_brackets.append(0)
            else:
                literal_brackets[depth] += 1
        elif char == "]":
            if literal_brackets[depth]:
                literal_brackets[depth] -= 1
            elif depth and _ends_nested_source(text, index):
                depth -= 1
                segment_starts.pop()
                segment_numbers.pop()
                literal_brackets.pop()
        elif char == "|":
            if depth == 0:
                parts.append(text[start:index])
                start = index + 1
            segment_starts[depth] = index + 1
            segment_numbers[depth] += 1
            literal_brackets[depth] = 0
    if depth:
        raise ValueError("Unclosed bracket in source specification.")
    parts.append(text[start:])
    return parts


def _starts_nested_source(
    text: str, segment_start: int, bracket: int, segment_number: int
) -> bool:
    """Return whether `text[bracket]` opens a nested specification."""
    if segment_number == 0:
        return False
    tag = text[segment_start:bracket]
    if tag.count(":") != 1 or not tag.endswith(":"):
        return False
    key = tag[:-1].strip()
    return bool(key) and key != "hint"


def _ends_nested_source(text: str, bracket: int) -> bool:
    """Return whether `text[bracket]` closes a nested specification."""
    return bracket + 1 == len(text) or text[bracket + 1] in "|]"


_PARSERS: tx.Dict[tx.Hashable, tx.Any] = {}


def register_parser(
    target: tx.Hashable,
) -> tx.Callable[[tx.Any], tx.Any]:
    """Register a decorated parser as the default for `target`."""

    def decorator(parser: tx.Any) -> tx.Any:
        _PARSERS[target] = parser
        return parser

    return decorator


_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})


@register_parser(bool)
def parse_bool(spec: SourceSpec) -> bool:
    """Parse a boolean option such as `log:true`.

    A dedicated parser is needed because every non-empty string is truthy, so
    `log:false` would otherwise read as true. The values `true`, `yes`, `on`
    and `1` and their opposites are accepted, in any letter case.

    Raises
    ------
    ValueError
        If the value is anything else, or if hints or options are given.
    """
    text = str(spec.path).strip().lower()
    if spec.hints or spec.options or (text not in _TRUE | _FALSE):
        raise ValueError(
            f"Expected a boolean option (true or false), not {str(spec)!r}."
        )
    return text in _TRUE


def parser_for(annotation: tx.Any) -> tx.Optional[tx.Any]:
    """Return the explicit or registered parser for an annotation, or `None`.

    For a union, the parser of its members is used when the members that have
    a parser all share the same one.
    """
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
        parsers = {
            parser
            for member in members
            if (parser := parser_for(member)) is not None
        }
        if len(parsers) == 1:
            return parsers.pop()
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
    """Return the hint paths of `cls` along each inheritance branch.

    A hint path lists the hints declared along an inheritance branch, from the
    most general to the most specific. [`format_hints`][] joins each path into
    a dotted hint such as `itk.displacements`. Each hint that `cls` declares
    also forms a path on its own.
    """
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
    """Return the plain and dotted hints of a format.

    For example, a format may answer to `itk` and `itk.displacements`.
    """
    paths: tx.Set[tx.Tuple[str, ...]] = set()
    for base in cls.__mro__:
        paths.update(_hint_paths(base))
    return frozenset(".".join(path) for path in paths)
