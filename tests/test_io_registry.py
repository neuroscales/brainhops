"""Unit tests for the file-format registry and its dispatch rules."""

import pytest
import typing_extensions as tx

from brainhops.io.base._base import (
    FileBasedObject,
    TextFileBasedObject,
    format_registry,
    register_format,
)
from brainhops.io.base.parsers import (
    AmbiguousFormatError,
    Confidence,
    ParserContentError,
)


@pytest.fixture
def root() -> type:
    """
    An isolated dispatcher, so tests never touch the real registries.

    Formats registered in a test would otherwise stay in
    `FileBasedObject._REGISTRY` for the rest of the session and leak into
    every later test.
    """

    @format_registry
    class Root(TextFileBasedObject):
        pass

    return Root


def _format(root: type, name: str, **attrs: tx.Any) -> type:
    """Build and register a tiny text format under `root`."""
    marker = attrs.pop("marker", name)

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return Confidence.CERTAIN if line.startswith(marker) else Confidence.NO

    @classmethod
    def from_line(cls, line, **kwargs) -> tx.Any:  # noqa: ANN001
        if not line.startswith(marker):
            raise ParserContentError(f"not a {name}")
        return {"format": name, "line": line}

    namespace = {"sniff_line": sniff_line, "from_line": from_line}
    namespace.update(attrs)
    return register_format(type(name, (root,), namespace))


# ----------------------------------------------------------------------
#   REGISTRATION
# ----------------------------------------------------------------------


def test_format_registry_gives_a_class_its_own_registry(root: type) -> None:
    assert "_REGISTRY" in root.__dict__
    assert root._REGISTRY == set()


def test_register_format_fills_every_ancestor_registry(root: type) -> None:
    fmt = _format(root, "A", EXTENSIONS=(".a",))
    assert fmt in root._REGISTRY
    assert fmt in FileBasedObject._REGISTRY
    assert fmt in TextFileBasedObject._REGISTRY
    # keep the real registries clean for the rest of the session
    for base in (FileBasedObject, TextFileBasedObject):
        base._REGISTRY.discard(fmt)


def test_dispatchers_are_not_registered_as_formats(root: type) -> None:
    # `root` dispatches; it must not appear in the registry it feeds into
    assert root not in FileBasedObject._REGISTRY


def test_registering_a_dispatcher_is_an_error(root: type) -> None:
    # Doing so would try every format below it twice, and make `from_`
    # recurse into itself.
    with pytest.raises(TypeError, match="owns a registry"):
        register_format(root)


def test_registering_twice_is_a_no_op(root: type) -> None:
    fmt = _format(root, "A", EXTENSIONS=(".a",))
    before = len(root._REGISTRY)
    register_format(fmt)
    assert len(root._REGISTRY) == before
    FileBasedObject._REGISTRY.discard(fmt)
    TextFileBasedObject._REGISTRY.discard(fmt)


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


def test_sniff_on_a_dispatcher_names_the_format(root: type) -> None:
    """
    A dispatcher has already compared the scores, so it answers the
    question they were compared to settle: *which* format is it.
    """
    fmt = _format(root, "A", EXTENSIONS=(".a",), marker="AAA")
    assert root.sniff_line("AAA hello") is fmt
    assert root.sniff_line("zzz") is None


def test_sniff_on_a_concrete_format_still_scores(root: type) -> None:
    """The number is still there; ask the format itself for it."""
    fmt = _format(root, "A", EXTENSIONS=(".a",), marker="AAA")
    assert fmt.sniff_line("AAA hello") == Confidence.CERTAIN
    assert fmt.sniff_line("zzz") == Confidence.NO


def test_sniff_agrees_with_what_load_actually_picks(root: type) -> None:
    """
    The two share their ranking, so an identification that disagreed
    with the parser eventually used would be a bug.
    """
    _format(root, "A", EXTENSIONS=(".a",), marker="AAA")
    _format(root, "B", EXTENSIONS=(".b",), marker="BBB")
    identified = root.sniff_line("BBB x")
    assert identified.__name__ == "B"
    assert root.from_line("BBB x")["format"] == identified.__name__


def test_dispatch_picks_the_format_that_sniffs_highest(root: type) -> None:
    _format(root, "A", EXTENSIONS=(".a",), marker="AAA")
    _format(root, "B", EXTENSIONS=(".b",), marker="BBB")
    assert root.from_line("BBB x")["format"] == "B"
    assert root.from_line("AAA x")["format"] == "A"


def test_longest_matching_extension_wins(root: type, tmp_path) -> None:  # noqa: ANN001
    """`.nii.gz` is a more specific claim than `.gz`."""

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return Confidence.MAYBE

    broad = register_format(
        type(
            "Broad",
            (root,),
            {
                "EXTENSIONS": (".gz",),
                "sniff_line": sniff_line,
                "from_line": classmethod(lambda cls, line, **kw: "broad"),
            },
        )
    )
    narrow = register_format(
        type(
            "Narrow",
            (root,),
            {
                "EXTENSIONS": (".nii.gz",),
                "sniff_line": sniff_line,
                "from_line": classmethod(lambda cls, line, **kw: "narrow"),
            },
        )
    )
    scan = tmp_path / "scan.nii.gz"
    scan.write_text("content\n")
    assert root.from_file(scan) == "narrow"
    for fmt in (broad, narrow):
        FileBasedObject._REGISTRY.discard(fmt)
        TextFileBasedObject._REGISTRY.discard(fmt)


def test_a_prefix_constraint_beats_an_unconstrained_format(
    root: type,
    tmp_path,  # noqa: ANN001
) -> None:
    """SPM's `y_` convention is the only thing separating two formats."""

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return Confidence.CERTAIN

    generic = register_format(
        type(
            "Generic",
            (root,),
            {
                "EXTENSIONS": (".nii",),
                "sniff_line": sniff_line,
                "from_line": classmethod(lambda cls, line, **kw: "generic"),
            },
        )
    )
    spm = register_format(
        type(
            "Spm",
            (root,),
            {
                "EXTENSIONS": (".nii",),
                "PREFIXES": ("y_",),
                "sniff_line": sniff_line,
                "from_line": classmethod(lambda cls, line, **kw: "spm"),
            },
        )
    )
    for name, expected in (("y_sub.nii", "spm"), ("sub.nii", "generic")):
        target = tmp_path / name
        target.write_text("content\n")
        assert root.from_file(target) == expected
    for fmt in (generic, spm):
        FileBasedObject._REGISTRY.discard(fmt)
        TextFileBasedObject._REGISTRY.discard(fmt)


def test_a_subclass_outranks_its_own_base(root: type) -> None:
    """The `singledispatch` rule: the more derived format is preferred."""
    base = _format(root, "Base", EXTENSIONS=(".x",), marker="X")
    derived = register_format(
        type(
            "Derived",
            (base,),
            {
                "from_line": classmethod(lambda cls, line, **kw: "derived"),
            },
        )
    )
    assert root.from_line("X hello") == "derived"
    for fmt in (base, derived):
        FileBasedObject._REGISTRY.discard(fmt)
        TextFileBasedObject._REGISTRY.discard(fmt)


def test_indistinguishable_formats_raise_rather_than_guess(
    root: type,
) -> None:
    """Two formats with nothing to tell them apart is a bug, not a coin
    flip: returning one at random would silently give the wrong type."""
    _format(root, "Twin1", EXTENSIONS=(".t",), marker="T")
    _format(root, "Twin2", EXTENSIONS=(".t",), marker="T")
    with pytest.raises(AmbiguousFormatError, match=r"Twin1[^\n]*\n.*Twin2"):
        root.from_line("T hello")


def test_an_apparent_tie_that_only_one_format_can_read_is_not_an_error(
    root: type,
) -> None:
    """Equal scores are fine as long as only one actually parses."""

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return Confidence.MAYBE

    _format(
        root, "Real", EXTENSIONS=(".t",), marker="T", sniff_line=sniff_line
    )
    register_format(
        type(
            "Broken",
            (root,),
            {
                "EXTENSIONS": (".t",),
                "sniff_line": sniff_line,
                "from_line": classmethod(
                    lambda cls, line, **kw: (_ for _ in ()).throw(
                        ParserContentError("nope")
                    )
                ),
            },
        )
    )
    assert root.from_line("T hello")["format"] == "Real"


# ----------------------------------------------------------------------
#   FAILURE REPORTING
# ----------------------------------------------------------------------


def test_failure_reports_what_each_parser_complained_about(
    root: type,
) -> None:
    """A bug in the right reader must not surface as 'cannot parse'."""
    _format(root, "A", EXTENSIONS=(".a",), marker="AAA")
    with pytest.raises(ParserContentError) as excinfo:
        root.from_line("zzz")
    assert "A" in str(excinfo.value)


def test_an_empty_registry_says_so(root: type) -> None:
    with pytest.raises(ParserContentError, match="no parser is registered"):
        root.from_line("anything")


def test_brute_force_is_opt_in(root: type) -> None:
    """A reader that declines to sniff is only run when asked."""
    register_format(
        type(
            "Silent",
            (root,),
            {
                "sniff_line": classmethod(
                    lambda cls, line, error=False, **kw: Confidence.NO
                ),
                "from_line": classmethod(lambda cls, line, **kw: "silent"),
            },
        )
    )
    with pytest.raises(ParserContentError):
        root.from_line("hello")
    assert root.from_line("hello", brute=True) == "silent"


def test_a_longer_required_prefix_is_more_specific(
    root: type,
    tmp_path,  # noqa: ANN001
) -> None:
    """Specificity is the length of the prefix that actually matched."""

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return Confidence.CERTAIN

    short = register_format(
        type(
            "Short",
            (root,),
            {
                "EXTENSIONS": (".nii",),
                "PREFIXES": ("sub",),
                "sniff_line": sniff_line,
                "from_line": classmethod(lambda cls, line, **kw: "short"),
            },
        )
    )
    long_ = register_format(
        type(
            "Long",
            (root,),
            {
                "EXTENSIONS": (".nii",),
                "PREFIXES": ("sub_y_",),
                "sniff_line": sniff_line,
                "from_line": classmethod(lambda cls, line, **kw: "long"),
            },
        )
    )
    target = tmp_path / "sub_y_01.nii"
    target.write_text("content\n")
    assert root.from_file(target) == "long"
    for fmt in (short, long_):
        FileBasedObject._REGISTRY.discard(fmt)
        TextFileBasedObject._REGISTRY.discard(fmt)


def test_declaring_more_prefixes_does_not_buy_specificity(
    root: type,
    tmp_path,  # noqa: ANN001
) -> None:
    """
    Declaring more prefixes is a *broader* claim, not a narrower one. It
    used to win ties, which had it exactly backwards; now the two are
    indistinguishable and dispatch says so.
    """

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return Confidence.CERTAIN

    for name, prefixes in (
        ("Many", ("y_", "iy_", "wy_")),
        ("One", ("iy_",)),
    ):
        register_format(
            type(
                name,
                (root,),
                {
                    "EXTENSIONS": (".nii",),
                    "PREFIXES": prefixes,
                    "sniff_line": sniff_line,
                    "from_line": classmethod(lambda cls, line, **kw: "x"),
                },
            )
        )
    target = tmp_path / "iy_sub.nii"
    target.write_text("content\n")
    with pytest.raises(AmbiguousFormatError, match=r"Many[^\n]*\n.*One"):
        root.from_file(target)


# ----------------------------------------------------------------------
#   THE AMBIGUITY MESSAGE
# ----------------------------------------------------------------------


def _twins(root: type) -> tx.Tuple[type, type]:
    """Two formats nothing tells apart, each with a hint of its own."""
    first = _format(
        root,
        "Raw",
        EXTENSIONS=(".t",),
        marker="T",
        HINTS=("raw",),
        __doc__="A table of raw samples.\n\nMore detail.",
    )
    second = _format(
        root,
        "Cooked",
        EXTENSIONS=(".t",),
        marker="T",
        HINTS=("cooked",),
        __doc__="A table of calibrated samples.",
    )
    return first, second


def test_the_ambiguity_message_gives_the_hint_that_selects_each_format(
    root: type,
) -> None:
    _twins(root)
    with pytest.raises(AmbiguousFormatError) as info:
        root.from_line("T hello")
    message = str(info.value)
    assert 'Raw (A table of raw samples): hint="raw"' in message
    assert 'Cooked (A table of calibrated samples): hint="cooked"' in message
    assert "`load(path, hint=" in message


@pytest.mark.parametrize("hint", ["raw", "cooked"])
def test_the_hints_in_the_message_resolve_the_ambiguity(
    root: type, hint: str
) -> None:
    """Following the advice of the message does choose that format."""
    _twins(root)
    with pytest.raises(AmbiguousFormatError, match=f'hint="{hint}"'):
        root.from_line("T hello")
    loaded = root.from_line("T hello", hint=hint)
    assert loaded["format"] == hint.capitalize()


def test_the_ambiguity_message_is_written_for_users(root: type) -> None:
    """How to fix the formats is for their maintainers, not the user."""
    _twins(root)
    with pytest.raises(AmbiguousFormatError) as info:
        root.from_line("T hello")
    message = str(info.value)
    assert "PRIORITY" not in message
    assert "sniffer" not in message


def test_a_format_no_hint_selects_is_read_with_its_own_load(
    root: type,
) -> None:
    """
    When every hint a format answers to is shared with the other, the
    message points at the format's own `load` instead.
    """
    _format(root, "Plain", EXTENSIONS=(".t",), marker="T", HINTS=("t",))
    _format(
        root, "Fancy", EXTENSIONS=(".t",), marker="T", HINTS=("t", "fancy")
    )
    with pytest.raises(AmbiguousFormatError) as info:
        root.from_line("T hello")
    message = str(info.value)
    assert "Plain: `Plain.load(path)`" in message
    assert 'Fancy: hint="fancy"' in message


def test_a_hint_that_only_settles_the_tie_is_offered(root: type) -> None:
    """
    A hint shared with a format that did not tie still settles the tie,
    so it is offered.
    """
    _format(root, "Left", EXTENSIONS=(".t",), marker="T", HINTS=("side",))
    _format(root, "Right", EXTENSIONS=(".t",), marker="T", HINTS=("r",))
    _format(root, "Other", EXTENSIONS=(".o",), marker="O", HINTS=("side",))
    with pytest.raises(AmbiguousFormatError) as info:
        root.from_line("T hello")
    assert 'Left: hint="side"' in str(info.value)
    assert root.from_line("T hello", hint="side")["format"] == "Left"


def test_sniff_reports_an_ambiguity_with_the_same_hints(root: type) -> None:
    _twins(root)
    assert root.sniff_line("T hello") is None
    with pytest.raises(AmbiguousFormatError) as info:
        root.sniff_line("T hello", error=True)
    message = str(info.value)
    assert 'hint="raw"' in message
    assert 'hint="cooked"' in message
