"""The syntax of an MNI transform file, as libminc reads and writes it.

The reader follows `input_transform` and `input_one_transform` in
libminc's `volume_io/MNI_formats/gen_xf_io.c`, and the tokens are read
as the functions of `volume_io/MNI_formats/mni_io.c` read them
(BIC-MNI/libminc, commit e71e590). The writer follows
`output_transform` and `output_one_transform` in the same file. The
package documentation, [`brainhops.io.transformations.minc`][], describes
the syntax.
"""

# stdlib
import math
import re

# dependencies
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import ParserContentError

HEADER = "MNI Transform File"
"""The first line of every MNI transform file."""

TYPE = "Transform_Type"
LINEAR = "Linear"
LINEAR_TRANSFORM = "Linear_Transform"
GRID = "Grid_Transform"
DISPLACEMENT_VOLUME = "Displacement_Volume"
THIN_PLATE_SPLINE = "Thin_Plate_Spline_Transform"
NUMBER_DIMENSIONS = "Number_Dimensions"
POINTS = "Points"
DISPLACEMENTS = "Displacements"
INVERT_FLAG = "Invert_Flag"
TRUE = "True"
FALSE = "False"

COMMENT_CHARACTERS = "%#"
"""The characters that start a comment, which ends with the line."""

_NUMBER = re.compile(
    r"[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)",
    re.IGNORECASE,
)
"""The longest prefix of a token that `sscanf("%lf")` reads."""

_INTEGER = re.compile(r"[+-]?\d+")
"""The longest prefix of a token that `sscanf("%d")` reads."""


class Cursor:
    """A position in the text of a transform file, read as libminc reads it.

    The methods mirror the functions of libminc's `mni_io.c`. Whitespace
    and comments are skipped only before a token, so a comment character
    inside a token, such as a file name, is part of the token. Each
    comment that is skipped is recorded in `comments`, from its comment
    character to the end of its line.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        """The text that is read."""
        self.position = 0
        """The index of the next character to read."""
        self.comments: tx.List[str] = []
        """The comments skipped so far, in the order of the file."""

    def character(self) -> tx.Optional[str]:
        """Return the next character, or `None` at the end of the text."""
        if self.position >= len(self.text):
            return None
        character = self.text[self.position]
        self.position += 1
        return character

    def nonwhite(self) -> tx.Optional[str]:
        """Return the next character that is neither blank nor in a comment.

        This is `mni_get_nonwhite_character`. `None` is returned at the
        end of the text.
        """
        in_comment = False
        while True:
            character = self.character()
            if character is None:
                return None
            if character in COMMENT_CHARACTERS:
                if not in_comment:
                    self._record_comment(self.position - 1)
                in_comment = True
            elif character == "\n":
                in_comment = False
            if not in_comment and character not in " \t\n\r":
                return character

    def string(self, terminators: str) -> tx.Optional[str]:
        """Read a token up to one of `terminators` or the end of the line.

        This is `mni_input_string`. A token that starts with a double
        quote ends at the next double quote instead, which is consumed.
        Another terminator is left to be read. Trailing spaces are
        removed and carriage returns are ignored. `None` is returned when
        the text ends before the token starts.

        Raises
        ------
        ParserContentError
            If the text ends inside the token.
        """
        character = self.nonwhite()
        if character is None:
            return None
        quoted = character == '"'
        if quoted:
            character = self.nonwhite()
            terminators = '"'
        token = []
        while (
            character is not None
            and character not in terminators
            and character != "\n"
        ):
            if character != "\r":
                token.append(character)
            character = self.character()
        if character is None:
            raise self.error("The file ends inside a token.")
        if not quoted:
            self.position -= 1
        return "".join(token).rstrip(" ")

    def expect(self, expected: str) -> None:
        """Read the next character that is not blank, which must be `expected`.

        This is `mni_skip_expected_character`.

        Raises
        ------
        ParserContentError
            If another character, or the end of the text, comes first.
        """
        character = self.nonwhite()
        if character != expected:
            found = "the end of the file" if character is None else character
            raise self.error(f"Expected {expected!r}, found {found!r}.")

    def keyword(self, keyword: str) -> None:
        """Read `keyword` followed by an equal sign.

        This is `mni_input_keyword_and_equal_sign`.

        Raises
        ------
        ParserContentError
            If something else comes first.
        """
        found = self.string("=")
        if found != keyword:
            raise self.error(f'Expected "{keyword} =", found {found!r}.')
        self.expect("=")

    def real(self) -> tx.Optional[float]:
        """Read a number, or return `None` if the next token is not one.

        This is `mni_input_real`, which reads the token with
        `sscanf("%lf")`, so trailing characters of the token are ignored.
        A token ends at a space, a tab, a semicolon or the end of the line.
        libminc does not end a token at a tab, and brainhops does, so that
        numbers separated by tabs are read. A token that is not a number is
        left to be read, and so is a semicolon.
        """
        start, comments = self.position, len(self.comments)
        token = self.string(" \t;")
        match = None if token is None else _NUMBER.match(token)
        if match is None:
            # The comments before the token are skipped again later.
            self.position = start
            del self.comments[comments:]
            return None
        return float(match.group())

    def reals(self, count: tx.Optional[int] = None) -> tx.List[float]:
        """Read `count` numbers, or all numbers up to a semicolon.

        With a count, this reads as many numbers as `mni_input_real` is
        called for in `input_one_transform`. Without one, it is
        `mni_input_reals`. The semicolon is not read.

        Raises
        ------
        ParserContentError
            If fewer than `count` numbers come first.
        """
        values = []
        while count is None or len(values) < count:
            value = self.real()
            if value is None:
                if count is None:
                    break
                raise self.error(
                    f"Expected {count} numbers, found {len(values)}."
                )
            values.append(value)
        return values

    def integer(self) -> int:
        """Read an integer, as `mni_input_int` does.

        Raises
        ------
        ParserContentError
            If the next token is not an integer.
        """
        token = self.string(" \t;")
        match = None if token is None else _INTEGER.match(token)
        if match is None:
            raise self.error(f"Expected an integer, found {token!r}.")
        return int(match.group())

    def error(self, message: str) -> ParserContentError:
        """Return the error of a file that libminc would not read."""
        line = self.text.count("\n", 0, self.position) + 1
        return ParserContentError(
            f"Not a valid MNI transform file, at line {line}: {message}"
        )

    def _record_comment(self, start: int) -> None:
        end = self.text.find("\n", start)
        end = len(self.text) if end < 0 else end
        self.comments.append(self.text[start:end].rstrip("\r"))


def format_number(value: float) -> str:
    """Return the text of a number, which reads back as the same number.

    libminc writes numbers with `%.15g`, which can lose the last digits of
    a double. The shortest text that reads back exactly is written
    instead, which `sscanf` reads as libminc does.
    """
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)
