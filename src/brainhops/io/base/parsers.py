"""Independent sniffing, reading and writing capabilities for file formats.

Native format parsers compose readers and writers independently. Writers
do not inherit reading or sniffing. Dispatchers, which choose between
formats, have their own API in
[`FileFormat`][brainhops.io.base.FileFormat] and inherit none of these
classes.

The errors that these classes raise are also importable from here.
"""

__all__ = [
    "Confidence",
    "FileReader",
    "FileWriter",
    "BinaryFileReader",
    "BinaryFileWriter",
    "TextFileReader",
    "TextFileWriter",
    "disabled",
]

# The errors that the classes raise are importable from here too.
from brainhops.errors import (  # noqa: F401
    AmbiguousFormatError,
    ParserContentError,
    ParserError,
    ParserExistsError,
    ParserNotImplementedError,
    ParserTypeError,
    SnifferContentError,
    SnifferError,
    SnifferExistsError,
    SnifferNotImplementedError,
    SnifferTypeError,
    UnrepresentableTransformationError,
    WriterError,
    WriterNotImplementedError,
)

from ._parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    FileReader,
    FileWriter,
    TextFileReader,
    TextFileWriter,
    disabled,
)
