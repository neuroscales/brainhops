"""The record of a Neuroglancer coordinate transform: its JSON object."""

__all__ = ["NeuroglancerTransformRaw"]

# stdlib
import copy
import json

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.datamodel.metadata._internal_api import takes_error_policy

# internals
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
)

# this format
from ._dimensions import Dimension, check_content, dimensions, is_content


class NeuroglancerTransformRaw(Raw, TextFileReader, TextFileWriter, eq=False):
    """Record of a Neuroglancer coordinate transform, as its JSON holds it.

    Neuroglancer, a viewer of large volumes in the browser, places each
    data source of a layer with a coordinate transform, which is the
    `transform` of the `source` of a layer in the state of the viewer:

    ```json
    {
      "matrix": [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30]],
      "outputDimensions": {"x": [1e-09, "m"], "y": [1e-09, "m"],
                           "z": [1e-09, "m"]},
      "inputDimensions": {"x": [4e-09, "m"], "y": [4e-09, "m"],
                          "z": [4e-08, "m"]}
    }
    ```

    The record holds that object, parsed, in `content`, and the text it
    was read from in `text`. The JSON is short and is read in one pass,
    so the matrix of the transform lives in the record, as the matrix of
    an LTA file does. The record checks its content as Neuroglancer reads
    it (see [`brainhops.io.transformations.neuroglancer`][] for the
    conventions and their sources), and it keeps the keys that
    Neuroglancer does not read.

    A record is never changed in place, so that several objects can
    share it. It is held by
    [`NeuroglancerTransformMetadata`][brainhops.io.transformations.neuroglancer.NeuroglancerTransformMetadata],
    and a
    [`NeuroglancerTransform`][brainhops.io.transformations.neuroglancer.NeuroglancerTransform]
    decodes its matrix and its coordinate systems from it.

    A record compares by identity. Two records hold the same transform
    when their `content` are equal.
    """

    content: tx.Dict[str, tx.Any]
    """The JSON object of the transform, with its keys in file order."""

    text: NoRepr[tx.Optional[str]] = None
    """The text that the content was read from.

    The text is written back as it is while it still parses to the
    content, so that a transform that is read and written again keeps
    its bytes, whichever program wrote it. Otherwise, and for a record
    built from its content alone, the content is written as JSON with
    one key, or one row of the matrix, on each line. The text is `None`
    for a record built from its content.
    """

    def __post_init__(self) -> None:
        check_content(self.content)

    # --- views --------------------------------------------------------

    @property
    def rank(self) -> int:
        """The number of dimensions on each side of the transform."""
        return len(self.content["outputDimensions"])

    @property
    def matrix(self) -> tx.Tuple[tx.Tuple[float, ...], ...]:
        """The `matrix` of the transform, row by row, as the file stores it.

        A transform without `matrix` is the identity, as Neuroglancer
        reads it, and the identity is returned. The rows have `rank + 1`
        numbers, the last of which is the translation.
        """
        rank = self.rank
        stored = self.content.get("matrix")
        if stored is None:
            return tuple(
                tuple(1.0 if i == j else 0.0 for j in range(rank + 1))
                for i in range(rank)
            )
        return tuple(tuple(float(v) for v in row) for row in stored)

    @property
    def output_dimensions(self) -> tx.Tuple[Dimension, ...]:
        """The output dimensions, with their prefixes folded into the scale."""
        return dimensions(self.content["outputDimensions"], "outputDimensions")

    @property
    def input_dimensions(self) -> tx.Optional[tx.Tuple[Dimension, ...]]:
        """The input dimensions, or `None` when the transform has none."""
        if "inputDimensions" not in self.content:
            return None
        return dimensions(self.content["inputDimensions"], "inputDimensions")

    @property
    def source_rank(self) -> int:
        """The number of input dimensions that the data source has.

        The other input dimensions only embed the source in a larger
        space. A transform without `sourceRank` has as many as its rank.
        """
        return self.content.get("sourceRank", self.rank)

    def copy(self) -> "NeuroglancerTransformRaw":
        """Return a copy of the record that can be changed freely.

        The content is copied deeply, and the text is kept, since
        [`to_text`][] writes it only while it still parses to the content.

        Returns
        -------
        NeuroglancerTransformRaw
            The copy.
        """
        return NeuroglancerTransformRaw(
            content=copy.deepcopy(self.content), text=self.text
        )

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score text as a Neuroglancer coordinate transform.

        A JSON file is recognized as a transform by its content alone
        only when its keys are among the four that Neuroglancer reads
        (`matrix`, `outputDimensions`, `inputDimensions` and
        `sourceRank`) and it is a valid transform. Other JSON is read only
        when the format is asked for, by its class or by a hint.

        Parameters
        ----------
        text : str
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not a transform.
        **kwargs : Any
            Options of the reader, which do not change the score.

        Returns
        -------
        float
            `Confidence.LIKELY` for a transform, and `Confidence.NO`
            otherwise.
        """
        if text.lstrip().startswith("{"):
            try:
                if is_content(json.loads(text)):
                    return Confidence.LIKELY
            except ValueError:
                pass
        if error:
            if error is True:
                error = SnifferContentError
            raise error("The text is not a Neuroglancer coordinate transform.")
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> tx.Self:
        """Read a record from the text of a JSON object.

        Parameters
        ----------
        text : str
            The JSON text of the transform.

        Returns
        -------
        NeuroglancerTransformRaw
            The record, which keeps the text.

        Raises
        ------
        ParserContentError
            If the text is not JSON, or not a coordinate transform.
        """
        try:
            content = json.loads(text)
        except ValueError as error:
            raise ParserContentError(
                f"A Neuroglancer coordinate transform is JSON: {error}"
            ) from error
        return cls(content=content, text=text)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str]) -> tx.Self:
        """Read a record from the lines of a JSON object.

        Parameters
        ----------
        lines : iterable of str
            The lines, without line endings.

        Returns
        -------
        NeuroglancerTransformRaw
            The record.
        """
        return cls.from_text("\n".join(lines) + "\n")

    # --- to -----------------------------------------------------------

    @takes_error_policy
    def to_text(self) -> str:
        """Return the JSON text of the transform.

        The text that the record was read from is returned while it
        still parses to the content. Otherwise the content is written as
        JSON with one key on each line and one row of the matrix on each
        line, ending with a newline.

        Returns
        -------
        str
            The text of the file.
        """
        if self.text is not None and _parses_to(self.text, self.content):
            return self.text
        return _dumps(self.content)

    @takes_error_policy
    def to_lines(self) -> tx.Iterator[str]:
        """Yield the lines of the JSON text, without line endings.

        Yields
        ------
        str
            The lines of [`to_text`][].
        """
        yield from self.to_text().splitlines()


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _parses_to(text: str, content: tx.Dict[str, tx.Any]) -> bool:
    # Whether the text still holds the content, keys in the same order.
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    return parsed == content and list(parsed) == list(content)


def _dumps(content: tx.Dict[str, tx.Any]) -> str:
    # One key on each line, and one row of the matrix on each line.
    lines = []
    for key, value in content.items():
        name = json.dumps(key, ensure_ascii=False)
        if key == "matrix" and isinstance(value, list) and value:
            rows = ",\n".join(f"    {json.dumps(row)}" for row in value)
            lines.append(f"  {name}: [\n{rows}\n  ]")
        else:
            lines.append(f"  {name}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(lines) + "\n}\n"
