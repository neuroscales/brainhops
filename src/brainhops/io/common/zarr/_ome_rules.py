"""Check of the OME metadata that a writer is about to write.

The OME-Zarr writers build their metadata through the typed models of
abczarr, which refuse malformed JSON. The check in this module runs on
the result, just before it is written, so that a writer that breaks a
rule of OME-NGFF raises a `WriterError` instead of writing a store that
other readers may refuse.

The metadata is first validated against the JSON schema of its version,
which abczarr bundles. The rules of OME-NGFF 0.6 that a schema cannot
express are then checked on the metadata of that version:

- a dataset is placed by one scale, one identity, or one sequence of a
  scale and a translation, and any other placement belongs to the
  coordinate transformations of the multiscale;
- the coordinate systems of a multiscale have distinct names;
- the input and the output of every coordinate transformation name a
  coordinate system of the multiscale or the path of one of its
  datasets.

Each axis must also be spelled out in full, but the schema already
requires the name of an axis, and the writers leave out only what they
do not know, such as an unknown unit.
"""

import typing_extensions as tx

from brainhops.io.base.parsers import WriterError

_TOP_LEVEL_VERSIONS = frozenset({"0.1", "0.2", "0.3", "0.4"})
"""The versions whose metadata is not stored under an `ome` key."""


def check_ome(ome: tx.Any) -> None:
    """
    Check the OME metadata that a writer is about to write.

    The metadata is validated against the JSON schema of its version.
    Metadata of OME-NGFF 0.6 is also checked against the rules that the
    module docstring lists.

    Parameters
    ----------
    ome : Any
        The typed OME metadata of abczarr, with its `version`.

    Raises
    ------
    WriterError
        If the metadata breaks a rule.
    """
    from abczarr.errors import SchemaValidationError
    from abczarr.ome import schemas

    version = str(ome.version)
    document = ome.to_json()
    instance = (
        document if version in _TOP_LEVEL_VERSIONS else {"ome": document}
    )
    try:
        schemas.validate(instance, version, "image")
    except SchemaValidationError as error:
        raise WriterError(
            f"The OME metadata to write does not follow the schema of "
            f"OME-NGFF {version}: {error}"
        ) from error
    if version == "0.6":
        for multiscale in document.get("multiscales", []):
            _check_multiscale(multiscale)


def _check_multiscale(multiscale: tx.Mapping[str, tx.Any]) -> None:
    # Check the rules of OME-NGFF 0.6 that the schema cannot express.
    names = [system["name"] for system in multiscale["coordinateSystems"]]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise WriterError(
            f"The OME metadata to write has several coordinate systems "
            f"named {', '.join(repr(name) for name in repeated)}."
        )
    paths = [dataset["path"] for dataset in multiscale["datasets"]]
    for dataset in multiscale["datasets"]:
        entries = dataset.get("coordinateTransformations", [])
        if len(entries) != 1 or not _is_placement(entries[0]):
            raise WriterError(
                f"The dataset {dataset['path']!r} of the OME metadata to "
                f"write is placed by {_kinds(entries)}, whereas a dataset "
                f"holds only a scale, an identity, or a sequence of a "
                f"scale and a translation."
            )
        for entry in entries:
            _check_references(entry, names, paths)
    for entry in multiscale.get("coordinateTransformations", []):
        _check_references(entry, names, paths)


def _is_placement(entry: tx.Mapping[str, tx.Any]) -> bool:
    # Return whether a transformation may place a dataset.
    if entry.get("type") in ("scale", "identity"):
        return True
    if entry.get("type") != "sequence":
        return False
    kinds = [part.get("type") for part in entry.get("transformations", [])]
    return kinds == ["scale", "translation"]


def _kinds(entries: tx.Sequence[tx.Mapping[str, tx.Any]]) -> str:
    # Describe the types of a list of transformations for a message.
    if not entries:
        return "nothing"
    kinds = []
    for entry in entries:
        kind = str(entry.get("type"))
        parts = entry.get("transformations")
        if parts:
            kind += "(" + ", ".join(str(p.get("type")) for p in parts) + ")"
        kinds.append(kind)
    return ", ".join(kinds)


def _check_references(
    entry: tx.Mapping[str, tx.Any],
    names: tx.Sequence[str],
    paths: tx.Sequence[str],
) -> None:
    # Check that the input and the output of a transformation name a
    # coordinate system or a dataset of the multiscale.
    for key in ("input", "output"):
        reference = entry.get(key)
        if reference is None:
            continue
        if "name" in reference and reference["name"] in names:
            continue
        if "name" not in reference and reference.get("path") in paths:
            continue
        raise WriterError(
            f"The {key} {reference!r} of a coordinate transformation of "
            f"the OME metadata to write names neither a coordinate system "
            f"nor a dataset of its multiscale."
        )
