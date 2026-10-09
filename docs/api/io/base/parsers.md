# Reading and writing capabilities

Reading, writing and format dispatch have separate APIs. `FileReader`
provides `sniff*`, `load` and `from_*` for one format; `FileWriter` provides
`save` and `to_*`. `Format` selects among registered formats.
Readers and dispatchers share private input adapter mixins, but neither
inherits the other's public API. In particular, a reader's `sniff*` returns
a confidence score; a dispatcher's `sniff*` returns a selected class or
`None`. These capabilities add no fields to the object's representation.
Text and binary variants supply the corresponding stream modes and adapters.

An internal format representation may use any fields or third-party type.
When implementing its own I/O, it can inherit a reader and a writer
independently:

```python
class LtaStruct(TextFileReader, TextFileWriter):
    # Native fields and implementations of from_lines / to_lines.
    ...
```

A write-only representation inherits only `TextFileWriter` or
`BinaryFileWriter`. It does not acquire `load`, `from_*`, or `sniff*`.
Conversely, a reader does not acquire writing methods.

Public format objects implement the Image/Transformation API and hold their
internal representation. They can use the shared readers and writers without
inheriting a parser. `Format` owns the format-selection methods and reuses
the input adapter mixins. `FileBasedObject` is the root registry for public
objects; `ImageFormat` and `TransformationFormat` scope it to images and
transformations. `TransformationFormat` also supplies the `xform` hint,
without a separate marker class. Neither dispatcher inherits `Image` or
`Transformation`; concrete objects supply their model. The dispatchers retain field-free construction support
from `DataModelBase` for cooperative `from_any` and `from_instance` calls.
A family owns a registry through `@format_registry`; a concrete format joins its ancestors' registries
through `@register_format`.

```python
@register_format
class MyTransformation(
    TextFileReader, TextFileWriter, TransformationFormat, Affine
):
    # Model properties wrap a private native representation.
    # from_lines reads that representation, then wraps it.
    # to_lines delegates to the representation of the current model.
    ...
```

These examples illustrate the bases rather than complete format
implementations. List the specialized text/binary adapters before generic
I/O bases so that their encoding and stream modes take precedence. Readers
and writers share no inheritance dependency; when both occur in a hierarchy,
keep their order consistent across its bases.

`Format` provides dispatch entry points, without inheriting `FileReader` or
`FileWriter`. A public export-only object combines its model with
`FileBasedObject` and a writer. Generic `io.save` selects writers from the
`FileBasedObject` registry. Sniffing and loading exclude writers that have
no implemented read route, so an export-only format is never selected for
reading just because its extension matches.

A standalone dispatcher such as the proposed `FileBasedMetadata` can inherit
`Format` and own its own registry without inheriting `FileBasedObject`.
Its concrete formats then participate only in that
dispatcher, not in generic object loading or saving. An object's metadata
view may share its native representation; this capability split imposes no
metadata schema or mutation policy.

The generic `FileParser`, `TextFileParser`, and `BinaryFileParser` bases are
replaced by `FileReader`, `TextFileReader`, and `BinaryFileReader`. Replace
the combined `*FileParserWriter` bases with the corresponding reader and
writer bases explicitly. Format-specific parser names, such as `LtaParser`
and `NiftiParser`, remain: a format parser may implement both routes.

The `WritableFileBased*` wrappers are removed as well. Use the corresponding
`FileBased*` class and obtain writing from the native parser or an explicit
writer base. Code that checks writing support should test `FileWriter`.
Dispatchers do not inherit `FileReader`; concrete public
objects can additionally compose readers for their own format.

The text/binary `FileBasedObject` convenience classes and standalone sniffer
bases are removed. Compose the appropriate reader or writer with the public
object registry; sniffing belongs to the reader API.

# ::: brainhops.io.base.parsers
