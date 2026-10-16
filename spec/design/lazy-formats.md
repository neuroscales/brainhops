# Formats that are imported when they are used

This memo describes how a file format can be registered before its module
is imported, so that importing brainhops does not import every format and
the optional libraries that the formats use. It replaces the central
table of format declarations proposed in #292. The mechanism is
implemented for the FSL transformations (FLIRT and FNIRT). The other
format families still register themselves when their modules are
imported, and are migrated one family at a time.

## The problem

Importing brainhops takes about 4 seconds, and about two thirds of that
time is spent importing `brainhops.io`, because every format module is
imported to register its format. Dispatch also needs every format to be
imported, because it calls the sniffer of every registered format on
every file that it loads. A user who reads one NIfTI file therefore
imports every reader that brainhops has.

## The decision

Each format is split into two classes that live in different modules.

The stub is a small class, defined in the `__init__` of the package of
the format. It derives from the dispatchers of the format, such as
`FslTransformationFormat`, and declares what dispatch reads before it
reads a file: the hints of the format, its file extensions and prefixes,
and its priority. The stub is decorated with
`@register_format("module:qualname")`, which names the format without
importing it. The stub is registered in the registry of every dispatcher
that it derives from, exactly as a format is.

```python
@register_format("._xform:FlirtTransform")
class FlirtTransformStub(FslAffineFormat):
    EXTENSIONS = (".mat",)
    HINTS = ("flirt",)
```

The format derives from its stub, and from whatever else it needs. Its
module imports what it uses at the top, as any other module does, because
it is only imported when the format is needed.

```python
class FlirtTransform(TextFileReader, FlirtTransformStub, Affine, ...):
    """Linear transformation stored in an FSL FLIRT `.mat` file."""
```

The first time dispatch needs a format, it imports the module of the
format and replaces the stub with the format in every registry. This step
is called resolving the stub. Resolving checks that the format derives
from its stub, and that it inherits the hints, the extensions, the
prefixes and the priority of the stub unchanged. A format that changes
them would be dispatched by the stub under different rules than its own,
so resolving it raises a `TypeError`. The stub is therefore the single
place where these attributes are written, and no test is needed to keep
two copies in agreement.

The package keeps its public names, such as
`brainhops.io.transformations.fsl.flirt.FlirtTransform`, through a
module-level `__getattr__` built by `brainhops._core.lazy.lazy_exports`
(see PEP 562), which imports the module that defines a name the first
time the name is accessed.

## Why a stub class and not a table or a keyword call

The hints, the extensions and the registries of a format are not
declared on one class. They are built up along its inheritance. For
example, `FnirtWarpField` declares only the hint `fnirt`. It answers to
`fsl` and to `xform.fsl.fnirt` because it derives from
`FslTransformationFormat`, and it claims `.nii` and `.nii.gz` because it
derives from the NIfTI base. A stub that derives from the same
dispatchers inherits all of this, and `register_format` finds its
registries by walking its bases, as it does for a format. A central
table, or a decorator that takes the hints as keywords, would have to
restate the hints and the registries of each format, far from the class,
and a test would have to keep the two copies in agreement.

A class that contributes hints or extensions without being a dispatcher
is split in the same way. `NiftiBasedTransformation` now derives from
`NiftiBasedTransformationStub`, which declares the hint `nifti` and the
NIfTI extensions and imports neither nibabel nor the data model, and the
stub of a NIfTI-based format derives from it.

## Dispatch

A registry is a set-like object that holds formats and stubs. Iterating
over a registry, or testing whether it holds a format, resolves its
stubs, so the code that reads a registry, such as the command line
interface and the tests, keeps working and always sees formats. Dispatch
reads the stubs instead.

`load` and `sniff` work in two passes. The first pass considers the
formats that are already imported, together with the stubs that may read
the input: the stubs that answer to the hints when hints are given, and
otherwise the stubs whose extension and prefix match the file name. Only
those stubs are resolved. If no format of the first pass reads the
input, the second pass considers every format, as dispatch did before. A
missing file is reported after the first pass. When options are given,
or when the input has no name, every stub is resolved at once, because
only a format can tell which options it accepts and a nameless input
gives nothing to choose by. `save` chooses its writer by extension, so it
resolves only the stubs whose extension matches the file name, or that
answer to the hints.

This changes the ranking in one case. A format that is not imported, and
whose extension does not match the file name, is no longer considered
when a format of the first pass reads the file. Before, it was sniffed
along with every other format, and it could win with a stronger sniffer
score or make the result ambiguous. A file that has the extension of its
format is read as before, and a misnamed file is still read in the
second pass.

## What a family needs before it can be migrated

The modules that a stub imports must be light, and so must the
`__init__` of every package on their path, because Python imports a
package before its submodules. For the FSL family this meant two
changes.

- `TransformationFormat` imported the conversions of transformations,
  which import the whole data model of transformations, at module level.
  It now imports them in the two methods that use them. With that change,
  the dispatchers of the FSL family import in about 0.25 seconds when the
  packages above them are light, and import neither the data model of
  transformations nor scipy.
- `FlirtMetadataStub` derives from `MetadataFormat`, which imports the
  data model of metadata, which imports the coordinate systems and the
  axes. `brainhops._core.typing` imports dask at module level, which
  costs about 0.5 seconds on its own. This is not fixed yet.

The packages above the FSL family, such as `brainhops.io.transformations`
and `brainhops.io.transformations.nifti`, still import their formats
eagerly. They become light once every family below them is migrated, and
only then does `import brainhops` become faster. Until then, the gain of
a migrated family is limited to the family itself: in a full import, the
FSL package now costs about 20 milliseconds instead of about 100, and the
FNIRT reader is not imported.

## Status

Implemented: the registry of stubs (`brainhops.io.base._registry`),
`register_format` with an import path, the two-pass dispatch in `load`,
`sniff` and `save`, `lazy_exports`, and the FSL family.

Not implemented: the other families, the light `__init__` of the
packages that aggregate them, and the removal of the module-level dask
import from `brainhops._core.typing`.
