# Design: public, internal and private names

Status: implemented (#591, and #592 for the merge of copied helpers).
This memo records which names of `brainhops` other code may import, and
from where. It defines three levels of visibility, a module named
`_internal_api.py` through which a package offers its internal names to
other packages, and a test that enforces the rules. Decisions are tagged
`V1` to `V6`.

## 1. The problem

A leading underscore is meant to say that a name is used only by the
module that defines it. In practice, 275 distinct names that start
with an underscore are imported by another module, in 555 places.
Most of these imports stay inside one package: 237 of those in `io`
import a name from a sibling module of the same format. About 250
cross from one package to another, mostly from the shared packages
under `io/common/` to the readers and writers of images and
transformations. Only 22 imports cross a top-level package: `io`
imports `_as_affine`, `_convert_withlog` and `_convert_withsplines`
from `datamodel`, and `datamodel` imports five sampling helpers from
`_core`.

The underscore therefore no longer tells a reader whether a change to a
name can break another module. Some helpers have also been copied
instead of shared: `_declined`, `_no_options` and `_sniffed_raw` are
defined separately in `io/common/afni`, `io/common/nrrd`,
`io/common/mrtrix` and `io/common/minc`.

## 2. Three levels

V1. Every name of `brainhops` has one of three levels of visibility.

| Level   | Spelling                                  | May be imported by      | Stability                |
| ------- | ----------------------------------------- | ----------------------- | ------------------------ |
| Public  | listed in `__all__` of a public module    | anyone                  | kept (see below)         |
| Internal | a plain name in a module whose name starts with `_` | any module of `brainhops` (see V3) | may change without notice |
| Private | a name that starts with `_`               | the defining module only | none                     |

A public name is kept from one release to the next, and a change to it
goes through a deprecation. This applies once a release has users. Until
then, a public name may change or be removed without a deprecation,
because no code outside the repository depends on it. An internal name
has a docstring and a signature that serves more than one caller,
because other modules depend on it. A private name is
never imported by another module, including tests.

A module counts as private when its name, or the name of a package that
holds it, starts with an underscore: the plain names of
`brainhops._core.path` are internal. A name that a public module lists
in its `__all__` is public, even when a private module defines it, as
`brainhops.backends` does for the names of `brainhops._core.backends`.

V2. A name that another module needs is promoted, not imported with its
underscore. It becomes internal when only `brainhops` needs it and
public when users need it too. `_as_affine` becomes the public
`to_affine()` (see [transformation-filters.md](transformation-filters.md)),
and `_convert_withlog` and `_convert_withsplines` become internal
because the converters of every file format use them.

## 3. The `_internal_api.py` modules

V3. A module may import an internal name directly from the module that
defines it when both modules are in the same package. A module in
another package imports internal names only from the `_internal_api.py`
module of the package that defines them. That module re-exports the
internal names that other packages use, and nothing else, so the
contract between packages is written in one place. The name says that
the module is an interface, which `_internal` alone would not, and
that it is not public.

A package is a directory that holds an `__init__.py` file, and two
sibling subpackages, such as `io/common/nifti` and `io/images/nifti`,
are different packages. A subpackage whose name starts with an
underscore, below the top-level packages of `brainhops`, is part of the
package that holds it: `datamodel/_transformations/compute` is part of
`datamodel`, while `_core` is a package of its own.

A test may import an internal name directly from the module that
defines it, so an `_internal_api.py` module never re-exports a name
that only tests use.

V4. A package has an `_internal_api.py` module only when another
package imports its internal names. The modules of a package never
import from their own `_internal_api.py`, which only re-exports, so the
module cannot create a circular import inside its package. The
`_internal_api.py` module of `_core` imports `bsplines` after its other
modules, because `bsplines` imports `brainhops.datamodel.enums`, which
imports names of `_core` through that module.

V5. The helpers that were copied in several packages under `io/common/`
are merged into one internal function in `io/base` and offered through
`io/base/_internal_api.py`. The module `io/base/_sniffing.py` defines
them:

- `no_options(kwargs, owner)` refuses the options given to a reader
  that takes none;
- `declined(error, message, failure=None)` declines sniffed content,
  and raises the error that the caller asks for;
- `checked_score(score, error, message)` returns a score, or raises
  when the score is zero and the caller asks for an error;
- `sniffed_raw(read, file)` reads the record of a stream for a sniffer,
  and returns the error instead of raising it.

The copies differed only in the words of their messages, which each
format now gives as the constants `OPTION_OWNER` and `DECLINE_MESSAGE`.
The AFNI sniffer also validates the record and restores the position of
the stream, which its function `read_validated_header` does before
`sniffed_raw` is called. The AFNI and MRtrix transformations, the N5
images and the precomputed images use the shared functions too.

Some helpers with the same names differ in behaviour and keep their own
definitions. The `no_options` of Imaris sorts the names of the options,
and its `_declined` raises without replacing the context of the error.
The other helpers named `no_options` or `_no_options`, in the
BigDataViewer, JPEG 2000, OpenSlide, Pillow and Zarr image formats and
in the MINC, c3d, NiftyReg and TrakEM2 transformation formats, word
their messages differently, and most of them name the class or the
method that refuses the options. The helpers named `score` or `_score`
of the BigDataViewer images and of the AFNI and MINC transformations
compute the score themselves.

## 4. Enforcement

V6. The test `tests/test_visibility.py` reads the import statements of
every module of `src/` and `tests/` with `ast`. It rejects an import of
a private name from another module, an import by a module of `src/` of
an internal name of another package from any module but that package's
`_internal_api.py`, an import of the `_internal_api.py` module of the
importing package, and an `_internal_api.py` module that re-exports a
name that no module of another package imports. The rules are also
stated in `CLAUDE.md`.

The ruff rule `PLC2701` (import-private-name) is not enabled, for three
reasons. It is a preview rule, which needs the preview mode of the
linter. Ruff treats `brainhops` as one package, so the rule reports no
import between the modules of `src/`. In `tests/`, it rejects the
imports from private modules such as `brainhops._core`, which the
internal level allows.

## 5. Implementation

The packages that have an `_internal_api.py` module are `_core`,
`_ext/invfield`, `datamodel`, `datamodel/metadata`, `io/base`,
`io/metadata`, `io/common`, and its subpackages `afni`, `bdv`,
`freesurfer`, `hdf5`, `mgh`, `minc`, `mrtrix`, `nifti`, `nrrd` and
`zarr`, `io/images/base`, `io/images/nifti`, `io/images/pillow`,
`io/images/tiff`, `io/transformations/base`,
`io/transformations/freesurfer`, `io/transformations/freesurfer/lta`,
`io/transformations/fsl`, `io/transformations/itk`,
`io/transformations/nifti` and `io/transformations/zarr`.

The `_internal_api.py` module of `_core` re-exports the affine versions
of `inv`, `expm`, `logm` and `sqrtm` of `_core/affines.py` as
`affine_inv`, `affine_expm`, `affine_logm` and `affine_sqrtm`, because
`_core/linalg.py` defines matrix functions with the same names.

When the plain spelling of a promoted name was already used by the
module or by one that imports it, the name says what the object is
instead, for example `header_of` for `_header` in `io/common/nifti`,
`parse_float` for `_float` in `io/common/nrrd`, and `TwoAxes` and
`ThreeAxes` for the type aliases `_2Axes` and `_3Axes`.

Public modules used to define private names that other modules
imported, which removing the underscore would not have made internal.
These names moved into private modules:

- `ComparesByIdMixin`, which `datamodel/base.py` listed in its
  `__all__`, is in `_core/compare.py`, since it does not depend on
  `datamodel`;
- `index2transform`, from `datamodel/geometry.py`, is in
  `datamodel/_indexing.py`, and `match_resolution`, from
  `datamodel/images.py`, is in `datamodel/_resolution.py`;
- the implementation of `io/base/parsers.py` is in `io/base/_parsers.py`,
  where its mixins and the helpers that mark how a `from_fileobj` reads
  its stream are internal, and `parsers.py` only re-exports the public
  classes of its `__all__` and the errors that they raise;
- `check_endpoint` and `compatible`, from
  `io/transformations/base/affines.py`, are in
  `io/transformations/base/_endpoints.py`. The constants `_VOXEL`,
  `_RAS` and `_LPS` are gone, and each use builds the system that it
  needs, since these systems compare by value.

The test therefore checks the rules without exceptions.
