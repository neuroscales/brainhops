# Design: public, internal and private names

Status: decided, not implemented (#591, and #592 for the merge of copied
helpers). This memo records which names of `brainhops` other code may
import, and from where. It defines three levels of visibility, a module
named `_internal_api.py` through which a package offers its internal
names to other packages, and a test that enforces the rules. Decisions
are tagged `V1` to `V6`.

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
| Public  | listed in `__all__` of a public module    | anyone                  | kept, with deprecations  |
| Internal | a plain name in a module whose name starts with `_` | any module of `brainhops` (see V3) | may change without notice |
| Private | a name that starts with `_`               | the defining module only | none                     |

An internal name has a docstring and a signature that serves more than
one caller, because other modules depend on it. A private name is
never imported by another module, including tests.

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

V4. A package has an `_internal_api.py` module only when another
package imports its internal names. The first ones are those of
`datamodel`, `_core`, `io/base`, `io/transformations/base`,
`io/transformations/itk`, and each package under `io/common/` whose
names are used by both the images and the transformations of its
format. The modules of a package never import from their own
`_internal_api.py`, which only re-exports, so the module cannot create
a circular import inside its package.

V5. The helpers that were copied in several packages under `io/common/`
are merged into one internal function in `io/base` and offered through
`io/base/_internal_api.py`.

## 4. Enforcement

V6. Two checks keep the rules true. The ruff rule `PLC2701`
(import-private-name) rejects an import of a name that starts with an
underscore from another module. A test under `tests/` reads the import
statements of every module with `ast` and rejects an import of an
internal name from a module of another package that is not that
package's `_internal_api.py`. Ruff cannot express the second rule,
because its list of banned imports applies to the whole repository and
not to one package. The rule is also stated in `CLAUDE.md`.
