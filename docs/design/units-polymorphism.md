# Design memo: units and `bagof.magic` polymorphism

Status: design only, no code. Relates to #54 (axes/units design) and #97
(adopt `bagof.magic` polymorphism across the datamodel).

> **Superseded by #289.** Units are now values backed by a private pint
> registry (see `brainhops.datamodel.units`): one interned `Unit` per
> unit, with `SpaceUnit`, `TimeUnit` and `IndexUnit` restricted to a
> dimension, and `Unit["<dimension>"]` for any other. The memo below
> describes the class-per-unit implementation that this replaced.

This memo audits the current units implementation
(`src/brainhops/datamodel/units.py`) and answers one question: does
`bagof.magic` polymorphism -- or another restructuring -- improve it? It
records the trade-offs, a recommended direction, and an honest call on
whether the work is worth doing now.

No units code is changed by the pull request that carries this memo.

## What units look like today

A unit is modelled as a *class*, and every concrete unit is a singleton
*instance* of one:

- `Unit` is the root. It carries no instance data at all: `name`,
  `scale` and `type` are `ClassVar`, and it is declared `init=False`,
  so an instance holds nothing but its class.
- `UnitSI` (with metaclass `_MetaUnitSI`) models a base SI unit plus an
  optional SI prefix. Its `name`, `symbol`, `scale` and `log10_scale`
  are class-level properties computed from two class variables, `base`
  and `prefix`.
- `KnownUnit` (with metaclass `_MetaKnownUnit`) models a fixed-scale
  named unit such as `inch` or `hour`, reading its scale and symbol out
  of the `UNITS` table.
- `TimeUnit` / `SpaceUnit` split the tree by dimension, and
  `TimeUnitSI` / `SpaceUnitSI` fix the base unit (`second`, `meter`).
- The `@siunit` decorator *generates* one subclass per SI prefix at
  import time (`Millimeter`, `Kilometer`, ...) with `type(name, bases,
  namespace)` and writes each into the module globals.
- `@register` interns one instance per class in `_REGISTERED_UNITS`.
- `Unit.__new__` does the real dispatch: it fuzzily parses the requested
  name with `_parse_unit_name` (matching full names, symbols, SI-prefix
  combinations, and the `µ`/`u`/`mc` spellings of *micro*), maps it to a
  `(prefix, base)` pair, looks up the generated class, and returns that
  class's interned singleton.

The module's own header carries the author's verdict: *"This is all
overly complicated and fiddly. But I am happy with the API for now."*
The API -- `SpaceUnit("millimeter")` returns the millimeter singleton --
is good; the implementation behind it is what is awkward.

## Does `bagof.magic` polymorphism fit?

`bagof.magic` polymorphism dispatches construction to a subclass by
reading **field values** off the call and matching them against each
subclass's `on={field: value}` registration (see the ITK and `Inverse`
work in this same effort). Measured against units, it is a poor fit, for
four independent reasons:

1. **There are no fields to match on.** Units are `init=False` with
   everything held as `ClassVar`. The `on={...}` mechanism reads a
   field's value out of the constructor arguments; with no init fields,
   there is nothing for it to read. The natural discriminant -- the unit
   *name* -- is not a field of the instance, so it cannot be the subject
   of an `on={"name": ...}` clause the way ITK's `type` field can.

2. **The discriminant is fuzzy, and the fuzziness is the actual
   complexity.** The hard part of units is `_parse_unit_name`: turning
   `"mm"`, `"millimetre"`, `"µm"` or `"mc"` into `(milli, meter)`.
   Polymorphism matches a value against an exact value, a set, a regex,
   or a type hint; it would still need `_parse_unit_name` to run first
   to produce a canonical key. Polymorphism replaces a registry lookup,
   not a parser, so it removes none of the code that is actually fiddly.

3. **Units are interned singletons; polymorphism builds fresh
   instances.** A matched polymorphic subclass is constructed with
   `target(*args, **kwargs)` -- a new object every time. The
   "same instance per name" guarantee that `Unit.__new__` provides would
   still have to live in a hand-written `__new__`, so the custom
   constructor does not go away.

4. **The subtypes are generated, not written.** The 25-ish prefixed
   classes per SI unit are produced by a loop in `@siunit`. Polymorphism
   registers a *fixed* set of subclasses named in source. One could loop
   and register each generated class with an `on=` clause, but that adds
   a second dispatch table beside the parser without deleting the first.

In short: the one place units *already* dispatch on a discriminant
(`Unit.__new__`) is dispatching on a parsed free-text name into a
dynamically generated, interned class hierarchy -- three properties that
each cut against the grain of the `on={field: value}` idiom. Adopting
polymorphism here would add machinery, not remove it.

## The restructuring that would help (and it is not polymorphism)

The awkwardness is not the dispatch; it is that **a unit is data wearing
the costume of a type**. A meter and a millimeter differ only in a scale
factor and a name, yet each is a distinct class with a generated
namespace and an interned instance, and two metaclasses exist only to
compute `name`/`symbol`/`scale` at the class level.

The direction that collapses all of this is a **value object**:

- One `Unit` (or one `Unit` per dimension: `SpaceUnit`, `TimeUnit`)
  holding plain fields -- `base` (e.g. `meter`), `prefix` (e.g. `milli`,
  or `None`), and the dimension. `scale`, `symbol`, `name` become
  ordinary instance properties, so both metaclasses disappear.
- The prefixed "classes" become *instances*: `Millimeter = Meter.scaled(milli)`,
  or simply module-level singletons built in the same loop that exists
  today -- but instances, not `type(...)` calls, so no class generation
  and no writing into globals.
- `Unit.parse(name)` (today's `_parse_unit_name`) stays exactly as is;
  it is the irreducible part. It returns a `Unit` value.
- Interning, if still wanted, becomes a cache keyed by `(dimension,
  base, prefix)` -- a normal `functools.lru_cache`-style memo on the
  parse/constructor, not a per-class registry.
- The `SpaceUnit` / `TimeUnit` split is kept as two thin value types (or
  a single type with a `dimension` field) so that an axis can still be
  typed as carrying a length unit rather than any unit.

This keeps the public API identical -- `SpaceUnit("millimeter")` still
answers with the millimeter -- while removing the class-per-unit
explosion, the two metaclasses, the dynamic class generation, and the
per-class singleton registry. `bagof.magic` still earns its keep here,
but as a plain frozen `Magic` value type with field converters (for the
name-parsing convert), not through its polymorphism option.

## Trade-offs

- **For the value-object rewrite.** Removes the two metaclasses, the
  `@siunit` class generation, and the class-keyed singleton registry;
  makes units comparable and hashable by value in the obvious way;
  makes a unit trivially serialisable as `(dimension, base, prefix)`.
- **Against, or at least the cost.** It is a breaking change to unit
  *identity*: today `type(SpaceUnit("mm"))` is `Millimeter`, and any
  consumer that branches on the unit's class (rather than its `name`,
  `scale` or `type`) would change. Every construction site -- the axis
  defaults (`SpaceUnit("millimeter")`, `TimeUnit("second")`), the io
  parsers that read units out of NIfTI/OME-Zarr, and any tests that
  assert on unit *type* -- must be checked. This is why #54 files units
  under "major rewrite; design phase" rather than a quick win.
- **Doing nothing** is a legitimate option: the API is good and the
  implementation, while fiddly, is self-contained behind `Unit.__new__`
  and well documented. The cost of the current form is paid by
  maintainers reading `units.py`, not by callers.

## Recommendation

1. **Do not apply `bagof.magic` polymorphism to units.** It does not fit
   the field-value dispatch model and would add a table without removing
   the parser or the singleton constructor.
2. **If and when units are revisited, prefer the value-object redesign**
   (single `Unit` value type per dimension, prefixed units as interned
   instances, `parse` as a classmethod) over both the current
   class-hierarchy form and over a polymorphic form.
3. **Sequence it as its own PR, after** the tractable polymorphism
   targets (ITK, `Inverse`) land, because it is a behaviour-affecting
   change to unit identity and touches every unit consumer. It is worth
   doing for maintainability, but it is not a low-risk refactor and
   should not ride along with the mechanical dispatch changes.

## One-line summary

Units already dispatch, but on a parsed free-text name into generated,
interned classes -- three things the `on={field: value}` idiom is the
wrong tool for; the real simplification is to make a unit a value object
rather than a class, and that is a deliberate follow-up rewrite, not part
of this polymorphism pass.
