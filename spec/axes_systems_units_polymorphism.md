# Axes, coordinate systems, and units: can `magic` polymorphism simplify them?

Status: audit and design memo for #54. No code. Deliverable is a recommended
direction and an honest call on whether it is worth doing.

## 0. Question

The datamodel proliferates subclasses that exist only to fix field values:

- Every anatomical axis (`LeftToRightAxis`, `PosteriorToAnteriorAxis`, …) fixes
  `orientation` (and `type`) with `HiddenConst`.
- Every named coordinate system (`RASCoordinateSystem`, `LPSCoordinateSystem`,
  and the F/C-voxel and 2D/3D combinations `FRASCoordinateSystem`,
  `CVoxelCoordinateSystem`, …) fixes the `axes` tuple.
- Units carry a bespoke machine (`units.py`): a singleton registry, a
  `__new__` that dispatches by parsing a unit name, a `siunit` decorator that
  code-generates a subclass per SI prefix, and metaclass properties that
  compute `scale`. The file's own header calls it "overly complicated and
  fiddly."

Can `bagof.magic`'s `polymorphic` option collapse these?

## 1. What `polymorphic` actually does

A class marked `polymorphic=True` lets each subclass declare the argument
values it stands for, and calling the base returns the matching subclass:

```python
class Chord(Magic, polymorphic=True):
    mode: str
class MinorChord(Chord, on={"mode": "minor"}): ...

Chord(mode="minor")   # -> MinorChord(...)
```

Constraints range over exact values, set membership, regex, and type hints.
When several subclasses match, the winner is chosen by `priority`, then by the
number of fields constrained, then by how precise the constraints are, then by
depth in the hierarchy. Ties raise rather than depend on import order.
`polymorphic="strict"` refuses to build the base when nothing matches.

The essential point: polymorphism is **value → subtype dispatch**. It is a way
to *hand back* one of the subclasses you keep. It does not, by itself, reduce
how many subclasses exist.

## 2. What actually causes the proliferation

Two mechanisms, and polymorphism addresses neither directly:

1. **Fixed values carried as types.** An anatomical axis is a `SpatialAxis`
   with `orientation` pinned by `HiddenConst`. The subclass exists to hold a
   constant. The adaptor (see the axis-order spec) matches axes on the
   `(type, orientation)` *field values*, not on the Python type. So no behaviour
   depends on `LeftToRightAxis` being a distinct class.
2. **Combinatorial multiple inheritance.** `FRASCoordinateSystem` is
   `RASCoordinateSystem` × `FVoxelCoordinateSystem`; the full set is
   {anatomical orientation} × {C, F order} × {2D, 3D}. This is a product of
   independent axes of variation expressed as a type lattice.

The lever that collapses both is the same: **move the distinction from the type
into a field value, and keep a named singleton for each canonical instance.**
`R` is already a singleton `LeftToRightAxis()`; the useful part is the value
`R`, not the class. `RAS` can be a singleton `SpatialCoordinateSystem(axes=(R,
A, S))` rather than a class.

## 3. Where polymorphism does earn its place

Polymorphism is complementary to that collapse, at exactly one boundary: a
reader has raw values and wants the right named object back.

- A NIfTI/ITK reader that recovers an orientation returns the canonical axis by
  constructing the base with the value, and dispatch names it.
- A file that states an axis tuple recovers the named system the same way.

So the recommendation is not "polymorphism instead of subclasses" nor "keep the
lattice." It is: reduce the lattice by moving fixed distinctions into fields,
keep singletons for the canonical instances, and where a named subclass is still
worth having, mark the base `polymorphic=True` so value-based construction still
returns it. Polymorphism becomes the constructor front door, not the taxonomy.

## 4. Axes

- **Recommended.** One `SpatialAxis` with `orientation` (and `name`) as ordinary
  fields; retire the per-orientation subclasses. Keep the six singletons
  `R, L, A, P, S, I`. The adaptor already matches on the field, so nothing that
  consumes axes regresses.
- **Optional.** If a reader benefits from `SpatialAxis(orientation=…) ->` a named
  type, add `polymorphic=True` on `SpatialAxis`. Low urgency: the singletons are
  the real API, and the classes are already thin.
- **Value:** modest. The axis classes are clean; the win is fewer names and one
  place for axis behaviour.

## 5. Coordinate systems

- **Recommended.** Represent the two independent axes of variation —
  orientation and array order — as the `axes` tuple on a small number of base
  classes (`CoordinateSystem2D/3D`, `SpatialCoordinateSystem`), and provide the
  named systems as **singletons** (`RAS`, `LPS`, `fRAS`, `cRAS`, …) rather than
  as a class per combination. This directly removes the multiple-inheritance
  product.
- **With polymorphism.** Mark `CoordinateSystem` (or `SpatialCoordinateSystem3D`)
  `polymorphic=True` and register the named systems `on={"axes": (…)}`, so
  `CoordinateSystem(axes=(R, A, S))` returns the `RAS` type for a reader. Exact
  tuple match is the narrowest constraint, so dispatch is unambiguous.
- **Caution.** Keep a genuine class only where one carries behaviour or a
  docstring that pulls its weight (`RAS` as a documented anatomical convention
  is defensible). A class that only fixes a tuple should become a singleton.
- **Value:** medium-high. This is where the combinatorial blow-up lives.

## 6. Units

- The current machine is the strongest candidate for replacement, and the least
  well served by polymorphism alone.
- **Recommended.** One `Unit` (or `UnitSI`) class holding `base` and `prefix` as
  fields, with `scale`, `symbol`, and `name` computed from them. Provide named
  singletons (`Meter`, `Millimeter`, `Second`, …) as instances. This retires the
  `siunit` code-generation, the singleton-dispatch `__new__`, and most of the
  metaclass property surface in one move — the SI prefixes become 24 values of a
  field, not 24 generated classes per base unit.
- **With polymorphism.** `Unit(name="mm")` or `Unit(base="meter", prefix="milli")`
  can dispatch to a named unit for ergonomics, but the real simplification is the
  field-based representation; polymorphism is a nicety on top.
- **Caveat.** Name parsing (`"mm"`, `"µm"`, `"micron"`) is a genuine feature that
  must survive the rewrite — as a converter/factory on the `name`/`base`/`prefix`
  fields, not as bespoke `__new__` logic. `magic`'s converter-from-type path is
  the right home for it.
- **Value:** high. The code self-identifies as a maintenance liability, and the
  representation change is well understood.

## 7. Honest call

Worth doing, in this order and with this scoping:

1. **Units** — highest value, clearest rewrite, self-flagged as fiddly. A
   field-based `Unit` with computed `scale` and a name converter, plus
   singletons. Polymorphism optional.
2. **Coordinate systems** — collapse the combinatorial classes to
   singletons over a few bases; add `polymorphic=True` for reader-side
   construct-from-`axes`. Do this *after* the adaptor (#10) lands, because the
   adaptor is the main consumer of system identity and will show which
   distinctions must remain types rather than field values.
3. **Axes** — smallest win; fold in with the coordinate-system pass since the
   two share the orientation vocabulary.

Not recommended: adopting `polymorphic` as the headline fix. It is the right
tool for the constructor boundary, but the proliferation is a representation
choice (values as types, products as multiple inheritance), and the durable
simplification is to change that representation. Polymorphism then makes the
retained named types reachable by value.

## 8. Risks and checks

- **`isinstance` call sites.** Before retiring any subclass, grep for
  `isinstance(x, LeftToRightAxis)` / `RASCoordinateSystem` and friends; each
  must become a field-value test. The audit above assumes the adaptor's
  field-based matching is the only consumer — verify per class.
- **`HiddenConst` semantics.** Moving a `HiddenConst` field to a plain field
  changes whether it is settable and whether it appears in `__init__`; confirm
  the singleton construction and any frozen-ness still hold.
- **3.8 runtime.** The unit rewrite must keep the wide-Python rules (no PEP
  604/585 in values; typing via `typing_extensions`).
- **Prior art.** Per the bagof-magic house rule, check how `pint` (units) and
  `pydantic`/`attrs` (discriminated construction) shape these before finalising
  the unit and polymorphic APIs.

## 9. Suggested process

This memo answers the design question at the level #54 asked for. The
implementation is three separate PRs (units; coordinate systems; axes), each
correctness-sensitive enough to warrant an Opus implementation and a Fable
review, and each gated on the checks in §8. The coordinate-system and axis
passes should follow the adaptor so the "which distinctions must stay types"
question is answered by a real consumer rather than guessed.
