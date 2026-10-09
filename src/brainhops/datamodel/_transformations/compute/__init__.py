"""The operations on transformations, and what implements them.

A transformation knows what it *is*; this subpackage knows what can be
*done* with one. Each operation comes as a pair of modules: the dispatcher
that defines it, and the implementations registered against it.

| Operation  | Dispatcher    | Implementations |
| ---------- | ------------- | --------------- |
| membership | `check`       | `checkers`      |
| conversion | `convert`     | `converters`    |
| conversion | `simplify`    | `simplifiers`   |
| product    | `compose`     | `composers`     |
| restriction| `restrict`    | `restrictors`   |

`adaptors` reconciles two consecutive transformations whose systems merely
reorder, rescale or flip their shared axes; `factor` rewrites a
transformation into its axis-group normal form; `separable` plans a
resampling as a sequence of per-axis passes; and `utils` holds what more
than one of them reads (axis counts, endpoints, the refusals).

Nothing here is imported for its own sake: the implementation modules
register themselves when they are imported, which `_transformations`
does, so importing the package is what makes the operations work.
"""
