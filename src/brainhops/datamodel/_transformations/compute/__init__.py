"""Operations on transformations.

Each operation is split into two modules: a dispatcher, which defines the
operation and chooses an implementation for its arguments, and a module of
implementations registered with that dispatcher.

| Operation  | Dispatcher    | Implementations |
| ---------- | ------------- | --------------- |
| membership | `check`       | `checkers`      |
| conversion | `convert`     | `converters`    |
| conversion | `simplify`    | `simplifiers`   |
| product    | `compose`     | `composers`     |
| restriction| `restrict`    | `restrictors`   |

The `adaptors` module bridges two consecutive transformations whose
coordinate systems differ only by a reordering, rescaling or flip of the
axes they share. The `factor` module rewrites a transformation into a
normal form built from groups of axes. The `separable` module plans a
resampling as a series of passes, each along a single axis, and the
`utils` module holds helpers that several of these modules use.

The implementation modules register their implementations when they are
imported. The parent package imports them, so importing the package is
enough to make every operation available.
"""
