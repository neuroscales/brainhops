"""Operations on transformations.

Each operation is split into a dispatcher and a module of registered
implementations.

| Operation  | Dispatcher    | Implementations |
| ---------- | ------------- | --------------- |
| membership | `check`       | `checkers`      |
| conversion | `convert`     | `converters`    |
| conversion | `simplify`    | `simplifiers`   |
| product    | `compose`     | `composers`     |
| restriction| `restrict`    | `restrictors`   |

The `adaptors` module bridges consecutive transformations whose systems
differ by a reordering, rescaling or flip of shared axes. The `factor`
module rewrites a transformation into a normal form over groups of
axes, `separable` plans a resampling as passes along single axes, and
`utils` holds shared helpers. The implementation modules register
themselves on import, which the parent package performs.
"""
