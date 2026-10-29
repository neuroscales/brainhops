# Batched transformations

Status: proposed, and not implemented. The rules that the operations
of Part A must follow for a batched transformation are recorded in
section A.10 of `compute-api-and-open-systems.md`. This memo
proposes how a batched transformation is represented, and it lists what
has to change in the data model, the compute machinery, the resampler,
the readers and writers, and the command line (#576).

Some transformations are a family of transformations indexed by an
axis. Motion correction, for example, estimates one spatial affine per
volume of a time series, and stores them as a stack of matrices of shape
`(N, 4, 4)`. A displacement field can also be stored for each volume,
with shape `(N, X, Y, Z, 3)`. brainhops cannot represent such a family
today, so the readers of the formats that store one either refuse it or
read a single element of it.

---

## 1. What a batched transformation is

A batch axis is an axis that a transformation carries through
unchanged, while its other outputs may depend on it. With one affine
per volume, the spatial outputs are `A(t) x + b(t)` and the time output
is `t`, so time is a batch axis. A batched transformation is a
transformation with at least one batch axis, whose restriction to one
value of the batch axes is an ordinary transformation, called an
element of the batch.

Two more properties follow from the formats that store batches.

- A batch is discrete. Each element sits at one position along the
  batch axes, such as the time of one volume, and the transformation
  is not defined between two positions. A transformation whose outputs
  vary continuously with time, such as a spatio-temporal deformation
  sampled at a few times and meant to be interpolated, is not a batch.
  It is a field over a grid that includes time.
- All the elements have the same numbers of input and output axes,
  but they need not have the same type. What makes a family a batch is
  the axis that selects its elements. A file that stores several
  transformations side by side without saying which axis selects them,
  such as an ITK file with several blocks and no composite header,
  holds a list of transformations, not a batch, and its readers keep
  returning one of them, selected by `position=`.

A batch may have several batch axes. BigStitcher, for example, stores
one registration per time point and per view, although only the time
points of one view belong to the same image (section 7).

## 2. Why the batch is not a leading dimension of the existing types

The simplest representation would let the data of each existing type
carry leading batch dimensions, so that an `Affine` would hold a matrix
of shape `(*batch, No, Ni+1)` and a `DisplacementField` a field of shape
`(*batch, *shape, ndim)`. This representation is rejected, for three
reasons.

The first reason is that the kind of a transformation is decided by its
class. `@kinds.Affine` registers the class `Affine`, so
`isinstance(t, kinds.Affine)` is true for every instance, whereas a
batched affine is not an affine map of its inputs `(t, x)`. The checks
of `compute` would admit it into a run of affines, `factor` would read
it as a plain affine and cut it without error, and the adaptors would
wrap a 3D batched affine in a `SubspaceTransformation` over `x`, `y` and
`z` that passes `t` through, so that the spatial outputs no longer read
`t`. This is the silent failure that A.10 forbids.

The second reason is that nothing checks the shape of the data against
the coordinate systems, and the code reads the shape in many places
that assume a single element. Some of these places raise on a
stacked array, such as the unpacking `rows, cols = matrix.shape` in the
kind checkers and in `axis_counts`. Others return a wrong result
without an error. `matrix[:, :-1]` takes a row instead of a column of a
stacked matrix, `.T` reverses every axis, `len(t.translation)` counts
the elements of the batch instead of the axes, and `axis_counts` counts
the batch dimensions of a coordinate field as input axes. The
composition of a stacked field with stacked coordinates in `pull_field`
samples every element at the coordinates of every other element, which
gives an output of shape `(B, B, ...)`.

The third reason is that the axes of the batch must be named. A stack of
matrices does not say which axis of the space selects an element, nor at
which positions along that axis the elements sit.

## 3. Proposed representation

A batched transformation is a new class, `BatchedTransformation`. It
holds one transformation per element, arranged on a grid whose
dimensions are the batch axes, and the positions of the elements along
these axes. The class has two forms, which share the same axes,
positions and rules.

- The general form holds an array of transformations, one per element.
  The elements may be of any type, such as affines, fields, sequences
  or lazy inverses, and they may differ from one element to the next,
  as long as they all have the same numbers of input and output axes.
- The stacked form holds elements of one matrix, vector or field type,
  with one encoding, and it stores their parameters in one array whose
  leading dimensions index the elements. Its typed subclasses, such as
  `BatchedAffine`, compute on the whole stack at once.

```python
# general form: an array of arbitrary transformations
BatchedTransformation(
    [chain_0, chain_1, chain_2],  # one transformation per element
    input_axes=["x", "y", "z"],  # every other axis is a batch axis
    positions=None,  # element k sits at index k
    input=...,  # (x, y, z, t)
    output=...,  # (x, y, z, t)
)

# stacked form: the parameters of all the elements in one array
Affine.batch(matrices, input_axes=["x", "y", "z"])  # (*batch, 3, 4)
```

The array of elements of the general form may be a nested list or a
NumPy array of objects, and its shape is the shape of the batch. The
data of the stacked form has the shape of the data of one element
preceded by one dimension per batch axis, and the keywords that encode
an element, such as `degree`, `bound`, `store` and `log`, are shared by
all the elements. `positions` gives, for each batch axis, the
coordinate of each element along that axis. Its default is the index of
the element, which is what a file that knows nothing about the timing
of the volumes can give.

The two forms are needed for different reasons. Some families are made
of transformations that one array cannot hold. BigStitcher, for
example, stores a chain of affines for each time point, and the chains
can differ in length when a time point receives an extra correction. A
batch of fields with different grids, or of lazy inverses, is in the
same situation. Other families are made of many small matrices, such as
the hundreds of affines of a motion correction, and the stacked form
computes their products and inverses with one call on the whole stack
instead of one call per element.

### Which axes are batch axes

`input_axes` and `output_axes` give the axes of the elements, by position
or by name (A.9), with the same meaning as for `SubspaceTransformation`.
Every other axis is a batch axis, and the batch axes are taken in the
same order on both sides and in the order of the dimensions of the
batch. A batched transformation is therefore a subspace transformation
whose inner transformation changes along the axes that it passes
through. A batch axis may sit at different positions in the input and
output systems, as an axis that a subspace transformation passes
through may. `output_axes` defaults to `input_axes` when only
`input_axes` is given.

When `input_axes` and `output_axes` are not given, the batch axes come
first and the axes of the elements follow, so that the space is
`(*batch_axes, *element_axes)`. This is the order of the dimensions of
the stacked data and the order that broadcasting uses. A batch of motion
parameters for a series `(x, y, z, t)`, whose time axis is last, names
its spatial axes, as in the examples above.

The number of batch axes is the number of axes of the space minus the
number of axes of the elements. The general form reads it from the
shape of its array of elements. The stacked form reads it from its data
for the types whose data has a fixed number of dimensions. The matrix
of an affine has two dimensions, so the data of a batch of affines has
two dimensions more than it has batch axes. The data of a field has one
dimension per axis of its grid and one for its components, so the shape
`(*batch, X, Y, Z, 3)` alone does not tell how many of its leading
dimensions are batch dimensions (#588). `input_axes` tells it, because
the grid of each element has one dimension per input axis of the
elements. The output system tells it too, because the output space has
the components of the elements and the batch axes. A stacked batch of
fields is refused when neither `input_axes` nor the output system gives
that number.

Taking the discrete axes of the systems as batch axes was rejected,
because the NIfTI reader never marks time as discrete, whereas a
channel axis is discrete without being a batch axis.

### Elements

`b[k]` returns element `k` along the first batch axis, and `b[k, v]`
indexes several batch axes. A slice, such as `b[2:5]`, returns a smaller
batch whose positions are sliced in the same way, and an integer
removes its batch axis from the space. Iterating over `b` gives its
elements one at a time. Element `k` is the same map as
`b.restrict(element_axes, element_axes, at={"t": positions[k]})`, and
the restrictor of section 4 returns it.

The general form returns the transformations that it holds. The stacked
form creates an element only when one is asked for, as an ordinary
transformation of the element type whose data is a view of the stack.
When the stack is a lazy array of a file, indexing it keeps it lazy, so
an element reads only its own part of the file.

### One class, with stacked subclasses

`BatchedTransformation` is the general form. It is polymorphic, and the
constructor builds a stacked subclass when it is given an element type
and stacked data instead of an array of elements, in the way that
`Inverse` builds an `InverseAffine` for the inverse of an `Affine`. The
stacked subclasses, such as `BatchedAffine` and
`BatchedDisplacementField`, give the views of the stack that the element
type has, such as `matrix`, `homogeneous_matrix`, `field` and
`coefficients`, and they let the composers and the other dispatched
functions register a rule that computes on the stack. Unlike
`InverseAffine`, which is also an `Affine`, a `BatchedAffine` does not
inherit from `Affine`.

A rule written for the general form applies to every batch, because it
reads the elements one at a time. A stacked subclass overrides a rule
only to compute it on the whole stack, so that the two forms always
give the same result.

No form is registered as any of the kinds of `kinds.py`. A batched
affine is therefore never mistaken for an affine, and every rule that
applies to it is written for it. The ordinary types, in turn, refuse
data with leading dimensions in their constructor, so that a stacked
array never enters an `Affine` by accident.

### Building a batch

Every matrix, vector and field type has a class method `batch`, which
builds a stacked batch of its own type from a stacked array without
creating any element. It takes the stack, or one of the views of the
class, such as `homogeneous_matrix`, `field` or `coefficients`, together
with `input_axes`, `output_axes` and `positions`. The encoding of the
elements, such as `degree`, `bound`, `store` and `log`, and their
systems are given as the other keywords of the constructor of the class,
whose defaults apply to the keywords that are not given.

```python
Affine.batch(matrices)  # space (*batch_axes, x, y, z)
Affine.batch(homogeneous_matrix=matrices, input_axes=[0, 1, 2])
DisplacementField.batch(fields, input_axes=[0, 1, 2], degree=3)
```

The systems that `batch` receives are those of the whole space, as for
the constructor of `BatchedTransformation`. With data that has one
batch dimension, the last two examples describe a space of four axes
whose last axis is the batch axis, such as a series `(x, y, z, t)`.

Since `batch` accepts the views of the class, the conversions between a
view and the stored data must work on stacks, which the first step of
section 9 provides. When `batch` is called on the class of a file
format, the element type is the data-model class of that format, because
the metadata of a format describes one record of a file and not one
element of a batch. A class that holds no data, such as `Identity`,
refuses `batch`, since a batch of identities is the identity. The meta
transformations, whose parameters are other transformations, have no
`batch` either, and a batch of them is a general batch.

`BatchedTransformation.stack(elements, batch_axis=...)` builds a batch
from transformations that already exist, along a new batch axis that
comes before the axes of the elements. The elements must have the same
numbers of axes and, when they have any, the same systems. The elements
do not know the batch axis, so `batch_axis` gives it, and an axis about
which nothing is known is added when it is not given. The result is a
general batch, which holds the elements as they are. It suits a format
that stores each element on its own, such as the directory of one file
per volume that MCFLIRT writes or the registrations of BigStitcher,
whereas a format that stores a stack passes its array to `batch`.

A general batch whose elements share one matrix, vector or field type
and one encoding can be converted to the stacked form, which copies
their data into one array. `simplify` makes this conversion, because
the stacked form is the cheaper one to compute with (section 4).

A stacked batch of fields covers a coordinate field of shape
`(*batch, *shape, n_out)`, which maps the batch axes and the axes of its
grid to the batch axes and `n_out` other axes. Its elements have `n_out`
components on a grid of dimension `len(shape)`, so the case `n_out !=
len(shape)` waits for #588, as it does for a field that is not batched.

### Values between and outside the positions

A point whose batch coordinate is not one of the positions has no
element, and its outputs are NaN. The same holds outside the range of
the positions. A coordinate is matched to a position up to a tolerance
relative to the smallest spacing between positions, so that a time
computed as `t0 + k * TR` in floating point still selects volume `k`.
NaN is preferred to the nearest element, which would silently choose a
transformation that was not estimated for that point, and to an error,
which would prevent the application of a batch to a grid that only
partly overlaps it. The resampler never asks for a value between two
positions, because it treats a batch axis as discrete (section 5).

### Positions and the time axis of an image

A motion-correction file gives one matrix per volume, and the image
whose volumes they correct gives the time of each volume. When the time
axis of the image is in seconds and the positions of the batch are
volume indices, the batch must be told where its elements sit before it
is applied. Two sources are proposed. A reader that is given the moving
image, as the FLIRT reader already is through `moving:` and as a reader
of MCFLIRT would be, sets the positions from the time axis of that
image. Otherwise, applying a batch to an image (`image(batch)`) maps the
positions onto the axis of the image with the same name, when the number
of elements equals the number of samples along that axis, and raises
otherwise. The second rule is an open question (section 10).

## 4. Rules in the compute machinery

Each rule below is registered through the dispatched functions of the
compute machinery: the pattern readers and block restrictors of
`compute/prune.py`, and the simplifiers, converters and composers.

### Dependency pattern

The pattern reader of a batch says that each output of the elements
reads the inputs that it reads in some element, and also every batch
axis, and that each batch output reads only its own input. This is the
"reports that dependence" of A.10, decision 27. The general form reads
the pattern of each element and takes their union. A stacked batch
reads the pattern of its element type once under the analytic policy,
and the exact zeros of the whole stack under the numeric policy. Under
the numeric policy, an output reads a batch axis only when the elements
are not all equal along that axis.

### Restriction

The restrictor of a batch handles two cases. When a batch axis is fixed
by `at`, it selects the elements at that position, which gives a batch
with one batch axis fewer, or a single element when no batch axis
remains. When the block that is kept contains every batch axis, it cuts
each element down to the block. The general form calls the restrictor of
each element, and declines when one of them declines. A stacked batch
indexes the last dimensions of its stack. A kept output that reads a
free batch axis raises at `compute`, as A.10 requires, because no single
map exists.

### Simplification

`simplify` removes a batch axis of length 1, and, under the numeric
policy, a batch axis along which the elements are all equal. A batch
with no batch axis left becomes its single element, wrapped in a
`SubspaceTransformation` when the space has other axes. `simplify` also
simplifies each element of a general batch, and converts a general
batch whose elements share one matrix, vector or field type and one
encoding to the stacked form. The cancellation of a transformation
against its own lazy inverse applies to batches as to any other
transformation.

### Inverse, square root, logarithm and exponential

`b.inverse()` is the batch of the inverses of the elements, with the
same batch axes and positions, and the same holds for `sqrt` and for the
conversions to and from the tangent classes. The general form holds the
inverse of each element, which may be a lazy inverse, and a stacked
batch computes the inverses on its stack. The functions of
`_core/affines.py` already accept stacks of matrices, except
`to_homogeneous`, whose padding row has no batch dimensions. That
function is used by `expm`, `logm` and `sqrtm`. The checks of
`_core/linalg.py` that require a square matrix also test
`matrix.ndim == 2`, and must test the last two dimensions instead.
`sqrtm`, `expm` and `logm` call SciPy, which accepts a stack only in
recent releases, so they apply to each matrix of a stack in turn. The
inversion of fields in `_ext/invfield` takes a single field, so the
inverse of a batch of fields loops over the elements.

### Composition

Four composers are proposed.

1. Two batches with the same batch axes compose element by element. A
   batch axis that only one of them has, or that has length 1 in one of
   them, is broadcast, as in NumPy. Their positions along a shared axis
   must be equal.
2. A batch composed with a transformation that neither reads nor writes
   the batch axes gives a batch, whose elements are the composition of
   each element with that transformation. A spatial affine composed with
   a batch of affines therefore gives one stack of matrices, computed by
   a single broadcast matrix product.
3. A batch applied to a `CoordinateField` reads the batch coordinate of
   each point, selects the element at that position, and applies it,
   which gives a `CoordinateField`. This is how a batch is evaluated at
   points, and the monolithic path of the resampler uses it.
4. A batch that follows a projection creating one of its batch axes at
   a value gives the elements at that value, as the restrictor does.
   The pruning leaves such a projection just before the element that
   reads the created axis, so this composer is how a composing
   `compute` selects an element (section 5).

On the general form, rules 1 and 2 compose the elements one at a time
with `compose`, and the result is a general batch, which `simplify` may
convert to the stacked form. Rule 3 groups the points by the element
that they select, and applies each element to its group. On a stacked
batch, rules 1 and 2 compute on the stacked arrays instead, with the
functions of `_core/affines.py` and `_core/linalg.py`, which accept
stacks of matrices. The element type of the result is the type that the
composition of two elements would have, such as a scaling for two
batches of scalings. The composers of the ordinary types are not used on
a stack, because they receive transformations, and an ordinary
transformation never holds a stack. They index the first dimensions of a
matrix (`matrix[:, :-1]`, `matrix.T`, `matrix.shape[0]`), which is
correct as long as their data has a single element. A composition that
reads or writes a batch axis in any other way, such as a scaling of the
time axis, has no composer, and the batch stays in the chain.

### Kinds and modes

`is_kind(b, K)` is false for every kind finer than `Transformation`. A
separate test, `b.elements_are(K)`, says whether every element is of
kind `K`. `compute(mode=Affine)` does not admit a batch into a run of
affines, so a chain that contains one is composed on each side of it
and keeps it as a separate element.

## 5. Resampling

The reslice planner in `compute/separable.py` reads the normal form of
`factor` and turns each group of axes into a gather, a one-dimensional
weight matrix or a pull. A batch couples its batch axis with the axes
of its elements, so the factor pass puts `t`, `x`, `y` and `z` in one
group. `_check_discrete` refuses such a group when `t` is marked
discrete, and the group is interpolated in four dimensions when it is
not.

A new kind of step is proposed for a group that contains a batch axis.
The planner loops over the samples of the output grid along that axis.
For each sample `k`, it restricts the chain with `at` set to `k` on the
grid axis, and plans the remaining group as it does today. The forward
pass of the pruning in `simplify` carries the value of the grid axis to
the first element that reads it, the voxel-to-world map of the image,
and a composing `compute` folds the value into that element by a matrix
product. The time output of the product is then a constant row, which
the numeric policy turns into a created axis, at the time of volume `k`.
When the created axis reaches the batch, the composer of a batch with a
projection that creates one of its batch axes selects the element at
that time, so that the batch reduces to one element. The output time
axis then reads no input, and the end `κ` of the normal form selects
one volume of the input data. This is the reduction that A.3 already
describes for one volume of a series.

The loop costs one plan per volume. A vectorised version would need a
pull whose coordinates differ along the batch dimensions. `pull` in
`_core/bsplines.py` loops over the leading dimensions of its input in
Python and applies the same coordinates to each of them, so that loop
could take one set of coordinates per item instead. This is an
optimisation for later.

The batch axes are discrete whatever the image says about its own time
axis. The NIfTI reader never marks time as discrete, and the planner
reads the `discrete` flag only on the systems of the grid and of the
data. The planner must also read the batches of the chain. When the
plan falls back to the monolithic pull, because the normal form has a
single group, `_check_discrete` is not called at all. The composers
refuse to sample a field along a discrete axis, but nothing refuses an
affine that couples a discrete axis with the others, so such a group is
interpolated without an error. This gap exists today and is fixed by
the same change.

The re-expression of vectors and tensors in `datamodel/_frames.py` needs
the Jacobian of the transformation. For an affine frame, `jacobian`
returns one matrix, which `_structure` reads as `affine.matrix[:, :-1]`.
For a batch of affines, it must return one matrix per element, of shape
`(*batch, n, n)`, placed along the batch axes of the data. The methods
`ComponentAxis.apply` and `TensorAxis.apply` already accept matrices of
shape `(*batch, n, n)` that broadcast against the values, so only
`_structure`, `frame_axes` and the affine branch of `jacobian` change.

## 6. Readers and writers

The formats below store batches. The column "Today" describes the
current reader.

| Format                            | How the batch is stored                    | Today                                              | Proposed                                               |
| --------------------------------- | ------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------ |
| X5                                | `ArrayLength = N`, `Transform` `(N, 4, 4)` | `ParserNotImplementedError`                        | batch of affines, written back with `ArrayLength`      |
| AFNI `.aff12.1D`                  | one row of 12 numbers per volume           | one volume, chosen by `volume=` (default 0)        | the whole batch when `volume` is not given             |
| FSL MCFLIRT                       | directory of `MAT_0000`, `MAT_0001`, ...   | not readable: no reader accepts a directory        | new reader of the directory                            |
| SPM realignment                   | `rp_*.txt` `(N, 6)`; `.mat` `(4, 4, N)`    | no reader                                          | new reader of `rp_*.txt`                               |
| elastix                           | `*StackTransform`                          | `ParserNotImplementedError`                        | batch of the inner transform                           |
| BigStitcher XML                   | one registration per time point and view   | one view, chosen by `setup=` and `timepoint=`      | general batch over time points for one setup           |
| NIfTI fields, SPM `y_`, ITK       | `(X, Y, Z, T, 3)`                          | `ParserContentError` when `T > 1`                  | batch of fields along `T`                              |
| OME-NGFF `coordinates`, `displacements` | array with an extra discrete axis    | the extra axis is read as a grid axis              | batch when the axis is discrete and is not an output   |

The AFNI reader reads the first volume by default today. Reading the
whole batch when `volume` is not given changes what `load` returns for
a file with several rows, so it needs a decision (section 10). The
directory of MCFLIRT also needs support in the base of the readers,
since `ArrayReader.sniff_filename` rejects a directory. SPM stores in
its realignment `.mat` file the voxel-to-world matrix of each volume,
which is the geometry of the image rather than a transformation, so it
belongs to the image readers (section 7).

The shared helpers in `io/transformations/base/` assume a single matrix
or a single field. `matrix_to_model` in `_records.py` drops the last row
with `[:-1]`, which drops the last matrix of a stack instead, and the
private copies of the record helpers in the MINC, c3d, BigStitcher,
TrakEM2 and FLIRT modules also assume a single matrix.
`_conversions.py`, `fields.py` and `_splines.py` read the grid of a
field as `data.shape[:-1]`. A reader of a batch therefore converts each
element with these helpers and stacks the results, instead of passing
them a stack.

OME-NGFF 0.6 has no type for a batch of affines. The writer refuses one,
and writes a batch of fields as a `coordinates` or `displacements` array
with the batch axis as an axis of its grid, inside a `byDimension` whose
other child is the identity on the batch axis. This requires the
multi-block `SubspaceTransformation` of A.8.

## 7. Images

The proposal concerns transformations, but two image formats carry one
transformation per volume. SPM writes the voxel-to-world matrix of each
volume of a series into a `.mat` file next to the image, and BigStitcher
stores one registration per time point. The geometry of such an image
is a chain that contains a batch over its time axis, which the
proposal supports without a new image type. Reading it is left to the
image readers, after the transformations work.

## 8. Command line

`brainhops reslice` applies each `-t` in turn and then reslices onto
the reference. Applying motion parameters to a series needs the
following changes.

- `-t` accepts a directory or a format that holds a batch, through the
  readers of section 6.
- The batch axis of the transformation is matched to the axis of the
  image with the same name, and an option names the image axis when the
  names differ.
- The output keeps the time axis of the input when the reference is 3D.
  The output grid is built from the reference alone today, and no test
  covers a 4D input resliced onto a 3D reference.
- `--clip` takes one pair per axis of the input data, time included,
  and `--degree` and `--bound` apply to every axis. These options should
  not apply along a batch axis, which is never interpolated.

`brainhops convert` writes a batch to the formats that accept one.

## 9. Order of the work

The work depends on Part A of `compute-api-and-open-systems.md`. The
selection of an element and the resampling loop use `Projection.at` and
the pruning in `simplify`, which are implemented, and
`Transformation.restrict`, which is not. Batches of fields whose number
of components differs from the dimension of their grid need #588.

1. Make the matrix functions of `_core` accept stacks, without changing
   their behaviour on a single matrix: `to_homogeneous` broadcasts its
   padding row, the square checks of `_core/linalg.py` read the last two
   dimensions, and `sqrtm`, `expm` and `logm` apply to each matrix of a
   stack. The ordinary matrix and vector types accept only data with
   the number of dimensions of one element (#608).
2. Add the data model: `BatchedTransformation` in its general form, its
   stacked subclasses for the matrix and vector types, the class method
   `batch` of these types, `stack`, indexing, the systems and `inverse`
   (#610).
3. Add the rules of section 4: the pattern reader, the restrictor, the
   simplifier and the four composers.
4. Add the resampling step of section 5, the check of discrete axes in
   the monolithic path, and the per-element Jacobian.
5. Add the stacked subclasses for fields, and the readers and writers of
   section 6, starting with X5, AFNI and MCFLIRT, which cover the
   motion-correction case.
6. Change the command line as in section 8.

Each step is tested against the element-by-element result: a batch of
`N` elements, applied or composed, must give the same values as each of
its `N` elements applied to the matching volume.

## 10. Open questions

1. Name of the class. `BatchedTransformation` follows
   `SubspaceTransformation`, and `TransformationStack` was suggested in
   #251.
2. Outputs at a batch coordinate that is not a position. This memo
   proposes NaN. The alternatives are the nearest element, which would
   suit a point slightly off its volume, and an error.
3. Whether the AFNI reader returns the whole batch by default, which
   changes the result of `load` for a file with several rows, or keeps
   `volume=0` and gains an option for the batch.
4. How the positions of a batch read from a file without an image meet
   the time axis of the image it is applied to. Section 3 proposes to
   match the axis by name and the elements to the samples by count.
5. Whether `positions` should accept irregular coordinates, as proposed,
   or only an origin and a spacing. BigStitcher numbers its time points
   with arbitrary integers, which a spacing cannot describe.
