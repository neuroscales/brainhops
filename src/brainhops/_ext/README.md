This folder contains modules that have a more general purpose than 
brainhops, and are used by it. They are intended to become their 
own packages one day (and then become external dependencies of brainhop).

While they are being developed and their API changing too often, they are 
kept here.

* `struct`: a library that provides similar tools as `dataclasses.@dataclass`,
  `attrs.@define`, or `pydantic.BaseModel`; with additional features.
  It is closer to pydantic in that it's default behaviour relies on inheritence
  rater than decorators (although a decorator is also available). However,
  its implementation relies heavily on ports from `dataclasses`.
  The main advantage for us is that it is ours so we have more flexibility
  when it comes to adding features we think we need for our models, and
  we can ensure backward compatibility to python versions of our choice.

* `invfield`: a compact implementation of John Ashburner's displacement 
  field inversion. It is an independant implementation based on his paper,
  not a port of the SPM implementation.

* `npfileobj`: a class that implements "lazy" array-like semantics for 
  array data that is stored contiguously on arbitrary file systems
  (not necessary local ones). It is based in parts on `nibabel.ArrayProxy`
  (except that it does not load data on `__getitem__`, but instead
  builds a "strided" proxy). It's in very early WIP, and will involve
  copying lots of nibabel code. 
  
  It might not even be useful (can we just wrap a `nibabel.ArrayProxy` 
  in a `dask.array`? Would chunking work correctly?)