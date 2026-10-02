"""
File formats whose chain has a fixed structure are immutable sequences.

An SPM `y_` field is `[ras2voxel, rasfield]`, a FNIRT warp is
`[ras2grid, field, grid2ras]`, an ITK NIfTI field is
`[lps2voxel, displacement|coordinates, voxel2lps]`, and an ITK block is
the chain of its named slots. Editing one in place would leave an object
whose class no longer says what it holds, so these formats inherit
[`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]:
the chain is a tuple and item assignment, deletion and insertion raise.

These tests pin that, and pin that nothing else changed: each format
still loads, rebuilds through `to(...)`, inverts, computes to the same
result as the plain sequence of its transformations, and -- where it has
a writer, which the ITK NIfTI fields and the ITK MATLAB blocks do --
round-trips through a file. An ITK file's list of blocks is
not fixed-structure, and stays editable.
"""

from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.io.transformations.fsl.fnirt import FnirtWarpField  # noqa: E402
from brainhops.io.transformations.itk import ItkTransform  # noqa: E402
from brainhops.io.transformations.itk._common import (  # noqa: E402
    ItkAffineBase,
    ItkDisplacementBase,
)
from brainhops.io.transformations.itk.mat import MatTransform  # noqa: E402
from brainhops.io.transformations.itk.nifti import (  # noqa: E402
    ItkNiftiCoordinatesField,
    ItkNiftiDisplacementField,
)
from brainhops.io.transformations.spm.y import (  # noqa: E402
    SpmCoordinatesField,
)

DATA = Path(__file__).parent / "data"
VECTOR = 1007  # NIFTI_INTENT_VECTOR

SHAPE = (4, 5, 6)
VOX2RAS = np.array(
    [
        [0.0, -3.0, 0.0, 10.0],
        [2.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _vectors() -> np.ndarray:
    """A smooth `(*SHAPE, 1, 3)` field, small enough to stay invertible."""
    grids = np.meshgrid(*map(np.arange, SHAPE), indexing="ij")
    vectors = np.stack(
        [0.1 * (d + 1) * g + d for d, g in enumerate(grids)], axis=-1
    )
    return vectors[:, :, :, None, :].astype("float32")


def _write_vector(path: Path, vectors: np.ndarray) -> Path:
    img = nb.Nifti1Image(vectors, VOX2RAS)
    img.header.set_intent(VECTOR)
    img.header.set_sform(VOX2RAS, code=1)
    img.header.set_qform(VOX2RAS, code=1)
    nb.save(img, str(path))
    return path


def _spm(tmp_path: Path) -> SpmCoordinatesField:
    # An SPM `y_` map is a field of absolute RAS coordinates.
    ijk = np.stack(np.meshgrid(*map(np.arange, SHAPE), indexing="ij"), -1)
    ras = ijk @ VOX2RAS[:3, :3].T + VOX2RAS[:3, 3]
    ras = ras[:, :, :, None, :] + _vectors()
    return io.transformations.load(
        _write_vector(tmp_path / "y_sub01.nii.gz", ras.astype("float32"))
    )


def _fnirt(kind: str) -> FnirtWarpField:
    return io.transformations.load(
        DATA / "fsl" / f"{kind}.nii.gz",
        moving=nb.load(str(DATA / "fsl" / "src.nii.gz")),
        reference=nb.load(str(DATA / "fsl" / "ref.nii.gz")),
    )


def _itk_nifti(cls: type, tmp_path: Path) -> xforms.Sequence:
    return cls.from_file(_write_vector(tmp_path / "warp.nii.gz", _vectors()))


def _itk_block(name: str) -> xforms.Sequence:
    if name.endswith(".h5"):
        pytest.importorskip("h5py")
    return io.transformations.load(DATA / name)[0]


FORMATS = {
    "spm": (SpmCoordinatesField, _spm),
    "fnirt-dense": (FnirtWarpField, lambda _: _fnirt("displacementfield")),
    "fnirt-coeff": (FnirtWarpField, lambda _: _fnirt("coefficientfield")),
    "itk-nifti-displacement": (
        ItkNiftiDisplacementField,
        lambda tmp: _itk_nifti(ItkNiftiDisplacementField, tmp),
    ),
    "itk-nifti-coordinates": (
        ItkNiftiCoordinatesField,
        lambda tmp: _itk_nifti(ItkNiftiCoordinatesField, tmp),
    ),
    "itk-block-affine": (
        ItkAffineBase,
        lambda _: _itk_block("itk_affine3d.tfm"),
    ),
    "itk-block-mat": (
        ItkAffineBase,
        lambda _: _itk_block("itk_affine3d_0GenericAffine.mat"),
    ),
    "itk-block-displacement": (
        ItkDisplacementBase,
        lambda _: _itk_block("itk_displacement3d.h5"),
    ),
    "itk-block-bspline": (
        ItkDisplacementBase,
        lambda _: _itk_block("itk_bspline3d.tfm"),
    ),
}


@pytest.fixture(params=sorted(FORMATS))
def loaded(request, tmp_path) -> tuple:  # noqa: ANN001
    """A `(name, expected class, loaded object)` triple, per format."""
    cls, load = FORMATS[request.param]
    return request.param, cls, load(tmp_path)


REGIONS = ["around", "before", "after"]
"""Where the fields are sampled, relative to their grid; see `_points`."""


def _points(name: str, region: str = "around") -> np.ndarray:
    """A few points in the space a format maps from, in and around its grid.

    The fields written here are sampled outside of their grid as well as
    inside it: `"around"` runs from before the start to past the end of
    every axis, and `"before"` and `"after"` lie entirely beyond one end
    of every axis -- the case that once sampled an empty crop of a dask
    field and read whatever memory followed it. The FNIRT fixtures and
    the ITK blocks are sampled on a fixed grid of world points.
    """
    if not name.startswith(("spm", "itk-nifti")):
        grids = np.meshgrid(*[np.linspace(-20.0, 20.0, 3)] * 3, indexing="ij")
        return np.stack(grids, -1)
    if region == "before":
        axes = [np.linspace(-10.0, -2.5, 3) for _ in SHAPE]
    elif region == "after":
        axes = [np.linspace(size + 1.5, size + 9.0, 3) for size in SHAPE]
    else:
        axes = [np.linspace(-4.0, size + 3.0, 5) for size in SHAPE]
    ijk = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    ras = ijk @ VOX2RAS[:3, :3].T + VOX2RAS[:3, 3]
    if name == "spm":
        return ras
    return ras * np.array([-1.0, -1.0, 1.0])  # ITK maps LPS


def _kinds(chain: tx.Iterable[xforms.Transformation]) -> list:
    # A derived chain (SPM) is rebuilt on every access, and its elements
    # hold arrays, so chains are compared by the kinds of their elements.
    return [type(t) for t in chain]


def _apply(xform: xforms.Transformation, points: np.ndarray) -> np.ndarray:
    # The points are declared in the space the transformation maps from,
    # so that no adaptor has to guess how their axes line up with it.
    points = xforms.CoordinatesField(field=points, output=xform.input)
    out = xforms.Sequence([points, xform])
    return np.asarray(out.compute().to(xforms.CoordinatesField).field)


# ----------------------------------------------------------------------
#   IMMUTABILITY
# ----------------------------------------------------------------------


def test_the_format_is_an_immutable_sequence(loaded) -> None:  # noqa: ANN001
    _, cls, obj = loaded
    assert isinstance(obj, cls)
    assert isinstance(obj, xforms.ImmutableSequence)
    assert isinstance(obj.transformations, tuple)
    assert len(obj) > 0


def test_in_place_edits_are_refused(loaded) -> None:  # noqa: ANN001
    _, _, obj = loaded
    kinds = _kinds(obj)
    with pytest.raises(TypeError, match="cannot be edited in place"):
        del obj[0]
    with pytest.raises(TypeError, match="cannot be edited in place"):
        obj[0] = xforms.Identity()
    with pytest.raises(TypeError, match="cannot be edited in place"):
        obj.insert(0, xforms.Identity())
    with pytest.raises(AttributeError):
        obj.transformations.append(xforms.Identity())
    assert _kinds(obj) == kinds


def test_an_assigned_chain_is_frozen_into_a_tuple(loaded) -> None:  # noqa: ANN001
    _, _, obj = loaded
    chain = list(obj.transformations)
    obj.transformations = chain
    assert isinstance(obj.transformations, tuple)
    assert all(a is b for a, b in zip(obj.transformations, chain))
    chain.pop()
    assert len(obj) == len(chain) + 1


# ----------------------------------------------------------------------
#   NOTHING ELSE CHANGED
# ----------------------------------------------------------------------


def test_replace_rebuilds_a_new_object(loaded) -> None:  # noqa: ANN001
    _, cls, obj = loaded
    chain = list(obj.transformations)
    copy = obj.to(transformations=chain[::-1])
    assert isinstance(copy, cls)
    assert copy is not obj
    assert isinstance(copy.transformations, tuple)
    assert len(copy) == len(chain)
    assert all(a is b for a, b in zip(copy.transformations, chain[::-1]))
    assert _kinds(obj) == _kinds(chain)


def test_the_flattened_chain_keeps_the_format(loaded) -> None:  # noqa: ANN001
    _, cls, obj = loaded
    flat = obj._flattened()
    assert isinstance(flat, cls)
    assert isinstance(flat.transformations, tuple)
    assert len(flat) == len(obj)


@pytest.mark.parametrize("region", REGIONS)
def test_the_format_computes_like_its_plain_chain(loaded, region) -> None:  # noqa: ANN001
    name, _, obj = loaded
    plain = xforms.Sequence(
        list(obj.transformations), input=obj.input, output=obj.output
    )
    assert type(obj.compute()) is type(plain.compute())
    points = _points(name, region)
    mapped = _apply(obj, points)
    assert mapped.shape == points.shape
    assert np.all(np.isfinite(mapped))
    assert np.abs(mapped).max() < 1e4
    np.testing.assert_allclose(
        mapped, _apply(plain, points), rtol=1e-6, atol=1e-6
    )


def test_the_inverse_is_a_plain_sequence(loaded) -> None:  # noqa: ANN001
    _, _, obj = loaded
    inverse = obj.inverse()
    assert not isinstance(inverse, xforms.ImmutableSequence)
    assert len(inverse) == len(obj)
    assert inverse.input == obj.output
    assert inverse.output == obj.input


@pytest.mark.parametrize("region", REGIONS)
@pytest.mark.parametrize(
    "cls", [ItkNiftiDisplacementField, ItkNiftiCoordinatesField]
)
def test_an_itk_nifti_field_round_trips(tmp_path, cls, region) -> None:  # noqa: ANN001
    first = _itk_nifti(cls, tmp_path)
    out = tmp_path / "out.nii.gz"
    first.save(out)
    second = cls.from_file(out)
    assert isinstance(second.transformations, tuple)
    assert _kinds(second) == _kinds(first)
    points = _points("itk-nifti", region)
    np.testing.assert_allclose(
        _apply(second, points), _apply(first, points), rtol=1e-5, atol=1e-4
    )


def test_an_itk_mat_block_round_trips(tmp_path) -> None:  # noqa: ANN001
    first = _itk_block("itk_affine3d_0GenericAffine.mat")
    out = tmp_path / "out.mat"
    MatTransform([first]).save(out)
    (second,) = io.transformations.load(out)
    assert type(second) is type(first)
    assert isinstance(second.transformations, tuple)
    assert _kinds(second) == _kinds(first)
    np.testing.assert_array_equal(second.parameters, first.parameters)
    np.testing.assert_array_equal(
        second.fixed_parameters, first.fixed_parameters
    )
    with pytest.raises(TypeError, match="cannot be edited in place"):
        second[0] = xforms.Identity()


# ----------------------------------------------------------------------
#   NOT FIXED-STRUCTURE
# ----------------------------------------------------------------------


def test_an_itk_files_list_of_blocks_stays_editable() -> None:
    """A composite ITK file holds any number of blocks, in any order."""
    xform = io.transformations.load(DATA / "itk_composite_affine3d.tfm")
    assert isinstance(xform, ItkTransform)
    assert not isinstance(xform, xforms.ImmutableSequence)
    first, second = xform
    del xform[0]
    assert list(xform) == [second]
    xform.transformations = [second, first]
    assert list(xform) == [second, first]
