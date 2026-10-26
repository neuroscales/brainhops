"""
Conversions between the formats of dense fields.

One small file is written in each format of dense fields, from the
same synthetic field. Every ordered pair of formats is then tested: the
file of the source is loaded, converted to the target, saved, loaded
again, and the reloaded field must map the points inside the grid as
the source does. A pair that is inherently lossy, because one side
interpolates its field with cubic splines, is converted with
`lossy=True` and compared at the nodes of the grid, where both agree.
A pair that is refused on purpose is listed with its reason, so that a
missing converter fails the test.
"""

from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io.transformations as transformations
from brainhops import io
from brainhops.datamodel import transformations as xforms
from brainhops.datamodel._transformations.compute.adaptors import bridge
from brainhops.datamodel.systems import (
    CoordinateSystem,
    LPSmm,
    RASmm,
    VoxelCoordinateSystem,
)
from brainhops.errors import ConversionError, LossyConversionError
from brainhops.io.transformations.afni import AfniBrikWarp, AfniNiftiWarp
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.freesurfer.m3z import M3zMorph
from brainhops.io.transformations.itk.nifti import (
    ItkNiftiCoordinateField,
    ItkNiftiDisplacementField,
)
from brainhops.io.transformations.minc import XfmTransform
from brainhops.io.transformations.mrtrix import (
    MrtrixCoordinateField,
    MrtrixDisplacementField,
)
from brainhops.io.transformations.nifti import NiftiRASDisplacementField
from brainhops.io.transformations.niftyreg import (
    NiftyRegDeformationField,
    NiftyRegDisplacementField,
    NiftyRegVelocityField,
)
from brainhops.io.transformations.spm import SpmCoordinateField
from brainhops.io.transformations.x5 import X5Transform
from brainhops.io.transformations.zarr import OmeZarrField

SHAPE = (4, 5, 6)
"""The shape of the grid: small, and no two axes of the same length."""

VOX2RAS = np.array(
    [
        [0.0, -2.0, 0.0, 10.0],
        [1.5, 0.0, 0.0, -20.0],
        [0.0, 0.0, 2.5, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""A grid whose axes run along the world axes, permuted and flipped."""

VOXEL = VoxelCoordinateSystem()

FORMATS = {
    "nifti": (NiftiRASDisplacementField, "nifti.nii.gz", None),
    "spm": (SpmCoordinateField, "y_spm.nii.gz", None),
    "mrtrix.disp": (MrtrixDisplacementField, "disp.mif", None),
    "mrtrix.coord": (MrtrixCoordinateField, "coord.mif", None),
    "afni.brik": (AfniBrikWarp, "afni+orig.HEAD", None),
    "afni.nifti": (AfniNiftiWarp, "afni.nii.gz", None),
    "itk.disp": (ItkNiftiDisplacementField, "itk.nii.gz", "itk"),
    "itk.coord": (ItkNiftiCoordinateField, "itkc.nii.gz", "itk.coordinates"),
    "niftyreg.disp": (NiftyRegDisplacementField, "nrd.nii.gz", None),
    "niftyreg.def": (NiftyRegDeformationField, "nrf.nii.gz", None),
    "niftyreg.vel": (NiftyRegVelocityField, "nrv.nii.gz", None),
    "m3z": (M3zMorph, "morph.m3z", None),
    "x5": (X5Transform, "field.x5", None),
    "minc": (XfmTransform, "grid.xfm", None),
    "ome": (OmeZarrField, "field.ome.zarr", None),
}
"""Each format: its class, the name of its file, and the hint to load it."""

CUBIC = {"minc"}
"""The formats that interpolate their field with cubic splines."""

CUBIC_KEPT = {"x5"}
"""The formats that keep a field interpolated with cubic splines."""

VELOCITY = {"niftyreg.vel"}
"""The formats that store a stationary velocity."""

UNKNOWN = {"ome"}
"""The formats whose files do not say the world of their field."""


def _refusal(source: str, target: str) -> str:
    """The reason a pair is refused on purpose, or an empty string."""
    if target in VELOCITY and source not in VELOCITY:
        return "computes no logarithm"
    if target.startswith("itk.") and source in UNKNOWN:
        return "endpoints are not known"
    return ""


def _lossy(source: str, target: str) -> bool:
    """Whether a pair converts with a loss, which is opted into.

    A field interpolated with cubic splines loses when it is stored as
    linear values, and a linear field loses when it is stored as cubic
    splines. X5 keeps a cubic field as a B-spline, which is exact.
    """
    if source in CUBIC:
        return target not in CUBIC | CUBIC_KEPT
    return target in CUBIC


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _field(log: bool = False) -> xforms.Sequence:
    rng = np.random.default_rng(0)
    scale = 0.1 if log else 1.0
    data = (scale * rng.normal(size=(*SHAPE, 3))).astype(np.float32)
    return xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=np.linalg.inv(VOX2RAS)[:-1]),
            xforms.DisplacementField(
                data=data, input=VOXEL, output=VOXEL, log=log
            ),
            VoxelToRAS(matrix=VOX2RAS[:-1]),
        ]
    )


def _voxels_to_ras(voxels: np.ndarray) -> np.ndarray:
    return voxels @ VOX2RAS[:3, :3].T + VOX2RAS[:3, 3]


def _inside() -> np.ndarray:
    """RAS points inside the grid, on and between its nodes."""
    rng = np.random.default_rng(1)
    upper = np.asarray(SHAPE, dtype=float) - 1
    return _voxels_to_ras(rng.uniform(0, 1, size=(20, 3)) * upper)


def _nodes() -> np.ndarray:
    """RAS points at the nodes of the grid, away from its edges."""
    voxels = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 1.0, 1.0]])
    return _voxels_to_ras(voxels)


def _in_world(
    t: xforms.Transformation,
    world: CoordinateSystem,
    default: tx.Optional[CoordinateSystem] = None,
) -> xforms.Transformation:
    # The map of `t` read from and written in `world`. An endpoint that
    # is not known is taken to be `default`, or `world` without one, as a
    # converter takes it.
    known = default or world
    return xforms.Sequence(
        transformations=[
            bridge(
                world, t.input or known, allow_type_grouped_positional=True
            ),
            t,
            bridge(
                t.output or known, world, allow_type_grouped_positional=True
            ),
        ]
    )


def _apply(t: xforms.Transformation, points: np.ndarray) -> np.ndarray:
    points = xforms.CoordinateField(field=points[:, None, None, :])
    out = xforms.Sequence(transformations=[points, t]).compute()
    return np.asarray(out.to(xforms.CoordinateField).field)[:, 0, 0, :]


def _same_map(
    a: xforms.Transformation, b: xforms.Transformation, points: np.ndarray
) -> None:
    # The points are in RAS. Both maps are compared in the world of `a`,
    # or in that of `b` when the world of `a` is not known, and in RAS
    # when neither is known. A field between LPS spaces is flipped, and a
    # field `a` whose world is not known is taken to be in that world, as
    # a converter takes it. A field `b` whose world is not known was
    # written by a converter, which writes a field of unknown world, such
    # as an OME-Zarr field, in RAS.
    world = a.input or b.input or RASmm()
    # The points are given in RAS, and are moved into that world.
    to_world = bridge(RASmm(), world, allow_type_grouped_positional=True)
    points = _apply(to_world, points)
    np.testing.assert_allclose(
        _apply(_in_world(a, world), points),
        _apply(_in_world(b, world, RASmm()), points),
        rtol=0,
        atol=1e-4,
    )


def _convert(t: xforms.Transformation, cls: type, lossy: bool) -> tx.Any:
    if not lossy:
        return t.to(cls)
    with pytest.raises(LossyConversionError):
        t.to(cls)
    return t.to(cls, lossy=True)


def _load(path: str, hint: tx.Optional[str]) -> xforms.Transformation:
    return transformations.load(path, hint=hint)


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One file per format, written from the synthetic field."""
    folder = tmp_path_factory.mktemp("fields")
    paths = {}
    for name, (cls, filename, _) in FORMATS.items():
        source = _field(log=name in VELOCITY)
        converted = _convert(source, cls, lossy=name in CUBIC)
        path = str(folder / filename)
        io.save(converted, path)
        paths[name] = path
    return paths


# ----------------------------------------------------------------------
#   TESTS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(FORMATS))
def test_each_file_is_read_back_as_its_format(files: dict, name: str) -> None:
    cls, _, hint = FORMATS[name]
    loaded = _load(files[name], hint)
    assert isinstance(loaded, cls)
    points = _nodes() if name in CUBIC else _inside()
    _same_map(_field(log=name in VELOCITY), loaded, points)


@pytest.mark.parametrize("target", list(FORMATS))
@pytest.mark.parametrize("source", list(FORMATS))
def test_a_field_converts_between_formats(
    files: dict, tmp_path: Path, source: str, target: str
) -> None:
    source_cls, _, source_hint = FORMATS[source]
    target_cls, filename, target_hint = FORMATS[target]
    loaded = _load(files[source], source_hint)
    reason = _refusal(source, target)
    if reason:
        with pytest.raises(ConversionError, match=reason):
            loaded.to(target_cls, lossy=True)
        return
    converted = _convert(loaded, target_cls, _lossy(source, target))
    assert isinstance(converted, target_cls)
    path = str(tmp_path / filename)
    io.save(converted, path)
    back = _load(path, target_hint)
    assert isinstance(back, target_cls)
    points = _nodes() if _lossy(source, target) else _inside()
    _same_map(loaded, back, points)


@pytest.mark.parametrize("name", list(FORMATS))
def test_a_file_saved_to_a_new_name_stays_in_its_format(
    files: dict, tmp_path: Path, name: str
) -> None:
    # The format of the object is chosen first, whatever else the name
    # could be written as.
    cls, filename, hint = FORMATS[name]
    loaded = _load(files[name], hint)
    path = str(tmp_path / ("copy-" + filename))
    if filename.startswith("y_"):
        path = str(tmp_path / ("y_copy-" + filename[2:]))
    io.save(loaded, path)
    assert isinstance(_load(path, hint), cls)


# ----------------------------------------------------------------------
#   CHOOSING A FORMAT ON SAVE
# ----------------------------------------------------------------------


def _generic(
    world: tx.Optional[CoordinateSystem], kind: str = "displacement"
) -> tx.Any:
    field = _field()
    data = field.transformations[1].data
    if kind == "coordinates":
        middle = xforms.CoordinateField(
            field=data + _grid(), input=VOXEL, output=VOXEL
        )
    else:
        middle = xforms.DisplacementField(
            field=data, input=VOXEL, output=VOXEL
        )
    return xforms.Sequence(
        transformations=[
            xforms.Affine(
                matrix=np.linalg.inv(VOX2RAS)[:-1], input=world, output=VOXEL
            ),
            middle,
            xforms.Affine(matrix=VOX2RAS[:-1], input=VOXEL, output=world),
        ]
    )


def _grid() -> np.ndarray:
    axes = [np.arange(n, dtype=np.float32) for n in SHAPE]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


@pytest.mark.parametrize(
    "world, kind, cls",
    [
        (None, "displacement", NiftiRASDisplacementField),
        (RASmm(), "displacement", NiftiRASDisplacementField),
        (LPSmm(), "displacement", ItkNiftiDisplacementField),
        (None, "coordinates", NiftiRASDisplacementField),
        (RASmm(), "coordinates", NiftiRASDisplacementField),
        (LPSmm(), "coordinates", ItkNiftiCoordinateField),
    ],
    ids=["unknown", "ras", "lps", "unknown-coord", "ras-coord", "lps-coord"],
)
def test_save_chooses_one_nifti_field(
    tmp_path: Path, world: tx.Optional[CoordinateSystem], kind: str, cls: type
) -> None:
    t = _generic(world, kind)
    path = str(tmp_path / "warp.nii.gz")
    io.save(t, path)
    hint = {ItkNiftiCoordinateField: "itk.coordinates"}.get(cls)
    if cls is ItkNiftiDisplacementField:
        hint = "itk"
    back = _load(path, hint)
    assert isinstance(back, cls)
    _same_map(
        t if world is not None else _generic(RASmm(), kind), back, _inside()
    )


@pytest.mark.parametrize(
    "kind, cls",
    [
        ("displacement", MrtrixDisplacementField),
        ("coordinates", MrtrixCoordinateField),
    ],
)
def test_save_chooses_the_mrtrix_field_of_the_same_kind(
    tmp_path: Path, kind: str, cls: type
) -> None:
    path = str(tmp_path / "warp.mif")
    io.save(_generic(None, kind), path)
    assert isinstance(_load(path, None), cls)


def test_save_writes_a_velocity_to_mrtrix_as_coordinates(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "warp.mif")
    io.save(_field(log=True), path)
    assert isinstance(_load(path, None), MrtrixCoordinateField)


# ----------------------------------------------------------------------
#   CHAINS, GEOMETRIES AND OPTIONS
# ----------------------------------------------------------------------


def test_x5_holds_a_chain_of_fields_and_affines(tmp_path: Path) -> None:
    # Each field becomes one transform, and the affines between them are
    # carried into the affine that places the next field, or become a
    # linear transform after the last one. The chain is in LPS, which is
    # flipped into RAS.
    shift = xforms.Translation([1.0, -2.0, 3.0], input=LPSmm(), output=LPSmm())
    field = _field().to(ItkNiftiDisplacementField)
    chain = xforms.Sequence(transformations=[field, shift, field])
    x5 = chain.to(X5Transform)
    kinds = [type(element).__name__ for element in x5.transformations]
    assert kinds.count("X5DisplacementField") == 2
    path = str(tmp_path / "chain.x5")
    io.save(x5, path)
    _same_map(chain, _load(path, None), _inside())


def test_x5_keeps_a_cubic_bspline() -> None:
    field = _field()
    cubic = xforms.DisplacementField(
        field=field.transformations[1].data,
        input=VOXEL,
        output=VOXEL,
        degree=3,
        bound="zeros",
        store="coefficients",
    )
    chain = xforms.Sequence(
        transformations=[
            field.transformations[0],
            cubic,
            field.transformations[2],
        ],
        input=RASmm(),
    )
    x5 = chain.to(X5Transform)
    assert type(x5.transformations[0]).__name__ == "X5BSplineField"


def test_an_oblique_field_is_sampled_for_afni(tmp_path: Path) -> None:
    angle = 0.3
    rotation = np.eye(4)
    rotation[:2, :2] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    vox2ras = rotation @ VOX2RAS
    field = _field()
    oblique = xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=np.linalg.inv(vox2ras)[:-1]),
            field.transformations[1],
            VoxelToRAS(matrix=vox2ras[:-1]),
        ]
    )
    with pytest.raises(LossyConversionError, match="oblique"):
        oblique.to(AfniBrikWarp)
    with pytest.raises(LossyConversionError, match="oblique"):
        oblique.to(OmeZarrField)
    assert isinstance(oblique.to(AfniBrikWarp, lossy=True), AfniBrikWarp)


def test_m3z_takes_the_geometry_of_the_source_image(tmp_path: Path) -> None:
    source = np.diag([1.0, 1.0, 1.0, 1.0])
    source[:3, 3] = [-30.0, -40.0, -10.0]
    morph = _field().to(M3zMorph, moving=(source, (64, 64, 32)))
    path = str(tmp_path / "morph.m3z")
    io.save(morph, path)
    back = _load(path, None)
    record = back.metadata.raw
    np.testing.assert_allclose(record.image_geometry.vox2ras, source)
    assert record.image_geometry.shape == (64, 64, 32)
    _same_map(_field(), back, _inside())


def test_minc_writes_its_displacement_volume(tmp_path: Path) -> None:
    path = tmp_path / "warp.xfm"
    io.save(_field().to(XfmTransform, lossy=True), str(path))
    assert (tmp_path / "warp_grid_0.mnc").exists()
    assert "warp_grid_0.mnc" in path.read_text()


def test_a_dask_field_stays_lazy(files: dict) -> None:
    da = pytest.importorskip("dask.array")
    from brainhops._core.backends import backend

    with backend("dask"):
        loaded = _load(files["nifti"], None)
        for cls in (
            MrtrixCoordinateField,
            ItkNiftiDisplacementField,
            NiftyRegDeformationField,
            OmeZarrField,
        ):
            converted = loaded.to(cls)
            if cls is OmeZarrField:
                data = converted.raw_levels[0]
            else:
                data = converted.transformations[1].data
            assert isinstance(data, da.Array), cls.__name__


@pytest.mark.parametrize("name", ["warp.h5", "warp.tfm"])
def test_a_dense_field_is_written_as_an_itk_displacement_field(
    tmp_path,  # noqa: ANN001
    name: str,
) -> None:
    # A text or HDF5 ITK file holds a DisplacementFieldTransform, which
    # stores the displacement of each node in LPS millimetres.
    pytest.importorskip("h5py")
    from test_io_convert_formats import (
        VOX2WORLD,
        _apply,
        _ras_displacement_chain,
        _voxels,
    )

    chain = _ras_displacement_chain()
    io.save(chain, tmp_path / name)
    back = io.transformations.load(tmp_path / name)
    inside = _voxels()[:6]
    ras = inside @ VOX2WORLD[:3, :3].T + VOX2WORLD[:3, 3]
    flip = np.array([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(
        _apply(back, ras * flip) * flip, _apply(chain, ras), atol=1e-8
    )


@pytest.mark.parametrize(
    "kind", ["ItkNiftiDisplacementField", "ItkNiftiCoordinateField"]
)
def test_an_itk_field_written_by_brainhops_reads_back_without_a_hint(
    tmp_path,  # noqa: ANN001
    kind: str,
) -> None:
    # A VECTOR image may hold ITK displacements or RAS coordinates, which
    # its header cannot tell apart. brainhops writes an intent name on the
    # ITK fields it writes, so they are read back as what they are.
    from brainhops.io.transformations import itk

    cls = getattr(itk.nifti, kind)
    source = transformations.load(
        Path(__file__).parent / "data" / "itk_displacement3d.h5"
    )
    path = tmp_path / "warp.nii.gz"
    source.to(cls).save(path)
    assert type(io.load(path)) is cls
