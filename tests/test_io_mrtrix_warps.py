"""
MRtrix transformations: deformation, displacement and warpfull fields
stored in MRtrix images, and linear transforms stored as text.

The fixtures are hand-built MRtrix files. The expected maps are computed
directly from what MRtrix3 does with each file (`convert.h`,
`compose.h`, `mrtransform.cpp`), not through the readers.
"""

import gzip
import math

import numpy as np
import pytest

import brainhops.io as io
from brainhops.datamodel import systems
from brainhops.datamodel import transformations as xforms
from brainhops.errors import ConversionError, IncompatibleSystemError
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.base.specs import format_hints
from brainhops.io.common.mrtrix import MrtrixRaw
from brainhops.io.common.mrtrix._codecs import parse_layout
from brainhops.io.common.mrtrix._data import MrtrixProxy, encode_data
from brainhops.io.common.mrtrix._geometry import split_voxel_to_scanner
from brainhops.io.images.mrtrix import MrtrixImage
from brainhops.io.transformations import TransformationFormat
from brainhops.io.transformations.matrix import TxtMatrixAffine
from brainhops.io.transformations.mrtrix import (
    MrtrixCoordinateField,
    MrtrixDisplacementField,
    MrtrixLinearTransform,
    MrtrixWarp,
    MrtrixWarpFull,
)

SHAPE = (4, 5, 6)
"""Grid shape: small, and no two axes of the same length."""

# A voxel-to-scanner affine with a permutation, a flip, anisotropic
# spacing and an offset, so that a field read in voxel units, or on the
# wrong grid, cannot pass.
VOX2RAS = np.array(
    [
        [0.0, -3.0, 0.0, 10.0],
        [2.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

WARPCONVERT = (
    "mrconvert in.nii in.mif  (version=3.0.4)\n"
    "/usr/local/bin/warpconvert def.mif deformation2displacement "
    "disp.mif  (version=3.0.4)"
)


def _grid_points(vox2ras=VOX2RAS, shape=SHAPE) -> np.ndarray:  # noqa: ANN001
    """The scanner coordinates of every voxel, `(X, Y, Z, 3)`."""
    ijk = np.stack(np.meshgrid(*map(np.arange, shape), indexing="ij"), -1)
    return ijk @ vox2ras[:3, :3].T + vox2ras[:3, 3]


def _ramp() -> np.ndarray:
    """An `(X, Y, Z, 3)` displacement whose entries name their voxel."""
    i, j, k = np.meshgrid(*map(np.arange, SHAPE), indexing="ij")
    return np.stack([1.0 + 0.1 * i, 2.0 + 0.2 * j, 3.0 + 0.3 * k], axis=-1)


def _write(  # noqa: ANN202
    path,  # noqa: ANN001
    values: np.ndarray,
    vox2ras=VOX2RAS,  # noqa: ANN001
    keyval=None,  # noqa: ANN001
    layout=None,  # noqa: ANN001
    transform=True,  # noqa: ANN001
):
    """Write `values` (indexed `[x, y, z, ...]`) as a single-file `.mif`,
    as MRtrix does: unit direction cosines and voxel sizes, `nan` on the
    non-spatial axes."""
    matrix, vox = split_voxel_to_scanner(vox2ras)
    header = MrtrixRaw(
        dim=values.shape,
        vox=list(vox) + [math.nan] * (values.ndim - 3),
        layout=None if layout is None else parse_layout(layout, values.ndim),
        datatype="Float32LE",
        transform=matrix if transform else None,
        keyval=keyval or {},
    )
    content = header.embedded() + encode_data(header, values.astype("<f4"))
    if str(path).endswith(".gz"):
        content = gzip.compress(content)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _apply(xform, points: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map scanner RAS points through a RAS-to-RAS transformation."""
    points = xforms.CoordinateField(field=np.asarray(points, float))
    out = xforms.Sequence(transformations=[points, *xform]).compute()
    return np.asarray(out.to(xforms.CoordinateField).field)


def _affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix[:3], [0, 0, 0, 1]])


@pytest.fixture
def deformation(tmp_path):  # noqa: ANN001, ANN201
    """A deformation: each voxel holds the scanner position it maps to,
    here its own position plus the ramp."""
    return _write(tmp_path / "warp.mif", _grid_points() + _ramp())


@pytest.fixture
def displacement(tmp_path):  # noqa: ANN001, ANN201
    """A displacement, as `warpconvert` writes it."""
    return _write(
        tmp_path / "disp.mif", _ramp(), keyval={"command_history": WARPCONVERT}
    )


# ----------------------------------------------------------------------
#   REGISTRATION AND HINTS
# ----------------------------------------------------------------------


def test_the_formats_are_registered() -> None:
    registry = TransformationFormat._REGISTRY
    for cls in (
        MrtrixCoordinateField,
        MrtrixDisplacementField,
        MrtrixWarpFull,
        MrtrixLinearTransform,
    ):
        assert cls in registry
    assert MrtrixWarp not in registry


@pytest.mark.parametrize(
    "cls, hints",
    [
        (
            MrtrixCoordinateField,
            {
                "mrtrix.field",
                "mrtrix.coord",
                "mrtrix.coordinates",
                "mrtrix.field.coord",
                "mrtrix.field.coordinates",
            },
        ),
        (
            MrtrixDisplacementField,
            {
                "mrtrix.field",
                "mrtrix.disp",
                "mrtrix.displacement",
                "mrtrix.field.disp",
                "mrtrix.field.displacement",
                "displacements",
            },
        ),
        (MrtrixWarpFull, {"mrtrix.field", "mrtrix.warpfull"}),
        (
            MrtrixLinearTransform,
            {"mrtrix.linear", "mrtrix.affine", "affine.linear"},
        ),
    ],
)
def test_hints(cls, hints) -> None:  # noqa: ANN001
    assert hints <= format_hints(cls)


# ----------------------------------------------------------------------
#   DEFORMATIONS
# ----------------------------------------------------------------------


def test_a_4d_warp_is_read_as_a_deformation(deformation) -> None:  # noqa: ANN001
    for load in (io.load, io.transformations.load):
        warp = load(deformation)
        assert type(warp) is MrtrixCoordinateField
        assert isinstance(warp.input, systems.RASmm)
        assert isinstance(warp.output, systems.RASmm)
        assert len(warp) == 2


def test_the_image_reader_still_reads_a_warp_as_an_image(deformation) -> None:  # noqa: ANN001
    image = io.images.load(deformation)
    assert type(image) is MrtrixImage
    assert image.data.shape == SHAPE + (3,)


def test_a_deformation_maps_each_voxel_to_its_value(deformation) -> None:  # noqa: ANN001
    """`mrtransform -warp` samples the moving image at the scanner
    position each voxel holds (`adapter/warp.h`)."""
    warp = io.load(deformation)
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(warp, points), points + _ramp().reshape(-1, 3), atol=1e-4
    )


def test_a_deformation_is_interpolated_linearly(tmp_path) -> None:  # noqa: ANN001
    """An affine deformation is reproduced exactly between voxels."""
    matrix = np.array(
        [[1.0, 0.1, 0.0, 2.0], [0.0, 0.9, 0.2, -1.0], [0.1, 0.0, 1.1, 0.5]]
    )
    path = _write(tmp_path / "w.mif", _affine(matrix, _grid_points()))
    ijk = np.array([[0.5, 1.25, 2.5], [2.2, 3.7, 4.1]])
    points = _affine(VOX2RAS, ijk)
    np.testing.assert_allclose(
        _apply(io.load(path), points), _affine(matrix, points), atol=1e-4
    )


@pytest.mark.parametrize("layout", ["+1,+2,+3,+0", "-0,+1,-2,+3"])
def test_a_deformation_in_any_layout(tmp_path, layout) -> None:  # noqa: ANN001
    """`warpinit` writes volume-contiguous warps (`+1,+2,+3,+0`)."""
    path = _write(tmp_path / "w.mif", _grid_points() + _ramp(), layout=layout)
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(io.load(path), points),
        points + _ramp().reshape(-1, 3),
        atol=1e-4,
    )


def test_a_deformation_without_a_transform_is_centred(tmp_path) -> None:  # noqa: ANN001
    """Without `transform`, MRtrix centres the grid on the origin."""
    vox = np.array([2.0, 3.0, 4.0])
    centred = np.diag([*vox, 1.0])
    centred[:3, 3] = -0.5 * (np.array(SHAPE) - 1) * vox
    values = _grid_points(centred) + 1.0
    path = _write(tmp_path / "w.mif", values, centred, transform=False)
    warp = io.load(path)
    np.testing.assert_allclose(
        np.linalg.inv(warp.ras2voxel.homogeneous_matrix), centred, atol=1e-6
    )
    points = _grid_points(centred).reshape(-1, 3)
    np.testing.assert_allclose(_apply(warp, points), points + 1, atol=1e-4)


@pytest.mark.parametrize(
    "history",
    [
        "warpinit template.mif warp.mif  (version=3.0.4)",
        "mrregister a.mif b.mif -nl_warp w1.mif w2.mif  (version=3.0.4)",
        "warpconvert w.mif displacement2deformation d.mif  (version=3)",
        "warpinvert w.mif winv.mif  (version=3.0.4)",
    ],
)
def test_commands_that_write_deformations(tmp_path, history) -> None:  # noqa: ANN001
    path = _write(
        tmp_path / "w.mif", _grid_points(), keyval={"command_history": history}
    )
    assert MrtrixCoordinateField.sniff(path) == Confidence.LIKELY
    assert MrtrixDisplacementField.sniff(path) == Confidence.WEAK
    assert type(io.load(path)) is MrtrixCoordinateField


def test_other_mrtrix_images_are_not_warps(tmp_path) -> None:  # noqa: ANN001
    for shape in [(4, 5, 6), (4, 5, 6, 4), (4, 5, 6, 3, 2)]:
        path = _write(tmp_path / "x.mif", np.zeros(shape))
        assert MrtrixCoordinateField.sniff(path) == 0
        assert MrtrixDisplacementField.sniff(path) == 0
        assert MrtrixWarpFull.sniff(path) == 0
        assert type(io.load(path)) is MrtrixImage


# ----------------------------------------------------------------------
#   DISPLACEMENTS
# ----------------------------------------------------------------------


def test_a_warpconvert_displacement_is_read_as_one(displacement) -> None:  # noqa: ANN001
    assert MrtrixDisplacementField.sniff(displacement) == Confidence.LIKELY
    assert MrtrixCoordinateField.sniff(displacement) == Confidence.WEAK
    warp = io.load(displacement)
    assert type(warp) is MrtrixDisplacementField
    assert isinstance(warp.input, systems.RASmm)
    assert isinstance(warp.output, systems.RASmm)
    assert len(warp) == 3


@pytest.mark.parametrize(
    "history",
    [
        "warpconvert w.mif warpfull2displacement d.mif -template t.mif",
        "warpinvert d.mif dinv.mif -displacement  (version=3.0.4)",
    ],
)
def test_commands_that_write_displacements(tmp_path, history) -> None:  # noqa: ANN001
    path = _write(
        tmp_path / "d.mif", _ramp(), keyval={"command_history": history}
    )
    assert type(io.load(path)) is MrtrixDisplacementField


def test_a_displacement_moves_each_voxel_by_its_value(displacement) -> None:  # noqa: ANN001
    """`displacement2deformation`: position + displacement."""
    warp = io.load(displacement)
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(warp, points), points + _ramp().reshape(-1, 3), atol=1e-4
    )


def test_a_displacement_without_history_is_sniffed_from_its_values(
    tmp_path,  # noqa: ANN001
) -> None:
    """Displacements stay near zero, and coordinates follow the grid."""
    path = _write(tmp_path / "d.mif", _ramp())
    assert MrtrixDisplacementField.sniff(path) == Confidence.MAYBE
    assert MrtrixCoordinateField.sniff(path) == Confidence.WEAK
    assert type(io.load(path)) is MrtrixDisplacementField
    path = _write(tmp_path / "c.mif", _grid_points() + _ramp())
    assert MrtrixCoordinateField.sniff(path) == Confidence.MAYBE
    assert MrtrixDisplacementField.sniff(path) == Confidence.WEAK
    assert type(io.load(path)) is MrtrixCoordinateField


def test_the_values_are_sniffed_through_a_mih_header(tmp_path) -> None:  # noqa: ANN001
    warp = MrtrixDisplacementField(data=_ramp())
    warp.consolidate(inplace=True, keyval={"brainhops_field_type": None})
    warp.save(tmp_path / "d.mih")
    assert type(io.load(tmp_path / "d.mih")) is MrtrixDisplacementField


def test_a_displacement_is_read_by_hint_when_nothing_decides(tmp_path) -> None:  # noqa: ANN001
    """A single voxel does not say whether it holds a position."""
    path = _write(tmp_path / "d.mif", _ramp()[:1, :1, :1])
    assert type(io.load(path)) is MrtrixCoordinateField
    for hint in ("mrtrix.disp", "mrtrix.field.displacement", "displacements"):
        warp = io.load(path, hint=hint)
        assert type(warp) is MrtrixDisplacementField
    path = _write(tmp_path / "c.mif", _ramp())
    for hint in ("mrtrix.coord", "mrtrix.field.coordinates"):
        assert type(io.load(path, hint=hint)) is MrtrixCoordinateField
    warp = MrtrixDisplacementField.from_file(path)
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(warp, points), points + _ramp().reshape(-1, 3), atol=1e-4
    )


@pytest.mark.parametrize(
    "cls, kind",
    [
        (MrtrixCoordinateField, "coordinates"),
        (MrtrixDisplacementField, "displacements"),
    ],
)
def test_brainhops_records_the_kind_of_warp(tmp_path, cls, kind) -> None:  # noqa: ANN001
    """The values are made to look like the other kind, so that only the
    key can decide."""
    values = _ramp() if cls is MrtrixCoordinateField else _grid_points()
    path = tmp_path / "w.mif"
    cls(data=values).save(path)
    record = MrtrixRaw.from_file(path)
    assert record.keyval["brainhops_field_type"] == kind
    assert record.keyval["command_history"].startswith("brainhops  (version=")
    assert cls.sniff(path) == Confidence.CERTAIN
    assert type(io.load(path)) is cls
    # Saving again does not repeat the `command_history` line.
    io.load(path).consolidate(inplace=True, datatype="float64").save(path)
    lines = MrtrixRaw.from_file(path).keyval["command_history"].split("\n")
    assert len(lines) == 1


@pytest.mark.parametrize("value", ["displacement", "displacements"])
def test_the_singular_displacement_key_still_reads(tmp_path, value) -> None:  # noqa: ANN001
    """Older versions of brainhops recorded `displacement`, in the
    singular, and such a file still reads as a displacement field."""
    path = _write(
        tmp_path / "d.mif",
        _grid_points(),
        keyval={
            "brainhops_field_type": value,
            "command_history": "brainhops  (version=1.0)",
        },
    )
    assert MrtrixDisplacementField.sniff(path) == Confidence.CERTAIN
    assert type(io.load(path)) is MrtrixDisplacementField


def test_a_brainhops_key_under_a_later_command(tmp_path) -> None:  # noqa: ANN001
    """MRtrix copies the key into the files that it derives, so a command
    that changes the kind of warp decides, and another one leaves the key
    likely."""
    brainhops = "brainhops  (version=1.0)"
    key = {"brainhops_field_type": "displacement"}
    path = _write(
        tmp_path / "c.mif",
        _grid_points(),
        keyval={
            **key,
            "command_history": brainhops
            + "\nwarpconvert d.mif displacement2deformation c.mif"
            + "  (version=3.0.4)",
        },
    )
    assert MrtrixCoordinateField.sniff(path) == Confidence.LIKELY
    assert type(io.load(path)) is MrtrixCoordinateField
    path = _write(
        tmp_path / "d.mif",
        _grid_points(),
        keyval={
            **key,
            "command_history": brainhops + "\nmrconvert d.mif d2.mif",
        },
    )
    assert MrtrixDisplacementField.sniff(path) == Confidence.LIKELY
    assert type(io.load(path)) is MrtrixDisplacementField


def test_a_changed_mrtrix_displacement_keeps_its_kind(displacement) -> None:  # noqa: ANN001
    """The history of `warpconvert` is kept, and brainhops writes last."""
    warp = io.load(displacement)
    warp.data = np.asarray(warp.data) + 1
    warp.save(displacement)
    history = MrtrixRaw.from_file(displacement).keyval["command_history"]
    assert history.startswith(WARPCONVERT + "\nbrainhops  (version=")
    assert MrtrixDisplacementField.sniff(displacement) == Confidence.CERTAIN


def test_a_deformation_and_its_displacement_agree(
    deformation,  # noqa: ANN001
    displacement,  # noqa: ANN001
) -> None:
    """`deformation2displacement` subtracts each voxel's position."""
    ijk = np.array([[0.3, 1.5, 2.25], [3.0, 0.0, 5.0], [1.9, 2.1, 0.4]])
    points = _affine(VOX2RAS, ijk)
    np.testing.assert_allclose(
        _apply(io.load(deformation), points),
        _apply(io.load(displacement), points),
        atol=1e-4,
    )


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["out.mif", "out.mif.gz", "out.mih"])
def test_a_deformation_round_trips(deformation, tmp_path, name) -> None:  # noqa: ANN001
    warp = io.load(deformation)
    io.save(warp, tmp_path / name)
    back = io.load(tmp_path / name)
    assert type(back) is MrtrixCoordinateField
    np.testing.assert_allclose(
        np.asarray(back.coordinates.field), np.asarray(warp.coordinates.field)
    )
    np.testing.assert_allclose(
        back.ras2voxel.homogeneous_matrix,
        warp.ras2voxel.homogeneous_matrix,
        atol=1e-12,
    )
    assert back.metadata.raw.dim == SHAPE + (3,)
    assert math.isnan(back.metadata.raw.vox[3])


@pytest.mark.parametrize("name", ["out.mif", "out.mih"])
def test_a_displacement_round_trips(displacement, tmp_path, name) -> None:  # noqa: ANN001
    warp = io.load(displacement)
    warp.save(tmp_path / name)
    back = io.load(tmp_path / name)
    # The history is written back, so the file is still a displacement.
    assert type(back) is MrtrixDisplacementField
    assert back.metadata.raw.keyval["command_history"] == WARPCONVERT
    np.testing.assert_allclose(back.data, _ramp(), rtol=1e-6, atol=1e-6)
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(back, points), _apply(warp, points), atol=1e-4
    )


def test_writer_options(deformation, tmp_path) -> None:  # noqa: ANN001
    warp = io.load(deformation)
    warp.save(
        tmp_path / "o.mif",
        layout="+1,+2,+3,+0",
        datatype="Float64BE",
        keyval={"comments": "made by a test"},
    )
    back = io.load(tmp_path / "o.mif")
    assert back.metadata.raw.layout == (2, 3, 4, 1)
    assert back.metadata.raw.datatype == "Float64BE"
    assert back.metadata.raw.keyval["comments"] == "made by a test"
    np.testing.assert_allclose(back.data, warp.data)
    with pytest.raises(TypeError):
        warp.to_bytes(bogus=1)


def test_a_field_built_from_a_chain_is_written(tmp_path) -> None:  # noqa: ANN001
    coordinates = (_grid_points() + _ramp()).astype("float32")
    chain = (
        xforms.Affine(matrix=np.linalg.inv(VOX2RAS)[:3]),
        xforms.CoordinateField(field=coordinates),
    )
    from brainhops.io.transformations.base.affines import RASToVoxel
    from brainhops.io.transformations.base.fields import RASCoordinateField

    warp = MrtrixCoordinateField(
        transformations=(
            RASToVoxel(matrix=chain[0].matrix),
            RASCoordinateField(field=coordinates),
        )
    )
    warp.save(tmp_path / "w.mif")
    back = io.load(tmp_path / "w.mif")
    np.testing.assert_allclose(back.data, coordinates)
    np.testing.assert_allclose(
        back.metadata.raw.voxel_to_scanner(), VOX2RAS, atol=1e-12
    )
    # An unnamed chain says nothing of scanner RAS, and is refused.
    with pytest.raises(UnrepresentableTransformationError):
        MrtrixCoordinateField(transformations=chain).save(tmp_path / "x.mif")


def test_a_field_without_data_cannot_be_written(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises((WriterError, ParserContentError)):
        MrtrixCoordinateField().save(tmp_path / "x.mif")


def _nifti_field(path, intent, name=""):  # noqa: ANN001, ANN202
    nb = pytest.importorskip("nibabel")
    vectors = _ramp().astype("float32")[:, :, :, None, :]
    image = nb.Nifti1Image(vectors, VOX2RAS)
    image.header.set_intent(intent, name=name)
    nb.save(image, str(path))
    return path


def test_nifti_displacements_are_written_as_mrtrix_warps(tmp_path) -> None:  # noqa: ANN001
    # DISPVECT: displacements
    source = io.load(_nifti_field(tmp_path / "d.nii.gz", 1006))
    io.save(source, tmp_path / "d.mif")
    back = MrtrixDisplacementField.from_file(tmp_path / "d.mif")
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(back, points), _apply(source, points), atol=1e-4
    )


def test_spm_coordinates_are_written_as_mrtrix_warps(tmp_path) -> None:  # noqa: ANN001
    # VECTOR with the intent name of SPM: coordinates
    path = _nifti_field(tmp_path / "y_c.nii.gz", 1007, "Mapping")
    source = io.load(path)
    io.save(source, tmp_path / "y_c.mif")
    back = MrtrixCoordinateField.from_file(tmp_path / "y_c.mif")
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(back, points), _apply(source, points), atol=1e-4
    )


def test_a_displacement_is_written_as_a_deformation(displacement) -> None:  # noqa: ANN001
    # The two are the same map inside the grid, so a field of
    # displacements is held as the position of each voxel.
    field = io.load(displacement)
    deformation = MrtrixCoordinateField.from_instance(field)
    assert isinstance(deformation, MrtrixCoordinateField)
    points = _grid_points()
    np.testing.assert_allclose(
        _apply(deformation, points), _apply(field, points), atol=1e-5
    )
    with pytest.raises(ConversionError):
        MrtrixWarpFull.from_instance(io.load(displacement))


# ----------------------------------------------------------------------
#   WARPFULL (5-D)
# ----------------------------------------------------------------------

MIDWAY = np.array(
    [
        [2.0, 0.0, 0.0, -9.0],
        [0.0, 2.0, 0.0, -10.0],
        [0.0, 0.0, 2.0, -11.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
MIDWAY_SHAPE = (10, 11, 12)

# Each warp is a translation, so that compositions are known exactly.
OFFSETS = [
    np.array([0.5, -0.3, 0.2]),  # im1_to_mid
    np.array([-0.5, 0.3, -0.2]),  # mid_to_im1
    np.array([0.1, 0.4, -0.6]),  # im2_to_mid
    np.array([-0.1, -0.4, 0.6]),  # mid_to_im2
]
LINEAR1 = np.array(
    [[1.0, 0.02, 0.0, 0.3], [-0.02, 1.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.1]]
)
LINEAR2 = np.array(
    [[1.0, 0.0, 0.01, -0.3], [0.0, 1.0, 0.0, 0.2], [-0.01, 0.0, 1.0, -0.1]]
)
POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -3.0], [-2.0, 1.0, 0.5]])


def _warps() -> np.ndarray:
    grid = _grid_points(MIDWAY, MIDWAY_SHAPE)
    return np.stack([grid + offset for offset in OFFSETS], axis=-1)


def _linear_text(matrix: np.ndarray) -> str:
    return "\n".join(" ".join(f"{v:g}" for v in row) for row in matrix)


@pytest.fixture
def warpfull(tmp_path):  # noqa: ANN001, ANN201
    """A warpfull file as `mrregister -nl_warp_full` writes it."""
    keyval = {
        "nl_scale": "0.25,0.5,1",
        "linear1": _linear_text(LINEAR1),
        "linear2": _linear_text(LINEAR2),
        "command_history": "mrregister a.mif b.mif -nl_warp_full w.mif",
    }
    return _write(tmp_path / "full.mif", _warps(), MIDWAY, keyval=keyval)


def test_a_5d_warp_is_read_as_a_warpfull(warpfull) -> None:  # noqa: ANN001
    assert MrtrixWarpFull.sniff(warpfull) == Confidence.CERTAIN
    assert MrtrixCoordinateField.sniff(warpfull) == 0
    warp = io.load(warpfull)
    assert type(warp) is MrtrixWarpFull
    assert isinstance(warp.input, systems.RASmm)
    assert isinstance(warp.output, systems.RASmm)
    np.testing.assert_allclose(warp.linear1.matrix, LINEAR1)
    np.testing.assert_allclose(warp.linear2.matrix, LINEAR2)


def test_a_5d_warp_without_linear_transforms(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path / "w.mif", _warps(), MIDWAY)
    assert MrtrixWarpFull.sniff(path) == Confidence.LIKELY
    warp = io.load(path)
    assert type(warp) is MrtrixWarpFull
    # MRtrix refuses it too (`parse_linear_transform`).
    with pytest.raises(ParserContentError):
        warp.linear1  # noqa: B018
    # The warps alone are still available.
    np.testing.assert_allclose(
        _apply(warp.im1_to_mid, POINTS), POINTS + OFFSETS[0], atol=1e-4
    )


def test_the_four_warps(warpfull) -> None:  # noqa: ANN001
    warp = io.load(warpfull)
    for name, offset in zip(
        ("im1_to_mid", "mid_to_im1", "im2_to_mid", "mid_to_im2"), OFFSETS
    ):
        np.testing.assert_allclose(
            _apply(getattr(warp, name), POINTS), POINTS + offset, atol=1e-4
        )


def test_the_full_warp_from_index_1(warpfull) -> None:  # noqa: ANN001
    """`compute_full_deformation(warp, template, from=1)`: image-2
    points go through `linear2⁻¹`, `mid_to_im2`, `im1_to_mid` and
    `linear1` to image-1 points."""
    warp = io.load(warpfull)
    inverse2 = np.linalg.inv(_homogeneous(LINEAR2))
    expected = _affine(
        LINEAR1, _affine(inverse2, POINTS) + OFFSETS[3] + OFFSETS[0]
    )
    np.testing.assert_allclose(_apply(warp, POINTS), expected, atol=1e-4)


def test_the_full_warp_from_index_2(warpfull) -> None:  # noqa: ANN001
    warp = io.load(warpfull, from_index=2)
    inverse1 = np.linalg.inv(_homogeneous(LINEAR1))
    expected = _affine(
        LINEAR2, _affine(inverse1, POINTS) + OFFSETS[1] + OFFSETS[2]
    )
    np.testing.assert_allclose(_apply(warp, POINTS), expected, atol=1e-4)


@pytest.mark.parametrize("from_index", [1, 2])
def test_the_midway_warps(warpfull, from_index) -> None:  # noqa: ANN001
    """`compute_midway_deformation`: `linear` after warp 0 (or 2)."""
    warp = io.load(warpfull, from_index=from_index, midway=True)
    linear, offset = (
        (LINEAR1, OFFSETS[0]) if from_index == 1 else (LINEAR2, OFFSETS[2])
    )
    expected = _affine(linear, POINTS + offset)
    np.testing.assert_allclose(_apply(warp, POINTS), expected, atol=1e-4)
    assert len(warp.transformations) == 3


def test_a_warpfull_round_trips(warpfull, tmp_path) -> None:  # noqa: ANN001
    warp = io.load(warpfull)
    warp.save(tmp_path / "o.mih")
    back = io.load(tmp_path / "o.mih")
    assert type(back) is MrtrixWarpFull
    assert back.metadata.raw.keyval == warp.metadata.raw.keyval
    np.testing.assert_allclose(back.data, warp.data)
    np.testing.assert_allclose(
        _apply(back, POINTS), _apply(warp, POINTS), atol=1e-6
    )


def test_a_warpfull_built_from_its_parts(tmp_path) -> None:  # noqa: ANN001
    warp = MrtrixWarpFull.from_warps(
        _warps().astype("float32"),
        MIDWAY,
        LINEAR1,
        xforms.Affine(matrix=LINEAR2),
    )
    warp.save(tmp_path / "w.mif")
    back = io.load(tmp_path / "w.mif")
    assert MrtrixWarpFull.sniff(tmp_path / "w.mif") == Confidence.CERTAIN
    np.testing.assert_allclose(back.linear1.matrix, LINEAR1)
    np.testing.assert_allclose(back.linear2.matrix, LINEAR2)
    np.testing.assert_allclose(
        _apply(back, POINTS), _apply(warp, POINTS), atol=1e-6
    )
    with pytest.raises(ValueError):
        MrtrixWarpFull.from_warps(np.zeros((2, 2, 2, 3, 3)), MIDWAY)
    with pytest.raises(WriterError):
        MrtrixWarpFull().save(tmp_path / "x.mif")


# ----------------------------------------------------------------------
#   LINEAR TRANSFORMS
# ----------------------------------------------------------------------

MATRIX = np.array(
    [[0.99, 0.01, 0.0, 1.5], [-0.01, 0.99, 0.0, -2.0], [0.0, 0.0, 1.0, 3.0]]
)
SAVED = """\
# command_history: mrregister a.mif b.mif -affine affine.txt  (version=3.0.4)
# centre: 1.5 -2 3
0.99 0.01 0 1.5
-0.01 0.99 0 -2
0 0 1 3
0 0 0 1
"""


def test_a_transform_written_by_mrtrix_is_recognised(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.txt"
    path.write_text(SAVED)
    assert MrtrixLinearTransform.sniff(path) == Confidence.CERTAIN
    xform = io.load(path)
    assert type(xform) is MrtrixLinearTransform
    np.testing.assert_allclose(xform.matrix, MATRIX)
    assert isinstance(xform.input, systems.RASmm)
    assert isinstance(xform.output, systems.RASmm)
    assert xform.metadata.raw.keyval["centre"] == "1.5 -2 3"


def test_the_matrix_maps_template_points_to_moving_points(tmp_path) -> None:  # noqa: ANN001
    """The "reverse" convention: no inversion (`adapter/reslice.h`)."""
    path = tmp_path / "affine.txt"
    path.write_text(SAVED)
    xform = io.load(path)
    np.testing.assert_allclose(
        _apply([xform], POINTS), _affine(MATRIX, POINTS), atol=1e-12
    )


def test_a_plain_matrix_is_read_by_hint(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "plain.txt"
    path.write_text("0.99,0.01,0,1.5\n-0.01;0.99;0;-2\n0\t0\t1\t3\n")
    assert MrtrixLinearTransform.sniff(path) == 0
    for hint in ("mrtrix.linear", "mrtrix.affine"):
        assert type(io.load(path, hint=hint)) is MrtrixLinearTransform
    xform = MrtrixLinearTransform.from_file(path)
    np.testing.assert_allclose(xform.matrix, MATRIX)
    assert "centre" not in xform.metadata.raw.keyval

    path = tmp_path / "space.txt"
    path.write_text("1 0 0 1\n0 1 0 2\n0 0 1 3\n0 0 0 1\n")
    assert type(io.load(path)) is TxtMatrixAffine


def test_only_the_first_three_rows_are_read(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "a.txt"
    path.write_text("1 0 0 1\n0 1 0 2\n0 0 1 3\n9 9 9 9\n")
    xform = MrtrixLinearTransform.from_file(path)
    np.testing.assert_allclose(
        xform.matrix, np.eye(3, 4) + [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]]
    )
    path.write_text("1 0 0\n0 1 0\n0 0 1\n")
    with pytest.raises(ParserContentError):
        MrtrixLinearTransform.from_file(path)


def test_a_linear_transform_round_trips(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.txt"
    path.write_text(SAVED)
    xform = io.load(path)
    xform.save(tmp_path / "out.txt")
    text = (tmp_path / "out.txt").read_text()
    assert text.splitlines()[:2] == SAVED.splitlines()[:2]
    assert text.splitlines()[-1] == "0 0 0 1"
    back = io.load(tmp_path / "out.txt")
    assert type(back) is MrtrixLinearTransform
    np.testing.assert_array_equal(back.matrix, xform.matrix)
    assert back.metadata.raw.keyval == xform.metadata.raw.keyval


def test_a_ras_affine_is_converted_to_an_mrtrix_transform(tmp_path) -> None:  # noqa: ANN001
    affine = xforms.Affine(
        matrix=MATRIX, input=systems.RASmm(), output=systems.RASmm()
    )
    affine.to(MrtrixLinearTransform).save(tmp_path / "a.txt")
    back = io.load(tmp_path / "a.txt")
    assert type(back) is MrtrixLinearTransform
    np.testing.assert_allclose(back.matrix, MATRIX)
    # A `.txt` name does not tell an MRtrix transform from a NiftyReg
    # affine, and `io.save` writes the latter (#343).
    io.save(affine, tmp_path / "b.txt")
    assert type(io.load(tmp_path / "b.txt", hint="niftyreg")).__name__ == (
        "NiftyRegAffine"
    )
    # An affine between LPS spaces is flipped into RAS by the conversion,
    # and refused by the format itself.
    lps = xforms.Affine(
        matrix=MATRIX, input=systems.LPSmm(), output=systems.LPSmm()
    )
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    expected = (flip @ _homogeneous(MATRIX) @ flip)[:3]
    np.testing.assert_allclose(lps.to(MrtrixLinearTransform).matrix, expected)
    with pytest.raises(IncompatibleSystemError):
        MrtrixLinearTransform(
            matrix=MATRIX, input=systems.LPSmm(), output=systems.LPSmm()
        )


def test_a_linear_transform_is_pending_until_consolidated(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.txt"
    path.write_text(SAVED)
    xform = io.load(path)
    metadata = xform.metadata
    assert not xform.data.flags.writeable
    # An untouched transform shares its record and keeps its bytes.
    assert xform.consolidate().metadata is metadata
    xform.save(tmp_path / "copy.txt")
    assert (tmp_path / "copy.txt").read_text() == SAVED
    # An assigned matrix is pending, and is encoded under the comments of
    # the record, which is not changed.
    xform.data = np.eye(3, 4)
    assert xform.metadata is metadata
    consolidated = xform.consolidate()
    assert consolidated.metadata is not metadata
    np.testing.assert_array_equal(
        consolidated.metadata.raw.matrix, np.eye(3, 4)
    )
    assert consolidated.metadata.raw.keyval == metadata.raw.keyval
    np.testing.assert_array_equal(metadata.raw.matrix, MATRIX)
    with pytest.raises(TypeError):
        MrtrixLinearTransform.from_file(path, bogus=1)
    with pytest.raises(TypeError):
        xform.to_text(bogus=1)


# ----------------------------------------------------------------------
#   PENDING STATE
# ----------------------------------------------------------------------


def test_a_warp_is_pending_until_consolidated(deformation, tmp_path) -> None:  # noqa: ANN001
    warp = io.load(deformation)
    raw, metadata = warp.rawdata, warp.metadata
    assert isinstance(raw, MrtrixProxy)
    if isinstance(warp.data, np.ndarray):
        assert not warp.data.flags.writeable
    # Assigned coordinates are pending, and drop the decoded chain.
    first = warp.transformations
    warp.data = np.asarray(warp.data) + 1
    assert warp.rawdata is raw and warp.metadata is metadata
    assert warp.transformations is not first
    consolidated = warp.consolidate()
    assert consolidated._data is None
    np.testing.assert_array_equal(consolidated.rawdata, np.asarray(raw) + 1)
    # The geometry and the keys of the record are kept, and the kind of
    # warp is recorded with a `command_history` line of brainhops.
    record = consolidated.metadata.raw
    np.testing.assert_array_equal(record.transform, metadata.raw.transform)
    added = set(record.keyval) - set(metadata.raw.keyval)
    assert added == {"brainhops_field_type", "command_history"}
    assert record.keyval["brainhops_field_type"] == "coordinates"
    assert record.keyval["command_history"].startswith("brainhops  (version=")
    # Saving over its own file re-points the warp at the new file.
    warp.save(deformation)
    assert isinstance(warp.rawdata, MrtrixProxy) and warp._data is None
    np.testing.assert_allclose(warp.data, io.load(deformation).data)


def test_a_warp_refuses_other_endpoints(deformation) -> None:  # noqa: ANN001
    warp = io.load(deformation)
    warp.input = systems.LPSmm()
    with pytest.raises(UnrepresentableTransformationError):
        warp.consolidate()
    with pytest.raises(TypeError):
        MrtrixCoordinateField.from_file(deformation, bogus=1)


def test_a_warpfull_refuses_an_assigned_chain(warpfull, tmp_path) -> None:  # noqa: ANN001
    warp = io.load(warpfull)
    warp.transformations = tuple(warp.transformations)
    with pytest.raises(UnrepresentableTransformationError):
        warp.save(tmp_path / "x.mif")
    with pytest.raises(ConversionError):
        xforms.Sequence(list(io.load(warpfull))).to(MrtrixWarpFull)
