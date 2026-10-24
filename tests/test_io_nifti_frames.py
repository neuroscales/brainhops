"""Tests for the frames of the vectors of NIfTI displacement images.

Each intent of a displacement field states in which coordinate system
its vectors are expressed. A DISPVECT file stores RAS vectors in the
world of its best affine, and an FSL file stores them in the FSL scaled
millimetres of its reference. The reader records that frame without
converting the values, and the writer re-expresses the values in the
world of the preferred transformation.
"""

import pathlib

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

from brainhops.datamodel import transformations as _xforms  # noqa: E402
from brainhops.datamodel._transformations.compute.multiscale import (  # noqa: E402
    _as_affine,
)
from brainhops.datamodel.axes import Axis  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    CoordinateSystem,
    LPSCoordinateSystem,
)
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    DisplacementField,
)
from brainhops.errors import LossyConversionError, WriterError  # noqa: E402
from brainhops.io import save  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations.fsl import (  # noqa: E402
    FnirtWarpField,
    FslCoordinateSystem,
)

_L = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
_SHAPE = (4, 5, 6)


def _systems(
    kind: str = "displacement", **options: tx.Any
) -> tx.Tuple[CoordinateSystem, CoordinateSystem]:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"] + [Axis("c", kind, **options)],
        name="voxel",
    )
    world = CoordinateSystem(
        [Axis(n, "space") for n in "xyz"] + [Axis("c", kind, **options)],
        name="RAS",
    )
    return voxel, world


def _affine(voxel: CoordinateSystem, world: CoordinateSystem) -> Affine:
    matrix = np.eye(5)[:-1]
    matrix[:3, :3] = _L
    matrix[:3, -1] = [1.0, 2.0, 3.0]
    return Affine(matrix=matrix, input=voxel, output=world)


def _stored(path: pathlib.Path) -> np.ndarray:
    # The vectors of a 3-D field are stored along the fifth NIfTI axis.
    return np.asarray(nb.load(str(path)).dataobj)[:, :, :, 0, :]


def _matrix(frame: tx.Any) -> np.ndarray:
    return np.asarray(_as_affine(frame).matrix)


@pytest.fixture
def values() -> np.ndarray:
    return np.random.default_rng(0).random((*_SHAPE, 3)).astype("f4")


def test_vectors_in_voxel_units_are_written_in_ras(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    a = _affine(*_systems())
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a]), path)

    assert nb.load(str(path)).header.get_intent()[0] == "displacement vector"
    assert np.allclose(_stored(path), values @ _L.T, atol=1e-5)


def test_vectors_in_ras_are_written_as_they_are(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    a = _affine(*_systems())
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a], frames={"c": a}), path)

    assert np.allclose(_stored(path), values)


def test_a_lazy_transformation_rotates_the_written_vectors(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    voxel, world = _systems()
    a = _affine(voxel, world)
    theta = 0.3
    rotation = np.eye(5)[:-1]
    rotation[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    u = Affine(matrix=rotation, input=world, output=world)
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a], frames={"c": a})(u), path)

    inverse = np.linalg.inv(rotation[:3, :3])
    assert np.allclose(_stored(path), values @ inverse.T, atol=1e-5)


def test_the_reader_records_the_frame_of_the_file(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    path = tmp_path / "field.nii"
    save(SingleScaleImage(values, [_affine(*_systems())]), path)

    image = NiftiImage.load(path)

    assert image.transformation.input.axes[4].type == "displacement"
    # The frames of the header are the same objects at every read.
    frame = image.frames[4]
    assert image.frames[4] is frame
    assert image.reframe(frame) is image
    # The values are already in the frame of the preferred transformation,
    # which is another object, so they are re-expressed without change.
    ras = np.asarray(image.reframe().data)[:, :, :, 0, :]
    assert np.allclose(ras, values @ _L.T, atol=1e-5)
    voxel = np.asarray(image.reframe(None).data)[:, :, :, 0, :]
    assert np.allclose(voxel, values, atol=1e-5)


def test_a_consolidated_image_reads_its_frames_from_the_header(
    values: np.ndarray,
) -> None:
    a = _affine(*_systems())
    image = NiftiImage.from_instance(SingleScaleImage(values, [a]))

    consolidated = image.consolidate()

    frame = consolidated.frames[4]
    assert consolidated.frames[4] is frame
    ras = np.asarray(consolidated.reframe().data)[:, :, :, 0, :]
    assert np.allclose(ras, values @ _L.T, atol=1e-5)


def test_a_frame_that_nifti_cannot_store_needs_lossy(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    voxel, world = _systems("vector", valence="covector")
    image = SingleScaleImage(values, [_affine(voxel, world)])
    path = tmp_path / "gradient.nii"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)
    with pytest.raises(LossyConversionError):
        NiftiImage.from_instance(image).consolidate()

    save(image, path, lossy=True)
    stored = np.asarray(nb.load(str(path)).dataobj)[:, :, :, 0, :]
    assert np.allclose(stored, values)


def test_a_frame_that_is_not_affine_needs_lossy(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    voxel, world = _systems()
    displacements = np.zeros((*_SHAPE, 1, 4))
    displacements[..., 0, 0] = 0.1 * np.arange(_SHAPE[1])[None, :, None]
    field = DisplacementField(data=displacements, input=voxel, output=voxel)
    image = SingleScaleImage(
        values, [_affine(voxel, world)], frames={"c": field}
    )
    path = tmp_path / "field.nii"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)

    save(image, path, lossy=True)
    # The field is linear, so its finite differences are exact.
    shear = np.eye(3)
    shear[0, 1] = 0.1
    expected = values @ (_L @ np.linalg.inv(shear)).T
    assert np.allclose(_stored(path), expected, atol=1e-5)


def test_a_two_dimensional_field_follows_the_first_axes(
    tmp_path: pathlib.Path,
) -> None:
    # The world of the file has three spatial axes, and the two components
    # follow the first two of them.
    nifti = nb.Nifti1Image(
        np.ones((4, 5, 1, 1, 2), "f4"), np.diag([2.0, 3.0, 4.0, 1.0])
    )
    nifti.header.set_intent("displacement vector")
    path = tmp_path / "field.nii"
    nb.save(nifti, str(path))

    image = NiftiImage.load(path)

    assert image.frames[4] is not None
    voxel = np.asarray(image.reframe(None).data)
    assert np.allclose(voxel[0, 0, 0, 0], [0.5, 1.0 / 3.0])


# ----------------------------------------------------------------------
#   FSL INTENTS
# ----------------------------------------------------------------------

FNIRT_DISPLACEMENT = 2006
CUBIC = 2007
DCT = 2008
QUADRATIC = 2009


def _fsl_file(
    path: pathlib.Path, data: np.ndarray, affine: np.ndarray, intent: int
) -> pathlib.Path:
    # The components are stored along the fifth axis, as the NIfTI standard
    # places them.
    nifti = nb.Nifti1Image(data[:, :, :, None, :].astype("f4"), affine)
    nifti.header["intent_code"] = intent
    nb.save(nifti, str(path))
    return path


def _reference_affine(neurological: bool) -> np.ndarray:
    affine = np.diag([2.0, 2.5, 3.0, 1.0])
    affine[:3, 3] = [-10.0, -12.0, -9.0]
    if not neurological:
        affine[0, 0] = -affine[0, 0]
    return affine


def _components(image: NiftiImage) -> np.ndarray:
    return np.asarray(image.data)[:, :, :, 0, :]


@pytest.mark.parametrize("neurological", [False, True])
def test_an_fnirt_displacement_field_is_in_scaled_millimetres(
    tmp_path: pathlib.Path, neurological: bool
) -> None:
    # FSL multiplies the voxel indices by the voxel sizes, and flips the
    # first axis when the best affine has a positive determinant.
    data = np.random.default_rng(2).normal(size=(*_SHAPE, 3))
    affine = _reference_affine(neurological)
    path = _fsl_file(tmp_path / "warp.nii", data, affine, FNIRT_DISPLACEMENT)

    image = NiftiImage.load(path)

    frame = image.frames[4]
    assert isinstance(frame.output, FslCoordinateSystem)
    sign = -1.0 if neurological else 1.0
    linear = np.diag([sign * 2.0, 2.5, 3.0])
    assert np.allclose(_matrix(frame)[:, :3], linear)
    voxel = _components(image.reframe(None))
    assert np.allclose(voxel, data / np.diag(linear), atol=1e-5)


@pytest.mark.parametrize("neurological", [False, True])
def test_an_fnirt_field_in_ras_matches_the_fnirt_reader(
    tmp_path: pathlib.Path, neurological: bool
) -> None:
    # With the reference as moving image, FSL maps the position of a voxel
    # to that position plus the displacement, so the displacement
    # re-expressed in RAS is the difference of the positions that the FNIRT
    # reader gives.
    data = np.random.default_rng(3).normal(size=(*_SHAPE, 3))
    affine = _reference_affine(neurological)
    path = _fsl_file(tmp_path / "warp.nii", data, affine, FNIRT_DISPLACEMENT)
    image = NiftiImage.load(path)
    warp = FnirtWarpField.load(path, deformation_type="relative")
    warp.moving = nb.load(str(path)).header

    ras = _components(image.reframe(-1))

    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in _SHAPE], indexing="ij"), -1
    ).astype(float)
    points = grid @ affine[:3, :3].T + affine[:3, 3]
    chain = _xforms.Sequence(
        transformations=[_xforms.CoordinatesField(field=points)]
        + list(warp.transformations)
    )
    moved = np.asarray(chain.compute().field)
    assert np.allclose(ras, moved - points, atol=1e-4)


_PREMAT = np.array(
    [
        [1.05, 0.02, 0.0, 1.5],
        [-0.03, 0.97, 0.05, -2.0],
        [0.0, 0.01, 1.02, 0.7],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _coefficient_file(
    path: pathlib.Path, intent: int, premat: np.ndarray
) -> tx.Tuple[pathlib.Path, np.ndarray]:
    # The header of a coefficient file, as FNIRT writes it: the knot
    # spacing in the pixdims, the voxel sizes of the reference in the
    # intent parameters, the shape of the reference in the offsets of the
    # qform and the initial affine in the sform.
    coef = np.random.default_rng(4).normal(size=(6, 5, 7, 1, 3))
    nifti = nb.Nifti1Image(coef.astype("f4"), None)
    header = nifti.header
    header["intent_code"] = intent
    header["intent_p1"] = 2.0
    header["intent_p2"] = 2.5
    header["intent_p3"] = 3.0
    header.set_zooms((3.0, 4.0, 2.0, 1.0, 1.0))
    qform = np.diag([3.0, 4.0, 2.0, 1.0])
    qform[:3, 3] = (11, 9, 8)
    nifti.set_qform(qform, 1)
    nifti.set_sform(premat, 1)
    nb.save(nifti, str(path))
    return path, coef[:, :, :, 0, :]


@pytest.mark.parametrize("intent", [CUBIC, QUADRATIC])
@pytest.mark.parametrize("premat", [np.eye(4), _PREMAT])
def test_fnirt_coefficients_are_in_the_frame_of_the_fnirt_reader(
    tmp_path: pathlib.Path, intent: int, premat: np.ndarray
) -> None:
    # The FNIRT reader stores the coefficients in units of the grid of
    # knots, moved by the initial affine, which is the array system of the
    # image.
    path, _ = _coefficient_file(tmp_path / "coef.nii", intent, premat)
    image = NiftiImage.load(path)
    warp = FnirtWarpField.load(path)
    warp.reference = nb.Nifti1Image(
        np.zeros((11, 9, 8), "f4"), _reference_affine(False)
    )
    warp.moving = warp.reference

    voxel = _components(image.reframe(None))

    assert isinstance(image.frames[4].output, FslCoordinateSystem)
    field = np.asarray(warp.transformations[1].data)
    assert np.allclose(voxel, field, atol=1e-5)


def test_a_dct_field_has_a_foreign_frame(tmp_path: pathlib.Path) -> None:
    path, _ = _coefficient_file(tmp_path / "dct.nii", DCT, np.eye(4))

    image = NiftiImage.load(path)

    assert image.frames[4] == FslCoordinateSystem()


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------

DISPVECT = 1006


def _intent(path: pathlib.Path) -> int:
    return int(nb.load(str(path)).header["intent_code"])


def test_a_vector_axis_is_written_as_displacements(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    a = _affine(*_systems("vector"))
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a]), path)

    assert _intent(path) == DISPVECT
    assert np.allclose(_stored(path), values @ _L.T, atol=1e-5)
    image = NiftiImage.load(path)
    assert image.transformation.input.axes[4].type == "displacement"


@pytest.mark.parametrize(
    "kind, size",
    [("tensor", 6), ("point", 3), ("coordinate", 3)],
)
def test_values_without_a_nifti_frame_need_lossy(
    tmp_path: pathlib.Path, kind: str, size: int
) -> None:
    values = np.random.default_rng(1).random((*_SHAPE, size))
    image = SingleScaleImage(values, [_affine(*_systems(kind))])
    path = tmp_path / "values.nii"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)

    save(image, path, lossy=True)
    assert _intent(path) != DISPVECT
    assert np.allclose(_stored(path), values, atol=1e-6)


def test_a_foreign_frame_needs_lossy(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    voxel, world = _systems()
    image = SingleScaleImage(
        values, [_affine(voxel, world)], frames={"c": FslCoordinateSystem()}
    )

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, tmp_path / "field.nii")


def test_a_two_dimensional_field_is_written_in_ras(
    tmp_path: pathlib.Path,
) -> None:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ij"] + [Axis("c", "displacement")],
        name="voxel",
    )
    world = CoordinateSystem(
        [Axis(n, "space") for n in "xy"] + [Axis("c", "displacement")],
        name="RAS",
    )
    matrix = np.eye(4)[:-1]
    matrix[:2, :2] = np.diag([2.0, 3.0])
    a = Affine(matrix=matrix, input=voxel, output=world)
    path = tmp_path / "field.nii"

    save(SingleScaleImage(np.ones((4, 5, 2)), [a]), path)

    assert _intent(path) == DISPVECT
    stored = np.asarray(nb.load(str(path)).dataobj)
    assert np.allclose(stored[0, 0, 0, 0], [2.0, 3.0])


def test_an_unchanged_fsl_field_keeps_its_intent_and_values(
    tmp_path: pathlib.Path,
) -> None:
    data = np.random.default_rng(5).normal(size=(*_SHAPE, 3))
    path = _fsl_file(
        tmp_path / "warp.nii",
        data,
        _reference_affine(True),
        FNIRT_DISPLACEMENT,
    )
    image = NiftiImage.load(path)
    image.data = np.asarray(image.data) * 2.0
    copy = tmp_path / "copy.nii"

    save(image, copy)

    assert _intent(copy) == FNIRT_DISPLACEMENT
    # The field is written as FSL writes it, with its components along the
    # fourth axis.
    stored = np.asarray(nb.load(str(copy)).dataobj)
    assert np.allclose(stored, 2.0 * data, atol=1e-5)


def _moved(image: NiftiImage) -> SingleScaleImage:
    world = image.transformation.output
    shift = np.eye(6)[:-1]
    shift[0, -1] = 7.0
    return image(Affine(matrix=shift, input=world, output=world))


def test_a_moved_fsl_field_is_written_in_ras_with_lossy(
    tmp_path: pathlib.Path,
) -> None:
    # Once its geometry changes, the field no longer lies on the grid of
    # its reference, so it is written with DISPVECT, in RAS. The vectors in
    # RAS are the displacements of the map only if the moving image has
    # the voxel-to-RAS map of the reference, so the write needs lossy=True.
    data = np.random.default_rng(6).normal(size=(*_SHAPE, 3))
    affine = _reference_affine(True)
    path = _fsl_file(tmp_path / "warp.nii", data, affine, FNIRT_DISPLACEMENT)
    image = NiftiImage.load(path)
    ras = _components(image.reframe(-1))
    copy = tmp_path / "copy.nii"

    with pytest.raises(WriterError, match="same voxel-to-RAS map"):
        save(_moved(image), copy)
    with pytest.raises(LossyConversionError, match="FSL scaled"):
        NiftiImage.from_instance(_moved(image)).consolidate()

    save(_moved(image), copy, lossy=True)

    assert _intent(copy) == DISPVECT
    assert np.allclose(_stored(copy), ras, atol=1e-4)


def test_moved_fsl_coefficients_need_lossy(tmp_path: pathlib.Path) -> None:
    path, coef = _coefficient_file(tmp_path / "coef.nii", CUBIC, np.eye(4))
    moved = _moved(NiftiImage.load(path))
    copy = tmp_path / "copy.nii"

    with pytest.raises(WriterError, match="lossy=True"):
        save(moved, copy)

    save(moved, copy, lossy=True)
    assert _intent(copy) == CUBIC
    # The intent comes from the metadata, not from the header of the
    # image, so the field keeps the layout of the NIfTI standard.
    assert np.allclose(_stored(copy), coef, atol=1e-6)


def test_a_moved_dispvect_field_keeps_its_vectors(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    # The frame read from the file ends with a projection that removes the
    # time axis of its world, and a translation of the world does not
    # change the vectors.
    path = tmp_path / "field.nii"
    save(SingleScaleImage(values, [_affine(*_systems())]), path)
    copy = tmp_path / "copy.nii"

    save(_moved(NiftiImage.load(path)), copy)

    assert _intent(copy) == DISPVECT
    assert np.allclose(_stored(copy), values @ _L.T, atol=1e-5)


# ----------------------------------------------------------------------
#   THE 4-D FILES OF FSL
# ----------------------------------------------------------------------


def _fsl_4d_file(
    path: pathlib.Path, data: np.ndarray, affine: np.ndarray
) -> pathlib.Path:
    # FSL writes the components along the fourth axis.
    nifti = nb.Nifti1Image(data.astype("f4"), affine)
    nifti.header["intent_code"] = FNIRT_DISPLACEMENT
    nb.save(nifti, str(path))
    return path


@pytest.mark.parametrize("neurological", [False, True])
def test_a_4d_fsl_field_holds_its_components_on_the_fourth_axis(
    tmp_path: pathlib.Path, neurological: bool
) -> None:
    data = np.random.default_rng(7).normal(size=(*_SHAPE, 3))
    affine = _reference_affine(neurological)
    path = _fsl_4d_file(tmp_path / "warp.nii", data, affine)
    image = NiftiImage.load(path)
    warp = FnirtWarpField.load(path, deformation_type="relative")
    warp.moving = nb.load(str(path)).header

    assert image.transformation.input.axes[3].type == "displacement"
    ras = np.asarray(image.reframe(-1).data)

    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in _SHAPE], indexing="ij"), -1
    ).astype(float)
    points = grid @ affine[:3, :3].T + affine[:3, 3]
    chain = _xforms.Sequence(
        transformations=[_xforms.CoordinatesField(field=points)]
        + list(warp.transformations)
    )
    moved = np.asarray(chain.compute().field)
    assert np.allclose(ras, moved - points, atol=1e-4)


def test_a_changed_4d_fsl_field_is_written_as_fsl_writes_it(
    tmp_path: pathlib.Path,
) -> None:
    data = np.random.default_rng(8).normal(size=(*_SHAPE, 3))
    path = _fsl_4d_file(tmp_path / "warp.nii", data, _reference_affine(True))
    image = NiftiImage.load(path)
    image.data = np.asarray(image.data) * 2.0
    copy = tmp_path / "copy.nii"

    save(image, copy)

    written = nb.load(str(copy))
    assert written.shape == (*_SHAPE, 3)
    assert _intent(copy) == FNIRT_DISPLACEMENT
    assert np.allclose(np.asarray(written.dataobj), 2.0 * data, atol=1e-5)


@pytest.mark.parametrize("shape", [(*_SHAPE, 2), (*_SHAPE, 1, 2)])
def test_an_fsl_field_of_two_components_is_read(
    tmp_path: pathlib.Path, shape: tx.Tuple[int, ...]
) -> None:
    # The frame of the FSL scaled millimetres has three axes, and the two
    # components follow the first two of them.
    nifti = nb.Nifti1Image(np.ones(shape, "f4"), _reference_affine(False))
    nifti.header["intent_code"] = FNIRT_DISPLACEMENT
    path = tmp_path / "warp.nii"
    nb.save(nifti, str(path))

    image = NiftiImage.load(path)

    voxel = np.asarray(image.reframe(None).data).reshape(-1, 2)
    assert np.allclose(voxel[0], [0.5, 0.4])


def test_a_plain_image_keeps_the_frames_of_a_nifti_image(
    tmp_path: pathlib.Path,
) -> None:
    data = np.random.default_rng(9).normal(size=(*_SHAPE, 3))
    path = _fsl_file(
        tmp_path / "warp.nii", data, _reference_affine(True), 2006
    )
    image = NiftiImage.load(path)

    for plain in (
        SingleScaleImage.from_instance(image),
        image.to(SingleScaleImage),
    ):
        assert type(plain) is SingleScaleImage
        assert isinstance(plain.frames[4].output, FslCoordinateSystem)


def test_vectors_in_centimetres_are_written_in_millimetres(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    # NIfTI stores its world in millimetres, so the writer converts the
    # matrix and the vectors from centimetres.
    voxel, _ = _systems()
    world = CoordinateSystem(
        [Axis(n, "space", unit="cm") for n in "xyz"] + [Axis("c", "channel")],
        name="RAS",
    )
    a = _affine(voxel, world)
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a], frames={"c": a}), path)

    assert np.allclose(_stored(path), 10.0 * values, atol=1e-5)
    image = NiftiImage.load(path)
    voxel_values = np.asarray(image.reframe(None).data)[:, :, :, 0, :]
    assert np.allclose(voxel_values @ _L.T, values, atol=1e-4)


def test_an_intent_given_by_keyword_does_not_relabel_the_values(
    tmp_path: pathlib.Path,
) -> None:
    # The values of an FSL field are not in the frame of DISPVECT, so the
    # writer re-expresses them in RAS, which rests on the orientation of
    # the moving image.
    data = np.random.default_rng(10).normal(size=(*_SHAPE, 3))
    affine = _reference_affine(True)
    path = _fsl_file(tmp_path / "warp.nii", data, affine, FNIRT_DISPLACEMENT)
    image = NiftiImage.load(path)
    ras = _components(image.reframe(-1))
    copy = tmp_path / "copy.nii"

    with pytest.raises(LossyConversionError):
        image.consolidate(intent_code=DISPVECT)
    with pytest.raises(LossyConversionError):
        image.consolidate(intent="displacement vector")

    save(image, copy, lossy=True, intent_code=DISPVECT)

    assert _intent(copy) == DISPVECT
    assert np.allclose(_stored(copy), ras, atol=1e-4)


def test_an_fsl_intent_given_by_keyword_needs_lossy(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    # The writer cannot re-express values in the frame of an FSL intent.
    image = NiftiImage.from_instance(
        SingleScaleImage(values, [_affine(*_systems())])
    )

    with pytest.raises(LossyConversionError):
        image.consolidate(intent_code=FNIRT_DISPLACEMENT)

    written = image.consolidate(intent_code=FNIRT_DISPLACEMENT, lossy=True)
    assert int(written.metadata.raw.header["intent_code"]) == 2006
    assert np.allclose(_components(written), values, atol=1e-6)


def test_a_template_does_not_lay_out_other_values_as_fsl(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    # A covector written as it is takes the intent of the template, but
    # keeps the layout of the NIfTI standard.
    like = _fsl_file(
        tmp_path / "warp.nii",
        np.zeros((*_SHAPE, 3)),
        _reference_affine(False),
        FNIRT_DISPLACEMENT,
    )
    voxel, world = _systems("vector", valence="covector")
    image = SingleScaleImage(values, [_affine(voxel, world)])
    path = tmp_path / "gradient.nii"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path, like=str(like))

    save(image, path, like=str(like), lossy=True)

    assert nb.load(str(path)).shape == (*_SHAPE, 1, 3)
    assert np.allclose(_stored(path), values, atol=1e-6)


def test_the_writer_keeps_the_frames_of_the_header(
    tmp_path: pathlib.Path,
) -> None:
    # Frames that are read from the header keep the file as it is. Frames
    # that are assigned, even the same objects, are frames given to the
    # image, which the writer re-expresses in RAS.
    data = np.random.default_rng(11).normal(size=(*_SHAPE, 3))
    affine = _reference_affine(True)
    path = _fsl_file(tmp_path / "warp.nii", data, affine, FNIRT_DISPLACEMENT)
    image = NiftiImage.load(path)
    image.data = np.asarray(image.data) + 1.0
    copy = tmp_path / "copy.nii"

    save(image, copy)
    assert _intent(copy) == FNIRT_DISPLACEMENT

    image.frames = image.frames
    with pytest.raises(LossyConversionError, match="FSL scaled"):
        image.consolidate()


def _dispvect(path: pathlib.Path, values: np.ndarray) -> NiftiImage:
    save(SingleScaleImage(values, [_affine(*_systems())]), path)
    return NiftiImage.load(path)


def test_the_frames_of_the_header_are_the_same_objects(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    image = _dispvect(tmp_path / "field.nii", values)

    assert image.frames is image.frames
    assert image.frames[4] is image.frames[4]


def test_the_frames_of_the_header_follow_the_image(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    # The operations of the data model read the frames through the
    # property, so the frames of the header reach their results.
    image = _dispvect(tmp_path / "field.nii", values)
    frame = image.frames[4]
    world = image.transformation.output
    shift = np.eye(6)[:-1]
    shift[0, -1] = 7.0
    u = Affine(matrix=shift, input=world, output=world)

    assert image(u).frames[4] is frame
    assert image.to(SingleScaleImage).frames[4] is frame
    indexed = image[1:3]
    assert indexed.frames[4] is not None
    ras = np.asarray(indexed.reframe(-1).data)[:, :, :, 0, :]
    assert np.allclose(ras, values[1:3] @ _L.T, atol=1e-5)
    resliced = image.reslice()
    assert resliced.frames[4] is not None
    ras = np.asarray(resliced.reframe(-1).data)[:, :, :, 0, :]
    assert np.allclose(ras, values @ _L.T, atol=1e-4)


def test_assigning_the_header_state_decodes_new_frames(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    image = _dispvect(tmp_path / "field.nii", values)
    first = image.frames[4]

    image.metadata = image.metadata
    second = image.frames[4]
    assert second is not first

    image.transformations = list(image.transformations)
    assert image.frames[4] is not second


def test_assigned_frames_stick(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    image = _dispvect(tmp_path / "field.nii", values)

    image.frames = {4: None}

    assert image.frames[4] is None
    assert image.frames is image.frames


def test_a_moving_image_with_another_origin_adds_an_offset(
    tmp_path: pathlib.Path,
) -> None:
    # The moving image has the orientation and the voxel sizes of the
    # reference, but its origin is shifted, so the displacements of the map
    # in RAS differ from the vectors re-expressed in RAS by a constant.
    data = np.random.default_rng(12).normal(size=(*_SHAPE, 3))
    affine = _reference_affine(False)
    path = _fsl_file(tmp_path / "warp.nii", data, affine, FNIRT_DISPLACEMENT)
    image = NiftiImage.load(path)
    shifted = affine.copy()
    shifted[:3, 3] += [4.0, -3.0, 2.0]
    warp = FnirtWarpField.load(path, deformation_type="relative")
    warp.moving = nb.Nifti1Image(np.zeros(_SHAPE, "f4"), shifted)

    ras = _components(image.reframe(-1))

    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in _SHAPE], indexing="ij"), -1
    ).astype(float)
    points = grid @ affine[:3, :3].T + affine[:3, 3]
    chain = _xforms.Sequence(
        transformations=[_xforms.CoordinatesField(field=points)]
        + list(warp.transformations)
    )
    moved = np.asarray(chain.compute().field)
    offset = (moved - points) - ras
    assert np.allclose(offset, [4.0, -3.0, 2.0], atol=1e-4)


def test_vectors_in_lps_are_written_in_ras(tmp_path: pathlib.Path) -> None:
    # NIfTI stores its world as RAS, so the writer flips the first two
    # components of vectors in LPS, as it flips the matrix.
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"] + [Axis("c", "displacement")],
        name="voxel",
    )
    world = CoordinateSystem(
        list(LPSCoordinateSystem().axes) + [Axis("c", "displacement")],
        name="LPS",
    )
    matrix = np.eye(5)[:-1]
    matrix[:3, :3] = np.diag([2.0, 3.0, 4.0])
    a = Affine(matrix=matrix, input=voxel, output=world)
    values = np.random.default_rng(13).normal(size=(*_SHAPE, 3))
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a], frames={"c": a}), path)

    assert np.allclose(nb.load(str(path)).affine[:3, :3], np.diag([-2, -3, 4]))
    expected = values * np.array([-1.0, -1.0, 1.0])
    assert np.allclose(_stored(path), expected, atol=1e-5)
    image = NiftiImage.load(path)
    voxel_values = np.asarray(image.reframe(None).data)[:, :, :, 0, :]
    assert np.allclose(voxel_values, values / [2.0, 3.0, 4.0], atol=1e-5)


def test_a_field_whose_world_mixes_its_axes_needs_lossy(
    tmp_path: pathlib.Path,
) -> None:
    # The two components follow the first two axes of a world in SAR,
    # which are the third and second axes of RAS, and DISPVECT holds two
    # components along the first two axes of RAS.
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"] + [Axis("c", "displacement")],
        name="voxel",
    )
    world = CoordinateSystem(
        [
            Axis("s", "space", orientation="inferior-to-superior"),
            Axis("a", "space", orientation="posterior-to-anterior"),
            Axis("r", "space", orientation="left-to-right"),
            Axis("c", "displacement"),
        ],
        name="SAR",
    )
    matrix = np.eye(5)[:-1]
    matrix[:3, :3] = np.diag([2.0, 3.0, 4.0])
    a = Affine(matrix=matrix, input=voxel, output=world)
    image = SingleScaleImage(np.ones((*_SHAPE, 2)), [a])
    path = tmp_path / "field.nii"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)
    with pytest.raises(LossyConversionError, match="first axes"):
        NiftiImage.from_instance(image).consolidate()

    save(image, path, lossy=True)
    assert _intent(path) == DISPVECT
