"""Tests for BIDS X5 (.x5) transformation files.

The fixtures are written with h5py in the layouts of nitransforms (version 1)
and fslpy (version '0.1.0'). When either library is installed, the tests
also check that both sides read each other's files alike.
"""

import io as _io
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from bagof.magic import replace

h5py = pytest.importorskip("h5py")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import systems  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    ParserContentError,
    ParserNotImplementedError,
    UnrepresentableTransformationError,
)
from brainhops.io.common.hdf5 import DelayedH5Array  # noqa: E402
from brainhops.io.transformations.x5 import (  # noqa: E402
    X5BSplineField,
    X5CoordinatesField,
    X5DisplacementField,
    X5Metadata,
    X5Raw,
    X5Transform,
)

data_dir = Path(__file__).parent / "data"

SHAPE = (4, 5, 6)
"""A small grid whose axis lengths all differ."""

# A permutation, flip, anisotropy and offset, so a misread grid fails.
VOX2RAS = np.array(
    [
        [0.0, -3.0, 0.0, 10.0],
        [2.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# A rotation about the superior axis, plus a translation.
AFFINE = np.array(
    [
        [0.0, -1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, -2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

KINDS = ("space", "space", "space", "vector")
IJK = np.array([[1, 2, 3], [0, 4, 5], [3, 1, 0], [2, 0, 4]])
"""Voxels where the fields are probed, avoiding interpolation."""


def _ramp() -> np.ndarray:
    """A field whose entries name their voxel."""
    i, j, k = np.meshgrid(*map(np.arange, SHAPE), indexing="ij")
    return np.stack(
        [1.0 + 0.1 * i, 2.0 + 0.2 * j, 3.0 + 0.3 * k], axis=-1
    ).astype("float32")


def _ras(ijk: np.ndarray) -> np.ndarray:
    return ijk @ VOX2RAS[:3, :3].T + VOX2RAS[:3, 3]


def _apply(xform, points: np.ndarray) -> np.ndarray:  # noqa: ANN001
    points = xforms.CoordinatesField(field=np.asarray(points, float))
    chain = list(xform) if isinstance(xform, xforms.Sequence) else [xform]
    out = xforms.Sequence(transformations=[points, *chain]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


def _affine_apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


# ----------------------------------------------------------------------
#   FIXTURES, IN NITRANSFORMS' LAYOUT
# ----------------------------------------------------------------------


def _write_node(group, node: dict) -> None:  # noqa: ANN001
    """Write a node as nitransforms.io.x5._write_x5_group does."""
    group.attrs["Type"] = node["type"]
    group.attrs["ArrayLength"] = node.get("array_length", 1)
    for key, attr in (
        ("subtype", "SubType"),
        ("representation", "Representation"),
    ):
        if node.get(key) is not None:
            group.attrs[attr] = node[key]
    if node.get("metadata") is not None:
        group.attrs["Metadata"] = json.dumps(node["metadata"])
    group.create_dataset("Transform", data=node["transform"])
    group.create_dataset(
        "DimensionKinds", data=np.asarray(node.get("kinds", KINDS), dtype="S")
    )
    if node.get("domain") is not None:
        size, mapping = node["domain"]
        domain = group.create_group("Domain")
        domain.create_dataset("Grid", data=np.uint8(1))
        domain.create_dataset("Size", data=np.asarray(size))
        domain.create_dataset("Mapping", data=mapping)
        domain.attrs["Coordinates"] = "cartesian"
    for key, name in (
        ("inverse", "Inverse"),
        ("additional_parameters", "AdditionalParameters"),
    ):
        if node.get(key) is not None:
            group.create_dataset(name, data=node[key])


def _write_x5(path: Path, nodes: list, chains: tuple = ()) -> Path:
    """Write an X5 file as nitransforms writes transforms and chains."""
    with h5py.File(path, "w") as f:
        f.attrs["Format"] = "X5"
        f.attrs["Version"] = np.uint16(1)
        group = f.create_group("TransformGroup")
        for i, node in enumerate(nodes):
            _write_node(group.create_group(str(i)), node)
        if chains:
            cgroup = f.create_group("TransformChain")
            for i, chain in enumerate(chains):
                cgroup.create_dataset(str(i), data="/".join(map(str, chain)))
    return path


def _linear(matrix: np.ndarray = AFFINE, **kwargs) -> dict:
    node = dict(
        type="linear",
        subtype="affine",
        representation="matrix",
        transform=matrix,
        metadata={"WrittenBy": "NiTransforms 25.1.0"},
        domain=(SHAPE, VOX2RAS),
    )
    node.update(kwargs)
    return node


def _field(representation: str = "displacements", **kwargs) -> dict:
    node = dict(
        type="nonlinear",
        subtype="densefield",
        representation=representation,
        transform=_ramp(),
        metadata={"WrittenBy": "NiTransforms 25.1.0"},
        domain=(SHAPE, VOX2RAS),
    )
    node.update(kwargs)
    return node


@pytest.fixture
def linear_x5(tmp_path: Path) -> Path:
    return _write_x5(tmp_path / "affine.x5", [_linear(inverse=np.eye(4))])


@pytest.fixture
def field_x5(tmp_path: Path) -> Path:
    return _write_x5(tmp_path / "warp.x5", [_field()])


@pytest.fixture
def chain_x5(tmp_path: Path) -> Path:
    return _write_x5(
        tmp_path / "chain.x5", [_linear(), _field()], chains=[(0, 1), (1, 0)]
    )


# ----------------------------------------------------------------------
#   LINEAR
# ----------------------------------------------------------------------


def test_linear_is_dispatched_by_extension_and_content(
    linear_x5: Path,
) -> None:
    xform = io.load(linear_x5)
    assert type(xform) is X5Transform
    assert io.transformations.sniff(linear_x5) is X5Transform


def test_linear_is_a_ras_to_ras_affine(linear_x5: Path) -> None:
    xform = io.load(linear_x5)
    assert len(xform) == 1
    affine = xform[0]
    assert isinstance(affine, xforms.Affine)
    assert isinstance(affine.input, systems.RASmm)
    assert isinstance(affine.output, systems.RASmm)
    np.testing.assert_allclose(affine.matrix, AFFINE[:3])
    points = _ras(IJK)
    np.testing.assert_allclose(
        _apply(xform, points), _affine_apply(AFFINE, points)
    )


def test_linear_keeps_its_raw_node(linear_x5: Path) -> None:
    xform = io.load(linear_x5)
    node = xform.metadata.raw.nodes[0]
    assert node.metadata == {"WrittenBy": "NiTransforms 25.1.0"}
    assert node.subtype == "affine"
    assert node.representation == "matrix"
    assert node.dimension_kinds == KINDS
    assert node.domain.size == SHAPE
    assert node.domain.coordinates == "cartesian"
    np.testing.assert_allclose(node.domain.mapping, VOX2RAS)
    np.testing.assert_allclose(node.inverse, np.eye(4))
    assert xform.metadata.raw.header.version == 1


def test_untouched_file_is_written_back_unchanged(
    chain_x5: Path, tmp_path: Path
) -> None:
    xform = io.load(chain_x5)
    out = tmp_path / "out.x5"
    xform.save(out)
    again = io.load(out)
    assert again.metadata.raw.header.chains == [(0, 1), (1, 0)]
    assert len(again.metadata.raw.nodes) == 2
    for a, b in zip(again.metadata.raw.nodes, xform.metadata.raw.nodes):
        assert a.type == b.type
        assert a.metadata == b.metadata
        assert a.dimension_kinds == b.dimension_kinds
        np.testing.assert_allclose(a.transform, b.transform)
        np.testing.assert_allclose(a.domain.mapping, b.domain.mapping)
    with h5py.File(out, "r") as f:
        assert f.attrs["Format"] == "X5"
        assert f.attrs["Version"] == 1
        assert f["TransformChain/1"][()] == b"1/0"
    # The fixture is written by h5py in the order that the writer uses,
    # with contiguous datasets, so the copy has the same bytes. A file
    # with chunked or compressed datasets is written back contiguous.
    assert out.read_bytes() == chain_x5.read_bytes()


def test_bytes_and_file_objects_round_trip(linear_x5: Path) -> None:
    xform = io.load(linear_x5)
    content = xform.to_bytes()
    again = X5Transform.from_bytes(content)
    np.testing.assert_allclose(again[0].matrix, AFFINE[:3])
    buffer = _io.BytesIO()
    xform.save(buffer)
    buffer.seek(0)
    again = io.transformations.load(buffer)
    assert type(again) is X5Transform
    node = again.metadata.raw.nodes[0]
    assert node.metadata == xform.metadata.raw.nodes[0].metadata


# ----------------------------------------------------------------------
#   NONLINEAR
# ----------------------------------------------------------------------


def test_displacements_move_each_voxel_centre_by_its_vector(
    field_x5: Path,
) -> None:
    """Displacements map x -> x + u(x) in RAS mm."""
    xform = io.load(field_x5)
    assert len(xform) == 1
    field = xform[0]
    assert isinstance(field, X5DisplacementField)
    assert isinstance(field.input, systems.RASmm)
    assert isinstance(field.output, systems.RASmm)
    points = _ras(IJK)
    expected = points + _ramp()[tuple(IJK.T)]
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)


def test_deformations_map_each_voxel_centre_to_its_coordinates(
    tmp_path: Path,
) -> None:
    path = _write_x5(tmp_path / "def.x5", [_field("deformations")])
    xform = io.load(path)
    assert isinstance(xform[0], X5CoordinatesField)
    points = _ras(IJK)
    expected = _ramp()[tuple(IJK.T)]
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)


def test_vector_axis_is_found_by_its_dimension_kind(tmp_path: Path) -> None:
    node = _field(
        transform=np.moveaxis(_ramp(), -1, 0),
        kinds=("vector", "space", "space", "space"),
    )
    xform = io.load(_write_x5(tmp_path / "vfirst.x5", [node]))
    points = _ras(IJK)
    expected = points + _ramp()[tuple(IJK.T)]
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)


def test_fields_are_read_lazily(field_x5: Path) -> None:
    xform = X5Transform.from_file(field_x5)
    node = xform.metadata.raw.nodes[0]
    # The field is read from the file when it is used, and the small
    # datasets are read at once.
    assert isinstance(node.transform, DelayedH5Array)
    assert isinstance(node.domain.mapping, np.ndarray)
    assert "_cache_transformations" not in vars(xform)
    points = _ras(IJK)
    expected = points + _ramp()[tuple(IJK.T)]
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)
    # Decoding the chain leaves the record as it was read.
    assert xform.metadata.raw.nodes[0] is node
    assert isinstance(node.transform, DelayedH5Array)


def test_fields_can_be_read_at_once(field_x5: Path) -> None:
    xform = X5Transform.from_file(field_x5, load=True)
    assert isinstance(xform.metadata.raw.nodes[0].transform, np.ndarray)
    # A stream cannot be opened again by name, so its fields are read at
    # once, and no open file is kept.
    with open(field_x5, "rb") as stream:
        xform = X5Transform.from_file(stream)
    assert isinstance(xform.metadata.raw.nodes[0].transform, np.ndarray)
    points = _ras(IJK)
    expected = points + _ramp()[tuple(IJK.T)]
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)


def test_no_open_file_is_kept(field_x5: Path) -> None:
    with pytest.raises(TypeError, match="keep_open"):
        X5Transform.from_file(field_x5, keep_open=True)
    xform = X5Transform.from_file(field_x5)
    assert not [
        value for value in vars(xform).values() if isinstance(value, h5py.File)
    ]


def test_decoded_matrices_are_read_only(linear_x5: Path) -> None:
    # The writer writes the record, not the decoded chain, so an edit in
    # place would be lost. A new chain is assigned instead.
    xform = io.load(linear_x5)
    with pytest.raises(ValueError, match="read-only"):
        xform[0].matrix[0, 0] = 2.0
    np.testing.assert_allclose(xform.metadata.raw.nodes[0].transform, AFFINE)


def test_a_record_is_never_changed_in_place(
    chain_x5: Path, tmp_path: Path
) -> None:
    xform = io.load(chain_x5)
    record = xform.metadata.raw
    copy = xform.metadata.to_raw()
    assert copy is not record and copy.nodes[0] is not record.nodes[0]
    copy.header.attrs["Edited"] = "yes"
    copy.nodes[0].metadata["Edited"] = True
    copy.nodes[1].metadata = {"edited": True}
    assert "Edited" not in record.header.attrs
    assert record.nodes[1].metadata == {"WrittenBy": "NiTransforms 25.1.0"}
    assert record.nodes[0].metadata == {"WrittenBy": "NiTransforms 25.1.0"}
    # Encoding a new chain builds a new record.
    xform.transformations = [xform[1]]
    xform.save(tmp_path / "out.x5")
    assert record.header.chains == [(0, 1), (1, 0)]
    assert len(record.nodes) == 2


def test_a_copy_made_with_replace_encodes_its_elements_again(
    chain_x5: Path, tmp_path: Path
) -> None:
    # Only the elements that a transformation decoded from its own record
    # are written as their nodes. The copy decodes its own elements, so an
    # element of the original is encoded again and loses its node metadata.
    xform = io.load(chain_x5)
    copy = replace(xform, transformations=[xform[1]])
    copy.save(tmp_path / "copy.x5")
    assert io.load(tmp_path / "copy.x5").metadata.raw.nodes[0].metadata is None
    xform.transformations = [xform[1]]
    xform.save(tmp_path / "same.x5")
    node = io.load(tmp_path / "same.x5").metadata.raw.nodes[0]
    assert node.metadata == {"WrittenBy": "NiTransforms 25.1.0"}


def test_a_selection_is_checked_against_a_record(chain_x5: Path) -> None:
    record = X5Metadata.load(chain_x5).raw
    with pytest.raises(ParserContentError, match="no chain 2"):
        X5Transform.from_raw(record, chain=2)
    with pytest.raises(ValueError, match="not both"):
        X5Transform.from_raw(record, chain=0, position=0)
    # An index assigned later is checked when the selection is read.
    xform = X5Transform.from_raw(record)
    xform.position = 5
    with pytest.raises(ParserContentError, match="no transform 5"):
        _ = xform.selection


def test_only_the_selected_chain_is_written_when_assigned(
    chain_x5: Path, tmp_path: Path
) -> None:
    xform = X5Transform.from_file(chain_x5, chain=1)
    xform.save(tmp_path / "whole.x5")
    whole = io.load(tmp_path / "whole.x5").metadata.raw
    assert whole.header.chains == [(0, 1), (1, 0)]
    xform.transformations = list(xform)
    xform.save(tmp_path / "selected.x5")
    selected = io.load(tmp_path / "selected.x5").metadata.raw
    assert selected.header.chains == [(0, 1)]
    assert [node.type for node in selected.nodes] == ["nonlinear", "linear"]


def test_other_metadata_gives_another_chain(chain_x5: Path) -> None:
    xform = io.load(chain_x5)
    first = xform.transformations
    assert isinstance(first[0], xforms.Affine)
    record = xform.metadata.to_raw()
    record.header.chains = [(1,)]
    xform.metadata = X5Metadata.from_raw(record)
    assert isinstance(xform[0], X5DisplacementField)
    assert xform.transformations is not first
    # Selecting another chain also drops the decoded chain.
    xform.chain = 0
    assert isinstance(xform[0], X5DisplacementField)
    xform.metadata = io.load(chain_x5).metadata
    xform.chain = 1
    assert xform.selection == (1, 0)
    assert isinstance(xform[0], X5DisplacementField)


def test_a_record_is_read_as_a_transformation(chain_x5: Path) -> None:
    record = X5Metadata.load(chain_x5).raw
    assert isinstance(record, X5Raw)
    xform = X5Transform.from_raw(record, chain=1)
    assert xform.metadata.raw is record
    assert xform.selection == (1, 0)
    assert X5Transform.from_any(record).selection == (0, 1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(domain=None), "Domain"),
        (dict(representation=None), "Representation"),
        (dict(transform=_ramp()[1:]), "Domain says"),
    ],
)
def test_malformed_fields_are_refused(
    tmp_path: Path, kwargs: dict, match: str
) -> None:
    xform = io.load(_write_x5(tmp_path / "bad.x5", [_field(**kwargs)]))
    with pytest.raises(ParserContentError, match=match):
        _ = xform.transformations


# ----------------------------------------------------------------------
#   B-SPLINES
# ----------------------------------------------------------------------

# The knots lie on an oblique grid, so coefficients in the wrong frame
# fail.
KNOTS = np.array(
    [
        [3.5, -1.0, 0.5, -12.0],
        [1.0, 4.0, -1.5, 7.0],
        [-0.5, 1.0, 5.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _cubic(d: np.ndarray) -> np.ndarray:
    """The centred cubic B-spline of nitransforms (_cubic_bspline)."""
    d = np.abs(d)
    near = (4.0 - 6.0 * d**2 + 3.0 * d**3) / 6.0
    far = np.clip(2.0 - d, 0.0, None) ** 3 / 6.0
    return np.where(d < 1.0, near, far)


def _bspline_map(
    coeffs: np.ndarray, knots: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """x + sum_k c_k B3(i(x) - k), as BSplineFieldTransform.map computes it."""
    ijk = _affine_apply(np.linalg.inv(knots), points)
    grid = np.stack(
        np.meshgrid(*map(np.arange, coeffs.shape[:-1]), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    weights = _cubic(ijk[:, None, :] - grid[None, :, :]).prod(-1)
    return points + weights @ coeffs.reshape(-1, 3).astype(np.float64)


def _bspline(**kwargs) -> dict:
    """A node as BSplineFieldTransform.to_x5 writes it."""
    node = dict(
        type="nonlinear",
        subtype="bspline",
        representation="coefficients",
        transform=_ramp() - 2.0,
        metadata={"WrittenBy": "NiTransforms 25.1.0"},
        domain=((20, 20, 20), np.diag([2.0, 2.0, 2.0, 1.0])),
        additional_parameters=KNOTS,
    )
    node.update(kwargs)
    return node


def _knot_probes() -> np.ndarray:
    """RAS points on the knots, between them, near the edges and outside."""
    rng = np.random.default_rng(0)
    ijk = np.concatenate(
        [IJK, rng.uniform(-3.0, np.add(SHAPE, 2.0), size=(64, 3))]
    )
    return _affine_apply(KNOTS, ijk)


def test_bspline_is_a_cubic_spline_of_ras_displacements(
    tmp_path: Path,
) -> None:
    """The coefficients are RAS displacements on the knot grid placed by
    AdditionalParameters; the Domain is unused.
    """
    path = _write_x5(tmp_path / "bspline.x5", [_bspline()])
    xform = io.load(path)
    field = xform[0]
    assert isinstance(field, X5BSplineField)
    assert isinstance(field.input, systems.RASmm)
    assert isinstance(field.output, systems.RASmm)
    assert field.displacement.store == "coefficients"
    assert int(field.displacement.degree) == 3
    points = _knot_probes()
    expected = _bspline_map(_ramp() - 2.0, KNOTS, points)
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)
    # Far from the knots, nothing moves.
    far = _affine_apply(KNOTS, np.array([[-5.0, 2.0, 2.0], [2.0, 9.0, 2.0]]))
    np.testing.assert_allclose(_apply(xform, far), far, atol=1e-6)


def test_bspline_agrees_with_the_itk_reader(tmp_path: Path) -> None:
    """An ITK B-spline and the same spline in X5 map points alike."""
    block = io.load(data_dir / "itk_bspline3d.h5")[-1]
    vox2lps = block.voxel2lps.to(xforms.Affine).homogeneous_matrix
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    vox2ras = flip @ vox2lps
    coeffs = np.asarray(block.field) @ (flip[:3, :3] @ vox2lps[:3, :3]).T
    path = _write_x5(
        tmp_path / "itk.x5",
        [_bspline(transform=coeffs, additional_parameters=vox2ras)],
    )
    xform = io.load(path)
    rng = np.random.default_rng(1)
    ras = _affine_apply(vox2ras, rng.uniform(-2.0, 9.0, size=(64, 3)))
    via_itk = _apply(block, ras @ flip[:3, :3]) @ flip[:3, :3]
    np.testing.assert_allclose(_apply(xform, ras), via_itk, atol=1e-6)
    np.testing.assert_allclose(
        _apply(xform, ras), _bspline_map(coeffs, vox2ras, ras), atol=1e-6
    )


def test_bspline_is_written_from_scratch(tmp_path: Path) -> None:
    field = X5BSplineField.from_ras(_ramp(), KNOTS)
    out = tmp_path / "bspline.x5"
    X5Transform(transformations=[field]).save(out)
    with h5py.File(out, "r") as f:
        node = f["TransformGroup/0"]
        assert node.attrs["Type"] == "nonlinear"
        assert node.attrs["SubType"] == "bspline"
        assert node.attrs["Representation"] == "coefficients"
        np.testing.assert_allclose(node["Transform"], _ramp(), atol=1e-5)
        np.testing.assert_allclose(node["AdditionalParameters"], KNOTS)
        # nitransforms requires a Domain, so the knot grid stands in.
        np.testing.assert_allclose(node["Domain/Mapping"], KNOTS)
        np.testing.assert_array_equal(node["Domain/Size"], SHAPE)
    points = _knot_probes()
    np.testing.assert_allclose(
        _apply(io.load(out), points),
        _bspline_map(_ramp(), KNOTS, points),
        atol=1e-5,
    )


def test_bspline_read_is_written_back_with_its_domain(
    tmp_path: Path,
) -> None:
    xform = io.load(_write_x5(tmp_path / "bspline.x5", [_bspline()]))
    xform.transformations = [xform[0]]
    out = tmp_path / "out.x5"
    xform.save(out)
    again = io.load(out)
    node = again.metadata.raw.nodes[0]
    assert node.domain.size == (20, 20, 20)
    assert node.metadata == {"WrittenBy": "NiTransforms 25.1.0"}
    assert isinstance(again[0], X5BSplineField)


@pytest.mark.parametrize(
    "kwargs",
    [dict(degree=1), dict(bound="nearest")],
    ids=["linear", "nearest"],
)
def test_splines_x5_cannot_hold_are_refitted(
    tmp_path: Path, kwargs: dict
) -> None:
    # X5 stores cubic zero-boundary coefficients, so others are refitted.
    out = tmp_path / "refit.x5"
    ras2vox, field, vox2ras = X5BSplineField.from_ras(_ramp(), KNOTS)
    spline = xforms.DisplacementField(
        data=field.data,
        input=field.input,
        output=field.output,
        store="coefficients",
        **{"degree": 3, "bound": "constant", **kwargs},
    )
    chain = xforms.Sequence(
        transformations=[ras2vox, spline, vox2ras],
        input=systems.RASmm(),
        output=systems.RASmm(),
    )
    X5Transform(transformations=[chain]).save(out)
    again = io.load(out)[0]
    assert isinstance(again, X5BSplineField)
    back = again[1]
    assert (back.store, int(back.degree)) == ("coefficients", 3)
    np.testing.assert_allclose(
        np.asarray(back.field), np.asarray(spline.field), atol=1e-4
    )


def test_stored_spline_coefficients_are_not_refitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cubic zero-boundary spline is written as is, without refitting.
    from brainhops.datamodel._transformations import concrete

    def refuse(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("the stored coefficients were refitted")

    monkeypatch.setattr(concrete, "value2coeff_field", refuse)
    monkeypatch.setattr(concrete, "coeff2value_field", refuse)
    out = tmp_path / "spline.x5"
    X5Transform(
        transformations=[X5BSplineField.from_ras(_ramp(), KNOTS)]
    ).save(out)
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(
            f["TransformGroup/0/Transform"], _ramp(), atol=1e-5
        )


def test_sampled_displacements_stay_sampled(tmp_path: Path) -> None:
    # The `store` flag selects the encoding; values are written as such.
    out = tmp_path / "dense.x5"
    X5Transform(
        transformations=[X5DisplacementField.from_ras(_ramp(), VOX2RAS)]
    ).save(out)
    with h5py.File(out, "r") as f:
        node = f["TransformGroup/0"]
        assert node.attrs["Representation"] == "displacements"
        np.testing.assert_allclose(node["Transform"], _ramp(), atol=1e-5)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (
            dict(additional_parameters=None),
            ParserContentError,
            "AdditionalParameters",
        ),
        (
            dict(transform=_ramp()[..., :2]),
            ParserContentError,
            "3-vector",
        ),
        (
            dict(representation="displacements"),
            ParserNotImplementedError,
            "coefficients",
        ),
        (
            dict(additional_parameters=np.eye(3)),
            ParserNotImplementedError,
            "4x4",
        ),
    ],
)
def test_malformed_bsplines_are_refused(
    tmp_path: Path, kwargs: dict, error: type, match: str
) -> None:
    xform = io.load(_write_x5(tmp_path / "bad.x5", [_bspline(**kwargs)]))
    with pytest.raises(error, match=match):
        _ = xform.transformations


# ----------------------------------------------------------------------
#   CHAINS
# ----------------------------------------------------------------------


def test_chain_is_a_sequence_in_application_order(chain_x5: Path) -> None:
    """The chain '0/1' applies f1(f0(x)), as Sequence([t0, t1]) does."""
    xform = io.load(chain_x5)
    assert xform.selection == (0, 1)
    assert isinstance(xform[0], xforms.Affine)
    assert isinstance(xform[1], X5DisplacementField)
    # Points that the affine sends to voxel centres of the field.
    targets = _ras(IJK)
    points = _affine_apply(np.linalg.inv(AFFINE), targets)
    expected = targets + _ramp()[tuple(IJK.T)]
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)


def test_chain_and_position_select_what_is_read(chain_x5: Path) -> None:
    second = X5Transform.from_file(chain_x5, chain=1)
    assert second.selection == (1, 0)
    assert isinstance(second[0], X5DisplacementField)
    field = X5Transform.from_file(chain_x5, position=1)
    assert len(field) == 1
    assert isinstance(field[0], X5DisplacementField)
    with pytest.raises(ParserContentError, match="no chain 2"):
        X5Transform.from_file(chain_x5, chain=2)
    with pytest.raises(ParserContentError, match="no transform 5"):
        X5Transform.from_file(chain_x5, position=5)


def test_unchained_nodes_read_the_first_with_a_warning(
    tmp_path: Path,
) -> None:
    path = _write_x5(tmp_path / "two.x5", [_linear(), _field()])
    xform = io.load(path)
    with pytest.warns(UserWarning, match="no chain"):
        chain = xform.transformations
    assert len(chain) == 1
    assert isinstance(chain[0], xforms.Affine)


def test_chain_is_written_from_scratch(tmp_path: Path) -> None:
    """A chain is written as one node per element plus a /TransformChain."""
    nifti = pytest.importorskip("nibabel")
    from brainhops.io.transformations.nifti import NiftiRASDisplacementField

    image = nifti.Nifti1Image(_ramp()[:, :, :, None, :], VOX2RAS)
    image.header.set_intent(1006)
    nifti.save(image, str(tmp_path / "warp.nii.gz"))
    warp = io.load(tmp_path / "warp.nii.gz")
    assert isinstance(warp, NiftiRASDisplacementField)
    affine = xforms.Affine(
        matrix=AFFINE[:3], input=systems.RASmm(), output=systems.RASmm()
    )
    xform = X5Transform(transformations=[affine, warp])
    out = tmp_path / "out.x5"
    xform.save(out)

    with h5py.File(out, "r") as f:
        assert f.attrs["Format"] == "X5"
        assert f.attrs["Version"] == 1
        assert f["TransformGroup/0"].attrs["Type"] == "linear"
        assert f["TransformGroup/1"].attrs["Type"] == "nonlinear"
        assert f["TransformGroup/1"].attrs["Representation"] == "displacements"
        np.testing.assert_allclose(f["TransformGroup/0/Transform"], AFFINE)
        np.testing.assert_allclose(
            f["TransformGroup/1/Domain/Mapping"], VOX2RAS
        )
        np.testing.assert_allclose(
            f["TransformGroup/1/Transform"], _ramp(), atol=1e-5
        )
        assert f["TransformChain/0"][()] == b"0/1"

    again = io.load(out)
    targets = _ras(IJK)
    points = _affine_apply(np.linalg.inv(AFFINE), targets)
    np.testing.assert_allclose(
        _apply(again, points), _apply(xform, points), atol=1e-5
    )


def test_reused_nodes_keep_their_metadata(
    chain_x5: Path, tmp_path: Path
) -> None:
    xform = io.load(chain_x5)
    xform.transformations = [xform[1]]
    out = tmp_path / "field.x5"
    xform.save(out)
    again = io.load(out)
    assert len(again.metadata.raw.nodes) == 1
    assert again.metadata.raw.header.chains == []
    node = again.metadata.raw.nodes[0]
    assert node.metadata == {"WrittenBy": "NiTransforms 25.1.0"}
    assert isinstance(again[0], X5DisplacementField)


def test_a_coordinates_field_is_written_as_deformations(
    tmp_path: Path,
) -> None:
    field = X5CoordinatesField.from_ras(_ramp(), VOX2RAS)
    out = tmp_path / "def.x5"
    X5Transform(transformations=[field]).save(out)
    with h5py.File(out, "r") as f:
        node = f["TransformGroup/0"]
        assert node.attrs["Representation"] == "deformations"
        np.testing.assert_allclose(node["Transform"], _ramp())
        np.testing.assert_allclose(node["Domain/Mapping"], VOX2RAS)
        assert "TransformChain" not in f


def test_coordinate_coefficients_are_written_as_deformations(
    tmp_path: Path,
) -> None:
    # X5 stores sampled coordinates, so a coefficient field is written as
    # values.
    ras2vox, field = X5CoordinatesField.from_ras(_ramp(), VOX2RAS)
    spline = field.to(degree=3).to(store="coefficients")
    chain = xforms.Sequence(
        transformations=[ras2vox, spline],
        input=systems.RASmm(),
        output=systems.RASmm(),
    )
    out = tmp_path / "def.x5"
    X5Transform(transformations=[chain]).save(out)
    with h5py.File(out, "r") as f:
        node = f["TransformGroup/0"]
        assert node.attrs["Representation"] == "deformations"
        np.testing.assert_allclose(node["Transform"], _ramp(), atol=1e-6)
    again = io.load(out)
    assert isinstance(again[0], X5CoordinatesField)
    np.testing.assert_allclose(
        np.asarray(again[0][1].field), np.asarray(field.field), atol=1e-6
    )


def test_non_ras_transformations_are_refused(tmp_path: Path) -> None:
    out = tmp_path / "lps.x5"
    affine = xforms.Affine(
        matrix=AFFINE[:3], input=systems.LPSmm(), output=systems.LPSmm()
    )
    with pytest.raises(UnrepresentableTransformationError, match="RAS"):
        X5Transform(transformations=[affine]).save(out)
    assert not out.exists()
    with pytest.raises(UnrepresentableTransformationError):
        X5Transform(transformations=[xforms.Affine(matrix=AFFINE[:3])]).save(
            out
        )


# ----------------------------------------------------------------------
#   NOT SUPPORTED, BUT KEPT
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "node, match",
    [
        (dict(type="composite", transform=np.zeros(1)), "composite"),
        (
            _linear(np.stack([AFFINE, AFFINE]), array_length=2),
            "stack",
        ),
        (_field(subtype="dct"), "dct"),
    ],
)
def test_unsupported_nodes_are_refused_but_round_trip(
    tmp_path: Path, node: dict, match: str
) -> None:
    path = _write_x5(tmp_path / "unsupported.x5", [node])
    xform = io.load(path)
    with pytest.raises(ParserNotImplementedError, match=match):
        _ = xform.transformations
    out = tmp_path / "out.x5"
    xform.save(out)
    again = io.load(out)
    written = again.metadata.raw.nodes[0]
    assert written.type == node["type"]
    np.testing.assert_allclose(written.transform, node["transform"])


# ----------------------------------------------------------------------
#   FSLPY'S EARLIER LAYOUT
# ----------------------------------------------------------------------


def _write_fslpy_affine(group, matrix: np.ndarray) -> None:  # noqa: ANN001
    group.attrs["Type"] = "affine"
    group.create_dataset("Matrix", data=matrix)
    group.create_dataset("Inverse", data=np.linalg.inv(matrix))


def _write_fslpy_space(group, shape, vox2ras) -> None:  # noqa: ANN001
    group.attrs["Type"] = "image"
    group.attrs["Size"] = np.asarray(shape, np.uint32)
    group.attrs["Scales"] = np.ones(3, np.float32)
    _write_fslpy_affine(group.create_group("Mapping"), vox2ras)


def _write_fslpy(path: Path, kind: str) -> Path:
    """Write a file as fsl.transform.x5.write(Non)LinearX5 does."""
    with h5py.File(path, "w") as f:
        f.attrs["Format"] = "X5"
        f.attrs["Version"] = "0.1.0"
        f.attrs["Metadata"] = json.dumps({"fslpy": "3.29.1"})
        _write_fslpy_space(f.create_group("A"), SHAPE, VOX2RAS)
        _write_fslpy_space(f.create_group("B"), SHAPE, np.eye(4))
        if kind == "linear":
            f.attrs["Type"] = "linear"
            _write_fslpy_affine(f.create_group("Transform"), AFFINE)
        else:
            f.attrs["Type"] = "nonlinear"
            group = f.create_group("Transform")
            group.attrs["Type"] = "deformation"
            group.attrs["SubType"] = kind
            group.create_dataset("Matrix", data=_ramp())
            _write_fslpy_affine(group.create_group("Mapping"), VOX2RAS)
    return path


def test_fslpy_linear_is_read(tmp_path: Path) -> None:
    xform = io.load(_write_fslpy(tmp_path / "fsl.x5", "linear"))
    assert xform.metadata.raw.header.legacy
    node = xform.metadata.raw.nodes[0]
    assert node.metadata == {"fslpy": "3.29.1"}
    np.testing.assert_allclose(xform[0].matrix, AFFINE[:3])
    np.testing.assert_allclose(node.inverse, np.linalg.inv(AFFINE))


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_fslpy_nonlinear_is_read(tmp_path: Path, kind: str) -> None:
    xform = io.load(_write_fslpy(tmp_path / "fsl.x5", kind))
    # The deformation is read when it is used, as in the current layout.
    assert isinstance(xform.metadata.raw.nodes[0].transform, DelayedH5Array)
    points = _ras(IJK)
    expected = _ramp()[tuple(IJK.T)] + (points if kind == "relative" else 0)
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)
    # The transformation is written back in the current layout.
    out = tmp_path / "out.x5"
    xform.save(out)
    with h5py.File(out, "r") as f:
        assert f.attrs["Version"] == 1
        assert "TransformGroup/0" in f
    np.testing.assert_allclose(
        _apply(io.load(out), points), expected, atol=1e-5
    )


# ----------------------------------------------------------------------
#   ITK HDF5 FILES ARE NOT X5 FILES
# ----------------------------------------------------------------------


def test_itk_h5_is_not_claimed() -> None:
    from brainhops.io.transformations.itk.h5 import H5Transform

    path = data_dir / "itk_affine3d.h5"
    assert X5Transform.sniff(path) == 0
    assert type(io.load(path)) is H5Transform


def test_x5_named_h5_is_read_as_x5(tmp_path: Path) -> None:
    path = _write_x5(tmp_path / "affine.h5", [_linear()])
    from brainhops.io.transformations.itk.h5 import H5Transform

    assert H5Transform.sniff(path) == 0
    assert type(io.load(path)) is X5Transform


# ----------------------------------------------------------------------
#   INTEROPERABILITY
# ----------------------------------------------------------------------


def _reference_image():  # noqa: ANN202
    nb = pytest.importorskip("nibabel")
    return nb.Nifti1Image(np.zeros(SHAPE, "uint8"), VOX2RAS)


def test_nitransforms_files_are_read_as_it_reads_them(
    tmp_path: Path,
) -> None:
    pytest.importorskip("nitransforms")
    from nitransforms.linear import Affine
    from nitransforms.manip import TransformChain
    from nitransforms.nonlinear import DenseFieldTransform

    ref = _reference_image()
    affine = Affine(AFFINE)
    field = DenseFieldTransform(_ramp(), is_deltas=True, reference=ref)
    chain = TransformChain([affine, field])
    path = tmp_path / "chain.x5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain.to_filename(path)
    xform = io.load(path)
    targets = _ras(IJK)
    points = _affine_apply(np.linalg.inv(AFFINE), targets)
    np.testing.assert_allclose(
        _apply(xform, points), chain.map(points), atol=1e-5
    )

    # brainhops files are read back by nitransforms.
    out = tmp_path / "rev.x5"
    X5Transform(transformations=list(xform)[::-1]).save(out)
    back = TransformChain.from_filename(out)
    np.testing.assert_allclose(
        back.map(targets),
        _affine_apply(AFFINE, targets + _ramp()[tuple(IJK.T)]),
        atol=1e-5,
    )


@pytest.mark.parametrize("deltas", [True, False])
def test_nitransforms_fields_match(tmp_path: Path, deltas: bool) -> None:
    pytest.importorskip("nitransforms")
    from nitransforms.io.x5 import to_filename
    from nitransforms.nonlinear import DenseFieldTransform

    field = DenseFieldTransform(
        _ramp(), is_deltas=deltas, reference=_reference_image()
    )
    path = tmp_path / "field.x5"
    to_filename(path, [field.to_x5()])
    xform = io.load(path)
    points = _ras(IJK)
    np.testing.assert_allclose(
        _apply(xform, points), field.map(points), atol=1e-5
    )


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_fslpy_files_are_read_as_it_reads_them(
    tmp_path: Path, kind: str
) -> None:
    fx5 = pytest.importorskip("fsl.transform.x5")
    from fsl.data.image import Image
    from fsl.transform.nonlinear import DeformationField

    src_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    src_affine[:3, 3] = -100
    src = Image(np.zeros((100, 100, 100), "float32"), xform=src_affine)
    ref = Image(np.zeros(SHAPE, "float32"), xform=VOX2RAS)

    path = tmp_path / "linear.x5"
    fx5.writeLinearX5(str(path), AFFINE, src, ref)
    np.testing.assert_allclose(io.load(path)[0].matrix, AFFINE[:3])

    field = DeformationField(
        _ramp() * 5,
        src=src,
        ref=ref,
        srcSpace="world",
        refSpace="world",
        defType=kind,
        xform=VOX2RAS,
    )
    path = tmp_path / "field.x5"
    fx5.writeNonLinearX5(str(path), field)
    points = _ras(IJK)
    np.testing.assert_allclose(
        _apply(io.load(path), points),
        field.transform(points, "world", "world"),
        atol=1e-5,
    )


def test_nitransforms_bsplines_match(tmp_path: Path) -> None:
    pytest.importorskip("nitransforms")
    nb = pytest.importorskip("nibabel")
    from nitransforms.io.x5 import to_filename
    from nitransforms.nonlinear import BSplineFieldTransform

    rng = np.random.default_rng(2)
    coeffs = rng.normal(scale=3.0, size=(*SHAPE, 3)).astype("float32")
    spline = BSplineFieldTransform(
        nb.Nifti1Image(coeffs, KNOTS), reference=_reference_image()
    )
    path = tmp_path / "bspline.x5"
    to_filename(path, [spline.to_x5()])
    xform = io.load(path)
    assert isinstance(xform[0], X5BSplineField)
    points = _knot_probes()
    expected = spline.map(points)
    # nitransforms weighs the knots in single precision.
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-5)

    # brainhops files are read back by nitransforms.
    out = tmp_path / "out.x5"
    X5Transform(transformations=[X5BSplineField.from_ras(coeffs, KNOTS)]).save(
        out
    )
    back = BSplineFieldTransform.from_filename(out)
    np.testing.assert_allclose(back.map(points), expected, atol=1e-5)


def test_assigned_systems_are_encoded(chain_x5: Path) -> None:
    # Systems that were assigned override the record, so the chain is
    # encoded, and X5 only stores maps from RAS to RAS.
    xform = io.load(chain_x5)
    xform.input = systems.RASmm()
    record = xform.to_raw()
    assert record is not xform.metadata.raw
    assert record.header.chains == [(0, 1)]
    assert record.nodes[0] is xform.metadata.raw.nodes[0]
    xform.input = systems.LPSmm()
    with pytest.raises(UnrepresentableTransformationError, match="RAS"):
        xform.to_raw()
    # The options only select what is decoded.
    selected = X5Transform.from_file(chain_x5, chain=1)
    assert selected.to_raw() is selected.metadata.raw


def test_decoded_fields_are_read_only(field_x5: Path) -> None:
    from brainhops.backends import backend

    with backend("numpy"):
        vectors = io.load(field_x5)[0].displacement.data
    assert isinstance(vectors, np.ndarray)
    assert not vectors.flags.writeable
