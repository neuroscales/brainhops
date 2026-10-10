"""Tests for FreeSurfer non-linear morphs (.m3z).

No morph ships with the tests, so files are encoded byte for byte by
`_encode`, a literal transcription of `__m3zWrite` in FreeSurfer's
utils/gcamorph.cpp that is independent of the reader.
"""

import gzip
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

from brainhops import io
from brainhops.datamodel import transformations as xforms
from brainhops.io.base.parsers import (
    ParserContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.freesurfer.m3z import (
    GCAM_RAS,
    GCAM_VOX,
    M3zMetadata,
    M3zMorph,
    M3zRaw,
)

SHAPE = (4, 3, 5)
"""Shape of the node grid."""

SPACING = 2

# The source image has 1.5 x 1 x 2 mm voxels and permuted axes.
IMAGE = dict(
    valid=1,
    shape=(20, 18, 16),
    size=(1.5, 1.0, 2.0),
    xras=(0.0, 0.0, -1.0),
    yras=(1.0, 0.0, 0.0),
    zras=(0.0, -1.0, 0.0),
    cras=(3.0, -7.0, 11.0),
    fname=b"/subjects/bert/mri/norm.mgz",
)
# The atlas is LIA with 1 mm voxels, and its shape is the node grid
# times the spacing.
ATLAS = dict(
    valid=1,
    shape=(8, 6, 10),
    size=(1.0, 1.0, 1.0),
    xras=(-1.0, 0.0, 0.0),
    yras=(0.0, 0.0, -1.0),
    zras=(0.0, 1.0, 0.0),
    cras=(0.5, 17.0, -18.0),
    fname=b"/usr/local/freesurfer/average/RB_all.gca",
)
LINEAR = np.array(
    [
        [1.1, 0.02, -0.01, -3.5],
        [-0.03, 0.95, 0.12, 21.25],
        [0.0, -0.1, 1.05, -12.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ----------------------------------------------------------------------
#   ENCODING, AS FREESURFER DOES
# ----------------------------------------------------------------------


def _vox2ras(geom: tx.Dict[str, tx.Any]) -> np.ndarray:
    """Transcribe VGgetVoxelToRasXform, which maps voxel shape / 2 to c_ras."""
    m = np.eye(4)
    m[:3, 0] = np.multiply(geom["xras"], geom["size"][0])
    m[:3, 1] = np.multiply(geom["yras"], geom["size"][1])
    m[:3, 2] = np.multiply(geom["zras"], geom["size"][2])
    centre = np.asarray(geom["shape"], float) / 2.0
    m[:3, 3] = np.asarray(geom["cras"]) - m[:3, :3] @ centre
    return m


def _positions(shape: tx.Tuple[int, int, int] = SHAPE) -> np.ndarray:
    """Return source voxel positions, affine in the node index plus a bump."""
    i, j, k = np.meshgrid(*map(np.arange, shape), indexing="ij")
    return np.stack(
        [
            2.0 + 1.5 * i + 0.25 * j + 0.01 * i * k,
            1.0 + 0.5 * k + 1.75 * j,
            3.0 + 0.8 * i - 0.3 * k,
        ],
        axis=-1,
    ).astype("float32")


def _geom(geom: tx.Dict[str, tx.Any]) -> bytes:
    """Transcribe VOL_GEOM::write."""
    out = struct.pack(">4i", geom["valid"], *geom["shape"])
    out += struct.pack(">3f", *geom["size"])
    for key in ("xras", "yras", "zras", "cras"):
        out += struct.pack(">3f", *geom[key])
    return out + geom["fname"].ljust(512, b"\0")


def _matrix(matrix: np.ndarray, keyword: str = "Matrix") -> str:
    return keyword + " " + " ".join(f"{v:10f}" for v in matrix.ravel())


def _encode(
    positions: np.ndarray,
    original: tx.Optional[np.ndarray] = None,
    spacing: int = SPACING,
    exp_k: float = 20.0,
    geometry: bool = True,
    gtype: tx.Optional[int] = GCAM_VOX,
    labels: bool = True,
    matrix: tx.Optional[np.ndarray] = LINEAR,
    xform_tag: int = 0,
    keyword: str = "Matrix",
    tail: bytes = b"",
    gz: bool = True,
) -> bytes:
    """Transcribe __m3zWrite."""
    width, height, depth = positions.shape[:3]
    if original is None:
        original = positions - 0.5
    out = struct.pack(">f", 1.0)
    out += struct.pack(">iiii", width, height, depth, spacing)
    out += struct.pack(">f", exp_k)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                out += struct.pack(">3f", *original[x, y, z])
                out += struct.pack(">3f", *positions[x, y, z])
                out += struct.pack(">3i", x // 2, y // 2, z // 2)
    if geometry:
        out += struct.pack(">i", 10) + _geom(IMAGE) + _geom(ATLAS)
    if gtype is not None:
        out += struct.pack(">ii", 11, gtype)
    if labels:
        out += struct.pack(">i", 12)
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    out += struct.pack(">i", 100 * x + 10 * y + z)
    if matrix is not None:
        text = _matrix(matrix, keyword).encode("ascii").ljust(1600, b"\0")
        out += struct.pack(">i", 31) + struct.pack(">iq", xform_tag, 1600)
        out += text
    out += tail
    return gzip.compress(out) if gz else out


def _write(tmp_path: Path, content: bytes, name: str = "t.m3z") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _apply(xform: tx.Any, points: np.ndarray) -> np.ndarray:
    """Map points through a transformation."""
    points = xforms.CoordinatesField(field=np.asarray(points, float))
    if isinstance(xform, (xforms.Sequence, list)):
        chain = list(xform)
    else:
        chain = [xform]
    out = xforms.Sequence(transformations=[points, *chain]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


def _affine_apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_read_nodes_in_freesurfer_order(tmp_path: Path) -> None:
    positions = _positions()
    morph = io.load(_write(tmp_path, _encode(positions)))
    assert isinstance(morph, M3zMorph)
    s = morph.metadata.raw
    assert s.shape == SHAPE
    assert s.spacing == SPACING
    assert s.exp_k == 20.0 and s.version == 1.0
    np.testing.assert_array_equal(s.positions, positions)
    np.testing.assert_array_equal(s.original, positions - 0.5)
    assert tuple(s.index[3, 2, 4]) == (1, 1, 2)
    assert s.labels[3, 1, 2] == 312
    assert s.type == GCAM_VOX and s.coordinates == GCAM_VOX
    assert s.tags == (10, 11, 12, 31)
    assert not s.invalid.any()


def test_read_geometries(tmp_path: Path) -> None:
    morph = io.load(_write(tmp_path, _encode(_positions())))
    image, atlas = morph.metadata.raw.image, morph.metadata.raw.atlas
    assert image.shape == IMAGE["shape"] and atlas.shape == ATLAS["shape"]
    assert image.filename == "/subjects/bert/mri/norm.mgz"
    assert atlas.filename.endswith("RB_all.gca")
    s = morph.metadata.raw
    np.testing.assert_allclose(s.image_geometry.vox2ras, _vox2ras(IMAGE))
    np.testing.assert_allclose(s.atlas_geometry.vox2ras, _vox2ras(ATLAS))
    node = _vox2ras(ATLAS) @ np.diag([SPACING] * 3 + [1])
    np.testing.assert_allclose(morph[0].matrix, np.linalg.inv(node)[:3])


@pytest.mark.parametrize("tag, keyword", [(0, "Matrix"), (33, "AutoAlign")])
def test_read_linear_transform(tmp_path: Path, tag: int, keyword: str) -> None:
    content = _encode(_positions(), xform_tag=tag, keyword=keyword)
    morph = io.load(_write(tmp_path, content))
    xform = morph.metadata.raw.xform
    assert xform.tag == tag
    np.testing.assert_allclose(xform.matrix, LINEAR, atol=1e-6)


def test_read_without_tags(tmp_path: Path) -> None:
    content = _encode(
        _positions(), geometry=False, gtype=None, labels=False, matrix=None
    )
    morph = io.load(_write(tmp_path, content))
    s = morph.metadata.raw
    assert s.tags == () and s.image is None and s.labels is None
    assert s.coordinates == GCAM_VOX and s.xform is None
    # The default FreeSurfer geometry is a centred 256^3 grid of 1 mm LIA
    # voxels.
    lia = _vox2ras(
        dict(
            shape=(256, 256, 256),
            size=(1, 1, 1),
            xras=(-1, 0, 0),
            yras=(0, 0, -1),
            zras=(0, 1, 0),
            cras=(0, 0, 0),
        )
    )
    np.testing.assert_allclose(s.image_geometry.vox2ras, lia)
    assert len(morph) == 3


def test_read_uncompressed(tmp_path: Path) -> None:
    positions = _positions()
    path = _write(tmp_path, _encode(positions, gz=False), "t.m3d")
    morph = io.load(path)
    assert isinstance(morph, M3zMorph)
    np.testing.assert_array_equal(morph.metadata.raw.positions, positions)


def test_invalid_nodes(tmp_path: Path) -> None:
    positions = _positions()
    original = positions - 0.5
    positions[1, 2, 3] = original[1, 2, 3] = 0
    content = _encode(positions, original=original)
    morph = io.load(_write(tmp_path, content))
    invalid = morph.metadata.raw.invalid
    assert invalid.sum() == 1 and invalid[1, 2, 3]


def test_truncated(tmp_path: Path) -> None:
    content = _encode(_positions(), gz=False)
    with pytest.raises(ParserContentError):
        M3zMorph.from_bytes(content[:200])
    with pytest.raises(ParserContentError):
        M3zMorph.from_bytes(content[: len(content) - 10])


# ----------------------------------------------------------------------
#   RECORD AND METADATA
# ----------------------------------------------------------------------


def test_the_field_is_a_view_of_the_record(tmp_path: Path) -> None:
    morph = io.load(_write(tmp_path, _encode(_positions())))
    assert isinstance(morph.metadata, M3zMetadata)
    record = morph.metadata.raw
    assert isinstance(record, M3zRaw)
    # The chain is decoded once, and its field holds a view of the array
    # of the record, which cannot be changed in place.
    assert morph.transformations is morph.transformations
    assert morph[1].data.base is record.positions
    assert not morph[1].data.flags.writeable
    for array in (record.positions, record.original, record.index):
        assert not array.flags.writeable
    assert not record.labels.flags.writeable
    with pytest.raises(ValueError):
        record.positions[0, 0, 0, 0] = 1.0


def test_an_unpickled_record_still_gives_a_read_only_field(
    tmp_path: Path,
) -> None:
    morph = io.load(_write(tmp_path, _encode(_positions())))
    metadata = pickle.loads(pickle.dumps(morph.metadata))
    morph = M3zMorph(metadata=metadata)
    assert not morph[1].data.flags.writeable
    np.testing.assert_array_equal(morph[1].data, _positions())


def test_a_copy_of_the_record_shares_its_arrays(tmp_path: Path) -> None:
    morph = io.load(_write(tmp_path, _encode(_positions())))
    record = morph.metadata.raw
    copy = morph.metadata.to_raw()
    assert copy is not record
    assert copy.positions is record.positions
    assert copy.labels is record.labels
    assert copy.xform is record.xform
    # The record of an untouched morph is written as it is.
    assert morph.to_raw() is record


def test_the_metadata_reads_and_writes_the_whole_file(tmp_path: Path) -> None:
    content = _encode(_positions(), gz=False)
    path = _write(tmp_path, gzip.compress(content))
    metadata = M3zMetadata.load(path)
    assert isinstance(io.metadata.MetadataFormat.load(path), M3zMetadata)
    np.testing.assert_array_equal(metadata.raw.positions, _positions())
    assert gzip.decompress(metadata.to_bytes()) == content
    assert metadata.to_bytes(compress=False) == content
    metadata.save(tmp_path / "copy.m3d")
    assert (tmp_path / "copy.m3d").read_bytes() == content
    with open(path, "rb") as f:
        assert M3zMetadata.from_fileobj(f).raw.shape == SHAPE
    with pytest.raises(WriterError):
        M3zMetadata().save(tmp_path / "empty.m3z")
    assert not (tmp_path / "empty.m3z").exists()


def test_other_metadata_gives_another_chain(tmp_path: Path) -> None:
    morph = io.load(_write(tmp_path, _encode(_positions())))
    chain = morph.transformations
    other = M3zMetadata.load(_write(tmp_path, _encode(_positions() * 2)))
    morph.metadata = other
    assert morph.transformations is not chain
    np.testing.assert_array_equal(morph[1].data, _positions() * 2)
    # An assigned chain is kept when the metadata changes.
    morph.transformations = list(chain)
    morph.metadata = M3zMetadata.from_raw(other.raw)
    assert morph.transformations == chain


def test_a_morph_reads_a_record_a_stream_and_bytes(tmp_path: Path) -> None:
    content = _encode(_positions())
    path = _write(tmp_path, content)
    record = M3zRaw.load(path)
    assert M3zMorph.from_raw(record).metadata.raw is record
    assert M3zMorph.from_any(record).metadata.raw is record
    with open(path, "rb") as f:
        morph = M3zMorph.load(f)
    np.testing.assert_array_equal(morph[1].data, _positions())
    morph = M3zMorph.from_bytes(content)
    np.testing.assert_array_equal(morph[1].data, _positions())


def test_a_morph_without_a_record_or_a_chain_is_not_written(
    tmp_path: Path,
) -> None:
    morph = M3zMorph()
    assert morph.metadata is None and morph.transformations == ()
    with pytest.raises(WriterError):
        morph.save(tmp_path / "out.m3z")
    assert not (tmp_path / "out.m3z").exists()


def test_writing_a_chain_leaves_the_record_alone(tmp_path: Path) -> None:
    morph = io.load(_write(tmp_path, _encode(_positions())))
    record = morph.metadata.raw
    ras2node, field, vox2ras = morph
    morph.transformations = [
        ras2node,
        xforms.CoordinatesField(field=_positions(SHAPE[::-1])),
        vox2ras,
    ]
    encoded = morph.to_raw()
    assert morph.metadata.raw is record
    np.testing.assert_array_equal(record.positions, _positions())
    # A new grid gets new arrays, which are read-only as well.
    assert encoded.shape == SHAPE[::-1]
    for array in (encoded.positions, encoded.index, encoded.labels):
        assert not array.flags.writeable
    assert not encoded.labels.any()


# ----------------------------------------------------------------------
#   SNIFFING AND HINTS
# ----------------------------------------------------------------------


def test_sniff(tmp_path: Path) -> None:
    path = _write(tmp_path, _encode(_positions()), "morph.bin")
    assert io.transformations.sniff(path) is M3zMorph
    assert M3zMorph.sniff(_encode(_positions(), gz=False)) == 1.0
    assert M3zMorph.sniff(b"\0" * 64) == 0.0
    assert M3zMorph.sniff(gzip.compress(b"not a morph" * 4)) == 0.0
    (tmp_path / "x.lta").write_text("type = 1\n")
    assert M3zMorph.sniff(tmp_path / "x.lta") == 0.0
    # FreeSurfer always writes big-endian, so a swapped header is no morph.
    swapped = struct.pack("<f4if", 1.0, *SHAPE, SPACING, 20.0)
    assert M3zMorph.sniff(swapped + bytes(64)) == 0.0
    # Only the head of an open file is read.
    with open(path, "rb") as f:
        assert M3zMorph.sniff(f) == 1.0 and f.tell() == 0


def test_the_record_is_the_interface() -> None:
    from brainhops.io.transformations.freesurfer import m3z

    assert {"M3zMetadata", "M3zRaw"} <= set(m3z.__all__)
    names = ("spacing", "image_vox2ras", "atlas_vox2ras", "affine", "struct")
    for name in names + ("to_struct", "from_struct"):
        assert not hasattr(M3zMorph, name)
    for name in ("M3zStruct", "M3zReaderWriter"):
        assert not hasattr(m3z, name)


@pytest.mark.parametrize("hint", ["m3z", "freesurfer", "freesurfer.m3z"])
def test_hints(tmp_path: Path, hint: str) -> None:
    path = _write(tmp_path, _encode(_positions()), "morph.bin")
    assert isinstance(io.load(path, hint=hint), M3zMorph)


# ----------------------------------------------------------------------
#   GEOMETRY: POINT CHECKS
# ----------------------------------------------------------------------


def _atlas_ras(atlas_vox: np.ndarray) -> np.ndarray:
    return _affine_apply(_vox2ras(ATLAS), atlas_vox)


def test_chain_shape(tmp_path: Path) -> None:
    morph = io.load(_write(tmp_path, _encode(_positions())))
    ras2node, field, vox2ras = morph
    assert isinstance(ras2node, RASToVoxel)
    assert isinstance(field, xforms.CoordinatesField)
    assert isinstance(vox2ras, VoxelToRAS)
    assert int(field.degree) == 1


def test_nodes_map_to_their_positions(tmp_path: Path) -> None:
    """Atlas voxel n * spacing lands at the position of node n."""
    positions = _positions()
    morph = io.load(_write(tmp_path, _encode(positions)))
    nodes = np.array([[0, 0, 0], [1, 2, 3], [3, 1, 4], [2, 0, 1]])
    expected = _affine_apply(
        _vox2ras(IMAGE), positions[tuple(nodes.T)].astype(float)
    )
    got = _apply(morph, _atlas_ras(nodes * SPACING))
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_between_nodes_is_trilinear(tmp_path: Path) -> None:
    """An atlas voxel between nodes interpolates the node positions."""
    positions = _positions()
    morph = io.load(_write(tmp_path, _encode(positions)))
    atlas_vox = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 5.0], [5.0, 3.0, 7.0]])
    expected = []
    for v in atlas_vox / SPACING:
        lo = np.floor(v).astype(int)
        t = v - lo
        acc = np.zeros(3)
        for corner in np.ndindex(2, 2, 2):
            w = np.prod(np.where(corner, t, 1 - t))
            acc += w * positions[tuple(lo + corner)]
        expected.append(acc)
    expected = _affine_apply(_vox2ras(IMAGE), np.asarray(expected))
    got = _apply(morph, _atlas_ras(atlas_vox))
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_ras_positions(tmp_path: Path) -> None:
    """GCAM_RAS positions are already in source scanner RAS."""
    positions = _positions() * 3 - 20
    morph = io.load(_write(tmp_path, _encode(positions, gtype=GCAM_RAS)))
    assert len(morph) == 2
    nodes = np.array([[0, 1, 2], [3, 2, 4]])
    got = _apply(morph, _atlas_ras(nodes * SPACING))
    np.testing.assert_allclose(got, positions[tuple(nodes.T)], atol=1e-4)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"xform_tag": 33, "keyword": "AutoAlign"},
        {"geometry": False, "gtype": None, "labels": False, "matrix": None},
        {"tail": struct.pack(">i", 99) + b"unknown tag payload"},
        {"tail": struct.pack(">i", 0)},
    ],
)
def test_round_trip_is_byte_exact(tmp_path: Path, options: dict) -> None:
    raw = _encode(_positions(), gz=False, **options)
    morph = io.load(_write(tmp_path, gzip.compress(raw)))
    out = tmp_path / "out.m3z"
    morph.save(out)
    assert gzip.decompress(out.read_bytes()) == raw
    out = tmp_path / "out.m3d"
    morph.save(out)
    assert out.read_bytes() == raw


def test_round_trip_through_io_save(tmp_path: Path) -> None:
    raw = _encode(_positions(), gz=False)
    morph = io.load(_write(tmp_path, raw, "t.m3d"))
    io.save(morph, tmp_path / "out.m3z")
    assert gzip.decompress((tmp_path / "out.m3z").read_bytes()) == raw


def test_write_edited_field(tmp_path: Path) -> None:
    morph = io.load(_write(tmp_path, _encode(_positions())))
    ras2node, field, vox2ras = morph
    moved = np.asarray(field.field) + 1.0
    morph.transformations = [
        ras2node,
        xforms.CoordinatesField(field=moved),
        vox2ras,
    ]
    morph.save(tmp_path / "out.m3z")
    old = morph.metadata.raw
    new = io.load(tmp_path / "out.m3z").metadata.raw
    np.testing.assert_allclose(new.positions, moved)
    # What the chain does not say is kept.
    np.testing.assert_array_equal(new.original, old.original)
    np.testing.assert_array_equal(new.labels, old.labels)
    np.testing.assert_allclose(new.xform.matrix, LINEAR, atol=1e-6)
    assert new.image.filename == IMAGE["fname"].decode()
    np.testing.assert_allclose(
        new.image_geometry.vox2ras, _vox2ras(IMAGE), atol=1e-5
    )
    np.testing.assert_allclose(
        new.atlas_geometry.vox2ras, _vox2ras(ATLAS), atol=1e-5
    )


def test_write_coefficients_as_positions(tmp_path: Path) -> None:
    # A morph stores sampled positions, so coefficients are written as values.
    morph = io.load(_write(tmp_path, _encode(_positions())))
    ras2node, field, vox2ras = morph
    moved = np.asarray(field.field) + 1.0
    coefficients = xforms.CoordinatesField(field=moved, degree=3)
    coefficients = coefficients.to(store="coefficients")
    morph.transformations = [ras2node, coefficients, vox2ras]
    morph.save(tmp_path / "out.m3z")
    back = io.load(tmp_path / "out.m3z")
    np.testing.assert_allclose(back.metadata.raw.positions, moved, atol=1e-4)


def test_write_from_scratch(tmp_path: Path) -> None:
    positions = _positions()
    node2ras = _vox2ras(ATLAS) @ np.diag([SPACING] * 3 + [1])
    voxel = xforms.CoordinatesField(field=positions)
    chain = [
        RASToVoxel(matrix=np.linalg.inv(node2ras)[:3]),
        voxel,
        VoxelToRAS(matrix=_vox2ras(IMAGE)[:3]),
    ]
    morph = M3zMorph(transformations=chain)
    with pytest.raises(WriterError):
        morph.save(tmp_path / "nope.m3z", spacing=SPACING)
    assert not (tmp_path / "nope.m3z").exists()
    morph.save(
        tmp_path / "out.m3z", spacing=SPACING, image_shape=IMAGE["shape"]
    )
    back = io.load(tmp_path / "out.m3z")
    assert back.metadata.raw.spacing == SPACING
    assert back.metadata.raw.atlas.shape == ATLAS["shape"]
    np.testing.assert_allclose(
        back.metadata.raw.atlas_geometry.vox2ras, _vox2ras(ATLAS), atol=1e-5
    )
    np.testing.assert_allclose(
        back.metadata.raw.image_geometry.vox2ras, _vox2ras(IMAGE), atol=1e-5
    )
    nodes = np.array([[1, 2, 3], [3, 0, 4]])
    points = _atlas_ras(nodes * SPACING + 0.5)
    np.testing.assert_allclose(
        _apply(back, points), _apply(chain, points), atol=1e-4
    )


def test_write_ras_chain(tmp_path: Path) -> None:
    positions = _positions() * 2
    morph = io.load(_write(tmp_path, _encode(positions, gtype=GCAM_RAS)))
    morph.transformations = list(morph)
    morph.save(tmp_path / "out.m3z")
    back = io.load(tmp_path / "out.m3z")
    assert back.metadata.raw.type == GCAM_RAS
    np.testing.assert_allclose(back.metadata.raw.positions, positions)


def test_write_refuses_other_chains(tmp_path: Path) -> None:
    morph = M3zMorph(transformations=[xforms.Affine(matrix=np.eye(4)[:3])])
    with pytest.raises(UnrepresentableTransformationError):
        morph.save(tmp_path / "out.m3z")
    assert not (tmp_path / "out.m3z").exists()
    field = xforms.DisplacementField(field=_positions())
    morph = M3zMorph(transformations=[RASToVoxel(), field, VoxelToRAS()])
    with pytest.raises(UnrepresentableTransformationError):
        morph.save(tmp_path / "out.m3z")


def test_a_morph_compares_by_identity() -> None:
    def make() -> M3zMorph:
        return M3zMorph(transformations=[xforms.Affine(matrix=np.eye(4)[:3])])

    morph = make()
    assert morph == morph
    assert morph != make()
    assert {morph: 1}[morph] == 1
    assert M3zMorph.__eq__ is object.__eq__
    assert M3zRaw.__eq__ is object.__eq__
