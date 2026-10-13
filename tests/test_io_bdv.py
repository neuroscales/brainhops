"""BigDataViewer datasets: their XML record, their images and the
registrations of their views.

The datasets are built by hand. The XML files follow the schema of
spimdata (`XmlIoAbstractSpimData`, `XmlIoAbstractSequenceDescription`,
`XmlIoBasicViewSetup`, `XmlIoTimePoints` and `XmlIoViewRegistrations`)
and the layout of JDOM's pretty format, which indents by two spaces and
ends lines with CRLF. The HDF5 files follow `bdv/img/hdf5/Util.java` and
are written with h5py.
"""

import os

import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io as io
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    DisplacementField,
    Sequence,
)
from brainhops.datamodel.units import Unit
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserExistsError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common.bdv import BdvMetadata, BdvRaw
from brainhops.io.common.bdv._geometry import (
    global_system,
    space_unit,
    voxel_system,
)
from brainhops.io.transformations.bdv import BdvViewRegistration

h5py = pytest.importorskip("h5py")

from brainhops.io.common.bdv._hdf5 import _BdvProxy  # noqa: E402
from brainhops.io.images.bdv import (  # noqa: E402
    BdvImage,
    BdvMultiScaleImage,
)

# ----------------------------------------------------------------------
#   DATASETS
# ----------------------------------------------------------------------

_TRANSLATION = (1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0)
"""A translation, which BigStitcher lists before the calibration."""

_CALIBRATION = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0)
"""The calibration that BigStitcher stores for a voxel size of
0.5 x 0.5 x 2.0: the voxel size divided by the smallest one."""


def _setup(index: int, size: str = "8 6 4", unit: str = "µm") -> str:
    return f"""      <ViewSetup>
        <id>{index}</id>
        <name>{index}</name>
        <size>{size}</size>
        <voxelSize>
          <unit>{unit}</unit>
          <size>0.5 0.5 2.0</size>
        </voxelSize>
      </ViewSetup>
"""


def _transform(values: tx.Sequence[float], name: tx.Optional[str]) -> str:
    named = f"        <Name>{name}</Name>\n" if name else ""
    numbers = " ".join(repr(float(v)) for v in values)
    return (
        '      <ViewTransform type="affine">\n'
        + named
        + f"        <affine>{numbers}</affine>\n"
        + "      </ViewTransform>\n"
    )


def _registration(timepoint: int, setup: int, transforms: str) -> str:
    return (
        f'    <ViewRegistration timepoint="{timepoint}" setup="{setup}">\n'
        + transforms
        + "    </ViewRegistration>\n"
    )


def _xml(
    setups: str = None,
    timepoints: str = None,
    registrations: str = None,
    loader: str = None,
    extra: str = "",
) -> bytes:
    """Return an XML file as BigDataViewer writes it, with CRLF lines."""
    if setups is None:
        setups = _setup(0)
    if timepoints is None:
        timepoints = (
            '    <Timepoints type="range">\n'
            "      <first>0</first>\n"
            "      <last>0</last>\n"
            "    </Timepoints>\n"
        )
    if registrations is None:
        registrations = _registration(
            0,
            0,
            _transform(_TRANSLATION, "Translation")
            + _transform(_CALIBRATION, "calibration"),
        )
    if loader is None:
        loader = (
            '    <ImageLoader format="bdv.hdf5">\n'
            '      <hdf5 type="relative">dataset.h5</hdf5>\n'
            "    </ImageLoader>\n"
        )
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<SpimData version="0.2">\n'
        '  <BasePath type="relative">.</BasePath>\n'
        "  <SequenceDescription>\n"
        + loader
        + "    <ViewSetups>\n"
        + setups
        + "    </ViewSetups>\n"
        + timepoints
        + extra
        + "  </SequenceDescription>\n"
        "  <ViewRegistrations>\n" + registrations + "  </ViewRegistrations>\n"
        "</SpimData>\n"
    )
    return text.replace("\n", "\r\n").encode("utf-8")


def _voxels(shape: tx.Sequence[int] = (8, 6, 4)) -> np.ndarray:
    """Return 16-bit voxels in the order x, y, z, some above 32767."""
    return (
        (np.arange(int(np.prod(shape))) * 997 % 65536)
        .astype("uint16")
        .reshape(shape)
    )


def _write_hdf5(
    path: tx.Any,
    levels: tx.Sequence[np.ndarray],
    resolutions: tx.Sequence[tx.Sequence[float]],
    data_type: tx.Optional[str] = None,
    setup: int = 0,
    timepoint: int = 0,
    mode: str = "w",
) -> None:
    """Write the levels of one view, stored z, y, x as BigDataViewer does.

    Without `data_type`, 16-bit unsigned voxels are stored as signed
    ones, as in a file written before the `dataType` attribute existed.
    """
    with h5py.File(path, mode) as f:
        group = f"s{setup:02d}"
        if group not in f:
            f.create_dataset(group + "/resolutions", data=resolutions)
            f.create_dataset(
                group + "/subdivisions",
                data=np.full((len(levels), 3), 4, "int32"),
            )
            if data_type is not None:
                f[group].attrs["dataType"] = data_type
        for index, level in enumerate(levels):
            stored = np.asarray(level).T
            if data_type is None:
                stored = stored.view("int16")
            name = f"t{timepoint:05d}/s{setup:02d}/{index}/cells"
            f.create_dataset(name, data=stored, chunks=True)


@pytest.fixture
def dataset(tmp_path: tx.Any) -> tx.Any:
    """A dataset of one view at two levels, in the older 16-bit layout."""
    xml = tmp_path / "dataset.xml"
    xml.write_bytes(_xml())
    voxels = _voxels()
    _write_hdf5(
        tmp_path / "dataset.h5",
        [voxels, voxels[::2, ::2]],
        [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
    )
    return xml


# ----------------------------------------------------------------------
#   THE RECORD
# ----------------------------------------------------------------------


def test_the_record_reads_the_schema_of_spimdata() -> None:
    timepoints = (
        '    <Timepoints type="pattern">\n'
        "      <integerpattern>0, 3-5, 20-10:5</integerpattern>\n"
        "    </Timepoints>\n"
    )
    missing = (
        "    <MissingViews>\n"
        '      <MissingView timepoint="3" setup="1" />\n'
        "    </MissingViews>\n"
    )
    raw = BdvRaw.from_bytes(
        _xml(
            setups=_setup(1) + _setup(0, size="2 3 4", unit="nm"),
            timepoints=timepoints,
            extra=missing,
        )
    )
    assert raw.version == "0.2"
    assert (raw.base_path, raw.base_path_relative) == (".", True)
    assert (raw.loader, raw.hdf5, raw.hdf5_relative) == (
        "bdv.hdf5",
        "dataset.h5",
        True,
    )
    assert [item.id for item in raw.setups] == [1, 0]
    assert raw.setup(0).size == (2, 3, 4)
    assert raw.setup(0).voxel_size == (0.5, 0.5, 2.0)
    assert raw.setup(0).unit == "nm"
    # IntegerPattern lists 0, the range 3-5, and 20 down to 10 by 5.
    assert raw.timepoints == (0, 3, 4, 5, 10, 15, 20)
    assert raw.missing == ((3, 1),)
    transforms = raw.registration(0, 0).transforms
    assert [item.name for item in transforms] == ["Translation", "calibration"]
    assert transforms[1].affine == _CALIBRATION
    assert raw.pyramids is None


def test_timepoints_may_be_a_list() -> None:
    timepoints = (
        '    <Timepoints type="list">\n'
        "      <Timepoint>\n        <id>7</id>\n      </Timepoint>\n"
        "      <Timepoint>\n        <id>2</id>\n      </Timepoint>\n"
        "    </Timepoints>\n"
    )
    assert BdvRaw.from_bytes(_xml(timepoints=timepoints)).timepoints == (2, 7)


def test_the_default_view_is_the_first_setup_and_time_point() -> None:
    timepoints = (
        '    <Timepoints type="range">\n'
        "      <first>3</first>\n      <last>4</last>\n"
        "    </Timepoints>\n"
    )
    raw = BdvRaw.from_bytes(
        _xml(setups=_setup(2) + _setup(1), timepoints=timepoints)
    )
    assert raw.view() == (1, 3)
    assert raw.view(2, 4) == (2, 4)
    with pytest.raises(ParserContentError, match="no setup 5"):
        raw.view(5)
    with pytest.raises(ParserContentError, match="no time point 0"):
        raw.view(None, 0)


def test_the_hdf5_path_resolves_against_the_base_path(tmp_path) -> None:  # noqa: ANN001
    content = _xml().replace(
        b'<BasePath type="relative">.</BasePath>',
        b'<BasePath type="relative">data</BasePath>',
    )
    raw = BdvRaw.from_bytes(content)
    expected = os.path.join(str(tmp_path), "data", "dataset.h5")
    assert str(raw.hdf5_filename(tmp_path / "dataset.xml")) == expected
    # A path without `type="relative"` is absolute, as for
    # `XmlHelpers.loadPath`.
    content = _xml(
        loader='    <ImageLoader format="bdv.hdf5">\n'
        "      <hdf5>/somewhere/else.h5</hdf5>\n"
        "    </ImageLoader>\n"
    )
    raw = BdvRaw.from_bytes(content)
    assert str(raw.hdf5_filename(tmp_path / "dataset.xml")).endswith("else.h5")


def test_other_xml_files_are_refused() -> None:
    assert BdvMetadata.sniff_bytes(b"<?xml version='1.0'?><OME/>") == 0
    with pytest.raises(ParserContentError, match="SpimData"):
        BdvRaw.from_bytes(b"<OME></OME>")
    with pytest.raises(ParserContentError, match="malformed"):
        BdvRaw.from_bytes(b"<SpimData><unclosed></SpimData>")
    with pytest.raises(TypeError, match="no option"):
        BdvRaw.from_bytes(_xml(), unknown=1)


# ----------------------------------------------------------------------
#   REGISTRATIONS
# ----------------------------------------------------------------------


def test_the_last_transform_of_the_file_is_applied_first() -> None:
    xform = BdvViewRegistration.from_bytes(_xml())
    first, second = xform.transformations
    # `ViewRegistration.updateModel` concatenates in the order of the
    # file, and `concatenate(b)` sets `a = a * b`, so the calibration,
    # listed last, is applied first.
    np.testing.assert_array_equal(
        first.matrix, np.reshape(_CALIBRATION, (3, 4))
    )
    np.testing.assert_array_equal(
        second.matrix, np.reshape(_TRANSLATION, (3, 4))
    )
    expected = np.eye(4)
    expected[:3] = np.reshape(_TRANSLATION, (3, 4))
    calibration = np.eye(4)
    calibration[:3] = np.reshape(_CALIBRATION, (3, 4))
    expected = expected @ calibration
    np.testing.assert_allclose(xform.to(Affine).matrix, expected[:3])
    # The voxel (1, 2, 3) is at z = 4 * 3 + 30 in the global space.
    point = xform.to(Affine).homogeneous_matrix @ [1.0, 2.0, 3.0, 1.0]
    np.testing.assert_allclose(point[:3], [11.0, 22.0, 42.0])
    assert xform.input == voxel_system()
    assert xform.output == global_system()
    assert first.input == voxel_system() and second.output == global_system()
    assert not first.matrix.flags.writeable


def test_an_untouched_registration_is_saved_byte_for_byte(tmp_path) -> None:  # noqa: ANN001
    content = _xml()
    xform = BdvViewRegistration.from_bytes(content)
    assert xform.to_bytes() == content
    xform.save(tmp_path / "copy.xml")
    assert (tmp_path / "copy.xml").read_bytes() == content
    assert BdvMetadata.from_bytes(content).to_bytes() == content


def test_a_changed_chain_rewrites_its_view_only() -> None:
    content = _xml()
    xform = BdvViewRegistration.from_bytes(content)
    old = xform.metadata.raw.registration(0, 0)
    scaling = Affine(matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:3])
    xform.transformations = list(xform.transformations) + [scaling]
    written = xform.to_bytes()
    # The bytes before and after the element of the view are kept, and so
    # are the elements of the transforms that were read, with their names.
    start, stop = old.span
    assert written[:start] == content[:start]
    assert written.endswith(content[stop:])
    for item in old.transforms:
        assert content[item.span[0] : item.span[1]] in written
    assert b"\n" not in written.replace(b"\r\n", b"")
    back = BdvViewRegistration.from_bytes(written)
    names = [
        item.name for item in back.metadata.raw.registration(0, 0).transforms
    ]
    # The scaling is applied last, so it is listed first, without a name.
    assert names == [None, "Translation", "calibration"]
    original = BdvViewRegistration.from_bytes(content).to(Affine)
    np.testing.assert_allclose(
        back.to(Affine).homogeneous_matrix,
        np.diag([2.0, 2.0, 2.0, 1.0]) @ original.homogeneous_matrix,
    )
    # The registration that was assigned is pending, and the object is
    # unchanged by the write.
    assert xform.metadata.raw.stored == content


def test_other_views_are_kept_when_one_view_changes() -> None:
    registrations = _registration(
        0, 0, _transform(_CALIBRATION, "calibration")
    ) + _registration(0, 1, _transform(_TRANSLATION, None))
    content = _xml(setups=_setup(0) + _setup(1), registrations=registrations)
    xform = BdvViewRegistration.from_bytes(content, setup=1)
    np.testing.assert_array_equal(
        xform.to(Affine).matrix, np.reshape(_TRANSLATION, (3, 4))
    )
    xform.transformations = []
    raw = BdvRaw.from_bytes(xform.to_bytes())
    assert raw.registration(0, 1).transforms == ()
    assert raw.registration(0, 0).transforms[0].name == "calibration"
    # An empty registration is the identity, as BigDataViewer reads it.
    back = BdvViewRegistration.from_raw(raw, setup=1)
    assert back.transformations == ()
    np.testing.assert_array_equal(back.to(Affine).matrix, np.eye(4)[:3])


def test_a_registration_built_from_a_chain_writes_a_new_file() -> None:
    chain = [Affine(matrix=np.reshape(_TRANSLATION, (3, 4)))]
    content = BdvViewRegistration(chain, setup=2, timepoint=5).to_bytes()
    raw = BdvRaw.from_bytes(content)
    assert content.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\r\n')
    assert raw.loader is None
    assert [item.id for item in raw.setups] == [2]
    assert raw.timepoints == (5,)
    assert raw.registration(5, 2).transforms[0].affine == _TRANSLATION


def test_a_generic_transform_is_kept_but_not_read(tmp_path) -> None:  # noqa: ANN001
    generic = (
        '      <ViewTransform type="generic" class="some.java.Class">\n'
        "        <Name>elastic</Name>\n"
        "      </ViewTransform>\n"
    )
    content = _xml(registrations=_registration(0, 0, generic))
    xform = BdvViewRegistration.from_bytes(content)
    assert xform.metadata.raw.registration(0, 0).transforms[0].type == (
        "generic"
    )
    with pytest.raises(ParserContentError, match="generic"):
        tuple(xform.transformations)
    assert xform.to_bytes() == content


def test_what_the_file_cannot_express_is_refused() -> None:
    xform = BdvViewRegistration.from_bytes(_xml())
    field = DisplacementField(np.zeros((2, 2, 2, 3)))
    xform.transformations = [field]
    with pytest.raises(UnrepresentableTransformationError, match="affine"):
        xform.to_bytes()
    xform = BdvViewRegistration.from_bytes(_xml())
    xform.output = CoordinateSystem(name="time", axes=[Axis("t", "time")])
    with pytest.raises(UnrepresentableTransformationError, match="output"):
        xform.consolidate()


def test_unknown_options_and_views_are_refused() -> None:
    with pytest.raises(TypeError):
        BdvViewRegistration.from_bytes(_xml(), level=1)
    with pytest.raises(ParserContentError, match="no setup 3"):
        BdvViewRegistration.from_bytes(_xml(), setup=3)
    with pytest.raises(ParserContentError, match="no registration"):
        BdvViewRegistration.from_bytes(
            _xml(setups=_setup(0) + _setup(1)), setup=1
        )


def test_a_chain_of_affines_converts_to_a_registration() -> None:
    chain = Sequence([Affine(matrix=np.reshape(_CALIBRATION, (3, 4)))])
    xform = BdvViewRegistration.from_instance(chain)
    assert xform.metadata is None
    raw = BdvRaw.from_bytes(xform.to_bytes())
    assert raw.registration(0, 0).transforms[0].affine == _CALIBRATION


# ----------------------------------------------------------------------
#   IMAGES
# ----------------------------------------------------------------------


def test_a_view_is_read_as_a_lazy_pyramid(dataset) -> None:  # noqa: ANN001
    pyramid = io.load(dataset)
    assert isinstance(pyramid, BdvMultiScaleImage)
    assert (pyramid.setup, pyramid.timepoint) == (0, 0)
    assert pyramid.transformations == []
    assert [level.level for level in pyramid.images] == [0, 1]
    level = pyramid.images[1]
    assert isinstance(level.raw, _BdvProxy)
    assert level.shape == (4, 3, 4)
    assert "_cache_data" not in vars(level)
    voxels = _voxels()
    np.testing.assert_array_equal(np.asarray(level.data), voxels[::2, ::2])
    # The older files store unsigned voxels in signed integers, which are
    # read back bit for bit.
    data = np.asarray(pyramid.images[0].data)
    assert data.dtype == np.uint16
    assert data.max() > 32767
    np.testing.assert_array_equal(data, voxels)
    assert level.metadata is pyramid.images[0].metadata
    assert level.metadata.raw.pyramid(0).resolutions == (
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 1.0),
    )


def test_a_level_is_placed_by_its_downsampling(dataset) -> None:  # noqa: ANN001
    level = io.images.load(dataset, level=1)
    assert isinstance(level, BdvImage)
    physical, world = level.transformations
    # Voxel i of a level downsampled by f is at f * i + (f - 1) / 2.
    np.testing.assert_allclose(
        physical.matrix,
        [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.25], [0.0, 0.0, 2.0, 0.0]],
    )
    np.testing.assert_allclose(
        world.matrix,
        [[2.0, 0.0, 0.0, 10.5], [0.0, 2.0, 0.0, 20.5], [0.0, 0.0, 4.0, 30.0]],
    )
    assert level.transformation is world
    assert physical.output.name == "physical"
    assert physical.output.axes[0].unit == Unit("um")
    assert world.output == global_system()
    assert world.output.axes[0].unit is None
    assert world.input == voxel_system()


def test_a_single_level_is_read_as_an_image(tmp_path) -> None:  # noqa: ANN001
    xml = tmp_path / "single.xml"
    xml.write_bytes(_xml().replace(b"dataset.h5", b"single.h5"))
    voxels = _voxels().astype("float32")
    _write_hdf5(tmp_path / "single.h5", [voxels], [[1, 1, 1]], "float32")
    image = io.images.load(xml)
    assert isinstance(image, BdvImage)
    assert np.asarray(image.data).dtype == np.float32
    np.testing.assert_array_equal(np.asarray(image.data), voxels)


def test_the_options_select_a_view(tmp_path) -> None:  # noqa: ANN001
    timepoints = (
        '    <Timepoints type="range">\n'
        "      <first>0</first>\n      <last>1</last>\n"
        "    </Timepoints>\n"
    )
    registrations = "".join(
        _registration(t, s, _transform(_TRANSLATION, None))
        for t in (0, 1)
        for s in (0, 1)
    )
    xml = tmp_path / "dataset.xml"
    xml.write_bytes(
        _xml(
            setups=_setup(0) + _setup(1),
            timepoints=timepoints,
            registrations=registrations,
        )
    )
    path = tmp_path / "dataset.h5"
    for t in (0, 1):
        for s in (0, 1):
            voxels = np.full((8, 6, 4), 10 * t + s, "uint8")
            _write_hdf5(path, [voxels], [[1, 1, 1]], "uint8", s, t, "a")
    image = io.images.load(xml, setup=1, timepoint=1)
    assert (image.setup, image.timepoint) == (1, 1)
    assert np.asarray(image.data).dtype == np.uint8
    assert np.all(np.asarray(image.data) == 11)
    with pytest.raises(Exception, match="no level 3"):
        BdvImage.load(xml, level=3)
    with pytest.raises(TypeError):
        BdvImage.load(xml, unknown=1)


def test_a_partition_is_read_through_its_external_link(tmp_path) -> None:  # noqa: ANN001
    xml = tmp_path / "dataset.xml"
    xml.write_bytes(_xml())
    voxels = _voxels()
    _write_hdf5(tmp_path / "dataset-00.h5", [voxels], [[1, 1, 1]])
    with h5py.File(tmp_path / "dataset.h5", "w") as f:
        f.create_dataset("s00/resolutions", data=[[1.0, 1.0, 1.0]])
        f.create_dataset("s00/subdivisions", data=[[4, 4, 4]])
        f["t00000/s00/0/cells"] = h5py.ExternalLink(
            "dataset-00.h5", "t00000/s00/0/cells"
        )
    image = BdvImage.load(xml)
    np.testing.assert_array_equal(np.asarray(image.data), voxels)


def test_an_empty_selection_reads_nothing(dataset) -> None:  # noqa: ANN001
    image = BdvImage.load(dataset)
    os.remove(dataset.parent / "dataset.h5")
    assert image.raw[0:0].shape == (0, 6, 8)


def test_what_cannot_be_read_as_an_image_is_refused(tmp_path) -> None:  # noqa: ANN001
    xml = tmp_path / "n5.xml"
    xml.write_bytes(
        _xml(
            loader='    <ImageLoader format="bdv.n5">\n'
            '      <n5 type="relative">dataset.n5</n5>\n'
            "    </ImageLoader>\n"
        )
    )
    with pytest.raises(ParserContentError, match="bdv.hdf5"):
        BdvImage.load(xml)
    # Such a dataset is still read as a registration.
    assert isinstance(io.load(xml), BdvViewRegistration)
    xml = tmp_path / "lost.xml"
    xml.write_bytes(_xml())
    with pytest.raises(ParserExistsError, match="dataset.h5"):
        BdvImage.load(xml)
    with pytest.raises(ParserContentError, match="path of the XML file"):
        BdvImage.from_bytes(_xml())


def test_a_missing_view_is_refused(tmp_path) -> None:  # noqa: ANN001
    missing = (
        "    <MissingViews>\n"
        '      <MissingView timepoint="0" setup="0" />\n'
        "    </MissingViews>\n"
    )
    xml = tmp_path / "dataset.xml"
    xml.write_bytes(_xml(extra=missing))
    _write_hdf5(tmp_path / "dataset.h5", [_voxels()], [[1, 1, 1]])
    with pytest.raises(ParserContentError, match="missing"):
        BdvImage.load(xml)


def test_an_image_cannot_be_written(dataset, tmp_path) -> None:  # noqa: ANN001
    image = BdvImage.load(dataset)
    assert not hasattr(image, "save")
    with pytest.raises(WriterError):
        io.save(image, tmp_path / "copy.xml")


def test_the_dataset_is_an_image_and_a_registration(dataset) -> None:  # noqa: ANN001
    assert isinstance(io.load(dataset), BdvMultiScaleImage)
    xform = io.transformations.load(dataset)
    assert isinstance(xform, BdvViewRegistration)
    image = io.images.load(dataset, level=0)
    np.testing.assert_allclose(
        image.transformation.matrix, xform.to(Affine).matrix
    )


@pytest.mark.parametrize(
    "text, unit",
    [
        ("µm", "micrometer"),
        ("um", "micrometer"),
        ("micron", "micrometer"),
        ("mm", "millimeter"),
        ("pixel", None),
        ("px", None),
        ("", None),
    ],
)
def test_the_unit_of_a_voxel_size(text: str, unit: tx.Optional[str]) -> None:
    found = space_unit(text)
    assert (None if found is None else str(found)) == unit
