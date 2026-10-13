"""Tests of the affines of TrakEM2 patches.

The project below is written by hand in the layout that TrakEM2 writes at
commit cdd8b8f: an ISO-8859-1 XML file with a document type whose
internal subset declares the elements, a `<t2_layer_set>` with its
`<t2_calibration>`, and `<t2_layer>` elements that hold `<t2_patch>`
elements, one attribute per line, each indented by tabs.
"""

import numpy as np
import pytest

from brainhops import io
from brainhops.datamodel.systems import RASmm
from brainhops.datamodel.transformations import Affine
from brainhops.errors import (
    ParserContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.transformations import load as load_transformation
from brainhops.io.transformations.trakem2 import (
    TrakEM2Metadata,
    TrakEM2PatchAffine,
    TrakEM2Raw,
)

PROJECT = """<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE trakem2_anything [
\t<!ELEMENT trakem2 (project,t2_layer_set,t2_display)>
\t<!ELEMENT t2_layer (t2_patch,t2_label,t2_layer_set,t2_profile)>
\t<!ATTLIST t2_layer oid NMTOKEN #REQUIRED>
\t<!ATTLIST t2_layer thickness NMTOKEN #REQUIRED>
\t<!ATTLIST t2_layer z NMTOKEN #REQUIRED>
{patch_element}
\t<!ATTLIST t2_patch oid NMTOKEN #REQUIRED>
\t<!ATTLIST t2_patch transform NMTOKEN #REQUIRED>
{padding}] >

<trakem2>
\t<project
\t\tid="0"
\t\ttitle="\xe9t\xe9.xml"
\t>
\t</project>
\t<t2_layer_set
\t\toid="3"
\t\twidth="2048.0"
\t\theight="2048.0"
\t\ttransform="matrix(1.0,0.0,0.0,1.0,0.0,0.0)"
\t\ttitle="Top Level"
\t\tlinks=""
\t\tlayer_width="2048.0"
\t\tlayer_height="2048.0"
\t>
\t\t<t2_calibration
\t\t\tpixelWidth="4.0"
\t\t\tpixelHeight="5.0"
\t\t\tpixelDepth="50.0"
\t\t\txOrigin="0.0"
\t\t\tyOrigin="0.0"
\t\t\tzOrigin="0.0"
\t\t\tinfo="null"
\t\t\tvalueUnit="Gray Value"
\t\t\ttimeUnit="sec"
\t\t\tunit="nm"
\t\t/>
\t\t<t2_layer oid="5"
\t\t\t thickness="12.5"
\t\t\t z="25.0"
\t\t\t title=""
\t\t>
\t\t\t<t2_patch
\t\t\t\toid="7"
\t\t\t\twidth="512.0"
\t\t\t\theight="256.0"
\t\t\t\ttransform="matrix(0.9,0.1,-0.1,0.9,100.5,-20.25)"
\t\t\t\ttitle="tile.tif"
\t\t\t\tlinks=""
\t\t\t\ttype="0"
\t\t\t\tfile_path="images/tile_0.tif"
\t\t\t\tstyle="fill-opacity:1.0;stroke:#ffff00;"
\t\t\t\to_width="512"
\t\t\t\to_height="256"
\t\t\t\tmin="0.0"
\t\t\t\tmax="255.0"
\t\t\t\tmres="32"
\t\t\t>
\t\t\t</t2_patch>
\t\t\t<t2_patch
\t\t\t\toid="8"
\t\t\t\twidth="512.0"
\t\t\t\theight="256.0"
\t\t\t\ttransform="matrix(1.0,0.0,0.0,1.0,400.0,0.0)"
\t\t\t\ttitle="tile.tif"
\t\t\t\tlinks=""
\t\t\t\to_width="512"
\t\t\t\to_height="256"
\t\t\t\tct_id="11"
\t\t\t>
<iict_transform class="mpicbg.trakem2.transform.SimilarityModel2D"
 data="0.5 0.1 3.0 4.0" />
\t\t\t</t2_patch>
\t\t</t2_layer>
\t\t<t2_layer oid="6"
\t\t\t thickness="12.5"
\t\t\t z="37.5"
\t\t\t title=""
\t\t>
\t\t\t<t2_patch
\t\t\t\toid="9"
\t\t\t\twidth="512.0"
\t\t\t\theight="256.0"
\t\t\t\ttransform="matrix(1.0,0.0,0.0,1.0,0.0,0.0)"
\t\t\t\ttitle="affine.tif"
\t\t\t\tct_id="12"
\t\t\t>
<iict_transform class="mpicbg.trakem2.transform.AffineModel2D"
 data="1.0 0.5 -0.5 2.0 7.0 8.0" />
\t\t\t</t2_patch>
\t\t\t<t2_patch
\t\t\t\toid="10"
\t\t\t\twidth="512.0"
\t\t\t\theight="256.0"
\t\t\t\ttransform="matrix(1.0,0.0,0.0,1.0,0.0,0.0)"
\t\t\t\ttitle="list.tif"
\t\t\t\tct_id="13"
\t\t\t>
<ict_transform_list>
\t<iict_transform class="mpicbg.trakem2.transform.AffineModel2D"
\t data="1.0 0.0 0.0 1.0 0.0 0.0" />
\t<ict_transform
\t class="mpicbg.trakem2.transform.ThinPlateSplineTransform"
\t data="2 1 2 0.0 0.0 1.0 1.0" />
</ict_transform_list>
\t\t\t</t2_patch>
\t\t\t<t2_patch
\t\t\t\toid="14"
\t\t\t\twidth="512.0"
\t\t\t\theight="256.0"
\t\t\t\ttransform="matrix(1.0,0.0,0.0,1.0,0.0,0.0)"
\t\t\t\ttitle="external.tif"
\t\t\t\tct_id="15"
\t\t\t>
\t\t\t</t2_patch>
\t\t</t2_layer>
\t</t2_layer_set>
</trakem2>
"""


PATCH_ELEMENT = (
    "\t<!ELEMENT t2_patch (t2_prop,t2_linked_prop,t2_annot,ict_transform,"
    "ict_transform_list,t2_filter)>"
)
"""The declaration of a patch, which TrakEM2 writes on one line."""


def _project(padding: int = 0) -> bytes:
    """Return the project, with a longer document type if asked."""
    comment = "\t<!-- " + "x" * padding + " -->\n" if padding else ""
    text = PROJECT.format(padding=comment, patch_element=PATCH_ELEMENT)
    return text.encode("latin-1")


@pytest.fixture
def project(tmp_path):  # noqa: ANN001, ANN201
    path = tmp_path / "project.xml"
    path.write_bytes(_project())
    return path


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_the_transform_attribute_is_read_in_awt_order(project) -> None:  # noqa: ANN001
    xform = io.load(project)
    assert type(xform) is TrakEM2PatchAffine
    assert xform.patch == 7
    # matrix(m00,m10,m01,m11,m02,m12)
    expected = [[0.9, -0.1, 100.5], [0.1, 0.9, -20.25]]
    assert np.array_equal(xform.matrix, expected)
    assert not xform.data.flags.writeable
    assert xform.input.name == "patch"
    assert [axis.name for axis in xform.input.axes] == ["x", "y"]
    assert xform.output.name == "layer"


def test_a_patch_is_selected_by_oid_or_title(project) -> None:  # noqa: ANN001
    xform = TrakEM2PatchAffine.load(project, patch=9)
    assert xform.patch == 9
    assert np.array_equal(xform.matrix, np.eye(3)[:2])
    xform = TrakEM2PatchAffine.load(project, patch="affine.tif")
    assert xform.patch == 9
    with pytest.raises(ParserContentError, match="oids 7, 8"):
        TrakEM2PatchAffine.load(project, patch="tile.tif")
    with pytest.raises(ParserContentError, match="no patch"):
        TrakEM2PatchAffine.load(project, patch=99)
    with pytest.raises(TypeError):
        TrakEM2PatchAffine.load(project, setup=0)


def test_the_record_keeps_the_layer_and_the_calibration(project) -> None:  # noqa: ANN001
    record = TrakEM2Metadata.load(project).raw
    assert [p.oid for p in record.patches] == [7, 8, 9, 10, 14]
    patch = record.patch(9)
    assert (patch.layer, patch.z, patch.thickness) == (6, 37.5, 12.5)
    calibration = patch.calibration
    assert calibration.pixel_width == 4.0
    assert calibration.pixel_height == 5.0
    assert calibration.pixel_depth == 50.0
    assert calibration.unit == "nm"
    assert record.patch(7).o_width == 512


def test_a_long_document_type_is_recognized(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "long.xml"
    path.write_bytes(_project(padding=10000))
    assert type(io.load(path)) is TrakEM2PatchAffine
    assert type(load_transformation(path)) is TrakEM2PatchAffine
    assert TrakEM2Raw.sniff_bytes(b"<?xml version='1.0'?><SpimData/>") == 0


# ----------------------------------------------------------------------
#   COORDINATE TRANSFORMS
# ----------------------------------------------------------------------


def test_an_affine_model_is_decoded(project) -> None:  # noqa: ANN001
    xform = TrakEM2PatchAffine.load(project, patch=9)
    # The affine maps the image after its coordinate transform.
    assert xform.input is None
    ct = xform.coordinate_transform()
    assert np.array_equal(ct.matrix, [[1.0, -0.5, 7.0], [0.5, 2.0, 8.0]])
    assert ct.input.name == "patch"
    assert ct.output is None


def test_a_similarity_model_is_decoded_in_single_precision(project) -> None:  # noqa: ANN001
    ct = TrakEM2PatchAffine.load(project, patch=8).coordinate_transform()
    s, c = float(np.float32(0.1)), 0.5
    assert np.array_equal(ct.matrix, [[c, -s, 3.0], [s, c, 4.0]])


def test_other_coordinate_transforms_are_refused(project) -> None:  # noqa: ANN001
    with pytest.raises(ParserContentError, match="ict_transform_list"):
        TrakEM2PatchAffine.load(project, patch=10).coordinate_transform()
    with pytest.raises(ParserContentError, match="separate file"):
        TrakEM2PatchAffine.load(project, patch=14).coordinate_transform()
    assert TrakEM2PatchAffine.load(project).coordinate_transform() is None
    # The list is kept as it is written.
    record = TrakEM2Metadata.load(project).raw
    item = record.patch(10).coordinate_transform
    assert item.element == "ict_transform_list"
    assert [child.cls for child in item.children] == [
        "mpicbg.trakem2.transform.AffineModel2D",
        "mpicbg.trakem2.transform.ThinPlateSplineTransform",
    ]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_an_untouched_project_is_written_byte_for_byte(
    project,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    target = tmp_path / "copy.xml"
    TrakEM2PatchAffine.load(project, patch=10).save(target)
    assert target.read_bytes() == project.read_bytes()


def test_only_the_transform_attribute_is_replaced(project, tmp_path) -> None:  # noqa: ANN001
    xform = TrakEM2PatchAffine.load(project, patch=8)
    xform.data = np.array([[1.0, 0.0, 1e-5], [0.0, 1.0, 12345678.0]])
    target = tmp_path / "changed.xml"
    xform.save(target)
    old = b"matrix(1.0,0.0,0.0,1.0,400.0,0.0)"
    new = b"matrix(1.0,0.0,0.0,1.0,1.0E-5,1.2345678E7)"
    assert target.read_bytes() == project.read_bytes().replace(old, new)
    back = TrakEM2PatchAffine.load(target, patch=8)
    assert np.array_equal(back.matrix, xform.matrix)
    assert np.array_equal(
        TrakEM2PatchAffine.load(target).matrix,
        TrakEM2PatchAffine.load(project).matrix,
    )


def test_an_equal_matrix_keeps_the_record(project) -> None:  # noqa: ANN001
    xform = TrakEM2PatchAffine.load(project)
    record = xform.metadata.raw
    xform.data = np.array(xform.data)
    assert xform.consolidate().metadata.raw is record


def test_a_matrix_that_is_not_a_2d_affine_is_refused(project) -> None:  # noqa: ANN001
    xform = TrakEM2PatchAffine.load(project)
    xform.data = np.eye(4)[:3]
    with pytest.raises(UnrepresentableTransformationError):
        xform.consolidate()
    xform.data = np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0]])
    with pytest.raises(WriterError):
        xform.consolidate()


def test_systems_are_honored_or_refused(project) -> None:  # noqa: ANN001
    xform = TrakEM2PatchAffine.load(project)
    xform.input = TrakEM2PatchAffine.load(project).input
    assert xform.consolidate().metadata.raw is xform.metadata.raw
    xform.output = RASmm()
    with pytest.raises(UnrepresentableTransformationError):
        xform.consolidate()
    # A patch with a coordinate transform takes no input system.
    other = TrakEM2PatchAffine.load(project, patch=9)
    other.input = xform.input
    with pytest.raises(UnrepresentableTransformationError):
        other.consolidate()


def test_an_affine_without_a_project_is_refused(tmp_path) -> None:  # noqa: ANN001
    xform = Affine(np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]))
    converted = xform.to(TrakEM2PatchAffine)
    assert type(converted) is TrakEM2PatchAffine
    assert np.array_equal(converted.matrix, xform.matrix)
    with pytest.raises(UnrepresentableTransformationError):
        converted.save(tmp_path / "new.xml")
    with pytest.raises(WriterError):
        TrakEM2Metadata().to_bytes()


def test_the_selected_patch_is_kept_by_copies(project) -> None:  # noqa: ANN001
    xform = TrakEM2PatchAffine.load(project, patch="affine.tif")
    copy = TrakEM2PatchAffine.from_instance(xform)
    assert copy.patch == 9
    assert copy.metadata is xform.metadata
    xform.data = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    consolidated = xform.consolidate()
    assert consolidated.patch == 9
    assert np.array_equal(
        consolidated.metadata.raw.patch(9).transform,
        (2.0, 0.0, 0.0, 2.0, 0.0, 0.0),
    )
