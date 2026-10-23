__all__ = [
    "AnatomicalAxis",
    "LeftToRightAxis",
    "RightToLeftAxis",
    "AnteriorToPosteriorAxis",
    "PosteriorToAnteriorAxis",
    "InferiorToSuperiorAxis",
    "SuperiorToInferiorAxis",
]

from bagof.magic import NoRepr

from ..orientations import (
    AnteriorToPosterior,
    InferiorToSuperior,
    LeftToRight,
    PosteriorToAnterior,
    RightToLeft,
    SuperiorToInferior,
)
from .base import Axis
from .dispatch import has_orientation, is_anatomical
from .types import OrientedSpaceAxis


# An anatomical orientation implies a spatial axis, but every parent of
# this class requires `type='space'`. The class is therefore also
# registered with `Axis` itself, under a condition on the orientation
# alone.
@Axis.register_polymorph(on={"orientation": is_anatomical})
class AnatomicalAxis(OrientedSpaceAxis, on={"orientation": is_anatomical}):
    """Spatial axis with an anatomical orientation.

    An anatomical orientation selects this class even without a type. The
    unit may be an index unit, for a voxel axis along that direction.

    Parameters
    ----------
    name : str, optional
        Name of the axis, such as `"x"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : Orientation, required
        Meaningful direction of the axis, such as left to right.
    """


class LeftToRightAxis(
    AnatomicalAxis, on={"orientation": has_orientation("left-to-right")}
):
    """Spatial axis oriented from left to right.

    Parameters
    ----------
    name : str, default="left-to-right"
        Name of the axis, such as `"x"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : {orientations.LR}, default=orientations.LR
        Pinned orientation.
    """

    name: ... = "left-to-right"
    orientation: NoRepr[LeftToRight] = LeftToRight()


class RightToLeftAxis(
    AnatomicalAxis, on={"orientation": has_orientation("right-to-left")}
):
    """Spatial axis oriented from right to left.

    Parameters
    ----------
    name : str, default="right-to-left"
        Name of the axis, such as `"x"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : {orientations.RL}, default=orientations.RL
        Pinned orientation.
    """

    name: ... = "right-to-left"
    orientation: NoRepr[RightToLeft] = RightToLeft()


class AnteriorToPosteriorAxis(
    AnatomicalAxis,
    on={"orientation": has_orientation("anterior-to-posterior")},
):
    """Spatial axis oriented from anterior to posterior.

    Parameters
    ----------
    name : str, default="anterior-to-posterior"
        Name of the axis, such as `"y"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : {orientations.AP}, default=orientations.AP
        Pinned orientation.
    """

    name: ... = "anterior-to-posterior"
    orientation: NoRepr[AnteriorToPosterior] = AnteriorToPosterior()


class PosteriorToAnteriorAxis(
    AnatomicalAxis,
    on={"orientation": has_orientation("posterior-to-anterior")},
):
    """Spatial axis oriented from posterior to anterior.

    Parameters
    ----------
    name : str, default="posterior-to-anterior"
        Name of the axis, such as `"y"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : {orientations.PA}, default=orientations.PA
        Pinned orientation.
    """

    name: ... = "posterior-to-anterior"
    orientation: NoRepr[PosteriorToAnterior] = PosteriorToAnterior()


class InferiorToSuperiorAxis(
    AnatomicalAxis, on={"orientation": has_orientation("inferior-to-superior")}
):
    """Spatial axis oriented from inferior to superior.

    Parameters
    ----------
    name : str, default="inferior-to-superior"
        Name of the axis, such as `"z"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : {orientations.IS}, default=orientations.IS
        Pinned orientation.
    """

    name: ... = "inferior-to-superior"
    orientation: NoRepr[InferiorToSuperior] = InferiorToSuperior()


class SuperiorToInferiorAxis(
    AnatomicalAxis, on={"orientation": has_orientation("superior-to-inferior")}
):
    """Spatial axis oriented from superior to inferior.

    Parameters
    ----------
    name : str, default="superior-to-inferior"
        Name of the axis, such as `"z"`.
    type : {"space"}, default="space"
        Pinned axis type.
    unit : SpaceUnit | IndexUnit | str, optional
        Unit of the coordinates along the axis.

    Other Parameters
    ----------------
    discrete : bool, optional
        Whether the axis is discrete, meaning that its elements have
        no specific order and should not be interpolated.
    orientation : {orientations.SI}, default=orientations.SI
        Pinned orientation.
    """

    name: ... = "superior-to-inferior"
    orientation: NoRepr[SuperiorToInferior] = SuperiorToInferior()
