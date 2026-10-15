__all__ = [
    "RASCoordinateSystem",
    "LPSCoordinateSystem",
    "RSACoordinateSystem",
    "RASmm",
    "LPSmm",
    "RSAmm",
    "FRASCoordinateSystem",
    "FLPSCoordinateSystem",
    "FRSACoordinateSystem",
    "CRASCoordinateSystem",
    "CLPSCoordinateSystem",
    "CRSACoordinateSystem",
]
import typing_extensions as tx
from bagof.magic import Factory, Inherit

from ..axes import A, AxisTuple, L, P, R, S
from .dispatch import _both, _is_anat, _is_mm
from .spatial import (
    CVoxelCoordinateSystem,
    FVoxelCoordinateSystem,
    MillimetricCoordinateSystem,
    SpatialCoordinateSystem3D,
)

# --- Base -------------------------------------------------------------


class RASCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RAS")}, priority=2
):
    """The RAS anatomical system, used by NIfTI and many neuroimaging formats.

    Coordinates increase towards the right, anterior and superior directions.
    """

    name: ... = "RAS"
    axes: Inherit[AxisTuple[R, A, S]] = Factory()


class LPSCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("LPS")}, priority=2
):
    """The LPS anatomical system, used by ITK and hence by ANTs and 3D Slicer.

    Coordinates increase towards the left, posterior and superior directions.
    """

    name: ... = "LPS"
    axes: Inherit[AxisTuple[L, P, S]] = Factory()


class RSACoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RSA")}, priority=2
):
    """The RSA anatomical system, found in some FreeSurfer LTA files.

    Coordinates increase towards the right, superior and anterior directions.
    """

    name: ... = "RSA"
    axes: Inherit[AxisTuple[R, S, A]] = Factory()


# --- Physical ---------------------------------------------------------


class RASmm(
    RASCoordinateSystem,
    MillimetricCoordinateSystem,
    on={"axes": _both(_is_anat("RAS"), _is_mm)},
):
    """A [`RASCoordinateSystem`][] in millimetres."""

    name: ... = "RAS"
    axes: Inherit[AxisTuple[R, A, S]] = Factory(
        lambda: (
            R("x", unit="mm"),
            A("y", unit="mm"),
            S("z", unit="mm"),
        )
    )


class LPSmm(
    LPSCoordinateSystem,
    MillimetricCoordinateSystem,
    on={"axes": _both(_is_anat("LPS"), _is_mm)},
):
    """An [`LPSCoordinateSystem`][] in millimetres."""

    name: ... = "LPS"
    axes: Inherit[AxisTuple[L, P, S]] = Factory(
        lambda: (
            L("x", unit="mm"),
            P("y", unit="mm"),
            S("z", unit="mm"),
        )
    )


class RSAmm(
    RSACoordinateSystem,
    MillimetricCoordinateSystem,
    on={"axes": _both(_is_anat("RSA"), _is_mm)},
):
    """An [`RSACoordinateSystem`][] in millimetres."""

    name: ... = "RSA"
    axes: Inherit[AxisTuple[R, S, A]] = Factory(
        lambda: (
            R("x", unit="mm"),
            S("y", unit="mm"),
            A("z", unit="mm"),
        )
    )


# --- Arrays -----------------------------------------------------------

# An F-ordered grid lists its axes as x, y, z, whereas a C-ordered grid
# lists them as z, y, x. The axes of an F-ordered RAS grid therefore point
# R, A, S, and those of a C-ordered RAS grid point S, A, R. An F-ordered
# system is selected when the constraints of both its parents hold. A
# C-ordered system lists its axes in an order that its anatomical parent
# does not accept, so it is excluded from automatic dispatch (`on=None`)
# and registered by hand with its C-ordered voxel parent, on its own axis
# order.


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("RAS"))
class FRASCoordinateSystem(RASCoordinateSystem, FVoxelCoordinateSystem):
    """An F-ordered voxel grid whose axes point in RAS order."""

    name: ... = "fRAS"
    axes: Inherit[AxisTuple[R, A, S]] = Factory(
        lambda: (
            R("x", unit="index"),
            A("y", unit="index"),
            S("z", unit="index"),
        )
    )


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("LPS"))
class FLPSCoordinateSystem(LPSCoordinateSystem, FVoxelCoordinateSystem):
    """An F-ordered voxel grid whose axes point in LPS order."""

    name: ... = "fLPS"
    axes: Inherit[AxisTuple[L, P, S]] = Factory(
        lambda: (
            L("x", unit="index"),
            P("y", unit="index"),
            S("z", unit="index"),
        )
    )


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("RSA"))
class FRSACoordinateSystem(RSACoordinateSystem, FVoxelCoordinateSystem):
    """An F-ordered voxel grid whose axes point in RSA order."""

    name: ... = "fRSA"
    axes: Inherit[AxisTuple[R, S, A]] = Factory(
        lambda: (
            R("x", unit="index"),
            S("y", unit="index"),
            A("z", unit="index"),
        )
    )


@CVoxelCoordinateSystem.register_polymorph(axes=_is_anat("SAR"))
class CRASCoordinateSystem(
    RASCoordinateSystem, CVoxelCoordinateSystem, on=None
):
    """A C-ordered RAS voxel grid, whose axes are listed S, A, R."""

    name: ... = "cRAS"
    order: tx.Literal["C"] = "C"
    axes: Inherit[AxisTuple[S, A, R]] = Factory(
        lambda: (
            S("z", unit="index"),
            A("y", unit="index"),
            R("x", unit="index"),
        )
    )


@CVoxelCoordinateSystem.register_polymorph(axes=_is_anat("SPL"))
class CLPSCoordinateSystem(
    LPSCoordinateSystem, CVoxelCoordinateSystem, on=None
):
    """A C-ordered LPS voxel grid, whose axes are listed S, P, L."""

    name: ... = "cLPS"
    order: tx.Literal["C"] = "C"
    axes: Inherit[AxisTuple[S, P, L]] = Factory(
        lambda: (
            S("z", unit="index"),
            P("y", unit="index"),
            L("x", unit="index"),
        )
    )


@CVoxelCoordinateSystem.register_polymorph(axes=_is_anat("ASR"))
class CRSACoordinateSystem(
    RSACoordinateSystem, CVoxelCoordinateSystem, on=None
):
    """A C-ordered RSA voxel grid, whose axes are listed A, S, R."""

    name: ... = "cRSA"
    order: tx.Literal["C"] = "C"
    axes: Inherit[AxisTuple[A, S, R]] = Factory(
        lambda: (
            A("z", unit="index"),
            S("y", unit="index"),
            R("x", unit="index"),
        )
    )
