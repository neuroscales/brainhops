# stdlib
from types import ModuleType

# internals
from .typing import ArrayProtocol, npt, cpt, dkt


def _get_array_package(x: ArrayProtocol) -> ModuleType:
    if npt.np and isinstance(x, npt.np.ndarray):
        return npt.np
    elif cpt.cp and isinstance(x, cpt.cp.ndarray):
        return cpt.cp
    elif dkt.da and isinstance(x, dkt.da.ndarray):
        return dkt.da
    else:
        raise TypeError(f"Unsupported array type: {type(x)}")