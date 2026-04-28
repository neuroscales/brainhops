__all__ = ["Options"]
import typing_extensions as _tx
from .constants import MISSING
from .utils import slots, SlotsBase


@slots(
    'init',             # Generate __init__ method
    'repr',             # Generate __repr__ method
    'eq',               # Generate __eq__ method
    'order',            # Generate __lt__ method
    'unsafe_hash',      # Always generate __hash__ method
    'frozen',           # Disable __setattr__ and __delattr__
    'match_args',       # Generate __match_args__ for pattern matching
    'kw_only',          # Make all fields keyword-only by default
    'positional_only',  # Make all fields positional-only by default
    'slots',            # Generate __slots__ and remove __dict__
    'weakref_slot',     # Generate a weakref slot in __slots__
    'factory',          # Use field type as factory if none is provided
    'convert',          # Use field type as converter if none is provided
    'validate',         # Use field type as validator if none is provided
    'mapping',          # Generate Mapping methods for dict-like behavior
    'reverse',          # Use the reverse MRO order when listing fields
)
class Options(SlotsBase):

    _DEFAULTS: _tx.Dict[str, bool] = dict(
        init=True,
        repr=True,
        eq=True,
        order=False,
        unsafe_hash=False,
        frozen=False,
        match_args=False,
        kw_only=False,
        positional_only=False,
        slots=False,
        weakref_slot=False,
        factory=False,
        convert=False,
        validate=False,
        mapping=False,
        reverse=False,
    )
    
    @staticmethod
    def make_default() -> _tx.Self:
        return Options(**Options._DEFAULTS)