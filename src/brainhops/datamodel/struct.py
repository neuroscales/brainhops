from brainhops.struct import Struct


class SpecializedStruct(Struct, convert=True):
    """
    We use this to set options that we want to propagate to all classes 
    in the hierarchy.
    """
    ...