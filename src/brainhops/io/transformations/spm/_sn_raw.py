"""The record of an SPM normalisation file (`*_sn.mat`).

The classes of this module hold what SPM's old spatial normalisation
saves, as `spm_normalise` (SPM2 to SPM8, and the `OldNorm` toolbox of
SPM12) and `spm_prep2sn` (the unified segmentation of SPM5 and SPM8)
write it. The package documentation of
[`brainhops.io.transformations.spm`][] gives the specification and the
sources that it follows.
"""

__all__ = ["SpmSnRaw", "SpmVolume", "read_sn"]

# stdlib
import io

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Magic, field, replace

# internals
from brainhops._core.streams import preserve_position
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    ParserNotImplementedError,
)
from brainhops.io.common._arrays import _reject, detect_container

# ----------------------------------------------------------------------
#   CONVENTIONS
# ----------------------------------------------------------------------

SPM_TO_NIFTI = np.array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""The affine that turns an SPM voxel-to-world matrix into a NIfTI one.

SPM numbers voxels from 1, as MATLAB does, and the `mat` of a volume
maps these indices to millimetres. NIfTI numbers voxels from 0. SPM's
`@nifti/private/decode_qform0.m` therefore reads a NIfTI header as
`mat = M * [eye(4,3) [-1 -1 -1 1]']`, and the NIfTI affine of a volume is
`mat @ SPM_TO_NIFTI`, which first adds 1 to each 0-based index.
"""

_VARIABLES = ("Affine", "Tr", "VG", "VF")
"""The variables that every `*_sn.mat` file holds, besides `flags`."""


# ----------------------------------------------------------------------
#   RECORD
# ----------------------------------------------------------------------


class SpmVolume(Magic, frozen=True, eq=False, repr=HIDE_IF_NONE):
    """One volume of an SPM normalisation file, as `spm_vol` describes it.

    SPM describes an image by a MATLAB structure that `spm_vol` returns,
    and a normalisation file keeps those of the template (`VG`) and of the
    source image (`VF`). The record keeps the fields that place the image
    in the world and name it. It drops the fields that describe how the
    voxels are stored (`dt`, `pinfo`, `n`) and `private`, which holds a
    MATLAB object.
    """

    fname: str = ""
    """The name of the image file, as SPM saw it when it estimated."""

    dim: tx.Tuple[int, int, int] = (0, 0, 0)
    """The number of voxels along each of the three axes."""

    mat: tx.Optional[np.ndarray] = field(default=None, repr=False)
    """The `(4, 4)` voxel-to-world matrix, for voxels numbered from 1."""

    descrip: str = ""
    """The description of the image, or an empty string."""

    @property
    def vox2world(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-world matrix, for voxels numbered from 0.

        It is the affine that a NIfTI header of the image stores, as SPM
        writes it (see [`SPM_TO_NIFTI`][]).
        """
        return self.mat @ SPM_TO_NIFTI


class SpmSnRaw(
    Magic,
    BinaryFileReader,
    frozen=True,
    eq=False,
    repr=HIDE_IF_NONE,
):
    """Record of an SPM normalisation file, which holds the whole file.

    A `*_sn.mat` file is a small MATLAB file that is read in one pass,
    so the record holds all that brainhops reads from it: the affine, the
    coefficients of the nonlinear warp, the templates, the source image
    and the options of the estimation. The metadata of a normalisation,
    [`SpmSnMetadata`][brainhops.io.transformations.spm.SpmSnMetadata],
    holds a record, and an
    [`SpmSnTransformation`][brainhops.io.transformations.spm.SpmSnTransformation]
    decodes its chain of transformations from the record of its metadata.

    A record is never changed in place, so that several objects can
    share it. The class is frozen and its arrays are read-only. Records
    compare by identity.

    Attributes
    ----------
    affine : (4, 4) array
        The variable `Affine`. It maps the voxels of the template to the
        voxels of the source image, both numbered from 1.
    tr : (Kx, Ky, Kz, 3) array
        The variable `Tr`. It holds the coefficients of the three
        components of the warp, a displacement in template voxels, in a
        basis of `Kx * Ky * Kz` products of discrete cosines. It is empty
        when the normalisation is affine only.
    vg : tuple of SpmVolume
        The variable `VG`: the templates. SPM uses the first one.
    vf : tuple of SpmVolume
        The variable `VF`: the source image. SPM uses the first one.
    flags : dict
        The variable `flags`: the options of the estimation. Only its
        text and numeric fields are kept. A number is a float, text is a
        string, and any other array is a read-only NumPy array.
    """

    affine: tx.Optional[np.ndarray] = field(default=None, repr=False)
    tr: tx.Optional[np.ndarray] = field(default=None, repr=False)
    vg: tx.Tuple[SpmVolume, ...] = ()
    vf: tx.Tuple[SpmVolume, ...] = ()
    flags: tx.Optional[tx.Dict[str, tx.Any]] = field(default=None, repr=False)

    def copy(self) -> "SpmSnRaw":
        """Return a copy of the record that shares its arrays.

        The record is frozen and its arrays are never changed in place, so
        the copy does not duplicate them.
        """
        return replace(self)

    @property
    def template(self) -> SpmVolume:
        """The template on whose grid SPM evaluates the warp (`VG(1)`)."""
        return self.vg[0]

    @property
    def source(self) -> SpmVolume:
        """The source image that the normalisation warps (`VF(1)`)."""
        return self.vf[0]

    @property
    def affine_only(self) -> bool:
        """Whether the normalisation has no nonlinear part (empty `Tr`)."""
        return self.tr is None or self.tr.size == 0

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open binary file.

        A MATLAB file lists its variables one after the other, so the
        whole file is read to list them, and the position of the stream
        is restored.

        Parameters
        ----------
        file : IO
            The open binary file, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an SPM
            normalisation file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        with preserve_position(file):
            content = file.read()
        return cls.sniff_bytes(content, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of a file.

        A MATLAB v5 to v7 file that holds the double arrays `Affine` and
        `Tr` and the structures `VG` and `VF` is an SPM normalisation file.
        The variables are listed with `scipy.io.whosmat`, which does not
        decode their values.

        Parameters
        ----------
        content : bytes
            The content of a file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not an SPM
            normalisation file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if detect_container(bytes(content)) != "mat5":
            return _reject(error, "Not a MATLAB v5 to v7 file.")
        import scipy.io

        try:
            listed = scipy.io.whosmat(io.BytesIO(bytes(content)))
        except Exception as e:  # noqa: BLE001  any reader error
            return _reject(error, f"Not a readable MATLAB file: {e}")
        classes = {name: kind for name, _, kind in listed}
        expected = {
            "Affine": "double",
            "Tr": "double",
            "VG": "struct",
            "VF": "struct",
        }
        if any(classes.get(n) != k for n, k in expected.items()):
            return _reject(
                error,
                "A MATLAB file is an SPM normalisation file when it holds "
                "the variables Affine, Tr, VG and VF.",
            )
        return Confidence.CERTAIN

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "SpmSnRaw":
        """Read a record from the rest of an open binary file.

        Parameters
        ----------
        file : IO
            The open binary file.
        **kwargs : Any
            Options of the reader. The record takes none.

        Returns
        -------
        SpmSnRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the content is not an SPM normalisation file.
        ParserNotImplementedError
            If the file is a MATLAB v7.3 file.
        """
        _no_options(kwargs)
        return read_sn(file.read())


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def read_sn(content: bytes) -> SpmSnRaw:
    """Read the record of an SPM normalisation file from its bytes.

    SPM saves the file with MATLAB's `save`, in the v6 format since SPM5
    (`-V6` in `spm_normalise`, and `defaults.mat.format = '-v6'` in
    SPM12), and in the format of the MATLAB of the user before that. The
    file is read with `scipy.io.loadmat`. A v7.3 file, which SPM12 writes
    only when its default is changed, is an HDF5 file in which a MATLAB
    structure has a layout of its own, and it is refused rather than
    guessed.

    Parameters
    ----------
    content : bytes
        The content of the file.

    Returns
    -------
    SpmSnRaw
        The record.

    Raises
    ------
    ParserContentError
        If the content is not an SPM normalisation file.
    ParserNotImplementedError
        If the file is a MATLAB v7.3 file.
    """
    content = bytes(content)
    kind = detect_container(content)
    if kind in ("mat73", "hdf5"):
        raise ParserNotImplementedError(
            "This SPM normalisation file was saved in the MATLAB v7.3 "
            "(HDF5) format, which brainhops does not read. SPM saves it in "
            "the v6 format unless defaults.mat.format is changed. In "
            "MATLAB, sn = load(name) followed by save(name, '-struct', "
            "'sn', '-v6') saves it again in that format."
        )
    if kind != "mat5":
        raise ParserContentError(
            "An SPM normalisation file is a MATLAB v5 to v7 file. The "
            "*_sn3d.mat files of SPM99 are MATLAB v4 files with other "
            "variables, which brainhops does not read."
        )
    import scipy.io

    try:
        variables = scipy.io.loadmat(
            io.BytesIO(content),
            squeeze_me=False,
            struct_as_record=True,
            chars_as_strings=True,
            mat_dtype=False,
        )
    except Exception as e:
        raise ParserContentError(f"Not a readable MATLAB file: {e}") from e
    missing = [name for name in _VARIABLES if name not in variables]
    if missing:
        raise ParserContentError(
            f"An SPM normalisation file holds the variables "
            f"{', '.join(_VARIABLES)}, and this file lacks "
            f"{', '.join(missing)}."
        )
    affine = _matrix(variables["Affine"], "Affine")
    tr = _coefficients(variables["Tr"])
    vg = _volumes(variables["VG"], "VG")
    vf = _volumes(variables["VF"], "VF")
    flags = _flags(variables.get("flags"))
    return SpmSnRaw(affine=affine, tr=tr, vg=vg, vf=vf, flags=flags)


def _no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options of a reader that takes none.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            f"Unknown SPM normalisation reader option(s): {', '.join(kwargs)}"
        )


def _read_only(array: tx.Any) -> np.ndarray:
    """Return a read-only float64 copy of a numeric MATLAB array."""
    array = np.array(array, dtype=np.float64)
    array.flags.writeable = False
    return array


def _numeric(value: tx.Any) -> bool:
    """Tell whether a value of a MATLAB file is a real numeric array."""
    return (
        isinstance(value, np.ndarray)
        and value.dtype.names is None
        and value.dtype.kind in "biuf"
    )


def _matrix(value: tx.Any, name: str) -> np.ndarray:
    """Read a `(4, 4)` matrix of the file.

    Raises
    ------
    ParserContentError
        If the value is not a `(4, 4)` numeric array.
    """
    if not _numeric(value) or value.shape != (4, 4):
        shape = getattr(value, "shape", None)
        raise ParserContentError(
            f"{name} of an SPM normalisation file is a 4x4 matrix, not "
            f"{type(value).__name__} of shape {shape}."
        )
    return _read_only(value)


def _coefficients(value: tx.Any) -> np.ndarray:
    """Read the coefficients `Tr`, which are empty for an affine only.

    MATLAB keeps the trailing axis of length 3, so a nonlinear warp is
    always four-dimensional, even when a basis has a single function
    along one axis.

    Raises
    ------
    ParserContentError
        If the coefficients are neither empty nor a `(Kx, Ky, Kz, 3)`
        numeric array.
    """
    if not _numeric(value):
        raise ParserContentError(
            "Tr of an SPM normalisation file is a numeric array."
        )
    if value.size == 0:
        return _read_only(np.zeros((0, 0, 0, 3)))
    if value.ndim != 4 or value.shape[3] != 3:
        raise ParserContentError(
            f"Tr of an SPM normalisation file has shape (Kx, Ky, Kz, 3), "
            f"not {value.shape}."
        )
    return _read_only(value)


def _volumes(value: tx.Any, name: str) -> tx.Tuple[SpmVolume, ...]:
    """Read an array of `spm_vol` structures, in MATLAB order.

    Raises
    ------
    ParserContentError
        If the value is not a non-empty structure array whose elements
        have a `mat` and a `dim`.
    """
    names = getattr(getattr(value, "dtype", None), "names", None) or ()
    if not {"mat", "dim"} <= set(names) or value.size == 0:
        raise ParserContentError(
            f"{name} of an SPM normalisation file is an array of spm_vol "
            f"structures, each with a mat and a dim."
        )
    volumes = []
    for element in value.ravel(order="F"):
        dim = element["dim"]
        if not _numeric(dim) or dim.size < 3:
            raise ParserContentError(
                f"The dim of {name} holds at least three numbers."
            )
        # The dim of SPM2, which reads Analyze headers, has a fourth
        # element, the data type.
        dim = tuple(int(d) for d in np.ravel(dim)[:3])
        volumes.append(
            SpmVolume(
                fname=_text(element["fname"]) if "fname" in names else "",
                dim=dim,
                mat=_matrix(element["mat"], f"The mat of {name}"),
                descrip=(
                    _text(element["descrip"]) if "descrip" in names else ""
                ),
            )
        )
    return tuple(volumes)


def _text(value: tx.Any) -> str:
    """Read a MATLAB character array, whose rows are joined by newlines."""
    if not isinstance(value, np.ndarray) or value.dtype.kind != "U":
        return ""
    return "\n".join(str(row) for row in value.ravel())


def _flags(value: tx.Any) -> tx.Dict[str, tx.Any]:
    """Read the text and numeric fields of the structure `flags`.

    A field that holds a single number becomes a float, text becomes a
    string, and another numeric array becomes a read-only float64 array.
    Fields of any other kind, such as nested structures and cell arrays,
    are dropped.
    """
    names = getattr(getattr(value, "dtype", None), "names", None)
    if not names or value.size == 0:
        return {}
    element = value.ravel(order="F")[0]
    flags = {}
    for name in names:
        item = element[name]
        if isinstance(item, np.ndarray) and item.dtype.kind == "U":
            flags[name] = _text(item)
        elif _numeric(item):
            flags[name] = float(item.item()) if item.size == 1 else (
                _read_only(item)
            )
    return flags
