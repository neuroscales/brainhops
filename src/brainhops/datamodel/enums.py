"""Enumerations shared across the data model.

The enumerations cover interpolation orders, boundary conditions,
anatomical orientations and the known terms of metadata fields.
"""

__all__ = [
    "BoundaryCondition",
    "InterpolationOrder",
    "StoreEnum",
    "OrientationType",
    "AnatomicalOrientationValue",
    "SimplifyPolicy",
    "SpaceEnum",
    "IntentEnum",
    "Manufacturer",
    "IlluminationType",
    "ContrastMethod",
]

from brainhops._core.enum import IntEnum, StrEnum


class SimplifyPolicy(StrEnum):
    """How far a transformation may be inspected when it is simplified.

    - `none`: only the declared type is used, and nothing is inspected or
      rewritten.
    - `analytic`: the structure is inspected (missing parameters, array
      shapes, axis lists, the contents of wrappers), but no value is read
      and no lazy inverse is materialized. Matrix transformations are
      assumed invertible when square, surjective when wide and injective
      when tall.
    - `numeric`: values are read as well, for zero tests, diagonality,
      orthogonality and rank. A typed inverse may be materialized when a
      leaf is downcast, and the rank replaces the shape assumption, so a
      singular square matrix is not invertible.
    """

    none = "none"
    analytic = "analytic"
    numeric = "numeric"


# ruff: disable[E501]
# fmt: off
class BoundaryCondition(StrEnum):
    """Boundary conditions for interpolation and resampling.

    | Value         | Aliases                        | Description                     |
    |---------------|--------------------------------|---------------------------------|
    | `nearest`     | `edge`, `border`               | <code>(a a a a &vert; a b c d &vert; d d d d)</code> |
    | `reflect`     | `symmetric`, `dct2`            | <code>(d c b a &vert; a b c d &vert; d c b a)</code> |
    | `mirror`      | `dct1`                         | <code>  (d c b &vert; a b c d &vert; c b a)  </code> |
    | `grid-wrap`   | `circular`, `circulant`, `dft` | <code>(a b c d &vert; a b c d &vert; a b c d)</code> |
    | `wrap`        |                                | <code>(d b c d &vert; a b c d &vert; b c a b)</code> |
    | `constant`    | `zero`, `zeros`                | <code>(0 0 0 0 &vert; a b c d &vert; 0 0 0 0)</code> |

    The value `"grid-wrap"` is the member `gridwrap`.
    """

    nearest = edge = border = "nearest"
    reflect = symmetric = dct2 = "reflect"
    mirror = dct1 = "mirror"
    gridwrap = circular = circulant = dft = "grid-wrap"
    wrap = "wrap"
    constant = zero = zeros = "constant"
# fmt: on
# ruff: enable[E501]


class StoreEnum(StrEnum):
    """What the `data` of a transformation field holds.

    The member is given by the `store` flag of
    [`TransformationField`][brainhops.datamodel.transformations.TransformationField].

    - `"values"`: `data` holds the field's values; its `field` view is
      `data` itself.
    - `"coefficients"`: `data` holds spline coefficients; its `field` view
      is `data` decoded to values.

    Each member also has an upper-case alias, and `coefficients` is also
    spelled `coeffs`.
    """

    COEFFICIENTS = COEFFS = coefficients = coeffs = "coefficients"
    VALUES = values = "values"

    @classmethod
    def from_coefficients(cls, coefficients: bool) -> "StoreEnum":
        """Return the member matching a boolean `coefficients` flag.

        File formats state whether they store coefficients as a boolean.
        """
        return cls.coefficients if coefficients else cls.values


class InterpolationOrder(IntEnum):
    """Interpolation order for interpolation and resampling.

    | Name          | Aliases     | Value | Description                     |
    |---------------|-------------|-------|---------------------------------|
    | `zeroth`      | `nearest`   | 0     | Nearest neighbor interpolation. |
    | `first`       | `linear`    | 1     | Linear interpolation.           |
    | `second`      | `quadratic` | 2     | Quadratic interpolation.        |
    | `third`       | `cubic`     | 3     | Cubic interpolation.            |
    | `fourth`      |             | 4     | Fourth-order interpolation.     |
    | `fifth`       |             | 5     | Fifth-order interpolation.      |
    | `barycentric` |             | -1    | Barycentric interpolation.      |
    | `fourier`     |             | -2    | Fourier interpolation.          |
    """

    zeroth = nearest = 0
    first = linear = 1
    second = quadratic = 2
    third = cubic = 3
    fourth = 4
    fifth = 5
    barycentric = -1
    fourier = -2


class AxisType(StrEnum):
    """Types of the axes of coordinate systems and transformations.

    | Name          | Value           | Description             |
    |---------------|-----------------|-------------------------|
    | `space`       | `"space"`       | Spatial axis.           |
    | `time`        | `"time"`        | Temporal axis.          |
    | `channel`     | `"channel"`     | Channel axis.           |
    | `displacement`| `"displacement"`| Displacement axis.     |
    | `coordinate`  | `"coordinate"`  | Coordinate axis.       |
    """

    space = "space"
    time = "time"
    channel = "channel"
    displacement = "displacement"
    coordinate = "coordinate"


class OrientationType(StrEnum):
    """Types of orientation, of which only `anatomical` exists for now.

    | Name          | Value          | Description             |
    |---------------|----------------|-------------------------|
    | `anatomical`  | `"anatomical"` | Anatomical orientation. |
    """

    anatomical = "anatomical"


# ruff: disable[E501]
# fmt: off
class AnatomicalOrientationValue(StrEnum):
    """Anatomical orientations of the axes of coordinate systems.

    | Name                    | Value                     | Description             |
    |-------------------------|---------------------------|-------------------------|
    | `left_to_right`         | `"left-to-right"`         |
    | `right_to_left`         | `"right-to-left"`         |
    | `proximal_to_distal`    | `"proximal-to-distal"`    |
    | `distal_to_proximal`    | `"distal-to-proximal"`    |
    | `anterior_to_posterior` | `"anterior-to-posterior"` | front-to-back
    | `posterior_to_anterior` | `"posterior-to-anterior"` | back-to-front
    | `inferior_to_superior`  | `"inferior-to-superior"`  | feet-to-head
    | `superior_to_inferior`  | `"superior-to-inferior"`  | head-to-feet
    | `dorsal_to_palmar`      | `"dorsal-to-palmar"`      | back of hand to palm
    | `palmar_to_dorsal`      | `"palmar-to-dorsal"`      | palm to back of hand
    | `dorsal_to_plantar`     | `"dorsal-to-plantar"`     | top of foot to sole
    | `plantar_to_dorsal`     | `"plantar-to-dorsal"`     | sole to top of foot
    | `rostral_to_caudal`     | `"rostral-to-caudal"`     | nose/beak-to-tail, especially for nervous system
    | `caudal_to_rostral`     | `"caudal-to-rostral"`     | tail-to-nose/beak, especially for nervous system
    | `cranial_to_caudal`     | `"cranial-to-caudal"`     | head-to-tail
    | `caudal_to_cranial`     | `"caudal-to-cranial"`     | tail-to-head
    | `dorsal_to_ventral`     | `"dorsal-to-ventral"`     | back/top-to-belly/bottom
    | `ventral_to_dorsal`     | `"ventral-to-dorsal"`     | belly/bottom-to-back/top
    | `superficial_to_deep`   | `"superficial-to-deep"`   | outer surface to inner depth, e.g. skin, gut, cortex
    | `deep_to_superficial`   | `"deep-to-superficial"`   | inner depth to outer surface
    | `apical_to_basal`       | `"apical-to-basal"`       | apical to basal surface, e.g. epithelial layers, polarized cells
    | `basal_to_apical`       | `"basal-to-apical"`       | basal to apical surface
    | `apex_to_base`          | `"apex-to-base"`          | tip to broad base, e.g. heart, lungs
    | `base_to_apex`          | `"base-to-apex"`          | broad base to tip
    """
    # Common to bipeds and quadrupeds.
    left_to_right = "left-to-right"
    right_to_left = "right-to-left"
    proximal_to_distal = "proximal-to-distal"
    distal_to_proximal = "distal-to-proximal"

    # Primarily for bipeds, such as humans.
    anterior_to_posterior = "anterior-to-posterior"
    posterior_to_anterior = "posterior-to-anterior"
    inferior_to_superior = "inferior-to-superior"
    superior_to_inferior = "superior-to-inferior"
    dorsal_to_palmar = "dorsal-to-palmar"
    palmar_to_dorsal = "palmar-to-dorsal"
    dorsal_to_plantar = "dorsal-to-plantar"
    plantar_to_dorsal = "plantar-to-dorsal"

    # Primarily for quadrupeds.
    rostral_to_caudal = "rostral-to-caudal"
    caudal_to_rostral = "caudal-to-rostral"
    cranial_to_caudal = "cranial-to-caudal"
    caudal_to_cranial = "caudal-to-cranial"
    dorsal_to_ventral = "dorsal-to-ventral"
    ventral_to_dorsal = "ventral-to-dorsal"

    # Layered and polarized tissues, local to the subject.
    superficial_to_deep = "superficial-to-deep"
    deep_to_superficial = "deep-to-superficial"
    apical_to_basal = "apical-to-basal"
    basal_to_apical = "basal-to-apical"
    apex_to_base = "apex-to-base"
    base_to_apex = "base-to-apex"
# fmt: on
# ruff: enable[E501]


# ----------------------------------------------------------------------
#   METADATA TERMS
# ----------------------------------------------------------------------
# Known terms of free-text metadata fields. A field typed `Union[<Enum>, str]`
# holds a member when its value matches one and the string otherwise, so these
# lists do not close the vocabulary.


# ruff: disable[E501]
# fmt: off
class SpaceEnum(StrEnum):
    """Known labels of a world space.

    The labels are used by the metadata fields `space`, `input_space` and
    `output_space`. An unlisted label is kept as a plain string. The members are grouped
    as follows.

    - NIfTI names of the `sform_code` and `qform_code` values:
      `scanner` (1, scanner-based anatomical coordinates), `aligned`
      (2, aligned to another file), `talairach` (3, Talairach-Tournoux
      atlas), `mni` (4, MNI 152) and `template` (5, another template).
    - BIDS standard volume templates, used for `SpatialReference` and the
      `space-` entity of file names: `ICBM452AirSpace`,
      `ICBM452Warp5Space`, `IXI549Space`, `MNI152Lin`,
      `MNI152NLin2009{a,b,c}{Asym,Sym}`, `MNI152NLin6Asym`,
      `MNI152NLin6Sym`, `MNI305`, `MNIColin27`, `MNIInfant`,
      `MNIPediatricAsym`, `NMT31Sym`, `OASIS30AntsOASISAnts`,
      `OASIS30Atropos`, `Talairach` and `UNCInfant`.
    - BIDS standard surface templates: `fsaverage`, `fsaverage3` to
      `fsaverage6`, `fsaveragesym`, `fsLR` and `fsnative`.
    - BIDS non-standard spaces: `orig`, `anat`, `T1w`, `T2w`,
      `individual` and `study`.
    """

    # NIfTI
    scanner = "scanner"
    aligned = "aligned"
    talairach = "talairach"
    mni = "mni"
    template = "template"

    # BIDS standard templates (volumes)
    ICBM452AirSpace = "ICBM452AirSpace"
    ICBM452Warp5Space = "ICBM452Warp5Space"
    IXI549Space = "IXI549Space"
    MNI152Lin = "MNI152Lin"
    MNI152NLin2009aAsym = "MNI152NLin2009aAsym"
    MNI152NLin2009aSym = "MNI152NLin2009aSym"
    MNI152NLin2009bAsym = "MNI152NLin2009bAsym"
    MNI152NLin2009bSym = "MNI152NLin2009bSym"
    MNI152NLin2009cAsym = "MNI152NLin2009cAsym"
    MNI152NLin2009cSym = "MNI152NLin2009cSym"
    MNI152NLin6Asym = "MNI152NLin6Asym"
    MNI152NLin6Sym = "MNI152NLin6Sym"
    MNI305 = "MNI305"
    MNIColin27 = "MNIColin27"
    MNIInfant = "MNIInfant"
    MNIPediatricAsym = "MNIPediatricAsym"
    NMT31Sym = "NMT31Sym"
    OASIS30AntsOASISAnts = "OASIS30AntsOASISAnts"
    OASIS30Atropos = "OASIS30Atropos"
    Talairach = "Talairach"
    UNCInfant = "UNCInfant"

    # BIDS standard templates (surfaces)
    fsaverage = "fsaverage"
    fsaverage3 = "fsaverage3"
    fsaverage4 = "fsaverage4"
    fsaverage5 = "fsaverage5"
    fsaverage6 = "fsaverage6"
    fsaveragesym = "fsaveragesym"
    fsLR = "fsLR"
    fsnative = "fsnative"

    # BIDS non-standard spaces
    orig = "orig"
    anat = "anat"
    T1w = "T1w"
    T2w = "T2w"
    individual = "individual"
    study = "study"


class IntentEnum(StrEnum):
    """Known values of the `intent` metadata field.

    The intent states what the values of an image represent. The members are the NIfTI intent names as nibabel spells them; each
    value is the member name with spaces for underscores. An unlisted
    intent is kept as a plain string. The NIfTI codes are listed below.

    - Statistical distributions: `correlation` (2), `t_test` (3),
      `f_test` (4), `z_score` (5), `chi2` (6), `beta` (7), `binomial` (8),
      `gamma` (9), `poisson` (10), `normal` (11), `non_central_f_test`
      (12), `non_central_chi2` (13), `logistic` (14), `laplace` (15),
      `uniform` (16), `non_central_t_test` (17), `weibull` (18), `chi`
      (19), `inverse_gaussian` (20), `extreme_value_1` (21), `p_value`
      (22), `log_p_value` (23) and `log10_p_value` (24).
    - Other kinds of values: `estimate` (1001), `label` (1002),
      `neuroname` (1003), `general_matrix` (1004), `symmetric_matrix`
      (1005), `displacement_vector` (1006), `vector` (1007), `pointset`
      (1008), `triangle` (1009), `quaternion` (1010), `dimensionless`
      (1011), `time_series` (2001), `node_index` (2002), `rgb_vector`
      (2003), `rgba_vector` (2004) and `shape` (2005).
    - FSL intents: `fnirt_disp_field` (2006), `fnirt_cubic_spline_coef`
      (2007), `fnirt_dct_coef` (2008), `fnirt_quad_spline_coef` (2009) and
      `topup_field` (2018).
    """

    correlation = "correlation"
    t_test = "t test"
    f_test = "f test"
    z_score = "z score"
    chi2 = "chi2"
    beta = "beta"
    binomial = "binomial"
    gamma = "gamma"
    poisson = "poisson"
    normal = "normal"
    non_central_f_test = "non central f test"
    non_central_chi2 = "non central chi2"
    logistic = "logistic"
    laplace = "laplace"
    uniform = "uniform"
    non_central_t_test = "non central t test"
    weibull = "weibull"
    chi = "chi"
    inverse_gaussian = "inverse gaussian"
    extreme_value_1 = "extreme value 1"
    p_value = "p value"
    log_p_value = "log p value"
    log10_p_value = "log10 p value"
    estimate = "estimate"
    label = "label"
    neuroname = "neuroname"
    general_matrix = "general matrix"
    symmetric_matrix = "symmetric matrix"
    displacement_vector = "displacement vector"
    vector = "vector"
    pointset = "pointset"
    triangle = "triangle"
    quaternion = "quaternion"
    dimensionless = "dimensionless"
    time_series = "time series"
    node_index = "node index"
    rgb_vector = "rgb vector"
    rgba_vector = "rgba vector"
    shape = "shape"
    fnirt_disp_field = "fnirt disp field"
    fnirt_cubic_spline_coef = "fnirt cubic spline coef"
    fnirt_dct_coef = "fnirt dct coef"
    fnirt_quad_spline_coef = "fnirt quad spline coef"
    topup_field = "topup field"
# fmt: on
# ruff: enable[E501]


class Manufacturer(StrEnum):
    """Known values of the `manufacturer` metadata field.

    The names follow BIDS `Manufacturer`, spelled as converters such as
    dcm2niix normalise the DICOM Manufacturer tag. Other manufacturers are
    kept as plain strings.
    """

    Siemens = "Siemens"
    GE = "GE"
    Philips = "Philips"
    Canon = "Canon"
    Toshiba = "Toshiba"
    Hitachi = "Hitachi"
    Bruker = "Bruker"
    UIH = "UIH"
    MRSolutions = "MRSolutions"


class IlluminationType(StrEnum):
    """Known values of the `illumination_type` metadata field.

    The members are those of the OME `Channel.IlluminationType`
    enumeration. Other values are kept as plain strings.
    """

    Transmitted = "Transmitted"
    Epifluorescence = "Epifluorescence"
    Oblique = "Oblique"
    NonLinear = "NonLinear"
    Other = "Other"


class ContrastMethod(StrEnum):
    """Known values of the `contrast_method` metadata field.

    The members are those of the OME `Channel.ContrastMethod` enumeration.
    Other values are kept as plain strings.
    """

    Brightfield = "Brightfield"
    Phase = "Phase"
    DIC = "DIC"
    HoffmanModulation = "HoffmanModulation"
    ObliqueIllumination = "ObliqueIllumination"
    PolarizedLight = "PolarizedLight"
    Darkfield = "Darkfield"
    Fluorescence = "Fluorescence"
    Other = "Other"
