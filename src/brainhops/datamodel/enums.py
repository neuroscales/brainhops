"""Enumerations shared across the data model, such as interpolation orders
and boundary conditions."""

__all__ = [
    "BoundaryCondition",
    "InterpolationOrder",
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
    """How hard a transformation may be looked at, and therefore how far it
    may be simplified.

    * `none` -- declared type only: nothing is inspected, nothing is
      rewritten.
    * `analytic` -- structure only: `None` parameters, array shapes, axis
      lists and wrapper contents are read; no value is read and no lazy
      inverse is materialized. Invertibility, injectivity and surjectivity
      of a matrix transformation are *assumed from its shape* at this level
      (a square matrix is presumed invertible, a wide one surjective, a tall
      one injective).
    * `numeric` -- values too: zero tests, diagonality, orthogonality, rank;
      a typed inverse may be materialized when a leaf is downcast. Rank
      replaces the analytic shape assumption, so a square singular matrix is
      *not* invertible here.
    """

    none = "none"
    analytic = "analytic"
    numeric = "numeric"


# ruff: disable[E501]
# fmt: off
class BoundaryCondition(StrEnum):
    """Boundary conditions for interpolation and resampling.

    The following boundary conditions are supported:

    | Value         | Aliases                        | Description                     |
    |---------------|--------------------------------|---------------------------------|
    | `nearest`     | `edge`, `border`               | <code>(a a a a &vert; a b c d &vert; d d d d)</code> |
    | `reflect`     | `symmetric`, `dct2`            | <code>(d c b a &vert; a b c d &vert; d c b a)</code> |
    | `mirror`      | `dct1`                         | <code>  (d c b &vert; a b c d &vert; c b a)  </code> |
    | `grid-wrap`   | `circular`, `circulant`, `dft` | <code>(a b c d &vert; a b c d &vert; a b c d)</code> |
    | `wrap`        |                                | <code>(d b c d &vert; a b c d &vert; b c a b)</code> |
    | `constant`    | `zero`, `zeros`                | <code>(0 0 0 0 &vert; a b c d &vert; 0 0 0 0)</code> |
    """

    nearest = edge = border = "nearest"                     # (a a a a | a b c d | d d d d)
    reflect = symmetric = dct2 = "reflect"                  # (d c b a | a b c d | d c b a)
    mirror = dct1 = "mirror"                                #   (d c b | a b c d | c b a)
    gridwrap = circular = circulant = dft = "grid-wrap"     # (a b c d | a b c d | a b c d)
    wrap = "wrap"                                           # (d b c d | a b c d | b c a b)
    constant = zero = zeros = "constant"                    # (0 0 0 0 | a b c d | 0 0 0 0)
# fmt: on
# ruff: enable[E501]


class InterpolationOrder(IntEnum):
    """Interpolation order for interpolation and resampling.

    The following interpolation orders are supported:

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
    """Axis types for coordinate systems and transformations.

    The following axis types are supported:

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
    """
    Orientation types for coordinate systems and transformations.

    Currently, only the `"anatomical"` orientation is supported.

    | Name          | Value          | Description             |
    |---------------|----------------|-------------------------|
    | `anatomical`  | `"anatomical"` | Anatomical orientation. |
    """

    anatomical = "anatomical"


# ruff: disable[E501]
# fmt: off
class AnatomicalOrientationValue(StrEnum):
    """
    Anatomical orientation values for coordinate systems and transformations.

    The following values are supported:

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
    | `caudal_to_rostral`     | ` "caudal-to-rostral"`    | tail-to-nose/beak, especially for nervous system
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
    # Common to both bipeds and quadrupeds
    left_to_right = "left-to-right"
    right_to_left = "right-to-left"
    proximal_to_distal = "proximal-to-distal"
    distal_to_proximal = "distal-to-proximal"

    # Primarily for bipeds (humans)
    anterior_to_posterior = "anterior-to-posterior"
    posterior_to_anterior = "posterior-to-anterior"
    inferior_to_superior = "inferior-to-superior"
    superior_to_inferior = "superior-to-inferior"
    dorsal_to_palmar = "dorsal-to-palmar"
    palmar_to_dorsal = "palmar-to-dorsal"
    dorsal_to_plantar = "dorsal-to-plantar"
    plantar_to_dorsal = "plantar-to-dorsal"

    # Primarily for quadrupeds:
    rostral_to_caudal = "rostral-to-caudal"
    caudal_to_rostral = "caudal-to-rostral"
    cranial_to_caudal = "cranial-to-caudal"
    caudal_to_cranial = "caudal-to-cranial"
    dorsal_to_ventral = "dorsal-to-ventral"
    ventral_to_dorsal = "ventral-to-dorsal"

    # For layered and polarized tissues (subject-local):
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
#
# The known terms of the free-text fields of `brainhops.datamodel.metadata`.
# A field typed `Union[<Enum>, str]` holds a member when its value is one
# of these, and the string otherwise: the lists help consistency and
# documentation, and do not close the vocabulary.


# ruff: disable[E501]
# fmt: off
class SpaceEnum(StrEnum):
    """
    The known labels of a world space, used by the `space`, `input_space`
    and `output_space` metadata fields.

    The first five members are the names that NIfTI gives to its
    `sform_code` and `qform_code` values. They are followed by the BIDS
    template identifiers, which BIDS uses for `SpatialReference` and for
    the `space-` entity of a file name, and by the spaces that BIDS
    defines without a template. A label that is not listed here is kept
    as a plain string.

    | Member                 | Value                    | Meaning                                                  |
    |------------------------|--------------------------|----------------------------------------------------------|
    | `scanner`              | `"scanner"`              | NIfTI xform code 1: scanner-based anatomical coordinates |
    | `aligned`              | `"aligned"`              | NIfTI xform code 2: aligned to another file              |
    | `talairach`            | `"talairach"`            | NIfTI xform code 3: Talairach-Tournoux atlas             |
    | `mni`                  | `"mni"`                  | NIfTI xform code 4: MNI 152                              |
    | `template`             | `"template"`             | NIfTI xform code 5: another template                     |
    | `ICBM452AirSpace`      | `"ICBM452AirSpace"`      | BIDS standard template (volume)                          |
    | `ICBM452Warp5Space`    | `"ICBM452Warp5Space"`    | BIDS standard template (volume)                          |
    | `IXI549Space`          | `"IXI549Space"`          | BIDS standard template (volume)                          |
    | `MNI152Lin`            | `"MNI152Lin"`            | BIDS standard template (volume)                          |
    | `MNI152NLin2009aAsym`  | `"MNI152NLin2009aAsym"`  | BIDS standard template (volume)                          |
    | `MNI152NLin2009aSym`   | `"MNI152NLin2009aSym"`   | BIDS standard template (volume)                          |
    | `MNI152NLin2009bAsym`  | `"MNI152NLin2009bAsym"`  | BIDS standard template (volume)                          |
    | `MNI152NLin2009bSym`   | `"MNI152NLin2009bSym"`   | BIDS standard template (volume)                          |
    | `MNI152NLin2009cAsym`  | `"MNI152NLin2009cAsym"`  | BIDS standard template (volume)                          |
    | `MNI152NLin2009cSym`   | `"MNI152NLin2009cSym"`   | BIDS standard template (volume)                          |
    | `MNI152NLin6Asym`      | `"MNI152NLin6Asym"`      | BIDS standard template (volume)                          |
    | `MNI152NLin6Sym`       | `"MNI152NLin6Sym"`       | BIDS standard template (volume)                          |
    | `MNI305`               | `"MNI305"`               | BIDS standard template (volume)                          |
    | `MNIColin27`           | `"MNIColin27"`           | BIDS standard template (volume)                          |
    | `MNIInfant`            | `"MNIInfant"`            | BIDS standard template (volume)                          |
    | `MNIPediatricAsym`     | `"MNIPediatricAsym"`     | BIDS standard template (volume)                          |
    | `NMT31Sym`             | `"NMT31Sym"`             | BIDS standard template (volume)                          |
    | `OASIS30AntsOASISAnts` | `"OASIS30AntsOASISAnts"` | BIDS standard template (volume)                          |
    | `OASIS30Atropos`       | `"OASIS30Atropos"`       | BIDS standard template (volume)                          |
    | `Talairach`            | `"Talairach"`            | BIDS standard template (volume)                          |
    | `UNCInfant`            | `"UNCInfant"`            | BIDS standard template (volume)                          |
    | `fsaverage`            | `"fsaverage"`            | BIDS standard template (surface)                         |
    | `fsaverage3`           | `"fsaverage3"`           | BIDS standard template (surface)                         |
    | `fsaverage4`           | `"fsaverage4"`           | BIDS standard template (surface)                         |
    | `fsaverage5`           | `"fsaverage5"`           | BIDS standard template (surface)                         |
    | `fsaverage6`           | `"fsaverage6"`           | BIDS standard template (surface)                         |
    | `fsaveragesym`         | `"fsaveragesym"`         | BIDS standard template (surface)                         |
    | `fsLR`                 | `"fsLR"`                 | BIDS standard template (surface)                         |
    | `fsnative`             | `"fsnative"`             | BIDS standard template (surface)                         |
    | `orig`                 | `"orig"`                 | BIDS non-standard space                                  |
    | `anat`                 | `"anat"`                 | BIDS non-standard space                                  |
    | `T1w`                  | `"T1w"`                  | BIDS non-standard space                                  |
    | `T2w`                  | `"T2w"`                  | BIDS non-standard space                                  |
    | `individual`           | `"individual"`           | BIDS non-standard space                                  |
    | `study`                | `"study"`                | BIDS non-standard space                                  |
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
    """
    The known values of the `intent` metadata field, which says what the
    values of an image represent.

    The members are the NIfTI intent names, spelled as `nibabel` spells
    them, with their NIfTI intent codes. Codes 2 to 24 name statistical
    distributions, codes 1001 to 2005 name other kinds of values, and the
    codes from 2006 onwards are FSL's own. An intent that is not listed
    here is kept as a plain string.

    | Member                    | Value                       | NIfTI code |
    |---------------------------|-----------------------------|------------|
    | `correlation`             | `"correlation"`             | 2          |
    | `t_test`                  | `"t test"`                  | 3          |
    | `f_test`                  | `"f test"`                  | 4          |
    | `z_score`                 | `"z score"`                 | 5          |
    | `chi2`                    | `"chi2"`                    | 6          |
    | `beta`                    | `"beta"`                    | 7          |
    | `binomial`                | `"binomial"`                | 8          |
    | `gamma`                   | `"gamma"`                   | 9          |
    | `poisson`                 | `"poisson"`                 | 10         |
    | `normal`                  | `"normal"`                  | 11         |
    | `non_central_f_test`      | `"non central f test"`      | 12         |
    | `non_central_chi2`        | `"non central chi2"`        | 13         |
    | `logistic`                | `"logistic"`                | 14         |
    | `laplace`                 | `"laplace"`                 | 15         |
    | `uniform`                 | `"uniform"`                 | 16         |
    | `non_central_t_test`      | `"non central t test"`      | 17         |
    | `weibull`                 | `"weibull"`                 | 18         |
    | `chi`                     | `"chi"`                     | 19         |
    | `inverse_gaussian`        | `"inverse gaussian"`        | 20         |
    | `extreme_value_1`         | `"extreme value 1"`         | 21         |
    | `p_value`                 | `"p value"`                 | 22         |
    | `log_p_value`             | `"log p value"`             | 23         |
    | `log10_p_value`           | `"log10 p value"`           | 24         |
    | `estimate`                | `"estimate"`                | 1001       |
    | `label`                   | `"label"`                   | 1002       |
    | `neuroname`               | `"neuroname"`               | 1003       |
    | `general_matrix`          | `"general matrix"`          | 1004       |
    | `symmetric_matrix`        | `"symmetric matrix"`        | 1005       |
    | `displacement_vector`     | `"displacement vector"`     | 1006       |
    | `vector`                  | `"vector"`                  | 1007       |
    | `pointset`                | `"pointset"`                | 1008       |
    | `triangle`                | `"triangle"`                | 1009       |
    | `quaternion`              | `"quaternion"`              | 1010       |
    | `dimensionless`           | `"dimensionless"`           | 1011       |
    | `time_series`             | `"time series"`             | 2001       |
    | `node_index`              | `"node index"`              | 2002       |
    | `rgb_vector`              | `"rgb vector"`              | 2003       |
    | `rgba_vector`             | `"rgba vector"`             | 2004       |
    | `shape`                   | `"shape"`                   | 2005       |
    | `fnirt_disp_field`        | `"fnirt disp field"`        | 2006       |
    | `fnirt_cubic_spline_coef` | `"fnirt cubic spline coef"` | 2007       |
    | `fnirt_dct_coef`          | `"fnirt dct coef"`          | 2008       |
    | `fnirt_quad_spline_coef`  | `"fnirt quad spline coef"`  | 2009       |
    | `topup_field`             | `"topup field"`             | 2018       |
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
    """
    The known values of the `manufacturer` metadata field (BIDS
    `Manufacturer`), spelled as converters such as `dcm2niix` normalise the
    DICOM `Manufacturer` tag. Any other manufacturer is kept as a plain
    string.

    | Member        | Value           |
    |---------------|-----------------|
    | `Siemens`     | `"Siemens"`     |
    | `GE`          | `"GE"`          |
    | `Philips`     | `"Philips"`     |
    | `Canon`       | `"Canon"`       |
    | `Toshiba`     | `"Toshiba"`     |
    | `Hitachi`     | `"Hitachi"`     |
    | `Bruker`      | `"Bruker"`      |
    | `UIH`         | `"UIH"`         |
    | `MRSolutions` | `"MRSolutions"` |
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
    """
    The known values of the `illumination_type` metadata field, which are
    the members of the OME `Channel.IlluminationType` enumeration. Any
    other value is kept as a plain string.

    | Member            | Value               |
    |-------------------|---------------------|
    | `Transmitted`     | `"Transmitted"`     |
    | `Epifluorescence` | `"Epifluorescence"` |
    | `Oblique`         | `"Oblique"`         |
    | `NonLinear`       | `"NonLinear"`       |
    | `Other`           | `"Other"`           |
    """

    Transmitted = "Transmitted"
    Epifluorescence = "Epifluorescence"
    Oblique = "Oblique"
    NonLinear = "NonLinear"
    Other = "Other"


class ContrastMethod(StrEnum):
    """
    The known values of the `contrast_method` metadata field, which are the
    members of the OME `Channel.ContrastMethod` enumeration. Any other
    value is kept as a plain string.

    | Member                | Value                   |
    |-----------------------|-------------------------|
    | `Brightfield`         | `"Brightfield"`         |
    | `Phase`               | `"Phase"`               |
    | `DIC`                 | `"DIC"`                 |
    | `HoffmanModulation`   | `"HoffmanModulation"`   |
    | `ObliqueIllumination` | `"ObliqueIllumination"` |
    | `PolarizedLight`      | `"PolarizedLight"`      |
    | `Darkfield`           | `"Darkfield"`           |
    | `Fluorescence`        | `"Fluorescence"`        |
    | `Other`               | `"Other"`               |
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
