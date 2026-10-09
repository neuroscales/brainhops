"""
Volume geometry shared by the FreeSurfer formats.

MGH headers, LTA files and `.m3z` files all describe a volume by its
shape, its voxel size in millimetres, the direction cosines `x_ras`,
`y_ras` and `z_ras` (the columns of the rotation part of the
voxel-to-RAS matrix), and the RAS coordinate `c_ras` of its centre.

Three coordinate systems are involved. Scanner RAS is the acquisition
space (`mri_info --vox2ras`). Tkr RAS, where surfaces live
(`mri_info --vox2ras-tkr`), has the same voxel sizes but conformed LIA
axes and its origin at the centre of the volume. The physical space,
used by `LINEAR_PHYSVOX` transforms, is the scaled voxel space shifted
to the centre of the volume.

!!! note "The centre of the volume"
    FreeSurfer puts the centre of the volume at voxel coordinate
    `shape / 2`, which lies half a voxel past the true centre along
    axes of even size. The functions follow this convention to match
    FreeSurfer and nibabel exactly.

All FreeSurfer formats derive from [`FreesurferFormat`][].
"""

__all__ = [
    "FreesurferFormat",
]

from ._formats import FreesurferFormat
