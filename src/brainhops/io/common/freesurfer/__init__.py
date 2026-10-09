"""
The volume geometry shared by every FreeSurfer format.

FreeSurfer describes the world placement of a volume the same way in
every file that records one -- the header of an MGH/MGZ image, the source
and destination blocks of an LTA transform, the source and atlas
geometries of a non-linear morph (`.m3z`), ... -- with:

- the volume's shape, in voxels (`width, height, depth`);
- the voxel size (`xsize, ysize, zsize`), in millimetres;
- the direction cosines of each voxel axis in RAS (`x_ras`, `y_ras`,
  `z_ras`), which form the columns of the rotation part of the
  voxel-to-RAS matrix;
- the RAS coordinates of the centre of the volume (`c_ras`, `Pxyz_c`).

Three coordinate systems derive from it:

- **scanner RAS**, the world space the volume was acquired in
  (`mri_info --vox2ras`);
- **tkr RAS** (also "surface RAS" or "tkregister RAS"), the space
  FreeSurfer surfaces live in. It has the same voxel sizes but drops the
  direction cosines and `c_ras`: the centre of the volume is the origin,
  and the axes are those of a conformed (LIA) volume
  (`mri_info --vox2ras-tkr`);
- **physical** (or "physvox"), the scaled voxel space shifted so that
  its origin is the centre of the volume. It is the space between
  voxels and scanner RAS used by LTA files of type `LINEAR_PHYSVOX`.

!!! note "The centre of the volume"
    FreeSurfer places the centre of the volume at voxel coordinate
    `shape / 2`, not at `(shape - 1) / 2`. With 0-based voxel
    coordinates whose integers are voxel centres, the centre therefore
    falls half a voxel past the true centre of an even-sized volume.
    This is FreeSurfer's convention, and every matrix here follows it so
    that it matches FreeSurfer and `nibabel` exactly.

The functions take the geometry as plain values, so that each format
reads it from wherever it stores it.

Every FreeSurfer format -- MGH/MGZ images, LTA transforms, morphs, ... --
derives from [`FreesurferFormat`][brainhops.io.common.freesurfer.
FreesurferFormat], so that the `"freesurfer"` hint selects them all.
"""

__all__ = [
    "FreesurferFormat",
]

from ._formats import FreesurferFormat
