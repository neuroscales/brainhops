"""
Format-specific machinery shared by the image and transformation readers.

Each subpackage holds what the image readers (`brainhops.io.images`) and
the transformation readers (`brainhops.io.transformations`) of one file
format have in common -- its parser, header, and geometry conventions:

- [`afni`][brainhops.io.common.afni]: AFNI `.HEAD`/`.BRIK` datasets;
- [`freesurfer`][brainhops.io.common.freesurfer]: the volume geometry
  shared by every FreeSurfer format;
- [`hdf5`][brainhops.io.common.hdf5]: formats stored in HDF5 files
  (needs `h5py`);
- [`jpeg2000`][brainhops.io.common.jpeg2000]: the header of JP2 files and
  JPEG 2000 codestreams;
- [`mgh`][brainhops.io.common.mgh]: FreeSurfer MGH/MGZ volumes (needs
  `nibabel`);
- [`minc`][brainhops.io.common.minc]: MINC1 and MINC2 volumes (needs
  `nibabel`, and `h5py` for MINC2);
- [`mrtrix`][brainhops.io.common.mrtrix]: MRtrix `.mif`/`.mih` images;
- [`nifti`][brainhops.io.common.nifti]: NIfTI-1 and NIfTI-2 files (needs
  `nibabel`);
- [`nrrd`][brainhops.io.common.nrrd]: NRRD `.nrrd`/`.nhdr` files;
- [`zarr`][brainhops.io.common.zarr]: Zarr stores (needs `abczarr` and
  one of its drivers).

None of them is imported here: a format whose optional dependencies are
missing must not break `brainhops.io`, so each one is imported by the
readers that need it, and only when those dependencies are installed.
"""

__all__ = []
