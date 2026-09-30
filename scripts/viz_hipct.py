"""Reslice randomly oriented patches from a HiP-CT Zarr through one warp.

One smooth deformation is drawn for the whole volume and shared by every
patch. Each patch contributes only an affine: its voxel size, rotation and
translation. The reslice is therefore ``source(warp).reslice(geometry)``,
which reads as "warp the source, then sample it on this patch grid".

The default input is the supplied public HiP-CT OME-Zarr HTTPS URL::

    python scripts/viz_hipct.py --output hipct_patches.png

With no ``--level`` the whole pyramid is resliced, and the level whose
resolution best matches each patch is chosen for it. Passing ``--level``
pins one level instead, which is useful for comparing the two.

The DANDI page for the source data is:
https://dandiarchive.org/dandiset/001278/draft/files?location=sourcedata/raw/sub-Hb1/micr
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from brainhops.backends import backend
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    DisplacementField,
    Transformation,
)
from brainhops.io.images.zarr import OmeZarrImage

HIPCT_ZARR_URL = (
    "https://dandiarchive.s3.amazonaws.com/zarr/"
    "6f11427f-ef86-42b8-9800-d3e09265c965/"
)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a random 3-D rotation matrix."""
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def smooth_warp(
    reference: SingleScaleImage,
    amplitude: float,
    nodes: Tuple[int, int, int],
    rng: np.random.Generator,
) -> Transformation:
    """One smooth world-to-world warp, shared by every patch.

    The field carries only ``nodes`` samples across the *whole* volume, so
    its node spacing is a large fraction of the source and the deformation
    is smooth on the scale of the volume rather than of a patch. A cubic
    spline interpolates between the nodes.

    `amplitude` sets the scale of the deformation, in voxels of
    `reference`: it bounds the node values, and the realised maximum comes
    out somewhat smaller, since a cubic spline does not reach its node
    values between them. The field is built in those units and then divided
    by the node spacing, because a displacement field displaces by its own
    input units and the conjugation below scales each component by that
    spacing.
    """
    source_voxels = reference.transformation.input
    node_voxels = CoordinateSystem(name="warp-node voxels")

    # The node grid spans the source, corner node to corner node.
    extent = np.asarray(reference.shape, dtype=float) - 1
    spacing = extent / (np.asarray(nodes) - 1)
    node_to_source = Affine(
        matrix=np.column_stack((np.diag(spacing), np.zeros(3))),
        input=node_voxels,
        output=source_voxels,
    )

    field = gaussian_filter(
        rng.normal(size=(*nodes, 3)), sigma=(1, 1, 1, 0), mode="reflect"
    )
    field *= amplitude / np.linalg.norm(field, axis=-1).max()
    field /= spacing

    deformation = DisplacementField(
        field=field,
        input=node_voxels,
        output=node_voxels,
        order=3,
        bound="reflect",
    )
    # Lift the coarse field into source voxels, then into world space, so the
    # warp composes with an image placed in world space.
    source_warp = node_to_source @ deformation @ node_to_source.inverse()
    voxel_to_world = reference.transformation
    return voxel_to_world @ source_warp @ voxel_to_world.inverse()


def patch_geometry(
    reference: SingleScaleImage,
    shape: Tuple[int, int, int],
    voxel_size: float,
    rng: np.random.Generator,
) -> Geometry:
    """One rotated output grid, as a pure affine.

    The patch carries only its voxel size, its rotation and its
    translation. The deformation belongs to the source, not to the patch,
    so nothing but an affine appears here.

    The patch centre is drawn so that the unwarped patch lies inside the
    source. The warp may still push individual samples just outside it,
    which is harmless: those samples take the boundary condition.
    """
    if reference.ndim != 3:
        raise ValueError("This example expects a single-channel 3-D CT image.")

    source_voxels = reference.transformation.input
    patch_voxels = CoordinateSystem(name="patch voxels")

    rotation = random_rotation(rng)
    linear = rotation * voxel_size
    half_extent = (np.asarray(shape) - 1) / 2
    # The rotated patch's half-width along each source axis.
    half_width = np.abs(linear) @ half_extent
    available = np.asarray(reference.shape) - 2 * half_width
    if np.any(available <= 0):
        raise ValueError(
            "The requested patch does not fit in this resolution level; "
            "use a smaller patch or a finer level."
        )
    center = half_width + rng.random(3) * available
    translation = center - linear @ half_extent

    patch_to_source = Affine(
        matrix=np.column_stack((linear, translation)),
        input=patch_voxels,
        output=source_voxels,
    )
    grid = CartesianField(shape=shape, input=patch_voxels, output=patch_voxels)
    return Geometry((grid, reference.transformation @ patch_to_source))


def reslice_patches(
    source: Union[SingleScaleImage, MultiScaleImage],
    reference: SingleScaleImage,
    warp: Transformation,
    count: int,
    shape: Tuple[int, int, int],
    voxel_size: float,
    seed: int,
) -> Sequence[np.ndarray]:
    """Reslice `count` random patches through one shared warp.

    `source` is what gets sampled: a whole pyramid, in which case the level
    matching each patch is chosen, or a single level. `reference` is the
    finest level, which fixes the world placement the patches are drawn in.
    """
    rng = np.random.default_rng(seed)
    # The warp is applied once, to the source. Every patch then samples the
    # same warped image on its own affine grid.
    warped = source(warp)
    patches = []
    for index in range(count):
        print(f"patch {index}", end=" ... ", flush=True)
        geometry = patch_geometry(reference, shape, voxel_size, rng)
        patches.append(np.asarray(warped.reslice(geometry, order=1)))
        print("done.", flush=True)
    return patches


def show_patches(patches: Sequence[np.ndarray], output: Path) -> None:
    """Save central axial, coronal, and sagittal slices for every patch."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(patches), 3, squeeze=False, figsize=(9, 3 * len(patches))
    )
    for row, patch in zip(axes, patches):
        middle = np.asarray(patch.shape) // 2
        views = (
            patch[:, :, middle[2]].T,
            patch[:, middle[1], :].T,
            patch[middle[0], :, :].T,
        )
        for axis, view, name in zip(
            row, views, ("axial", "coronal", "sagittal")
        ):
            axis.imshow(view, cmap="gray", origin="lower")
            axis.set_title(name)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=180)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zarr_url",
        nargs="?",
        default=HIPCT_ZARR_URL,
        help="public HTTPS URL of an OME-Zarr asset",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="pin one OME-Zarr resolution level; by default the whole "
        "pyramid is resliced and the matching level is chosen per patch",
    )
    parser.add_argument(
        "--count", type=int, default=4, help="number of patches"
    )
    parser.add_argument("--shape", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=2.0,
        help="output spacing, in voxels of the finest level",
    )
    parser.add_argument(
        "--warp-amplitude",
        type=float,
        default=64.0,
        help="scale of the deformation, in voxels of the finest level; "
        "the realised maximum is somewhat smaller",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        nargs=3,
        default=(4, 4, 4),
        help="deformation nodes across the whole volume; few nodes make a "
        "deformation that is smooth on the scale of the volume",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("hipct_patches.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    with backend("dask"):
        # `from_store` rather than `load`: a store is a directory, and a
        # directory has no object of its own over plain HTTPS, so `load`'s
        # "does this path exist" check rejects a remote store URL.
        image = OmeZarrImage.from_store(args.zarr_url)
        # The finest level fixes the world placement and the voxel units the
        # patch and warp arguments are expressed in.
        reference = image.to_singlescale(0)
        if args.level is None:
            source = image
        else:
            source = image.to_singlescale(args.level)
        warp = smooth_warp(
            reference,
            args.warp_amplitude,
            tuple(args.nodes),
            np.random.default_rng(args.seed),
        )
        patches = reslice_patches(
            source,
            reference,
            warp,
            args.count,
            tuple(args.shape),
            args.voxel_size,
            args.seed,
        )
    show_patches(patches, args.output)
    print(f"Saved {len(patches)} patches to {args.output}")


if __name__ == "__main__":
    main()
