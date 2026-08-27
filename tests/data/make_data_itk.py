import json
import os.path as op

import numpy as np
import SimpleITK as sitk

this_dir = op.dirname(op.abspath(__file__))

center3d_vector = [50.0, 50.0, 50.0]
translation3d_vector = [10.0, 5.0, 2.0]
scale3d_vector = [1.2, 0.8, 1.0]
affine3d_matrix = np.array([
    [0.9, 0.1, 0.0],
    [-0.1, 0.9, 0.0],
    [0.0, 0.0, 1.0]
])

# --- Image 3D ---------------------------------------------------------

shape = [64, 92, 128]
dtype = sitk.sitkFloat32
num_channels = 1
voxel_size = [0.97, 0.97, 1.25]
origin_coords = [-120.5, -120.5, -60.0]
orientation_matrix = [
    -1.0,  0.0,  0.0,
    0.0, -1.0,  0.0,
    0.0,  0.0,  1.0
]

image = sitk.Image(shape, dtype, num_channels)
image.SetSpacing(voxel_size)
image.SetOrigin(origin_coords)
image.SetDirection(orientation_matrix)

# --- IdentityTransform 3D ---------------------------------------------

identity3d = sitk.Transform(3, sitk.sitkIdentity)

sitk.WriteTransform(identity3d, op.join(this_dir, "itk_identity3d.tfm"))
sitk.WriteTransform(identity3d, op.join(this_dir, "itk_identity3d.h5"))
json.dump(
    {
        "type": "IdentityTransform",
        "ndim": 3
    },
    open(op.join(this_dir, "itk_identity3d.json"), "w"),
    indent=4
)

# --- TranslationTransform 3D ------------------------------------------

translation3d = sitk.TranslationTransform(3)
translation3d.SetOffset(translation3d_vector)

sitk.WriteTransform(translation3d, op.join(this_dir, "itk_translation3d.tfm"))
sitk.WriteTransform(translation3d, op.join(this_dir, "itk_translation3d.h5"))
json.dump(
    {
        "type": "TranslationTransform",
        "ndim": 3,
        "offset": translation3d_vector
    },
    open(op.join(this_dir, "itk_translation3d.json"), "w"),
    indent=4
)


# --- ScaleTransform 3D ------------------------------------------------

scale3d = sitk.ScaleTransform(3)
scale3d.SetScale(scale3d_vector)

sitk.WriteTransform(scale3d, op.join(this_dir, "itk_scale3d.tfm"))
sitk.WriteTransform(scale3d, op.join(this_dir, "itk_scale3d.h5"))
json.dump(
    {
        "type": "ScaleTransform",
        "ndim": 3,
        "scale": scale3d_vector
    },
    open(op.join(this_dir, "itk_scale3d.json"), "w"),
    indent=4
)

# --- AffineTransform 2D -----------------------------------------------

affine2d = sitk.AffineTransform(2)
affine2d.SetMatrix(affine3d_matrix[:2, :2].flatten().tolist())
affine2d.SetTranslation(translation3d_vector[:2])
affine2d.SetCenter(center3d_vector[:2])

sitk.WriteTransform(affine2d, op.join(this_dir, "itk_affine2d.tfm"))
sitk.WriteTransform(affine2d, op.join(this_dir, "itk_affine2d.h5"))
json.dump(
    {
        "type": "AffineTransform",
        "ndim": 2,
        "matrix": affine3d_matrix[:2, :2].tolist(),
        "translation": translation3d_vector[:2],
        "center": center3d_vector[:2]
    },
    open(op.join(this_dir, "itk_affine2d.json"), "w"),
    indent=4
)

# --- AffineTransform 3D -----------------------------------------------

affine3d = sitk.AffineTransform(3)
affine3d.SetMatrix(affine3d_matrix.flatten().tolist())
affine3d.SetTranslation(translation3d_vector)
affine3d.SetCenter(center3d_vector)

sitk.WriteTransform(affine3d, op.join(this_dir, "itk_affine3d.tfm"))
sitk.WriteTransform(affine3d, op.join(this_dir, "itk_affine3d.h5"))
json.dump(
    {
        "type": "AffineTransform",
        "ndim": 3,
        "matrix": affine3d_matrix.tolist(),
        "translation": translation3d_vector,
        "center": center3d_vector
    },
    open(op.join(this_dir, "itk_affine3d.json"), "w"),
    indent=4
)

# --- BSplineTransform 3D -----------------------------------------------


mesh_size = [5] * image.GetDimension()
bspline3d = sitk.BSplineTransformInitializer(image, mesh_size)

coeff3d = np.random.randn(bspline3d.GetNumberOfParameters())
bspline3d.SetParameters(coeff3d.tolist())

sitk.WriteTransform(bspline3d, op.join(this_dir, "itk_bspline3d.tfm"))
sitk.WriteTransform(bspline3d, op.join(this_dir, "itk_bspline3d.h5"))
json.dump(
    {
        "type": "BSplineTransform",
        "ndim": 3,
        "parameters": bspline3d.GetParameters(),
        "fixed_parameters": bspline3d.GetFixedParameters()
    },
    open(op.join(this_dir, "itk_bspline3d.json"), "w"),
    indent=4
)

# --- DisplacementFieldTransform 3D ------------------------------------

values3d = np.random.randn(*image.GetSize(), image.GetDimension())

displacement3d = sitk.GetImageFromArray(values3d.transpose(2, 1, 0, 3))
displacement3d.CopyInformation(image)
displacement3d = sitk.DisplacementFieldTransform(displacement3d)

sitk.WriteTransform(displacement3d, op.join(
    this_dir, "itk_displacement3d.tfm"))
sitk.WriteTransform(displacement3d, op.join(this_dir, "itk_displacement3d.h5"))


# --- CompositeTransform 3D --------------------------------------------

composite_affine3d = sitk.CompositeTransform(3)
composite_affine3d.AddTransform(affine3d)
composite_affine3d.AddTransform(scale3d)

sitk.WriteTransform(composite_affine3d, op.join(
    this_dir, "itk_composite_affine3d.tfm"))
sitk.WriteTransform(composite_affine3d, op.join(
    this_dir, "itk_composite_affine3d.h5"))

composite_bsplines3d = sitk.CompositeTransform(3)
composite_bsplines3d.AddTransform(affine3d)
composite_bsplines3d.AddTransform(bspline3d)

sitk.WriteTransform(composite_bsplines3d, op.join(
    this_dir, "itk_composite_bsplines3d.tfm"))
sitk.WriteTransform(composite_bsplines3d, op.join(
    this_dir, "itk_composite_bsplines3d.h5"))

composite_displacement3d = sitk.CompositeTransform(3)
composite_displacement3d.AddTransform(affine3d)
composite_displacement3d.AddTransform(displacement3d)

sitk.WriteTransform(composite_displacement3d, op.join(
    this_dir, "itk_composite_displacement3d.tfm"))
sitk.WriteTransform(composite_displacement3d, op.join(
    this_dir, "itk_composite_displacement3d.h5"))
