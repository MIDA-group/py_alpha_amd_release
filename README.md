# Project: py_alpha_amd
Python/NumPy/SciPy implementation of a registration framework using the Alpha AMD similarity measure, which combines intensity and spatial information. (Ref: https://arxiv.org/abs/1807.11599)

The Alpha AMD similarity measure typically exhibits few local optima in the transformation parameter search space, compared to other commonly used measures, which makes the registration substantially more robust/less sensitive to the starting position, when used in a local search framework.

This framework consists of a collection of common transformation models which can be composed, combined and even extended with new transformations, driven by a gradient descent optimizer to find the transformation parameters which reduces the cost function.

# Features
- The framework/measure supports images, represented by numpy arrays, with additional (anisotropic) voxel-sizes.
- 2D and 3D registration (ND for affine transformations)
- Completely python/numpy/scipy based codebase. No C/C++/... code, which should facilitate portability and understandability.

# Example
Try out the provided sample script:
python2 register\_example.py ./test_images/reference\_example.png ./test_images/floating\_example.png

# License
The registration framework is licensed under the permissive MIT license.

# Author/Copyright
Framework was written by (and copyright reserved for) Johan Ofverstedt.
