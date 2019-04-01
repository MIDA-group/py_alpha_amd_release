
#
# Py-Alpha-AMD Registration Framework
# Author: Johan Ofverstedt
# Reference: Fast and Robust Symmetric Image Registration Based on Distances Combining Intensity and Spatial Information
#
# Copyright 2019 Johan Ofverstedt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

#
# Example script for affine registration
#

# Import Numpy/Scipy
import numpy as np
import scipy as sp
import scipy.misc

# Import transforms
from transforms import CompositeTransform
from transforms import AffineTransform
from transforms import Rigid2DTransform
from transforms import Rotate2DTransform
from transforms import TranslationTransform
from transforms import ScalingTransform
import transforms

# Import optimizers
from optimizers import GradientDescentOptimizer

# Import generators and filters
import generators
import filters

# Import registration framework
from register import Register

# Import misc
import math
import sys
import time
import os

# Registration Parameters
alpha_levels = 7
# Pixel-size
spacing = np.array([1.0, 1.0])
# Run the symmetric version of the registration framework
symmetric_measure = True
# Use the squared Euclidean distance
squared_measure = False

# The number of iterations
param_iterations = 100
# The fraction of the points to sample randomly (0.0-1.0)
param_sampling_fraction = 0.1
# Number of iterations between each printed output (with current distance/gradient/parameters)
param_report_freq = 100

def main():
    np.random.seed(1000)
    
    if len(sys.argv) < 3:
        print('register_example.py: Too few parameters. Give the path to two gray-scale image files.')
        print('Example: python2 register_example.py reference_image floating_image')
        return False

    ref_im_path = sys.argv[1]
    flo_im_path = sys.argv[2]

    ref_im = scipy.misc.imread(ref_im_path, 'L')
    flo_im = scipy.misc.imread(flo_im_path, 'L')

    # Save copies of original images
    ref_im_orig = ref_im.copy()
    flo_im_orig = flo_im.copy()

    ref_im = filters.normalize(ref_im, 0.0, None)
    flo_im = filters.normalize(flo_im, 0.0, None)
    
    diag = 0.5 * (transforms.image_diagonal(ref_im, spacing) + transforms.image_diagonal(flo_im, spacing))

    weights1 = np.ones(ref_im.shape)
    mask1 = np.ones(ref_im.shape, 'bool')
    weights2 = np.ones(flo_im.shape)
    mask2 = np.ones(flo_im.shape, 'bool')

    # Initialize registration framework for 2d images
    reg = Register(2)

    reg.set_report_freq(param_report_freq)
    reg.set_alpha_levels(alpha_levels)

    reg.set_reference_image(ref_im)
    reg.set_reference_mask(mask1)
    reg.set_reference_weights(weights1)

    reg.set_floating_image(flo_im)
    reg.set_floating_mask(mask2)
    reg.set_floating_weights(weights2)

    # Setup the Gaussian pyramid resolution levels
    
    reg.add_pyramid_level(4, 5.0)
    reg.add_pyramid_level(2, 3.0)
    reg.add_pyramid_level(1, 0.0)

    # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...] (for each pyramid level)
    step_lengths = np.array([[1., 1.], [1., 1.], [1., 1e-1]]) * 1e-1

    # Create the transform and add it to the registration framework (switch between affine/rigid transforms by commenting/uncommenting)
    # Affine
    #1.0/diag, 1.0/diag, 1.0/diag, 1.0/diag, 1.0, 1.0
    reg.add_initial_transform(AffineTransform(2), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    # Rigid 2D
    #reg.add_initial_transform(Rigid2DTransform(2), np.array([1.0/diag, 1.0, 1.0]))

    # Set the parameters
    reg.set_iterations(param_iterations)
    reg.set_gradient_magnitude_threshold(0.001)
    reg.set_sampling_fraction(param_sampling_fraction)
    reg.set_step_lengths(step_lengths)
    reg.set_optimizer('adam')

    # Create output directory
    directory = os.path.dirname('./test_images/output/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start the pre-processing
    reg.initialize('./test_images/output/')
    
    # Control the formatting of numpy
    np.set_printoptions(suppress=True, linewidth=200)

    # Start the registration
    reg.run()

    (transform, value) = reg.get_output(0)

    ### Warp final image
    c = transforms.make_image_centered_transform(transform, ref_im, flo_im, spacing, spacing)

    # Print out transformation parameters
    print('Transformation parameters: %s.' % str(transform.get_params()))

    # Create the output image
    ref_im_warped = np.zeros(ref_im.shape)

    # Transform the floating image into the reference image space by applying transformation 'c'
    c.warp(In = flo_im_orig, Out = ref_im_warped, in_spacing=spacing, out_spacing=spacing, mode='spline', bg_value = 0.0)

    # Save the registered image
    scipy.misc.imsave('./test_images/output/registered.png', ref_im_warped)

    # Compute the absolute difference image between the reference and registered images
    D1 = np.abs(ref_im_orig-ref_im_warped)
    err = np.mean(D1)
    print("Err: %f" % err)

    scipy.misc.imsave('./test_images/output/diff.png', D1)

    return True

if __name__ == '__main__':
    start_time = time.time()
    res = main()
    end_time = time.time()
    if res == True:
        print("Elapsed time: " + str((end_time-start_time)))
