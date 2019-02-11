
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
# Registration framework
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

# Import distances
from distances import QuantizedImage
from distances import alpha_amd
from distances import symmetric_amd_distance

# Import optimizers
from optimizers import GradientDescentOptimizer

# Import generators and filters
import generators
import filters

# Import misc
import math
import sys
import time
import cProfile, pstats, StringIO

class Register:
    def __init__(self, dim):
        self.dim = dim
        self.sampling_fraction = 1.0
        self.step_lengths = np.array([[0.1, 1.0]])
        self.iterations = 1500
        self.alpha_levels = 7
        self.gradient_magnitude_threshold = 0.00001
        
        self.ref_im = None
        self.flo_im = None
        self.ref_mask = None
        self.flo_mask = None
        self.ref_weights = None
        self.flo_weights = None

        # Transforms
        self.initial_transforms = []
        self.transforms_param_scaling = []
        self.output_transforms = []
        self.values = []
        self.value_history = []

        # Resolution pyramid levels
        self.pyramid_factors = []
        self.pyramid_sigmas = []

        self.distances = []
        
        # Reporting/Output
        self.report_func = None
        self.report_freq = 25

    def add_initial_transform(self, transform, param_scaling=None):
        if param_scaling is None:
            param_scaling = np.ones((transforms.get_param_count(),))
        self.initial_transforms.append(transform)
        self.transforms_param_scaling.append(param_scaling)
    
    def add_initial_transforms(self, transforms, param_scaling=None):
        for i, t in enumerate(transforms):
            if param_scaling is None:
                pscaling = np.ones((transforms.get_param_count(),))
            else:
                pscaling = param_scaling[i]
            self.add_initial_transform(t, pscaling)
    
    def clear_transforms(self):
        self.initial_transforms = []
        self.output_transforms = []
        self.transforms_param_scaling = []
        self.values = []
        self.value_history = []
    
    def get_output(self, index):
        return self.output_transforms[index], self.values[index]

    def get_value_history(self, index, level):
        return self.value_history[index][level]

    def add_pyramid_level(self, factor, sigma):
        self.pyramid_factors.append(factor)
        self.pyramid_sigmas.append(sigma)
    
    def add_pyramid_levels(self, factors, sigmas):
        for i in xrange(len(factors)):
            self.add_pyramid_level(factors[i], sigmas[i])

    def get_pyramid_level_count(self):
        return len(self.pyramid_factors)

    def set_sampling_fraction(self, sampling_fraction):
        self.sampling_fraction = sampling_fraction
    
    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_alpha_levels(self, alpha_levels):
        self.alpha_levels = alpha_levels
    
    def set_step_lengths(self, step_lengths):
        self.step_lengths = np.array(step_lengths)#np.array([start_step_length, end_step_length])
    
    def set_reference_image(self, image, spacing = None):
        self.ref_im = image
        if spacing is None:
            self.ref_spacing = np.ones(image.ndim)
        else:
            self.ref_spacing = spacing
        
    def set_floating_image(self, image, spacing = None):
        self.flo_im = image
        if spacing is None:
            self.flo_spacing = np.ones(image.ndim)
        else:
            self.flo_spacing = spacing

    def set_reference_mask(self, mask):
        self.ref_mask = mask

    def set_floating_mask(self, mask):
        self.flo_mask = mask

    def set_reference_weights(self, weights):
        self.ref_weights = weights

    def set_floating_weights(self, weights):
        self.flo_weights = weights

    def set_gradient_magnitude_threshold(self, t):
        self.gradient_magnitude_threshold = t

    def set_report_freq(self, freq):
        self.report_freq = freq
    
    def set_report_func(self, func):
        self.report_func = func
  
    def initialize(self, pyramid_images_output_path=None):
        if len(self.pyramid_factors) == 0:
            self.add_pyramid_level(1, 0.0)
        if len(self.initial_transforms) == 0:
            self.add_initial_transform(AffineTransform(self.dim))
        
        ### Preprocessing

        pyramid_levels = len(self.pyramid_factors)

        for i in xrange(pyramid_levels):
            factor = self.pyramid_factors[i]

            ref_resampled = filters.downsample(filters.gaussian_filter(self.ref_im, self.pyramid_sigmas[i]), factor)
            flo_resampled = filters.downsample(filters.gaussian_filter(self.flo_im, self.pyramid_sigmas[i]), factor)
            
            ref_mask_resampled = filters.downsample(self.ref_mask, factor)
            flo_mask_resampled = filters.downsample(self.flo_mask, factor)

            ref_resampled = filters.normalize(ref_resampled, 0.0, ref_mask_resampled)
            flo_resampled = filters.normalize(flo_resampled, 0.0, flo_mask_resampled)

            if pyramid_images_output_path is not None and ref_resampled.ndim == 2:
                scipy.misc.imsave('%sref_resampled_%d.png' % (pyramid_images_output_path, i+1), ref_resampled)
                scipy.misc.imsave('%sflo_resampled_%d.png' % (pyramid_images_output_path, i+1), flo_resampled)
            
            if self.ref_weights is None:
                ref_weights = np.zeros(ref_resampled.shape)
                ref_weights[ref_mask_resampled] = 1.0
            else:
                ref_weights = filters.downsample(self.ref_weights, factor)
            if self.flo_weights is None:
                flo_weights = np.zeros(flo_resampled.shape)
                flo_weights[flo_mask_resampled] = 1.0
            else:
                flo_weights = filters.downsample(self.flo_weights, factor)

            ref_diag = np.sqrt(np.square(np.array(ref_resampled.shape)*self.ref_spacing).sum())
            flo_diag = np.sqrt(np.square(np.array(flo_resampled.shape)*self.flo_spacing).sum())

            q_ref = QuantizedImage(ref_resampled, self.alpha_levels, ref_weights, self.ref_spacing*factor, remove_zero_weight_pnts = True)
            q_flo = QuantizedImage(flo_resampled, self.alpha_levels, flo_weights, self.flo_spacing*factor, remove_zero_weight_pnts = True)

            tf_ref = alpha_amd.AlphaAMD(q_ref, self.alpha_levels, ref_diag, self.ref_spacing*factor, ref_mask_resampled, ref_mask_resampled, interpolator_mode='linear', dt_fun = None, mask_out_edges = True)
            tf_flo = alpha_amd.AlphaAMD(q_flo, self.alpha_levels, flo_diag, self.flo_spacing*factor, flo_mask_resampled, flo_mask_resampled, interpolator_mode='linear', dt_fun = None, mask_out_edges = True)

            symmetric_measure = True
            squared_measure = False

            sym_dist = symmetric_amd_distance.SymmetricAMDDistance(symmetric_measure=symmetric_measure, squared_measure=squared_measure)

            sym_dist.set_ref_image_source(q_ref)
            sym_dist.set_ref_image_target(tf_ref)

            sym_dist.set_flo_image_source(q_flo)
            sym_dist.set_flo_image_target(tf_flo)

            sym_dist.set_sampling_fraction(self.sampling_fraction)

            sym_dist.initialize()

            self.distances.append(sym_dist)

    def run(self):
        pyramid_level_count = len(self.pyramid_factors)
        transform_count = len(self.initial_transforms)

        for t_it in xrange(transform_count):
            init_transform = self.initial_transforms[t_it]
            param_scaling = self.transforms_param_scaling[t_it]

            self.value_history.append([])

            for lvl_it in xrange(pyramid_level_count):
             
                opt = GradientDescentOptimizer(self.distances[lvl_it], init_transform.copy())

                if self.step_lengths.ndim == 1:
                    opt.set_step_length(self.step_lengths[0], self.step_lengths[1])
                else:
                    opt.set_step_length(self.step_lengths[lvl_it, 0], self.step_lengths[lvl_it, 1])
                opt.set_scalings(param_scaling)
                opt.set_gradient_magnitude_threshold(self.gradient_magnitude_threshold)
                opt.set_report_freq(self.report_freq)
                if type(self.report_func) is list or type(self.report_func) is tuple:
                    opt.set_report_callback(self.report_func[t_it])
                else:
                    opt.set_report_callback(self.report_func)

                if isinstance(self.iterations, int):
                    itercount = self.iterations
                else:
                    assert(len(self.iterations) == pyramid_level_count)
                    itercount = self.iterations[lvl_it]
                
                opt.optimize(itercount)

                if lvl_it + 1 == pyramid_level_count:
                    self.output_transforms.append(opt.get_transform())
                    self.values.append(opt.get_value())
                    self.initial_transforms[t_it] = opt.get_transform()
                else:
                    init_transform = opt.get_transform()

                self.value_history[-1].append(opt.get_value_history())

   
        
        