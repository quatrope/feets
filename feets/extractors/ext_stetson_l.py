#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOC
# =============================================================================

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class StetsonL(Extractor):

    data = ['aligned_magnitude', 'aligned_magnitude2',
            'aligned_error', 'aligned_error2']
    features = ["StetsonL"]

    def fit(self, aligned_magnitude, aligned_magnitude2,
            aligned_error, aligned_error2):
        magnitude, magnitude2 = aligned_magnitude, aligned_magnitude2
        error, error2 = aligned_error, aligned_error2

        N = len(magnitude)

        mean_mag = (np.sum(magnitude/(error*error)) /
                    np.sum(1.0 / (error * error)))
        mean_mag2 = (np.sum(magnitude2/(error2*error2)) /
                     np.sum(1.0 / (error2 * error2)))

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (magnitude[:N] - mean_mag) /
                  error)

        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                  (magnitude2[:N] - mean_mag2) /
                  error2)
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) *
             np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i))))

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigma_i)) / np.sqrt(np.sum(sigma_i ** 2)))

        return J * K / 0.798
