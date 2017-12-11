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

class RCS(Extractor):
    """
    **Rcs** - Range of cumulative sum (:math:`R_{cs}`)

    :math:`R_{cs}` is the range of a cumulative sum (Ellaway 1978) of each
    light-curve and is defined as:

    .. math::

        R_{cs} = max(S) - min(S) \\
        S = \frac{1}{N \sigma} \sum_{i=1}^l (m_i - \bar{m})

    where max(min) is the maximum (minimum) value of S and
    :math:`l=1,2, \dots, N`.

    :math:`R_{cs}` should take a value close to zero for any symmetric
    distribution:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Rcs'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Rcs': 0.0094459606901065168}


    """

    data = ['magnitude']
    features = ['Rcs']

    def fit(self, magnitude):
        sigma = np.std(magnitude)
        N = len(magnitude)
        m = np.mean(magnitude)
        s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return {"Rcs": R}
