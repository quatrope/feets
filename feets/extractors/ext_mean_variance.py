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

class MeanVariance(Extractor):
    r"""
    **Meanvariance** (:math:`\frac{\sigma}{\bar{m}}`)

    This is a simple variability index and is defined as the ratio of the
    standard deviation :math:`\sigma`, to the mean magnitude, :math:`\bar{m}`.
    If a light curve has strong  variability, :math:`\frac{\sigma}{\bar{m}}`
    of the light curve is generally  large.

    For a uniform distribution from 0 to 1, the mean is equal to 0.5 and the
    variance is equal to 1/12, thus the mean-variance should take a value
    close to 0.577:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Meanvariance'])
        >>> features, values = fs.extract(**lc_uniform)
        >>> dict(zip(features, values))
        {'Meanvariance': 0.5816791217381897}

    References
    ----------

    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    """

    features = ['Meanvariance']

    def __init__(self):
        pass

    def extract(self, magnitude):
        return {"Meanvariance": np.std(magnitude) / np.mean(magnitude)}
