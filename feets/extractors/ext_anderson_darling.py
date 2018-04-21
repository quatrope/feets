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

from scipy import stats

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class AndersonDarling(Extractor):
    """
    **AndersonDarling**

    The Anderson-Darling test is a statistical test of whether a given
    sample of data is drawn from a given probability distribution. When
    applied to testing if a normal distribution adequately describes a set of
    data, it is one of the most powerful statistical tools for detecting most
    departures from normality.

    For a normal distribution the Anderson-Darling statistic should take values
    close to 0.25.

    References
    ----------

    .. [kim2009trending] Kim, D. W., Protopapas, P., Alcock, C.,
       Byun, Y. I., & Bianco, F. (2009). De-Trending Time Series for
       Astronomical Variability Surveys. Monthly Notices of the Royal
       Astronomical Society, 397(1), 558-568.
       Doi:10.1111/j.1365-2966.2009.14967.x.

    """

    data = ['magnitude']
    features = ["AndersonDarling"]
    warnings = [
        ("The original FATS documentation says that the result of "
         "AndersonDarling must be ~0.25 for gausian distribution but the  "
         "result is ~-0.60")]

    def fit(self, magnitude):
        ander = stats.anderson(magnitude)[0]
        return {"AndersonDarling": 1 / (1.0 + np.exp(-10 * (ander - 0.3)))}
