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
# DOC
# =============================================================================

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Gskew(Extractor):
    r"""Median-of-magnitudes based measure of the skew.

    .. math::

        Gskew = m_{q3} + m_{q97} - 2m

    Where:

    - :math:`m_{q3}` is the median of magnitudes lesser or equal than the
      quantile 3.
    - :math:`m_{q97}` is the median of magnitudes greater or equal than the
      quantile 97.
    - :math:`m` is the median of magnitudes.

    """

    features = ["Gskew"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        median_mag = np.median(magnitude)
        F_3_value = np.percentile(magnitude, 3)
        F_97_value = np.percentile(magnitude, 97)

        skew = (
            np.median(magnitude[magnitude <= F_3_value])
            + np.median(magnitude[magnitude >= F_97_value])
            - 2 * median_mag
        )
        return {"Gskew": skew}
