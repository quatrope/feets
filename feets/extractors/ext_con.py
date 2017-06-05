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

from six.moves import range

import numpy as np

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class Con(Extractor):
    """Index introduced for selection of variable starts from OGLE database.


    To calculate Con, we counted the number of three consecutive measurements
    that are out of 2sigma range, and normalized by N-2
    Pavlos not happy
    """
    data = ['magnitude']
    features = ["Con"]
    params = {"consecutiveStar": 3}

    def fit(self, magnitude, consecutiveStar):

        N = len(magnitude)
        if N < consecutiveStar:
            return 0
        sigma = np.std(magnitude)
        m = np.mean(magnitude)
        count = 0

        for i in range(N - consecutiveStar + 1):
            flag = 0
            for j in range(consecutiveStar):
                if(magnitude[i + j] > m + 2 * sigma or
                   magnitude[i + j] < m - 2 * sigma):
                    flag = 1
                else:
                    flag = 0
                    break
            if flag:
                count = count + 1
        return count * 1.0 / (N - consecutiveStar + 1)
