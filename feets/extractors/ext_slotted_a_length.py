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

class SlottedA_length(Extractor):
    r"""
    **SlottedA_length** - Slotted Autocorrelation

    In slotted autocorrelation, time lags are defined as intervals or slots
    instead of single values. The slotted autocorrelation function at a
    certain time lag slot is computed by averaging the cross product between
    samples whose time differences fall in the given slot.

    .. math::

        \hat{\rho}(\tau=kh) = \frac {1}{\hat{\rho}(0)\,N_\tau}
            \sum_{t_i}\sum_{t_j= t_i+(k-1/2)h }^{t_i+(k+1/2)h}
            \bar{y}_i(t_i)\,\, \bar{y}_j(t_j)

    Where :math:`h` is the slot size, :math:`\bar{y}` is the normalized
    magnitude, :math:`\hat{\rho}(0)` is the slotted autocorrelation for the
    first lag, and :math:`N_\tau` is the number of pairs that fall in the
    given slot.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(
        ...     only=['SlottedA_length'], SlottedA_length={"t": 1})
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'SlottedA_length': 1.}

    **Parameters**

    - ``T``: tau - slot size in days (default=1).

    """

    data = ["magnitude", "time"]
    features = ["SlottedA_length"]
    params = {"T": 1}

    def slotted_autocorrelation(self, data, time, T, K,
                                second_round=False, K1=100):

        slots, i = np.zeros((K, 1)), 1

        # make time start from 0
        time = time - np.min(time)

        # subtract mean from mag values
        m = np.mean(data)
        data = data - m

        prod = np.zeros((K, 1))
        pairs = np.subtract.outer(time, time)
        pairs[np.tril_indices_from(pairs)] = 10000000

        ks = np.int64(np.floor(np.abs(pairs) / T + 0.5))

        # We calculate the slotted autocorrelation for k=0 separately
        idx = np.where(ks == 0)
        prod[0] = ((sum(data ** 2) + sum(data[idx[0]] *
                   data[idx[1]])) / (len(idx[0]) + len(data)))
        slots[0] = 0

        # We calculate it for the rest of the ks
        if second_round is False:
            for k in np.arange(1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
        else:
            for k in np.arange(K1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i - 1] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
            np.trim_zeros(prod, trim='b')

        slots = np.trim_zeros(slots, trim='b')
        return prod / prod[0], np.int64(slots).flatten()

    def start_conditions(self, magnitude, time, T):
        N = len(time)

        if T is None:
            deltaT = time[1:] - time[:-1]
            sorted_deltaT = np.sort(deltaT)
            T = sorted_deltaT[int(N * 0.05)+1]

        K = 100

        SAC, slots = self.slotted_autocorrelation(magnitude, time, T, K)
        SAC2 = SAC[slots]

        return T, K, slots, SAC2

    def fit(self, magnitude, time, T):
        T, K, slots, SAC2 = self.start_conditions(magnitude, time, T)

        k = next((index for index, value in
                 enumerate(SAC2) if value < np.exp(-1)), None)

        while k is None:
            K = K + K
            if K > (np.max(time) - np.min(time)) / T:
                break
            else:
                SAC, slots = self.slotted_autocorrelation(
                    magnitude, time, T, K, second_round=True, K1=K/2)
                SAC2 = SAC[slots]
                k = next((index for index, value in
                         enumerate(SAC2) if value < np.exp(-1)), None)
        return {"SlottedA_length": slots[k] * T}
