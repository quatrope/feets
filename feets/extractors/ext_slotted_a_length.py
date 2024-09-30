#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# FUNCTIONS
# =============================================================================


def slotted_autocorrelation(data, time, T, K, second_round=False, K1=100):

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
    prod[0] = (sum(data**2) + sum(data[idx[0]] * data[idx[1]])) / (
        len(idx[0]) + len(data)
    )
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
                prod[k] = np.inf
    else:
        for k in np.arange(K1, K):
            idx = np.where(ks == k)
            if len(idx[0]) != 0:
                prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                slots[i - 1] = k
                i = i + 1
            else:
                prod[k] = np.inf
        np.trim_zeros(prod, trim="b")

    slots = np.trim_zeros(slots, trim="b")
    return prod / prod[0], np.int64(slots).flatten()


def start_conditions(magnitude, time, T):
    N = len(time)

    if T is None:
        deltaT = time[1:] - time[:-1]
        sorted_deltaT = np.sort(deltaT)
        T = sorted_deltaT[int(N * 0.05) + 1]

    K = 100

    SAC, slots = slotted_autocorrelation(magnitude, time, T, K)
    SAC2 = SAC[slots]

    return T, K, slots, SAC2


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class SlottedA_length(Extractor):
    r"""Slotted Autocorrelation.

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

    References
    ----------

    .. [huijse2012information] Huijse, P., Estevez, P. A., Protopapas, P.,
       Zegers, P., & Principe, J. C. (2012). An information theoretic algorithm
       for finding periodicities in stellar light curves. IEEE Transactions on
       Signal Processing, 60(10), 5135-5145.

    """

    features = ["SlottedA_length"]

    def __init__(self, T=1):
        self.T = T

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, time):
        T, K, slots, SAC2 = start_conditions(magnitude, time, self.T)

        k = next(
            (index for index, value in enumerate(SAC2) if value < np.exp(-1)),
            None,
        )

        while k is None:
            K = K + K
            if K > (np.max(time) - np.min(time)) / T:
                break
            else:
                SAC, slots = slotted_autocorrelation(
                    magnitude, time, T, K, second_round=True, K1=int(K / 2)
                )
                SAC2 = SAC[slots]
                k = next(
                    (
                        index
                        for index, value in enumerate(SAC2)
                        if value < np.exp(-1)
                    ),
                    None,
                )

        val = np.nan if k is None else slots[k] * T
        return {"SlottedA_length": val}
