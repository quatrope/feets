#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2018 Bruno Sanchez

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

class DeltamDeltat(Extractor):
    """
    Deltas features described in
    Configure the bins as desired.

    It is a map of n observations to n chosen 2 dupla of observations.
    **Eta_color** (:math:`\eta_{color}`)

    Variability index Eta_e (:math:`\eta^e`)
    calculated from the color light-curve.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Eta_color'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Eta_color': 1.991749074648397}

    References
    ----------
    Mahabal et. al 2017 (arxiv:1709.06257)

    """
    data = ['magnitude', 'time']
    params = {"dt_bins": np.hstack([0., np.logspace(-3., 3.5, num=23)]),
              "dm_bins": np.hstack([-1.*np.logspace(1, -1, num=12), 0,
                                    np.logspace(-1, 1, num=12)])}

    features = []
    for i in range(len(params["dm_bins"]) - 1):
        for j in range(len(params["dt_bins"]) - 1):
            features.append("DeltamDeltat_dt_{}_dm_{}".format(j, i))

    # this variable stores a sorted version of the features
    # because feets only stores a frozenset of the original features
    # for future validation.
    sorted_features = tuple(features)

    del i, j

    def fit(self, magnitude, time, dt_bins, dm_bins):

        lc_len = len(time)
        n_vals = int(0.5 * lc_len * (lc_len - 1))

        deltam = []
        deltat = []
        for i in range(lc_len-1):
            t0 = time[i]
            m0 = magnitude[i]

            dtimes = time[i + 1:] - t0
            dmags = magnitude[i + 1:] - m0

            deltat.append(dtimes)
            deltam.append(dmags)

        deltat = np.hstack(deltat)
        deltam = np.hstack(deltam)

        deltat[np.where(deltat < 0)] *= -1
        deltam[np.where(deltat < 0)] *= -1

        bins = [dt_bins, dm_bins]
        counts = np.histogram2d(deltat, deltam, bins=bins, normed=False)[0]
        counts = np.fix(255. * counts/n_vals + 0.999).astype(int)

        result = zip(self.sorted_features,
                     counts.reshape((len(dt_bins) - 1) * (len(dm_bins) - 1)))

        return dict(result)
