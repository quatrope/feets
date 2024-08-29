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

    features = ["DeltamDeltat"]

    def __init__(
        self,
        dt_bins=np.hstack([0.0, np.logspace(-3.0, 3.5, num=23)]),
        dm_bins=np.hstack(
            [-1.0 * np.logspace(1, -1, num=12), 0, np.logspace(-1, 1, num=12)]
        ),
    ):
        feature_attrs = []
        for i in range(len(dm_bins) - 1):
            for j in range(len(dt_bins) - 1):
                feature_attrs.append(f"dt_{j}_dm_{i}")

        self.dt_bins = dt_bins
        self.dm_bins = dm_bins
        self.feature_attrs = tuple(feature_attrs)

    def extract(self, magnitude, time):
        def delta_calc(idx):
            t0 = time[idx]
            m0 = magnitude[idx]
            deltat = time[idx + 1 :] - t0
            deltam = magnitude[idx + 1 :] - m0

            deltat[np.where(deltat < 0)] *= -1
            deltam[np.where(deltat < 0)] *= -1

            return np.column_stack((deltat, deltam))

        lc_len = len(time)
        n_vals = int(0.5 * lc_len * (lc_len - 1))

        deltas = np.vstack([delta_calc(idx) for idx in range(lc_len - 1)])

        deltat = deltas[:, 0]
        deltam = deltas[:, 1]

        dt_bins, dm_bins = self.dt_bins, self.dm_bins
        bins = [dt_bins, dm_bins]
        counts = np.histogram2d(deltat, deltam, bins=bins)[0]
        counts = np.fix(255.0 * counts / n_vals + 0.999).astype(int)

        result = zip(
            self.feature_attrs,
            counts.reshape((len(dt_bins) - 1) * (len(dm_bins) - 1)),
        )

        return dict(result)
