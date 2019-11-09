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

""""""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class DeltaMDeltaT(Extractor):
    r"""

    **DMDT (Delta Magnitude - Delta Time Mapping)**

    The 2D representations - called dmdt-images hereafter -
    reflect the underlying structure from variability of the source.

    The dmdt-images are translation independent as they consider
    only the differences in time.

    For each pair of points in a light curve we determine
    the difference in magnitude (dm) and the difference in
    time (dt). This gives us $p = n/2 = n ∗ (n − 1)/2$ points
    for a light curve of length $n$. . These points
    are then binned for a range of dm and dt values. The
    resulting binned 2D representation is our 2D mapping from
    the light curve.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['DMDT'])
        >>> rs = fs.extract(**lc_normal)
        >>> rs.as_dict()
        {'DMDT': array([[0, 0, 1, 1, ..., ]])},

    References
    ----------

    .. [Mahabal2017] Mahabal, A., Sheth, K., Gieseke, F., Pai, A.,
       Djorgovski, S.G., Drake, A. J., & Graham, M. J. (2017, November).
       Deep-learn classification of light curves. In 2017 IEEE Symposium
       Series on Computational Intelligence (SSCI) (pp. 1-8). IEEE.

    """
    data = ['magnitude', 'time']
    params = {"dt_bins": np.hstack([0., np.logspace(-3., 3.5, num=23)]),
              "dm_bins": np.hstack([-1. * np.logspace(1, -1, num=12), 0,
                                    np.logspace(-1, 1, num=12)])}
    features = ["DMDT"]

    def fit(self, magnitude, time, dt_bins, dm_bins):

        def delta_calc(idx):
            t0 = time[idx]
            m0 = magnitude[idx]
            deltat = time[idx + 1:] - t0
            deltam = magnitude[idx + 1:] - m0

            deltat[np.where(deltat < 0)] *= -1
            deltam[np.where(deltat < 0)] *= -1

            return np.column_stack((deltat, deltam))

        lc_len = len(time)
        n_vals = int(0.5 * lc_len * (lc_len - 1))

        deltas = np.vstack(
            tuple(delta_calc(idx) for idx in range(lc_len - 1)))

        deltat = deltas[:, 0]
        deltam = deltas[:, 1]

        bins = [dt_bins, dm_bins]
        counts = np.histogram2d(deltat, deltam, bins=bins, normed=False)[0]
        counts = np.fix(255. * counts / n_vals + 0.999).astype(int)

        return {"DMDT": counts}
