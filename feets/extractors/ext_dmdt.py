#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  dmdt_extractor.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#


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
        deltam = np.empty(n_vals)
        deltat = np.empty(n_vals)

        k = 0
        for i in range(lc_len):
            t0 = time[i]
            m0 = magnitude[i]
            for j in range(i + 1, lc_len):
                tp = time[j]
                mp = magnitude[j]
                if tp > t0:
                    dt = tp - t0
                    dm = mp - m0
                else:
                    dt = t0 - tp
                    dm = m0 - mp

                deltat[k] = dt
                deltam[k] = dm
                k += 1

        bins = [dt_bins, dm_bins]
        counts = np.histogram2d(deltat, deltam, bins=bins, normed=False)[0]
        counts = np.fix(255. * counts/n_vals + 0.999).astype(int)
        result = zip(self.sorted_features,
                     counts.reshape((len(dt_bins) - 1) * (len(dm_bins) - 1)))

        return dict(result)
