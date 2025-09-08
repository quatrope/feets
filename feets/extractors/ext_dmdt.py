#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

"""Delta m and Delta t extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import copy

import numpy as np

from .extractor import Extractor
from ..libs import doctools

__all__ = ["DeltamDeltat"]

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DT_BINS = np.hstack([0.0, np.logspace(-3.0, 3.5, num=23)])

DEFAULT_DM_BINS = np.hstack(
    [-1.0 * np.logspace(1, -1, num=12), 0, np.logspace(-1, 1, num=12)]
)


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class DeltamDeltat(Extractor):
    r"""Delta m and Delta t extractor.

    **DeltamDeltat**

    The 2D histogram of the differences in magnitude (Delta m) and time
    (Delta t) between all pairs of observations in a light curve.

    Parameters
    ----------
    dt_bins : array-like, optional
        The bins for the time differences.
    dm_bins : array-like, optional
        The bins for the magnitude differences.

    References
    ----------
    .. [astro-ph.IM] Mahabal, A. A., Sheth, K., Gieseke, F., Pai, A.,
       Djorgovski, S. G., Drake, A. J., & Graham, M. J. (2017). Deep-learnt
       classification of light curves. 2017 IEEE Symposium Series on
       Computational Intelligence (SSCI), 1-8.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["DeltamDeltat"])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'DeltamDeltat': {'dt_0_dm_0': np.int64(0),
      'dt_1_dm_0': np.int64(0),
       ...
     'dt_22_dm_23': np.int64(0)}}
    """

    features = ["DeltamDeltat"]

    def __init__(self, dt_bins=None, dm_bins=None):
        self.dt_bins = np.asarray(
            copy.deepcopy(DEFAULT_DT_BINS) if dt_bins is None else dt_bins
        )
        self.dm_bins = np.asarray(
            copy.deepcopy(DEFAULT_DM_BINS) if dm_bins is None else dm_bins
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, time):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like
        """

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
        labels = tuple(
            f"dt_{j}_dm_{i}"
            for i in range(len(dm_bins) - 1)
            for j in range(len(dt_bins) - 1)
        )

        counts = np.histogram2d(deltat, deltam, bins=bins)[0]
        counts = np.fix(255.0 * counts / n_vals + 0.999).astype(int)

        result = zip(
            labels,
            counts.reshape((len(dt_bins) - 1) * (len(dm_bins) - 1)),
        )

        return {"DeltamDeltat": dict(result)}
