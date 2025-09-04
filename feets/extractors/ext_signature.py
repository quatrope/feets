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

"""Signature extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Signature(Extractor):
    r"""Signature extractor.

    **Signature**

    The signature is a 2D histogram of the light-curve in the phase-magnitude
    space. The phase is calculated as the time modulo the period, and the
    magnitude is normalized by the amplitude.

    Parameters
    ----------
    phase_bins : int, optional, default: `18`
        Number of phase bins.
    mag_bins : int, optional, default: `12`
        Number of magnitude bins.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["Signature"])
    >>> features = fs.extract(**lc_periodic)
    >>> features[0]
    {'Signature': array([{'ph_0_mag_0': np.float64(3.273060645417755), ...,
             'ph_17_mag_11': np.float64(0.0)},
            {'ph_0_mag_0': np.float64(6.546121290835849), ...,
             'ph_17_mag_11': np.float64(0.0)},
            {'ph_0_mag_0': np.float64(3.273060645418243), ...,
             'ph_17_mag_11': np.float64(0.0)}],
           dtype=object)}
    """

    features = ["Signature"]

    def __init__(self, phase_bins=18, mag_bins=12):
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, time, PeriodLS, MedianAmplitude):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like
        PeriodLS : array-like
        MedianAmplitude : float
        """
        phase_bins, mag_bins = self.phase_bins, self.mag_bins
        bins = (phase_bins, mag_bins)
        labels = tuple(
            f"ph_{j}_mag_{i}"
            for i in range(mag_bins)
            for j in range(phase_bins)
        )

        lc_yaxis = (magnitude - np.min(magnitude)) / np.float64(
            MedianAmplitude
        )

        # Shift time to the minimum value
        loc = np.argmin(lc_yaxis)

        signatures = np.full(len(PeriodLS), None, dtype=object)
        for idx, period_ls in enumerate(PeriodLS):
            lc_phases = np.remainder(time - time[loc], period_ls) / period_ls

            count = np.histogram2d(
                lc_phases, lc_yaxis, bins=bins, density=True
            )[0]

            signature = zip(labels, count.reshape(phase_bins * mag_bins))
            signatures[idx] = dict(signature)

        return {"Signature": signatures}
