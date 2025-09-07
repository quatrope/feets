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

"""Lomb-Scargle extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import Periodogram as _Periodogram

import numpy as np

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveLombScargle(LightCurveExtractor):
    r"""Peaks of Lomb-Scargle periodogram.

    Periodogram :math:`P(\omega)` is an estimate of spectral density of
    unevenly time series.

    The `peaks` argument corresponds to a number of the most significant
    spectral density peaks to return.

    For each peak its period and "signal to noise" ratio is returned:

    .. math::

        \mathrm{signal~to~noise~of~peak} =
            \frac{P(\omega_\mathrm{peak}) - \langle P(\omega) \rangle}
            {\sigma_{P(\omega)}}

    Parameters
    ----------
    peaks : int or None, default=3
        Number of peaks to find, default is 3.
    resolution : float or None, default=10
        Resolution of frequency grid, default is 10.
    max_freq_factor : float or None, default=1
        Multiplier for Nyquist frequency, default is 1.
    nyquist : str or float or None, default='average'
        Type of Nyquist frequency. Could be one of:

        - 'average': "Average" Nyquist frequency.
        - 'median': Nyquist frequency is defined by median time interval
          between observations.
        - float: Nyquist frequency is defined by given quantile of time
          intervals between observations.

        Default is 'average'.
    fast : bool or None, default=True
        Use "Fast" (approximate and FFT-based) or direct periodogram
        algorithm, default is True.

    """

    features = ["Periodogram_Peaks", "Periodogram_S_to_N"]

    def __init__(
        self,
        peaks=3,
        resolution=10,
        max_freq_factor=1,
        nyquist="average",
        fast=True,
    ):
        self.peaks = peaks
        self.resolution = resolution
        self.max_freq_factor = max_freq_factor
        self.nyquist = nyquist
        self.fast = fast

        self._extract = _Periodogram(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        """
        Parameters
        ----------
        time : array-like
        magnitude : array-like
        error : array-like, optional
        """
        periodogram = self._extract(time, magnitude, error)
        transpose = np.reshape(periodogram, (-1, 2))
        [period, period_s_to_n] = np.transpose(transpose)

        return {
            "Periodogram_Peaks": period,
            "Periodogram_S_to_N": period_s_to_n,
        }
