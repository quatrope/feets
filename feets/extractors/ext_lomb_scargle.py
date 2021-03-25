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
# DOC
# =============================================================================

""""""


# =============================================================================
# IMPORTS
# =============================================================================

from astropy.timeseries import lombscargle

import numpy as np

from .core import Extractor
from ..libs import ls_fap

# =============================================================================
# CONSTANTS
# =============================================================================

EPS = np.finfo(float).eps


# =============================================================================
# FUNCTIONS
# =============================================================================


def argmaxes(arr, n):
    arr = -np.asarray(arr)
    sort = np.argsort(arr)
    return sort[:n]


def lscargle(
    time, magnitude, error=None, model_kwds=None, autopower_kwds=None
):

    model_kwds = model_kwds or {}
    autopower_kwds = autopower_kwds or {}
    model = lombscargle.LombScargle(time, magnitude, error, **model_kwds)
    frequency, power = model.autopower(**autopower_kwds)

    return frequency, power


def fap(
    max_power,
    fmax,
    time,
    magnitude,
    error,
    method,
    normalization,
    method_kwds=None,
):
    method_kwds = method_kwds or {}
    dy = 0.01 if error is None else error
    return ls_fap.false_alarm_probability(
        Z=max_power,
        fmax=fmax,
        t=time,
        y=magnitude,
        dy=dy,
        method=method,
        normalization=normalization,
        method_kwds=method_kwds,
    )


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LombScargle(Extractor):
    r"""
    **PeriodLS**

    The Lomb-Scargle (L-S) algorithm (Scargle, 1982) is a variation of the
    Discrete Fourier Transform (DFT), in which a time series is decomposed
    into a linear combination of sinusoidal functions. The basis of sinusoidal
    functions transforms the data from the time domain to the frequency domain.
    DFT techniques often assume evenly spaced data points in the time series,
    but this is rarely the case with astrophysical time-series data. Scargle
    has derived a formula for transform coefficients that is similiar to the
    DFT in the limit of evenly spaced observations. In addition, an adjustment
    of the values used to calculate the transform coefficients makes the
    transform invariant to time shifts.

    The Lomb-Scargle periodogram is optimized to identify sinusoidal-shaped
    periodic signals in time-series data. Particular applications include
    radial velocity data and searches for pulsating variable stars. L-S is not
    optimal for detecting signals from transiting exoplanets, where the shape
    of the periodic light-curve is not sinusoidal.

    **Period_fit**

    The false alarm probability of the `peaks`-largest periodogram value. Let's
    test it for a normal distributed data and for a periodic one.

    **Psi_CS** (:math:`\Psi_{CS}`)

    :math:`R_{CS}` applied to the phase-folded light curve (generated using
    the periods estimated from the Lomb-Scargle method).

    **Psi_eta** (:math:`\Psi_{\eta}`)

    :math:`\eta^e` index calculated from the folded light curve.


    References
    ----------

    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    """

    data = ["magnitude", "time", "error"]
    optional = ["error"]
    features = ["PeriodLS", "Period_fit", "Psi_CS", "Psi_eta"]
    params = {
        "peaks": 1,
        "lscargle_kwds": {
            "autopower_kwds": {
                "normalization": "standard",
                "nyquist_factor": 100,
            }
        },
        "fap_kwds": {"normalization": "standard", "method": "simple"},
    }

    def _compute_ls(self, magnitude, time, error, peaks, lscargle_kwds):
        frequency, power = lscargle(
            time=time, magnitude=magnitude, error=error, **lscargle_kwds
        )

        fmaxs = argmaxes(power, peaks)
        best_periods = 1 / frequency[fmaxs]

        return frequency, power, fmaxs, best_periods

    def _compute_fap(self, max_power, fmax, time, magnitude, error, fap_kwds):
        return fap(
            max_power=max_power,
            fmax=fmax,
            time=time,
            magnitude=magnitude,
            error=error,
            **fap_kwds,
        )

    def _compute_cs(self, folded_data, N):
        sigma = np.std(folded_data)
        m = np.mean(folded_data)
        s = np.cumsum(folded_data - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return R

    def _compute_eta(self, folded_data, N):
        sigma2 = np.var(folded_data)
        Psi_eta = (
            1.0
            / ((N - 1) * sigma2)
            * np.sum(np.power(folded_data[1:] - folded_data[:-1], 2))
        )
        return Psi_eta

    def fit(self, magnitude, time, error, peaks, lscargle_kwds, fap_kwds):
        # first we retrieve the frequencies, power,
        # max frequencies and best_periods
        frequency, power, fmaxs, best_periods = self._compute_ls(
            magnitude=magnitude,
            time=time,
            error=error,
            peaks=peaks,
            lscargle_kwds=lscargle_kwds,
        )

        faps, Rs, Psi_etas = np.empty(peaks), np.empty(peaks), np.empty(peaks)
        for idx, fmax in enumerate(fmaxs):
            # false alarm probability
            faps[idx] = self._compute_fap(
                max_power=power[fmax],
                fmax=fmax,
                time=time,
                magnitude=magnitude,
                error=error,
                fap_kwds=fap_kwds,
            )

            # fold the data
            best_period = best_periods[idx]

            new_time = np.mod(time, 2 * best_period) / (2 * best_period)
            folded_data = magnitude[np.argsort(new_time)]
            N = len(folded_data)

            # CS and Psi_eta
            Rs[idx] = self._compute_cs(folded_data, N)
            Psi_etas[idx] = self._compute_eta(folded_data, N)

        return {
            "PeriodLS": best_periods,
            "Period_fit": faps,
            "Psi_CS": Rs,
            "Psi_eta": Psi_etas,
        }
