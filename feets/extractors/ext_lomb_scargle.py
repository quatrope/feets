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

from astropy.stats import lombscargle

from ..libs import ls_fap

from .core import Extractor


# =============================================================================
# FUNCTIONS
# =============================================================================

def lscargle(time, magnitude, error=None,
             model_kwds=None, autopower_kwds=None):

    model_kwds = model_kwds or {}
    autopower_kwds = autopower_kwds or {}
    model = lombscargle.LombScargle(time, magnitude, error, **model_kwds)
    frequency, power = model.autopower(**autopower_kwds)

    fmax = np.argmax(power)

    return frequency, power, fmax


def fap(power, fmax, time, mag, method, normalization, method_kwds)
    ls_fap.false_alarm_probability(power.max(), fmax, t, y, dy=eps, method="simple", normalization="standard")


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class LombScargle(Extractor):

    data = ['magnitude', 'time']
    features = ["PeriodLS", "Period_fit", "Psi_CS", "Psi_eta"]
    params = {"lscargle_kwds": None}

    def _compute_ls(self, magnitude, time, lscargle_kwds):
        frequency, power, fmax = lscargle(time, magnitude, **lscargle_kwds)

        best_period = 1 / frequency[fmax]

        return frequency, power, fmax, best_period


    def _compute_cs(self, folded_data, N):
        sigma = np.std(folded_data)
        m = np.mean(folded_data)
        s = np.cumsum(folded_data - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return R

    def _compute_eta(self, folded_data, N):
        sigma2 = np.var(folded_data)
        Psi_eta = (1.0 / ((N - 1) * sigma2) *
                   np.sum(np.power(folded_data[1:] - folded_data[:-1], 2)))
        return Psi_eta

    def fit(self, magnitude, time, lscargle_kwds, fap_kwds):
        lscargle_kwds = lscargle_kwds or {}

        frequency, power, fmax, best_period, new_time = self._compute_ls(
            magnitude, time, lscargle_kwds)

        fap = self._compute_fap(power, fmax, time, magnitude, **fap_kwds)

        new_time = np.mod(time, 2 * best_period) / (2 * best_period)
        folded_data = magnitude[np.argsort(new_time)]
        N = len(folded_data)

        R = self._compute_cs(folded_data, N)
        Psi_eta = self._compute_eta(folded_data, N)

        return {"PeriodLS": best_period, "Period_fit": fap,
                "Psi_CS": R, "Psi_eta": Psi_eta}
