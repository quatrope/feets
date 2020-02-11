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

"""feets.extractors.ext_lomb_scargle Tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

import numpy as np

import pandas as pd

from feets import extractors, FeatureSpace

import pytest

from ..core import DATA_PATH


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def periodic_lc():
    random = np.random.RandomState(42)

    N = 100
    mjd_periodic = np.arange(N)
    Period = 20
    cov = np.zeros([N, N])
    mean = np.zeros(N)
    for i in np.arange(N):
        for j in np.arange(N):
            cov[i, j] = np.exp(-(np.sin((np.pi / Period) * (i - j)) ** 2))
    data_periodic = random.multivariate_normal(mean, cov)
    error = random.normal(size=100, loc=0.001)
    lc = {
        "magnitude": data_periodic,
        "time": mjd_periodic,
        "error": error}
    return lc


# =============================================================================
# Test cases
# =============================================================================

def test_lscargle_vs_feets():

    # extract the module for make short code
    ext_lomb_scargle = extractors.ext_lomb_scargle

    # load the data
    path = os.path.join(
        DATA_PATH, "bad_results.pandas.pkl")
    tseries = pd.read_pickle(path)

    # the ls params
    ext_params = ext_lomb_scargle.LombScargle.get_default_params()
    lscargle_kwds = ext_params["lscargle_kwds"]

    # create the feature space
    fs = FeatureSpace(only=["PeriodLS"])

    ls_periods, feets_periods = [], []
    for src_id in tseries.bm_src_id.unique():

        # extract the timeseries
        sobs = tseries[tseries.bm_src_id == src_id]
        time = sobs.pwp_stack_src_hjd.values
        magnitude = sobs.pwp_stack_src_mag3.values
        error = sobs.pwp_stack_src_mag_err3.values

        # "pure" lomb scargle (without the entire feets pipeline)
        frequency, power = ext_lomb_scargle.lscargle(
            time=time, magnitude=magnitude, error=error, **lscargle_kwds)
        fmax = np.argmax(power)
        ls_periods.append(1 / frequency[fmax])

        # extract the period from the feets pipele
        rs = fs.extract(time=time, magnitude=magnitude, error=error)
        feets_periods.append(rs.values['PeriodLS'])

    feets_periods = np.array(feets_periods).flatten()

    np.testing.assert_array_equal(ls_periods, feets_periods)


def test_lscargle_peaks(periodic_lc):

    for peaks in [1, 2, 3, 10]:
        ext = extractors.LombScargle(peaks=peaks)
        feats = ext.extract(features={}, **periodic_lc)
        for v in feats.values():
            assert len(v) == peaks
