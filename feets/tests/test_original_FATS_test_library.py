#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2015, 2016, 2017, 2018
# Isadora Nun, Jorge Martínez Palomera, Juan B Cabral

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


"""Original tests of FATS:


Removed:

-   Commented ones
-   The Stetson test because don't work anyway (the original test not provides
    all the data required for the Stetson indexes.


Orignal Version:

https://github.com/isadoranun/FATS/blob/b45b5c1/FATS/test_library.py

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from six.moves import range

from ..core import FeatureSpace


# =============================================================================
# FIXTURES
# =============================================================================

# FIX the random state
random = np.random.RandomState(42)


@pytest.fixture
def white_noise():
    data = random.normal(size=10000)
    mjd = np.arange(10000)
    error = random.normal(loc=0.01, scale=0.8, size=10000)
    second_data = random.normal(size=10000)
    aligned_data = data
    aligned_second_data = second_data
    aligned_mjd = mjd
    lc = {
        "magnitude": data,
        "time": mjd,
        "error": error,
        "magnitude2": second_data,
        "aligned_magnitude": aligned_data,
        "aligned_magnitude2": aligned_second_data,
        "aligned_time": aligned_mjd}
    return lc


@pytest.fixture
def periodic_lc():
    N = 100
    mjd_periodic = np.arange(N)
    Period = 20
    cov = np.zeros([N, N])
    mean = np.zeros(N)
    for i in np.arange(N):
        for j in np.arange(N):
            cov[i, j] = np.exp(-(np.sin((np.pi/Period) * (i-j)) ** 2))
    data_periodic = random.multivariate_normal(mean, cov)
    lc = {
        "magnitude": data_periodic,
        "time": mjd_periodic}
    return lc


@pytest.fixture
def uniform_lc():
    mjd_uniform = np.arange(1000000)
    data_uniform = random.uniform(size=1000000)
    lc = {
        "magnitude": data_uniform,
        "time": mjd_uniform}
    return lc


@pytest.fixture
def random_walk():
    N = 10000
    alpha = 1.
    sigma = 0.5
    data_rw = np.zeros([N, 1])
    data_rw[0] = 1
    time_rw = range(1, N)
    for t in time_rw:
        data_rw[t] = alpha * data_rw[t-1] + random.normal(loc=0.0, scale=sigma)
    time_rw = np.array(range(0, N)) + 1 * random.uniform(size=N)
    data_rw = data_rw.squeeze()
    lc = {
        "magnitude": data_rw,
        "time": time_rw}
    return lc


# =============================================================================
# TESTS
# =============================================================================

def test_Beyond1Std(white_noise):
    fs = FeatureSpace(only=['Beyond1Std'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= 0.30 and result <= 0.40


def test_Mean(white_noise):
    fs = FeatureSpace(only=['Mean'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= -0.1 and result <= 0.1


def test_Con(white_noise):
    fs = FeatureSpace(only=['Con'], Con={"consecutiveStar": 1})
    result = fs.extract(**white_noise)[1][0]
    assert result >= 0.04 and result <= 0.05


def test_Eta_color(white_noise):
    fs = FeatureSpace(only=['Eta_color'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= 1.9 and result <= 2.1


def test_Eta_e(white_noise):
    fs = FeatureSpace(only=['Eta_e'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= 1.9 and result <= 2.1


def test_FluxPercentile(white_noise):
    fs = FeatureSpace(only=[
        'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35',
        'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65',
        'FluxPercentileRatioMid80'])
    result = fs.extract(**white_noise)[1]
    assert result[0] >= 0.145 and result[0] <= 0.160
    assert result[1] >= 0.260 and result[1] <= 0.290
    assert result[2] >= 0.350 and result[2] <= 0.450
    assert result[3] >= 0.540 and result[3] <= 0.580
    assert result[4] >= 0.760 and result[4] <= 0.800


def test_LinearTrend(white_noise):
    fs = FeatureSpace(only=['LinearTrend'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= -0.1 and result <= 0.1


def test_Meanvariance(uniform_lc):
    fs = FeatureSpace(only=['Meanvariance'])
    result = fs.extract(**uniform_lc)[1][0]
    assert result >= 0.575 and result <= 0.580


def test_MedianAbsDev(white_noise):
    fs = FeatureSpace(only=['MedianAbsDev'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= 0.630 and result <= 0.700


def test_PairSlopeTrend(white_noise):
    fs = FeatureSpace(only=['PairSlopeTrend'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= -0.25 and result <= 0.25


def test_Period_Psi(periodic_lc):
    params = {
        "lscargle_kwds": {
            "autopower_kwds": {
                "normalization": "standard",
                "nyquist_factor": 1,
            }
        }
    }

    fs = FeatureSpace(only=['PeriodLS'], LombScargle=params)
    result = fs.extract(**periodic_lc)[1][0]
    assert result >= 19 and result <= 21


def test_Q31(white_noise):
    fs = FeatureSpace(only=['Q31'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= 1.30 and result <= 1.38


def test_Rcs(white_noise):
    fs = FeatureSpace(only=['Rcs'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= 0 and result <= 0.1


def test_Skew(white_noise):
    fs = FeatureSpace(only=['Skew'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= -0.1 and result <= 0.1


def test_SmallKurtosis(white_noise):
    fs = FeatureSpace(only=['SmallKurtosis'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= -0.2 and result <= 0.2


def test_Std(white_noise):
    fs = FeatureSpace(only=['Std'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= 0.9 and result <= 1.1


def test_Gskew(white_noise):
    fs = FeatureSpace(only=['Gskew'])
    result = fs.extract(**white_noise)[1][0]
    assert result >= -0.2 and result <= 0.2


def test_StructureFunction(random_walk):
    fs = FeatureSpace(only=[
        'StructureFunction_index_21',
        'StructureFunction_index_31',
        'StructureFunction_index_32'])
    result = fs.extract(**random_walk)[1]
    assert(result[0] >= 1.520 and result[0] <= 2.067)
    assert(result[1] >= 1.821 and result[1] <= 3.162)
    assert(result[2] >= 1.243 and result[2] <= 1.562)
