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

# =============================================================================
# DOCS
# =============================================================================

"""Original tests of FATS:


Removed:

-   Commented ones
-   The Stetson test because don't work anyway (the original test not provides
    all the data required for the Stetson indexes.
-   Added the test from the tutorial


Orignal Version:

https://github.com/isadoranun/FATS/blob/b45b5c1/FATS/test_library.py

"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings

import numpy as np

import pandas as pd

import pytest

from ..core import FeatureSpace
from .core import FeetsTestCase, DATA_PATH


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
            cov[i, j] = np.exp(-(np.sin((np.pi / Period) * (i - j)) ** 2))
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
        data_rw[t] = (
            alpha * data_rw[t - 1] + random.normal(loc=0.0, scale=sigma))
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
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= 0.30 and result <= 0.40


def test_Mean(white_noise):
    fs = FeatureSpace(only=['Mean'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= -0.1 and result <= 0.1


def test_Con(white_noise):
    fs = FeatureSpace(only=['Con'], Con={"consecutiveStar": 1})
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= 0.04 and result <= 0.05


def test_Eta_color(white_noise):
    fs = FeatureSpace(only=['Eta_color'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= 1.9 and result <= 2.1


def test_Eta_e(white_noise):
    fs = FeatureSpace(only=['Eta_e'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= 1.9 and result <= 2.1


def test_FluxPercentile(white_noise):
    fs = FeatureSpace(only=[
        'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35',
        'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65',
        'FluxPercentileRatioMid80'])
    result = fs.extract(**white_noise).as_arrays()[1]
    assert result[0] >= 0.145 and result[0] <= 0.160
    assert result[1] >= 0.260 and result[1] <= 0.290
    assert result[2] >= 0.350 and result[2] <= 0.450
    assert result[3] >= 0.540 and result[3] <= 0.580
    assert result[4] >= 0.760 and result[4] <= 0.800


def test_LinearTrend(white_noise):
    fs = FeatureSpace(only=['LinearTrend'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= -0.1 and result <= 0.1


def test_Meanvariance(uniform_lc):
    fs = FeatureSpace(only=['Meanvariance'])
    result = fs.extract(**uniform_lc).as_arrays()[1][0]
    assert result >= 0.575 and result <= 0.580


def test_MedianAbsDev(white_noise):
    fs = FeatureSpace(only=['MedianAbsDev'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= 0.630 and result <= 0.700


def test_PairSlopeTrend(white_noise):
    fs = FeatureSpace(only=['PairSlopeTrend'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= -0.25 and result <= 0.25


def test_Period(periodic_lc):
    params = {
        "lscargle_kwds": {
            "autopower_kwds": {
                "normalization": "standard",
                "nyquist_factor": 1,
            }
        }
    }

    fs = FeatureSpace(only=['PeriodLS'], LombScargle=params)
    result = fs.extract(**periodic_lc).as_arrays()[1][0]
    assert result >= 19 and result <= 21


def test_Q31(white_noise):
    fs = FeatureSpace(only=['Q31'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= 1.30 and result <= 1.38


def test_Rcs(white_noise):
    fs = FeatureSpace(only=['Rcs'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= 0 and result <= 0.1


def test_Skew(white_noise):
    fs = FeatureSpace(only=['Skew'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= -0.1 and result <= 0.1


def test_SmallKurtosis(white_noise):
    fs = FeatureSpace(only=['SmallKurtosis'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= -0.2 and result <= 0.2


def test_Std(white_noise):
    fs = FeatureSpace(only=['Std'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= 0.9 and result <= 1.1


def test_Gskew(white_noise):
    fs = FeatureSpace(only=['Gskew'])
    result = fs.extract(**white_noise).as_arrays()[1][0]
    assert result >= -0.2 and result <= 0.2


def test_StructureFunction(random_walk):
    fs = FeatureSpace(only=[
        'StructureFunction_index_21',
        'StructureFunction_index_31',
        'StructureFunction_index_32'])
    result = fs.extract(**random_walk).as_arrays()[1]
    assert(result[0] >= 1.520 and result[0] <= 2.067)
    assert(result[1] >= 1.821 and result[1] <= 3.162)
    assert(result[2] >= 1.243 and result[2] <= 1.562)


# =============================================================================
# TUTORIAL TEST CASE
# =============================================================================

class FATSTutorialTestCase(FeetsTestCase):

    def shuffle(self, mag, error, time, mag2, aligned_mag, aligned_mag2,
                aligned_time, aligned_error, aligned_error2):

        N = len(mag)
        shuffle = np.arange(0, N)
        index = self.random.permutation(shuffle)
        index = np.sort(index[0:int(N / 2)])

        mag_test = mag[index]
        time_test = time[index]
        error_test = error[index]

        N2 = len(mag2)
        shuffle2 = np.arange(0, N2)
        index2 = self.random.permutation(shuffle2)
        index2 = np.sort(index2[0:int(N2 / 2)])

        mag2_test = mag2[index2]

        N3 = len(aligned_mag)
        shuffle3 = np.arange(0, N3)
        index3 = self.random.permutation(shuffle3)
        index3 = np.sort(index3[0:int(N3 / 2)])

        aligned_mag_test = aligned_mag[index3]
        aligned_mag2_test = aligned_mag2[index3]
        aligned_time_test = aligned_time[index3]
        aligned_error_test = aligned_error[index3]
        aligned_error2_test = aligned_error2[index3]

        return {
            "magnitude": mag_test,
            "time": time_test,
            "error": error_test,
            "magnitude2": mag2_test,
            "aligned_magnitude": aligned_mag_test,
            "aligned_magnitude2": aligned_mag2_test,
            "aligned_time": aligned_time_test,
            "aligned_error": aligned_error_test,
            "aligned_error2": aligned_error2_test}

    def setUp(self):
        self.random = np.random.RandomState(42)
        self.lc_path = os.path.join(DATA_PATH, "FATS_aligned.npz")
        with np.load(self.lc_path) as npz:
            self.lc = dict(npz)

    def test_invariance_to_unequal_sampling(self):
        # tests performed to the features in order to check their invariance
        # to unequal sampling. To do so, we take random observations of a
        # light-curve and compare the resulting features with the ones obtained
        # from the original data.

        fs = FeatureSpace()

        # We calculate the features values for fifty random samples of the
        # original light-curve:
        features_values = []
        for i in range(50):
            sample = self.shuffle(**self.lc)
            features, values = fs.extract(**sample)
            result = dict(zip(features, values))
            features_values.append(result)

        # We obtain the mean and standard deviation of each calculated feature:
        stats = pd.DataFrame(features_values).aggregate([np.mean, np.std])

        # Original light-curve:
        features, values = fs.extract(
            magnitude=self.lc["mag"],
            time=self.lc["time"],
            error=self.lc["error"],
            magnitude2=self.lc["mag2"],
            aligned_magnitude=self.lc["aligned_mag"],
            aligned_magnitude2=self.lc["aligned_mag2"],
            aligned_time=self.lc["aligned_time"],
            aligned_error=self.lc["aligned_error"],
            aligned_error2=self.lc["aligned_error2"])

        def normalize(c):
            name, value = c.name, c[0]
            mean, std = stats[name]["mean"], stats[name]["std"]
            normalized = (value - mean) / std
            return normalized

        original = pd.DataFrame([dict(zip(features, values))])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message="invalid value encountered in double_scalars")
            result = original.apply(normalize)

        self.assertLess(np.abs(result.mean()), 0.12)
        self.assertLess(result.std(), 1.09)
