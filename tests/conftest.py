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

"""Pytest configuration"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

import feets
from feets.datasets import macho

import numpy as np

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

# FIX the random state
random = np.random.RandomState(42)

DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


# =============================================================================
# UTILS
# =============================================================================


class Bunch(dict):
    def __getattr__(self, k):
        return self[k]


# =============================================================================
# FIXTURES
# =============================================================================


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
        "aligned_time": aligned_mjd,
    }
    return Bunch(lc)


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
    lc = {"magnitude": data_periodic, "time": mjd_periodic}
    return Bunch(lc)


@pytest.fixture
def periodic_lc_werror():

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
    lc = {"magnitude": data_periodic, "time": mjd_periodic, "error": error}
    return Bunch(lc)


@pytest.fixture
def uniform_lc():
    mjd_uniform = np.arange(1000000)
    data_uniform = random.uniform(size=1000000)
    lc = {"magnitude": data_uniform, "time": mjd_uniform}
    return Bunch(lc)


@pytest.fixture
def random_walk():
    N = 10000
    alpha = 1.0
    sigma = 0.5
    data_rw = np.zeros([N, 1])
    data_rw[0] = 1
    time_rw = range(1, N)
    for t in time_rw:
        data_rw[t] = alpha * data_rw[t - 1] + random.normal(
            loc=0.0, scale=sigma
        )
    time_rw = np.array(range(0, N)) + 1 * random.uniform(size=N)
    data_rw = data_rw.squeeze()
    lc = {"magnitude": data_rw, "time": time_rw}
    return Bunch(lc)


@pytest.fixture(scope="session")
def denoised_MACHO_by_FATS():
    preprc_path = os.path.join(DATA_PATH, "FATS_preprc.npz")
    with np.load(preprc_path) as npz:
        return Bunch(npz)


@pytest.fixture(scope="session")
def aligned_MACHO_by_FATS():
    lc_path = os.path.join(DATA_PATH, "FATS_aligned.npz")
    with np.load(lc_path) as npz:
        return Bunch(npz)


@pytest.fixture(scope="session")
def MACHO_example():
    lc = macho.load_MACHO_example()
    return Bunch(
        time=lc.data.R.time,
        mag=lc.data.R.magnitude,
        error=lc.data.R.error,
        time2=lc.data.B.time,
        mag2=lc.data.B.magnitude,
        error2=lc.data.B.error,
    )


@pytest.fixture(scope="session")
def FATS_results():
    # recreate the FATS result
    FATS_result_path = os.path.join(DATA_PATH, "FATS_result.npz")
    with np.load(FATS_result_path) as npz:
        features = npz["features"].astype("U")
        values = dict(zip(features, npz["values"]))
    return Bunch(features=features, fvalues=values)


@pytest.fixture
def mock_extractors_register(monkeypatch):
    monkeypatch.setattr(feets.extractors, "_extractors", {})


@pytest.fixture
def foo_extractor(mock_extractors_register):
    @feets.register_extractor
    class MockExtractor(feets.Extractor):
        data = ["magnitude"]
        features = ["foo"]

        def plot_feature(self, **kwargs):
            pass

        def fit(self, magnitude):
            return {"foo": magnitude}

    return MockExtractor()
