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

"""This tests was madded to always check that the results expected by the
original FATS documentation are the same of feets.

"""


# =============================================================================
# IMPORTS
# =============================================================================

from feets import extractors

import numpy as np

import pytest

# =============================================================================
# CASES
# =============================================================================


def test_FATS_doc_Amplitude():
    ext = extractors.Amplitude()
    value = ext.fit(np.arange(0, 1001))["Amplitude"]
    assert value == 475


@pytest.mark.xfail(reason="FATS say must be 0.2, but actual is -0.60")
def test_FATS_doc_AndersonDarling():
    random = np.random.RandomState(42)

    ext = extractors.AndersonDarling()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = ext.fit(mags)["AndersonDarling"]
    np.testing.assert_allclose(values.mean(), 0.25)


def test_FATS_doc_Beyond1Std():
    random = np.random.RandomState(42)

    ext = extractors.Beyond1Std()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        errors = random.normal(scale=0.001, size=1000)
        values[idx] = ext.fit(mags, errors)["Beyond1Std"]
    np.testing.assert_allclose(values.mean(), 0.32972600000000002)


def test_FATS_doc_Con():
    random = np.random.RandomState(42)

    ext = extractors.Con()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = ext.fit(mags, consecutiveStar=1)["Con"]
    np.testing.assert_allclose(values.mean(), 0.045557)


def test_FATS_doc_MeanVariance():
    random = np.random.RandomState(42)

    ext = extractors.MeanVariance()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.uniform(size=1000)
        values[idx] = ext.fit(magnitude=mags)["Meanvariance"]
    np.testing.assert_allclose(values.mean(), 0.57664232208148747)


def test_FATS_doc_MedianAbsDev():
    random = np.random.RandomState(42)

    ext = extractors.MedianAbsDev()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = ext.fit(magnitude=mags)["MedianAbsDev"]
    np.testing.assert_allclose(values.mean(), 0.67490807679242459)


def test_FATS_doc_RCS():
    random = np.random.RandomState(42)

    ext = extractors.RCS()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.uniform(size=1000)
        values[idx] = ext.fit(magnitude=mags)["Rcs"]
    np.testing.assert_allclose(values.mean(), 0.03902862976795655)


def test_FATS_doc_Skew():
    random = np.random.RandomState(42)

    ext = extractors.Skew()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = ext.fit(magnitude=mags)["Skew"]
    np.testing.assert_allclose(values.mean(), -0.0017170680368871292)


def test_FATS_doc_SmallKurtosis():
    random = np.random.RandomState(42)

    ext = extractors.SmallKurtosis()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = ext.fit(magnitude=mags)["SmallKurtosis"]
    np.testing.assert_allclose(values.mean(), 0.00040502517673364258)


def test_FATS_doc_Std():
    random = np.random.RandomState(42)

    ext = extractors.Std()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = ext.fit(magnitude=mags)["Std"]
    np.testing.assert_allclose(values.mean(), 0.9994202277548033)


@pytest.mark.xfail(reason="FATS say must be 0, but actual is -0.41")
def test_FATS_doc_StetsonJ():
    random = np.random.RandomState(42)

    ext = extractors.StetsonJ()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        mags2 = mags * random.uniform(0, 1.5, mags.size)
        errors = random.normal(scale=0.001, size=1000)
        errors2 = random.normal(scale=0.001, size=1000)
        values[idx] = ext.fit(
            aligned_magnitude=mags,
            aligned_magnitude2=mags2,
            aligned_error=errors,
            aligned_error2=errors2,
        )["StetsonJ"]
    np.testing.assert_allclose(values.mean(), 0)


@pytest.mark.xfail(reason="FATS say must be 2/pi, but actual is -0.20")
def test_FATS_doc_StetsonK():
    random = np.random.RandomState(42)

    ext = extractors.StetsonK()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        errors = random.normal(scale=0.001, size=1000)
        values[idx] = ext.fit(magnitude=mags, error=errors)["StetsonK"]
    np.testing.assert_allclose(values.mean(), 0.798)


def test_FATS_doc_StetsonL():
    random = np.random.RandomState(42)

    ext = extractors.StetsonL()
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        mags2 = mags * random.uniform(0, 1.5, mags.size)
        errors = random.normal(scale=0.001, size=1000)
        errors2 = random.normal(scale=0.001, size=1000)
        values[idx] = ext.fit(
            aligned_magnitude=mags,
            aligned_magnitude2=mags2,
            aligned_error=errors,
            aligned_error2=errors2,
        )["StetsonL"]
    np.testing.assert_allclose(values.mean(), -0.0470713296883)
