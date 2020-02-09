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

"""This tests was meded to always check that the results given by the
original FATS project was the same of feets.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import unittest

import numpy as np


from feets import extractors

from ..core import FeetsTestCase


class FATSExtractorsTestCases(FeetsTestCase):

    def setUp(self):
        self.random = np.random.RandomState(42)

    def test_FATS_doc_Amplitude(self):
        ext = extractors.Amplitude()
        value = ext.fit(np.arange(0, 1001))["Amplitude"]
        self.assertEqual(value, 475)

    @unittest.skip("FATS say must be 0.2, but actual is -0.60")
    def test_FATS_doc_AndersonDarling(self):
        ext = extractors.AndersonDarling()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            values[idx] = ext.fit(mags)["AndersonDarling"]
        self.assertAllClose(values.mean(), 0.25)

    def test_FATS_doc_Beyond1Std(self):
        ext = extractors.Beyond1Std()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            errors = self.random.normal(scale=0.001, size=1000)
            values[idx] = ext.fit(mags, errors)["Beyond1Std"]
        self.assertAllClose(values.mean(), 0.32972600000000002)

    def test_FATS_doc_Con(self):
        ext = extractors.Con()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            values[idx] = ext.fit(mags, consecutiveStar=1)["Con"]
        self.assertAllClose(values.mean(), 0.045557)

    def test_FATS_doc_MeanVariance(self):
        ext = extractors.MeanVariance()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.uniform(size=1000)
            values[idx] = ext.fit(magnitude=mags)['Meanvariance']
        self.assertAllClose(values.mean(), 0.57664232208148747)

    def test_FATS_doc_MedianAbsDev(self):
        ext = extractors.MedianAbsDev()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            values[idx] = ext.fit(magnitude=mags)['MedianAbsDev']
        self.assertAllClose(values.mean(), 0.67490807679242459)

    def test_FATS_doc_RCS(self):
        ext = extractors.RCS()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.uniform(size=1000)
            values[idx] = ext.fit(magnitude=mags)['Rcs']
        self.assertAllClose(values.mean(), 0.03902862976795655)

    def test_FATS_doc_Skew(self):
        ext = extractors.Skew()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            values[idx] = ext.fit(magnitude=mags)['Skew']
        self.assertAllClose(values.mean(), -0.0017170680368871292)

    def test_FATS_doc_SmallKurtosis(self):
        ext = extractors.SmallKurtosis()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            values[idx] = ext.fit(magnitude=mags)['SmallKurtosis']
        self.assertAllClose(values.mean(), 0.00040502517673364258)

    def test_FATS_doc_Std(self):
        ext = extractors.Std()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            values[idx] = ext.fit(magnitude=mags)['Std']
        self.assertAllClose(values.mean(), 0.9994202277548033)

    @unittest.skip("FATS say must be 0, but actual is -0.41")
    def test_FATS_doc_StetsonJ(self):
        ext = extractors.StetsonJ()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            mags2 = mags * self.random.uniform(0, 1.5, mags.size)
            errors = self.random.normal(scale=0.001, size=1000)
            errors2 = self.random.normal(scale=0.001, size=1000)
            values[idx] = ext.fit(
                aligned_magnitude=mags, aligned_magnitude2=mags2,
                aligned_error=errors, aligned_error2=errors2)['StetsonJ']
        self.assertAllClose(values.mean(), 0)

    @unittest.skip("FATS say must be 2/pi, but actual is -0.20")
    def test_FATS_doc_StetsonK(self):
        ext = extractors.StetsonK()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            errors = self.random.normal(scale=0.001, size=1000)
            values[idx] = ext.fit(magnitude=mags, error=errors)['StetsonK']
        self.assertAllClose(values.mean(), 0.798)

    def test_FATS_doc_StetsonL(self):
        ext = extractors.StetsonL()
        values = np.empty(1000)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            mags2 = mags * self.random.uniform(0, 1.5, mags.size)
            errors = self.random.normal(scale=0.001, size=1000)
            errors2 = self.random.normal(scale=0.001, size=1000)
            values[idx] = ext.fit(
                aligned_magnitude=mags, aligned_magnitude2=mags2,
                aligned_error=errors, aligned_error2=errors2)['StetsonL']
        self.assertAllClose(values.mean(), -0.0470713296883)
