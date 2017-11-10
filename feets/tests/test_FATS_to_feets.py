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

__doc__ = """FATS to feets compatibility testing"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

import numpy as np

from .. import FeatureSpace, datasets, preprocess

from .core import FeetsTestCase


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


# =============================================================================
# CLASSES
# =============================================================================

class FATSPreprocessRegressionTestCase(FeetsTestCase):

    def setUp(self):
        d = datasets.load_MACHO_example()
        self.time, self.mag, self.error = d.lc[0]
        self.time2, self.mag2, self.error2 = d.lc[1]

        self.preprc_path = os.path.join(DATA_PATH, "FATS_preprc.npz")
        with np.load(self.preprc_path) as npz:
            self.pF_time, self.pF_time2 = npz["time"], npz["time2"]
            self.pF_mag, self.pF_mag2 = npz["mag"], npz["mag2"]
            self.pF_error, self.pF_error2 = npz["error"], npz["error2"]

        self.lc_path = os.path.join(DATA_PATH, "FATS_aligned.npz")
        with np.load(self.lc_path) as npz:
            self.aF_time = npz['aligned_time']
            self.aF_mag = npz['aligned_mag']
            self.aF_mag2 = npz['aligned_mag2']
            self.aF_error = npz['aligned_error']
            self.aF_error2 = npz['aligned_error2']

    def test_remove_noise(self):
        p_time, p_mag, p_error = preprocess.remove_noise(
            self.time, self.mag, self.error)
        p_time2, p_mag2, p_error2 = preprocess.remove_noise(
            self.time2, self.mag2, self.error2)
        self.assertArrayEqual(p_time, self.pF_time)
        self.assertArrayEqual(p_time2, self.pF_time2)
        self.assertArrayEqual(p_mag, self.pF_mag)
        self.assertArrayEqual(p_mag2, self.pF_mag2)
        self.assertArrayEqual(p_error, self.pF_error)
        self.assertArrayEqual(p_error2, self.pF_error2)

    def test_align(self):
        a_time, a_mag, a_mag2, a_error, a_error2 = preprocess.align(
            self.pF_time, self.pF_time2,
            self.pF_mag, self.pF_mag2,
            self.pF_error, self.pF_error2)
        self.assertArrayEqual(a_time, self.aF_time)
        self.assertArrayEqual(a_mag, self.aF_mag)
        self.assertArrayEqual(a_mag2, self.aF_mag2)
        self.assertArrayEqual(a_error, self.aF_error)
        self.assertArrayEqual(a_error2, self.aF_error2)


class FATSRegressionTestCase(FeetsTestCase):

    def setUp(self):
        # the paths
        self.lc_path = os.path.join(DATA_PATH, "FATS_aligned.npz")
        self.FATS_result_path = os.path.join(DATA_PATH, "FATS_result.npz")

        # recreate light curve
        with np.load(self.lc_path) as npz:
            self.lc = (
                npz['mag'],
                npz['time'],
                npz['error'],
                npz['mag2'],
                npz['aligned_mag'],
                npz['aligned_mag2'],
                npz['aligned_time'],
                npz['aligned_error'],
                npz['aligned_error2'])

        # recreate the FATS result
        with np.load(self.FATS_result_path) as npz:
            self.features = npz["features"]
            self.features = self.features.astype("U")
            self.FATS_result = dict(zip(self.features, npz["values"]))

        # creates an template for all error, messages
        self.err_template = ("Feature '{feature}' missmatch.")

    def exclude_value_feature_evaluation(self, feature):
        return (
            "_harmonics_" in feature or
            feature in ["PeriodLS", "Period_fit", "Psi_CS", "Psi_eta"])

    def assertFATS(self, feets_result):
        for feature in self.features:
            if feature not in feets_result:
                self.fail("Missing feature {}".format(feature))
            if self.exclude_value_feature_evaluation(feature):
                continue
            feets_value = feets_result[feature]
            FATS_value = self.FATS_result[feature]
            err_msg = self.err_template.format(feature=feature)
            self.assertAllClose(feets_value, FATS_value, err_msg=err_msg)

    def test_FATS_to_feets_extract_one(self):
        fs = FeatureSpace()
        result = fs.extract_one(self.lc)
        feets_result = dict(zip(*result))
        self.assertFATS(feets_result)

    def test_FATS_to_feets_extract(self):
        fs = FeatureSpace()
        rfeatures, rdatas = fs.extract([self.lc] * 3)
        for result in rdatas:
            feets_result = dict(zip(rfeatures, result))
            self.assertFATS(feets_result)
        self.assertEqual(len(rdatas), 3)
