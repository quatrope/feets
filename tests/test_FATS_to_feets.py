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

"""FATS to feets compatibility testing"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

import numpy as np

from feets import FeatureSpace, preprocess
from feets.datasets import macho

from .core import FeetsTestCase, DATA_PATH


# =============================================================================
# CLASSES
# =============================================================================

class FATSPreprocessRegressionTestCase(FeetsTestCase):

    def setUp(self):
        lc = macho.load_MACHO_example()
        self.time = lc.data.R.time
        self.mag = lc.data.R.magnitude
        self.error = lc.data.R.error
        self.time2 = lc.data.B.time
        self.mag2 = lc.data.B.magnitude
        self.error2 = lc.data.B.error

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
                npz['time'],
                npz['mag'],
                npz['error'],
                npz['mag2'],
                npz['aligned_time'],
                npz['aligned_mag'],
                npz['aligned_mag2'],
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
        return "_harmonics_" in feature

    def assert_feature_params(self, feature):
        feature_params = {
            "PeriodLS": {"atol": 1e-04},
            "Period_fit": {"atol": 1e-40},
            "Psi_CS": {"atol": 1e-02},
            "Psi_eta": {"atol": 1e-01}}
        params = {"err_msg": self.err_template.format(feature=feature)}
        params .update(feature_params.get(feature, {}))
        return params

    def assertFATS(self, feets_result):
        for feature in self.features:
            if feature not in feets_result:
                self.fail("Missing feature {}".format(feature))
            if self.exclude_value_feature_evaluation(feature):
                continue
            feets_value = feets_result[feature]
            FATS_value = self.FATS_result[feature]
            params = self.assert_feature_params(feature)
            self.assertAllClose(feets_value, FATS_value, **params)

    def test_FATS_to_feets_extract_one(self):
        fs = FeatureSpace(
            SlottedA_length={"T": None},
            StetsonKAC={"T": None})
        result = fs.extract(*self.lc)
        feets_result = dict(zip(*result))

        feets_result.update({
            "PeriodLS": feets_result.pop("PeriodLS_0"),
            "Period_fit": feets_result.pop("Period_fit_0"),
            "Psi_eta": feets_result.pop("Psi_eta_0"),
            "Psi_CS": feets_result.pop("Psi_CS_0")})

        self.assertFATS(feets_result)
