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

from .core import FeetsTestCase, DATA_PATH


# =============================================================================
# CASES
# =============================================================================

def test_F2f_remove_noise(MACHO_example, denoised_MACHO_by_FATS):
    me, dMF = MACHO_example, denoised_MACHO_by_FATS

    p_time, p_mag, p_error = preprocess.remove_noise(me.time, me.mag, me.error)
    p_time2, p_mag2, p_error2 = preprocess.remove_noise(
        me.time2, me.mag2, me.error2)

    np.testing.assert_array_equal(p_time, dMF.time)
    np.testing.assert_array_equal(p_time2, dMF.time2)
    np.testing.assert_array_equal(p_mag, dMF.mag)
    np.testing.assert_array_equal(p_mag2, dMF.mag2)
    np.testing.assert_array_equal(p_error, dMF.error)
    np.testing.assert_array_equal(p_error2, dMF.error2)


def test_F2f_align(denoised_MACHO_by_FATS, aligned_MACHO_by_FATS):
    dMF, aMF = denoised_MACHO_by_FATS, aligned_MACHO_by_FATS
    a_time, a_mag, a_mag2, a_error, a_error2 = preprocess.align(
        dMF.time, dMF.time2,
        dMF.mag, dMF.mag2,
        dMF.error, dMF.error2)
    np.testing.assert_array_equal(a_time, aMF.aligned_time)
    np.testing.assert_array_equal(a_mag, aMF.aligned_mag)
    np.testing.assert_array_equal(a_mag2, aMF.aligned_mag2)
    np.testing.assert_array_equal(a_error, aMF.aligned_error)
    np.testing.assert_array_equal(a_error2, aMF.aligned_error2)


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
        params.update(feature_params.get(feature, {}))
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
