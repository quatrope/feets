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

# import os

import feets

import numpy as np

import pytest

import tests.conftest as conftest

# from .c import FeetsTestCase, DATA_PATH


# =============================================================================
# CLASSES
# =============================================================================


@pytest.fixture
def MACHO_LC():
    lc = feets.datasets.load_MACHO_example()
    lc = {
        "time": lc.data.R.time,
        "magnitude": lc.data.R.magnitude,
        "error": lc.data.R.error,
        "time2": lc.data.B.time,
        "magnitude2": lc.data.B.magnitude,
        "error2": lc.data.B.error,
    }
    return lc


@pytest.fixture
def FATS_MACHO_LC_remove_noise_result():
    path = conftest.TEST_DATASET_PATH / "FATS_preprc.npz"
    lc = {}
    with np.load(path) as npz:
        lc = {
            "time": npz["time"],
            "time2": npz["time2"],
            "magnitude": npz["mag"],
            "magnitude2": npz["mag2"],
            "error": npz["error"],
            "error2": npz["error2"],
        }
    return lc


@pytest.fixture
def FATS_MACHO_LC_aligned():
    path = conftest.TEST_DATASET_PATH / "FATS_aligned.npz"
    lc = {}
    with np.load(path) as npz:
        lc = {
            "time": npz["aligned_time"],
            "magnitude": npz["aligned_mag"],
            "magnitude2": npz["aligned_mag2"],
            "error": npz["aligned_error"],
            "error2": npz["aligned_error2"],
        }
    return lc


@pytest.mark.skip
def test_FATS2feets_remove_noise(MACHO_LC, FATS_MACHO_LC_remove_noise_result):
    p_time, p_mag, p_error = feets.preprocess.remove_noise(
        MACHO_LC["time"], MACHO_LC["magnitude"], MACHO_LC["error"]
    )
    p_time2, p_mag2, p_error2 = feets.preprocess.remove_noise(
        MACHO_LC["time2"], MACHO_LC["magnitude2"], MACHO_LC["error2"]
    )
    np.testing.assert_array_equal(
        p_time, FATS_MACHO_LC_remove_noise_result["time"]
    )
    np.testing.assert_array_equal(
        p_time2, FATS_MACHO_LC_remove_noise_result["time2"]
    )
    np.testing.assert_array_equal(
        p_mag, FATS_MACHO_LC_remove_noise_result["magnitude"]
    )
    np.testing.assert_array_equal(
        p_mag2, FATS_MACHO_LC_remove_noise_result["magnitude2"]
    )
    np.testing.assert_array_equal(
        p_error, FATS_MACHO_LC_remove_noise_result["error"]
    )
    np.testing.assert_array_equal(
        p_error2, FATS_MACHO_LC_remove_noise_result["error2"]
    )

@pytest.mark.skip
def test_FATS2feets_align(MACHO_LC, FATS_MACHO_LC_remove_noise_aligned):
    a_time, a_mag, a_mag2, a_error, a_error2 = feets.preprocess.align(
        **MACHO_LC
    )
    np.testing.assert_array_equal(
        a_time, FATS_MACHO_LC_remove_noise_aligned["time"]
    )
    np.testing.assert_array_equal(
        a_mag, FATS_MACHO_LC_remove_noise_aligned["magnitude"]
    )
    np.testing.assert_array_equal(
        a_mag2, FATS_MACHO_LC_remove_noise_aligned["magnitude2"]
    )
    np.testing.assert_array_equal(
        a_error, FATS_MACHO_LC_remove_noise_aligned["error"]
    )
    np.testing.assert_array_equal(
        a_error2, FATS_MACHO_LC_remove_noise_aligned["error2"]
    )


# NO implementemos ESTO AUN!
# class FATSRegressionTestCase(FeetsTestCase):

#     def setUp(self):
#         # the paths
#         self.lc_path = os.path.join(DATA_PATH, "FATS_aligned.npz")
#         self.FATS_result_path = os.path.join(DATA_PATH, "FATS_result.npz")

#         # recreate light curve
#         with np.load(self.lc_path) as npz:
#             self.lc = (
#                 npz["time"],
#                 npz["mag"],
#                 npz["error"],
#                 npz["mag2"],
#                 npz["aligned_time"],
#                 npz["aligned_mag"],
#                 npz["aligned_mag2"],
#                 npz["aligned_error"],
#                 npz["aligned_error2"],
#             )

#         # recreate the FATS result
#         with np.load(self.FATS_result_path) as npz:
#             self.features = npz["features"]
#             self.features = self.features.astype("U")
#             self.FATS_result = dict(zip(self.features, npz["values"]))

#         # creates an template for all error, messages
#         self.err_template = "Feature '{feature}' missmatch."

#     def exclude_value_feature_evaluation(self, feature):
#         return "_harmonics_" in feature

#     def assert_feature_params(self, feature):
#         feature_params = {
#             "PeriodLS": {"atol": 1e-04},
#             "Period_fit": {"atol": 1e-40},
#             "Psi_CS": {"atol": 1e-02},
#             "Psi_eta": {"atol": 1e-01},
#         }
#         params = {"err_msg": self.err_template.format(feature=feature)}
#         params.update(feature_params.get(feature, {}))
#         return params

#     def assertFATS(self, feets_result):
#         for feature in self.features:
#             if feature not in feets_result:
#                 self.fail("Missing feature {}".format(feature))
#             if self.exclude_value_feature_evaluation(feature):
#                 continue
#             feets_value = feets_result[feature]
#             FATS_value = self.FATS_result[feature]
#             params = self.assert_feature_params(feature)
#             self.assertAllClose(feets_value, FATS_value, **params)


# @pytest.mark.xfail
# def test_FATS_to_feets_extract_one(self):
#     fs = FeatureSpace(SlottedA_length={"T": None}, StetsonKAC={"T": None})
#     result = fs.extract(*self.lc)
#     feets_result = dict(zip(*result))
#     self.assertFATS(feets_result)
