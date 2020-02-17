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

import numpy as np

import pytest

from feets import FeatureSpace, preprocess


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


# =============================================================================
# CHECK IF OUR CHANGES DONT BREAK THE ORIGInaL FATS IMPLEMENTATION
# =============================================================================

def get_feature_assert_params(feature):
    feature_params = {
        "PeriodLS": {"atol": 1e-04},
        "Period_fit": {"atol": 1e-40},
        "Psi_CS": {"atol": 1e-02},
        "Psi_eta": {"atol": 1e-01}}
    params = {"err_msg": f"Feature '{feature}' missmatch."}
    params.update(feature_params.get(feature, {}))
    return params


def assertFATS(feets_result, features, FATS_values):
    for feature in features:

        if feature not in feets_result:
            pytest.fail("Missing feature {}".format(feature))

        # some features changes the values explicity  and must
        # not be evaluates
        if "_harmonics_" in feature:
            continue

        feets_value = feets_result[feature]
        FATS_value = FATS_values[feature]
        params = get_feature_assert_params(feature)
        np.testing.assert_allclose(feets_value, FATS_value, **params)


def test_F2f_extract_one_same_values(aligned_MACHO_by_FATS, FATS_results):
    lc = (
        aligned_MACHO_by_FATS.time,
        aligned_MACHO_by_FATS.mag,
        aligned_MACHO_by_FATS.error,
        aligned_MACHO_by_FATS.mag2,
        aligned_MACHO_by_FATS.aligned_time,
        aligned_MACHO_by_FATS.aligned_mag,
        aligned_MACHO_by_FATS.aligned_mag2,
        aligned_MACHO_by_FATS.aligned_error,
        aligned_MACHO_by_FATS.aligned_error2)

    features, FATS_values = FATS_results.features, FATS_results.fvalues

    fs = FeatureSpace(
        SlottedA_length={"T": None},
        StetsonKAC={"T": None})
    result = fs.extract(*lc)
    feets_result = dict(zip(*result))

    feets_result.update({
        "PeriodLS": feets_result.pop("PeriodLS_0"),
        "Period_fit": feets_result.pop("Period_fit_0"),
        "Psi_eta": feets_result.pop("Psi_eta_0"),
        "Psi_CS": feets_result.pop("Psi_CS_0")})

    assertFATS(feets_result, features, FATS_values)
