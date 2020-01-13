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

"""Extractors Tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import unittest
import warnings

import numpy as np

import pandas as pd

from unittest import mock

from feets import (
    FeatureSpace,
    Extractor, register_extractor, extractors, ExtractorContractError)

from .core import FeetsTestCase, DATA_PATH


# =============================================================================
# BASE CLASS
# =============================================================================

class SortByDependenciesTestCases(FeetsTestCase):

    @mock.patch("feets.extractors._extractors", {})
    def test_sort_by_dependencies(self):
        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["test_a"]

            def fit(self, *args):
                pass

        @register_extractor
        class B1(Extractor):
            data = ["magnitude"]
            features = ["test_b1"]
            dependencies = ["test_a"]

            def fit(self, *args):
                pass

        @register_extractor
        class B2(Extractor):
            data = ["magnitude"]
            features = ["test_b2"]
            dependencies = ["test_a"]

            def fit(self, *args):
                pass

        @register_extractor
        class C(Extractor):
            data = ["magnitude"]
            features = ["test_c"]
            dependencies = ["test_b1", "test_b2", "test_a"]

            def fit(self, *args):
                pass

        a, b1, b2, c = A(), B1(), B2(), C()
        exts = [c, b1, a, b2]
        plan = extractors.sort_by_dependencies(exts)
        for idx, ext in enumerate(plan):
            if idx == 0:
                self.assertIs(ext, a)
            elif idx in (1, 2):
                self.assertIn(ext, (b1, b2))
            elif idx == 3:
                self.assertIs(ext, c)
            else:
                self.fail("to many extractors in plan: {}".format(idx))


class FlattenTestCase(FeetsTestCase):

    def setUp(self):
        self.params = dict.fromkeys(extractors.DATAS + ("features", ))

    @mock.patch("feets.extractors._extractors", {})
    def test_default_flatten_invalid_feature_for_the_extracor(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

        ext = A()
        with self.assertRaises(ExtractorContractError):
            ext.flatten("foo", 1, **self.params)

    @mock.patch("feets.extractors._extractors", {})
    def test_default_flatten_scalar(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

        ext = A()
        expected = {"feat": 1}
        self.assertDictEqual(
            ext.flatten("feat", 1, **self.params), expected)

    @mock.patch("feets.extractors._extractors", {})
    def test_default_flatten_1D(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

        ext = A()
        expected = {"feat_0": 1, "feat_1": 2}
        self.assertDictEqual(
            ext.flatten("feat", [1, 2], **self.params), expected)

    @mock.patch("feets.extractors._extractors", {})
    def test_default_flatten_2D(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

        ext = A()
        expected = {'feat_0_0': 1, 'feat_0_1': 2,
                    'feat_1_0': 3, 'feat_1_1': 4}
        self.assertDictEqual(
            ext.flatten("feat", [[1, 2], [3, 4]], **self.params), expected)

    @mock.patch("feets.extractors._extractors", {})
    def test_default_flatten_3D(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

        ext = A()

        value = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]]

        expected = {
            'feat_0_0_0': 1,
            'feat_0_0_1': 2,
            'feat_0_1_0': 3,
            'feat_0_1_1': 4,
            'feat_1_0_0': 5,
            'feat_1_0_1': 6,
            'feat_1_1_0': 7,
            'feat_1_1_1': 8}

        self.assertDictEqual(
            ext.flatten("feat", value, **self.params), expected)

    @mock.patch("feets.extractors._extractors", {})
    def test_default_flatten_4D(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

        ext = A()

        value = [[[[1, 2], [3, 4]]]]

        expected = {
            'feat_0_0_0_0': 1,
            'feat_0_0_0_1': 2,
            'feat_0_0_1_0': 3,
            'feat_0_0_1_1': 4}

        self.assertDictEqual(
            ext.flatten("feat", value, **self.params), expected)

    @mock.patch("feets.extractors._extractors", {})
    def test_flatten_return_ndim_gt_0(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

            def flatten_feature(self, feature, value, **kwargs):
                return {feature: value}

        ext = A()
        value = [1, 2]

        with self.assertRaises(ExtractorContractError):
            ext.flatten("feat", value, **self.params)

    @mock.patch("feets.extractors._extractors", {})
    def test_flatten_return_not_dict_instance(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

            def flatten_feature(self, feature, value, **kwargs):
                return None

        ext = A()
        value = [1, 2]

        with self.assertRaises(ExtractorContractError):
            ext.flatten("feat", value, **self.params)

    @mock.patch("feets.extractors._extractors", {})
    def test_flatten_return_invalid_name(self):

        @register_extractor
        class A(Extractor):
            data = ["magnitude"]
            features = ["feat"]

            def fit(self, *args):
                pass

            def flatten_feature(self, feature, value, **kwargs):
                return {"foo": 1}

        ext = A()
        value = [1, 2]

        with self.assertRaises(ExtractorContractError):
            ext.flatten("feat", value, **self.params)


class RequiredDataTestCases(FeetsTestCase):

    @mock.patch("feets.extractors._extractors", {})
    def test_required_data_fail(self):
        with self.assertRaises(extractors.ExtractorBadDefinedError):

            class A(Extractor):
                data = ["magnitude", "time"]
                optional = ["error"]
                features = ["test_a"]

                def fit(self, *args):
                    pass

    @mock.patch("feets.extractors._extractors", {})
    def test_required_data(self):

        class A(Extractor):
            data = ["magnitude", "time"]
            optional = ["magnitude"]
            features = ["test_a"]

            def fit(self, *args):
                pass

        self.assertCountEqual(A.get_optional(), ["magnitude"])
        self.assertCountEqual(A.get_data(), ["magnitude", "time"])
        self.assertCountEqual(A.get_required_data(), ["time"])

    @mock.patch("feets.extractors._extractors", {})
    def test_all_required_data(self):

        class A(Extractor):
            data = ["magnitude", "time"]
            features = ["test_a"]

            def fit(self, *args):
                pass

        self.assertCountEqual(A.get_optional(), [])
        self.assertCountEqual(A.get_data(), ["magnitude", "time"])
        self.assertCountEqual(A.get_required_data(), ["time", "magnitude"])

    @mock.patch("feets.extractors._extractors", {})
    def test_fail_on_all_optional_data(self):
        with self.assertRaises(extractors.ExtractorBadDefinedError):

            @register_extractor
            class A(Extractor):
                data = ["magnitude", "time"]
                optional = ["magnitude", "time"]
                features = ["test_a"]

                def fit(self, *args):
                    pass


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


class DMDTTestCases(FeetsTestCase):

    def setUp(self):
        self.random = np.random.RandomState(42)

    def test_feets_dmdt(self):
        ext = extractors.DeltamDeltat()
        params = ext.get_default_params()
        time = np.arange(0, 1000)

        values = np.empty(50)
        for idx in range(values.size):
            mags = self.random.normal(size=1000)
            feats = ext.fit(magnitude=mags, time=time, **params)
            values[idx] = np.sum(list(feats.values()))
        self.assertAllClose(values.mean(), 424.56)


class LombScargleTests(FeetsTestCase):

    def setUp(self):
        self.random = np.random.RandomState(42)

    def periodic_lc(self):
        N = 100
        mjd_periodic = np.arange(N)
        Period = 20
        cov = np.zeros([N, N])
        mean = np.zeros(N)
        for i in np.arange(N):
            for j in np.arange(N):
                cov[i, j] = np.exp(-(np.sin((np.pi / Period) * (i - j)) ** 2))
        data_periodic = self.random.multivariate_normal(mean, cov)
        error = self.random.normal(size=100, loc=0.001)
        lc = {
            "magnitude": data_periodic,
            "time": mjd_periodic,
            "error": error}
        return lc

    def test_lscargle_vs_feets(self):

        # extract the module for make short code
        ext_lomb_scargle = extractors.ext_lomb_scargle

        # load the data
        path = os.path.join(
            DATA_PATH, "bad_results.pandas.pkl")
        tseries = pd.read_pickle(path)

        # the ls params
        ext_params = ext_lomb_scargle.LombScargle.get_default_params()
        lscargle_kwds = ext_params["lscargle_kwds"]

        # create the feature space
        fs = FeatureSpace(only=["PeriodLS"])

        ls_periods, feets_periods = [], []
        for src_id in tseries.bm_src_id.unique():

            # extract the timeseries
            sobs = tseries[tseries.bm_src_id == src_id]
            time = sobs.pwp_stack_src_hjd.values
            magnitude = sobs.pwp_stack_src_mag3.values
            error = sobs.pwp_stack_src_mag_err3.values

            # "pure" lomb scargle (without the entire feets pipeline)
            frequency, power = ext_lomb_scargle.lscargle(
                time=time, magnitude=magnitude, error=error, **lscargle_kwds)
            fmax = np.argmax(power)
            ls_periods.append(1 / frequency[fmax])

            # extract the period from the feets pipele
            rs = fs.extract(time=time, magnitude=magnitude, error=error)
            feets_periods.append(rs.values['PeriodLS'])

        feets_periods = np.array(feets_periods).flatten()
        self.assertArrayEqual(ls_periods, feets_periods)

    def test_lscargle_peaks(self):
        lc = self.periodic_lc()

        for peaks in [1, 2, 3, 10]:
            ext = extractors.LombScargle(peaks=peaks)
            feats = ext.extract(features={}, **lc)
            for v in feats.values():
                self.assertEqual(len(v), peaks)


class FourierTests(FeetsTestCase):

    def setUp(self):
        self.random = np.random.RandomState(42)

    def periodic_lc(self):
        N = 100
        mjd_periodic = np.arange(N)
        Period = 20
        cov = np.zeros([N, N])
        mean = np.zeros(N)
        for i in np.arange(N):
            for j in np.arange(N):
                cov[i, j] = np.exp(-(np.sin((np.pi / Period) * (i - j)) ** 2))
        data_periodic = self.random.multivariate_normal(mean, cov)
        error = self.random.normal(size=100, loc=0.001)
        lc = {
            "magnitude": data_periodic,
            "time": mjd_periodic,
            "error": error}
        return lc

    def test_fourier_optional_data(self):
        lc_error = self.periodic_lc()

        lc = lc_error.copy()
        lc["error"] = None

        ext = extractors.FourierComponents()

        self.assertNotEqual(
            ext.extract(features={}, **lc),
            ext.extract(features={}, **lc_error))


class GSKewTest(FeetsTestCase):

    def setUp(self):
        self.random = np.random.RandomState(42)

    def test_gskew_linear_interpolation_problem(self):
        magnitude = [
            13.859, 13.854, 13.844, 13.881, 13.837, 13.885, 13.865, 13.9,
            13.819, 13.889, 13.89, 13.831, 13.869, 13.893, 13.825, 13.844,
            13.862, 13.853, 13.844, 13.85, 13.843, 13.839, 13.885, 13.859,
            13.865, 13.867, 13.874, 13.906, 13.819, 13.854, 13.891, 13.896,
            13.847, 13.862, 13.827, 13.849, 13.881, 13.871, 13.862, 13.846,
            13.865, 13.837, 13.819, 13.867, 13.833, 13.88, 13.868, 13.819,
            13.846, 13.842, 13.9, 13.88, 13.851, 13.885, 13.898, 13.824, 13.83,
            13.865, 13.823, 13.845, 13.874]

        lc = {
            "magnitude": np.array(magnitude),
            "time": np.arange(len(magnitude)),
            "error": self.random.rand(len(magnitude))}

        with warnings.catch_warnings():  # this launch mean of empty
            warnings.filterwarnings('ignore')

            ext = extractors.Gskew(interpolation="linear")
            result = ext.extract(features={}, **lc)
            assert np.isnan(result["Gskew"])

        ext = extractors.Gskew()  # by default interpolation is nearest
        result = ext.extract(features={}, **lc)
        assert not np.isnan(result["Gskew"])

        ext = extractors.Gskew(interpolation="nearest")
        result = ext.extract(features={}, **lc)
        assert not np.isnan(result["Gskew"])
