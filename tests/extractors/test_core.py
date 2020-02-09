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

from unittest import mock

from feets import (
    Extractor, register_extractor, extractors, ExtractorContractError)

from ..core import FeetsTestCase


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
