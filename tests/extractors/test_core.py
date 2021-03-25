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
    Extractor,
    ExtractorContractError,
    extractors,
    register_extractor,
)

import pytest

from pytest_unordered import unordered


# =============================================================================
# FIXTURES
# =============================================================================

FLATTEN_PARAMS = dict.fromkeys(extractors.DATAS + ("features",))


# =============================================================================
# DEPENDENCIES TEST CASES
# =============================================================================


@mock.patch("feets.extractors._extractors", {})
def test_sort_by_dependencies():
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
            assert ext is a
        elif idx in (1, 2):
            assert ext in (b1, b2)
        elif idx == 3:
            assert ext is c
        else:
            pytest.fail("to many extractors in plan: {}".format(idx))


# =============================================================================
# FLATTEN TESTCASES
# =============================================================================


@mock.patch("feets.extractors._extractors", {})
def test_default_flatten_invalid_feature_for_the_extracor():
    @register_extractor
    class A(Extractor):
        data = ["magnitude"]
        features = ["feat"]

        def fit(self, *args):
            pass

    ext = A()
    with pytest.raises(ExtractorContractError):
        ext.flatten("foo", 1, **FLATTEN_PARAMS)


@mock.patch("feets.extractors._extractors", {})
def test_default_flatten_scalar():
    @register_extractor
    class A(Extractor):
        data = ["magnitude"]
        features = ["feat"]

        def fit(self, *args):
            pass

    ext = A()
    expected = {"feat": 1}
    assert ext.flatten("feat", 1, **FLATTEN_PARAMS) == expected


@mock.patch("feets.extractors._extractors", {})
def test_default_flatten_1D():
    @register_extractor
    class A(Extractor):
        data = ["magnitude"]
        features = ["feat"]

        def fit(self, *args):
            pass

    ext = A()
    expected = {"feat_0": 1, "feat_1": 2}
    assert ext.flatten("feat", [1, 2], **FLATTEN_PARAMS) == expected


@mock.patch("feets.extractors._extractors", {})
def test_default_flatten_2D():
    @register_extractor
    class A(Extractor):
        data = ["magnitude"]
        features = ["feat"]

        def fit(self, *args):
            pass

    ext = A()
    expected = {"feat_0_0": 1, "feat_0_1": 2, "feat_1_0": 3, "feat_1_1": 4}
    assert ext.flatten("feat", [[1, 2], [3, 4]], **FLATTEN_PARAMS) == expected


@mock.patch("feets.extractors._extractors", {})
def test_default_flatten_3D():
    @register_extractor
    class A(Extractor):
        data = ["magnitude"]
        features = ["feat"]

        def fit(self, *args):
            pass

    ext = A()

    value = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    expected = {
        "feat_0_0_0": 1,
        "feat_0_0_1": 2,
        "feat_0_1_0": 3,
        "feat_0_1_1": 4,
        "feat_1_0_0": 5,
        "feat_1_0_1": 6,
        "feat_1_1_0": 7,
        "feat_1_1_1": 8,
    }

    assert ext.flatten("feat", value, **FLATTEN_PARAMS) == expected


@mock.patch("feets.extractors._extractors", {})
def test_default_flatten_4D():
    @register_extractor
    class A(Extractor):
        data = ["magnitude"]
        features = ["feat"]

        def fit(self, *args):
            pass

    ext = A()

    value = [[[[1, 2], [3, 4]]]]

    expected = {
        "feat_0_0_0_0": 1,
        "feat_0_0_0_1": 2,
        "feat_0_0_1_0": 3,
        "feat_0_0_1_1": 4,
    }

    assert ext.flatten("feat", value, **FLATTEN_PARAMS) == expected


@mock.patch("feets.extractors._extractors", {})
def test_flatten_return_ndim_gt_0():
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

    with pytest.raises(ExtractorContractError):
        ext.flatten("feat", value, **FLATTEN_PARAMS)


@mock.patch("feets.extractors._extractors", {})
def test_flatten_return_not_dict_instance():
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

    with pytest.raises(ExtractorContractError):
        ext.flatten("feat", value, **FLATTEN_PARAMS)


@mock.patch("feets.extractors._extractors", {})
def test_flatten_return_invalid_name():
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

    with pytest.raises(ExtractorContractError):
        ext.flatten("feat", value, **FLATTEN_PARAMS)


# =============================================================================
# RequiredDataTestCases
# =============================================================================


@mock.patch("feets.extractors._extractors", {})
def test_required_data_fail():
    with pytest.raises(extractors.ExtractorBadDefinedError):

        class A(Extractor):
            data = ["magnitude", "time"]
            optional = ["error"]
            features = ["test_a"]

            def fit(self, *args):
                pass


@mock.patch("feets.extractors._extractors", {})
def test_required_data():
    class A(Extractor):
        data = ["magnitude", "time"]
        optional = ["magnitude"]
        features = ["test_a"]

        def fit(self, *args):
            pass

    assert A.get_optional() == unordered(["magnitude"])
    assert A.get_data() == unordered(["magnitude", "time"])
    assert A.get_required_data() == unordered(["time"])


@mock.patch("feets.extractors._extractors", {})
def test_all_required_data():
    class A(Extractor):
        data = ["magnitude", "time"]
        features = ["test_a"]

        def fit(self, *args):
            pass

    assert A.get_optional() == unordered([])
    assert A.get_data() == unordered(["magnitude", "time"])
    assert A.get_required_data() == unordered(["time", "magnitude"])


@mock.patch("feets.extractors._extractors", {})
def test_fail_on_all_optional_data():
    with pytest.raises(extractors.ExtractorBadDefinedError):

        @register_extractor
        class A(Extractor):
            data = ["magnitude", "time"]
            optional = ["magnitude", "time"]
            features = ["test_a"]

            def fit(self, *args):
                pass


# =============================================================================
# TEST has PLOTS
# =============================================================================


@pytest.mark.parametrize("ename, ext_cls", extractors._extractors.items())
def test_implement_plot_feature(ename, ext_cls):
    msg = f"Extractor {ename} must implement plot_fature() method."
    assert ext_cls.plot_feature is not Extractor.plot_feature, msg
