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

"""All feets base tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

from matplotlib import axes

import pytest

from pytest_unordered import unordered

from feets import (
    extractors,
    FeatureSpace,
    FeatureSpaceError,
    FeatureNotFound,
    Extractor,
    register_extractor,
    ExtractorContractError,
)
from feets.core import FeatureSet


# =============================================================================
# CONSTANTS
# =============================================================================

TIME_SERIE = dict.fromkeys(extractors.DATAS)


# =============================================================================
# RESULTS
# =============================================================================


def test_invalid_feature():
    with pytest.raises(FeatureNotFound):
        FeatureSet(
            features_names=["Fail"],
            values={"fail": 1},
            timeserie=TIME_SERIE,
            extractors={},
        )


def test_iter(foo_extractor):
    rs = FeatureSet(
        features_names=["foo"],
        values={"foo": 1},
        timeserie=TIME_SERIE,
        extractors={"foo": foo_extractor},
    )
    feats, values = rs
    assert list(feats) == unordered(["foo"])
    assert list(values) == unordered([1])


def test_getitem(foo_extractor):
    rs = FeatureSet(
        features_names=["foo"],
        values={"foo": 1},
        timeserie=TIME_SERIE,
        extractors={"foo": foo_extractor},
    )

    assert rs["foo"] == 1
    with pytest.raises(KeyError):
        rs["faaa"]


def test_as_array(foo_extractor):
    rs = FeatureSet(
        features_names=["foo"],
        values={"foo": 1},
        timeserie=TIME_SERIE,
        extractors={"foo": foo_extractor},
    )
    feats, values = rs.as_arrays()
    assert list(feats) == unordered(["foo"])
    assert list(values) == unordered([1])


def test_as_dict(foo_extractor):
    rs = FeatureSet(
        features_names=["foo"],
        values={"foo": 1},
        timeserie=TIME_SERIE,
        extractors={"foo": foo_extractor},
    )
    assert rs.as_dict() == {"foo": 1}


def test_as_dataframe(foo_extractor):
    rs = FeatureSet(
        features_names=["foo"],
        values={"foo": 1},
        timeserie=TIME_SERIE,
        extractors={"foo": foo_extractor},
    )

    expected = pd.DataFrame([{"foo": 1.0}])
    assert rs.as_dataframe().equals(expected)


def test_repr(foo_extractor):
    timeserie = TIME_SERIE.copy()
    timeserie.update(time=1, error=2)

    rs = FeatureSet(
        features_names=["foo"],
        values={"foo": 1},
        timeserie=timeserie,
        extractors={"foo": foo_extractor},
    )
    expected = "FeatureSet(features=<foo>, timeserie=<time, error>)"
    assert repr(rs) == str(rs) == expected


def test_plot(foo_extractor):
    rs = FeatureSet(
        features_names=["foo"],
        values={"foo": 1},
        timeserie=TIME_SERIE,
        extractors={"foo": foo_extractor},
    )

    assert isinstance(rs.plot("foo"), axes.Axes)


# =============================================================================
# SPACE
# =============================================================================


def test_extract():
    space = FeatureSpace(only=["Amplitude"])
    magnitude = np.array(
        [
            0.46057565,
            0.51372940,
            0.70136533,
            0.21454228,
            0.54792300,
            0.33433717,
            0.44879870,
            0.55571062,
            0.24388037,
            0.44793366,
            0.30175873,
            0.88326381,
            0.12208977,
            0.37088649,
            0.59457310,
            0.74705894,
            0.24551664,
            0.36009236,
            0.80661981,
            0.04961063,
            0.87747311,
            0.97388975,
            0.95775496,
            0.34195989,
            0.54201036,
            0.87854618,
            0.07388174,
            0.21543205,
            0.59295337,
            0.56771493,
        ]
    )

    features, values = space.extract(magnitude=magnitude)
    assert len(features) == 1 and features[0] == "Amplitude"
    np.testing.assert_allclose(values[features == "Amplitude"], 0.45203809)


def test_features_order(mock_extractors_register):
    @register_extractor
    class ReturnSame(Extractor):
        data = ["magnitude"]
        features = ["Same"]

        def fit(self, magnitude):
            return {"Same": magnitude[0]}

    space = FeatureSpace(only=["Same"])

    for _ in range(200):
        data = np.unique(np.random.randint(1, 1000, 10))
        np.random.shuffle(data)

        features, values_col = space.extract(magnitude=data)
        np.testing.assert_array_equal(data[0], values_col)


def test_features_kwargs():
    # ok
    FeatureSpace(only=["CAR_sigma"], CAR={"minimize_method": "powell"})

    # invalid parameter
    with pytest.raises(ExtractorContractError):
        FeatureSpace(only=["CAR_sigma"], CAR={"o": 1})

    # invalid parameter with valid parameter
    with pytest.raises(ExtractorContractError):
        FeatureSpace(
            only=["CAR_sigma"], CAR={"o": 1, "minimize_method": "powell"}
        )


def test_remove_by_dependencies(mock_extractors_register):
    @register_extractor
    class A(Extractor):
        data = ["magnitude"]
        features = ["test_a", "test_a2"]

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
    class C(Extractor):
        data = ["magnitude"]
        features = ["test_c"]

        def fit(self, *args):
            pass

    fs = FeatureSpace(exclude=["test_a"])
    assert list(fs.features_) == unordered(["test_c", "test_a2"])


def test_with_optional_data(mock_extractors_register):
    @register_extractor
    class A(Extractor):
        data = ["magnitude", "time"]
        optional = ["magnitude"]
        features = ["test_a"]

        def fit(self, *args):
            pass

    fs = FeatureSpace(data=["time"])
    assert len(fs.features_extractors_) == 1
    assert isinstance(list(fs.features_extractors_)[0], A)

    fs = FeatureSpace(data=["time", "magnitude"])
    assert len(fs.features_extractors_) == 1
    assert isinstance(list(fs.features_extractors_)[0], A)

    with pytest.raises(FeatureSpaceError):
        fs = FeatureSpace(data=["magnitude"])


def test_with_optional_data_call(mock_extractors_register):
    @register_extractor
    class A(Extractor):
        data = ["magnitude", "time"]
        optional = ["magnitude"]
        features = ["time_arg", "magnitude_arg"]

        def fit(self, time, magnitude):
            return {"time_arg": time, "magnitude_arg": magnitude}

    time, magnitude = [1, 2, 3], [4, 5, 6]

    fs = FeatureSpace(data=["time"])
    result = fs.extract(time=time, magnitude=magnitude)
    np.testing.assert_array_equal(result["time_arg"], time)
    np.testing.assert_array_equal(result["magnitude_arg"], magnitude)

    result = fs.extract(time=time)
    np.testing.assert_array_equal(result["time_arg"], time)
    np.testing.assert_array_equal(result["magnitude_arg"], None)
