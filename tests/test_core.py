#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.core import FeatureSet, FeatureSpace

import numpy as np

import pytest


# =============================================================================
# FAKE CLASSES AND FIXTURES FOR TESTING
# =============================================================================


class FakeExtractorRegistry:
    def __init__(self, extractors):
        self._extractors = extractors

    def get_execution_plan(self, *, data=None, only=None, exclude=None):
        return self._extractors


@pytest.fixture
def mock_extractor_registry(mocker):

    def maker(extractors):
        mocker.patch(
            "feets.core.extractors.extractor_registry",
            FakeExtractorRegistry(extractors),
        )

    return maker


@pytest.fixture
def mock_run(mocker):
    def maker(results):
        def fake_run(*args, **kwargs):
            return results

        mocker.patch("feets.runner.run", fake_run)

    return maker


@pytest.fixture
def fake_extractor_cls():
    def maker(features, data=None, default_params=None):
        if data is None:
            data = []
        if default_params is None:
            default_params = {}

        class FakeExtractor:
            def __init__(self, **kwargs):
                self.kawrgs = kwargs

            @classmethod
            def get_features(cls):
                return frozenset(features)

            @classmethod
            def get_data(cls):
                return frozenset(data)

            @classmethod
            def get_default_params(cls):
                return default_params

        return FakeExtractor

    return maker


# =============================================================================
# FEATURE SPACE TESTS
# =============================================================================


def test_FeatureSpace_init(mock_extractor_registry, fake_extractor_cls):
    extractor_clss = [
        fake_extractor_cls(features=["feature1", "feature2"], data=["data1"]),
        fake_extractor_cls(features=["feature3", "feature4"], data=["data2"]),
    ]
    mock_extractor_registry(extractor_clss)

    fs = FeatureSpace()

    assert isinstance(fs._extractors[0], extractor_clss[0])
    assert isinstance(fs._extractors[1], extractor_clss[1])

    np.testing.assert_equal(
        fs._selected_features,
        frozenset(["feature1", "feature2", "feature3", "feature4"]),
    )
    np.testing.assert_equal(
        fs._required_data,
        frozenset(["data1", "data2"]),
    )


def test_FeatureSpace_init_kwargs(mock_extractor_registry, fake_extractor_cls):
    extractor_clss = [
        fake_extractor_cls(
            features=["feature1"], default_params={"param1": 1}
        ),
        fake_extractor_cls(
            features=["feature2"], default_params={"param2": 2}
        ),
    ]
    mock_extractor_registry(extractor_clss)

    fs = FeatureSpace()

    np.testing.assert_equal(fs._extractors[0].kawrgs, {"param1": 1})
    np.testing.assert_equal(fs._extractors[1].kawrgs, {"param2": 2})


def test_FeatureSpace_init_only(mock_extractor_registry, fake_extractor_cls):
    extractor_clss = [
        fake_extractor_cls(features=["feature1", "feature2"]),
        fake_extractor_cls(features=["feature3", "feature4"]),
        fake_extractor_cls(features=["feature5"]),
    ]
    mock_extractor_registry(extractor_clss)

    fs = FeatureSpace(only=["feature1", "feature3"])

    np.testing.assert_equal(
        fs._selected_features,
        frozenset(["feature1", "feature3"]),
    )


def test_FeatureSpace_repr(mock_extractor_registry, fake_extractor_cls):
    extractor_clss = [
        fake_extractor_cls(features=["feature1"]),
        fake_extractor_cls(features=["feature2"]),
    ]
    mock_extractor_registry(extractor_clss)
    fs = FeatureSpace()
    np.testing.assert_equal(
        repr(fs), f"<FeatureSpace: {fs._extractors[0]}, {fs._extractors[1]}>"
    )


def test_FeatureSpace_extract(
    mock_extractor_registry, fake_extractor_cls, mock_run
):
    extractor_clss = [
        fake_extractor_cls(features=["feature1", "feature2", "feature3"])
    ]
    mock_extractor_registry(extractor_clss)
    mock_run({"feature1": 1, "feature2": 2, "feature3": 3})

    fs = FeatureSpace()

    features = fs.extract()

    expected = FeatureSet(
        "features",
        {
            "feature1": 1,
            "feature2": 2,
            "feature3": 3,
        },
    )

    np.testing.assert_equal(
        features,
        expected,
    )


def test_FeatureSpace_properties(fake_extractor_cls, mock_extractor_registry):
    extractor_clss = [
        fake_extractor_cls(features=["feature1", "feature2"]),
        fake_extractor_cls(features=["feature3", "feature4"]),
    ]
    mock_extractor_registry(extractor_clss)

    fs = FeatureSpace()

    np.testing.assert_equal(
        fs.features,
        frozenset(["feature1", "feature2", "feature3", "feature4"]),
    )

    assert isinstance(fs.execution_plan[0], extractor_clss[0])
    assert isinstance(fs.execution_plan[1], extractor_clss[1])
