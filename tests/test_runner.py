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

from feets.runner import DataRequiredError, run

import numpy as np

import pytest


# =============================================================================
# FAKE CLASSES AND FIXTURES FOR TESTING
# =============================================================================


class FakeExtractor:
    def __init__(self, *, features, data=None, dependencies=None):
        self.features = features
        self.data = data or []
        self.dependencies = dependencies or []

    def select_kwargs(self, datas, delayed_features):
        kwargs = {data: datas[data] for data in self.data}
        kwargs.update(
            {
                feature: delayed_features[feature]
                for feature in self.dependencies
            }
        )
        return kwargs

    def extract_and_validate(self, kwargs):
        data_sum = sum(kwargs[data] for data in self.data)
        dependency_sum = sum(kwargs[feature] for feature in self.dependencies)
        return {
            feature: data_sum + dependency_sum for feature in self.features
        }

    def get_features(self):
        return self.features


# =============================================================================
# TESTS
# =============================================================================


def test_run():
    features = run(
        extractors=[
            FakeExtractor(features=["feature1", "feature2"], data=["data1"]),
            FakeExtractor(features=["feature3", "feature4"], data=["data1"]),
        ],
        selected_features=["feature1", "feature2", "feature3"],
        required_data=["data1"],
        data1=123,
    )
    np.testing.assert_equal(
        features,
        {
            "feature1": 123,
            "feature2": 123,
            "feature3": 123,
        },
    )


def test_run_dependencies():
    features = run(
        extractors=[
            FakeExtractor(features=["feature1"], data=["data1"]),
            FakeExtractor(
                features=["feature2"],
                data=["data1"],
                dependencies=["feature1"],
            ),
        ],
        selected_features=["feature1", "feature2"],
        required_data=["data1"],
        data1=123,
    )
    np.testing.assert_equal(
        features,
        {
            "feature1": 123,
            "feature2": 246,
        },
    )


def test_run_empty_data():
    features = run(
        extractors=[
            FakeExtractor(features=["feature1"], data=[]),
        ],
        selected_features=["feature1"],
        required_data=[],
    )
    np.testing.assert_equal(
        features,
        {
            "feature1": 0,
        },
    )


def test_run_missing_data():
    with pytest.raises(
        DataRequiredError, match="Required data 'data1' not found"
    ):
        run(
            extractors=[FakeExtractor(features=["feature1"])],
            selected_features=["feature1"],
            required_data=["data1"],
        )
