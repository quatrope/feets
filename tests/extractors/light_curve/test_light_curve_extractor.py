#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.extractors.extractor import (
    DATA_ERROR,
    DATA_FLUX,
    DATA_FLUX_ERROR,
    DATA_MAGNITUDE,
    DATA_MAGNITUDE2,
    DATA_TIME,
    ExtractorValidationError,
)
from feets.extractors.light_curve.light_curve_extractor import (
    LightCurveExtractor,
)

import numpy as np

import pytest


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def TestExtractor():
    class TestExtractor(LightCurveExtractor):
        features = {"test_feature_1", "test_feature_2"}

        def __init__(
            self, test_param_1=None, test_param_2=None, test_param_3=None
        ):
            self.test_param_1 = test_param_1
            self.test_param_2 = test_param_2
            self.test_param_3 = test_param_3

        def extract(
            self,
            time,
            flux,
            flux_error,
            test_dependency_1,
            test_dependency_2,
            magnitude=None,
            error=None,
        ):
            pass

    return TestExtractor


# =============================================================================
# EXTRACTOR TESTS
# =============================================================================


def test_LightCurveExtractor_init():
    with pytest.raises(TypeError):
        LightCurveExtractor()


def test_LightCurveExtractor_init_subclass(TestExtractor):
    with pytest.raises(AttributeError):
        TestExtractor.features


def test_LightCurveExtractor_abstract_subclass():
    class AbstractExtractor(LightCurveExtractor):
        __abstractclass__ = True

    with pytest.raises(TypeError):
        AbstractExtractor()


@pytest.mark.parametrize(
    ("data", "dependencies", "expected"),
    [
        (
            {
                DATA_TIME: [0, 1, 2],
                DATA_FLUX: [10, 20, 30],
                DATA_FLUX_ERROR: [0.1, 0.2, 0.3],
                DATA_MAGNITUDE: [10, 20, 30],
                DATA_ERROR: [0.1, 0.2, 0.3],
                DATA_MAGNITUDE2: [100, 200, 300],
            },
            {
                "test_dependency_1": 1,
                "test_dependency_2": 2,
                "test_dependency_3": 3,
            },
            {
                DATA_TIME: [0, 1, 2],
                DATA_FLUX: [10, 20, 30],
                DATA_FLUX_ERROR: [100, 25, 11.111111],
                DATA_MAGNITUDE: [10, 20, 30],
                DATA_ERROR: [100, 25, 11.111111],
                "test_dependency_1": 1,
                "test_dependency_2": 2,
            },
        ),
        (
            {
                DATA_TIME: [0, 1, 2],
                DATA_FLUX: [10, 20, 30],
                DATA_FLUX_ERROR: [0.1, 0.2, 0.3],
            },
            {
                "test_dependency_1": 1,
                "test_dependency_2": 2,
            },
            {
                DATA_TIME: [0, 1, 2],
                DATA_FLUX: [10, 20, 30],
                DATA_FLUX_ERROR: [100, 25, 11.111111],
                DATA_MAGNITUDE: [0, 0, 0],
                DATA_ERROR: [1, 1, 1],
                "test_dependency_1": 1,
                "test_dependency_2": 2,
            },
        ),
    ],
)
def test_LightCurveExtractor_prepare_extract(
    TestExtractor, data, dependencies, expected
):
    result = TestExtractor.prepare_extract(data, dependencies)

    # keys
    np.testing.assert_equal(result.keys(), expected.keys())

    # required data
    np.testing.assert_allclose(result[DATA_TIME], expected[DATA_TIME])
    np.testing.assert_allclose(result[DATA_FLUX], expected[DATA_FLUX])
    np.testing.assert_allclose(
        result[DATA_FLUX_ERROR], expected[DATA_FLUX_ERROR]
    )

    # optional data
    np.testing.assert_allclose(
        result[DATA_MAGNITUDE], expected[DATA_MAGNITUDE]
    )
    np.testing.assert_allclose(result[DATA_ERROR], expected[DATA_ERROR])

    # dependencies
    np.testing.assert_equal(
        result["test_dependency_1"], expected["test_dependency_1"]
    )
    np.testing.assert_equal(
        result["test_dependency_2"], expected["test_dependency_2"]
    )


def test_LightCurveExtractor_prepare_extract_missing_dependency(TestExtractor):
    data = {DATA_TIME: [0, 1, 2], DATA_FLUX: [10, 20, 30]}
    dependencies = {
        "test_dependency_1": 1,
    }

    with pytest.raises(ExtractorValidationError):
        TestExtractor().prepare_extract(data, dependencies)


def test_LightCurveExtractor_prepare_extract_missing_required_data(
    TestExtractor,
):
    data = {DATA_TIME: [0, 1, 2]}
    dependencies = {"test_dependency_1": 1, "test_dependency_2": 2}

    with pytest.raises(ExtractorValidationError):
        TestExtractor().prepare_extract(data, dependencies)
