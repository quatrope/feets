#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

from feets.extractors.extractor import (
    Extractor,
    ExtractorBadDefinedError,
    ExtractorTransformError,
    ExtractorValidationError,
    ExtractorWarning,
    FeatureExtractionWarning,
    extractor_warning,
    feature_warning,
)

import numpy as np

import pytest


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def TestExtractor():
    class TestExtractor(Extractor):
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
            magnitude,
            test_dependency_1,
            test_dependency_2,
            error=None,
            magnitude2=None,
        ):
            pass

    return TestExtractor


@pytest.fixture
def test_extractor(TestExtractor):
    return TestExtractor(
        test_param_1=1, test_param_2={"sub_param_1": 21, "sub_param_2": 22}
    )


# =============================================================================
# WARNING TESTS
# =============================================================================


def test_extractor_warning():
    with pytest.warns(ExtractorWarning):
        extractor_warning("test warning")


def test_feature_warning():
    with pytest.warns(FeatureExtractionWarning):
        feature_warning("test warning")


# =============================================================================
# EXTRACTOR TESTS
# =============================================================================


def test_Extractor_init():
    with pytest.raises(TypeError):
        Extractor()


def test_Extractor_init_subclass(TestExtractor):
    with pytest.raises(AttributeError):
        TestExtractor.features


def test_Extractor_abstract_subclass():
    class AbstractExtractor(Extractor):
        __abstractclass__ = True

    with pytest.raises(TypeError):
        AbstractExtractor()


def test_Extractor_no_features():
    with pytest.raises(ExtractorBadDefinedError):

        class BadExtractor(Extractor):
            def extract(self):
                pass


def test_Extractor_invalid_feature():
    with pytest.raises(ExtractorBadDefinedError):

        class BadExtractor(Extractor):
            features = ["magnitude"]

            def extract(self):
                pass


@pytest.mark.parametrize(
    "feature",
    [
        None,
        0,
        (1, 2, 3),
    ],
)
def test_Extractor_invalid_feature_format(feature):
    with pytest.raises(ExtractorBadDefinedError):

        class BadExtractor(Extractor):
            features = [feature]

            def extract(self):
                pass


@pytest.mark.parametrize(
    "feature",
    [
        "   ",
        "while",
        "(1, 2, 3)",
    ],
)
def test_Extractor_invalid_feature_name(feature):
    with pytest.raises(ExtractorBadDefinedError):

        class BadExtractor(Extractor):
            features = [feature]

            def extract(self):
                pass


def test_Extractor_repeated_feature():
    with pytest.raises(ExtractorBadDefinedError):

        class BadExtractor(Extractor):
            features = ["test_feature", "test_feature"]

            def extract(self):
                pass


def test_Extractor_dependency_has_default():
    with pytest.raises(ExtractorBadDefinedError):

        class BadExtractor(Extractor):
            features = ["test_feature"]

            def extract(self, test_dependency=None):
                pass


def test_Extractor_parameter_has_no_default():
    with pytest.raises(ExtractorBadDefinedError):

        class BadExtractor(Extractor):
            features = ["test_feature"]

            def __init__(self, test_param_1):
                pass

            def extract(self):
                pass


def test_Extractor_extract_not_implemented():
    class BadExtractor(Extractor):
        features = ["test_feature"]

        def extract(self):
            super().extract()

    with pytest.raises(NotImplementedError):
        BadExtractor().extract()


def test_Extractor_get_features(TestExtractor):
    np.testing.assert_equal(
        TestExtractor.get_features(), {"test_feature_1", "test_feature_2"}
    )


def test_Extractor_get_data(TestExtractor):
    np.testing.assert_equal(
        TestExtractor.get_data(), {"time", "magnitude", "error", "magnitude2"}
    )


def test_Extractor_get_optional_data(TestExtractor):
    np.testing.assert_equal(
        TestExtractor.get_optional_data(), {"error", "magnitude2"}
    )


def test_Extractor_get_required_data(TestExtractor):
    np.testing.assert_equal(
        TestExtractor.get_required_data(), {"time", "magnitude"}
    )


def test_Extractor_get_dependencies(TestExtractor):
    np.testing.assert_equal(
        TestExtractor.get_dependencies(),
        {"test_dependency_1", "test_dependency_2"},
    )


def test_Extractor_get_default_params(TestExtractor):
    print(TestExtractor.get_default_params())
    np.testing.assert_equal(
        TestExtractor.get_default_params(),
        {"test_param_1": None, "test_param_2": None, "test_param_3": None},
    )


def test_Extractor_prepare_extract(TestExtractor):
    data = {
        "time": [0, 1, 2],
        "magnitude": [10, 20, 30],
        "error": [0.1, 0.2, 0.3],
        "flux": [100, 200, 300],
    }
    dependencies = {
        "test_dependency_1": 1,
        "test_dependency_2": 2,
        "test_dependency_3": 3,
        "test_dependency_4": 4,
    }

    np.testing.assert_equal(
        TestExtractor.prepare_extract(data, dependencies),
        {
            "time": [0, 1, 2],
            "magnitude": [10, 20, 30],
            "error": [0.1, 0.2, 0.3],
            "test_dependency_1": 1,
            "test_dependency_2": 2,
        },
    )


def test_Extractor_prepare_extract_missing_required_dependency(TestExtractor):
    data = {
        "time": [0, 1, 2],
        "magnitude": [10, 20, 30],
    }
    dependencies = {
        "test_dependency_1": 1,
    }

    with pytest.raises(ExtractorValidationError):
        TestExtractor.prepare_extract(data, dependencies)


def test_Extractor_prepare_extract_missing_required_data(TestExtractor):
    data = {
        "time": [0, 1, 2],
    }
    dependencies = {
        "test_dependency_1": 1,
        "test_dependency_2": 2,
    }

    with pytest.raises(ExtractorValidationError):
        TestExtractor.prepare_extract(data, dependencies)


def test_Extractor_validate_extract(TestExtractor):
    features = {
        "test_feature_1": 1,
        "test_feature_2": 2,
    }

    np.testing.assert_equal(TestExtractor.validate_extract(features), None)


@pytest.mark.parametrize(
    "features",
    [
        {"test_feature_1": 1},
        {"test_feature_1": 1, "test_feature_2": 2, "test_feature_3": 3},
        {"test_feature_3": 3},
    ],
)
def test_Extractor_validate_extract_features_mismatch(TestExtractor, features):
    with pytest.raises(ExtractorValidationError):
        TestExtractor.validate_extract(features)


def test_Extractor_validate_flatten(TestExtractor):
    feature = "test_feature_1"
    flattened = {
        "test_feature_1_0": 0,
        "test_feature_1_1": 1,
        "test_feature_1_2": 2,
    }

    np.testing.assert_equal(
        TestExtractor.validate_flatten(feature, flattened), None
    )


@pytest.mark.parametrize(
    "flattened",
    [
        None,
        (1, 2, 3),
        {None: 1},
        {(1, 2, 3): 1},
        {"test_feature_1_0": None},
        {"test_feature_1_0": (1, 2, 3)},
    ],
)
def test_Extractor_validate_flatten_invalid_format(TestExtractor, flattened):
    feature = "test_feature_1"
    with pytest.raises(ExtractorValidationError):
        TestExtractor.validate_flatten(feature, flattened)


def test_Extractor_params(test_extractor):
    params = {
        "test_param_1": 1,
        "test_param_2": {"sub_param_1": 21, "sub_param_2": 22},
        "test_param_3": None,
    }
    np.testing.assert_equal(test_extractor.params, params)


def test_Extractor_to_dict(test_extractor):
    params = {
        "test_param_1": 1,
        "test_param_2": {"sub_param_1": 21, "sub_param_2": 22},
        "test_param_3": None,
    }
    np.testing.assert_equal(
        test_extractor.to_dict(),
        {"TestExtractor": params},
    )


def test_Extractor_repr(test_extractor):
    np.testing.assert_equal(
        repr(test_extractor),
        ("TestExtractor(test_param_1=1, test_param_2=..., test_param_3=None)"),
    )


@pytest.mark.parametrize(
    ["value", "flattened"],
    [
        [1, {"test_feature_1": 1}],
        ["value", {"test_feature_1": "value"}],
        [
            [0, 1, 2],
            {
                "test_feature_1_0": 0,
                "test_feature_1_1": 1,
                "test_feature_1_2": 2,
            },
        ],
        [
            {"key": "value"},
            {"test_feature_1_key": "value"},
        ],
        [
            {"key": [0, 1, 2]},
            {
                "test_feature_1_key_0": 0,
                "test_feature_1_key_1": 1,
                "test_feature_1_key_2": 2,
            },
        ],
    ],
)
def test_Extractor_flatten_feature(test_extractor, value, flattened):
    np.testing.assert_equal(
        test_extractor.flatten_feature("test_feature_1", value),
        flattened,
    )


@pytest.mark.parametrize(
    "value",
    [set(), map(lambda x: x, [1, 2, 3])],
)
def test_Extractor_flatten_feature_invalid_value_format(test_extractor, value):
    with pytest.raises(ExtractorTransformError):
        test_extractor.flatten_feature("test_feature_1", value)
