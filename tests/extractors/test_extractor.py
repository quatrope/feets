#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# IMPORRTS
# =============================================================================

from feets.extractors.extractor import (
    Extractor,
    ExtractorBadDefinedError,
    FeatureExtractionWarning,
    _ExtractorConf,
)

import numpy as np

import pytest


# =============================================================================
# EXTRACTOR CONF
# =============================================================================


mock_DATAS = ("data1", "data2", "data3")


def create_mock_extractor(
    feature_names=None, init_method=None, extract_method=None
):
    if feature_names is None:
        feature_names = ["feature1", "feature2"]

    if init_method is None:

        def init_method(self):
            pass

    if extract_method is None:

        def extract_method(self):
            pass

    class MockExtractor:
        features = feature_names
        __init__ = init_method
        extract = extract_method

    return MockExtractor


@pytest.mark.parametrize(
    "features",
    [[], [123], ["data1"], ["duplicate_feature", "duplicate_feature"]],
    ids=[
        "no_features",
        "feature_is_not_string",
        "feature_in_DATAS",
        "duplicate_feature",
    ],
)
def test_ExtractorConf_get_feature_conf_errors(mocker, features):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    ecls = create_mock_extractor(feature_names=features)
    with pytest.raises(ExtractorBadDefinedError):
        _ExtractorConf._get_feature_conf(ecls)


@pytest.mark.parametrize(
    ["features", "expected"],
    [
        (["feature1"], {"feature1"}),
        (
            ["feature1", "feature2", "feature3"],
            {"feature1", "feature2", "feature3"},
        ),
    ],
    ids=[
        "single_feature",
        "multiple_features",
    ],
)
def test_ExtractorConf_get_feature_conf_success(mocker, features, expected):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    ecls = create_mock_extractor(feature_names=features)
    feature_conf = _ExtractorConf._get_feature_conf(ecls)
    np.testing.assert_equal(feature_conf, expected)


@pytest.mark.parametrize(
    "extract_method",
    [lambda self, dependency1=123: None],
    ids=["dependency_has_default"],
)
def test_ExtractorConf_get_extract_method_parameters_errors(
    mocker, extract_method
):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    ecls = create_mock_extractor(extract_method=extract_method)
    with pytest.raises(ExtractorBadDefinedError):
        _ExtractorConf._get_extract_method_parameters(ecls)


@pytest.mark.parametrize(
    [
        "extract_method",
        "expected_required",
        "expected_optional",
        "expected_dependencies",
    ],
    [
        (None, set(), set(), set()),
        (
            lambda self, data1, data2, data3: None,
            {"data1", "data2", "data3"},
            set(),
            set(),
        ),
        (
            lambda self, data1=123, data2=456, data3=789: None,
            set(),
            {"data1", "data2", "data3"},
            set(),
        ),
        (
            lambda self, dependency1, dependency2: None,
            set(),
            set(),
            {"dependency1", "dependency2"},
        ),
        (
            lambda self, data1, dependency1, data2=123: None,
            {"data1"},
            {"data2"},
            {"dependency1"},
        ),
    ],
    ids=[
        "no_params",
        "required_data",
        "optional_data",
        "dependencies",
        "multiple_params",
    ],
)
def test_ExtractorConf_get_extract_method_parameters_success(
    mocker,
    extract_method,
    expected_required,
    expected_optional,
    expected_dependencies,
):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    ecls = create_mock_extractor(extract_method=extract_method)
    required, optional, dependencies = (
        _ExtractorConf._get_extract_method_parameters(ecls)
    )
    np.testing.assert_equal(required, expected_required)
    np.testing.assert_equal(optional, expected_optional)
    np.testing.assert_equal(dependencies, expected_dependencies)


@pytest.mark.parametrize(
    "init_method",
    [lambda self, param1: None],
    ids=["missing_default"],
)
def test_ExtractorConf_get_init_method_parameters_error(mocker, init_method):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    ecls = create_mock_extractor(init_method=init_method)
    with pytest.raises(ExtractorBadDefinedError):
        _ExtractorConf._get_init_method_parameters(ecls)


@pytest.mark.parametrize(
    ["init_method", "expected"],
    [
        (lambda self: None, {}),
        (
            lambda self, param1=123, param2=456, param3=789: None,
            {"param1": 123, "param2": 456, "param3": 789},
        ),
    ],
    ids=["no_params", "multiple_params"],
)
def test_ExtractorConf_get_init_method_parameters_success(
    mocker, init_method, expected
):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    ecls = create_mock_extractor(init_method=init_method)
    parameters = _ExtractorConf._get_init_method_parameters(ecls)
    np.testing.assert_equal(parameters, expected)


@pytest.mark.parametrize(
    ["features", "init_method", "extract_method", "expected"],
    [
        (
            ["feature1", "feature2", "feature3"],
            None,
            None,
            _ExtractorConf(
                features={"feature1", "feature2", "feature3"},
                required=set(),
                optional=set(),
                dependencies=set(),
                parameters={},
            ),
        ),
        (
            None,
            lambda self, param1=123, param2=456: None,
            None,
            _ExtractorConf(
                features={"feature1", "feature2"},
                required=set(),
                optional=set(),
                dependencies=set(),
                parameters={"param1": 123, "param2": 456},
            ),
        ),
        (
            None,
            None,
            lambda self, data1, data2, featureA, featureB, data3=123: None,
            _ExtractorConf(
                features={"feature1", "feature2"},
                required={"data1", "data2"},
                optional={"data3"},
                dependencies={"featureA", "featureB"},
                parameters={},
            ),
        ),
        (
            ["feature1", "feature2", "feature3"],
            lambda self, param1=123, param2=456: None,
            lambda self, data1, data2, featureA, featureB, data3=123: None,
            _ExtractorConf(
                features={"feature1", "feature2", "feature3"},
                required={"data1", "data2"},
                optional={"data3"},
                dependencies={"featureA", "featureB"},
                parameters={"param1": 123, "param2": 456},
            ),
        ),
    ],
    ids=[
        "no_params",
        "init_params",
        "extract_params",
        "multiple_params",
    ],
)
def test_ExtractorConf_from_extractor_class(
    mocker,
    features,
    init_method,
    extract_method,
    expected,
):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    ecls = create_mock_extractor(
        feature_names=features,
        init_method=init_method,
        extract_method=extract_method,
    )
    conf = _ExtractorConf.from_extractor_class(ecls)
    np.testing.assert_equal(conf, expected)


# =============================================================================
# EXTRACTOR
# =============================================================================


class MockExtractorConf:
    @staticmethod
    def from_extractor_class(cls):
        return "mock_conf"


def test_Extractor_init_subclass_success(mocker):
    mocker.patch(
        "feets.extractors.extractor._ExtractorConf",
        MockExtractorConf,
    )

    class MockExtractor(Extractor):
        features = ["feature1", "feature2"]

        def extract(self):
            pass

    np.testing.assert_equal(MockExtractor._conf, "mock_conf")


def test_Extractor_init_subclass_empty_feature_list(mocker):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    with pytest.raises(ExtractorBadDefinedError):

        class MockExtractor(Extractor):
            def extract(self):
                pass


def test_Extractor_init_subclass_method_not_redefined(mocker):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)
    with pytest.raises(ExtractorBadDefinedError):

        class MockExtractor(Extractor):
            features = ["feature1", "feature2"]


@pytest.mark.parametrize(
    "getter_name, expected",
    [
        ("get_features", {"feature1", "feature2"}),
        ("get_data", {"data1", "data2", "data3"}),
        ("get_optional", {"data3"}),
        ("get_required_data", {"data1", "data2"}),
        ("get_dependencies", {"dependency1"}),
        (
            "get_default_params",
            {
                "parameter1": 123,
                "parameter2": 456,
                "parameter3": 789,
            },
        ),
    ],
    ids=[
        "features",
        "data",
        "optional",
        "required",
        "dependencies",
        "parameters",
    ],
)
def test_Extractor_getters(mocker, getter_name, expected):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)

    class MockExtractor(Extractor):
        features = {"feature1", "feature2"}

        def __init__(self, parameter1=123, parameter2=456, parameter3=789):
            pass

        def extract(self, data1, data2, dependency1, data3=123):
            pass

    result = getattr(MockExtractor, getter_name)()
    np.testing.assert_equal(result, expected)


def test_Extractor_feature_warning(mocker):
    mocker.patch(
        "feets.extractors.extractor._ExtractorConf",
        MockExtractorConf,
    )

    class MockExtractor(Extractor):
        features = ["feature1", "feature2"]

        def extract(self):
            pass

    extractor = MockExtractor()
    with pytest.warns(FeatureExtractionWarning):
        extractor.feature_warning("Test warning message")


def test_Extractor_extract(mocker):
    mocker.patch("feets.extractors.extractor.DATAS", mock_DATAS)

    class MockExtractor(Extractor):
        features = ["feature1", "feature2"]

        def extract(self):
            super().extract()

    with pytest.raises(NotImplementedError):
        MockExtractor().extract()
