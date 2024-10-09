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

from typing import OrderedDict
from attr import dataclass

from feets.extractors.extractor import (
    Extractor,
    ExtractorBadDefinedError,
    ExtractorContractError,
    ExtractorTransformError,
    ExtractorWarning,
    FeatureExtractionWarning,
    _ExtractorConf,
)

import numpy as np

import pytest


# =============================================================================
# FAKE CLASSES AND FIXTURES FOR TESTING
# =============================================================================


@pytest.fixture
def mock_DATAS(mocker):
    def maker(fake_DATAS=None):
        if fake_DATAS is None:
            fake_DATAS = ("data1", "data2", "data3")

        mocker.patch("feets.extractors.extractor.DATAS", fake_DATAS)

    return maker


@pytest.fixture
def fake_ecls():
    def maker(feature_names=None, init_method=None, extract_method=None):
        if feature_names is None:
            feature_names = ["feature1", "feature2"]

        if init_method is None:

            def init_method(self):
                pass

        if extract_method is None:

            def extract_method(self):
                pass

        class FakeExtractor:
            features = feature_names
            __init__ = init_method
            extract = extract_method

        return FakeExtractor

    return maker


@pytest.fixture
def fake_extractor_conf_cls():
    def maker(
        *,
        features,
        required=None,
        optional=None,
        dependencies=None,
        default_params=None,
    ):
        if required is None:
            required = []
        if optional is None:
            optional = []
        if dependencies is None:
            dependencies = []
        if default_params is None:
            default_params = {}

        @dataclass
        class FakeExtractorConf:
            features: frozenset
            data: frozenset
            required: frozenset
            optional: frozenset
            dependencies: frozenset
            parameters: dict

            @classmethod
            def from_extractor_class(cls, *args, **kwargs):
                return FakeExtractorConf(
                    frozenset(features),
                    frozenset(required + optional),
                    frozenset(required),
                    frozenset(optional),
                    frozenset(dependencies),
                    dict(default_params),
                )

        return FakeExtractorConf

    return maker


@pytest.fixture
def mock_extractor_conf(mocker):
    def maker(extractor_conf_cls):
        mocker.patch(
            "feets.extractors.extractor._ExtractorConf", extractor_conf_cls
        )

    return maker


# =============================================================================
# EXTRACTOR CONF TESTS
# =============================================================================


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
def test_ExtractorConf_get_feature_conf(
    mock_DATAS, fake_ecls, features, expected
):
    mock_DATAS()
    ecls = fake_ecls(feature_names=features)
    feature_conf = _ExtractorConf._get_feature_conf(ecls)
    np.testing.assert_equal(feature_conf, expected)


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
def test_ExtractorConf_get_feature_conf_raises_ExtractorBadDefinedError(
    mock_DATAS, fake_ecls, features
):
    mock_DATAS()
    ecls = fake_ecls(feature_names=features)
    with pytest.raises(ExtractorBadDefinedError):
        _ExtractorConf._get_feature_conf(ecls)


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
def test_ExtractorConf_get_extract_method_parameters(
    mock_DATAS,
    fake_ecls,
    extract_method,
    expected_required,
    expected_optional,
    expected_dependencies,
):
    mock_DATAS()
    ecls = fake_ecls(extract_method=extract_method)
    required, optional, dependencies = (
        _ExtractorConf._get_extract_method_parameters(ecls)
    )
    np.testing.assert_equal(required, expected_required)
    np.testing.assert_equal(optional, expected_optional)
    np.testing.assert_equal(dependencies, expected_dependencies)


@pytest.mark.parametrize(
    "extract_method",
    [lambda self, dependency1=123: None],
    ids=["dependency_has_default"],
)
def test_ExtractorConf_get_extract_method_parameters_raises_ExtractorBadDefinedError(
    mock_DATAS, fake_ecls, extract_method
):
    mock_DATAS()
    ecls = fake_ecls(extract_method=extract_method)
    with pytest.raises(ExtractorBadDefinedError):
        _ExtractorConf._get_extract_method_parameters(ecls)


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
def test_ExtractorConf_get_init_method_parameters(
    mock_DATAS, fake_ecls, init_method, expected
):
    mock_DATAS()
    ecls = fake_ecls(init_method=init_method)
    parameters = _ExtractorConf._get_init_method_parameters(ecls)
    np.testing.assert_equal(parameters, expected)


@pytest.mark.parametrize(
    "init_method",
    [lambda self, param1: None],
    ids=["missing_default"],
)
def test_ExtractorConf_get_init_method_parameters_raises_ExtractorBadDefinedError(
    mock_DATAS, fake_ecls, init_method
):
    mock_DATAS()
    ecls = fake_ecls(init_method=init_method)
    with pytest.raises(ExtractorBadDefinedError):
        _ExtractorConf._get_init_method_parameters(ecls)


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
    mock_DATAS,
    fake_ecls,
    features,
    init_method,
    extract_method,
    expected,
):
    mock_DATAS()
    ecls = fake_ecls(
        feature_names=features,
        init_method=init_method,
        extract_method=extract_method,
    )
    conf = _ExtractorConf.from_extractor_class(ecls)
    np.testing.assert_equal(conf, expected)


@pytest.mark.parametrize(
    ["required", "optional", "expected"],
    [
        (set(), set(), set()),
        ({"data1", "data2"}, set(), {"data1", "data2"}),
        (set(), {"data1", "data2"}, {"data1", "data2"}),
        ({"data1"}, {"data2", "data3"}, {"data1", "data2", "data3"}),
    ],
    ids=[
        "no_data",
        "required",
        "optional",
        "required_and_optional",
    ],
)
def test_ExtractorConf_data(
    mock_DATAS, fake_ecls, required, optional, expected
):
    mock_DATAS()
    extractor_conf = _ExtractorConf(
        required=required,
        optional=optional,
        features=set(),
        dependencies=set(),
        parameters={},
    )
    np.testing.assert_equal(extractor_conf.data, expected)


# =============================================================================
# EXTRACTOR TESTS
# =============================================================================


def test_Extractor_init_subclass(fake_extractor_conf_cls, mock_extractor_conf):
    extractor_conf_cls = fake_extractor_conf_cls(
        features=["feature1", "feature2"],
        required=["data1"],
        optional=["data2"],
        dependencies=["dependency1"],
        default_params={
            "parameter1": 1,
            "parameter2": 2,
            "parameter3": 3,
        },
    )
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self):
            pass

    np.testing.assert_equal(
        TestExtractor._conf, extractor_conf_cls.from_extractor_class()
    )


def test_Extractor_init_subclass_raises_ExtractorBadDefinedError():
    with pytest.raises(ExtractorBadDefinedError):

        class TestExtractorA(Extractor):
            def extract(self):
                pass

    with pytest.raises(ExtractorBadDefinedError):

        class TestExtractorB(Extractor):
            features = ["feature1"]


@pytest.mark.parametrize(
    ["method", "expected"],
    (
        ("get_features", {"feature1", "feature2"}),
        ("get_data", {"data1", "data2"}),
        ("get_required_data", {"data1"}),
        ("get_optional", {"data2"}),
        ("get_dependencies", {"dependency1"}),
        (
            "get_default_params",
            {
                "parameter1": 1,
                "parameter2": 2,
                "parameter3": 3,
            },
        ),
    ),
    ids=(
        "features",
        "data",
        "required_data",
        "optional",
        "dependencies",
        "default_params",
    ),
)
def test_Extractor_getters(
    fake_extractor_conf_cls, mock_extractor_conf, method, expected
):
    extractor_conf_cls = fake_extractor_conf_cls(
        features=["feature1", "feature2"],
        required=["data1"],
        optional=["data2"],
        dependencies=["dependency1"],
        default_params={
            "parameter1": 1,
            "parameter2": 2,
            "parameter3": 3,
        },
    )
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self):
            return None

    np.testing.assert_equal(getattr(TestExtractor, method)(), expected)


def test_Extractor_warnings(fake_extractor_conf_cls, mock_extractor_conf):
    extractor_conf_cls = fake_extractor_conf_cls(features=["feature1"])
    mock_extractor_conf(extractor_conf_cls)
    message = "Test warning message"

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self):
            pass

    extractor = TestExtractor()

    with pytest.warns(FeatureExtractionWarning, match=message):
        extractor.feature_warning(message)

    with pytest.warns(ExtractorWarning, match=message):
        extractor.extractor_warning(message)


def test_Extractor_select_kwargs(fake_extractor_conf_cls, mock_extractor_conf):
    extractor_conf_cls = fake_extractor_conf_cls(
        features=["feature1"],
        required=["data1", "data2"],
        optional=["data3", "data4"],
        dependencies=["dependency1", "dependency2"],
    )
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self):
            pass

    data = {f"data{i+1}": i + 1 for i in range(10)}
    dependencies = {f"dependency{i+1}": i + 11 for i in range(10)}
    kwargs = TestExtractor().select_kwargs(data, dependencies)

    np.testing.assert_equal(
        kwargs,
        {
            "data1": 1,
            "data2": 2,
            "data3": 3,
            "data4": 4,
            "dependency1": 11,
            "dependency2": 12,
        },
    )


def test_Extractor_select_kwargs_raises_KeyError(
    fake_extractor_conf_cls, mock_extractor_conf
):
    extractor_conf_cls = fake_extractor_conf_cls(
        features=["feature1"],
        required=["data1", "data2"],
        optional=["data3", "data4"],
        dependencies=["dependency1", "dependency2"],
    )
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self):
            pass

    dependencies = {"dependency1": 11, "dependency2": 12}

    # missing required data
    data = {"data3": 3, "data4": 4}
    with pytest.raises(KeyError):
        TestExtractor().select_kwargs(data, dependencies)

    # missing optional data
    data = {"data1": 1, "data2": 2}
    with pytest.raises(KeyError):
        TestExtractor().select_kwargs(data, dependencies)

    # missing dependencies
    data = {"data1": 1, "data2": 2, "data3": 3, "data4": 4}
    with pytest.raises(KeyError):
        TestExtractor().select_kwargs(data, {})


def test_Extractor_extract_and_validate(
    fake_extractor_conf_cls, mock_extractor_conf
):
    extractor_conf_cls = fake_extractor_conf_cls(
        features=["feature1"],
        required=["data1"],
        optional=["data2"],
        dependencies=["dependency1"],
    )
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self, data1, dependency1, data2=2):
            return {"feature1": data1 + dependency1 + data2}

    results = TestExtractor().extract_and_validate(
        {"data1": 1, "data2": 2, "dependency1": 3}
    )
    np.testing.assert_equal(results, {"feature1": 6})


def test_Extractor_extract_and_validate_raises_ExtractorContractError(
    fake_extractor_conf_cls,
    mock_extractor_conf,
):
    extractor_conf_cls = fake_extractor_conf_cls(
        features=["feature1"],
        required=["data1"],
        optional=["data2"],
        dependencies=["dependency1"],
    )
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self, data1, dependency1, data2=2):
            return {"feature2": data1 + dependency1 + data2}

    with pytest.raises(ExtractorContractError):
        TestExtractor().extract_and_validate(
            {"data1": 1, "data2": 2, "dependency1": 3}
        )


def test_Extractor_flatten_and_validate(
    fake_extractor_conf_cls,
    mock_extractor_conf,
):
    extractor_conf_cls = fake_extractor_conf_cls(
        features=["feature1"],
    )
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self, data1, dependency1, data2=2):
            return {"feature1": 1}

        def flatten_feature(self, feature, value):
            return {feature: value}

    np.testing.assert_equal(
        TestExtractor().flatten_and_validate("feature1", 1),
        {"feature1": 1},
    )


@pytest.mark.parametrize(
    "flatten_result",
    ["feature1", {("feature1", 1): 1}, {"feature1": [1, 2, 3]}],
    ids=["not_dict", "name_not_str", "value_not_scalar"],
)
def test_Extractor_flatten_and_validate_raises_ExtractorContractError(
    fake_extractor_conf_cls,
    mock_extractor_conf,
    flatten_result,
):
    extractor_conf_cls = fake_extractor_conf_cls(
        features=["feature1"],
    )
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self, data1, dependency1, data2=2):
            pass

        def flatten_feature(self, feature, value):
            return flatten_result

    with pytest.raises(ExtractorContractError):
        TestExtractor().flatten_and_validate("feature1", 1)


def test_Extractor_extract_default(
    fake_extractor_conf_cls, mock_extractor_conf
):
    extractor_conf_cls = fake_extractor_conf_cls(features=["feature1"])
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self):
            super().extract()

    with pytest.raises(NotImplementedError):
        TestExtractor().extract()


@pytest.mark.parametrize(
    ["raw_value", "expected"],
    [
        (1, {"feature1": 1}),
        ("string", {"feature1": "string"}),
        ((0, 1, 2), {"feature1_0": 0, "feature1_1": 1, "feature1_2": 2}),
        ([0, 1, 2], {"feature1_0": 0, "feature1_1": 1, "feature1_2": 2}),
        ({"key": "value"}, {"feature1_key": "value"}),
        (
            {"key": [0, 1, {"first": 2, "last": 3}]},
            {
                "feature1_key_0": 0,
                "feature1_key_1": 1,
                "feature1_key_2_first": 2,
                "feature1_key_2_last": 3,
            },
        ),
        (
            OrderedDict([("key1", 1), ("key2", 2)]),
            {"feature1_key1": 1, "feature1_key2": 2},
        ),
    ],
)
def test_Extractor_flatten_feature_default(
    fake_extractor_conf_cls, mock_extractor_conf, raw_value, expected
):
    extractor_conf_cls = fake_extractor_conf_cls(features=["feature1"])
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self):
            return {"feature1": raw_value}

    np.testing.assert_equal(
        TestExtractor().flatten_feature("feature1", raw_value), expected
    )


@pytest.mark.parametrize(
    "raw_value",
    [
        None,
        {"result1", "result2"},
        lambda x: x,
        (
            np.array([1, 2, 3]),
            {"feature1_0": 1, "feature1_1": 2, "feature1_2": 3},
        ),
    ],
    ids=["none", "set", "function", "numpy_array"],
)
def test_Extractor_flatten_feature_default_raises_ExtractorTransformError(
    fake_extractor_conf_cls, mock_extractor_conf, raw_value
):
    extractor_conf_cls = fake_extractor_conf_cls(features=["feature1"])
    mock_extractor_conf(extractor_conf_cls)

    class TestExtractor(Extractor):
        features = ["feature1"]

        def extract(self):
            return {"feature1": raw_value}

    with pytest.raises(ExtractorTransformError):
        TestExtractor().flatten_feature("feature1", raw_value)
