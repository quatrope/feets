from inspect import Parameter

from feets.extractors.extractor import ExtractorBadDefinedError, _ExtractorConf

import numpy as np
from numpy.testing import assert_raises

import pytest

# =============================================================================
# EXTRACTOR CONF
# =============================================================================


@pytest.fixture
def ExtractorConf(mocker):
    mocker.patch(
        "feets.extractors.extractor.DATAS",
        ("valid_data_1", "valid_data_2", "valid_data_3"),
    )
    return _ExtractorConf


@pytest.fixture
def extractor_class(mocker):
    mocker.patch(
        "feets.extractors.extractor.DATAS",
        ("valid_data_1", "valid_data_2", "valid_data_3"),
    )

    def maker(*, features=None, init=None, extract=None):
        return type(
            "MyExtractor",
            (object,),
            {"features": features or [], "__init__": init, "extract": extract},
        )

    return maker


def function_empty(self):
    pass


def init_multiple(self, parameter_1=123, parameter_2=456, parameter_3=789):
    pass


def extract_multiple(
    self, valid_data_1, valid_data_2, feature_A, feature_B, valid_data_3=123
):
    pass


@pytest.mark.parametrize(
    "feature, features, exception",
    [
        (123, set(), ExtractorBadDefinedError),
        ("valid_data_1", set(), ExtractorBadDefinedError),
        ("duplicate_feature", {"duplicate_feature"}, ExtractorBadDefinedError),
    ],
    ids=[
        "error_non_string_feature",
        "error_feature_in_DATAS",
        "error_duplicate_feature",
    ],
)
def test_ExtractorConf_validate_and_add_feature_error(
    feature, features, exception, ExtractorConf
):
    features_attr = "MyExtractor.features"
    with assert_raises(exception):
        ExtractorConf._validate_and_add_feature(
            feature, features, features_attr
        )


@pytest.mark.parametrize(
    "feature, features, expected",
    [
        ("new_feature", set(), {"new_feature"}),
        (
            "new_feature",
            {"existing_feature"},
            {"existing_feature", "new_feature"},
        ),
    ],
    ids=[
        "success_new_feature",
        "success_existing_feature",
    ],
)
def test_ExtractorConf_validate_and_add_feature_success(
    feature, features, expected, ExtractorConf
):
    features_attr = "MyExtractor.features"
    ExtractorConf._validate_and_add_feature(feature, features, features_attr)
    np.testing.assert_equal(features, expected)


@pytest.mark.parametrize(
    "param, exception",
    [
        (
            Parameter(
                "dependency",
                Parameter.POSITIONAL_OR_KEYWORD,
                default=123,
            ),
            ExtractorBadDefinedError,
        ),
    ],
    ids=[
        "error_dependency_has_default",
    ],
)
def test_ExtractorConf_validate_and_add_extract_param_error(
    param,
    exception,
    ExtractorConf,
):
    required, optional, dependencies = set(), set(), set()
    ecls_name = "MyExtractor"

    with assert_raises(exception):
        ExtractorConf._validate_and_add_extract_param(
            param,
            required,
            optional,
            dependencies,
            ecls_name,
        )


@pytest.mark.parametrize(
    "param, expected",
    [
        (
            Parameter("valid_data_1", Parameter.POSITIONAL_OR_KEYWORD),
            ({"valid_data_1"}, set(), set()),
        ),
        (
            Parameter(
                "valid_data_1", Parameter.POSITIONAL_OR_KEYWORD, default=123
            ),
            (set(), {"valid_data_1"}, set()),
        ),
        (
            Parameter("dependency", Parameter.POSITIONAL_OR_KEYWORD),
            (set(), set(), {"dependency"}),
        ),
    ],
    ids=[
        "success_new_required",
        "success_new_optional",
        "success_new_dependency",
    ],
)
def test_ExtractorConf_validate_and_add_extract_param_success(
    param,
    expected,
    ExtractorConf,
):
    required, optional, dependencies = set(), set(), set()
    ecls_name = "MyExtractor"

    ExtractorConf._validate_and_add_extract_param(
        param,
        required,
        optional,
        dependencies,
        ecls_name,
    )

    np.testing.assert_equal((required, optional, dependencies), expected)


@pytest.mark.parametrize(
    "param, exception",
    [
        (
            Parameter("valid_data_1", Parameter.POSITIONAL_OR_KEYWORD),
            ExtractorBadDefinedError,
        ),
    ],
    ids=[
        "error_missing_default",
    ],
)
def test_ExtractorConf_validate_and_add_init_param_error(
    param,
    exception,
    ExtractorConf,
):
    parameters = set()
    ecls_name = "MyExtractor"

    with assert_raises(exception):
        ExtractorConf._validate_and_add_init_param(
            param,
            parameters,
            ecls_name,
        )


@pytest.mark.parametrize(
    "param, expected",
    [
        (
            Parameter(
                "parameter", Parameter.POSITIONAL_OR_KEYWORD, default=123
            ),
            {"parameter": 123},
        ),
    ],
    ids=[
        "success_new_param",
    ],
)
def test_ExtractorConf_validate_and_add_init_param_success(
    param, expected, ExtractorConf
):
    parameters = {}
    ecls_name = "MyExtractor"

    ExtractorConf._validate_and_add_init_param(
        param,
        parameters,
        ecls_name,
    )

    np.testing.assert_equal(parameters, expected)


@pytest.mark.parametrize(
    "features, exception",
    [
        ([], ExtractorBadDefinedError),
    ],
    ids=[
        "error_empty_feature_list",
    ],
)
def test_ExtractorConf_get_feature_conf_error(
    features,
    exception,
    ExtractorConf,
    extractor_class,
):
    ecls = extractor_class(features=features)
    with assert_raises(exception):
        ExtractorConf._get_feature_conf(ecls)


@pytest.mark.parametrize(
    "features, expected",
    [
        (
            ["feature_1", "feature_2", "feature_3"],
            {"feature_1", "feature_2", "feature_3"},
        ),
    ],
    ids=[
        "success_full_feature_list",
    ],
)
def test_ExtractorConf_get_feature_conf_success(
    features,
    expected,
    ExtractorConf,
    extractor_class,
):
    ecls = extractor_class(features=features)
    feature_conf = ExtractorConf._get_feature_conf(ecls)
    np.testing.assert_equal(feature_conf, expected)


@pytest.mark.parametrize(
    "extract, expected",
    [
        (function_empty, (set(), set(), set())),
        (
            lambda self, valid_data_1, valid_data_2: None,
            ({"valid_data_1", "valid_data_2"}, set(), set()),
        ),
        (
            lambda self, valid_data_1=123, valid_data_2=123: None,
            (set(), {"valid_data_1", "valid_data_2"}, set()),
        ),
        (
            lambda self, feature_1, feature_2: None,
            (set(), set(), {"feature_1", "feature_2"}),
        ),
        (
            lambda self, valid_data_1, feature_1, valid_data_2=123: None,
            ({"valid_data_1"}, {"valid_data_2"}, {"feature_1"}),
        ),
    ],
    ids=[
        "success_no_parameters",
        "success_required",
        "success_optional",
        "success_dependencies",
        "success_all",
    ],
)
def test_ExtractorConf_get_extract_method_parameters_success(
    extract,
    expected,
    ExtractorConf,
    extractor_class,
):
    ecls = extractor_class(extract=extract)
    required, optional, dependencies = (
        ExtractorConf._get_extract_method_parameters(ecls)
    )
    np.testing.assert_equal((required, optional, dependencies), expected)


@pytest.mark.parametrize(
    "init, expected",
    [
        (lambda self: None, {}),
        (
            lambda self, parameter_1=123, parameter_2=456: None,
            {"parameter_1": 123, "parameter_2": 456},
        ),
    ],
    ids=[
        "success_no_parameters",
        "success_multiple_parameters",
    ],
)
def test_ExtractorConf_get_init_method_parameters_success(
    init,
    expected,
    ExtractorConf,
    extractor_class,
):
    ecls = extractor_class(init=init)
    parameters = ExtractorConf._get_init_method_parameters(ecls)
    np.testing.assert_equal(parameters, expected)


@pytest.mark.parametrize(
    "features, init, extract, expected_kwargs",
    [
        (
            ["feature_1", "feature_2"],
            function_empty,
            function_empty,
            {
                "features": {"feature_1", "feature_2"},
                "required": set(),
                "optional": set(),
                "dependencies": set(),
                "parameters": {},
            },
        ),
        (
            ["feature_1", "feature_2"],
            init_multiple,
            function_empty,
            {
                "features": {"feature_1", "feature_2"},
                "required": set(),
                "optional": set(),
                "dependencies": set(),
                "parameters": {
                    "parameter_1": 123,
                    "parameter_2": 456,
                    "parameter_3": 789,
                },
            },
        ),
        (
            ["feature_1", "feature_2"],
            function_empty,
            extract_multiple,
            {
                "features": {"feature_1", "feature_2"},
                "required": {"valid_data_1", "valid_data_2"},
                "optional": {"valid_data_3"},
                "dependencies": {"feature_A", "feature_B"},
                "parameters": {},
            },
        ),
        (
            ["feature_1", "feature_2"],
            init_multiple,
            extract_multiple,
            {
                "features": {"feature_1", "feature_2"},
                "required": {"valid_data_1", "valid_data_2"},
                "optional": {"valid_data_3"},
                "dependencies": {"feature_A", "feature_B"},
                "parameters": {
                    "parameter_1": 123,
                    "parameter_2": 456,
                    "parameter_3": 789,
                },
            },
        ),
    ],
    ids=[
        "success_no_parameters",
        "success_init",
        "success_extract",
        "success_init_extract",
    ],
)
def test_ExtractorConf_from_extractor_class(
    features,
    init,
    extract,
    expected_kwargs,
    ExtractorConf,
    extractor_class,
):
    ecls = extractor_class(features=features, init=init, extract=extract)
    conf = ExtractorConf.from_extractor_class(ecls)
    expected = ExtractorConf(**expected_kwargs)
    np.testing.assert_equal(conf, expected)
