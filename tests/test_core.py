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

from feets.core import FeatureSpace
from feets.extractors.registry import RegistryError

import numpy as np

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_1 = "test_data_1"
DATA_2 = "test_data_2"
DATA_3 = "test_data_3"
DATA_4 = "test_data_4"
ALL_DATA = (DATA_1, DATA_2, DATA_3, DATA_4)

FEATURE_1 = "test_feature_1"
FEATURE_2 = "test_feature_2"
FEATURE_3 = "test_feature_3"

PARAM_1 = "test_param_1"
PARAM_2 = "test_param_2"
PARAM_3 = "test_param_3"

DASK_OPTION_1 = "test_dask_option_1"
DASK_OPTION_2 = "test_dask_option_2"
DASK_OPTIONS = {DASK_OPTION_1: 1, DASK_OPTION_2: 2}


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def Extractor_mocker(mocker):
    def maker(
        *,
        name="Extractor",
        features=set(),
        required_data=set(),
        params=dict(),
    ):
        mock = mocker.Mock()
        mock.__name__ = name
        mock.get_features.return_value = frozenset(features)
        mock.get_required_data.return_value = frozenset(required_data)
        mock.return_value.params.return_value = dict(params)
        mock.return_value.to_dict.return_value = {name: params}

        return mock

    return maker


@pytest.fixture
def get_execution_plan_mock(mocker):
    return mocker.patch(
        "feets.extractors.extractor_registry.get_execution_plan",
        return_value=[],
    )


@pytest.fixture
def run_mock(mocker):
    return mocker.patch("feets.core.run", autospec=True)


@pytest.fixture
def Features_mock(mocker):
    return mocker.patch("feets.core.Features", autospec=True)


@pytest.fixture
def store_json_mock(mocker):
    return mocker.patch("feets.io.store_json", autospec=True)


@pytest.fixture
def store_yaml_mock(mocker):
    return mocker.patch("feets.io.store_yaml", autospec=True)


@pytest.fixture(scope="module")
def DATAS_mock(module_mocker):
    return module_mocker.patch("feets.core.DATAS", ALL_DATA)


@pytest.fixture
def fs(mocker, Extractor_mocker, get_execution_plan_mock):
    Extractor1 = Extractor_mocker(
        name="Extractor1",
        features={FEATURE_1},
        required_data={DATA_1, DATA_2},
        params={PARAM_1: 1, PARAM_2: 2},
    )
    Extractor2 = Extractor_mocker(
        name="Extractor2",
        features={FEATURE_2, FEATURE_3},
        required_data={DATA_2, DATA_3},
        params={PARAM_1: 2, PARAM_3: 3},
    )
    get_execution_plan_mock.return_value = [Extractor1, Extractor2]

    fs = FeatureSpace(
        Extractor1={PARAM_1: 1},
        Extractor2={PARAM_3: 3},
        dask_options=DASK_OPTIONS,
    )

    return fs


# =============================================================================
# TESTS
# =============================================================================


def test_FeatureSpace_init(Extractor_mocker, get_execution_plan_mock):
    Extractor1 = Extractor_mocker(
        features={FEATURE_1, FEATURE_2},
        required_data={DATA_1},
    )
    Extractor2 = Extractor_mocker(
        features={FEATURE_3},
        required_data={DATA_2, DATA_3},
    )
    get_execution_plan_mock.return_value = [Extractor1, Extractor2]

    fs = FeatureSpace(dask_options=DASK_OPTIONS)

    np.testing.assert_equal(
        fs.extractors,
        [Extractor1.return_value, Extractor2.return_value],
    )
    np.testing.assert_equal(
        fs.selected_features, {FEATURE_1, FEATURE_2, FEATURE_3}
    )
    np.testing.assert_equal(fs.required_data, {DATA_1, DATA_2, DATA_3})
    np.testing.assert_equal(fs.dask_options, DASK_OPTIONS)


def test_FeatureSpace_init_filters(get_execution_plan_mock):
    get_execution_plan_mock.return_value = []

    filters = {
        "data": {DATA_1, DATA_2, DATA_3},
        "only": {FEATURE_1, FEATURE_2},
        "exclude": {FEATURE_3},
    }

    fs = FeatureSpace(**filters)

    get_execution_plan_mock.assert_called_once_with(**filters)


def test_FeatureSpace_init_conflicting_filters(get_execution_plan_mock):
    get_execution_plan_mock.side_effect = RegistryError()

    with pytest.raises(ValueError):
        FeatureSpace(only={FEATURE_1}, exclude={FEATURE_1})


def test_FeatureSpace_init_kwargs(Extractor_mocker, get_execution_plan_mock):
    Extractor1 = Extractor_mocker(name="Extractor1")
    Extractor2 = Extractor_mocker(name="Extractor2")
    get_execution_plan_mock.return_value = [Extractor1, Extractor2]

    fs = FeatureSpace(
        Extractor1={PARAM_1: 1, PARAM_2: 2},
        Extractor2={PARAM_3: 3},
    )

    np.testing.assert_equal(
        fs.extractors,
        [Extractor1.return_value, Extractor2.return_value],
    )

    Extractor1.assert_called_once_with(**{PARAM_1: 1, PARAM_2: 2})
    Extractor2.assert_called_once_with(**{PARAM_3: 3})


def test_FeatureSpace_from_lightcurves(DATAS_mock, get_execution_plan_mock):
    lc1 = {DATA_1: 1, DATA_2: 2, DATA_3: 3}
    lc2 = {DATA_2: 2, DATA_3: 3, DATA_4: 4}

    fs = FeatureSpace.from_lightcurves(lc1, lc2)

    get_execution_plan_mock.assert_called_once_with(
        data={DATA_2, DATA_3}, only=None, exclude=None
    )


def test_FeatureSpace_from_lightcurves_empty(
    DATAS_mock, get_execution_plan_mock
):
    fs = FeatureSpace.from_lightcurves()

    get_execution_plan_mock.assert_called_once_with(
        data=set(ALL_DATA), only=None, exclude=None
    )


def test_FeatureSpace_from_lightcurves_disjoint(
    DATAS_mock, get_execution_plan_mock
):
    lc1 = {DATA_1: 1, DATA_2: 2}
    lc2 = {DATA_3: 3, DATA_4: 4}

    with pytest.raises(ValueError):
        FeatureSpace.from_lightcurves(lc1, lc2)


def test_FeatureSpace_from_lightcurve(DATAS_mock, get_execution_plan_mock):
    lc = {DATA_1: 1, DATA_2: 2, DATA_3: 3}

    fs = FeatureSpace.from_lightcurve(**lc)

    get_execution_plan_mock.assert_called_once_with(
        data={DATA_1, DATA_2, DATA_3}, only=None, exclude=None
    )


def test_FeatureSpace_repr(mocker, fs):
    [extractor_1, extractor_2] = fs.extractors
    np.testing.assert_equal(
        repr(fs),
        f"<FeatureSpace: {repr(extractor_1)}, {repr(extractor_2)}>",
    )


def test_FeatureSpace_from_dict(
    Extractor_mocker, DATAS_mock, get_execution_plan_mock
):
    Extractor1 = Extractor_mocker(
        name="Extractor1",
        features={FEATURE_1},
        required_data={DATA_1, DATA_2},
    )
    Extractor2 = Extractor_mocker(
        name="Extractor2",
        features={FEATURE_2, FEATURE_3},
        required_data={DATA_2, DATA_3},
    )
    get_execution_plan_mock.return_value = [Extractor1, Extractor2]

    fs_dict = {
        "dask_options": DASK_OPTIONS,
        "extractors": [
            {"Extractor1": {PARAM_1: 1, PARAM_2: 2}},
            {"Extractor2": {PARAM_3: 3}},
        ],
        "required_data": {DATA_1, DATA_2, DATA_3},
        "selected_features": {FEATURE_1, FEATURE_2, FEATURE_3},
    }

    fs = FeatureSpace.from_dict(fs_dict)

    np.testing.assert_equal(fs.dask_options, DASK_OPTIONS)
    np.testing.assert_equal(
        fs.selected_features,
        {FEATURE_1, FEATURE_2, FEATURE_3},
    )
    np.testing.assert_equal(
        fs.extractors,
        [Extractor1.return_value, Extractor2.return_value],
    )

    get_execution_plan_mock.assert_called_once_with(
        data=None, only={FEATURE_1, FEATURE_2, FEATURE_3}, exclude=None
    )
    Extractor1.assert_called_once_with(**{PARAM_1: 1, PARAM_2: 2})
    Extractor2.assert_called_once_with(**{PARAM_3: 3})


@pytest.mark.parametrize(
    "fs_dict",
    [
        {
            "dask_options": DASK_OPTIONS,
            "selected_features": {FEATURE_1},
        },
        {
            "extractors": [],
            "selected_features": {FEATURE_1},
        },
        {
            "dask_options": DASK_OPTIONS,
            "extractors": [],
        },
    ],
)
def test_FeatureSpace_from_dict_missing_keys(
    DATAS_mock, get_execution_plan_mock, fs_dict
):
    with pytest.raises(KeyError):
        FeatureSpace.from_dict(fs_dict)


def test_FeatureSpace_to_dict(fs):
    fs_dict = fs.to_dict()

    np.testing.assert_equal(fs_dict["dask_options"], DASK_OPTIONS)

    np.testing.assert_equal(
        fs_dict["extractors"],
        [
            {"Extractor1": {PARAM_1: 1, PARAM_2: 2}},
            {"Extractor2": {PARAM_1: 2, PARAM_3: 3}},
        ],
    )
    np.testing.assert_equal(fs_dict["required_data"], {DATA_1, DATA_2, DATA_3})
    np.testing.assert_equal(
        fs_dict["selected_features"], {FEATURE_1, FEATURE_2, FEATURE_3}
    )


def test_FeatureSpace_to_json(fs, store_json_mock):
    store_json_mock.return_value = "FeatureSpace_JSON"

    fs_json = fs.to_json()

    np.testing.assert_equal(fs_json, "FeatureSpace_JSON")


def test_FeatureSpace_to_yaml(fs, store_yaml_mock):
    store_yaml_mock.return_value = "FeatureSpace_YAML"

    fs_yaml = fs.to_yaml()

    np.testing.assert_equal(fs_yaml, "FeatureSpace_YAML")


def test_FeatureSpace_extract_many(fs, run_mock, Features_mock):
    lc1 = {DATA_1: 1, DATA_2: 2, DATA_3: 3}
    lc2 = {DATA_2: 2, DATA_3: 3, DATA_4: 4}

    result = fs.extract_many(lc1, lc2)

    run_mock.assert_called_once_with(
        extractors=fs.extractors,
        selected_features=fs.selected_features,
        required_data=fs.required_data,
        dask_options=fs.dask_options,
        lcs=[lc1, lc2],
    )

    Features_mock.assert_called_once_with(
        features=run_mock.return_value, extractors=fs.extractors
    )

    np.testing.assert_equal(result, Features_mock.return_value)


def test_FeatureSpace_extract(fs, run_mock, Features_mock):
    lc = {DATA_1: 1, DATA_2: 2, DATA_3: 3}

    result = fs.extract(**lc)

    run_mock.assert_called_once_with(
        extractors=fs.extractors,
        selected_features=fs.selected_features,
        required_data=fs.required_data,
        dask_options=fs.dask_options,
        lcs=[lc],
    )

    Features_mock.assert_called_once_with(
        features=run_mock.return_value, extractors=fs.extractors
    )

    np.testing.assert_equal(result, Features_mock.return_value)
