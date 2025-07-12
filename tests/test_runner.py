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
# CONSTANTS
# =============================================================================

DATA_1 = "test_data_1"
DATA_2 = "test_data_2"
DATA_3 = "test_data_3"

FEATURE_1 = "test_feature_1"
FEATURE_2 = "test_feature_2"
FEATURE_3 = "test_feature_3"
FEATURE_4 = "test_feature_4"

DASK_OPTIONS = {"scheduler": "synchronous"}

LCS_SINGLE = [{DATA_1: None, DATA_2: None}]
LCS_MULTIPLE = [
    {DATA_1: None, DATA_2: None},
    {DATA_1: None, DATA_2: None, DATA_3: None},
]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def Extractor_mocker(mocker):
    def maker(*, features=None):
        mock = mocker.Mock()
        features = features or []
        mock.get_features.return_value = frozenset(features)
        mock.extract.return_value = {feature: None for feature in features}
        return mock

    return maker


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    "lcs",
    [LCS_SINGLE, LCS_MULTIPLE],
)
def test_run_single_extractor(Extractor_mocker, lcs):
    extractor = Extractor_mocker(features={FEATURE_1, FEATURE_2, FEATURE_3})
    extractor.prepare_extract.return_value = {DATA_1: None, DATA_2: None}

    features = run(
        extractors=[extractor],
        selected_features=[FEATURE_1, FEATURE_2],
        required_data=[DATA_1, DATA_2],
        lcs=lcs,
        dask_options=DASK_OPTIONS,
    )

    extractor.extract.assert_called_with(**{DATA_1: None, DATA_2: None})
    extractor.validate_extract.assert_called_with(
        {FEATURE_1: None, FEATURE_2: None, FEATURE_3: None}
    )
    np.testing.assert_equal(
        features,
        [{FEATURE_1: None, FEATURE_2: None}] * len(lcs),
    )


@pytest.mark.parametrize(
    "lcs",
    [LCS_SINGLE, LCS_MULTIPLE],
)
def test_run_multiple_extractors(Extractor_mocker, lcs):
    extractor_1 = Extractor_mocker(features={FEATURE_1, FEATURE_3})
    extractor_1.prepare_extract.return_value = {DATA_1: None}

    extractor_2 = Extractor_mocker(features={FEATURE_2, FEATURE_4})
    extractor_2.prepare_extract.return_value = {DATA_2: None}

    features = run(
        extractors=[extractor_1, extractor_2],
        selected_features=[FEATURE_1, FEATURE_2],
        required_data=[DATA_1, DATA_2],
        lcs=lcs,
        dask_options=DASK_OPTIONS,
    )

    extractor_1.extract.assert_called_with(**{DATA_1: None})
    extractor_1.validate_extract.assert_called_with(
        {FEATURE_1: None, FEATURE_3: None}
    )

    extractor_2.extract.assert_called_with(**{DATA_2: None})
    extractor_2.validate_extract.assert_called_with(
        {FEATURE_2: None, FEATURE_4: None}
    )

    np.testing.assert_equal(
        features,
        [{FEATURE_1: None, FEATURE_2: None}] * len(lcs),
    )


@pytest.mark.parametrize(
    "lcs",
    [LCS_SINGLE, LCS_MULTIPLE],
)
def test_run_missing_data(lcs):
    with pytest.raises(DataRequiredError):
        run(
            extractors=None,
            selected_features=None,
            required_data=[DATA_3],
            lcs=lcs,
        )
