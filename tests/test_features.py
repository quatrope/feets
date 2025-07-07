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

from feets.core import Features

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

FEATURE_1 = "test_feature_1"
FEATURE_2 = "test_feature_2"
FEATURE_3 = "test_feature_3"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def features():
    return [
        {FEATURE_1: 1, FEATURE_2: 2},
        {FEATURE_1: 10, FEATURE_2: 20},
        {FEATURE_1: 100, FEATURE_2: 200},
    ]


@pytest.fixture
def extractor_mock(mocker):
    def maker(feature):
        extractor = mocker.Mock()
        extractor.get_features.return_value = frozenset({feature})
        extractor.flatten_feature = lambda x, y: {x: y}
        return extractor

    return maker


@pytest.fixture
def extractors(extractor_mock):
    return [extractor_mock(FEATURE_1), extractor_mock(FEATURE_2)]


@pytest.fixture
def results(features, extractors):
    return Features(features=features, extractors=extractors)


@pytest.fixture
def joblib_mock(mocker):
    mocker.patch("joblib.cpu_count", return_value=1)
    mocker.patch("joblib.delayed", lambda x: x)
    Parallel = mocker.patch("joblib.Parallel")
    Parallel.return_value.__enter__.return_value = lambda x: x
    return Parallel


# =============================================================================
# TESTS
# =============================================================================


def test_Features_init(results, features, extractors):
    assert isinstance(results, Features)

    np.testing.assert_equal(results.features, features)
    np.testing.assert_equal(results.extractors, extractors)
    np.testing.assert_equal(results.feature_names, {FEATURE_1, FEATURE_2})
    np.testing.assert_equal(results.length, len(features))


def test_Features_repr(results, features):
    message = (
        f"<Features feature_names={set({FEATURE_1, FEATURE_2})}, "
        f"length={len(features)}>"
    )
    np.testing.assert_equal(repr(results), message)


def test_Features_getattr(results):
    np.testing.assert_equal(getattr(results, FEATURE_1), [1, 10, 100])
    np.testing.assert_equal(getattr(results, FEATURE_2), [2, 20, 200])


def test_Features_getattr_invalid_feature(results):
    message = f"'Features' object has no feature by the name {FEATURE_3!r}"
    with pytest.raises(AttributeError, match=message):
        getattr(results, FEATURE_3)


def test_Features_getitem(results):
    np.testing.assert_equal(results[0], {FEATURE_1: 1, FEATURE_2: 2})
    np.testing.assert_equal(results[1], {FEATURE_1: 10, FEATURE_2: 20})
    np.testing.assert_equal(results[2], {FEATURE_1: 100, FEATURE_2: 200})


def test_Features_getitem_slice(results):
    np.testing.assert_equal(
        results[0:2],
        [{FEATURE_1: 1, FEATURE_2: 2}, {FEATURE_1: 10, FEATURE_2: 20}],
    )
    np.testing.assert_equal(
        results[:],
        [
            {FEATURE_1: 1, FEATURE_2: 2},
            {FEATURE_1: 10, FEATURE_2: 20},
            {FEATURE_1: 100, FEATURE_2: 200},
        ],
    )


def test_Features_getitem_out_of_bounds(results, features):
    message = (
        f"index 3 is out of bounds for 'Features' object with length "
        f"{len(features)}"
    )
    with pytest.raises(IndexError, match=message):
        results[3]


def test_Features_getitem_len(results, features):
    np.testing.assert_equal(len(results), len(features))


def test_Features_dir(results):
    attributes = set(dir(results))
    assert FEATURE_1 in attributes
    assert FEATURE_2 in attributes
    assert FEATURE_3 not in attributes


def test_Features_as_frame(joblib_mock, results):
    df = results.as_frame()

    assert isinstance(df, pd.DataFrame)

    np.testing.assert_equal(
        df.to_dict(orient="list"),
        {FEATURE_1: [1, 10, 100], FEATURE_2: [2, 20, 200]},
    )


@pytest.mark.parametrize(
    ["kwargs", "expected_kwargs"],
    [
        ({}, {"prefer": "processes", "n_jobs": 1}),
        ({"prefer": "threads"}, {"prefer": "threads", "n_jobs": 1}),
        (
            {"n_jobs": 2, "kwarg_1": "value_1"},
            {"prefer": "processes", "n_jobs": 2, "kwarg_1": "value_1"},
        ),
        (
            {
                "kwarg_1": "value_1",
                "kwarg_2": "value_2",
                "kwarg_3": "value_3",
            },
            {
                "prefer": "processes",
                "n_jobs": 1,
                "kwarg_1": "value_1",
                "kwarg_2": "value_2",
                "kwarg_3": "value_3",
            },
        ),
    ],
)
def test_Features_as_frame_kwargs(
    joblib_mock, results, kwargs, expected_kwargs
):
    df = results.as_frame(**kwargs)

    assert isinstance(df, pd.DataFrame)

    joblib_mock.assert_called_once_with(**expected_kwargs)
