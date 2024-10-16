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

from feets.core import FeatureSpace, Features

import numpy as np

import pandas as pd

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

            def flatten_feature(self, feature, value):
                return {f"flat_{feature}": value}

        return FakeExtractor

    return maker


# =============================================================================
# FEATURES TESTS
# =============================================================================


@pytest.mark.parametrize(
    ["features_by_lc", "feature_names", "length"],
    [
        ([{"feature1": 1, "feature2": 2}], {"feature1", "feature2"}, 1),
        ([{"feature1": 1}] * 3, {"feature1"}, 3),
    ],
    ids=["simple", "multiple"],
)
def test_Features_init(
    fake_extractor_cls, features_by_lc, feature_names, length
):
    fake_extractors = [
        fake_extractor_cls(features=["feature1"])(),
        fake_extractor_cls(features=["feature2"])(),
    ]
    features = Features(features=features_by_lc, extractors=fake_extractors)

    np.testing.assert_equal(features.features, features_by_lc)
    np.testing.assert_equal(features.extractors, fake_extractors)
    np.testing.assert_equal(features.feature_names, feature_names)
    np.testing.assert_equal(features.length, length)


@pytest.mark.parametrize(
    ["features_by_lc", "feature_names", "length"],
    [
        ([{"feature1": 1, "feature2": 2}], {"feature1", "feature2"}, 1),
        ([{"feature1": 1}] * 3, {"feature1"}, 3),
    ],
    ids=["simple", "multiple"],
)
def test_Features_repr(
    fake_extractor_cls, features_by_lc, feature_names, length
):
    fake_extractors = [
        fake_extractor_cls(features=["feature1"])(),
        fake_extractor_cls(features=["feature2"])(),
    ]
    features = Features(features=features_by_lc, extractors=fake_extractors)

    np.testing.assert_equal(
        repr(features),
        f"Features({feature_names=}, {length=})",
    )


@pytest.mark.parametrize(
    ["features_by_lc", "feature_name", "expected"],
    [
        ([{"feature1": 1, "feature2": 2}], "feature1", [1]),
        ([{"feature1": 1}] * 3, "feature1", [1] * 3),
    ],
    ids=["simple", "multiple"],
)
def test_Features_getattr(
    fake_extractor_cls, features_by_lc, feature_name, expected
):
    fake_extractors = [
        fake_extractor_cls(features=["feature1"])(),
        fake_extractor_cls(features=["feature2"])(),
    ]
    features = Features(features=features_by_lc, extractors=fake_extractors)

    np.testing.assert_equal(getattr(features, feature_name), expected)


@pytest.mark.parametrize(
    ["features_by_lc", "slicer", "expected"],
    [
        ([{"feature1": 1, "feature2": 2}], 0, {"feature1": 1, "feature2": 2}),
        ([{"feature1": 1}] * 3, 0, {"feature1": 1}),
        (
            [{"feature1": 1}, {"feature1": 2}, {"feature1": 3}],
            slice(0, 2),
            [{"feature1": 1}, {"feature1": 2}],
        ),
    ],
    ids=["simple", "multiple", "multiple_slice"],
)
def test_Features_getitem(
    fake_extractor_cls, features_by_lc, slicer, expected
):
    fake_extractors = [
        fake_extractor_cls(features=["feature1"])(),
        fake_extractor_cls(features=["feature2"])(),
    ]
    features = Features(features=features_by_lc, extractors=fake_extractors)

    np.testing.assert_equal(features[slicer], expected)


@pytest.mark.parametrize(
    ["features_by_lc", "expected"],
    [
        ([{"feature1": 1, "feature2": 2}], 1),
        ([{"feature1": 1}] * 3, 3),
    ],
    ids=["simple", "multiple"],
)
def test_Features_len(fake_extractor_cls, features_by_lc, expected):
    fake_extractors = [
        fake_extractor_cls(features=["feature1"])(),
        fake_extractor_cls(features=["feature2"])(),
    ]
    features = Features(features=features_by_lc, extractors=fake_extractors)

    np.testing.assert_equal(len(features), expected)


@pytest.mark.parametrize(
    ["features_by_lc", "feature_names"],
    [
        ([{"feature1": 1, "feature2": 2}], ["feature1", "feature2"]),
        ([{"feature1": 1}] * 3, ["feature1"]),
    ],
    ids=["simple", "multiple"],
)
def test_Features_dir(fake_extractor_cls, features_by_lc, feature_names):
    fake_extractors = [
        fake_extractor_cls(features=["feature1"])(),
        fake_extractor_cls(features=["feature2"])(),
    ]
    features = Features(features=features_by_lc, extractors=fake_extractors)

    assert set(feature_names).issubset(dir(features))


@pytest.mark.parametrize(
    ["features_by_lc", "expected"],
    [
        (
            [{"feature1": 1, "feature2": 2}],
            pd.DataFrame({"flat_feature1": [1], "flat_feature2": [2]}),
        ),
        ([{"feature1": 1}] * 3, pd.DataFrame({"flat_feature1": [1] * 3})),
    ],
    ids=["simple", "multiple"],
)
def test_Features_as_frame(fake_extractor_cls, features_by_lc, expected):
    fake_extractors = [
        fake_extractor_cls(features=["feature1"])(),
        fake_extractor_cls(features=["feature2"])(),
    ]
    features = Features(features=features_by_lc, extractors=fake_extractors)
    expected.columns.name = "Features"

    pd.testing.assert_frame_equal(features.as_frame(), expected)


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
    mock_run([{"feature1": 1, "feature2": 2, "feature3": 3}])

    fs = FeatureSpace()

    features = fs.extract()

    expected = Features(
        features=([{"feature1": 1, "feature2": 2, "feature3": 3}]),
        extractors=fs._extractors,
    )

    assert features == expected


def test_FeatureSpace_extract_raises_ValueError(
    mock_extractor_registry, fake_extractor_cls, mock_run
):
    extractor_clss = [
        fake_extractor_cls(features=["feature1", "feature2", "feature3"])
    ]
    mock_extractor_registry(extractor_clss)

    fs = FeatureSpace()

    lc = {"data1": [1, 2, 3]}

    with pytest.raises(ValueError):
        fs.extract(lc, **lc)


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
