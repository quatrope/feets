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

from io import StringIO

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
            "feets.extractors.extractor_registry",
            FakeExtractorRegistry(extractors),
        )

    return maker


@pytest.fixture
def mock_available_data(mocker):
    def maker(data):
        mocker.patch("feets.extractors.DATAS", tuple(data))

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

            def to_dict(self):
                return {"FakeExtractor": {"kwargs": self.kawrgs}}

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


@pytest.mark.parametrize(
    ["lc", "expected_data"],
    [
        (
            {"data1": [1, 2, 3]},
            {"data1"},
        ),
        (
            {"data1": [1, 2, 3], "data2": [4, 5, 6]},
            {"data1", "data2"},
        ),
    ],
)
def test_FeatureSpace_from_lightcurves_single(
    mock_available_data,
    mocker,
    lc,
    expected_data,
):
    data = ["data1", "data2", "data3"]
    mock_available_data(data)

    def _fake_init(self, data):
        self.data = data

    mocker.patch.object(FeatureSpace, "__init__", _fake_init)

    fs = FeatureSpace.from_lightcurves(**lc)

    np.testing.assert_equal(fs.data, expected_data)


@pytest.mark.parametrize(
    ["lcs", "expected_data"],
    [
        (
            [
                {"data1": [1, 2, 3]},
                {"data1": [1, 2, 3], "data2": [4, 5, 6]},
            ],
            {"data1"},
        ),
        (
            [
                {"data1": [1, 2, 3], "data2": [4, 5, 6]},
                {"data2": [4, 5, 6], "data3": [7, 8, 9]},
            ],
            {"data2"},
        ),
        (
            [{"data1": [1, 2, 3]}, {"data2": [4, 5, 6]}, {"data3": [7, 8, 9]}],
            set(),
        ),
    ],
)
def test_FeatureSpace_from_lightcurves_multiple(
    mock_available_data,
    mocker,
    lcs,
    expected_data,
):
    data = ["data1", "data2", "data3"]
    mock_available_data(data)

    def _fake_init(self, data):
        self.data = data

    mocker.patch.object(FeatureSpace, "__init__", _fake_init)

    fs = FeatureSpace.from_lightcurves(*lcs)

    np.testing.assert_equal(fs.data, expected_data)


def test_FeatureSpace_from_lightcurves_raises_ValueError():
    lc = {"data1": [1, 2, 3]}
    lcs = [lc, lc, lc]

    with pytest.raises(ValueError):
        FeatureSpace.from_lightcurves(*lcs, **lc)


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


def test_FeatureSpace_to_dict(mock_extractor_registry, fake_extractor_cls):
    extractor_clss = [
        fake_extractor_cls(features=["feature1"], data=["data1"]),
        fake_extractor_cls(features=["feature2"], data=["data2"]),
    ]
    mock_extractor_registry(extractor_clss)

    fake_dask_options = {"key": "value"}

    fs = FeatureSpace(dask_options=fake_dask_options)

    expected = {
        "selected_features": list(fs._selected_features),
        "required_data": list(fs._required_data),
        "dask_options": fs._dask_options,
        "extractors": [ext.to_dict() for ext in fs._extractors],
    }

    np.testing.assert_equal(fs.to_dict(), expected)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (
            "to_json",
            [
                '{"selected_features": ["feature1", "feature2"], "required_data": ["data1", "data2"], "dask_options": {"key": "value"}, "extractors": [{"FakeExtractor": {"kwargs": {}}}, {"FakeExtractor": {"kwargs": {}}}]}',
                '{"selected_features": ["feature2", "feature1"], "required_data": ["data2", "data1"], "dask_options": {"key": "value"}, "extractors": [{"FakeExtractor": {"kwargs": {}}}, {"FakeExtractor": {"kwargs": {}}}]}',
                '{"selected_features": ["feature1", "feature2"], "required_data": ["data2", "data1"], "dask_options": {"key": "value"}, "extractors": [{"FakeExtractor": {"kwargs": {}}}, {"FakeExtractor": {"kwargs": {}}}]}',
                '{"selected_features": ["feature2", "feature1"], "required_data": ["data1", "data2"], "dask_options": {"key": "value"}, "extractors": [{"FakeExtractor": {"kwargs": {}}}, {"FakeExtractor": {"kwargs": {}}}]}',
            ],
        ),
        (
            "to_yaml",
            [
                "dask_options:\n  key: value\nextractors:\n- FakeExtractor:\n    kwargs: {}\n- FakeExtractor:\n    kwargs: {}\nrequired_data:\n- data1\n- data2\nselected_features:\n- feature1\n- feature2\n",
                "dask_options:\n  key: value\nextractors:\n- FakeExtractor:\n    kwargs: {}\n- FakeExtractor:\n    kwargs: {}\nrequired_data:\n- data2\n- data1\nselected_features:\n- feature2\n- feature1\n",
                "dask_options:\n  key: value\nextractors:\n- FakeExtractor:\n    kwargs: {}\n- FakeExtractor:\n    kwargs: {}\nrequired_data:\n- data2\n- data1\nselected_features:\n- feature1\n- feature2\n",
                "dask_options:\n  key: value\nextractors:\n- FakeExtractor:\n    kwargs: {}\n- FakeExtractor:\n    kwargs: {}\nrequired_data:\n- data1\n- data2\nselected_features:\n- feature2\n- feature1\n",
            ],
        ),
    ],
    ids=["json", "yaml"],
)
def test_FeatureSpace_persistence_string(
    mock_extractor_registry,
    fake_extractor_cls,
    method,
    expected,
):
    extractor_clss = [
        fake_extractor_cls(features=["feature1"], data=["data1"]),
        fake_extractor_cls(features=["feature2"], data=["data2"]),
    ]
    mock_extractor_registry(extractor_clss)

    fake_dask_options = {"key": "value"}

    fs = FeatureSpace(dask_options=fake_dask_options)

    assert getattr(fs, method)() in expected


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (
            "to_json",
            [
                '{"selected_features": ["feature1", "feature2"], "required_data": ["data1", "data2"], "dask_options": {"key": "value"}, "extractors": [{"FakeExtractor": {"kwargs": {}}}, {"FakeExtractor": {"kwargs": {}}}]}',
                '{"selected_features": ["feature2", "feature1"], "required_data": ["data2", "data1"], "dask_options": {"key": "value"}, "extractors": [{"FakeExtractor": {"kwargs": {}}}, {"FakeExtractor": {"kwargs": {}}}]}',
                '{"selected_features": ["feature1", "feature2"], "required_data": ["data2", "data1"], "dask_options": {"key": "value"}, "extractors": [{"FakeExtractor": {"kwargs": {}}}, {"FakeExtractor": {"kwargs": {}}}]}',
                '{"selected_features": ["feature2", "feature1"], "required_data": ["data1", "data2"], "dask_options": {"key": "value"}, "extractors": [{"FakeExtractor": {"kwargs": {}}}, {"FakeExtractor": {"kwargs": {}}}]}',
            ],
        ),
        (
            "to_yaml",
            [
                "dask_options:\n  key: value\nextractors:\n- FakeExtractor:\n    kwargs: {}\n- FakeExtractor:\n    kwargs: {}\nrequired_data:\n- data1\n- data2\nselected_features:\n- feature1\n- feature2\n",
                "dask_options:\n  key: value\nextractors:\n- FakeExtractor:\n    kwargs: {}\n- FakeExtractor:\n    kwargs: {}\nrequired_data:\n- data2\n- data1\nselected_features:\n- feature2\n- feature1\n",
                "dask_options:\n  key: value\nextractors:\n- FakeExtractor:\n    kwargs: {}\n- FakeExtractor:\n    kwargs: {}\nrequired_data:\n- data2\n- data1\nselected_features:\n- feature1\n- feature2\n",
                "dask_options:\n  key: value\nextractors:\n- FakeExtractor:\n    kwargs: {}\n- FakeExtractor:\n    kwargs: {}\nrequired_data:\n- data1\n- data2\nselected_features:\n- feature2\n- feature1\n",
            ],
        ),
    ],
    ids=["json", "yaml"],
)
def test_FeatureSpace_persistence_file(
    mock_extractor_registry,
    fake_extractor_cls,
    method,
    expected,
):
    extractor_clss = [
        fake_extractor_cls(features=["feature1"], data=["data1"]),
        fake_extractor_cls(features=["feature2"], data=["data2"]),
    ]
    mock_extractor_registry(extractor_clss)

    fake_dask_options = {"key": "value"}

    fs = FeatureSpace(dask_options=fake_dask_options)

    outfile = StringIO()
    getattr(fs, method)(stream_or_buff=outfile)
    outfile.seek(0)
    content = outfile.read()

    assert content in expected


@pytest.mark.parametrize(
    "method",
    ["to_json", "to_yaml"],
    ids=["json", "yaml"],
)
def test_FeatureSpace_persistence_path(
    mock_extractor_registry,
    fake_extractor_cls,
    mocker,
    method,
):
    extractor_clss = [
        fake_extractor_cls(features=["feature1"], data=["data1"]),
        fake_extractor_cls(features=["feature2"], data=["data2"]),
    ]
    mock_extractor_registry(extractor_clss)

    fake_dask_options = {"key": "value"}

    fs = FeatureSpace(dask_options=fake_dask_options)

    fake_path = "output"
    open_mock = mocker.mock_open()

    mocker.patch("feets.core.open", open_mock, create=True)
    getattr(fs, method)(stream_or_buff=fake_path)

    open_mock.assert_called_with("output", "w")


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
