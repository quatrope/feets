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

import datetime as dt
from io import StringIO

from feets.core import FeatureSpace
from feets.io import (
    CustomJSONEncoder,
    none_open_or_buffer,
    read_json,
    read_yaml,
    store_json,
    store_yaml,
)

import numpy as np

import pytest

# =============================================================================
# MOCKS AND FIXTURES FOR TESTING
# =============================================================================


class MockWriter:
    """Collect all written data."""

    def __init__(self):
        self.contents = ""

    def write(self, data):
        self.contents += data


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ["data", "expected"],
    [
        ((1, 2, 3), [1, 2, 3]),
        ({1, 2, 3}, [1, 2, 3]),
        (frozenset([1, 2, 3]), [1, 2, 3]),
        (
            dt.datetime(2024, 11, 6, 0, 41, 57, 812214),
            dt.datetime(2024, 11, 6, 0, 41, 57, 812214).isoformat(),
        ),
        (np.int32(42), int(42)),
        (np.float32(3.14), float(np.float32(3.14))),
        (np.complex64(1 + 2j), complex(1, 2)),
        (np.bool_(True), True),
        (np.array([1, 2, 3]), [1, 2, 3]),
    ],
    ids=[
        "tuple",
        "set",
        "frozenset",
        "datetime",
        "numpy_int",
        "numpy_float",
        "numpy_complex",
        "numpy_bool",
        "numpy_array",
    ],
)
def test_CustomJSONEncoder_default(data, expected):
    encoder = CustomJSONEncoder()
    np.testing.assert_equal(encoder.default(data), expected)


def test_CustomJSONEncoder_default_raises_TypeError():
    encoder = CustomJSONEncoder()
    with pytest.raises(TypeError):
        encoder.default(object())


def test_none_open_or_buffer_none():
    with none_open_or_buffer(None, "w") as buffer:
        assert isinstance(buffer, StringIO)
        buffer.write("test")
        np.testing.assert_equal(buffer.getvalue(), "test")


def test_none_open_or_buffer_path(mocker):
    open_mock = mocker.mock_open()
    mocker.patch("feets.io.open", open_mock)
    with none_open_or_buffer("output", "w") as buffer:
        assert buffer is open_mock()
        buffer.write("test")
        open_mock().write.assert_called_with("test")


def test_none_open_or_buffer_file_like():
    file_like = StringIO()
    with none_open_or_buffer(file_like, "w") as buffer:
        buffer.write("test")
        np.testing.assert_equal(buffer.getvalue(), "test")


def test_store_json_to_string(mocker):
    fspace = mocker.Mock(spec=FeatureSpace)
    fspace.to_dict.return_value = {"feature": {"key": "value"}}

    result = store_json(fspace)
    expected = '{\n  "feature": {\n    "key": "value"\n  }\n}'

    np.testing.assert_equal(result, expected)


def test_store_json_to_file(mocker):
    fspace = mocker.Mock(spec=FeatureSpace)
    fspace.to_dict.return_value = {"feature": {"key": "value"}}

    open_mock = mocker.mock_open()
    mocker.patch("feets.io.open", open_mock)

    writer = MockWriter()
    open_mock.return_value.write = writer.write

    store_json(fspace, "output.json")
    expected = '{\n  "feature": {\n    "key": "value"\n  }\n}'

    open_mock.assert_called_once_with("output.json", "w")
    np.testing.assert_equal(writer.contents, expected)


def test_store_json_with_kwargs(mocker):
    fspace = mocker.Mock(spec=FeatureSpace)
    fspace.to_dict.return_value = {"feature": {"key": "value"}}

    result = store_json(fspace, indent=4)
    expected = '{\n    "feature": {\n        "key": "value"\n    }\n}'

    np.testing.assert_equal(result, expected)


def test_store_yaml_to_string(mocker):
    fspace = mocker.Mock(spec=FeatureSpace)
    fspace.to_dict.return_value = {"feature": {"key": "value"}}

    result = store_yaml(fspace)
    expected = "feature:\n  key: value\n"

    np.testing.assert_equal(result, expected)


def test_store_yaml_to_file(mocker):
    fspace = mocker.Mock(spec=FeatureSpace)
    fspace.to_dict.return_value = {"feature": {"key": "value"}}

    open_mock = mocker.mock_open()
    mocker.patch("feets.io.open", open_mock)

    writer = MockWriter()
    open_mock.return_value.write = writer.write

    store_yaml(fspace, "output.json")
    expected = "feature:\n  key: value\n"

    open_mock.assert_called_once_with("output.json", "w")
    np.testing.assert_equal(writer.contents, expected)


def test_store_yaml_with_kwargs(mocker):
    fspace = mocker.Mock(spec=FeatureSpace)
    fspace.to_dict.return_value = {"feature": {"key": "value"}}

    result = store_yaml(fspace, indent=4)
    expected = "feature:\n    key: value\n"

    np.testing.assert_equal(result, expected)


def test_read_json(mocker):
    fspace = mocker.Mock(spec=FeatureSpace)
    mocker.patch("feets.io.FeatureSpace.from_dict", return_value=fspace)

    json_data = '{\n  "feature": {\n    "key": "value"\n  }\n}'
    fspace_dict = {"feature": {"key": "value"}}

    open_mock = mocker.mock_open(read_data=json_data)
    mocker.patch("feets.io.open", open_mock)

    result = read_json("input.json")

    open_mock.assert_called_once_with("input.json", "r")
    FeatureSpace.from_dict.assert_called_once_with(fspace_dict)
    assert result is fspace


def test_read_yaml(mocker):
    fspace = mocker.Mock(spec=FeatureSpace)
    mocker.patch("feets.io.FeatureSpace.from_dict", return_value=fspace)

    yaml_data = "feature:\n  key: value\n"
    fspace_dict = {"feature": {"key": "value"}}

    open_mock = mocker.mock_open(read_data=yaml_data)
    mocker.patch("feets.io.open", open_mock)

    result = read_yaml("input.yaml")

    open_mock.assert_called_once_with("input.yaml", "r")
    FeatureSpace.from_dict.assert_called_once_with(fspace_dict)
    assert result is fspace
