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
# FIXTURES
# =============================================================================


@pytest.fixture
def patch_open(mocker):
    def maker(*args, **kwargs):
        mocked_open = mocker.mock_open(*args, **kwargs)
        mocked_open.return_value.write = mocker.Mock()
        return mocker.patch("feets.io.open", mocked_open)

    return maker


@pytest.fixture
def patch_from_dict(mocker):
    def maker(*args, **kwargs):
        return mocker.patch("feets.io.FeatureSpace.from_dict", *args, **kwargs)

    return maker


@pytest.fixture
def fspace_mock(mocker):
    fspace = mocker.Mock()

    fs_dict = {"nested": {"foo": "bar"}}

    fspace.to_dict.return_value = fs_dict
    return fspace


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ["obj", "expected"],
    [
        ((1, 2, 3), [1, 2, 3]),
        ({1, 2, 3}, [1, 2, 3]),
        (frozenset({1, 2, 3}), [1, 2, 3]),
        (np.array([1, 2, 3]), [1, 2, 3]),
        (
            dt.datetime(2024, 11, 6, 0, 41, 57, 812214),
            "2024-11-06T00:41:57.812214",
        ),
        (np.int64(42), int(42)),
        (np.float64(3.14), 3.14),
        (np.complex128(1 + 2j), complex(1, 2)),
        (np.True_, True),
        (np.False_, False),
    ],
)
def test_CustomJSONEncoder_default(obj, expected):
    encoder = CustomJSONEncoder()
    np.testing.assert_equal(encoder.default(obj), expected)


def test_CustomJSONEncoder_default_invalid_type():
    encoder = CustomJSONEncoder()
    with pytest.raises(TypeError):
        encoder.default("invalid_type")


def test_none_open_or_buffer_none():
    with none_open_or_buffer(None, "w") as fp:
        assert isinstance(fp, StringIO)
        fp.write("test")
        np.testing.assert_equal(fp.getvalue(), "test")


def test_none_open_or_buffer_path(patch_open):
    open_mock = patch_open()
    with none_open_or_buffer("output", "w") as fp:
        assert fp is open_mock.return_value
        fp.write("test")
        open_mock.return_value.write.assert_called_with("test")


def test_none_open_or_buffer_file_like():
    file_like = StringIO()
    with none_open_or_buffer(file_like, "w") as fp:
        fp.write("test")
        np.testing.assert_equal(fp.getvalue(), "test")


def test_store_json_to_string(fspace_mock):
    result = store_json(fspace_mock)
    expected = '{\n  "nested": {\n    "foo": "bar"\n  }\n}'

    np.testing.assert_equal(result, expected)


def test_store_json_to_file(patch_open, fspace_mock):
    open_mock = patch_open()

    store_json(fspace_mock, "output.json")
    expected = '{\n  "nested": {\n    "foo": "bar"\n  }\n}'

    calls = open_mock.return_value.write.mock_calls
    contents = "".join(call.args[0] for call in calls)

    open_mock.assert_called_once_with("output.json", "w")
    np.testing.assert_equal(contents, expected)


def test_store_json_kwargs(fspace_mock):
    result = store_json(fspace_mock, indent=4)
    expected = '{\n    "nested": {\n        "foo": "bar"\n    }\n}'

    np.testing.assert_equal(result, expected)


def test_store_json_unserializable(fspace_mock):
    fspace_mock.to_dict.return_value = {"nested": {"foo": object()}}
    with pytest.raises(TypeError):
        store_json(fspace_mock)


def test_store_yaml_to_string(fspace_mock):
    result = store_yaml(fspace_mock)
    expected = "nested:\n  foo: bar\n"

    np.testing.assert_equal(result, expected)


def test_store_yaml_to_file(patch_open, fspace_mock):
    open_mock = patch_open()

    store_yaml(fspace_mock, "output.json")
    expected = "nested:\n  foo: bar\n"

    calls = open_mock.return_value.write.mock_calls
    contents = "".join(call.args[0] for call in calls)

    open_mock.assert_called_once_with("output.json", "w")
    np.testing.assert_equal(contents, expected)


def test_store_yaml_kwargs(fspace_mock):
    result = store_yaml(fspace_mock, indent=4)
    expected = "nested:\n    foo: bar\n"

    np.testing.assert_equal(result, expected)


def test_store_yaml_unserializable(fspace_mock):
    fspace_mock.to_dict.return_value = {"nested": {"foo": object()}}
    with pytest.raises(TypeError):
        store_yaml(fspace_mock)


def test_read_json(patch_open, patch_from_dict, fspace_mock):
    from_dict_mock = patch_from_dict(return_value=fspace_mock)

    json_data = '{\n  "nested": {\n    "foo": "bar"\n  }\n}'
    open_mock = patch_open(read_data=json_data)

    result = read_json("test")

    open_mock.assert_called_once_with("test", "r")
    from_dict_mock.assert_called_once_with(fspace_mock.to_dict())
    np.testing.assert_equal(result, fspace_mock)


def test_read_yaml(patch_open, patch_from_dict, fspace_mock):
    from_dict_mock = patch_from_dict(return_value=fspace_mock)

    yaml_data = "nested:\n  foo: bar\n"
    open_mock = patch_open(read_data=yaml_data)

    result = read_yaml("test")

    open_mock.assert_called_once_with("test", "r")
    from_dict_mock.assert_called_once_with(fspace_mock.to_dict())
    np.testing.assert_equal(result, fspace_mock)
