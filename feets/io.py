#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# This code was ripped of from scikit-neuromsi on 06-nov-2024.
# https://github.com/renatoparedes/scikit-neuromsi/blob/f197a3c/skneuromsi/utils/custom_json.py
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# All rights reserved.


# =============================================================================
# DOCS
# =============================================================================

"""Serialize and deserialize `feets.FeatureSpace` objects."""

# =============================================================================
# IMPORTS
# =============================================================================

import contextlib
import datetime as dt
import io
import json
import pathlib

import numpy as np

import yaml

from .core import FeatureSpace

# =============================================================================
# CUSTOM JSON ENCODER
# =============================================================================


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON <https://json.org> encoder for `feets.FeatureSpace` objects.

    This class extends the `json.JSONEncoder` to add support for the following
    objects and types:

    +----------------------------------------------------+---------------+
    | Python                                             | JSON          |
    +====================================================+===============+
    | tuple, set, frozenset, np.ndarray                  | array         |
    +----------------------------------------------------+---------------+
    | datetime                                           | string        |
    +----------------------------------------------------+---------------+
    | np.integer, np.floating, np.complexfloating        | number        |
    +----------------------------------------------------+---------------+
    | np.true                                            | true          |
    +----------------------------------------------------+---------------+
    | np.false                                           | false         |
    +----------------------------------------------------+---------------+

    Attributes
    ----------
    CONVERTERS : dict
        A dictionary mapping data types to their corresponding converter
        functions.

    See Also
    --------
    json.JSONEncoder :
        Extensible JSON https://json.org encoder for Python data structures.
    """

    CONVERTERS = (
        (tuple, list),
        (set, list),
        (frozenset, list),
        (dt.datetime, dt.datetime.isoformat),
        (np.integer, int),
        (np.floating, float),
        (np.complexfloating, complex),
        (np.bool_, bool),
        (np.ndarray, np.ndarray.tolist),
    )

    def default(self, obj):
        """Serialize an object to a JSON-serializable format.

        This method overrides the default method of the `json.JSONEncoder`
        class to provide custom serialization for the data structures defined
        in the `CONVERTERS` attribute, or calls the base implementation for
        any other object.

        Returns
        -------
        object
            The JSON-serializable representation of the object.

        Raises
        ------
        TypeError
            If the object does not match any of the types in `CONVERTERS`.

        """
        for cls, converter in self.CONVERTERS:
            if isinstance(obj, cls):
                return converter(obj)
        return super(CustomJSONEncoder, self).default(obj)


# =============================================================================
# API
# =============================================================================


@contextlib.contextmanager
def none_open_or_buffer(path_or_buffer, mode):
    """Context manager to handle file paths or buffers as file-like objects.

    This function provides a unified way to handle file paths, buffers, or
    in-memory buffers, and yields a file-like object for reading or writing.

    Parameters
    ----------
    path_or_buffer : str, pathlib.Path, file-like object or None
        - If `str` or `pathlib.Path`, the file at this given path is opened
            with the specified `mode`.
        - If a file-like object, it is yielded directly.
        - If `None`, an `io.StringIO` in-memory buffer is created and yielded.
    mode : str
        The mode in which to open the file (e.g., 'r', 'w'). This is ignored
        if `path_or_buffer` is not a path.

    Yields
    ------
    file-like object
        An open, ready-to-use file-like object.
    """
    if path_or_buffer is None:
        yield io.StringIO()

    elif isinstance(path_or_buffer, (str, pathlib.Path)):
        with open(path_or_buffer, mode) as fp:
            yield fp
    else:
        yield path_or_buffer


def store_json(fspace, path_or_buffer=None, **kwargs):
    """Serialize a `feets.FeatureSpace` to a JSON formatted string or file.

    Parameters
    ----------
    fspace : feets.FeatureSpace
        The `feets.FeatureSpace` object to serialize. This object must
        implement a `to_dict` method that returns a serializable
        representation.
    path_or_buffer : str, pathlib.Path, file-like object or None, default=None
        The file path, buffer, or stream to write the JSON data to.
        If `None`, the JSON data is returned as a string.
    **kwargs
        Additional keyword arguments passed to `json.dump` when serializing
        the feature space.

    Returns
    -------
    str or None
        If `path_or_buffer` is `None`, returns a JSON formatted string
        representing the feature space. Otherwise, writes the JSON data to the
        specified file or buffer and returns `None`.

    Raises
    ------
    TypeError
        If the provided feature space contains non-serializable objects.

    See Also
    --------
    feets.FeatureSpace :
        Class to select and extract features from a time series.
    read_json
    json.dump
    """
    data = fspace.to_dict()

    kwargs.setdefault("indent", 2)
    with none_open_or_buffer(path_or_buffer, "w") as fp:
        json.dump(data, fp=fp, cls=CustomJSONEncoder, **kwargs)

    if path_or_buffer is None:
        return fp.getvalue()


def store_yaml(fspace, path_or_buffer=None, **kwargs):
    """Serialize a `feets.FeatureSpace` to a YAML formatted string or file.

    Parameters
    ----------
    fspace : feets.FeatureSpace
        The `feets.FeatureSpace` object to serialize. This object must
        implement a `to_dict` method that returns a serializable
        representation.
    path_or_buffer : str, pathlib.Path, file-like object or None, default=None
        The file path, buffer, or stream to write the YAML data to.
        If `None`, the YAML data is returned as a string.
    **kwargs
        Additional keyword arguments passed to `yaml.safe_dump` when serializing
        the feature space.

    Returns
    -------
    str or None
        If `path_or_buffer` is `None`, returns a YAML formatted string
        representing the feature space. Otherwise, writes the YAML data to the
        specified file or buffer and returns `None`.

    Raises
    ------
    TypeError
        If the provided feature space contains non-serializable objects.

    See Also
    --------
    feets.FeatureSpace :
        Class to select and extract features from a time series.
    read_yaml
    yaml.safe_dump
    """
    json_str = store_json(fspace, path_or_buffer=None, indent=None)
    data = json.loads(json_str)

    with none_open_or_buffer(path_or_buffer, "w") as fp:
        yaml.safe_dump(data, stream=fp, **kwargs)

    if path_or_buffer is None:
        return fp.getvalue()


def read_json(path_or_buffer):
    """Deserialize a JSON formatted string or file to `feets.FeatureSpace`.

    Parameters
    ----------
    path_or_buffer : str, pathlib.Path or file-like object
        The file path, buffer, or stream to read the JSON data from.

    Returns
    -------
    feets.FeatureSpace
        A `feets.FeatureSpace` object containing the deserialized data.

    See Also
    --------
    feets.FeatureSpace :
        Class to select and extract features from a time series.
    store_json
    """
    with none_open_or_buffer(path_or_buffer, "r") as fp:
        data = json.load(fp)
    return FeatureSpace.from_dict(data)


def read_yaml(path_or_buffer):
    """Deserialize a YAML formatted string or file to `feets.FeatureSpace`.

    Parameters
    ----------
    path_or_buffer : str, pathlib.Path or file-like object
        The file path, buffer, or stream to read the YAML data from.

    Returns
    -------
    feets.FeatureSpace
        A `feets.FeatureSpace` object containing the deserialized data.

    See Also
    --------
    feets.FeatureSpace :
        Class to select and extract features from a time series.
    store_yaml
    """
    with none_open_or_buffer(path_or_buffer, "r") as fp:
        data = yaml.safe_load(fp)
    return FeatureSpace.from_dict(data)
