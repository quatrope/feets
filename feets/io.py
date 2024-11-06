#!/usr/bin/env python

# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
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

"""Functionalities for serializing and deserializing FeatureSpace objects."""

# =============================================================================
# IMPORTS
# =============================================================================

import contextlib
import datetime as dt
import io
import json
import pathlib

import yaml

import numpy as np

from .core import FeatureSpace

# =============================================================================
# CUSTOM JSON ENCODER
# =============================================================================


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder class that extends the default JSONEncoder.

    This class provides additional functionality for encoding various data
    types that are not supported by the default JSONEncoder, such as tuples,
    sets, frozensets, datetime objects, NumPy types, and NumPy arrays.

    Attributes
    ----------
    CONVERTERS : dict
        A dictionary mapping data types to their corresponding converter
        functions. The converter functions are used to convert the data types
        to JSON-serializable representations.

    Methods
    -------
    default(obj)
        Overrides the default method of JSONEncoder to handle additional data
        types. If the object is an instance of any of the data types specified
        in the CONVERTERS dictionary, the corresponding converter function is
        applied. Otherwise, the default behavior of the superclass is used.
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
        """
        Override the default method to handle additional data types.

        Parameters
        ----------
        obj : object
            The object to be encoded.

        Returns
        -------
        object
            The JSON-serializable representation of the object.
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
    """Context manager that yields a file-like object for reading or writing.

    This context manager handles opening a file or a file-like object for
    reading or writing.

    If `path_or_buffer` is `None`, it creates and yields a `StringIO` object.
    Otherwise, it opens the file using the specified mode.

    Parameters
    ----------
    path_or_buffer : str, pathlib.Path, file-like object or None
        The file path or file-like object to read from or write to. If `None`,
        a `StringIO` object is yielded.
    mode : str
        The mode in which to open the file, determining whether it is for
        reading, writing, or both, and whether to create the file if it
        does not exist.

    Yields
    ------
    file-like object
        A file-like object for reading or writing data.
    """
    if path_or_buffer is None:
        yield io.StringIO()

    elif isinstance(path_or_buffer, (str, pathlib.Path)):
        with open(path_or_buffer, mode) as fp:
            yield fp
    else:
        yield path_or_buffer


def store_json(fspace, path_or_buffer=None, **kwargs):
    """Serialize a feature space to a JSON formatted string or file.

    Parameters
    ----------
    fspace : FeatureSpace
        The feature space to serialize.
    path_or_buffer : str, pathlib.Path, file-like object or None, optional
        The file path or buffer to write the JSON data to. If `None`, the JSON
        data is returned as a string. Defaults to `None`.
    **kwargs
        Additional keyword arguments to pass to `json.dump`.

    Returns
    -------
    str
        The JSON formatted string if `path_or_buffer` is None.

    Raises
    ------
    TypeError
        If the dictionary contains non-serializable objects.
    """
    data = fspace.to_dict()

    kwargs.setdefault("indent", 2)
    with none_open_or_buffer(path_or_buffer, "w") as fp:
        json.dump(data, fp=fp, cls=CustomJSONEncoder, **kwargs)

    if path_or_buffer is None:
        return fp.getvalue()


def store_yaml(fspace, path_or_buffer=None, **kwargs):
    """Serialize a feature space to a YAML formatted string or file.

    Parameters
    ----------
    fspace : FeatureSpace
        The feature space to serialize.
    path_or_buffer : str, pathlib.Path, file-like object or None, optional
        The file path or buffer to write the YAML data to. If `None`, the JSON
        data is returned as a string. Defaults to `None`.
    **kwargs
        Additional keyword arguments to pass to `json.dump`.

    Returns
    -------
    str
        The YAML formatted string if `path_or_buffer` is None.

    Raises
    ------
    TypeError
        If the dictionary contains non-serializable objects.
    """
    json_str = store_json(fspace, path_or_buffer=None, indent=None)
    data = json.loads(json_str)

    with none_open_or_buffer(path_or_buffer, "w") as fp:
        yaml.safe_dump(data, stream=fp, **kwargs)

    if path_or_buffer is None:
        return fp.getvalue()


def read_json(path_or_buffer):
    """Deserialize a JSON formatted string or file to a feature space.

    Parameters
    ----------
    path_or_buffer : str, pathlib.Path, file-like object
        The file path or buffer to read the JSON data from.

    Returns
    -------
    FeatureSpace
        The deserialized feature space.
    """
    with none_open_or_buffer(path_or_buffer, "r") as fp:
        data = json.load(fp)
    return FeatureSpace.from_dict(data)


def read_yaml(path_or_buffer):
    """Deserialize a YAML formatted string or file to a feature space.

    Parameters
    ----------
    path_or_buffer : str, pathlib.Path, file-like object
        The file path or buffer to read the YAML data from.

    Returns
    -------
    FeatureSpace
        The deserialized feature space.
    """
    with none_open_or_buffer(path_or_buffer, "r") as fp:
        data = yaml.safe_load(fp)
    return FeatureSpace.from_dict(data)
