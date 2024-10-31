#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Custom JSON encoding and decoding module.

This module provides custom JSON encoding and decoding functionality by
extending the default JSONEncoder and providing additional converter
functions for various data types that are not supported by the default
encoder.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime as dt
import json

import numpy as np

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


def dump(obj, fp, **kwargs):
    """Serialize obj as a JSON formatted stream to fp (.write()-supporting \
    file-like object).

    Parameters
    ----------
    obj : object
        The object to be serialized.
    fp : file-like object
        A .write()-supporting file-like object to write the JSON formatted
        stream to.
    **kwargs
        Additional keyword arguments to be passed to the underlying json.dump()
        function.

    Returns
    -------
    None
    """
    kwargs.setdefault("cls", CustomJSONEncoder)
    return json.dump(obj, fp, **kwargs)


def dumps(obj, **kwargs):
    """Serialize obj to a JSON formatted str.

    Parameters
    ----------
    obj : object
        The object to be serialized.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        json.dumps() function.

    Returns
    -------
    str
        The JSON formatted string representation of the object.
    """
    kwargs.setdefault("cls", CustomJSONEncoder)
    return json.dumps(obj, **kwargs)


def load(fp, **kwargs):
    """Deserialize fp (.read()-supporting file-like object containing a JSON \
    document) to a Python object.

    Parameters
    ----------
    fp : file-like object
        A .read()-supporting file-like object containing a JSON document.
    **kwargs
        Additional keyword arguments to be passed to the underlying json.load()
        function.

    Returns
    -------
    object
        The Python object deserialized from the JSON document.
    """
    return json.load(fp, **kwargs)


def loads(text, **kwargs):
    """Deserialize text (str, bytes or bytearray instance containing a JSON \
    document) to a Python object.

    Parameters
    ----------
    text : str, bytes or bytearray
        A string, bytes or bytearray instance containing a JSON document.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        json.loads() function.

    Returns
    -------
    object
        The Python object deserialized from the JSON document.
    """
    return json.loads(text, **kwargs)
