#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """core functionalities of feets"""

__all__ = ["DataRequiredError", "FeatureSpace"]


# =============================================================================
# IMPORTS
# =============================================================================

import logging

import numpy as np

from . import extractors
from .extractors.extractor import (
    DATA_ALIGNED_ERROR,
    DATA_ALIGNED_ERROR2,
    DATA_ALIGNED_MAGNITUDE,
    DATA_ALIGNED_MAGNITUDE2,
    DATA_ALIGNED_TIME,
    DATA_ERROR,
    DATA_MAGNITUDE,
    DATA_MAGNITUDE2,
    DATA_TIME,
)
from .libs import bunch

# =============================================================================
# CONSTANTS
# =============================================================================

TABULATE_PARAMS = {
    "headers": "firstrow",
    "numalign": "center",
    "stralign": "center",
}


# =============================================================================
# LOG
# =============================================================================

logger = logging.getLogger("feets")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class DataRequiredError(ValueError):
    pass


# =============================================================================
# FEATURE SET
# =============================================================================


class FeatureSet(bunch.Bunch):
    pass


# =============================================================================
# FEATURE SPACE
# =============================================================================


class FeatureSpace:
    """Wrapper class, to allow user select the
    features based on the available time series vectors (magnitude, time,
    error, second magnitude, etc.) or specify a list of features.
    The finally selected features for the execution plan are are those that
    satisfy all the filters.

    Parameters
    ----------

    data : array-like, optional, default ``None``
        available time series vectors, which will
        output all the features that need this data to be calculated.

    only : array-like, optional, default ``None``
        List of features, which will output
        all the features in the list.

    exclude : array-like, optional, default ``None``
        List of features, which will not output

    kwargs
        Extra configuration for the feature extractors.
        format is ``Feature_name={param1: value, param2: value, ...}``

    Examples
    --------

    **List of features as an input:**

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Std'])
        >>> features, values = fs.extract(*lc)
        >>> dict(zip(features, values))
        {"Std": .42}

    **Available data as an input:**

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(data=['magnitude','time'])
        >>> features, values = fs.extract(*lc)
        >>> dict(zip(features, values))
        {...}

    **List of features and available data as an input:**

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(
        ...     only=['Mean','Beyond1Std', 'CAR_sigma','Color'],
        ...     data=['magnitude', 'error'])
        >>> features, values = fs.extract(*lc)
        >>> dict(zip(features, values))
        {"Beyond1Std": ..., "Mean": ...}

    **Excluding list as an input**

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(
        ...     only=['Mean','Beyond1Std','CAR_sigma','Color'],
        ...     data=['magnitude', 'error'],
        ...     exclude=["Beyond1Std"])
        >>> features, values = fs.extract(**lc)
        >>> dict(zip(features, values))
        {"Mean": 23}

    """

    def __init__(self, data=None, only=None, exclude=None, **kwargs):
        extractors_clss = extractors.register.get_plan(
            data=data, only=only, exclude=exclude
        )

        exts = []
        for extractor_cls in extractors_clss:
            ext_kwargs = {
                pname: kwargs.get(pname, pvalue)
                for pname, pvalue in extractor_cls.get_default_params()
            }
            extractor = extractor_cls(**ext_kwargs)
            exts.append(extractor)

        self._extractors = np.array(exts)

    def __repr__(self):
        return str(self)

    def __str__(self):
        space = ", ".join(str(extractor) for extractor in self._extractors)
        return f"<FeatureSpace: {space}>"

    def dict_data_as_array(self, data):
        array_data = {}
        for k, v in data.items():
            if k in self._required_data and v is None:
                raise DataRequiredError(k)
            array_data[k] = v if v is None else np.asarray(v)
        return array_data

    def extract(
        self,
        time=None,
        magnitude=None,
        error=None,
        magnitude2=None,
        aligned_time=None,
        aligned_magnitude=None,
        aligned_magnitude2=None,
        aligned_error=None,
        aligned_error2=None,
    ):

        kwargs = self.dict_data_as_array(
            {
                DATA_TIME: time,
                DATA_MAGNITUDE: magnitude,
                DATA_ERROR: error,
                DATA_MAGNITUDE2: magnitude2,
                DATA_ALIGNED_TIME: aligned_time,
                DATA_ALIGNED_MAGNITUDE: aligned_magnitude,
                DATA_ALIGNED_MAGNITUDE2: aligned_magnitude2,
                DATA_ALIGNED_ERROR: aligned_error,
                DATA_ALIGNED_ERROR2: aligned_error2,
            }
        )

        features = {}
        for extractor in self._extractors:
            result = extractor.extract(features=features, **kwargs)
            features.update(result)

        return FeatureSet("features", features)

    @property
    def kwargs(self):
        return dict(self._kwargs)

    @property
    def features_(self):
        return self._features

    @property
    def features_extractors_(self):
        return self._features_extractors

    @property
    def features_as_array_(self):
        return self._features_as_array

    @property
    def excecution_plan_(self):
        return self._extractors
