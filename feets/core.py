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

__all__ = ["FeatureSpace"]


# =============================================================================
# IMPORTS
# =============================================================================

import logging

import numpy as np

from . import extractors
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
        extractor_clss = extractors.register.get_execution_plan(
            data=data, only=only, exclude=exclude
        )

        selected_extractors = []
        selected_features = set()
        required_data = set()
        for extractor_cls in extractor_clss:
            extractor_kwargs = {
                pname: kwargs.get(pname, pvalue)
                for pname, pvalue in extractor_cls.get_default_params()
            }

            extractor = extractor_cls(**extractor_kwargs)
            features = extractor_cls.get_features()
            data = extractor_cls.get_data()

            selected_extractors.append(extractor)
            selected_features.update(features)
            required_data.update(data)

        self._extractors = np.array(selected_extractors)
        self._selected_features = frozenset(
            selected_features.intersection(only or [])
        )
        self._required_data = frozenset(required_data)

    def __repr__(self):
        space = ", ".join(str(extractor) for extractor in self._extractors)
        return f"<FeatureSpace: {space}>"

    def extract(self, **data):
        for dname in self._required_data:
            if data.get(dname, None) is None:
                raise DataRequiredError(dname)

        features = {}
        for extractor in self._extractors:
            results = extractor.select_extract_and_validate(
                data=data,
                dependencies=features,
                selected_features=self._selected_features,
            )
            features.update(results)

        return FeatureSet("features", features)

    @property
    def features(self):
        return self._selected_features

    @property
    def execution_plan(self):
        return self._extractors
