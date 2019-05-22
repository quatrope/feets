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

"""core functionalities of feets"""

__all__ = [
    "FeatureNotFound",
    "DataRequiredError",
    "FeatureSpace"]


# =============================================================================
# IMPORTS
# =============================================================================

import logging

import numpy as np

import attr

from . import extractors
from .extractors.core import (
    DATA_MAGNITUDE,
    DATA_TIME,
    DATA_ERROR,
    DATA_MAGNITUDE2,
    DATA_ALIGNED_MAGNITUDE,
    DATA_ALIGNED_MAGNITUDE2,
    DATA_ALIGNED_TIME,
    DATA_ALIGNED_ERROR,
    DATA_ALIGNED_ERROR2)


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

class FeatureNotFound(ValueError):
    pass


class DataRequiredError(ValueError):
    pass


# =============================================================================
# FEATURE RESULT
# =============================================================================

@attr.s(frozen=True, auto_attribs=True)
class ResultSet:

    features: np.ndarray = attr.ib(repr=True)
    values: np.ndarray = attr.ib(repr=False)
    extractors_conf: dict = attr.ib(repr=False)
    lc: dict = attr.ib(repr=False)

    def __iter__(self):
        return iter(self.as_arrays())

    def extractor_of(self, feature):
        # regenerate the extractor
        fcls = extractors.extractor_of(feature)
        extractor_params = self.extractors_conf.get(fcls.__name__, {})
        extractor = fcls(**extractor_params)
        return extractor

    def as_arrays(self):
        flatten_features = {}
        for feature, value in zip(self.features, self.values):
            extractor = self.extractor_of(feature)
            flatten_value = extractor.flatten_feature(feature, value)
            flatten_features.update(flatten_value)

        features = np.empty(len(flatten_features), dtype=object)
        values = np.empty(len(flatten_features))
        for idx, fv in enumerate(flatten_features.items()):
            features[idx], values[idx] = fv

        return features, values

    def __getitem__(self, feature):
        idx, = np.where(self.features == feature)
        if not len(idx):
            raise KeyError(f"{feature}")
        return self.values[idx[0]]

    def as_dict(self):
        return dict(zip(self.features, self.values))


    # def plot(self, feature, ax=None, **kwargs):
    #     if feature not in self.features:
    #         raise FeatureNotFound(feature)



    #     plot = fex.make_plot(
    #         feature_name=feature, features=features,
    #         lc=self.data, ax=ax, plot_kwargs=kwargs)


# =============================================================================
# FEATURE EXTRACTORS
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
        # retrieve all the extractors
        exts = extractors.registered_extractors()

        # store all the parameters for the extractors
        self._kwargs = kwargs

        # get all possible features by data
        if data:
            fbdata = []
            for fname, f in exts.items():
                if not f.get_data().difference(data):
                    fbdata.append(fname)
        else:
            fbdata = exts.keys()
        self._data = frozenset(data or extractors.DATAS)
        self._features_by_data = frozenset(fbdata)

        # validate the list of features or select all of them
        if only:
            for f in only:
                if f not in exts:
                    raise FeatureNotFound(f)
        self._only = frozenset(only or exts.keys())

        # select the features to exclude or not exclude anything
        if exclude:
            for f in exclude:
                if f not in exts:
                    raise FeatureNotFound(f)
        self._exclude = frozenset(exclude or ())

        # the candidate to be the features to be extracted
        candidates = self._features_by_data.intersection(
            self._only).difference(self._exclude)

        # remove by dependencies
        if only or exclude:
            final = set()
            for f in candidates:
                fcls = exts[f]
                dependencies = fcls.get_dependencies()
                if dependencies.issubset(candidates):
                    final.add(f)
        else:
            final = candidates

        # the final features
        self._features = frozenset(final)

        # create a ndarray for all the results
        self._features_as_array = np.array(sorted(self._features))

        # initialize the extractors and determine the required data only
        features_extractors, features_extractors_names = set(), set()
        required_data = set()
        for fcls in set(exts.values()):
            if fcls.get_features().intersection(self._features):

                params = self._kwargs.get(fcls.__name__, {})
                fext = fcls(**params)

                features_extractors.add(fext)
                features_extractors_names.add(fext.name)
                required_data.update(fext.get_data())

        self._features_extractors = frozenset(features_extractors)
        self._features_extractors_names = frozenset(features_extractors_names)
        self._required_data = frozenset(required_data)

        # excecution order by dependencies
        self._execution_plan = extractors.sort_by_dependencies(
            features_extractors)

        not_found = set(self._kwargs).difference(
            self._features_extractors_names)
        if not_found:
            msg = (
                "This space not found feature(s) extractor(s) {} "
                "to assign the given parameter(s)"
            ).format(", ".join(not_found))
            raise FeatureNotFound(msg)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not hasattr(self, "__str"):
            extractors = [str(extractor) for extractor in self._execution_plan]
            space = ", ".join(extractors)
            self.__str = "<FeatureSpace: {}>".format(space)
        return self.__str

    def dict_data_as_array(self, d):
        array_data = {}
        for k, v in d.items():
            if k in self._required_data and v is None:
                raise DataRequiredError(k)
            array_data[k] = v if v is None else np.asarray(v)
        return array_data

    def extract(self, time=None, magnitude=None, error=None,
                magnitude2=None, aligned_time=None,
                aligned_magnitude=None, aligned_magnitude2=None,
                aligned_error=None, aligned_error2=None):

        lc = self.dict_data_as_array({
            DATA_TIME: time,
            DATA_MAGNITUDE: magnitude,
            DATA_ERROR: error,
            DATA_MAGNITUDE2: magnitude2,
            DATA_ALIGNED_TIME: aligned_time,
            DATA_ALIGNED_MAGNITUDE: aligned_magnitude,
            DATA_ALIGNED_MAGNITUDE2: aligned_magnitude2,
            DATA_ALIGNED_ERROR: aligned_error,
            DATA_ALIGNED_ERROR2: aligned_error2})

        features = {}
        for fextractor in self._execution_plan:
            result = fextractor.extract(features=features, **lc)
            features.update(result)

        fvalues = np.array([
            features[fname] for fname in self._features_as_array])

        rs = ResultSet(
            features=self._features_as_array, values=fvalues,
            lc=lc, extractors_conf=self.extractors_conf)
        return rs

    @property
    def extractors_conf(self):
        return dict(self._kwargs)

    @property
    def data(self):
        return self._data

    @property
    def only(self):
        return self._only

    @property
    def exclude(self):
        return self._exclude

    @property
    def features_by_data_(self):
        return self._features_by_data

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
        return self._execution_plan

    @property
    def required_data_(self):
        return self._required_data
