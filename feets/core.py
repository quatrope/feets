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
# FUTURE
# =============================================================================

from __future__ import unicode_literals, print_function


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """core functionalities of feets"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import time
import inspect
import warnings

import numpy as np
import pandas as pd

from . import extractors

class FeatureError(Exception): pass


class FeatureNotFound(ValueError): pass


class FeatureWarning(Warning): pass


class FeatureSpace(object):
    """
    This Class is a wrapper class, to allow user select the
    features based on the available time series vectors (magnitude, time,
    error, second magnitude, etc.) or specify a list of features.

    __init__ will take in the list of the available data and featureList.

    User could only specify the available time series vectors, which will
    output all the features that need this data to be calculated.

    User could only specify featureList, which will output
    all the features in the list.

    User could specify a list of the available time series vectors and
    featureList, which will output all the features in the List that
    use the available data.

    Additional parameters are used for individual features.
    Format is featurename = [parameters]

    usage:
    data = np.random.randint(0,10000, 100000000)
    # automean is the featurename and [0,0] is the parameter for the feature
    a = FeatureSpace(category='all', automean=[0,0])
    print a.featureList
    a=a.calculateFeature(data)
    print a.result(method='array')
    print a.result(method='dict')

    """
    def __init__(self, data=None, only=None, exclude=None, **kwargs):
        # retrieve all the extractors
        exts = extractors.registered_extractors()

        # store all the parameters for the extractors
        self._kwargs = kwargs

        # get all posible features by data
        if data:
            fbdata = []
            for fname, f in exts.items():
                if not f._conf.data.difference(data):
                    fbdata.append(fname)
        else:
            fbdata = exts.keys()
        self._data = frozenset(data or extractors.DATAS)
        self._features_by_data = frozenset(fbdata)
        import ipdb; ipdb.set_trace()

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

        # TODO: remove by dependencies

        # final list of features
        self._features = self._features_by_data.intersection(
            self._only).difference(self._exclude)

        # create a ndarray for all the results
        self._features_as_array = np.array(sorted(self._features))

        # initialize the extractors
        features_extractors = set()
        for fcls in set(exts.values()):
            if fcls._conf.features.intersection(self._features):
                self._fetures_extractors.add(fcls(self))
        self._features_extractors = frozenset(features_extractors)

        # TODO: excecution_order by dependencies
        self._execution_plan = tuple(features_extractors)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not hasattr(self, "__str"):
            extractors = []
            for fname in self._features_as_array:
                extractor = self._features_extractors[fname]
                extractors.append(str(extractor))
            space = ", ".join(extractors)
            self.__str = "<FeatureSpace: {}>".format(space)
        return self.__str

    def params_by_features(self, features):
        params = {}
        for f in features:
            params.update(self._kwargs.get(f, {}))
        return params

    def extract(self, data):
        data, features = np.asarray(data), {}
        for fname in self._execution_plan:
            features.update(fextractor.extract(data, features))
         fvalues = np.array([
            features[fname] for fname in self._features_as_array])
        return self._features_as_array, fvalues

    @property
    def kwargs(self):
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
