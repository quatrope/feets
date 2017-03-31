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
    def __init__(self, data=None, features=None, exclude=None, **kwargs):

        self._kwargs = kwargs

        # get all posible features by data
        if data:
            fbdata = []
            for fname, f in extractors.extractors.items():
                if not f.data.difference(data):
                    fbdata.append(fname)
        else:
            fbdata = extractors.extractors.keys()
        self.features_by_data = frozenset(fbdata)

        self.data = frozenset(data or extractors.DATAS)


        # validate the list of features or select all of them
        if features:
            for f in features:
                if f not in extractors.extractors:
                    raise FeatureNotFound(f)
        self.features = frozenset(features or extractors.extractors.keys())

        # select the features to exclude or not exclude anything
        if exclude:
            for f in exclude:
                if f not in extractors.extractors:
                    raise FeatureNotFound(f)
        self.exclude = frozenset(exclude or ())

        # TODO: remove by dependencies

        # final list of features
        self.selected_features = self.features_by_data.intersection(
                self.features).difference(self.exclude)

        # create a ndarray for all the results
        self._features_as_array = np.array(sorted(self.selected_features))

        # TODO: excecution_order by dependencies
        self._execution_plan = tuple(self._features_as_array)

        # retrieve only the relevant kwargs
        self._selected_kwargs = {
            fname: params for fname, params in kwargs.items()
            if fname in self.selected_features}

        # initialize the extractors
        self._features_extractors = {}
        for fname in self.selected_features:
            Extractor = extractors.extractors[fname]
            params = self._selected_kwargs.get(fname, {})
            self._features_extractors[fname] = Extractor(**params)

    def __repr__(self):
        return str(self)

    def __str__(self):
        extractors = []
        for fname in self._features_as_array:
            fparams = ", ".join([
                "{}={}".format(k, v)
                for k, v in self._selected_kwargs.get(fname, {}).items()])
            extractors.append("{}({})".format(fname, fparams))
        space = ", ".join(extractors)
        return "<FeatureSpace: {}>".format(space)

    @property
    def kwargs(self):
        return dict(self._kwargs)

    @property
    def selected_kwargs(self):
        return dict(self._selected_kwargs)

    def calculateFeature(self, data):
        data, features = np.asarray(data), {}
        for fname in self._execution_plan:
            fextractor = self._features_extractors[fname]
            features[fname] = fextractor.extract(data, features)
        fvalues = np.array([
            result[fname] for fname in self._features_as_array])
        return self._features_as_array, fvalues
