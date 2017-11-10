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

__all__ = [
    "FeatureNotFound",
    "Features",
    "FeatureSpace"]


# =============================================================================
# IMPORTS
# =============================================================================

import sys
import logging
from collections import Mapping

import six
from six.moves import zip

from tabulate import tabulate

import numpy as np

from . import extractors


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


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

class Features(Mapping):

    def __init__(self, names, values):
        self._names = names
        self._values = values

    def __dir__(self):
        dlist = list(super(Features, self).__dir__())
        dlist += list(self._names)
        return dlist

    def __getitem__(self, name):
        index = np.where(self._names == name)[0]
        if index.size == 0:
            raise KeyError(name)
        return self._values[index][0]

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return self._names.size

    def __unicode__(self):
        return self.to_str()

    def __bytes__(self):
        encoding = sys.getdefaultencoding()
        return self.__unicode__().encode(encoding, 'replace')

    def __str__(self):
        """Return a string representation for a particular Object
        Invoked by str(df) in both py2/py3.
        Yields Bytestring in Py2, Unicode String in py3.
        """
        if six.PY3:
            return self.__unicode__()
        return self.__bytes__()

    def __repr__(self):
        return str(self)

    def _repr_html_(self):
        return self.to_str(tablefmt="html")

    def to_str(self, **params):
        """String representation of the Features object.
        Parameters
        ----------
        kwargs :
            Parameters to configure
            `tabulate <https://bitbucket.org/astanin/python-tabulate>`_
        Return
        ------
        str :
            String representation of the Data object.
        """

        params.update({
            k: v for k, v in TABULATE_PARAMS.items() if k not in params})
        headers = [["Feature", "Value"]]
        rows = [[n, v] for n, v in zip(self._names, self._values)]
        return tabulate(headers + rows, **params)

    def raw(self):
        return self._names, self._values


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

        # TODO: remove by dependencies

        # final list of features
        self._features = self._features_by_data.intersection(
            self._only).difference(self._exclude)

        # create a ndarray for all the results
        self._features_as_array = np.array(sorted(self._features))

        # initialize the extractors
        features_extractors = set()
        for fcls in set(exts.values()):
            if fcls.get_features().intersection(self._features):
                features_extractors.add(fcls(self))
        self._features_extractors = frozenset(features_extractors)

        # excecution order by dependencies
        self._execution_plan = extractors.sort_by_dependencies(
            features_extractors)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not hasattr(self, "__str"):
            extractors = [str(extractor) for extractor in self._execution_plan]
            space = ", ".join(extractors)
            self.__str = "<FeatureSpace: {}>".format(space)
        return self.__str

    def params_by_features(self, features):
        params = {}
        for f in features:
            params.update(self._kwargs.get(f, {}))
        return params

    def _extract_one(self, data):
        data, features = np.asarray(data), {}
        for fextractor in self._execution_plan:
            features.update(fextractor.extract(data, features))
        fvalues = np.array([
            features[fname] for fname in self._features_as_array])
        return fvalues

    def extract_one(self, data):
        return Features(self._features_as_array,
                        self._extract_one(data))

    def extract(self, data):
        result = []
        for chunk in data:
            result.append(self._extract_one(chunk))
        return Features(
            self._features_as_array,
            np.asarray(result))

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
