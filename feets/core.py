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

import logging
import multiprocessing as mp

import numpy as np

from . import extractors, util, err


# =============================================================================
# CONSTANTS
# =============================================================================

CPU_COUNT = mp.cpu_count()


# =============================================================================
# LOG
# =============================================================================

logger = logging.getLogger("feets")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

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

        # validate the list of features or select all of them
        if only:
            for f in only:
                if f not in exts:
                    raise err.FeatureNotFound(f)
        self._only = frozenset(only or exts.keys())

        # select the features to exclude or not exclude anything
        if exclude:
            for f in exclude:
                if f not in exts:
                    raise err.FeatureNotFound(f)
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
                features_extractors.add(fcls(self))
        self._features_extractors = frozenset(features_extractors)

        # TODO: excecution_order by dependencies
        self._execution_plan = tuple(util.fero(features_extractors))

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

    def _extract(self, data):
        data, features = np.asarray(data), {}
        for fextractor in self._execution_plan:
            features.update(fextractor.extract(data, features))
        fvalues = np.array([
            features[fname] for fname in self._features_as_array])
        return self._features_as_array, fvalues

    def extract(self, data):
        return self._extract(data)

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


# =============================================================================
# MULTIPROCESS
# =============================================================================

class FeatureSpaceProcess(mp.Process):

    def __init__(self, space, data, **kwargs):
        super(FeatureSpaceProcess, self).__init__(**kwargs)
        self._space = space
        self._data = data
        self._queue = mp.Queue()

    def run(self):
        result = []
        for data in self._data:
            result.append(self._space._extract(self._data)[1])
        self._queue.put(result)

    @property
    def space(self):
        return self._space

    @property
    def data(self):
        return self._data

    @property
    def queue(self):
        return self._queue

    @property
    def result_(self):
        return self._result


class MPFeatureSpace(FeatureSpace):
    """Multiprocess version of FeatureSpace

    """
    def __init__(self, data=None, only=None, exclude=None,
                 proccls=FeatureSpaceProcess, **kwargs):
        super(MPFeatureSpace, self).__init__(
            data=data, only=only, exclude=exclude, **kwargs)
        self._proccls = self.proc_cls

    def __str__(self):
        if not hasattr(self, "__str"):
            extractors = [str(extractor) for extractor in self._execution_plan]
            space = ", ".join(extractors)
            self.__str = "<MPFeatureSpace: {}>".format(space)
        return self.__str

    def chunk_it(self, seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return sorted(out, reverse=True)

    def extract(self, data, procs=CPU_COUNT, **kwargs):
        procs, fvalues = [], []
        for chunk in self.chunk_it(data, procs):
            proc = self._proccls(self, chunk)
            proc.start()
        for proc in procs:
            proc.join()
            fvalues.append(proc.result_)
        return self._features_as_array, tuple(fvalues)

    @property
    def proccls(self):
        return self._proccls


# =============================================================================
#
# =============================================================================
