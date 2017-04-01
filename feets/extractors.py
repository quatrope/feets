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

__doc__ = """Features extractors classes and register utilities"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
import math
import bisect
import abc
import inspect

import six

import numpy as np

from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from statsmodels.tsa import stattools
from scipy.interpolate import interp1d

#~ import lomb

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_MAGNITUDE = "magnitude"
DATA_TIME = "time"
DATA_ERROR = "error"

DATA_IDXS = {
    DATA_MAGNITUDE: 0,
    DATA_TIME: 1,
    DATA_ERROR: 2,
}

DATAS = tuple([d[0] for d in sorted(DATA_IDXS.items(), key=lambda di: di[1])])


# =============================================================================
# REGISTER UTILITY
# =============================================================================

_extractors = {}


def register_extractor(cls, name=None):

    name = name or cls.__name__

    if not inspect.isclass(cls) or not issubclass(cls, Extractor):
        msg = "'cls' must be a subclass of Extractor. Found: {}"
        raise TypeError(msg.format(cls))

    _extractors[name] = cls


def registered():
    return dict(_extractors)


# =============================================================================
# BASE CLASS
# =============================================================================

class ExtractorError(Exception):
    pass


class ExtractorMeta(type):

    def __new__(mcls, name, bases, namespace):
        cls = super(ExtractorMeta, mcls).__new__(mcls, name, bases, namespace)

        check = False
        try:
            check = cls != Extractor
        except NameError:
            pass

        if check:
            if not hasattr(cls, "data"):
                msg = "'{}' must redefine {}"
                raise ExtractorError(msg.format(cls, "data attribute"))
            if not cls.data:
                msg = "'data' can't be empty"
                raise ExtractorError(msg)
            for d in cls.data:
                if d not in DATAS:
                    msg = "'data' must be a iterable with values in {}. Found {}"
                    raise ExtractorError(msg.format(DATAS, cls.data))
            if len(set(cls.data)) != len(cls.data):
                msg = "'data' has duplicated values: {}"
                raise ExtractorError(msg.format(cls.data))

            if cls.fit == Extractor.fit:
                msg = "'{}' must redefine {}"
                raise ExtractorError(msg.format(cls, "fit method"))

            for c in cls.dependencies:
                if not inspect.isclass(c) or not issubclass(c, Extractor):
                    msg = ("All dependencies of one extractor must be "
                           "subclasses of Extractor.")
                    raise TypeError(msg)

            cls.data = frozenset(cls.data)
            cls.dependencies == frozenset(cls.dependencies)

        return cls


@six.add_metaclass(ExtractorMeta)
class Extractor(object):

    dependencies = []

    def setup(self):
        pass

    def fit(self):
        raise NotImplementedError()

    def teardown(self):
        pass

    def extract(self, data, features):
        kwargs = {fname: features[name] for k in self.dependencies}
        for d in self.data:
            idx = DATA_IDXS[d]
            kwargs[d] = data[idx]
        try:
            self.setup()
            return self.fit(**kwargs)
        finally:
            self.teardown()


# =============================================================================
# EXTRACTORS
# =============================================================================

class Amplitude(Extractor):
    """Half the difference between the maximum and the minimum magnitude"""

    data = ['magnitude']

    def fit(self, magnitude):
        N = len(magnitude)
        sorted_mag = np.sort(magnitude)

        return (np.median(sorted_mag[-math.ceil(0.05 * N):]) -
                np.median(sorted_mag[0:math.ceil(0.05 * N)])) / 2.0


class Rcs(Extractor):
    """Range of cumulative sum"""

    data = ['magnitude']

    def fit(self, magnitude):
        sigma = np.std(magnitude)
        N = len(magnitude)
        m = np.mean(magnitude)
        s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return R


class StetsonK(Extractor):
    data = ['magnitude', 'error']

    def fit(self, magnitude, error):
        mean_mag = (np.sum(magnitude/(error*error)) /
                    np.sum(1.0 / (error * error)))

        N = len(magnitude)
        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (magnitude - mean_mag) / error)

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

        return K


class Meanvariance(Extractor):
    """variability index"""

    data = ['magnitude']

    def fit(self, magnitude):
        return np.std(magnitude) / np.mean(magnitude)


class Autocor_length(Extractor):
    data = ['magnitude']

    def __init__(self, lags=100):
        self.nlags = lags

    def fit(self, magnitude):
        AC = stattools.acf(magnitude, nlags=self.nlags)
        k = next((index for index, value in
                 enumerate(AC) if value < np.exp(-1)), None)

        while k is None:
            self.nlags = self.nlags + 100
            AC = stattools.acf(magnitude, nlags=self.nlags)
            k = next((index for index, value in
                      enumerate(AC) if value < np.exp(-1)), None)

        return k


# =============================================================================
# REGISTERS
# =============================================================================

for cls in Extractor.__subclasses__():
    register_extractor(cls)
del cls
