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
    "FeatureSpaceError",
    "FeatureSpace",
]


# =============================================================================
# IMPORTS
# =============================================================================

import copy
import itertools as it
from collections import Counter
from collections.abc import Mapping

import attr

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from . import extractors
from .extractors.core import (
    DATAS,
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

# =============================================================================
# EXCEPTIONS
# =============================================================================


class FeatureNotFound(ValueError):
    """Raises when a non-available feature are requested.

    A non-available feature can be:

    - The feature don't exist in any of the registered extractor.
    - The feature can't be requested with the available data.

    """


class DataRequiredError(ValueError):
    """Raised when the feature-space required another data."""


class FeatureSpaceError(ValueError):
    """The FeatureSpace can't be configured with the given parameters."""


# =============================================================================
# RESULTSET
# =============================================================================


class _Map(Mapping):
    """Internal representation of a immutable dict"""

    def __init__(self, d):
        self._keys = tuple(d.keys())
        self._values = tuple(d.values())

    def __getitem__(self, k):
        """x.__getitem__(y) <==> x[y]"""
        if k not in self._keys:
            raise KeyError(k)
        idx = self._keys.index(k)
        return self._values[idx]

    def __iter__(self):
        """x.__iter__() <==> iter(x)"""
        return iter(self._keys)

    def __len__(self):
        """x.__len__() <==> len(x)"""
        return len(self._keys)


@attr.s(frozen=True, auto_attribs=True, repr=False)
class FeatureSet:
    """Container of features.

    The FeatureSet object is capable of convert the features into
    dicts, numpy arrays and also provides
    analysis capabilities like plots thought the matplotlib
    and seaborn library.

    """

    features_names: tuple = attr.ib(converter=tuple)
    values: dict = attr.ib(converter=_Map)
    extractors: dict = attr.ib(converter=_Map)
    timeserie: dict = attr.ib(converter=_Map)

    def __attrs_post_init__(self):
        cnt = Counter(
            it.chain(self.features_names, self.values, self.extractors)
        )
        diff = set(k for k, v in cnt.items() if v < 3)
        if diff:
            joined_diff = ", ".join(diff)
            raise FeatureNotFound(
                f"The features '{joined_diff}' must be in 'features_names' "
                "'values' and 'extractors'"
            )

    def __iter__(self):
        """x.__iter__() <==> iter(x)"""
        return iter(self.as_arrays())

    def __getitem__(self, k):
        """x.__getitem__(y) <==> x[y]"""
        return copy.deepcopy(self.values[k])

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        feats = ", ".join(self.features_names)
        ts = ", ".join(d for d in DATAS if self.timeserie.get(d) is not None)
        return f"FeatureSet(features=<{feats}>, timeserie=<{ts}>)"

    def extractor_of(self, feature):
        """Retrieve the  extractor instance used for create the feature."""
        return copy.deepcopy(self.extractors[feature])

    def plot(self, feature, ax=None, **plot_kws):
        """If is available, draw the plot-representation of the feature.

        Parameters
        ----------

        feature : str
            The feature to plot.
        ax : matplotlib axes object, default None.
        `**plot_kws` : keywords
            Options to pass to extractor and matplotlib plotting method.

        Returns
        -------
        axes : matplotlib.axes.Axes or np.ndarray of them

        """
        ax = plt.gca() if ax is None else ax

        all_features = self.as_dict()
        extractor = self.extractor_of(feature)
        value = self[feature]
        try:
            ax = extractor.plot(
                feature=feature,
                value=value,
                ax=ax,
                plot_kws=plot_kws,
                features=all_features,
                **self.timeserie,
            )
        except NotImplementedError:
            ax.remove()
            raise

        return ax

    def as_arrays(self):
        """Convert the feature values into two 1D numpy arrays.

        The first one contains all the names of the features (with suffixes
        if the array was flattened). And the second one the values.

        Internally this method uses the ``flatten_feature()`` method.

        """

        all_features, flatten_features = self.as_dict(), {}

        for fname, fvalue in self.values.items():

            extractor = self.extractors[fname]

            flatten_value = extractor.flatten(
                feature=fname,
                value=fvalue,
                features=all_features,
                **self.timeserie,
            )

            flatten_features.update(flatten_value)

        features = np.empty(len(flatten_features), dtype=object)
        values = np.empty(len(flatten_features))
        for idx, fv in enumerate(flatten_features.items()):
            features[idx], values[idx] = fv

        return features, values

    def as_dict(self):
        """Return a copy of values"""
        return dict(self.values)

    def as_dataframe(self):
        """Convert the entire features into a ``pandas.DataFrame``.

        The multimensional features are first *flattened* with the
        ``flatten_feature()`` method.

        """
        features, values = self.as_arrays()
        return pd.DataFrame([values], columns=features)


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
        >>> features = fs.extract(*lc)
        >>> features.as_dict()
        {"Std": .42}

    **Available data as an input:**

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(data=['magnitude','time'])
        >>> features = fs.extract(**lc)
        >>> features.as_dict()
        {...}

    **List of features and available data as an input:**

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(
        ...     only=['Mean','Beyond1Std', 'CAR_sigma','Color'],
        ...     data=['magnitude', 'error'])
        >>> features = fs.extract(**lc)
        >>> features.as_dict()
        {"Beyond1Std": ..., "Mean": ...}

    **Excluding list as an input**

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(
        ...     only=['Mean','Beyond1Std','CAR_sigma','Color'],
        ...     data=['magnitude', 'error'],
        ...     exclude=["Beyond1Std"])
        >>> features = fs.extract(**lc)
        >>> features.as_dict()
        {"Mean": 23}

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
                if not f.get_required_data().difference(data):
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
            self._only
        ).difference(self._exclude)

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
                required_data.update(fext.get_required_data())

        if not features_extractors:
            raise FeatureSpaceError("No feature extractor was selected")

        self._features_extractors = frozenset(features_extractors)
        self._features_extractors_names = frozenset(features_extractors_names)
        self._required_data = frozenset(required_data)

        # excecution order by dependencies
        self._execution_plan = extractors.sort_by_dependencies(
            features_extractors
        )

        not_found = set(self._kwargs).difference(
            self._features_extractors_names
        )
        if not_found:
            joined_not_found = ", ".join(not_found)
            raise FeatureNotFound(
                "This space not found feature(s) extractor(s) "
                f"{joined_not_found} to assign the given parameter(s)"
            )

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return str(self)

    def __str__(self):
        """x.__str__() <==> str(x)"""
        if not hasattr(self, "__str"):
            extractors = [str(extractor) for extractor in self._execution_plan]
            space = ", ".join(extractors)
            self.__str = "<FeatureSpace: {}>".format(space)
        return self.__str

    def preprocess_timeserie(self, d):
        """Validate if the required values of the time-serie exist with
        non ``None`` values in the dict ``d``. Finally returns a
        new dictionary whose non-null values have been converted to
        ``np.ndarray``

        """
        array_data = {}
        for k, v in d.items():
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
        """Extract the features from a given time-series.

        This method must be provided with the required timeseries data
        specified in the attribute ``required_data_``.

        Parameters
        ----------
        time : iterable, optional
        magnitude : iterable, optional
        error : iterable, optional
        magnitude2 : iterable, optional
        aligned_time : iterable, optional
        aligned_magnitude : iterable, optional
        aligned_magnitude2 : iterable, optional
        aligned_error : iterable, optional
        aligned_error2 : iterable, optional

        Returns
        -------
        feets.core.FeatureSet
            Container of a calculated features.

        """
        timeserie = self.preprocess_timeserie(
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

        features, extractors = {}, {}
        for fextractor in self._execution_plan:
            result = fextractor.extract(features=features, **timeserie)
            for fname, fvalue in result.items():
                features[fname] = fvalue
                extractors[fname] = copy.deepcopy(fextractor)

        # remove all the not needed features and extractors
        flt_features, flt_extractors = {}, {}
        for fname in self._features_as_array:
            flt_features[fname] = features[fname]
            flt_extractors[fname] = extractors[fname]

        rs = FeatureSet(
            features_names=self._features_as_array,
            values=flt_features,
            extractors=flt_extractors,
            timeserie=timeserie,
        )
        return rs

    @property
    def extractors_conf(self):
        return copy.deepcopy(self._kwargs)

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
