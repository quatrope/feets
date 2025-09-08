#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""Manage and manipulate feature extraction results."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections.abc import Sequence

import joblib

import numpy as np

import pandas as pd


# =============================================================================
# FEATURE SET
# =============================================================================


class Features(Sequence):
    """Class to manage and manipulate feature extraction results.

    The `Features` class encapsulates the results of feature extraction
    performed on multiple light curves. It provides an interface to access
    the extracted features either by feature name or by light curve index.

    Parameters
    ----------
    features : array_like of dict
        The results of the feature extraction for each of the light curves.
    extractors : array_like of Extractor
        The extractor instances used to compute the features.

    Attributes
    ----------
    features : np.ndarray
        The extracted features by light curve.
    extractors : np.ndarray
        The extractor instances used to compute the features.
    feature_names : frozenset
        The names of the extracted features.
    length : int
        The number of light curves.

    Examples
    --------
    >>> from feets import FeatureSpace
    >>> fs = FeatureSpace(only=["Std", "Mean"])
    >>> results = fs.extract_many(
    ...     {"magnitude": [1, 1.5, 2]},
    ...     {"magnitude": [1, 2, 3]}
    ... )
    >>> results
    <Features feature_names={'Std', 'Mean'}, length=2>

    Accessing results by feature name:

    >>> results.Mean
    array([1.5, 2. ])
    >>> results.Std
    array([0.5, 1. ])

    Accessing results by light curve index:

    >>> results[0]
    {'Std': np.float64(0.5), 'Mean': np.float64(1.5)}
    >>> results[1]
    {'Std': np.float64(1.0), 'Mean': np.float64(2.0)}
    """

    # CONSTRUCTOR =============================================================

    def __init__(self, features, extractors):
        self.features = np.array(features, dtype=dict)
        self.extractors = np.array(extractors, dtype=object)

    # PROPERTIES ==============================================================

    @property
    def feature_names(self):
        """frozenset: The names of the extracted features."""
        return frozenset(self.features[0])

    @property
    def length(self):
        """int: The number of light curves."""
        return len(self.features)

    # MAGIC ===================================================================

    def __repr__(self):
        """String representation of the `Features` object."""
        return (
            f"<Features feature_names={set(self.feature_names)}, "
            f"length={self.length}>"
        )

    def __getattr__(self, feature_name):
        """Access feature results by feature name."""
        try:
            return np.array([feat[feature_name] for feat in self.features])
        except KeyError:
            message = (
                f"{type(self).__name__!r} object has no feature by the name "
                f"{feature_name!r}"
            )
            raise AttributeError(message)

    def __getitem__(self, slicer):
        """Access light curve results by index or slice."""
        try:
            return self.features.__getitem__(slicer)
        except IndexError:
            message = (
                f"index {slicer} is out of bounds for "
                f"{type(self).__name__!r} object with length {self.length}"
            )
            raise IndexError(message)

    def __len__(self):
        """Get the number of light curves."""
        return self.length

    def __dir__(self):
        """Get the list of attributes of the `Features` object."""
        return list(vars(type(self))) + list(self.feature_names)

    # API =====================================================================

    def _extractors_by_feature(self):
        extractors_by_feature = {}
        for extractor in self.extractors:
            extractor_by_feature = dict.fromkeys(
                extractor.get_features(), extractor
            )
            extractors_by_feature.update(extractor_by_feature)

        return extractors_by_feature

    def _get_default_jobs(self):
        jobs = min(len(self.features), joblib.cpu_count())
        return jobs

    @staticmethod
    def _features_as_serie(features, extractors_by_feature):
        data = {}
        for fname, fvalue in features.items():
            extractor = extractors_by_feature[fname]
            flattened = extractor.flatten_feature(fname, fvalue)
            extractor.validate_flatten(fname, flattened)
            data.update(flattened)
        return pd.Series(data)

    def as_frame(self, **kwargs):
        """Convert the extraction results into a `pandas.DataFrame`.

        This method transforms the extracted features into a `pandas.DataFrame`,
        where each row corresponds to a light curve and each column represents
        a feature.

        The conversion process can be parallelized to improve performance on
        large datasets.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the `joblib.Parallel` constructor,
            used when parallel processing the `pandas.DataFrame` conversion.

        Returns
        -------
        pandas.DataFrame
            A `pandas.DataFrame` representation of the extracted features.
            Each row corresponds to a light curve and each column represents
            a feature.

        Examples
        --------
        >>> from feets import FeatureSpace
        >>> fs = FeatureSpace(only=["Std", "Mean"])
        >>> results = fs.extract_many(
        ...     {"magnitude": [1, 1.5, 2]},
        ...     {"magnitude": [1, 2, 3]}
        ... )
        >>> results.as_frame()
        Features     Std  Mean
        Light Curve
        0            0.5   1.5
        1            1.0   2.0
        """
        extractors_by_feature = self._extractors_by_feature()

        kwargs.setdefault("prefer", "processes")
        kwargs.setdefault("n_jobs", self._get_default_jobs())

        with joblib.Parallel(**kwargs) as P:
            features_as_serie = joblib.delayed(self._features_as_serie)
            all_series = P(
                features_as_serie(features, extractors_by_feature)
                for features in self.features
            )
        df = pd.DataFrame(all_series)
        df.index.name = "Light Curve"
        df.columns.name = "Features"
        return df
