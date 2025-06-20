#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities of feets."""

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

    Parameters
    ----------
    features : array_like
        The extracted features by light curve.
    extractors : array_like
        The extractors used to compute the features.

    Attributes
    ----------
    features : np.ndarray
        The extracted features by light curve.
    extractors : np.ndarray
        The extractors used to compute the features.
    feature_names : frozenset
        The names of the extracted features.
    length : int
        The number of light curves.
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
        """String representation of the Features object."""
        return f"<Features feature_names={set(self.feature_names)}, length={self.length}>"

    def __getattr__(self, feature_name):
        """Access feature values by using name as attribute."""
        return np.array([feat[feature_name] for feat in self.features])

    def __getitem__(self, slicer):
        """Access features by index or slice."""
        return self.features.__getitem__(slicer)

    def __len__(self):
        """Return the number of light curves."""
        return self.length

    def __dir__(self):
        """Return the list of attributes of the object."""
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
        """Return the features as a pandas DataFrame.

        Parameters
        ----------
        **kwargs
            Extra parameters that are passed to the joblib.Parallel constructor.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the extracted features by lightcurve.
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
