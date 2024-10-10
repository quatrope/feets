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
import logging

import attrs

import joblib

import numpy as np

import pandas as pd

from . import extractors, runner

__all__ = ["FeatureSpace"]


# =============================================================================
# LOG
# =============================================================================

logger = logging.getLogger("feets")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


# =============================================================================
# FEATURE SET
# =============================================================================


@attrs.define(frozen=True)
class Features(Sequence):
    features: tuple = attrs.field(converter=np.array, repr=False)
    extractors: np.ndarray = attrs.field(repr=False, converter=tuple)
    feature_names: np.ndarray = attrs.field(init=False, repr=True)
    length: int = attrs.field(init=False, repr=True)

    @feature_names.default
    def _feature_names_defaults(self):
        return tuple(set(self.features[0]))

    @length.default
    def _length_defaults(self):
        return len(self.features)

    def __attrs_post_init__(self):
        self.features.setflags(write=False)

    def __getattr__(self, a):
        """Allow to access the features as attributes."""
        return np.array([feat[a] for feat in self.features])

    def __getitem__(self, slicer):
        return self.features.__getitem__(slicer)

    def __len__(self):
        return self.length

    def __dir__(self):
        return list(vars(type(self))) + list(self.feature_names)

    def _extractors_by_features(self):
        all_extractors_by_features = {}
        for extractor in self.extractors:
            extractor_by_feature = dict.fromkeys(
                extractor.get_features(), extractor
            )
            all_extractors_by_features.update(extractor_by_feature)
        return all_extractors_by_features

    def _get_default_jobs(self):
        jobs = min(len(self.features), joblib.cpu_count())
        return jobs

    def _features_as_serie(self, features, extractors_by_feature):
        data = {}
        for fname, fvalue in features.items():
            extractor = extractors_by_feature[fname]
            fflattened = extractor.flatten_feature(fname, fvalue)
            data.update(fflattened)
        return pd.Series(data)

    def as_frame(self, **kwargs):
        # a = [1, 2, 3] ==> ["a_0": 1, "a_1": 2, "a_2": 3]
        # b = {"hola": [1,2,3], "chau": 1} ==>
        #   ["b_hola_0": 1, "b_hola_1": 2, "b_hola_2": 3, "b_chau": 1]

        extractors_by_features = self._extractors_by_features()

        kwargs.setdefault("prefer", "processes")
        kwargs.setdefault("n_jobs", self._get_default_jobs())

        with joblib.Parallel(**kwargs) as P:
            features_as_serie = joblib.delayed(self._features_as_serie)
            all_series = P(
                features_as_serie(features, extractors_by_features)
                for features in self.features
            )
        df = pd.DataFrame(all_series)
        df.columns.name = "Features"
        return df


# =============================================================================
# FEATURE SPACE
# =============================================================================


class FeatureSpace:
    """Class to manage the selection and extraction of features from a time series.

    The `FeatureSpace` class allows for the selection of features based on the
    available time series vectors (e.g., magnitude, time, error, second magnitude),
    or on a specified list of features.

    The final set of features for the execution plan are those that satisfy all
    the provided filters. If no filter is provided, all features are included.

    Parameters
    ----------
    data : array_like, optional
        List of available time series vectors to be used by the feature
        extractors. If provided, only the extractors that require some subset
        of the selected data will be included.
    only : array_like, optional
        List of features to be included in the output. If provided, only the
        selected features will be extracted.
    exclude : array_like, optional
        List of features to be excluded from the output. If provided, all
        features except the selected ones will be extracted. It must be
        disjoint with `only`.
    **kwargs
        Extra parameters that are passed to the feature extractors.

    Methods
    -------
    extract(**kwargs)
        Extract all the selected features from the provided data.

    Examples
    --------
    **List of features as an input:**

    >>> fs = feets.FeatureSpace(only=['Std'])
    >>> fs.extract(**lc)
    <features {'Std'}>

    **List of available data as an input:**

    >>> fs = feets.FeatureSpace(data=['magnitude','time'])
    >>> fs.extract(**lc)
    <features {...}>

    **List of features and available data as an input:**

    >>> fs = feets.FeatureSpace(
    ...     only=['Mean','Beyond1Std', 'CAR_sigma','Color'],
    ...     data=['magnitude', 'error'])
    >>> fs.extract(**lc)
    <features {'Mean', 'Beyond1Std'}>

    **List of exclusions as an input:**

    >>> fs = feets.FeatureSpace(data=['magnitude'])
    >>> fs.extract(**lc)
    <features {'Mean', 'Std', ...}>
    >>> fs = feets.FeatureSpace(data=['magnitude'], exclude=['Mean'])
    >>> fs.extract(**lc)
    <features {'Std', ...}>
    """

    def __init__(
        self, data=None, only=None, exclude=None, dask_options=None, **kwargs
    ):
        extractor_clss = extractors.extractor_registry.get_execution_plan(
            data=data, only=only, exclude=exclude
        )

        extractor_instances = []
        selected_features = set()
        required_data = set()

        for extractor_cls in extractor_clss:
            extractor_instance = self._init_extractor(extractor_cls, **kwargs)
            extractor_instances.append(extractor_instance)

            features = extractor_instance.get_features()
            if only is not None:
                features = features.intersection(only)
            selected_features.update(features)

            required_data.update(extractor_instance.get_data())

        self._extractors = np.array(extractor_instances, dtype=object)
        self._selected_features = frozenset(selected_features)
        self._required_data = frozenset(required_data)
        self._dask_options = dask_options

    def _init_extractor(self, extractor_cls, **kwargs):
        default_params = extractor_cls.get_default_params()
        params = {
            param: kwargs.get(param, default)
            for param, default in default_params.items()
        }
        return extractor_cls(**params)

    def __repr__(self):
        """Return a string representation of the FeatureSpace object."""
        space = ", ".join(str(extractor) for extractor in self._extractors)
        return f"<FeatureSpace: {space}>"

    def extract(self, *lcs, **lc):
        """Extract all the selected features from the provided data.

        Parameters
        ----------
        dask_options : dict, optional
            Options to be passed to the Dask scheduler.
        **kwargs
            The time series data required by the extractors.

        Returns
        -------
        FeatureSet
            A collection of extracted features.

        Examples
        --------
        >>> fs = feets.FeatureSpace(only=['Std'])
        >>> fs.extract(**lc)
        <features {'Std'}>
        """
        if lc and lcs:
            raise ValueError("O una curva separadas o muchas en diccionarios")

        lcs = [lc] if lc else lcs

        features = runner.run(
            extractors=self._extractors,
            selected_features=self._selected_features,
            required_data=self._required_data,
            dask_options=self._dask_options,
            lc=lc,
        )
        return Features(features=[features], extractors=self._extractors)

    @property
    def features(self):
        """frozenset: The selected features."""
        return self._selected_features

    @property
    def execution_plan(self):
        """np.ndarray: The extractor instances in order of their dependencies."""
        return self._extractors
