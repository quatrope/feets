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

__doc__ = """core functionalities of feets"""

__all__ = ["FeatureSpace"]


# =============================================================================
# IMPORTS
# =============================================================================

import logging
from itertools import chain

import numpy as np

import ray

from . import extractors
from .libs import bunch

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


class DataRequiredError(ValueError):
    pass


# =============================================================================
# FEATURE SET
# =============================================================================


class FeatureSet(bunch.Bunch):
    pass


# =============================================================================
# FEATURE SPACE
# =============================================================================


def _feature_from_extraction(extraction, feature):
    return extraction[feature]


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
        ray.init()

        extractor_clss = extractors.register.get_execution_plan(
            data=data, only=only, exclude=exclude
        )

        feature_extractors = []
        required_data = set()
        for extractor_cls in extractor_clss:
            extractor, features, data = self.init_extractor(
                extractor_cls, only=only, **kwargs
            )
            feature_extractors.append((extractor, features))
            required_data.update(data)

        self._feature_extractors = np.asarray(feature_extractors)
        self._extractors = np.asarray(self._feature_extractors[:, 0])
        self._selected_features = frozenset(
            chain.from_iterable(self._feature_extractors[:, 1])
        )
        self._required_data = frozenset(required_data)

    def init_extractor(self, extractor_cls, only=None, **kwargs):
        # parameters needed to initialize the extractor
        default_params = extractor_cls.get_default_params().items()
        extractor_params = {
            pname: kwargs.get(pname, pvalue)
            for pname, pvalue in default_params
        }

        # initialize the extractor
        extractor = extractor_cls(**extractor_params)

        # expected features
        features = extractor_cls.get_features()
        if only is not None:
            features = features.intersection(only)

        # required data
        data = extractor_cls.get_data()

        return extractor, features, data

    def __repr__(self):
        space = ", ".join(str(extractor) for extractor in self._extractors)
        return f"<FeatureSpace: {space}>"

    def extract(self, **kwargs):
        for key in self._required_data:
            if kwargs.get(key, None) is None:
                raise DataRequiredError(key)
        data = {key: np.asarray(value) for key, value in kwargs.items()}

        actor_refs = []
        result_refs_by_feature = {}
        for extractor, features in self._feature_extractors:
            actor_ref = extractors.actor.ExtractorActor.remote(
                extractor, result_refs_by_feature
            )
            actor_refs.append(actor_ref)

            result_ref = actor_ref.select_extract_and_validate.remote(data)

            for feature in features:
                result_refs_by_feature[feature] = result_ref

        features = {}
        for feature in self._selected_features:
            result_ref = result_refs_by_feature[feature]
            result = ray.get(result_ref)
            features[feature] = result[feature]

        for actor_ref in actor_refs:
            ray.kill(actor_ref)

        return FeatureSet("features", features)

    @property
    def features(self):
        return self._selected_features
