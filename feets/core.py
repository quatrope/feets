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

import numpy as np

from . import extractors, runner
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
        extractor_clss = extractors.extractor_registry.get_execution_plan(
            data=data, only=only, exclude=exclude
        )

        extractor_instances = []
        features = set()
        required_data = set()

        for extractor_cls in extractor_clss:
            extractor_instance = self._init_extractor(extractor_cls, **kwargs)

            extractor_instances.append(extractor_instance)
            features.update(extractor_instance.get_features())
            required_data.update(extractor_instance.get_data())

        self._extractors = np.array(extractor_instances, dtype=object)
        self._selected_features = frozenset(features if only is None else only)
        self._required_data = frozenset(required_data)

    def _init_extractor(self, extractor_cls, **kwargs):
        default_params = extractor_cls.get_default_params()
        params = {
            param: kwargs.get(param, default)
            for param, default in default_params.items()
        }
        return extractor_cls(**params)

    def __repr__(self):
        space = ", ".join(str(extractor) for extractor in self._extractors)
        return f"<FeatureSpace: {space}>"

    def extract(self, **kwargs):
        features = runner.run(
            extractors=self._extractors,
            selected_features=self._selected_features,
            required_data=self._required_data,
            **kwargs,
        )

        return FeatureSet("features", features)

    @property
    def features(self):
        return self._selected_features

    @property
    def execution_plan(self):
        return self._extractors
