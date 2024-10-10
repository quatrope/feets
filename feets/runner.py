#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# DOC
# =============================================================================

"""Functionalities for running multiple extractors in parallel."""


# =============================================================================
# IMPORTS
# =============================================================================

import copy

import dask
from dask.delayed import delayed

import numpy as np

__all__ = ["run"]


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DASK_OPTIONS = {"scheduler": "threads"}


# =============================================================================
# EXCEPTIONS
# =============================================================================


class DataRequiredError(ValueError):
    """Some required data is missing.

    Parameters
    ----------
    data : str
        The name of the required data that was not found.
    """

    def __init__(self, data):
        super().__init__(f"Required data '{data}' not found")


# =============================================================================
# RUNNER
# =============================================================================


@delayed
def _get_feature(results, feature):
    return results[feature]


def _preprocess_data(required_data, kwargs):
    datas = {}
    for required in required_data:
        data = kwargs.get(required)
        if data is None:
            raise DataRequiredError(required)
        datas[required] = np.asarray(data)

    return datas


def _extract_selected_features(extractors, data, selected_features):
    delayed_features = {}

    for extractor in extractors:
        kwargs = extractor.select_kwargs(data, delayed_features)

        results = delayed(extractor.extract_and_validate)(kwargs)

        for feature in extractor.get_features():
            delayed_features[feature] = _get_feature(results, feature)

    return {
        feature: delayed_features[feature] for feature in selected_features
    }


def run(
    *,
    extractors,
    selected_features,
    required_data,
    dask_options=None,
    lc,
):
    """Run the extractors on the given data and return the selected features.

    This function executes a series of feature extractors on provided data
    and returns a dictionary of the selected features. It assumes that the
    given extractors are sorted in order of their dependencies.

    Parameters
    ----------
    extractors : np.ndarray of Extractor
        List of extractor instances to run. Must be sorted in order of
        dependencies.
    selected_features : array-like of str
        The features to return.
    required_data : array-like of str
        The data required by the extractors.
    dask_options : dict, optional
        Options to be passed to the Dask scheduler.
    **kwargs
        The data to feed the extractors.

    Returns
    -------
    features : dict
        The extracted features.
        Example: {"feature1": 123, "feature2": 456}

    Raises
    ------
    DataRequiredError
        If some required data is missing.

    Notes
    -----
    The extractors are run in parallel using Dask:
    https://docs.dask.org/en/stable/
    """
    if dask_options is None:
        dask_options = copy.deepcopy(DEFAULT_DASK_OPTIONS)

    data = _preprocess_data(required_data, lc)

    delayed_features = _extract_selected_features(
        extractors, data, selected_features
    )

    (features,) = dask.compute(delayed_features, **dask_options)

    return features
