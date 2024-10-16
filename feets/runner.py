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
import dask.bag as db
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


def _run_single(
    *,
    extractors,
    selected_features,
    required_data,
    lc,
):

    data = _preprocess_data(required_data, lc)
    delayed_features = _extract_selected_features(
        extractors, data, selected_features
    )
    return delayed_features


def run(
    *,
    extractors,
    selected_features,
    required_data,
    dask_options=None,
    lcs,
):
    """Run extractors and select features from the given light curves.

    This function runs a series of feature extractors on the provided light
    curves and selects only the desired features. The result is a list
    containing the selected features for each light curve.

    The extractors should be sorted based on their dependencies to ensure
    proper execution.

    Parameters
    ----------
    extractors : np.ndarray of Extractor
        Array of extractor instances to run. Must be sorted based on dependencies.
    selected_features : array-like of str
        The features to extract.
    required_data : array-like of str
        The data required by the extractors.
    dask_options : dict, optional
        Options to be passed to the Dask scheduler.
    lcs : list of dict
        The light curves to process.

    Returns
    -------
    list of dict
        The extracted features for each light curve. The order of the list is preserved.

    Raises
    ------
    DataRequiredError
        If any required data is missing from the light curves.

    Examples
    --------
    >>> import numpy as np
    >>> from feets.extractors.ext_mean import Mean
    >>> lcs = [{"magnitude": [1, 2, 3]}, {"magnitude": [4, 5, 6]}]
    >>> run(extractors=np.array([Mean()]),
    ...     selected_features=["Mean"],
    ...     required_data=["magnitude"],
    ...     lcs=lcs)
    [{'Mean': np.float64(2.0)}, {'Mean': np.float64(5.0)}]

    Notes
    -----
    The feature extraction is performed in parallel using Dask, and can be
    configured using the `dask_options` parameter.

    For more information on Dask, visit: https://docs.dask.org/en/stable/
    """
    if dask_options is None:
        dask_options = copy.deepcopy(DEFAULT_DASK_OPTIONS)

    delayed_features_by_lc = db.from_sequence(lcs).map(
        lambda lc: dask.compute(
            _run_single(
                extractors=extractors,
                selected_features=selected_features,
                required_data=required_data,
                lc=lc,
            ),
            **dask_options,
        )[0]
    )

    return delayed_features_by_lc.compute(**dask_options)
