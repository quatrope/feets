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

__all__ = ["run"]


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DASK_OPTIONS = {"scheduler": "processes"}


# =============================================================================
# RUNNER
# =============================================================================


@delayed
def _get_feature(results, feature):
    return results[feature]


@delayed
def _extract_and_validate(extractor, kwargs):
    results = extractor.extract(**kwargs)
    extractor.validate_extract(results)
    return results


def _extract_selected_features(extractors, data, selected_features):
    delayed_features = {}

    for extractor in extractors:
        kwargs = extractor.prepare_extract(data, delayed_features)
        delayed_results = _extract_and_validate(extractor, kwargs)
        for feature in extractor.get_features():
            delayed_features[feature] = _get_feature(delayed_results, feature)

    return {
        feature: delayed_features[feature] for feature in selected_features
    }


def _run_single(*, extractors, selected_features, lc):
    delayed_features = _extract_selected_features(
        extractors, lc, selected_features
    )

    return delayed_features


def run(
    *,
    extractors,
    selected_features,
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
    dask_options : dict, optional
        Options to be passed to the Dask scheduler.
    lcs : list of dict
        The light curves to process.

    Returns
    -------
    list of dict
        The extracted features for each light curve. The order of the list is preserved.

    Examples
    --------
    >>> import numpy as np
    >>> from feets.extractors.ext_mean import Mean
    >>> lcs = [{"magnitude": [1, 2, 3]}, {"magnitude": [4, 5, 6]}]
    >>> run(extractors=np.array([Mean()]),
    ...     selected_features=["Mean"],
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

    delayed_features_by_lc = [
        _run_single(
            extractors=extractors,
            selected_features=selected_features,
            lc=lc,
        )
        for lc in lcs
    ]

    features_by_lc = dask.compute(*delayed_features_by_lc, **dask_options)

    return features_by_lc
