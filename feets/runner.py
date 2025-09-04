#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# DOC
# =============================================================================

"""Run multiple feature extractors in parallel."""


# =============================================================================
# IMPORTS
# =============================================================================

import copy

import dask
from dask.delayed import delayed

__all__ = ["run", "DataRequiredError"]


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DASK_OPTIONS = {"scheduler": "processes"}

# =============================================================================
# EXCEPTIONS
# =============================================================================


class DataRequiredError(ValueError):
    """A required data vector is missing from the light curve."""

    pass


# =============================================================================
# RUNNER
# =============================================================================


def _validate_required_data_single(*, required_data, lc):
    missing_data = set(required_data).difference(lc)
    if missing_data:
        missing_str = ", ".join(missing_data)
        raise DataRequiredError(
            f"Missing required data vectors in light curve: {missing_str}"
        )


def _validate_required_data(*, required_data, lcs, dask_options):
    validations = [
        _validate_required_data_single(
            required_data=required_data,
            lc=lc,
        )
        for lc in lcs
    ]

    dask.compute(*validations, **dask_options)


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
    required_data,
    lcs,
    dask_options=None,
):
    """Run instances of feature extractors on a collection of light curves.

    Executes the specified extractor instances on each provided light curve,
    returning the extracted features for each. Feature extraction is performed
    in parallel using Dask, enabling efficient computation across multiple
    light curves. The order of execution respects dependencies between
    extractors; ensure that the `extractors` list is topologically sorted so
    that dependencies are satisfied.

    Parameters
    ----------
    extractors : array_like of feets.extractors.Extractor
        Feature extractor instances to apply. Must be sorted so that any
        extractor appears after those it depends on.
    selected_features : array_like of str
        Names of features to extract from each light curve.
    required_data : array_like of str
        Names of required data fields that must be present in each light curve.
    lcs : array_like of dict
        Light curves to process, each represented as a dictionary of data
        vectors.
    dask_options : dict, optional
        Options for the Dask scheduler. Defaults to
        ``{"scheduler": "processes"}``.

    Returns
    -------
    list of dict
        List of dictionaries, one per input light curve, with the extracted
        feature values. Each dictionary contains the extracted features
        specified in `selected_features`. The order of the list matches the
        input `lcs`.

    Raises
    ------
    DataRequiredError
        If any of the required data vectors are missing from a light curve.

    See Also
    --------
    feets.Extractor : Abstract base class for feature extractors.
    feets.FeatureSpace :
        Class to select and extract features from a time series.
    dask.compute

    Notes
    -----
    Feature extraction is parallelized using Dask. You can control parallelism
    and scheduler behavior via the `dask_options` parameter.

    For more information on Dask, see: https://docs.dask.org/en/stable/

    Examples
    --------
    >>> from feets.extractors import Mean
    >>>
    >>> # Instantiate the feature extractor
    >>> mean_extractor = Mean()
    >>>
    >>> # Light curves to process
    >>> lcs = [{"magnitude": [1, 2, 3]}, {"magnitude": [4, 5, 6]}]
    >>>
    >>> # Run the feature extraction
    >>> run(
    ...     extractors=[mean_extractor],
    ...     selected_features=["Mean"],
    ...     required_data=["magnitude"],
    ...     lcs=lcs
    ... )
    [{'Mean': np.float64(2.0)}, {'Mean': np.float64(5.0)}]
    """
    if dask_options is None:
        dask_options = copy.deepcopy(DEFAULT_DASK_OPTIONS)

    _validate_required_data(
        required_data=required_data, lcs=lcs, dask_options=dask_options
    )

    delayed_features_by_lc = [
        _run_single(
            extractors=extractors,
            selected_features=selected_features,
            lc=lc,
        )
        for lc in lcs
    ]

    features_by_lc = dask.compute(*delayed_features_by_lc, **dask_options)

    return list(features_by_lc)
