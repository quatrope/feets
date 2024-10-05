#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import dask
from dask.delayed import delayed

import numpy as np

# =============================================================================
# EXCEPTIONS
# =============================================================================


class DataRequiredError(ValueError):
    pass


# =============================================================================
# RUNNER
# =============================================================================


def _preprocess_data(required_data, kwargs):
    datas = {}
    for required in required_data:
        data = kwargs.get(required)
        if data is None:
            raise DataRequiredError(f"Required data {required} not found")
        datas[required] = np.asarray(data)

    return datas

@delayed
def _get_feature(results, feature):
    return results[feature]


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


def run(*, extractors, selected_features, required_data, **kwargs):
    data = _preprocess_data(required_data, kwargs)

    delayed_features = _extract_selected_features(
        extractors, data, selected_features
    )

    (features,) = dask.compute(delayed_features)

    return features
