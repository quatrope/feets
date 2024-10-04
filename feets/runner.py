# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import dask
from dask.delayed import delayed

# =============================================================================
# EXCEPTIONS
# =============================================================================


class DataRequiredError(ValueError):
    pass


# =============================================================================
# RUNNER
# =============================================================================


def _preprocess_data(required_data, kwargs):
    return {
        data: np.asarray(kwargs[data])
        for data in required_data
        if kwargs.get(data) is not None
    }


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
