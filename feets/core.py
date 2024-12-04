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

import logging
from collections.abc import Sequence

import joblib

import numpy as np

import pandas as pd

from . import extractors, runner

__all__ = ["Features", "FeatureSpace"]


# =============================================================================
# LOG
# =============================================================================

logger = logging.getLogger("feets")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


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
        df.index.name = "Light Curve"
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

    Attributes
    ----------
    features : frozenset
        The selected features.
    extractors : np.ndarray
        The extractor instances in order of their dependencies.
    required_data : frozenset
        The data vectors required by the extractors.
    dask_options : dict
        Options to be passed to the Dask scheduler.

    Examples
    --------
    **List of features as an input:**

    >>> fs = feets.FeatureSpace(only=['Std'])
    >>> fs.extract(**lc)
    <Features feature_names={'Std'}, length=1>

    **List of available data as an input:**

    >>> fs = feets.FeatureSpace(data=['magnitude','time'])
    >>> fs.extract(**lc)
    <Features feature_names={...}, length=1>

    **List of features and available data as an input:**

    >>> fs = feets.FeatureSpace(
    ...     only=['Mean','Beyond1Std', 'CAR_sigma','Color'],
    ...     data=['magnitude', 'error'])
    >>> fs.extract(**lc)

    >>> fs = feets.FeatureSpace(data=['magnitude','time'])
    >>> fs.extract(**lc)
    <Features feature_names={'Mean', 'Beyond1Std'}, length=1>

    **List of exclusions as an input:**

    >>> fs = feets.FeatureSpace(data=['magnitude'])
    >>> fs.extract(**lc)
    <Features feature_names={'Mean', 'Std', ...}, length=1>
    >>> fs = feets.FeatureSpace(data=['magnitude'], exclude=['Mean'])
    >>> fs.extract(**lc)
    <Features feature_names={'Std', ...}, length=1>
    """

    # CONSTRUCTOR =============================================================

    def _init_extractor(self, extractor_cls, **kwargs):
        default_params = extractor_cls.get_default_params()
        params = {
            param: kwargs.get(param, default)
            for param, default in default_params.items()
        }
        return extractor_cls(**params)

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

            required_data.update(extractor_instance.get_required_data())

        self._extractors = np.array(extractor_instances, dtype=object)
        self._selected_features = frozenset(selected_features)
        self._required_data = frozenset(required_data)
        self.dask_options = dask_options

    # FROM LC =================================================================

    @classmethod
    def _coerce_lightcurves(cls, *, single_lc, multiple_lc):
        if single_lc and multiple_lc:
            raise ValueError(
                "Please provide either a single light curve or a list of light "
                "curves, but not both."
            )

        return [single_lc] if single_lc else multiple_lc

    @classmethod
    def from_lightcurves(cls, *lcs, **lc):
        """Return a FeatureSpace object from a set of light curves data \
        present in one of multiple lightcurves.

        Parameters
        ----------
        *lcs : optional
            A list of light curves represented as dictionaries.
        **lc : optional
            A single light curve represented as a dictionary.

        Returns
        -------
        FeatureSpace
            A FeatureSpace object with the features that can be extracted from
            the provided light curves.

        Examples
        --------
        >>> fs = feets.FeatureSpace.from_lightcurves(lc)
        >>> fs.extract(**lc)
        Features(feature_names={...}, length=1)

        >>> fs = feets.FeatureSpace.from_lightcurves(*lcs)
        >>> fs.extract(*lcs)
        Features(feature_names={...}, length=1)

        """
        lcs = cls._coerce_lightcurves(single_lc=lc, multiple_lc=lcs)
        selected_data = set(extractors.DATAS)
        for lc in lcs:
            selected_data.intersection_update(lc)
        return cls(data=selected_data)

    # PROPERTIES ==============================================================

    @property
    def selected_features(self):
        """frozenset: The selected features."""
        return self._selected_features

    @property
    def extractors(self):
        """np.ndarray: The extractor instances in order of their dependencies."""
        return self._extractors

    @property
    def required_data(self):
        """frozenset: The data vectors required by the extractors."""
        return self._required_data

    # MAGIC ===================================================================

    def __repr__(self):
        """String representation of the FeatureSpace object."""
        space = ", ".join(str(extractor) for extractor in self._extractors)
        return f"<FeatureSpace: {space}>"

    # PERSISTENCE ==============================================================

    @classmethod
    def from_dict(cls, data):
        """Create a FeatureSpace instance from a dictionary representation.

        Parameters
        ----------
        data : dict
            A dictionary representation of the feature space, including selected
            features, required data, dask options, and extractors.

        Returns
        -------
        FeatureSpace
            A FeatureSpace object with the features, required data, dask options,
            and extractors from the provided dictionary.
        """
        only = data["selected_features"]
        dask_options = data["dask_options"]
        kwargs = {}
        for extractor in data["extractors"]:
            name = list(extractor).pop()
            kwargs.update(extractor[name])

        return cls(
            only=only,
            dask_options=dask_options,
            **kwargs,
        )

    def to_dict(self):
        """Convert the feature space to a dictionary representation.

        Returns
        -------
        dict
            A dictionary representation of the feature space, including selected
            features, required data, dask options, and extractors.
        """
        return {
            "selected_features": list(self._selected_features),
            "required_data": list(self._required_data),
            "dask_options": self.dask_options,
            "extractors": [
                extractor.to_dict() for extractor in self._extractors
            ],
        }

    def to_json(self, *, path_or_buffer=None, **kwargs):
        """Serialize the feature space to a JSON formatted string or file.

        Parameters
        ----------
        path_or_buffer : str, pathlib.Path, file-like object or None, optional
            The file path or buffer to write the JSON data to. If `None`, the JSON
            data is returned as a string. Defaults to `None`.
        **kwargs
            Additional parameters to pass to `io.store_json`.

        Returns
        -------
        str
            The JSON formatted string if `path_or_buffer` is None.
        """
        from . import io  # noqa

        return io.store_json(self, path_or_buffer=path_or_buffer, **kwargs)

    def to_yaml(self, *, path_or_buffer=None, **kwargs):
        """Serialize the feature space to a YAML formatted string or file.

        Parameters
        ----------
        path_or_buffer : str, pathlib.Path, file-like object or None, optional
            The file path or buffer to write the YAML data to. If `None`, the JSON
            data is returned as a string. Defaults to `None`.
        **kwargs
            Additional parameters to pass to `io.store_json`.

        Returns
        -------
        str
            The YAML formatted string if `path_or_buffer` is None.
        """
        from . import io  # noqa

        return io.store_yaml(self, path_or_buffer=path_or_buffer, **kwargs)

    # API =====================================================================

    def extract(self, *lcs, **lc):
        """Extract the selected features from the provided light curves.

        Note that only one of `lcs` or `lc` can be provided.

        Parameters
        ----------
        *lcs : array_like of dict, optional
            A list of light curves represented as dictionaries.
        **lc : dict, optional
            A single light curve represented as a dictionary.

        Raises
        ------
        ValueError
            Both `lc` and `lcs` are provided.

        Returns
        -------
        Features
            A collection of extracted features of the provided light curves.

        Examples
        --------
        **Single light curve:**

        >>> fs = feets.FeatureSpace(only=['Std'])
        >>> fs.extract(magnitude=[1, 2, 3])
        Features(feature_names={'Std'}, length=1)

        **Multiple light curves:**

        >>> fs = feets.FeatureSpace(only=['Std'])
        >>> fs.extract({'magnitude': [1, 2, 3]}, {'magnitude': [4, 5, 6]})
        Features(feature_names={'Std'}, length=2)

        """
        lcs = self._coerce_lightcurves(single_lc=lc, multiple_lc=lcs)

        features_by_lc = runner.run(
            extractors=self._extractors,
            selected_features=self._selected_features,
            required_data=self._required_data,
            dask_options=self.dask_options,
            lcs=lcs,
        )

        return Features(features=features_by_lc, extractors=self._extractors)
