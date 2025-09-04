#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
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

import numpy as np

from .extractors import DATAS, extractor_registry
from .extractors.registry import RegistryError
from .features import Features
from .runner import run

# =============================================================================
# LOG
# =============================================================================

logger = logging.getLogger("feets")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


# =============================================================================
# FEATURE SPACE
# =============================================================================


class FeatureSpace:
    """Class to select and extract features from a time series.

    The `FeatureSpace` class allows for the extraction of selected features
    from the available data vectors (e.g., magnitude, time, error,
    second magnitude) of one or more time series.

    The `data`, `only`, and `exclude` filters can be combined to control the
    selection of features to be extracted. If no filter is provided, the
    selection will include all the available features.

    Parameters
    ----------
    data : array_like, optional
        List of available data vectors to extract from. If provided, only the
        features that can be computed on some of the selected vectors will be
        included.
    only : array_like, optional
        List of features to be extracted. If provided, only the selected
        features will be included. It must be disjoint with `exclude`.
    exclude : array_like, optional
        List of features to be excluded from the extraction. If provided, all
        features except the selected ones will be included. It must be
        disjoint with `only`.
    **kwargs
        Additional parameters used to initialize the extractors.

    Attributes
    ----------
    features : frozenset
        The features selected for extraction, based on the provided filters.
    extractors : np.ndarray
        The extractor instances used to compute the features, ordered by their
        dependencies.
    required_data : frozenset
        The data vectors required for the extraction.
    dask_options : dict
        Options to be passed to the Dask scheduler.

    Raises
    ------
    ValueError
        If an invalid combination of `data`, `only`, and `exclude` is provided.

    See Also
    --------
    feets.Features : Class to manage and manipulate feature extraction results.
    feets.Extractor : Abstract base class for feature extractors.
    dask.compute : Compute several dask collections at once.

    Examples
    --------
    Using `data` filter to specify the available data vectors:

    >>> fs = FeatureSpace(data=['magnitude', 'time'])
    >>> # The resulting `FeatureSpace` will only extract the features that
    >>> # depend on 'magnitude' and/or 'time'.
    >>> fs.extract(**lc)
    <Features feature_names={'Mean', 'Std', 'PeriodLS', 'Signature', ...}, length=1>

    Using `only` filter to select specific features for extraction:

    >>> fs = FeatureSpace(only=['Mean', 'Std'])
    >>> # The resulting `FeatureSpace` will only extract the 'Mean' and 'Std'
    >>> # features, regardless of the available data vectors.
    >>> fs.extract(**lc)
    <Features feature_names={'Mean', 'Std'}, length=1>

    Using `exclude` filter to exclude specific features from extraction:

    >>> fs = FeatureSpace(exclude=['Mean', 'Std'])
    >>> # The resulting `FeatureSpace` will extract all features except for
    >>> # 'Mean' and 'Std', regardless of the available data vectors.
    >>> fs.extract(**lc)
    <Features feature_names={'PeriodLS', 'Signature', ...}, length=1>

    Configuring the extractors with additional parameters:

    >>> fs = FeatureSpace(
    ...     data=['magnitude', 'time'],
    ...     PeriodLS={'nperiods': 5},
    ...     Signature={'phase_bins': 20, 'mag_bins': 15}
    ... )
    >>> # The resulting `FeatureSpace` will extract features that depend on
    >>> # 'magnitude' and 'time', with the specified parameters for the
    >>> # `PeriodLS` and `Signature` extractors.
    >>> fs.extract(**lc)
    <Features feature_names={'Mean', 'Std', 'PeriodLS', 'Signature', ...}, length=1>
    """

    # CONSTRUCTOR =============================================================

    def _init_extractor(self, extractor_cls, **kwargs):
        params = kwargs.get(extractor_cls.__name__, {})
        extractor = extractor_cls(**params)
        return extractor

    def __init__(
        self, data=None, only=None, exclude=None, dask_options=None, **kwargs
    ):
        try:
            extractor_clss = extractor_registry.get_execution_plan(
                data=data, only=only, exclude=exclude
            )
        except RegistryError as exc:
            raise ValueError(str(exc))

        extractor_instances = []
        selected_features = set()
        required_data = set()

        for extractor_cls in extractor_clss:
            extractor_instance = self._init_extractor(extractor_cls, **kwargs)
            extractor_instances.append(extractor_instance)

            features = extractor_cls.get_features()
            if only is not None:
                features = features.intersection(only)
            selected_features.update(features)

            required_data.update(extractor_cls.get_required_data())

        self._extractors = np.array(extractor_instances, dtype=object)
        self._selected_features = frozenset(selected_features)
        self._required_data = frozenset(required_data)
        self.dask_options = dask_options

    # FROM LC =================================================================

    @classmethod
    def from_lightcurves(cls, *lcs):
        """Create a `FeatureSpace` for the provided light curves.

        This method determines the common data vectors (e.g., 'magnitude',
        'time') present across all provided light curves. It then creates a
        `FeatureSpace` configured to extract only the features that can be
        computed from this common set of data vectors.

        Parameters
        ----------
        *lcs : list of dict
            A list of light curves, where each light curve is a dictionary
            mapping data vector names to their values.

        Returns
        -------
        FeatureSpace
            A `FeatureSpace` instance configured for the common data vectors.

        Raises
        ------
        ValueError
            If no common data vectors are found among the light curves.

        See Also
        --------
        from_lightcurve

        Examples
        --------
        >>> lc1 = {'magnitude': [1, 2, 3]}
        >>> lc2 = {'time': [0.1, 0.2, 0.3], 'magnitude': [4, 5, 6]}
        >>>
        >>> # The common data vector is 'magnitude'.
        >>> fs = FeatureSpace.from_lightcurves(lc1, lc2)
        >>>
        >>> # The resulting `FeatureSpace` will only extract features that
        >>> # depend on 'magnitude'.
        >>> fs.extract(**lc1)
        Features(feature_names={'Mean', 'Std', ...}, length=1)
        """
        selected_data = set(DATAS)
        for lc in lcs:
            selected_data.intersection_update(lc)
        if not selected_data:
            raise ValueError(
                "No common data vectors found in the provided light curves."
            )
        return cls(data=selected_data)

    @classmethod
    def from_lightcurve(cls, **lc):
        """Create a `FeatureSpace` for the provided light curve.

        The resulting `FeatureSpace` will be configured to extract only the
        features that can be computed from the data vectors present in the
        provided light curve.

        Parameters
        ----------
        **lc : dict
            A light curve represented as a dictionary, mapping data vector names
            to their values.

        Returns
        -------
        FeatureSpace
            A `FeatureSpace` instance configured for the provided light curve.

        See Also
        --------
        from_lightcurves

        Examples
        --------
        >>> lc = {'magnitude': [1, 2, 3]}
        >>> fs = FeatureSpace.from_lightcurve(**lc)
        >>>
        >>> # The resulting `FeatureSpace` will only extract features that
        >>> # depend on 'magnitude'.
        >>> fs.extract(**lc)
        Features(feature_names={'Mean', 'Std', ...}, length=1)
        """
        return cls.from_lightcurves(lc)

    # PROPERTIES ==============================================================

    @property
    def selected_features(self):
        """frozenset: The features selected for extraction."""
        return self._selected_features

    @property
    def extractors(self):
        """np.ndarray: The extractor instances used to compute the features.

        The extractors are ordered according to their dependencies, meaning that
        the extractors that depend on others come after those they depend on.

        """
        return self._extractors

    @property
    def required_data(self):
        """frozenset: The data vectors required for the extraction."""
        return self._required_data

    # MAGIC ===================================================================

    def __repr__(self):
        """String representation of the `FeatureSpace` object."""
        space = ", ".join(map(repr, self._extractors))
        return f"<FeatureSpace: {space}>"

    # PERSISTENCE ==============================================================

    @classmethod
    def from_dict(cls, data):
        """Create a `FeatureSpace` object from a dictionary representation.

        Parameters
        ----------
        data : dict
            A dictionary representation of the `FeatureSpace`, including the data
            vectors required for extraction, the selected features, the list of
            extractors with their parameters, and the Dask options.

        Returns
        -------
        FeatureSpace
            A `FeatureSpace` object configured with the features, required
            data vectors, dask options, and extractors from the provided
            dictionary.

        See Also
        --------
        to_dict

        """
        only = data["selected_features"]
        dask_options = data["dask_options"]
        kwargs = {}
        for extractor in data["extractors"]:
            ((ename, ekwargs),) = extractor.items()
            kwargs.update({ename: ekwargs})

        return cls(
            only=only,
            dask_options=dask_options,
            **kwargs,
        )

    def to_dict(self):
        """Convert the `FeatureSpace` object to a dictionary representation.

        Returns
        -------
        dict
            A dictionary representation of the `FeatureSpace`, including the data
            vectors required for extraction, the selected features, the list of
            extractors with their parameters, and the Dask options.

        See Also
        --------
        from_dict, to_json, to_yaml

        """
        return {
            "selected_features": set(self._selected_features),
            "required_data": set(self._required_data),
            "dask_options": self.dask_options,
            "extractors": [
                extractor.to_dict() for extractor in self._extractors
            ],
        }

    def to_json(self, *, path_or_buffer=None, **kwargs):
        """Serialize the `FeatureSpace` to a JSON formatted string or file.

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

        See Also
        --------
        to_dict, to_yaml

        """
        from . import io  # noqa

        return io.store_json(self, path_or_buffer=path_or_buffer, **kwargs)

    def to_yaml(self, *, path_or_buffer=None, **kwargs):
        """Serialize the `FeatureSpace` to a YAML formatted string or file.

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

    def extract_many(self, *lcs):
        """Extract the selected features from the provided light curves.

        Parameters
        ----------
        *lcs : list of dict
            A list of light curves, where each light curve is a dictionary
            mapping data vector names to their values.

        Returns
        -------
        Features
            A collection of extracted features of the provided light curves.

        See Also
        --------
        feets.Features :
            Class to manage and manipulate feature extraction results.
        extract



        Examples
        --------
        >>> fs = FeatureSpace(only=['Mean'])
        >>> fs.extract_many({'magnitude': [1, 2, 3]}, {'magnitude': [4, 5, 6]})
        Features(feature_names={'Mean'}, length=2)
        """
        features_by_lc = run(
            extractors=self._extractors,
            selected_features=self._selected_features,
            required_data=self._required_data,
            dask_options=self.dask_options,
            lcs=list(lcs),
        )

        return Features(features=features_by_lc, extractors=self._extractors)

    def extract(self, **lc):
        """Extract the selected features from the provided light curve.

        Parameters
        ----------
        **lc : dict
            A light curve represented as a dictionary, mapping data vector names
            to their values.

        Returns
        -------
        Features
            A collection of extracted features of the provided light curves.

        See Also
        --------
        feets.Features :
            Class to manage and manipulate feature extraction results.
        extract_many

        Examples
        --------
        >>> fs = FeatureSpace(only=['Mean'])
        >>> fs.extract(magnitude=[1, 2, 3])
        Features(feature_names={'Mean'}, length=1)
        """
        return self.extract_many(lc)
