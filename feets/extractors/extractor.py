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

"""Feature extractor base classes."""

# =============================================================================
# IMPORTS
# =============================================================================

import abc
import inspect
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from keyword import iskeyword

import numpy as np

__all__ = [
    "DATAS",
    "extractor_warning",
    "Extractor",
    "ExtractorBadDefinedError",
    "ExtractorValidationError",
    "ExtractorWarning",
    "feature_warning",
    "FeatureExtractionWarning",
]

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_MAGNITUDE = "magnitude"
DATA_TIME = "time"
DATA_ERROR = "error"
DATA_MAGNITUDE2 = "magnitude2"
DATA_ALIGNED_MAGNITUDE = "aligned_magnitude"
DATA_ALIGNED_MAGNITUDE2 = "aligned_magnitude2"
DATA_ALIGNED_TIME = "aligned_time"
DATA_ALIGNED_ERROR = "aligned_error"
DATA_ALIGNED_ERROR2 = "aligned_error2"
DATA_FLUX = "flux"
DATA_FLUX_ERROR = "flux_error"

DATAS = (
    DATA_TIME,
    DATA_MAGNITUDE,
    DATA_ERROR,
    DATA_MAGNITUDE2,
    DATA_ALIGNED_TIME,
    DATA_ALIGNED_MAGNITUDE,
    DATA_ALIGNED_MAGNITUDE2,
    DATA_ALIGNED_ERROR,
    DATA_ALIGNED_ERROR2,
    DATA_FLUX,
    DATA_FLUX_ERROR,
)

# =============================================================================
# EXCEPTIONS
# =============================================================================


class ExtractorBadDefinedError(TypeError):
    """The extractor class is not defined properly."""

    pass


class ExtractorValidationError(ValueError):
    """Some value used by the extractor is missing or invalid."""

    pass


class ExtractorTransformError(RuntimeError):
    """The extractor can't transform the data into the expected format."""

    pass


class ExtractorWarning(UserWarning):
    """Warn about the Extractor behavior."""

    pass


class FeatureExtractionWarning(UserWarning):
    """Warn about the calculation of some feature."""

    pass


warnings.simplefilter("always", ExtractorWarning)
warnings.simplefilter("always", FeatureExtractionWarning)


def extractor_warning(msg):
    """Issue a warning about the extractor behaviour.

    Parameters
    ----------
    msg : str
        The warning message to be issued.
    """
    warnings.warn(msg, ExtractorWarning, 1)


def feature_warning(msg):
    """Issue a warning about the feature extraction process.

    Parameters
    ----------
    msg : str
        The warning message to be issued.
    """
    warnings.warn(msg, FeatureExtractionWarning, 1)


# =============================================================================
# EXTRACTOR CONF & UTILS FOR META PROGRAMMING
# =============================================================================


def _iter_method_parameters(method):
    signature = inspect.signature(method)
    parameters = tuple(signature.parameters.values())[1:]
    return iter(parameters)


def _is_valid_name(name):
    return name.isidentifier() and not iskeyword(name)


def _flatten_data(data, prefix):
    result = {}

    if np.isscalar(data):
        result[prefix] = data
    elif isinstance(data, Mapping):
        for key, value in data.items():
            result.update(_flatten_data(value, f"{prefix}_{key}"))
    elif isinstance(data, (Sequence, np.ndarray)):
        for index, item in enumerate(data):
            result.update(_flatten_data(item, f"{prefix}_{index}"))
    else:
        raise ExtractorTransformError(
            f"Can't transform data {data!r} of type {type(data)} into a "
            f"scalar format."
        )

    return result


@dataclass(frozen=True)
class _ExtractorConf:
    features: frozenset
    optional: frozenset
    required: frozenset
    dependencies: frozenset
    parameters: dict

    @staticmethod
    def _validate_and_add_feature(feature, features, features_attr):
        if not isinstance(feature, str):
            raise ExtractorBadDefinedError(
                f"Feature name must be an instance of string. "
                f"Found {type(feature)}, please check {features_attr!r}"
            )
        if not _is_valid_name(feature):
            raise ExtractorBadDefinedError(
                f"Feature name must be a valid variable identifier. "
                f"Found {feature!r}, please check {features_attr!r}"
            )
        if feature in DATAS:
            raise ExtractorBadDefinedError(
                f"Feature can't be in {DATAS!r}. Check {features_attr!r}"
            )
        if feature in features:
            raise ExtractorBadDefinedError(
                f"Duplicated feature {feature!r} in {features_attr!r}"
            )
        features.add(feature)

    @staticmethod
    def _validate_and_add_extract_param(
        param, required, optional, dependencies, ecls_name
    ):
        pname = param.name
        has_default = param.default is not param.empty

        if pname in DATAS:
            if has_default:
                optional.add(pname)
            else:
                required.add(pname)
            return

        if has_default:
            raise ExtractorBadDefinedError(
                "Dependencies can't have default values. "
                f"Check {pname!r} in '{ecls_name}.extract()' method"
            )
        dependencies.add(pname)

    @staticmethod
    def _validate_and_add_init_param(param, parameters, ecls_name):
        pname = param.name
        if param.default is param.empty:
            raise ExtractorBadDefinedError(
                f"All parameters in the '{ecls_name}.__init__()' method"
                f"must have a default value. Check {pname!r}."
            )
        parameters[pname] = param.default

    @classmethod
    def _get_feature_conf(cls, ecls):
        features_attr = f"{ecls.__qualname__}.features"
        features = set()

        for feature in getattr(ecls, "features", []):
            cls._validate_and_add_feature(feature, features, features_attr)

        if not features:
            raise ExtractorBadDefinedError(
                f"{features_attr!r} must be a non-empty sequence"
            )

        return frozenset(features)

    @classmethod
    def _get_extract_method_parameters(cls, ecls):
        ecls_name = ecls.__qualname__
        required, optional, dependencies = set(), set(), set()

        extract_params = _iter_method_parameters(ecls.extract)
        for param in extract_params:
            cls._validate_and_add_extract_param(
                param, required, optional, dependencies, ecls_name
            )

        return (
            frozenset(required),
            frozenset(optional),
            frozenset(dependencies),
        )

    @classmethod
    def _get_init_method_parameters(cls, ecls):
        ecls_name = ecls.__name__
        parameters = {}

        init_params = _iter_method_parameters(ecls.__init__)
        for param in init_params:
            cls._validate_and_add_init_param(param, parameters, ecls_name)

        return dict(parameters)

    @classmethod
    def from_extractor_class(cls, ecls):
        features = cls._get_feature_conf(ecls)
        (
            required,
            optional,
            dependencies,
        ) = cls._get_extract_method_parameters(ecls)
        parameters = cls._get_init_method_parameters(ecls)

        return _ExtractorConf(
            features=features,
            required=required,
            optional=optional,
            dependencies=dependencies,
            parameters=parameters,
        )

    @property
    def data(self):
        return frozenset(self.required.union(self.optional))


# =============================================================================
# EXTRACTOR
# =============================================================================


class Extractor(abc.ABC):
    """Abstract base class for feature extractors.

    To create a feature extractor, define a subclass of the `Extractor` class
    that defines a `features` attribute with the names of the new features, and
    implement the `extract()` method with the logic needed to compute them.

    Once defined, the new extractor class must be registered in the
    extractor registry to make it available to new `FeatureSpace` instances for
    automatic discovery and usage.

    A feature extractor may also expose optional parameters to customize its
    behavior. To add such parameters, implement the `__init__()` method and
    specify them as keyword arguments.

    For representation purposes, the features returned by the `extract()`
    method must be normalizable into a flat dictionary of scalar values. This
    is internally accomplished by the `flatten_feature()` method, which can
    be extended to add support for custom formats.

    Parameters
    ----------
    **kwargs
        Optional parameters to change the behavior of the extractor.

    Attributes
    ----------
    features : array_like of str
        The features that can be computed with the `extract()` method.

    Methods
    -------
    extract(**kwargs)
        Implement this method in a subclass such that it returns a dictionary
        containing the computed values for all of the features defined in the
        `features` attribute.
    flatten_feature(feature, value)
        By default, it handles the normalization of scalars, sequences and
        dictionaries. Extend this method to add support to more complex
        formats.


    See Also
    --------
    feets.FeatureSpace :
        Class to select and extract features from a time series.
    feets.extractor_registry :
        Extractor registry of available feature extractors.

    Examples
    --------
    Extractor that computes the sum of the `magnitude` data vector:

    >>> magnitude = [1, 2, 3, 4]
    ...
    >>> class SumExtractor(Extractor):
    ...     features = ["Sum"]
    ...
    ...     def extract(self, magnitude):
    ...         return {"Sum": sum(magnitude)}
    ...
    >>> sum_ext = SumExtractor()
    >>> sum_results = ext.extract(magnitude)
    >>> sum_results
    {'Sum': 10}

    Extractor that depends on the previously computed `Sum` feature to compute
    the mean of the `magnitude` data vector:

    >>> class MeanExtractor(Extractor):
    ...     features = ["Mean"]
    ...
    ...     def extract(self, magnitude, Sum):
    ...         return {"Mean": Sum / len(magnitude)}
    ...
    >>> mean_ext = MeanExtractor()
    >>> mean_results = mean_ext.extract(magnitude, sum_results['Sum'])
    >>> mean_results
    {'Mean': 2.5}

    Extractor that implements normalization for custom feature formats:

    >>> class CustomFormatExtractor(Extractor):
    ...     features = ["Min", "Parity", "NoDuplicates", "Squared"]
    ...
    ...     def extract(self, magnitude):
    ...         return {
    ...             "Min": min(magnitude), # number
    ...             "Parity": {
    ...                  "even": [x for x in magnitude if int(x) % 2 == 0],
    ...                  "odd": [x for x in magnitude if int(x) % 2 != 0]
    ...             }, # dict[string, list of number]
    ...             "NoDuplicates": set(magnitude), # set of number
    ...             "Squared": map(lambda x: x**2, magnitude) # map of number
    ...         }
    ...
    ...     def flatten_feature(self, feature, value):
    ...         if feature in ("NoDuplicates", "Squared"):
    ...             # add support for sets and maps
    ...             return {
    ...                 f"{feature}_{i}": item for i, item in enumerate(value)
    ...             }
    ...         else:
    ...             # fallback to default behavior
    ...             return super().flatten_feature(feature, value)
    ...
    >>> custom_format_ext = CustomFormatExtractor()
    >>> custom_format_results = custom_format_ext.extract(magnitude)
    >>> custom_format_results
    {
        'Min': 1,
        'Parity': {'even': [2, 4], 'odd': [1, 3]},
        'NoDuplicates': {1, 2, 3, 4},
        'Squared': <map object at 0x7fc9c3d7a350>
    }
    >>> custom_format_ext.flatten_feature(
    ...     "Min", custom_format_results['Min']
    ... )
    {'Min': 1}
    >>> custom_format_ext.flatten_feature(
    ...     "Parity", custom_format_results['Parity']
    ... )
    {
        'Parity_even_0': 2,
        'Parity_even_1': 4,
        'Parity_odd_0': 1,
        'Parity_odd_1': 3
    }
    >>> custom_format_ext.flatten_feature(
    ...     "NoDuplicates", custom_format_results['NoDuplicates']
    ... )
    {
        'NoDuplicates_0': 1,
        'NoDuplicates_1': 2,
        'NoDuplicates_2': 3,
        'NoDuplicates_3': 4
    }
    >>> custom_format_ext.flatten_feature(
    ...     "Squared", custom_format_results['Squared']
    ... )
    {
        'Squared_0': 1,
        'Squared_1': 4,
        'Squared_2': 9,
        'Squared_3': 16
    }
    """

    def __init_subclass__(cls, **kwargs):
        """Initialize and validate an `Extractor` subclass.

        Upon creation of an `Extractor` subclass, set the class attributes
        and validate that the `extract()` method is implemented.

        Raises
        ------
        ExtractorBadDefinedError
            If the `Extractor` subclass does not implement the `extract()`
            method.

        """
        if inspect.isabstract(cls):
            return

        cls._conf = _ExtractorConf.from_extractor_class(cls)

        cls_init = cls.__init__

        def __init__(self, **kwargs):
            cls._params = kwargs
            cls_init(self, **kwargs)

        cls.__init__ = __init__

        del cls.features

    # GETTERS =================================================================

    @classmethod
    def get_features(cls):
        """Get the features that can be computed by the feature extractor.

        Returns
        -------
        frozenset
            The features that the `extract()` method can compute.

        See Also
        --------
        extract
        """
        return cls._conf.features

    @classmethod
    def get_data(cls):
        """Get the data vectors that can be used by the feature extractor.

        The result is the union of the required and optional data vectors.

        Returns
        -------
        frozenset
            Time series data vectors that the `extract()` method can use to
            compute the features.

        See Also
        --------
        get_optional_data, get_required_data
        extract
        """
        return cls._conf.data

    @classmethod
    def get_optional_data(cls):
        """Get the data vectors optionally used by the feature extractor.

        Returns
        -------
        frozenset
            Time series data vectors that can be optionally passed to the
            `extract()` method to compute the features.

        See Also
        --------
        get_data, get_required_data
        extract
        """
        return cls._conf.optional

    @classmethod
    def get_required_data(cls):
        """Get the data vectors required by the feature extractor.

        Returns
        -------
        frozenset
            Time series data vectors that are required for the `extract()`
            method to compute the features.

        See Also
        --------
        get_data, get_optional_data
        extract
        """
        return cls._conf.required

    @classmethod
    def get_dependencies(cls):
        """Get the feature dependencies required by the feature extractor.

        Returns
        -------
        frozenset
            Features that should be previously computed by other extractors,
            and are required for the `extract()` method to compute its own
            features.

        See Also
        --------
        extract
        """
        return cls._conf.dependencies

    @classmethod
    def get_default_params(cls):
        """Get the default values for the feature extractor parameters.

        Returns
        -------
        dict
            The default values for the parameters defined by the `__init__()`
            method.

        See Also
        --------
        params
        """
        return cls._conf.parameters

    # API =====================================================================

    @classmethod
    def prepare_extract(cls, data, dependencies):
        """Build keyword arguments for the `extract()` method.

        Combine the required features from `dependencies` and the data vectors
        from `data` into the dictionary of keyword arguments that should be
        passed to the `extract()` method.

        Parameters
        ----------
        data : dict
            The available time series data vectors.
        dependencies : dict
            The available features computed by other extractors.

        Raises
        ------
        ExtractorValidationError
            A required data vector or feature dependency is missing from the
            provided values.

        Returns
        -------
        dict
            The keyword arguments for the `extract()` method.
        """
        cls_name = cls.__qualname__

        kwargs = {}

        # select dependencies
        for d in cls.get_dependencies():
            if d not in dependencies:
                raise ExtractorValidationError(
                    f"Missing required dependency {d!r} for extractor {cls_name}"
                )
            kwargs[d] = dependencies[d]

        # select data
        for d in cls.get_required_data():
            if d not in data:
                raise ExtractorValidationError(
                    f"Missing required data {d!r} for extractor {cls_name}"
                )
            kwargs[d] = np.asarray(data[d])

        for d in cls.get_optional_data():
            if d not in data or d in kwargs:
                continue
            kwargs[d] = np.asarray(data[d])

        return kwargs

    @classmethod
    def validate_extract(cls, features):
        """Validate the results of the `extract()` method.

        Validate that the extracted features match the `features` attribute.

        Parameters
        ----------
        features : dict
            The results extracted with the `extract()` method.

        Raises
        ------
        ExtractorValidationError
            If the extracted features don't match the ones defined in the
            `features` attribute.
        """
        expected_features = cls.get_features()
        diff = expected_features.symmetric_difference(features)

        if diff:
            cls_name = cls.__qualname__
            expected_str = ", ".join(map(repr, expected_features))
            results_str = ", ".join(map(repr, features.keys()))
            raise ExtractorValidationError(
                f"The extractor '{cls_name}' expected the features "
                f"{expected_str}. Found: {results_str!r}"
            )

    @classmethod
    def validate_flatten(cls, feature, flattened):
        """Validate the results of the `flatten_feature()` method.

        Parameters
        ----------
        feature : str
            The name of the flattened feature.
        flattened : object
            The flattened feature value.

        Raises
        ------
        ExtractorValidationError
            If the format of the flattened feature is not valid.
        """
        if not isinstance(flattened, dict):
            raise ExtractorValidationError(
                f"The 'flatten_feature()' method must return a dictionary. "
                f"Found {type(flattened)} for feature {feature!r}"
            )

        for key, val in flattened.items():
            if not isinstance(key, str):
                raise ExtractorValidationError(
                    f"The keys of the flattened feature must be strings. "
                    f"Found {type(key)} for feature {feature!r}"
                )
            if not np.isscalar(val):
                raise ExtractorValidationError(
                    f"The values of the flattened feature must be scalars. "
                    f"Found {type(val)} for feature {feature!r}"
                )

    # PERSISTENCE =============================================================

    @property
    def params(self):
        """Feature extractor initial parameters.

        Returns
        -------
        dict
            The parameters passed to the `__init__()` method.

        See Also
        --------
        get_default_params
        """
        params = self.get_default_params()
        params.update(self._params)
        return params

    def to_dict(self):
        """Convert the `Extractor` object to a dictionary representation.

        Returns
        -------
        dict
            A dictionary representation of the `Extractor`, including the
            values of the parameters.
        """
        cls_name = type(self).__name__
        params = self.params
        return {cls_name: params}

    # MAGIC ===================================================================

    def __repr__(self):
        """String representation of the `Extractor` object."""
        cls_name = type(self).__name__
        params = self.params

        param_strs = [
            (
                f"{pname}=..."
                if len(repr(pvalue)) > 20
                else f"{pname}={pvalue!r}"
            )
            for pname, pvalue in params.items()
        ]
        state_str = ", ".join(param_strs)
        extractor_str = f"{cls_name}({state_str})"

        return extractor_str

    # TO REDEFINE =============================================================

    def __init__(self):
        pass

    @abc.abstractmethod
    def extract(self, *args, **kwargs):
        """Extract `features` from time series data vectors.

        Parameters
        ----------
        *args
            Includes the time series data vectors that are required as inputs
            for the extraction, as well as the necessary feature dependencies.
        **kwargs
            Additional time series data vectors that can be used as optional
            inputs for the extraction.

        Returns
        -------
        dict
            The computed values for all of the features defined in the
            `features` attribute.

        See Also
        --------
        feets.Extractor : Abstract base class for feature extractors.
        """
        raise NotImplementedError()

    def flatten_feature(self, feature, value):
        """Normalize a feature value into a dictionary of scalars.

        This method is called internally to better represent the returned
        features of the `extract()` method.

        Parameters
        ----------
        feature : str
            The name of the feature.
        value : object
            The raw value as received by the `extract()` method.

        Returns
        -------
        dict
            A dictionary of scalars representing the flattened feature.

        See Also
        --------
        feets.Extractor : Abstract base class for feature extractors.
        """
        return _flatten_data(value, feature)
