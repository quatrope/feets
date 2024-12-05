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

"""Feature extractors base classes."""


# =============================================================================
# IMPORTS
# =============================================================================

import abc
import inspect
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_VALUES_TO_REPR = 10

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


class ExtractorValidationError(ValueError):
    """Some value used by the extractor is missing or invalid."""


class ExtractorTransformError(RuntimeError):
    """The extractor can't transform the data into the expected format."""


class ExtractorWarning(UserWarning):
    """Warn about the Extractor behavior."""


class FeatureExtractionWarning(UserWarning):
    """Warn about the calculation of some feature."""


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


def _is_abstract_class(cls):
    return getattr(cls, "__abstractclass__", False)


def _is_abstract_method(attr):
    return getattr(attr, "__isabstractmethod__", False)


def _iter_method_parameters(method):
    signature = inspect.signature(method)
    parameters = tuple(signature.parameters.values())[1:]
    return iter(parameters)


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
    """Abstract base class for all feature extractors.

    Extractors are classes that implement feature extraction logic. They must
    override the `extract()` method, which is called by the `FeatureSpace`
    class to extract features from a time series.

    An extractor class may also override the `__init__()` method with custom
    arguments that are considered parameters of the extractor. All of these
    parameters must have default values.

    Extractors must define a `features` attribute, which is the list of
    features that the extractor can compute.

    Additionally, extractors can override the `flatten_feature` method to
    normalize a feature into a dictionary of scalar subfeatures for
    representation purposes.

    Attributes
    ----------
    features : array_like
        The features computed by the extractor.

    Methods
    -------
    extract(**kwargs)
        Extract features from the time series.

    flatten_feature(feature, data)
        Flatten the feature value for representation.

    Notes on the `extract()` method
    -------------------------------
    The `extract()` method must be overridden by the user to implement the
    feature extraction logic.

    It must accept as arguments the data vectors present in the time series
    (e.g., magnitude, time, error) from which it will compute the features, as
    well as the features computed by other extractors on which the calculation
    may also depend.

    The return value must be a dictionary containing the results of the feature
    extraction, where the keys represent the feature names and the values are
    the computed results. Additionally, the features returned in the dictionary
    must be those defined in the `features` attribute.

    Notes on the `flatten_feature()` method
    ---------------------------------------
    The `flatten_feature()` method can be overridden by the user to normalize a
    feature whose value comes in a complex format into a flat dictionary of
    subfeatures for representation purposes.

    It must accept as arguments the feature name and the feature computed
    value, and return a flat dictionary where the keys represent the
    subfeature names and the values are the normalized results as numpy
    scalars.

    The built-in implementation of `flatten_feature()` is already able to
    handle dictionaries and sequences, but it may need to be overridden if the
    feature comes in a different format.

    Examples
    --------
    **An extractor that computes the sum of the magnitude vector:**
    >>> class SumExtractor(Extractor):
    ...     features = ["sum_feature"]
    ...
    ...     def extract(self, magnitude):
    ...         return {"sum_feature": sum(magnitude)}
    >>> ext = SumExtractor()
    >>> ext.extract(magnitude=[1, 2, 3, 4])
    {'sum_feature': 10}

    **An extractor that depends on the previously computed sum feature and
    computes the mean of the magnitude vector:**
    >>> class MeanExtractor(Extractor):
    ...     features = ["mean_feature"]
    ...
    ...     def extract(self, magnitude, sum_feature):
    ...         return {"mean_feature": sum_feature / len(magnitude)}
    >>> ext = MeanExtractor()
    >>> ext.extract(magnitude=[1, 2, 3, 4], sum_feature=10)
    {'mean_feature': 2.5}

    **An extractor that flattens a feature into subfeatures:**
    >>> class FlattenExtractor(Extractor):
    ...     features = ["my_feature"]
    ...
    ...     def extract(self, magnitude):
    ...         return {"my_feature": magnitude}
    ...
    ...     def flatten_feature(self, feature, data):
    ...         return {f"{feature}_{i}": val for i, val in enumerate(data)}
    >>> ext = FlattenExtractor()
    >>> ext.flatten_feature("my_feature", [1, 2, 3])
    {'my_feature_0': 1, 'my_feature_1': 2, 'my_feature_2': 3}
    """

    def __init_subclass__(cls):
        """Initialize the extractor subclass and validate its definition.

        This method is called when a new subclass of the Extractor class is
        created. It validates that the subclass implements the `extract()`
        method and sets the configuration attributes for the extractor.

        Raises
        ------
        ExtractorBadDefinedError
            If the extractor subclass does not implement the `extract()` method.

        """
        cls_name = cls.__qualname__

        if cls.is_abstract():
            return

        if _is_abstract_method(cls.extract):
            raise ExtractorBadDefinedError(
                f"'{cls_name}.extract()' method must be redefined"
            )

        cls._conf = _ExtractorConf.from_extractor_class(cls)
        del cls.features

    # GETTERS =================================================================

    @classmethod
    def is_abstract(cls):
        """Check if the extractor class is an abstract class.

        Returns
        -------
        bool
            True if the extractor class is an abstract class, False otherwise.
        """
        return _is_abstract_class(cls)

    @classmethod
    def get_features(cls):
        """Retrieve the features that can be computed by the extractor.

        Returns
        -------
        frozenset
            A set of features that the extractor computes from the time series
            data. These features are the ones returned by the `extract()` method.
        """
        return cls._conf.features

    @classmethod
    def get_data(cls):
        """Retrieve the data vectors utilized by the extractor.

        Returns
        -------
        frozenset
            A set of time series data vectors that are used by the extractor
            during feature extraction. This includes both required and optional
            data vectors.

            These vectors are passed as arguments to the `extract()` method.
        """
        return cls._conf.data

    @classmethod
    def get_optional_data(cls):
        """Retrieve the optional data vectors for the extractor.

        Returns
        -------
        frozenset
            A set of optional time series data vectors that the extractor can
            use to compute its features. These data vectors may be provided as
            arguments to the `extract()` method, but they are not required.
        """
        return cls._conf.optional

    @classmethod
    def get_required_data(cls):
        """Retrieve the required data vectors for the extractor.

        Returns
        -------
        frozenset
            A set of time series data vectors that that are necessary for the
            extractor to compute its features. These data vectors must be
            provided as arguments to the `extract()` method.
        """
        return cls._conf.required

    @classmethod
    def get_dependencies(cls):
        """Retrieve the dependencies required by the extractor.

        Returns
        -------
        frozenset
            A set of features computed by other extractors that are necessary
            for this extractor to compute its own features. These dependencies
            must be provided as arguments to the `extract()` method.
        """
        return cls._conf.dependencies

    @classmethod
    def get_default_params(cls):
        """Retrieve the default parameters for the extractor.

        Returns
        -------
        dict
            A dictionary containing the default values for the parameters
            defined in the `__init__()` method of the extractor class.
        """
        return cls._conf.parameters

    # PERSISTENCE =============================================================

    @property
    def params(self):
        param_names = self.get_default_params().keys()
        return {pname: getattr(self, pname) for pname in param_names}

    def to_dict(self):
        """Convert the extractor to a dictionary representation.

        Returns
        -------
        dict
            A dictionary containing the parameters of the extractor instance.
        """
        cls_name = type(self).__name__
        params = self.params
        return {cls_name: params}

    # MAGIC ===================================================================

    def __repr__(self):
        """Return a string representation of the Extractor object."""
        cls_name = type(self).__name__
        params = self.params
        state = {}
        for pname, pvalue in params.items():
            if len(repr(pvalue)) > 20:
                pvalue = "<MANY CONFIGURATIONS>"
            state[pname] = pvalue
        return f"<{cls_name} {state}>" if state else f"<{cls_name}>"

    # API =====================================================================

    def prepare_extract(self, data, dependencies):
        """Prepare keyword arguments for the `extract()` method.

        Combine the necessary features from the `dependencies` and the required
        data from the provided `data` into a dictionary of keyword arguments
        that should be passed to the `extract()` method.

        Parameters
        ----------
        data : dict
            The available time series data vectors.
        dependencies : dict
            The available features computed by other extractors.

        Raises
        ------
        ExtractorValidationError
            If any required data or dependency is missing from the provided
            inputs.

        Returns
        -------
        dict
            A dictionary containing the keyword arguments for the `extract()`
            method.
        """
        cls_name = type(self).__qualname__

        kwargs = {}

        # select dependencies
        for d in self.get_dependencies():
            if d not in dependencies:
                raise ExtractorValidationError(
                    f"Missing required dependency {d!r} for extractor {cls_name}"
                )
            kwargs[d] = dependencies[d]

        # select data
        for d in self.get_required_data():
            if d not in data:
                raise ExtractorValidationError(
                    f"Missing required data {d!r} for extractor {cls_name}"
                )
            kwargs[d] = np.asarray(data[d])

        for d in self.get_optional_data():
            if d not in data:
                continue
            kwargs[d] = np.asarray(data[d])

        return kwargs

    def validate_extract(self, features):
        """Validate the extracted features.

        Validate that the extracted features match the expected features
        defined in the extractor.

        Parameters
        ----------
        features : dict
            The extracted features from the time series.

        Raises
        ------
        ExtractorValidationError
            If the extracted features don't match the expected features defined
            in the `features` attribute.
        """
        expected_features = self.get_features()
        diff = expected_features.symmetric_difference(features)

        if diff:
            cls_name = type(self).__qualname__
            expected_str = ", ".join(expected_features)
            results_str = ", ".join(features.keys())
            raise ExtractorValidationError(
                f"The extractor '{cls_name}' expected the features "
                f"{expected_str}. Found: {results_str!r}"
            )

    def validate_flatten(self, feature, flattened):
        """Validate that a flattened feature is a dictionary of numpy scalars.

        Parameters
        ----------
        feature : str
            The name of the flattened feature.
        flattend : object
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

    # TO REDEFINE =============================================================

    def __init__(self):
        pass

    @abc.abstractmethod
    def extract(self):
        """Extract features from the time series.

        Returns
        -------
        dict
            The dictionary of features extracted from the time series.
        """
        raise NotImplementedError()

    def flatten_feature(self, feature, value):
        """Flatten the feature value for representation.

        Parameters
        ----------
        feature : str
            The name of the feature to flatten.
        value : object
            The value of the feature.

        Returns
        -------
        dict
            A dictionary containing the flattened feature value as subfeatures.
        """
        return _flatten_data(value, feature)
