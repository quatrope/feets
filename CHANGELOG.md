# Changelog

All notable changes to this project will be documented in this file.

## [1.0] - 2025-09

New major version of `feets`, with a full redesign and modernization.

Support for older versions of Python has been removed. `feets` now officially supports Python 3.10+.

### Highlights of this release

#### Parallelized workflow & batch processing

A new [Dask](https://www.dask.org/)-powered parallelized model has been introduced for feature extraction.

-   Dramatically reduces runtime for large datasets.
-   Enables scalable batch processing of multiple light curves simultaneously using the new `FeatureSpace.extract_many()` method.
-   Simplifies large-scale analyses and integration with machine learning pipelines by processing multiple datasets in a single call.

#### Integration with `light-curve`

This release incorporates all extractors available in the [light-curve](https://github.com/light-curve/) library.

-   Existing extractors in `feets` have been replaced by optimized implementations from `light-curve`, improving speed, numerical stability, and reliability.
-   Additional extractors from `light-curve` have been added, expanding the feature set.
-   New compatibility with `flux` and `flux_error` data vectors, allowing extractors to operate directly on flux-based inputs as well as magnitudes.

#### Configuration persistence

New functionality for saving and loading `FeatureSpace` configurations has been introduced.

-   Configurations can now be exported and imported in **JSON** and **YAML** formats.
-   Methods `FeatureSpace.to_json()` and `FeatureSpace.to_yaml()`, as well as functions `feets.read_json()` and `feets.read_yaml()`, enable reproducible workflows and easy sharing of feature extraction setups.
-   This makes it straightforward to store, reuse, and distribute predefined extraction pipelines across projects.

### Main changes to the API

#### `feets.Extractor`

The extractor system has been fully redesigned for clarity, configurability, and performance.

-   **Removed:** the `data`, `dependencies`, and `params` attributes.
-   **Changed:** the `fit()` method has been renamed to `extract()`.
-   **Changed:** data requirements and feature dependencies are now inferred directly from the signature of the `extract()` method.
-   **Changed:** extractor parameters are now defined in the `__init__()` constructor instead of a `params` dictionary.
-   **Added:** `flatten_feature()` method for custom flattening of multi-value features.
-   **Added:** compatibility with new data vectors (`flux`, `flux_error`).

#### `feets.FeatureSpace`

-   **Changed:** the `extract()` method now accepts light-curve data as keyword arguments (e.g., `extract(time=t, magnitude=m)`).
-   **Added:** `extract_many()` method for batch feature extraction from multiple light curves.
-   **Added:** `from_dict()`, `to_dict()`, `to_json()`, and `to_yaml()` methods for configuration persistence and reproducibility.
-   **Added:** `from_lightcurve()` and `from_lightcurves()` for automatic feature selection based on the available data.

#### `feets.features`

-   **Added**: `Features` class for a structured representation of extracted features. Encapsulates feature values, names, and associated metadata in a single object, with support for easy conversion to dictionaries, `pandas.DataFrame`s, and other formats for downstream analysis.

#### `feets.registry`

-   **Added:** `ExtractorRegistry` class for registration, discovery, and ordering of extractors.
-   **Added:** a central `extractor_registry` object that stores available extractors and features, ensuring correct execution order and dependency resolution.

#### `feets.io`

-   **Added:** `load_json()` and `load_yaml()` functions for reading existing `FeatureSpace` configurations.

#### `feets.runner`

-   **Added:** `run()` function for running instances of feature extractors in parallel with `dask`.

## [0.4] 2018-04

Early prerelease version with experimental features.

APIs subject to change.
