.. FILE AUTO GENERATED !! 

All notable changes to this project will be documented in this file.

[Unreleased]
------------

Added
^^^^^


* Separate development dependencies into ``requirements_dev.txt``.
* Add ``.travis.yml`` for continuous integration.
* Add extractors with flux parameters.
* Add missing magnitude extractor from ``light-curve``.
* Initial support for ``light-curve`` objects.
* Add ``io`` module for data persistence.
* Add ``custom_json`` for serialization.
* Add persistence methods in ``FeatureSpace``.
* Add ``to_dict`` methods for ``FeatureSpace`` and ``Extractor``.
* Add benchmarks for performance evaluation.
* Add ``from_lc`` method in ``FeatureSpace``.
* Add ``flatten_feature()`` method to the ``Extractor`` abstract class.
* Add ``runner`` module for parallel execution.
* Add headers to source files.
* Add extractor plan getter for ``FeatureSpace``.
* Add ``Signature`` and ``DeltamDeltat`` extractors.
* Add basic skeleton of ``FeatureSet``.
* Add two new feature sets.

Changed
^^^^^^^


* Re-enable ``pytest-xdist`` for parallel testing.
* Decorate slow tests with ``@pytest.mark.slow``.
* Update ``.readthedocs.yml`` configuration.
* Bump version for development.
* Update license file.
* Revamp ``datasets`` module.
* Improve documentation and testing for various modules.
* Refactor ``FeatureSpace`` to separate single and multiple lightcurve handling.
* Extract ``Features`` class into its own module.
* Setup style for Python 13.
* Simplify ``lightcurve`` extractor API and rework parameter handling.
* Validate required data before extraction.
* Bump ``astropy`` to v7.
* Move warnings out of ``Extractor`` class.
* Update transform defaults and add multiple parametrization for extractors.
* Rename and refactor ``light-curve`` extractors.
* Refactor ``Extractor`` API and add new ``LightCurveExtractor`` class.
* Extend documentation and rename ``io`` module.
* Refactor ``FeatureSpace`` to use ``dask.delayed``.
* Refactor extractor registry.
* Use ``dask.delayed`` instead of custom graph for parallel execution.
* Use FAP implementation from ``astropy``.
* Update ``LombScargle`` extractor to return N periods.
* Update various extractors.
* Format code with ``black``.

Fixed
^^^^^


* Fix data values being overridden by ``light-curve`` extractors.
* Remove documentation warnings.
* Revert renaming of ``io`` module.
* Revert usage of chunks in runner and default to multiprocessing scheduler.
* Fix parallel extract implementation.
* Fix default init params for extractors.
* Fix execution plan.
* Fix various extractor issues.
* Fix invalid escape sequence warnings.
* Fix some ``flake8`` warnings.
* Fix ``StetsonJ`` and ``StetsonL`` tests.
* Fix ``pyproject.toml`` configuration.
* Fix ``numpy`` and ``astropy`` imports.
* Fix various merge conflicts.
* Fix typos in documentation.
* Fix ``CAR`` and ``SlottedALength`` extractors.

Removed
^^^^^^^


* Remove duplicated files.
* Remove ``attrs`` dependency.
* Remove execution graph from repository.
* Remove ``travis`` and ``ez_setup``.
* Remove ``unicode_literals``.
* Remove unnecessary ``dict`` declarations.

[0.4] - 2023-10-27
------------------

Added
^^^^^


* Filter features by dependencies.
* Add synthetic lightcurve generation.
* Add ``features`` object.
* Add ``LombScargle`` extractor.
* Add ``StructureFunction`` feature.
* Add ``Gskew`` feature.
* Add regression tests.
* Add test infrastructure.

Changed
^^^^^^^


* Improve API for feature extraction.
* Improve documentation.
* Improve testing.
* Use ``astropy`` implementation of FAP.
* Optimize FFT calculations.
* Improve ``sort_by_dependencies``.
* Make ``Extractor`` parameters stricter.
* Make API clearer.
* Standardize ``Extractor`` representation.
* Group extractors by file.

Fixed
^^^^^


* Remove warnings in ``CAR`` extractor.
* Fix ``StructureFunctions`` undefined error.
* Fix ``CAR`` and ``SlottedALength`` extractors.
* Fix JavaScript issues in documentation.
* Fix ``travis`` configuration for Python 3.7.
* Fix bug in ``register_extractor`` order.
* Fix ``Fourier`` extractor.
* Fix bug in ``align_lc``.
* Fix ``FluxPercentile`` features.
* Fix ``Color`` extractor.
* Fix ``Signature`` extractor.

Removed
^^^^^^^


* Remove support for Python 3.4.
* Remove ``multiprocess`` in favor of ``dask``.
* Remove ``numba`` dependency.
* Remove unused ``FATS`` code.
