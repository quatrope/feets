feets: feATURE eXTRACTOR FOR tIME sERIES
========================================

.. image:: https://travis-ci.org/carpyncho/feets.svg?branch=master
    :target: https://travis-ci.org/carpyncho/feets
    :alt: Build Status

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://tldrlegal.com/license/mit-license
   :alt: License

.. image:: https://img.shields.io/badge/python-2.7-blue.svg
   :target: https://badge.fury.io/py/scikit-criteria
   :alt: Python 2.7

.. image:: https://img.shields.io/badge/python-3.5-blue.svg
   :target: https://badge.fury.io/py/scikit-criteria
   :alt: Python 3.5

Based on FATS:

- **Authors:** Isadora Nun and Pavlos Protopapas
- **Contributors:** Karim Pichara, Rahul Dave, Daniel Acuña, Nicolás Castro,
  Cristobal Mackenzie, Andrés Riveros and Ming Zhu

Main difference with FATS:

- Removed licurve retrieval from public surveys (we only do one thing here)
- Removed all the ``sys.exit()`` calls and replaced with Exceptions
- PEP-8
- Python 2 and 3
- Warnings instead of prints
- Only one type of results: numpy array.
- Posibility of register more FeaturesExtractors.
- Multiprocessing.


Installation
------------

Clone this repository and do ``pip install -e``.

Or ``pip install feets`` for the latest stable version.


Description
-----------

In time-domain astronomy, data gathered from the telescopes is usually
represented in the form of light-curves. These are time series that show the
brightness variation of an object through a period of time
(for a visual representation see video below). Based on the variability
characteristics of the light-curves, celestial objects can be classified into
different groups (quasars, long period variables, eclipsing binaries, etc.)
and consequently be studied in depth independentely.

In order to characterize this variability, some of the existing methods use
machine learning algorithms that build their decision on the light-curves
features. Features, the topic of the following work, are numerical descriptors
that aim to characterize and distinguish the different variability classes.
They can go from basic statistical measures such as the mean or the standard
deviation, to complex time-series characteristics such as the autocorrelation
function.

In this package we present a library with a compilation of some of the
existing light-curve features. The main goal is to create a collaborative and
open tool where every user can characterize or analyze an astronomical
photometric database while also contributing to the library by adding new
features. However, it is important to highlight that **this library is not**
**restricted to the astronomical field** and could also be applied to any kind
of time series.

Our vision is to be capable of analyzing and comparing light-curves from all
the available astronomical catalogs in a standard and universal way. This
would facilitate and make more efficient tasks as modelling, classification,
data cleaning, outlier detection and data analysis in general. Consequently,
when studying light-curves, astronomers and data analysts would be on the same
wavelength and would not have the necessity to find a way of comparing or
matching different features. In order to achieve this goal, the library should
be run in every existent survey (MACHO, EROS, OGLE, Catalina, Pan-STARRS, etc)
and future surveys (LSST) and the results should be ideally shared in the same
open way as this library.
