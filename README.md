![logo](https://github.com/quatrope/feets/raw/master/res/logo_medium.png)


<!-- BODY -->

# feets: feATURE eXTRACTOR FOR tIME sERIES

[![PyPi Version](https://badge.fury.io/py/feets.svg)](https://badge.fury.io/py/feets)
[![ascl:1806.001](https://img.shields.io/badge/ascl-1806.001-blue.svg?colorB=262255)](http://ascl.net/1806.001)
[![ReadTheDocs.org](https://img.shields.io/badge/docs-passing-brightgreen.svg)](http://feets.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://tldrlegal.com/license/mit-license)
[![Python 3.10 3.11 3.12 3.13](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://badge.fury.io/py/feets)


## Description


In time-domain astronomy, data gathered from the telescopes is usually represented in the form of light-curves. These are time series that show the brightness variation of an object through a period of time. Based on the variability characteristics of the light-curves, celestial objects can be classified into different groups (quasars, long period variables, eclipsing binaries, etc.) and consequently be studied in depth independentely.

In order to characterize this variability, some of the existing methods use machine learning algorithms that build their decision on the light-curves features. Features, the topic of the following work, are numerical descriptors that aim to characterize and distinguish the different variability classes. They can go from basic statistical measures such as the mean or the standard deviation, to complex time-series characteristics such as the autocorrelation function.

In this package we present a library with a compilation of some of the existing light-curve features. The main goal is to create a collaborative and open tool where every user can characterize or analyze an astronomical photometric database while also contributing to the library by adding new features. However, it is important to highlight that **this library is not restricted to the astronomical field** and could also be applied to any kind of time series.

Our vision is to be capable of analyzing and comparing light-curves from all the available astronomical catalogs in a standard and universal way. This would facilitate and make more efficient tasks as modelling, classification, data cleaning, outlier detection and data analysis in general. Consequently, when studying light-curves, astronomers and data analysts would be on the same wavelength and would not have the necessity to find a way of comparing or matching different features. In order to achieve this goal, the library should be run in every existent survey (MACHO, EROS, OGLE, Catalina, Pan-STARRS, etc) and future surveys (LSST) and the results should be ideally shared in the same open way as this library.

## Installation

### Basic install

Execute:

```sh
$ pip install feets
```

### Development install

Clone this repo, and once inside execute

```sh
$ pip install -r requirements_dev.txt
```

## Documentation and tutorial

For a full documentation and tutorial, visit https://feets.readthedocs.io

## Authors

### Juan BC

jbc.develop@gmail.com

[IATE](https://iate.oac.uncor.edu/) - [UNR](https://unr.edu.ar/)


### Felipe Clariá

felipe.claria@unc.edu.ar

[UNC](https://www.unc.edu.ar/)


## License

feets is under [The MIT License](https://raw.githubusercontent.com/quatrope/feets/master/LICENSE).

A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

## Citation

If you use feets in a scientific publication, we would appreciate citations to the following paper:

> Cabral, J. B., Sánchez, B., Ramos, F., Gurovich, S., Granitto, P., & Vanderplas, J. (2018).
> From FATS to feets: Further improvements to an astronomical feature extraction tool based on machine learning.
> Astronomy and Computing.


### Bibtex entry

```tex
    @article{cabral2018fats,
      title={
        From FATS to feets: Further improvements to
        an astronomical feature extraction tool based on
        machine learning},
      author={
        Cabral, JB and S{\'a}nchez, B and Ramos, F and
        Gurovich, S and Granitto, P and Vanderplas, J},
      journal={Astronomy and Computing},
      year={2018},
      publisher={Elsevier}
    }
```

### Full publication

http://adsabs.harvard.edu/abs/2018arXiv180902154C


----

Based on "FATS" by Isadora Nun and Pavlos Protopapas (https://github.com/isadoranun/FATS)

Main difference with FATS:

- Removed all the ``sys.exit()`` calls and replaced with Exceptions
- PEP-8
- Python 2 and 3
- Warnings instead of prints
- Only one type of results: numpy array.
- Posibility of register more FeaturesExtractors.


## Code of conduct

`feets` endorses [the Astropy Project code of conduct](http://www.astropy.org/code_of_conduct.html).
