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

"""Synthetic light curve dataset generators."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .base import LightCurveDataset
from ..extractors.extractor import DATA_ERROR, DATA_MAGNITUDE, DATA_TIME


# =============================================================================
# CONSTANTS
# =============================================================================

DATASET_NAME = "feets-synthetic"

DATASET_DESCRIPTION = "Lightcurve created with random numbers"

DATASET_BANDS = ("B", "V")

DATASET_METADATA = None

DEFAULT_SIZE = 10000

DEFAULT_TIMEF_PARAMS = {"start": 0.0, "stop": 1.0}


# =============================================================================
# FUNCTIONS
# =============================================================================


def create_random(
    magf,
    magf_params,
    errf,
    errf_params,
    timef=np.linspace,
    timef_params=None,
    size=DEFAULT_SIZE,
    _id=None,
    name=DATASET_NAME,
    description=DATASET_DESCRIPTION,
    bands=DATASET_BANDS,
    metadata=DATASET_METADATA,
):
    """Generate a light curve dataset with any given random generator.

    Parameters
    ----------
    magf : callable
        Function to generate the magnitudes.
    magf_params : dict-like
        Parameters to feed the `magf` function.
    errf : callable
        Function to generate the magnitudes.
    errf_params : dict-like
        Parameters to feed the `errf` function.
    timef : callable, default=numpy.linspace
        Function to generate the times.
    timef_params : dict-like or None, (default={"start": 0., "stop": 1.})
        Parameters to feed the `timef` callable.
    size : int, default=10000
        Number of observations to generate for each light curve.
    _id : object, optional
        Id for the generated dataset.
    name : str, default="feets-synthetic"
        Name for the generated dataset.
    description : str, default="Lightcurve created with random numbers"
        Description for the generated dataset.
    bands : tuple of strings, default=("B", "V")
        The bands to generate.
    metadata : dict-like, optional
        The metadata for the generated dataset

    Returns
    -------
    LightCurveDataset
        Dataset with randomly generated light curve data vectors.

    Examples
    --------
    >>> from numpy import random
    >>>  create_random(
    ...     magf=random.normal, magf_params={"loc": 0, "scale": 1},
    ...     errf=random.normal, errf_params={"loc": 0, "scale": 0.008})
    LightCurveDataset(_id=None, name='feets-synthetic', bands=('B', 'V'))
    """
    timef_params = (
        DEFAULT_TIMEF_PARAMS.copy()
        if timef_params is None
        else timef_params.copy()
    )
    timef_params.update(num=size)

    magf_params = magf_params.copy()
    magf_params.update(size=size)

    errf_params = errf_params.copy()
    errf_params.update(size=size)

    data = {}
    for band in bands:
        data[band] = {
            DATA_TIME: timef(**timef_params),
            DATA_MAGNITUDE: magf(**magf_params),
            DATA_ERROR: errf(**errf_params),
        }
    return LightCurveDataset(
        id=_id,
        name=name,
        description=description,
        bands=bands,
        data=data,
        metadata=metadata,
    )


def create_normal(
    mu=0.0, sigma=1.0, mu_err=0.0, sigma_err=1.0, seed=None, **kwargs
):
    """Generate a light curve dataset that follows a Gaussian distribution.

    Both the magnitudes and the errors are drawn from Gaussian distributions.

    Parameters
    ----------
    mu : float, default=0.0
        Mean of the Gaussian distribution for the magnitudes.
    sigma : float, default=1.0
        Standard deviation of the Gaussian distribution for the magnitudes.
    mu_err : float, default=0.0
        Mean of the Gaussian distribution for the magnitudes errors.
    sigma_err : float, default=1.0
        Standard deviation of the Gaussian distribution for the magnitudes
        errors.
    seed : int or array_like, optional
        Random seed used to initialize the pseudo-random number generator.
    kwargs : optional
        extra arguments for `create_random()`.

    Returns
    -------
    LightCurveDataset
        Dataset with randomly generated light curve data vectors.

    See Also
    --------
    create_random

    Examples
    --------
    >>> ds = create_normal(0, 1, 0, .0008, seed=42)
    >>> ds
    LightCurveDataset(_id=None, name='feets-synthetic', bands=('B', 'V'))
    >>> ds.data.B
    LightCurve(time[10000], magnitude[10000], error[10000])
    >>> ds.data.B.time
    array([  0.00000000e+00,   1.00010001e-04,   2.00020002e-04, ...,
                9.99799980e-01,   9.99899990e-01,   1.00000000e+00])
    """
    random = np.random.RandomState(seed)
    return create_random(
        magf=random.normal,
        magf_params={"loc": mu, "scale": sigma},
        errf=random.normal,
        errf_params={"loc": mu_err, "scale": sigma_err},
        **kwargs,
    )


def create_uniform(
    low=0.0, high=1.0, mu_err=0.0, sigma_err=1.0, seed=None, **kwargs
):
    """Generate a light curve dataset that follows a uniform distribution.

    The magnitude values follow a uniform distribution, meanwhile the
    error values are drawn from a Gaussian distribution.

    Parameters
    ----------
    low : float, default=0.0
        Lower boundary of the magnitude interval. All values generated will be
        greater than or equal to `low`.
    high : float, default=1.0
        Upper boundary of the magnitude interval. All values generated will be
        less than `high`.
    mu_err : float, default=0.0
        Mean of the Gaussian distribution for the magnitude errors.
    sigma_err : float, default=1.0
        Standard deviation of the Gaussian distribution for the magnitude
        errors.
    seed : int or array_like, optional
        Random seed used to initialize the pseudo-random number generator.
    kwargs : optional
        extra arguments for `create_random()`.

    Returns
    -------
    LightCurveDataset
        Dataset with randomly generated light curve data vectors.

    See Also
    --------
    create_random

    Examples
    --------
    >>> ds = synthetic.create_uniform(1, 2, 0, .0008, 42)
    >>> ds
    LightCurveDataset(_id=None, name='feets-synthetic', bands=('B', 'V'))
    >>> ds.data.B.magnitude
    array([ 1.37454012,  1.95071431,  1.73199394, ...,  1.94670792,
            1.39748799,  1.2171404 ])
    """
    random = np.random.RandomState(seed)
    return create_random(
        magf=random.uniform,
        magf_params={"low": low, "high": high},
        errf=random.normal,
        errf_params={"loc": mu_err, "scale": sigma_err},
        **kwargs,
    )


def create_periodic(mu_err=0.0, sigma_err=1.0, seed=None, **kwargs):
    """Generate a light curve dataset with periodic variability.

    The magnitude values follow a sinusoidal pattern with added Gaussian noise,
    while the error values are drawn from a Gaussian distribution.

    Parameters
    ----------
    mu_err : float, default=0.0
        Mean of the Gaussian distribution for the magnitude errors.
    sigma_err : float, default=1.0
        Standard deviation of the Gaussian distribution for the magnitude
        errors.
    seed : int or array_like, optional
        Random seed used to initialize the pseudo-random number generator.
    kwargs : optional
        extra arguments for `create_random()`.

    Returns
    -------
    LightCurveDataset
        Dataset with randomly generated light curve data vectors.

    Examples
    --------
    >>> ds = synthetic.create_periodic(bands=["Ks"])
    >>> ds
    LightCurveDataset(_id=None, name='feets-synthetic', bands=('Ks',))
    >>> ds.data.Ks.magnitude
    array([ 0.95428053,  0.73022685,  0.03005121, ..., -0.26305297,
            2.57880082,  1.03376863])
    """
    random = np.random.RandomState(seed)

    size = kwargs.get("size", DEFAULT_SIZE)

    times, mags, errors = [], [], []
    for _ in kwargs.get("bands", DATASET_BANDS):
        time = 100 * random.rand(size)
        error = random.normal(size=size, loc=mu_err, scale=sigma_err)
        mag = np.sin(2 * np.pi * time) + error * random.randn(size)
        times.append(time)
        errors.append(error)
        mags.append(mag)

    times, mags, errors = iter(times), iter(mags), iter(errors)

    return create_random(
        magf=lambda **k: next(mags),
        magf_params={},
        errf=lambda **k: next(errors),
        errf_params={},
        timef=lambda **k: next(times),
        timef_params={},
        **kwargs,
    )
