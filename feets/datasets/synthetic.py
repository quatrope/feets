#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =============================================================================
# DOCS
# =============================================================================

"""Synthetic light curve generator.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .base import Data


# =============================================================================
# CONSTANTS
# =============================================================================

DS_NAME = "feets-synthetic"

DESCRIPTION = "Lightcurve created with random numbers"

BANDS = ("B", "V")

METADATA = None

DEFAULT_SIZE = 10000


# =============================================================================
# FUNCTIONS
# =============================================================================

def create_random(magf, magf_params, errf, errf_params,
                  timef=np.linspace, timef_params=None, size=DEFAULT_SIZE,
                  id=None, ds_name=DS_NAME, description=DESCRIPTION,
                  bands=BANDS, metadata=METADATA):
    """Generate a data with any given random function.

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
    timef : callable, (default=numpy.linspace)
        Function to generate the times.
    timef_params : dict-like or None, (default={"start": 0., "stop": 1.})
        Parameters to feed the `timef` callable.
    size : int (default=10000)
        Number of obervation of the light curves
    id : object (default=None)
        Id of the created data.
    ds_name : str (default="feets-synthetic")
        Name of the dataset
    description : str (default="Lightcurve created with random numbers")
        Description of the data
    bands : tuple of strings (default=("B", "V"))
        The bands to be created
    metadata : dict-like or None (default=None)
        The metadata of the created data

    Returns
    -------

    data
        A Data object with a random lightcurves.

    Examples
    --------

    .. code-block:: pycon

        >>> from numpy import random
        >>>  create_random(
        ...     magf=random.normal, magf_params={"loc": 0, "scale": 1},
        ...     errf=random.normal, errf_params={"loc": 0, "scale": 0.008})
        Data(id=None, ds_name='feets-synthetic', bands=('B', 'V'))

    """
    timef_params = (
        {"start": 0., "stop": 1.}
        if timef_params is None else
        timef_params.copy())
    timef_params.update(num=size)

    magf_params = magf_params.copy()
    magf_params.update(size=size)

    errf_params = errf_params.copy()
    errf_params.update(size=size)

    data = {}
    for band in bands:
        data[band] = {
            "time": timef(**timef_params),
            "magnitude": magf(**magf_params),
            "error": errf(**errf_params)}
    return Data(
        id=id, ds_name=ds_name, description=description,
        bands=bands, metadata=metadata, data=data)


def create_normal(mu=0., sigma=1., mu_err=0.,
                  sigma_err=1., seed=None, **kwargs):
    """Generate a data with magnitudes that follows a Gaussian
     distribution. Also their errors are gaussian.

    Parameters
    ----------

    mu : float (default=0)
        Mean of the gaussian distribution of magnitudes
    sigma : float (default=1)
        Standar deviation of the gaussian distribution of magnitude errors
    mu_err : float (default=0)
        Mean of the gaussian distribution of magnitudes
    sigma_err : float (default=1)
        Standar deviation of the gaussian distribution of magnitude errorrs
    seed : {None, int, array_like}, optional
        Random seed used to initialize the pseudo-random number generator.
        Can be any integer between 0 and 2**32 - 1 inclusive, an
        array (or other sequence) of such integers, or None (the default).
        If seed is None, then RandomState will try to read data from
        /dev/urandom (or the Windows analogue) if available or seed from
        the clock otherwise.
    kwargs : optional
        extra arguments for create_random.

    Returns
    -------

    data
        A Data object with a random lightcurves.

    Examples
    --------

    .. code-block:: pycon

        >>> ds = create_normal(0, 1, 0, .0008, seed=42)
        >>> ds
        Data(id=None, ds_name='feets-synthetic', bands=('B', 'V'))
        >>> ds.data.B
        LightCurve(time[10000], magnitude[10000], error[10000])
        >>> ds.data.B.time
        array([  0.00000000e+00,   1.00010001e-04,   2.00020002e-04, ...,
                 9.99799980e-01,   9.99899990e-01,   1.00000000e+00])

    """

    random = np.random.RandomState(seed)
    return create_random(
        magf=random.normal, magf_params={"loc": mu, "scale": sigma},
        errf=random.normal, errf_params={"loc": mu_err, "scale": sigma_err},
        **kwargs)


def create_uniform(low=0., high=1., mu_err=0., sigma_err=1.,
                   seed=None, **kwargs):
    """Generate a data with magnitudes that follows a uniform
     distribution; the error instead are gaussian.

    Parameters
    ----------

    low : float, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float, optional
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    mu_err : float (default=0)
        Mean of the gaussian distribution of magnitudes
    sigma_err : float (default=1)
        Standar deviation of the gaussian distribution of magnitude errorrs
    seed : {None, int, array_like}, optional
        Random seed used to initialize the pseudo-random number generator.
        Can be any integer between 0 and 2**32 - 1 inclusive, an
        array (or other sequence) of such integers, or None (the default).
        If seed is None, then RandomState will try to read data from
        /dev/urandom (or the Windows analogue) if available or seed from
        the clock otherwise.
    kwargs : optional
        extra arguments for create_random.

    Returns
    -------

    data
        A Data object with a random lightcurves.

    Examples
    --------

    .. code-block:: pycon

        >>> ds = synthetic.create_uniform(1, 2, 0, .0008, 42)
        >>> ds
        Data(id=None, ds_name='feets-synthetic', bands=('B', 'V'))
        >>> ds.data.B.magnitude
        array([ 1.37454012,  1.95071431,  1.73199394, ...,  1.94670792,
                1.39748799,  1.2171404 ])

    """
    random = np.random.RandomState(seed)
    return create_random(
        magf=random.uniform, magf_params={"low": low, "high": high},
        errf=random.normal, errf_params={"loc": mu_err, "scale": sigma_err},
        **kwargs)


def create_periodic(mu_err=0., sigma_err=1., seed=None, **kwargs):
    """Generate a data with magnitudes with periodic variability
     distribution; the error instead are gaussian.

    Parameters
    ----------
    mu_err : float (default=0)
        Mean of the gaussian distribution of magnitudes
    sigma_err : float (default=1)
        Standar deviation of the gaussian distribution of magnitude errorrs
    seed : {None, int, array_like}, optional
        Random seed used to initialize the pseudo-random number generator.
        Can be any integer between 0 and 2**32 - 1 inclusive, an
        array (or other sequence) of such integers, or None (the default).
        If seed is None, then RandomState will try to read data from
        /dev/urandom (or the Windows analogue) if available or seed from
        the clock otherwise.
    kwargs : optional
        extra arguments for create_random.

    Returns
    -------

    data
        A Data object with a random lightcurves.

    Examples
    --------

    .. code-block:: pycon

        >>> ds = synthetic.create_periodic(bands=["Ks"])
        >>> ds
        Data(id=None, ds_name='feets-synthetic', bands=('Ks',))
        >>> ds.data.Ks.magnitude
        array([ 0.95428053,  0.73022685,  0.03005121, ..., -0.26305297,
                2.57880082,  1.03376863])

    """

    random = np.random.RandomState(seed)

    size = kwargs.get("size", DEFAULT_SIZE)

    times, mags, errors = [], [], []
    for b in kwargs.get("bands", BANDS):
        time = 100 * random.rand(size)
        error = random.normal(size=size, loc=mu_err, scale=sigma_err)
        mag = np.sin(2 * np.pi * time) + error * random.randn(size)
        times.append(time)
        errors.append(error)
        mags.append(mag)

    times, mags, errors = iter(times), iter(mags), iter(errors)

    return create_random(
        magf=lambda **k: next(mags), magf_params={},
        errf=lambda **k: next(errors), errf_params={},
        timef=lambda **k: next(times), timef_params={}, **kwargs)
