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

from .base import Dataset


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

        timef_params = (
            {"start": 0., "stop": 1.}
            if timef_params is None else
            timef_params.copy())
        timef_params.update(num=size)

        data = {}
        for band in bands:
            data[band] = {
                "time": timef(**timef_params),
                "magnitude": magf(size=size, **magf_params),
                "error":  errf(size=size, **errf_params)}
        return Dataset(
            id=id, ds_name=ds_name, description=description,
            bands=bands, metadata=metadata, data=data)


def create_normal(mu=0., sigma=1., mu_err=0.,
                  sigma_err=1., seed=None, **kwargs):

    random = np.random.RandomState(seed)
    return create_random(
        magf=random.normal, magf_params={"loc": mu, "scale": sigma},
        errf=random.normal, errf_params={"loc": mu_err, "scale": sigma_err},
        **kwargs)


def create_uniform(low=0., high=1., mu_err=0., sigma_err=1.,
                   seed=None, **kwargs):

    random = np.random.RandomState(seed)
    return create_random(
        magf=random.uniform, magf_params={"low": low, "high": high},
        errf=random.normal, errf_params={"loc": mu_err, "scale": sigma_err},
        **kwargs)


def create_periodic(mu_err=0., sigma_err=1., period=1, seed=None, **kwargs):
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
