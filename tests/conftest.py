#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

import os
import pathlib

import numpy as np

import pytest

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

TEST_DATASET_PATH = PATH / "data"


@pytest.fixture(scope="session")
def normal():
    def maker(*, random=None, size=100, loc=0.0, scale=1.0):
        random = np.random.default_rng(random)
        return random.normal(loc=loc, scale=scale, size=size)

    return maker


@pytest.fixture(scope="session")
def uniform():
    def maker(*, random=None, size=100, low=0.0, high=1.0):
        random = np.random.default_rng(random)
        return random.uniform(low=low, high=high, size=size)

    return maker


@pytest.fixture(scope="session")
def periodic():
    def maker(*, random=None, size=100, mean=None, cov=None, period=10):
        random = np.random.default_rng(random)

        if mean is None:
            mean = np.zeros(size)

        if cov is None:
            cov = np.zeros([size, size])
            for i in np.arange(size):
                for j in np.arange(size):
                    cov[i, j] = np.exp(
                        -(np.sin((np.pi / period) * (i - j)) ** 2)
                    )

        return random.multivariate_normal(mean=mean, cov=cov)

    return maker
