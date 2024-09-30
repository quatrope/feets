#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

import os
import pathlib

from feets.extractors import extractor

import numpy as np

import pytest

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

TEST_DATASET_PATH = PATH / "data"


@pytest.fixture(scope="session")
def normal_light_curve():
    def maker(*, data=None, size=100, random=None, **kwargs):
        random = np.random.default_rng(random)

        data = extractor.DATAS if data is None else data
        diff = set(data).difference(extractor.DATAS)
        if diff:
            raise ValueError(f"Invalid data/s {diff}")

        lc = {}
        for data_name in data:
            data_loc, data_scale = f"{data_name}_loc", f"{data_name}_scale"
            loc, scale = kwargs.get(data_loc, 0.0), kwargs.get(data_scale, 1.0)
            lc[data_name] = random.normal(loc=loc, scale=scale, size=size)
        return lc

    return maker


@pytest.fixture(scope="session")
def uniform_light_curve():
    def maker(*, data=None, size=100, random=None, **kwargs):
        random = np.random.default_rng(random)

        data = extractor.DATAS if data is None else data
        diff = set(data).difference(extractor.DATAS)
        if diff:
            raise ValueError(f"Invalid data/s {diff}")

        lc = {}
        for data_name in data:
            data_low, data_high = f"{data_name}_low", f"{data_name}_high"
            low, high = kwargs.get(data_low, 0.0), kwargs.get(data_high, 1.0)
            lc[data_name] = random.uniform(low=low, high=high, size=size)
        return lc

    return maker


@pytest.fixture(scope="session")
def periodic_light_curve():
    def maker(*, data=None, size=100, random=None, **kwargs):
        random = np.random.default_rng(random)

        data = extractor.DATAS if data is None else data
        diff = set(data).difference(extractor.DATAS)
        if diff:
            raise ValueError(f"Invalid data/s {diff}")

        lc = {}
        for data_name in data:
            data_mean, data_cov, data_period = (
                f"{data_name}_mean",
                f"{data_name}_cov",
                f"{data_name}_period",
            )
            mean, cov, period = (
                kwargs.get(data_mean, np.zeros(size)),
                kwargs.get(data_cov, None),
                kwargs.get(data_period, 10),
            )
            if cov is None:
                cov = np.zeros([size, size])
                for i in np.arange(size):
                    for j in np.arange(size):
                        cov[i, j] = np.exp(
                            -(np.sin((np.pi / period) * (i - j)) ** 2)
                        )
            lc[data_name] = random.multivariate_normal(mean=mean, cov=cov)
        return lc

    return maker
