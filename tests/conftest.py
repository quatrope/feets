from feets.extractors import core

import numpy as np

import pytest


@pytest.fixture(scope="session")
def normal_light_curve():
    def maker(*, data=None, size=100, seed=None, **kwargs):
        random = np.random.default_rng(seed)

        data = core.DATAS if data is None else data
        diff = set(data).difference(core.DATAS)
        if diff:
            raise ValueError(f"Invalid data/s {diff}")

        lc = {}
        for data_name in data:
            data_loc, data_scale = f"{data_name}_loc", f"{data_name}_scale"
            loc, scale = kwargs.get(data_loc, 0.0), kwargs.get(data_scale, 1.0)
            lc[data_name] = random.normal(loc=loc, scale=scale, size=size)
        return lc

    return maker
