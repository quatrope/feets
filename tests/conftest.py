import os
import pathlib

from feets.extractors import core

import numpy as np

import pytest

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

TEST_DATASET_PATH = PATH / "data"


@pytest.fixture(scope="session")
def normal_light_curve():
    def maker(*, data=None, size=100, random=None, **kwargs):
        random = np.random.default_rng(random)

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
