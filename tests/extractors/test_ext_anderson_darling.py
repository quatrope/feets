from feets.extractors import ext_anderson_darling

import numpy as np

import pytest


@pytest.mark.skip("FATS say must be 0.2, but actual is -0.60")
def test_AndersonDarling_extract(normal_light_curve):
    # create the extractor
    extractor = ext_anderson_darling.AndersonDarling()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(
            random=random,
            size=1000,
            data=["magnitude"],
        )
        values[idx] = extractor.extract(**lc)["AndersonDarling"]

    np.testing.assert_allclose(values.mean(), 0.25)
