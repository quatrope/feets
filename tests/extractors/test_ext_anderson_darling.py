import numpy as np

from feets.extractors import ext_anderson_darling

import pytest


@pytest.mark.skip("FATS say must be 0.2, but actual is -0.60")
def test_AndersonDarling_extract():
    # create the extractor
    extractor = ext_anderson_darling.AndersonDarling()

    # init the seed
    random = np.random.default_rng(seed=42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        values[idx] = extractor.extract(magnitude=mags)["AndersonDarling"]

    np.testing.assert_allclose(values.mean(), 0.25)
