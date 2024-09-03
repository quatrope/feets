from feets.extractors import ext_slotted_a_length

import numpy as np

import pytest


@pytest.mark.skip(reason="Slow test")
def test_SlottedA_length_extract(normal_light_curve):
    # create the extractor
    extractor = ext_slotted_a_length.SlottedA_length()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=10000, data=["magnitude"])
        lc["time"] = np.arange(10000)
        values[idx] = extractor.extract(**lc)["SlottedA_length"]

    np.testing.assert_allclose(values.mean(), 1.0)
