from feets.extractors import ext_autocor_length

import numpy as np


def test_AutocorLength_extract(normal_light_curve):
    # create the extractor
    extractor = ext_autocor_length.AutocorLength()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["Autocor_length"]

    np.testing.assert_allclose(values.mean(), 1.0)
