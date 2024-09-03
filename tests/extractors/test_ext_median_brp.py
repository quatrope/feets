from feets.extractors import ext_median_brp

import numpy as np


def test_MedianBRP_extract(normal_light_curve):
    # create the extractor
    extractor = ext_median_brp.MedianBRP()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=10000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["MedianBRP"]

    np.testing.assert_allclose(values.mean(), 0.5596438)
