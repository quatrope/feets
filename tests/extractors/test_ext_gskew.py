from feets.extractors import ext_gskew

import numpy as np


def test_Gskew_extract(normal_light_curve):
    # create the extractor
    extractor = ext_gskew.Gskew()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["Gskew"]

    np.testing.assert_allclose(values.mean(), -0.0007054194429618122)
