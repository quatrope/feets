from feets.extractors import ext_rcs

import numpy as np


def test_RCS_extract(uniform_light_curve):
    # create the extractor
    extractor = ext_rcs.RCS()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = uniform_light_curve(random=random, size=1000, data=["magnitude"])
        values[idx] = extractor.extract(**lc)["Rcs"]

    np.testing.assert_allclose(values.mean(), 0.038746172489149244)
