from feets.extractors import ext_con

import numpy as np


def test_Con_extract(normal_light_curve):
    # create the extractor
    extractor = ext_con.Con(consecutiveStar=1)

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
        values[idx] = extractor.extract(**lc)["Con"]

    np.testing.assert_allclose(values.mean(), 0.04554)
