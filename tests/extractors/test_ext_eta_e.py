from feets.extractors import ext_eta_e

import numpy as np


def test_Eta_e_extract(normal_light_curve):
    # create the extractor
    extractor = ext_eta_e.Eta_e()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    values = np.empty(1000)
    for idx in range(values.size):
        lc = normal_light_curve(random=random, size=1000, data=["magnitude"])
        lc["time"] = np.arange(1000)
        values[idx] = extractor.extract(**lc)["Eta_e"]

    np.testing.assert_allclose(values.mean(), 1.9978701893508646)
