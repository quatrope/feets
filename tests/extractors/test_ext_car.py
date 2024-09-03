from feets.extractors import ext_car

import numpy as np


def test_CAR_extract(periodic_light_curve):
    # create the extractor
    extractor = ext_car.CAR()

    # init the seed
    random = np.random.default_rng(42)

    # excute the simulation
    sims = 100
    size = 100

    time = np.arange(size)
    values = np.empty([sims, 3])
    for idx in range(sims):
        lc = periodic_light_curve(
            random=random,
            size=size,
            data=["magnitude"],
        )
        lc["time"] = time
        lc["error"] = random.normal(loc=1, scale=0.008, size=size)
        results = extractor.extract(**lc)
        values[idx] = (
            results["CAR_mean"],
            results["CAR_sigma"],
            results["CAR_tau"],
        )

    # test
    np.testing.assert_allclose(values[:, 0].mean(), -0.11888100485725793)
    np.testing.assert_allclose(values[:, 1].mean(), 0.008015313327483975)
    np.testing.assert_allclose(values[:, 2].mean(), 0.6470569371786853)
